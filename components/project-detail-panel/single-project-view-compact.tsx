'use client'

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { MiningProject } from "@/lib/types/mining-project"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { ExternalLink, AlertTriangle, FileText, Users, Bookmark, BookmarkCheck, ImageIcon, MessageSquare, Loader2 } from "lucide-react"
import { SensitivityAnalysis } from "./sensitivity-analysis"
import { AIInsightsPanel } from "@/components/ai-insights-panel"
import { InlinePDFViewerWrapper } from "./inline-pdf-viewer-wrapper"
import { cn } from "@/lib/utils"
import { ExportDropdown, ExportFormat } from "@/components/ui/export-dropdown"
import { exportProjects } from "@/lib/export-utils"
import { toast } from "sonner"
import { Toaster } from "@/components/ui/toaster"
import { supabase } from "@/lib/supabase/client"
import { useGlobalChat } from "@/lib/global-chat-context"
import { useChat } from "@/lib/chat-context"
import { formatNumber, formatCurrency, formatPercent, formatTonnes } from "@/lib/format-utils"
import { InteractiveMapbox } from "@/components/ui/interactive-mapbox"

interface SingleProjectViewProps {
  project: MiningProject
  onProjectSelect?: (projectId: string) => void
  onClose?: () => void
  initialPdfUrl?: string | null
  initialPdfTitle?: string | null
}

const getRiskBadgeColor = (risk: string) => {
  switch (risk.toLowerCase()) {
    case "low":
      return "bg-green-100 text-green-800"
    case "medium":
      return "bg-yellow-100 text-yellow-800"
    case "high":
      return "bg-red-100 text-red-800"
    default:
      return "bg-gray-100 text-gray-800"
  }
}

export function SingleProjectView({ project, onProjectSelect, onClose, initialPdfUrl, initialPdfTitle }: SingleProjectViewProps) {
  const router = useRouter()
  const [updatingWatchlist, setUpdatingWatchlist] = useState(false)
  const [generatingImage, setGeneratingImage] = useState(false)
  const [isWatchlisted, setIsWatchlisted] = useState(project.watchlist || false)
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(project.generated_image_url || null)
  const [pdfViewerOpen, setPdfViewerOpen] = useState(false)
  const [selectedPdfUrl, setSelectedPdfUrl] = useState<string | null>(null)
  const [selectedPdfTitle, setSelectedPdfTitle] = useState<string | null>(null)
  const [currentProject, setCurrentProject] = useState(project)
  const [similarProjects, setSimilarProjects] = useState<any[]>([])
  const [loadingSimilar, setLoadingSimilar] = useState(false)
  const { setInput } = useGlobalChat()
  const { toggleChat } = useChat()

  const handleViewPdf = (url: string, title: string) => {
    setSelectedPdfUrl(url)
    setSelectedPdfTitle(title)
    setPdfViewerOpen(true)
  }

  const handleClosePdf = () => {
    setPdfViewerOpen(false)
    setSelectedPdfUrl(null)
    setSelectedPdfTitle(null)
  }

  const handleProjectUpdated = async () => {
    // Refresh project data from database after extraction
    console.log('🔄 Refreshing project data after extraction...')
    try {
      const { data, error } = await supabase
        .from('projects')
        .select('*')
        .eq('id', project.id)
        .single()

      if (error) {
        console.error('Error refreshing project:', error)
      } else if (data) {
        console.log('✅ Project data refreshed:', data)
        console.log('📊 Resource value:', data.resource)
        console.log('📊 Reserve value:', data.reserve)
        setCurrentProject(data as MiningProject)
      }
    } catch (error) {
      console.error('Error refreshing project:', error)
    }
  }

  // Sync currentProject state when project prop changes
  React.useEffect(() => {
    console.log('🔄 Project prop changed, updating state:', project.id, project.name)
    setCurrentProject(project)
    setIsWatchlisted(project.watchlist || false)
    setGeneratedImageUrl(project.generated_image_url || null)
    // Reset PDF viewer when switching projects
    setPdfViewerOpen(false)
    setSelectedPdfUrl(null)
    setSelectedPdfTitle(null)
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [project.id])

  // Load similar projects
  React.useEffect(() => {
    async function loadSimilarProjects() {
      if (!project.id) return

      setLoadingSimilar(true)
      try {
        const response = await fetch(`/api/projects/${project.id}/similar`)
        if (response.ok) {
          const data = await response.json()
          setSimilarProjects(data.similar || [])
        }
      } catch (error) {
        console.error('Error loading similar projects:', error)
      } finally {
        setLoadingSimilar(false)
      }
    }

    loadSimilarProjects()
  }, [project.id])

  // Load full project data on mount including resource/reserve
  React.useEffect(() => {
    async function loadFullProjectData() {
      const { data, error } = await supabase
        .from('projects')
        .select(`
          *,
          companies (
            id,
            name,
            ticker
          )
        `)
        .eq('id', project.id)
        .single()

      if (error) {
        console.error('Error loading full project data:', error)
        setCurrentProject(project)
      } else if (data) {
        console.log('✅ Full project data loaded:', data)
        console.log('📊 Resource:', data.resource)
        console.log('📊 Reserve:', data.reserve)
        // Set the company name from the joined data
        const projectWithCompany = {
          ...data,
          company: data.companies?.name || data.company || null
        } as MiningProject
        setCurrentProject(projectWithCompany)
      }
    }

    loadFullProjectData()
    setIsWatchlisted(project.watchlist || false)
    setGeneratedImageUrl(project.generated_image_url || null)
  }, [project.id])

  // Auto-open PDF viewer if initial PDF URL is provided
  React.useEffect(() => {
    if (initialPdfUrl && initialPdfTitle) {
      handleViewPdf(initialPdfUrl, initialPdfTitle)
    }
  }, [initialPdfUrl, initialPdfTitle])

  const handleToggleWatchlist = async () => {
    try {
      setUpdatingWatchlist(true)
      const newWatchlistStatus = !isWatchlisted

      console.log('Updating watchlist for project:', project.id, 'to:', newWatchlistStatus)

      const { data, error } = await supabase
        .from('projects')
        .update({
          watchlist: newWatchlistStatus,
          watchlisted_at: newWatchlistStatus ? new Date().toISOString() : null
        })
        .eq('id', project.id)
        .select()

      if (error) {
        console.error('Supabase error:', error)
        throw error
      }

      console.log('Update successful:', data)

      // Update local state immediately for visual feedback
      setIsWatchlisted(newWatchlistStatus)

      toast.success(newWatchlistStatus ? 'Added to watchlist' : 'Removed from watchlist')

      // Trigger refresh
      window.dispatchEvent(new CustomEvent('refreshProjects'))
    } catch (error: any) {
      console.error('Error updating watchlist:', error)
      toast.error(`Failed to update watchlist: ${error.message || 'Unknown error'}`)
    } finally {
      setUpdatingWatchlist(false)
    }
  }

  const handleExport = (format: ExportFormat) => {
    exportProjects([project], format, `${project.project}-details`)
    toast.success(`Exported project as ${format.toUpperCase()}`)
  }

  const handleGenerateImage = async () => {
    try {
      setGeneratingImage(true)

      const response = await fetch('/api/projects/generate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ projectId: project.id })
      })

      const data = await response.json()

      if (!response.ok) {
        console.error('API Error:', data)
        throw new Error(data.error || data.details || 'Failed to generate image')
      }

      const { imageUrl } = data

      // Update local state with the generated image
      setGeneratedImageUrl(imageUrl)

      toast.success('Image generated successfully')

      // Trigger refresh after a small delay to ensure database is updated
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('refreshProjects'))
      }, 500)
    } catch (error: any) {
      console.error('Error generating image:', error)
      toast.error(error.message || 'Failed to generate image')
    } finally {
      setGeneratingImage(false)
    }
  }

  const handleAddToChat = () => {
    const projectContext = `Analyze this mining project:

Project: ${project.name}
Company: ${project.company || 'N/A'}
Location: ${project.location || 'N/A'}
Commodities: ${(project.commodities || []).join(', ')}${project.stage ? `\nStage: ${project.stage}` : ''}${project.status ? `\nStatus: ${project.status}` : ''}
Ownership: ${project.ownership_percentage !== null ? `${project.ownership_percentage}%` : 'N/A'}
Resource Estimate: ${currentProject.resource || 'N/A'}
Reserve Estimate: ${currentProject.reserve || 'N/A'}

What is your assessment of this project?`

    // Set the input in the chat
    setInput(projectContext)

    // Close the detail panel if there's an onClose handler
    if (onClose) {
      onClose()
    }

    // Open the chat panel
    toggleChat()

    toast.success('Project added to chat')
  }

  // If PDF viewer is open, show it instead of the main content
  if (pdfViewerOpen && selectedPdfUrl) {
    return (
      <div className="h-full flex flex-col">
        <InlinePDFViewerWrapper
          url={selectedPdfUrl}
          title={selectedPdfTitle || undefined}
          onClose={handleClosePdf}
          projectId={project.id}
          onProjectUpdated={handleProjectUpdated}
        />
      </div>
    )
  }

  return (
    <div className="space-y-4 p-6">
      {/* Company Info */}
      <div className="space-y-3">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-medium">{currentProject.name}</h2>
              {currentProject.is_private && (
                <Badge variant="secondary" className="text-[10px] px-2 py-0 bg-purple-100 text-purple-700 border-purple-300">
                  Private
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              {currentProject.company && currentProject.location
                ? `${currentProject.company} • ${currentProject.location}`
                : currentProject.company || currentProject.location || ''}
            </p>
            {currentProject.commodities && currentProject.commodities.length > 0 && (
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-muted-foreground">Primary:</span>
                <Badge variant="default" className="text-xs font-medium">
                  {currentProject.commodities[0]}
                </Badge>
                {currentProject.commodities.length > 1 && (
                  <span className="text-xs text-muted-foreground">
                    +{currentProject.commodities.length - 1} more
                  </span>
                )}
              </div>
            )}
          </div>
          <Badge className={cn("text-xs", getRiskBadgeColor(currentProject.riskLevel || 'Medium'))}>
            {currentProject.riskLevel || 'Medium'} Risk
          </Badge>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button
            variant={isWatchlisted ? "default" : "outline"}
            size="sm"
            onClick={handleToggleWatchlist}
            disabled={updatingWatchlist}
          >
            {updatingWatchlist ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : isWatchlisted ? (
              <BookmarkCheck className="h-4 w-4 mr-2 fill-current" />
            ) : (
              <Bookmark className="h-4 w-4 mr-2" />
            )}
            {isWatchlisted ? 'Remove from Watchlist' : 'Add to Watchlist'}
          </Button>

          <ExportDropdown onExport={handleExport} size="sm" />

          <Button
            variant="outline"
            size="sm"
            onClick={handleGenerateImage}
            disabled={generatingImage}
          >
            {generatingImage ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <ImageIcon className="h-4 w-4 mr-2" />
            )}
            Generate Image
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleAddToChat}
          >
            <MessageSquare className="h-4 w-4 mr-2" />
            Add to Chat
          </Button>

          {project.technicalReportUrl && (
            <Button
              variant="outline"
              size="sm"
              asChild
            >
              <a
                href={project.technicalReportUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                <FileText className="h-4 w-4 mr-2" />
                Technical Report
              </a>
            </Button>
          )}
        </div>

        {/* Location Map */}
        {currentProject.location && (
          <InteractiveMapbox
            location={currentProject.location}
            latitude={currentProject.latitude}
            longitude={currentProject.longitude}
            width={600}
            height={300}
            initialZoom={8}
          />
        )}

        {/* Generated Image Display */}
        {generatedImageUrl && (
          <div className="mt-4">
            <div className="relative rounded-lg overflow-hidden border">
              <img
                src={generatedImageUrl}
                alt={`AI visualization for ${project.project}`}
                className="w-full h-auto"
              />
              <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
                AI Generated
              </div>
            </div>
          </div>
        )}
      </div>

      <Separator />

      {/* Key Metrics Table */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Financial Metrics</h3>
        <div className="border rounded-lg divide-y">
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">NPV (Net Present Value)</span>
              <span className="text-sm font-medium">
                {currentProject.npv !== null && currentProject.npv !== undefined
                  ? `$${currentProject.npv.toFixed(0)}M`
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">IRR (Internal Rate of Return)</span>
              <span className="text-sm font-medium">
                {currentProject.irr !== null && currentProject.irr !== undefined
                  ? `${currentProject.irr.toFixed(1)}%`
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">CAPEX (Capital Expenditure)</span>
              <span className="text-sm font-medium">
                {currentProject.capex !== null && currentProject.capex !== undefined
                  ? `$${currentProject.capex.toFixed(0)}M`
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">AISC (All-In Sustaining Cost)</span>
              <span className="text-sm font-medium">
                {currentProject.aisc !== null && currentProject.aisc !== undefined
                  ? `$${currentProject.aisc.toFixed(0)}/unit`
                  : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <Separator />

      {/* Project Information */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Project Information</h3>
        <div className="border rounded-lg divide-y">
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Location</span>
              <span className="text-sm font-medium">{currentProject.location || 'N/A'}</span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Latitude</span>
              <span className="text-sm font-medium">
                {currentProject.latitude !== null && currentProject.latitude !== undefined
                  ? currentProject.latitude.toFixed(4)
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Longitude</span>
              <span className="text-sm font-medium">
                {currentProject.longitude !== null && currentProject.longitude !== undefined
                  ? currentProject.longitude.toFixed(4)
                  : 'N/A'}
              </span>
            </div>
          </div>
          {project.stage && (
            <div className="p-2.5">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Stage</span>
                <Badge variant="outline" className="text-xs">{project.stage}</Badge>
              </div>
            </div>
          )}
          {project.status && (
            <div className="p-2.5">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Status</span>
                <Badge variant="outline" className="text-xs">{project.status}</Badge>
              </div>
            </div>
          )}
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Commodities</span>
              <div className="flex flex-wrap gap-1 justify-end">
                {(project.commodities || []).map((commodity, i) => (
                  <Badge key={i} variant="outline" className="text-xs">{commodity}</Badge>
                ))}
              </div>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Ownership</span>
              <span className="text-sm font-medium">
                {project.ownership_percentage !== null && project.ownership_percentage !== undefined
                  ? `${project.ownership_percentage}%`
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Resource Estimate</span>
              <span className="text-sm font-medium">
                {(() => {
                  console.log('🔍 Displaying resource:', currentProject.resource, 'Type:', typeof currentProject.resource)
                  return currentProject.resource || 'N/A'
                })()}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Reserve Estimate</span>
              <span className="text-sm font-medium">
                {(() => {
                  console.log('🔍 Displaying reserve:', currentProject.reserve, 'Type:', typeof currentProject.reserve)
                  return currentProject.reserve || 'N/A'
                })()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Qualified Persons */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium">Qualified Persons</h3>
        <div className="border rounded-lg">
          {currentProject.qualified_persons && currentProject.qualified_persons.length > 0 ? (
            <div className="divide-y">
              {currentProject.qualified_persons.map((qp, idx) => (
                <div key={idx} className="p-2.5">
                  <div className="text-sm font-medium">{qp.name}</div>
                  <div className="text-xs text-muted-foreground">{qp.credentials}</div>
                  {qp.company && qp.company !== 'Not specified' && (
                    <div className="text-xs text-muted-foreground mt-0.5">{qp.company}</div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="p-2.5">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Qualified Persons</span>
                <span className="text-sm font-medium">-</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Description */}
      {project.description && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Description</h3>
          <p className="text-sm text-muted-foreground">{project.description}</p>
        </div>
      )}

      {/* Project Documents - Prefer Supabase Storage over FactSet URLs */}
      {(currentProject.document_storage_path || (project.urls && project.urls.length > 0)) && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Technical Documents</h3>
          <div className="flex flex-wrap gap-2">
            {/* Show Supabase Storage document if available */}
            {currentProject.document_storage_path && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const url = currentProject.document_storage_path!
                  const isPdf = url.toLowerCase().includes('.pdf')
                  const isHtml = url.toLowerCase().includes('.html')

                  if (isPdf || isHtml) {
                    handleViewPdf(url, `${project.name} - Technical Report`)
                  } else {
                    window.open(url, '_blank')
                  }
                }}
              >
                <FileText className="h-3 w-3 mr-1" />
                Technical Report
              </Button>
            )}

            {/* Show FactSet URLs only if no Supabase document available */}
            {!currentProject.document_storage_path && project.urls && project.urls.map((url, i) => {
              const isPdf = url.toLowerCase().includes('.pdf')
              const isHtml = url.toLowerCase().includes('.html')
              return (
                <Button
                  key={i}
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (isPdf || isHtml) {
                      handleViewPdf(url, `${project.name} - Document ${i + 1}`)
                    } else {
                      window.open(url, '_blank')
                    }
                  }}
                >
                  <FileText className="h-3 w-3 mr-1" />
                  {isPdf ? `PDF ${i + 1}` : isHtml ? `HTML ${i + 1}` : `Source ${i + 1}`}
                </Button>
              )
            })}
          </div>
        </div>
      )}

      {/* Project Details Table */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Additional Details</h3>
        <div className="border rounded-lg divide-y">
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Last Updated</span>
              <span className="text-xs">{new Date(project.updated_at).toLocaleDateString()}</span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Primary Commodity</span>
              <span className="text-sm">{project.primaryCommodity || 'N/A'}</span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Resource Grade</span>
              <span className="text-sm">
                {formatNumber(project.resourceGrade, { decimals: 2, suffix: ` ${project.gradeUnit || '%'}` })}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Contained Metal</span>
              <span className="text-sm">
                {formatTonnes(project.containedMetal, { decimals: 0, unit: 't' })}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">ESG Score</span>
              <Badge variant="outline" className="text-xs">{project.esgScore}</Badge>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Technical Report</span>
              {project.technicalReportUrl ? (
                <a
                  href={project.technicalReportUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-blue-600 hover:underline flex items-center gap-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  View Report
                  <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              ) : (
                <span className="text-sm text-muted-foreground">N/A</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Risk Alerts */}
      {typeof project.redFlags === 'number' && project.redFlags > 0 ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-600" />
            <span className="text-sm font-medium text-red-900">
              {project.redFlags} Risk Flags Identified
            </span>
          </div>
        </div>
      ) : null}

      {/* Tabs */}
      <Tabs defaultValue="insights" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="technical">Technical Report</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="experts">Experts</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-4 mt-4">
          <Card className="p-4">
            <h4 className="text-sm font-medium mb-3">Production Profile</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Annual Production</span>
                <span>{project.annualProduction || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Resource</span>
                <span>{project.totalResource || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Strip Ratio</span>
                <span>{project.stripRatio || 'N/A'}</span>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <h4 className="text-sm font-medium mb-3">Infrastructure</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Power Supply</span>
                <span>{project.powerSupply || 'Grid connected'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Water Source</span>
                <span>{project.waterSource || 'Local aquifer'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Transport</span>
                <span>{project.transport || 'Road & rail access'}</span>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="technical" className="mt-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Technical Documentation</h4>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              {project.document_storage_path ? (
                <Button
                  variant="outline"
                  className="w-full justify-start text-sm"
                  onClick={() => {
                    // Construct the full Supabase storage URL
                    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
                    const bucketPath = project.document_storage_path

                    // If it's already a full URL, use it as is
                    if (project.document_storage_path.startsWith('http')) {
                      handleViewPdf(project.document_storage_path, `${project.name} - Technical Report`)
                    } else {
                      // Otherwise construct the Supabase storage URL
                      const fullUrl = `${supabaseUrl}/storage/v1/object/public/${bucketPath}`
                      handleViewPdf(fullUrl, `${project.name} - Technical Report`)
                    }
                  }}
                >
                  <FileText className="mr-2 h-4 w-4" />
                  Technical Report - View Document
                </Button>
              ) : project.urls && project.urls.length > 0 ? (
                // Fallback to URLs array if document_storage_path is not available
                project.urls.map((url, i) => {
                  const isPdf = url.toLowerCase().endsWith('.pdf')
                  const isHtml = url.toLowerCase().endsWith('.html')
                  return (
                    <Button
                      key={i}
                      variant="outline"
                      className="w-full justify-start text-sm"
                      onClick={() => {
                        if (isPdf || isHtml) {
                          handleViewPdf(url, `${project.name} - Document ${i + 1}`)
                        } else {
                          window.open(url, '_blank')
                        }
                      }}
                    >
                      <FileText className="mr-2 h-4 w-4" />
                      Technical Report {project.urls && project.urls.length > 1 ? `(${i + 1})` : ''}
                    </Button>
                  )
                })
              ) : (
                <div className="text-sm text-muted-foreground p-2 text-center">
                  No technical documentation available
                </div>
              )}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="mt-4">
          <AIInsightsPanel
            projectId={project.id}
            projectName={project.name}
            autoGenerate={true}
          />
        </TabsContent>

        <TabsContent value="experts" className="mt-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Industry Experts</h4>
              <Users className="h-4 w-4 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">
              Connect with verified mining industry experts for deeper insights.
            </p>
          </Card>
        </TabsContent>
      </Tabs>

      <Separator />

      {/* Sensitivity Analysis */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Sensitivity Analysis</h3>
        <SensitivityAnalysis project={currentProject} />
      </div>

      <Separator />

      {/* Similar Projects */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Similar Projects</h3>
        {loadingSimilar ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : similarProjects.length > 0 ? (
          <div className="grid grid-cols-2 gap-3">
            {similarProjects.map((similar) => (
              <Card
                key={similar.id}
                className="p-3 cursor-pointer hover:bg-muted/50 hover:shadow-md transition-all active:scale-95"
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  console.log('🔗 Navigating to similar project:', similar.id, similar.name)
                  router.push(`/projects/${similar.id}`)
                }}
              >
                <div className="space-y-2 pointer-events-none">
                  <h4 className="text-sm font-medium line-clamp-1">{similar.name}</h4>
                  {similar.commodities && similar.commodities.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {similar.commodities.slice(0, 2).map((commodity: string, idx: number) => (
                        <Badge key={idx} variant="outline" className="text-xs px-1.5 py-0">
                          {commodity}
                        </Badge>
                      ))}
                    </div>
                  )}
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{similar.npv ? `NPV: $${similar.npv}M` : 'NPV: N/A'}</span>
                    <span>{similar.irr ? `IRR: ${similar.irr}%` : 'IRR: N/A'}</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-4">No similar projects found</p>
        )}
      </div>

      {/* Quick Actions */}
      <div className="sticky bottom-0 bg-background border-t pt-4 -mx-6 px-6 -mb-6">
        <div className="flex gap-2">
          <Button
            className="flex-1"
            size="sm"
            onClick={() => {
              // Close the detail panel first if there's an onClose handler
              if (onClose) {
                onClose()
              }

              // Navigate back to global projects with this project pre-selected
              // Use setTimeout to ensure the page loads before we try to select the row
              router.push(`/global-projects`)

              setTimeout(() => {
                // Store the project ID in sessionStorage so the screener can pre-select it
                sessionStorage.setItem('preselect-project', project.id)

                toast.info('Select more projects to compare', {
                  description: 'This project is selected. Pick 1+ more and click Compare.'
                })
              }, 100)
            }}
          >
            Compare Projects
          </Button>
          <Button
            className="flex-1"
            variant="outline"
            size="sm"
            onClick={() => handleExport('excel')}
          >
            Export Analysis
          </Button>
        </div>
      </div>
      <Toaster />
    </div>
  )
} 