'use client'

import React, { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { X, ArrowLeft, ExternalLink, Bookmark, BookmarkCheck, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Card } from "@/components/ui/card"
import { Company } from "@/lib/hooks/use-companies"
import { supabase } from "@/lib/supabase/client"
import { toast } from "sonner"
import { CompanyComparisonView } from "./company-comparison-view"
import { formatCurrency } from "@/lib/format-utils"

interface Project {
  id: string
  name: string
  location?: string
  stage?: string
  commodities?: string[]
  description?: string
  status?: string
}

interface CompanyDetailPanelProps {
  isOpen: boolean
  onClose: () => void
  companies: Company[]
  mode?: "single" | "comparison"
  onProjectSelect?: (projectId: string) => void
}

export function CompanyDetailPanel({
  isOpen,
  onClose,
  companies,
  mode = "single",
  onProjectSelect,
}: CompanyDetailPanelProps) {
  const router = useRouter()
  const company = companies[0] // For single mode
  const [updatingWatchlist, setUpdatingWatchlist] = useState(false)
  const [isWatchlisted, setIsWatchlisted] = useState(company?.watchlist || false)
  const [companyProjects, setCompanyProjects] = useState<Project[]>([])
  const [loadingProjects, setLoadingProjects] = useState(false)

  // Update local state when company prop changes
  React.useEffect(() => {
    if (company) {
      setIsWatchlisted(company.watchlist || false)
    }
  }, [company])

  // Fetch projects belonging to this company
  useEffect(() => {
    async function fetchCompanyProjects() {
      if (!company?.id || mode !== "single") return

      setLoadingProjects(true)
      try {
        const { data, error } = await supabase
          .from('projects')
          .select('id, name, location, stage, commodities, description, status')
          .eq('company_id', company.id)
          .order('name')

        if (error) {
          console.error('Error fetching company projects:', error)
        } else {
          setCompanyProjects(data || [])
        }
      } catch (error) {
        console.error('Error fetching company projects:', error)
      } finally {
        setLoadingProjects(false)
      }
    }

    if (isOpen) {
      fetchCompanyProjects()
    }
  }, [company?.id, isOpen, mode])

  // Close on ESC key
  React.useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose()
      }
    }

    document.addEventListener("keydown", handleEscape)
    return () => document.removeEventListener("keydown", handleEscape)
  }, [isOpen, onClose])

  // Prevent body scroll when panel is open
  React.useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden"
    } else {
      document.body.style.overflow = "unset"
    }

    return () => {
      document.body.style.overflow = "unset"
    }
  }, [isOpen])

  const handleToggleWatchlist = async () => {
    if (!company) return

    try {
      setUpdatingWatchlist(true)
      const newWatchlistStatus = !isWatchlisted

      const { error } = await supabase
        .from('companies')
        .update({ watchlist: newWatchlistStatus })
        .eq('id', company.id)

      if (error) throw error

      setIsWatchlisted(newWatchlistStatus)
      toast.success(newWatchlistStatus ? 'Added to watchlist' : 'Removed from watchlist')

      // Trigger refresh
      window.dispatchEvent(new CustomEvent('refreshCompanies'))
    } catch (error: any) {
      console.error('Error updating watchlist:', error)
      toast.error(`Failed to update watchlist: ${error.message || 'Unknown error'}`)
    } finally {
      setUpdatingWatchlist(false)
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          "fixed inset-0 bg-black/20 backdrop-blur-sm transition-opacity z-40",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full bg-background shadow-xl transition-transform duration-300 ease-in-out z-50",
          "w-full",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 bg-background border-b px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Companies
              </Button>
              <h2 className="text-lg font-semibold">
                {mode === "comparison"
                  ? `Comparing ${companies.length} Companies`
                  : company?.name || "Company Details"}
              </h2>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="h-[calc(100%-73px)] overflow-y-auto">
          {mode === "comparison" ? (
            <CompanyComparisonView companies={companies} />
          ) : company ? (
          <div className="space-y-4 p-6">
            {/* Company Info */}
            <div className="space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-medium">{company.name}</h2>
                  {company.ticker && company.exchange && (
                    <p className="text-sm text-muted-foreground">
                      {company.ticker} • {company.exchange} • {company.country || 'N/A'}
                    </p>
                  )}
                </div>
                {company.market_cap && (
                  <Badge variant="outline" className="text-sm">
                    {formatCurrency(company.market_cap, { decimals: company.market_cap >= 1000 ? 1 : 0, unit: 'M' })}
                  </Badge>
                )}
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

                {company.website && (
                  <Button
                    variant="outline"
                    size="sm"
                    asChild
                  >
                    <a
                      href={company.website}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="h-4 w-4 mr-2" />
                      Visit Website
                    </a>
                  </Button>
                )}
              </div>
            </div>

            <Separator />

            {/* Company Information */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium">Company Information</h3>
              <div className="border rounded-lg divide-y">
                <div className="p-2.5">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Name</span>
                    <span className="text-sm font-medium">{company.name}</span>
                  </div>
                </div>
                {company.ticker && (
                  <div className="p-2.5">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Ticker</span>
                      <Badge variant="outline" className="text-xs">{company.ticker}</Badge>
                    </div>
                  </div>
                )}
                {company.exchange && (
                  <div className="p-2.5">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Exchange</span>
                      <Badge variant="outline" className="text-xs">{company.exchange}</Badge>
                    </div>
                  </div>
                )}
                {company.country && (
                  <div className="p-2.5">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Country</span>
                      <span className="text-sm font-medium">{company.country}</span>
                    </div>
                  </div>
                )}
                {company.market_cap && (
                  <div className="p-2.5">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Market Cap</span>
                      <span className="text-sm font-medium">
                        {formatCurrency(company.market_cap, { decimals: company.market_cap >= 1000 ? 1 : 0, unit: 'M' })}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Description */}
            {company.description && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Description</h3>
                <p className="text-sm text-muted-foreground">{company.description}</p>
              </div>
            )}

            {/* Resources */}
            {company.urls && company.urls.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Resources</h3>
                <div className="flex flex-wrap gap-2">
                  {company.urls.map((url, i) => (
                    <Button
                      key={i}
                      variant="outline"
                      size="sm"
                      asChild
                    >
                      <a
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs"
                      >
                        <ExternalLink className="h-3 w-3 mr-1" />
                        Source {i + 1}
                      </a>
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Additional Details */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium">Additional Details</h3>
              <div className="border rounded-lg divide-y">
                <div className="p-2.5">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Last Updated</span>
                    <span className="text-xs">{new Date(company.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>
                <div className="p-2.5">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Website</span>
                    {company.website ? (
                      <a
                        href={company.website}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:underline text-sm flex items-center gap-1"
                      >
                        <ExternalLink className="h-3 w-3" />
                        Link
                      </a>
                    ) : (
                      <span className="text-sm text-gray-400">N/A</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <Separator />

            {/* Company Projects */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium">
                Projects {companyProjects.length > 0 && `(${companyProjects.length})`}
              </h3>
              {loadingProjects ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : companyProjects.length > 0 ? (
                <div className="grid grid-cols-2 gap-3">
                  {companyProjects.map((project) => (
                    <Card
                      key={project.id}
                      className="p-3 cursor-pointer hover:bg-muted/50 hover:shadow-md transition-all active:scale-95"
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        console.log('🔗 Navigating to project from company:', project.id, project.name)
                        router.push(`/projects/${project.id}`)
                      }}
                    >
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium line-clamp-1">{project.name}</h4>
                        {project.commodities && project.commodities.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {project.commodities.slice(0, 3).map((commodity, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs">
                                {commodity}
                              </Badge>
                            ))}
                            {project.commodities.length > 3 && (
                              <Badge variant="outline" className="text-xs">
                                +{project.commodities.length - 3}
                              </Badge>
                            )}
                          </div>
                        )}
                        <div className="flex gap-3 text-xs text-muted-foreground">
                          {project.location && (
                            <span className="line-clamp-1">{project.location}</span>
                          )}
                          {project.stage && (
                            <Badge variant="outline" className="text-xs">
                              {project.stage}
                            </Badge>
                          )}
                        </div>
                        {project.status && (
                          <div className="text-xs text-muted-foreground">
                            Status: {project.status}
                          </div>
                        )}
                      </div>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground text-center py-6">
                  No projects linked to this company yet.
                </div>
              )}
            </div>
          </div>
          ) : null}
        </div>
      </div>
    </>
  )
}
