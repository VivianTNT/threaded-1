"use client"

import * as React from "react"
import { Viewer, Worker, SpecialZoomLevel, type RenderViewer } from "@react-pdf-viewer/core"
import { highlightPlugin, Trigger } from "@react-pdf-viewer/highlight"
import { pageNavigationPlugin } from "@react-pdf-viewer/page-navigation"
import { searchPlugin, RenderSearchProps } from "@react-pdf-viewer/search"
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { X, Download, Maximize2, Sparkles, Loader2, Search, ChevronDown, ChevronUp, GripHorizontal, ExternalLink, FileText } from "lucide-react"

import "@react-pdf-viewer/core/lib/styles/index.css"
import "@react-pdf-viewer/highlight/lib/styles/index.css"
import "@react-pdf-viewer/search/lib/styles/index.css"

interface InlinePDFViewerProps {
  url: string
  title?: string
  onClose: () => void
  projectId?: string | null
  onProjectUpdated?: () => void
}

interface Highlight {
  id: string
  content: string
  highlightAreas: Array<{
    pageIndex: number
    left: number
    top: number
    width: number
    height: number
  }>
  quote: string
}

export function InlinePDFViewer({ url, title, onClose, projectId, onProjectUpdated }: InlinePDFViewerProps) {
  const [highlights, setHighlights] = React.useState<Highlight[]>([])
  const [loading, setLoading] = React.useState(true)
  const [autoExtracting, setAutoExtracting] = React.useState(false)
  const [documentLoaded, setDocumentLoaded] = React.useState(false)
  const [pdfError, setPdfError] = React.useState<string | null>(null)
  const hasAttemptedAutoExtract = React.useRef(false)
  const jumpToPageRef = React.useRef<((pageIndex: number) => void) | null>(null)

  // Auto-extract function
  const autoExtractKeyData = React.useCallback(async () => {
    if (!projectId || hasAttemptedAutoExtract.current) return

    hasAttemptedAutoExtract.current = true
    setAutoExtracting(true)
    try {
      const response = await fetch('/api/pdf/extract-highlights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pdfUrl: url,
          projectId: projectId,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        if (data.highlights) {
          setHighlights(data.highlights)
          console.log('✅ Auto-extracted highlights:', data.highlights.length)

          if (data.projectUpdated) {
            console.log('✅ Project database updated with extracted data')
            // Trigger refresh of project data in parent component
            if (onProjectUpdated) {
              onProjectUpdated()
            }
          }
        }
      }
    } catch (error) {
      console.error('Error auto-extracting:', error)
    } finally {
      setAutoExtracting(false)
    }
  }, [url, projectId, onProjectUpdated])

  // Load existing highlights from database and create fallback from project data
  React.useEffect(() => {
    async function loadHighlights() {
      try {
        const response = await fetch(`/api/pdf/extract-highlights?pdfUrl=${encodeURIComponent(url)}`)
        if (response.ok) {
          const data = await response.json()
          console.log('📥 Highlight API response:', data)
          if (data.highlights?.highlights && data.highlights.highlights.length > 0) {
            console.log('✅ Loaded existing highlights:', data.highlights.highlights)
            setHighlights(data.highlights.highlights)
          } else {
            console.log('⚠️ No highlights found')
            // If we have projectId, create synthetic highlights from project data
            if (projectId) {
              await createSyntheticHighlights()
            } else {
              // Try auto-extraction
              autoExtractKeyData()
            }
          }
        } else {
          console.log('⚠️ Highlight API request failed:', response.status)
          // Fallback to synthetic highlights if we have projectId
          if (projectId) {
            await createSyntheticHighlights()
          }
        }
      } catch (error) {
        console.error('❌ Error loading highlights:', error)
        // Fallback to synthetic highlights if we have projectId
        if (projectId) {
          await createSyntheticHighlights()
        }
      } finally {
        setLoading(false)
      }
    }

    async function createSyntheticHighlights() {
      try {
        const projectResponse = await fetch(`/api/projects/${projectId}`)
        if (projectResponse.ok) {
          const project = await projectResponse.json()
          const syntheticHighlights: any[] = []

          if (project.npv) {
            syntheticHighlights.push({
              id: `synthetic-npv-${Date.now()}`,
              content: `NPV: $${project.npv}M`,
              quote: `Net Present Value: $${project.npv} million`,
              highlightAreas: [],
              dataType: 'npv',
              value: project.npv,
              page: null,
            })
          }

          if (project.irr) {
            syntheticHighlights.push({
              id: `synthetic-irr-${Date.now()}`,
              content: `IRR: ${project.irr}%`,
              quote: `Internal Rate of Return: ${project.irr}%`,
              highlightAreas: [],
              dataType: 'irr',
              value: project.irr,
              page: null,
            })
          }

          if (project.capex) {
            syntheticHighlights.push({
              id: `synthetic-capex-${Date.now()}`,
              content: `CAPEX: $${project.capex}M`,
              quote: `Capital Expenditure: $${project.capex} million`,
              highlightAreas: [],
              dataType: 'capex',
              value: project.capex,
              page: null,
            })
          }

          if (project.resource) {
            syntheticHighlights.push({
              id: `synthetic-resources-${Date.now()}`,
              content: project.resource,
              quote: project.resource,
              highlightAreas: [],
              dataType: 'resources',
              page: null,
            })
          }

          if (project.reserve) {
            syntheticHighlights.push({
              id: `synthetic-reserves-${Date.now()}`,
              content: project.reserve,
              quote: project.reserve,
              highlightAreas: [],
              dataType: 'reserves',
              page: null,
            })
          }

          if (syntheticHighlights.length > 0) {
            console.log('✅ Created synthetic highlights from project data:', syntheticHighlights)
            setHighlights(syntheticHighlights)
          }
        }
      } catch (error) {
        console.error('Error creating synthetic highlights:', error)
      }
    }

    loadHighlights()
  }, [url, projectId, autoExtractKeyData])

  // Save highlights to database
  const saveHighlights = React.useCallback(async (newHighlights: Highlight[]) => {
    try {
      const response = await fetch('/api/pdf/extract-highlights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pdfUrl: url,
          highlights: newHighlights,
        }),
      })
      if (!response.ok) {
        console.error('Failed to save highlights')
      }
    } catch (error) {
      console.error('Error saving highlights:', error)
    }
  }, [url])

  const renderHighlightTarget = (props: any) => (
    <div
      style={{
        background: "#fef08a",
        display: "flex",
        position: "absolute",
        left: `${props.selectionRegion.left}%`,
        top: `${props.selectionRegion.top + props.selectionRegion.height}%`,
        transform: "translate(0, 8px)",
        zIndex: 1,
      }}
    >
      <Button
        size="sm"
        variant="secondary"
        onClick={() => {
          const newHighlight: Highlight = {
            id: Math.random().toString(36).substr(2, 9),
            content: props.selectedText,
            highlightAreas: props.highlightAreas,
            quote: props.selectedText,
          }
          const updatedHighlights = [...highlights, newHighlight]
          setHighlights(updatedHighlights)
          saveHighlights(updatedHighlights)
          props.cancel()
        }}
        className="h-7 text-xs"
      >
        Highlight
      </Button>
    </div>
  )

  const renderHighlights = (props: any) => (
    <div>
      {highlights.map((highlight) => (
        <React.Fragment key={highlight.id}>
          {Array.isArray(highlight.highlightAreas) &&
            highlight.highlightAreas
              .filter((area) => area.pageIndex === props.pageIndex)
              .map((area, idx) => (
                <div
                  key={idx}
                  className="highlight-area"
                  style={{
                    ...props.getCssProperties(area, props.rotation),
                    background: "rgba(250, 204, 21, 0.3)",
                    border: "1px solid rgba(250, 204, 21, 0.6)",
                    cursor: "pointer",
                  }}
                  title={`Highlighted: "${highlight.quote}"`}
                />
              ))}
        </React.Fragment>
      ))}
    </div>
  )

  const highlightPluginInstance = highlightPlugin({
    renderHighlightTarget,
    renderHighlights,
    trigger: Trigger.TextSelection,
  })
  const { jumpToHighlightArea } = highlightPluginInstance

  const pageNavigationPluginInstance = pageNavigationPlugin()
  const { jumpToPage } = pageNavigationPluginInstance

  // Store jumpToPage in ref for reliable access
  React.useEffect(() => {
    jumpToPageRef.current = jumpToPage
  }, [jumpToPage])

  const searchPluginInstance = searchPlugin()

  function handleDownload() {
    const link = document.createElement("a")
    link.href = url
    link.download = title || "document.pdf"
    link.target = "_blank"
    link.click()
  }

  function openFullScreen() {
    window.open(url, "_blank")
  }

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header Controls */}
      <div className="sticky top-0 z-10 bg-background border-b px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
            <h3 className="font-semibold text-sm truncate max-w-xs">
              {title || "PDF Document"}
            </h3>
          </div>

          <div className="flex items-center gap-1">
            <Button
              variant="default"
              size="sm"
              onClick={autoExtractKeyData}
              disabled={autoExtracting || highlights.length > 0}
              className="h-8 gap-1"
            >
              {autoExtracting ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Sparkles className="h-3 w-3" />
              )}
              <span className="text-xs">
                {highlights.length > 0 ? 'Data Extracted' : 'Extract Key Data'}
              </span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={openFullScreen}
              className="h-8 gap-1"
            >
              <Maximize2 className="h-3 w-3" />
              <span className="text-xs">Open</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDownload}
              className="h-8 gap-1"
            >
              <Download className="h-3 w-3" />
              <span className="text-xs">Save</span>
            </Button>
          </div>
        </div>

        {/* Search Bar */}
        {searchPluginInstance.Search && (
          <searchPluginInstance.Search>
            {(renderSearchProps: RenderSearchProps) => (
              <div className="mt-3 flex items-center gap-2">
                <div className="flex-1 flex items-center gap-1 bg-muted rounded-md px-2 py-1">
                  <Search className="h-3.5 w-3.5 text-muted-foreground" />
                  <Input
                    type="text"
                    placeholder="Search in document..."
                    value={renderSearchProps.keyword}
                    onChange={(e) => {
                      renderSearchProps.setKeyword(e.target.value)
                    }}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        renderSearchProps.search()
                      }
                    }}
                    className="h-7 border-0 bg-transparent text-xs focus-visible:ring-0 focus-visible:ring-offset-0"
                  />
                  {renderSearchProps.keyword && (
                    <button
                      onClick={() => {
                        renderSearchProps.clearKeyword()
                      }}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
                {renderSearchProps.numberOfMatches > 0 && (
                  <div className="text-xs text-muted-foreground whitespace-nowrap">
                    {renderSearchProps.currentMatch}/{renderSearchProps.numberOfMatches}
                  </div>
                )}
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={renderSearchProps.jumpToPreviousMatch}
                    disabled={!renderSearchProps.keyword || renderSearchProps.numberOfMatches === 0}
                    className="h-7 w-7"
                    title="Previous match"
                  >
                    <ChevronUp className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={renderSearchProps.jumpToNextMatch}
                    disabled={!renderSearchProps.keyword || renderSearchProps.numberOfMatches === 0}
                    className="h-7 w-7"
                    title="Next match"
                  >
                    <ChevronDown className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            )}
          </searchPluginInstance.Search>
        )}
      </div>

      {/* PDF Content with Resizable Panels */}
      <div className="flex-1 overflow-hidden">
        <PanelGroup direction="vertical">
          <Panel defaultSize={70} minSize={30}>
            <div className="h-full overflow-auto bg-background">
              <Worker workerUrl="/pdf.worker.min.js">
                <div className="h-full">
                  {pdfError ? (
                    <div className="flex flex-col items-center justify-center h-full p-8">
                      <div className="text-destructive mb-4">
                        <FileText className="h-12 w-12 opacity-50" />
                      </div>
                      <h3 className="text-lg font-semibold mb-2">Unable to load document</h3>
                      <p className="text-sm text-muted-foreground text-center mb-4">
                        {pdfError}
                      </p>
                      <Button
                        variant="outline"
                        onClick={() => window.open(url, '_blank')}
                        className="gap-2"
                      >
                        <ExternalLink className="h-4 w-4" />
                        Try opening in a new tab
                      </Button>
                    </div>
                  ) : (
                    <Viewer
                      fileUrl={url}
                      plugins={[highlightPluginInstance, pageNavigationPluginInstance, searchPluginInstance]}
                      defaultScale={SpecialZoomLevel.PageWidth}
                      theme={{
                        theme: "light",
                      }}
                      onDocumentLoad={() => {
                        console.log('📄 PDF document fully loaded and ready for navigation')
                        setDocumentLoaded(true)
                        setLoading(false)
                        setPdfError(null)
                      }}
                      onLoadError={(error: any) => {
                        console.error('❌ Failed to load PDF:', error)
                        setLoading(false)
                        setPdfError('Failed to load the document. The file may be unavailable or in an unsupported format.')
                      }}
                    />
                  )}
                </div>
              </Worker>
            </div>
          </Panel>

          {/* Extracted Data Highlights Section */}
          {highlights.length > 0 && (
            <>
              <PanelResizeHandle className="h-1 bg-border hover:bg-accent transition-colors flex items-center justify-center group">
                <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                  <GripHorizontal className="h-3 w-3 text-muted-foreground" />
                </div>
              </PanelResizeHandle>
              <Panel defaultSize={35} minSize={20} maxSize={60}>
                <div className="h-full overflow-auto border-t p-4 bg-background">
                  <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-yellow-600" />
                    Extracted Key Data ({highlights.length})
                  </h4>
                  <p className="text-xs text-muted-foreground mb-4">
                    Click any item to jump to the source in the document
                  </p>

                  <div className="space-y-3">
                    {highlights.map((highlight: any) => {
                      return (
                        <div
                          key={highlight.id}
                          className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg hover:bg-yellow-100 transition-colors cursor-pointer"
                          onClick={() => {
                            console.log('🖱️ Clicked highlight:', highlight)
                            console.log('  - Page:', highlight.page)
                            console.log('  - Document loaded:', documentLoaded)
                            console.log('  - jumpToPageRef available:', !!jumpToPageRef.current)

                            // Ensure document is loaded before attempting navigation
                            if (!documentLoaded) {
                              console.warn('⚠️ Document not fully loaded yet, waiting...')
                              // Wait a bit and try again
                              setTimeout(() => {
                                if (jumpToPageRef.current && highlight.page !== null && highlight.page !== undefined) {
                                  console.log('  - Retrying jump to page', highlight.page - 1)
                                  jumpToPageRef.current(highlight.page - 1)
                                }
                              }, 500)
                              return
                            }

                            // Use ref to ensure we have the latest jumpToPage function
                            if (highlight.page !== null && highlight.page !== undefined && jumpToPageRef.current) {
                              console.log('  - Jumping to page', highlight.page - 1)
                              // Add small delay to ensure viewer is ready
                              setTimeout(() => {
                                if (jumpToPageRef.current) {
                                  jumpToPageRef.current(highlight.page - 1)
                                }
                              }, 100)
                            } else {
                              console.log('  - No navigation info available')
                            }
                          }}
                        >
                          {/* Header: DATATYPE → PAGE X */}
                          <div className="text-xs font-bold uppercase text-yellow-800 mb-2">
                            {highlight.dataType || 'Data'} → PAGE {highlight.page}
                          </div>

                          {/* Source Quote */}
                          <div className="text-sm text-yellow-900 leading-relaxed">
                            {highlight.quote}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </Panel>
            </>
          )}
        </PanelGroup>
      </div>
    </div>
  )
}
