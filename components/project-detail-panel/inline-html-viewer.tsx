"use client"

import React, { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { X, ExternalLink, Download, Loader2, ChevronRight, ArrowLeft, Sparkles, CheckCircle2 } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"

interface InlineHTMLViewerProps {
  url: string
  title?: string
  onClose: () => void
  projectId?: string | null
  onProjectUpdated?: () => void
}

interface ExtractedHighlight {
  id: string
  content: string
  dataType: string
  value?: any
  highlightArea?: {
    elementId?: string
    sectionId?: string
  }
}

export function InlineHTMLViewer({
  url,
  title,
  onClose,
  projectId,
  onProjectUpdated
}: InlineHTMLViewerProps) {
  const [extracting, setExtracting] = useState(false)
  const [highlights, setHighlights] = useState<ExtractedHighlight[]>([])
  const [extractedData, setExtractedData] = useState<any>(null)
  const [hasExistingData, setHasExistingData] = useState(false)
  const [htmlContent, setHtmlContent] = useState<string>('')
  const [loadingHtml, setLoadingHtml] = useState(true)
  const iframeRef = useRef<HTMLIFrameElement>(null)

  // Fetch HTML content on mount
  useEffect(() => {
    fetchHTMLContent()
    checkExistingExtraction()
  }, [url])

  const fetchHTMLContent = async () => {
    try {
      setLoadingHtml(true)
      const response = await fetch(url)
      const html = await response.text()
      setHtmlContent(html)
    } catch (error) {
      console.error('Error fetching HTML:', error)
      alert('Failed to load document')
    } finally {
      setLoadingHtml(false)
    }
  }

  const checkExistingExtraction = async () => {
    try {
      const response = await fetch(`/api/html/extract-highlights?htmlUrl=${encodeURIComponent(url)}`)
      if (response.ok) {
        const data = await response.json()
        if (data.highlights && data.highlights.highlights) {
          setHighlights(data.highlights.highlights)
          setExtractedData(data.highlights.extractedData)
          setHasExistingData(true)
          console.log('✅ Loaded existing extraction:', data.highlights)
        }
      }
    } catch (error) {
      console.error('Error checking existing extraction:', error)
    }
  }

  const handleExtractData = async () => {
    if (!projectId) {
      alert('Project ID is required for extraction')
      return
    }

    setExtracting(true)
    try {
      console.log('🚀 Starting extraction for:', url)
      const response = await fetch('/api/html/extract-highlights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          htmlUrl: url,
          projectId: projectId,
        }),
      })

      console.log('📡 Response status:', response.status)
      const responseText = await response.text()
      console.log('📄 Response body:', responseText.substring(0, 500))

      if (!response.ok) {
        throw new Error(`Extraction failed: ${response.status} - ${responseText}`)
      }

      const data = JSON.parse(responseText)
      console.log('✅ Extraction complete:', data)

      if (data.highlights) {
        setHighlights(data.highlights)
        setExtractedData(data.extractedData)
        setHasExistingData(true)

        // Notify parent that project was updated
        if (data.projectUpdated && onProjectUpdated) {
          onProjectUpdated()
        }

        // Apply highlights to iframe
        setTimeout(() => applyHighlights(data.highlights), 1000)
      }
    } catch (error: any) {
      console.error('❌ Error extracting data:', error)
      alert(`Failed to extract data: ${error.message}`)
    } finally {
      setExtracting(false)
    }
  }

  const applyHighlights = (highlightsData: ExtractedHighlight[]) => {
    if (!iframeRef.current) return

    try {
      const iframeDoc = iframeRef.current.contentDocument
      if (!iframeDoc) {
        console.log('⚠️  Cannot access iframe document')
        return
      }

      console.log('🎨 Applying highlights to iframe...')

      // Inject CSS for highlights
      const existingStyle = iframeDoc.getElementById('lithos-highlight-style')
      if (existingStyle) {
        existingStyle.remove()
      }

      const style = iframeDoc.createElement('style')
      style.id = 'lithos-highlight-style'
      style.textContent = `
        .lithos-highlight {
          background-color: yellow !important;
          padding: 2px 4px;
          border-radius: 2px;
          cursor: pointer;
        }
        .lithos-highlight:hover {
          background-color: #ffeb3b !important;
          box-shadow: 0 0 0 2px rgba(255, 193, 7, 0.5);
        }
        .lithos-highlight-npv { border-left: 3px solid #4caf50; }
        .lithos-highlight-irr { border-left: 3px solid #2196f3; }
        .lithos-highlight-capex { border-left: 3px solid #ff9800; }
        .lithos-highlight-companyName { border-left: 3px solid #9c27b0; }
        .lithos-highlight-location { border-left: 3px solid #f44336; }
      `
      iframeDoc.head.appendChild(style)

      // Apply highlights
      let highlightsApplied = 0
      highlightsData.forEach(highlight => {
        if (highlight.highlightArea?.elementId) {
          // Highlight by element ID
          const element = iframeDoc.getElementById(highlight.highlightArea.elementId)
          if (element) {
            element.classList.add('lithos-highlight', `lithos-highlight-${highlight.dataType}`)
            element.setAttribute('data-highlight-id', highlight.id)
            highlightsApplied++
          } else {
            console.log(`⚠️  Element not found: ${highlight.highlightArea.elementId}`)
          }
        } else if (highlight.highlightArea?.sectionId) {
          // Highlight by section ID
          const section = iframeDoc.getElementById(highlight.highlightArea.sectionId)
          if (section) {
            section.classList.add('lithos-highlight', `lithos-highlight-${highlight.dataType}`)
            section.setAttribute('data-highlight-id', highlight.id)
            highlightsApplied++
          } else {
            console.log(`⚠️  Section not found: ${highlight.highlightArea.sectionId}`)
          }
        }
      })

      console.log(`✅ Applied ${highlightsApplied} / ${highlightsData.length} highlights`)
    } catch (error) {
      console.error('Error applying highlights:', error)
    }
  }

  const scrollToHighlight = (highlight: ExtractedHighlight) => {
    if (!iframeRef.current) return

    try {
      const iframeDoc = iframeRef.current.contentDocument
      if (!iframeDoc) return

      let targetElement: HTMLElement | null = null

      if (highlight.highlightArea?.elementId) {
        targetElement = iframeDoc.getElementById(highlight.highlightArea.elementId)
      } else if (highlight.highlightArea?.sectionId) {
        // Check if sectionId is in "Table X" format
        const tableMatch = highlight.highlightArea.sectionId.match(/Table (\d+)/)
        if (tableMatch) {
          const tableNum = tableMatch[1]
          targetElement = iframeDoc.getElementById(`lithos-table-${tableNum}`)
          console.log(`📍 Looking for table: lithos-table-${tableNum}`, targetElement ? 'Found!' : 'Not found')
        } else {
          targetElement = iframeDoc.getElementById(highlight.highlightArea.sectionId)
        }
      }

      if (targetElement) {
        targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' })

        // Flash highlight
        targetElement.style.transition = 'all 0.3s'
        targetElement.style.backgroundColor = '#ffeb3b'
        setTimeout(() => {
          targetElement!.style.backgroundColor = ''
        }, 1000)
      } else {
        console.log('⚠️  Target element not found for highlight:', highlight)
      }
    } catch (error) {
      console.error('Error scrolling to highlight:', error)
    }
  }

  const getDataTypeLabel = (dataType: string) => {
    const labels: Record<string, string> = {
      npv: 'NPV',
      irr: 'IRR',
      capex: 'CAPEX',
      opex: 'OPEX',
      resources: 'Resources',
      reserves: 'Reserves',
      production: 'Production',
      commodities: 'Commodities',
      companyName: 'Company',
      location: 'Location'
    }
    return labels[dataType] || dataType
  }

  const getDataTypeColor = (dataType: string) => {
    const colors: Record<string, string> = {
      npv: 'bg-green-100 text-green-800',
      irr: 'bg-blue-100 text-blue-800',
      capex: 'bg-orange-100 text-orange-800',
      opex: 'bg-yellow-100 text-yellow-800',
      resources: 'bg-purple-100 text-purple-800',
      reserves: 'bg-indigo-100 text-indigo-800',
      production: 'bg-pink-100 text-pink-800',
      commodities: 'bg-teal-100 text-teal-800',
      companyName: 'bg-violet-100 text-violet-800',
      location: 'bg-red-100 text-red-800'
    }
    return colors[dataType] || 'bg-gray-100 text-gray-800'
  }

  const handleIframeLoad = () => {
    console.log('🎬 Iframe loaded')

    // Enable text selection and add table IDs in iframe
    try {
      const iframeDoc = iframeRef.current?.contentDocument
      if (iframeDoc) {
        // Add unique IDs to all tables for navigation
        const tables = iframeDoc.querySelectorAll('table')
        tables.forEach((table, index) => {
          if (!table.id) {
            table.id = `lithos-table-${index + 1}`
          }
        })
        console.log(`✅ Added IDs to ${tables.length} tables`)

        const style = iframeDoc.createElement('style')
        style.textContent = `
          * {
            user-select: text !important;
            -webkit-user-select: text !important;
            -moz-user-select: text !important;
            -ms-user-select: text !important;
          }
          ::selection {
            background-color: #b3d9ff;
            color: #000;
          }
          ::-moz-selection {
            background-color: #b3d9ff;
            color: #000;
          }
        `
        iframeDoc.head.appendChild(style)
      }
    } catch (error) {
      console.error('Error enabling text selection:', error)
    }

    if (hasExistingData && highlights.length > 0) {
      // Apply highlights after iframe loads
      setTimeout(() => applyHighlights(highlights), 1000)
    }
  }

  const handleDownload = () => {
    // Create a blob and download
    const blob = new Blob([htmlContent], { type: 'text/html' })
    const downloadUrl = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = downloadUrl
    a.download = `${title || 'document'}.html`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(downloadUrl)
  }

  if (loadingHtml) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-10 w-10 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading HTML document...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex bg-background">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b p-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold">{title || "HTML Document"}</h2>
            <Badge variant="outline" className="bg-yellow-100 text-yellow-800">
              FactSet Filing
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            {projectId && !hasExistingData && (
              <Button
                variant="default"
                size="sm"
                onClick={handleExtractData}
                disabled={extracting}
              >
                {extracting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Extracting...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Extract Key Data
                  </>
                )}
              </Button>
            )}
            {projectId && hasExistingData && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleExtractData}
                disabled={extracting}
              >
                {extracting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Re-extracting...
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="h-4 w-4 mr-2 text-green-600" />
                    Data Extracted
                  </>
                )}
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
            >
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open(url, '_blank')}
            >
              <ExternalLink className="h-4 w-4 mr-2" />
              Open in New Tab
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Document Content */}
        <div className="flex-1 overflow-hidden">
          <iframe
            ref={iframeRef}
            srcDoc={htmlContent}
            className="w-full h-full border-0"
            sandbox="allow-same-origin allow-scripts"
            title={title || "Document Viewer"}
            onLoad={handleIframeLoad}
            style={{
              transform: 'scale(1.25)',
              transformOrigin: 'top left',
              width: '80%',
              height: '80%'
            }}
          />
        </div>

        {/* Info Footer */}
        {!hasExistingData && (
          <div className="border-t p-3 bg-muted/50">
            <p className="text-xs text-muted-foreground">
              📄 This is a FactSet SEC/SEDAR filing. Click "Extract Key Data" to automatically extract financial metrics (NPV, IRR, CAPEX) and company information.
            </p>
          </div>
        )}
      </div>

      {/* Sidebar - Extracted Data */}
      {hasExistingData && highlights.length > 0 && (
        <div className="w-80 border-l bg-muted/50 flex flex-col">
          <div className="p-4 border-b">
            <h3 className="font-semibold text-sm">Extracted Key Data</h3>
            <p className="text-xs text-muted-foreground mt-1">
              Click any field to navigate to its location in the document
            </p>
          </div>

          <ScrollArea className="flex-1 p-4">
            <div className="space-y-3">
              {highlights.map((highlight) => (
                <Card
                  key={highlight.id}
                  className="p-3 cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => scrollToHighlight(highlight)}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <Badge className={getDataTypeColor(highlight.dataType)}>
                      {getDataTypeLabel(highlight.dataType)}
                    </Badge>
                    <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  </div>

                  {highlight.value !== undefined && (
                    <div className="font-semibold text-sm mb-1">
                      {Array.isArray(highlight.value)
                        ? highlight.value.join(', ')
                        : typeof highlight.value === 'number'
                        ? highlight.value.toLocaleString()
                        : highlight.value}
                    </div>
                  )}

                  <div className="text-xs text-muted-foreground line-clamp-2">
                    {highlight.content}
                  </div>

                  {highlight.highlightArea?.sectionId && (
                    <div className="text-xs text-blue-600 mt-1">
                      Section: {highlight.highlightArea.sectionId}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          </ScrollArea>

          <Separator />

          <div className="p-4">
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={handleExtractData}
              disabled={extracting}
            >
              {extracting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Re-extracting...
                </>
              ) : (
                'Re-extract Data'
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
