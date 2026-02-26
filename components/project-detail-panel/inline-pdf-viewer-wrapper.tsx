"use client"

import dynamic from "next/dynamic"
import * as React from "react"
import { Button } from "@/components/ui/button"
import { X, ExternalLink } from "lucide-react"

// Dynamically import InlinePDFViewer to prevent SSR issues
const InlinePDFViewer = dynamic(
  () => import("./inline-pdf-viewer").then((mod) => ({ default: mod.InlinePDFViewer })),
  {
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center gap-3 py-12">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
        <p className="text-xs text-muted-foreground">Loading viewer...</p>
      </div>
    ),
  }
)

// Dynamically import InlineHTMLViewer for HTML documents
const InlineHTMLViewer = dynamic(
  () => import("./inline-html-viewer").then((mod) => ({ default: mod.InlineHTMLViewer })),
  {
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center gap-3 py-12">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
        <p className="text-xs text-muted-foreground">Loading HTML viewer...</p>
      </div>
    ),
  }
)

interface InlinePDFViewerWrapperProps {
  url: string
  title?: string
  onClose: () => void
  projectId?: string | null
  onProjectUpdated?: () => void
}

export function InlinePDFViewerWrapper({ url, title, onClose, projectId, onProjectUpdated }: InlinePDFViewerWrapperProps) {
  // Detect if this is an HTML document (FactSet filing) or PDF
  const isHtmlDocument = url.endsWith('.html') || url.includes('.html')

  if (isHtmlDocument) {
    // Render HTML document with enhanced viewer
    return <InlineHTMLViewer url={url} title={title} onClose={onClose} projectId={projectId} onProjectUpdated={onProjectUpdated} />
  }

  // Render PDF with full PDF viewer
  return <InlinePDFViewer url={url} title={title} onClose={onClose} projectId={projectId} onProjectUpdated={onProjectUpdated} />
}
