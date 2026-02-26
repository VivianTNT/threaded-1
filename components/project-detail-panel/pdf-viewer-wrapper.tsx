"use client"

import dynamic from "next/dynamic"
import * as React from "react"

// Dynamically import PDFViewer to prevent SSR issues
const PDFViewer = dynamic(
  () => import("./pdf-viewer").then((mod) => ({ default: mod.PDFViewer })),
  {
    ssr: false,
    loading: () => (
      <div className="fixed inset-0 z-50 bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
          <p className="text-sm text-muted-foreground">Loading PDF viewer...</p>
        </div>
      </div>
    ),
  }
)

interface PDFViewerWrapperProps {
  url: string
  title?: string
  onClose: () => void
}

export function PDFViewerWrapper({ url, title, onClose }: PDFViewerWrapperProps) {
  return <PDFViewer url={url} title={title} onClose={onClose} />
}
