"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Download, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface PDFViewerProps {
  url: string
  title?: string
  onClose: () => void
}

export function PDFViewer({ url, title, onClose }: PDFViewerProps) {
  const [scale, setScale] = React.useState<number>(100)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)

  function zoomIn() {
    setScale((prevScale) => Math.min(prevScale + 10, 200))
  }

  function zoomOut() {
    setScale((prevScale) => Math.max(prevScale - 10, 50))
  }

  function handleDownload() {
    const link = document.createElement("a")
    link.href = url
    link.download = title || "document.pdf"
    link.target = "_blank"
    link.click()
  }

  return (
    <div className="fixed inset-0 z-50 bg-background flex flex-col">
      {/* Header Controls */}
      <div className="bg-background border-b px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-9 w-9"
            >
              <X className="h-5 w-5" />
            </Button>
            <h2 className="font-semibold text-base truncate max-w-md">
              {title || "PDF Document"}
            </h2>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            Download
          </Button>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={zoomOut}
            disabled={scale <= 50}
            className="h-8 w-8 p-0"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-sm font-medium min-w-[50px] text-center">
            {scale}%
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={zoomIn}
            disabled={scale >= 200}
            className="h-8 w-8 p-0"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* PDF Content using iframe */}
      <div className="flex-1 overflow-auto bg-background">
        <iframe
          ref={iframeRef}
          src={url}
          className="w-full h-full border-0"
          title={title || "PDF Document"}
          style={{ minHeight: '100%' }}
        />
      </div>
    </div>
  )
}
