"use client"

import * as React from "react"
import { Sparkles, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { toast } from "sonner"

interface FashionAgentButtonProps {
  onAgentRun?: () => void
}

export function FashionAgentButton({ onAgentRun }: FashionAgentButtonProps) {
  const [isRunning, setIsRunning] = React.useState(false)
  const [progress, setProgress] = React.useState(0)

  const startAgent = async () => {
    setIsRunning(true)
    setProgress(0)

    toast.info('Fashion Agent started', {
      description: 'Discovering new items for you...'
    })

    // Simulate agent progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsRunning(false)
          toast.success('Fashion Agent completed!', {
            description: '47 new items found matching your style'
          })
          onAgentRun?.()
          return 100
        }
        return prev + 10
      })
    }, 800)
  }

  return (
    <div className="space-y-2">
      <Button
        onClick={startAgent}
        disabled={isRunning}
        className="w-full bg-gradient-to-r from-primary to-primary/80"
        size="lg"
      >
        {isRunning ? (
          <>
            <Loader2 className="h-5 w-5 mr-2 animate-spin" />
            Finding items...
          </>
        ) : (
          <>
            <Sparkles className="h-5 w-5 mr-2" />
            Run Fashion Agent
          </>
        )}
      </Button>
      {isRunning && (
        <div className="space-y-1">
          <Progress value={progress} className="h-2" />
          <p className="text-xs text-muted-foreground text-center">
            Scanning retailers and matching your style... {progress}%
          </p>
        </div>
      )}
    </div>
  )
}
