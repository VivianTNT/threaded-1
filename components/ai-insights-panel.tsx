"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  AlertCircle,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Sparkles,
  MapPin,
  Scale,
  Package,
  Users,
  CheckCircle2,
  XCircle,
  Clock
} from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

// Parse markdown-style links [text](url) and render as clickable anchors
function parseMarkdownLinks(text: string) {
  const parts: (string | JSX.Element)[] = []
  let lastIndex = 0

  // Regex to match [text](url) format
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g
  let match

  while ((match = linkRegex.exec(text)) !== null) {
    // Add text before the link
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index))
    }

    // Add the clickable link
    const linkText = match[1]
    const url = match[2]
    parts.push(
      <a
        key={match.index}
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 dark:text-blue-400 hover:underline"
      >
        {linkText}
      </a>
    )

    lastIndex = match.index + match[0].length
  }

  // Add remaining text after the last link
  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex))
  }

  return parts.length > 0 ? parts : text
}

interface AIInsight {
  id: string
  project_id: string
  geography_risk_score: number
  geography_risk_analysis: string
  legal_risk_score: number
  legal_risk_analysis: string
  commodity_risk_score: number
  commodity_risk_analysis: string
  team_risk_score: number
  team_risk_analysis: string
  overall_risk_score: number
  risk_summary: string
  key_opportunities: string[]
  key_threats: string[]
  investment_recommendation: string
  recommendation_rationale: string
  generated_at: string
  expires_at: string
  generation_time_ms: number
  from_cache?: boolean
}

interface AIInsightsPanelProps {
  projectId: string
  projectName?: string
  autoGenerate?: boolean
  className?: string
}

function getRiskColor(score: number) {
  if (score <= 3) return "text-green-600 bg-green-50 border-green-200"
  if (score <= 5) return "text-yellow-600 bg-yellow-50 border-yellow-200"
  if (score <= 7) return "text-orange-600 bg-orange-50 border-orange-200"
  return "text-red-600 bg-red-50 border-red-200"
}

function getRiskLabel(score: number) {
  if (score <= 3) return "Low Risk"
  if (score <= 5) return "Moderate Risk"
  if (score <= 7) return "High Risk"
  return "Very High Risk"
}

function getRecommendationColor(recommendation: string) {
  switch (recommendation) {
    case "Strong Buy":
      return "bg-green-100 text-green-800 border-green-300"
    case "Buy":
      return "bg-blue-100 text-blue-800 border-blue-300"
    case "Hold":
      return "bg-yellow-100 text-yellow-800 border-yellow-300"
    case "Pass":
      return "bg-red-100 text-red-800 border-red-300"
    default:
      return "bg-gray-100 text-gray-800 border-gray-300"
  }
}

export function AIInsightsPanel({
  projectId,
  projectName,
  autoGenerate = true,
  className
}: AIInsightsPanelProps) {
  const [insight, setInsight] = useState<AIInsight | null>(null)
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (projectId) {
      // Clear previous insight when projectId changes to show skeleton
      setInsight(null)
      setError(null)
      fetchInsight()
    }
  }, [projectId])

  const fetchInsight = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch(`/api/ai-insights?projectId=${projectId}`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch insights')
      }

      if (data.insight) {
        setInsight(data.insight)
      } else if (autoGenerate) {
        // No cached insight found, generate new one
        await generateInsight()
      }
    } catch (err: any) {
      console.error('Error fetching insight:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const generateInsight = async (forceRegenerate: boolean = false) => {
    try {
      setGenerating(true)
      setError(null)

      const response = await fetch('/api/ai-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId,
          forceRegenerate,
          useWebSearch: false // Can be enabled for enhanced analysis
        })
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('AI insights API error:', response.status, errorText)
        throw new Error(`Failed to generate insights: ${response.status}`)
      }

      const data = await response.json()

      setInsight(data.insight)

      if (data.insight.from_cache) {
        toast.info('Loaded cached AI insights')
      } else {
        toast.success('AI insights generated successfully', {
          description: `Analysis completed in ${(data.insight.generation_time_ms / 1000).toFixed(1)}s`
        })
      }
    } catch (err: any) {
      console.error('Error generating insight:', err)
      setError(err.message)
      toast.error('Failed to generate AI insights', {
        description: err.message
      })
    } finally {
      setGenerating(false)
    }
  }

  if ((loading || generating) && !insight) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-600 animate-pulse" />
            <CardTitle>AI Risk Analysis</CardTitle>
          </div>
          <CardDescription>
            {generating ? 'Generating AI insights...' : 'Loading insights...'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Overall Risk Score Skeleton */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-6 w-24" />
            </div>
            <Skeleton className="h-2 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>

          {/* Investment Recommendation Skeleton */}
          <Skeleton className="h-24 w-full rounded-lg" />

          {/* Risk Categories Tabs Skeleton */}
          <div className="space-y-4">
            <Skeleton className="h-10 w-full" />
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-1.5 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
          </div>

          {/* Opportunities and Threats Skeleton */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-16 w-full" />
            </div>
            <div className="space-y-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-16 w-full" />
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error && !insight) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-600" />
            <CardTitle>AI Risk Analysis</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <Button onClick={() => generateInsight()} disabled={generating}>
              {generating ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate AI Insights
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!insight) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-600" />
            <CardTitle>AI Risk Analysis</CardTitle>
          </div>
          <CardDescription>
            Get AI-powered risk analysis for {projectName || 'this project'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Sparkles className="h-12 w-12 text-purple-600 mb-4" />
            <p className="text-sm text-muted-foreground mb-4">
              No insights available yet. Generate comprehensive risk analysis powered by AI.
            </p>
            <Button onClick={() => generateInsight()} disabled={generating}>
              {generating ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate AI Insights
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-600" />
            <CardTitle>AI Risk Analysis</CardTitle>
            {insight.from_cache && (
              <Badge variant="outline" className="text-xs">
                <Clock className="mr-1 h-3 w-3" />
                Cached
              </Badge>
            )}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => generateInsight(true)}
            disabled={generating}
          >
            {generating ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
        </div>
        <CardDescription>
          Generated {new Date(insight.generated_at).toLocaleDateString()} •
          Expires {new Date(insight.expires_at).toLocaleDateString()}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overall Risk Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Overall Risk Score</h3>
            <Badge className={cn("text-sm", getRiskColor(insight.overall_risk_score))}>
              {insight.overall_risk_score.toFixed(1)} / 10 - {getRiskLabel(insight.overall_risk_score)}
            </Badge>
          </div>
          <Progress value={insight.overall_risk_score * 10} className="h-2" />
          <p className="text-sm text-muted-foreground">{parseMarkdownLinks(insight.risk_summary)}</p>
        </div>

        {/* Investment Recommendation */}
        <div className={cn("p-4 rounded-lg border-2", getRecommendationColor(insight.investment_recommendation))}>
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-semibold">Investment Recommendation</h3>
            <Badge variant="outline" className="font-bold">
              {insight.investment_recommendation}
            </Badge>
          </div>
          <p className="text-sm">{parseMarkdownLinks(insight.recommendation_rationale)}</p>
        </div>

        {/* Risk Categories Tabs */}
        <Tabs defaultValue="geography" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="geography" className="text-xs">
              <MapPin className="mr-1 h-3 w-3" />
              Geography
            </TabsTrigger>
            <TabsTrigger value="legal" className="text-xs">
              <Scale className="mr-1 h-3 w-3" />
              Legal
            </TabsTrigger>
            <TabsTrigger value="commodity" className="text-xs">
              <Package className="mr-1 h-3 w-3" />
              Commodity
            </TabsTrigger>
            <TabsTrigger value="team" className="text-xs">
              <Users className="mr-1 h-3 w-3" />
              Team
            </TabsTrigger>
          </TabsList>

          <TabsContent value="geography" className="space-y-2 mt-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Geographic Risk</h4>
              <Badge className={cn("text-xs", getRiskColor(insight.geography_risk_score))}>
                {insight.geography_risk_score.toFixed(1)} / 10
              </Badge>
            </div>
            <Progress value={insight.geography_risk_score * 10} className="h-1.5" />
            <p className="text-sm text-muted-foreground">{parseMarkdownLinks(insight.geography_risk_analysis)}</p>
          </TabsContent>

          <TabsContent value="legal" className="space-y-2 mt-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Legal & Regulatory Risk</h4>
              <Badge className={cn("text-xs", getRiskColor(insight.legal_risk_score))}>
                {insight.legal_risk_score.toFixed(1)} / 10
              </Badge>
            </div>
            <Progress value={insight.legal_risk_score * 10} className="h-1.5" />
            <p className="text-sm text-muted-foreground">{parseMarkdownLinks(insight.legal_risk_analysis)}</p>
          </TabsContent>

          <TabsContent value="commodity" className="space-y-2 mt-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Commodity Market Risk</h4>
              <Badge className={cn("text-xs", getRiskColor(insight.commodity_risk_score))}>
                {insight.commodity_risk_score.toFixed(1)} / 10
              </Badge>
            </div>
            <Progress value={insight.commodity_risk_score * 10} className="h-1.5" />
            <p className="text-sm text-muted-foreground">{parseMarkdownLinks(insight.commodity_risk_analysis)}</p>
          </TabsContent>

          <TabsContent value="team" className="space-y-2 mt-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Management Team Risk</h4>
              <Badge className={cn("text-xs", getRiskColor(insight.team_risk_score))}>
                {insight.team_risk_score.toFixed(1)} / 10
              </Badge>
            </div>
            <Progress value={insight.team_risk_score * 10} className="h-1.5" />
            <p className="text-sm text-muted-foreground">{parseMarkdownLinks(insight.team_risk_analysis)}</p>
          </TabsContent>
        </Tabs>

        {/* Opportunities and Threats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-green-600" />
              <h4 className="text-sm font-medium">Key Opportunities</h4>
            </div>
            <ul className="space-y-1.5">
              {insight.key_opportunities.map((opp, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                  <span className="text-muted-foreground">{parseMarkdownLinks(opp)}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-red-600" />
              <h4 className="text-sm font-medium">Key Threats</h4>
            </div>
            <ul className="space-y-1.5">
              {insight.key_threats.map((threat, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <XCircle className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
                  <span className="text-muted-foreground">{parseMarkdownLinks(threat)}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
