"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import {
  Sparkles,
  RefreshCw,
  Search,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Filter
} from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import { MiningProject } from "@/lib/types/mining-project"
import { AIInsightsPanel } from "./ai-insights-panel"

interface AIInsight {
  id: string
  project_id: string
  overall_risk_score: number
  investment_recommendation: string
  generated_at: string
  expires_at: string
}

interface AIInsightsTabProps {
  projects: MiningProject[]
  loading?: boolean
  className?: string
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

export function AIInsightsTab({ projects, loading: projectsLoading, className }: AIInsightsTabProps) {
  const [insights, setInsights] = useState<Record<string, AIInsight>>({})
  const [loadingInsights, setLoadingInsights] = useState(false)
  const [generatingAll, setGeneratingAll] = useState(false)
  const [selectedProject, setSelectedProject] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [filterRecommendation, setFilterRecommendation] = useState<string>("all")
  const [sortBy, setSortBy] = useState<string>("risk-asc")

  useEffect(() => {
    if (projects.length > 0) {
      fetchExistingInsights()
    }
  }, [projects])

  const fetchExistingInsights = async () => {
    try {
      setLoadingInsights(true)
      const projectIds = projects.map(p => p.id).join(',')

      const response = await fetch(`/api/ai-insights?projectIds=${projectIds}`)

      if (!response.ok) {
        console.error('Failed to fetch AI insights:', response.status)
        return
      }

      const data = await response.json()

      if (data.insights) {
        const insightsMap: Record<string, AIInsight> = {}
        data.insights.forEach((insight: AIInsight) => {
          insightsMap[insight.project_id] = insight
        })
        setInsights(insightsMap)
      }
    } catch (error) {
      console.error('Error fetching insights:', error)
    } finally {
      setLoadingInsights(false)
    }
  }

  const generateAllInsights = async () => {
    try {
      setGeneratingAll(true)
      const projectIds = projects.map(p => p.id)

      const response = await fetch('/api/ai-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectIds,
          forceRegenerate: false,
          useWebSearch: false
        })
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('AI insights batch generation error:', response.status, errorText)
        throw new Error(`Failed to generate insights: ${response.status}`)
      }

      const data = await response.json()

      if (data.insights) {
        const insightsMap: Record<string, AIInsight> = {}
        data.insights.forEach((insight: AIInsight) => {
          insightsMap[insight.project_id] = insight
        })
        setInsights(insightsMap)

        toast.success(`Generated insights for ${data.successful} projects`, {
          description: data.failed > 0 ? `${data.failed} failed` : undefined
        })
      }
    } catch (error: any) {
      console.error('Error generating insights:', error)
      toast.error('Failed to generate insights', {
        description: error.message
      })
    } finally {
      setGeneratingAll(false)
    }
  }

  // Filter and sort projects
  const filteredProjects = projects.filter(project => {
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      if (
        !project.name.toLowerCase().includes(query) &&
        !project.location?.toLowerCase().includes(query) &&
        !project.commodities?.some(c => c.toLowerCase().includes(query))
      ) {
        return false
      }
    }

    // Recommendation filter
    if (filterRecommendation !== "all") {
      const insight = insights[project.id]
      if (!insight || insight.investment_recommendation !== filterRecommendation) {
        return false
      }
    }

    return true
  })

  // Sort projects
  const sortedProjects = [...filteredProjects].sort((a, b) => {
    const insightA = insights[a.id]
    const insightB = insights[b.id]

    switch (sortBy) {
      case "risk-asc":
        if (!insightA) return 1
        if (!insightB) return -1
        return insightA.overall_risk_score - insightB.overall_risk_score
      case "risk-desc":
        if (!insightA) return 1
        if (!insightB) return -1
        return insightB.overall_risk_score - insightA.overall_risk_score
      case "name":
        return a.name.localeCompare(b.name)
      default:
        return 0
    }
  })

  const insightsCount = Object.keys(insights).length
  const projectsWithInsights = projects.filter(p => insights[p.id])
  const strongBuyCount = projectsWithInsights.filter(p => insights[p.id]?.investment_recommendation === "Strong Buy").length
  const buyCount = projectsWithInsights.filter(p => insights[p.id]?.investment_recommendation === "Buy").length
  const holdCount = projectsWithInsights.filter(p => insights[p.id]?.investment_recommendation === "Hold").length
  const passCount = projectsWithInsights.filter(p => insights[p.id]?.investment_recommendation === "Pass").length

  if (projectsLoading) {
    return (
      <div className={cn("space-y-4", className)}>
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header with Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>Total Projects</CardDescription>
            <CardTitle className="text-3xl">{projects.length}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>Insights Generated</CardDescription>
            <CardTitle className="text-3xl">{insightsCount}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>
              <span className="text-green-600 font-semibold">Strong Buy</span>
            </CardDescription>
            <CardTitle className="text-3xl text-green-600">{strongBuyCount}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>
              <span className="text-blue-600 font-semibold">Buy</span>
            </CardDescription>
            <CardTitle className="text-3xl text-blue-600">{buyCount}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>
              <span className="text-yellow-600 font-semibold">Hold</span> / <span className="text-red-600 font-semibold">Pass</span>
            </CardDescription>
            <CardTitle className="text-3xl">{holdCount + passCount}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-purple-600" />
              <CardTitle>AI Insights</CardTitle>
            </div>
            <Button
              onClick={generateAllInsights}
              disabled={generatingAll || projects.length === 0}
              size="sm"
            >
              {generatingAll ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate All ({projects.length - insightsCount} remaining)
                </>
              )}
            </Button>
          </div>
          <CardDescription>
            AI-powered risk analysis for {projects.length} projects
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Filters */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search projects..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={filterRecommendation} onValueChange={setFilterRecommendation}>
              <SelectTrigger className="w-full md:w-[200px]">
                <Filter className="mr-2 h-4 w-4" />
                <SelectValue placeholder="Filter by recommendation" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Recommendations</SelectItem>
                <SelectItem value="Strong Buy">Strong Buy</SelectItem>
                <SelectItem value="Buy">Buy</SelectItem>
                <SelectItem value="Hold">Hold</SelectItem>
                <SelectItem value="Pass">Pass</SelectItem>
              </SelectContent>
            </Select>
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-full md:w-[200px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="risk-asc">Risk: Low to High</SelectItem>
                <SelectItem value="risk-desc">Risk: High to Low</SelectItem>
                <SelectItem value="name">Name: A to Z</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Project List with Insights */}
      {sortedProjects.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No projects match your filters</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {sortedProjects.map(project => {
            const insight = insights[project.id]
            return (
              <Card key={project.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-lg">{project.name}</CardTitle>
                      <CardDescription>
                        {project.location} • {Array.isArray(project.commodities) ? project.commodities.join(', ') : 'N/A'}
                      </CardDescription>
                    </div>
                    {insight && (
                      <Badge className={cn("text-xs", getRecommendationColor(insight.investment_recommendation))}>
                        {insight.investment_recommendation}
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <AIInsightsPanel
                    projectId={project.id}
                    projectName={project.name}
                    autoGenerate={false}
                  />
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}
