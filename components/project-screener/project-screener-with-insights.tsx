"use client"

import * as React from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ProjectScreener } from "./project-screener"
import { AIInsightsTab } from "../ai-insights-tab"
import { useWatchlistProjects } from "@/lib/hooks/use-watchlist-projects"
import { Sparkles, Table } from "lucide-react"

export function ProjectScreenerWithInsights() {
  const { projects, loading, error } = useWatchlistProjects()

  return (
    <Tabs defaultValue="table" className="w-full">
      <TabsList className="grid w-full grid-cols-2 mb-4">
        <TabsTrigger value="table" className="flex items-center gap-2">
          <Table className="h-4 w-4" />
          Projects Table
        </TabsTrigger>
        <TabsTrigger value="ai-insights" className="flex items-center gap-2">
          <Sparkles className="h-4 w-4" />
          AI Insights
        </TabsTrigger>
      </TabsList>

      <TabsContent value="table" className="mt-0">
        <ProjectScreener />
      </TabsContent>

      <TabsContent value="ai-insights" className="mt-0">
        <AIInsightsTab projects={projects} loading={loading} />
      </TabsContent>
    </Tabs>
  )
}
