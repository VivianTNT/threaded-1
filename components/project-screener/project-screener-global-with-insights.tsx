"use client"

import * as React from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ProjectScreenerGlobal } from "./project-screener-global"
import { AIInsightsTab } from "../ai-insights-tab"
import { useProjects } from "@/lib/hooks/use-projects"
import { Sparkles, Table } from "lucide-react"

export function ProjectScreenerGlobalWithInsights() {
  const { projects, loading, error } = useProjects()

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
        <ProjectScreenerGlobal />
      </TabsContent>

      <TabsContent value="ai-insights" className="mt-0">
        <AIInsightsTab projects={projects} loading={loading} />
      </TabsContent>
    </Tabs>
  )
}
