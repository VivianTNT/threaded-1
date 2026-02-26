'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppSidebar } from '@/components/app-sidebar'
import { SiteHeader } from '@/components/site-header'
import { ChatLayout } from '@/components/chat-layout'
import { useRequireAuth } from '@/lib/auth-utils'
import {
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar'
import { ProjectDetailPanel } from '@/components/project-detail-panel'
import { supabase } from '@/lib/supabase/client'
import { MiningProject } from '@/lib/types/mining-project'
import { Loader2, ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function ProjectDetailPage() {
  const { user, isLoading: authLoading } = useRequireAuth()
  const params = useParams()
  const router = useRouter()
  const projectId = params.id as string

  const [project, setProject] = useState<MiningProject | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!projectId) return

    fetchProjectDetails()
  }, [projectId])

  const fetchProjectDetails = async () => {
    try {
      setLoading(true)
      setError(null)

      // Fetch project
      const { data: projectData, error: projectError } = await supabase
        .from('projects')
        .select('*')
        .eq('id', projectId)
        .single()

      if (projectError) throw projectError

      if (!projectData) {
        setError('Project not found')
        return
      }

      // Fetch company if project has company_id
      let companyName = 'Unknown'
      if (projectData.company_id) {
        const { data: companyData } = await supabase
          .from('companies')
          .select('name')
          .eq('id', projectData.company_id)
          .single()

        if (companyData) {
          companyName = companyData.name
        }
      }

      // Transform to MiningProject type
      const transformedProject: MiningProject = {
        // Database fields
        id: projectData.id,
        company_id: projectData.company_id,
        name: projectData.name,
        location: projectData.location,
        stage: projectData.stage,
        commodities: projectData.commodities,
        resource_estimate: projectData.resource_estimate,
        reserve_estimate: projectData.reserve_estimate,
        ownership_percentage: projectData.ownership_percentage,
        status: projectData.status,
        description: projectData.description,
        urls: projectData.urls,
        watchlist: projectData.watchlist || false,
        created_at: projectData.created_at,
        updated_at: projectData.updated_at,

        // User upload fields
        user_id: projectData.user_id,
        is_private: projectData.is_private || false,
        uploaded_at: projectData.uploaded_at,
        document_storage_path: projectData.document_storage_path,

        // Financial metrics
        npv: projectData.npv,
        irr: projectData.irr,
        capex: projectData.capex,

        // Computed/display fields
        project: projectData.name,
        company: companyName,
        primaryCommodity: projectData.commodities?.[0] || 'Unknown',
        jurisdiction: projectData.location || 'Unknown',
        riskLevel: 'Medium' as const,

        // Optional fields
        project_id: projectData.id,
        watchlisted_at: projectData.watchlisted_at,
        technicalReportUrl: projectData.urls?.[0],
      }

      setProject(transformedProject)
    } catch (err) {
      console.error('Error fetching project:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch project')
    } finally {
      setLoading(false)
    }
  }

  if (authLoading || loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!user) {
    return null // Will redirect to login
  }

  if (error) {
    return (
      <ChatLayout>
        <SidebarProvider
          style={
            {
              "--sidebar-width": "calc(var(--spacing) * 72)",
              "--header-height": "calc(var(--spacing) * 12)",
            } as React.CSSProperties
          }
        >
          <AppSidebar variant="inset" />
          <SidebarInset>
            <SiteHeader />
            <div className="flex flex-1 flex-col items-center justify-center gap-4">
              <div className="text-lg text-muted-foreground">{error}</div>
              <Button onClick={() => router.push('/global-projects')} variant="outline">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Projects
              </Button>
            </div>
          </SidebarInset>
        </SidebarProvider>
      </ChatLayout>
    )
  }

  return (
    <ChatLayout>
      <SidebarProvider
        style={
          {
            "--sidebar-width": "calc(var(--spacing) * 72)",
            "--header-height": "calc(var(--spacing) * 12)",
          } as React.CSSProperties
        }
      >
        <AppSidebar variant="inset" />
        <SidebarInset>
          <SiteHeader />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
                <div className="px-4 lg:px-6">
                  <Button
                    onClick={() => router.push('/global-projects')}
                    variant="ghost"
                    className="mb-4"
                  >
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back to Projects
                  </Button>

                  {project && (
                    <ProjectDetailPanel
                      isOpen={true}
                      onClose={() => router.push('/global-projects')}
                      projects={[project]}
                      mode="single"
                      onProjectSelect={() => {}}
                    />
                  )}
                </div>
              </div>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>
    </ChatLayout>
  )
}
