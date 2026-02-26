import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params

    // Fetch project from Supabase
    const { data: project, error } = await supabase
      .from('projects')
      .select('*')
      .eq('id', id)
      .single()

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 404 })
    }

    // Capitalize location and commodities consistently
    if (project.location) {
      project.location = project.location
        .split(',')
        .map((part: string) => part.trim().charAt(0).toUpperCase() + part.trim().slice(1).toLowerCase())
        .join(', ')
    }

    if (Array.isArray(project.commodities)) {
      project.commodities = project.commodities.map((c: string) =>
        c.charAt(0).toUpperCase() + c.slice(1).toLowerCase()
      )
    }

    return NextResponse.json(project)
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
