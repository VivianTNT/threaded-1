import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

interface Project {
  id: string
  name: string
  commodities: string[]
  capex: number | null
  npv: number | null
  irr: number | null
  stage: string | null
  location: string | null
}

// Calculate Jaccard similarity for commodity overlap
function jaccardSimilarity(arr1: string[], arr2: string[]): number {
  if (!arr1 || !arr2 || arr1.length === 0 || arr2.length === 0) return 0

  const set1 = new Set(arr1.map(c => c.toLowerCase()))
  const set2 = new Set(arr2.map(c => c.toLowerCase()))

  const intersection = new Set([...set1].filter(x => set2.has(x)))
  const union = new Set([...set1, ...set2])

  return intersection.size / union.size
}

// Extract country from location string
function extractCountry(location: string | null): string {
  if (!location) return ''

  // Simple country extraction - takes last part after comma
  const parts = location.split(',')
  return parts[parts.length - 1].trim().toLowerCase()
}

// Calculate similarity score between two projects
function calculateSimilarity(project1: Project, project2: Project): number {
  let score = 0

  // Commodity overlap (Jaccard similarity) - weight 40%
  const commodityOverlap = jaccardSimilarity(project1.commodities || [], project2.commodities || [])
  score += commodityOverlap * 0.4

  // CAPEX similarity - weight 20%
  if (project1.capex && project2.capex) {
    const maxCapex = Math.max(project1.capex, project2.capex)
    const minCapex = Math.min(project1.capex, project2.capex)
    const capexSimilarity = minCapex / maxCapex
    score += capexSimilarity * 0.2
  }

  // NPV similarity - weight 20%
  if (project1.npv && project2.npv) {
    const maxNpv = Math.max(project1.npv, project2.npv)
    const minNpv = Math.min(project1.npv, project2.npv)
    const npvSimilarity = minNpv / maxNpv
    score += npvSimilarity * 0.2
  }

  // Stage match - weight 10%
  if (project1.stage && project2.stage && project1.stage === project2.stage) {
    score += 0.1
  }

  // Location similarity (country match) - weight 10%
  const country1 = extractCountry(project1.location)
  const country2 = extractCountry(project2.location)
  if (country1 && country2 && country1 === country2) {
    score += 0.1
  }

  return score
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params

    // Fetch the current project
    const { data: currentProject, error: currentError } = await supabase
      .from('projects')
      .select('id, name, commodities, capex, npv, irr, stage, location')
      .eq('id', id)
      .single()

    if (currentError || !currentProject) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Fetch all other projects
    const { data: allProjects, error: allError } = await supabase
      .from('projects')
      .select('id, name, commodities, capex, npv, irr, stage, location')
      .neq('id', id)

    if (allError || !allProjects) {
      return NextResponse.json({ error: 'Failed to fetch projects' }, { status: 500 })
    }

    // Calculate similarity scores
    const projectsWithScores = allProjects.map(project => ({
      ...project,
      similarityScore: calculateSimilarity(currentProject as Project, project as Project)
    }))

    // Sort by similarity score and take top 4
    const similarProjects = projectsWithScores
      .sort((a, b) => b.similarityScore - a.similarityScore)
      .slice(0, 4)
      .map(({ similarityScore, ...project }) => project)

    return NextResponse.json({ similar: similarProjects })
  } catch (error: any) {
    console.error('Error calculating similar projects:', error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
