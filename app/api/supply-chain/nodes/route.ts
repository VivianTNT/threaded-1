import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const nodeType = searchParams.get('node_type')
    const commodity = searchParams.get('commodity')
    const country = searchParams.get('country')
    const status = searchParams.get('status')
    const limit = parseInt(searchParams.get('limit') || '100')

    let query = supabase
      .from('supply_chain_nodes')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit)

    if (nodeType) {
      query = query.eq('node_type', nodeType)
    }

    if (commodity) {
      query = query.contains('commodities', [commodity])
    }

    if (country) {
      query = query.eq('country', country)
    }

    if (status) {
      query = query.eq('status', status)
    }

    const { data, error } = await query

    if (error) throw error

    return NextResponse.json({ nodes: data || [] })
  } catch (error: any) {
    console.error('Error fetching supply chain nodes:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch supply chain nodes' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const { data, error } = await supabase
      .from('supply_chain_nodes')
      .insert(body)
      .select()
      .single()

    if (error) throw error

    return NextResponse.json({ node: data })
  } catch (error: any) {
    console.error('Error creating supply chain node:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to create supply chain node' },
      { status: 500 }
    )
  }
}
