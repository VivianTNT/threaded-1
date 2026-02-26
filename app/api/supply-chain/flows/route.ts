import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const commodity = searchParams.get('commodity')
    const status = searchParams.get('status')
    const sourceNodeId = searchParams.get('source_node_id')
    const targetNodeId = searchParams.get('target_node_id')
    const limit = parseInt(searchParams.get('limit') || '100')

    let query = supabase
      .from('supply_chain_flows')
      .select(`
        *,
        source_node:source_node_id(id, name, node_type, location, country, commodities, status),
        target_node:target_node_id(id, name, node_type, location, country, commodities, status)
      `)
      .order('created_at', { ascending: false })
      .limit(limit)

    if (commodity) {
      query = query.eq('commodity', commodity)
    }

    if (status) {
      query = query.eq('status', status)
    }

    if (sourceNodeId) {
      query = query.eq('source_node_id', sourceNodeId)
    }

    if (targetNodeId) {
      query = query.eq('target_node_id', targetNodeId)
    }

    const { data, error } = await query

    if (error) throw error

    return NextResponse.json({ flows: data || [] })
  } catch (error: any) {
    console.error('Error fetching supply chain flows:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch supply chain flows' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const { data, error } = await supabase
      .from('supply_chain_flows')
      .insert(body)
      .select(`
        *,
        source_node:source_node_id(*),
        target_node:target_node_id(*)
      `)
      .single()

    if (error) throw error

    return NextResponse.json({ flow: data })
  } catch (error: any) {
    console.error('Error creating supply chain flow:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to create supply chain flow' },
      { status: 500 }
    )
  }
}
