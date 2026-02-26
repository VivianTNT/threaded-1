import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const eventType = searchParams.get('event_type')
    const severity = searchParams.get('severity')
    const status = searchParams.get('status')
    const commodity = searchParams.get('commodity')
    const limit = parseInt(searchParams.get('limit') || '50')

    let query = supabase
      .from('supply_chain_events')
      .select('*')
      .order('event_date', { ascending: false })
      .limit(limit)

    if (eventType) {
      query = query.eq('event_type', eventType)
    }

    if (severity) {
      query = query.eq('severity', severity)
    }

    if (status) {
      query = query.eq('status', status)
    }

    if (commodity) {
      query = query.contains('commodities', [commodity])
    }

    const { data, error } = await query

    if (error) throw error

    return NextResponse.json({ events: data || [] })
  } catch (error: any) {
    console.error('Error fetching supply chain events:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch supply chain events' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const { data, error } = await supabase
      .from('supply_chain_events')
      .insert(body)
      .select()
      .single()

    if (error) throw error

    return NextResponse.json({ event: data })
  } catch (error: any) {
    console.error('Error creating supply chain event:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to create supply chain event' },
      { status: 500 }
    )
  }
}
