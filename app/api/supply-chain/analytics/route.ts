import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

export async function GET(request: NextRequest) {
  try {
    // Fetch nodes count by type
    const { data: nodes, error: nodesError } = await supabase
      .from('supply_chain_nodes')
      .select('node_type, status, commodities, capacity')

    if (nodesError) throw nodesError

    // Fetch flows statistics
    const { data: flows, error: flowsError } = await supabase
      .from('supply_chain_flows')
      .select('status, commodity, volume')

    if (flowsError) throw flowsError

    // Fetch active events by severity
    const { data: events, error: eventsError } = await supabase
      .from('supply_chain_events')
      .select('severity, event_type, status')
      .eq('status', 'active')

    if (eventsError) throw eventsError

    // Process analytics
    const nodesByType = nodes?.reduce((acc: any, node: any) => {
      acc[node.node_type] = (acc[node.node_type] || 0) + 1
      return acc
    }, {}) || {}

    const capacityByComm = nodes?.reduce((acc: any, node: any) => {
      if (node.commodities && node.capacity) {
        node.commodities.forEach((commodity: string) => {
          acc[commodity] = (acc[commodity] || 0) + (node.capacity || 0)
        })
      }
      return acc
    }, {}) || {}

    const volumeByCommodity = flows?.reduce((acc: any, flow: any) => {
      if (flow.commodity && flow.volume) {
        acc[flow.commodity] = (acc[flow.commodity] || 0) + (flow.volume || 0)
      }
      return acc
    }, {}) || {}

    const riskDistribution = events?.reduce((acc: any, event: any) => {
      acc[event.severity] = (acc[event.severity] || 0) + 1
      return acc
    }, {}) || {}

    const activeFlows = flows?.filter((f: any) => f.status === 'active').length || 0
    const disruptedFlows = flows?.filter((f: any) => f.status === 'disrupted').length || 0

    // Get top risks (most severe/recent events)
    const topRisks = events
      ?.sort((a: any, b: any) => {
        const severityWeight: any = { critical: 4, high: 3, medium: 2, low: 1 }
        return severityWeight[b.severity] - severityWeight[a.severity]
      })
      .slice(0, 5) || []

    // Calculate concentration metrics (simplified)
    const commodities = Array.from(
      new Set(nodes?.flatMap((n: any) => n.commodities || []))
    )

    const concentrationMetrics = commodities.map((commodity: any) => {
      const nodesForComm = nodes?.filter((n: any) =>
        n.commodities?.includes(commodity) && n.capacity
      ) || []

      const totalCapacity = nodesForComm.reduce((sum: number, n: any) =>
        sum + (n.capacity || 0), 0
      )

      const topSuppliers = nodesForComm
        .map((n: any) => ({
          node_id: n.id,
          name: n.name,
          market_share: totalCapacity > 0 ? (n.capacity / totalCapacity) * 100 : 0
        }))
        .sort((a: any, b: any) => b.market_share - a.market_share)
        .slice(0, 3)

      // Calculate Herfindahl-Hirschman Index (HHI)
      const hhi = nodesForComm.reduce((sum: number, n: any) => {
        const share = totalCapacity > 0 ? (n.capacity / totalCapacity) * 100 : 0
        return sum + (share * share)
      }, 0)

      return {
        commodity,
        top_suppliers: topSuppliers,
        herfindahl_index: Math.round(hhi)
      }
    })

    const analytics = {
      total_nodes: nodes?.length || 0,
      nodes_by_type: nodesByType,
      total_flows: flows?.length || 0,
      active_flows: activeFlows,
      disrupted_flows: disruptedFlows,
      total_capacity: capacityByComm,
      total_volume: volumeByCommodity,
      risk_distribution: riskDistribution,
      top_risks: topRisks,
      concentration_metrics: concentrationMetrics
    }

    return NextResponse.json({ analytics })
  } catch (error: any) {
    console.error('Error fetching supply chain analytics:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch supply chain analytics' },
      { status: 500 }
    )
  }
}
