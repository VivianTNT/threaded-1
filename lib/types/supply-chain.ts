/**
 * Supply Chain Types
 * Types for tracking critical minerals supply chains from mines to end users
 */

export type SupplyChainNodeType =
  | 'mine'
  | 'supplier'
  | 'smelter'
  | 'refinery'
  | 'manufacturer'
  | 'logistics_provider'

export type FlowStatus = 'active' | 'disrupted' | 'at_risk' | 'inactive'

export type RiskSeverity = 'low' | 'medium' | 'high' | 'critical'

export type EventType =
  | 'disruption'
  | 'delay'
  | 'price_shock'
  | 'geopolitical'
  | 'esg_incident'
  | 'weather'
  | 'logistics'
  | 'regulatory'

/**
 * Supply Chain Node - represents an entity in the supply chain
 */
export interface SupplyChainNode {
  id: string
  name: string
  node_type: SupplyChainNodeType
  location: string | null
  country: string | null
  latitude: number | null
  longitude: number | null
  commodities: string[] | null // Commodities handled
  capacity: number | null // Annual capacity in tonnes
  capacity_unit: string | null
  status: FlowStatus
  parent_company: string | null
  certifications: string[] | null // ISO, sustainability certs
  description: string | null
  website: string | null
  created_at: string
  updated_at: string

  // Relations
  project_id?: string | null // Link to mining project if node_type is 'mine'
}

/**
 * Supply Chain Flow - represents material flows between nodes
 */
export interface SupplyChainFlow {
  id: string
  source_node_id: string
  target_node_id: string
  commodity: string
  volume: number | null // Annual volume in tonnes
  volume_unit: string | null
  contract_type: string | null // 'offtake', 'spot', 'long_term'
  contract_start_date: string | null
  contract_end_date: string | null
  status: FlowStatus
  transportation_mode: string[] | null // ['rail', 'ship', 'truck']
  avg_transit_time_days: number | null
  cost_per_unit: number | null
  currency: string | null
  notes: string | null
  created_at: string
  updated_at: string

  // Computed for display
  source_node?: SupplyChainNode
  target_node?: SupplyChainNode
}

/**
 * Supply Chain Event - tracks disruptions, risks, and news
 */
export interface SupplyChainEvent {
  id: string
  event_type: EventType
  title: string
  description: string | null
  severity: RiskSeverity
  status: 'active' | 'monitoring' | 'resolved'
  affected_node_ids: string[] // Nodes impacted
  affected_flow_ids: string[] | null // Flows impacted
  commodities: string[] | null
  source_url: string | null
  source_name: string | null
  sentiment_score: number | null // -1 to 1
  event_date: string
  resolution_date: string | null
  impact_description: string | null
  created_at: string
  updated_at: string

  // Computed for display
  affected_nodes?: SupplyChainNode[]
  affected_flows?: SupplyChainFlow[]
}

/**
 * Supply Chain Risk Metric - calculated risk scores
 */
export interface SupplyChainRiskMetric {
  id: string
  node_id: string | null
  flow_id: string | null
  commodity: string
  risk_category: string // 'concentration', 'geopolitical', 'esg', 'logistics', 'financial'
  risk_score: number // 0-100
  risk_level: RiskSeverity
  factors: Record<string, any> // JSON with contributing factors
  calculated_at: string
  created_at: string

  // Computed for display
  node?: SupplyChainNode
  flow?: SupplyChainFlow
}

/**
 * Processing Stage - tracks material transformation stages
 */
export interface ProcessingStage {
  id: string
  node_id: string
  input_commodity: string
  output_commodity: string
  conversion_rate: number // e.g., 0.95 for 95% yield
  processing_method: string | null
  energy_consumption_kwh: number | null
  water_consumption_m3: number | null
  emissions_co2_tonnes: number | null
  created_at: string
  updated_at: string

  node?: SupplyChainNode
}

/**
 * Supply Chain Analytics - aggregated metrics
 */
export interface SupplyChainAnalytics {
  total_nodes: number
  nodes_by_type: Record<SupplyChainNodeType, number>
  total_flows: number
  active_flows: number
  disrupted_flows: number
  total_capacity: Record<string, number> // By commodity
  total_volume: Record<string, number> // By commodity
  risk_distribution: Record<RiskSeverity, number>
  top_risks: SupplyChainEvent[]
  concentration_metrics: {
    commodity: string
    top_suppliers: Array<{
      node_id: string
      name: string
      market_share: number
    }>
    herfindahl_index: number // Market concentration measure
  }[]
}

/**
 * Supply Chain Filter - for querying and filtering
 */
export interface SupplyChainFilter {
  node_types?: SupplyChainNodeType[]
  commodities?: string[]
  countries?: string[]
  status?: FlowStatus[]
  risk_levels?: RiskSeverity[]
  search_query?: string
  has_active_events?: boolean
}

/**
 * Network Visualization Data - formatted for graph rendering
 */
export interface SupplyChainNetworkData {
  nodes: Array<{
    id: string
    label: string
    type: SupplyChainNodeType
    size: number
    color: string
    risk_level: RiskSeverity
    metadata: SupplyChainNode
  }>
  edges: Array<{
    id: string
    source: string
    target: string
    label: string
    weight: number
    status: FlowStatus
    metadata: SupplyChainFlow
  }>
}
