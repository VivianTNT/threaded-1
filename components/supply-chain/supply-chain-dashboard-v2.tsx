"use client"

import * as React from "react"
import { IconTrendingUp, IconTrendingDown } from "@tabler/icons-react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, CardAction } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { AlertCircle, TrendingUp, MapPin, Factory, ArrowRight, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { SupplyChainNodeDetail } from "./supply-chain-node-detail"
import { SupplyChainEventCard } from "./supply-chain-event-card"

interface SupplyChainNode {
  id: string
  name: string
  node_type: string
  location: string
  country: string
  commodities: string[]
  status: string
  capacity: number
  capacity_unit: string
  parent_company: string
}

interface SupplyChainFlow {
  id: string
  source_node: SupplyChainNode
  target_node: SupplyChainNode
  commodity: string
  volume: number
  volume_unit: string
  status: string
  transportation_mode: string[]
  avg_transit_time_days: number
}

interface SupplyChainEvent {
  id: string
  event_type: string
  title: string
  description: string
  severity: string
  status: string
  event_date: string
  commodities: string[]
}

interface Analytics {
  total_nodes: number
  total_flows: number
  active_flows: number
  disrupted_flows: number
  risk_distribution: Record<string, number>
}

export function SupplyChainDashboardV2() {
  const [nodes, setNodes] = React.useState<SupplyChainNode[]>([])
  const [flows, setFlows] = React.useState<SupplyChainFlow[]>([])
  const [events, setEvents] = React.useState<SupplyChainEvent[]>([])
  const [analytics, setAnalytics] = React.useState<Analytics | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [selectedNode, setSelectedNode] = React.useState<SupplyChainNode | null>(null)
  const [selectedCommodity, setSelectedCommodity] = React.useState<string>("all")
  const [currentPage, setCurrentPage] = React.useState(0)
  const itemsPerPage = 10

  React.useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      const [nodesRes, flowsRes, eventsRes, analyticsRes] = await Promise.all([
        fetch('/api/supply-chain/nodes?limit=100'),
        fetch('/api/supply-chain/flows?limit=100'),
        fetch('/api/supply-chain/events?status=active&limit=10'),
        fetch('/api/supply-chain/analytics')
      ])

      const [nodesData, flowsData, eventsData, analyticsData] = await Promise.all([
        nodesRes.json(),
        flowsRes.json(),
        eventsRes.json(),
        analyticsRes.json()
      ])

      setNodes(nodesData.nodes || [])
      setFlows(flowsData.flows || [])
      setEvents(eventsData.events || [])
      setAnalytics(analyticsData.analytics)
    } catch (error) {
      console.error('Error fetching supply chain data:', error)
    } finally {
      setLoading(false)
    }
  }

  const commodities = Array.from(new Set(nodes.flatMap(n => n.commodities || []))).sort()

  const filteredNodes = selectedCommodity === "all"
    ? nodes
    : nodes.filter(n => n.commodities?.includes(selectedCommodity))

  const filteredFlows = selectedCommodity === "all"
    ? flows
    : flows.filter(f => f.commodity === selectedCommodity)

  // Pagination logic
  const totalPages = Math.ceil(filteredNodes.length / itemsPerPage)
  const startIndex = currentPage * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const paginatedNodes = filteredNodes.slice(startIndex, endIndex)

  // Reset to first page when commodity filter changes
  React.useEffect(() => {
    setCurrentPage(0)
  }, [selectedCommodity])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 hover:bg-green-100'
      case 'disrupted': return 'bg-red-100 text-red-800 hover:bg-red-100'
      case 'at_risk': return 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100'
      default: return 'bg-gray-100 text-gray-800 hover:bg-gray-100'
    }
  }

  const activeRisks = events.filter(e => e.status === 'active' && (e.severity === 'high' || e.severity === 'critical'))

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">Loading supply chain intelligence...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Supply Chain Intelligence</h2>
        <p className="text-sm text-muted-foreground">
          Track critical minerals flows from mines to manufacturers with real-time risk monitoring
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 gap-4 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
          <CardHeader>
            <CardDescription>Supply Chain Nodes</CardDescription>
            <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
              {analytics?.total_nodes || 0}
            </CardTitle>
            <CardAction>
              <Badge variant="outline">
                <IconTrendingUp />
                Active
              </Badge>
            </CardAction>
          </CardHeader>
          <CardFooter className="flex-col items-start gap-1.5 text-sm">
            <div className="line-clamp-1 flex gap-2 font-medium">
              Tracked entities <Factory className="size-4" />
            </div>
            <div className="text-muted-foreground">
              Mines, refineries, manufacturers
            </div>
          </CardFooter>
        </Card>

        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
          <CardHeader>
            <CardDescription>Active Material Flows</CardDescription>
            <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
              {analytics?.active_flows || 0}
            </CardTitle>
            <CardAction>
              <Badge variant="outline">
                <IconTrendingUp />
                +{Math.round(((analytics?.active_flows || 0) / (analytics?.total_flows || 1)) * 100)}%
              </Badge>
            </CardAction>
          </CardHeader>
          <CardFooter className="flex-col items-start gap-1.5 text-sm">
            <div className="line-clamp-1 flex gap-2 font-medium">
              Operational flows <TrendingUp className="size-4" />
            </div>
            <div className="text-muted-foreground">
              Out of {analytics?.total_flows || 0} total
            </div>
          </CardFooter>
        </Card>

        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
          <CardHeader>
            <CardDescription>Supply Disruptions</CardDescription>
            <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
              {analytics?.disrupted_flows || 0}
            </CardTitle>
            <CardAction>
              <Badge variant="outline" className="bg-red-100 text-red-800 hover:bg-red-100">
                <AlertCircle className="size-3" />
                Alert
              </Badge>
            </CardAction>
          </CardHeader>
          <CardFooter className="flex-col items-start gap-1.5 text-sm">
            <div className="line-clamp-1 flex gap-2 font-medium">
              Disrupted flows <AlertCircle className="size-4" />
            </div>
            <div className="text-muted-foreground">
              Require immediate attention
            </div>
          </CardFooter>
        </Card>

        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
          <CardHeader>
            <CardDescription>Active Risk Events</CardDescription>
            <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
              {activeRisks.length}
            </CardTitle>
            <CardAction>
              <Badge variant="outline" className={activeRisks.length > 0 ? "bg-orange-100 text-orange-800 hover:bg-orange-100" : ""}>
                {activeRisks.length > 0 ? <IconTrendingDown /> : <IconTrendingUp />}
                {activeRisks.length > 0 ? 'Watch' : 'Clear'}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardFooter className="flex-col items-start gap-1.5 text-sm">
            <div className="line-clamp-1 flex gap-2 font-medium">
              High-severity events <AlertCircle className="size-4" />
            </div>
            <div className="text-muted-foreground">
              Real-time monitoring
            </div>
          </CardFooter>
        </Card>
      </div>

      {/* Commodity Filter */}
      <div className="flex items-center gap-4">
        <Select value={selectedCommodity} onValueChange={setSelectedCommodity}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Filter by commodity" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Commodities</SelectItem>
            {commodities.map((commodity) => (
              <SelectItem key={commodity} value={commodity}>
                {commodity}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Badge variant="outline" className="text-sm">
          {filteredNodes.length} nodes • {filteredFlows.length} flows
        </Badge>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 @container lg:grid-cols-3">
        {/* Left Column - Supply Chain Nodes */}
        <div className="lg:col-span-2 space-y-4">
          {/* Supply Chain Nodes Table */}
          <div className="rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Entity</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Location</TableHead>
                  <TableHead>Commodities</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedNodes.map((node) => (
                  <TableRow
                    key={node.id}
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => setSelectedNode(node)}
                  >
                    <TableCell>
                      <div className="space-y-1">
                        <div className="font-medium">{node.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {node.parent_company}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="text-xs">
                        {node.node_type.replace('_', ' ')}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1 text-sm">
                        <MapPin className="h-3 w-3 text-muted-foreground" />
                        {node.country}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {node.commodities?.slice(0, 2).map((commodity) => (
                          <Badge key={commodity} variant="outline" className="text-xs">
                            {commodity}
                          </Badge>
                        ))}
                        {node.commodities && node.commodities.length > 2 && (
                          <Badge variant="outline" className="text-xs">
                            +{node.commodities.length - 2}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className={cn("text-xs", getStatusColor(node.status))}>
                        {node.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <div className="flex items-center justify-end space-x-2">
            <div className="flex-1 text-sm text-muted-foreground">
              Showing {startIndex + 1} to {Math.min(endIndex, filteredNodes.length)} of {filteredNodes.length} results
            </div>
            <div className="space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(prev => Math.max(0, prev - 1))}
                disabled={currentPage === 0}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(prev => Math.min(totalPages - 1, prev + 1))}
                disabled={currentPage >= totalPages - 1}
              >
                Next
              </Button>
            </div>
          </div>
        </div>

        {/* Right Column - Key Material Flows */}
        <div className="space-y-4">
          <div className="rounded-lg border h-[650px] flex flex-col">
            <div className="border-b px-6 py-4">
              <h3 className="font-semibold">Key Material Flows</h3>
              <p className="text-sm text-muted-foreground mt-1">Major supply chain connections</p>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {filteredFlows.map((flow) => (
                <div
                  key={flow.id}
                  className="border-l-2 border-primary/20 pl-3 py-2 hover:border-primary/40 hover:bg-muted/30 transition-colors"
                >
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <span className="truncate">{flow.source_node?.name}</span>
                    <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                    <span className="truncate">{flow.target_node?.name}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <Badge variant="outline" className="text-xs">
                      {flow.commodity}
                    </Badge>
                    {flow.volume && (
                      <span className="text-xs text-muted-foreground">
                        {flow.volume.toLocaleString()} {flow.volume_unit}/yr
                      </span>
                    )}
                  </div>
                  {flow.avg_transit_time_days && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Transit: {flow.avg_transit_time_days} days
                    </div>
                  )}
                  <Badge
                    variant="outline"
                    className={cn("text-xs mt-2", getStatusColor(flow.status))}
                  >
                    {flow.status}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Active Supply Chain Events */}
      {events.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Active Supply Chain Events</CardTitle>
            <CardDescription>
              Real-time monitoring of disruptions and risks
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-2">
              {events.slice(0, 4).map((event) => (
                <SupplyChainEventCard key={event.id} event={event} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Node Detail Panel */}
      {selectedNode && (
        <SupplyChainNodeDetail
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  )
}
