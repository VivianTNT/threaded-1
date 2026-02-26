"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Factory,
  TrendingDown,
  TrendingUp,
  AlertCircle,
  MapPin,
  Package,
  ArrowRight
} from "lucide-react"
import { cn } from "@/lib/utils"

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
}

interface SupplyChainFlow {
  id: string
  source_node: SupplyChainNode
  target_node: SupplyChainNode
  commodity: string
  volume: number
  status: string
}

export function SupplyChainOverview() {
  const [nodes, setNodes] = React.useState<SupplyChainNode[]>([])
  const [flows, setFlows] = React.useState<SupplyChainFlow[]>([])
  const [loading, setLoading] = React.useState(true)
  const [selectedCommodity, setSelectedCommodity] = React.useState<string | null>(null)

  React.useEffect(() => {
    fetchSupplyChainData()
  }, [])

  const fetchSupplyChainData = async () => {
    try {
      setLoading(true)
      const [nodesRes, flowsRes] = await Promise.all([
        fetch('/api/supply-chain/nodes'),
        fetch('/api/supply-chain/flows')
      ])

      const nodesData = await nodesRes.json()
      const flowsData = await flowsRes.json()

      setNodes(nodesData.nodes || [])
      setFlows(flowsData.flows || [])
    } catch (error) {
      console.error('Error fetching supply chain data:', error)
    } finally {
      setLoading(false)
    }
  }

  // Get unique commodities
  const commodities = Array.from(
    new Set(nodes.flatMap(n => n.commodities || []))
  ).sort()

  // Filter nodes and flows by selected commodity
  const filteredNodes = selectedCommodity
    ? nodes.filter(n => n.commodities?.includes(selectedCommodity))
    : nodes

  const filteredFlows = selectedCommodity
    ? flows.filter(f => f.commodity === selectedCommodity)
    : flows

  // Group nodes by type
  const nodesByType = filteredNodes.reduce((acc, node) => {
    if (!acc[node.node_type]) acc[node.node_type] = []
    acc[node.node_type].push(node)
    return acc
  }, {} as Record<string, SupplyChainNode[]>)

  const getNodeTypeIcon = (type: string) => {
    switch (type) {
      case 'mine': return '⛏️'
      case 'smelter': return '🏭'
      case 'refinery': return '⚗️'
      case 'manufacturer': return '🏢'
      case 'supplier': return '📦'
      case 'logistics_provider': return '🚢'
      default: return '📍'
    }
  }

  const getNodeTypeLabel = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 border-green-200'
      case 'disrupted': return 'bg-red-100 text-red-800 border-red-200'
      case 'at_risk': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'inactive': return 'bg-gray-100 text-gray-800 border-gray-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">Loading supply chain data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Total Nodes</p>
                <p className="text-2xl font-semibold">{filteredNodes.length}</p>
              </div>
              <Factory className="h-8 w-8 text-blue-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Active Flows</p>
                <p className="text-2xl font-semibold">
                  {filteredFlows.filter(f => f.status === 'active').length}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Disrupted</p>
                <p className="text-2xl font-semibold">
                  {filteredFlows.filter(f => f.status === 'disrupted').length}
                </p>
              </div>
              <AlertCircle className="h-8 w-8 text-red-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Commodities</p>
                <p className="text-2xl font-semibold">{commodities.length}</p>
              </div>
              <Package className="h-8 w-8 text-purple-600 opacity-60" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Commodity Filter */}
      <Card>
        <CardHeader>
          <CardTitle>Filter by Commodity</CardTitle>
          <CardDescription>
            Select a commodity to view its supply chain network
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Button
              variant={selectedCommodity === null ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedCommodity(null)}
            >
              All Commodities
            </Button>
            {commodities.map((commodity) => (
              <Button
                key={commodity}
                variant={selectedCommodity === commodity ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedCommodity(commodity)}
              >
                {commodity}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Supply Chain Network Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Supply Chain Network</CardTitle>
          <CardDescription>
            {selectedCommodity
              ? `Showing ${selectedCommodity} supply chain`
              : 'Showing all commodity supply chains'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {Object.entries(nodesByType).map(([type, typeNodes]) => (
              <div key={type} className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{getNodeTypeIcon(type)}</span>
                  <h3 className="text-lg font-semibold">
                    {getNodeTypeLabel(type)} ({typeNodes.length})
                  </h3>
                </div>
                <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                  {typeNodes.map((node) => (
                    <Card key={node.id} className="overflow-hidden">
                      <CardContent className="p-4 space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <div className="space-y-1 flex-1 min-w-0">
                            <p className="font-semibold text-sm truncate">
                              {node.name}
                            </p>
                            <div className="flex items-center gap-1 text-xs text-muted-foreground">
                              <MapPin className="h-3 w-3" />
                              <span className="truncate">{node.country}</span>
                            </div>
                          </div>
                          <Badge
                            variant="outline"
                            className={cn("text-xs", getStatusColor(node.status))}
                          >
                            {node.status}
                          </Badge>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {node.commodities?.map((commodity) => (
                            <Badge key={commodity} variant="secondary" className="text-xs">
                              {commodity}
                            </Badge>
                          ))}
                        </div>
                        {node.capacity && (
                          <p className="text-xs text-muted-foreground">
                            Capacity: {node.capacity.toLocaleString()} {node.capacity_unit}
                          </p>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {filteredNodes.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              No supply chain nodes found for the selected commodity
            </div>
          )}
        </CardContent>
      </Card>

      {/* Key Flows */}
      {filteredFlows.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Key Material Flows</CardTitle>
            <CardDescription>
              Major supply chain connections and volumes
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {filteredFlows.slice(0, 10).map((flow) => (
                <div
                  key={flow.id}
                  className="flex items-center gap-4 p-4 border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm truncate">
                        {flow.source_node?.name}
                      </span>
                      <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      <span className="font-medium text-sm truncate">
                        {flow.target_node?.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs">
                        {flow.commodity}
                      </Badge>
                      {flow.volume && (
                        <span className="text-xs text-muted-foreground">
                          {flow.volume.toLocaleString()} tonnes/year
                        </span>
                      )}
                    </div>
                  </div>
                  <Badge
                    variant="outline"
                    className={cn("text-xs", getStatusColor(flow.status))}
                  >
                    {flow.status}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
