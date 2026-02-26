"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ArrowRight, Search, Ship, Truck, Train, MapPin } from "lucide-react"
import { cn } from "@/lib/utils"

interface SupplyChainFlow {
  id: string
  source_node: any
  target_node: any
  commodity: string
  volume: number
  volume_unit: string
  status: string
  contract_type: string
  transportation_mode: string[]
  avg_transit_time_days: number
  cost_per_unit: number
  currency: string
}

export function SupplyChainFlowsView() {
  const [flows, setFlows] = React.useState<SupplyChainFlow[]>([])
  const [loading, setLoading] = React.useState(true)
  const [searchQuery, setSearchQuery] = React.useState("")
  const [statusFilter, setStatusFilter] = React.useState<string>("all")
  const [commodityFilter, setCommodityFilter] = React.useState<string>("all")

  React.useEffect(() => {
    fetchFlows()
  }, [])

  const fetchFlows = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/supply-chain/flows')
      const data = await response.json()
      setFlows(data.flows || [])
    } catch (error) {
      console.error('Error fetching flows:', error)
    } finally {
      setLoading(false)
    }
  }

  // Get unique commodities for filter
  const commodities = Array.from(new Set(flows.map(f => f.commodity))).sort()

  // Filter flows
  const filteredFlows = flows.filter(flow => {
    const matchesSearch = searchQuery === "" ||
      flow.source_node?.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      flow.target_node?.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      flow.commodity?.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesStatus = statusFilter === "all" || flow.status === statusFilter
    const matchesCommodity = commodityFilter === "all" || flow.commodity === commodityFilter

    return matchesSearch && matchesStatus && matchesCommodity
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 border-green-200'
      case 'disrupted': return 'bg-red-100 text-red-800 border-red-200'
      case 'at_risk': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'inactive': return 'bg-gray-100 text-gray-800 border-gray-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getTransportIcon = (mode: string) => {
    switch (mode.toLowerCase()) {
      case 'ship': return <Ship className="h-4 w-4" />
      case 'truck': return <Truck className="h-4 w-4" />
      case 'rail': return <Train className="h-4 w-4" />
      default: return <MapPin className="h-4 w-4" />
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">Loading supply chain flows...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Filter Flows</CardTitle>
          <CardDescription>
            Search and filter material flows across the supply chain
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search flows..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={commodityFilter} onValueChange={setCommodityFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All Commodities" />
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
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All Statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="disrupted">Disrupted</SelectItem>
                <SelectItem value="at_risk">At Risk</SelectItem>
                <SelectItem value="inactive">Inactive</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Flows Table */}
      <Card>
        <CardHeader>
          <CardTitle>Material Flows ({filteredFlows.length})</CardTitle>
          <CardDescription>
            Track commodity flows from mines to processors and manufacturers
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Source → Destination</TableHead>
                  <TableHead>Commodity</TableHead>
                  <TableHead className="text-right">Volume</TableHead>
                  <TableHead>Transport</TableHead>
                  <TableHead className="text-right">Transit Time</TableHead>
                  <TableHead>Contract</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredFlows.map((flow) => (
                  <TableRow key={flow.id}>
                    <TableCell>
                      <div className="space-y-1 min-w-[250px]">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">
                            {flow.source_node?.name || 'Unknown'}
                          </span>
                          <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                          <span className="font-medium text-sm">
                            {flow.target_node?.name || 'Unknown'}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <span>{flow.source_node?.country}</span>
                          <span>→</span>
                          <span>{flow.target_node?.country}</span>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{flow.commodity}</Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {flow.volume ? (
                        <div className="space-y-0.5">
                          <div className="font-medium">
                            {flow.volume.toLocaleString()}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {flow.volume_unit || 'tonnes'}/year
                          </div>
                        </div>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {flow.transportation_mode && flow.transportation_mode.length > 0 ? (
                        <div className="flex gap-2">
                          {flow.transportation_mode.map((mode, idx) => (
                            <div
                              key={idx}
                              className="flex items-center gap-1 text-muted-foreground"
                              title={mode}
                            >
                              {getTransportIcon(mode)}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {flow.avg_transit_time_days ? (
                        <span>{flow.avg_transit_time_days} days</span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {flow.contract_type ? (
                        <Badge variant="secondary" className="text-xs">
                          {flow.contract_type.replace('_', ' ')}
                        </Badge>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={cn("text-xs", getStatusColor(flow.status))}
                      >
                        {flow.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            {filteredFlows.length === 0 && (
              <div className="text-center py-12 text-muted-foreground">
                No flows found matching your filters
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Flow Statistics */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="p-6">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Total Volume</p>
              <p className="text-2xl font-semibold">
                {filteredFlows
                  .reduce((sum, f) => sum + (f.volume || 0), 0)
                  .toLocaleString()}
              </p>
              <p className="text-xs text-muted-foreground">tonnes/year</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Avg Transit Time</p>
              <p className="text-2xl font-semibold">
                {Math.round(
                  filteredFlows
                    .filter(f => f.avg_transit_time_days)
                    .reduce((sum, f) => sum + (f.avg_transit_time_days || 0), 0) /
                  filteredFlows.filter(f => f.avg_transit_time_days).length || 1
                )}
              </p>
              <p className="text-xs text-muted-foreground">days</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Active Contracts</p>
              <p className="text-2xl font-semibold">
                {filteredFlows.filter(f => f.status === 'active').length}
              </p>
              <p className="text-xs text-muted-foreground">
                out of {filteredFlows.length} total
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
