"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts"
import { TrendingUp, AlertTriangle, Package, Factory } from "lucide-react"

interface SupplyChainAnalytics {
  total_nodes: number
  nodes_by_type: Record<string, number>
  total_flows: number
  active_flows: number
  disrupted_flows: number
  total_capacity: Record<string, number>
  total_volume: Record<string, number>
  risk_distribution: Record<string, number>
  top_risks: any[]
  concentration_metrics: Array<{
    commodity: string
    top_suppliers: Array<{
      node_id: string
      name: string
      market_share: number
    }>
    herfindahl_index: number
  }>
}

export function SupplyChainAnalytics() {
  const [analytics, setAnalytics] = React.useState<SupplyChainAnalytics | null>(null)
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    fetchAnalytics()
  }, [])

  const fetchAnalytics = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/supply-chain/analytics')
      const data = await response.json()
      setAnalytics(data.analytics)
    } catch (error) {
      console.error('Error fetching analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">Loading analytics...</div>
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">No analytics data available</div>
      </div>
    )
  }

  // Prepare chart data
  const nodesByTypeData = Object.entries(analytics.nodes_by_type).map(([type, count]) => ({
    name: type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
    count
  }))

  const capacityData = Object.entries(analytics.total_capacity)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([commodity, capacity]) => ({
      name: commodity,
      capacity: Math.round(capacity)
    }))

  const volumeData = Object.entries(analytics.total_volume)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([commodity, volume]) => ({
      name: commodity,
      volume: Math.round(volume)
    }))

  const riskDistributionData = Object.entries(analytics.risk_distribution).map(([severity, count]) => ({
    name: severity.charAt(0).toUpperCase() + severity.slice(1),
    value: count
  }))

  const COLORS = ['#ef4444', '#f97316', '#eab308', '#3b82f6', '#8b5cf6']

  const getConcentrationRating = (hhi: number) => {
    if (hhi < 1500) return { label: 'Low Concentration', color: 'bg-green-100 text-green-800' }
    if (hhi < 2500) return { label: 'Moderate Concentration', color: 'bg-yellow-100 text-yellow-800' }
    return { label: 'High Concentration', color: 'bg-red-100 text-red-800' }
  }

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Total Nodes</p>
                <p className="text-2xl font-semibold">{analytics.total_nodes}</p>
              </div>
              <Factory className="h-8 w-8 text-blue-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Total Flows</p>
                <p className="text-2xl font-semibold">{analytics.total_flows}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Active Flows</p>
                <p className="text-2xl font-semibold">{analytics.active_flows}</p>
                <p className="text-xs text-muted-foreground">
                  {Math.round((analytics.active_flows / analytics.total_flows) * 100)}% of total
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
                <p className="text-sm text-muted-foreground">Disruptions</p>
                <p className="text-2xl font-semibold">{analytics.disrupted_flows}</p>
                <p className="text-xs text-muted-foreground">
                  {analytics.total_flows > 0
                    ? Math.round((analytics.disrupted_flows / analytics.total_flows) * 100)
                    : 0}% of total
                </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-600 opacity-60" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Nodes by Type */}
        <Card>
          <CardHeader>
            <CardTitle>Supply Chain Nodes by Type</CardTitle>
            <CardDescription>Distribution of entities across the supply chain</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={nodesByTypeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Active Risk Distribution</CardTitle>
            <CardDescription>Current supply chain disruptions by severity</CardDescription>
          </CardHeader>
          <CardContent>
            {riskDistributionData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskDistributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                No active risk events
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Total Capacity */}
        <Card>
          <CardHeader>
            <CardTitle>Production Capacity by Commodity</CardTitle>
            <CardDescription>Annual production capacity in tonnes</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={capacityData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={80} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="capacity" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Total Volume */}
        <Card>
          <CardHeader>
            <CardTitle>Flow Volume by Commodity</CardTitle>
            <CardDescription>Annual flow volume in tonnes</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={volumeData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={80} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="volume" fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Concentration Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Supply Concentration Analysis</CardTitle>
          <CardDescription>
            Market concentration metrics (HHI) for key commodities
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {analytics.concentration_metrics.map((metric) => {
              const rating = getConcentrationRating(metric.herfindahl_index)
              return (
                <div key={metric.commodity} className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Package className="h-5 w-5 text-muted-foreground" />
                      <h4 className="font-semibold">{metric.commodity}</h4>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className={rating.color}>
                        {rating.label}
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        HHI: {metric.herfindahl_index}
                      </span>
                    </div>
                  </div>
                  <div className="pl-8 space-y-2">
                    <p className="text-sm text-muted-foreground">Top Suppliers:</p>
                    <div className="space-y-2">
                      {metric.top_suppliers.map((supplier, idx) => (
                        <div
                          key={supplier.node_id}
                          className="flex items-center justify-between text-sm"
                        >
                          <span>
                            {idx + 1}. {supplier.name}
                          </span>
                          <Badge variant="secondary">
                            {supplier.market_share.toFixed(1)}% share
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
