"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  ExternalLink,
  Calendar,
  TrendingDown,
  TrendingUp
} from "lucide-react"
import { cn } from "@/lib/utils"
import { format } from "date-fns"

interface SupplyChainEvent {
  id: string
  event_type: string
  title: string
  description: string
  severity: string
  status: string
  affected_node_ids: string[]
  commodities: string[]
  source_url: string
  source_name: string
  sentiment_score: number
  event_date: string
  resolution_date: string
  impact_description: string
}

export function SupplyChainRiskMonitor() {
  const [events, setEvents] = React.useState<SupplyChainEvent[]>([])
  const [loading, setLoading] = React.useState(true)
  const [severityFilter, setSeverityFilter] = React.useState<string>("all")
  const [statusFilter, setStatusFilter] = React.useState<string>("active")
  const [eventTypeFilter, setEventTypeFilter] = React.useState<string>("all")

  React.useEffect(() => {
    fetchEvents()
  }, [])

  const fetchEvents = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/supply-chain/events')
      const data = await response.json()
      setEvents(data.events || [])
    } catch (error) {
      console.error('Error fetching events:', error)
    } finally {
      setLoading(false)
    }
  }

  // Filter events
  const filteredEvents = events.filter(event => {
    const matchesSeverity = severityFilter === "all" || event.severity === severityFilter
    const matchesStatus = statusFilter === "all" || event.status === statusFilter
    const matchesType = eventTypeFilter === "all" || event.event_type === eventTypeFilter
    return matchesSeverity && matchesStatus && matchesType
  })

  const getSeverityConfig = (severity: string) => {
    switch (severity) {
      case 'critical':
        return {
          color: 'bg-red-100 text-red-800 border-red-200',
          icon: <AlertCircle className="h-5 w-5 text-red-600" />,
          label: 'Critical'
        }
      case 'high':
        return {
          color: 'bg-orange-100 text-orange-800 border-orange-200',
          icon: <AlertTriangle className="h-5 w-5 text-orange-600" />,
          label: 'High'
        }
      case 'medium':
        return {
          color: 'bg-yellow-100 text-yellow-800 border-yellow-200',
          icon: <AlertTriangle className="h-5 w-5 text-yellow-600" />,
          label: 'Medium'
        }
      case 'low':
        return {
          color: 'bg-blue-100 text-blue-800 border-blue-200',
          icon: <AlertCircle className="h-5 w-5 text-blue-600" />,
          label: 'Low'
        }
      default:
        return {
          color: 'bg-gray-100 text-gray-800 border-gray-200',
          icon: <AlertCircle className="h-5 w-5 text-gray-600" />,
          label: severity
        }
    }
  }

  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'active':
        return {
          color: 'bg-red-100 text-red-800',
          icon: <AlertTriangle className="h-4 w-4" />
        }
      case 'monitoring':
        return {
          color: 'bg-yellow-100 text-yellow-800',
          icon: <AlertCircle className="h-4 w-4" />
        }
      case 'resolved':
        return {
          color: 'bg-green-100 text-green-800',
          icon: <CheckCircle className="h-4 w-4" />
        }
      default:
        return {
          color: 'bg-gray-100 text-gray-800',
          icon: <AlertCircle className="h-4 w-4" />
        }
    }
  }

  const getEventTypeLabel = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  const getSentimentIcon = (score: number | null) => {
    if (score === null) return null
    if (score > 0) return <TrendingUp className="h-4 w-4 text-green-600" />
    if (score < 0) return <TrendingDown className="h-4 w-4 text-red-600" />
    return null
  }

  // Risk distribution
  const riskDistribution = {
    critical: filteredEvents.filter(e => e.severity === 'critical').length,
    high: filteredEvents.filter(e => e.severity === 'high').length,
    medium: filteredEvents.filter(e => e.severity === 'medium').length,
    low: filteredEvents.filter(e => e.severity === 'low').length,
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">Loading risk events...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Risk Summary */}
      <div className="grid gap-4 md:grid-cols-4">
        {Object.entries(riskDistribution).map(([severity, count]) => {
          const config = getSeverityConfig(severity)
          return (
            <Card key={severity}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground capitalize">
                      {severity} Risk
                    </p>
                    <p className="text-2xl font-semibold">{count}</p>
                  </div>
                  {config.icon}
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Filter Events</CardTitle>
          <CardDescription>
            Filter supply chain disruptions and risk events
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All Statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="monitoring">Monitoring</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
              </SelectContent>
            </Select>

            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All Severities" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>

            <Select value={eventTypeFilter} onValueChange={setEventTypeFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All Types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="disruption">Disruption</SelectItem>
                <SelectItem value="delay">Delay</SelectItem>
                <SelectItem value="price_shock">Price Shock</SelectItem>
                <SelectItem value="geopolitical">Geopolitical</SelectItem>
                <SelectItem value="esg_incident">ESG Incident</SelectItem>
                <SelectItem value="weather">Weather</SelectItem>
                <SelectItem value="logistics">Logistics</SelectItem>
                <SelectItem value="regulatory">Regulatory</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Events List */}
      <div className="space-y-4">
        {filteredEvents.map((event) => {
          const severityConfig = getSeverityConfig(event.severity)
          const statusConfig = getStatusConfig(event.status)

          return (
            <Card key={event.id}>
              <CardContent className="p-6">
                <div className="space-y-4">
                  {/* Header */}
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3 flex-1 min-w-0">
                      {severityConfig.icon}
                      <div className="space-y-1 flex-1 min-w-0">
                        <h3 className="font-semibold text-base">
                          {event.title}
                        </h3>
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge
                            variant="outline"
                            className={cn("text-xs", severityConfig.color)}
                          >
                            {severityConfig.label}
                          </Badge>
                          <Badge
                            variant="outline"
                            className={cn("text-xs flex items-center gap-1", statusConfig.color)}
                          >
                            {statusConfig.icon}
                            {event.status}
                          </Badge>
                          <Badge variant="secondary" className="text-xs">
                            {getEventTypeLabel(event.event_type)}
                          </Badge>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Calendar className="h-3 w-3" />
                      {format(new Date(event.event_date), 'MMM d, yyyy')}
                    </div>
                  </div>

                  {/* Description */}
                  {event.description && (
                    <p className="text-sm text-muted-foreground">
                      {event.description}
                    </p>
                  )}

                  {/* Commodities */}
                  {event.commodities && event.commodities.length > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        Affected commodities:
                      </span>
                      <div className="flex flex-wrap gap-1">
                        {event.commodities.map((commodity) => (
                          <Badge key={commodity} variant="outline" className="text-xs">
                            {commodity}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Impact & Sentiment */}
                  <div className="flex items-center gap-4 text-sm">
                    {event.affected_node_ids && (
                      <div className="text-muted-foreground">
                        {event.affected_node_ids.length} node(s) affected
                      </div>
                    )}
                    {event.sentiment_score !== null && (
                      <div className="flex items-center gap-1">
                        {getSentimentIcon(event.sentiment_score)}
                        <span className="text-muted-foreground">
                          Sentiment: {event.sentiment_score > 0 ? 'Positive' : 'Negative'}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Source Link */}
                  {event.source_url && (
                    <div className="pt-2 border-t">
                      <a
                        href={event.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
                      >
                        <ExternalLink className="h-3 w-3" />
                        {event.source_name || 'View source'}
                      </a>
                    </div>
                  )}

                  {/* Resolution Date */}
                  {event.resolution_date && (
                    <div className="text-xs text-muted-foreground">
                      Resolved on {format(new Date(event.resolution_date), 'MMM d, yyyy')}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )
        })}

        {filteredEvents.length === 0 && (
          <Card>
            <CardContent className="p-12">
              <div className="text-center text-muted-foreground">
                <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-600" />
                <p className="text-lg font-medium">No events matching your filters</p>
                <p className="text-sm mt-2">
                  {statusFilter === 'active'
                    ? 'No active supply chain disruptions detected'
                    : 'Try adjusting your filters to see more events'}
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
