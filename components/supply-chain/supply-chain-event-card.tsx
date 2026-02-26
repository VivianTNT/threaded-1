"use client"

import * as React from "react"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, AlertCircle, CheckCircle, Calendar, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { format } from "date-fns"

interface SupplyChainEvent {
  id: string
  event_type: string
  title: string
  description: string
  severity: string
  status: string
  event_date: string
  commodities: string[]
  source_url?: string
}

interface SupplyChainEventCardProps {
  event: SupplyChainEvent
}

export function SupplyChainEventCard({ event }: SupplyChainEventCardProps) {
  const getSeverityConfig = (severity: string) => {
    switch (severity) {
      case 'critical':
        return {
          color: 'bg-red-100 text-red-800 hover:bg-red-100',
          icon: <AlertCircle className="h-4 w-4 text-red-600" />
        }
      case 'high':
        return {
          color: 'bg-orange-100 text-orange-800 hover:bg-orange-100',
          icon: <AlertTriangle className="h-4 w-4 text-orange-600" />
        }
      case 'medium':
        return {
          color: 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100',
          icon: <AlertTriangle className="h-4 w-4 text-yellow-600" />
        }
      default:
        return {
          color: 'bg-blue-100 text-blue-800 hover:bg-blue-100',
          icon: <AlertCircle className="h-4 w-4 text-blue-600" />
        }
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge variant="outline" className="bg-red-100 text-red-800 hover:bg-red-100 text-xs">Active</Badge>
      case 'monitoring':
        return <Badge variant="outline" className="bg-yellow-100 text-yellow-800 hover:bg-yellow-100 text-xs">Monitoring</Badge>
      case 'resolved':
        return <Badge variant="outline" className="bg-green-100 text-green-800 hover:bg-green-100 text-xs">Resolved</Badge>
      default:
        return <Badge variant="outline" className="text-xs">{status}</Badge>
    }
  }

  const severityConfig = getSeverityConfig(event.severity)

  return (
    <div className="border rounded-lg p-4 hover:bg-muted/30 transition-colors">
      <div className="flex items-start gap-3">
        {severityConfig.icon}
        <div className="flex-1 space-y-2">
          <div className="flex items-start justify-between gap-2">
            <h4 className="font-medium text-sm leading-tight">{event.title}</h4>
            {getStatusBadge(event.status)}
          </div>

          {event.description && (
            <p className="text-xs text-muted-foreground line-clamp-2">
              {event.description}
            </p>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className={cn("text-xs", severityConfig.color)}>
              {event.severity}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {event.event_type.replace('_', ' ')}
            </Badge>
            {event.commodities && event.commodities.length > 0 && (
              <Badge variant="outline" className="text-xs">
                {event.commodities[0]}
              </Badge>
            )}
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Calendar className="h-3 w-3" />
              {format(new Date(event.event_date), 'MMM d')}
            </div>
          </div>

          {event.source_url && (
            <a
              href={event.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              <ExternalLink className="h-3 w-3" />
              View source
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
