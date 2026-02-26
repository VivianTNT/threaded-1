"use client"

import * as React from "react"
import { X, MapPin, Factory, Package, TrendingUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
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
  parent_company: string
  description?: string
  website?: string
  latitude?: number
  longitude?: number
}

interface SupplyChainNodeDetailProps {
  node: SupplyChainNode
  onClose: () => void
}

export function SupplyChainNodeDetail({ node, onClose }: SupplyChainNodeDetailProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 hover:bg-green-100'
      case 'disrupted': return 'bg-red-100 text-red-800 hover:bg-red-100'
      case 'at_risk': return 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100'
      default: return 'bg-gray-100 text-gray-800 hover:bg-gray-100'
    }
  }

  const getNodeTypeLabel = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-auto">
        <CardHeader className="flex flex-row items-start justify-between space-y-0">
          <div className="space-y-1">
            <CardTitle className="text-2xl">{node.name}</CardTitle>
            <CardDescription>{node.parent_company}</CardDescription>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Status and Type */}
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className={cn("text-sm", getStatusColor(node.status))}>
              {node.status}
            </Badge>
            <Badge variant="secondary" className="text-sm">
              {getNodeTypeLabel(node.node_type)}
            </Badge>
          </div>

          {/* Location */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Location
            </h3>
            <p className="text-sm text-muted-foreground">
              {node.location}, {node.country}
            </p>
            {node.latitude && node.longitude && (
              <p className="text-xs text-muted-foreground">
                Coordinates: {node.latitude.toFixed(4)}, {node.longitude.toFixed(4)}
              </p>
            )}
          </div>

          {/* Commodities */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Package className="h-4 w-4" />
              Commodities
            </h3>
            <div className="flex flex-wrap gap-2">
              {node.commodities?.map((commodity) => (
                <Badge key={commodity} variant="outline">
                  {commodity}
                </Badge>
              ))}
            </div>
          </div>

          {/* Capacity */}
          {node.capacity && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Factory className="h-4 w-4" />
                Production Capacity
              </h3>
              <p className="text-2xl font-semibold tabular-nums">
                {node.capacity.toLocaleString()} <span className="text-sm font-normal text-muted-foreground">{node.capacity_unit}/year</span>
              </p>
            </div>
          )}

          {/* Description */}
          {node.description && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold">About</h3>
              <p className="text-sm text-muted-foreground">
                {node.description}
              </p>
            </div>
          )}

          {/* Website */}
          {node.website && (
            <div>
              <a
                href={node.website}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-blue-600 hover:text-blue-800 underline"
              >
                Visit website
              </a>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
