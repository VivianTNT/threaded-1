"use client"

import * as React from "react"
import { Company } from "@/lib/hooks/use-companies"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Building, Globe, TrendingUp, DollarSign } from "lucide-react"

interface CompanyDetailViewProps {
  company: Company
}

export function CompanyDetailView({ company }: CompanyDetailViewProps) {
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold">{company.name}</h1>
          {company.ticker && (
            <div className="flex items-center gap-2">
              <Badge variant="outline">{company.ticker}</Badge>
              {company.exchange && (
                <span className="text-sm text-muted-foreground">{company.exchange}</span>
              )}
            </div>
          )}
        </div>
      </div>

      <Separator />

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Market Cap</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {company.market_cap !== null && company.market_cap !== undefined
                ? `$${company.market_cap.toFixed(0)}M`
                : 'N/A'}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Location</CardTitle>
            <Globe className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {company.country || 'N/A'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Company Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building className="h-5 w-5" />
            Company Information
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {company.description && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Description</h3>
              <p className="text-sm">{company.description}</p>
            </div>
          )}

          {company.website && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Website</h3>
              <a
                href={company.website}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-blue-600 hover:underline"
              >
                {company.website}
              </a>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 pt-4 border-t">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-1">Ticker</h3>
              <p className="text-sm font-medium">{company.ticker || 'N/A'}</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-1">Exchange</h3>
              <p className="text-sm font-medium">{company.exchange || 'N/A'}</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-1">Country</h3>
              <p className="text-sm font-medium">{company.country || 'N/A'}</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-1">Market Cap</h3>
              <p className="text-sm font-medium">
                {company.market_cap !== null && company.market_cap !== undefined
                  ? `$${company.market_cap.toFixed(0)}M`
                  : 'N/A'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Note about AISC */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="h-5 w-5" />
            Cost Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            AISC (All-In Sustaining Cost) is reported at the project/mine level, not company-wide.
            Different projects produce different commodities (gold = $/oz, copper = $/lb, lithium = $/tonne),
            making a company-wide average AISC meaningless. View individual project pages to see project-specific AISC values.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
