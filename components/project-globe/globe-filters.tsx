"use client"

import * as React from "react"
import { Filter, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { MiningProject } from "@/lib/types/mining-project"

interface GlobeFiltersProps {
  projects: MiningProject[]
  onFilterChange: (filteredProjects: MiningProject[]) => void
  filteredCount?: number
}

const stages = [
  "Exploration",
  "PEA",
  "PFS",
  "DFS",
  "Development",
  "Production",
  "Care & Maintenance",
]

const commodities = [
  "Lithium",
  "Copper",
  "Nickel",
  "Cobalt",
  "Graphite",
  "Rare Earths",
  "Uranium",
  "Gold",
  "Silver",
]

export function GlobeFilters({ projects, onFilterChange, filteredCount }: GlobeFiltersProps) {
  const [selectedStages, setSelectedStages] = React.useState<string[]>([])
  const [selectedCommodities, setSelectedCommodities] = React.useState<string[]>([])
  const [isOpen, setIsOpen] = React.useState(false)

  // Apply filters whenever selections change
  React.useEffect(() => {
    let filtered = projects

    // Filter by stage
    if (selectedStages.length > 0) {
      filtered = filtered.filter(p =>
        p.stage && selectedStages.includes(p.stage)
      )
    }

    // Filter by commodity
    if (selectedCommodities.length > 0) {
      filtered = filtered.filter(p =>
        p.commodities && p.commodities.some(c =>
          selectedCommodities.some(sc => c.toLowerCase().includes(sc.toLowerCase()))
        )
      )
    }

    onFilterChange(filtered)
  }, [selectedStages, selectedCommodities, projects])

  const handleStageToggle = (stage: string) => {
    setSelectedStages(prev =>
      prev.includes(stage)
        ? prev.filter(s => s !== stage)
        : [...prev, stage]
    )
  }

  const handleCommodityToggle = (commodity: string) => {
    setSelectedCommodities(prev =>
      prev.includes(commodity)
        ? prev.filter(c => c !== commodity)
        : [...prev, commodity]
    )
  }

  const clearFilters = () => {
    setSelectedStages([])
    setSelectedCommodities([])
  }

  const activeFiltersCount = selectedStages.length + selectedCommodities.length

  return (
    <Card className="p-2">
      <div className="mb-2">
        <div className="text-xs text-muted-foreground mb-1">Total Projects</div>
        <div className="text-2xl font-semibold">{projects.length.toLocaleString()}</div>
      </div>
      <div>
        <Popover open={isOpen} onOpenChange={setIsOpen}>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="relative w-full">
              <Filter className="h-3 w-3 mr-1.5" />
              Filter
              {activeFiltersCount > 0 && (
                <Badge
                  variant="secondary"
                  className="ml-1.5 h-4 w-4 p-0 flex items-center justify-center rounded-full text-[9px]"
                >
                  {activeFiltersCount}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-80" align="start">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold text-sm">Filters</h4>
                {activeFiltersCount > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearFilters}
                    className="h-7 text-xs"
                  >
                    Clear all
                  </Button>
                )}
              </div>

              {/* Stage Filter */}
              <div>
                <Label className="text-xs font-semibold mb-2 block">Stage</Label>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {stages.map((stage) => (
                    <div key={stage} className="flex items-center space-x-2">
                      <Checkbox
                        id={`stage-${stage}`}
                        checked={selectedStages.includes(stage)}
                        onCheckedChange={() => handleStageToggle(stage)}
                      />
                      <label
                        htmlFor={`stage-${stage}`}
                        className="text-xs font-normal leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                      >
                        {stage}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Commodity Filter */}
              <div>
                <Label className="text-xs font-semibold mb-2 block">Commodity</Label>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {commodities.map((commodity) => (
                    <div key={commodity} className="flex items-center space-x-2">
                      <Checkbox
                        id={`commodity-${commodity}`}
                        checked={selectedCommodities.includes(commodity)}
                        onCheckedChange={() => handleCommodityToggle(commodity)}
                      />
                      <label
                        htmlFor={`commodity-${commodity}`}
                        className="text-xs font-normal leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                      >
                        {commodity}
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      </div>

      {/* Active Filters */}
      {activeFiltersCount > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {selectedStages.map((stage) => (
            <Badge
              key={stage}
              variant="secondary"
              className="text-[10px] h-5 px-1.5"
            >
              {stage}
              <button
                onClick={() => handleStageToggle(stage)}
                className="ml-1 hover:text-foreground"
              >
                <X className="h-2.5 w-2.5" />
              </button>
            </Badge>
          ))}
          {selectedCommodities.map((commodity) => (
            <Badge
              key={commodity}
              variant="secondary"
              className="text-[10px] h-5 px-1.5"
            >
              {commodity}
              <button
                onClick={() => handleCommodityToggle(commodity)}
                className="ml-1 hover:text-foreground"
              >
                <X className="h-2.5 w-2.5" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </Card>
  )
}
