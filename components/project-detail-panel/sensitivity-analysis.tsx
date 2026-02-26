"use client"

import * as React from "react"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MiningProject } from "@/lib/types/mining-project"
import { Sparkles } from "lucide-react"

interface SensitivityAnalysisProps {
  project: MiningProject
}

interface ActualValue {
  value: number
  unit: string
  source: string
}

interface SensitivityParameter {
  name: string
  key: string
  min: number
  max: number
  baseline: number
  unit: string
  step: number
  category: 'market' | 'production' | 'costs'
}

// Helper function to format large dollar amounts
function formatDollarAmount(value: number, includeDecimal: boolean = true): string {
  if (value >= 1000) {
    // Convert to billions
    const billions = value / 1000;
    return `$${Math.round(billions)}B`;
  } else {
    // Keep in millions
    return includeDecimal ? `$${value.toFixed(1)}M` : `$${Math.round(value)}M`;
  }
}

export function SensitivityAnalysis({ project }: SensitivityAnalysisProps) {
  const [actualValues, setActualValues] = React.useState<Record<string, ActualValue>>({})
  const [loadingPrices, setLoadingPrices] = React.useState(true)
  const [aiCalculatedValues, setAiCalculatedValues] = React.useState<{ npv: number; irr: number; aisc: number } | null>(null)

  const parameters: SensitivityParameter[] = [
    {
      name: "Commodity Price",
      key: "price",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 0.1,
      category: 'market'
    },
    {
      name: "Throughput",
      key: "throughput",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 0.1,
      category: 'production'
    },
    {
      name: "Head Grade",
      key: "grade",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 0.1,
      category: 'production'
    },
    {
      name: "Operating Costs",
      key: "opex",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 0.1,
      category: 'costs'
    },
    {
      name: "Capital Costs",
      key: "capex",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 0.1,
      category: 'costs'
    },
    {
      name: "Recovery Rate",
      key: "recovery",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 0.1,
      category: 'production'
    },
  ]

  const [values, setValues] = React.useState<Record<string, number>>(
    parameters.reduce((acc, param) => ({ ...acc, [param.key]: param.baseline }), {})
  )
  const [aiInsight, setAiInsight] = React.useState<string>("")
  const [isGeneratingInsight, setIsGeneratingInsight] = React.useState(false)
  const debounceTimer = React.useRef<NodeJS.Timeout | null>(null)

  const baselineNPV = project.npv || 0
  const baselineIRR = project.irr || 0
  const baselineAISC = project.aisc || 0

  const calculateNPVForValues = (vals: Record<string, number>) => {
    if (project.npv === null || project.npv === undefined) return 0

    const priceImpact = (vals.price - 100) * 0.02
    const throughputImpact = (vals.throughput - 100) * 0.015
    const gradeImpact = (vals.grade - 100) * 0.018
    const recoveryImpact = (vals.recovery - 100) * 0.012
    const opexImpact = -(vals.opex - 100) * 0.012
    const capexImpact = -(vals.capex - 100) * 0.008

    const totalImpact = 1 + priceImpact + throughputImpact + gradeImpact + recoveryImpact + opexImpact + capexImpact
    return project.npv * totalImpact
  }

  const calculateIRRForValues = (vals: Record<string, number>) => {
    if (project.irr === null || project.irr === undefined) return 0

    const priceImpact = (vals.price - 100) * 0.015
    const throughputImpact = (vals.throughput - 100) * 0.01
    const gradeImpact = (vals.grade - 100) * 0.012
    const recoveryImpact = (vals.recovery - 100) * 0.008
    const opexImpact = -(vals.opex - 100) * 0.008
    const capexImpact = -(vals.capex - 100) * 0.006

    const totalImpact = priceImpact + throughputImpact + gradeImpact + recoveryImpact + opexImpact + capexImpact
    return project.irr + totalImpact
  }

  const calculateAISCForValues = (vals: Record<string, number>) => {
    if (project.aisc === null || project.aisc === undefined) return 0

    // AISC impacts (inverse for throughput, grade, recovery)
    const throughputImpact = -(vals.throughput - 100) * 0.007 // Higher throughput spreads fixed costs
    const gradeImpact = -(vals.grade - 100) * 0.008 // Higher grade reduces cost per unit
    const opexImpact = (vals.opex - 100) * 0.009 // Direct impact
    const capexImpact = (vals.capex - 100) * 0.002 // Sustaining capex component
    const recoveryImpact = -(vals.recovery - 100) * 0.007 // Higher recovery reduces cost per unit

    const totalImpact = 1 + throughputImpact + gradeImpact + opexImpact + capexImpact + recoveryImpact
    return project.aisc * totalImpact
  }

  const generateAIInsight = React.useCallback(async (currentValues: Record<string, number>) => {
    const changes = parameters
      .filter(p => Math.abs(currentValues[p.key] - p.baseline) > 0.1)
      .map(p => ({
        name: p.name,
        value: currentValues[p.key],
        change: ((currentValues[p.key] - p.baseline) / p.baseline * 100).toFixed(1)
      }))

    if (changes.length === 0) {
      const insights = [`Base case scenario for ${project.name}.`]
      if (baselineNPV) insights.push(`NPV: ${formatDollarAmount(baselineNPV)}`)
      if (baselineIRR) insights.push(`IRR: ${baselineIRR.toFixed(1)}%`)
      if (baselineAISC) insights.push(`AISC: $${baselineAISC.toFixed(2)}`)
      if (project.capex) insights.push(`CAPEX: ${formatDollarAmount(project.capex)}`)
      if (project.mine_life) insights.push(`Mine Life: ${project.mine_life} years`)
      setAiInsight(insights.join(' | '))
      return
    }

    setIsGeneratingInsight(true)
    try {
      // Calculate parameter changes as percentages from baseline
      const parameterChanges = {
        commodityPrice: currentValues.price - 100,
        throughput: currentValues.throughput - 100,
        grade: currentValues.grade - 100,
        opex: currentValues.opex - 100,
        capex: currentValues.capex - 100,
        recovery: currentValues.recovery - 100
      }

      // Call new sensitivity analysis API with GPT integration
      const response = await fetch('/api/sensitivity-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          baseCase: {
            npv: baselineNPV,
            irr: baselineIRR,
            aisc: baselineAISC
          },
          parameters: parameterChanges,
          projectContext: {
            name: project.name,
            commodity: project.commodities?.[0],
            mineLife: project.mine_life,
            annualProduction: project.annual_production
          },
          actualValues: {
            commodityPrice: actualValues.price,
            aisc: actualValues.opex,
            capex: actualValues.capex
          }
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      if (data.success && data.result) {
        const { npv, irr, aisc, explanation, assumptions, riskFactors } = data.result

        // Store AI-calculated values
        setAiCalculatedValues({ npv, irr, aisc })

        let insightText = explanation

        if (assumptions && assumptions.length > 0) {
          insightText += `\n\nKey Assumptions: ${assumptions.join('; ')}`
        }

        if (riskFactors && riskFactors.length > 0) {
          insightText += `\n\nRisk Factors: ${riskFactors.join('; ')}`
        }

        setAiInsight(insightText)
      } else {
        throw new Error('Invalid API response')
      }
    } catch (error) {
      console.error('AI Insight error:', error)
      // Fallback to detailed calculation-based insight
      const npvCalc = calculateNPVForValues(currentValues)
      const irrCalc = calculateIRRForValues(currentValues)
      const aiscCalc = calculateAISCForValues(currentValues)
      const npvChange = baselineNPV ? ((npvCalc - baselineNPV) / baselineNPV) * 100 : 0
      const irrChange = baselineIRR ? irrCalc - baselineIRR : 0
      const aiscChange = baselineAISC ? ((aiscCalc - baselineAISC) / baselineAISC) * 100 : 0

      const insights = [
        `${project.name} with ${changes.length} parameter${changes.length > 1 ? 's' : ''} adjusted:`
      ]

      if (baselineNPV) {
        insights.push(`NPV: ${formatDollarAmount(npvCalc, false)} (${npvChange > 0 ? '+' : ''}${npvChange.toFixed(1)}%)`)
      }
      if (baselineIRR) {
        insights.push(`IRR: ${irrCalc.toFixed(1)}% (${irrChange > 0 ? '+' : ''}${irrChange.toFixed(1)} pts)`)
      }
      if (baselineAISC) {
        insights.push(`AISC: $${aiscCalc.toFixed(2)} (${aiscChange > 0 ? '+' : ''}${aiscChange.toFixed(1)}%)`)
      }

      setAiInsight(insights.join(' | '))
    } finally {
      setIsGeneratingInsight(false)
    }
  }, [parameters, baselineNPV, baselineIRR, baselineAISC, calculateNPVForValues, calculateAISCForValues, project])

  const debouncedGenerateInsight = React.useCallback((newValues: Record<string, number>) => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      generateAIInsight(newValues)
    }, 1000)
  }, [generateAIInsight])

  const handleSliderChange = (param: string, value: number) => {
    const newValues = { ...values, [param]: value }
    setValues(newValues)
    debouncedGenerateInsight(newValues)
  }

  const handleInputChange = (param: string, value: string) => {
    const numValue = parseFloat(value)
    if (!isNaN(numValue)) {
      const newValues = { ...values, [param]: numValue }
      setValues(newValues)
      debouncedGenerateInsight(newValues)
    }
  }

  const calculateNPV = () => aiCalculatedValues?.npv ?? calculateNPVForValues(values)
  const calculateIRR = () => aiCalculatedValues?.irr ?? calculateIRRForValues(values)
  const calculateAISC = () => aiCalculatedValues?.aisc ?? calculateAISCForValues(values)

  const getPercentChange = (current: number, base: number) => {
    return ((current - base) / base * 100).toFixed(1)
  }

  const npv = calculateNPV()
  const irr = calculateIRR()
  const aisc = calculateAISC()
  const npvTrend = baselineNPV > 0 ? ((npv - baselineNPV) / baselineNPV) * 100 : 0
  const irrTrend = baselineIRR > 0 ? irr - baselineIRR : 0
  const aiscTrend = baselineAISC > 0 ? ((aisc - baselineAISC) / baselineAISC) * 100 : 0

  const groupedParams = {
    market: parameters.filter(p => p.category === 'market'),
    production: parameters.filter(p => p.category === 'production'),
    costs: parameters.filter(p => p.category === 'costs'),
  }

  // Fetch actual commodity prices and other baseline values
  React.useEffect(() => {
    async function fetchActualValues() {
      try {
        setLoadingPrices(true)
        const response = await fetch('/api/commodity-prices', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            projectId: project.id,
            commodities: project.commodities || [],
            aisc: project.aisc,
            capex: project.capex
          })
        })

        if (response.ok) {
          const data = await response.json()
          setActualValues(data.actualValues || {})
        }
      } catch (error) {
        console.error('Error fetching actual values:', error)
      } finally {
        setLoadingPrices(false)
      }
    }

    fetchActualValues()
  }, [project.id, project.commodities])

  React.useEffect(() => {
    if (!loadingPrices) {
      generateAIInsight(values)
    }
  }, [loadingPrices])

  const InputRow = ({ param }: { param: SensitivityParameter }) => {
    const [localValue, setLocalValue] = React.useState(values[param.key])

    // Sync local value with global state when values change externally (e.g., reset button)
    React.useEffect(() => {
      setLocalValue(values[param.key])
    }, [values[param.key]])

    const handleLocalSliderChange = (newValue: number[]) => {
      setLocalValue(newValue[0])
    }

    const handleSliderCommit = (newValue: number[]) => {
      handleSliderChange(param.key, newValue[0])
    }

    const actualValue = actualValues[param.key]

    // Format the current actual value based on parameter type
    const getCurrentActualValueDisplay = () => {
      if (!actualValue) return null;

      const currentValue = actualValue.value * (values[param.key] / 100);

      // For CAPEX (in millions), check if we need to convert to billions
      if (param.key === 'capex' && actualValue.unit === 'USD M') {
        if (currentValue === 0) {
          return 'USD M';
        }
        if (currentValue >= 1000) {
          return `$${Math.round(currentValue / 1000)}B`;
        }
        return `$${Math.round(currentValue)}M`;
      }

      // For commodity price
      if (param.key === 'price') {
        if (currentValue === 0) {
          return actualValue.unit;
        }
        return `$${Math.round(currentValue)} ${actualValue.unit}`;
      }

      // For operating costs (AISC)
      if (param.key === 'opex') {
        if (currentValue === 0) {
          return actualValue.unit;
        }
        return `$${currentValue.toFixed(2)} ${actualValue.unit}`;
      }

      // For production parameters (throughput, grade, recovery) - always show values now
      // No special handling needed, fall through to default formatting

      // For other values, 2 decimals
      return `${currentValue.toFixed(2)} ${actualValue.unit}`;
    };

    // Get display label with commodity name for price
    const getDisplayLabel = () => {
      if (param.key === 'price' && project.commodities?.[0]) {
        return `${project.commodities[0]} Price`
      }
      return param.name
    }

    return (
      <div className="mb-5">
        <div className="flex justify-between items-center mb-2">
          <div className="flex flex-col">
            <label className="text-sm font-medium">{getDisplayLabel()}</label>
            {actualValue && (
              <span className="text-xs text-muted-foreground">
                {getCurrentActualValueDisplay()}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              value={values[param.key]}
              onChange={(e) => handleInputChange(param.key, e.target.value)}
              className="w-16 h-7 px-2 text-xs text-right"
              step={param.step}
            />
            <span className="text-xs text-muted-foreground">{param.unit}</span>
            {Math.abs(parseFloat(getPercentChange(values[param.key], param.baseline))) > 0.1 && (
              <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                {getPercentChange(values[param.key], param.baseline)}%
              </span>
            )}
          </div>
        </div>
        <Slider
          value={[localValue]}
          onValueChange={handleLocalSliderChange}
          onValueCommit={handleSliderCommit}
          min={param.min}
          max={param.max}
          step={param.step}
          className="mb-1.5"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{param.min}{param.unit}</span>
          <span>Base: {param.baseline}{param.unit}</span>
          <span>{param.max}{param.unit}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Sensitivity Analysis</h3>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            const baseValues = parameters.reduce((acc, param) => ({ ...acc, [param.key]: param.baseline }), {})
            setValues(baseValues)
            generateAIInsight(baseValues)
          }}
        >
          Reset to Base
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Left - Parameters */}
        <div className="space-y-4">
          <div>
            <h4 className="text-xs font-semibold text-muted-foreground mb-3 uppercase tracking-wide">Market</h4>
            {groupedParams.market.map((param) => (
              <InputRow key={param.key} param={param} />
            ))}
          </div>

          <div className="border-t pt-4">
            <h4 className="text-xs font-semibold text-muted-foreground mb-3 uppercase tracking-wide">Production</h4>
            {groupedParams.production.map((param) => (
              <InputRow key={param.key} param={param} />
            ))}
          </div>

          <div className="border-t pt-4">
            <h4 className="text-xs font-semibold text-muted-foreground mb-3 uppercase tracking-wide">Costs</h4>
            {groupedParams.costs.map((param) => (
              <InputRow key={param.key} param={param} />
            ))}
          </div>
        </div>

        {/* Right - Results */}
        <div className="space-y-4">
          {/* Metrics */}
          <div className="grid grid-cols-3 gap-3">
            <Card>
              <CardContent className="p-4">
                <div className="text-xs text-muted-foreground mb-1">NPV</div>
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-300">
                  {baselineNPV > 0 ? formatDollarAmount(npv, false) : 'N/A'}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {baselineNPV > 0 && npvTrend !== 0
                    ? `${npvTrend > 0 ? '+' : ''}${npvTrend.toFixed(1)}%`
                    : '0.0%'}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="text-xs text-muted-foreground mb-1">IRR</div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-300">
                  {baselineIRR > 0 ? `${irr.toFixed(1)}%` : 'N/A'}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {baselineIRR > 0 && irrTrend !== 0
                    ? `${irrTrend > 0 ? '+' : ''}${irrTrend.toFixed(1)} pts`
                    : '0.0 pts'}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="text-xs text-muted-foreground mb-1">AISC</div>
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-300">
                  {baselineAISC ? `$${aisc.toFixed(2)}` : 'N/A'}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {baselineAISC && aiscTrend !== 0
                    ? `${aiscTrend > 0 ? '+' : ''}${aiscTrend.toFixed(1)}%`
                    : baselineAISC ? '0.0%' : 'N/A'}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* AI Insight */}
          <Card className="bg-muted/30">
            <CardContent className="p-3">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-semibold">AI Insight</h4>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {isGeneratingInsight ? "Analyzing scenario..." : aiInsight || "Base case scenario."}
              </p>
            </CardContent>
          </Card>

          {/* Comparison Table */}
          <Card>
            <CardContent className="p-3">
              <h4 className="text-xs font-semibold mb-2">Scenario Comparison</h4>
              <div className="text-xs">
                <table className="w-full">
                  <thead className="border-b">
                    <tr className="text-muted-foreground">
                      <th className="text-left py-2 font-medium">Parameter</th>
                      <th className="text-right py-2 font-medium">Base</th>
                      <th className="text-right py-2 font-medium">Current</th>
                      <th className="text-right py-2 font-medium">Δ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    <tr>
                      <td className="py-2">Commodity Price</td>
                      <td className="py-2 text-right text-muted-foreground">{parameters[0].baseline}%</td>
                      <td className="py-2 text-right font-medium">{values.price}%</td>
                      <td className={`py-2 text-right font-medium ${
                        parseFloat(getPercentChange(values.price, parameters[0].baseline)) >= 0
                          ? 'text-green-600 dark:text-green-300'
                          : 'text-red-600 dark:text-red-300'
                      }`}>
                        {getPercentChange(values.price, parameters[0].baseline)}%
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2">Head Grade</td>
                      <td className="py-2 text-right text-muted-foreground">{parameters[2].baseline}%</td>
                      <td className="py-2 text-right font-medium">{values.grade}%</td>
                      <td className={`py-2 text-right font-medium ${
                        parseFloat(getPercentChange(values.grade, parameters[2].baseline)) >= 0
                          ? 'text-green-600 dark:text-green-300'
                          : 'text-red-600 dark:text-red-300'
                      }`}>
                        {getPercentChange(values.grade, parameters[2].baseline)}%
                      </td>
                    </tr>
                    <tr className="bg-muted/30">
                      <td className="py-2 font-semibold">NPV</td>
                      <td className="py-2 text-right text-muted-foreground">{formatDollarAmount(baselineNPV, false)}</td>
                      <td className="py-2 text-right font-semibold">{formatDollarAmount(npv, false)}</td>
                      <td className={`py-2 text-right font-semibold ${npvTrend >= 0 ? 'text-green-600 dark:text-green-300' : 'text-red-600 dark:text-red-300'}`}>
                        {npvTrend.toFixed(1)}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
} 