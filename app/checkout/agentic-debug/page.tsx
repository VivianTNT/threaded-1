'use client'

import * as React from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Bot,
  CheckCircle2,
  XCircle,
  Clock,
  SkipForward,
  Loader2,
  Globe,
  ShoppingCart,
  Eye,
  MousePointer,
  Shield,
  ArrowRight,
} from 'lucide-react'

interface PurchaseStep {
  step: number
  action: string
  description: string
  status: 'success' | 'failed' | 'skipped'
  detail?: string
  timestamp: string
}

interface PurchaseResult {
  success: boolean
  product_url: string
  steps: PurchaseStep[]
  cart_status?: string
  error?: string
  elapsed_ms: number
}

const STEP_ICONS: Record<string, React.ReactNode> = {
  launch_browser: <Globe className="h-4 w-4" />,
  navigate: <ArrowRight className="h-4 w-4" />,
  dismiss_overlays: <Shield className="h-4 w-4" />,
  analyze_page: <Eye className="h-4 w-4" />,
  check_availability: <CheckCircle2 className="h-4 w-4" />,
  select_size: <MousePointer className="h-4 w-4" />,
  set_quantity: <MousePointer className="h-4 w-4" />,
  dry_run: <Clock className="h-4 w-4" />,
  add_to_cart: <ShoppingCart className="h-4 w-4" />,
  verify_cart: <CheckCircle2 className="h-4 w-4" />,
  error: <XCircle className="h-4 w-4" />,
  init: <Bot className="h-4 w-4" />,
}

const STATUS_COLORS: Record<string, string> = {
  success: 'bg-green-500/10 text-green-600 border-green-200',
  failed: 'bg-red-500/10 text-red-600 border-red-200',
  skipped: 'bg-yellow-500/10 text-yellow-600 border-yellow-200',
}

const STATUS_BADGE: Record<string, string> = {
  success: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
  skipped: 'bg-yellow-100 text-yellow-700',
}

// Default test items from the database
const TEST_ITEMS = [
  {
    name: 'NYLON DUCK BIFOLD WALLET',
    url: 'https://www.carhartt.com/product/801983/nylon-duck-bifold-wallet',
    size: 'OS',
    price: '$28.49',
  },
  {
    name: 'Leatherman Rebar Multitool',
    url: 'https://www.llbean.com/llb/shop/128862',
    size: 'OS',
    price: '$79.95',
  },
  {
    name: 'CashSoft Turtleneck Sweater',
    url: 'https://www.gap.com/browse/product.do?pid=844100002&vid=1&pcid=1165627&cid=3051166#pdp-page-content',
    size: 'M',
    price: '$54.99',
  },
  {
    name: 'Zip-Up Short Jacket',
    url: 'https://www.uniqlo.com/us/en/products/E479208-000/00?colorDisplayCode=69&sizeDisplayCode=003',
    size: 'M',
    price: '$49.90',
  },
]

export default function AgenticDebugPage() {
  const [isRunning, setIsRunning] = React.useState(false)
  const [result, setResult] = React.useState<PurchaseResult | null>(null)
  const [selectedItem, setSelectedItem] = React.useState(TEST_ITEMS[0])
  const [customUrl, setCustomUrl] = React.useState('')
  const [customSize, setCustomSize] = React.useState('')
  const [dryRun, setDryRun] = React.useState(true)
  const [visibleSteps, setVisibleSteps] = React.useState<PurchaseStep[]>([])

  const runPurchase = async () => {
    setIsRunning(true)
    setResult(null)
    setVisibleSteps([])

    const url = customUrl || selectedItem.url
    const size = customSize || selectedItem.size

    try {
      const response = await fetch('/api/agentic-checkout/purchase', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product_url: url,
          product_name: selectedItem.name || 'Test Product',
          selected_size: size,
          quantity: 1,
          dry_run: dryRun,
        }),
      })

      const data: PurchaseResult = await response.json()
      setResult(data)

      // Animate steps appearing one by one
      for (let i = 0; i < data.steps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 300))
        setVisibleSteps(prev => [...prev, data.steps[i]])
      }
    } catch (error: any) {
      setResult({
        success: false,
        product_url: url,
        steps: [],
        error: error.message || 'Request failed',
        elapsed_ms: 0,
      })
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <main className="min-h-screen bg-muted/20 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center">
            <Bot className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Agentic Checkout — Debug Console</h1>
            <p className="text-sm text-muted-foreground">
              Step-by-step browser automation for purchasing items from retailer websites
            </p>
          </div>
        </div>

        {/* Config Panel */}
        <Card className="p-6 space-y-4">
          <h2 className="font-semibold text-lg">Test Configuration</h2>

          {/* Quick select from DB products */}
          <div>
            <label className="text-sm font-medium mb-2 block">
              Select a product from your database:
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {TEST_ITEMS.map((item, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setSelectedItem(item)
                    setCustomUrl('')
                    setCustomSize('')
                  }}
                  className={`text-left p-3 rounded-lg border transition-colors ${
                    selectedItem.url === item.url && !customUrl
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:border-primary/40'
                  }`}
                >
                  <div className="font-medium text-sm truncate">{item.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {item.price} · Size: {item.size}
                  </div>
                  <div className="text-xs text-muted-foreground/60 truncate mt-0.5">
                    {new URL(item.url).hostname}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Custom URL override */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="sm:col-span-2">
              <label className="text-sm font-medium mb-1 block">Product URL (or use selection above)</label>
              <input
                type="url"
                value={customUrl}
                onChange={e => setCustomUrl(e.target.value)}
                placeholder="https://www.example.com/product/..."
                className="w-full px-3 py-2 rounded-md border text-sm bg-background"
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1 block">Size Override</label>
              <input
                type="text"
                value={customSize}
                onChange={e => setCustomSize(e.target.value)}
                placeholder="e.g. M, L, XL, 10"
                className="w-full px-3 py-2 rounded-md border text-sm bg-background"
              />
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-4">
            <Button onClick={runPurchase} disabled={isRunning} size="lg">
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Agent Running...
                </>
              ) : (
                <>
                  <Bot className="h-4 w-4 mr-2" />
                  Run Agentic Purchase
                </>
              )}
            </Button>

            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={e => setDryRun(e.target.checked)}
                className="rounded border-border"
              />
              <span className="font-medium">Dry Run</span>
              <span className="text-muted-foreground">(stops before actual purchase)</span>
            </label>
          </div>
        </Card>

        {/* Results Panel */}
        {(visibleSteps.length > 0 || result) && (
          <Card className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-lg">Agent Execution Log</h2>
              {result && (
                <div className="flex items-center gap-3">
                  <Badge
                    variant="outline"
                    className={result.success ? 'bg-green-50 text-green-700 border-green-200' : 'bg-red-50 text-red-700 border-red-200'}
                  >
                    {result.success ? 'SUCCESS' : 'FAILED'}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {result.elapsed_ms}ms
                  </span>
                  {result.cart_status && (
                    <Badge variant="outline" className="text-xs">
                      {result.cart_status}
                    </Badge>
                  )}
                </div>
              )}
            </div>

            {/* Step-by-step log */}
            <div className="space-y-2">
              {visibleSteps.map((step, i) => (
                <div
                  key={i}
                  className={`flex items-start gap-3 p-3 rounded-lg border transition-all animate-in slide-in-from-left-2 ${STATUS_COLORS[step.status]}`}
                >
                  <div className="mt-0.5 flex-shrink-0">
                    {step.status === 'success' ? (
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                    ) : step.status === 'failed' ? (
                      <XCircle className="h-4 w-4 text-red-600" />
                    ) : (
                      <SkipForward className="h-4 w-4 text-yellow-600" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-muted-foreground">
                        Step {step.step}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {STEP_ICONS[step.action]}
                      </span>
                      <Badge variant="outline" className={`text-[10px] ${STATUS_BADGE[step.status]}`}>
                        {step.status}
                      </Badge>
                    </div>
                    <p className="text-sm font-medium mt-0.5">{step.description}</p>
                    {step.detail && (
                      <p className="text-xs text-muted-foreground mt-0.5 truncate">{step.detail}</p>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground/60 flex-shrink-0">
                    {new Date(step.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              ))}

              {isRunning && (
                <div className="flex items-center gap-3 p-3 rounded-lg border border-primary/20 bg-primary/5">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  <span className="text-sm text-primary font-medium">Agent is working...</span>
                </div>
              )}
            </div>

            {/* Error display */}
            {result?.error && (
              <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
                <strong>Error:</strong> {result.error}
              </div>
            )}
          </Card>
        )}

        {/* Architecture Info */}
        <Card className="p-6">
          <h2 className="font-semibold text-lg mb-3">How It Works</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
            <div className="space-y-1">
              <div className="font-medium flex items-center gap-2">
                <Globe className="h-4 w-4 text-blue-500" /> 1. Browser Launch
              </div>
              <p className="text-muted-foreground">
                Playwright opens a headless Chromium instance and navigates to the product URL.
              </p>
            </div>
            <div className="space-y-1">
              <div className="font-medium flex items-center gap-2">
                <Eye className="h-4 w-4 text-purple-500" /> 2. AI Analysis
              </div>
              <p className="text-muted-foreground">
                OpenAI analyzes the page accessibility tree to identify buttons, size selectors, and cart controls.
              </p>
            </div>
            <div className="space-y-1">
              <div className="font-medium flex items-center gap-2">
                <ShoppingCart className="h-4 w-4 text-green-500" /> 3. Execute & Verify
              </div>
              <p className="text-muted-foreground">
                The agent selects size, dismisses popups, clicks add-to-cart, and verifies the item was added.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </main>
  )
}
