'use client'

import * as React from 'react'
import { AppSidebar } from '@/components/app-sidebar'
import { SiteHeader } from '@/components/site-header'
import { ChatLayout } from '@/components/chat-layout'
import { useRequireAuth } from '@/lib/auth-utils'
import { useCart } from '@/lib/cart-context'
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { ShoppingBag, Trash2, Minus, Plus, ExternalLink, X, Bot, CheckCircle2, Loader2 } from 'lucide-react'
import { toast } from 'sonner'

interface AgentItemStatus {
  productId: string
  productName: string
  productUrl: string
  status: 'pending' | 'running' | 'success' | 'failed'
  stepCount?: number
  detail?: string
  elapsed?: number
}

export default function CartPage() {
  const { user, isLoading } = useRequireAuth()
  const { items, removeFromCart, updateQuantity, getCartTotal, clearCart } = useCart()
  const [isCheckingOut, setIsCheckingOut] = React.useState(false)
  const [agentItems, setAgentItems] = React.useState<AgentItemStatus[]>([])
  const [showAgentPanel, setShowAgentPanel] = React.useState(false)

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  const total = getCartTotal()
  const itemCount = items.reduce((sum, item) => sum + item.quantity, 0)

  // Run the agentic checkout for each item one-by-one
  const runAgenticCheckout = async () => {
    const itemsWithUrls = items.filter(item => item.product.product_url)
    if (!itemsWithUrls.length) return

    // Initialize statuses
    const statuses: AgentItemStatus[] = itemsWithUrls.map(item => ({
      productId: item.product.id,
      productName: item.product.name,
      productUrl: item.product.product_url!,
      status: 'pending',
    }))
    setAgentItems(statuses)
    setShowAgentPanel(true)

    // Process each item one-by-one via the agentic API
    for (const item of itemsWithUrls) {
      // Mark as running
      setAgentItems(prev =>
        prev.map(t =>
          t.productId === item.product.id ? { ...t, status: 'running' } : t
        )
      )
      toast.info(`Agent starting: ${item.product.name}`, {
        description: `Purchasing from ${new URL(item.product.product_url!).hostname}...`,
      })

      try {
        const res = await fetch('/api/agentic-checkout/purchase', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            product_url: item.product.product_url,
            product_name: item.product.name,
            selected_size: item.selectedSize || null,
            quantity: item.quantity,
            dry_run: false,
            headless: false, // visible browser so user can watch
            user_id: user?.id || null,
          }),
        })
        const data = await res.json()

        setAgentItems(prev =>
          prev.map(t =>
            t.productId === item.product.id
              ? {
                  ...t,
                  status: data.success ? 'success' : 'failed',
                  stepCount: data.steps?.length || 0,
                  detail: data.success
                    ? `${data.cart_status} in ${data.steps?.length || 0} steps`
                    : data.error || 'Failed',
                  elapsed: data.elapsed_ms,
                }
              : t
          )
        )

        if (data.success) {
          toast.success(`Agent completed: ${item.product.name}`, {
            description: `${data.cart_status} — ${data.steps?.length} steps, ${Math.round((data.elapsed_ms || 0) / 1000)}s`,
          })
        } else {
          toast.error(`Agent failed: ${item.product.name}`, {
            description: data.error || 'Could not complete purchase',
          })
        }
      } catch (err: any) {
        setAgentItems(prev =>
          prev.map(t =>
            t.productId === item.product.id
              ? { ...t, status: 'failed', detail: err.message || 'Network error' }
              : t
          )
        )
        toast.error(`Agent error: ${item.product.name}`, {
          description: err.message || 'Request failed',
        })
      }
    }
  }

  const handleCheckout = async () => {
    if (!items.length || isCheckingOut) return

    setIsCheckingOut(true)
    try {
      // Step 1: Run agentic checkout for each item (opens visible browser per item)
      await runAgenticCheckout()

      // Step 2: Create Stripe checkout session (payment on our side)
      const response = await fetch('/api/checkout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userEmail: user?.email || null,
          items: items.map((item) => ({
            id: item.product.id,
            name: item.product.name,
            brand: item.product.brand,
            image_url: item.product.image_url,
            product_url: item.product.product_url,
            price: item.product.price,
            quantity: item.quantity,
            currency: item.product.currency || 'USD',
            selectedSize: item.selectedSize || null,
          })),
        }),
      })

      const data = await response.json()
      if (!response.ok || !data?.success || typeof data?.checkoutUrl !== 'string') {
        throw new Error(data?.message || 'Failed to start checkout.')
      }

      await new Promise(r => setTimeout(r, 2000))
      window.location.href = data.checkoutUrl
    } catch (error: any) {
      toast.error('Checkout failed', {
        description: error?.message || 'Unable to create Stripe checkout session.',
      })
      setIsCheckingOut(false)
    }
  }

  return (
    <ChatLayout>
      <SidebarProvider
        style={
          {
            "--sidebar-width": "calc(var(--spacing) * 72)",
            "--header-height": "calc(var(--spacing) * 12)",
          } as React.CSSProperties
        }
      >
        <AppSidebar variant="inset" />
        <SidebarInset>
          <SiteHeader />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
                <div className="px-4 lg:px-6">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-2xl font-semibold flex items-center gap-2">
                        <ShoppingBag className="h-6 w-6" />
                        Shopping Cart
                      </h2>
                      <p className="text-sm text-muted-foreground">
                        {itemCount > 0
                          ? `${itemCount} item${itemCount > 1 ? 's' : ''} in your cart`
                          : 'Your cart is empty'}
                      </p>
                    </div>
                    {items.length > 0 && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={clearCart}
                        className="text-muted-foreground"
                      >
                        <X className="h-4 w-4 mr-2" />
                        Clear Cart
                      </Button>
                    )}
                  </div>

                  {items.length === 0 ? (
                    <Card className="p-12 text-center">
                      <div className="flex flex-col items-center gap-4">
                        <div className="rounded-full bg-muted p-6">
                          <ShoppingBag className="h-12 w-12 text-muted-foreground" />
                        </div>
                        <div>
                          <h3 className="font-semibold mb-1">Your cart is empty</h3>
                          <p className="text-sm text-muted-foreground mb-4">
                            Discover amazing fashion items tailored to your style
                          </p>
                        </div>
                        <Button asChild>
                          <a href="/">Browse Recommendations</a>
                        </Button>
                      </div>
                    </Card>
                  ) : (
                    <div className="grid lg:grid-cols-3 gap-6">
                      {/* Cart Items */}
                      <div className="lg:col-span-2 space-y-4">
                        {items.map((item) => (
                          <Card key={item.product.id} className="p-4">
                            <div className="flex gap-4">
                              {/* Product Image */}
                              <div className="relative w-24 h-32 flex-shrink-0 rounded-lg overflow-hidden bg-muted">
                                <img
                                  src={item.product.image_url}
                                  alt={item.product.name}
                                  className="w-full h-full object-cover"
                                />
                              </div>

                              {/* Product Details */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex-1 min-w-0">
                                    <p className="text-xs text-muted-foreground font-medium">
                                      {item.product.brand}
                                    </p>
                                    <h3 className="font-semibold text-sm line-clamp-2 mb-1">
                                      {item.product.name}
                                    </h3>
                                    <div className="flex flex-wrap gap-2 mb-2">
                                      {item.product.color && item.product.color.length > 0 && (
                                        <Badge variant="outline" className="text-xs">
                                          {item.product.color[0]}
                                        </Badge>
                                      )}
                                      {item.selectedSize && (
                                        <Badge variant="outline" className="text-xs">
                                          Size {item.selectedSize}
                                        </Badge>
                                      )}
                                    </div>
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-8 w-8 flex-shrink-0"
                                    onClick={() => removeFromCart(item.product.id)}
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>

                                <div className="flex items-center justify-between mt-3">
                                  {/* Quantity Controls */}
                                  <div className="flex items-center gap-2">
                                    <Button
                                      variant="outline"
                                      size="icon"
                                      className="h-8 w-8"
                                      onClick={() =>
                                        updateQuantity(item.product.id, item.quantity - 1)
                                      }
                                    >
                                      <Minus className="h-3 w-3" />
                                    </Button>
                                    <span className="text-sm font-medium w-8 text-center">
                                      {item.quantity}
                                    </span>
                                    <Button
                                      variant="outline"
                                      size="icon"
                                      className="h-8 w-8"
                                      onClick={() =>
                                        updateQuantity(item.product.id, item.quantity + 1)
                                      }
                                    >
                                      <Plus className="h-3 w-3" />
                                    </Button>
                                  </div>

                                  {/* Price */}
                                  <div className="text-right">
                                    <p className="font-semibold">
                                      ${(item.product.price * item.quantity).toLocaleString()}
                                    </p>
                                    {item.quantity > 1 && (
                                      <p className="text-xs text-muted-foreground">
                                        ${item.product.price} each
                                      </p>
                                    )}
                                  </div>
                                </div>

                                {/* View on Site Link */}
                                {item.product.product_url && (
                                  <Button
                                    variant="link"
                                    size="sm"
                                    className="h-auto p-0 mt-2 text-xs"
                                    onClick={() => window.open(item.product.product_url!, '_blank')}
                                  >
                                    <ExternalLink className="h-3 w-3 mr-1" />
                                    View on brand website
                                  </Button>
                                )}
                              </div>
                            </div>
                          </Card>
                        ))}
                      </div>

                      {/* Order Summary */}
                      <div className="lg:col-span-1 space-y-4">
                        <Card className="p-6 sticky top-4">
                          <h3 className="font-semibold mb-4">Order Summary</h3>

                          <div className="space-y-3 mb-4">
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground">Subtotal</span>
                              <span className="font-medium">${total.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground">Shipping</span>
                              <span className="text-sm text-muted-foreground">Calculated at checkout</span>
                            </div>

                            <Separator />

                            <div className="flex justify-between">
                              <span className="font-semibold">Total</span>
                              <span className="font-bold text-lg">${total.toLocaleString()}</span>
                            </div>
                          </div>

                          <div className="space-y-3">
                            <Button className="w-full" size="lg" onClick={handleCheckout} disabled={isCheckingOut}>
                              {isCheckingOut ? (
                                <>
                                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                  Processing...
                                </>
                              ) : (
                                <>
                                  <Bot className="h-4 w-4 mr-2" />
                                  Checkout & Purchase Items
                                </>
                              )}
                            </Button>
                            <Button variant="outline" className="w-full" asChild>
                              <a href="/">Continue Shopping</a>
                            </Button>
                          </div>

                          <div className="flex items-center gap-2 mt-4 text-xs text-muted-foreground text-center justify-center">
                            <Bot className="h-3 w-3" />
                            <span>Agent opens each item on its retailer site</span>
                          </div>
                          <p className="text-xs text-muted-foreground text-center mt-1">
                            Secure payment powered by Stripe
                          </p>
                        </Card>

                        {/* Agent Status Panel */}
                        {showAgentPanel && agentItems.length > 0 && (
                          <Card className="p-4">
                            <div className="flex items-center gap-2 mb-3">
                              <Bot className="h-4 w-4 text-primary" />
                              <h4 className="font-semibold text-sm">Agent Purchase Status</h4>
                            </div>
                            <div className="space-y-2">
                              {agentItems.map((item) => (
                                <div
                                  key={item.productId}
                                  className={`flex items-center gap-2 p-2 rounded-md text-xs border ${
                                    item.status === 'success'
                                      ? 'bg-green-50 border-green-200 text-green-700'
                                      : item.status === 'running'
                                      ? 'bg-blue-50 border-blue-200 text-blue-700'
                                      : item.status === 'failed'
                                      ? 'bg-red-50 border-red-200 text-red-700'
                                      : 'bg-muted/50 border-border text-muted-foreground'
                                  }`}
                                >
                                  {item.status === 'success' ? (
                                    <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0" />
                                  ) : item.status === 'running' ? (
                                    <Loader2 className="h-3.5 w-3.5 flex-shrink-0 animate-spin" />
                                  ) : item.status === 'failed' ? (
                                    <X className="h-3.5 w-3.5 flex-shrink-0" />
                                  ) : (
                                    <Bot className="h-3.5 w-3.5 flex-shrink-0" />
                                  )}
                                  <div className="flex-1 min-w-0">
                                    <div className="font-medium truncate">{item.productName}</div>
                                    <div className="text-[10px] opacity-70 truncate">
                                      {new URL(item.productUrl).hostname}
                                      {item.elapsed ? ` · ${Math.round(item.elapsed / 1000)}s` : ''}
                                      {item.stepCount ? ` · ${item.stepCount} steps` : ''}
                                    </div>
                                    {item.detail && (
                                      <div className="text-[10px] opacity-60 truncate mt-0.5">{item.detail}</div>
                                    )}
                                  </div>
                                  <Badge
                                    variant="outline"
                                    className={`text-[10px] flex-shrink-0 ${
                                      item.status === 'success'
                                        ? 'bg-green-100 text-green-700 border-green-300'
                                        : item.status === 'failed'
                                        ? 'bg-red-100 text-red-700 border-red-300'
                                        : ''
                                    }`}
                                  >
                                    {item.status === 'success'
                                      ? 'Purchased'
                                      : item.status === 'running'
                                      ? 'Agent working...'
                                      : item.status === 'failed'
                                      ? 'Failed'
                                      : 'Queued'}
                                  </Badge>
                                </div>
                              ))}
                            </div>
                            <p className="text-[10px] text-muted-foreground mt-2">
                              The AI agent opens a browser and purchases each item from the retailer site automatically.
                            </p>
                          </Card>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>
    </ChatLayout>
  )
}