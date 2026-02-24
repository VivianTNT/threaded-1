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
import { ShoppingBag, Trash2, Minus, Plus, ExternalLink, X } from 'lucide-react'
import { toast } from 'sonner'

export default function CartPage() {
  const { user, isLoading } = useRequireAuth()
  const { items, removeFromCart, updateQuantity, getCartTotal, clearCart } = useCart()
  const [isCheckingOut, setIsCheckingOut] = React.useState(false)

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

  const handleCheckout = async () => {
    if (!items.length || isCheckingOut) return

    setIsCheckingOut(true)
    try {
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
                      <div className="lg:col-span-1">
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
                              {isCheckingOut ? 'Redirecting...' : 'Proceed to Checkout'}
                            </Button>
                            <Button variant="outline" className="w-full" asChild>
                              <a href="/">Continue Shopping</a>
                            </Button>
                          </div>

                          <p className="text-xs text-muted-foreground text-center mt-4">
                            Secure checkout powered by Stripe.
                          </p>
                        </Card>
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
