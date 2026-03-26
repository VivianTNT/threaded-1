'use client'

import * as React from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { CheckCircle2, Bot, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useCart } from '@/lib/cart-context'

function CheckoutSuccessContent() {
  const { items, clearCart } = useCart()
  const searchParams = useSearchParams()
  const sessionId = searchParams.get('session_id')
  const clearedRef = React.useRef(false)

  // Capture items before clearing for the retailer tab reminder
  const [purchasedItems] = React.useState(() =>
    items.filter(item => item.product.product_url).map(item => ({
      name: item.product.name,
      brand: item.product.brand,
      url: item.product.product_url!,
    }))
  )

  React.useEffect(() => {
    if (clearedRef.current) return
    if (items.length > 0) {
      clearCart()
    }
    clearedRef.current = true
  }, [items.length, clearCart])

  return (
    <main className="min-h-screen flex items-center justify-center p-4 bg-muted/20">
      <Card className="w-full max-w-lg p-8 text-center space-y-5">
        <div className="mx-auto w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center">
          <CheckCircle2 className="h-8 w-8 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-semibold mb-2">Payment Successful</h1>
          <p className="text-sm text-muted-foreground">
            Thank you for your purchase. Your order has been confirmed.
          </p>
        </div>
        {sessionId && (
          <p className="text-xs text-muted-foreground break-all">
            Session: {sessionId}
          </p>
        )}

        {/* Agent purchase reminder */}
        {purchasedItems.length > 0 && (
          <div className="bg-muted/50 rounded-lg p-4 text-left space-y-3">
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">Complete Your Purchases</span>
            </div>
            <p className="text-xs text-muted-foreground">
              We opened tabs for each item. Complete checkout on the retailer sites:
            </p>
            <div className="space-y-2">
              {purchasedItems.map((item, i) => (
                <button
                  key={i}
                  onClick={() => window.open(item.url, '_blank')}
                  className="flex items-center gap-2 w-full p-2 rounded-md border bg-background hover:bg-muted/50 transition-colors text-left"
                >
                  <ExternalLink className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate">{item.name}</div>
                    <div className="text-[10px] text-muted-foreground">{item.brand} · {new URL(item.url).hostname}</div>
                  </div>
                  <Badge variant="outline" className="text-[10px] flex-shrink-0">Open</Badge>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex gap-3 justify-center">
          <Button asChild>
            <Link href="/">Back to Recommendations</Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href="/cart">View Cart</Link>
          </Button>
        </div>
      </Card>
    </main>
  )
}

function CheckoutSuccessFallback() {
  return (
    <main className="min-h-screen flex items-center justify-center p-4 bg-muted/20">
      <Card className="w-full max-w-lg p-8 text-center space-y-5">
        <div>
          <h1 className="text-2xl font-semibold mb-2">Finalizing your order...</h1>
          <p className="text-sm text-muted-foreground">
            Please wait while we confirm your checkout session.
          </p>
        </div>
      </Card>
    </main>
  )
}

export default function CheckoutSuccessPage() {
  return (
    <React.Suspense fallback={<CheckoutSuccessFallback />}>
      <CheckoutSuccessContent />
    </React.Suspense>
  )
}
