'use client'

import * as React from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useCart } from '@/lib/cart-context'

function CheckoutSuccessContent() {
  const { items, clearCart } = useCart()
  const searchParams = useSearchParams()
  const sessionId = searchParams.get('session_id')
  const clearedRef = React.useRef(false)

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
