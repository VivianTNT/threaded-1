import Link from 'next/link'
import { XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'

export default function CheckoutCancelPage() {
  return (
    <main className="min-h-screen flex items-center justify-center p-4 bg-muted/20">
      <Card className="w-full max-w-lg p-8 text-center space-y-5">
        <div className="mx-auto w-14 h-14 rounded-full bg-muted flex items-center justify-center">
          <XCircle className="h-8 w-8 text-muted-foreground" />
        </div>
        <div>
          <h1 className="text-2xl font-semibold mb-2">Checkout Canceled</h1>
          <p className="text-sm text-muted-foreground">
            Your payment was not completed. Your cart items are still saved.
          </p>
        </div>
        <div className="flex gap-3 justify-center">
          <Button asChild>
            <Link href="/cart">Return to Cart</Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href="/">Continue Shopping</Link>
          </Button>
        </div>
      </Card>
    </main>
  )
}
