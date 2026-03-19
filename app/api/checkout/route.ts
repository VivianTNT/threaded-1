import { NextResponse } from 'next/server'

type CheckoutItemInput = {
  id?: unknown
  name?: unknown
  brand?: unknown
  image_url?: unknown
  product_url?: unknown
  price?: unknown
  quantity?: unknown
  currency?: unknown
  selectedSize?: unknown
}

type CheckoutRequestBody = {
  items?: unknown
  userEmail?: unknown
}

type NormalizedCheckoutItem = {
  id: string
  name: string
  brand: string | null
  imageUrl: string | null
  productUrl: string | null
  unitAmountCents: number
  quantity: number
  currency: string
  selectedSize: string | null
}

function asString(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  return trimmed.length ? trimmed : null
}

function asPositiveInteger(value: unknown): number | null {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return null
  const intVal = Math.floor(parsed)
  return intVal > 0 ? intVal : null
}

function asUsdCents(value: unknown): number | null {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  const cents = Math.round(parsed * 100)
  return cents > 0 ? cents : null
}

function normalizeCurrency(value: unknown): string {
  const raw = (asString(value) || 'USD').toLowerCase()
  return /^[a-z]{3}$/.test(raw) ? raw : 'usd'
}

function normalizeItem(raw: CheckoutItemInput): NormalizedCheckoutItem | null {
  const name = asString(raw.name)
  const quantity = asPositiveInteger(raw.quantity)
  const unitAmountCents = asUsdCents(raw.price)
  if (!name || !quantity || !unitAmountCents) return null

  return {
    id: asString(raw.id) || '',
    name,
    brand: asString(raw.brand),
    imageUrl: asString(raw.image_url),
    productUrl: asString(raw.product_url),
    unitAmountCents,
    quantity,
    currency: normalizeCurrency(raw.currency),
    selectedSize: asString(raw.selectedSize),
  }
}

function getOrigin(request: Request): string {
  const headerOrigin = asString(request.headers.get('origin'))
  if (headerOrigin) return headerOrigin

  const configured = asString(process.env.NEXT_PUBLIC_APP_URL)
  if (configured) return configured

  return new URL(request.url).origin
}

export async function POST(request: Request) {
  try {
    const stripeSecretKey = asString(process.env.STRIPE_SECRET_KEY)
    if (!stripeSecretKey) {
      return NextResponse.json(
        { success: false, message: 'Missing STRIPE_SECRET_KEY configuration.' },
        { status: 500 }
      )
    }

    const body = (await request.json()) as CheckoutRequestBody
    const rawItems = Array.isArray(body?.items) ? (body.items as CheckoutItemInput[]) : []
    const items = rawItems
      .map(normalizeItem)
      .filter((item): item is NormalizedCheckoutItem => item !== null)

    if (!items.length) {
      return NextResponse.json(
        { success: false, message: 'No valid cart items were provided for checkout.' },
        { status: 400 }
      )
    }

    const origin = getOrigin(request)
    const userEmail = asString(body?.userEmail)

    const params = new URLSearchParams()
    params.set('mode', 'payment')
    params.set('success_url', `${origin}/checkout/success?session_id={CHECKOUT_SESSION_ID}`)
    params.set('cancel_url', `${origin}/checkout/cancel`)
    params.set('submit_type', 'pay')
    params.set('allow_promotion_codes', 'true')

    if (userEmail) {
      params.set('customer_email', userEmail)
    }

    items.forEach((item, idx) => {
      params.set(`line_items[${idx}][quantity]`, String(item.quantity))
      params.set(`line_items[${idx}][price_data][currency]`, item.currency)
      params.set(`line_items[${idx}][price_data][unit_amount]`, String(item.unitAmountCents))
      params.set(`line_items[${idx}][price_data][product_data][name]`, item.name)

      const descriptionParts = [item.brand, item.selectedSize ? `Size ${item.selectedSize}` : null].filter(Boolean)
      if (descriptionParts.length) {
        params.set(`line_items[${idx}][price_data][product_data][description]`, descriptionParts.join(' • '))
      }

      if (item.imageUrl && /^https?:\/\//i.test(item.imageUrl)) {
        params.set(`line_items[${idx}][price_data][product_data][images][0]`, item.imageUrl)
      }

      if (item.id) {
        params.set(`line_items[${idx}][price_data][product_data][metadata][product_id]`, item.id)
      }
      if (item.productUrl) {
        params.set(`line_items[${idx}][price_data][product_data][metadata][product_url]`, item.productUrl)
      }
    })

    const stripeResponse = await fetch('https://api.stripe.com/v1/checkout/sessions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${stripeSecretKey}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: params.toString(),
    })

    const responseText = await stripeResponse.text()
    let responseJson: any = null
    try {
      responseJson = JSON.parse(responseText)
    } catch {
      responseJson = null
    }

    if (!stripeResponse.ok) {
      const message = responseJson?.error?.message || responseText || 'Failed to create Stripe checkout session.'
      console.error('[checkout][POST] stripe_error:', {
        status: stripeResponse.status,
        message,
      })
      return NextResponse.json({ success: false, message }, { status: 500 })
    }

    const checkoutUrl = asString(responseJson?.url)
    const sessionId = asString(responseJson?.id)
    if (!checkoutUrl || !sessionId) {
      return NextResponse.json(
        { success: false, message: 'Stripe did not return a valid checkout URL.' },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      checkoutUrl,
      sessionId,
    })
  } catch (error: any) {
    console.error('[checkout][POST] error:', error)
    return NextResponse.json(
      { success: false, message: error?.message || 'Failed to start checkout.' },
      { status: 500 }
    )
  }
}
