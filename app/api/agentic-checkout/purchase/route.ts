import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

/**
 * Agentic Checkout — Stagehand-Powered Browser Agent
 *
 * Uses Stagehand (@browserbasehq/stagehand) which wraps Playwright with
 * three AI primitives:
 *   - page.act("natural language instruction")  — click, type, select
 *   - page.observe("what to look for")          — find elements
 *   - page.extract("what data to pull")         — extract structured data
 *
 * Each step is a single natural-language command. Stagehand handles overlay
 * dismissal, custom dropdowns, and element targeting via vision + DOM analysis.
 */

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

// ── Types ────────────────────────────────────────────────────────────────────

interface UserCheckoutDetails {
  email: string | null
  firstName: string | null
  lastName: string | null
  phone: string | null
  shippingAddress: string | null
  shippingCity: string | null
  shippingState: string | null
  shippingZip: string | null
  shippingCountry: string | null
}

interface PurchaseStep {
  step: number
  action: string
  description: string
  status: 'success' | 'failed' | 'skipped'
  detail?: string
  timestamp: string
}

interface PurchaseRequest {
  product_url: string
  product_name?: string
  selected_size?: string
  quantity?: number
  dry_run?: boolean
  user_id?: string
  userDetails?: UserCheckoutDetails | null
  headless?: boolean
}

interface PurchaseResult {
  success: boolean
  product_url: string
  steps: PurchaseStep[]
  cart_status?: 'item_added' | 'checkout_reached' | 'failed' | 'dry_run_stopped'
  error?: string
  elapsed_ms: number
}

// ── Supabase helper ──────────────────────────────────────────────────────────

async function getUserCheckoutDetails(userId: string): Promise<UserCheckoutDetails | null> {
  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey)
    const { data, error } = await supabase
      .from('users')
      .select('email, first_name, last_name, phone, shipping_address, shipping_city, shipping_state, shipping_zip, shipping_country')
      .eq('id', userId)
      .single()
    if (error || !data) return null
    return {
      email: data.email,
      firstName: data.first_name,
      lastName: data.last_name,
      phone: data.phone,
      shippingAddress: data.shipping_address,
      shippingCity: data.shipping_city,
      shippingState: data.shipping_state,
      shippingZip: data.shipping_zip,
      shippingCountry: data.shipping_country || 'US',
    }
  } catch {
    return null
  }
}

// ── Step logger ──────────────────────────────────────────────────────────────

function createStepLogger() {
  const steps: PurchaseStep[] = []
  let stepNum = 0
  return {
    steps,
    add(action: string, description: string, status: PurchaseStep['status'], detail?: string) {
      stepNum++
      steps.push({ step: stepNum, action, description, status, detail, timestamp: new Date().toISOString() })
      console.log(`  [step ${stepNum}] ${status.toUpperCase()} ${action}: ${description}${detail ? ' — ' + detail : ''}`)
    },
  }
}

// ── Stagehand agent ──────────────────────────────────────────────────────────

async function executePurchase(req: PurchaseRequest): Promise<PurchaseResult> {
  const startTime = Date.now()
  const log = createStepLogger()

  // Dynamic import — Stagehand is an ESM package
  let Stagehand: any
  try {
    const mod = await import('@browserbasehq/stagehand')
    Stagehand = mod.Stagehand
  } catch (e: any) {
    log.add('init', 'Failed to import Stagehand', 'failed', e.message)
    return { success: false, product_url: req.product_url, steps: log.steps, error: 'Stagehand not available: ' + e.message, elapsed_ms: Date.now() - startTime }
  }

  let stagehand: any = null

  try {
    // ── Init ─────────────────────────────────────────────────────────────
    const headless = req.headless !== false
    log.add('init', headless ? 'Launching headless browser' : 'Launching VISIBLE browser — watch your screen', 'success')

    stagehand = new Stagehand({
      env: 'LOCAL',
      modelName: 'gpt-4o',
      modelClientOptions: { apiKey: process.env.OPENAI_API_KEY },
      headless,
      enableCaching: false,
      verbose: 0,
    })
    await stagehand.init()
    // In Stagehand v3, page is accessed via context.pages()
    const page = stagehand.context.pages()[0]

    // ── Navigate ─────────────────────────────────────────────────────────
    log.add('navigate', `Opening ${req.product_url}`, 'success')
    await page.goto(req.product_url, { waitUntil: 'domcontentloaded', timeout: 30000 })
    await page.waitForTimeout(3000)

    // ── Accept cookies first ────────────────────────────────────────────
    try {
      await stagehand.act('click "Accept All Cookies", "Accept All", "Accept", or "Allow All" button on the cookie banner')
      log.add('accept_cookies', 'Accepted cookies', 'success')
    } catch {
      log.add('accept_cookies', 'No cookie banner found', 'skipped')
    }
    await page.waitForTimeout(1000)

    // ── Dismiss any remaining popups ─────────────────────────────────────
    try {
      await stagehand.act('close any remaining popup overlay, newsletter signup, or dialog that is blocking the page')
      log.add('dismiss_popups', 'Dismissed overlays/popups', 'success')
    } catch {
      log.add('dismiss_popups', 'No popups to dismiss', 'skipped')
    }
    await page.waitForTimeout(1000)

    // ── Select size ────────────────────────────────────────────────────
    {
      let sizeSelected = false
      const sizeToTry = (req.selected_size && req.selected_size.toUpperCase() !== 'OS') ? req.selected_size : 'XL'

      // Attempt 1: click the size button
      try {
        await stagehand.act(`click the "${sizeToTry}" size button`)
        log.add('select_size', `Clicked size ${sizeToTry}`, 'success')
        sizeSelected = true
      } catch {}

      // Attempt 2: try XL if first attempt wasn't XL
      if (!sizeSelected && sizeToTry !== 'XL') {
        try {
          await stagehand.act('click the "XL" size button')
          log.add('select_size', 'Clicked size XL', 'success')
          sizeSelected = true
        } catch {}
      }

      // Attempt 3: try size 8
      if (!sizeSelected) {
        try {
          await stagehand.act('click the "8" size button')
          log.add('select_size', 'Clicked size 8', 'success')
          sizeSelected = true
        } catch {}
      }

      if (!sizeSelected) {
        log.add('select_size', 'Could not select a size', 'skipped')
      }
      await page.waitForTimeout(500)
    }

    // ── Add to bag (one click only) ─────────────────────────────────────
    try {
      await stagehand.act('click the "Add to bag" or "Add to Cart" button')
      log.add('add_to_bag', 'Clicked Add to Bag', 'success')
    } catch (e: any) {
      log.add('add_to_bag', 'Failed to click Add to Bag', 'failed', e.message)
      return { success: false, product_url: req.product_url, steps: log.steps, cart_status: 'failed', error: 'Could not add to bag', elapsed_ms: Date.now() - startTime }
    }
    await page.waitForTimeout(2000)

    // ── A popup appears — click "Go to bag" inside it ────────────────
    try {
      await stagehand.act('click the "Go to bag" button in the popup')
      log.add('go_to_bag', 'Clicked Go to bag', 'success')
    } catch {
      try {
        const currentUrl = new URL(page.url())
        await page.goto(`${currentUrl.origin}/cart`, { waitUntil: 'domcontentloaded' })
        log.add('go_to_bag', 'Navigated to /cart directly', 'success')
      } catch {
        log.add('go_to_bag', 'Could not get to bag', 'failed')
      }
    }
    await page.waitForTimeout(2000)

    // ── Dry run stop ─────────────────────────────────────────────────────
    if (req.dry_run) {
      log.add('dry_run', 'Dry run — stopping after add-to-cart', 'success')
      return { success: true, product_url: req.product_url, steps: log.steps, cart_status: 'dry_run_stopped', elapsed_ms: Date.now() - startTime }
    }

    // ── Click "Proceed to Secure Checkout" on the bag page ─────────────
    try { await stagehand.act('close any popup or overlay blocking the page') } catch {}
    await page.waitForTimeout(500)

    try {
      await stagehand.act('click "Proceed to Secure Checkout", "Secure Checkout", "Checkout", or "Proceed to Checkout" button (NOT PayPal, NOT Google Pay, NOT "Continue Shopping")')
      log.add('checkout', 'Clicked Proceed to Secure Checkout', 'success')
    } catch {
      try {
        const currentUrl = new URL(page.url())
        await page.goto(`${currentUrl.origin}/checkout`, { waitUntil: 'domcontentloaded' })
        log.add('checkout', 'Navigated to /checkout directly', 'success')
      } catch (e2: any) {
        log.add('checkout', 'Could not click checkout', 'failed', e2.message)
        return { success: false, product_url: req.product_url, steps: log.steps, cart_status: 'item_added', error: 'Could not reach checkout', elapsed_ms: Date.now() - startTime }
      }
    }
    await page.waitForTimeout(3000)

    // ── Guest checkout gate ──
    // Flow: "Continue as Guest" → enter email → "Continue as Guest" again → shipping form
    const d = req.userDetails

    // Step A: First "Continue as Guest" button (before email)
    try {
      await stagehand.act('click "Continue as Guest", "Guest Checkout", or "Checkout as Guest" button (do NOT log in, do NOT create an account)')
      log.add('guest_step1', 'Clicked Continue as Guest (step 1)', 'success')
      await page.waitForTimeout(2000)
    } catch {
      log.add('guest_step1', 'No initial guest prompt', 'skipped')
    }

    // Step B: Enter email address
    if (d?.email) {
      try {
        await stagehand.act(`type "${d.email}" into the email address field`)
        log.add('gate_email', 'Entered email', 'success', d.email)
        await page.waitForTimeout(1000)
      } catch {
        log.add('gate_email', 'No email gate field', 'skipped')
      }
    }

    // Step C: Second "Continue as Guest" button (after email)
    try {
      await stagehand.act('click "Continue as Guest", "Guest Checkout", or "Continue" button (do NOT log in, do NOT create an account)')
      log.add('guest_step2', 'Clicked Continue as Guest (step 2)', 'success')
      await page.waitForTimeout(3000)
    } catch {
      log.add('guest_step2', 'No second guest prompt (already on shipping form)', 'skipped')
    }

    // ── Helper: try act(), then rephrase, then observe+fill ──────────
    const STATE_NAMES: Record<string, string> = {
      AL:'Alabama',AK:'Alaska',AZ:'Arizona',AR:'Arkansas',CA:'California',CO:'Colorado',
      CT:'Connecticut',DE:'Delaware',FL:'Florida',GA:'Georgia',HI:'Hawaii',ID:'Idaho',
      IL:'Illinois',IN:'Indiana',IA:'Iowa',KS:'Kansas',KY:'Kentucky',LA:'Louisiana',
      ME:'Maine',MD:'Maryland',MA:'Massachusetts',MI:'Michigan',MN:'Minnesota',MS:'Mississippi',
      MO:'Missouri',MT:'Montana',NE:'Nebraska',NV:'Nevada',NH:'New Hampshire',NJ:'New Jersey',
      NM:'New Mexico',NY:'New York',NC:'North Carolina',ND:'North Dakota',OH:'Ohio',OK:'Oklahoma',
      OR:'Oregon',PA:'Pennsylvania',RI:'Rhode Island',SC:'South Carolina',SD:'South Dakota',
      TN:'Tennessee',TX:'Texas',UT:'Utah',VT:'Vermont',VA:'Virginia',WA:'Washington',
      WV:'West Virginia',WI:'Wisconsin',WY:'Wyoming',DC:'District of Columbia',
    }

    async function fillField(
      fieldName: string,
      value: string,
      primaryInstruction: string,
      retryInstruction: string,
      observeDescription: string,
    ) {
      // Attempt 1: primary act()
      try {
        await stagehand.act(primaryInstruction)
        log.add(`fill_${fieldName}`, `Filled ${fieldName}`, 'success', `"${value}"`)
        return
      } catch {}

      // Attempt 2: rephrased act()
      try {
        await stagehand.act(retryInstruction)
        log.add(`fill_${fieldName}`, `Filled ${fieldName} (retry)`, 'success', `"${value}"`)
        return
      } catch {}

      // Attempt 3: observe() + Playwright fill()
      try {
        const elements = await stagehand.observe(observeDescription)
        if (elements && elements.length > 0 && elements[0].selector) {
          const el = await page.$(elements[0].selector)
          if (el) {
            await el.scrollIntoViewIfNeeded().catch(() => {})
            await el.click({ force: true }).catch(() => {})
            await el.fill(value)
            log.add(`fill_${fieldName}`, `Filled ${fieldName} (observe fallback)`, 'success', `"${value}"`)
            return
          }
        }
      } catch {}

      // Attempt 4: direct Playwright — find by common selectors, label text, or placeholder
      try {
        const filled = await page.evaluate(({ name, val }: { name: string; val: string }) => {
          const searchTerms: Record<string, string[]> = {
            first_name: ['first', 'fname', 'given'],
            last_name: ['last', 'lname', 'family', 'surname'],
            email: ['email', 'e-mail'],
            phone: ['phone', 'tel', 'mobile'],
            address: ['address', 'street', 'addr'],
            city: ['city', 'town', 'locality'],
            zip: ['zip', 'postal', 'postcode'],
          }
          const terms = searchTerms[name] || [name.replace('_', '')]
          const inputs = Array.from(document.querySelectorAll('input:not([type="hidden"]):not([type="checkbox"]):not([type="radio"]), textarea')) as HTMLInputElement[]
          for (const input of inputs) {
            if (input.offsetParent === null || input.disabled) continue
            const id = (input.id || '').toLowerCase()
            const nm = (input.name || '').toLowerCase()
            const ph = (input.placeholder || '').toLowerCase()
            const label = (input.getAttribute('aria-label') || '').toLowerCase()
            const allText = `${id} ${nm} ${ph} ${label}`
            if (terms.some(t => allText.includes(t)) && !input.value) {
              input.focus()
              input.value = val
              input.dispatchEvent(new Event('input', { bubbles: true }))
              input.dispatchEvent(new Event('change', { bubbles: true }))
              input.dispatchEvent(new Event('blur', { bubbles: true }))
              return `Filled via JS: ${input.id || input.name || input.placeholder}`
            }
          }
          return null
        }, { name: fieldName, val: value })
        if (filled) {
          log.add(`fill_${fieldName}`, `Filled ${fieldName} (JS fallback)`, 'success', filled)
          return
        }
      } catch {}

      log.add(`fill_${fieldName}`, `Could not fill ${fieldName}`, 'failed')
    }

    // ── Fill shipping/delivery form ─────────────────────────────────────
    if (d) {
      // 1. Fill fields that are visible right away: First Name, Last Name, Phone
      if (d.firstName) {
        await fillField('first_name', d.firstName,
          `click on the First Name field and type "${d.firstName}"`,
          `fill in "${d.firstName}" as the first name`,
          'the first name input field')
        await page.waitForTimeout(400)
      }
      if (d.lastName) {
        await fillField('last_name', d.lastName,
          `click on the Last Name field and type "${d.lastName}"`,
          `fill in "${d.lastName}" as the last name`,
          'the last name input field')
        await page.waitForTimeout(400)
      }
      if (d.phone) {
        const cleanPhone = d.phone.replace(/[^0-9+]/g, '')
        await fillField('phone', cleanPhone,
          `click on the Phone field and type "${cleanPhone}"`,
          `fill the phone field with "${cleanPhone}"`,
          'the phone number input field')
        await page.waitForTimeout(400)
      }
      // 2. Click "Enter Address Manually" if visible (expands hidden address fields)
      try {
        await stagehand.act('click "Enter Address Manually" or "Enter address manually" link if visible')
        log.add('expand_address', 'Clicked Enter Address Manually', 'success')
        await page.waitForTimeout(1500)
      } catch {
        log.add('expand_address', 'No manual address link (fields already visible)', 'skipped')
      }

      // 3. Fill address fields (now visible after expanding)
      if (d.shippingAddress) {
        await fillField('address', d.shippingAddress,
          `type "${d.shippingAddress}" into the Address or Address Line 1 field`,
          `fill the street address field with "${d.shippingAddress}"`,
          'the address or street address input field')
        await page.waitForTimeout(400)
      }
      if (d.shippingCity) {
        await fillField('city', d.shippingCity,
          `type "${d.shippingCity}" into the City or Town field`,
          `fill the city field with "${d.shippingCity}"`,
          'the city input field')
        await page.waitForTimeout(400)
      }
      if (d.shippingZip) {
        await fillField('zip', d.shippingZip,
          `type "${d.shippingZip}" into the Zip Code or Postal Code field`,
          `fill the zip code field with "${d.shippingZip}"`,
          'the zip code or postal code input field')
        await page.waitForTimeout(400)
      }

      // 4. State dropdown
      if (d.shippingState) {
        const stateCode = d.shippingState.toUpperCase()
        const stateName = STATE_NAMES[stateCode] || d.shippingState
        let stateSet = false

        try {
          await stagehand.act(`select "${stateName}" from the State dropdown`)
          log.add('fill_state', `Selected state ${stateName}`, 'success')
          stateSet = true
        } catch {}

        if (!stateSet) {
          try {
            await stagehand.act(`choose "${stateCode}" in the state selector`)
            log.add('fill_state', `Selected state ${stateCode}`, 'success')
            stateSet = true
          } catch {}
        }

        if (!stateSet) {
          try {
            const elements = await stagehand.observe('the state dropdown or select element')
            if (elements?.[0]?.selector) {
              await page.selectOption(elements[0].selector, { label: stateName }).catch(async () => {
                await page.selectOption(elements[0].selector, stateCode).catch(() => {})
              })
              log.add('fill_state', `Selected state via fallback`, 'success', stateName)
              stateSet = true
            }
          } catch {}
        }

        if (!stateSet) {
          log.add('fill_state', `Could not select state ${stateName}`, 'failed')
        }
      }

      // 5. Click "Save Address" if that popup shows up
      try {
        await stagehand.act('click "Save Address" or "Use This Address" button if visible')
        log.add('save_address', 'Clicked Save Address', 'success')
        await page.waitForTimeout(1500)
      } catch {
        log.add('save_address', 'No save address popup', 'skipped')
      }

      // 6. Click "Continue to Payment" / submit the delivery form
      await page.waitForTimeout(1000)
      try {
        await stagehand.act('click "CONTINUE TO PAYMENT", "Continue to Payment", "Continue", or the submit button to proceed to payment')
        log.add('submit_shipping', 'Clicked Continue to Payment', 'success')
      } catch (e: any) {
        log.add('submit_shipping', 'Could not submit shipping form', 'failed', e.message)
      }
      await page.waitForTimeout(3000)

      // 7. Check if we reached payment
      const finalUrl = page.url()
      const reachedPayment = /payment|billing|review|order/i.test(finalUrl)
      if (reachedPayment) {
        log.add('checkout_complete', `Reached payment page: ${finalUrl}`, 'success')
      } else {
        log.add('checkout_complete', `Current URL: ${finalUrl}`, 'skipped', 'Browser left open for user to complete')
      }
    }

    return {
      success: true,
      product_url: req.product_url,
      steps: log.steps,
      cart_status: 'checkout_reached',
      elapsed_ms: Date.now() - startTime,
    }
  } catch (error: any) {
    log.add('error', error.message, 'failed', error.stack?.substring(0, 200))
    return { success: false, product_url: req.product_url, steps: log.steps, cart_status: 'failed', error: error.message, elapsed_ms: Date.now() - startTime }
  } finally {
    // Keep browser open so user can finish manually if needed
    // Only close in headless mode
    if (stagehand && req.headless) {
      try { await stagehand.close() } catch {}
    }
  }
}

// ── API Route ────────────────────────────────────────────────────────────────

export async function POST(request: Request) {
  try {
    const body = await request.json()

    const product_url = body.product_url?.trim()
    if (!product_url || !/^https?:\/\//i.test(product_url)) {
      return NextResponse.json({ success: false, message: 'A valid product_url is required.' }, { status: 400 })
    }

    let userDetails: UserCheckoutDetails | null = null
    if (body.user_id) {
      userDetails = await getUserCheckoutDetails(body.user_id)
    }

    const req: PurchaseRequest = {
      product_url,
      product_name: body.product_name || 'Unknown Product',
      selected_size: body.selected_size || null,
      quantity: Math.max(1, parseInt(body.quantity) || 1),
      dry_run: body.dry_run !== false,
      user_id: body.user_id || null,
      userDetails,
      headless: body.headless !== false,
    }

    console.log('[agentic-checkout] Starting Stagehand agent:', { url: req.product_url, name: req.product_name, dry_run: req.dry_run, headless: req.headless })
    const result = await executePurchase(req)
    console.log('[agentic-checkout] Done:', { success: result.success, cart_status: result.cart_status, steps: result.steps.length, elapsed: result.elapsed_ms })

    return NextResponse.json(result)
  } catch (error: any) {
    console.error('[agentic-checkout] Fatal:', error)
    return NextResponse.json({ success: false, message: error.message || 'Agentic checkout failed.' }, { status: 500 })
  }
}
