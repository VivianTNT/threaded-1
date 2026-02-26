#!/usr/bin/env npx tsx
/**
 * Extract financial data from already-stored PDFs
 * Uses the working /api/pdf/extract-highlights endpoint
 */

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('💰 EXTRACTING FINANCIAL DATA FROM STORED PDFs')
  console.log('='.repeat(80))

  // Get all projects with URLs
  const { data: projects } = await supabase
    .from('projects')
    .select('*')
    .not('urls', 'is', null)
    .order('name')

  if (!projects || projects.length === 0) {
    console.log('No projects found')
    return
  }

  console.log(`Found ${projects.length} projects with PDFs\n`)

  let extracted = 0
  let failed = 0

  for (const project of projects) {
    console.log(`\n📊 ${project.name}`)

    if (!project.urls || project.urls.length === 0) {
      console.log(`   ⚠️  No URLs`)
      failed++
      continue
    }

    const pdfUrl = project.urls[0] // Use first PDF
    console.log(`   📄 ${pdfUrl.substring(pdfUrl.lastIndexOf('/') + 1)}`)

    try {
      console.log(`   🔍 Calling extraction API...`)

      const response = await fetch('http://localhost:3002/api/pdf/extract-highlights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdfUrl, projectId: project.id })
      })

      if (!response.ok) {
        console.log(`   ❌ API error: ${response.status}`)
        failed++
        continue
      }

      const result = await response.json()

      if (result.error) {
        console.log(`   ❌ ${result.error}`)
        failed++
        continue
      }

      // Extract financial metrics
      const updates: any = {}

      if (result.npv?.value) {
        updates.post_tax_npv_usd_m = result.npv.value
        console.log(`   💰 NPV: $${result.npv.value}M`)
      }

      if (result.irr?.value) {
        updates.irr_percent = result.irr.value
        console.log(`   📈 IRR: ${result.irr.value}%`)
      }

      if (result.capex?.value) {
        updates.capex_usd_m = result.capex.value
        console.log(`   🏗️  CAPEX: $${result.capex.value}M`)
      }

      if (Object.keys(updates).length > 0) {
        await supabase
          .from('projects')
          .update(updates)
          .eq('id', project.id)

        console.log(`   ✅ Updated database`)
        extracted++
      } else {
        console.log(`   ⚠️  No metrics found`)
        failed++
      }

    } catch (error) {
      console.log(`   ❌ Error: ${error}`)
      failed++
    }

    // Rate limit
    await new Promise(r => setTimeout(r, 3000))
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ Extracted: ${extracted}`)
  console.log(`❌ Failed: ${failed}`)
  console.log(`📊 Total: ${projects.length}`)
  console.log(`${'='.repeat(80)}`)
}

main().catch(console.error)
