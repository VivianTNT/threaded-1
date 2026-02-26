#!/usr/bin/env npx tsx
/**
 * Direct PDF Financial Extraction
 * Downloads PDFs from Supabase storage and extracts NPV, IRR, CAPEX using OpenAI
 */

import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import pdf from 'pdf-parse'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

async function extractTextFromPDF(buffer: Buffer): Promise<string> {
  try {
    const data = await pdf(buffer)
    return data.text
  } catch (error) {
    console.log(`      ⚠️  PDF parse error: ${error}`)
    return ''
  }
}

async function extractFinancialMetrics(text: string, projectName: string) {
  try {
    // Use first 20000 chars (executive summary + economics)
    const relevantText = text.substring(0, 20000)

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `Extract financial metrics from this NI 43-101 mining technical report for ${projectName}. Return ONLY a JSON object with numeric values in millions USD for NPV and CAPEX, and percentage for IRR. Look for: Post-Tax NPV, After-Tax NPV, IRR, Internal Rate of Return, Initial CAPEX, Capital Cost, Total CAPEX. Return {"npv": number, "irr": number, "capex": number} or {} if not found.`
        },
        {
          role: 'user',
          content: relevantText
        }
      ],
      response_format: { type: 'json_object' },
      temperature: 0.1
    })

    const result = JSON.parse(response.choices[0].message.content || '{}')

    return {
      npv: result.npv || result.NPV || null,
      irr: result.irr || result.IRR || null,
      capex: result.capex || result.CAPEX || null
    }
  } catch (error) {
    console.log(`      ⚠️  AI extraction failed: ${error}`)
    return { npv: null, irr: null, capex: null }
  }
}

async function main() {
  console.log('💰 DIRECT PDF FINANCIAL EXTRACTION')
  console.log('='.repeat(80))

  // Get all projects with PDF URLs
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

    const pdfUrl = project.urls[0]
    const fileName = pdfUrl.substring(pdfUrl.lastIndexOf('/') + 1)
    console.log(`   📄 ${fileName}`)

    try {
      // Download PDF from Supabase
      console.log(`   📥 Downloading...`)
      const pdfResponse = await fetch(pdfUrl)

      if (!pdfResponse.ok) {
        console.log(`   ❌ Download failed: ${pdfResponse.status}`)
        failed++
        continue
      }

      const arrayBuffer = await pdfResponse.arrayBuffer()
      const buffer = Buffer.from(arrayBuffer)
      const sizeInMB = (buffer.length / 1024 / 1024).toFixed(2)
      console.log(`   ✅ Downloaded ${sizeInMB}MB`)

      // Extract text
      console.log(`   📖 Extracting text...`)
      const text = await extractTextFromPDF(buffer)

      if (text.length < 1000) {
        console.log(`   ⚠️  Too little text extracted`)
        failed++
        continue
      }

      console.log(`   ✅ Extracted ${(text.length / 1000).toFixed(0)}KB text`)

      // Extract financial metrics with AI
      console.log(`   🤖 Analyzing with AI...`)
      const metrics = await extractFinancialMetrics(text, project.name)

      // Update database
      const updates: any = {}
      if (metrics.npv) {
        updates.post_tax_npv_usd_m = metrics.npv
        console.log(`   💰 NPV: $${metrics.npv}M`)
      }
      if (metrics.irr) {
        updates.irr_percent = metrics.irr
        console.log(`   📈 IRR: ${metrics.irr}%`)
      }
      if (metrics.capex) {
        updates.capex_usd_m = metrics.capex
        console.log(`   🏗️  CAPEX: $${metrics.capex}M`)
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
    await new Promise(r => setTimeout(r, 2000))
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ Extracted: ${extracted}`)
  console.log(`❌ Failed: ${failed}`)
  console.log(`📊 Total: ${projects.length}`)
  console.log(`${'='.repeat(80)}`)
}

main().catch(console.error)
