#!/usr/bin/env node

/**
 * Download Large NI 43-101 Technical Reports from FactSet
 * Specifically searches for full technical reports (typically 100+ pages)
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'
import fs from 'fs/promises'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)
const BASE_URL = 'https://api.factset.com/content/global-filings/v2'

// Companies with known large NI 43-101 reports
const COMPANIES = [
  { name: 'Ivanhoe Mines', ticker: 'IVN-CA' },
  { name: 'Lundin Mining', ticker: 'LUN-CA' },
  { name: 'Teck Resources', ticker: 'TECK-CA' },
  { name: 'First Quantum Minerals', ticker: 'FM-CA' },
]

interface Filing {
  headline: string
  filingsDateTime: string
  formTypes?: string[]
  documentId: string
  filingsLink: string
  filingSize?: string
}

function createAuthHeader(): string {
  return 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
}

async function searchNI43101Reports(ticker: string): Promise<Filing[]> {
  console.log(`\n🔍 Searching for NI 43-101 reports for ${ticker}...`)

  const url = new URL(`${BASE_URL}/search`)
  url.searchParams.append('sources', 'SEDAR')
  url.searchParams.append('sources', 'EDG') // EDGAR for US filings
  url.searchParams.append('startDate', '2020-01-01')
  url.searchParams.append('endDate', '2025-12-31')
  url.searchParams.append('includeFormTypes', 'true')

  // Search for ticker + NI 43-101 keywords
  const query = `(${ticker}) AND ("NI 43-101" OR "technical report" OR "feasibility study" OR "preliminary economic assessment")`
  url.searchParams.append('searchText', query)

  const response = await fetch(url.toString(), {
    headers: {
      Authorization: createAuthHeader(),
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    console.error(`❌ Search failed: ${response.status} ${response.statusText}`)
    const errorText = await response.text()
    console.error('Error details:', errorText)
    return []
  }

  const data = await response.json()
  const filings: Filing[] = data.data || []

  // Filter for likely technical reports (typically larger files)
  const technicalReports = filings.filter((filing) => {
    const headline = filing.headline?.toLowerCase() || ''
    const isNI43101 =
      headline.includes('ni 43-101') ||
      headline.includes('technical report') ||
      headline.includes('feasibility study') ||
      headline.includes('preliminary economic assessment') ||
      headline.includes('pea') ||
      headline.includes('mineral resource')

    return isNI43101
  })

  console.log(`   Found ${technicalReports.length} potential NI 43-101 reports`)

  return technicalReports.slice(0, 5) // Top 5 results
}

async function downloadDocument(filing: Filing, ticker: string): Promise<{ success: boolean; size?: number; path?: string }> {
  try {
    console.log(`\n📥 Downloading: ${filing.headline}`)
    console.log(`   Date: ${filing.filingsDateTime}`)
    console.log(`   Size: ${filing.filingSize || 'Unknown'}`)

    const url = `${BASE_URL}/document/${filing.documentId}`
    const response = await fetch(url, {
      headers: {
        Authorization: createAuthHeader(),
        Accept: 'application/pdf',
      },
    })

    if (!response.ok) {
      console.error(`   ❌ Download failed: ${response.status}`)
      return { success: false }
    }

    const buffer = await response.arrayBuffer()
    const sizeKB = Math.round(buffer.byteLength / 1024)
    console.log(`   Downloaded: ${sizeKB} KB`)

    // Only keep files larger than 500 KB (likely to be full reports)
    if (sizeKB < 500) {
      console.log(`   ⚠️  Skipping - too small (< 500 KB), likely not a full technical report`)
      return { success: false }
    }

    // Upload to Supabase
    const fileName = `${filing.documentId}.pdf`
    const storagePath = `${ticker.replace('-CA', '')}/${fileName}`

    console.log(`   📤 Uploading to Supabase: ${storagePath}`)

    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(storagePath, Buffer.from(buffer), {
        contentType: 'application/pdf',
        upsert: true,
      })

    if (error) {
      console.error(`   ❌ Upload failed:`, error.message)
      return { success: false }
    }

    const { data: publicUrlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(storagePath)

    console.log(`   ✅ Uploaded successfully!`)
    console.log(`   URL: ${publicUrlData.publicUrl}`)

    return { success: true, size: sizeKB, path: publicUrlData.publicUrl }
  } catch (error: any) {
    console.error(`   ❌ Error:`, error.message)
    return { success: false }
  }
}

async function main() {
  console.log('🔍 Searching for large NI 43-101 technical reports...\n')

  const results: Array<{ company: string; url: string; size: number; headline: string }> = []

  for (const company of COMPANIES) {
    console.log(`\n${'='.repeat(80)}`)
    console.log(`📊 ${company.name} (${company.ticker})`)
    console.log('='.repeat(80))

    const filings = await searchNI43101Reports(company.ticker)

    if (filings.length === 0) {
      console.log('⚠️  No NI 43-101 reports found')
      continue
    }

    // Try to download each filing
    for (const filing of filings) {
      const result = await downloadDocument(filing, company.ticker)

      if (result.success && result.path && result.size) {
        results.push({
          company: company.name,
          url: result.path,
          size: result.size,
          headline: filing.headline,
        })
      }

      // Rate limiting
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  }

  console.log('\n\n' + '='.repeat(80))
  console.log('✅ Download Summary')
  console.log('='.repeat(80))

  if (results.length === 0) {
    console.log('⚠️  No large technical reports were downloaded.')
    console.log('   This might mean:')
    console.log('   - The documents are not available via FactSet API')
    console.log('   - They are behind paywalls or restricted access')
    console.log('   - The search query needs refinement')
  } else {
    console.log(`\n📄 Downloaded ${results.length} technical reports:\n`)
    results.forEach((r, i) => {
      console.log(`${i + 1}. ${r.company}`)
      console.log(`   Size: ${r.size} KB`)
      console.log(`   Title: ${r.headline}`)
      console.log(`   URL: ${r.url}\n`)
    })
  }
}

main().catch(console.error)
