#!/usr/bin/env node

/**
 * Download NI 43-101 Reports using FactSet Global Filings API
 * Uses proper /search endpoint with company IDs
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)
const BASE_URL = 'https://api.factset.com/content/global-filings/v2'

// Canadian mining companies with FactSet IDs
const COMPANIES = [
  { name: 'Ivanhoe Mines', id: 'IVN-CA' },
  { name: 'Lundin Mining', id: 'LUN-CA' },
  { name: 'Teck Resources', id: 'TECK-CA' },
  { name: 'First Quantum Minerals', id: 'FM-CA' },
]

function createAuthHeader(): string {
  return 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
}

async function searchNI43101Documents(companyId: string, companyName: string) {
  console.log(`\n🔍 Searching for ${companyName} (${companyId}) documents...`)

  // Search for documents using FactSet Global Filings API
  const url = new URL(`${BASE_URL}/search`)
  url.searchParams.append('ids', companyId)
  url.searchParams.append('sources', 'SDRP') // SEDAR PLUS (current)
  url.searchParams.append('sources', 'SDR')  // SEDAR (legacy)
  url.searchParams.append('startDate', '20180101') // Go back further
  url.searchParams.append('endDate', '20251231')
  url.searchParams.append('_paginationLimit', '200') // More results
  url.searchParams.append('_sort', '-filingsDateTime') // Most recent first

  try {
    const response = await fetch(url.toString(), {
      headers: {
        Authorization: createAuthHeader(),
        Accept: 'application/json',
      },
    })

    if (!response.ok) {
      console.error(`   ❌ Search failed: ${response.status}`)
      const errorText = await response.text()
      console.error(`   Details: ${errorText}`)
      return []
    }

    const data = await response.json()
    const results = data.data?.[0]?.documents || []

    // Filter for FULL NI 43-101 technical reports (not consent forms or certificates)
    const technicalReports = results.filter((doc: any) => {
      const headline = doc.headline?.toLowerCase() || ''

      // Exclude consent/certificate documents
      if (headline.includes('consent') || headline.includes('certificate')) {
        return false
      }

      // Include only full reports
      return (
        (headline.includes('ni 43-101') && headline.includes('report')) ||
        headline.includes('feasibility study') ||
        headline.includes('pre-feasibility') ||
        headline.includes('pea ') ||
        headline.includes('preliminary economic assessment') ||
        (headline.includes('mineral resource') && headline.includes('estimate')) ||
        (headline.includes('mineral reserve') && headline.includes('estimate'))
      )
    })

    console.log(`   Found ${results.length} total documents, ${technicalReports.length} full technical reports`)

    return technicalReports
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`)
    return []
  }
}

async function downloadDocument(doc: any, companyId: string, companyName: string) {
  console.log(`\n📥 Downloading: ${doc.headline}`)
  console.log(`   Date: ${doc.filingsDateTime}`)
  console.log(`   Size: ${doc.filingSize || 'Unknown'}`)
  console.log(`   Document ID: ${doc.documentId}`)

  try {
    // Use the filingsLink provided in the response
    const url = doc.filingsLink

    const response = await fetch(url, {
      headers: {
        Authorization: createAuthHeader(),
        Accept: '*/*',
      },
    })

    if (!response.ok) {
      console.error(`   ❌ Download failed: ${response.status}`)
      return null
    }

    const contentType = response.headers.get('content-type')
    console.log(`   Content-Type: ${contentType}`)

    const buffer = await response.arrayBuffer()
    const sizeKB = Math.round(buffer.byteLength / 1024)
    const sizeMB = (sizeKB / 1024).toFixed(2)

    console.log(`   ✅ Downloaded: ${sizeKB} KB (${sizeMB} MB)`)

    // Check if it's actually a PDF
    if (!contentType?.includes('pdf')) {
      console.log(`   ⚠️  Skipping - not a PDF file (${contentType})`)
      return null
    }

    // Only keep files larger than 500 KB (likely to be full technical reports)
    if (sizeKB < 500) {
      console.log(`   ⚠️  Skipping - too small (< 500 KB), likely not a full technical report`)
      return null
    }

    // Upload to Supabase
    const fileName = `${doc.documentId}.pdf`
    const storagePath = `${companyId.replace('-CA', '')}/${fileName}`

    console.log(`   📤 Uploading to Supabase: ${storagePath}`)

    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(storagePath, Buffer.from(buffer), {
        contentType: 'application/pdf',
        upsert: true,
      })

    if (error) {
      console.error(`   ❌ Upload failed:`, error.message)
      return null
    }

    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(storagePath)

    console.log(`   ✅ Uploaded successfully!`)
    console.log(`   URL: ${urlData.publicUrl}`)

    return {
      company: companyName,
      url: urlData.publicUrl,
      sizeKB,
      sizeMB,
      headline: doc.headline,
      date: doc.filingsDateTime,
    }
  } catch (error: any) {
    console.error(`   ❌ Error:`, error.message)
    return null
  }
}

async function main() {
  console.log('🔍 Searching for NI 43-101 technical reports using FactSet API...\n')

  const results: any[] = []

  for (const company of COMPANIES) {
    console.log('\n' + '='.repeat(80))
    console.log(`📊 ${company.name}`)
    console.log('='.repeat(80))

    const documents = await searchNI43101Documents(company.id, company.name)

    if (documents.length === 0) {
      console.log('⚠️  No NI 43-101 documents found')
      continue
    }

    // Try to download the top 5 most recent technical reports
    const topDocs = documents.slice(0, 5)

    for (const doc of topDocs) {
      const result = await downloadDocument(doc, company.id, company.name)

      if (result) {
        results.push(result)
      }

      // Rate limiting
      await new Promise((resolve) => setTimeout(resolve, 2000))
    }
  }

  console.log('\n\n' + '='.repeat(80))
  console.log('✅ Download Summary')
  console.log('='.repeat(80))

  if (results.length === 0) {
    console.log('\n⚠️  No large technical reports were downloaded.')
    console.log('   Possible reasons:')
    console.log('   - Documents may be too small (consent forms, amendments)')
    console.log('   - Full technical reports may not be in FactSet coverage')
    console.log('   - API access limitations')
  } else {
    console.log(`\n📄 Downloaded ${results.length} technical reports:\n`)
    results.forEach((r, i) => {
      console.log(`${i + 1}. ${r.company}`)
      console.log(`   Size: ${r.sizeKB} KB (${r.sizeMB} MB)`)
      console.log(`   Date: ${r.date}`)
      console.log(`   Title: ${r.headline}`)
      console.log(`   URL: ${r.url}\n`)
    })

    console.log('\n🧪 To test extraction on the largest document, run:')
    console.log('npx tsx scripts/test-pdf-extraction.ts')
  }
}

main().catch(console.error)
