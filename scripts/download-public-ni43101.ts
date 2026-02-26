#!/usr/bin/env node

/**
 * Download a known large NI 43-101 technical report from public sources
 * These are typically 200+ page documents from SEDAR or company websites
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Known public NI 43-101 reports (these are real URLs from SEDAR/company websites)
const PUBLIC_REPORTS = [
  {
    company: 'Ivanhoe Mines',
    project: 'Kamoa-Kakula',
    url: 'https://www.ivanhoemines.com/site/assets/files/4290/2023_kamoa-kakula_ni_43-101_technical_report_final.pdf',
    ticker: 'IVN',
  },
  {
    company: 'Teck Resources',
    project: 'QB2',
    url: 'https://www.teck.com/media/43-101-Technical-Report-Quebrada-Blanca-Phase-2-Project-Chile.pdf',
    ticker: 'TECK',
  },
  {
    company: 'First Quantum',
    project: 'Sentinel',
    url: 'https://www.first-quantum.com/English/operations/sentinel/default.aspx',
    ticker: 'FM',
    note: 'Direct download link may need to be found on website',
  },
]

async function downloadReport(report: typeof PUBLIC_REPORTS[0]) {
  console.log(`\n📥 Downloading: ${report.company} - ${report.project}`)
  console.log(`   URL: ${report.url}`)

  try {
    const response = await fetch(report.url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
      },
    })

    if (!response.ok) {
      console.log(`   ❌ Failed to download: ${response.status} ${response.statusText}`)
      return null
    }

    const contentType = response.headers.get('content-type')
    if (!contentType?.includes('pdf')) {
      console.log(`   ⚠️  Not a PDF file (${contentType}), skipping...`)
      return null
    }

    const buffer = await response.arrayBuffer()
    const sizeKB = Math.round(buffer.byteLength / 1024)
    const sizeMB = (sizeKB / 1024).toFixed(2)

    console.log(`   ✅ Downloaded: ${sizeKB} KB (${sizeMB} MB)`)

    // Upload to Supabase
    const fileName = `${report.project.toLowerCase().replace(/\s+/g, '-')}.pdf`
    const storagePath = `${report.ticker}/${fileName}`

    console.log(`   📤 Uploading to Supabase...`)

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
    console.log(`   Public URL: ${urlData.publicUrl}`)

    return {
      company: report.company,
      project: report.project,
      url: urlData.publicUrl,
      sizeKB,
      sizeMB,
    }
  } catch (error: any) {
    console.error(`   ❌ Error:`, error.message)
    return null
  }
}

async function main() {
  console.log('🔍 Downloading publicly available NI 43-101 technical reports...\n')

  const results = []

  for (const report of PUBLIC_REPORTS) {
    if (report.note) {
      console.log(`\n⚠️  ${report.company}: ${report.note}`)
      continue
    }

    const result = await downloadReport(report)
    if (result) {
      results.push(result)
    }

    // Rate limiting
    await new Promise((resolve) => setTimeout(resolve, 2000))
  }

  console.log('\n\n' + '='.repeat(80))
  console.log('✅ Download Summary')
  console.log('='.repeat(80))

  if (results.length === 0) {
    console.log('\n⚠️  No reports were successfully downloaded.')
    console.log('\nAlternative: Try downloading from SEDAR manually:')
    console.log('1. Go to https://www.sedarplus.ca')
    console.log('2. Search for companies like "Ivanhoe Mines" or "Teck Resources"')
    console.log('3. Look for "Technical Report" or "NI 43-101" documents')
    console.log('4. Download the PDF manually and save to downloads folder')
  } else {
    console.log(`\n📄 Successfully downloaded ${results.length} technical reports:\n`)
    results.forEach((r, i) => {
      console.log(`${i + 1}. ${r.company} - ${r.project}`)
      console.log(`   Size: ${r.sizeKB} KB (${r.sizeMB} MB)`)
      console.log(`   URL: ${r.url}\n`)
    })

    console.log('\n🧪 To test extraction, run:')
    console.log('npx tsx scripts/test-pdf-extraction.ts')
  }
}

main().catch(console.error)
