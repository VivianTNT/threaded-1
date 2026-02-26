#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!

const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Canadian mining companies for SEDAR NI 43-101 reports
const MINING_COMPANIES = [
  { ticker: 'FM-CA', name: 'First Quantum Minerals', primary_commodity: 'Copper' },
  { ticker: 'ABX-CA', name: 'Barrick Gold', primary_commodity: 'Gold' },
  { ticker: 'K-CA', name: 'Kinross Gold', primary_commodity: 'Gold' },
  { ticker: 'AEM-CA', name: 'Agnico Eagle Mines', primary_commodity: 'Gold' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', primary_commodity: 'Copper' },
  { ticker: 'CS-CA', name: 'Capstone Copper', primary_commodity: 'Copper' },
  { ticker: 'TECK-CA', name: 'Teck Resources', primary_commodity: 'Coal' },
  { ticker: 'IMG-CA', name: 'IAMGOLD', primary_commodity: 'Gold' }
]

async function searchSEDARFilings(ticker: string, companyName: string) {
  console.log(`\n🔍 Searching SEDAR for ${companyName} (${ticker})...`)

  try {
    const params = new URLSearchParams({
      ids: ticker,
      sources: 'SDR',  // SEDAR
      startDate: '20200101',
      endDate: '20241231',
      searchText: 'Technical Report',
      _paginationLimit: '20'
    })

    const url = `https://api.factset.com/content/global-filings/v2/search?${params.toString()}`

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/json'
      }
    })

    if (!response.ok) {
      console.log(`   ❌ Search failed (${response.status})`)
      return []
    }

    const data = await response.json()
    const filings: any[] = []

    if (data.data && Array.isArray(data.data)) {
      for (const result of data.data) {
        if (result.documents && Array.isArray(result.documents)) {
          console.log(`   ✅ Found ${result.documents.length} filings`)

          for (const doc of result.documents) {
            // Check if this is a PDF by trying to get the PDF version
            const pdfUrl = doc.filingsLink?.replace('report=story', 'report=pdf')

            filings.push({
              companyName,
              ticker,
              headline: doc.headline,
              filingDate: doc.filingsDateTime,
              documentId: doc.documentId,
              htmlUrl: doc.filingsLink,
              pdfUrl: pdfUrl,
              size: doc.filingSize
            })

            console.log(`      📄 ${doc.headline?.substring(0, 60)}...`)
          }
        }
      }
    }

    return filings
  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`)
    return []
  }
}

async function downloadPDF(url: string, documentId: string): Promise<Buffer | null> {
  console.log(`   📥 Attempting PDF download...`)

  try {
    // Try with Accept: application/pdf header
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf'
      }
    })

    if (!response.ok) {
      console.log(`      ❌ Failed (${response.status})`)
      return null
    }

    const contentType = response.headers.get('content-type')
    console.log(`      📋 Content-Type: ${contentType}`)

    const buffer = Buffer.from(await response.arrayBuffer())

    // Check if it's actually a PDF
    const pdfHeader = buffer.slice(0, 5).toString()
    if (pdfHeader === '%PDF-') {
      console.log(`      ✅ Valid PDF downloaded (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`)
      return buffer
    } else {
      console.log(`      ⚠️  Not a PDF, trying alternative method...`)
      return null
    }
  } catch (error: any) {
    console.log(`      ❌ Error: ${error.message}`)
    return null
  }
}

async function downloadHTMLAndConvert(url: string): Promise<Buffer | null> {
  console.log(`   📄 Downloading HTML version...`)

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'text/html'
      }
    })

    if (!response.ok) {
      return null
    }

    const html = await response.text()
    console.log(`      ✅ Downloaded HTML (${(html.length / 1024).toFixed(2)} KB)`)

    // Return as buffer for storage
    return Buffer.from(html, 'utf-8')
  } catch (error: any) {
    console.log(`      ❌ Error: ${error.message}`)
    return null
  }
}

async function processAndUpload() {
  console.log('🏭 FACTSET PDF DOWNLOAD AND UPLOAD PIPELINE')
  console.log('='.repeat(60))

  // Clear existing projects
  console.log('\n🗑️  Clearing existing projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')

  let totalUploaded = 0

  for (const company of MINING_COMPANIES.slice(0, 5)) {
    console.log(`\n${'='.repeat(60)}`)
    console.log(`📊 Processing ${company.name}`)
    console.log('='.repeat(60))

    const filings = await searchSEDARFilings(company.ticker, company.name)

    if (filings.length === 0) {
      console.log(`   ⚠️  No filings found, skipping...`)
      continue
    }

    const filing = filings[0]

    // Try to get PDF version first
    let pdfBuffer = await downloadPDF(filing.pdfUrl, filing.documentId)

    let fileExtension = 'pdf'
    let contentType = 'application/pdf'
    let buffer = pdfBuffer

    // If PDF doesn't work, fall back to HTML
    if (!pdfBuffer) {
      console.log(`   ℹ️  PDF not available, using HTML version`)
      buffer = await downloadHTMLAndConvert(filing.htmlUrl)
      fileExtension = 'html'
      contentType = 'text/html'
    }

    if (!buffer) {
      console.log(`   ❌ Failed to download document, skipping...`)
      continue
    }

    // Upload to Supabase Storage
    const projectId = crypto.randomUUID()
    const fileName = `${projectId}.${fileExtension}`
    const filePath = `factset-documents/${fileName}`

    console.log(`   ☁️  Uploading to Supabase Storage...`)

    const { error: uploadError } = await supabase.storage
      .from('mining-documents')
      .upload(filePath, buffer, {
        contentType,
        upsert: true
      })

    if (uploadError) {
      console.log(`   ❌ Upload failed: ${uploadError.message}`)
      continue
    }

    // Get public URL
    const { data: { publicUrl } } = supabase.storage
      .from('mining-documents')
      .getPublicUrl(filePath)

    console.log(`   ✅ Uploaded: ${publicUrl}`)

    // Create project WITHOUT financial data - users will extract via PDF viewer
    const projectData = {
      id: projectId,
      name: company.name,
      location: 'Canada',
      stage: 'Unknown',
      commodities: [company.primary_commodity],
      status: 'Active',
      description: `${filing.headline}\n\nFiled: ${filing.filingDate?.substring(0, 10)}\nSource: FactSet SEDAR\n\nUse the PDF viewer to extract financial metrics.`,
      urls: [publicUrl],
      npv: null,
      irr: null,
      capex: null,
      watchlist: false
    }

    const { error: insertError } = await supabase.from('projects').insert(projectData)

    if (!insertError) {
      console.log(`   ✅ Created project: ${company.name}`)
      totalUploaded++
    } else {
      console.log(`   ❌ Failed to create project: ${insertError.message}`)
    }

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 2000))
  }

  console.log('\n' + '='.repeat(60))
  console.log('📊 SUMMARY')
  console.log('='.repeat(60))
  console.log(`✅ Documents uploaded: ${totalUploaded}`)
  console.log('\n✨ Documents are now stored in Supabase and accessible!')
}

processAndUpload().catch(console.error)