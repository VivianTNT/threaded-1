#!/usr/bin/env node

/**
 * Download FactSet Documents and Upload to Supabase
 *
 * For a list of sample mining companies:
 * 1. Search for technical documents (NI 43-101, technical reports, etc.)
 * 2. Download PDFs from FactSet
 * 3. Upload to Supabase Storage
 * 4. Link documents to projects in database
 * 5. Make PDFs accessible via public URLs
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

// Sample Canadian mining companies
const SAMPLE_COMPANIES = [
  { name: 'Ivanhoe Mines', ticker: 'IVN-CA', priority: true },
  { name: 'Teck Resources', ticker: 'TECK-CA', priority: true },
  { name: 'First Quantum Minerals', ticker: 'FM-CA', priority: false },
  { name: 'Lundin Mining', ticker: 'LUN-CA', priority: false },
  { name: 'Hudbay Minerals', ticker: 'HBM-CA', priority: false }
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

async function searchTechnicalDocuments(ticker: string): Promise<Filing[]> {
  const params = new URLSearchParams({
    ids: ticker,
    sources: 'SDR,SDRP', // Both old and new SEDAR
    startDate: '20230101', // Last 2 years
    _paginationLimit: '20',
    timeZone: 'America/Toronto',
    _sort: '-filingsDateTime'
  })

  const response = await fetch(`${BASE_URL}/search?${params.toString()}`, {
    headers: {
      'Authorization': createAuthHeader(),
      'Accept': 'application/json'
    }
  })

  if (!response.ok) {
    throw new Error(`Search failed: ${await response.text()}`)
  }

  const data = await response.json()
  const allDocs = data.data?.[0]?.documents || []

  // Filter for technical documents
  return allDocs.filter((doc: Filing) => {
    const headline = doc.headline.toLowerCase()
    const formType = doc.formTypes?.[0]?.toLowerCase() || ''

    return (
      headline.includes('43-101') ||
      headline.includes('technical report') ||
      headline.includes('feasibility') ||
      headline.includes('pre-feasibility') ||
      headline.includes('mineral resource') ||
      headline.includes('mineral reserve') ||
      formType.includes('43-101') ||
      formType.includes('technical')
    )
  })
}

async function downloadDocument(filing: Filing): Promise<Buffer | null> {
  try {
    const response = await fetch(filing.filingsLink, {
      headers: {
        'Authorization': createAuthHeader(),
        'Accept': '*/*'
      }
    })

    if (!response.ok) {
      console.error(`      ❌ Download failed: ${response.status} ${response.statusText}`)
      return null
    }

    return Buffer.from(await response.arrayBuffer())
  } catch (error: any) {
    console.error(`      ❌ Download error: ${error.message}`)
    return null
  }
}

async function uploadToSupabase(
  ticker: string,
  filing: Filing,
  pdfBuffer: Buffer
): Promise<string | null> {
  try {
    const filename = `${ticker.replace('-CA', '')}/${filing.documentId}.pdf`

    // Upload to Supabase Storage
    const { error: uploadError } = await supabase.storage
      .from('factset-documents')
      .upload(filename, pdfBuffer, {
        contentType: 'application/pdf',
        upsert: true
      })

    if (uploadError) {
      console.error(`      ❌ Upload failed: ${uploadError.message}`)
      return null
    }

    // Get public URL
    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(filename)

    return urlData.publicUrl
  } catch (error: any) {
    console.error(`      ❌ Upload error: ${error.message}`)
    return null
  }
}

async function findOrCreateProject(companyName: string, ticker: string): Promise<string | null> {
  // First, try to find existing company
  const { data: company } = await supabase
    .from('companies')
    .select('id')
    .ilike('name', `%${companyName}%`)
    .single()

  let companyId = company?.id

  // If company doesn't exist, create it
  if (!companyId) {
    const { data: newCompany, error: createError } = await supabase
      .from('companies')
      .insert({
        name: companyName,
        ticker: ticker.replace('-CA', ''),
        exchange: 'TSX'
      })
      .select('id')
      .single()

    if (createError || !newCompany) {
      console.error(`      ⚠️  Could not create company: ${createError?.message}`)
      return null
    }

    companyId = newCompany.id
  }

  // Find or create a project for this company
  const { data: project } = await supabase
    .from('projects')
    .select('id')
    .eq('company_id', companyId)
    .limit(1)
    .single()

  if (project?.id) {
    return project.id
  }

  // Create a default project
  const { data: newProject, error: projectError } = await supabase
    .from('projects')
    .insert({
      company_id: companyId,
      name: `${companyName} Operations`,
      status: 'Active'
    })
    .select('id')
    .single()

  if (projectError || !newProject) {
    console.error(`      ⚠️  Could not create project: ${projectError?.message}`)
    return null
  }

  return newProject.id
}

async function saveDocumentMetadata(
  filing: Filing,
  projectId: string,
  publicUrl: string,
  fileSize: number
): Promise<boolean> {
  try {
    const { error } = await supabase
      .from('factset_documents')
      .upsert({
        document_id: filing.documentId,
        project_id: projectId,
        headline: filing.headline,
        filing_date: filing.filingsDateTime,
        form_type: filing.formTypes?.[0],
        storage_path: publicUrl.split('/').slice(-2).join('/'),
        public_url: publicUrl,
        file_size: fileSize
      }, {
        onConflict: 'document_id'
      })

    if (error) {
      console.error(`      ⚠️  Database error: ${error.message}`)
      return false
    }

    return true
  } catch (error: any) {
    console.error(`      ⚠️  Metadata error: ${error.message}`)
    return false
  }
}

async function main() {
  console.log('='.repeat(70))
  console.log('FACTSET TECHNICAL DOCUMENTS → SUPABASE')
  console.log('='.repeat(70))

  let totalSearched = 0
  let totalTechnicalDocs = 0
  let totalDownloaded = 0
  let totalUploaded = 0

  for (const company of SAMPLE_COMPANIES) {
    console.log(`\n📊 ${company.name} (${company.ticker})`)
    console.log('-'.repeat(70))

    totalSearched++

    try {
      // Search for technical documents
      console.log(`   🔍 Searching for technical documents...`)
      const technicalDocs = await searchTechnicalDocuments(company.ticker)

      if (technicalDocs.length === 0) {
        console.log(`   ℹ️  No technical documents found`)
        continue
      }

      totalTechnicalDocs += technicalDocs.length
      console.log(`   ✅ Found ${technicalDocs.length} technical document(s)`)

      // Find or create project
      const projectId = await findOrCreateProject(company.name, company.ticker)
      if (!projectId) {
        console.log(`   ⚠️  Could not find/create project, skipping...`)
        continue
      }

      // Process documents (limit to first 3 for testing)
      const docsToProcess = technicalDocs.slice(0, 3)

      for (let i = 0; i < docsToProcess.length; i++) {
        const doc = docsToProcess[i]
        console.log(`\n   📄 Document ${i + 1}/${docsToProcess.length}:`)
        console.log(`      Title: ${doc.headline.substring(0, 70)}...`)
        console.log(`      Date: ${new Date(doc.filingsDateTime).toLocaleDateString()}`)
        console.log(`      Form: ${doc.formTypes?.[0] || 'N/A'}`)

        // Download
        console.log(`      📥 Downloading...`)
        const pdfBuffer = await downloadDocument(doc)

        if (!pdfBuffer) {
          continue
        }

        totalDownloaded++
        const sizeKB = Math.round(pdfBuffer.length / 1024)
        console.log(`      ✅ Downloaded ${sizeKB} KB`)

        // Upload to Supabase
        console.log(`      📤 Uploading to Supabase...`)
        const publicUrl = await uploadToSupabase(company.ticker, doc, pdfBuffer)

        if (!publicUrl) {
          continue
        }

        totalUploaded++
        console.log(`      ✅ Uploaded successfully`)
        console.log(`      🔗 URL: ${publicUrl}`)

        // Save metadata
        console.log(`      💾 Saving metadata...`)
        const saved = await saveDocumentMetadata(doc, projectId, publicUrl, pdfBuffer.length)

        if (saved) {
          console.log(`      ✅ Metadata saved`)
        }

        // Small delay between downloads
        await new Promise(resolve => setTimeout(resolve, 500))
      }

    } catch (error: any) {
      console.error(`   ❌ Error: ${error.message}`)
    }
  }

  // Summary
  console.log('\n' + '='.repeat(70))
  console.log('SUMMARY')
  console.log('='.repeat(70))
  console.log(`\n📊 Results:`)
  console.log(`   Companies searched: ${totalSearched}`)
  console.log(`   Technical documents found: ${totalTechnicalDocs}`)
  console.log(`   Documents downloaded: ${totalDownloaded}`)
  console.log(`   Documents uploaded to Supabase: ${totalUploaded}`)

  if (totalUploaded > 0) {
    console.log(`\n✅ SUCCESS! ${totalUploaded} document(s) now accessible via Supabase URLs`)
    console.log(`\n💡 Next steps:`)
    console.log(`   1. View documents in Supabase Storage: factset-documents bucket`)
    console.log(`   2. Check metadata: SELECT * FROM factset_documents`)
    console.log(`   3. Access PDFs from frontend using the public_url column`)
  } else {
    console.log(`\n⚠️  No documents were uploaded`)
  }

  console.log('\n' + '='.repeat(70))
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error.message)
  process.exit(1)
})
