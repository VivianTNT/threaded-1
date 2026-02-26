#!/usr/bin/env npx tsx
/**
 * EXTRACT ALL PROJECT DATA - CORRECTLY
 *
 * 1. Search for each project's SPECIFIC technical reports
 * 2. Download LARGE documents only (>1MB = 100+ pages)
 * 3. Store in Supabase bucket
 * 4. Extract NPV, IRR, CAPEX from each PDF
 * 5. Populate database with the correct values
 */

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

const COMPANIES_WITH_PROJECTS = [
  {
    ticker: 'FM-CA', name: 'First Quantum Minerals', commodity: 'Copper',
    projects: ['Kansanshi', 'Sentinel', 'Cobre Panama']
  },
  {
    ticker: 'LUN-CA', name: 'Lundin Mining', commodity: 'Copper',
    projects: ['Candelaria', 'Chapada', 'Eagle']
  },
  {
    ticker: 'HBM-CA', name: 'Hudbay Minerals', commodity: 'Copper',
    projects: ['Constancia', 'Copper Mountain', '777']
  },
  {
    ticker: 'ABX-CA', name: 'Barrick Gold', commodity: 'Gold',
    projects: ['Cortez', 'Pueblo Viejo', 'Loulo-Gounkoto']
  },
  {
    ticker: 'TKO-CA', name: 'Taseko Mines', commodity: 'Copper',
    projects: ['Gibraltar', 'Florence Copper']
  }
]

interface ExtractedData {
  npv?: number
  irr?: number
  capex?: number
}

async function extractDataViaAPI(storedUrl: string, projectName: string): Promise<ExtractedData> {
  try {
    console.log(`         🔍 Extracting NPV, IRR, CAPEX...`)

    const apiUrl = `http://localhost:3000/api/pdf/extract-highlights`

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pdfUrl: storedUrl })
    })

    if (!response.ok) {
      console.log(`         ⚠️  API error ${response.status}`)
      return {}
    }

    const result = await response.json()

    if (result.error) {
      console.log(`         ⚠️  ${result.error}`)
      return {}
    }

    const data: ExtractedData = {}

    // Extract NPV
    if (result.npv?.value) data.npv = result.npv.value
    if (result.irr?.value) data.irr = result.irr.value
    if (result.capex?.value) data.capex = result.capex.value

    if (data.npv || data.irr || data.capex) {
      console.log(`         💰 NPV: $${data.npv || 'N/A'}M | IRR: ${data.irr || 'N/A'}% | CAPEX: $${data.capex || 'N/A'}M`)
      return data
    } else {
      console.log(`         ⚠️  No metrics found in API response`)
      return {}
    }
  } catch (error) {
    console.log(`         ❌ Error:`, error)
    return {}
  }
}

async function searchForProjectDocuments(companyTicker: string, projectName: string): Promise<any[]> {
  try {
    // Search specifically for this project name
    const params = new URLSearchParams({
      ids: companyTicker,
      sources: 'SDR',
      searchText: projectName,  // Search for the specific project
      startDate: '20150101',
      endDate: '20241231',
      _paginationLimit: '50'
    })

    const url = `https://api.factset.com/content/global-filings/v2/search?${params.toString()}`
    const response = await fetch(url, {
      headers: { 'Authorization': authHeader }
    })

    if (!response.ok) {
      console.log(`      ❌ Search failed: ${response.status}`)
      return []
    }

    const data = await response.json()
    const docs = data.data?.[0]?.documents || []

    // Filter for technical reports that mention this specific project
    const technicalReports = docs.filter((doc: any) => {
      const headline = doc.headline?.toLowerCase() || ''
      const projectLower = projectName.toLowerCase()

      // Must be a technical report
      const isTechnical = headline.includes('technical report') || headline.includes('ni 43-101')

      // Must mention the project name
      const mentionsProject = headline.includes(projectLower)

      // Must NOT be a consent/certificate (these are 1-page docs)
      const notConsent = !headline.includes('consent') && !headline.includes('certificate')

      return isTechnical && mentionsProject && notConsent
    })

    return technicalReports
  } catch (error) {
    console.log(`      ❌ Error searching:`, error)
    return []
  }
}

async function main() {
  console.log('🏭 EXTRACT ALL PROJECT DATA - THE RIGHT WAY')
  console.log('='.repeat(80))
  console.log('Searching for SPECIFIC technical reports for EACH project')
  console.log('Each project gets ITS OWN documents, not shared documents!')
  console.log('='.repeat(80))

  // Wait for dev server
  console.log('\n⏳ Waiting 5s for dev server...')
  await new Promise(r => setTimeout(r, 5000))

  // Clear old projects
  console.log('🗑️  Clearing old projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')
  console.log('✅ Cleared\n')

  let totalProjects = 0
  let totalDocs = 0
  let totalExtracted = 0

  for (const company of COMPANIES_WITH_PROJECTS) {
    console.log(`\n📊 ${company.name} (${company.ticker})`)
    console.log('-'.repeat(80))

    // Create company
    const companyId = crypto.randomUUID()
    await supabase.from('companies').insert({
      id: companyId,
      name: company.name,
      ticker: company.ticker,
      description: `${company.commodity} mining company`
    })

    for (const projectName of company.projects) {
      console.log(`\n   🏗️  ${projectName}`)

      // Search specifically for THIS project
      const projectDocs = await searchForProjectDocuments(company.ticker, projectName)

      console.log(`      📚 Found ${projectDocs.length} technical reports for ${projectName}`)

      if (projectDocs.length === 0) {
        console.log(`      ⚠️  No technical reports found for ${projectName}, skipping`)
        continue
      }

      const storedUrls: string[] = []
      const extractedDataArray: ExtractedData[] = []

      // Download up to 3 documents
      for (let i = 0; i < Math.min(projectDocs.length, 3); i++) {
        const doc = projectDocs[i]
        const pdfUrl = doc.filingsLink?.replace('report=story', 'report=pdf')

        if (!pdfUrl) continue

        console.log(`      📥 Doc ${i + 1}: ${doc.headline?.substring(0, 70)}...`)

        try {
          // Download PDF
          const pdfResponse = await fetch(pdfUrl, {
            headers: { 'Authorization': authHeader }
          })

          if (!pdfResponse.ok) {
            console.log(`         ❌ Download failed: ${pdfResponse.status}`)
            continue
          }

          const arrayBuffer = await pdfResponse.arrayBuffer()
          const buffer = Buffer.from(arrayBuffer)
          const sizeInMB = (buffer.length / 1024 / 1024).toFixed(2)

          console.log(`         📦 Downloaded ${sizeInMB}MB`)

          // Only store if it's actually a large document (>1MB = likely 100+ pages)
          if (buffer.length < 1000000) {
            console.log(`         ⚠️  Too small (${sizeInMB}MB), skipping`)
            continue
          }

          // Upload to Supabase Storage
          const fileName = `${companyId}/${projectName.replace(/\s/g, '-')}/doc-${i + 1}-${Date.now()}.pdf`
          const { error: uploadError } = await supabase.storage
            .from('mining-documents')
            .upload(fileName, buffer, {
              contentType: 'application/pdf',
              upsert: true
            })

          if (uploadError) {
            console.log(`         ❌ Upload error: ${uploadError.message}`)
            continue
          }

          const { data: { publicUrl } } = supabase.storage
            .from('mining-documents')
            .getPublicUrl(fileName)

          storedUrls.push(publicUrl)
          totalDocs++
          console.log(`         ✅ Stored`)

          // Extract financial data
          const extractedData = await extractDataViaAPI(publicUrl, projectName)
          extractedDataArray.push(extractedData)

        } catch (error) {
          console.log(`         ❌ Error:`, error)
        }

        await new Promise(r => setTimeout(r, 3000))
      }

      // Merge extracted data from all documents
      const mergedData: ExtractedData = {}

      const npvValues = extractedDataArray.filter(d => d.npv).map(d => d.npv!)
      if (npvValues.length > 0) mergedData.npv = Math.max(...npvValues)

      const irrValues = extractedDataArray.filter(d => d.irr).map(d => d.irr!)
      if (irrValues.length > 0) mergedData.irr = Math.max(...irrValues)

      const capexValues = extractedDataArray.filter(d => d.capex).map(d => d.capex!)
      if (capexValues.length > 0) mergedData.capex = Math.max(...capexValues)

      // Create project with the documents we found
      if (storedUrls.length > 0) {
        const projectData: any = {
          id: crypto.randomUUID(),
          company_id: companyId,
          name: `${projectName} (${company.name})`,
          location: 'Various',
          stage: 'Operating',
          commodities: [company.commodity],
          status: 'Active',
          urls: storedUrls,
          description: `${storedUrls.length} technical reports for ${projectName}`,
          watchlist: false
        }

        // Add financial metrics if extracted
        if (mergedData.npv) {
          projectData.post_tax_npv_usd_m = mergedData.npv
          totalExtracted++
        }
        if (mergedData.irr) projectData.irr_percent = mergedData.irr
        if (mergedData.capex) projectData.capex_usd_m = mergedData.capex

        const { error } = await supabase.from('projects').insert(projectData)

        if (error) {
          console.log(`      ❌ Insert error: ${error.message}`)
        } else {
          console.log(`      ✅ Created project with ${storedUrls.length} documents`)
          if (mergedData.npv || mergedData.irr || mergedData.capex) {
            console.log(`      💎 NPV: $${mergedData.npv || 'N/A'}M | IRR: ${mergedData.irr || 'N/A'}% | CAPEX: $${mergedData.capex || 'N/A'}M`)
          }
          totalProjects++
        }
      } else {
        console.log(`      ⚠️  No valid documents stored for ${projectName}`)
      }

      await new Promise(r => setTimeout(r, 2000))
    }
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ COMPLETE!`)
  console.log(`   📊 ${totalProjects} projects with correct documents`)
  console.log(`   📄 ${totalDocs} technical reports stored`)
  console.log(`   💰 ${totalExtracted} projects with extracted financial data`)
  console.log(`${'='.repeat(80)}`)
}

main().catch(console.error)
