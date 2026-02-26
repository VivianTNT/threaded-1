#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'
import * as pdf from 'pdf-parse'
const pdfParse = pdf.default || pdf
import OpenAI from 'openai'
import { v4 as uuidv4 } from 'uuid'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})

interface ExtractedData {
  projectName: string
  companyName: string
  location: string
  commodities: string[]
  npv?: number
  irr?: number
  capex?: number
  opex?: number
  resource?: string
  reserve?: string
  mineLife?: number
  discountRate?: number
  production?: string
  grade?: string
}

async function extractDataFromPDF(pdfBuffer: Buffer, fileName: string): Promise<ExtractedData | null> {
  try {
    console.log(`   📖 Parsing PDF: ${fileName}`)
    const data = await pdfParse(pdfBuffer)
    const text = data.text
    const pageCount = data.numpages

    console.log(`   📄 Pages: ${pageCount}, Characters: ${text.length}`)

    if (text.length < 1000) {
      console.log(`   ⚠️  Document too short, skipping`)
      return null
    }

    // Extract key sections for analysis
    const sections = {
      executive: text.slice(0, 15000), // First ~15k chars for executive summary
      financial: extractSection(text, ['economic', 'financial', 'npv', 'irr', 'capex'], 10000),
      resource: extractSection(text, ['resource', 'reserve', 'mineral', 'ore'], 10000),
      project: extractSection(text, ['project', 'property', 'location', 'mine'], 10000)
    }

    const prompt = `Extract mining project information from this technical report. Return ONLY a JSON object with these fields:
{
  "projectName": "name of the mining project",
  "companyName": "company operating the project",
  "location": "location (country, state/province)",
  "commodities": ["list", "of", "commodities"],
  "npv": number in millions USD (null if not found),
  "irr": percentage as decimal (e.g., 0.15 for 15%),
  "capex": number in millions USD,
  "opex": number per unit (e.g., $/oz, $/lb),
  "resource": "measured + indicated resource string",
  "reserve": "proven + probable reserve string",
  "mineLife": years as number,
  "discountRate": percentage as decimal,
  "production": "annual production rate",
  "grade": "average grade"
}

Document sections:
EXECUTIVE SUMMARY: ${sections.executive.slice(0, 3000)}
FINANCIAL: ${sections.financial}
RESOURCE: ${sections.resource}
PROJECT: ${sections.project}

Return ONLY valid JSON, no explanation.`

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 1000
    })

    const content = response.choices[0]?.message?.content
    if (!content) return null

    // Clean and parse JSON
    const jsonStr = content.replace(/```json\n?|\n?```/g, '').trim()
    const extracted = JSON.parse(jsonStr) as ExtractedData

    // Validate extracted data
    if (!extracted.projectName && !extracted.companyName) {
      console.log(`   ⚠️  No project or company name extracted`)
      return null
    }

    console.log(`   ✅ Extracted: ${extracted.projectName || 'Unknown'} by ${extracted.companyName || 'Unknown'}`)

    return extracted
  } catch (error: any) {
    console.log(`   ❌ Extraction error: ${error.message}`)
    return null
  }
}

function extractSection(text: string, keywords: string[], maxLength: number = 5000): string {
  const lowerText = text.toLowerCase()

  for (const keyword of keywords) {
    const index = lowerText.indexOf(keyword.toLowerCase())
    if (index !== -1) {
      return text.slice(Math.max(0, index - 500), Math.min(text.length, index + maxLength))
    }
  }

  return ''
}

async function processDocuments() {
  console.log('=' .repeat(80))
  console.log('📊 EXTRACTING DATA FROM MINING DOCUMENTS')
  console.log('=' .repeat(80))

  // List all documents in mining-documents folder
  console.log('\n📁 Fetching documents from technical-documents/mining-documents/')

  const { data: files, error: listError } = await supabase.storage
    .from('technical-documents')
    .list('mining-documents', { limit: 100 })

  if (listError || !files) {
    console.error('Error listing files:', listError)
    return
  }

  const pdfFiles = files.filter(f => f.name.endsWith('.pdf'))
  console.log(`Found ${pdfFiles.length} PDF documents to process\n`)

  // Process each PDF
  const extractedProjects: any[] = []

  for (let i = 0; i < pdfFiles.length; i++) {
    const file = pdfFiles[i]
    const filePath = `mining-documents/${file.name}`
    const sizeMB = ((file.metadata?.size || 0) / 1024 / 1024).toFixed(2)

    console.log(`\n[${i + 1}/${pdfFiles.length}] Processing: ${file.name} (${sizeMB} MB)`)
    console.log('-' .repeat(60))

    try {
      // Download the PDF
      const { data: pdfData, error: downloadError } = await supabase.storage
        .from('technical-documents')
        .download(filePath)

      if (downloadError || !pdfData) {
        console.log(`   ❌ Download failed: ${downloadError?.message}`)
        continue
      }

      // Convert blob to buffer
      const buffer = Buffer.from(await pdfData.arrayBuffer())

      // Extract data using OpenAI
      const extractedData = await extractDataFromPDF(buffer, file.name)

      if (extractedData) {
        // Generate public URL for the document
        const { data: urlData } = supabase.storage
          .from('technical-documents')
          .getPublicUrl(filePath)

        // Check if company exists or create one
        let companyId = null
        if (extractedData.companyName) {
          const { data: existingCompany } = await supabase
            .from('companies')
            .select('id')
            .ilike('name', extractedData.companyName)
            .single()

          if (existingCompany) {
            companyId = existingCompany.id
            console.log(`   🏢 Found existing company: ${extractedData.companyName}`)
          } else {
            const { data: newCompany, error: companyError } = await supabase
              .from('companies')
              .insert({
                name: extractedData.companyName,
                ticker: extractedData.companyName.slice(0, 5).toUpperCase(),
                sector: 'Mining',
                market_cap: null
              })
              .select()
              .single()

            if (newCompany) {
              companyId = newCompany.id
              console.log(`   🏢 Created company: ${extractedData.companyName}`)
            } else if (companyError) {
              console.log(`   ⚠️  Company creation failed: ${companyError.message}`)
            }
          }
        }

        // Create project record
        const projectData = {
          id: uuidv4(),
          name: extractedData.projectName || `Mining Project ${i + 1}`,
          company_id: companyId,
          location: extractedData.location || 'Unknown',
          commodities: extractedData.commodities || [],
          npv: extractedData.npv,
          irr: extractedData.irr,
          capex: extractedData.capex,
          opex: extractedData.opex,
          resource: extractedData.resource,
          reserve: extractedData.reserve,
          mine_life_years: extractedData.mineLife,
          discount_rate: extractedData.discountRate,
          production_capacity: extractedData.production,
          grade: extractedData.grade,
          document_storage_path: filePath,
          document_urls: [urlData.publicUrl],
          technical_report_url: urlData.publicUrl,
          source: 'factset',
          created_at: new Date().toISOString()
        }

        const { data: project, error: projectError } = await supabase
          .from('projects')
          .insert(projectData)
          .select()
          .single()

        if (project) {
          console.log(`   ✅ Created project: ${project.name}`)
          extractedProjects.push(project)

          // Add PDF highlights for key metrics
          const highlights = []

          if (extractedData.npv) {
            highlights.push({
              project_id: project.id,
              data_type: 'NPV',
              page: 1,
              value: `$${extractedData.npv}M`,
              quote: `Net Present Value: $${extractedData.npv} million`
            })
          }

          if (extractedData.irr) {
            highlights.push({
              project_id: project.id,
              data_type: 'IRR',
              page: 1,
              value: `${(extractedData.irr * 100).toFixed(1)}%`,
              quote: `Internal Rate of Return: ${(extractedData.irr * 100).toFixed(1)}%`
            })
          }

          if (extractedData.capex) {
            highlights.push({
              project_id: project.id,
              data_type: 'CAPEX',
              page: 1,
              value: `$${extractedData.capex}M`,
              quote: `Initial Capital Expenditure: $${extractedData.capex} million`
            })
          }

          if (highlights.length > 0) {
            const { error: highlightError } = await supabase
              .from('pdf_highlights')
              .insert(highlights)

            if (!highlightError) {
              console.log(`   📍 Added ${highlights.length} highlights`)
            }
          }
        } else if (projectError) {
          console.log(`   ❌ Project creation failed: ${projectError.message}`)
        }
      }
    } catch (error: any) {
      console.log(`   ❌ Processing error: ${error.message}`)
    }
  }

  console.log('\n' + '=' .repeat(80))
  console.log('📊 EXTRACTION SUMMARY')
  console.log('=' .repeat(80))
  console.log(`Documents processed: ${pdfFiles.length}`)
  console.log(`Projects created: ${extractedProjects.length}`)

  if (extractedProjects.length > 0) {
    console.log('\n📋 Created Projects:')
    extractedProjects.forEach(p => {
      console.log(`  • ${p.name}`)
      if (p.npv || p.irr || p.capex) {
        const metrics = []
        if (p.npv) metrics.push(`NPV: $${p.npv}M`)
        if (p.irr) metrics.push(`IRR: ${(p.irr * 100).toFixed(1)}%`)
        if (p.capex) metrics.push(`CAPEX: $${p.capex}M`)
        console.log(`    ${metrics.join(' | ')}`)
      }
    })
  }

  console.log('\n✅ Extraction complete!')
}

processDocuments().catch(console.error)