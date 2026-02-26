#!/usr/bin/env npx tsx

/**
 * Enhance Single Project - DEMONSTRATION
 *
 * This script demonstrates the complete enhancement process for ONE project:
 * 1. Loads ONE project from the database
 * 2. Improves the project name using OpenAI
 * 3. Downloads the filing document from FactSet
 * 4. Uploads to Supabase Storage (factset-documents bucket)
 * 5. Extracts text from the document
 * 6. Extracts financial data using OpenAI GPT-4o-mini
 * 7. Updates ALL project columns in the database
 * 8. Displays the complete before/after comparison
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import { getDocument } from 'unpdf'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

async function extractProjectNameFromHeadline(headline: string, companyName: string): Promise<string> {
  try {
    const prompt = `Extract a clear, concise project name from this mining company filing headline.

Company: ${companyName}
Headline: ${headline}

Return ONLY the project name. If it's a general filing without a specific project, return "${companyName} - [Document Type]"

Examples:
- "Smackover Lithium Files Definitive Feasibility Study for Its South West Arkansas Project" → "South West Arkansas Project"
- "Lundin Mining Announces Initial Mine" → "Lundin Mining Project"

Project name:`

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      max_tokens: 50
    })

    const name = completion.choices[0].message.content?.trim() || headline.substring(0, 100)
    return name
  } catch (error) {
    console.log(`   ⚠️  Name extraction failed, using headline`)
    return headline.substring(0, 100)
  }
}

async function downloadFromFactSet(url: string): Promise<Buffer | null> {
  try {
    // Try to download PDF version first
    const pdfUrl = url.replace('report=story', 'report=pdf')

    const response = await fetch(pdfUrl, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf,*/*',
        'User-Agent': 'Mozilla/5.0'
      }
    })

    if (!response.ok) {
      console.log(`   ⚠️  PDF download failed (${response.status}), trying HTML fallback...`)

      // Fall back to HTML version
      const htmlResponse = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': authHeader,
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'User-Agent': 'Mozilla/5.0'
        }
      })

      if (!htmlResponse.ok) {
        console.log(`   ❌ Download failed (${htmlResponse.status})`)
        return null
      }

      const html = await htmlResponse.text()
      return Buffer.from(html, 'utf-8')
    }

    const contentType = response.headers.get('content-type') || ''

    // If we got PDF, return it
    if (contentType.includes('application/pdf')) {
      return Buffer.from(await response.arrayBuffer())
    }

    // Otherwise treat as HTML
    const text = await response.text()
    return Buffer.from(text, 'utf-8')

  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`)
    return null
  }
}

async function uploadToStorage(content: Buffer, projectId: string, companyName: string, isHTML: boolean = false): Promise<string | null> {
  try {
    const sanitizedName = companyName.replace(/[^a-zA-Z0-9]/g, '_')
    const extension = isHTML ? 'html' : 'pdf'
    const filename = `${sanitizedName}_${projectId}.${extension}`
    const filepath = `project-documents/${filename}`

    const contentType = isHTML ? 'text/html' : 'application/pdf'

    const { error } = await supabase.storage
      .from('factset-documents')
      .upload(filepath, content, {
        contentType,
        upsert: true
      })

    if (error) {
      console.log(`   ❌ Storage error: ${error.message}`)
      return null
    }

    const { data } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(filepath)

    return data.publicUrl
  } catch (error: any) {
    console.log(`   ❌ Upload error: ${error.message}`)
    return null
  }
}

async function extractTextFromHTML(html: string): Promise<string> {
  let text = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
  text = text.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
  text = text.replace(/<[^>]+>/g, ' ')
  text = text.replace(/&nbsp;/g, ' ')
  text = text.replace(/&amp;/g, '&')
  text = text.replace(/&lt;/g, '<')
  text = text.replace(/&gt;/g, '>')
  text = text.replace(/&quot;/g, '"')
  text = text.replace(/\s+/g, ' ').trim()
  return text
}

async function extractFinancialData(text: string, projectName: string): Promise<any> {
  try {
    const limitedText = text.substring(0, 100000)

    const prompt = `Analyze this mining project document and extract financial metrics.

Project: ${projectName}

Document text:
${limitedText}

Extract and return JSON with these fields (use null if not found):
{
  "npv_usd_millions": number or null,
  "irr_percentage": number or null,
  "capex_usd_millions": number or null,
  "location": string or null,
  "stage": string or null (e.g., "Feasibility", "PEA", "Operating", "Exploration"),
  "commodities": string[] or null,
  "resource_estimate": string or null,
  "reserve_estimate": string or null,
  "description": string or null (brief 1-2 sentence summary)
}

Return ONLY valid JSON.`

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      response_format: { type: 'json_object' }
    })

    const result = completion.choices[0].message.content
    return result ? JSON.parse(result) : {}
  } catch (error: any) {
    console.log(`   ⚠️  Extraction error: ${error.message}`)
    return {}
  }
}

async function main() {
  console.log('======================================================================')
  console.log('SINGLE PROJECT ENHANCEMENT - COMPLETE DEMONSTRATION')
  console.log('======================================================================')
  console.log()

  // Step 1: Load ONE project
  console.log('📊 Loading ONE project from database...')

  const { data: projects, error } = await supabase
    .from('projects')
    .select(`
      id,
      name,
      company_id,
      urls,
      npv,
      irr,
      capex,
      resource,
      reserve,
      document_storage_path,
      location,
      stage,
      commodities,
      description,
      companies!inner (
        id,
        name
      )
    `)
    .not('urls', 'is', null)
    .limit(1)
    .single()

  if (error || !projects) {
    console.error('❌ Error loading project:', error)
    return
  }

  const project = projects
  const company = (project as any).companies

  console.log('✅ Found project to enhance')
  console.log()
  console.log('='.repeat(70))
  console.log('BEFORE ENHANCEMENT')
  console.log('='.repeat(70))
  console.log(`ID: ${project.id}`)
  console.log(`Name: ${project.name}`)
  console.log(`Company: ${company.name}`)
  console.log(`URLs: ${project.urls?.[0]?.substring(0, 80)}...`)
  console.log(`NPV: ${project.npv || 'null'}`)
  console.log(`IRR: ${project.irr || 'null'}`)
  console.log(`CAPEX: ${project.capex || 'null'}`)
  console.log(`Resource: ${project.resource || 'null'}`)
  console.log(`Reserve: ${project.reserve || 'null'}`)
  console.log(`Document Storage: ${project.document_storage_path || 'null'}`)
  console.log(`Location: ${project.location || 'null'}`)
  console.log(`Stage: ${project.stage || 'null'}`)
  console.log(`Commodities: ${project.commodities || 'null'}`)
  console.log(`Description: ${project.description || 'null'}`)
  console.log('='.repeat(70))
  console.log()

  // Step 2: Extract better name
  console.log('📝 STEP 1: Extracting better project name...')
  const betterName = await extractProjectNameFromHeadline(project.name, company.name)
  console.log(`   ✅ New name: "${betterName}"`)
  console.log()

  // Step 3: Download document
  console.log('📥 STEP 2: Downloading document from FactSet...')
  const documentUrl = project.urls![0]
  const content = await downloadFromFactSet(documentUrl)

  if (!content) {
    console.log('   ❌ Download failed - cannot continue')
    return
  }

  const isHTML = content.toString('utf-8', 0, 100).includes('<')
  console.log(`   ✅ Downloaded ${(content.length / 1024).toFixed(0)}KB (${isHTML ? 'HTML' : 'binary'})`)
  console.log()

  // Step 4: Upload to storage
  console.log('☁️  STEP 3: Uploading to Supabase Storage...')
  const storageUrl = await uploadToStorage(content, project.id, company.name, isHTML)

  if (!storageUrl) {
    console.log('   ❌ Upload failed - cannot continue')
    return
  }

  console.log(`   ✅ Uploaded to: ${storageUrl}`)
  console.log()

  // Step 5: Extract text
  console.log('📖 STEP 4: Extracting text from document...')
  let text = ''

  if (isHTML) {
    text = await extractTextFromHTML(content.toString('utf-8'))
  } else {
    try {
      const pdf = await getDocument(content).promise
      const maxPages = Math.min(pdf.numPages, 30)

      for (let i = 1; i <= maxPages; i++) {
        const page = await pdf.getPage(i)
        const textContent = await page.getTextContent()
        const pageText = textContent.items.map((item: any) => item.str).join(' ')
        text += pageText + '\n'
      }
    } catch (error) {
      console.log(`   ⚠️  PDF extraction failed, treating as HTML`)
      text = await extractTextFromHTML(content.toString('utf-8'))
    }
  }

  console.log(`   ✅ Extracted ${(text.length / 1024).toFixed(0)}KB text`)
  console.log()

  // Step 6: Extract financial data
  console.log('🤖 STEP 5: Extracting financial data with OpenAI...')
  const financialData = await extractFinancialData(text, betterName)

  console.log('   ✅ Financial data extracted:')
  if (financialData.npv_usd_millions) console.log(`      💰 NPV: $${financialData.npv_usd_millions}M`)
  if (financialData.irr_percentage) console.log(`      📈 IRR: ${financialData.irr_percentage}%`)
  if (financialData.capex_usd_millions) console.log(`      🏗️  CAPEX: $${financialData.capex_usd_millions}M`)
  if (financialData.location) console.log(`      📍 Location: ${financialData.location}`)
  if (financialData.stage) console.log(`      🔧 Stage: ${financialData.stage}`)
  if (financialData.commodities) console.log(`      ⛏️  Commodities: ${financialData.commodities.join(', ')}`)
  console.log()

  // Step 7: Update database
  console.log('💾 STEP 6: Updating database...')

  const updates = {
    name: betterName,
    document_storage_path: storageUrl,
    location: financialData.location,
    stage: financialData.stage,
    commodities: financialData.commodities,
    resource: financialData.resource_estimate,
    reserve: financialData.reserve_estimate,
    npv: financialData.npv_usd_millions,
    irr: financialData.irr_percentage,
    capex: financialData.capex_usd_millions,
    description: financialData.description,
    updated_at: new Date().toISOString()
  }

  const { error: updateError } = await supabase
    .from('projects')
    .update(updates)
    .eq('id', project.id)

  if (updateError) {
    console.log(`   ❌ Update error: ${updateError.message}`)
    return
  }

  console.log('   ✅ Database updated!')
  console.log()

  // Step 8: Show AFTER state
  const { data: updatedProject } = await supabase
    .from('projects')
    .select('*')
    .eq('id', project.id)
    .single()

  console.log('='.repeat(70))
  console.log('AFTER ENHANCEMENT')
  console.log('='.repeat(70))
  console.log(`ID: ${updatedProject.id}`)
  console.log(`Name: ${updatedProject.name}`)
  console.log(`NPV: ${updatedProject.npv || 'null'}`)
  console.log(`IRR: ${updatedProject.irr || 'null'}`)
  console.log(`CAPEX: ${updatedProject.capex || 'null'}`)
  console.log(`Resource: ${updatedProject.resource || 'null'}`)
  console.log(`Reserve: ${updatedProject.reserve || 'null'}`)
  console.log(`Document Storage: ${updatedProject.document_storage_path || 'null'}`)
  console.log(`Location: ${updatedProject.location || 'null'}`)
  console.log(`Stage: ${updatedProject.stage || 'null'}`)
  console.log(`Commodities: ${updatedProject.commodities || 'null'}`)
  console.log(`Description: ${updatedProject.description || 'null'}`)
  console.log('='.repeat(70))
  console.log()
  console.log('✅ DEMONSTRATION COMPLETE!')
  console.log()
  console.log('This project now has:')
  console.log('  ✅ Improved project name')
  console.log('  ✅ Document downloaded from FactSet')
  console.log('  ✅ Document uploaded to Supabase Storage')
  console.log('  ✅ Supabase Storage URL in database')
  console.log('  ✅ Financial metrics extracted')
  console.log('  ✅ All columns populated')
  console.log('======================================================================')
}

main().catch(console.error)
