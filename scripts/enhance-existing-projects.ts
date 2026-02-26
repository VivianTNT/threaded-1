#!/usr/bin/env npx tsx

/**
 * Enhance Existing Projects Script
 *
 * This script:
 * 1. Loads existing projects from database
 * 2. Improves project names using OpenAI
 * 3. Downloads PDF documents from FactSet filing URLs
 * 4. Uploads PDFs to Supabase Storage
 * 5. Extracts financial data using OpenAI
 * 6. Updates all project columns
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

interface Project {
  id: string
  name: string
  company_id: string
  urls: string[] | null
}

interface Company {
  id: string
  name: string
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function extractProjectNameFromHeadline(headline: string, companyName: string): Promise<string> {
  try {
    const prompt = `Extract a clear, concise project name from this mining company filing headline.

Company: ${companyName}
Headline: ${headline}

Return ONLY the project name. If it's a general filing without a specific project, return "${companyName} - [Document Type]"

Examples:
- "Smackover Lithium Files Definitive Feasibility Study for Its South West Arkansas Project" → "South West Arkansas Project"
- "Standard Lithium Ltd. - Filing 51ee219649e4682e731d5ff54e656dad" → "Standard Lithium - Technical Report"
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
    console.log(`      ⚠️  Name extraction failed, using headline`)
    return headline.substring(0, 100)
  }
}

async function downloadPDFFromFilingLink(url: string): Promise<Buffer | null> {
  try {
    // Convert the filing link to PDF format by replacing report=story with report=pdf
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
      console.log(`      ⚠️  PDF download failed (${response.status}), trying HTML...`)

      // Fall back to HTML version if PDF not available
      const htmlResponse = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': authHeader,
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'User-Agent': 'Mozilla/5.0'
        }
      })

      if (!htmlResponse.ok) {
        return null
      }

      const html = await htmlResponse.text()
      return Buffer.from(html, 'utf-8')
    }

    const contentType = response.headers.get('content-type') || ''

    // We got the PDF
    if (contentType.includes('application/pdf')) {
      return Buffer.from(await response.arrayBuffer())
    }

    // Fallback: if it's not PDF, treat as HTML
    const text = await response.text()
    return Buffer.from(text, 'utf-8')

  } catch (error: any) {
    console.log(`      ⚠️  Error: ${error.message}`)
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
      console.log(`      ⚠️  Storage error: ${error.message}`)
      return null
    }

    const { data } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(filepath)

    return data.publicUrl
  } catch (error: any) {
    console.log(`      ⚠️  Upload error: ${error.message}`)
    return null
  }
}

async function extractTextFromHTML(html: string): Promise<string> {
  // Remove script and style tags
  let text = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
  text = text.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')

  // Remove HTML tags
  text = text.replace(/<[^>]+>/g, ' ')

  // Decode HTML entities
  text = text.replace(/&nbsp;/g, ' ')
  text = text.replace(/&amp;/g, '&')
  text = text.replace(/&lt;/g, '<')
  text = text.replace(/&gt;/g, '>')
  text = text.replace(/&quot;/g, '"')

  // Clean up whitespace
  text = text.replace(/\s+/g, ' ').trim()

  return text
}

async function extractFinancialData(text: string, projectName: string): Promise<any> {
  try {
    // Limit text to first 100KB for API limits
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
  "opex_usd_millions": number or null,
  "payback_period_years": number or null,
  "mine_life_years": number or null,
  "production_rate": string or null,
  "discount_rate_percentage": number or null,
  "location": string or null,
  "stage": string or null (e.g., "Feasibility", "PEA", "Operating", "Exploration"),
  "commodities": string[] or null,
  "resource_estimate": string or null,
  "reserve_estimate": string or null
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
    console.log(`      ⚠️  Extraction error: ${error.message}`)
    return {}
  }
}

async function updateProject(projectId: string, updates: any): Promise<boolean> {
  try {
    const { error } = await supabase
      .from('projects')
      .update({
        ...updates,
        updated_at: new Date().toISOString()
      })
      .eq('id', projectId)

    if (error) {
      console.log(`      ⚠️  Update error: ${error.message}`)
      return false
    }

    return true
  } catch (error: any) {
    console.log(`      ⚠️  Update failed: ${error.message}`)
    return false
  }
}

async function processProject(project: Project, company: Company, index: number, total: number): Promise<boolean> {
  console.log(`\n[${index}/${total}] ${project.name}`)
  console.log(`   Company: ${company.name}`)

  // Step 1: Extract better project name
  console.log('   📝 Extracting project name...')
  const betterName = await extractProjectNameFromHeadline(project.name, company.name)
  console.log(`   ✅ New name: ${betterName}`)

  // Step 2: Download document from first URL
  if (!project.urls || project.urls.length === 0) {
    console.log('   ⚠️  No URLs found for this project')
    return false
  }

  const documentUrl = project.urls[0]
  console.log('   ⬇️  Downloading document...')

  const content = await downloadPDFFromFilingLink(documentUrl)

  if (!content) {
    console.log('   ❌ Download failed')
    return false
  }

  const isHTML = content.toString('utf-8', 0, 100).includes('<')
  console.log(`   ✅ Downloaded ${(content.length / 1024).toFixed(0)}KB (${isHTML ? 'HTML' : 'binary'})`)

  // Step 3: Upload to storage
  console.log('   ☁️  Uploading to storage...')
  const storageUrl = await uploadToStorage(content, project.id, company.name, isHTML)

  if (!storageUrl) {
    console.log('   ❌ Upload failed')
    return false
  }

  console.log('   ✅ Uploaded to storage')

  // Step 4: Extract text
  console.log('   📖 Extracting text...')
  let text = ''

  if (isHTML) {
    text = await extractTextFromHTML(content.toString('utf-8'))
  } else {
    // Try to extract as PDF
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
      console.log(`      ⚠️  PDF extraction failed, treating as HTML`)
      text = await extractTextFromHTML(content.toString('utf-8'))
    }
  }

  if (!text || text.length < 500) {
    console.log('   ⚠️  Insufficient text extracted')
    // Still update name and storage URL
    await updateProject(project.id, {
      name: betterName,
      document_storage_path: storageUrl
    })
    return false
  }

  console.log(`   ✅ Extracted ${(text.length / 1024).toFixed(0)}KB text`)

  // Step 5: Extract financial data
  console.log('   🤖 Extracting financial data...')
  const financialData = await extractFinancialData(text, betterName)

  // Step 6: Update project
  console.log('   💾 Updating project...')

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
    capex: financialData.capex_usd_millions
  }

  const success = await updateProject(project.id, updates)

  if (success) {
    console.log('   ✅ Project updated!')

    // Log extracted metrics
    if (financialData.npv_usd_millions) {
      console.log(`      💰 NPV: $${financialData.npv_usd_millions}M`)
    }
    if (financialData.irr_percentage) {
      console.log(`      📈 IRR: ${financialData.irr_percentage}%`)
    }
    if (financialData.capex_usd_millions) {
      console.log(`      🏗️  CAPEX: $${financialData.capex_usd_millions}M`)
    }
    if (financialData.location) {
      console.log(`      📍 Location: ${financialData.location}`)
    }
  }

  return success
}

async function main() {
  console.log('======================================================================')
  console.log('ENHANCE EXISTING PROJECTS - DOWNLOAD, EXTRACT & UPDATE')
  console.log('======================================================================')
  console.log()

  // Load all projects with their company info
  console.log('📊 Loading projects from database...')

  const { data: projects, error } = await supabase
    .from('projects')
    .select(`
      id,
      name,
      company_id,
      urls,
      companies!inner (
        id,
        name
      )
    `)
    .not('urls', 'is', null)
    .order('created_at', { ascending: false })

  if (error) {
    console.error('❌ Error loading projects:', error)
    return
  }

  console.log(`✅ Found ${projects.length} projects with URLs`)
  console.log()

  let successCount = 0
  let failCount = 0

  for (let i = 0; i < projects.length; i++) {
    const project = projects[i]
    const company = (project as any).companies

    const success = await processProject(
      {
        id: project.id,
        name: project.name,
        company_id: project.company_id,
        urls: project.urls
      },
      company,
      i + 1,
      projects.length
    )

    if (success) {
      successCount++
    } else {
      failCount++
    }

    // Rate limiting
    await sleep(300)
  }

  console.log()
  console.log('======================================================================')
  console.log('ENHANCEMENT COMPLETE')
  console.log('======================================================================')
  console.log()
  console.log('📊 Summary:')
  console.log(`   Total projects: ${projects.length}`)
  console.log(`   Successfully enhanced: ${successCount}`)
  console.log(`   Failed: ${failCount}`)
  console.log()
  console.log('✅ All projects have been enhanced with:')
  console.log('   - Improved project names')
  console.log('   - Downloaded documents in storage')
  console.log('   - Extracted financial metrics')
  console.log('======================================================================')
}

main().catch(console.error)
