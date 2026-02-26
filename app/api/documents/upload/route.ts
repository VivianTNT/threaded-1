import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { extractText, getDocumentProxy } from 'unpdf'
import OpenAI from 'openai'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

// Regex patterns to identify relevant sections (same as factset approach)
const SECTION_PATTERNS = {
  economics: /economic\s+analysis|financial\s+analysis|project\s+economics|economic\s+evaluation/i,
  summary: /executive\s+summary|summary\s+of\s+results|highlights|key\s+findings/i,
  npv: /net\s+present\s+value|npv|post-tax\s+npv|after-tax\s+npv|pre-tax\s+npv/i,
  irr: /internal\s+rate\s+of\s+return|irr/i,
  capex: /capital\s+cost|initial\s+capital|capex|capital\s+expenditure/i,
  resources: /mineral\s+resource|resource\s+estimate|measured.*indicated.*inferred/i,
  reserves: /mineral\s+reserve|reserve\s+estimate|proven.*probable/i,
  location: /location|jurisdiction|property\s+location|geographic\s+setting/i,
  commodity: /commodity|metal|mineral|deposit\s+type/i
}

/**
 * Find relevant sections - ALWAYS extract, don't filter by keywords
 * This ensures we extract from ALL documents, even if they don't match patterns
 */
function findRelevantSections(fullText: string): string {
  console.log(`   🔍 Preparing text for extraction (no keyword filtering)...`)

  // Strategy: Send the most relevant parts, but ALWAYS extract
  // 1. Try to find high-scoring chunks
  // 2. If no high-scoring chunks, use beginning of document
  // 3. Always send up to 80KB for better extraction

  const chunkSize = 5000
  const chunks: { text: string; score: number; index: number }[] = []

  for (let i = 0; i < fullText.length; i += chunkSize) {
    const chunk = fullText.substring(i, Math.min(i + chunkSize, fullText.length))

    // Score chunk based on keyword matches (for prioritization, not filtering)
    const score = [
      SECTION_PATTERNS.economics.test(chunk),
      SECTION_PATTERNS.summary.test(chunk),
      SECTION_PATTERNS.npv.test(chunk),
      SECTION_PATTERNS.irr.test(chunk),
      SECTION_PATTERNS.capex.test(chunk),
      SECTION_PATTERNS.resources.test(chunk),
      SECTION_PATTERNS.reserves.test(chunk)
    ].filter(Boolean).length

    chunks.push({ text: chunk, score, index: i / chunkSize })
  }

  // Sort by score (highest first)
  chunks.sort((a, b) => b.score - a.score)

  // Take top chunks (up to 80KB total for comprehensive extraction)
  let totalLength = 0
  const relevantChunks: string[] = []

  for (const chunk of chunks) {
    if (totalLength + chunk.text.length > 80000) break
    relevantChunks.push(chunk.text)
    totalLength += chunk.text.length
  }

  console.log(`   📊 Using ${relevantChunks.length} chunks (${(totalLength / 1024).toFixed(1)}KB) - scored chunks: ${chunks.filter(c => c.score > 0).length}/${chunks.length}`)

  // ALWAYS return text for extraction - never skip
  if (relevantChunks.length > 0) {
    return relevantChunks.join('\n\n')
  }

  // Fallback: if no chunks (empty document?), return first 80KB
  console.log(`   ⚠️  No chunks found, using first 80KB of document`)
  return fullText.substring(0, 80000)
}

/**
 * Extract structured data using OpenAI (same approach as factset)
 * Uses TWO-PASS extraction for better accuracy
 */
async function extractDataWithAI(text: string, fileName: string) {
  console.log(`   🤖 Extracting data with OpenAI GPT-4o-mini (two-pass approach)...`)

  try {
    // PASS 1: Extract with standard prompt
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `You are extracting financial and technical data from mining technical reports (NI 43-101, JORC, PFS, FS, etc.).

Extract the following data from the provided text for the document "${fileName}":

1. Company Name - the company that owns/operates the project (e.g., "Agnico Eagle Mines", "Barrick Gold")
2. Project Name - the mining project name
3. NPV (Net Present Value) - in millions USD (post-tax/after-tax preferred, but pre-tax/before-tax acceptable if clearly stated)
4. IRR (Internal Rate of Return) - as percentage (post-tax/after-tax preferred)
5. CAPEX (Capital Expenditure) - in millions USD (initial capital, upfront capital, or total capital costs)
6. Location - specific location (country, state/province)
7. Commodity - primary commodity or commodities (e.g., ["Gold"], ["Copper", "Gold"])
8. Resource Estimate - brief summary (e.g., "M&I: 2.5M oz Au @ 1.2g/t, Inferred: 0.8M oz")
9. Reserve Estimate - brief summary (e.g., "P&P: 1.8M oz Au @ 1.1g/t")
10. Description - 2-3 sentence project description
11. Stage - project development stage

CRITICAL EXTRACTION TIPS:
- Company Name: Look for "Prepared by", "Issued by", cover page, or "Owner" sections
- Look for tables labeled "Economic Analysis", "Project Economics", "Financial Summary"
- NPV and IRR are often in the same table or paragraph
- CAPEX may appear as "Total Initial Capital", "Upfront CAPEX", "LOM Capital"
- Look for phrases like "Base Case NPV", "After-Tax NPV @ X% discount rate"
- Numbers can be in format: $1,485M, $1.485B, US$1,485 million
- Extract ONLY the numeric value and convert to millions (e.g., $1.485B → 1485)
- If you find "Pre-Tax" and "Post-Tax" values, prefer Post-Tax
- Resources/Reserves: Look for M&I (Measured & Indicated), Inferred, P&P (Proven & Probable)

Return ONLY a valid JSON object with these exact keys:
{
  "company_name": string or null,
  "project_name": string or null,
  "npv": number or null,
  "irr": number or null,
  "capex": number or null,
  "location": string or null,
  "commodities": string[] or null,
  "resource": string or null,
  "reserve": string or null,
  "description": string or null,
  "stage": string or null
}

Example extractions:
- "Prepared by Agnico Eagle Mines Limited" → company_name: "Agnico Eagle Mines"
- "Post-Tax NPV (5%): $1,485M" → npv: 1485
- "After-Tax IRR: 17.5%" → irr: 17.5
- "Total Initial Capital: US$1.736 billion" → capex: 1736
- "Gold: 3.2 Moz @ 1.1 g/t Au" → commodities: ["Gold"]`
        },
        {
          role: 'user',
          content: text
        }
      ],
      temperature: 0.1,
      max_tokens: 2000
    })

    const content = response.choices[0].message.content
    if (!content) {
      console.log(`   ❌ No response from OpenAI (Pass 1)`)
      return null
    }

    // Parse JSON response
    const jsonMatch = content.match(/\{[\s\S]*\}/)
    if (!jsonMatch) {
      console.log(`   ❌ No JSON found in response (Pass 1)`)
      return null
    }

    const extracted = JSON.parse(jsonMatch[0])
    console.log(`   ✅ Pass 1 extracted:`, {
      npv: extracted.npv,
      irr: extracted.irr,
      capex: extracted.capex,
      location: extracted.location,
      commodities: extracted.commodities
    })

    // PASS 2: If we're missing critical financial data, try again with more focused prompt
    const missingFinancials = !extracted.npv || !extracted.irr || !extracted.capex

    if (missingFinancials && text.length > 10000) {
      console.log(`   🔄 Pass 2: Missing financials, trying targeted extraction...`)

      const pass2Response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: `You are a financial data extraction specialist for mining projects.

The following text is from "${fileName}". Your ONLY job is to find these THREE metrics:

1. NPV (Net Present Value) - Look for:
   - Tables with "NPV", "Net Present Value", "After-Tax NPV", "Post-Tax NPV"
   - Phrases like "NPV at 5% discount rate", "Base Case NPV"
   - Extract numeric value in millions (e.g., "$1,485M" → 1485, "$1.485B" → 1485)

2. IRR (Internal Rate of Return) - Look for:
   - Tables with "IRR", "Internal Rate of Return", "After-Tax IRR"
   - Usually appears as percentage (e.g., "17.5%", "IRR: 17.5%")
   - Extract just the number (e.g., "17.5%" → 17.5)

3. CAPEX (Capital Expenditure) - Look for:
   - Tables with "Total Capital", "Initial Capital", "CAPEX", "Capital Cost"
   - May be labeled "Total Initial Capital", "Upfront CAPEX", "LOM Capital"
   - Extract numeric value in millions (e.g., "$1.736B" → 1736)

Return ONLY JSON:
{
  "npv": number or null,
  "irr": number or null,
  "capex": number or null
}

BE VERY THOROUGH - scan the entire text for tables and financial summaries.`
          },
          {
            role: 'user',
            content: text
          }
        ],
        temperature: 0.1,
        max_tokens: 500
      })

      const pass2Content = pass2Response.choices[0].message.content
      if (pass2Content) {
        const pass2Match = pass2Content.match(/\{[\s\S]*\}/)
        if (pass2Match) {
          const pass2Data = JSON.parse(pass2Match[0])
          console.log(`   ✅ Pass 2 extracted:`, pass2Data)

          // Merge Pass 2 data into Pass 1 (Pass 2 overrides if values found)
          if (pass2Data.npv) extracted.npv = pass2Data.npv
          if (pass2Data.irr) extracted.irr = pass2Data.irr
          if (pass2Data.capex) extracted.capex = pass2Data.capex

          console.log(`   ✅ Final merged data:`, {
            npv: extracted.npv,
            irr: extracted.irr,
            capex: extracted.capex
          })
        }
      }
    }

    return extracted
  } catch (error: any) {
    console.log(`   ❌ OpenAI extraction failed: ${error.message}`)
    return null
  }
}

/**
 * POST handler for document upload
 */
export async function POST(req: NextRequest) {
  try {
    console.log('📤 Document upload request received')

    // Get the user from the Authorization header
    const authHeader = req.headers.get('authorization')
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const token = authHeader.substring(7)
    const { data: { user }, error: authError } = await supabase.auth.getUser(token)

    if (authError || !user) {
      console.error('❌ Auth error:', authError)
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    console.log(`✅ Authenticated user: ${user.id}`)

    // Parse FormData
    const formData = await req.formData()
    const file = formData.get('file') as File | null

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 })
    }

    console.log(`📄 File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`)

    // Validate file type
    if (!file.type.includes('pdf')) {
      return NextResponse.json({ error: 'Only PDF files are supported' }, { status: 400 })
    }

    // Upload to Supabase Storage
    const timestamp = Date.now()
    const sanitizedFileName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_')
    const storagePath = `${user.id}/${timestamp}_${sanitizedFileName}`

    console.log(`☁️  Uploading to storage: ${storagePath}`)

    const arrayBuffer = await file.arrayBuffer()
    const fileBuffer = Buffer.from(arrayBuffer)

    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('user-documents')
      .upload(storagePath, fileBuffer, {
        contentType: 'application/pdf',
        upsert: false
      })

    if (uploadError) {
      console.error('❌ Upload error:', uploadError)
      return NextResponse.json({ error: 'Failed to upload file' }, { status: 500 })
    }

    console.log(`✅ File uploaded successfully`)

    // Extract text from PDF using unpdf
    console.log(`📖 Extracting text from PDF...`)
    const uint8Array = new Uint8Array(arrayBuffer)
    const pdf = await getDocumentProxy(uint8Array)
    const { text: fullText, totalPages } = await extractText(pdf, { mergePages: true })

    console.log(`✅ Extracted ${totalPages} pages, ${(fullText.length / 1024).toFixed(1)}KB text`)

    // Find relevant sections using regex
    const relevantText = findRelevantSections(fullText)

    // Extract structured data using OpenAI
    const extracted = await extractDataWithAI(relevantText, file.name)

    if (!extracted) {
      console.log('⚠️  Failed to extract data, but document was uploaded')
    }

    // Generate a signed URL for authenticated access (valid for 1 year)
    // This allows the user to access their private document through the PDF viewer
    const { data: urlData, error: urlError } = await supabase.storage
      .from('user-documents')
      .createSignedUrl(storagePath, 31536000) // 1 year in seconds

    if (urlError) {
      console.error('❌ Error creating signed URL:', urlError)
    }

    const documentUrl = urlData?.signedUrl || null

    // Look up or create company if company name was extracted
    let companyId: string | null = null
    if (extracted?.company_name) {
      console.log(`🏢 Looking up company: ${extracted.company_name}`)

      // Try to find existing company by name (case-insensitive)
      const { data: existingCompanies } = await supabase
        .from('companies')
        .select('id, name')
        .ilike('name', extracted.company_name)
        .limit(1)

      if (existingCompanies && existingCompanies.length > 0) {
        companyId = existingCompanies[0].id
        console.log(`   ✅ Found existing company: ${existingCompanies[0].name} (${companyId})`)
      } else {
        // Create new company
        const { data: newCompany, error: companyError } = await supabase
          .from('companies')
          .insert({
            name: extracted.company_name,
            created_at: new Date().toISOString()
          })
          .select('id')
          .single()

        if (!companyError && newCompany) {
          companyId = newCompany.id
          console.log(`   ✅ Created new company: ${extracted.company_name} (${companyId})`)
        } else {
          console.log(`   ⚠️  Failed to create company: ${companyError?.message}`)
        }
      }
    }

    // Create project entry in database
    const projectData = {
      name: extracted?.project_name || file.name.replace('.pdf', ''),
      npv: extracted?.npv,
      irr: extracted?.irr,
      capex: extracted?.capex,
      location: extracted?.location,
      commodities: extracted?.commodities,
      resource: extracted?.resource,
      reserve: extracted?.reserve,
      description: extracted?.description,
      stage: extracted?.stage,
      user_id: user.id,
      is_private: true,
      uploaded_at: new Date().toISOString(),
      document_storage_path: storagePath,
      urls: documentUrl ? [documentUrl] : null,
      status: 'Active',
      company_id: companyId,
      watchlist: false
    }

    console.log(`💾 Creating project entry in database...`)

    const { data: project, error: projectError } = await supabase
      .from('projects')
      .insert(projectData)
      .select()
      .single()

    if (projectError) {
      console.error('❌ Project creation error:', projectError)
      // Clean up uploaded file
      await supabase.storage.from('user-documents').remove([storagePath])
      return NextResponse.json({ error: 'Failed to create project entry' }, { status: 500 })
    }

    console.log(`✅ Project created: ${project.id}`)

    // Fetch company name if company_id is set
    let companyName: string | null = null
    if (project.company_id) {
      const { data: companyData } = await supabase
        .from('companies')
        .select('name')
        .eq('id', project.company_id)
        .single()

      companyName = companyData?.name || null
    }

    // Return the ACTUAL database values, not the extraction attempt
    // This ensures we show what was successfully saved to the database
    return NextResponse.json({
      success: true,
      project: {
        id: project.id,
        name: project.name,
        storagePath,
        url: documentUrl
      },
      extracted: {
        company_name: companyName,
        project_name: project.name,
        location: project.location,
        npv: project.npv,
        irr: project.irr,
        capex: project.capex,
        commodities: project.commodities,
        resource: project.resource,
        reserve: project.reserve,
        stage: project.stage,
        description: project.description
      },
      message: `Document uploaded and ${extracted ? 'parsed' : 'stored'} successfully`
    })

  } catch (error: any) {
    console.error('❌ Upload error:', error)
    return NextResponse.json({
      error: 'Internal server error',
      details: error.message
    }, { status: 500 })
  }
}

/**
 * GET handler - retrieve user's uploaded documents
 */
export async function GET(req: NextRequest) {
  try {
    const authHeader = req.headers.get('authorization')
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const token = authHeader.substring(7)
    const { data: { user }, error: authError } = await supabase.auth.getUser(token)

    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Fetch user's private documents
    const { data: projects, error } = await supabase
      .from('projects')
      .select('*')
      .eq('user_id', user.id)
      .eq('is_private', true)
      .order('uploaded_at', { ascending: false })

    if (error) {
      console.error('Error fetching documents:', error)
      return NextResponse.json({ error: 'Failed to fetch documents' }, { status: 500 })
    }

    return NextResponse.json({ documents: projects })

  } catch (error: any) {
    console.error('Error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
