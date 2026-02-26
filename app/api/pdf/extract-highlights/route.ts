import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'
import { createClient } from '@supabase/supabase-js'
import {
  MINING_DATA_PATTERNS,
  findPagesWithPattern,
  findKeyPages,
  extractContext,
} from '@/lib/pdf-extraction-utils'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

interface ExtractedData {
  commodities?: { text: string; page: number; value?: string[] }
  location?: { text: string; page: number; value?: string }
  npv?: { value: number; text: string; page: number }
  irr?: { value: number; text: string; page: number }
  capex?: { value: number; text: string; page: number }
  opex?: { value: number; text: string; page: number }
  resources?: { text: string; page: number }
  reserves?: { text: string; page: number }
  production?: { text: string; page: number }
}

// Simple Cmd+F style text search to find coordinates
// Just searches for the text in the page and returns coordinates
function findTextCoordinatesInPage(
  searchText: string,
  page: any,
  pageHeight: number,
  pageWidth: number
): { left: number; top: number; width: number; height: number } | null {
  if (!page || !page.textItems || !searchText) {
    return null
  }

  const textItems = page.textItems

  // Normalize search text - remove extra spaces and lowercase
  const searchNormalized = searchText
    .toLowerCase()
    .replace(/\s+/g, ' ')  // Replace multiple spaces with single space
    .trim()

  // Build text with original spacing preserved
  const fullText = textItems.map((item: any) => item.str || '').join(' ')
  const fullTextNormalized = fullText
    .toLowerCase()
    .replace(/\s+/g, ' ')  // Normalize spaces in page text too

  // Try exact match first
  let searchIndex = fullTextNormalized.indexOf(searchNormalized)

  // If exact match fails, try finding a substring (first 30 chars)
  if (searchIndex === -1 && searchNormalized.length > 30) {
    const shortSearch = searchNormalized.substring(0, 30)
    searchIndex = fullTextNormalized.indexOf(shortSearch)

    if (searchIndex !== -1) {
      console.log(`📝 Partial match found for: "${shortSearch}..."`)
      // Use the partial match
      return findCoordinatesForTextRange(
        textItems,
        pageHeight,
        pageWidth,
        searchIndex,
        searchIndex + shortSearch.length,
        fullText
      )
    }
  }

  // If still not found, try key phrases (first significant part)
  if (searchIndex === -1) {
    // Extract key numbers or phrases from the search text
    const keyPhrases = searchNormalized.match(/\d+[\d,.]* ?mt|[\d.]+%|[\d,]+\.[\d]+|\b\w+\s+\w+\s+\w+/g)

    if (keyPhrases && keyPhrases.length > 0) {
      for (const phrase of keyPhrases) {
        searchIndex = fullTextNormalized.indexOf(phrase)
        if (searchIndex !== -1) {
          console.log(`🔍 Found key phrase: "${phrase}"`)
          return findCoordinatesForTextRange(
            textItems,
            pageHeight,
            pageWidth,
            searchIndex,
            searchIndex + phrase.length,
            fullText
          )
        }
      }
    }
  }

  if (searchIndex === -1) {
    console.log(`⚠️ Text not found: "${searchText.substring(0, 30)}..."`)
    return null
  }

  console.log(`✅ Found text: "${searchText.substring(0, 30)}..."`)

  // Find the text items that contain this text range
  return findCoordinatesForTextRange(
    textItems,
    pageHeight,
    pageWidth,
    searchIndex,
    searchIndex + searchNormalized.length,
    fullText
  )
}

function findCoordinatesForTextRange(
  textItems: any[],
  pageHeight: number,
  pageWidth: number,
  startIndex: number,
  endIndex: number,
  fullText: string
): { left: number; top: number; width: number; height: number } | null {
  // Simple approach: find which text items contain our search range
  let currentPos = 0
  const matchedItems: any[] = []

  for (const item of textItems) {
    const itemText = item.str || ''
    const itemLength = itemText.length
    const itemEndPos = currentPos + itemLength + 1 // +1 for the space we add between items

    // Check if this item is within our search range
    if (currentPos < endIndex && itemEndPos > startIndex) {
      matchedItems.push(item)
    }

    currentPos = itemEndPos
  }

  if (matchedItems.length === 0) {
    return null
  }

  // Get bounding box from matched items
  const bounds = matchedItems.map((item: any) => {
    const transform = item.transform || [1, 0, 0, 1, 0, 0]
    const x = transform[4]  // x position
    const y = transform[5]  // y position (from bottom in PDF coordinates)
    const fontSize = Math.abs(transform[3])

    // Simple width estimation if not provided
    const width = item.width || (item.str?.length || 0) * fontSize * 0.6

    return {
      left: x,
      bottom: y,
      right: x + width,
      top: y + fontSize
    }
  })

  // Calculate overall bounding box
  const left = Math.min(...bounds.map(b => b.left))
  const right = Math.max(...bounds.map(b => b.right))
  const bottom = Math.min(...bounds.map(b => b.bottom))
  const top = Math.max(...bounds.map(b => b.top))

  // Convert to viewer percentages (top-left origin)
  // Add small padding for visibility
  const padding = 2 // pixels
  const viewerLeft = Math.max(0, ((left - padding) / pageWidth) * 100)
  const viewerTop = Math.max(0, ((pageHeight - top - padding) / pageHeight) * 100)
  const viewerWidth = Math.min(100 - viewerLeft, ((right - left + 2 * padding) / pageWidth) * 100)
  const viewerHeight = Math.min(100 - viewerTop, ((top - bottom + 2 * padding) / pageHeight) * 100)

  return {
    left: viewerLeft,
    top: viewerTop,
    width: viewerWidth,
    height: viewerHeight
  }
}

export async function POST(req: NextRequest) {
  try {
    const { pdfUrl, projectId } = await req.json()

    if (!pdfUrl) {
      return NextResponse.json({ error: 'PDF URL is required' }, { status: 400 })
    }

    console.log('📄 Extracting key data from PDF:', pdfUrl)

    // Download PDF content
    const pdfResponse = await fetch(pdfUrl)
    if (!pdfResponse.ok) {
      throw new Error('Failed to fetch PDF')
    }

    const pdfBuffer = await pdfResponse.arrayBuffer()

    // Extract text page-by-page from PDF
    console.log('📖 Extracting text from PDF...')
    // Custom page rendering to preserve page numbers and coordinates
    let currentPage = 0
    const pages: {
      pageNumber: number
      text: string
      textItems?: any[]
      viewport?: { height: number; width: number }
    }[] = []

    // Use unpdf for ESM-compatible PDF parsing
    const { extractText, getDocumentProxy } = await import('unpdf')

    // Convert Buffer to Uint8Array for unpdf
    console.log('🔧 Converting Buffer to Uint8Array...')
    const uint8Array = new Uint8Array(pdfBuffer)
    console.log('✅ Uint8Array created, length:', uint8Array.length)

    // Extract text with page information
    console.log('📖 Calling extractText...')
    const { text: fullText, totalPages } = await extractText(uint8Array, {
      mergePages: false,
    })

    console.log(`✅ Extracted ${totalPages} pages`)

    // Get document proxy for detailed page access
    console.log('🔧 Getting document proxy...')
    const pdfDoc = await getDocumentProxy(uint8Array)

    // Extract detailed page information
    for (let i = 1; i <= totalPages; i++) {
      const page = await pdfDoc.getPage(i)
      const textContent = await page.getTextContent()
      const viewport = page.getViewport({ scale: 1.0 })
      const pageText = textContent.items.map((item: any) => item.str).join(' ')

      pages.push({
        pageNumber: i,
        text: pageText,
        textItems: textContent.items,
        viewport: { height: viewport.height, width: viewport.width },
      })
    }

    const numPages = totalPages
    console.log(`✅ Processed ${pages.length} pages with detailed text items`)

    // Step 1: Find pages with specific patterns using regex
    console.log('🔍 Searching for key data patterns...')
    const npvPages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.npv)
    const irrPages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.irr)
    const capexPages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.capex)
    const opexPages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.opex)
    const resourcePages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.resources)
    const reservePages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.reserves)
    const productionPages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.production)
    const commodityPages = findPagesWithPattern(pages, MINING_DATA_PATTERNS.commodities)

    // Step 2: Find key sections
    const keyPages = findKeyPages(pages)

    console.log(`📊 Found patterns:`)
    console.log(`  - NPV: ${npvPages.length} pages`)
    console.log(`  - IRR: ${irrPages.length} pages`)
    console.log(`  - CAPEX: ${capexPages.length} pages`)
    console.log(`  - Resources: ${resourcePages.length} pages`)
    console.log(`  - Reserves: ${reservePages.length} pages`)
    console.log(`  - Executive Summary: ${keyPages.executiveSummary.length} pages`)

    // Step 3: Prepare context for AI extraction (only relevant pages)
    const relevantPages = new Set<number>()

    ;[
      ...npvPages,
      ...irrPages,
      ...capexPages,
      ...opexPages,
      ...resourcePages,
      ...reservePages,
      ...productionPages,
      ...keyPages.executiveSummary,
      ...keyPages.financialMetrics,
    ].forEach(match => relevantPages.add(match.pageNumber))

    // Limit to top 20 most relevant pages
    const topPages = Array.from(relevantPages)
      .sort((a, b) => a - b)
      .slice(0, 20)
      .map(pageNum => pages.find(p => p.pageNumber === pageNum)!)
      .filter(Boolean)

    const contextText = topPages
      .map(p => `[Page ${p.pageNumber}]\n${p.text.substring(0, 2000)}`)
      .join('\n\n---\n\n')

    console.log(`🤖 Sending ${topPages.length} relevant pages to AI for precise extraction...`)

    // Step 4: Use AI for precise extraction from relevant pages only
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: `You are an expert at extracting key financial and technical data from mining technical reports (NI 43-101, feasibility studies).

Extract the following data points from the provided pages and return in JSON format:
- npv: {value: number in millions, text: exact quote, page: page number}
- irr: {value: number as percentage, text: exact quote, page: page number}
- capex: {value: number in millions, text: exact quote, page: page number}
- opex: {value: number, text: exact quote, page: page number}
- resources: {text: summary quote, page: page number}
- reserves: {text: summary quote, page: page number}
- production: {text: summary quote, page: page number}
- commodities: {text: quote, page: page number, value: array of metals}
- location: {text: quote, page: page number, value: location string}

For each value:
1. Extract the EXACT text snippet containing it (10-50 words)
2. Note the specific [Page X] number where it was found
3. Extract the numeric value where applicable

Return null for values not found. Be precise with page numbers.`,
        },
        {
          role: 'user',
          content: `Extract key mining project data from these pages:\n\n${contextText.substring(0, 100000)}`,
        },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.1,
    })

    const extractedDataStr = completion.choices[0]?.message?.content
    if (!extractedDataStr) {
      throw new Error('No response from OpenAI')
    }

    const extractedData: ExtractedData = JSON.parse(extractedDataStr)
    console.log('✅ Extracted data:', Object.keys(extractedData))

    // Step 5: Convert extracted data to highlights format with exact pages
    const highlights: any[] = []

    // Helper to create highlight with proper coordinate conversion
    const createHighlight = (
      dataType: string,
      text: string,
      pageNum: number,
      value?: any
    ) => {
      const page = pages.find(p => p.pageNumber === pageNum)
      let coords = null

      if (page && page.viewport && page.textItems) {
        // Simple Cmd+F style search for the text on the page
        coords = findTextCoordinatesInPage(
          text,
          page,
          page.viewport.height,
          page.viewport.width
        )
        if (!coords) {
          console.log(`⚠️ Text not found on page ${pageNum} for ${dataType}`)
        }
      }

      return {
        id: `auto-${dataType}-${Date.now()}-${Math.random()}`,
        content: text,
        quote: text,
        highlightAreas: coords ? [{ pageIndex: pageNum - 1, ...coords }] : [],
        dataType,
        value,
        page: pageNum,
      }
    }

    if (extractedData.npv && extractedData.npv.text) {
      highlights.push(
        createHighlight('npv', extractedData.npv.text, extractedData.npv.page, extractedData.npv.value)
      )
    }

    if (extractedData.irr && extractedData.irr.text) {
      highlights.push(
        createHighlight('irr', extractedData.irr.text, extractedData.irr.page, extractedData.irr.value)
      )
    }

    if (extractedData.capex && extractedData.capex.text) {
      highlights.push(
        createHighlight('capex', extractedData.capex.text, extractedData.capex.page, extractedData.capex.value)
      )
    }

    if (extractedData.resources && extractedData.resources.text) {
      highlights.push(
        createHighlight('resources', extractedData.resources.text, extractedData.resources.page)
      )
    }

    if (extractedData.reserves && extractedData.reserves.text) {
      highlights.push(
        createHighlight('reserves', extractedData.reserves.text, extractedData.reserves.page)
      )
    }

    console.log(`💾 Saving ${highlights.length} highlights to database...`)

    // Save to database - check if exists first, then update or insert
    const { data: existing } = await supabase
      .from('pdf_highlights')
      .select('id')
      .eq('document_url', pdfUrl)
      .single()

    let savedHighlights
    let saveError

    const highlightData = {
      document_url: pdfUrl,
      project_id: projectId,
      highlight_data: {
        highlights,
        extractedData,
        extractedAt: new Date().toISOString(),
        numPages,
        relevantPages: topPages.map(p => p.pageNumber),
      },
      updated_at: new Date().toISOString(),
    }

    if (existing) {
      // Update existing record
      const result = await supabase
        .from('pdf_highlights')
        .update(highlightData)
        .eq('document_url', pdfUrl)
        .select()
      savedHighlights = result.data
      saveError = result.error
    } else {
      // Insert new record
      const result = await supabase
        .from('pdf_highlights')
        .insert(highlightData)
        .select()
      savedHighlights = result.data
      saveError = result.error
    }

    if (saveError) {
      console.error('Error saving highlights:', saveError)
    }

    console.log('✅ Extraction complete!')

    // Step 6: Update projects table with extracted financial data if projectId is provided
    let projectUpdated = false
    if (projectId && extractedData) {
      console.log('💾 Updating project with extracted financial data...')
      console.log('📊 Project ID:', projectId)
      console.log('📊 Extracted data summary:', {
        hasNPV: !!extractedData.npv?.value,
        npvValue: extractedData.npv?.value,
        hasIRR: !!extractedData.irr?.value,
        irrValue: extractedData.irr?.value,
        hasCAPEX: !!extractedData.capex?.value,
        capexValue: extractedData.capex?.value,
        hasLocation: !!extractedData.location?.value,
        hasCommodities: !!extractedData.commodities?.value,
      })

      const projectUpdateData: any = {
        updated_at: new Date().toISOString(),
        financial_metrics_updated_at: new Date().toISOString(),
      }

      // Add NPV if extracted (use simple column name that frontend expects)
      if (extractedData.npv?.value) {
        projectUpdateData.npv = extractedData.npv.value
        console.log('  ✓ Adding NPV:', extractedData.npv.value)
      }

      // Add IRR if extracted
      if (extractedData.irr?.value) {
        projectUpdateData.irr = extractedData.irr.value
        console.log('  ✓ Adding IRR:', extractedData.irr.value)
      }

      // Add CAPEX if extracted
      if (extractedData.capex?.value) {
        projectUpdateData.capex = extractedData.capex.value
        console.log('  ✓ Adding CAPEX:', extractedData.capex.value)
      }

      // Add location if extracted
      if (extractedData.location?.value) {
        projectUpdateData.location = extractedData.location.value
        console.log('  ✓ Adding location:', extractedData.location.value)
      }

      // Add commodities if extracted
      if (extractedData.commodities?.value && Array.isArray(extractedData.commodities.value)) {
        projectUpdateData.commodities = extractedData.commodities.value
        console.log('  ✓ Adding commodities:', extractedData.commodities.value)
      }

      console.log('📝 Updating with data:', projectUpdateData)

      // Update project in database
      const { data: updatedProject, error: projectError } = await supabase
        .from('projects')
        .update(projectUpdateData)
        .eq('id', projectId)
        .select()

      if (projectError) {
        console.error('❌ Error updating project:', projectError)
      } else {
        console.log('✅ Project updated successfully!')
        console.log('✅ Updated project data:', updatedProject)
        projectUpdated = true
      }
    } else {
      console.log('⚠️ Skipping project update - projectId:', projectId, 'extractedData:', !!extractedData)
    }

    // Manually serialize highlights to avoid DataCloneError from unpdf objects
    // Only include serializable fields
    const serializedHighlights = highlights.map(h => ({
      id: h.id,
      content: h.content,
      quote: h.quote,
      highlightAreas: h.highlightAreas ? h.highlightAreas.map((area: any) => ({
        pageIndex: area.pageIndex,
        left: area.left,
        top: area.top,
        width: area.width,
        height: area.height,
      })) : [],
      dataType: h.dataType,
      value: h.value,
      page: h.page,
    }))

    // Serialize extractedData to remove any unpdf objects
    const serializedExtractedData = extractedData ? {
      npv: extractedData.npv ? { text: extractedData.npv.text, value: extractedData.npv.value, page: extractedData.npv.page } : null,
      irr: extractedData.irr ? { text: extractedData.irr.text, value: extractedData.irr.value, page: extractedData.irr.page } : null,
      capex: extractedData.capex ? { text: extractedData.capex.text, value: extractedData.capex.value, page: extractedData.capex.page } : null,
      resources: extractedData.resources ? { text: extractedData.resources.text, page: extractedData.resources.page } : null,
      reserves: extractedData.reserves ? { text: extractedData.reserves.text, page: extractedData.reserves.page } : null,
      location: extractedData.location ? { text: extractedData.location.text, value: extractedData.location.value, page: extractedData.location.page } : null,
      commodities: extractedData.commodities ? { text: extractedData.commodities.text, value: extractedData.commodities.value, page: extractedData.commodities.page } : null,
    } : null

    return NextResponse.json({
      success: true,
      highlights: serializedHighlights,
      extractedData: serializedExtractedData,
      numPages,
      relevantPages: topPages.map((p: any) => p.pageNumber),
      saved: !!savedHighlights,
      projectUpdated,
    })
  } catch (error: any) {
    console.error('❌ Error extracting highlights:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to extract highlights' },
      { status: 500 }
    )
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const pdfUrl = searchParams.get('pdfUrl')

    if (!pdfUrl) {
      return NextResponse.json({ error: 'PDF URL is required' }, { status: 400 })
    }

    // Fetch existing highlights
    const { data: highlights, error } = await supabase
      .from('pdf_highlights')
      .select('*')
      .eq('document_url', pdfUrl)
      .single()

    if (error && error.code !== 'PGRST116') {
      throw error
    }

    return NextResponse.json({
      highlights: highlights?.highlight_data || null,
    })
  } catch (error: any) {
    console.error('Error fetching highlights:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch highlights' },
      { status: 500 }
    )
  }
}
