import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'
import { createClient } from '@supabase/supabase-js'
import * as cheerio from 'cheerio'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// CSS.escape polyfill for Node.js
function cssEscape(value: string): string {
  const string = String(value)
  const length = string.length
  let index = -1
  let codeUnit
  let result = ''
  const firstCodeUnit = string.charCodeAt(0)

  while (++index < length) {
    codeUnit = string.charCodeAt(index)

    // Note: there's no need to special-case astral symbols, surrogate
    // pairs, or lone surrogates.

    // If the character is NULL (U+0000), then the REPLACEMENT CHARACTER
    // (U+FFFD).
    if (codeUnit === 0x0000) {
      result += '\uFFFD'
      continue
    }

    if (
      // If the character is in the range [\1-\1F] (U+0001 to U+001F) or is
      // U+007F, […]
      (codeUnit >= 0x0001 && codeUnit <= 0x001F) || codeUnit === 0x007F ||
      // If the character is the first character and is in the range [0-9]
      // (U+0030 to U+0039), […]
      (index === 0 && codeUnit >= 0x0030 && codeUnit <= 0x0039) ||
      // If the character is the second character and is in the range [0-9]
      // (U+0030 to U+0039) and the first character is a `-` (U+002D), […]
      (
        index === 1 &&
        codeUnit >= 0x0030 && codeUnit <= 0x0039 &&
        firstCodeUnit === 0x002D
      )
    ) {
      // https://drafts.csswg.org/cssom/#escape-a-character-as-code-point
      result += '\\' + codeUnit.toString(16) + ' '
      continue
    }

    if (
      // If the character is the first character and is a `-` (U+002D), and
      // there is no second character, […]
      index === 0 &&
      length === 1 &&
      codeUnit === 0x002D
    ) {
      result += '\\' + string.charAt(index)
      continue
    }

    // If the character is not handled by one of the above rules and is
    // greater than or equal to U+0080, is `-` (U+002D) or `_` (U+005F), or
    // is in one of the ranges [0-9] (U+0030 to U+0039), [A-Z] (U+0041 to
    // U+005A), or [a-z] (U+0061 to U+007A), […]
    if (
      codeUnit >= 0x0080 ||
      codeUnit === 0x002D ||
      codeUnit === 0x005F ||
      codeUnit >= 0x0030 && codeUnit <= 0x0039 ||
      codeUnit >= 0x0041 && codeUnit <= 0x005A ||
      codeUnit >= 0x0061 && codeUnit <= 0x007A
    ) {
      // the character itself
      result += string.charAt(index)
      continue
    }

    // Otherwise, the escaped character.
    // https://drafts.csswg.org/cssom/#escape-a-character
    result += '\\' + string.charAt(index)
  }

  return result
}

interface ExtractedData {
  companyName?: { text: string; elementId: string; value: string }
  location?: { text: string; elementId: string; value: string }
  commodities?: { text: string; sectionId: string; value: string[] }
  npv?: { value: number; text: string; sectionId: string }
  irr?: { value: number; text: string; sectionId: string }
  capex?: { value: number; text: string; sectionId: string }
  opex?: { value: number; text: string; sectionId: string }
  resources?: { text: string; sectionId: string }
  reserves?: { text: string; sectionId: string }
  production?: { text: string; sectionId: string }
}

interface HighlightArea {
  elementId?: string
  sectionId?: string
  text: string
}

// Extract company name from inline XBRL tags
function extractCompanyInfo(html: string, $: cheerio.CheerioAPI): {
  companyName?: { text: string; elementId: string; value: string }
  location?: { text: string; elementId: string; value: string }
} {
  const result: any = {}

  // Extract company name from dei:EntityRegistrantName
  const companyNameEl = $('ix\\:nonnumeric[name="dei:EntityRegistrantName"], [name="dei:EntityRegistrantName"]').first()
  if (companyNameEl.length) {
    const value = companyNameEl.text().trim()
    const elementId = companyNameEl.attr('id') || ''
    result.companyName = {
      text: value,
      elementId,
      value
    }
  }

  // Extract location from address fields
  const cityEl = $('ix\\:nonnumeric[name="dei:EntityAddressCityOrTown"], [name="dei:EntityAddressCityOrTown"]').first()
  const stateEl = $('ix\\:nonnumeric[name="dei:EntityAddressStateOrProvince"], [name="dei:EntityAddressStateOrProvince"]').first()
  const countryEl = $('ix\\:nonnumeric[name="dei:EntityIncorporationStateCountryCode"], [name="dei:EntityIncorporationStateCountryCode"]').first()

  if (cityEl.length || stateEl.length || countryEl.length) {
    const city = cityEl.text().trim()
    const state = stateEl.text().trim()
    const country = countryEl.text().trim()
    const locationParts = [city, state, country].filter(Boolean)
    const locationValue = locationParts.join(', ')

    result.location = {
      text: locationValue,
      elementId: cityEl.attr('id') || stateEl.attr('id') || countryEl.attr('id') || '',
      value: locationValue
    }
  }

  return result
}

// Find relevant sections using regex patterns
function findFinancialSections(html: string): {
  financialStatements: string[]
  managementDiscussion: string[]
  balanceSheet: string[]
  incomeStatement: string[]
  cashFlow: string[]
} {
  const result = {
    financialStatements: [] as string[],
    managementDiscussion: [] as string[],
    balanceSheet: [] as string[],
    incomeStatement: [] as string[],
    cashFlow: [] as string[]
  }

  // Regex patterns to find financial sections
  const patterns = {
    financialStatements: /<div[^>]*id="([^"]*)"[^>]*>[\s\S]*?(?:financial statements?|consolidated statements?|statements? of (?:operations|income|financial position))[\s\S]{0,500}?<\/div>/gi,
    managementDiscussion: /<div[^>]*id="([^"]*)"[^>]*>[\s\S]*?(?:management'?s? discussion|md&a|operating results)[\s\S]{0,500}?<\/div>/gi,
    balanceSheet: /<div[^>]*id="([^"]*)"[^>]*>[\s\S]*?(?:balance sheet|financial position)[\s\S]{0,500}?<\/div>/gi,
    incomeStatement: /<div[^>]*id="([^"]*)"[^>]*>[\s\S]*?(?:income statement|statement of operations|statement of earnings)[\s\S]{0,500}?<\/div>/gi,
    cashFlow: /<div[^>]*id="([^"]*)"[^>]*>[\s\S]*?(?:cash flow|statement of cash flows)[\s\S]{0,500}?<\/div>/gi
  }

  // Extract section IDs
  for (const [key, pattern] of Object.entries(patterns)) {
    let match
    while ((match = pattern.exec(html)) !== null) {
      const sectionId = match[1]
      if (sectionId && !result[key as keyof typeof result].includes(sectionId)) {
        result[key as keyof typeof result].push(sectionId)
      }
    }
  }

  return result
}

// Extract tables from sections and add IDs for navigation
function extractTablesFromSections(html: string, $: cheerio.CheerioAPI, sectionIds: string[]): {
  tablesText: string
  tableMap: Map<string, string> // Maps "Table X" to actual element ID
} {
  let tablesText = ''
  const tableMap = new Map<string, string>()
  let globalTableIndex = 0

  for (const sectionId of sectionIds.slice(0, 10)) { // Limit to first 10 sections
    const section = $(`#${cssEscape(sectionId)}`)
    if (section.length) {
      // Get section title
      const sectionTitle = section.find('span[style*="font-weight:700"]').first().text().trim()

      // Extract all tables in this section
      const tables = section.find('table')
      tables.each((i, table) => {
        const $table = $(table)

        // Add a unique ID to the table if it doesn't have one
        if (!$table.attr('id')) {
          const tableId = `lithos-table-${globalTableIndex}`
          $table.attr('id', tableId)
          tableMap.set(`Table ${globalTableIndex + 1}`, tableId)
        } else {
          tableMap.set(`Table ${globalTableIndex + 1}`, $table.attr('id')!)
        }

        const rows: string[] = []

        $table.find('tr').each((j, tr) => {
          const cells: string[] = []
          $(tr).find('td, th').each((k, cell) => {
            cells.push($(cell).text().trim())
          })
          if (cells.length > 0) {
            rows.push(cells.join(' | '))
          }
        })

        if (rows.length > 0) {
          tablesText += `\n\n[SectionId: ${sectionId}]\n[TableId: Table ${globalTableIndex + 1}]\n${rows.join('\n')}\n`
        }

        globalTableIndex++
      })

      // Also get text content around tables (context)
      const sectionText = section.text().trim().substring(0, 2000)
      if (sectionText) {
        tablesText += `\n[Section Text: ${sectionTitle || sectionId}]\n${sectionText}\n`
      }
    }
  }

  return { tablesText, tableMap }
}

export async function POST(req: NextRequest) {
  try {
    const { htmlUrl, projectId } = await req.json()

    if (!htmlUrl) {
      return NextResponse.json({ error: 'HTML URL is required' }, { status: 400 })
    }

    console.log('📄 Extracting key data from HTML:', htmlUrl)

    // Download HTML content
    const htmlResponse = await fetch(htmlUrl)
    if (!htmlResponse.ok) {
      throw new Error('Failed to fetch HTML')
    }

    const htmlText = await htmlResponse.text()
    console.log(`✅ Downloaded HTML: ${htmlText.length} characters`)

    // Parse HTML with cheerio
    const $ = cheerio.load(htmlText)

    // Step 1: Extract company info from inline XBRL tags
    console.log('🏢 Extracting company information from XBRL tags...')
    const companyInfo = extractCompanyInfo(htmlText, $)
    console.log('✅ Company info extracted:', companyInfo)

    // Step 2: Find financial sections using regex
    console.log('🔍 Finding financial sections...')
    const sections = findFinancialSections(htmlText)
    console.log('📊 Found sections:', {
      financialStatements: sections.financialStatements.length,
      managementDiscussion: sections.managementDiscussion.length,
      balanceSheet: sections.balanceSheet.length,
      incomeStatement: sections.incomeStatement.length,
      cashFlow: sections.cashFlow.length
    })

    // Step 3: Extract tables from relevant sections
    console.log('📊 Extracting tables from sections...')
    const allSectionIds = [
      ...sections.financialStatements,
      ...sections.managementDiscussion,
      ...sections.balanceSheet,
      ...sections.incomeStatement,
      ...sections.cashFlow
    ].filter((v, i, a) => a.indexOf(v) === i) // unique

    const { tablesText, tableMap } = extractTablesFromSections(htmlText, $, allSectionIds)
    console.log(`✅ Extracted tables: ${tablesText.length} characters`)
    console.log(`📋 Table map created: ${tableMap.size} tables`)

    // Step 4: Use AI for precise extraction
    console.log('🤖 Sending to AI for financial data extraction...')
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: `You are an expert at extracting key financial and operational data from SEC/SEDAR HTML filings.

Extract the following data points and return in JSON format:
- companyName: Company name (if not already provided)
- location: Company headquarters location (if not already provided)
- commodities: {text: exact quote from document, tableId: "Table X" where X is the table number, value: array of metals/minerals}
- npv: {value: number in millions, text: exact quote, tableId: "Table X"}
- irr: {value: number as percentage, text: exact quote, tableId: "Table X"}
- capex: {value: number in millions, text: exact quote, tableId: "Table X"}
- opex: {value: number, text: exact quote, tableId: "Table X"}
- resources: {text: summary quote, tableId: "Table X"}
- reserves: {text: summary quote, tableId: "Table X"}
- production: {text: summary quote, tableId: "Table X"}

IMPORTANT: When you see [TableId: Table X], use exactly "Table X" as the tableId in your response.

For mining companies, look for:
- Production volumes (tonnes, ounces, pounds)
- Operating metrics (cash costs, AISC)
- Capital expenditures
- Resources and reserves
- Commodity prices

Return null for values not found. Be precise with table identifiers.`,
        },
        {
          role: 'user',
          content: `Company: ${companyInfo.companyName?.value || 'Unknown'}
Location: ${companyInfo.location?.value || 'Unknown'}

Extract mining/operational data from these tables and sections:

${tablesText.substring(0, 80000)}`,
        },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.1,
    })

    const extractedDataStr = completion.choices[0]?.message?.content
    if (!extractedDataStr) {
      throw new Error('No response from OpenAI')
    }

    let extractedData: ExtractedData = JSON.parse(extractedDataStr)
    console.log('✅ Extracted data:', Object.keys(extractedData))

    // Merge company info from XBRL tags
    if (companyInfo.companyName) {
      extractedData.companyName = companyInfo.companyName
    }
    if (companyInfo.location) {
      extractedData.location = companyInfo.location
    }

    // Step 5: Convert extracted data to highlights format
    const highlights: any[] = []

    const createHighlight = (
      dataType: string,
      text: string,
      location: { elementId?: string; sectionId?: string },
      value?: any
    ) => {
      return {
        id: `auto-${dataType}-${Date.now()}-${Math.random()}`,
        content: text,
        quote: text,
        highlightArea: location,
        dataType,
        value,
      }
    }

    if (extractedData.companyName) {
      highlights.push(
        createHighlight(
          'companyName',
          extractedData.companyName.text,
          { elementId: extractedData.companyName.elementId },
          extractedData.companyName.value
        )
      )
    }

    if (extractedData.location) {
      highlights.push(
        createHighlight(
          'location',
          extractedData.location.text,
          { elementId: extractedData.location.elementId },
          extractedData.location.value
        )
      )
    }

    if (extractedData.npv) {
      highlights.push(
        createHighlight(
          'npv',
          extractedData.npv.text,
          { sectionId: extractedData.npv.sectionId },
          extractedData.npv.value
        )
      )
    }

    if (extractedData.irr) {
      highlights.push(
        createHighlight(
          'irr',
          extractedData.irr.text,
          { sectionId: extractedData.irr.sectionId },
          extractedData.irr.value
        )
      )
    }

    if (extractedData.capex) {
      highlights.push(
        createHighlight(
          'capex',
          extractedData.capex.text,
          { sectionId: extractedData.capex.sectionId },
          extractedData.capex.value
        )
      )
    }

    if (extractedData.commodities) {
      highlights.push(
        createHighlight(
          'commodities',
          extractedData.commodities.text,
          { sectionId: extractedData.commodities.sectionId },
          extractedData.commodities.value
        )
      )
    }

    if (extractedData.resources) {
      highlights.push(
        createHighlight(
          'resources',
          extractedData.resources.text,
          { sectionId: extractedData.resources.sectionId }
        )
      )
    }

    if (extractedData.reserves) {
      highlights.push(
        createHighlight(
          'reserves',
          extractedData.reserves.text,
          { sectionId: extractedData.reserves.sectionId }
        )
      )
    }

    console.log(`💾 Saving ${highlights.length} highlights to database...`)

    // Save to database
    const { data: existing } = await supabase
      .from('html_highlights')
      .select('id')
      .eq('document_url', htmlUrl)
      .single()

    let savedHighlights
    let saveError

    const highlightData = {
      document_url: htmlUrl,
      project_id: projectId,
      highlight_data: {
        highlights,
        extractedData,
        extractedAt: new Date().toISOString(),
        sections: allSectionIds,
      },
      updated_at: new Date().toISOString(),
    }

    if (existing) {
      const result = await supabase
        .from('html_highlights')
        .update(highlightData)
        .eq('document_url', htmlUrl)
        .select()
      savedHighlights = result.data
      saveError = result.error
    } else {
      const result = await supabase
        .from('html_highlights')
        .insert(highlightData)
        .select()
      savedHighlights = result.data
      saveError = result.error
    }

    if (saveError) {
      console.error('Error saving highlights:', saveError)
    }

    console.log('✅ Extraction complete!')

    // Step 6: Update projects table with extracted data
    let projectUpdated = false
    if (projectId && extractedData) {
      console.log('💾 Updating project with extracted data...')

      const projectUpdateData: any = {
        updated_at: new Date().toISOString(),
        financial_metrics_updated_at: new Date().toISOString(),
      }

      if (extractedData.npv?.value) {
        projectUpdateData.npv = extractedData.npv.value
      }
      if (extractedData.irr?.value) {
        projectUpdateData.irr = extractedData.irr.value
      }
      if (extractedData.capex?.value) {
        projectUpdateData.capex = extractedData.capex.value
      }
      if (extractedData.location?.value) {
        projectUpdateData.location = extractedData.location.value
      }
      if (extractedData.commodities?.value) {
        projectUpdateData.commodities = extractedData.commodities.value
      }
      if (extractedData.resources?.text) {
        projectUpdateData.resource = extractedData.resources.text
      }
      if (extractedData.reserves?.text) {
        projectUpdateData.reserve = extractedData.reserves.text
      }

      // Update company name if different from "Unknown"
      if (extractedData.companyName?.value) {
        // Get company_id, update or create company
        const { data: company } = await supabase
          .from('companies')
          .select('id')
          .eq('name', extractedData.companyName.value)
          .single()

        if (company) {
          projectUpdateData.company_id = company.id
        }
      }

      const { data: updatedProject, error: projectError } = await supabase
        .from('projects')
        .update(projectUpdateData)
        .eq('id', projectId)
        .select()

      if (projectError) {
        console.error('❌ Error updating project:', projectError)
      } else {
        console.log('✅ Project updated successfully!')
        projectUpdated = true
      }
    }

    return NextResponse.json({
      success: true,
      highlights,
      extractedData,
      sections: allSectionIds,
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
    const htmlUrl = searchParams.get('htmlUrl')

    if (!htmlUrl) {
      return NextResponse.json({ error: 'HTML URL is required' }, { status: 400 })
    }

    // Fetch existing highlights
    const { data: highlights, error } = await supabase
      .from('html_highlights')
      .select('*')
      .eq('document_url', htmlUrl)
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
