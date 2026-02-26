/**
 * Utility functions for extracting data from large PDF mining reports
 * Uses hybrid regex + AI approach for efficiency
 */

export interface PageMatch {
  pageNumber: number
  text: string
  matches: string[]
}

export interface DataPattern {
  name: string
  patterns: RegExp[]
  keywords: string[]
}

// Common patterns for mining report data
export const MINING_DATA_PATTERNS: Record<string, DataPattern> = {
  npv: {
    name: 'NPV',
    patterns: [
      /NPV[\s@]*(?:of\s+)?(?:\$|USD|CAD)?\s*[\d,]+(?:\.\d+)?\s*(?:million|M|B)/gi,
      /Net\s+Present\s+Value[\s:]*(?:\$|USD|CAD)?\s*[\d,]+(?:\.\d+)?\s*(?:million|M)/gi,
      /(?:\$|USD|CAD)\s*[\d,]+(?:\.\d+)?\s*(?:million|M).*NPV/gi,
    ],
    keywords: ['NPV', 'Net Present Value', 'after-tax NPV', 'pre-tax NPV'],
  },
  irr: {
    name: 'IRR',
    patterns: [
      /IRR[\s:]*[\d.]+\s*%/gi,
      /Internal\s+Rate\s+of\s+Return[\s:]*[\d.]+\s*%/gi,
      /[\d.]+\s*%.*IRR/gi,
    ],
    keywords: ['IRR', 'Internal Rate of Return', 'after-tax IRR', 'pre-tax IRR'],
  },
  capex: {
    name: 'CAPEX',
    patterns: [
      /CAPEX[\s:]*(?:\$|USD|CAD)?\s*[\d,]+(?:\.\d+)?\s*(?:million|M|B)/gi,
      /Capital\s+(?:Cost|Expenditure)[\s:]*(?:\$|USD|CAD)?\s*[\d,]+(?:\.\d+)?\s*(?:million|M)/gi,
      /Initial\s+Capital[\s:]*(?:\$|USD|CAD)?\s*[\d,]+(?:\.\d+)?\s*(?:million|M)/gi,
    ],
    keywords: ['CAPEX', 'Capital Cost', 'Capital Expenditure', 'Initial Capital'],
  },
  opex: {
    name: 'OPEX',
    patterns: [
      /OPEX[\s:]*(?:\$|USD|CAD)?\s*[\d,]+(?:\.\d+)?\s*(?:\/t|per\s+tonne|per\s+ton)/gi,
      /Operating\s+Cost[\s:]*(?:\$|USD|CAD)?\s*[\d,]+(?:\.\d+)?\s*(?:\/t|per\s+tonne)/gi,
    ],
    keywords: ['OPEX', 'Operating Cost', 'Operating Expenditure', 'C1 Cash Cost'],
  },
  resources: {
    name: 'Mineral Resources',
    patterns: [
      /(?:Measured|Indicated|Inferred)\s+(?:Resource|Resources)[\s:]*[\d,]+(?:\.\d+)?\s*(?:Mt|million tonnes|Kt)/gi,
      /Total\s+(?:Mineral\s+)?Resources[\s:]*[\d,]+(?:\.\d+)?\s*(?:Mt|million tonnes)/gi,
    ],
    keywords: ['Mineral Resources', 'Measured Resources', 'Indicated Resources', 'Inferred Resources', 'Resource Estimate'],
  },
  reserves: {
    name: 'Mineral Reserves',
    patterns: [
      /(?:Proven|Probable)\s+Reserves?[\s:]*[\d,]+(?:\.\d+)?\s*(?:Mt|million tonnes|Kt)/gi,
      /Total\s+Reserves?[\s:]*[\d,]+(?:\.\d+)?\s*(?:Mt|million tonnes)/gi,
    ],
    keywords: ['Mineral Reserves', 'Proven Reserves', 'Probable Reserves', 'Reserve Estimate'],
  },
  production: {
    name: 'Production',
    patterns: [
      /(?:Annual|Average)\s+Production[\s:]*[\d,]+(?:\.\d+)?\s*(?:kt|thousand tonnes|Mlb|million pounds)/gi,
      /Production\s+Rate[\s:]*[\d,]+(?:\.\d+)?\s*(?:tpd|tonnes per day|ktpa)/gi,
    ],
    keywords: ['Annual Production', 'Production Rate', 'Life of Mine', 'LOM Production'],
  },
  commodities: {
    name: 'Commodities',
    patterns: [
      /\b(?:Copper|Gold|Silver|Zinc|Lead|Nickel|Cobalt|Lithium|Molybdenum)\b/gi,
      /\b(?:Cu|Au|Ag|Zn|Pb|Ni|Co|Li|Mo)\b(?=\s*(?:grade|content|\%))/gi,
    ],
    keywords: ['copper', 'gold', 'silver', 'zinc', 'lead', 'nickel', 'cobalt', 'lithium', 'molybdenum'],
  },
}

/**
 * Search for specific patterns in page text
 */
export function findPagesWithPattern(
  pages: { pageNumber: number; text: string }[],
  pattern: DataPattern
): PageMatch[] {
  const matches: PageMatch[] = []

  for (const page of pages) {
    const pageMatches: string[] = []

    // Check if page contains any keywords
    const hasKeyword = pattern.keywords.some(keyword =>
      page.text.toLowerCase().includes(keyword.toLowerCase())
    )

    if (hasKeyword) {
      // Try to extract with regex patterns
      for (const regex of pattern.patterns) {
        const found = page.text.match(regex)
        if (found) {
          pageMatches.push(...found)
        }
      }

      // If we found matches or keywords, include this page
      if (pageMatches.length > 0 || hasKeyword) {
        matches.push({
          pageNumber: page.pageNumber,
          text: page.text,
          matches: pageMatches,
        })
      }
    }
  }

  return matches
}

/**
 * Find pages likely to contain executive summary or key metrics
 */
export function findKeyPages(pages: { pageNumber: number; text: string }[]): {
  executiveSummary: PageMatch[]
  financialMetrics: PageMatch[]
  mineralResources: PageMatch[]
  mineralReserves: PageMatch[]
} {
  const executiveSummary: PageMatch[] = []
  const financialMetrics: PageMatch[] = []
  const mineralResources: PageMatch[] = []
  const mineralReserves: PageMatch[] = []

  for (const page of pages) {
    const lowerText = page.text.toLowerCase()

    // Executive Summary (usually in first 20 pages)
    if (
      page.pageNumber <= 20 &&
      (lowerText.includes('executive summary') ||
        lowerText.includes('summary of results') ||
        lowerText.includes('key highlights'))
    ) {
      executiveSummary.push({
        pageNumber: page.pageNumber,
        text: page.text,
        matches: ['Executive Summary Section'],
      })
    }

    // Financial Metrics (NPV, IRR, CAPEX typically together)
    if (
      (lowerText.includes('npv') || lowerText.includes('net present value')) &&
      (lowerText.includes('irr') || lowerText.includes('internal rate'))
    ) {
      financialMetrics.push({
        pageNumber: page.pageNumber,
        text: page.text,
        matches: ['Financial Metrics Section'],
      })
    }

    // Mineral Resources
    if (
      lowerText.includes('mineral resource') ||
      (lowerText.includes('measured') && lowerText.includes('indicated'))
    ) {
      mineralResources.push({
        pageNumber: page.pageNumber,
        text: page.text,
        matches: ['Mineral Resources Section'],
      })
    }

    // Mineral Reserves
    if (
      lowerText.includes('mineral reserve') ||
      (lowerText.includes('proven') && lowerText.includes('probable'))
    ) {
      mineralReserves.push({
        pageNumber: page.pageNumber,
        text: page.text,
        matches: ['Mineral Reserves Section'],
      })
    }
  }

  return {
    executiveSummary,
    financialMetrics,
    mineralResources,
    mineralReserves,
  }
}

/**
 * Extract surrounding context from a page given a match position
 */
export function extractContext(text: string, matchText: string, contextChars: number = 200): string {
  const index = text.indexOf(matchText)
  if (index === -1) return matchText

  const start = Math.max(0, index - contextChars)
  const end = Math.min(text.length, index + matchText.length + contextChars)

  return text.substring(start, end)
}
