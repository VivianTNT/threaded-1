/**
 * Mining Company Extraction Script
 *
 * This script uses Firecrawl API to extract mining companies and their projects
 * from major mining exchanges and data sources.
 *
 * Target: 3-4 figure mining companies (100-9999 companies)
 *
 * Run with: npx tsx scripts/extract-mining-companies.ts
 */

import FirecrawlApp from '@mendable/firecrawl-js';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

interface MiningCompany {
  name: string;
  ticker?: string;
  exchange?: string;
  projects?: string[];
  headquarters?: string;
  commodities?: string[];
  url?: string;
}

// Initialize Firecrawl
const firecrawlApiKey = process.env.FIRECRAWL_API_KEY;
if (!firecrawlApiKey) {
  console.error('❌ FIRECRAWL_API_KEY not found in environment variables');
  process.exit(1);
}

const firecrawl = new FirecrawlApp({ apiKey: firecrawlApiKey });

// Major mining data sources
const DATA_SOURCES = [
  {
    name: 'ASX Mining Companies',
    url: 'https://www.asx.com.au/markets/trade-our-cash-market/directory/asx-listed-companies',
    sector: 'Materials'
  },
  {
    name: 'TSX/TSXV Mining',
    url: 'https://www.tsx.com/listings/listing-with-us/sector-and-product-profiles/mining',
    sector: 'Mining'
  },
  {
    name: 'LSE Mining',
    url: 'https://www.londonstockexchange.com/stock-screener?filterBy=sector',
    sector: 'Mining'
  },
  {
    name: 'Mining.com Top Companies',
    url: 'https://www.mining.com/markets/',
    sector: 'Mining'
  }
];

/**
 * Extract mining companies from a given URL using Firecrawl
 */
async function extractCompaniesFromUrl(url: string, sourceName: string): Promise<MiningCompany[]> {
  console.log(`\n🔍 Scraping: ${sourceName}`);
  console.log(`   URL: ${url}`);

  try {
    const result = await firecrawl.scrapeUrl(url, {
      formats: ['markdown', 'html'],
      timeout: 45000,
      waitFor: 2000,
      includeTags: ['table', 'tbody', 'tr', 'td', 'li', 'a', 'div', 'span'],
      excludeTags: ['nav', 'header', 'footer', 'aside', 'script', 'style', 'advertisement']
    });

    if (!result.success || !result.markdown) {
      console.log(`   ⚠️  Failed to scrape ${sourceName}`);
      return [];
    }

    console.log(`   ✓ Scraped ${result.markdown.length} characters`);

    // Parse the markdown to extract company information
    const companies = parseCompaniesFromMarkdown(result.markdown, sourceName);
    console.log(`   ✓ Extracted ${companies.length} companies`);

    return companies;

  } catch (error) {
    console.error(`   ❌ Error scraping ${sourceName}:`, error);
    return [];
  }
}

/**
 * Parse markdown content to extract company information
 */
function parseCompaniesFromMarkdown(markdown: string, source: string): MiningCompany[] {
  const companies: MiningCompany[] = [];
  const lines = markdown.split('\n');

  // Mining-related keywords to filter companies
  const miningKeywords = [
    'mining', 'mineral', 'metals', 'gold', 'silver', 'copper', 'lithium',
    'nickel', 'cobalt', 'rare earth', 'exploration', 'resources', 'ore',
    'zinc', 'lead', 'iron', 'coal', 'uranium', 'platinum', 'palladium'
  ];

  // Pattern to match company listings (varies by source)
  const patterns = [
    // ASX pattern: "COMPANY NAME (TICKER)"
    /([A-Z][A-Za-z\s&\.-]+)\s*\(([A-Z]{2,5})\)/g,
    // Generic pattern: ticker followed by company name
    /([A-Z]{2,5})\s+([A-Z][A-Za-z\s&\.-]+)/g,
    // Table row pattern with pipes
    /\|\s*([A-Z][A-Za-z\s&\.-]+)\s*\|\s*([A-Z]{2,5})\s*\|/g
  ];

  for (const line of lines) {
    const lowerLine = line.toLowerCase();

    // Check if line contains mining-related keywords
    const hasMiningKeyword = miningKeywords.some(keyword => lowerLine.includes(keyword));

    for (const pattern of patterns) {
      const matches = Array.from(line.matchAll(pattern));

      for (const match of matches) {
        const name = match[1]?.trim();
        const ticker = match[2]?.trim();

        if (name && name.length > 2 && name.length < 100) {
          // Add company if it has mining keywords or comes from a mining-specific source
          if (hasMiningKeyword || source.toLowerCase().includes('mining')) {
            companies.push({
              name,
              ticker,
              exchange: source.includes('ASX') ? 'ASX' :
                       source.includes('TSX') ? 'TSX' :
                       source.includes('LSE') ? 'LSE' : undefined,
              url: undefined
            });
          }
        }
      }
    }
  }

  // Deduplicate by name
  const uniqueCompanies = Array.from(
    new Map(companies.map(c => [c.name, c])).values()
  );

  return uniqueCompanies;
}

/**
 * Search for mining companies using Firecrawl search API
 */
async function searchMiningCompanies(query: string, limit: number = 10): Promise<MiningCompany[]> {
  console.log(`\n🔎 Searching: ${query}`);

  try {
    const searchResults = await firecrawl.search(query, {
      limit,
      scrapeOptions: {
        formats: ['markdown'],
        timeout: 30000,
        waitFor: 1000
      }
    });

    if (!searchResults.success || !searchResults.data) {
      console.log('   ⚠️  Search failed or returned no results');
      return [];
    }

    console.log(`   ✓ Found ${searchResults.data.length} results`);

    const companies: MiningCompany[] = [];

    for (const result of searchResults.data) {
      // Extract company info from search results
      const company = extractCompanyFromSearchResult(result);
      if (company) {
        companies.push(company);
      }
    }

    console.log(`   ✓ Extracted ${companies.length} companies from search`);
    return companies;

  } catch (error) {
    console.error(`   ❌ Search error:`, error);
    return [];
  }
}

/**
 * Extract company information from a search result
 */
function extractCompanyFromSearchResult(result: any): MiningCompany | null {
  const title = result.title || '';
  const markdown = result.markdown || '';
  const url = result.url || '';

  // Look for company name patterns
  const nameMatch = title.match(/([A-Z][A-Za-z\s&\.-]+(?:Mining|Minerals|Resources|Metals|Gold|Silver|Copper|Lithium))/);

  if (nameMatch) {
    return {
      name: nameMatch[1].trim(),
      url,
      commodities: extractCommodities(markdown)
    };
  }

  return null;
}

/**
 * Extract commodity types from text
 */
function extractCommodities(text: string): string[] {
  const commodityMap: Record<string, string> = {
    'gold': 'Gold',
    'silver': 'Silver',
    'copper': 'Copper',
    'lithium': 'Lithium',
    'nickel': 'Nickel',
    'cobalt': 'Cobalt',
    'zinc': 'Zinc',
    'lead': 'Lead',
    'iron ore': 'Iron Ore',
    'coal': 'Coal',
    'uranium': 'Uranium',
    'rare earth': 'Rare Earth Elements',
    'platinum': 'Platinum',
    'palladium': 'Palladium',
    'graphite': 'Graphite',
    'manganese': 'Manganese'
  };

  const lowerText = text.toLowerCase();
  const found: string[] = [];

  for (const [keyword, commodity] of Object.entries(commodityMap)) {
    if (lowerText.includes(keyword)) {
      found.push(commodity);
    }
  }

  return [...new Set(found)]; // Remove duplicates
}

/**
 * Main execution function
 */
async function main() {
  console.log('🚀 Mining Company Extraction System');
  console.log('=' .repeat(60));
  console.log('Target: Extract 3-4 figure mining companies worldwide');
  console.log('Method: Firecrawl API scraping and search');
  console.log('=' .repeat(60));

  const allCompanies: MiningCompany[] = [];

  // Strategy 1: Search for mining company lists
  const searchQueries = [
    'list of mining companies ASX TSX LSE',
    'top gold mining companies publicly traded',
    'lithium mining companies stock exchange',
    'copper mining companies listed',
    'junior mining companies exploration'
  ];

  for (const query of searchQueries) {
    const companies = await searchMiningCompanies(query, 5);
    allCompanies.push(...companies);

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Strategy 2: Scrape specific exchange pages (commented out for now to avoid rate limits)
  /*
  for (const source of DATA_SOURCES.slice(0, 2)) { // Start with first 2 sources
    const companies = await extractCompaniesFromUrl(source.url, source.name);
    allCompanies.push(...companies);

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  */

  // Deduplicate companies by name
  const uniqueCompanies = Array.from(
    new Map(allCompanies.map(c => [c.name.toLowerCase(), c])).values()
  );

  console.log('\n' + '='.repeat(60));
  console.log('📊 EXTRACTION RESULTS');
  console.log('='.repeat(60));
  console.log(`Total unique companies found: ${uniqueCompanies.length}`);
  console.log('');

  // Display subset of companies (first 50)
  const displayCount = Math.min(50, uniqueCompanies.length);
  console.log(`\n📋 Displaying ${displayCount} companies:\n`);

  uniqueCompanies.slice(0, displayCount).forEach((company, index) => {
    console.log(`${(index + 1).toString().padStart(3)}. ${company.name}`);
    if (company.ticker) console.log(`     Ticker: ${company.ticker}`);
    if (company.exchange) console.log(`     Exchange: ${company.exchange}`);
    if (company.commodities && company.commodities.length > 0) {
      console.log(`     Commodities: ${company.commodities.join(', ')}`);
    }
    if (company.url) console.log(`     URL: ${company.url}`);
    console.log('');
  });

  console.log('='.repeat(60));
  console.log('✅ Extraction complete');
  console.log('='.repeat(60));

  // Return companies for potential further processing
  return uniqueCompanies;
}

// Execute
main()
  .then(() => {
    console.log('\n✨ Process finished successfully');
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n❌ Fatal error:', error);
    process.exit(1);
  });
