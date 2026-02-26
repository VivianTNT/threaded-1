/**
 * Commodity Price Fetcher
 * Extracts commodity prices from technical documents or uses fallback current market prices
 */

import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

// Fallback commodity prices (updated November 2025, USD per unit)
const FALLBACK_PRICES: Record<string, { price: number; unit: string }> = {
  'Gold': { price: 4180, unit: 'USD/oz' },
  'Silver': { price: 36, unit: 'USD/oz' },
  'Copper': { price: 10880, unit: 'USD/t' },
  'Lithium': { price: 11800, unit: 'USD/t' }, // Lithium carbonate
  'Nickel': { price: 14880, unit: 'USD/t' },
  'Cobalt': { price: 28000, unit: 'USD/t' },
  'Zinc': { price: 2740, unit: 'USD/t' },
  'Lead': { price: 2000, unit: 'USD/t' },
  'Iron Ore': { price: 110, unit: 'USD/t' },
  'Platinum': { price: 950, unit: 'USD/oz' },
  'Palladium': { price: 1000, unit: 'USD/oz' },
  'Uranium': { price: 90, unit: 'USD/lb' },
  'Rare Earth Elements': { price: 50000, unit: 'USD/t' },
  'Graphite': { price: 800, unit: 'USD/t' },
  'Molybdenum': { price: 50000, unit: 'USD/t' }, // Updated to ~50K USD/t based on Nov 2025 data
  'Manganese': { price: 1800, unit: 'USD/t' },
  'Vanadium': { price: 22000, unit: 'USD/t' },
  'Tungsten': { price: 28000, unit: 'USD/t' },
  'Tin': { price: 25000, unit: 'USD/t' },
  'Aluminum': { price: 2400, unit: 'USD/t' },
  'Bauxite': { price: 50, unit: 'USD/t' },
  'Phosphate': { price: 150, unit: 'USD/t' },
  'Potash': { price: 350, unit: 'USD/t' },
};

interface CommodityPriceData {
  commodity: string;
  price: number;
  unit: string;
  source: 'technical_document' | 'project_metadata' | 'fallback';
  extractedAt?: string;
}

/**
 * Extract commodity price from technical document text
 */
function extractPriceFromDocument(text: string, commodity: string): { price: number; unit: string } | null {
  const commodityLower = commodity.toLowerCase();

  // Common price patterns for different commodities
  const patterns = [
    // "gold price of $2,050/oz" or "gold at $2,050/oz"
    new RegExp(`${commodityLower}[\\s\\w]*(?:price|at|of)[\\s:]*\\$?([\\d,]+)\\s*\\/\\s*(oz|lb|t|tonne)`, 'i'),
    // "$2,050/oz gold"
    new RegExp(`\\$?([\\d,]+)\\s*\\/\\s*(oz|lb|t|tonne)\\s*${commodityLower}`, 'i'),
    // "assumed gold price: $2,050 per ounce"
    new RegExp(`${commodityLower}\\s+price[\\s:]*\\$?([\\d,]+)\\s*(?:per|\/)?\\s*(oz|ounce|lb|pound|t|tonne)`, 'i'),
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      const price = parseFloat(match[1].replace(/,/g, ''));
      let unit = match[2].toLowerCase();

      // Normalize units
      if (unit === 'ounce') unit = 'oz';
      if (unit === 'pound') unit = 'lb';
      if (unit === 'tonne') unit = 't';

      if (!isNaN(price) && price > 0) {
        return { price, unit: `USD/${unit}` };
      }
    }
  }

  return null;
}

/**
 * Get commodity price for a project
 * Priority: 1) Technical documents 2) Fallback market prices
 */
export async function getCommodityPrice(
  projectId: string,
  commodity: string
): Promise<CommodityPriceData> {
  try {
    // Step 1: Try to extract from technical documents
    const { data: pdfHighlights } = await supabase
      .from('pdf_highlights')
      .select('highlight_data, created_at')
      .eq('project_id', projectId)
      .order('created_at', { ascending: false })
      .limit(3);

    if (pdfHighlights && pdfHighlights.length > 0) {
      for (const pdf of pdfHighlights) {
        if (pdf.highlight_data && typeof pdf.highlight_data === 'object') {
          const data = pdf.highlight_data as any;
          let documentText = '';

          if (data.extractedText) {
            documentText += data.extractedText;
          }
          if (data.highlights) {
            documentText += JSON.stringify(data.highlights);
          }

          const extracted = extractPriceFromDocument(documentText, commodity);
          if (extracted) {
            return {
              commodity,
              price: extracted.price,
              unit: extracted.unit,
              source: 'technical_document',
              extractedAt: pdf.created_at
            };
          }
        }
      }
    }

    // Step 2: Fallback to market prices
    const fallback = FALLBACK_PRICES[commodity];
    if (fallback) {
      return {
        commodity,
        price: fallback.price,
        unit: fallback.unit,
        source: 'fallback'
      };
    }

    // Step 3: Default if commodity not found
    return {
      commodity,
      price: 0,
      unit: 'USD/unit',
      source: 'fallback'
    };

  } catch (error) {
    console.error('Error fetching commodity price:', error);

    // Return fallback on error
    const fallback = FALLBACK_PRICES[commodity];
    return {
      commodity,
      price: fallback?.price || 0,
      unit: fallback?.unit || 'USD/unit',
      source: 'fallback'
    };
  }
}

/**
 * Get all relevant commodity prices for a project (handles multi-commodity projects)
 */
export async function getProjectCommodityPrices(
  projectId: string,
  commodities: string[]
): Promise<CommodityPriceData[]> {
  const prices = await Promise.all(
    commodities.map(commodity => getCommodityPrice(projectId, commodity))
  );
  return prices;
}
