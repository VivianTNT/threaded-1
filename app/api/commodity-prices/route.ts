import { NextRequest, NextResponse } from 'next/server';
import { getCommodityPrice } from '@/lib/commodity-price-fetcher';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

interface CommodityPricesRequest {
  projectId: string;
  commodities: string[];
  aisc?: number | null;
  capex?: number | null;
}

/**
 * POST /api/commodity-prices
 * Fetch actual commodity prices and other baseline values for sensitivity analysis
 */
export async function POST(req: NextRequest) {
  try {
    const body: CommodityPricesRequest = await req.json();
    const { projectId, commodities, aisc, capex } = body;

    if (!projectId || !commodities || commodities.length === 0) {
      return NextResponse.json(
        { error: 'projectId and commodities are required' },
        { status: 400 }
      );
    }

    // Get price for primary commodity (first in list)
    const primaryCommodity = commodities[0];
    const commodityPriceData = await getCommodityPrice(projectId, primaryCommodity);

    // Fetch technical document data for production parameters
    const { data: pdfHighlights } = await supabase
      .from('pdf_highlights')
      .select('highlight_data')
      .eq('project_id', projectId)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    let throughput = null;
    let grade = null;
    let recovery = null;

    // Extract production parameters from technical documents
    if (pdfHighlights?.highlight_data) {
      const data = pdfHighlights.highlight_data as any;
      let documentText = '';

      if (data.extractedText) documentText += data.extractedText;
      if (data.highlights) documentText += JSON.stringify(data.highlights);

      // Extract throughput (tonnes per day/year)
      const throughputMatch = documentText.match(/(?:throughput|mill\s+capacity|processing\s+rate)[^\d]*([\d,]+)\s*(tpd|tpa|t\/d|t\/a|tonnes?\s*per\s*(?:day|year))/i);
      if (throughputMatch) {
        throughput = parseFloat(throughputMatch[1].replace(/,/g, ''));
      }

      // Extract grade (g/t or %)
      const gradeMatch = documentText.match(/(?:head\s+grade|average\s+grade|ore\s+grade)[^\d]*([\d.]+)\s*(g\/t|%)/i);
      if (gradeMatch) {
        grade = { value: parseFloat(gradeMatch[1]), unit: gradeMatch[2] };
      }

      // Extract recovery rate (%)
      const recoveryMatch = documentText.match(/(?:recovery|metallurgical\s+recovery)[^\d]*([\d.]+)\s*%/i);
      if (recoveryMatch) {
        recovery = parseFloat(recoveryMatch[1]);
      }
    }

    const actualValues: Record<string, { value: number; unit: string; source: string }> = {};

    // Commodity price
    if (commodityPriceData.price > 0) {
      actualValues['price'] = {
        value: commodityPriceData.price,
        unit: commodityPriceData.unit,
        source: commodityPriceData.source
      };
    }

    // Throughput - use default typical value if not found
    actualValues['throughput'] = throughput ? {
      value: throughput,
      unit: 'tpd',
      source: 'technical_document'
    } : {
      value: 10000, // Typical medium-scale mine throughput
      unit: 'tpd',
      source: 'default'
    };

    // Grade - use default typical value if not found
    actualValues['grade'] = grade ? {
      value: grade.value,
      unit: grade.unit,
      source: 'technical_document'
    } : {
      value: 1.5, // Typical grade for many commodities
      unit: 'g/t',
      source: 'default'
    };

    // Recovery - use default typical value if not found
    actualValues['recovery'] = recovery ? {
      value: recovery,
      unit: '%',
      source: 'technical_document'
    } : {
      value: 85, // Typical metallurgical recovery rate
      unit: '%',
      source: 'default'
    };

    // AISC (Operating Costs) - use default based on commodity if not in project data
    if (aisc && aisc > 0) {
      actualValues['opex'] = {
        value: aisc,
        unit: commodityPriceData.unit, // Same unit as commodity price
        source: 'project_metadata'
      };
    } else {
      // Default AISC based on commodity type (typically 60-70% of commodity price)
      const defaultAISC = commodityPriceData.price * 0.65;
      actualValues['opex'] = {
        value: defaultAISC,
        unit: commodityPriceData.unit,
        source: 'default'
      };
    }

    // CAPEX (already in project data) - use typical default if not available
    if (capex && capex > 0) {
      actualValues['capex'] = {
        value: capex,
        unit: 'USD M',
        source: 'project_metadata'
      };
    } else {
      // Default CAPEX for medium-scale mine (typical range $500M - $2B)
      actualValues['capex'] = {
        value: 1000, // $1B typical for medium-scale operation
        unit: 'USD M',
        source: 'default'
      };
    }

    return NextResponse.json({
      success: true,
      actualValues,
      primaryCommodity
    });

  } catch (error: any) {
    console.error('Error fetching commodity prices:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch commodity prices' },
      { status: 500 }
    );
  }
}
