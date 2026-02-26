import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
});

interface SensitivityAnalysisRequest {
  baseCase: {
    npv: number;
    irr: number;
    aisc: number;
  };
  parameters: {
    commodityPrice?: number; // % change
    throughput?: number; // % change
    grade?: number; // % change
    opex?: number; // % change
    capex?: number; // % change
    recovery?: number; // % change
  };
  projectContext?: {
    name?: string;
    commodity?: string;
    mineLife?: number;
    annualProduction?: number;
  };
  actualValues?: {
    commodityPrice?: { value: number; unit: string };
    aisc?: { value: number; unit: string };
    capex?: { value: number; unit: string };
  };
}

interface SensitivityAnalysisResult {
  npv: number;
  irr: number;
  aisc: number;
  explanation: string;
  assumptions: string[];
  riskFactors: string[];
}

/**
 * POST /api/sensitivity-analysis
 * Use GPT to predict NPV/IRR/AISC changes based on parameter adjustments
 */
export async function POST(req: NextRequest) {
  try {
    const body: SensitivityAnalysisRequest = await req.json();
    const { baseCase, parameters, projectContext, actualValues } = body;

    // Validate base case
    if (!baseCase || baseCase.npv === undefined || baseCase.irr === undefined || baseCase.aisc === undefined) {
      return NextResponse.json(
        { error: 'Base case NPV, IRR, and AISC are required' },
        { status: 400 }
      );
    }

    // Build context for GPT with actual values
    let actualValuesString = '';
    if (actualValues) {
      actualValuesString = '\n**Actual Baseline Values:**';
      if (actualValues.commodityPrice) {
        const baseCommodityPrice = actualValues.commodityPrice.value;
        const newCommodityPrice = baseCommodityPrice * (1 + (parameters.commodityPrice || 0) / 100);
        actualValuesString += `\n- Base Commodity Price: ${baseCommodityPrice} ${actualValues.commodityPrice.unit}`;
        if (parameters.commodityPrice) {
          actualValuesString += `\n- New Commodity Price: ${newCommodityPrice.toFixed(0)} ${actualValues.commodityPrice.unit} (${parameters.commodityPrice > 0 ? '+' : ''}${parameters.commodityPrice}%)`;
        }
      }
      if (actualValues.aisc) {
        actualValuesString += `\n- Base AISC: ${actualValues.aisc.value.toFixed(2)} ${actualValues.aisc.unit}`;
      }
      if (actualValues.capex) {
        actualValuesString += `\n- Base CAPEX: $${actualValues.capex.value.toFixed(1)}M`;
      }
    }

    const contextString = `
**Base Case Financial Metrics:**
- NPV: $${baseCase.npv.toFixed(1)}M
- IRR: ${baseCase.irr.toFixed(1)}%
- AISC: $${baseCase.aisc.toFixed(2)}/unit
${actualValuesString}

**Parameter Changes:**
${parameters.commodityPrice ? `- Commodity Price: ${parameters.commodityPrice > 0 ? '+' : ''}${parameters.commodityPrice}%` : ''}
${parameters.throughput ? `- Throughput: ${parameters.throughput > 0 ? '+' : ''}${parameters.throughput}%` : ''}
${parameters.grade ? `- Grade: ${parameters.grade > 0 ? '+' : ''}${parameters.grade}%` : ''}
${parameters.opex ? `- Operating Costs (OPEX): ${parameters.opex > 0 ? '+' : ''}${parameters.opex}%` : ''}
${parameters.capex ? `- Capital Expenditure (CAPEX): ${parameters.capex > 0 ? '+' : ''}${parameters.capex}%` : ''}
${parameters.recovery ? `- Metallurgical Recovery: ${parameters.recovery > 0 ? '+' : ''}${parameters.recovery}%` : ''}

${projectContext ? `
**Project Context:**
- Project: ${projectContext.name || 'N/A'}
- Commodity: ${projectContext.commodity || 'N/A'}
- Mine Life: ${projectContext.mineLife ? `${projectContext.mineLife} years` : 'N/A'}
- Annual Production: ${projectContext.annualProduction || 'N/A'}
` : ''}
`.trim();

    const systemPrompt = `You are a mining financial analyst specializing in sensitivity analysis and economic modeling.

**CRITICAL REQUIREMENT - QUOTE ACTUAL NUMBERS:**
You MUST reference the specific base case values and calculated results. Do NOT provide generic analysis.

**MANDATORY RULES FOR EXPLANATION:**
1. Quote the EXACT base case values (e.g., "Starting from base NPV of $485M, IRR of 22.4%, AISC of $850/oz...")
2. Quote the EXACT parameter changes (e.g., "With commodity price +10%, throughput -5%...")
3. Quote the EXACT new calculated values (e.g., "New NPV of $627M (+29%), IRR of 26.8% (+4.4 pts), AISC of $823/oz (-3%)")
4. Explain WHY each metric changed using actual numbers
5. Reference the project name if provided

**Key Relationships (use these for calculations):**
1. **NPV Impact:**
   - Commodity price: ~1.5-2.5x leverage (10% price increase → 15-25% NPV increase)
   - Throughput: ~1.0-1.5x leverage
   - Grade: ~1.2-1.8x leverage
   - OPEX: ~0.8-1.2x leverage (inverse)
   - CAPEX: ~0.3-0.6x leverage (inverse)
   - Recovery: ~1.0-1.5x leverage

2. **IRR Impact:**
   - Commodity price: ~0.8-1.2x percentage point change
   - Throughput: ~0.5-0.8x
   - Grade: ~0.6-1.0x
   - OPEX: ~0.5-0.8x (inverse)
   - CAPEX: ~0.4-0.7x (inverse)
   - Recovery: ~0.5-0.9x

3. **AISC Impact:**
   - Commodity price: minimal (0-5%)
   - Throughput: ~0.6-0.9x (inverse, due to fixed cost spreading)
   - Grade: ~0.7-1.0x (inverse)
   - OPEX: ~0.8-1.0x (direct)
   - CAPEX: ~0.1-0.3x (sustaining capex component)
   - Recovery: ~0.6-0.9x (inverse)

**Instructions:**
1. Calculate new NPV, IRR, and AISC using the leverage ratios above
2. In "explanation": Quote base case → quote parameters changed → quote new values → explain the impact
3. In "assumptions": List specific assumptions with numbers (e.g., "NPV discount rate remains at 8%", "Mine life of 15 years unchanged")
4. In "riskFactors": Identify specific risks with numbers (e.g., "AISC of $823/oz still above industry median of $750/oz")

Return a JSON object with this exact structure:
{
  "npv": number (in millions USD),
  "irr": number (as percentage),
  "aisc": number (in USD/unit),
  "explanation": string (MUST quote actual base and new values),
  "assumptions": string[] (MUST include specific numbers),
  "riskFactors": string[] (MUST reference calculated values)
}`;

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: contextString }
      ],
      response_format: { type: "json_object" },
      temperature: 0.2,
      max_tokens: 1000
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error('No response from OpenAI');
    }

    const result = JSON.parse(content) as SensitivityAnalysisResult;

    // Validate response structure
    if (result.npv === undefined || result.irr === undefined || result.aisc === undefined) {
      throw new Error('Invalid response structure from GPT');
    }

    return NextResponse.json({
      success: true,
      result
    });

  } catch (error: any) {
    console.error('Sensitivity analysis error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to perform sensitivity analysis' },
      { status: 500 }
    );
  }
}
