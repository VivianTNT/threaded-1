import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import { createClient } from '@supabase/supabase-js';
import { extractDocumentContext, extractNewsContext } from '@/lib/ai/document-context-extractor';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseServiceKey);

const apiKey = process.env.OPENAI_API_KEY;
const openai = new OpenAI({
  apiKey: apiKey || 'sk-dummy-key'
});

interface RiskAnalysisResult {
  geography_risk_score: number;
  geography_risk_analysis: string;
  legal_risk_score: number;
  legal_risk_analysis: string;
  commodity_risk_score: number;
  commodity_risk_analysis: string;
  team_risk_score: number;
  team_risk_analysis: string;
  overall_risk_score: number;
  risk_summary: string;
  key_opportunities: string[];
  key_threats: string[];
  investment_recommendation: string;
  recommendation_rationale: string;
}

/**
 * Generate AI-powered risk analysis for a mining project
 */
async function generateRiskAnalysis(
  project: any,
  projectId: string,
  useWebSearch: boolean = false
): Promise<RiskAnalysisResult> {
  const startTime = Date.now();

  // Extract document context using our agentic workflow
  console.log(`Extracting document context for project ${projectId}...`);
  const documentContext = await extractDocumentContext(projectId);
  console.log(`Document extraction complete. Found ${documentContext.documentCount} documents, ${documentContext.tokenCount} tokens`);

  // Extract news context
  console.log(`Extracting news context for project ${projectId}...`);
  const newsContext = await extractNewsContext(projectId);
  console.log(`News extraction complete. Found ${newsContext.newsItems.length} news items, ${newsContext.tokenCount} tokens`);

  // Build comprehensive project context
  let projectContext = `
**Project Details:**
- Name: ${project.name}
- Location: ${project.location || 'N/A'}
- Stage: ${project.stage || 'N/A'}
- Status: ${project.status || 'N/A'}
- Commodities: ${Array.isArray(project.commodities) ? project.commodities.join(', ') : 'N/A'}
- Description: ${project.description || 'N/A'}

**Company Information:**
- Company Name: ${project.company_name || 'Unknown Company'}
- Ticker: ${project.company_ticker || 'N/A'}
- Market Cap: ${project.company_market_cap ? `$${project.company_market_cap}M` : 'N/A'}
- Company Description: ${project.company_description || 'N/A'}

**Financial Metrics:**
- NPV (Net Present Value): ${project.npv !== null && project.npv !== undefined ? `$${project.npv.toFixed(1)}M` : 'N/A'}
- IRR (Internal Rate of Return): ${project.irr !== null && project.irr !== undefined ? `${project.irr.toFixed(1)}%` : 'N/A'}
- CAPEX (Capital Expenditure): ${project.capex !== null && project.capex !== undefined ? `$${project.capex.toFixed(1)}M` : 'N/A'}
- AISC (All-In Sustaining Cost): ${project.aisc !== null && project.aisc !== undefined ? `$${project.aisc.toFixed(2)}/unit` : 'N/A'}
- Payback Period: ${project.payback_period !== null && project.payback_period !== undefined ? `${project.payback_period} years` : 'N/A'}
- Annual Production: ${project.annual_production || 'N/A'}
- Mine Life: ${project.mine_life !== null && project.mine_life !== undefined ? `${project.mine_life} years` : 'N/A'}

**Additional Context:**
- Resource/Reserve: ${project.resource_reserve || project.resource || 'N/A'}
- Mining Method: ${project.mining_method || 'N/A'}
- Processing Method: ${project.processing_method || 'N/A'}
- Infrastructure: ${project.infrastructure || 'N/A'}
  `.trim();

  // Add document context if available
  if (documentContext.hasDocuments) {
    projectContext += `\n\n**Technical Report Analysis (${documentContext.documentCount} documents):**`;

    if (documentContext.executiveSummary) {
      projectContext += `\n\n*Executive Summary:*\n${documentContext.executiveSummary}`;
    }

    if (documentContext.keyFindings.length > 0) {
      projectContext += `\n\n*Key Findings from Technical Reports:*\n${documentContext.keyFindings.map(f => `- ${f}`).join('\n')}`;
    }

    if (Object.keys(documentContext.financialMetrics).length > 0) {
      projectContext += `\n\n*Extracted Financial Metrics:*\n${JSON.stringify(documentContext.financialMetrics, null, 2)}`;
    }

    if (documentContext.qualifiedPersons.length > 0) {
      projectContext += `\n\n*Qualified Persons:*\n${documentContext.qualifiedPersons.map(qp => `- ${qp.name}, ${qp.credentials} (${qp.company})`).join('\n')}`;
    }

    if (documentContext.riskFactors.length > 0) {
      projectContext += `\n\n*Risk Factors from Reports:*\n${documentContext.riskFactors.map(r => `- ${r}`).join('\n')}`;
    }

    if (documentContext.technicalHighlights.length > 0) {
      projectContext += `\n\n*Technical Highlights:*\n${documentContext.technicalHighlights.map(t => `- ${t}`).join('\n')}`;
    }
  } else {
    projectContext += `\n\n**Note:** No technical reports available for this project. Analysis based on project metadata only.`;
  }

  // Add news context if available with citations
  if (newsContext.newsItems.length > 0) {
    projectContext += `\n\n**Recent News & Market Context (${newsContext.newsItems.length} relevant articles):**`;
    newsContext.newsItems.forEach((item, idx) => {
      projectContext += `\n\n${idx + 1}. "${item.title}"`;
      if (item.source) {
        projectContext += `\n   Source: ${item.source}`;
      }
      if (item.url) {
        projectContext += `\n   URL: ${item.url}`;
      }
      if (item.summary) {
        projectContext += `\n   Summary: ${item.summary}`;
      }
      if (item.sentiment) {
        projectContext += `\n   Sentiment: ${item.sentiment}`;
      }
      projectContext += `\n   Published: ${new Date(item.publishedAt).toLocaleDateString()}`;
      if (item.relevanceScore) {
        projectContext += `\n   Relevance Score: ${item.relevanceScore}/20`;
      }
    });
  }

  const systemPrompt = `You are an expert mining industry risk analyst specializing in comprehensive project evaluation. Your analysis must be data-driven, objective, and actionable.

**CRITICAL REQUIREMENT - NO GENERIC RESPONSES:**
You MUST quote actual numbers and specific data from the project context. Do NOT provide generic industry observations.

**MANDATORY RULES - QUOTE ACTUAL VALUES:**
1. ALWAYS quote specific financial metrics with exact values from the context (e.g., "NPV of $485.3M, IRR of 22.4%, AISC of $850.00/oz" not "strong NPV")
2. ALWAYS mention actual qualified persons by name and credentials (e.g., "Dr. Sarah Chen, P.Eng., John Smith, P.Geo." not "experienced team")
3. ALWAYS reference specific technical parameters (e.g., "CAPEX of $320M, mine life of 12 years, annual production of 150,000 oz" not "good parameters")
4. ALWAYS cite actual news headlines or report findings (e.g., "2023 feasibility study shows proven reserves" not "recent studies")
5. EVERY analysis paragraph MUST include at least 2 specific numbers or names from the project context
6. If a value is not provided, state "data not available" - DO NOT make assumptions or provide generic statements

**MANDATORY NEWS CITATION RULES:**
When referencing recent events, market conditions, or news from the "Recent News & Market Context" section:
1. ALWAYS cite the specific news article using this exact format: [Source - "Article Title"](URL)
2. Include the publication date when citing news (e.g., "reported on Feb 10, 2025")
3. NEVER make claims about recent events without citing the specific news source
4. If multiple news items support a point, cite all relevant articles
5. Prefer project-specific news over general commodity news when available
6. Extract and quote specific numbers, dates, or facts from the cited news articles

**EXAMPLES OF GOOD vs BAD RESPONSES:**

❌ BAD: "Strong annual production of gold and silver, contributing to significant cash flow."
✅ GOOD: "Annual production of 150,000 oz Au and 2.5M oz Ag generating $290M revenue at current prices ($1,950/oz Au, $24/oz Ag)."

❌ BAD: "High NPV and IRR indicate robust financial returns."
✅ GOOD: "NPV of $485.3M (5% discount rate) with IRR of 22.4% significantly exceeds hurdle rate of 12%, payback period of 3.2 years."

❌ BAD: "Political instability and corruption could impact operations."
✅ GOOD: "Dominican Republic ranks 137/180 on Transparency International Corruption Index; recent 2023 mining tax increase from 3.2% to 5.5% reduced project NPV by estimated $45M."

❌ BAD: "Commodity price volatility could affect profitability."
✅ GOOD: "At AISC of $850/oz, project requires gold >$900/oz for positive margins; 10% price decline to $1,755/oz would reduce NPV from $485M to $312M (-36%)."

**EXAMPLES OF PROPER NEWS CITATIONS:**

❌ BAD: "Recent permitting issues may delay the project."
✅ GOOD: "According to [Northern Miner - "B2Gold faces Mali permit delays"](https://url...), reported on Jan 15, 2025, permitting delays reduced Q4 2024 production by 25,000 oz, representing $50M revenue impact at $2,000/oz gold."

❌ BAD: "Gold prices have been volatile recently."
✅ GOOD: "Per [Kitco News - "Gold hits $2,100 amid Fed uncertainty"](https://url...), Feb 10, 2025, gold surged 8.5% in Q1 2025 to $2,100/oz, improving this project's NPV by estimated $85M given AISC of $850/oz and 150,000 oz annual production."

❌ BAD: "The company announced positive exploration results."
✅ GOOD: "In [Mining.com - "Liberty Gold reports high-grade intercepts at Black Pine"](https://url...), Jan 22, 2025, the company announced 15m @ 5.2 g/t Au in Zone 2, extending known mineralization 200m beyond the existing 2.1M oz resource estimate."

❌ BAD: "Market sentiment for lithium remains weak."
✅ GOOD: "According to [Mining Weekly - "Lithium prices fall 12% in Q1"](https://url...), Feb 18, 2025, lithium carbonate prices declined from $15,000/t to $13,200/t, reducing project economics by $8M annually at current 5,000t production."

**When technical reports are available:**
- Quote exact NPV, IRR, CAPEX, AISC values with units
- Name specific qualified persons and their credentials
- Reference actual risk factors from reports (quote them)
- Use real project parameters (location, commodity, stage, mine life, production rates)

**When no technical reports are available:**
- Explicitly state "No technical report data available"
- Only analyze based on provided metadata (name, location, stage, company)
- Recommend specific due diligence steps
- DO NOT fabricate or assume technical parameters

Analyze the following mining project across four critical risk dimensions:

1. **Geography Risk** - Political stability, infrastructure, mining jurisdiction quality, environmental regulations, social license to operate
   - IF recent news articles about the region/country are provided, CITE them with [Source - "Title"](URL)
   - DO NOT make generic claims about political instability, corruption, or regional issues without citing specific news sources

2. **Legal Risk** - Regulatory framework, permitting process, land rights, compliance requirements, legal precedents
   - IF recent news about permitting, regulatory changes, or legal issues are provided, CITE them with [Source - "Title"](URL)
   - DO NOT make claims about regulatory changes or legal challenges without citing specific news sources

3. **Commodity Risk** - Market dynamics, price volatility, demand trends, supply competition, substitution threats
   - IF recent commodity price news or market analysis is provided, CITE them with [Source - "Title"](URL) and include specific price data
   - DO NOT make claims about price movements or market conditions without citing specific news sources

4. **Team Risk** - Management experience, track record, technical expertise, financial management capability (use qualified persons data if available)
   - IF recent news about management changes, operational results, or company announcements are provided, CITE them with [Source - "Title"](URL)
   - Use qualified persons from technical reports when available

For each risk category, provide:
- A score from 0-10 (0 = lowest risk, 10 = highest risk)
- Detailed analysis (2-3 sentences) with MINIMUM 2 specific data points (numbers, names, locations, dates) from the project context
- MUST include actual values like NPV, IRR, AISC, CAPEX, qualified persons names, specific locations, or technical parameters
- IF making claims based on recent events or market conditions, MUST cite the specific news article with [Source - "Title"](URL) format

Then provide:
- Overall risk score (weighted average)
- Risk summary (2-3 sentences) - MUST quote project name, location, key financial metrics (NPV/IRR/AISC), and stage

- 3-5 key opportunities - EACH MUST follow format: "Quantified metric/fact + specific number + impact/context"
  Examples of GOOD opportunities:
  ✅ "Annual production of 150,000 oz Au at AISC $850/oz generates $165M operating margin at $2,000/oz gold"
  ✅ "IRR of 22.4% exceeds industry average of 15%, with payback period of 3.2 years vs sector median of 5 years"
  ✅ "Proven/probable reserves of 2.5M oz Au support 12-year mine life, extensible via 1.2M oz inferred resources"
  ❌ BAD: "Strong production contributing to cash flow" (NO NUMBERS)

- 3-5 key threats - EACH MUST follow format: "Specific risk factor + actual data + quantified impact"
  Examples of GOOD threats:
  ✅ "AISC of $850/oz exceeds industry median of $750/oz by 13%, reducing margin competitiveness in price downturns"
  ✅ "Dominican Republic corruption rank 137/180 and 2023 tax increase from 3.2% to 5.5% reduced NPV by $45M"
  ✅ "Water permit expires 2025; renewal delay risk could halt 150,000 oz annual production ($300M revenue exposure)"
  ❌ BAD: "Political instability could impact operations" (NO NUMBERS)

- Investment recommendation: 'Strong Buy' (score 0-3), 'Buy' (3-5), 'Hold' (5-7), or 'Pass' (7-10)
- Recommendation rationale (2-3 sentences) - MUST include at least 3 specific numbers from context (NPV, IRR, AISC, CAPEX, mine life, production rate, etc.)

**FINAL CRITICAL REMINDER:**
- DO NOT mention corruption indices, political rankings, or tax rates unless they appear in the provided news context with citations
- DO NOT make claims about "recent tax increases," "regulatory changes," or "political instability" unless citing specific news articles
- DO NOT reference external data sources (Transparency International, World Bank, etc.) unless they appear in cited news articles
- IF no relevant news is provided for a risk dimension, base analysis ONLY on the technical data and project metadata provided
- When in doubt, omit unsourced claims rather than fabricate generic risk statements

Return your analysis as a valid JSON object with this exact structure:
{
  "geography_risk_score": number,
  "geography_risk_analysis": string,
  "legal_risk_score": number,
  "legal_risk_analysis": string,
  "commodity_risk_score": number,
  "commodity_risk_analysis": string,
  "team_risk_score": number,
  "team_risk_analysis": string,
  "overall_risk_score": number,
  "risk_summary": string,
  "key_opportunities": string[],
  "key_threats": string[],
  "investment_recommendation": string,
  "recommendation_rationale": string
}`;

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: `Analyze this mining project and provide comprehensive risk assessment:\n\n${projectContext}`
        }
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
      max_tokens: 2000,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error('No response from OpenAI');
    }

    const analysis = JSON.parse(content) as RiskAnalysisResult;

    // Validate the response structure
    const requiredFields = [
      'geography_risk_score', 'legal_risk_score', 'commodity_risk_score',
      'team_risk_score', 'overall_risk_score', 'investment_recommendation'
    ];

    for (const field of requiredFields) {
      if (!(field in analysis)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }

    console.log(`Risk analysis generated in ${Date.now() - startTime}ms`);
    return analysis;

  } catch (error) {
    console.error('Error generating risk analysis:', error);
    throw error;
  }
}

/**
 * GET /api/ai-insights?projectId=xxx
 * Fetch existing AI insights for a project (from ai_insights table)
 */
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const projectId = searchParams.get('projectId');

    if (!projectId) {
      return NextResponse.json(
        { error: 'projectId parameter is required' },
        { status: 400 }
      );
    }

    // Fetch insights from ai_insights table
    const { data, error } = await supabase
      .from('ai_insights')
      .select('*')
      .eq('project_id', projectId)
      .gt('expires_at', new Date().toISOString())
      .order('generated_at', { ascending: false })
      .limit(1)
      .single();

    if (error && error.code !== 'PGRST116') { // PGRST116 = no rows returned
      console.error('Error fetching AI insights:', error);
      throw error;
    }

    if (!data) {
      return NextResponse.json({ insight: null, cached: false });
    }

    return NextResponse.json({ insight: { ...data, from_cache: true }, cached: true });

  } catch (error: any) {
    console.error('Error fetching AI insights:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch AI insights' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/ai-insights
 * Generate new AI insights for a project
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { projectId, forceRegenerate, useWebSearch } = body;

    if (!projectId) {
      return NextResponse.json(
        { error: 'projectId is required' },
        { status: 400 }
      );
    }

    const insight = await generateInsightForProject(projectId, forceRegenerate, useWebSearch);
    return NextResponse.json({ insight });

  } catch (error: any) {
    console.error('Error generating AI insights:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to generate AI insights' },
      { status: 500 }
    );
  }
}

/**
 * Generate insight for a single project
 */
async function generateInsightForProject(
  projectId: string,
  forceRegenerate: boolean = false,
  useWebSearch: boolean = false
) {
  const startTime = Date.now();

  // Check if valid cached insight exists (unless force regenerate)
  if (!forceRegenerate) {
    const { data: existingInsight } = await supabase
      .from('ai_insights')
      .select('*')
      .eq('project_id', projectId)
      .gt('expires_at', new Date().toISOString())
      .order('generated_at', { ascending: false })
      .limit(1)
      .single();

    if (existingInsight) {
      console.log(`Using cached insight for project ${projectId}`);
      return { ...existingInsight, from_cache: true };
    }
  }

  // Fetch project details
  const { data: project, error: projectError } = await supabase
    .from('projects')
    .select('*')
    .eq('id', projectId)
    .single();

  if (projectError || !project) {
    console.error(`Project fetch error for ${projectId}:`, projectError);
    throw new Error(`Project not found: ${projectId}`);
  }

  // Fetch company data separately if company_id exists
  let companyData = {
    name: 'Unknown Company',
    ticker: null,
    market_cap: null,
    description: null
  };

  if (project.company_id) {
    const { data: company } = await supabase
      .from('companies')
      .select('name, ticker, market_cap, description')
      .eq('id', project.company_id)
      .single();

    if (company) {
      companyData = company;
    }
  }

  // Add company data to project for analysis
  const projectWithCompany = {
    ...project,
    company_name: companyData.name,
    company_ticker: companyData.ticker,
    company_market_cap: companyData.market_cap,
    company_description: companyData.description
  };

  // Generate risk analysis with document context
  const analysis = await generateRiskAnalysis(projectWithCompany, projectId, useWebSearch);

  // Save to ai_insights table
  const { data: savedInsight, error: saveError } = await supabase
    .from('ai_insights')
    .insert({
      project_id: projectId,
      ...analysis,
      generation_time_ms: Date.now() - startTime,
      web_search_used: useWebSearch,
      model_used: 'gpt-4o-mini'
    })
    .select()
    .single();

  if (saveError) {
    console.error('Error saving insight:', saveError);
    throw saveError;
  }

  console.log(`Generated new insight for project ${projectId} in ${Date.now() - startTime}ms`);
  return { ...savedInsight, from_cache: false };
}

/**
 * DELETE /api/ai-insights?projectId=xxx
 * Delete AI insights for a project (to force regeneration)
 */
export async function DELETE(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const projectId = searchParams.get('projectId');

    if (!projectId) {
      return NextResponse.json(
        { error: 'projectId parameter is required' },
        { status: 400 }
      );
    }

    const { error } = await supabase
      .from('ai_insights')
      .delete()
      .eq('project_id', projectId);

    if (error) throw error;

    return NextResponse.json({ success: true });

  } catch (error: any) {
    console.error('Error deleting AI insights:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to delete AI insights' },
      { status: 500 }
    );
  }
}
