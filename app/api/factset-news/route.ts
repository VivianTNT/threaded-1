import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import { createClient } from '@supabase/supabase-js';
import { ApiClient, HeadlinesApi } from '@factset/sdk-streetaccountnews';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
});

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// FactSet StreetAccount credentials
const FACTSET_USERNAME = 'LITHOS-2220379';
const FACTSET_PASSWORD = 'YaCeW9yKju608Q4rWUt8yxn0hb2AGQQmpyLde8IG';

interface FactSetNewsItem {
  headline?: string;
  publishedDate?: string;
  source?: string;
  content?: string;
  storyId?: string;
  tickers?: Array<{ ticker: string }>;
  categories?: string[];
  regions?: string[];
  sectors?: string[];
}

interface EnrichedNewsItem {
  title: string;
  urls: string[];
  source: string;
  published_at: string;
  summary: string;
  commodities: string[];
  project_ids: string[];
  sentiment: string;
}

/**
 * POST /api/factset-news
 * Fetch mining news from FactSet StreetAccount and enrich with AI
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      limit = 100,
      sectors = ['Materials', 'Metals & Mining'],
      regions = ['North America', 'South America', 'Australia', 'Africa'],
      startDate,
      endDate
    } = body;

    console.log(`Fetching FactSet news with limit: ${limit}`);

    // Fetch headlines from FactSet StreetAccount
    const factsetNews = await fetchFactSetHeadlines({
      limit,
      sectors,
      regions,
      startDate,
      endDate
    });

    console.log(`Fetched ${factsetNews.length} headlines from FactSet`);

    // Enrich news with AI (sentiment, commodity extraction, etc.)
    const enrichedNews = await enrichNewsWithAI(factsetNews);

    console.log(`Enriched ${enrichedNews.length} news items with AI`);

    // Save to Supabase
    const savedCount = await saveNewsToSupabase(enrichedNews);

    return NextResponse.json({
      success: true,
      fetched: factsetNews.length,
      enriched: enrichedNews.length,
      saved: savedCount,
      message: `Successfully processed ${savedCount} news articles`
    });

  } catch (error: any) {
    console.error('Error fetching FactSet news:', error);
    return NextResponse.json(
      {
        error: error.message || 'Failed to fetch FactSet news',
        details: error.stack
      },
      { status: 500 }
    );
  }
}

/**
 * Fetch headlines from FactSet StreetAccount API
 */
async function fetchFactSetHeadlines(options: {
  limit: number;
  sectors: string[];
  regions: string[];
  startDate?: string;
  endDate?: string;
}): Promise<FactSetNewsItem[]> {
  const { limit, sectors, regions, startDate, endDate } = options;

  // Initialize FactSet API client
  const apiClient = ApiClient.instance;
  const factSetApiKey = apiClient.authentications['FactSetApiKey'];
  factSetApiKey.username = FACTSET_USERNAME;
  factSetApiKey.password = FACTSET_PASSWORD;

  const headlinesApi = new HeadlinesApi();

  // Build request body
  const requestBody: any = {
    data: {}
  };

  // Add filters
  if (sectors.length > 0) {
    requestBody.data.sectors = sectors;
  }

  if (regions.length > 0) {
    requestBody.data.regions = regions;
  }

  // Add date range (use predefined range for simplicity)
  if (!startDate && !endDate) {
    // Default to last month
    requestBody.data.predefinedRange = 'oneMonth';
  } else if (startDate || endDate) {
    // Custom date range
    requestBody.data.searchTime = {
      start: startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
      end: endDate || new Date().toISOString()
    };
  }

  // Add pagination
  requestBody.meta = {
    pagination: {
      limit: Math.min(limit, 200), // API max is 200
      offset: 0
    }
  };

  try {
    // Call the API
    const response = await headlinesApi.getStreetAccountHeadlines(requestBody);

    // Extract headlines from response
    if (response && response.data && Array.isArray(response.data)) {
      return response.data;
    } else {
      console.warn('Unexpected FactSet response structure:', response);
      return [];
    }
  } catch (error: any) {
    console.error('FactSet API error:', error);
    throw new Error(`FactSet API error: ${error.message}`);
  }
}

/**
 * Enrich news items with AI-powered analysis
 */
async function enrichNewsWithAI(newsItems: FactSetNewsItem[]): Promise<EnrichedNewsItem[]> {
  const enrichedNews: EnrichedNewsItem[] = [];

  // Process in batches to avoid rate limits
  const batchSize = 10;
  for (let i = 0; i < newsItems.length; i += batchSize) {
    const batch = newsItems.slice(i, i + batchSize);

    const batchPromises = batch.map(async (item) => {
      try {
        const analysis = await analyzeNewsWithAI(item);
        return {
          title: item.headline || 'No title',
          urls: [], // FactSet may not provide direct URLs
          source: 'FactSet StreetAccount',
          published_at: item.publishedDate || new Date().toISOString(),
          summary: analysis.summary,
          commodities: analysis.commodities,
          project_ids: [], // Will be populated later based on commodities/content
          sentiment: analysis.sentiment
        };
      } catch (error) {
        console.error(`Error enriching news item: ${item.headline}`, error);
        // Return basic item without AI enrichment
        return {
          title: item.headline || 'No title',
          urls: [],
          source: 'FactSet StreetAccount',
          published_at: item.publishedDate || new Date().toISOString(),
          summary: item.content?.substring(0, 500) || item.headline || '',
          commodities: item.sectors || [],
          project_ids: [],
          sentiment: 'Neutral'
        };
      }
    });

    const batchResults = await Promise.all(batchPromises);
    enrichedNews.push(...batchResults);

    // Rate limit: 10 requests/second
    if (i + batchSize < newsItems.length) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  return enrichedNews;
}

/**
 * Analyze a single news item with OpenAI
 */
async function analyzeNewsWithAI(newsItem: FactSetNewsItem): Promise<{
  summary: string;
  commodities: string[];
  sentiment: string;
}> {
  const prompt = `Analyze this mining industry news headline and content:

Headline: ${newsItem.headline}
Content: ${newsItem.content || 'N/A'}
Sectors: ${newsItem.sectors?.join(', ') || 'N/A'}

Extract:
1. A concise 1-2 sentence summary
2. All relevant mining commodities mentioned (e.g., Gold, Copper, Lithium, Silver, etc.)
3. Sentiment (Positive, Negative, or Neutral) - consider impact on mining industry

Return as JSON:
{
  "summary": "...",
  "commodities": ["..."],
  "sentiment": "Positive|Negative|Neutral"
}`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'system',
        content: 'You are a mining industry analyst. Extract structured information from news articles about mining, metals, and commodities.'
      },
      {
        role: 'user',
        content: prompt
      }
    ],
    response_format: { type: 'json_object' },
    temperature: 0.3,
    max_tokens: 500
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    throw new Error('No response from OpenAI');
  }

  const analysis = JSON.parse(content);

  return {
    summary: analysis.summary || newsItem.headline,
    commodities: analysis.commodities || [],
    sentiment: analysis.sentiment || 'Neutral'
  };
}

/**
 * Save enriched news to Supabase
 */
async function saveNewsToSupabase(newsItems: EnrichedNewsItem[]): Promise<number> {
  let savedCount = 0;

  for (const item of newsItems) {
    try {
      // Check if news already exists (by title and published date)
      const { data: existing } = await supabase
        .from('news')
        .select('id')
        .eq('title', item.title)
        .eq('published_at', item.published_at)
        .single();

      if (existing) {
        console.log(`Skipping duplicate: ${item.title}`);
        continue;
      }

      // Insert new news item
      const { error } = await supabase
        .from('news')
        .insert({
          title: item.title,
          urls: item.urls.length > 0 ? item.urls : null,
          source: item.source,
          published_at: item.published_at,
          summary: item.summary,
          commodities: item.commodities.length > 0 ? item.commodities : null,
          project_ids: item.project_ids.length > 0 ? item.project_ids : null,
          sentiment: item.sentiment,
          watchlist: false
        });

      if (error) {
        console.error(`Error saving news: ${item.title}`, error);
      } else {
        savedCount++;
      }
    } catch (error) {
      console.error(`Error processing news item: ${item.title}`, error);
    }
  }

  return savedCount;
}
