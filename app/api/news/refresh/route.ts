import { NextResponse } from 'next/server';
import { FirecrawlNewsScraper } from '@/lib/news-scraper/firecrawl-news-scraper';

export async function POST(request: Request) {
  try {
    // Get optional parameters from request body
    const body = await request.json().catch(() => ({}));
    const { 
      maxPerSource = 5,  // Number of articles per source
      sources = null     // Specific sources to scrape (null = all)
    } = body;

    console.log('🔄 Starting news refresh...');
    console.log(`📊 Max articles per source: ${maxPerSource}`);

    // Initialize the scraper
    const scraper = new FirecrawlNewsScraper();

    // Scrape news from all configured sources
    const results = await scraper.scrapeAllSources(maxPerSource);

    console.log('\n✅ News refresh completed!');
    console.log(`📈 Total articles scraped: ${results.totalScraped}`);
    console.log(`💾 Total articles saved: ${results.totalSaved}`);

    // Return detailed results
    return NextResponse.json({
      success: true,
      message: 'News refreshed successfully',
      totalScraped: results.totalScraped,
      totalSaved: results.totalSaved,
      sources: results.sources,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('❌ Error refreshing news:', error);
    
    return NextResponse.json({ 
      success: false,
      error: 'Failed to refresh news',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { 
      status: 500 
    });
  }
}

// GET endpoint to check refresh status or last refresh time
export async function GET(request: Request) {
  try {
    // This could be enhanced to check the last refresh time from the database
    // For now, return a simple status
    return NextResponse.json({
      status: 'ready',
      message: 'News refresh endpoint is ready. Send a POST request to refresh news.',
      lastRefresh: null, // Could query database for last fetched_at
      sources: [
        'Mining Weekly',
        'Mining Review Africa',
        'Northern Miner',
        'Australian Mining',
        'Kitco News',
        'Canadian Mining Journal',
        'International Mining',
        'Mining Magazine',
        'Engineering & Mining Journal',
        'Pit & Quarry',
        'Rock to Road',
        'Fastmarkets',
        'Argus Metals',
        'Reuters Commodities',
        'Mining People'
      ]
    });
  } catch (error) {
    return NextResponse.json({ 
      error: 'Internal server error' 
    }, { 
      status: 500 
    });
  }
}