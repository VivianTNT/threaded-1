-- Create news table
CREATE TABLE IF NOT EXISTS news (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  urls TEXT[], -- Array of URLs from multiple sources
  source TEXT, -- News source name
  published_at TIMESTAMPTZ,
  summary TEXT,
  commodities TEXT[], -- Array of relevant commodities
  project_ids UUID[], -- Array of related project IDs
  sentiment TEXT, -- e.g., 'Positive', 'Negative', 'Neutral'
  watchlist BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_commodities ON news USING GIN(commodities);
CREATE INDEX IF NOT EXISTS idx_news_project_ids ON news USING GIN(project_ids);
CREATE INDEX IF NOT EXISTS idx_news_watchlist ON news(watchlist) WHERE watchlist = TRUE;

-- Create updated_at trigger
CREATE TRIGGER update_news_updated_at
  BEFORE UPDATE ON news
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
