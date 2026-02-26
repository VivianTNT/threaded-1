-- Create table for QuoteMedia news headlines
CREATE TABLE IF NOT EXISTS quotemedia_news (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    news_id BIGINT UNIQUE NOT NULL,
    symbol TEXT,
    company_name TEXT,
    headline TEXT NOT NULL,
    summary TEXT,
    source TEXT,
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    vendor_time TIMESTAMP WITH TIME ZONE,
    story_url TEXT,
    permalink TEXT,
    thumbnail_url TEXT,
    video_url TEXT,
    video_image_url TEXT,

    -- Topics and categorization
    topics TEXT[],
    primary_commodity TEXT,
    commodities TEXT[],
    news_category TEXT, -- 'company', 'commodity', 'market', 'regulatory', etc.

    -- Relevance and metrics
    is_mining_related BOOLEAN DEFAULT false,
    is_project_related BOOLEAN DEFAULT false,
    mentions_financials BOOLEAN DEFAULT false,
    mentions_technical_report BOOLEAN DEFAULT false,
    sentiment_score NUMERIC(3,2),
    relevance_score INTEGER,

    -- Processing flags
    is_processed BOOLEAN DEFAULT false,
    is_archived BOOLEAN DEFAULT false,
    processing_status TEXT DEFAULT 'pending',-
    processing_notes TEXT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,

    -- Indexes for performance
    CONSTRAINT news_id_unique UNIQUE (news_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_quotemedia_news_symbol ON quotemedia_news(symbol);
CREATE INDEX IF NOT EXISTS idx_quotemedia_news_datetime ON quotemedia_news(datetime DESC);
CREATE INDEX IF NOT EXISTS idx_quotemedia_news_mining ON quotemedia_news(is_mining_related) WHERE is_mining_related = true;
CREATE INDEX IF NOT EXISTS idx_quotemedia_news_commodity ON quotemedia_news(primary_commodity);
CREATE INDEX IF NOT EXISTS idx_quotemedia_news_topics ON quotemedia_news USING GIN(topics);
CREATE INDEX IF NOT EXISTS idx_quotemedia_news_created ON quotemedia_news(created_at DESC);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_quotemedia_news_updated_at
    BEFORE UPDATE ON quotemedia_news
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE quotemedia_news IS 'News headlines from QuoteMedia API for mining companies and commodities';
COMMENT ON COLUMN quotemedia_news.news_id IS 'Unique QuoteMedia news ID (16 digits)';
COMMENT ON COLUMN quotemedia_news.symbol IS 'Stock symbol associated with the news';
COMMENT ON COLUMN quotemedia_news.topics IS 'Array of all topics/symbols mentioned in the article';
COMMENT ON COLUMN quotemedia_news.primary_commodity IS 'Main commodity mentioned in the news';
COMMENT ON COLUMN quotemedia_news.sentiment_score IS 'Sentiment score from -1 (negative) to 1 (positive)';