-- Create companies table
CREATE TABLE IF NOT EXISTS companies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  ticker TEXT,
  exchange TEXT, -- e.g., 'ASX', 'TSX', 'NYSE', 'LSE'
  country TEXT,
  website TEXT,
  description TEXT,
  market_cap NUMERIC,
  urls TEXT[], -- Array of URLs from multiple sources
  watchlist BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index on ticker and exchange for fast lookups
CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
CREATE INDEX IF NOT EXISTS idx_companies_exchange ON companies(exchange);
CREATE INDEX IF NOT EXISTS idx_companies_watchlist ON companies(watchlist) WHERE watchlist = TRUE;

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_companies_updated_at
  BEFORE UPDATE ON companies
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
