-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  location TEXT, -- Country or region
  stage TEXT, -- e.g., 'Exploration', 'Development', 'Production'
  commodities TEXT[], -- Array of commodities: ['Lithium', 'Cobalt', 'Nickel']
  resource_estimate TEXT, -- Summary of resource estimate
  reserve_estimate TEXT, -- Summary of reserve estimate
  ownership_percentage NUMERIC, -- Company's ownership stake (0-100)
  status TEXT, -- e.g., 'Active', 'On Hold', 'Closed'
  description TEXT,
  urls TEXT[], -- Array of URLs from multiple sources
  watchlist BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_projects_company_id ON projects(company_id);
CREATE INDEX IF NOT EXISTS idx_projects_commodities ON projects USING GIN(commodities);
CREATE INDEX IF NOT EXISTS idx_projects_stage ON projects(stage);
CREATE INDEX IF NOT EXISTS idx_projects_watchlist ON projects(watchlist) WHERE watchlist = TRUE;

-- Create updated_at trigger
CREATE TRIGGER update_projects_updated_at
  BEFORE UPDATE ON projects
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
