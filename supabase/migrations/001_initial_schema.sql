-- ============================================================================
-- INITIAL DATABASE SCHEMA FOR LITHOS
-- ============================================================================
-- This migration sets up the core tables needed for the application:
-- - companies: Mining companies
-- - projects: Mining projects
-- - news: News and announcements
-- - usr: User accounts
-- - chat_history: Chat conversation history
-- ============================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

DO $$ BEGIN
    CREATE TYPE project_stage AS ENUM (
        'Exploration',
        'Resource Definition', 
        'Pre-Feasibility',
        'Feasibility',
        'Construction',
        'Production',
        'Care & Maintenance',
        'Closed'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE commodity_type AS ENUM (
        'Lithium', 'Copper', 'Nickel', 'Cobalt', 'Rare Earths',
        'Gold', 'Silver', 'Uranium', 'Graphite', 'Other'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE risk_level AS ENUM ('Low', 'Medium', 'High', 'Very High');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE esg_grade AS ENUM ('A', 'B', 'C', 'D', 'F');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- ============================================================================
-- COMPANIES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    ticker VARCHAR(20),
    exchange VARCHAR(50),
    headquarters_location VARCHAR(255),
    country VARCHAR(100),
    website_url TEXT,
    market_cap_usd DECIMAL(15,2),
    
    -- Metrics
    total_projects INTEGER DEFAULT 0,
    production_projects INTEGER DEFAULT 0,
    development_projects INTEGER DEFAULT 0,
    exploration_projects INTEGER DEFAULT 0,
    total_npv_usd DECIMAL(15,2),
    avg_irr_percent DECIMAL(5,2),
    total_capex_usd DECIMAL(15,2),
    
    -- Commodities and locations
    primary_commodities TEXT[],
    operating_countries TEXT[],
    
    -- Risk and ESG
    risk_level VARCHAR(20),
    esg_score VARCHAR(2),
    
    -- Metadata
    description TEXT,
    logo_url TEXT,
    founded_year INTEGER,
    employees_count INTEGER,
    
    -- Watchlist
    watchlist BOOLEAN DEFAULT FALSE,
    watchlisted_at TIMESTAMP WITH TIME ZONE,
    watchlisted_by UUID[],
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name);
CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
CREATE INDEX IF NOT EXISTS idx_companies_country ON companies(country);
CREATE INDEX IF NOT EXISTS idx_companies_watchlist ON companies(watchlist) WHERE watchlist = TRUE;

-- ============================================================================
-- PROJECTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS projects (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id INTEGER REFERENCES companies(id) ON DELETE SET NULL,
    
    -- Basic Information
    project_name VARCHAR(255) NOT NULL,
    company_name VARCHAR(255),
    project_description TEXT,
    
    -- Location
    jurisdiction VARCHAR(255),
    country VARCHAR(100),
    location GEOGRAPHY(POINT, 4326),
    
    -- Stage and Timeline
    stage project_stage DEFAULT 'Exploration',
    mine_life_years DECIMAL(5,2),
    production_start_date DATE,
    
    -- Financial Metrics
    post_tax_npv_usd_m DECIMAL(12,2),
    irr_percent DECIMAL(5,2),
    payback_years DECIMAL(5,2),
    capex_usd_m DECIMAL(12,2),
    opex_usd_per_tonne DECIMAL(10,2),
    
    -- Resources
    primary_commodity commodity_type,
    secondary_commodities commodity_type[],
    annual_production_tonnes DECIMAL(15,2),
    total_resource_tonnes DECIMAL(15,2),
    resource_grade DECIMAL(10,4),
    
    -- Risk
    jurisdiction_risk risk_level DEFAULT 'Medium',
    esg_score esg_grade DEFAULT 'C',
    
    -- Watchlist
    watchlist BOOLEAN DEFAULT FALSE,
    watchlisted_at TIMESTAMP WITH TIME ZONE,
    watchlisted_by UUID[],
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(project_name, company_name)
);

CREATE INDEX IF NOT EXISTS idx_projects_company_id ON projects(company_id);
CREATE INDEX IF NOT EXISTS idx_projects_stage ON projects(stage);
CREATE INDEX IF NOT EXISTS idx_projects_commodity ON projects(primary_commodity);
CREATE INDEX IF NOT EXISTS idx_projects_country ON projects(country);
CREATE INDEX IF NOT EXISTS idx_projects_watchlist ON projects(watchlist) WHERE watchlist = TRUE;

-- ============================================================================
-- NEWS TABLE (UNIFIED_NEWS)
-- ============================================================================

CREATE TABLE IF NOT EXISTS unified_news (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Core fields
    headline TEXT NOT NULL,
    summary TEXT,
    url TEXT UNIQUE NOT NULL,
    published_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Source
    source_name TEXT NOT NULL,
    source_type TEXT DEFAULT 'news',
    
    -- Company association
    company_id INTEGER REFERENCES companies(id) ON DELETE SET NULL,
    company_name TEXT,
    symbol TEXT,
    exchange TEXT,
    
    -- Categorization
    primary_commodity TEXT,
    commodities TEXT[],
    topics TEXT[],
    countries TEXT[],
    project_names TEXT[],
    news_category TEXT,
    
    -- Mining-specific flags
    is_mining_related BOOLEAN DEFAULT false,
    is_project_related BOOLEAN DEFAULT false,
    is_exploration_news BOOLEAN DEFAULT false,
    is_production_news BOOLEAN DEFAULT false,
    
    -- Content mentions
    mentions_financials BOOLEAN DEFAULT false,
    mentions_technical_report BOOLEAN DEFAULT false,
    mentions_resource_estimate BOOLEAN DEFAULT false,
    mentions_feasibility_study BOOLEAN DEFAULT false,
    mentions_environmental BOOLEAN DEFAULT false,
    mentions_permits BOOLEAN DEFAULT false,
    mentions_acquisition BOOLEAN DEFAULT false,
    
    -- Sentiment and scoring
    sentiment_score NUMERIC(3,2),
    relevance_score INTEGER,
    importance_level TEXT,
    
    -- Status
    is_featured BOOLEAN DEFAULT false,
    is_archived BOOLEAN DEFAULT false,
    
    -- Watchlist
    watchlist BOOLEAN DEFAULT FALSE,
    watchlisted_at TIMESTAMP WITH TIME ZONE,
    watchlisted_by UUID[],
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add missing columns to existing unified_news table if they don't exist
DO $$ 
BEGIN
    -- Add exchange column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='exchange') THEN
        ALTER TABLE unified_news ADD COLUMN exchange TEXT;
    END IF;
    
    -- Add project_names column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='project_names') THEN
        ALTER TABLE unified_news ADD COLUMN project_names TEXT[];
    END IF;
    
    -- Add news_category column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='news_category') THEN
        ALTER TABLE unified_news ADD COLUMN news_category TEXT;
    END IF;
    
    -- Add mining-specific flags
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='is_mining_related') THEN
        ALTER TABLE unified_news ADD COLUMN is_mining_related BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='is_project_related') THEN
        ALTER TABLE unified_news ADD COLUMN is_project_related BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='is_exploration_news') THEN
        ALTER TABLE unified_news ADD COLUMN is_exploration_news BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='is_production_news') THEN
        ALTER TABLE unified_news ADD COLUMN is_production_news BOOLEAN DEFAULT false;
    END IF;
    
    -- Add content mention flags
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='mentions_financials') THEN
        ALTER TABLE unified_news ADD COLUMN mentions_financials BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='mentions_technical_report') THEN
        ALTER TABLE unified_news ADD COLUMN mentions_technical_report BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='mentions_resource_estimate') THEN
        ALTER TABLE unified_news ADD COLUMN mentions_resource_estimate BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='mentions_feasibility_study') THEN
        ALTER TABLE unified_news ADD COLUMN mentions_feasibility_study BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='mentions_environmental') THEN
        ALTER TABLE unified_news ADD COLUMN mentions_environmental BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='mentions_permits') THEN
        ALTER TABLE unified_news ADD COLUMN mentions_permits BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='mentions_acquisition') THEN
        ALTER TABLE unified_news ADD COLUMN mentions_acquisition BOOLEAN DEFAULT false;
    END IF;
    
    -- Add importance_level column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='importance_level') THEN
        ALTER TABLE unified_news ADD COLUMN importance_level TEXT;
    END IF;
    
    -- Add status flags
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='is_featured') THEN
        ALTER TABLE unified_news ADD COLUMN is_featured BOOLEAN DEFAULT false;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='unified_news' AND column_name='is_archived') THEN
        ALTER TABLE unified_news ADD COLUMN is_archived BOOLEAN DEFAULT false;
    END IF;
END $$;

-- Keep the old news table as an alias/view for backward compatibility
DROP VIEW IF EXISTS news;
CREATE VIEW news AS SELECT * FROM unified_news;

CREATE INDEX IF NOT EXISTS idx_unified_news_published ON unified_news(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_unified_news_company_id ON unified_news(company_id);
CREATE INDEX IF NOT EXISTS idx_unified_news_source ON unified_news(source_name);
CREATE INDEX IF NOT EXISTS idx_unified_news_watchlist ON unified_news(watchlist) WHERE watchlist = TRUE;
CREATE INDEX IF NOT EXISTS idx_unified_news_archived ON unified_news(is_archived);
CREATE INDEX IF NOT EXISTS idx_unified_news_category ON unified_news(news_category);
CREATE INDEX IF NOT EXISTS idx_unified_news_commodity ON unified_news(primary_commodity);

-- ============================================================================
-- USR TABLE (Users)
-- ============================================================================

CREATE TABLE IF NOT EXISTS usr (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    company_id INTEGER REFERENCES companies(id) ON DELETE SET NULL,
    role VARCHAR(50) DEFAULT 'member',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usr_email ON usr(email);
CREATE INDEX IF NOT EXISTS idx_usr_company_id ON usr(company_id);

-- ============================================================================
-- CHAT HISTORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    message TEXT NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created ON chat_history(created_at DESC);

-- ============================================================================
-- JUNCTION TABLES
-- ============================================================================

-- News to Projects relationship
CREATE TABLE IF NOT EXISTS news_projects (
    id SERIAL PRIMARY KEY,
    news_id UUID NOT NULL REFERENCES unified_news(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    impact_type VARCHAR(50),
    impact_score INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(news_id, project_id)
);

CREATE INDEX IF NOT EXISTS idx_news_projects_news ON news_projects(news_id);
CREATE INDEX IF NOT EXISTS idx_news_projects_project ON news_projects(project_id);

-- News to Companies relationship
CREATE TABLE IF NOT EXISTS news_companies (
    id SERIAL PRIMARY KEY,
    news_id UUID NOT NULL REFERENCES unified_news(id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    impact_type VARCHAR(50),
    impact_score INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(news_id, company_id)
);

CREATE INDEX IF NOT EXISTS idx_news_companies_news ON news_companies(news_id);
CREATE INDEX IF NOT EXISTS idx_news_companies_company ON news_companies(company_id);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
CREATE TRIGGER update_companies_updated_at
BEFORE UPDATE ON companies
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
CREATE TRIGGER update_projects_updated_at
BEFORE UPDATE ON projects
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_unified_news_updated_at ON unified_news;
CREATE TRIGGER update_unified_news_updated_at
BEFORE UPDATE ON unified_news
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_usr_updated_at ON usr;
CREATE TRIGGER update_usr_updated_at
BEFORE UPDATE ON usr
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

-- Grant permissions for authenticated users
GRANT SELECT, INSERT, UPDATE ON companies TO authenticated;
GRANT SELECT, INSERT, UPDATE ON projects TO authenticated;
GRANT SELECT, INSERT, UPDATE ON unified_news TO authenticated;
GRANT SELECT, INSERT, UPDATE ON usr TO authenticated;
GRANT SELECT, INSERT ON chat_history TO authenticated;
GRANT SELECT, INSERT, DELETE ON news_projects TO authenticated;
GRANT SELECT, INSERT, DELETE ON news_companies TO authenticated;

-- Grant all permissions to service role
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;

-- ============================================================================
-- Enable Row Level Security
-- ============================================================================

ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE unified_news ENABLE ROW LEVEL SECURITY;
ALTER TABLE usr ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

-- Create basic RLS policies (adjust as needed)
CREATE POLICY "Enable read access for all users" ON companies FOR SELECT USING (true);
CREATE POLICY "Enable read access for all users" ON projects FOR SELECT USING (true);
CREATE POLICY "Enable read access for all users" ON unified_news FOR SELECT USING (true);
CREATE POLICY "Users can view own profile" ON usr FOR SELECT USING (auth.uid()::text = email);
CREATE POLICY "Users can view own chat history" ON chat_history FOR SELECT USING (auth.uid() = user_id);

-- ============================================================================
-- FINAL MESSAGE
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Initial database schema created successfully!';
    RAISE NOTICE 'Tables created: companies, projects, unified_news, usr, chat_history';
    RAISE NOTICE 'Junction tables: news_projects, news_companies';
    RAISE NOTICE 'Views created: news (alias for unified_news)';
END $$;
