-- ========================================
-- User-Specific Watchlist Migration
-- ========================================
-- This migration creates per-user watchlist tables to replace the global
-- watchlist boolean columns. This ensures each user has their own isolated
-- watchlist that doesn't interfere with other users.
--
-- IMPORTANT: This is a breaking change. After running this migration,
-- the old watchlist boolean columns will be deprecated in favor of
-- junction tables.
-- ========================================

-- Create user_project_watchlist junction table
CREATE TABLE IF NOT EXISTS user_project_watchlist (
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, project_id)
);

-- Create user_company_watchlist junction table
CREATE TABLE IF NOT EXISTS user_company_watchlist (
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, company_id)
);

-- Create user_news_watchlist junction table
CREATE TABLE IF NOT EXISTS user_news_watchlist (
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  news_id UUID NOT NULL REFERENCES news(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, news_id)
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_user_project_watchlist_user ON user_project_watchlist(user_id);
CREATE INDEX IF NOT EXISTS idx_user_project_watchlist_project ON user_project_watchlist(project_id);

CREATE INDEX IF NOT EXISTS idx_user_company_watchlist_user ON user_company_watchlist(user_id);
CREATE INDEX IF NOT EXISTS idx_user_company_watchlist_company ON user_company_watchlist(company_id);

CREATE INDEX IF NOT EXISTS idx_user_news_watchlist_user ON user_news_watchlist(user_id);
CREATE INDEX IF NOT EXISTS idx_user_news_watchlist_news ON user_news_watchlist(news_id);

-- Enable RLS on watchlist tables
ALTER TABLE user_project_watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_company_watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_news_watchlist ENABLE ROW LEVEL SECURITY;

-- RLS Policies for user_project_watchlist
CREATE POLICY "Users can view their own project watchlist"
ON user_project_watchlist FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can add to their own project watchlist"
ON user_project_watchlist FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can remove from their own project watchlist"
ON user_project_watchlist FOR DELETE
USING (auth.uid() = user_id);

-- RLS Policies for user_company_watchlist
CREATE POLICY "Users can view their own company watchlist"
ON user_company_watchlist FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can add to their own company watchlist"
ON user_company_watchlist FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can remove from their own company watchlist"
ON user_company_watchlist FOR DELETE
USING (auth.uid() = user_id);

-- RLS Policies for user_news_watchlist
CREATE POLICY "Users can view their own news watchlist"
ON user_news_watchlist FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can add to their own news watchlist"
ON user_news_watchlist FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can remove from their own news watchlist"
ON user_news_watchlist FOR DELETE
USING (auth.uid() = user_id);

-- Add comment to old watchlist columns to indicate they are deprecated
COMMENT ON COLUMN projects.watchlist IS 'DEPRECATED: Use user_project_watchlist table instead. This column is kept for backward compatibility but should not be used for new features.';
COMMENT ON COLUMN companies.watchlist IS 'DEPRECATED: Use user_company_watchlist table instead. This column is kept for backward compatibility but should not be used for new features.';
COMMENT ON COLUMN news.watchlist IS 'DEPRECATED: Use user_news_watchlist table instead. This column is kept for backward compatibility but should not be used for new features.';
