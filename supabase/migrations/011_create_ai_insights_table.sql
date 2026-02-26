-- Create AI insights table for caching risk analysis
-- Migration 011: AI-generated risk insights with caching

CREATE TABLE IF NOT EXISTS ai_insights (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

  -- Risk Analysis Scores (0-10 scale)
  geography_risk_score NUMERIC CHECK (geography_risk_score >= 0 AND geography_risk_score <= 10),
  geography_risk_analysis TEXT,

  legal_risk_score NUMERIC CHECK (legal_risk_score >= 0 AND legal_risk_score <= 10),
  legal_risk_analysis TEXT,

  commodity_risk_score NUMERIC CHECK (commodity_risk_score >= 0 AND commodity_risk_score <= 10),
  commodity_risk_analysis TEXT,

  team_risk_score NUMERIC CHECK (team_risk_score >= 0 AND team_risk_score <= 10),
  team_risk_analysis TEXT,

  -- Overall risk assessment
  overall_risk_score NUMERIC CHECK (overall_risk_score >= 0 AND overall_risk_score <= 10),
  risk_summary TEXT,
  key_opportunities TEXT[],
  key_threats TEXT[],

  -- Investment recommendation
  investment_recommendation TEXT, -- 'Strong Buy', 'Buy', 'Hold', 'Pass'
  recommendation_rationale TEXT,

  -- Metadata
  generated_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days'), -- Cache for 7 days
  generation_time_ms INTEGER, -- Track generation performance
  model_used TEXT DEFAULT 'gpt-4o-mini',
  web_search_used BOOLEAN DEFAULT FALSE,

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_ai_insights_project_id ON ai_insights(project_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_expires_at ON ai_insights(expires_at);
CREATE INDEX IF NOT EXISTS idx_ai_insights_overall_risk ON ai_insights(overall_risk_score);
CREATE INDEX IF NOT EXISTS idx_ai_insights_recommendation ON ai_insights(investment_recommendation);

-- Create updated_at trigger
CREATE TRIGGER update_ai_insights_updated_at
  BEFORE UPDATE ON ai_insights
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Add unique constraint: one active insight per project
CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_insights_unique_active
  ON ai_insights(project_id)
  WHERE expires_at > NOW();

-- Comments for documentation
COMMENT ON TABLE ai_insights IS 'AI-generated risk analysis and insights for mining projects';
COMMENT ON COLUMN ai_insights.geography_risk_score IS 'Geographic/political risk score (0=lowest risk, 10=highest risk)';
COMMENT ON COLUMN ai_insights.legal_risk_score IS 'Legal/regulatory risk score (0=lowest risk, 10=highest risk)';
COMMENT ON COLUMN ai_insights.commodity_risk_score IS 'Commodity market risk score (0=lowest risk, 10=highest risk)';
COMMENT ON COLUMN ai_insights.team_risk_score IS 'Management team risk score (0=lowest risk, 10=highest risk)';
COMMENT ON COLUMN ai_insights.overall_risk_score IS 'Weighted average of all risk scores';
COMMENT ON COLUMN ai_insights.expires_at IS 'When this cached insight expires and should be regenerated';
COMMENT ON COLUMN ai_insights.generation_time_ms IS 'Time taken to generate this insight in milliseconds';
