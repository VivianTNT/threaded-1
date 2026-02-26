-- Add financial metrics columns to projects table
-- Migration 010: Add NPV, IRR, and CAPEX columns

-- Add NPV (Net Present Value) column in millions USD
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS npv_usd_millions NUMERIC;

-- Add IRR (Internal Rate of Return) as percentage
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS irr_percentage NUMERIC;

-- Add CAPEX (Capital Expenditure) in millions USD
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS capex_usd_millions NUMERIC;

-- Add optional metadata for financial metrics
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS financial_metrics_updated_at TIMESTAMPTZ;

-- Add optional discount rate used for NPV calculation
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS discount_rate_percentage NUMERIC;

-- Create index for financial filtering
CREATE INDEX IF NOT EXISTS idx_projects_npv ON projects(npv_usd_millions) WHERE npv_usd_millions IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_projects_irr ON projects(irr_percentage) WHERE irr_percentage IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_projects_capex ON projects(capex_usd_millions) WHERE capex_usd_millions IS NOT NULL;

-- Add comment for documentation
COMMENT ON COLUMN projects.npv_usd_millions IS 'Net Present Value in millions USD';
COMMENT ON COLUMN projects.irr_percentage IS 'Internal Rate of Return as percentage (e.g., 15.5 for 15.5%)';
COMMENT ON COLUMN projects.capex_usd_millions IS 'Capital Expenditure in millions USD';
COMMENT ON COLUMN projects.discount_rate_percentage IS 'Discount rate used for NPV calculation (e.g., 8.0 for 8%)';
COMMENT ON COLUMN projects.financial_metrics_updated_at IS 'Timestamp when financial metrics were last updated';
