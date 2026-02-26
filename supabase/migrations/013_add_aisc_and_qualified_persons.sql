-- Migration: Add AISC and Qualified Persons to Projects and Companies
-- Date: 2025-01-09
-- Purpose: Support AISC metric display and qualified persons information

-- Add AISC column to projects table
ALTER TABLE projects ADD COLUMN IF NOT EXISTS aisc NUMERIC;
COMMENT ON COLUMN projects.aisc IS 'All-In Sustaining Cost (AISC) in USD per unit (oz, lb, tonne)';

-- Add qualified persons JSONB column to projects table
ALTER TABLE projects ADD COLUMN IF NOT EXISTS qualified_persons JSONB DEFAULT '[]'::jsonb;
COMMENT ON COLUMN projects.qualified_persons IS 'Array of qualified persons who prepared technical reports: [{name, credentials, company}]';

-- Create index for AISC filtering on projects
CREATE INDEX IF NOT EXISTS idx_projects_aisc ON projects(aisc) WHERE aisc IS NOT NULL;

-- NOTE: We do NOT add AISC to companies table because:
-- 1. Companies have multiple projects with different commodities ($/oz vs $/lb vs $/tonne)
-- 2. AISC varies significantly by project stage, location, and mining method
-- 3. A company-wide AISC average is not a meaningful metric
-- 4. Industry practice is to report AISC per mine/project, not per company

-- Create GIN index for qualified_persons JSONB searching
CREATE INDEX IF NOT EXISTS idx_projects_qualified_persons ON projects USING GIN (qualified_persons);
