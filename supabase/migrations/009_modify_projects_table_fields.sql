-- Remove hard-to-populate fields that consistently show "Not specified" or "N/A"
ALTER TABLE projects DROP COLUMN IF EXISTS resource_estimate;
ALTER TABLE projects DROP COLUMN IF EXISTS reserve_estimate;
ALTER TABLE projects DROP COLUMN IF EXISTS ownership_percentage;

-- Add easily-scrapable, useful fields that are commonly published
ALTER TABLE projects ADD COLUMN IF NOT EXISTS operator TEXT; -- Operating company name
ALTER TABLE projects ADD COLUMN IF NOT EXISTS production_rate TEXT; -- e.g., "100,000 tonnes per year", "500k oz Au/year"
ALTER TABLE projects ADD COLUMN IF NOT EXISTS mine_life TEXT; -- e.g., "20 years", "2025-2045"
ALTER TABLE projects ADD COLUMN IF NOT EXISTS capex TEXT; -- Capital expenditure, e.g., "$500M", "A$1.2B"
ALTER TABLE projects ADD COLUMN IF NOT EXISTS first_production DATE; -- Expected or actual first production date
ALTER TABLE projects ADD COLUMN IF NOT EXISTS project_type TEXT; -- e.g., "Open Pit", "Underground", "Heap Leach"

-- Create indexes on new fields for better query performance
CREATE INDEX IF NOT EXISTS idx_projects_operator ON projects(operator);
CREATE INDEX IF NOT EXISTS idx_projects_project_type ON projects(project_type);
