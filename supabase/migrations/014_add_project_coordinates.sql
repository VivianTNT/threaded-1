-- Migration 014: Add latitude and longitude to projects table
-- This enables exact project locations on maps instead of country-level geocoding

-- Add latitude column
ALTER TABLE projects ADD COLUMN IF NOT EXISTS latitude NUMERIC;
COMMENT ON COLUMN projects.latitude IS 'Latitude of project location in decimal degrees (-90 to 90)';

-- Add longitude column
ALTER TABLE projects ADD COLUMN IF NOT EXISTS longitude NUMERIC;
COMMENT ON COLUMN projects.longitude IS 'Longitude of project location in decimal degrees (-180 to 180)';

-- Create index for geospatial queries (optional but recommended for performance)
CREATE INDEX IF NOT EXISTS idx_projects_coordinates ON projects(latitude, longitude) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Add constraint to validate latitude range
ALTER TABLE projects ADD CONSTRAINT check_latitude_range CHECK (latitude IS NULL OR (latitude >= -90 AND latitude <= 90));

-- Add constraint to validate longitude range
ALTER TABLE projects ADD CONSTRAINT check_longitude_range CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180));
