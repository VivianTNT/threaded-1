-- Create pdf_highlights table for storing persistent PDF annotations
CREATE TABLE IF NOT EXISTS pdf_highlights (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_url TEXT NOT NULL UNIQUE,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  highlight_data JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_pdf_highlights_project_id ON pdf_highlights(project_id);

-- Add comment
COMMENT ON TABLE pdf_highlights IS 'Stores PDF highlight annotations and extracted key data points';
