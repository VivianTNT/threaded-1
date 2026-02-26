-- Create table for storing FactSet document metadata
CREATE TABLE IF NOT EXISTS factset_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id TEXT UNIQUE NOT NULL,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

  -- Document metadata
  headline TEXT NOT NULL,
  filing_date TIMESTAMP WITH TIME ZONE NOT NULL,
  form_type TEXT,

  -- Storage info
  storage_path TEXT NOT NULL,
  public_url TEXT NOT NULL,
  file_size INTEGER,

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_factset_documents_project_id ON factset_documents(project_id);
CREATE INDEX IF NOT EXISTS idx_factset_documents_filing_date ON factset_documents(filing_date);
CREATE INDEX IF NOT EXISTS idx_factset_documents_form_type ON factset_documents(form_type);

-- Add document_urls column to projects table if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'projects' AND column_name = 'document_urls'
  ) THEN
    ALTER TABLE projects ADD COLUMN document_urls TEXT[];
  END IF;
END $$;

-- Add factset_document_count column to projects table
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'projects' AND column_name = 'factset_document_count'
  ) THEN
    ALTER TABLE projects ADD COLUMN factset_document_count INTEGER DEFAULT 0;
  END IF;
END $$;

-- Enable Row Level Security
ALTER TABLE factset_documents ENABLE ROW LEVEL SECURITY;

-- Create policy to allow read access for authenticated users
CREATE POLICY "Allow read access for authenticated users"
  ON factset_documents FOR SELECT
  TO authenticated
  USING (true);

-- Create policy to allow insert/update for service role
CREATE POLICY "Allow insert for service role"
  ON factset_documents FOR INSERT
  TO service_role
  WITH CHECK (true);

CREATE POLICY "Allow update for service role"
  ON factset_documents FOR UPDATE
  TO service_role
  USING (true);
