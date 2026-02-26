-- Add columns to support user-uploaded private documents
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id),
ADD COLUMN IF NOT EXISTS is_private BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS document_storage_path TEXT;

-- Create index for faster private project queries
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id) WHERE is_private = TRUE;
CREATE INDEX IF NOT EXISTS idx_projects_is_private ON projects(is_private);

-- Enable RLS on projects table
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Policy: Users can see all public projects
CREATE POLICY "Public projects are viewable by everyone"
ON projects FOR SELECT
USING (is_private = FALSE OR is_private IS NULL);

-- Policy: Users can see their own private projects
CREATE POLICY "Users can view their own private projects"
ON projects FOR SELECT
USING (is_private = TRUE AND user_id = auth.uid());

-- Policy: Authenticated users can insert their own private projects
CREATE POLICY "Users can insert their own private projects"
ON projects FOR INSERT
WITH CHECK (is_private = TRUE AND user_id = auth.uid());

-- Policy: Users can update their own private projects
CREATE POLICY "Users can update their own private projects"
ON projects FOR UPDATE
USING (is_private = TRUE AND user_id = auth.uid());

-- Policy: Users can delete their own private projects
CREATE POLICY "Users can delete their own private projects"
ON projects FOR DELETE
USING (is_private = TRUE AND user_id = auth.uid());

-- Create storage bucket for user documents (if not exists)
INSERT INTO storage.buckets (id, name, public)
VALUES ('user-documents', 'user-documents', false)
ON CONFLICT (id) DO NOTHING;

-- Storage RLS policies
CREATE POLICY "Users can upload their own documents"
ON storage.objects FOR INSERT
WITH CHECK (
  bucket_id = 'user-documents' AND
  auth.role() = 'authenticated' AND
  (storage.foldername(name))[1] = auth.uid()::text
);

CREATE POLICY "Users can view their own documents"
ON storage.objects FOR SELECT
USING (
  bucket_id = 'user-documents' AND
  auth.role() = 'authenticated' AND
  (storage.foldername(name))[1] = auth.uid()::text
);

CREATE POLICY "Users can delete their own documents"
ON storage.objects FOR DELETE
USING (
  bucket_id = 'user-documents' AND
  auth.role() = 'authenticated' AND
  (storage.foldername(name))[1] = auth.uid()::text
);
