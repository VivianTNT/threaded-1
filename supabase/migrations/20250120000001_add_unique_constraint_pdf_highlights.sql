-- Add UNIQUE constraint to document_url in pdf_highlights table
-- This allows upsert operations based on document_url

-- Drop the existing index since we'll replace it with a unique constraint
DROP INDEX IF EXISTS idx_pdf_highlights_document_url;

-- Add unique constraint
ALTER TABLE pdf_highlights
ADD CONSTRAINT pdf_highlights_document_url_key UNIQUE (document_url);
