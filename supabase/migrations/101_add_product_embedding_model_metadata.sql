-- Add metadata columns for recommendation embedding provenance/versioning.
ALTER TABLE IF EXISTS product_embeddings
ADD COLUMN IF NOT EXISTS model_name TEXT,
ADD COLUMN IF NOT EXISTS model_version TEXT,
ADD COLUMN IF NOT EXISTS embedding_source TEXT DEFAULT 'image_content',
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_product_embeddings_model_name
ON product_embeddings(model_name);

CREATE INDEX IF NOT EXISTS idx_product_embeddings_model_version
ON product_embeddings(model_version);
