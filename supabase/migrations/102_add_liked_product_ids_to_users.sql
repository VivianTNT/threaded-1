ALTER TABLE users
ADD COLUMN IF NOT EXISTS liked_product_ids TEXT[] NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_users_liked_product_ids
ON users
USING GIN (liked_product_ids);
