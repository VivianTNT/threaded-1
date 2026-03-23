CREATE TABLE IF NOT EXISTS user_recommendation_snapshots (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  snapshot_key TEXT NOT NULL,
  catalog_marker TIMESTAMPTZ,
  mode TEXT NOT NULL,
  engine TEXT NOT NULL,
  recommended_product_ids TEXT[] NOT NULL DEFAULT '{}',
  recommendation_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
  total INTEGER NOT NULL DEFAULT 0,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_recommendation_snapshots_expires_at
ON user_recommendation_snapshots(expires_at);
