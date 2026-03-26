ALTER TABLE user_recommendation_snapshots
ADD COLUMN IF NOT EXISTS request_strategy TEXT NOT NULL DEFAULT 'hybrid';

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'user_recommendation_snapshots_pkey'
      AND conrelid = 'user_recommendation_snapshots'::regclass
  ) THEN
    ALTER TABLE user_recommendation_snapshots
    DROP CONSTRAINT user_recommendation_snapshots_pkey;
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'user_recommendation_snapshots_pkey'
      AND conrelid = 'user_recommendation_snapshots'::regclass
  ) THEN
    ALTER TABLE user_recommendation_snapshots
    ADD CONSTRAINT user_recommendation_snapshots_pkey PRIMARY KEY (user_id, request_strategy);
  END IF;
END $$;
