-- Add email column to users table for Penn database
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/owbuzpxttovfssoarzwc/sql

ALTER TABLE users
ADD COLUMN IF NOT EXISTS email TEXT;

-- Add unique constraint on email
ALTER TABLE users
ADD CONSTRAINT users_email_unique UNIQUE (email);

-- Add index for faster email lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Add auth_user_id column to link with Supabase Auth
ALTER TABLE users
ADD COLUMN IF NOT EXISTS auth_user_id UUID;

-- Add index for auth_user_id
CREATE INDEX IF NOT EXISTS idx_users_auth_user_id ON users(auth_user_id);

-- Update trigger for updated_at if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
