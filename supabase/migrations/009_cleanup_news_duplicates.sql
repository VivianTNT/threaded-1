-- Migration: Clean up duplicate and bad quality news entries
-- This migration removes:
-- 1. Duplicate news entries (keeping the oldest one)
-- 2. Entries with image URLs
-- 3. Entries with teaser/image titles
-- 4. Entries with titles that are too short (<15 chars) or too long (>250 chars)
-- 5. Entries with invalid URLs

-- First, let's see what we're working with (commented out for actual migration)
-- SELECT COUNT(*) as total_before FROM news;

-- Remove entries with image URLs
DELETE FROM news
WHERE EXISTS (
  SELECT 1 FROM unnest(urls) AS url
  WHERE url ~* '\.(jpg|jpeg|png|gif|webp|svg|bmp|ico)(\?|$)'
);

-- Remove entries with teaser/image titles
DELETE FROM news
WHERE title ~* '^(image|photo|video|teaser|thumbnail|icon)[\s:]';

-- Remove entries with titles that are too short or too long
DELETE FROM news
WHERE LENGTH(title) < 15 OR LENGTH(title) > 250;

-- Remove entries where URLs don't start with http
DELETE FROM news
WHERE EXISTS (
  SELECT 1 FROM unnest(urls) AS url
  WHERE url !~ '^https?://'
);

-- Remove entries with empty or null titles
DELETE FROM news
WHERE title IS NULL OR TRIM(title) = '';

-- Remove entries with empty URLs array
DELETE FROM news
WHERE urls IS NULL OR array_length(urls, 1) IS NULL OR array_length(urls, 1) = 0;

-- Remove duplicates based on URL (keep oldest entry based on created_at)
DELETE FROM news a
USING news b
WHERE a.id > b.id
  AND a.urls && b.urls;  -- Arrays have overlapping elements

-- Remove duplicates based on exact title match (keep oldest entry)
DELETE FROM news a
USING news b
WHERE a.id > b.id
  AND a.title = b.title;

-- Create an index on urls for better query performance (if not exists)
CREATE INDEX IF NOT EXISTS idx_news_urls ON news USING GIN(urls);

-- Create an index on title for better duplicate detection
CREATE INDEX IF NOT EXISTS idx_news_title ON news(title);

-- Create an index on published_at for sorting
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news(published_at DESC);

-- See results (commented out for actual migration)
-- SELECT COUNT(*) as total_after FROM news;
-- SELECT
--   COUNT(*) as total,
--   COUNT(DISTINCT title) as unique_titles,
--   MIN(LENGTH(title)) as min_title_length,
--   MAX(LENGTH(title)) as max_title_length
-- FROM news;
