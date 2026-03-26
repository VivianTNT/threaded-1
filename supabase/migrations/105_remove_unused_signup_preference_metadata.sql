UPDATE users
SET metadata = COALESCE(metadata, '{}'::jsonb)
  - 'style_preferences'
  - 'budget_range'
  - 'favorite_colors'
  - 'bio'
WHERE COALESCE(metadata, '{}'::jsonb) ?| ARRAY[
  'style_preferences',
  'budget_range',
  'favorite_colors',
  'bio'
];
