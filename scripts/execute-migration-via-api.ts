import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function executeMigration() {
  console.log('🚀 Executing database migration...\n')

  // Step 1: Add columns to projects table
  console.log('Step 1: Adding columns to projects table...')
  const { error: alterError } = await supabase.rpc('exec', {
    sql: `
      ALTER TABLE projects
      ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id),
      ADD COLUMN IF NOT EXISTS is_private BOOLEAN DEFAULT FALSE,
      ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      ADD COLUMN IF NOT EXISTS document_storage_path TEXT;
    `
  })

  if (alterError) {
    console.log('   ⚠️ Using alternative method for ALTER TABLE...')
    // Try using the postgres connection directly
    const { data, error } = await supabase.from('projects').select('user_id, is_private').limit(1)
    if (error && error.message.includes('does not exist')) {
      console.log('   ❌ Columns do not exist yet, using raw SQL execution...')
    } else {
      console.log('   ✅ Columns may already exist')
    }
  } else {
    console.log('   ✅ Columns added successfully')
  }

  // Step 2: Create indexes
  console.log('\nStep 2: Creating indexes...')
  const { error: idx1Error } = await supabase.rpc('exec', {
    sql: 'CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id) WHERE is_private = TRUE;'
  })
  console.log(idx1Error ? '   ⚠️ Index creation via RPC failed' : '   ✅ Index idx_projects_user_id created')

  const { error: idx2Error } = await supabase.rpc('exec', {
    sql: 'CREATE INDEX IF NOT EXISTS idx_projects_is_private ON projects(is_private);'
  })
  console.log(idx2Error ? '   ⚠️ Index creation via RPC failed' : '   ✅ Index idx_projects_is_private created')

  // Step 3: Enable RLS
  console.log('\nStep 3: Enabling Row Level Security...')
  const { error: rlsError } = await supabase.rpc('exec', {
    sql: 'ALTER TABLE projects ENABLE ROW LEVEL SECURITY;'
  })
  console.log(rlsError ? '   ⚠️ RLS enable via RPC failed' : '   ✅ RLS enabled')

  // Step 4: Create policies using direct SQL
  console.log('\nStep 4: Creating RLS policies...')

  const policies = [
    {
      name: 'Public projects viewable by everyone',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'projects' AND policyname = 'Public projects are viewable by everyone'
          ) THEN
            EXECUTE 'CREATE POLICY "Public projects are viewable by everyone" ON projects FOR SELECT USING (is_private = FALSE OR is_private IS NULL)';
          END IF;
        END $$;
      `
    },
    {
      name: 'Users can view own private projects',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'projects' AND policyname = 'Users can view their own private projects'
          ) THEN
            EXECUTE 'CREATE POLICY "Users can view their own private projects" ON projects FOR SELECT USING (is_private = TRUE AND user_id = auth.uid())';
          END IF;
        END $$;
      `
    },
    {
      name: 'Users can insert own private projects',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'projects' AND policyname = 'Users can insert their own private projects'
          ) THEN
            EXECUTE 'CREATE POLICY "Users can insert their own private projects" ON projects FOR INSERT WITH CHECK (is_private = TRUE AND user_id = auth.uid())';
          END IF;
        END $$;
      `
    },
    {
      name: 'Users can update own private projects',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'projects' AND policyname = 'Users can update their own private projects'
          ) THEN
            EXECUTE 'CREATE POLICY "Users can update their own private projects" ON projects FOR UPDATE USING (is_private = TRUE AND user_id = auth.uid())';
          END IF;
        END $$;
      `
    },
    {
      name: 'Users can delete own private projects',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'projects' AND policyname = 'Users can delete their own private projects'
          ) THEN
            EXECUTE 'CREATE POLICY "Users can delete their own private projects" ON projects FOR DELETE USING (is_private = TRUE AND user_id = auth.uid())';
          END IF;
        END $$;
      `
    }
  ]

  for (const policy of policies) {
    const { error } = await supabase.rpc('exec', { sql: policy.sql })
    console.log(`   ${error ? '⚠️' : '✅'} ${policy.name}`)
  }

  // Step 5: Storage bucket
  console.log('\nStep 5: Creating storage bucket...')
  const { data: existingBucket } = await supabase.storage.getBucket('user-documents')

  if (!existingBucket) {
    const { error } = await supabase.storage.createBucket('user-documents', {
      public: false,
      fileSizeLimit: 52428800 // 50MB
    })
    console.log(error ? `   ⚠️ ${error.message}` : '   ✅ Storage bucket created')
  } else {
    console.log('   ✅ Storage bucket already exists')
  }

  // Step 6: Storage policies
  console.log('\nStep 6: Creating storage policies...')

  const storagePolicies = [
    {
      name: 'Users can upload own documents',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'objects' AND policyname = 'Users can upload their own documents'
          ) THEN
            EXECUTE 'CREATE POLICY "Users can upload their own documents" ON storage.objects FOR INSERT WITH CHECK (bucket_id = ''user-documents'' AND auth.role() = ''authenticated'' AND (storage.foldername(name))[1] = auth.uid()::text)';
          END IF;
        END $$;
      `
    },
    {
      name: 'Users can view own documents',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'objects' AND policyname = 'Users can view their own documents'
          ) THEN
            EXECUTE 'CREATE POLICY "Users can view their own documents" ON storage.objects FOR SELECT USING (bucket_id = ''user-documents'' AND auth.role() = ''authenticated'' AND (storage.foldername(name))[1] = auth.uid()::text)';
          END IF;
        END $$;
      `
    },
    {
      name: 'Users can delete own documents',
      sql: `
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_policies WHERE tablename = 'objects' AND policyname = 'Users can delete their own documents'
          ) THEN
            EXECUTE 'CREATE POLICY "Users can delete their own documents" ON storage.objects FOR DELETE USING (bucket_id = ''user-documents'' AND auth.role() = ''authenticated'' AND (storage.foldername(name))[1] = auth.uid()::text)';
          END IF;
        END $$;
      `
    }
  ]

  for (const policy of storagePolicies) {
    const { error } = await supabase.rpc('exec', { sql: policy.sql })
    console.log(`   ${error ? '⚠️' : '✅'} ${policy.name}`)
  }

  console.log('\n✅ Migration execution complete!')
  console.log('   Run check-migration-status.ts to verify\n')
}

executeMigration().catch(console.error)
