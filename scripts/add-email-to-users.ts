import { createClient } from '@supabase/supabase-js';

const projectUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const secretKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!projectUrl || !secretKey) {
  throw new Error('Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
}

const supabase = createClient(projectUrl, secretKey);

async function addEmailColumn() {
  console.log('🔧 Adding email column to users table...\n');

  try {
    // Use raw SQL to add the email column
    const { data, error } = await supabase.rpc('exec_sql', {
      query: `
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS email TEXT UNIQUE;

        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
      `
    });

    if (error) {
      console.log('❌ RPC method not available. Trying alternative approach...\n');

      // Alternative: try to insert a test record to see if column exists
      const testEmail = `test-${Date.now()}@example.com`;
      const { error: insertError } = await supabase
        .from('users')
        .insert({
          handle: 'test_user',
          email: testEmail
        });

      if (insertError) {
        if (insertError.message.includes('column "email" of relation "users" does not exist')) {
          console.log('⚠️  Email column does not exist.');
          console.log('❌ Unable to add column via API. Manual SQL migration required.\n');
          console.log('Please run this SQL in Supabase SQL Editor:');
          console.log('─'.repeat(60));
          console.log(`
ALTER TABLE users
ADD COLUMN IF NOT EXISTS email TEXT UNIQUE;

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
          `);
          console.log('─'.repeat(60));
          return;
        } else {
          // Column might already exist
          console.log('✅ Email column appears to already exist!');
          console.log('Cleaning up test record...');
          await supabase.from('users').delete().eq('email', testEmail);
        }
      } else {
        console.log('✅ Email column exists and is working!');
        console.log('Cleaning up test record...');
        await supabase.from('users').delete().eq('email', testEmail);
      }
    } else {
      console.log('✅ Email column added successfully!');
    }
  } catch (err: any) {
    console.error('❌ Error:', err.message);
  }
}

addEmailColumn()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
