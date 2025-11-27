import { createClient } from '@supabase/supabase-js';

const projectUrl = 'https://owbuzpxttovfssoarzwc.supabase.co';
const secretKey = 'sb_secret_yszkEDSnLE0ljW__Z15MJg_8rIIyFoP';

const supabase = createClient(projectUrl, secretKey);

async function checkUsersStructure() {
  console.log('🔍 Checking Penn Users Table Structure...\n');

  // Get all users to see the full structure
  const { data: users, error } = await supabase
    .from('users')
    .select('*');

  if (error) {
    console.log('❌ Error:', error.message);
    return;
  }

  console.log(`Found ${users?.length || 0} users\n`);

  if (users && users.length > 0) {
    console.log('Sample user data:');
    users.forEach((user, idx) => {
      console.log(`\nUser ${idx + 1}:`);
      console.log(JSON.stringify(user, null, 2));
    });
  }

  // Check if Supabase Auth is enabled
  console.log('\n═'.repeat(60));
  console.log('\n🔐 Checking Supabase Auth integration...\n');

  const { data: authData, error: authError } = await supabase.auth.getSession();

  if (authError) {
    console.log('Auth check result:', authError.message);
  } else {
    console.log('Auth is configured and accessible');
  }

  console.log('\n✅ Check complete!');
}

checkUsersStructure()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
