import { createClient } from '@supabase/supabase-js';

const projectUrl = 'https://owbuzpxttovfssoarzwc.supabase.co';
const secretKey = 'sb_secret_yszkEDSnLE0ljW__Z15MJg_8rIIyFoP';

const supabase = createClient(projectUrl, secretKey);

async function checkUsers() {
  console.log('🔍 Checking existing users in Penn database...\n');

  // Check users table
  const { data: users, error } = await supabase
    .from('users')
    .select('*');

  if (error) {
    console.log('❌ Error fetching users:', error.message);
    return;
  }

  console.log(`📊 Found ${users?.length || 0} users in users table:\n`);

  if (users && users.length > 0) {
    users.forEach((user, idx) => {
      console.log(`User ${idx + 1}:`);
      console.log(`  ID: ${user.id}`);
      console.log(`  Handle: ${user.handle}`);
      console.log(`  Email: ${user.email || 'NO EMAIL SET'}`);
      console.log(`  Created: ${user.created_at}`);
      console.log('');
    });
  }

  // Check Supabase Auth users
  console.log('\n🔐 Checking Supabase Auth users...\n');

  const { data: authData, error: authError } = await supabase.auth.admin.listUsers();

  if (authError) {
    console.log('❌ Error fetching auth users:', authError.message);
  } else {
    console.log(`📊 Found ${authData.users.length} users in Supabase Auth:\n`);
    authData.users.forEach((user, idx) => {
      console.log(`Auth User ${idx + 1}:`);
      console.log(`  ID: ${user.id}`);
      console.log(`  Email: ${user.email}`);
      console.log(`  Created: ${user.created_at}`);
      console.log('');
    });
  }
}

checkUsers()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
