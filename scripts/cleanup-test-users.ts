import { createClient } from '@supabase/supabase-js';

const projectUrl = 'https://owbuzpxttovfssoarzwc.supabase.co';
const secretKey = 'sb_secret_yszkEDSnLE0ljW__Z15MJg_8rIIyFoP';

const supabase = createClient(projectUrl, secretKey);

async function cleanupTestUsers() {
  console.log('🧹 Cleaning up test users...\n');

  // Get all auth users
  const { data: authData } = await supabase.auth.admin.listUsers();

  if (authData && authData.users.length > 0) {
    console.log(`Found ${authData.users.length} auth users to delete:\n`);

    for (const user of authData.users) {
      console.log(`Deleting auth user: ${user.email} (${user.id})`);
      const { error } = await supabase.auth.admin.deleteUser(user.id);

      if (error) {
        console.log(`  ❌ Error: ${error.message}`);
      } else {
        console.log(`  ✅ Deleted from Supabase Auth`);
      }
    }
  }

  // Delete from users table
  const { data: users } = await supabase.from('users').select('*');

  if (users && users.length > 0) {
    console.log(`\nFound ${users.length} users in users table to delete:\n`);

    for (const user of users) {
      console.log(`Deleting user: ${user.handle} (${user.id})`);
      const { error } = await supabase
        .from('users')
        .delete()
        .eq('id', user.id);

      if (error) {
        console.log(`  ❌ Error: ${error.message}`);
      } else {
        console.log(`  ✅ Deleted from users table`);
      }
    }
  }

  console.log('\n✅ Cleanup complete! You can now sign up with a fresh account.');
}

cleanupTestUsers()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
