import { createClient } from '@supabase/supabase-js';

const projectUrl = 'https://owbuzpxttovfssoarzwc.supabase.co';
const secretKey = 'sb_secret_yszkEDSnLE0ljW__Z15MJg_8rIIyFoP';

const supabase = createClient(projectUrl, secretKey);

async function verifyStructure() {
  console.log('🔍 Verifying users table structure...\n');

  // Try inserting and immediately deleting a test user to see all columns
  const testData = {
    id: '00000000-0000-0000-0000-000000000000',
    handle: 'structure_test',
    email: `test-${Date.now()}@example.com`,
    metadata: { test: true },
    uploaded_images: ['test.jpg']
  };

  const { data, error } = await supabase
    .from('users')
    .insert(testData)
    .select()
    .single();

  if (error) {
    console.log('Insert error (this helps us see which columns exist):');
    console.log(error.message);
  } else {
    console.log('✅ Successfully created test user. Structure confirmed:');
    console.log(JSON.stringify(data, null, 2));

    // Clean up
    await supabase.from('users').delete().eq('id', testData.id);
    console.log('\n✅ Test user cleaned up');
  }

  console.log('\n📊 Current users in table:');
  const { data: users } = await supabase.from('users').select('*');
  console.log(`Total users: ${users?.length || 0}`);
}

verifyStructure()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
