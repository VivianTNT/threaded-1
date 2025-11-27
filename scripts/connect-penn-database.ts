import { createClient } from '@supabase/supabase-js';
import fetch from 'node-fetch';

const projectId = 'owbuzpxttovfssoarzwc';
const projectUrl = 'https://owbuzpxttovfssoarzwc.supabase.co';
const publishableKey = 'sb_publishable_Jmbxhkw40JYw0VUUlRwFhQ_YllbVbkN';
const secretKey = 'sb_secret_yszkEDSnLE0ljW__Z15MJg_8rIIyFoP';

async function connectToDatabase() {
  console.log('🔍 Attempting to connect to Penn Supabase database...\n');
  console.log(`Project ID: ${projectId}`);
  console.log(`Project URL: ${projectUrl}\n`);

  // Try using the secret key as the service role key directly
  console.log('Method 1: Using secret key as service role key...');
  try {
    const supabase1 = createClient(projectUrl, secretKey);
    const { data, error } = await supabase1
      .from('users')
      .select('*', { count: 'exact', head: true });

    if (!error) {
      console.log(`✅ Success! Connected with secret key. Users table has ${data || 0} rows.\n`);
      return supabase1;
    } else {
      console.log(`❌ Error: ${error.message}\n`);
    }
  } catch (err: any) {
    console.log(`❌ Failed: ${err.message}\n`);
  }

  // Try using the publishable key as anon key
  console.log('Method 2: Using publishable key as anon key...');
  try {
    const supabase2 = createClient(projectUrl, publishableKey);
    const { data, error } = await supabase2
      .from('users')
      .select('*', { count: 'exact', head: true });

    if (!error) {
      console.log(`✅ Success! Connected with publishable key. Users table has ${data || 0} rows.\n`);
      return supabase2;
    } else {
      console.log(`❌ Error: ${error.message}\n`);
    }
  } catch (err: any) {
    console.log(`❌ Failed: ${err.message}\n`);
  }

  // Try to get the actual API keys via Management API
  console.log('Method 3: Fetching actual API keys from Management API...');
  try {
    const response = await fetch(
      `https://api.supabase.com/v1/projects/${projectId}/api-keys`,
      {
        headers: {
          'Authorization': `Bearer ${secretKey}`,
          'Content-Type': 'application/json',
        },
      }
    );

    if (response.ok) {
      const keys = await response.json();
      console.log('✅ Retrieved API keys:');
      console.log(JSON.stringify(keys, null, 2));
    } else {
      console.log(`❌ API returned: ${response.status} ${response.statusText}`);
      const errorText = await response.text();
      console.log(`Response: ${errorText}\n`);
    }
  } catch (err: any) {
    console.log(`❌ Failed: ${err.message}\n`);
  }

  console.log('❌ All connection methods failed.');
  console.log('\n⚠️  The publishable/secret keys are Management API keys, not database API keys.');
  console.log('You need to get the actual "anon public" and "service_role" keys from:');
  console.log(`https://supabase.com/dashboard/project/${projectId}/settings/api`);
}

connectToDatabase()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
