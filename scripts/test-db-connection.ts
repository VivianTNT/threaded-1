import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';
import * as path from 'path';

// Load .env.local
dotenv.config({ path: path.join(__dirname, '..', '.env.local') });

console.log('\n🔍 Testing Database Connection\n');
console.log('Environment variables loaded:');
console.log('NEXT_PUBLIC_SUPABASE_URL:', process.env.NEXT_PUBLIC_SUPABASE_URL);
console.log('SUPABASE_SERVICE_ROLE_KEY:', process.env.SUPABASE_SERVICE_ROLE_KEY?.substring(0, 30) + '...');

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);

async function testConnection() {
  try {
    console.log('\n📊 Fetching products...\n');

    const { data, error, count } = await supabase
      .from('products')
      .select('*', { count: 'exact' })
      .limit(5);

    if (error) {
      console.log('❌ Error:', error);
      return;
    }

    console.log(`✅ Success! Found ${count} total products`);
    console.log(`\nFirst 5 products:`);
    data?.forEach((product, idx) => {
      console.log(`\n${idx + 1}. ${product.name}`);
      console.log(`   Brand: ${product.brand_name || 'N/A'}`);
      console.log(`   Price: $${product.price || 'N/A'}`);
      console.log(`   Domain: ${product.domain}`);
    });
  } catch (err: any) {
    console.log('❌ Unexpected error:', err.message);
  }
}

testConnection()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
