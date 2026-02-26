import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!supabaseUrl || !supabaseServiceKey) {
  throw new Error('Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function checkAllTables() {
  console.log('🔍 Checking access to all tables from schema diagram...\n');

  const tables = [
    'products',
    'product_embeddings',
    'brands',
    'users',
    'user_events'
  ];

  for (const tableName of tables) {
    console.log(`\n📋 Table: ${tableName}`);
    console.log('═'.repeat(60));

    try {
      // Check row count
      const { count, error: countError } = await supabase
        .from(tableName)
        .select('*', { count: 'exact', head: true });

      if (countError) {
        console.log(`❌ Error accessing table: ${countError.message}`);
        console.log(`   Code: ${countError.code}`);
        continue;
      }

      console.log(`✅ Access successful! Row count: ${count || 0}`);

      // Get sample data to show columns
      const { data: sampleData, error: dataError } = await supabase
        .from(tableName)
        .select('*')
        .limit(1);

      if (dataError) {
        console.log(`⚠️  Could not fetch sample data: ${dataError.message}`);
      } else if (sampleData && sampleData.length > 0) {
        console.log('\nSample row:');
        console.log(JSON.stringify(sampleData[0], null, 2));
      } else {
        console.log('\n📊 Table is empty (0 rows)');

        // For empty tables, try to get column info another way
        console.log('\nAttempting to get column structure...');
      }

    } catch (err: any) {
      console.log(`❌ Unexpected error: ${err.message}`);
    }
  }

  console.log('\n' + '═'.repeat(60));
  console.log('✅ Table access check complete!');
}

checkAllTables()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
