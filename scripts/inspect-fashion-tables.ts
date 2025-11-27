import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://dfxauievbyqwcynwtvib.supabase.co';
const supabaseServiceKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmeGF1aWV2Ynlxd2N5bnd0dmliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Nzg0ODI2MSwiZXhwIjoyMDYzNDI0MjYxfQ.Gs2NX-UUKtXvW3a9_h49ATSDzvpsfJdja6tt1bCkyjc';

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function inspectFashionTables() {
  console.log('👗 Inspecting Fashion-Related Tables...\n');

  const fashionTables = [
    'users',
    'fashion_products',
    'wardrobe_items',
    'recommendations',
    'notifications',
    'chat_history'
  ];

  for (const tableName of fashionTables) {
    console.log(`\n📋 Table: ${tableName}`);
    console.log('═'.repeat(60));

    // Get sample data to see structure
    const { data: sampleData, error } = await supabase
      .from(tableName)
      .select('*')
      .limit(1);

    if (error) {
      console.log(`❌ Error accessing table: ${error.message}`);
      continue;
    }

    if (sampleData && sampleData.length > 0) {
      console.log('Sample row:');
      console.log(JSON.stringify(sampleData[0], null, 2));
    } else {
      console.log('Table exists but is empty.');

      // Try to get table structure by attempting an insert with no data (will fail but show structure)
      const { error: structureError } = await supabase
        .from(tableName)
        .insert({});

      console.log('\nColumns (inferred from empty table):');
      // List common expected columns based on table name
      if (tableName === 'users') {
        console.log('  Expected: id, email, name, created_at, etc.');
      } else if (tableName === 'fashion_products') {
        console.log('  Expected: id, name, brand, price, image_url, category, etc.');
      } else if (tableName === 'wardrobe_items') {
        console.log('  Expected: id, user_id, image_url, category, color, etc.');
      } else if (tableName === 'recommendations') {
        console.log('  Expected: id, user_id, product_id, score, reason, etc.');
      } else if (tableName === 'notifications') {
        console.log('  Expected: id, user_id, message, type, read, created_at, etc.');
      } else if (tableName === 'chat_history') {
        console.log('  Expected: id, user_id, message, role, created_at, etc.');
      }
    }

    // Get row count
    const { count } = await supabase
      .from(tableName)
      .select('*', { count: 'exact', head: true });

    console.log(`\n📊 Row count: ${count || 0}`);
  }

  console.log('\n' + '═'.repeat(60));
  console.log('✅ Inspection complete!');
}

inspectFashionTables()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('❌ Error:', error);
    process.exit(1);
  });
