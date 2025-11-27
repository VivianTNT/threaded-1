import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://dfxauievbyqwcynwtvib.supabase.co';
const supabaseServiceKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmeGF1aWV2Ynlxd2N5bnd0dmliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Nzg0ODI2MSwiZXhwIjoyMDYzNDI0MjYxfQ.Gs2NX-UUKtXvW3a9_h49ATSDzvpsfJdja6tt1bCkyjc';

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function inspectDatabase() {
  console.log('🔍 Inspecting Supabase Database Schema...\n');

  // Query to get all tables in the public schema
  const { data: tables, error: tablesError } = await supabase
    .from('information_schema.tables')
    .select('table_name')
    .eq('table_schema', 'public')
    .order('table_name');

  if (tablesError) {
    console.error('Error fetching tables:', tablesError);

    // Alternative approach: try to query known tables
    console.log('\n⚠️  Direct query failed. Trying to list tables via RPC...\n');

    const { data: rpcData, error: rpcError } = await supabase.rpc('get_tables');

    if (rpcError) {
      console.log('RPC also failed. Let me try querying common tables directly...\n');

      // Try common table names
      const commonTables = [
        'users', 'companies', 'projects', 'supply_chain_nodes',
        'news_items', 'chat_history', 'notifications', 'fashion_products',
        'wardrobe_items', 'recommendations'
      ];

      for (const tableName of commonTables) {
        try {
          const { data, error, count } = await supabase
            .from(tableName)
            .select('*', { count: 'exact', head: true });

          if (!error) {
            console.log(`✅ Found table: ${tableName} (${count || 0} rows)`);

            // Get one row to inspect columns
            const { data: sampleData } = await supabase
              .from(tableName)
              .select('*')
              .limit(1);

            if (sampleData && sampleData.length > 0) {
              console.log(`   Columns: ${Object.keys(sampleData[0]).join(', ')}`);
            }
            console.log('');
          }
        } catch (err) {
          // Table doesn't exist, skip
        }
      }
      return;
    }
  }

  if (tables && tables.length > 0) {
    console.log(`📊 Found ${tables.length} tables:\n`);

    for (const table of tables) {
      const tableName = table.table_name;
      console.log(`\n📋 Table: ${tableName}`);
      console.log('─'.repeat(50));

      // Get column information
      const { data: columns, error: columnsError } = await supabase
        .from('information_schema.columns')
        .select('column_name, data_type, is_nullable, column_default')
        .eq('table_schema', 'public')
        .eq('table_name', tableName)
        .order('ordinal_position');

      if (!columnsError && columns) {
        console.log('Columns:');
        columns.forEach((col: any) => {
          const nullable = col.is_nullable === 'YES' ? '(nullable)' : '(required)';
          const defaultVal = col.column_default ? ` [default: ${col.column_default}]` : '';
          console.log(`  • ${col.column_name}: ${col.data_type} ${nullable}${defaultVal}`);
        });
      }

      // Get row count
      const { count } = await supabase
        .from(tableName)
        .select('*', { count: 'exact', head: true });

      console.log(`\nRow count: ${count || 0}`);
    }
  } else {
    console.log('No tables found or unable to access schema information.');
  }
}

inspectDatabase()
  .then(() => {
    console.log('\n✅ Database inspection complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Error:', error);
    process.exit(1);
  });
