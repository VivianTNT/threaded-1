import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://dfxauievbyqwcynwtvib.supabase.co';
const supabaseServiceKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmeGF1aWV2Ynlxd2N5bnd0dmliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Nzg0ODI2MSwiZXhwIjoyMDYzNDI0MjYxfQ.Gs2NX-UUKtXvW3a9_h49ATSDzvpsfJdja6tt1bCkyjc';

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function listActualTables() {
  console.log('🔍 Querying actual tables in the database...\n');

  // Use raw SQL to query pg_catalog
  const { data, error } = await supabase.rpc('exec_sql', {
    query: `
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE'
      ORDER BY table_name;
    `
  });

  if (error) {
    console.log('RPC failed, trying alternative method...\n');

    // Try querying known tables from earlier
    const knownTables = [
      'companies', 'projects', 'supply_chain_nodes', 'unified_news',
      'news_items', 'chat_history', 'usr', 'news_projects', 'news_companies',
      'products', 'brands', 'users', 'user_events', 'product_embeddings',
      'wardrobe_items', 'recommendations', 'notifications', 'cart_items',
      'ai_insights', 'factset_documents', 'pdf_highlights'
    ];

    console.log('📊 Testing table access:\n');

    for (const tableName of knownTables) {
      try {
        const { count, error } = await supabase
          .from(tableName)
          .select('*', { count: 'exact', head: true });

        if (!error) {
          console.log(`✅ ${tableName.padEnd(30)} - ${count || 0} rows`);
        }
      } catch (err) {
        // Skip
      }
    }
    return;
  }

  console.log('Tables found:', data);
}

listActualTables()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
