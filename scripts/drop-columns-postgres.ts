import pkg from 'pg';
const { Client } = pkg;

// Construct PostgreSQL connection string
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const dbPassword = process.env.DATABASE_PASSWORD!;

// Extract project ref from URL
const ref = supabaseUrl.replace('https://', '').replace('.supabase.co', '');

// Supabase pooler connection string format
const connectionString = `postgresql://postgres.${ref}:${dbPassword}@aws-0-us-west-1.pooler.supabase.com:5432/postgres`;

async function dropColumns() {
  console.log('🗑️  Dropping optional columns from projects table in Supabase...\n');

  const client = new Client({ connectionString });

  try {
    await client.connect();
    console.log('✓ Connected to Supabase PostgreSQL\n');

    const columnsToRemove = [
      'resource_estimate',
      'reserve_estimate',
      'ownership_percentage',
      'operator',
      'production_rate',
      'mine_life',
      'capex',
      'first_production',
      'project_type'
    ];

    for (const column of columnsToRemove) {
      console.log(`🗑️  Dropping column: ${column}...`);
      await client.query(`ALTER TABLE projects DROP COLUMN IF EXISTS ${column}`);
      console.log(`   ✓ Dropped ${column}`);
    }

    console.log('\n✅ All optional columns removed from projects table!');
    console.log('\n📋 Remaining columns:');

    const { rows } = await client.query(`
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name = 'projects'
      ORDER BY ordinal_position
    `);

    rows.forEach(row => {
      console.log(`   • ${row.column_name} (${row.data_type})`);
    });

  } catch (error: any) {
    console.error('❌ Error:', error.message);
    console.log('\n💡 Trying alternative connection methods...\n');

    // Try direct connection without pooler
    const directConnection = `postgresql://postgres:${dbPassword}@db.${ref}.supabase.co:5432/postgres`;
    const directClient = new Client({ connectionString: directConnection });

    try {
      await directClient.connect();
      console.log('✓ Connected via direct connection\n');

      const columnsToRemove = [
        'resource_estimate',
        'reserve_estimate',
        'ownership_percentage',
        'operator',
        'production_rate',
        'mine_life',
        'capex',
        'first_production',
        'project_type'
      ];

      for (const column of columnsToRemove) {
        console.log(`🗑️  Dropping column: ${column}...`);
        await directClient.query(`ALTER TABLE projects DROP COLUMN IF EXISTS ${column}`);
        console.log(`   ✓ Dropped ${column}`);
      }

      console.log('\n✅ All optional columns removed!');

      await directClient.end();
    } catch (directError: any) {
      console.error('❌ Direct connection also failed:', directError.message);
      console.log('\n📝 Manual steps required:');
      console.log('1. Go to Supabase Dashboard: https://supabase.com/dashboard/project/' + ref);
      console.log('2. Navigate to SQL Editor');
      console.log('3. Run this SQL:\n');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS resource_estimate;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS reserve_estimate;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS ownership_percentage;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS operator;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS production_rate;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS mine_life;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS capex;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS first_production;');
      console.log('ALTER TABLE projects DROP COLUMN IF EXISTS project_type;');
    }
  } finally {
    await client.end();
  }
}

dropColumns();
