import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function ensureTop10UniqueProjects() {
  console.log('🔧 Ensuring top 10 projects are unique and accurate...\n')

  // First, delete ALL existing duplicates
  console.log('1️⃣ Removing all duplicates...\n')

  const { data: allProjects } = await supabase
    .from('projects')
    .select('id, name, created_at')
    .order('created_at', { ascending: true })

  if (!allProjects) return

  const nameMap = new Map<string, any[]>()
  allProjects.forEach(project => {
    const existing = nameMap.get(project.name) || []
    existing.push(project)
    nameMap.set(project.name, existing)
  })

  let deletedCount = 0
  for (const [name, projects] of nameMap.entries()) {
    if (projects.length > 1) {
      const toDelete = projects.slice(1)
      for (const dup of toDelete) {
        await supabase.from('projects').delete().eq('id', dup.id)
        deletedCount++
      }
    }
  }

  console.log(`✓ Removed ${deletedCount} duplicates\n`)

  // Get the 16 accurate projects we added
  const accurateProjectNames = [
    'Greenbushes Lithium Mine',
    'Pilgangoora Lithium-Tantalum Project',
    'Mount Marion Lithium Project',
    'Wodgina Lithium Project',
    'Altura Lithium Project',
    'Escondida Copper Mine',
    'Collahuasi Copper Mine',
    'Grasberg Copper-Gold Mine',
    'Norilsk-Talnakh Nickel-Copper-PGM Complex',
    'Sorowako Nickel Mine',
    'Weda Bay Nickel Project',
    'Tenke Fungurume Copper-Cobalt Mine',
    'Mutanda Copper-Cobalt Mine',
    'Bayan Obo Rare Earth Mine',
    'Mountain Pass Rare Earth Mine',
    'Mount Weld Rare Earth Mine'
  ]

  // Update the updated_at timestamp of these projects to ensure they appear first
  console.log('2️⃣ Setting accurate projects to appear first...\n')

  const now = new Date().toISOString()

  for (let i = 0; i < accurateProjectNames.length; i++) {
    const projectName = accurateProjectNames[i]
    // Set updated_at to now minus index (so they appear in order)
    const timestamp = new Date(Date.now() - (i * 1000)).toISOString()

    const { error } = await supabase
      .from('projects')
      .update({ updated_at: timestamp })
      .eq('name', projectName)

    if (error) {
      console.error(`✗ Failed to update ${projectName}:`, error)
    } else {
      console.log(`✓ Updated: ${projectName}`)
    }
  }

  // Verify the top 10
  console.log('\n3️⃣ Verifying top 10 projects...\n')

  const { data: topProjects } = await supabase
    .from('projects')
    .select('name, company_id, companies(name)')
    .order('updated_at', { ascending: false })
    .limit(10)

  console.log('Top 10 projects (as they will appear on the frontend):\n')
  topProjects?.forEach((p, i) => {
    console.log(`${i + 1}. ${p.name} - ${(p as any).companies?.name || 'NO COMPANY'}`)
  })

  // Count total unique projects
  const { count } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  console.log(`\n✅ Done! Total unique projects: ${count}`)
}

ensureTop10UniqueProjects()
