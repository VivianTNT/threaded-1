#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import fetch from 'node-fetch'
import * as fs from 'fs/promises'
import * as path from 'path'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Large technical reports that are publicly accessible
const TECHNICAL_REPORTS = [
  {
    projectName: 'Escondida',
    url: 'https://www.bhp.com/-/media/documents/media/reports-and-presentations/2024/240827_bhpresultsfortheyearended30june2024.pdf',
    fileName: 'bhp-annual-report-2024.pdf'
  },
  {
    projectName: 'Nevada Gold Mines',
    url: 'https://s25.q4cdn.com/322814910/files/doc_financials/2023/ar/barrick-ar-2023.pdf',
    fileName: 'barrick-annual-report-2023.pdf'
  },
  {
    projectName: 'Oyu Tolgoi',
    url: 'https://www.riotinto.com/-/media/Content/Documents/Invest/Annual-reports/RT-2023-annual-report.pdf',
    fileName: 'rio-tinto-annual-report-2023.pdf'
  }
]

async function downloadAndUploadReport(report: any) {
  console.log(`\n📥 Processing: ${report.projectName}`)
  
  try {
    // For now, just update the project with the direct URL
    const { data: project } = await supabase
      .from('projects')
      .select('id, urls')
      .eq('name', report.projectName)
      .single()

    if (project) {
      const { error } = await supabase
        .from('projects')
        .update({
          urls: [report.url],
          updated_at: new Date().toISOString()
        })
        .eq('id', project.id)

      if (!error) {
        console.log(`  ✅ Updated ${report.projectName} with working URL`)
        return true
      }
    }
    return false
  } catch (error) {
    console.log(`  ❌ Error: ${error}`)
    return false
  }
}

async function main() {
  console.log('🚀 Updating Projects with Working Document URLs')
  console.log('=' .repeat(60))

  for (const report of TECHNICAL_REPORTS) {
    await downloadAndUploadReport(report)
  }

  console.log('\n✨ Done!')
}

main().catch(console.error)
