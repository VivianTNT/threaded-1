#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!

const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

async function downloadAndUploadToSupabase() {
  console.log('📥 DOWNLOADING FACTSET DOCUMENTS TO SUPABASE STORAGE')
  console.log('='.repeat(60))

  // Get all projects with FactSet URLs
  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, urls')
    .not('urls', 'is', null)

  if (error || !projects) {
    console.log('❌ Failed to fetch projects:', error)
    return
  }

  console.log(`✅ Found ${projects.length} projects with documents\n`)

  let uploaded = 0
  let failed = 0

  for (const project of projects) {
    if (!project.urls || project.urls.length === 0) continue

    const factsetUrl = project.urls[0]

    // Only process FactSet URLs
    if (!factsetUrl.includes('api.factset.com')) {
      console.log(`⏭️  Skipping ${project.name} (not a FactSet URL)`)
      continue
    }

    console.log(`\n📄 Processing: ${project.name}`)
    console.log(`   Source: ${factsetUrl.substring(0, 80)}...`)

    try {
      // Download from FactSet
      console.log(`   📥 Downloading from FactSet...`)
      const response = await fetch(factsetUrl, {
        method: 'GET',
        headers: {
          'Authorization': authHeader,
          'Accept': 'text/html,application/pdf,*/*'
        }
      })

      if (!response.ok) {
        console.log(`   ❌ Download failed (${response.status})`)
        failed++
        continue
      }

      const buffer = Buffer.from(await response.arrayBuffer())
      const sizeInMB = (buffer.length / 1024 / 1024).toFixed(2)
      console.log(`   ✅ Downloaded ${sizeInMB} MB`)

      // Upload to Supabase Storage
      const fileName = `${project.id}.html`
      const filePath = `factset-documents/${fileName}`

      console.log(`   ☁️  Uploading to Supabase Storage...`)

      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('mining-documents')
        .upload(filePath, buffer, {
          contentType: 'text/html',
          upsert: true
        })

      if (uploadError) {
        console.log(`   ❌ Upload failed: ${uploadError.message}`)
        failed++
        continue
      }

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from('mining-documents')
        .getPublicUrl(filePath)

      console.log(`   ✅ Uploaded to: ${publicUrl}`)

      // Update project with new URL
      const { error: updateError } = await supabase
        .from('projects')
        .update({ urls: [publicUrl] })
        .eq('id', project.id)

      if (updateError) {
        console.log(`   ⚠️  Failed to update URL: ${updateError.message}`)
      } else {
        console.log(`   ✅ Updated project with public URL`)
        uploaded++
      }

    } catch (error: any) {
      console.log(`   ❌ Error: ${error.message}`)
      failed++
    }

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 1000))
  }

  console.log('\n' + '='.repeat(60))
  console.log('📊 SUMMARY')
  console.log('='.repeat(60))
  console.log(`✅ Successfully uploaded: ${uploaded}`)
  console.log(`❌ Failed: ${failed}`)
  console.log('\n✨ All FactSet documents are now publicly accessible!')
  console.log('🔗 Documents are stored in Supabase Storage: mining-documents/')
}

downloadAndUploadToSupabase().catch(console.error)