import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'
import { parseVector, vectorToPg } from '@/lib/recommendations/image-content'

type ProductEmbeddingRow = {
  product_id: string
  image_embedding: unknown
}

type EmbedItemsResponse = {
  items?: Array<{
    id?: unknown
    embedding?: unknown
    used_model?: unknown
  }>
  used_model?: unknown
  model_loaded?: unknown
  fallback_reason?: unknown
}

// Auto-load local environment variables for standalone script execution.
config({ path: path.resolve(process.cwd(), '.env.local') })
config()

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY
const MODEL_SERVICE_URL = (process.env.RECOMMENDER_MODEL_SERVICE_URL || 'http://127.0.0.1:8001').replace(/\/+$/, '')
const BATCH_SIZE = Number(process.env.BACKFILL_BATCH_SIZE || '200')
const DRY_RUN = process.env.DRY_RUN === '1'
const MODEL_NAME = process.env.TWO_TOWER_MODEL_NAME || 'content_two_tower_hm'
const MODEL_VERSION = process.env.TWO_TOWER_MODEL_VERSION || 'v1'

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  throw new Error('Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.')
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

function normalizeEmbedding(value: unknown): number[] | null {
  const vec = parseVector(value)
  if (!vec || vec.length === 0) return null
  return vec
}

async function callEmbedItems(
  items: Array<{ id: string; embedding: number[] }>
): Promise<Array<{ id: string; embedding: number[]; usedModel: boolean }>> {
  const response = await fetch(`${MODEL_SERVICE_URL}/embed-items`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Model service embed-items failed (${response.status}): ${text}`)
  }

  const json = (await response.json()) as EmbedItemsResponse
  const out: Array<{ id: string; embedding: number[]; usedModel: boolean }> = []

  for (const row of json.items || []) {
    const id = typeof row.id === 'string' ? row.id : row.id != null ? String(row.id) : ''
    const embedding = normalizeEmbedding(row.embedding)
    const usedModel = Boolean(row.used_model)
    if (!id || !embedding) continue
    out.push({ id, embedding, usedModel })
  }

  if (!out.length) {
    throw new Error('Model service returned no valid embeddings.')
  }

  return out
}

async function upsertEmbeddings(
  rows: Array<{ id: string; embedding: number[]; usedModel: boolean }>,
  includeMetadataColumns: boolean
): Promise<boolean> {
  const now = new Date().toISOString()
  const payload = rows.map((row) => {
    const base: Record<string, unknown> = {
      product_id: row.id,
      image_embedding: vectorToPg(row.embedding),
    }

    if (includeMetadataColumns) {
      base.model_name = MODEL_NAME
      base.model_version = MODEL_VERSION
      base.embedding_source = row.usedModel ? 'two_tower_item_tower' : 'two_tower_fallback'
      base.updated_at = now
    }
    return base
  })

  const { error } = await supabase
    .from('product_embeddings')
    .upsert(payload, { onConflict: 'product_id' })

  if (!error) return includeMetadataColumns

  const missingColumnMessage =
    /column .* does not exist/i.test(error.message) ||
    /could not find the .* column .* in the schema cache/i.test(error.message)

  if (includeMetadataColumns && missingColumnMessage) {
    console.warn('Metadata columns missing on product_embeddings; retrying without metadata columns.')
    return upsertEmbeddings(rows, false)
  }

  throw new Error(`Failed to upsert embeddings: ${error.message}`)
}

async function run(): Promise<void> {
  console.log('Starting two-tower embedding backfill')
  console.log(`Model service: ${MODEL_SERVICE_URL}`)
  console.log(`Batch size: ${BATCH_SIZE}`)
  console.log(`Dry run: ${DRY_RUN ? 'yes' : 'no'}`)

  let offset = 0
  let totalSeen = 0
  let totalUpdated = 0
  let includeMetadataColumns = true

  for (;;) {
    const { data, error } = await supabase
      .from('product_embeddings')
      .select('product_id,image_embedding')
      .not('image_embedding', 'is', null)
      .order('product_id', { ascending: true })
      .range(offset, offset + BATCH_SIZE - 1)

    if (error) {
      throw new Error(`Failed loading product_embeddings batch: ${error.message}`)
    }

    const batch = (data || []) as ProductEmbeddingRow[]
    if (!batch.length) break

    totalSeen += batch.length

    const items = batch
      .map((row) => {
        const id = String(row.product_id)
        const embedding = normalizeEmbedding(row.image_embedding)
        if (!id || !embedding) return null
        return { id, embedding }
      })
      .filter((row): row is { id: string; embedding: number[] } => row !== null)

    if (!items.length) {
      offset += BATCH_SIZE
      continue
    }

    const embedded = await callEmbedItems(items)
    if (!DRY_RUN) {
      includeMetadataColumns = await upsertEmbeddings(embedded, includeMetadataColumns)
    }

    totalUpdated += embedded.length
    offset += BATCH_SIZE
    console.log(`Processed ${totalSeen} rows, prepared ${totalUpdated} updates`)
  }

  console.log('Backfill complete')
  console.log(`Rows scanned: ${totalSeen}`)
  console.log(`Rows updated: ${totalUpdated}`)
}

run().catch((error) => {
  console.error(error)
  process.exit(1)
})
