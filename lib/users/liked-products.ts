import { SupabaseClient } from '@supabase/supabase-js'
import { buildUserEmbeddingWithTwoTower } from '@/lib/recommendations/model-service'
import { l2Normalize, parseVector } from '@/lib/recommendations/image-content'

const DEFAULT_MAX_STORED_LIKED_PRODUCT_IDS = 100
const DEFAULT_EMBEDDING_CONTEXT_WINDOW = 50
const PRODUCT_ID_BATCH_SIZE = 200

type UserLikesSource = {
  liked_product_ids?: unknown
  metadata?: {
    liked_product_ids?: unknown
  } | null
}

type ProductEmbeddingRow = {
  product_id: string | number
  image_embedding: unknown
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback
  return Math.floor(parsed)
}

export const MAX_STORED_LIKED_PRODUCT_IDS = parsePositiveInt(
  process.env.MAX_STORED_LIKED_PRODUCT_IDS,
  DEFAULT_MAX_STORED_LIKED_PRODUCT_IDS
)

export const USER_EMBEDDING_CONTEXT_WINDOW = parsePositiveInt(
  process.env.USER_EMBEDDING_CONTEXT_WINDOW,
  DEFAULT_EMBEDDING_CONTEXT_WINDOW
)

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []

  const out: string[] = []
  for (const item of value) {
    const id = String(item || '').trim()
    if (!id) continue
    out.push(id)
  }
  return out
}

function uniquePreservingOrder(ids: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []

  for (const id of ids) {
    if (seen.has(id)) continue
    seen.add(id)
    out.push(id)
  }

  return out
}

function chunkArray<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size))
  }
  return chunks
}

export function normalizeLikedProductIds(value: unknown, maxItems = MAX_STORED_LIKED_PRODUCT_IDS): string[] {
  const uniqueIds = uniquePreservingOrder(toStringArray(value))
  if (uniqueIds.length <= maxItems) return uniqueIds
  return uniqueIds.slice(-maxItems)
}

export function getStoredLikedProductIds(userRow: UserLikesSource | null | undefined): string[] {
  const columnIds = normalizeLikedProductIds(userRow?.liked_product_ids)
  if (columnIds.length > 0) return columnIds
  return normalizeLikedProductIds(userRow?.metadata?.liked_product_ids)
}

export function addLikedProductIds(
  existing: unknown,
  additions: unknown,
  maxItems = MAX_STORED_LIKED_PRODUCT_IDS
): string[] {
  const existingIds = normalizeLikedProductIds(existing, maxItems)
  const newIds = normalizeLikedProductIds(additions, maxItems)
  if (!newIds.length) return existingIds

  const newIdSet = new Set(newIds)
  const merged = [...existingIds.filter((id) => !newIdSet.has(id)), ...newIds]
  return merged.slice(-maxItems)
}

export function removeLikedProductId(
  existing: unknown,
  productId: string,
  maxItems = MAX_STORED_LIKED_PRODUCT_IDS
): string[] {
  const targetId = String(productId || '').trim()
  if (!targetId) return normalizeLikedProductIds(existing, maxItems)
  return normalizeLikedProductIds(existing, maxItems).filter((id) => id !== targetId)
}

function recencyWeightedAverage(vectors: number[][]): number[] {
  if (!vectors.length) return []

  const dim = vectors[0].length
  const out = new Array(dim).fill(0)
  let totalWeight = 0

  for (let index = 0; index < vectors.length; index += 1) {
    const vec = vectors[index]
    const weight = index + 1
    totalWeight += weight

    for (let dimIndex = 0; dimIndex < dim; dimIndex += 1) {
      out[dimIndex] += vec[dimIndex] * weight
    }
  }

  return l2Normalize(out.map((value) => value / totalWeight))
}

export async function computeUserImageEmbeddingFromLikedProducts(
  supabase: SupabaseClient<any, any, any>,
  likedProductIds: string[]
): Promise<number[] | null> {
  const recentLikedIds = normalizeLikedProductIds(likedProductIds, USER_EMBEDDING_CONTEXT_WINDOW)
  if (!recentLikedIds.length) return null

  const embeddingsByProductId = new Map<string, number[]>()
  for (const chunk of chunkArray(recentLikedIds, PRODUCT_ID_BATCH_SIZE)) {
    const { data, error } = await supabase
      .from('product_embeddings')
      .select('product_id,image_embedding')
      .in('product_id', chunk)
      .not('image_embedding', 'is', null)

    if (error) {
      throw new Error(`Failed to load liked product embeddings: ${error.message}`)
    }

    for (const row of (data || []) as ProductEmbeddingRow[]) {
      const vec = parseVector(row.image_embedding)
      if (!vec || vec.length === 0) continue
      embeddingsByProductId.set(String(row.product_id), l2Normalize(vec))
    }
  }

  const orderedVectors = recentLikedIds
    .map((productId) => embeddingsByProductId.get(productId))
    .filter((vec): vec is number[] => Array.isArray(vec) && vec.length > 0)

  if (!orderedVectors.length) return null

  const sourceDim = orderedVectors[0]?.length || 0
  const modelEmbedding = await buildUserEmbeddingWithTwoTower(orderedVectors)
  if (modelEmbedding && modelEmbedding.length === sourceDim) {
    return l2Normalize(modelEmbedding)
  }

  return recencyWeightedAverage(orderedVectors)
}
