import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { transformPennProduct, PennProduct } from '@/lib/penn-products'
import { cosineSimilarity, l2Normalize, parseVector } from '@/lib/recommendations/image-content'
import { rankWithTwoTower } from '@/lib/recommendations/model-service'
import { getStoredLikedProductIds } from '@/lib/users/liked-products'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseKey)

type UserRecRow = {
  id: string
  email: string | null
  metadata: any
  liked_product_ids: string[] | null
  user_image_embedding_avg: unknown
}

type ProductEmbeddingRow = {
  product_id: string
  image_embedding: unknown
}

type ScoredId = { id: string; score: number }
type HybridCatalogResult = { product_id: string | number; score: number }
type RecommendationEngine =
  | 'faiss_two_tower_hybrid'
  | 'faiss_two_tower_image_hybrid'
  | 'two_tower_or_cosine_fallback'
  | 'latest_products_fallback'

const DEFAULT_RECSYS_BASE_URL = 'http://127.0.0.1:8000'
const HYBRID_CATALOG_LIMIT = 1000
const HYBRID_API_TIMEOUT_MS = parseInt(process.env.RECSYS_TIMEOUT_MS || '60000', 10)
const SUPABASE_SELECT_PAGE_SIZE = 500
const SUPABASE_IN_FILTER_BATCH_SIZE = 200

function chunkArray<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size))
  }
  return chunks
}

function getRecsysBaseUrl(): string {
  const raw = process.env.RECSYS_API_URL || DEFAULT_RECSYS_BASE_URL
  return raw.replace(/\/+$/, '')
}

function asMetadataStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((v) => String(v))
}

async function getUserRow(userId: string | null, userEmail: string | null): Promise<UserRecRow | null> {
  if (!userId && !userEmail) return null

  let query = supabase
    .from('users')
    .select('id,email,metadata,liked_product_ids,user_image_embedding_avg')
    .limit(1)

  if (userId) {
    query = query.eq('id', userId)
  } else if (userEmail) {
    query = query.eq('email', userEmail)
  }

  const { data, error } = await query.maybeSingle()
  if (error) {
    throw new Error(`Failed to load user profile: ${error.message}`)
  }
  return (data as UserRecRow | null) || null
}

async function getProductsByIds(ids: string[]): Promise<PennProduct[]> {
  if (!ids.length) return []

  const uniqueIds = Array.from(new Set(ids.map(String)))
  const productsById = new Map<string, PennProduct>()

  for (const chunk of chunkArray(uniqueIds, SUPABASE_IN_FILTER_BATCH_SIZE)) {
    const { data, error } = await supabase
      .from('products')
      .select('*')
      .in('id', chunk)

    if (error) {
      throw new Error(`Failed to load products by ids: ${error.message}`)
    }

    for (const row of (data as PennProduct[] || [])) {
      productsById.set(String(row.id), row)
    }
  }

  return uniqueIds.map((id) => productsById.get(id)).filter((p): p is PennProduct => Boolean(p))
}

async function getLatestProducts(limit: number, offset: number): Promise<{ products: PennProduct[]; total: number }> {
  const { data, error, count } = await supabase
    .from('products')
    .select('*', { count: 'exact' })
    .range(offset, offset + limit - 1)
    .order('created_at', { ascending: false })

  if (error) {
    throw new Error(`Failed to fetch products: ${error.message}`)
  }

  return {
    products: (data as PennProduct[]) || [],
    total: count || 0,
  }
}

async function getCatalogProductsForHybrid(limit: number): Promise<PennProduct[]> {
  const { data, error } = await supabase
    .from('products')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(limit)

  if (error) {
    throw new Error(`Failed to fetch hybrid catalog products: ${error.message}`)
  }
  return (data as PennProduct[]) || []
}

function toRecommendationScore(score: number): number {
  // Hybrid score is often centered near [-1, 1].
  const normalized = (score + 1) * 50
  return Math.max(0, Math.min(100, Math.round(normalized)))
}

function normalizeHybridScore(score: number): number {
  return Math.max(0, Math.min(1, (score + 1) / 2))
}

function denormalizeHybridScore(score: number): number {
  return Math.max(-1, Math.min(1, score * 2 - 1))
}

async function getProductEmbeddingsByIds(ids: string[]): Promise<Map<string, number[]>> {
  if (!ids.length) return new Map()

  const out = new Map<string, number[]>()
  for (const chunk of chunkArray(Array.from(new Set(ids.map(String))), SUPABASE_IN_FILTER_BATCH_SIZE)) {
    const { data, error } = await supabase
      .from('product_embeddings')
      .select('product_id,image_embedding')
      .in('product_id', chunk)
      .not('image_embedding', 'is', null)

    if (error) {
      throw new Error(`Failed to load product embeddings by ids: ${error.message}`)
    }

    for (const row of (data || []) as ProductEmbeddingRow[]) {
      const vec = parseVector(row.image_embedding)
      if (!vec || !vec.length) continue
      out.set(String(row.product_id), l2Normalize(vec))
    }
  }

  return out
}

async function rerankHybridResultsWithImageSignals(
  hybridScored: ScoredId[],
  likedIds: string[]
): Promise<ScoredId[]> {
  if (!hybridScored.length || !likedIds.length) return hybridScored

  const embeddingIds = Array.from(new Set([...likedIds, ...hybridScored.map((row) => row.id)]))
  const embeddings = await getProductEmbeddingsByIds(embeddingIds)
  const likedVectors = likedIds
    .map((id) => embeddings.get(id))
    .filter((vec): vec is number[] => Array.isArray(vec) && vec.length > 0)

  if (!likedVectors.length) return hybridScored

  const imageUserVec = l2Normalize(
    likedVectors[0].map((_, index) => {
      let sum = 0
      for (const vec of likedVectors) sum += vec[index]
      return sum / likedVectors.length
    })
  )

  const reranked = hybridScored.map((row) => {
    const candidateVec = embeddings.get(row.id)
    if (!candidateVec || candidateVec.length !== imageUserVec.length) {
      return row
    }

    const textNorm = normalizeHybridScore(row.score)
    const imageNorm = normalizeHybridScore(cosineSimilarity(imageUserVec, candidateVec))
    const fusedNorm = 0.7 * textNorm + 0.3 * imageNorm

    return {
      id: row.id,
      score: denormalizeHybridScore(fusedNorm),
    }
  })

  reranked.sort((a, b) => b.score - a.score)
  return reranked
}

async function scoreProductsWithHybridCatalogApi(
  likedProducts: PennProduct[],
  catalogProducts: PennProduct[],
  excludedIds: Set<string>,
  topK: number
): Promise<ScoredId[]> {
  if (!likedProducts.length || !catalogProducts.length) return []

  const baseUrl = getRecsysBaseUrl()
  const filteredCatalog = catalogProducts.filter((p) => !excludedIds.has(String(p.id)))
  if (!filteredCatalog.length) return []

  const payload = {
    liked_products: likedProducts.map((p) => ({
      id: p.id,
      title: p.name || '',
      name: p.name || '',
      description: p.description || '',
    })),
    catalog_products: filteredCatalog.map((p) => ({
      id: p.id,
      title: p.name || '',
      name: p.name || '',
      description: p.description || '',
    })),
    top_k: topK,
    strategy: 'hybrid',
    exclude_ids: Array.from(excludedIds),
  }

  const controller = new AbortController()
  const startedAt = Date.now()
  const timeout = setTimeout(() => controller.abort(), HYBRID_API_TIMEOUT_MS)
  try {
    const response = await fetch(`${baseUrl}/recommend/hybrid/catalog`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })

    if (!response.ok) {
      const body = await response.text().catch(() => '')
      throw new Error(
        `Hybrid API error (${response.status}) after ${Date.now() - startedAt}ms: ${body || 'no body'}`
      )
    }

    const json = await response.json()
    const recs: HybridCatalogResult[] = Array.isArray(json?.recommendations) ? json.recommendations : []

    return recs
      .map((r) => ({
        id: String(r.product_id),
        score: Number(r.score),
      }))
      .filter((r) => Number.isFinite(r.score))
  } catch (error: any) {
    const elapsedMs = Date.now() - startedAt
    console.error('Hybrid catalog API request failed', {
      baseUrl,
      elapsedMs,
      timeoutMs: HYBRID_API_TIMEOUT_MS,
      likedProducts: likedProducts.length,
      catalogProducts: filteredCatalog.length,
      error: error?.name || 'UnknownError',
      message: error?.message || String(error),
    })
    throw error
  } finally {
    clearTimeout(timeout)
  }
}

async function scoreProductsByUserImageEmbedding(
  userImageEmbedding: number[],
  excludedIds: Set<string>
): Promise<ScoredId[]> {
  const userVec = l2Normalize(userImageEmbedding)

  const candidates: Array<{ id: string; embedding: number[] }> = []
  for (let from = 0; ; from += SUPABASE_SELECT_PAGE_SIZE) {
    const to = from + SUPABASE_SELECT_PAGE_SIZE - 1
    const { data, error } = await supabase
      .from('product_embeddings')
      .select('product_id,image_embedding')
      .not('image_embedding', 'is', null)
      .order('product_id', { ascending: true })
      .range(from, to)

    if (error) {
      throw new Error(`Failed to load product embeddings: ${error.message}`)
    }

    for (const row of (data || []) as ProductEmbeddingRow[]) {
      const id = String(row.product_id)
      if (excludedIds.has(id)) continue

      const vec = parseVector(row.image_embedding)
      if (!vec || !vec.length) continue
      const normalizedVec = l2Normalize(vec)
      candidates.push({ id, embedding: normalizedVec })
    }

    if (!data || data.length < SUPABASE_SELECT_PAGE_SIZE) {
      break
    }
  }

  const modelRanked = await rankWithTwoTower(userVec, candidates)
  if (modelRanked && modelRanked.length > 0) {
    return modelRanked.map((row) => ({ id: row.id, score: row.score }))
  }

  const scored: ScoredId[] = []
  for (const candidate of candidates) {
    if (candidate.embedding.length !== userVec.length) continue
    const score = cosineSimilarity(userVec, candidate.embedding)
    scored.push({ id: candidate.id, score })
  }

  scored.sort((a, b) => b.score - a.score)
  return scored
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')
    const userId = searchParams.get('userId')
    const userEmail = searchParams.get('userEmail')
    const personalized = searchParams.get('personalized') !== 'false'

    const userRow = await getUserRow(userId, userEmail)
    const likedIds = getStoredLikedProductIds(userRow)
    const shownIds = asMetadataStringArray(userRow?.metadata?.shown_product_ids)
    const likedProductsRaw = likedIds.length ? await getProductsByIds(likedIds) : []
    const likedProducts = likedProductsRaw.map((p) => ({
      ...transformPennProduct(p),
      is_liked: true,
    }))

    // Skip the expensive personalization path when the client wants a fast initial render.
    if (personalized && userRow) {
      const excluded = new Set<string>([...likedIds, ...shownIds])
      const topK = Math.max(limit + offset, limit)

      if (likedProductsRaw.length > 0) {
        try {
          const catalogProducts = await getCatalogProductsForHybrid(HYBRID_CATALOG_LIMIT)
          const hybridScored = await scoreProductsWithHybridCatalogApi(
            likedProductsRaw,
            catalogProducts,
            excluded,
            topK
          )

          if (hybridScored.length > 0) {
            const multimodalScored = await rerankHybridResultsWithImageSignals(hybridScored, likedIds)
            const pageScored = multimodalScored.slice(offset, offset + limit)
            const pageIds = pageScored.map((s) => s.id)
            const products = await getProductsByIds(pageIds)
            const scoreMap = new Map(pageScored.map((s) => [s.id, s.score]))

            const fashionProducts = products.map((p) => {
              const fp = transformPennProduct(p)
              const sim = scoreMap.get(p.id)
              return {
                ...fp,
                recommendation_score: typeof sim === 'number' ? toRecommendationScore(sim) : null,
                recommendation_reason:
                  typeof sim === 'number'
                    ? 'Recommended by multimodal ranker (text, image, and two-tower signals).'
                    : null,
              }
            })

            return NextResponse.json({
              success: true,
              products: fashionProducts,
              likedProducts,
              total: multimodalScored.length,
              limit,
              offset,
              mode: 'personalized_hybrid_api',
              engine: 'faiss_two_tower_image_hybrid' satisfies RecommendationEngine,
            })
          }
        } catch (error) {
          console.error('Hybrid catalog API path failed, falling back to local scorer:', error)
        }
      }

      // Fallback personalization path (image-only) when user embedding exists.
      const userVec = parseVector(userRow.user_image_embedding_avg)
      if (userVec && userVec.length > 0) {
        const scored = await scoreProductsByUserImageEmbedding(userVec, excluded)
        const pageScored = scored.slice(offset, offset + limit)
        const pageIds = pageScored.map((s) => s.id)

        const products = await getProductsByIds(pageIds)
        const scoreMap = new Map(pageScored.map((s) => [s.id, s.score]))

        const fashionProducts = products.map((p) => {
          const fp = transformPennProduct(p)
          const sim = scoreMap.get(p.id)
          return {
            ...fp,
            recommendation_score: typeof sim === 'number' ? Math.round(sim * 100) : null,
            recommendation_reason:
              typeof sim === 'number' ? 'Recommended from your liked picks using personalized ranking.' : null,
          }
        })

        return NextResponse.json({
          success: true,
          products: fashionProducts,
          likedProducts,
          total: scored.length,
          limit,
          offset,
          mode: 'personalized_image',
          engine: 'two_tower_or_cosine_fallback' satisfies RecommendationEngine,
        })
      }
    }

    // Fallback path: latest products.
    const { products, total } = await getLatestProducts(limit, offset)
    const fashionProducts = (products || []).map(transformPennProduct)

    return NextResponse.json({
      success: true,
      products: fashionProducts,
      likedProducts,
      total,
      limit,
      offset,
      mode: 'latest_fallback',
      engine: 'latest_products_fallback' satisfies RecommendationEngine,
    })
  } catch (error: any) {
    console.error('Products API error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
