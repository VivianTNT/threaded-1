import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { transformPennProduct, PennProduct } from '@/lib/penn-products'
import { cosineSimilarity, l2Normalize, parseVector } from '@/lib/recommendations/image-content'
import { rankWithTwoTower } from '@/lib/recommendations/model-service'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseKey)

type UserRecRow = {
  id: string
  email: string | null
  metadata: any
  user_image_embedding_avg: unknown
}

type ProductEmbeddingRow = {
  product_id: string
  image_embedding: unknown
}

type ScoredId = { id: string; score: number }
type HybridCatalogResult = { product_id: string | number; score: number }

const DEFAULT_RECSYS_BASE_URL = 'http://127.0.0.1:8000'
const HYBRID_CATALOG_LIMIT = 1000

function getRecsysBaseUrl(): string {
  const raw = process.env.RECSYS_API_URL || DEFAULT_RECSYS_BASE_URL
  return raw.replace(/\/+$/, '')
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((v) => String(v))
}

async function getUserRow(userId: string | null, userEmail: string | null): Promise<UserRecRow | null> {
  if (!userId && !userEmail) return null

  let query = supabase
    .from('users')
    .select('id,email,metadata,user_image_embedding_avg')
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
  const { data, error } = await supabase
    .from('products')
    .select('*')
    .in('id', ids)

  if (error) {
    throw new Error(`Failed to load products by ids: ${error.message}`)
  }

  const map = new Map<string, PennProduct>((data as PennProduct[] || []).map((p) => [p.id, p]))
  return ids.map((id) => map.get(id)).filter((p): p is PennProduct => Boolean(p))
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
  const timeout = setTimeout(() => controller.abort(), 20000)
  try {
    const response = await fetch(`${baseUrl}/recommend/hybrid/catalog`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new Error(`Hybrid API error (${response.status})`)
    }

    const json = await response.json()
    const recs: HybridCatalogResult[] = Array.isArray(json?.recommendations) ? json.recommendations : []

    return recs
      .map((r) => ({
        id: String(r.product_id),
        score: Number(r.score),
      }))
      .filter((r) => Number.isFinite(r.score))
  } finally {
    clearTimeout(timeout)
  }
}

async function scoreProductsByUserImageEmbedding(
  userImageEmbedding: number[],
  excludedIds: Set<string>
): Promise<ScoredId[]> {
  const userVec = l2Normalize(userImageEmbedding)
  const { data, error } = await supabase
    .from('product_embeddings')
    .select('product_id,image_embedding')
    .not('image_embedding', 'is', null)

  if (error) {
    throw new Error(`Failed to load product embeddings: ${error.message}`)
  }

  const candidates: Array<{ id: string; embedding: number[] }> = []
  for (const row of (data || []) as ProductEmbeddingRow[]) {
    const id = String(row.product_id)
    if (excludedIds.has(id)) continue

    const vec = parseVector(row.image_embedding)
    if (!vec || !vec.length) continue
    const normalizedVec = l2Normalize(vec)
    candidates.push({ id, embedding: normalizedVec })
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

    const userRow = await getUserRow(userId, userEmail)
    const likedIds = asStringArray(userRow?.metadata?.liked_product_ids)
    const shownIds = asStringArray(userRow?.metadata?.shown_product_ids)
    const likedProductsRaw = likedIds.length ? await getProductsByIds(likedIds) : []
    const likedProducts = likedProductsRaw.map((p) => ({
      ...transformPennProduct(p),
      is_liked: true,
    }))

    // Personalization path (hybrid recommender API) using liked products + catalog.
    if (userRow) {
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
            const pageScored = hybridScored.slice(offset, offset + limit)
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
                    ? 'Recommended by hybrid ranker (content + two-tower).'
                    : null,
              }
            })

            return NextResponse.json({
              success: true,
              products: fashionProducts,
              likedProducts,
              total: hybridScored.length,
              limit,
              offset,
              mode: 'personalized_hybrid_api',
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
    })
  } catch (error: any) {
    console.error('Products API error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
