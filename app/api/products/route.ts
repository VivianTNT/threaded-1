import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { createHash } from 'crypto'
import { transformPennProduct, PennProduct } from '@/lib/penn-products'
import { cosineSimilarity, l2Normalize, parseVector } from '@/lib/recommendations/image-content'
import { rankWithTwoTower } from '@/lib/recommendations/model-service'
import {
  computeUserImageEmbeddingFromLikedProducts,
  computeUserImageEmbeddingFromVectors,
  getStoredLikedProductIds,
} from '@/lib/users/liked-products'

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

type PersonalizedRecommendationMode = 'personalized_hybrid_api' | 'personalized_image'
type RecommendationRequestStrategy = 'hybrid' | 'content'

type RecommendationSnapshotRow = {
  user_id: string
  request_strategy: RecommendationRequestStrategy
  snapshot_key: string
  catalog_marker: string | null
  mode: PersonalizedRecommendationMode
  engine: RecommendationEngine
  recommended_product_ids: string[] | null
  recommendation_scores: Record<string, number> | null
  total: number | null
  expires_at: string
}

type RecommendationSnapshotPayload = {
  userId: string
  requestStrategy: RecommendationRequestStrategy
  snapshotKey: string
  catalogMarker: string | null
  mode: PersonalizedRecommendationMode
  engine: RecommendationEngine
  recommendedProductIds: string[]
  recommendationScores: Record<string, number>
  total: number
}

type PersonalizedRecommendationResult = {
  recommendedProductIds: string[]
  recommendationScores: Record<string, number>
  total: number
  mode: PersonalizedRecommendationMode
  engine: RecommendationEngine
}

const DEFAULT_RECSYS_BASE_URL = 'http://127.0.0.1:8000'
const HYBRID_CATALOG_LIMIT = 3000
const HYBRID_API_TIMEOUT_MS = parseInt(process.env.RECSYS_TIMEOUT_MS || '60000', 10)
const SUPABASE_SELECT_PAGE_SIZE = 500
const SUPABASE_IN_FILTER_BATCH_SIZE = 200
const DEFAULT_RECOMMENDATION_SNAPSHOT_TTL_MINUTES = 30
const DEFAULT_RECOMMENDATION_SNAPSHOT_LIMIT = 200
const RECOMMENDATION_SNAPSHOT_SCHEMA_VERSION = 3

function chunkArray<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size))
  }
  return chunks
}

function shuffleInPlace<T>(items: T[]): void {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1))
    const current = items[index]
    items[index] = items[swapIndex]
    items[swapIndex] = current
  }
}

function getRecsysBaseUrl(): string {
  const raw = process.env.RECSYS_API_URL || DEFAULT_RECSYS_BASE_URL
  return raw.replace(/\/+$/, '')
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback
  return Math.floor(parsed)
}

function getRecommendationSnapshotTtlMinutes(): number {
  return parsePositiveInt(
    process.env.RECOMMENDATION_SNAPSHOT_TTL_MINUTES,
    DEFAULT_RECOMMENDATION_SNAPSHOT_TTL_MINUTES
  )
}

function getRecommendationSnapshotLimit(): number {
  return parsePositiveInt(
    process.env.RECOMMENDATION_SNAPSHOT_LIMIT,
    DEFAULT_RECOMMENDATION_SNAPSHOT_LIMIT
  )
}

function asMetadataStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((v) => String(v))
}

function isMissingSnapshotTableError(message: string | undefined): boolean {
  const normalized = String(message || '').toLowerCase()
  return (
    normalized.includes('user_recommendation_snapshots') ||
    normalized.includes('recommendation_snapshot') ||
    normalized.includes('request_strategy') ||
    normalized.includes('on conflict specification') ||
    normalized.includes('no unique or exclusion constraint')
  )
}

function normalizeRecommendationRequestStrategy(
  raw: string | null | undefined
): RecommendationRequestStrategy {
  return raw === 'content' ? 'content' : 'hybrid'
}

function buildRecommendationSnapshotKey(
  likedIds: string[],
  shownIds: string[],
  requestStrategy: RecommendationRequestStrategy
): string {
  return createHash('sha256')
    .update(
      JSON.stringify({
        version: RECOMMENDATION_SNAPSHOT_SCHEMA_VERSION,
        requestStrategy,
        likedIds,
        shownIds,
      })
    )
    .digest('hex')
}

function getRecommendationReason(mode: PersonalizedRecommendationMode): string {
  return mode === 'personalized_hybrid_api'
    ? 'Recommended by multimodal ranker (text, image, and two-tower signals).'
    : 'Recommended from your liked picks using personalized ranking.'
}

function toRecommendationScoreRecord(rows: ScoredId[]): Record<string, number> {
  const out: Record<string, number> = {}
  for (const row of rows) {
    if (!row.id || !Number.isFinite(row.score)) continue
    out[row.id] = row.score
  }
  return out
}

function toDisplayRecommendationScore(mode: PersonalizedRecommendationMode, score: number): number {
  return mode === 'personalized_hybrid_api' ? toRecommendationScore(score) : Math.round(score * 100)
}

async function getCatalogMarker(): Promise<string | null> {
  const { data, error } = await supabase
    .from('products')
    .select('created_at')
    .order('created_at', { ascending: false })
    .limit(1)

  if (error) {
    throw new Error(`Failed to fetch catalog marker: ${error.message}`)
  }

  return data?.[0]?.created_at || null
}

async function getRecommendationSnapshot(
  userId: string,
  requestStrategy: RecommendationRequestStrategy
): Promise<RecommendationSnapshotRow | null> {
  const { data, error } = await supabase
    .from('user_recommendation_snapshots')
    .select(
      'user_id,request_strategy,snapshot_key,catalog_marker,mode,engine,recommended_product_ids,recommendation_scores,total,expires_at'
    )
    .eq('user_id', userId)
    .eq('request_strategy', requestStrategy)
    .maybeSingle()

  if (error) {
    if (isMissingSnapshotTableError(error.message)) return null
    throw new Error(`Failed to load recommendation snapshot: ${error.message}`)
  }

  return (data as RecommendationSnapshotRow | null) || null
}

async function saveRecommendationSnapshot(payload: RecommendationSnapshotPayload): Promise<void> {
  const expiresAt = new Date(Date.now() + getRecommendationSnapshotTtlMinutes() * 60 * 1000).toISOString()
  const { error } = await supabase
    .from('user_recommendation_snapshots')
    .upsert(
      {
        user_id: payload.userId,
        request_strategy: payload.requestStrategy,
        snapshot_key: payload.snapshotKey,
        catalog_marker: payload.catalogMarker,
        mode: payload.mode,
        engine: payload.engine,
        recommended_product_ids: payload.recommendedProductIds,
        recommendation_scores: payload.recommendationScores,
        total: payload.total,
        computed_at: new Date().toISOString(),
        expires_at: expiresAt,
      },
      { onConflict: 'user_id,request_strategy' }
    )

  if (error && !isMissingSnapshotTableError(error.message)) {
    throw new Error(`Failed to save recommendation snapshot: ${error.message}`)
  }
}

function isRecommendationSnapshotUsable(
  snapshot: RecommendationSnapshotRow | null,
  snapshotKey: string,
  catalogMarker: string | null,
  requiredCount: number
): boolean {
  if (!snapshot) return false
  if (snapshot.snapshot_key !== snapshotKey) return false
  if ((snapshot.catalog_marker || null) !== catalogMarker) return false
  if (!snapshot.expires_at || new Date(snapshot.expires_at).getTime() <= Date.now()) return false
  if ((snapshot.recommended_product_ids || []).length < requiredCount) return false
  return true
}

async function hydrateRecommendationProducts(
  recommendedProductIds: string[],
  recommendationScores: Record<string, number>,
  mode: PersonalizedRecommendationMode,
  offset: number,
  limit: number
): Promise<ReturnType<typeof transformPennProduct>[]> {
  const pageIds = recommendedProductIds.slice(offset, offset + limit)
  const products = await getProductsByIds(pageIds)
  const reason = getRecommendationReason(mode)

  return products.map((p) => {
    const fp = transformPennProduct(p)
    const rawScore = recommendationScores[p.id]
    return {
      ...fp,
      recommendation_score: Number.isFinite(rawScore) ? toDisplayRecommendationScore(mode, rawScore) : null,
      recommendation_reason: Number.isFinite(rawScore) ? reason : null,
    }
  })
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
    .select('id')

  if (error) {
    throw new Error(`Failed to fetch hybrid catalog product ids: ${error.message}`)
  }

  const allIds = ((data as Array<{ id: string }> | null) || []).map((row) => String(row.id))
  if (allIds.length <= limit) {
    return getProductsByIds(allIds)
  }

  shuffleInPlace(allIds)
  return getProductsByIds(allIds.slice(0, limit))
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
  userImageSignal: number[] | null
): Promise<ScoredId[]> {
  if (!hybridScored.length || !userImageSignal?.length) return hybridScored

  const imageUserVec = l2Normalize(userImageSignal)
  const embeddings = await getProductEmbeddingsByIds(hybridScored.map((row) => row.id))

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

async function computePersonalizedRecommendations(
  likedProductsRaw: PennProduct[],
  likedIds: string[],
  shownIds: string[],
  userRow: UserRecRow,
  requiredCount: number,
  requestStrategy: RecommendationRequestStrategy
): Promise<PersonalizedRecommendationResult | null> {
  const snapshotLimit = Math.max(requiredCount, getRecommendationSnapshotLimit())
  const excluded = new Set<string>([...likedIds, ...shownIds])
  const likedProductUserVec = likedIds.length
    ? await computeUserImageEmbeddingFromLikedProducts(supabase, likedIds)
    : null
  const uploadedImageUserVec = parseVector(userRow.metadata?.uploaded_image_embedding_avg)
  const recomputedUserVec = await computeUserImageEmbeddingFromVectors(
    [likedProductUserVec, uploadedImageUserVec].filter((vec): vec is number[] => Array.isArray(vec) && vec.length > 0)
  )
  const userVec = recomputedUserVec && recomputedUserVec.length > 0
    ? recomputedUserVec
    : parseVector(userRow.user_image_embedding_avg)

  if (requestStrategy === 'hybrid' && likedProductsRaw.length > 0) {
    try {
      const catalogProducts = await getCatalogProductsForHybrid(HYBRID_CATALOG_LIMIT)
      const hybridScored = await scoreProductsWithHybridCatalogApi(
        likedProductsRaw,
        catalogProducts,
        excluded,
        snapshotLimit
      )

      if (hybridScored.length > 0) {
        const multimodalScored = await rerankHybridResultsWithImageSignals(hybridScored, userVec)
        const storedRows = multimodalScored.slice(0, snapshotLimit)

        return {
          recommendedProductIds: storedRows.map((row) => row.id),
          recommendationScores: toRecommendationScoreRecord(storedRows),
          total: multimodalScored.length,
          mode: 'personalized_hybrid_api',
          engine: 'faiss_two_tower_image_hybrid',
        }
      }
    } catch (error) {
      console.error('Hybrid catalog API path failed, falling back to local scorer:', error)
    }
  }

  if (userVec && userVec.length > 0) {
    const scored = await scoreProductsByUserImageEmbedding(userVec, excluded)
    const storedRows = scored.slice(0, snapshotLimit)

    return {
      recommendedProductIds: storedRows.map((row) => row.id),
      recommendationScores: toRecommendationScoreRecord(storedRows),
      total: scored.length,
      mode: 'personalized_image',
      engine: 'two_tower_or_cosine_fallback',
    }
  }

  return null
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')
    const userId = searchParams.get('userId')
    const userEmail = searchParams.get('userEmail')
    const requestStrategy = normalizeRecommendationRequestStrategy(searchParams.get('strategy'))
    const forceRefresh = searchParams.get('forceRefresh') === '1'

    const userRow = await getUserRow(userId, userEmail)
    const likedIds = getStoredLikedProductIds(userRow)
    const shownIds = asMetadataStringArray(userRow?.metadata?.shown_product_ids)
    const likedProductsRaw = likedIds.length ? await getProductsByIds(likedIds) : []
    const likedProducts = likedProductsRaw.map((p) => ({
      ...transformPennProduct(p),
      is_liked: true,
    }))

    // Personalization path (hybrid recommender API) using liked products + catalog.
    if (userRow) {
      const requiredCount = Math.max(limit + offset, limit)
      const catalogMarker = await getCatalogMarker()
      const snapshotKey = buildRecommendationSnapshotKey(likedIds, shownIds, requestStrategy)
      const snapshot = forceRefresh ? null : await getRecommendationSnapshot(userRow.id, requestStrategy)

      if (isRecommendationSnapshotUsable(snapshot, snapshotKey, catalogMarker, requiredCount)) {
        const fashionProducts = await hydrateRecommendationProducts(
          snapshot!.recommended_product_ids || [],
          snapshot!.recommendation_scores || {},
          snapshot!.mode,
          offset,
          limit
        )

        return NextResponse.json({
          success: true,
          products: fashionProducts,
          likedProducts,
          total: snapshot?.total || (snapshot?.recommended_product_ids || []).length,
          limit,
          offset,
          mode: snapshot!.mode,
          engine: snapshot!.engine,
          cached: true,
        })
      }

      const personalized = await computePersonalizedRecommendations(
        likedProductsRaw,
        likedIds,
        shownIds,
        userRow,
        requiredCount,
        requestStrategy
      )

      if (personalized) {
        await saveRecommendationSnapshot({
          userId: userRow.id,
          requestStrategy,
          snapshotKey,
          catalogMarker,
          mode: personalized.mode,
          engine: personalized.engine,
          recommendedProductIds: personalized.recommendedProductIds,
          recommendationScores: personalized.recommendationScores,
          total: personalized.total,
        })

        const fashionProducts = await hydrateRecommendationProducts(
          personalized.recommendedProductIds,
          personalized.recommendationScores,
          personalized.mode,
          offset,
          limit
        )

        return NextResponse.json({
          success: true,
          products: fashionProducts,
          likedProducts,
          total: personalized.total,
          limit,
          offset,
          mode: personalized.mode,
          engine: personalized.engine,
          cached: false,
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
