import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import {
  DEFAULT_SIGNUP_MIN_LIKES,
  DEFAULT_SIGNUP_SAMPLE_SIZE,
  DEFAULT_SIGNUP_TOP_K,
  EmbeddingRow,
  ProductRow,
  cosineSimilarity,
  l2Normalize,
  meanVectors,
  parseVector,
  shuffleInPlace,
  toProductCard,
} from '@/lib/recommendations/image-content'
import { buildUserEmbeddingWithTwoTower, rankWithTwoTower } from '@/lib/recommendations/model-service'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

const SIGNUP_SAMPLE_SIZE = DEFAULT_SIGNUP_SAMPLE_SIZE
const SIGNUP_MIN_LIKES = DEFAULT_SIGNUP_MIN_LIKES
const SIGNUP_TOP_K = DEFAULT_SIGNUP_TOP_K
const DEFAULT_RECSYS_BASE_URL = 'http://127.0.0.1:8000'
const SIGNUP_HYBRID_CATALOG_LIMIT = 1000

type HybridCatalogResult = { product_id: string | number; score: number }

function getRecsysBaseUrl(): string {
  const raw = process.env.RECSYS_API_URL || DEFAULT_RECSYS_BASE_URL
  return raw.replace(/\/+$/, '')
}

async function loadAllImageEmbeddings(): Promise<Map<string, number[]>> {
  const { data, error } = await supabase
    .from('product_embeddings')
    .select('product_id,image_embedding')
    .not('image_embedding', 'is', null)

  if (error) {
    throw new Error(`Failed to load product embeddings: ${error.message}`)
  }

  const map = new Map<string, number[]>()
  for (const row of (data || []) as EmbeddingRow[]) {
    const productId = String(row.product_id)
    const vec = parseVector(row.image_embedding)
    if (vec && vec.length > 0) {
      map.set(productId, l2Normalize(vec))
    }
  }
  return map
}

async function loadProductsByIds(ids: string[]): Promise<ProductRow[]> {
  if (!ids.length) return []
  const { data, error } = await supabase
    .from('products')
    .select('id,name,brand_name,image_url,price,product_url,category,description')
    .in('id', ids)

  if (error) {
    throw new Error(`Failed to load products: ${error.message}`)
  }
  return (data || []) as ProductRow[]
}

async function loadLatestProductsForHybrid(limit: number): Promise<ProductRow[]> {
  const { data, error } = await supabase
    .from('products')
    .select('id,name,brand_name,image_url,price,product_url,category,description')
    .not('image_url', 'is', null)
    .order('created_at', { ascending: false })
    .limit(limit)

  if (error) {
    throw new Error(`Failed to load hybrid signup catalog: ${error.message}`)
  }

  return (data || []) as ProductRow[]
}

async function scoreProductsWithHybridCatalogApi(
  likedProducts: ProductRow[],
  catalogProducts: ProductRow[],
  excludedIds: Set<string>,
  topK: number
): Promise<Array<{ id: string; score: number }>> {
  if (!likedProducts.length || !catalogProducts.length) return []

  const response = await fetch(`${getRecsysBaseUrl()}/recommend/hybrid/catalog`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      liked_products: likedProducts.map((p) => ({
        id: p.id,
        title: p.name || '',
        name: p.name || '',
        description: p.description || '',
      })),
      catalog_products: catalogProducts
        .filter((p) => !excludedIds.has(String(p.id)))
        .map((p) => ({
          id: p.id,
          title: p.name || '',
          name: p.name || '',
          description: p.description || '',
        })),
      top_k: topK,
      strategy: 'hybrid',
      exclude_ids: Array.from(excludedIds),
    }),
  })

  if (!response.ok) {
    throw new Error(`Hybrid API error (${response.status})`)
  }

  const json = await response.json()
  const recs: HybridCatalogResult[] = Array.isArray(json?.recommendations) ? json.recommendations : []
  return recs
    .map((row) => ({ id: String(row.product_id), score: Number(row.score) }))
    .filter((row) => row.id && Number.isFinite(row.score))
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const requested = Number(searchParams.get('sampleSize') || SIGNUP_SAMPLE_SIZE)
    const sampleSize = Number.isFinite(requested) && requested > 0 ? Math.floor(requested) : SIGNUP_SAMPLE_SIZE

    const embeddingMap = await loadAllImageEmbeddings()
    const embeddingIds = Array.from(embeddingMap.keys())
    if (!embeddingIds.length) {
      return NextResponse.json({ success: true, products: [], sampleSize: 0 })
    }

    // Fetch all product cards that have embeddings, then random sample.
    const allProducts = await loadProductsByIds(embeddingIds)
    const eligible = allProducts.filter((p) => Boolean(p.image_url))
    shuffleInPlace(eligible)

    const sampled = eligible.slice(0, Math.min(sampleSize, eligible.length)).map(toProductCard)
    const embeddingDim = embeddingMap.values().next().value?.length || null

    return NextResponse.json({
      success: true,
      products: sampled,
      sampleSize: sampled.length,
      embeddingDim,
    })
  } catch (error: any) {
    console.error('[signup/image-recommendations][GET] error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'Failed to load signup sample' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const likedProductIds: string[] = Array.isArray(body?.likedProductIds) ? body.likedProductIds.map(String) : []
    const shownProductIds: string[] = Array.isArray(body?.shownProductIds) ? body.shownProductIds.map(String) : []
    const requestedTopK = Number(body?.topK || SIGNUP_TOP_K)
    const topK = Number.isFinite(requestedTopK) && requestedTopK > 0 ? Math.floor(requestedTopK) : SIGNUP_TOP_K

    if (likedProductIds.length < SIGNUP_MIN_LIKES) {
      return NextResponse.json(
        {
          success: false,
          message: `Please select at least ${SIGNUP_MIN_LIKES} liked items.`,
          minLikes: SIGNUP_MIN_LIKES,
        },
        { status: 400 }
      )
    }

    const likedProductsRaw = await loadProductsByIds(likedProductIds)
    const excluded = new Set<string>([...likedProductIds, ...shownProductIds])

    if (likedProductsRaw.length > 0) {
      try {
        const hybridCatalog = await loadLatestProductsForHybrid(SIGNUP_HYBRID_CATALOG_LIMIT)
        const hybridScored = await scoreProductsWithHybridCatalogApi(
          likedProductsRaw,
          hybridCatalog,
          excluded,
          topK
        )

        if (hybridScored.length > 0) {
          const recProductsRaw = await loadProductsByIds(hybridScored.map((s) => s.id))
          const recMap = new Map(recProductsRaw.map((p) => [p.id, p]))
          const recommendations = hybridScored
            .map((r) => {
              const row = recMap.get(r.id)
              if (!row) return null
              return {
                ...toProductCard(row),
                similarity: r.score,
              }
            })
            .filter((v): v is ReturnType<typeof toProductCard> & { similarity: number } => v !== null)

          return NextResponse.json({
            success: true,
            likedProducts: likedProductsRaw.map(toProductCard),
            recommendations,
            recommendedProductIds: recommendations.map((r) => r.id),
            userImageEmbeddingAvg: null,
            embeddingDim: null,
            mode: 'signup_hybrid_api',
            tunables: {
              sampleSize: SIGNUP_SAMPLE_SIZE,
              minLikes: SIGNUP_MIN_LIKES,
              topK,
            },
          })
        }
      } catch (error) {
        console.error('[signup/image-recommendations][POST] hybrid API failed, falling back:', error)
      }
    }

    const embeddingMap = await loadAllImageEmbeddings()
    const likedVecs = likedProductIds
      .map((id) => embeddingMap.get(id))
      .filter((v): v is number[] => Array.isArray(v) && v.length > 0)

    if (!likedVecs.length) {
      return NextResponse.json(
        { success: false, message: 'No valid image embeddings found for liked products.' },
        { status: 400 }
      )
    }

    const sourceDim = likedVecs[0]?.length || 0
    const modelUserVec = await buildUserEmbeddingWithTwoTower(likedVecs)
    const userVec = l2Normalize(
      modelUserVec && modelUserVec.length === sourceDim ? modelUserVec : meanVectors(likedVecs)
    )
    const candidates: Array<{ id: string; embedding: number[] }> = []
    for (const [productId, vec] of embeddingMap.entries()) {
      if (excluded.has(productId)) continue
      candidates.push({ id: productId, embedding: vec })
    }

    let scored: Array<{ id: string; score: number }> = []
    const modelRanked = await rankWithTwoTower(userVec, candidates, topK)
    if (modelRanked && modelRanked.length > 0) {
      scored = modelRanked.map((row) => ({ id: row.id, score: row.score }))
    } else {
      for (const candidate of candidates) {
        if (candidate.embedding.length !== userVec.length) continue
        scored.push({ id: candidate.id, score: cosineSimilarity(userVec, candidate.embedding) })
      }
      scored.sort((a, b) => b.score - a.score)
    }

    const topScored = scored.slice(0, topK)

    const recProductsRaw = await loadProductsByIds(topScored.map((s) => s.id))

    const likedMap = new Map(likedProductsRaw.map((p) => [p.id, p]))
    const recMap = new Map(recProductsRaw.map((p) => [p.id, p]))

    const likedProducts = likedProductIds
      .map((id) => likedMap.get(id))
      .filter((p): p is ProductRow => Boolean(p))
      .map(toProductCard)

    const recommendations = topScored
      .map((r) => {
        const row = recMap.get(r.id)
        if (!row) return null
        return {
          ...toProductCard(row),
          similarity: r.score,
        }
      })
      .filter((v): v is ReturnType<typeof toProductCard> & { similarity: number } => v !== null)

    return NextResponse.json({
      success: true,
      likedProducts,
      recommendations,
      recommendedProductIds: recommendations.map((r) => r.id),
      userImageEmbeddingAvg: userVec,
      embeddingDim: userVec.length,
      tunables: {
        sampleSize: SIGNUP_SAMPLE_SIZE,
        minLikes: SIGNUP_MIN_LIKES,
        topK,
      },
    })
  } catch (error: any) {
    console.error('[signup/image-recommendations][POST] error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'Failed to generate recommendations' },
      { status: 500 }
    )
  }
}
