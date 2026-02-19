import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { transformPennProduct, PennProduct } from '@/lib/penn-products'
import { cosineSimilarity, l2Normalize, parseVector } from '@/lib/recommendations/image-content'

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

  const scored: ScoredId[] = []
  for (const row of (data || []) as ProductEmbeddingRow[]) {
    const id = String(row.product_id)
    if (excludedIds.has(id)) continue

    const vec = parseVector(row.image_embedding)
    if (!vec || !vec.length || vec.length !== userVec.length) continue
    const score = cosineSimilarity(userVec, l2Normalize(vec))
    scored.push({ id, score })
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

    // Personalization path (image-only) when user embedding exists.
    if (userRow) {
      const userVec = parseVector(userRow.user_image_embedding_avg)
      const excluded = new Set<string>([...likedIds, ...shownIds])

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
              typeof sim === 'number' ? 'Recommended based on image similarity to your liked picks.' : null,
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
