import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { vectorToPg } from '@/lib/recommendations/image-content'
import {
  addLikedProductIds,
  computeUserImageEmbeddingFromLikedProducts,
  getStoredLikedProductIds,
  removeLikedProductId,
} from '@/lib/users/liked-products'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

type UserLikesRow = {
  id: string
  metadata: Record<string, unknown> | null
  liked_product_ids: string[] | null
}

function getMetadataObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  return { ...(value as Record<string, unknown>) }
}

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 })
    }

    const token = authHeader.substring(7)
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser(token)

    if (authError || !user) {
      return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 })
    }

    const body = await request.json()
    const productId = String(body?.productId || '').trim()
    const action = body?.action === 'unlike' ? 'unlike' : 'like'

    if (!productId) {
      return NextResponse.json({ success: false, message: 'productId is required' }, { status: 400 })
    }

    const { data: userRow, error: userError } = await supabase
      .from('users')
      .select('id,metadata,liked_product_ids')
      .eq('id', user.id)
      .maybeSingle()

    if (userError) {
      throw new Error(`Failed to load user likes: ${userError.message}`)
    }

    if (!userRow) {
      return NextResponse.json({ success: false, message: 'User profile not found' }, { status: 404 })
    }

    const existingLikedProductIds = getStoredLikedProductIds(userRow as UserLikesRow)
    const nextLikedProductIds =
      action === 'unlike'
        ? removeLikedProductId(existingLikedProductIds, productId)
        : addLikedProductIds(existingLikedProductIds, [productId])

    const metadata = getMetadataObject((userRow as UserLikesRow).metadata)
    metadata.liked_product_ids = nextLikedProductIds

    const userImageEmbeddingAvg = await computeUserImageEmbeddingFromLikedProducts(supabase, nextLikedProductIds)

    const updatePayload: Record<string, unknown> = {
      liked_product_ids: nextLikedProductIds,
      metadata,
      updated_at: new Date().toISOString(),
      user_image_embedding_avg: userImageEmbeddingAvg ? vectorToPg(userImageEmbeddingAvg) : null,
    }
    const retryablePayload: Record<string, unknown> = { ...updatePayload }

    let updateResult = await supabase
      .from('users')
      .update(retryablePayload)
      .eq('id', user.id)
      .select('id,liked_product_ids')
      .single()

    if (updateResult.error && updateResult.error.message?.includes('liked_product_ids')) {
      delete retryablePayload.liked_product_ids
      updateResult = await supabase
        .from('users')
        .update(retryablePayload)
        .eq('id', user.id)
        .select('id')
        .single()
    }

    if (updateResult.error && updateResult.error.message?.includes('user_image_embedding_avg')) {
      delete retryablePayload.user_image_embedding_avg
      updateResult = await supabase
        .from('users')
        .update(retryablePayload)
        .eq('id', user.id)
        .select('id')
        .single()
    }

    if (updateResult.error) {
      throw new Error(`Failed to update user likes: ${updateResult.error.message}`)
    }

    return NextResponse.json({
      success: true,
      likedProductIds: nextLikedProductIds,
      totalLikes: nextLikedProductIds.length,
      action,
    })
  } catch (error: any) {
    console.error('[users/likes][POST] error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'Failed to update user likes' },
      { status: 500 }
    )
  }
}
