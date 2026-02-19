import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import {
  l2Normalize,
  meanVectors,
  parseVector,
  vectorToPg,
} from '@/lib/recommendations/image-content'

// Use service role key for server-side operations
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

const MIN_LIKES_TO_STORE_EMBEDDING = 1
const USER_IMAGE_EMBEDDING_DIM = 512

async function computeUserImageEmbeddingAvgFromLikes(likedProductIds: string[]): Promise<number[] | null> {
  if (!likedProductIds.length) return null

  const { data, error } = await supabase
    .from('product_embeddings')
    .select('product_id,image_embedding')
    .in('product_id', likedProductIds)
    .not('image_embedding', 'is', null)

  if (error) {
    throw new Error(`Failed to load liked product embeddings: ${error.message}`)
  }

  const vectors: number[][] = []
  for (const row of data || []) {
    const vec = parseVector((row as any).image_embedding)
    if (vec && vec.length > 0) {
      vectors.push(l2Normalize(vec))
    }
  }

  if (!vectors.length) return null
  return l2Normalize(meanVectors(vectors))
}

export async function POST(request: Request) {
  try {
    const { email, password, userData } = await request.json()

    console.log('Signup request for:', email)

    // First, sign up with Supabase Auth
    const { data: authData, error: signUpError } = await supabase.auth.admin.createUser({
      email,
      password,
      email_confirm: true, // Auto-confirm for development - change to false in production
      user_metadata: {
        name: userData.name
      }
    })

    if (signUpError) {
      console.error('Auth signup error:', signUpError)
      return NextResponse.json(
        { success: false, message: signUpError.message },
        { status: 400 }
      )
    }

    if (!authData.user) {
      return NextResponse.json(
        { success: false, message: 'Failed to create auth user' },
        { status: 400 }
      )
    }

    console.log('Auth user created:', authData.user.id)

    // Check if email already exists in users table
    const { data: existingUsers } = await supabase
      .from('users')
      .select('id')
      .eq('email', email)
      .limit(1)

    if (existingUsers && existingUsers.length > 0) {
      // Clean up auth user if profile creation fails
      await supabase.auth.admin.deleteUser(authData.user.id)
      return NextResponse.json(
        { success: false, message: 'Email already taken' },
        { status: 400 }
      )
    }

    const likedProductIds: string[] = Array.isArray(userData?.likedProductIds)
      ? userData.likedProductIds.map((id: any) => String(id))
      : []
    const shownProductIds: string[] = Array.isArray(userData?.shownProductIds)
      ? userData.shownProductIds.map((id: any) => String(id))
      : []
    const recommendedProductIds: string[] = Array.isArray(userData?.recommendedProductIds)
      ? userData.recommendedProductIds.map((id: any) => String(id))
      : []

    const userImageEmbeddingAvg = likedProductIds.length >= MIN_LIKES_TO_STORE_EMBEDDING
      ? await computeUserImageEmbeddingAvgFromLikes(likedProductIds)
      : null

    const baseInsertPayload: any = {
      id: authData.user.id, // Use Auth user ID as primary key
      handle: userData.name,
      email: email,
      metadata: {
        style_preferences: userData.stylePreferences || [],
        budget_range: userData.budgetRange || '',
        favorite_colors: userData.favoriteColors || '',
        bio: userData.bio || '',
        liked_product_ids: likedProductIds,
        shown_product_ids: shownProductIds,
        recommended_product_ids: recommendedProductIds,
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    if (userImageEmbeddingAvg && userImageEmbeddingAvg.length === USER_IMAGE_EMBEDDING_DIM) {
      baseInsertPayload.user_image_embedding_avg = vectorToPg(userImageEmbeddingAvg)
    }

    // Insert user data into the users table
    let insertResult = await supabase
      .from('users')
      .insert(baseInsertPayload)
      .select()
      .single()

    // Fallback for environments that have not applied the new embedding column migration yet.
    if (insertResult.error && insertResult.error.message?.includes('user_image_embedding_avg')) {
      const fallbackPayload = { ...baseInsertPayload }
      delete fallbackPayload.user_image_embedding_avg
      insertResult = await supabase
        .from('users')
        .insert(fallbackPayload)
        .select()
        .single()
    }

    const { data: newUser, error: insertError } = insertResult

    if (insertError) {
      console.error('Error inserting user:', insertError)
      // Clean up auth user if profile creation fails
      await supabase.auth.admin.deleteUser(authData.user.id)

      if (insertError.code === '23505') { // Unique violation
        return NextResponse.json(
          { success: false, message: 'Email already taken' },
          { status: 400 }
        )
      }
      return NextResponse.json(
        { success: false, message: `Failed to create user profile: ${insertError.message}` },
        { status: 500 }
      )
    }

    console.log('User profile created:', newUser.id)

    return NextResponse.json({
      success: true,
      message: 'Account created successfully! Redirecting to onboarding...'
    })
  } catch (error: any) {
    console.error('Signup error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
