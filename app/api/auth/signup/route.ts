import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { vectorToPg } from '@/lib/recommendations/image-content'
import {
  addLikedProductIds,
  computeUserImageEmbeddingFromLikedProducts,
} from '@/lib/users/liked-products'

// Use service role key for server-side operations
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

const MIN_LIKES_TO_STORE_EMBEDDING = 1

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
    const storedLikedProductIds = addLikedProductIds([], likedProductIds)

    const userImageEmbeddingAvg = storedLikedProductIds.length >= MIN_LIKES_TO_STORE_EMBEDDING
      ? await computeUserImageEmbeddingFromLikedProducts(supabase, storedLikedProductIds)
      : null

    const baseInsertPayload: any = {
      id: authData.user.id, // Use Auth user ID as primary key
      handle: userData.name,
      email: email,
      liked_product_ids: storedLikedProductIds,
      metadata: {
        style_preferences: userData.stylePreferences || [],
        budget_range: userData.budgetRange || '',
        favorite_colors: userData.favoriteColors || '',
        bio: userData.bio || '',
        liked_product_ids: storedLikedProductIds,
        shown_product_ids: shownProductIds,
        recommended_product_ids: recommendedProductIds,
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    if (userImageEmbeddingAvg && userImageEmbeddingAvg.length > 0) {
      baseInsertPayload.user_image_embedding_avg = vectorToPg(userImageEmbeddingAvg)
    }

    // Insert user data into the users table
    let insertResult = await supabase
      .from('users')
      .insert(baseInsertPayload)
      .select()
      .single()

    // Fallback for environments that have not applied new profile columns yet.
    if (
      insertResult.error &&
      (
        insertResult.error.message?.includes('user_image_embedding_avg') ||
        insertResult.error.message?.includes('liked_product_ids')
      )
    ) {
      const fallbackPayload = { ...baseInsertPayload }
      delete fallbackPayload.user_image_embedding_avg
      delete fallbackPayload.liked_product_ids
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
