import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { vectorToPg } from '@/lib/recommendations/image-content'
import {
  addLikedProductIds,
  computeUserImageEmbeddingFromLikedProducts,
  computeUserImageEmbeddingFromVectors,
} from '@/lib/users/liked-products'

// Use service role key for server-side operations
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

const MIN_LIKES_TO_STORE_EMBEDDING = 1
const DEFAULT_RECSYS_BASE_URL = 'http://127.0.0.1:8000'
const SIGNUP_IMAGE_UPLOAD_LIMIT = 5
const SIGNUP_IMAGE_MAX_BYTES = 10 * 1024 * 1024

function getRecsysBaseUrl(): string {
  const raw = process.env.RECSYS_API_URL || DEFAULT_RECSYS_BASE_URL
  return raw.replace(/\/+$/, '')
}

function sanitizeFileName(fileName: string): string {
  return fileName.replace(/[^a-zA-Z0-9._-]+/g, '-')
}

async function parseSignupPayload(request: Request): Promise<{
  email: string
  password: string
  userData: any
  uploadedClothingPhotos: File[]
}> {
  const contentType = request.headers.get('content-type') || ''

  if (contentType.includes('multipart/form-data')) {
    const formData = await request.formData()
    const email = String(formData.get('email') || '')
    const password = String(formData.get('password') || '')
    const userDataRaw = String(formData.get('userData') || '{}')
    let userData: any = {}

    try {
      userData = JSON.parse(userDataRaw)
    } catch {
      userData = {}
    }

    const uploadedClothingPhotos = formData
      .getAll('uploadedClothingPhotos')
      .filter((value): value is File => value instanceof File)

    return { email, password, userData, uploadedClothingPhotos }
  }

  const { email, password, userData } = await request.json()
  return { email, password, userData, uploadedClothingPhotos: [] }
}

async function uploadSignupClothingPhotos(userId: string, files: File[]): Promise<string[]> {
  const uploadedPaths: string[] = []

  for (const file of files.slice(0, SIGNUP_IMAGE_UPLOAD_LIMIT)) {
    const storagePath = `${userId}/signup-wardrobe/${Date.now()}-${crypto.randomUUID()}-${sanitizeFileName(file.name || 'upload.jpg')}`
    const arrayBuffer = await file.arrayBuffer()
    const { error } = await supabase
      .storage
      .from('user-documents')
      .upload(storagePath, arrayBuffer, {
        contentType: file.type || 'application/octet-stream',
        upsert: false,
      })

    if (error) {
      throw new Error(`Failed to upload clothing photo: ${error.message}`)
    }

    uploadedPaths.push(storagePath)
  }

  return uploadedPaths
}

async function embedSignupClothingPhotos(files: File[]): Promise<number[][]> {
  const embeddings: number[][] = []

  for (const file of files.slice(0, SIGNUP_IMAGE_UPLOAD_LIMIT)) {
    if (!file.type.startsWith('image/')) continue
    if (file.size > SIGNUP_IMAGE_MAX_BYTES) {
      throw new Error(`Clothing photo "${file.name}" exceeds the 10MB limit`)
    }

    const formData = new FormData()
    const blob = new Blob([await file.arrayBuffer()], { type: file.type || 'application/octet-stream' })
    formData.append('file', blob, file.name || 'upload.jpg')

    const response = await fetch(`${getRecsysBaseUrl()}/embed/image`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const text = await response.text().catch(() => '')
      throw new Error(`Failed to embed clothing photo (${response.status}): ${text || 'no body'}`)
    }

    const json = await response.json()
    const embedding = Array.isArray(json?.embedding)
      ? json.embedding.map((value: unknown) => Number(value)).filter((value: number) => Number.isFinite(value))
      : []

    if (embedding.length > 0) {
      embeddings.push(embedding)
    }
  }

  return embeddings
}

export async function POST(request: Request) {
  let createdUserId: string | null = null
  let uploadedImagePaths: string[] = []

  try {
    const { email, password, userData, uploadedClothingPhotos } = await parseSignupPayload(request)

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

    createdUserId = authData.user.id
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
    const uploadedPhotoFiles = uploadedClothingPhotos.slice(0, SIGNUP_IMAGE_UPLOAD_LIMIT)
    uploadedImagePaths = uploadedPhotoFiles.length > 0
      ? await uploadSignupClothingPhotos(authData.user.id, uploadedPhotoFiles)
      : []
    const uploadedImageEmbeddings = uploadedPhotoFiles.length > 0
      ? await embedSignupClothingPhotos(uploadedPhotoFiles)
      : []

    const likedProductsEmbedding = storedLikedProductIds.length >= MIN_LIKES_TO_STORE_EMBEDDING
      ? await computeUserImageEmbeddingFromLikedProducts(supabase, storedLikedProductIds)
      : null
    const uploadedImageEmbeddingAvg = await computeUserImageEmbeddingFromVectors(uploadedImageEmbeddings)
    const userImageEmbeddingAvg = await computeUserImageEmbeddingFromVectors(
      [likedProductsEmbedding, uploadedImageEmbeddingAvg].filter((vec): vec is number[] => Array.isArray(vec) && vec.length > 0)
    )

    const baseInsertPayload: any = {
      id: authData.user.id, // Use Auth user ID as primary key
      handle: userData.name,
      email: email,
      liked_product_ids: storedLikedProductIds,
      metadata: {
        liked_product_ids: storedLikedProductIds,
        shown_product_ids: shownProductIds,
        recommended_product_ids: recommendedProductIds,
        uploaded_image_paths: uploadedImagePaths,
        uploaded_image_count: uploadedImagePaths.length,
        uploaded_image_embedding_avg: uploadedImageEmbeddingAvg,
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

    if (uploadedImagePaths.length > 0) {
      await supabase.storage.from('user-documents').remove(uploadedImagePaths).catch(() => undefined)
    }

    if (createdUserId) {
      await supabase.auth.admin.deleteUser(createdUserId).catch(() => undefined)
    }

    return NextResponse.json(
      { success: false, message: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
