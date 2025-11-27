import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

// Use service role key for server-side operations
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

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

    // Insert user data into the users table
    const { data: newUser, error: insertError } = await supabase
      .from('users')
      .insert({
        id: authData.user.id, // Use Auth user ID as primary key
        handle: userData.name,
        email: email,
        metadata: {
          style_preferences: userData.stylePreferences || [],
          budget_range: userData.budgetRange || '',
          favorite_colors: userData.favoriteColors || '',
          bio: userData.bio || ''
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .select()
      .single()

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
