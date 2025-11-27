import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { transformPennProduct, PennProduct } from '@/lib/penn-products'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

console.log('🔍 API Route Environment Check:');
console.log('SUPABASE_URL:', supabaseUrl);
console.log('SUPABASE_KEY:', supabaseKey?.substring(0, 20) + '...');

const supabase = createClient(supabaseUrl, supabaseKey)

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')

    // Fetch products from Penn database
    const { data: products, error, count } = await supabase
      .from('products')
      .select('*', { count: 'exact' })
      .range(offset, offset + limit - 1)
      .order('created_at', { ascending: false })

    if (error) {
      console.error('Error fetching products:', error)
      return NextResponse.json(
        { success: false, message: 'Failed to fetch products', error: error.message },
        { status: 500 }
      )
    }

    // Transform Penn products to FashionProduct format
    const fashionProducts = (products as PennProduct[] || []).map(transformPennProduct)

    return NextResponse.json({
      success: true,
      products: fashionProducts,
      total: count || 0,
      limit,
      offset
    })
  } catch (error: any) {
    console.error('Products API error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
