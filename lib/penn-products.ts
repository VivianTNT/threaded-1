import { FashionProduct } from './types/fashion-product'

// Penn database product structure
export interface PennProduct {
  id: string
  product_url: string
  image_url: string
  name: string
  price: number | null
  domain: string
  created_at: string
  brand_name: string | null
  category: string | null
  description: string
  currency: string | null
}

// Transform Penn product to FashionProduct
export function transformPennProduct(pennProduct: PennProduct): FashionProduct {
  // Generate a placeholder image if none exists
  const placeholderImage = 'https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=800&h=1200&fit=crop'

  // Parse category if available
  const categoryMap: Record<string, any> = {
    'tops': 'Tops',
    'bottoms': 'Bottoms',
    'dresses': 'Dresses',
    'outerwear': 'Outerwear',
    'shoes': 'Shoes',
    'accessories': 'Accessories',
    'bags': 'Bags'
  }

  const category = pennProduct.category
    ? categoryMap[pennProduct.category.toLowerCase()] || 'Accessories'
    : 'Accessories'

  return {
    id: pennProduct.id,
    user_id: null,
    name: pennProduct.name || 'Fashion Item',
    brand: pennProduct.brand_name || pennProduct.domain?.replace('.com', '').replace('www.', '') || 'Unknown Brand',
    category,
    style: ['Casual'], // Default style
    price: pennProduct.price || 0,
    currency: pennProduct.currency || 'USD',
    color: ['Various'], // Default
    material: ['Mixed Materials'], // Default
    season: ['All Season'],
    gender: 'Unisex',
    sizes_available: ['S', 'M', 'L', 'XL'], // Default sizes
    description: pennProduct.description || null,
    image_url: pennProduct.image_url || placeholderImage,
    product_url: pennProduct.product_url || null,
    in_stock: true, // Default to in stock
    sustainability_rating: null,
    created_at: pennProduct.created_at,
    updated_at: pennProduct.created_at,

    // User interaction fields
    is_liked: false,
    is_saved: false,
    in_cart: false,
    added_to_wardrobe: false,

    // Recommendation metadata
    recommendation_score: null,
    recommendation_reason: null,
    similar_to_wardrobe_item_id: null
  }
}
