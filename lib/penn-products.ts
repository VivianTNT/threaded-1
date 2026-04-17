import { FashionProduct } from './types/fashion-product'
import { normalizeProductGender } from './product-gender'

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
  gender: string | null
  description: string
  currency: string | null
}

export function formatDomainAsBrand(domain: string | null | undefined): string {
  const raw = String(domain || '').trim().toLowerCase()
  if (!raw) return 'Unknown Brand'

  const withoutProtocol = raw.replace(/^https?:\/\//, '')
  const hostname = withoutProtocol.split('/')[0]?.replace(/^www\./, '') || ''
  const segments = hostname.split('.').filter(Boolean)

  if (!segments.length) return 'Unknown Brand'

  let brandSegment = segments[0]
  if (segments.length >= 2) {
    const last = segments[segments.length - 1]
    const secondLast = segments[segments.length - 2]

    if (last.length === 2 && ['co', 'com', 'org', 'net'].includes(secondLast) && segments.length >= 3) {
      brandSegment = segments[segments.length - 3]
    } else {
      brandSegment = secondLast
    }
  }

  const cleaned = brandSegment.replace(/[-_]+/g, ' ').trim()
  if (!cleaned) return 'Unknown Brand'

  return cleaned.replace(/\b\w/g, (char) => char.toUpperCase())
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
    brand: formatDomainAsBrand(pennProduct.domain),
    category,
    style: ['Casual'], // Default style
    price: pennProduct.price || 0,
    currency: pennProduct.currency || 'USD',
    color: ['Various'], // Default
    material: ['Mixed Materials'], // Default
    season: ['All Season'],
    gender: normalizeProductGender(pennProduct.gender),
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
