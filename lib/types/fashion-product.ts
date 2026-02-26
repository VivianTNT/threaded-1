export type ProductCategory =
  | 'Tops'
  | 'Bottoms'
  | 'Dresses'
  | 'Outerwear'
  | 'Activewear'
  | 'Innerwear'
  | 'Shoes'
  | 'Accessories'
  | 'Bags';

export type ClothingStyle =
  | 'Casual'
  | 'Formal'
  | 'Business Casual'
  | 'Athleisure'
  | 'Evening'
  | 'Streetwear'
  | 'Minimalist'
  | 'Bohemian';

export type PriceRange = 'Budget' | 'Mid-Range' | 'Premium' | 'Luxury';

export type Season = 'Spring' | 'Summer' | 'Fall' | 'Winter' | 'All Season';

export type Gender = 'Women' | 'Men' | 'Unisex';

// Brand interface
export interface Brand {
  name: string;
  tier: 'Luxury' | 'Premium' | 'Contemporary' | 'Fast Fashion';
  website: string;
  sustainability_score?: number; // 0-100
}

// Database schema for fashion products
export interface FashionProduct {
  id: string;
  user_id: string | null;
  name: string;
  brand: string;
  category: ProductCategory;
  style: ClothingStyle[];
  price: number;
  currency: string;
  color: string[];
  material: string[];
  season: Season[];
  gender: Gender;
  sizes_available: string[];
  description: string | null;
  image_url: string;
  product_url: string | null;
  in_stock: boolean;
  sustainability_rating: number | null; // 0-5 stars
  created_at: string;
  updated_at: string;

  // User interaction fields
  is_liked: boolean;
  is_saved: boolean;
  in_cart: boolean;
  added_to_wardrobe: boolean;

  // Recommendation metadata
  recommendation_score: number | null; // 0-100
  recommendation_reason: string | null;
  similar_to_wardrobe_item_id: string | null;

  // Computed/display fields
  price_range?: PriceRange;
  brand_tier?: string;
}

export interface ProductFilter {
  categories?: ProductCategory[];
  styles?: ClothingStyle[];
  brands?: string[];
  colors?: string[];
  priceRange?: { min: number; max: number };
  seasons?: Season[];
  genders?: Gender[];
  inStock?: boolean;
  searchQuery?: string;
}

export interface SavedView {
  id: string;
  name: string;
  filters: ProductFilter;
  visibleColumns: string[];
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface WardrobeItem {
  id: string;
  user_id: string;
  image_url: string;
  category: ProductCategory;
  color: string[];
  style: ClothingStyle[];
  brand?: string;
  notes?: string;
  created_at: string;
}

export interface Notification {
  id: string;
  user_id: string;
  type: 'new_recommendation' | 'price_drop' | 'back_in_stock' | 'similar_item' | 'agent_complete';
  title: string;
  message: string;
  product_id: string | null;
  image_url: string | null;
  is_read: boolean;
  created_at: string;
}

export interface FashionAgentRun {
  id: string;
  user_id: string;
  status: 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  items_found: number;
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
}
