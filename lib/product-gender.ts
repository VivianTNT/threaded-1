import { Gender } from './types/fashion-product'

type RawGender = string | null | undefined

export function normalizeProductGender(gender: RawGender): Gender {
  const normalized = String(gender || '').trim().toLowerCase()

  if (!normalized) return 'Unisex'
  if (normalized === 'm' || normalized === 'men' || normalized === "men's") return 'Men'
  if (normalized === 'w' || normalized === 'women' || normalized === "women's") return 'Women'
  return 'Unisex'
}

export function getProductGenderLabel(gender: RawGender): string {
  const normalized = normalizeProductGender(gender)

  if (normalized === 'Men') return "Men's"
  if (normalized === 'Women') return "Women's"
  return 'Unisex'
}
