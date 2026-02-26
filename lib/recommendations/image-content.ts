export type ProductRow = {
  id: string
  name: string | null
  brand_name: string | null
  image_url: string | null
  price: number | null
  product_url: string | null
  category: string | null
  description: string | null
}

export type ProductCard = {
  id: string
  name: string
  brand: string
  image_url: string
  price: number | null
  product_url: string | null
  category: string | null
  description: string | null
}

export type EmbeddingRow = {
  product_id: string
  image_embedding: unknown
}

export const DEFAULT_SIGNUP_SAMPLE_SIZE = 16
export const DEFAULT_SIGNUP_MIN_LIKES = 3
export const DEFAULT_SIGNUP_TOP_K = 12

export function parseVector(value: unknown): number[] | null {
  if (!value) return null

  if (Array.isArray(value)) {
    const nums = value.map((v) => Number(v)).filter((v) => Number.isFinite(v))
    return nums.length ? nums : null
  }

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null

    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (Array.isArray(parsed)) {
          const nums = parsed.map((v) => Number(v)).filter((v) => Number.isFinite(v))
          return nums.length ? nums : null
        }
      } catch {
        // Fall through to regex parse.
      }
    }

    const matches = trimmed.match(/[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?/g)
    if (!matches || !matches.length) return null
    const nums = matches.map((v) => Number(v)).filter((v) => Number.isFinite(v))
    return nums.length ? nums : null
  }

  return null
}

export function l2Normalize(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0))
  if (norm <= 1e-12) return vec.map(() => 0)
  return vec.map((v) => v / norm)
}

export function meanVectors(vectors: number[][]): number[] {
  if (!vectors.length) return []
  const dim = vectors[0].length
  const out = new Array(dim).fill(0)

  for (const vec of vectors) {
    for (let i = 0; i < dim; i += 1) {
      out[i] += vec[i]
    }
  }
  for (let i = 0; i < dim; i += 1) {
    out[i] /= vectors.length
  }
  return out
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  const dim = Math.min(a.length, b.length)
  for (let i = 0; i < dim; i += 1) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  if (na <= 1e-12 || nb <= 1e-12) return 0
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

export function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1))
    const tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
  }
}

export function vectorToPg(value: number[]): string {
  return `[${value.join(',')}]`
}

export function toProductCard(row: ProductRow): ProductCard {
  return {
    id: row.id,
    name: row.name || 'Fashion Item',
    brand: row.brand_name || 'Unknown Brand',
    image_url: row.image_url || '',
    price: row.price ?? null,
    product_url: row.product_url ?? null,
    category: row.category ?? null,
    description: row.description ?? null,
  }
}
