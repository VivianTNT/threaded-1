type CandidateEmbedding = {
  id: string
  embedding: number[]
}

type RankedCandidate = {
  id: string
  score: number
}

type RankResponse = {
  results?: Array<{ id?: unknown; score?: unknown }>
}

type EmbedUserResponse = {
  embedding?: unknown
}

const DEFAULT_TIMEOUT_MS = 4000

function parseTimeoutMs(): number {
  const raw = Number(process.env.RECOMMENDER_MODEL_TIMEOUT_MS || DEFAULT_TIMEOUT_MS)
  if (!Number.isFinite(raw) || raw <= 0) return DEFAULT_TIMEOUT_MS
  return Math.floor(raw)
}

function getBaseUrl(): string | null {
  const raw = process.env.RECOMMENDER_MODEL_SERVICE_URL
  if (!raw) return null
  const trimmed = raw.trim()
  return trimmed ? trimmed.replace(/\/+$/, '') : null
}

function isFiniteVector(value: unknown): value is number[] {
  return Array.isArray(value) && value.length > 0 && value.every((v) => Number.isFinite(Number(v)))
}

function normalizeVector(value: unknown): number[] | null {
  if (!isFiniteVector(value)) return null
  return value.map((v) => Number(v))
}

async function postJson<T>(path: string, payload: unknown): Promise<T | null> {
  const baseUrl = getBaseUrl()
  if (!baseUrl) return null

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), parseTimeoutMs())

  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })

    if (!response.ok) {
      return null
    }

    return (await response.json()) as T
  } catch {
    return null
  } finally {
    clearTimeout(timeout)
  }
}

export function modelServiceEnabled(): boolean {
  return Boolean(getBaseUrl())
}

export async function rankWithTwoTower(
  userEmbedding: number[],
  candidates: CandidateEmbedding[],
  topK?: number
): Promise<RankedCandidate[] | null> {
  if (!modelServiceEnabled() || !isFiniteVector(userEmbedding) || candidates.length === 0) {
    return null
  }

  const response = await postJson<RankResponse>('/rank', {
    user_embedding: userEmbedding,
    candidates: candidates.map((c) => ({
      id: c.id,
      embedding: c.embedding,
    })),
    top_k: Number.isFinite(topK) && typeof topK === 'number' && topK > 0 ? Math.floor(topK) : undefined,
  })

  if (!response?.results || !Array.isArray(response.results)) {
    return null
  }

  const out: RankedCandidate[] = []
  for (const result of response.results) {
    const id = typeof result?.id === 'string' ? result.id : result?.id != null ? String(result.id) : null
    const score = Number(result?.score)
    if (!id || !Number.isFinite(score)) continue
    out.push({ id, score })
  }

  if (!out.length) return null
  return out
}

export async function buildUserEmbeddingWithTwoTower(likedEmbeddings: number[][]): Promise<number[] | null> {
  if (!modelServiceEnabled() || !likedEmbeddings.length) {
    return null
  }

  const cleanEmbeddings = likedEmbeddings.filter((vec) => isFiniteVector(vec))
  if (!cleanEmbeddings.length) return null

  const response = await postJson<EmbedUserResponse>('/embed-user', {
    liked_embeddings: cleanEmbeddings,
  })

  return normalizeVector(response?.embedding)
}
