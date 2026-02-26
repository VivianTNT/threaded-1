import { NextRequest, NextResponse } from 'next/server'

// In-memory cache for geocoded locations
const geocodeCache = new Map<string, { lat: number; lng: number } | null>()

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const location = searchParams.get('location')

  if (!location) {
    return NextResponse.json({ error: 'Location parameter required' }, { status: 400 })
  }

  // Check cache first
  if (geocodeCache.has(location)) {
    const coords = geocodeCache.get(location)
    if (coords) {
      return NextResponse.json(coords)
    }
    return NextResponse.json({ error: 'Location not found' }, { status: 404 })
  }

  try {
    // Use Nominatim API with proper User-Agent header
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(location)}&format=json&limit=1`,
      {
        headers: {
          'User-Agent': 'Lithos Mining Projects (contact@lithos.ai)'
        }
      }
    )

    if (!response.ok) {
      throw new Error(`Geocoding failed: ${response.status}`)
    }

    const data = await response.json()

    if (data && data.length > 0) {
      const coords = {
        lat: parseFloat(data[0].lat),
        lng: parseFloat(data[0].lon)
      }

      // Cache the result
      geocodeCache.set(location, coords)
      return NextResponse.json(coords)
    }

    // Cache null result to avoid repeated failed lookups
    geocodeCache.set(location, null)
    return NextResponse.json({ error: 'Location not found' }, { status: 404 })
  } catch (error) {
    console.error(`Failed to geocode "${location}":`, error)
    geocodeCache.set(location, null)
    return NextResponse.json({ error: 'Geocoding error' }, { status: 500 })
  }
}
