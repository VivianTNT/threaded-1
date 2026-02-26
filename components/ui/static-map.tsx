'use client'

import React, { useEffect, useState } from 'react'
import { Card } from '@/components/ui/card'
import { MapPin, Loader2 } from 'lucide-react'

interface StaticMapProps {
  location: string
  width?: number
  height?: number
  zoom?: number
  className?: string
}

interface Coordinates {
  lat: number
  lng: number
}

// In-memory cache for geocoded locations
const geocodeCache = new Map<string, Coordinates | null>()

// Helper function to geocode using our API endpoint
async function geocodeLocation(location: string): Promise<Coordinates | null> {
  // Check cache first
  if (geocodeCache.has(location)) {
    return geocodeCache.get(location) || null
  }

  try {
    const response = await fetch(`/api/geocode?location=${encodeURIComponent(location)}`)

    if (!response.ok) {
      // Cache null result to avoid repeated failed lookups
      geocodeCache.set(location, null)
      return null
    }

    const coords: Coordinates = await response.json()

    // Cache the result
    geocodeCache.set(location, coords)
    return coords
  } catch (error) {
    console.error(`Failed to geocode "${location}":`, error)
    geocodeCache.set(location, null)
    return null
  }
}

// Helper function to get coordinates from location string with fallback
function getCoordinatesFromLocationFallback(location: string): Coordinates | null {
  const locationMap: Record<string, Coordinates> = {
    // North America
    'USA': { lat: 37.0902, lng: -95.7129 },
    'United States': { lat: 37.0902, lng: -95.7129 },
    'Canada': { lat: 56.1304, lng: -106.3468 },
    'Ontario': { lat: 51.2538, lng: -85.3232 },
    'Northeastern Ontario': { lat: 47.5448, lng: -81.0043 },
    'British Columbia': { lat: 53.7267, lng: -127.6476 },
    'Quebec': { lat: 52.9399, lng: -73.5491 },
    'Saskatchewan': { lat: 52.9399, lng: -106.4509 },
    'Mexico': { lat: 23.6345, lng: -102.5528 },

    // South America
    'Chile': { lat: -35.6751, lng: -71.5430 },
    'Peru': { lat: -9.1900, lng: -75.0152 },
    'Brazil': { lat: -14.2350, lng: -51.9253 },
    'Argentina': { lat: -38.4161, lng: -63.6167 },
    'Bolivia': { lat: -16.2902, lng: -63.5887 },

    // Europe
    'Sweden': { lat: 60.1282, lng: 18.6435 },
    'Finland': { lat: 61.9241, lng: 25.7482 },
    'Norway': { lat: 60.4720, lng: 8.4689 },
    'UK': { lat: 55.3781, lng: -3.4360 },
    'United Kingdom': { lat: 55.3781, lng: -3.4360 },
    'Spain': { lat: 40.4637, lng: -3.7492 },
    'Portugal': { lat: 39.3999, lng: -8.2245 },
    'France': { lat: 46.2276, lng: 2.2137 },
    'Germany': { lat: 51.1657, lng: 10.4515 },

    // Africa
    'South Africa': { lat: -30.5595, lng: 22.9375 },
    'DRC': { lat: -4.0383, lng: 21.7587 },
    'Congo': { lat: -4.0383, lng: 21.7587 },
    'Zambia': { lat: -13.1339, lng: 27.8493 },
    'Tanzania': { lat: -6.3690, lng: 34.8888 },
    'Namibia': { lat: -22.9576, lng: 18.4904 },
    'Botswana': { lat: -22.3285, lng: 24.6849 },
    'Zimbabwe': { lat: -19.0154, lng: 29.1549 },

    // Asia
    'China': { lat: 35.8617, lng: 104.1954 },
    'Mongolia': { lat: 46.8625, lng: 103.8467 },
    'Kazakhstan': { lat: 48.0196, lng: 66.9237 },
    'India': { lat: 20.5937, lng: 78.9629 },
    'Indonesia': { lat: -0.7893, lng: 113.9213 },
    'Philippines': { lat: 12.8797, lng: 121.7740 },

    // Oceania
    'Australia': { lat: -25.2744, lng: 133.7751 },
    'Western Australia': { lat: -25.0423, lng: 121.6384 },
    'Papua New Guinea': { lat: -6.3150, lng: 143.9555 },
    'New Zealand': { lat: -40.9006, lng: 174.8860 },
  }

  // Try exact match first
  if (locationMap[location]) {
    return locationMap[location]
  }

  // Try partial match
  for (const [key, coords] of Object.entries(locationMap)) {
    if (location.toLowerCase().includes(key.toLowerCase())) {
      return coords
    }
  }

  return null
}

export function StaticMap({
  location,
  width = 600,
  height = 300,
  zoom = 5,
  className
}: StaticMapProps) {
  const [coords, setCoords] = useState<Coordinates | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    let mounted = true

    async function fetchCoordinates() {
      setLoading(true)
      setError(false)

      // Try fallback first (instant)
      const fallbackCoords = getCoordinatesFromLocationFallback(location)
      if (fallbackCoords && mounted) {
        setCoords(fallbackCoords)
        setLoading(false)
        return
      }

      // Try geocoding API
      try {
        const geocodedCoords = await geocodeLocation(location)
        if (mounted) {
          if (geocodedCoords) {
            setCoords(geocodedCoords)
          } else {
            setError(true)
          }
          setLoading(false)
        }
      } catch (err) {
        if (mounted) {
          setError(true)
          setLoading(false)
        }
      }
    }

    fetchCoordinates()

    return () => {
      mounted = false
    }
  }, [location])

  if (loading) {
    return (
      <Card className={`flex items-center justify-center bg-muted ${className || ''}`} style={{ width, height }}>
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <Loader2 className="h-6 w-6 animate-spin" />
          <p className="text-sm">Loading map...</p>
        </div>
      </Card>
    )
  }

  if (error || !coords) {
    return (
      <Card className={`flex items-center justify-center bg-muted ${className || ''}`} style={{ width, height }}>
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <MapPin className="h-8 w-8" />
          <p className="text-sm">Location map unavailable</p>
          <p className="text-xs">{location}</p>
        </div>
      </Card>
    )
  }

  // Mapbox Static Images API with Satellite Imagery
  // Official documentation: https://docs.mapbox.com/api/maps/static-images/
  // Format: https://api.mapbox.com/styles/v1/{username}/{style_id}/static/{overlay}/{lon},{lat},{zoom},{bearing},{pitch}/{width}x{height}{@2x}?access_token={token}
  //
  // Style: satellite-streets-v12 - Latest satellite imagery with street labels and POI data
  // Resolution: Up to 5cm per pixel at high zoom levels, Maxar Vivid imagery (zoom 8-18)
  // Marker: pin-l+e63946 - Large red marker for clear visibility on satellite imagery
  // @2x: High-DPI rendering for retina displays
  const mapboxToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN
  const mapUrl = `https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/static/pin-l+e63946(${coords.lng},${coords.lat})/${coords.lng},${coords.lat},${zoom},0,0/${width}x${height}@2x?access_token=${mapboxToken}`

  return (
    <div className={className}>
      <img
        src={mapUrl}
        alt={`Map of ${location}`}
        width={width}
        height={height}
        className="rounded-lg border shadow-sm w-full h-auto"
        loading="lazy"
      />
    </div>
  )
}
