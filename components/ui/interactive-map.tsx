'use client'

import React, { useEffect, useState } from 'react'
import { Card } from '@/components/ui/card'
import { MapPin, Loader2 } from 'lucide-react'
import dynamic from 'next/dynamic'

// Dynamically import Leaflet components to avoid SSR issues
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
)
const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
)
const Marker = dynamic(
  () => import('react-leaflet').then((mod) => mod.Marker),
  { ssr: false }
)
const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
)

interface InteractiveMapProps {
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

export function InteractiveMap({
  location,
  width = 600,
  height = 300,
  zoom = 10,
  className
}: InteractiveMapProps) {
  const [coords, setCoords] = useState<Coordinates | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const [mounted, setMounted] = useState(false)

  // Handle client-side mounting
  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    let isActive = true

    async function fetchCoordinates() {
      setLoading(true)
      setError(false)

      // Try fallback first (instant)
      const fallbackCoords = getCoordinatesFromLocationFallback(location)
      if (fallbackCoords && isActive) {
        setCoords(fallbackCoords)
        setLoading(false)
        return
      }

      // Try geocoding API
      try {
        const geocodedCoords = await geocodeLocation(location)
        if (isActive) {
          if (geocodedCoords) {
            setCoords(geocodedCoords)
          } else {
            setError(true)
          }
          setLoading(false)
        }
      } catch (err) {
        if (isActive) {
          setError(true)
          setLoading(false)
        }
      }
    }

    fetchCoordinates()

    return () => {
      isActive = false
    }
  }, [location])

  // Import Leaflet CSS only on client side
  useEffect(() => {
    if (mounted) {
      import('leaflet/dist/leaflet.css')
    }
  }, [mounted])

  if (!mounted || loading) {
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

  return (
    <div className={className} style={{ width, height }}>
      <MapContainer
        center={[coords.lat, coords.lng]}
        zoom={zoom}
        style={{ width: '100%', height: '100%', borderRadius: '0.5rem', border: '1px solid hsl(var(--border))' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={[coords.lat, coords.lng]}>
          <Popup>
            <strong>{location}</strong>
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  )
}
