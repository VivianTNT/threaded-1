'use client'

import React, { useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { Card } from '@/components/ui/card'
import { MapPin, Loader2 } from 'lucide-react'

interface InteractiveMapboxProps {
  location: string
  latitude?: number | null
  longitude?: number | null
  width?: number
  height?: number
  initialZoom?: number
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
  if (geocodeCache.has(location)) {
    return geocodeCache.get(location) || null
  }

  try {
    const response = await fetch(`/api/geocode?location=${encodeURIComponent(location)}`)
    if (!response.ok) {
      geocodeCache.set(location, null)
      return null
    }

    const coords: Coordinates = await response.json()
    geocodeCache.set(location, coords)
    return coords
  } catch (error) {
    console.error(`Failed to geocode "${location}":`, error)
    geocodeCache.set(location, null)
    return null
  }
}

// Fallback coordinates database
function getCoordinatesFromLocationFallback(location: string): Coordinates | null {
  const locationMap: Record<string, Coordinates> = {
    'USA': { lat: 37.0902, lng: -95.7129 },
    'United States': { lat: 37.0902, lng: -95.7129 },
    'Canada': { lat: 56.1304, lng: -106.3468 },
    'Ontario': { lat: 51.2538, lng: -85.3232 },
    'Northeastern Ontario': { lat: 47.5448, lng: -81.0043 },
    'British Columbia': { lat: 53.7267, lng: -127.6476 },
    'Quebec': { lat: 52.9399, lng: -73.5491 },
    'Saskatchewan': { lat: 52.9399, lng: -106.4509 },
    'Mexico': { lat: 23.6345, lng: -102.5528 },
    'Chile': { lat: -35.6751, lng: -71.5430 },
    'Peru': { lat: -9.1900, lng: -75.0152 },
    'Brazil': { lat: -14.2350, lng: -51.9253 },
    'Argentina': { lat: -38.4161, lng: -63.6167 },
    'Bolivia': { lat: -16.2902, lng: -63.5887 },
    'Sweden': { lat: 60.1282, lng: 18.6435 },
    'Finland': { lat: 61.9241, lng: 25.7482 },
    'Norway': { lat: 60.4720, lng: 8.4689 },
    'UK': { lat: 55.3781, lng: -3.4360 },
    'United Kingdom': { lat: 55.3781, lng: -3.4360 },
    'Spain': { lat: 40.4637, lng: -3.7492 },
    'Portugal': { lat: 39.3999, lng: -8.2245 },
    'France': { lat: 46.2276, lng: 2.2137 },
    'Germany': { lat: 51.1657, lng: 10.4515 },
    'South Africa': { lat: -30.5595, lng: 22.9375 },
    'DRC': { lat: -4.0383, lng: 21.7587 },
    'Congo': { lat: -4.0383, lng: 21.7587 },
    'Zambia': { lat: -13.1339, lng: 27.8493 },
    'Tanzania': { lat: -6.3690, lng: 34.8888 },
    'Namibia': { lat: -22.9576, lng: 18.4904 },
    'Botswana': { lat: -22.3285, lng: 24.6849 },
    'Zimbabwe': { lat: -19.0154, lng: 29.1549 },
    'China': { lat: 35.8617, lng: 104.1954 },
    'Mongolia': { lat: 46.8625, lng: 103.8467 },
    'Kazakhstan': { lat: 48.0196, lng: 66.9237 },
    'India': { lat: 20.5937, lng: 78.9629 },
    'Indonesia': { lat: -0.7893, lng: 113.9213 },
    'Philippines': { lat: 12.8797, lng: 121.7740 },
    'Australia': { lat: -25.2744, lng: 133.7751 },
    'Western Australia': { lat: -25.0423, lng: 121.6384 },
    'Papua New Guinea': { lat: -6.3150, lng: 143.9555 },
    'New Zealand': { lat: -40.9006, lng: 174.8860 },
  }

  if (locationMap[location]) {
    return locationMap[location]
  }

  for (const [key, coords] of Object.entries(locationMap)) {
    if (location.toLowerCase().includes(key.toLowerCase())) {
      return coords
    }
  }

  return null
}

export function InteractiveMapbox({
  location,
  latitude,
  longitude,
  width = 600,
  height = 300,
  initialZoom = 8,
  className
}: InteractiveMapboxProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [coords, setCoords] = useState<Coordinates | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  // Fetch coordinates
  useEffect(() => {
    let mounted = true

    async function fetchCoordinates() {
      setLoading(true)
      setError(false)

      // Priority 1: Use provided latitude/longitude if available
      if (latitude !== null && latitude !== undefined &&
          longitude !== null && longitude !== undefined) {
        if (mounted) {
          setCoords({ lat: latitude, lng: longitude })
          setLoading(false)
        }
        return
      }

      // Priority 2: Try fallback location mapping
      const fallbackCoords = getCoordinatesFromLocationFallback(location)
      if (fallbackCoords && mounted) {
        setCoords(fallbackCoords)
        setLoading(false)
        return
      }

      // Priority 3: Try geocoding API
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
  }, [location, latitude, longitude])

  // Initialize map
  useEffect(() => {
    if (!coords || !mapContainer.current || map.current) return

    mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: [coords.lng, coords.lat],
      zoom: initialZoom,
      attributionControl: false // Remove attribution
    })

    // Add navigation controls (zoom buttons)
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

    // Add marker
    new mapboxgl.Marker({ color: '#e63946' })
      .setLngLat([coords.lng, coords.lat])
      .addTo(map.current)

    return () => {
      map.current?.remove()
      map.current = null
    }
  }, [coords, initialZoom])

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

  return (
    <div className={className} style={{ width, height }}>
      <div
        ref={mapContainer}
        className="w-full h-full rounded-lg border shadow-sm"
      />
    </div>
  )
}
