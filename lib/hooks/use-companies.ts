import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase/client'

export interface Company {
  id: string
  name: string
  ticker: string | null
  exchange: string | null
  country: string | null
  website: string | null
  description: string | null
  market_cap: number | null
  urls: string[] | null
  watchlist: boolean
  created_at: string
  updated_at: string
  // NOTE: AISC is NOT included at company level because:
  // - Different projects have different commodities ($/oz vs $/lb vs $/tonne)
  // - AISC is project/mine-specific, not company-wide
  // - Use project-level AISC instead
}

export function useCompanies() {
  const [companies, setCompanies] = useState<Company[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchCompanies()

    // Subscribe to realtime changes
    const client = supabase
    if (client) {
      const subscription = client
        .channel('companies')
        .on(
          'postgres_changes',
          { event: '*', schema: 'public', table: 'companies' },
          () => {
            fetchCompanies()
          }
        )
        .subscribe()

      return () => {
        subscription.unsubscribe()
      }
    }
  }, [])

  const fetchCompanies = async () => {
    try {
      setLoading(true)

      const client = supabase
      if (!client) {
        setError('Supabase client not available')
        setLoading(false)
        return
      }

      // Fetch companies with pagination
      let allData: any[] = []
      let offset = 0
      const pageSize = 1000
      let hasMore = true

      while (hasMore) {
        const { data, error } = await client
          .from('companies')
          .select('*')
          .order('market_cap', { ascending: false, nullsFirst: false })
          .range(offset, offset + pageSize - 1)

        if (error) throw error

        if (data && data.length > 0) {
          allData = [...allData, ...data]
          offset += pageSize
          hasMore = data.length === pageSize
        } else {
          hasMore = false
        }
      }

      // Get current user's watchlist
      const { data: { user } } = await client.auth.getUser()
      let userWatchlistSet = new Set<string>()

      if (user) {
        const { data: watchlistData } = await client
          .from('user_company_watchlist')
          .select('company_id')
          .eq('user_id', user.id)

        userWatchlistSet = new Set(watchlistData?.map(w => w.company_id) || [])
      }

      // Update watchlist status based on user's watchlist
      const companiesWithWatchlist = allData.map(company => ({
        ...company,
        watchlist: userWatchlistSet.has(company.id)
      }))

      setCompanies(companiesWithWatchlist)
      setError(null)
    } catch (err) {
      console.error('Error fetching companies:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch companies')
    } finally {
      setLoading(false)
    }
  }

  return { companies, loading, error, refetch: fetchCompanies }
}
