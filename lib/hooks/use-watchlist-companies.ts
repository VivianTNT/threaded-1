import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase/client'
import { Company } from './use-companies'

export function useWatchlistCompanies() {
  const [companies, setCompanies] = useState<Company[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchCompanies()

    // Subscribe to realtime changes
    const client = supabase
    if (client) {
      const subscription = client
        .channel('watchlist-companies')
        .on(
          'postgres_changes',
          { event: '*', schema: 'public', table: 'user_company_watchlist' },
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

      // Get current user
      const { data: { user } } = await client.auth.getUser()
      if (!user) {
        setCompanies([])
        setError(null)
        setLoading(false)
        return
      }

      // Fetch user's watchlisted companies using the junction table
      const { data: watchlistData, error: watchlistError } = await client
        .from('user_company_watchlist')
        .select('company_id')
        .eq('user_id', user.id)

      if (watchlistError) throw watchlistError

      // If no watchlisted companies, return empty array
      if (!watchlistData || watchlistData.length === 0) {
        setCompanies([])
        setError(null)
        setLoading(false)
        return
      }

      const companyIds = watchlistData.map(w => w.company_id)

      // Fetch the actual company data
      const { data, error } = await client
        .from('companies')
        .select('*')
        .in('id', companyIds)
        .order('updated_at', { ascending: false })

      if (error) throw error

      setCompanies(data || [])
      setError(null)
    } catch (err) {
      console.error('Error fetching watchlisted companies:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch watchlisted companies')
    } finally {
      setLoading(false)
    }
  }

  return { companies, loading, error, refetch: fetchCompanies }
}
