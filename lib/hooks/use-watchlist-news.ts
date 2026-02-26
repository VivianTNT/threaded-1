import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase/client'
import { NewsItem } from './use-news'

export function useWatchlistNews() {
  const [news, setNews] = useState<NewsItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchNews = async () => {
    try {
      setLoading(true)

      // Get current user
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        setNews([])
        setError(null)
        setLoading(false)
        return
      }

      // Fetch user's watchlisted news using the junction table
      const { data: watchlistData, error: watchlistError } = await supabase
        .from('user_news_watchlist')
        .select('news_id')
        .eq('user_id', user.id)

      if (watchlistError) throw watchlistError

      // If no watchlisted news, return empty array
      if (!watchlistData || watchlistData.length === 0) {
        setNews([])
        setError(null)
        setLoading(false)
        return
      }

      const newsIds = watchlistData.map(w => w.news_id)

      // Fetch the actual news data
      const { data, error: fetchError } = await supabase
        .from('news')
        .select('*')
        .in('id', newsIds)
        .order('published_at', { ascending: false, nullsFirst: false })
        .order('created_at', { ascending: false })

      if (fetchError) throw fetchError

      setNews(data || [])
      setError(null)
    } catch (err: any) {
      console.error('Error fetching watchlist news:', err)
      setError(err.message || 'Failed to fetch watchlist news')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchNews()

    // Subscribe to real-time updates
    const channel = supabase
      .channel('watchlist-news-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'user_news_watchlist',
        },
        () => {
          fetchNews()
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  return { news, loading, error, refetch: fetchNews }
}
