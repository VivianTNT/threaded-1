'use client'

import * as React from 'react'
import { AppSidebar } from '@/components/app-sidebar'
import { SiteHeader } from '@/components/site-header'
import { ChatLayout } from '@/components/chat-layout'
import { useRequireAuth } from '@/lib/auth-utils'
import { FashionGrid } from '@/components/fashion-recommendations/fashion-grid'
import { ProductDetailPanel } from '@/components/fashion-recommendations/product-detail-panel'
import { FashionProduct } from '@/lib/types/fashion-product'
import { Sparkles, Filter, RefreshCw, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'

import {
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar'

type ProductsResponse = {
  success: boolean
  products: FashionProduct[]
  likedProducts: FashionProduct[]
  total?: number
  limit?: number
  offset?: number
  mode?: string
  engine?: string
  cached?: boolean
}

type RecommendationRefreshStrategy = 'hybrid' | 'content'

type RecommendationViewCache = {
  userId: string
  data: ProductsResponse
  selectedProductId: string | null
  showFilters: boolean
  autoRefreshOnLike: boolean
  preferredRefreshStrategy: RecommendationRefreshStrategy
  recommendationsNeedRefresh: boolean
  savedAt: number
}

const RECOMMENDATION_VIEW_CACHE_KEY = 'threaded:recommendation-view-cache'
const CLIENT_REVALIDATE_AFTER_MS = 30 * 1000

let recommendationViewMemoryCache: RecommendationViewCache | null = null

function readRecommendationViewCache(userId: string): RecommendationViewCache | null {
  if (recommendationViewMemoryCache?.userId === userId) {
    return recommendationViewMemoryCache
  }

  if (typeof window === 'undefined') return null

  try {
    const raw = window.sessionStorage.getItem(RECOMMENDATION_VIEW_CACHE_KEY)
    if (!raw) return null

    const parsed = JSON.parse(raw) as RecommendationViewCache
    if (!parsed || parsed.userId !== userId) return null

    recommendationViewMemoryCache = parsed
    return parsed
  } catch {
    return null
  }
}

function writeRecommendationViewCache(cache: RecommendationViewCache): void {
  recommendationViewMemoryCache = cache
  if (typeof window === 'undefined') return

  try {
    window.sessionStorage.setItem(RECOMMENDATION_VIEW_CACHE_KEY, JSON.stringify(cache))
  } catch {
    // Ignore storage write failures.
  }
}

function resolveSelectedProduct(
  likedProducts: FashionProduct[],
  products: FashionProduct[],
  selectedProductId: string | null
): FashionProduct | null {
  if (!selectedProductId) return null
  return [...likedProducts, ...products].find((product) => product.id === selectedProductId) || null
}

function upsertProduct(products: FashionProduct[], product: FashionProduct): FashionProduct[] {
  const without = products.filter((entry) => entry.id !== product.id)
  return [product, ...without]
}

// --- Extract Domain for Brand ---
function extractDomain(url?: string | null): string {
  if (!url) return 'Unknown'
  try {
    const hostname = new URL(url).hostname
    return hostname.replace(/^www\./, '')
  } catch {
    return 'Unknown'
  }
}

export default function Page() {
  const { user, session, isLoading } = useRequireAuth()
  
  // --- CORE STATE ---
  const [selectedProduct, setSelectedProduct] = React.useState<FashionProduct | null>(null)
  const [products, setProducts] = React.useState<FashionProduct[]>([])
  const [likedProducts, setLikedProducts] = React.useState<FashionProduct[]>([])
  const [recommendationMode, setRecommendationMode] = React.useState<string>('latest_fallback')
  const [recommendationEngine, setRecommendationEngine] = React.useState<string>('latest_products_fallback')
  const [showFilters, setShowFilters] = React.useState(false)
  const [isLoadingProducts, setIsLoadingProducts] = React.useState(true)
  const [pendingLikeProductId, setPendingLikeProductId] = React.useState<string | null>(null)
  const [autoRefreshOnLike, setAutoRefreshOnLike] = React.useState(true)
  const [preferredRefreshStrategy, setPreferredRefreshStrategy] = React.useState<RecommendationRefreshStrategy>('hybrid')
  const [isRefreshingRecommendations, setIsRefreshingRecommendations] = React.useState(false)
  const [recommendationsNeedRefresh, setRecommendationsNeedRefresh] = React.useState(false)

  // --- FILTER STATE ---
  const [activeFilters, setActiveFilters] = React.useState({
    searchQuery: '',
    brands: [] as string[],
    genders: [] as string[],
    minPrice: 0,
    maxPrice: 5000,
  })

  const allProducts = React.useMemo(() => {
    return [...likedProducts, ...products]
  }, [likedProducts, products])
  
  const likedItemIds = React.useMemo(() => {
    return new Set(likedProducts.map((product) => product.id))
  }, [likedProducts])

  // --- REFS FOR CALLBACKS ---
  const selectedProductIdRef = React.useRef<string | null>(null)
  const showFiltersRef = React.useRef(showFilters)
  const autoRefreshOnLikeRef = React.useRef(autoRefreshOnLike)
  const preferredRefreshStrategyRef = React.useRef<RecommendationRefreshStrategy>(preferredRefreshStrategy)

  React.useEffect(() => { selectedProductIdRef.current = selectedProduct?.id || null }, [selectedProduct])
  React.useEffect(() => { showFiltersRef.current = showFilters }, [showFilters])
  React.useEffect(() => { autoRefreshOnLikeRef.current = autoRefreshOnLike }, [autoRefreshOnLike])
  React.useEffect(() => { preferredRefreshStrategyRef.current = preferredRefreshStrategy }, [preferredRefreshStrategy])

  // --- DYNAMIC FILTER OPTIONS ---
  const availableBrands = React.useMemo(() => {
    const domains = products.map(p => extractDomain(p.product_url)).filter(d => d !== 'Unknown')
    return Array.from(new Set(domains)).sort()
  }, [products])
  
  const availableGenders = React.useMemo(() => {
    const genders = products.map(p => p.gender).filter(Boolean) as string[]
    return Array.from(new Set(genders)).sort()
  }, [products])

  const absoluteMaxPrice = React.useMemo(() => {
    const max = Math.max(...products.map(p => p.price || 0), 100)
    return Math.ceil(max / 10) * 10
  }, [products])

  // --- FILTER LOGIC ---
  const filteredProducts = React.useMemo(() => {
    return products.filter((product) => {
      // Search Query
      if (activeFilters.searchQuery) {
        const query = activeFilters.searchQuery.toLowerCase()
        const matchName = (product.name || '').toLowerCase().includes(query)
        const matchDesc = (product.description || '').toLowerCase().includes(query)
        if (!matchName && !matchDesc) return false
      }

      // Brand (Derived from Domain)
      if (activeFilters.brands.length > 0) {
        const productBrand = extractDomain(product.product_url)
        if (!activeFilters.brands.includes(productBrand)) return false
      }

      // Gender
      if (activeFilters.genders.length > 0) {
        if (!product.gender || !activeFilters.genders.includes(product.gender)) return false
      }

      // Price Bounds
      if (product.price < activeFilters.minPrice) return false
      if (product.price > activeFilters.maxPrice) return false
      
      return true
    })
  }, [products, activeFilters])

  // Initialize max price once products load
  React.useEffect(() => {
    if (products.length > 0 && activeFilters.maxPrice === 5000) {
      setActiveFilters(prev => ({ ...prev, maxPrice: absoluteMaxPrice }))
    }
  }, [products, absoluteMaxPrice, activeFilters.maxPrice])

  const toggleFilterArray = (key: 'brands' | 'genders', value: string) => {
    setActiveFilters(prev => {
      const currentArr = prev[key]
      return {
        ...prev,
        [key]: currentArr.includes(value) 
          ? currentArr.filter(v => v !== value)
          : [...currentArr, value]
      }
    })
  }

  const clearFilters = () => {
    setActiveFilters({
      searchQuery: '',
      brands: [],
      genders: [],
      minPrice: 0,
      maxPrice: absoluteMaxPrice,
    })
  }

  // --- FETCH & CACHE LOGIC ---
  const applyProductsResponse = React.useCallback(
    (data: ProductsResponse, selectedProductId: string | null, nextShowFilters: boolean) => {
      const nextProducts = Array.isArray(data.products) ? data.products : []
      const nextLikedProducts = Array.isArray(data.likedProducts) ? data.likedProducts : []

      setProducts(nextProducts)
      setLikedProducts(nextLikedProducts)
      setRecommendationMode(typeof data.mode === 'string' ? data.mode : 'latest_fallback')
      setRecommendationEngine(
        typeof data.engine === 'string' ? data.engine : 'latest_products_fallback'
      )
      setShowFilters(nextShowFilters)
      setSelectedProduct(resolveSelectedProduct(nextLikedProducts, nextProducts, selectedProductId))
    },
    []
  )

  const fetchProducts = React.useCallback(async ({
    showLoading = true,
    strategy = preferredRefreshStrategyRef.current,
    forceRefresh = false,
  }: {
    showLoading?: boolean
    strategy?: RecommendationRefreshStrategy
    forceRefresh?: boolean
  } = {}) => {
    if (!user) return

    try {
      setIsRefreshingRecommendations(true)
      if (showLoading) {
        setIsLoadingProducts(true)
      }
      const params = new URLSearchParams({
        limit: '50',
        userId: user.id || '',
        userEmail: user.email || '',
        strategy,
      })
      if (forceRefresh) {
        params.set('forceRefresh', '1')
      }
      const response = await fetch(`/api/products?${params.toString()}`)
      const data = await response.json()

      if (data.success) {
        applyProductsResponse(
          data as ProductsResponse,
          selectedProductIdRef.current,
          showFiltersRef.current
        )
        setPreferredRefreshStrategy(strategy)
        setRecommendationsNeedRefresh(false)

        writeRecommendationViewCache({
          userId: user.id,
          data: data as ProductsResponse,
          selectedProductId: selectedProductIdRef.current,
          showFilters: showFiltersRef.current,
          autoRefreshOnLike: autoRefreshOnLikeRef.current,
          preferredRefreshStrategy: strategy,
          recommendationsNeedRefresh: false,
          savedAt: Date.now(),
        })
      } else {
        console.error('Failed to fetch products:', data.message)
      }
    } catch (error) {
      console.error('Error fetching products:', error)
    } finally {
      setIsLoadingProducts(false)
      setIsRefreshingRecommendations(false)
    }
  }, [user, applyProductsResponse])

  React.useEffect(() => {
    if (user) {
      const cachedView = readRecommendationViewCache(user.id)
      if (cachedView?.data?.success) {
        const cachedStrategy = cachedView.preferredRefreshStrategy === 'content' ? 'content' : 'hybrid'
        applyProductsResponse(
          cachedView.data,
          cachedView.selectedProductId,
          cachedView.showFilters
        )
        setAutoRefreshOnLike(cachedView.autoRefreshOnLike ?? true)
        setPreferredRefreshStrategy(cachedStrategy)
        setRecommendationsNeedRefresh(Boolean(cachedView.recommendationsNeedRefresh))
        setIsLoadingProducts(false)

        const shouldRevalidate = Date.now() - cachedView.savedAt > CLIENT_REVALIDATE_AFTER_MS
        if (shouldRevalidate) {
          void fetchProducts({ showLoading: false, strategy: cachedStrategy })
        }
        return
      }

      let cancelled = false

      const loadInitialRecommendations = async () => {
        await fetchProducts({ showLoading: true, strategy: 'content' })
        if (cancelled) return
        await fetchProducts({ showLoading: false, strategy: 'hybrid' })
      }

      void loadInitialRecommendations()

      return () => {
        cancelled = true
      }
    }
  }, [user, applyProductsResponse, fetchProducts])

  React.useEffect(() => {
    if (!user) return
    if (products.length === 0 && likedProducts.length === 0) return

    writeRecommendationViewCache({
      userId: user.id,
      data: {
        success: true,
        products,
        likedProducts,
        mode: recommendationMode,
        engine: recommendationEngine,
      },
      selectedProductId: selectedProduct?.id || null,
      showFilters,
      autoRefreshOnLike,
      preferredRefreshStrategy,
      recommendationsNeedRefresh,
      savedAt: Date.now(),
    })
  }, [
    user,
    products,
    likedProducts,
    recommendationMode,
    recommendationEngine,
    selectedProduct,
    showFilters,
    autoRefreshOnLike,
    preferredRefreshStrategy,
    recommendationsNeedRefresh,
  ])

  const applyOptimisticLikeToggle = React.useCallback(
    (product: FashionProduct, action: 'like' | 'unlike') => {
      const optimisticProduct = {
        ...product,
        is_liked: action === 'like',
      }

      if (action === 'like') {
        setProducts((prev) => prev.filter((entry) => entry.id !== product.id))
        setLikedProducts((prev) => upsertProduct(prev, optimisticProduct))
      } else {
        setLikedProducts((prev) => prev.filter((entry) => entry.id !== product.id))
        setProducts((prev) => {
          const updated = prev.map((entry) => (
            entry.id === product.id
              ? optimisticProduct
              : entry
          ))
          return updated.some((entry) => entry.id === product.id)
            ? updated
            : [optimisticProduct, ...updated]
        })
      }

      setSelectedProduct((current) => (
        current?.id === product.id
          ? optimisticProduct
          : current
      ))
    },
    []
  )

  const handleToggleLike = async (product: FashionProduct) => {
    if (!session?.access_token || pendingLikeProductId) return

    const action = likedItemIds.has(product.id) ? 'unlike' : 'like'
    const previousProducts = products
    const previousLikedProducts = likedProducts
    const previousSelectedProduct = selectedProduct

    applyOptimisticLikeToggle(product, action)
    setPendingLikeProductId(product.id)

    try {
      const response = await fetch('/api/users/likes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          productId: product.id,
          action,
        }),
      })

      const data = await response.json()
      if (!response.ok || !data.success) {
        throw new Error(data.message || 'Failed to update liked products')
      }
      
      if (autoRefreshOnLike) {
        void fetchProducts({
          showLoading: false,
          strategy: preferredRefreshStrategy,
          forceRefresh: true,
        })
      } else {
        setRecommendationsNeedRefresh(true)
      }
    } catch (error) {
      console.error('Failed to toggle like:', error)
      setProducts(previousProducts)
      setLikedProducts(previousLikedProducts)
      setSelectedProduct(previousSelectedProduct)
    } finally {
      setPendingLikeProductId(null)
    }
  }

  const handleRefreshRecommendations = (strategy: RecommendationRefreshStrategy) => {
    setPreferredRefreshStrategy(strategy)
    void fetchProducts({
      showLoading: products.length === 0 && likedProducts.length === 0,
      strategy,
      forceRefresh: true,
    })
  }

  // Get similar products based on domain and gender
  const getSimilarProducts = (product: FashionProduct): FashionProduct[] => {
    const productDomain = extractDomain(product.product_url);
    return allProducts
      .filter(p =>
        p.id !== product.id &&
        (p.gender === product.gender || extractDomain(p.product_url) === productDomain)
        )
      .slice(0, 4)
  }

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return null // Will redirect to login
  }

  const selectedModelLabel =
    preferredRefreshStrategy === 'hybrid'
      ? 'Two-Tower Neural Network selected'
      : 'Content Filtering selected'

  const selectedModelDescription =
    preferredRefreshStrategy === 'hybrid'
      ? 'The Two-Tower neural network blends multiple signals to surface more diverse and novel recommendations.'
      : 'Content filtering compares image embeddings only, so it is best for finding items that look very similar to your liked products and uploaded photos.'

  const engineBadgeLabel =
    recommendationEngine === 'faiss_two_tower_hybrid'
      ? 'FAISS + two-tower active'
      : recommendationEngine === 'faiss_two_tower_image_hybrid'
        ? 'Text + image + two-tower active'
      : recommendationEngine === 'two_tower_or_cosine_fallback'
        ? 'Fallback ranking active'
        : 'Latest-products fallback'

  const engineBadgeVariant =
    recommendationEngine === 'faiss_two_tower_hybrid' ||
    recommendationEngine === 'faiss_two_tower_image_hybrid'
      ? 'default'
      : 'outline'

  return (
    <ChatLayout>
      <SidebarProvider
        style={
          {
            "--sidebar-width": "calc(var(--spacing) * 72)",
            "--header-height": "calc(var(--spacing) * 12)",
          } as React.CSSProperties
        }
      >
        <AppSidebar variant="inset" />
        <SidebarInset>
          <SiteHeader />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">

                {/* Hero Section */}
                <div className="px-4 lg:px-6">
                  <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-background p-6 rounded-lg border">
                    <div className="flex items-center gap-2 mb-2">
                      <Sparkles className="h-5 w-5 text-primary" />
                      <h1 className="text-2xl font-bold">Your Fashion Recommendations</h1>
                    </div>
                    <p className="text-muted-foreground mb-4">
                      Discover curated items tailored to your personal style. Our AI analyzes your preferences to find pieces you'll love.
                    </p>

                    {/* Refresh Settings Block */}
                    <div className="mb-4 flex flex-col gap-3 rounded-lg border bg-background/80 p-4">
                      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-3">
                            <Switch
                              id="auto-refresh-recommendations"
                              checked={autoRefreshOnLike}
                              onCheckedChange={setAutoRefreshOnLike}
                            />
                            <Label htmlFor="auto-refresh-recommendations">
                              Auto-refresh recommendations after likes/unlikes
                            </Label>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {autoRefreshOnLike
                              ? `Likes rerun your ${preferredRefreshStrategy === 'hybrid' ? 'Two-Tower neural network' : 'content filtering model'} in the background.`
                              : 'Likes update instantly, and you choose when to rerun recommendations.'}
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <Button
                            variant={preferredRefreshStrategy === 'hybrid' ? 'default' : 'outline'}
                            size="sm"
                            disabled={isRefreshingRecommendations}
                            onClick={() => handleRefreshRecommendations('hybrid')}
                          >
                            <RefreshCw className={`mr-2 h-4 w-4 ${isRefreshingRecommendations && preferredRefreshStrategy === 'hybrid' ? 'animate-spin' : ''}`} />
                            Refresh Hybrid Model
                          </Button>
                          <Button
                            variant={preferredRefreshStrategy === 'content' ? 'default' : 'outline'}
                            size="sm"
                            disabled={isRefreshingRecommendations}
                            onClick={() => handleRefreshRecommendations('content')}
                          >
                            <RefreshCw className={`mr-2 h-4 w-4 ${isRefreshingRecommendations && preferredRefreshStrategy === 'content' ? 'animate-spin' : ''}`} />
                            Refresh Content Filtering
                          </Button>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {selectedModelDescription}
                      </p>
                      {recommendationsNeedRefresh && (
                        <p className="text-sm text-amber-700">
                          Your likes changed. Refresh recommendations when you are ready.
                        </p>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-2">
                      {recommendationMode === 'personalized_image' || recommendationMode === 'personalized_hybrid_api' ? (
                        <>
                          <Badge variant="secondary">{selectedModelLabel}</Badge>
                          <Badge variant={engineBadgeVariant}>{engineBadgeLabel}</Badge>
                          {isRefreshingRecommendations && recommendationMode === 'personalized_image' && (
                            <Badge variant="outline">Upgrading to hybrid...</Badge>
                          )}
                          <Badge variant="outline">{products.length} recommendations</Badge>
                        </>
                      ) : (
                        <>
                          <Badge variant="secondary">Latest products</Badge>
                          <Badge variant={engineBadgeVariant}>{engineBadgeLabel}</Badge>
                          <Badge variant="outline">Set likes at signup for personalization</Badge>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                {/* Already Liked Section */}
                {likedProducts.length > 0 && (
                  <div className="px-4 lg:px-6">
                    <div className="mb-3">
                      <h2 className="text-xl font-semibold">Already Liked</h2>
                      <p className="text-sm text-muted-foreground">
                        {likedProducts.length} items you liked
                      </p>
                    </div>
                    <FashionGrid
                      products={likedProducts}
                      onProductClick={setSelectedProduct}
                      likedItems={likedItemIds}
                      onToggleLike={handleToggleLike}
                      pendingLikeProductId={pendingLikeProductId}
                      maxTopPickCount={0}
                    />
                  </div>
                )}

                {/* Filters Bar */}
                <div className="px-4 lg:px-6">
                  <div className="flex items-center justify-between border-b pb-4">
                    <div>
                      <h2 className="text-xl font-semibold">Recommended for You</h2>
                      <p className="text-sm text-muted-foreground">
                        Showing {filteredProducts.length} of {products.length} items
                      </p>
                    </div>
                    <Button
                      variant={showFilters ? "secondary" : "outline"}
                      size="sm"
                      onClick={() => setShowFilters(!showFilters)}
                    >
                      <Filter className="h-4 w-4 mr-2" />
                      {showFilters ? 'Hide Filters' : 'Show Filters'}
                    </Button>
                  </div>
                </div>

                {/* Main Content Area (Grid + Sidebar) */}
                <div className="px-4 lg:px-6 flex flex-col md:flex-row gap-8 items-start">
                  
                  {/* Filtering Sidebar Panel */}
                  {showFilters && (
                    <div className="w-full md:w-64 flex-shrink-0 space-y-6 border rounded-lg p-5 bg-card text-card-foreground">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold">Filters</h3>
                        <Button variant="ghost" size="sm" onClick={clearFilters} className="h-8 px-2 text-xs">
                          Clear All
                        </Button>
                      </div>

                      {/* Text Search */}
                      <div className="space-y-3">
                        <h4 className="text-sm font-semibold">Search Items</h4>
                        <div className="relative">
                          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                          <Input
                            type="text"
                            placeholder="e.g. shirt, jeans..."
                            className="pl-9 h-9 text-sm"
                            value={activeFilters.searchQuery}
                            onChange={(e) => setActiveFilters(prev => ({ ...prev, searchQuery: e.target.value }))}
                          />
                        </div>
                      </div>

                      {/* Min/Max Price Inputs */}
                      <div className="space-y-3">
                        <h4 className="text-sm font-semibold">Price Range</h4>
                        <div className="flex items-center space-x-2">
                          <div className="flex flex-col w-full">
                            <span className="text-xs text-muted-foreground mb-1">Min ($)</span>
                            <Input
                              type="number"
                              min={0}
                              max={activeFilters.maxPrice}
                              value={activeFilters.minPrice || ''}
                              onChange={(e) => setActiveFilters(prev => ({ ...prev, minPrice: parseInt(e.target.value) || 0 }))}
                              className="h-8 text-sm px-2"
                              placeholder="0"
                            />
                          </div>
                          <span className="text-muted-foreground mt-4">-</span>
                          <div className="flex flex-col w-full">
                            <span className="text-xs text-muted-foreground mb-1">Max ($)</span>
                            <Input
                              type="number"
                              min={activeFilters.minPrice}
                              max={absoluteMaxPrice}
                              value={activeFilters.maxPrice || ''}
                              onChange={(e) => setActiveFilters(prev => ({ ...prev, maxPrice: parseInt(e.target.value) || 0 }))}
                              className="h-8 text-sm px-2"
                              placeholder={absoluteMaxPrice.toString()}
                            />
                          </div>
                        </div>
                      </div>

                      {/* Genders */}
                      {availableGenders.length > 0 && (
                        <div className="space-y-3">
                          <h4 className="text-sm font-semibold">Gender</h4>
                          <div className="space-y-2 max-h-40 overflow-y-auto pr-2">
                            {availableGenders.map(gender => (
                              <label key={gender} className="flex items-center space-x-2 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={activeFilters.genders.includes(gender)}
                                  onChange={() => toggleFilterArray('genders', gender)}
                                  className="rounded border-gray-300 text-primary focus:ring-primary"
                                />
                                <span className="text-sm text-muted-foreground capitalize">{gender}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Brands (Derived from Domain) */}
                      {availableBrands.length > 0 && (
                        <div className="space-y-3">
                          <h4 className="text-sm font-semibold">Brands / Domains</h4>
                          <div className="space-y-2 max-h-60 overflow-y-auto pr-2">
                            {availableBrands.map(brand => (
                              <label key={brand} className="flex items-center space-x-2 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={activeFilters.brands.includes(brand)}
                                  onChange={() => toggleFilterArray('brands', brand)}
                                  className="rounded border-gray-300 text-primary focus:ring-primary"
                                />
                                <span className="text-sm text-muted-foreground">{brand}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Product Grid Area */}
                  <div className="flex-1 w-full">
                  {isLoadingProducts ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="text-muted-foreground">Loading products...</div>
                    </div>
                  ) : filteredProducts.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12">
                      <div className="text-muted-foreground mb-2">No products found</div>
                      <p className="text-sm text-muted-foreground">Adjust filters or check back soon</p>
                      <Button variant="link" onClick={clearFilters}>
                        Clear all filters
                      </Button>
                    </div>
                  ) : (
                    <FashionGrid
                      products={filteredProducts}
                      onProductClick={setSelectedProduct}
                      likedItems={likedItemIds}
                      onToggleLike={handleToggleLike}
                      pendingLikeProductId={pendingLikeProductId}
                      maxTopPickCount={8}
                    />
                  )}
                  </div>

                </div>
              </div>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>

      {/* Product Detail Panel */}
      {selectedProduct && (
        <ProductDetailPanel
          product={selectedProduct}
          similarProducts={getSimilarProducts(selectedProduct)}
          onClose={() => setSelectedProduct(null)}
          onSimilarProductClick={setSelectedProduct}
          isLiked={likedItemIds.has(selectedProduct.id)}
          isTogglingLike={pendingLikeProductId === selectedProduct.id}
          onToggleLike={handleToggleLike}
        />
      )}
    </ChatLayout>
  )
}
