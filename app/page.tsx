'use client'

import * as React from 'react'
import { AppSidebar } from '@/components/app-sidebar'
import { SiteHeader } from '@/components/site-header'
import { ChatLayout } from '@/components/chat-layout'
import { useRequireAuth } from '@/lib/auth-utils'
import { FashionGrid } from '@/components/fashion-recommendations/fashion-grid'
import { ProductDetailPanel } from '@/components/fashion-recommendations/product-detail-panel'
import { FashionProduct } from '@/lib/types/fashion-product'
import { Sparkles, Filter } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

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

type RecommendationViewCache = {
  userId: string
  data: ProductsResponse
  selectedProductId: string | null
  showFilters: boolean
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

export default function Page() {
  const { user, session, isLoading } = useRequireAuth()
  const [selectedProduct, setSelectedProduct] = React.useState<FashionProduct | null>(null)
  const [products, setProducts] = React.useState<FashionProduct[]>([])
  const [likedProducts, setLikedProducts] = React.useState<FashionProduct[]>([])
  const [recommendationMode, setRecommendationMode] = React.useState<string>('latest_fallback')
  const [recommendationEngine, setRecommendationEngine] = React.useState<string>('latest_products_fallback')
  const [showFilters, setShowFilters] = React.useState(false)
  const [isLoadingProducts, setIsLoadingProducts] = React.useState(true)
  const [pendingLikeProductId, setPendingLikeProductId] = React.useState<string | null>(null)
  const allProducts = React.useMemo(() => {
    return [...likedProducts, ...products]
  }, [likedProducts, products])
  const likedItemIds = React.useMemo(() => {
    return new Set(likedProducts.map((product) => product.id))
  }, [likedProducts])

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

  const fetchProducts = async ({ showLoading = true }: { showLoading?: boolean } = {}) => {
    if (!user) return

    try {
      if (showLoading) {
        setIsLoadingProducts(true)
      }
      const params = new URLSearchParams({
        limit: '50',
        userId: user.id || '',
        userEmail: user.email || '',
      })
      const response = await fetch(`/api/products?${params.toString()}`)
      const data = await response.json()

      if (data.success) {
        applyProductsResponse(
          data as ProductsResponse,
          selectedProduct?.id || null,
          showFilters
        )

        writeRecommendationViewCache({
          userId: user.id,
          data: data as ProductsResponse,
          selectedProductId: selectedProduct?.id || null,
          showFilters,
          savedAt: Date.now(),
        })
      } else {
        console.error('Failed to fetch products:', data.message)
      }
    } catch (error) {
      console.error('Error fetching products:', error)
    } finally {
      setIsLoadingProducts(false)
    }
  }

  React.useEffect(() => {
    if (user) {
      const cachedView = readRecommendationViewCache(user.id)
      if (cachedView?.data?.success) {
        applyProductsResponse(
          cachedView.data,
          cachedView.selectedProductId,
          cachedView.showFilters
        )
        setIsLoadingProducts(false)

        const shouldRevalidate = Date.now() - cachedView.savedAt > CLIENT_REVALIDATE_AFTER_MS
        if (shouldRevalidate) {
          void fetchProducts({ showLoading: false })
        }
        return
      }

      void fetchProducts({ showLoading: true })
    }
  }, [user, applyProductsResponse])

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
      savedAt: Date.now(),
    })
  }, [user, products, likedProducts, recommendationMode, recommendationEngine, selectedProduct, showFilters])

  const handleToggleLike = async (product: FashionProduct) => {
    if (!session?.access_token || pendingLikeProductId) return

    const action = likedItemIds.has(product.id) ? 'unlike' : 'like'
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

      setSelectedProduct((current) => {
        if (!current || current.id !== product.id) return current
        return {
          ...current,
          is_liked: action === 'like',
        }
      })

      await fetchProducts()
    } catch (error) {
      console.error('Failed to toggle like:', error)
    } finally {
      setPendingLikeProductId(null)
    }
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

  // Get similar products based on category and style
  const getSimilarProducts = (product: FashionProduct): FashionProduct[] => {
    return allProducts
      .filter(p =>
        p.id !== product.id &&
        (p.category === product.category ||
         p.style.some(s => product.style.includes(s)))
      )
      .slice(0, 4)
  }

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
                    <div className="flex flex-wrap gap-2">
                      {recommendationMode === 'personalized_image' || recommendationMode === 'personalized_hybrid_api' ? (
                        <>
                          <Badge variant="secondary">
                            {recommendationMode === 'personalized_hybrid_api'
                              ? 'Personalized by hybrid ranker'
                              : 'Personalized by image likes'}
                          </Badge>
                          <Badge variant={engineBadgeVariant}>
                            {engineBadgeLabel}
                          </Badge>
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

                {likedProducts.length > 0 && (
                  <div className="px-4 lg:px-6">
                    <div className="mb-3">
                      <h2 className="text-xl font-semibold">Already Liked</h2>
                      <p className="text-sm text-muted-foreground">
                        {likedProducts.length} items you selected during signup
                      </p>
                    </div>
                    <FashionGrid
                      products={likedProducts}
                      onProductClick={setSelectedProduct}
                      likedItems={likedItemIds}
                      onToggleLike={handleToggleLike}
                      pendingLikeProductId={pendingLikeProductId}
                    />
                  </div>
                )}

                {/* Filters Bar */}
                <div className="px-4 lg:px-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-semibold">Recommended for You</h2>
                      <p className="text-sm text-muted-foreground">
                        {products.length} items
                      </p>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowFilters(!showFilters)}
                    >
                      <Filter className="h-4 w-4 mr-2" />
                      Filters
                    </Button>
                  </div>
                </div>

                {/* Product Grid */}
                <div className="px-4 lg:px-6">
                  {isLoadingProducts ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="text-muted-foreground">Loading products...</div>
                    </div>
                  ) : products.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12">
                      <div className="text-muted-foreground mb-2">No products found</div>
                      <p className="text-sm text-muted-foreground">Check back soon for new recommendations</p>
                    </div>
                  ) : (
                    <FashionGrid
                      products={products}
                      onProductClick={setSelectedProduct}
                      likedItems={likedItemIds}
                      onToggleLike={handleToggleLike}
                      pendingLikeProductId={pendingLikeProductId}
                    />
                  )}
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
