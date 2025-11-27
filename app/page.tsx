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

export default function Page() {
  const { user, isLoading } = useRequireAuth()
  const [selectedProduct, setSelectedProduct] = React.useState<FashionProduct | null>(null)
  const [products, setProducts] = React.useState<FashionProduct[]>([])
  const [showFilters, setShowFilters] = React.useState(false)
  const [isLoadingProducts, setIsLoadingProducts] = React.useState(true)

  // Fetch products from Penn database
  React.useEffect(() => {
    const fetchProducts = async () => {
      try {
        setIsLoadingProducts(true)
        const response = await fetch('/api/products?limit=50')
        const data = await response.json()

        if (data.success) {
          setProducts(data.products)
        } else {
          console.error('Failed to fetch products:', data.message)
        }
      } catch (error) {
        console.error('Error fetching products:', error)
      } finally {
        setIsLoadingProducts(false)
      }
    }

    if (user) {
      fetchProducts()
    }
  }, [user])

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
    return products
      .filter(p =>
        p.id !== product.id &&
        (p.category === product.category ||
         p.style.some(s => product.style.includes(s)))
      )
      .slice(0, 4)
  }

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
                      <Badge variant="secondary">12 New Items Today</Badge>
                      <Badge variant="outline">95% Match Rate</Badge>
                    </div>
                  </div>
                </div>

                {/* Filters Bar */}
                <div className="px-4 lg:px-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-semibold">Recommended for You</h2>
                      <p className="text-sm text-muted-foreground">
                        {products.length} items • Updated 30 minutes ago
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
        />
      )}
    </ChatLayout>
  )
}
