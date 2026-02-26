"use client"

import * as React from "react"
import { Heart, ShoppingCart, ExternalLink, Sparkles } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { FashionProduct } from "@/lib/types/fashion-product"
import { cn } from "@/lib/utils"
import { useCart } from "@/lib/cart-context"

interface FashionGridProps {
  products: FashionProduct[]
  onProductClick: (product: FashionProduct) => void
}

export function FashionGrid({ products, onProductClick }: FashionGridProps) {
  const [likedItems, setLikedItems] = React.useState<Set<string>>(new Set())
  const { addToCart, isInCart } = useCart()

  const toggleLike = (productId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setLikedItems(prev => {
      const newSet = new Set(prev)
      if (newSet.has(productId)) {
        newSet.delete(productId)
      } else {
        newSet.add(productId)
      }
      return newSet
    })
  }

  const handleAddToCart = (product: FashionProduct, e: React.MouseEvent) => {
    e.stopPropagation()
    addToCart(product)
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {products.map((product) => (
        <Card
          key={product.id}
          className="group cursor-pointer overflow-hidden transition-all hover:shadow-lg"
          onClick={() => onProductClick(product)}
        >
          <div className="relative aspect-[3/4] overflow-hidden bg-muted">
            <img
              src={product.image_url || 'https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=800&h=1200&fit=crop'}
              alt={product.name}
              className="h-full w-full object-cover transition-transform group-hover:scale-105"
              onError={(e) => {
                const target = e.target as HTMLImageElement
                target.src = 'https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=800&h=1200&fit=crop'
              }}
            />
            <div className="absolute top-2 right-2 flex flex-col gap-2">
              <Button
                size="icon"
                variant="secondary"
                className={cn(
                  "h-8 w-8 rounded-full bg-white/90 backdrop-blur-sm hover:bg-white",
                  likedItems.has(product.id) && "text-red-500"
                )}
                onClick={(e) => toggleLike(product.id, e)}
              >
                <Heart className={cn("h-4 w-4", likedItems.has(product.id) && "fill-current")} />
              </Button>
            </div>
            {product.recommendation_score && product.recommendation_score >= 90 && (
              <div className="absolute top-2 left-2">
                <Badge className="bg-primary/90 backdrop-blur-sm flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  Top Pick
                </Badge>
              </div>
            )}
            {!product.in_stock && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <Badge variant="secondary">Out of Stock</Badge>
              </div>
            )}
          </div>
          <div className="p-4 space-y-2">
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground font-medium">{product.brand}</div>
              <h3 className="font-medium text-sm line-clamp-2 leading-tight">{product.name}</h3>
            </div>
            <div className="flex items-center justify-between">
              <div className="font-semibold">
                {product.price && product.price > 0 ? `$${product.price.toLocaleString()}` : 'Price not available'}
              </div>
              {product.sustainability_rating && product.sustainability_rating >= 4 && (
                <Badge variant="outline" className="text-xs">
                  Sustainable
                </Badge>
              )}
            </div>
            {product.recommendation_reason && (
              <p className="text-xs text-muted-foreground line-clamp-2">
                {product.recommendation_reason}
              </p>
            )}
            <div className="flex gap-2 pt-2">
              <Button
                size="sm"
                variant={isInCart(product.id) ? "default" : "outline"}
                className="flex-1"
                onClick={(e) => handleAddToCart(product, e)}
                disabled={!product.in_stock}
              >
                <ShoppingCart className="h-3.5 w-3.5 mr-1.5" />
                {isInCart(product.id) ? 'In Cart' : 'Add to Cart'}
              </Button>
              {product.product_url && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation()
                    window.open(product.product_url!, '_blank')
                  }}
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
