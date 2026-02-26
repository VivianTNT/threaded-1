"use client"

import * as React from "react"
import { X, Heart, ShoppingCart, ExternalLink, Share2, Star } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { FashionProduct } from "@/lib/types/fashion-product"
import { cn } from "@/lib/utils"
import { useCart } from "@/lib/cart-context"

interface ProductDetailPanelProps {
  product: FashionProduct
  similarProducts: FashionProduct[]
  onClose: () => void
  onSimilarProductClick: (product: FashionProduct) => void
}

export function ProductDetailPanel({
  product,
  similarProducts,
  onClose,
  onSimilarProductClick,
}: ProductDetailPanelProps) {
  const [isLiked, setIsLiked] = React.useState(false)
  const { addToCart, isInCart, removeFromCart } = useCart()
  const inCart = isInCart(product.id)

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed inset-y-0 right-0 w-full md:w-[600px] bg-background border-l shadow-2xl z-50 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b flex-shrink-0">
          <h2 className="text-lg font-semibold">Product Details</h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-5 w-5" />
          </Button>
        </div>

        <ScrollArea className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6">
          {/* Product Image */}
          <div className="relative aspect-[3/4] overflow-hidden rounded-lg bg-muted">
            <img
              src={product.image_url || 'https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=800&h=1200&fit=crop'}
              alt={product.name}
              className="h-full w-full object-cover"
              onError={(e) => {
                const target = e.target as HTMLImageElement
                target.src = 'https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=800&h=1200&fit=crop'
              }}
            />
            {!product.in_stock && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <Badge variant="secondary" className="text-base">Out of Stock</Badge>
              </div>
            )}
          </div>

          {/* Product Info */}
          <div className="space-y-4">
            <div>
              <div className="text-sm text-muted-foreground font-medium mb-1">{product.brand}</div>
              <h3 className="text-2xl font-semibold">{product.name}</h3>
            </div>

            <div className="flex items-center justify-between">
              <div className="text-3xl font-bold">
                {product.price && product.price > 0 ? `$${product.price.toLocaleString()}` : 'Price not available'}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setIsLiked(!isLiked)}
                >
                  <Heart className={cn("h-5 w-5", isLiked && "fill-current text-red-500")} />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                >
                  <Share2 className="h-5 w-5" />
                </Button>
              </div>
            </div>

            {/* Badges */}
            <div className="flex flex-wrap gap-2">
              {product.style.map((style) => (
                <Badge key={style} variant="secondary">{style}</Badge>
              ))}
              {product.sustainability_rating && product.sustainability_rating >= 4 && (
                <Badge variant="outline" className="border-green-500 text-green-700">
                  <Star className="h-3 w-3 mr-1 fill-current" />
                  Sustainable
                </Badge>
              )}
            </div>

            {/* Description */}
            {product.description && (
              <div>
                <h4 className="font-semibold mb-2">Description</h4>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {product.description}
                </p>
              </div>
            )}

            {/* Details - Only show if values exist */}
            {(product.category || product.gender || (product.material && product.material.length > 0) || (product.season && product.season.length > 0)) && (
              <div className="grid grid-cols-2 gap-4 text-sm">
                {product.category && (
                  <div>
                    <div className="text-muted-foreground">Category</div>
                    <div className="font-medium">{product.category}</div>
                  </div>
                )}
                {product.gender && (
                  <div>
                    <div className="text-muted-foreground">Gender</div>
                    <div className="font-medium">{product.gender}</div>
                  </div>
                )}
                {product.material && product.material.length > 0 && (
                  <div>
                    <div className="text-muted-foreground">Material</div>
                    <div className="font-medium">{product.material.join(', ')}</div>
                  </div>
                )}
                {product.season && product.season.length > 0 && (
                  <div>
                    <div className="text-muted-foreground">Season</div>
                    <div className="font-medium">{product.season.join(', ')}</div>
                  </div>
                )}
              </div>
            )}

            {/* Colors - Only show if colors exist */}
            {product.color && product.color.length > 0 && (
              <div>
                <div className="text-sm text-muted-foreground mb-2">Available Colors</div>
                <div className="flex gap-2">
                  {product.color.map((color) => (
                    <Badge key={color} variant="outline">{color}</Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Sizes - Only show if sizes exist */}
            {product.sizes_available && product.sizes_available.length > 0 && (
              <div>
                <div className="text-sm text-muted-foreground mb-2">Available Sizes</div>
                <div className="flex flex-wrap gap-2">
                  {product.sizes_available.map((size) => (
                    <Button key={size} variant="outline" size="sm">{size}</Button>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendation Reason */}
            {product.recommendation_reason && (
              <div className="bg-primary/5 p-4 rounded-lg">
                <h4 className="font-semibold mb-1 text-sm">Why we recommend this</h4>
                <p className="text-sm text-muted-foreground">{product.recommendation_reason}</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3">
              <Button
                className="flex-1"
                size="lg"
                variant={inCart ? "outline" : "default"}
                onClick={() => inCart ? removeFromCart(product.id) : addToCart(product)}
                disabled={!product.in_stock}
              >
                <ShoppingCart className="h-5 w-5 mr-2" />
                {inCart ? 'Remove from Cart' : 'Add to Cart'}
              </Button>
              {product.product_url && (
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => window.open(product.product_url!, '_blank')}
                >
                  <ExternalLink className="h-5 w-5 mr-2" />
                  View on Site
                </Button>
              )}
            </div>
          </div>

          {/* Similar Products */}
          {similarProducts.length > 0 && (
            <>
              <Separator />
              <div className="space-y-4">
                <h4 className="font-semibold">Similar Items</h4>
                <div className="grid grid-cols-2 gap-4">
                  {similarProducts.slice(0, 4).map((similar) => (
                    <Card
                      key={similar.id}
                      className="cursor-pointer overflow-hidden hover:shadow-md transition-shadow"
                      onClick={() => onSimilarProductClick(similar)}
                    >
                      <div className="relative aspect-square overflow-hidden bg-muted">
                        <img
                          src={similar.image_url}
                          alt={similar.name}
                          className="h-full w-full object-cover"
                        />
                      </div>
                      <div className="p-2 space-y-1">
                        <div className="text-xs text-muted-foreground">{similar.brand}</div>
                        <div className="text-sm font-medium line-clamp-2">{similar.name}</div>
                        <div className="text-sm font-semibold">${similar.price.toLocaleString()}</div>
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </ScrollArea>
      </div>
    </>
  )
}

function Card({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("border rounded-lg", className)} {...props}>
      {children}
    </div>
  )
}
