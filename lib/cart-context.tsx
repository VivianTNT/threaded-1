"use client"

import React, { createContext, useContext, useState, useEffect } from "react"
import { FashionProduct } from "@/lib/types/fashion-product"
import { toast } from "sonner"

interface CartItem {
  product: FashionProduct
  quantity: number
  selectedSize?: string
  addedAt: string
}

interface CartContextType {
  items: CartItem[]
  addToCart: (product: FashionProduct, size?: string) => void
  removeFromCart: (productId: string) => void
  updateQuantity: (productId: string, quantity: number) => void
  isInCart: (productId: string) => boolean
  getCartTotal: () => number
  getCartCount: () => number
  clearCart: () => void
}

const CartContext = createContext<CartContextType | undefined>(undefined)

export function CartProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<CartItem[]>([])

  // Load cart from localStorage on mount
  useEffect(() => {
    const savedCart = localStorage.getItem('threaded-cart')
    if (savedCart) {
      try {
        setItems(JSON.parse(savedCart))
      } catch (error) {
        console.error('Error loading cart:', error)
      }
    }
  }, [])

  // Save cart to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('threaded-cart', JSON.stringify(items))
  }, [items])

  const addToCart = (product: FashionProduct, size?: string) => {
    setItems(prev => {
      const existingItem = prev.find(item => item.product.id === product.id)

      if (existingItem) {
        // Update quantity if item already exists
        toast.success('Updated quantity in cart')
        return prev.map(item =>
          item.product.id === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        )
      } else {
        // Add new item
        toast.success('Added to cart', {
          description: product.name
        })
        return [...prev, {
          product,
          quantity: 1,
          selectedSize: size,
          addedAt: new Date().toISOString()
        }]
      }
    })
  }

  const removeFromCart = (productId: string) => {
    setItems(prev => prev.filter(item => item.product.id !== productId))
    toast.success('Removed from cart')
  }

  const updateQuantity = (productId: string, quantity: number) => {
    if (quantity <= 0) {
      removeFromCart(productId)
      return
    }

    setItems(prev =>
      prev.map(item =>
        item.product.id === productId
          ? { ...item, quantity }
          : item
      )
    )
  }

  const isInCart = (productId: string) => {
    return items.some(item => item.product.id === productId)
  }

  const getCartTotal = () => {
    return items.reduce((total, item) => total + (item.product.price * item.quantity), 0)
  }

  const getCartCount = () => {
    return items.reduce((count, item) => count + item.quantity, 0)
  }

  const clearCart = () => {
    setItems([])
    toast.success('Cart cleared')
  }

  return (
    <CartContext.Provider
      value={{
        items,
        addToCart,
        removeFromCart,
        updateQuantity,
        isInCart,
        getCartTotal,
        getCartCount,
        clearCart
      }}
    >
      {children}
    </CartContext.Provider>
  )
}

export function useCart() {
  const context = useContext(CartContext)
  if (context === undefined) {
    throw new Error("useCart must be used within a CartProvider")
  }
  return context
}
