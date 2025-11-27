"use client"

import { type Icon } from "@tabler/icons-react"
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Badge } from '@/components/ui/badge'
import { useCart } from '@/lib/cart-context'

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar'

export function NavMain({
  items,
}: {
  items: {
    title: string
    url: string
    icon?: Icon
  }[]
}) {
  const pathname = usePathname()
  const { getCartCount } = useCart()
  const cartCount = getCartCount()

  return (
    <SidebarGroup>
      <SidebarGroupContent className="flex flex-col gap-2">
        <SidebarMenu>
          {items.map((item) => (
            <SidebarMenuItem key={item.title}>
              <SidebarMenuButton
                tooltip={item.title}
                asChild
                isActive={pathname === item.url}
              >
                <Link href={item.url} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {item.icon && <item.icon />}
                    <span>{item.title}</span>
                  </div>
                  {item.title === "Cart" && cartCount > 0 && (
                    <Badge variant="secondary" className="ml-auto">
                      {cartCount}
                    </Badge>
                  )}
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
