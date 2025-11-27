"use client"

import * as React from "react"
import {
  IconSparkles,
  IconHelp,
  IconSearch,
  IconSettings,
  IconBell,
  IconShoppingBag,
} from "@tabler/icons-react"

import { NavMain } from '@/components/nav-main'
import { NavSecondary } from '@/components/nav-secondary'
import { NavUser } from '@/components/nav-user'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar'
import { useAuth } from '@/lib/auth-context'
import { useUserData } from '@/lib/hooks/use-user-data'
import { useCart } from '@/lib/cart-context'
import { Badge } from '@/components/ui/badge'

const data = {
  navMain: [
    {
      title: "Recommendations",
      url: "/",
      icon: IconSparkles,
    },
    {
      title: "Cart",
      url: "/cart",
      icon: IconShoppingBag,
    },
    {
      title: "Notifications",
      url: "/notifications",
      icon: IconBell,
    },
  ],
  navSecondary: [
    {
      title: "Settings",
      url: "#",
      icon: IconSettings,
    },
    {
      title: "Get Help",
      url: "#",
      icon: IconHelp,
    },
    {
      title: "Search",
      url: "#",
      icon: IconSearch,
    },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { user } = useAuth()
  const { userData: dbUserData } = useUserData()
  
  const userData = {
    name: dbUserData?.name || user?.user_metadata?.name || "User",
    email: user?.email || "user@example.com",
    avatar: "/placeholder-user.jpg",
  }
  
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <a href="#" className="flex items-center gap-2">
                <img src="/favicon.avif" alt="Threaded" className="h-6 w-6 rounded" />
                <span className="text-base font-semibold">Threaded</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={userData} />
      </SidebarFooter>
    </Sidebar>
  )
}
