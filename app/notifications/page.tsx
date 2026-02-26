'use client'

import { AppSidebar } from '@/components/app-sidebar'
import { SiteHeader } from '@/components/site-header'
import { ChatLayout } from '@/components/chat-layout'
import { useRequireAuth } from '@/lib/auth-utils'
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar'
import { mockNotifications } from '@/lib/data/mock-fashion-products'
import { Bell, Sparkles, TrendingDown, Package, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { formatDistanceToNow } from 'date-fns'
import * as React from 'react'

export default function NotificationsPage() {
  const { user, isLoading } = useRequireAuth()
  const [notifications, setNotifications] = React.useState(mockNotifications)

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  const markAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(notif =>
        notif.id === id ? { ...notif, is_read: true } : notif
      )
    )
  }

  const deleteNotification = (id: string) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id))
  }

  const markAllAsRead = () => {
    setNotifications(prev =>
      prev.map(notif => ({ ...notif, is_read: true }))
    )
  }

  const getIcon = (type: string) => {
    switch (type) {
      case 'new_recommendation':
        return <Sparkles className="h-5 w-5 text-primary" />
      case 'price_drop':
        return <TrendingDown className="h-5 w-5 text-green-600" />
      case 'back_in_stock':
        return <Package className="h-5 w-5 text-blue-600" />
      case 'agent_complete':
        return <Bell className="h-5 w-5 text-purple-600" />
      default:
        return <Bell className="h-5 w-5" />
    }
  }

  const unreadCount = notifications.filter(n => !n.is_read).length

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
                <div className="px-4 lg:px-6">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-2xl font-semibold">Notifications</h2>
                      <p className="text-sm text-muted-foreground">
                        {unreadCount > 0
                          ? `You have ${unreadCount} unread notification${unreadCount > 1 ? 's' : ''}`
                          : 'All caught up!'}
                      </p>
                    </div>
                    {unreadCount > 0 && (
                      <Button variant="outline" size="sm" onClick={markAllAsRead}>
                        Mark all as read
                      </Button>
                    )}
                  </div>

                  <div className="space-y-2">
                    {notifications.length === 0 ? (
                      <div className="text-center py-12 text-muted-foreground">
                        <Bell className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>No notifications yet</p>
                        <p className="text-sm">We'll notify you about new recommendations and updates</p>
                      </div>
                    ) : (
                      notifications.map((notification) => (
                        <div
                          key={notification.id}
                          className={`flex items-start gap-4 p-4 rounded-lg border transition-colors ${
                            !notification.is_read
                              ? 'bg-primary/5 border-primary/20'
                              : 'bg-background hover:bg-muted/50'
                          }`}
                        >
                          <div className="flex-shrink-0 mt-1">
                            {getIcon(notification.type)}
                          </div>

                          {notification.image_url && (
                            <div className="flex-shrink-0">
                              <img
                                src={notification.image_url}
                                alt=""
                                className="w-16 h-16 rounded object-cover"
                              />
                            </div>
                          )}

                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between gap-2">
                              <div>
                                <h3 className="font-medium text-sm">{notification.title}</h3>
                                <p className="text-sm text-muted-foreground mt-0.5">
                                  {notification.message}
                                </p>
                                <p className="text-xs text-muted-foreground mt-2">
                                  {formatDistanceToNow(new Date(notification.created_at), { addSuffix: true })}
                                </p>
                              </div>
                              <div className="flex items-center gap-2">
                                {!notification.is_read && (
                                  <Badge variant="default" className="shrink-0">New</Badge>
                                )}
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8"
                                  onClick={() => deleteNotification(notification.id)}
                                >
                                  <X className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                            {!notification.is_read && (
                              <Button
                                variant="link"
                                size="sm"
                                className="h-auto p-0 mt-2 text-xs"
                                onClick={() => markAsRead(notification.id)}
                              >
                                Mark as read
                              </Button>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>
    </ChatLayout>
  )
}
