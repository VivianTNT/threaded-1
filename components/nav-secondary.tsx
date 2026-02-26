"use client"

import * as React from "react"
import { type Icon } from "@tabler/icons-react"
import { toast } from 'sonner'

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar'
import { SettingsDialog } from '@/components/settings-dialog'
import { HelpDialog } from '@/components/help-dialog'
import { useChat } from '@/lib/chat-context'
import { useGlobalChat } from '@/lib/global-chat-context'

export function NavSecondary({
  items,
  ...props
}: {
  items: {
    title: string
    url: string
    icon: Icon
  }[]
} & React.ComponentPropsWithoutRef<typeof SidebarGroup>) {
  const [settingsOpen, setSettingsOpen] = React.useState(false)
  const [helpOpen, setHelpOpen] = React.useState(false)
  const { toggleChat } = useChat()
  const { setInput } = useGlobalChat()

  // Add keyboard shortcut for Cmd/Ctrl+K
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setInput("")
        toggleChat()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [toggleChat, setInput])

  const handleClick = (title: string, e: React.MouseEvent) => {
    e.preventDefault()

    switch (title) {
      case "Search":
        // Open Threaded AI stylist
        setInput("") // Clear any previous input
        toggleChat() // Open the chat sidebar
        toast.success('Threaded AI Stylist opened', {
          description: 'Ask me anything about fashion, styling, or products'
        })
        break

      case "Settings":
        setSettingsOpen(true)
        break

      case "Get Help":
        setHelpOpen(true)
        break

      default:
        break
    }
  }

  return (
    <>
      <SidebarGroup {...props}>
        <SidebarGroupContent>
          <SidebarMenu>
            {items.map((item) => (
              <SidebarMenuItem key={item.title}>
                <SidebarMenuButton asChild>
                  <a
                    href={item.url}
                    onClick={(e) => handleClick(item.title, e)}
                    className="cursor-pointer"
                    {...(item.title === "Search" && { 'data-search-trigger': 'true' })}
                  >
                    <item.icon />
                    <span>{item.title}</span>
                  </a>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>

      {/* Dialogs */}
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
      <HelpDialog open={helpOpen} onOpenChange={setHelpOpen} />
    </>
  )
}
