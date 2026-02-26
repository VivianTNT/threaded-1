"use client"

import * as React from "react"
import { Moon, Sun, Save, Settings } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { toast } from "sonner"

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const [darkMode, setDarkMode] = React.useState(false)
  const [saveChats, setSaveChats] = React.useState(true)

  // Load settings from localStorage on mount
  React.useEffect(() => {
    const storedDarkMode = localStorage.getItem('threaded-dark-mode')
    const storedSaveChats = localStorage.getItem('threaded-save-chats')

    if (storedDarkMode !== null) {
      setDarkMode(storedDarkMode === 'true')
    }
    if (storedSaveChats !== null) {
      setSaveChats(storedSaveChats === 'true')
    }
  }, [])

  const handleDarkModeChange = (checked: boolean) => {
    setDarkMode(checked)
    localStorage.setItem('threaded-dark-mode', String(checked))

    // Apply dark mode class to document
    if (checked) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }

    toast.success(`Dark mode ${checked ? 'enabled' : 'disabled'}`)
  }

  const handleSaveChatsChange = (checked: boolean) => {
    setSaveChats(checked)
    localStorage.setItem('threaded-save-chats', String(checked))

    toast.success(`Chat history ${checked ? 'will be saved' : 'will not be saved'}`)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Settings
          </DialogTitle>
          <DialogDescription>
            Manage your Threaded preferences and account settings
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Appearance Settings */}
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium mb-3">Appearance</h3>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="dark-mode" className="text-sm font-medium cursor-pointer">
                    Dark Mode
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Toggle dark mode theme
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {darkMode ? (
                    <Moon className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <Sun className="h-4 w-4 text-muted-foreground" />
                  )}
                  <Switch
                    id="dark-mode"
                    checked={darkMode}
                    onCheckedChange={handleDarkModeChange}
                  />
                </div>
              </div>
            </div>
          </div>

          <Separator />

          {/* Chat Settings */}
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium mb-3">Chat</h3>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="save-chats" className="text-sm font-medium cursor-pointer">
                    Save Chat History
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Automatically save your chat conversations
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Save className="h-4 w-4 text-muted-foreground" />
                  <Switch
                    id="save-chats"
                    checked={saveChats}
                    onCheckedChange={handleSaveChatsChange}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
