"use client"

import * as React from "react"
import { BookOpen, Mail, Sparkles, Upload, ShoppingBag, MessageSquare, ArrowRight, Bell } from "lucide-react"
import { useRouter } from "next/navigation"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"

interface HelpDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function HelpDialog({ open, onOpenChange }: HelpDialogProps) {
  const router = useRouter()
  const [activeTab, setActiveTab] = React.useState<"guide" | "features">("guide")

  const handleNavigate = (path: string) => {
    onOpenChange(false)
    router.push(path)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Help & Documentation
          </DialogTitle>
          <DialogDescription>
            Quick start guide and feature overview
          </DialogDescription>
        </DialogHeader>

        <div className="flex gap-8 border-b mb-4">
          <button
            onClick={() => setActiveTab("guide")}
            className={`pb-3 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === "guide"
                ? "border-foreground text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            Quick Start
          </button>
          <button
            onClick={() => setActiveTab("features")}
            className={`pb-3 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === "features"
                ? "border-foreground text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            Features
          </button>
        </div>

        <Tabs value={activeTab} className="w-full flex-1 flex flex-col overflow-hidden">
          <TabsList className="hidden">
            <TabsTrigger value="guide">Quick Start</TabsTrigger>
            <TabsTrigger value="features">Features</TabsTrigger>
          </TabsList>

          <TabsContent value="guide" className="flex-1 flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto space-y-3 pr-2">
              <div className="space-y-2">
                <h3 className="text-sm font-semibold">Your Workflow</h3>
                <div className="space-y-2">
                <div className="w-full text-left p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Upload className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium text-sm">1. Upload Your Wardrobe (Optional)</div>
                      <div className="text-xs text-muted-foreground">Add photos of your clothes to get personalized recommendations</div>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => handleNavigate('/')}
                  className="w-full text-left p-3 border rounded-lg hover:bg-muted transition-colors group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Sparkles className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <div className="font-medium text-sm">2. Browse Recommendations</div>
                        <div className="text-xs text-muted-foreground">Discover curated fashion items tailored to your style</div>
                      </div>
                    </div>
                    <ArrowRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </button>

                <div className="w-full text-left p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <ShoppingBag className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium text-sm">3. Explore & Shop</div>
                      <div className="text-xs text-muted-foreground">View details, find similar items, add to cart</div>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => handleNavigate('/notifications')}
                  className="w-full text-left p-3 border rounded-lg hover:bg-muted transition-colors group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Bell className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <div className="font-medium text-sm">4. Stay Updated</div>
                        <div className="text-xs text-muted-foreground">Get notified about new finds and price drops</div>
                      </div>
                    </div>
                    <ArrowRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </button>

                <button
                  onClick={() => {
                    onOpenChange(false)
                    const searchButton = document.querySelector('[data-search-trigger]') as HTMLElement
                    if (searchButton) {
                      searchButton.click()
                    }
                  }}
                  className="w-full text-left p-3 border rounded-lg bg-gradient-to-r from-primary/10 to-primary/5 hover:from-primary/20 hover:to-primary/10 transition-all group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <MessageSquare className="h-5 w-5 text-primary" />
                      <div>
                        <div className="font-medium text-sm">5. Ask AI Stylist</div>
                        <div className="text-xs text-muted-foreground">Press Cmd/Ctrl+K for fashion advice</div>
                      </div>
                    </div>
                    <ArrowRight className="h-4 w-4 text-primary opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </button>
              </div>
            </div>

            <Separator />

              <div className="space-y-2">
                <h3 className="text-sm font-semibold">Key Features</h3>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>• <strong>Fashion Agent:</strong> AI finds new items matching your style preferences</p>
                  <p>• <strong>Similar Items:</strong> Discover alternatives for products you love</p>
                  <p>• <strong>Smart Filters:</strong> Search by brand, style, price, and more</p>
                  <p>• <strong>Save & Like:</strong> Curate your personal wishlist effortlessly</p>
                </div>
              </div>
            </div>

            <div className="border-t mt-3 pt-3 flex items-center justify-between flex-shrink-0">
              <div className="text-sm text-muted-foreground">Need help?</div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.location.href = 'mailto:hello@threaded.app?subject=Help Request'}
              >
                <Mail className="h-4 w-4 mr-2" />
                Contact Support
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="features" className="flex-1 flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto space-y-3 pr-2">
              <div className="flex items-start gap-3 p-3 border rounded-lg">
                <Sparkles className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div className="text-sm space-y-1">
                  <p className="font-medium">AI Fashion Agent</p>
                  <p className="text-muted-foreground text-xs">Our AI agent continuously scans premium retailers to find items that match your style preferences, wardrobe, and budget. Get notified when new recommendations are available.</p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 border rounded-lg">
                <Upload className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div className="text-sm space-y-1">
                  <p className="font-medium">Wardrobe Upload</p>
                  <p className="text-muted-foreground text-xs">Upload photos of your existing clothes to help our AI understand your style better. The more items you add, the more personalized your recommendations become.</p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 border rounded-lg">
                <ShoppingBag className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div className="text-sm space-y-1">
                  <p className="font-medium">Smart Shopping</p>
                  <p className="text-muted-foreground text-xs">Find similar items, compare prices, save favorites, and add to cart. One-click access to purchase directly from brand websites.</p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 border rounded-lg">
                <Bell className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div className="text-sm space-y-1">
                  <p className="font-medium">Real-Time Notifications</p>
                  <p className="text-muted-foreground text-xs">Get instant alerts for new recommendations, price drops, and when items you've saved come back in stock.</p>
                </div>
              </div>

              <Separator />

              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="h-4 w-4" />
                  <p className="text-sm font-medium">AI Stylist Chat</p>
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                  <p>Ask questions about any item, get styling advice, or request specific types of clothing. Our AI stylist is here to help with:</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2">
                    <p>• Outfit combinations</p>
                    <p>• Sizing guidance</p>
                    <p>• Style recommendations</p>
                    <p>• Occasion-based looks</p>
                    <p>• Trend insights</p>
                    <p>• Sustainability info</p>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
