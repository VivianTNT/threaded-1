'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { useRequireAuth } from '@/lib/auth-utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Upload, Camera, Sparkles, ArrowRight, Check } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

export default function OnboardingPage() {
  const { user, isLoading } = useRequireAuth()
  const router = useRouter()
  const [uploadedImages, setUploadedImages] = React.useState<string[]>([])
  const [isDragging, setIsDragging] = React.useState(false)

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

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files) {
      const newImages = Array.from(files).map(file => URL.createObjectURL(file))
      setUploadedImages(prev => [...prev, ...newImages])
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files) {
      const newImages = Array.from(files).map(file => URL.createObjectURL(file))
      setUploadedImages(prev => [...prev, ...newImages])
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const removeImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index))
  }

  const handleSkip = () => {
    router.push('/')
  }

  const handleContinue = () => {
    // Would save wardrobe items to database here
    router.push('/')
  }

  const progress = uploadedImages.length > 0 ? Math.min((uploadedImages.length / 10) * 100, 100) : 0

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="container max-w-5xl mx-auto px-4 py-8 md:py-12">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <img src="/favicon.avif" alt="Threaded" className="h-8 w-8 rounded" />
            <span className="text-xl font-semibold">Threaded</span>
          </div>
          <h1 className="text-3xl md:text-4xl font-bold mb-2">Build Your Digital Wardrobe</h1>
          <p className="text-muted-foreground text-lg">
            Upload photos of your favorite pieces to get personalized recommendations
          </p>
        </div>

        {/* Progress */}
        {uploadedImages.length > 0 && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">
                {uploadedImages.length} item{uploadedImages.length !== 1 ? 's' : ''} uploaded
              </span>
              <Badge variant="outline">
                {progress.toFixed(0)}% Complete
              </Badge>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        )}

        {/* Upload Area */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Wardrobe Items
            </CardTitle>
            <CardDescription>
              Add photos of clothing, shoes, and accessories you already own
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-muted-foreground/25 hover:border-primary/50'
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="rounded-full bg-primary/10 p-4">
                  <Camera className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Drag and drop images here</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    or click to browse from your device
                  </p>
                </div>
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Button type="button" variant="outline">
                    <Upload className="h-4 w-4 mr-2" />
                    Choose Files
                  </Button>
                  <input
                    id="file-upload"
                    type="file"
                    multiple
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                </label>
                <p className="text-xs text-muted-foreground">
                  Supports: JPG, PNG, HEIC • Max 10MB per image
                </p>
              </div>
            </div>

            {/* Uploaded Images Grid */}
            {uploadedImages.length > 0 && (
              <div className="mt-6">
                <h4 className="font-medium mb-3">Your Wardrobe</h4>
                <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
                  {uploadedImages.map((image, index) => (
                    <div key={index} className="relative group aspect-square">
                      <img
                        src={image}
                        alt={`Wardrobe item ${index + 1}`}
                        className="w-full h-full object-cover rounded-lg"
                      />
                      <button
                        onClick={() => removeImage(index)}
                        className="absolute top-1 right-1 bg-black/70 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <svg
                          className="h-4 w-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                          />
                        </svg>
                      </button>
                      <div className="absolute bottom-1 right-1 bg-primary/90 text-white rounded-full p-1">
                        <Check className="h-3 w-3" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Why This Helps */}
        <Card className="mb-6 bg-gradient-to-r from-primary/5 to-primary/10 border-primary/20">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <div className="rounded-full bg-primary/20 p-2 flex-shrink-0">
                <Sparkles className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h4 className="font-semibold mb-1">Why upload your wardrobe?</h4>
                <p className="text-sm text-muted-foreground">
                  Our AI analyzes your existing pieces to understand your style preferences, favorite colors,
                  and silhouettes. This helps us recommend items that complement what you already own and
                  suggest outfits you'll love.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Action Buttons */}
        <div className="flex items-center justify-between gap-4">
          <Button
            variant="ghost"
            onClick={handleSkip}
            className="text-muted-foreground"
          >
            Skip for now
          </Button>
          <Button
            onClick={handleContinue}
            disabled={uploadedImages.length === 0}
            className="bg-gradient-to-r from-primary to-primary/80"
          >
            {uploadedImages.length > 0 ? 'Continue' : 'Upload items to continue'}
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </div>

        {/* Tips */}
        <div className="mt-8 text-center">
          <p className="text-sm text-muted-foreground mb-2">
            Tips for best results:
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            <Badge variant="secondary" className="text-xs">Clear, well-lit photos</Badge>
            <Badge variant="secondary" className="text-xs">Single item per image</Badge>
            <Badge variant="secondary" className="text-xs">Different angles welcome</Badge>
          </div>
        </div>
      </div>
    </div>
  )
}
