"use client"

import { useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import { useRedirectIfAuthenticated } from "@/lib/auth-utils"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { CheckCircle, AlertCircle, Info, Upload, Camera, X } from "lucide-react"

const SIGNUP_SAMPLE_SIZE = 16
const SIGNUP_MIN_LIKES = 3
const SIGNUP_UPLOAD_LIMIT = 5
const SIGNUP_UPLOAD_MAX_BYTES = 10 * 1024 * 1024

type OnboardingProductCard = {
  id: string
  name: string
  brand: string
  image_url: string
  price: number | null
  product_url: string | null
  category: string | null
  description: string | null
  similarity?: number
}

type UploadedClothingPhoto = {
  file: File
  previewUrl: string
}

export function SignupForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const [step, setStep] = useState<"account" | "preferences">("account")
  const router = useRouter()
  const { signUp } = useAuth()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [alert, setAlert] = useState<{
    type: "success" | "error" | "info" | null;
    message: string;
  }>({ type: null, message: "" })
  const [sampleProducts, setSampleProducts] = useState<OnboardingProductCard[]>([])
  const [likedProductIds, setLikedProductIds] = useState<string[]>([])
  const [isLoadingSample, setIsLoadingSample] = useState(false)
  const [uploadedClothingPhotos, setUploadedClothingPhotos] = useState<UploadedClothingPhoto[]>([])
  const uploadedClothingPhotosRef = useRef<UploadedClothingPhoto[]>([])

  // Form data state
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  })

  // Optional checkout details
  const [checkoutDetails, setCheckoutDetails] = useState({
    firstName: "",
    lastName: "",
    phone: "",
    shippingAddress: "",
    shippingCity: "",
    shippingState: "",
    shippingZip: "",
    shippingCountry: "US",
  })
  const [showCheckoutFields, setShowCheckoutFields] = useState(false)

  const handleCheckoutChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { id, value } = e.target
    setCheckoutDetails(prev => ({ ...prev, [id]: value }))
  }

  // Redirect if already authenticated
  useRedirectIfAuthenticated()

  useEffect(() => {
    if (step !== "preferences" || sampleProducts.length > 0) return
    void loadSignupSample()
  }, [step, sampleProducts.length])

  useEffect(() => {
    uploadedClothingPhotosRef.current = uploadedClothingPhotos
  }, [uploadedClothingPhotos])

  useEffect(() => {
    return () => {
      uploadedClothingPhotosRef.current.forEach((photo) => URL.revokeObjectURL(photo.previewUrl))
    }
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target
    setFormData(prev => ({ ...prev, [id]: value }))
  }

  const handleNextStep = (e: React.FormEvent) => {
    e.preventDefault()
    setAlert({ type: null, message: "" })

    // Validate account information
    if (!formData.name || !formData.email || !formData.password) {
      setAlert({
        type: "error",
        message: "Please fill in all required fields"
      })
      return
    }

    // Password validation
    if (formData.password.length < 8) {
      setAlert({
        type: "error",
        message: "Password must be at least 8 characters long"
      })
      return
    }

    setStep("preferences")
  }

  const loadSignupSample = async () => {
    setIsLoadingSample(true)
    try {
      const response = await fetch(`/api/signup/image-recommendations?sampleSize=${SIGNUP_SAMPLE_SIZE}`)
      const data = await response.json()
      if (!response.ok || !data.success) {
        throw new Error(data.message || "Failed to load products for signup sample")
      }
      setSampleProducts(Array.isArray(data.products) ? data.products : [])
      setLikedProductIds([])
    } catch (error: any) {
      setAlert({
        type: "error",
        message: error?.message || "Could not load signup sample products"
      })
    } finally {
      setIsLoadingSample(false)
    }
  }

  const toggleLikedProduct = (productId: string) => {
    setLikedProductIds((prev) => {
      return prev.includes(productId)
        ? prev.filter((id) => id !== productId)
        : [...prev, productId]
    })
  }

  const addUploadedClothingPhotos = (files: FileList | null) => {
    if (!files) return

    const remainingSlots = SIGNUP_UPLOAD_LIMIT - uploadedClothingPhotos.length
    if (remainingSlots <= 0) {
      setAlert({
        type: "info",
        message: `You can upload up to ${SIGNUP_UPLOAD_LIMIT} clothing photos.`
      })
      return
    }

    const nextPhotos: UploadedClothingPhoto[] = []
    for (const file of Array.from(files).slice(0, remainingSlots)) {
      if (!file.type.startsWith("image/")) {
        setAlert({
          type: "error",
          message: `${file.name} is not a supported image file.`
        })
        continue
      }

      if (file.size > SIGNUP_UPLOAD_MAX_BYTES) {
        setAlert({
          type: "error",
          message: `${file.name} is larger than 10MB.`
        })
        continue
      }

      nextPhotos.push({
        file,
        previewUrl: URL.createObjectURL(file),
      })
    }

    if (nextPhotos.length > 0) {
      setUploadedClothingPhotos((prev) => [...prev, ...nextPhotos])
    }
  }

  const removeUploadedClothingPhoto = (index: number) => {
    setUploadedClothingPhotos((prev) => {
      const photo = prev[index]
      if (photo) {
        URL.revokeObjectURL(photo.previewUrl)
      }
      return prev.filter((_, photoIndex) => photoIndex !== index)
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setAlert({ type: null, message: "" })

    setIsSubmitting(true)

    try {
      const { error, success, message } = await signUp(
        formData.email,
        formData.password,
        {
          name: formData.name,
          likedProductIds,
          shownProductIds: sampleProducts.map((p) => p.id),
          uploadedClothingPhotos: uploadedClothingPhotos.map((photo) => photo.file),
          checkoutDetails: showCheckoutFields ? checkoutDetails : undefined,
        }
      )

      if (success) {
        setAlert({
          type: "success",
          message: message || "Account created! Redirecting to onboarding..."
        })
        setTimeout(() => {
          router.push("/onboarding")
        }, 2000)
      } else {
        if (message === 'Email already taken') {
          setAlert({
            type: "error",
            message: "Email already taken"
          })
        } else if (message?.includes('verification')) {
          setAlert({
            type: "info",
            message: "Please verify your email address before logging in"
          })
        } else {
          setAlert({
            type: "error",
            message: message || "Failed to create account"
          })
        }
      }
    } catch (error) {
      setAlert({
        type: "error",
        message: "An unexpected error occurred"
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card>
        <CardHeader className="text-center">
          <CardTitle className="text-xl">Create your account</CardTitle>
          <CardDescription>
            Join Threaded to discover your perfect style
          </CardDescription>
        </CardHeader>
        <CardContent>
          {alert.type && (
            <Alert
              variant={alert.type === "success" ? "default" : alert.type === "error" ? "destructive" : "default"}
              className="mb-6"
            >
              {alert.type === "success" && <CheckCircle className="h-4 w-4" />}
              {alert.type === "error" && <AlertCircle className="h-4 w-4" />}
              {alert.type === "info" && <Info className="h-4 w-4" />}
              <AlertDescription>{alert.message}</AlertDescription>
            </Alert>
          )}
          <Tabs value={step} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="account" disabled={step === "preferences"}>Account</TabsTrigger>
              <TabsTrigger value="preferences" disabled={step === "account"}>Pick Products</TabsTrigger>
            </TabsList>
            <TabsContent value="account">
              <form onSubmit={handleNextStep}>
                <div className="grid gap-6">
                  <div className="grid gap-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      placeholder="Your Name"
                      required
                      value={formData.name}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      required
                      value={formData.email}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="password">Password</Label>
                    <Input
                      id="password"
                      type="password"
                      required
                      value={formData.password}
                      onChange={handleChange}
                    />
                    <p className="text-xs text-muted-foreground">
                      Password must be at least 8 characters long
                    </p>
                  </div>
                  {/* Optional: Checkout / Shipping Details */}
                  <div className="grid gap-3 rounded-lg border border-dashed p-4">
                    <button
                      type="button"
                      onClick={() => setShowCheckoutFields(!showCheckoutFields)}
                      className="flex items-center gap-2 text-left w-full"
                    >
                      <div className="flex-1">
                        <p className="text-sm font-medium">Shipping & billing details (optional)</p>
                        <p className="text-xs text-muted-foreground">
                          Add now for faster checkout — our agent will auto-fill retailer forms for you.
                        </p>
                      </div>
                      <svg className={`h-4 w-4 text-muted-foreground transition-transform ${showCheckoutFields ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    {showCheckoutFields && (
                      <div className="grid gap-3 pt-2">
                        <div className="grid grid-cols-2 gap-3">
                          <div className="grid gap-1">
                            <Label htmlFor="firstName" className="text-xs">First Name</Label>
                            <Input id="firstName" placeholder="John" value={checkoutDetails.firstName} onChange={handleCheckoutChange} className="h-9 text-sm" />
                          </div>
                          <div className="grid gap-1">
                            <Label htmlFor="lastName" className="text-xs">Last Name</Label>
                            <Input id="lastName" placeholder="Doe" value={checkoutDetails.lastName} onChange={handleCheckoutChange} className="h-9 text-sm" />
                          </div>
                        </div>
                        <div className="grid gap-1">
                          <Label htmlFor="phone" className="text-xs">Phone</Label>
                          <Input id="phone" type="tel" placeholder="(555) 123-4567" value={checkoutDetails.phone} onChange={handleCheckoutChange} className="h-9 text-sm" />
                        </div>
                        <div className="grid gap-1">
                          <Label htmlFor="shippingAddress" className="text-xs">Address</Label>
                          <Input id="shippingAddress" placeholder="123 Main St, Apt 4B" value={checkoutDetails.shippingAddress} onChange={handleCheckoutChange} className="h-9 text-sm" />
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                          <div className="grid gap-1">
                            <Label htmlFor="shippingCity" className="text-xs">City</Label>
                            <Input id="shippingCity" placeholder="New York" value={checkoutDetails.shippingCity} onChange={handleCheckoutChange} className="h-9 text-sm" />
                          </div>
                          <div className="grid gap-1">
                            <Label htmlFor="shippingState" className="text-xs">State</Label>
                            <Input id="shippingState" placeholder="NY" value={checkoutDetails.shippingState} onChange={handleCheckoutChange} className="h-9 text-sm" />
                          </div>
                          <div className="grid gap-1">
                            <Label htmlFor="shippingZip" className="text-xs">ZIP</Label>
                            <Input id="shippingZip" placeholder="10001" value={checkoutDetails.shippingZip} onChange={handleCheckoutChange} className="h-9 text-sm" />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  <Button type="submit" className="w-full">
                    Continue
                  </Button>
                  <div className="grid gap-3 rounded-lg border border-dashed p-4">
                    <div className="flex items-center gap-2">
                      <Camera className="h-4 w-4 text-primary" />
                      <div>
                        <p className="text-sm font-medium">Upload clothing photos (optional)</p>
                        <p className="text-xs text-muted-foreground">
                          Add up to {SIGNUP_UPLOAD_LIMIT} photos of pieces you own so we can better understand your style.
                        </p>
                      </div>
                    </div>
                    <label
                      htmlFor="signup-clothing-photos"
                      className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-border bg-muted/30 px-4 py-6 text-center transition hover:border-primary/50"
                    >
                      <Upload className="mb-2 h-5 w-5 text-primary" />
                      <span className="text-sm font-medium">Choose photos</span>
                      <span className="text-xs text-muted-foreground">
                        JPG, PNG, HEIC up to 10MB each
                      </span>
                      <input
                        id="signup-clothing-photos"
                        type="file"
                        accept="image/*"
                        multiple
                        className="hidden"
                        onChange={(e) => {
                          addUploadedClothingPhotos(e.target.files)
                          e.currentTarget.value = ""
                        }}
                      />
                    </label>
                    {uploadedClothingPhotos.length > 0 && (
                      <div className="grid grid-cols-3 gap-3 md:grid-cols-5">
                        {uploadedClothingPhotos.map((photo, index) => (
                          <div key={`${photo.file.name}-${index}`} className="group relative aspect-square overflow-hidden rounded-md border">
                            <img
                              src={photo.previewUrl}
                              alt={photo.file.name}
                              className="h-full w-full object-cover"
                            />
                            <button
                              type="button"
                              onClick={() => removeUploadedClothingPhoto(index)}
                              className="absolute right-1 top-1 rounded-full bg-black/70 p-1 text-white opacity-0 transition group-hover:opacity-100"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </form>
            </TabsContent>
            <TabsContent value="preferences">
              <form onSubmit={handleSubmit}>
                <div className="grid gap-6">
                  <div className="grid gap-2">
                    <div className="flex items-center justify-between gap-3">
                      <Label>Pick products you like (image-based)</Label>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => loadSignupSample()}
                        disabled={isLoadingSample}
                      >
                        {isLoadingSample ? "Loading..." : "Refresh Sample"}
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Select at least {SIGNUP_MIN_LIKES}. We use these likes to build your personalized recommendation profile for the home page.
                    </p>

                    {isLoadingSample ? (
                      <div className="text-sm text-muted-foreground py-4">Loading product images...</div>
                    ) : sampleProducts.length === 0 ? (
                      <div className="text-sm text-muted-foreground py-4">No sample products found.</div>
                    ) : (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {sampleProducts.map((product) => {
                          const isLiked = likedProductIds.includes(product.id)
                          return (
                            <button
                              key={product.id}
                              type="button"
                              onClick={() => toggleLikedProduct(product.id)}
                              className={cn(
                                "text-left rounded-lg border overflow-hidden transition",
                                isLiked ? "border-primary ring-2 ring-primary/20" : "border-border hover:border-primary/50"
                              )}
                            >
                              <div className="aspect-[3/4] bg-muted">
                                {product.image_url ? (
                                  <img
                                    src={product.image_url}
                                    alt={product.name}
                                    className="h-full w-full object-cover"
                                  />
                                ) : (
                                  <div className="h-full w-full flex items-center justify-center text-xs text-muted-foreground">
                                    No image
                                  </div>
                                )}
                              </div>
                              <div className="p-2">
                                <p className="text-xs font-medium line-clamp-2">{product.name}</p>
                                <p className="text-xs text-muted-foreground">{product.brand}</p>
                              </div>
                            </button>
                          )
                        })}
                      </div>
                    )}
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-muted-foreground">
                        Selected: {likedProductIds.length}
                      </p>
                      {likedProductIds.length < SIGNUP_MIN_LIKES ? (
                        <p className="text-xs text-muted-foreground">
                          Pick {SIGNUP_MIN_LIKES - likedProductIds.length} more for better personalization
                        </p>
                      ) : (
                        <p className="text-xs text-muted-foreground">Great, you are ready to continue</p>
                      )}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setStep("account")}
                    >
                      Back
                    </Button>
                    <Button type="submit" disabled={isSubmitting}>
                      {isSubmitting ? "Creating Account..." : "Complete Setup"}
                    </Button>
                  </div>
                </div>
              </form>
            </TabsContent>
          </Tabs>
        </CardContent>
        <CardFooter className="flex justify-center border-t px-6 py-4">
          <div className="text-center text-sm">
            Already have an account?{" "}
            <Link href="/login" className="underline underline-offset-4">
              Log in
            </Link>
          </div>
        </CardFooter>
      </Card>
      <div className="text-balance text-center text-xs text-muted-foreground [&_a]:underline [&_a]:underline-offset-4 [&_a]:hover:text-primary">
        By clicking continue, you agree to our <a href="#">Terms of Service</a>{" "}
        and <a href="#">Privacy Policy</a>.
      </div>
    </div>
  )
}
