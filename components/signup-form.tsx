"use client"

import { useState } from "react"
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
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useAuth } from "@/lib/auth-context"
import { useRedirectIfAuthenticated } from "@/lib/auth-utils"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { CheckCircle, AlertCircle, Info } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"

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

  // Form data state
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    stylePreferences: [] as string[],
    budgetRange: "",
    favoriteColors: "",
    bio: ""
  })

  // Redirect if already authenticated
  useRedirectIfAuthenticated()

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { id, value } = e.target
    setFormData(prev => ({ ...prev, [id]: value }))
  }

  const handleSelectChange = (value: string) => {
    setFormData(prev => ({ ...prev, budgetRange: value }))
  }

  const toggleStylePreference = (style: string) => {
    setFormData(prev => ({
      ...prev,
      stylePreferences: prev.stylePreferences.includes(style)
        ? prev.stylePreferences.filter(s => s !== style)
        : [...prev.stylePreferences, style]
    }))
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
          stylePreferences: formData.stylePreferences,
          budgetRange: formData.budgetRange,
          favoriteColors: formData.favoriteColors,
          bio: formData.bio
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

  const styleOptions = [
    "Minimalist",
    "Classic",
    "Casual",
    "Business Casual",
    "Athleisure",
    "Bohemian",
    "Streetwear",
    "Elegant"
  ]

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
              <TabsTrigger value="preferences" disabled={step === "account"}>Style Preferences</TabsTrigger>
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
                  <Button type="submit" className="w-full">
                    Continue
                  </Button>
                </div>
              </form>
            </TabsContent>
            <TabsContent value="preferences">
              <form onSubmit={handleSubmit}>
                <div className="grid gap-6">
                  <div className="grid gap-3">
                    <Label>Style Preferences (select all that apply)</Label>
                    <div className="grid grid-cols-2 gap-3">
                      {styleOptions.map((style) => (
                        <div key={style} className="flex items-center space-x-2">
                          <Checkbox
                            id={style}
                            checked={formData.stylePreferences.includes(style)}
                            onCheckedChange={() => toggleStylePreference(style)}
                          />
                          <label
                            htmlFor={style}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                          >
                            {style}
                          </label>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="budgetRange">Budget Range (optional)</Label>
                    <Select value={formData.budgetRange} onValueChange={handleSelectChange}>
                      <SelectTrigger id="budgetRange">
                        <SelectValue placeholder="Select budget range" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="budget">Budget Friendly ($0-$50)</SelectItem>
                        <SelectItem value="midrange">Mid-Range ($50-$200)</SelectItem>
                        <SelectItem value="premium">Premium ($200-$500)</SelectItem>
                        <SelectItem value="luxury">Luxury ($500+)</SelectItem>
                        <SelectItem value="mixed">Mixed Range</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="favoriteColors">Favorite Colors (optional)</Label>
                    <Input
                      id="favoriteColors"
                      placeholder="e.g., Black, Navy, Beige, Emerald"
                      value={formData.favoriteColors}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="bio">About You (optional)</Label>
                    <Textarea
                      id="bio"
                      placeholder="Tell us about your style goals, what occasions you dress for, or anything that helps us understand your fashion needs..."
                      className="min-h-24"
                      value={formData.bio}
                      onChange={handleChange}
                    />
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
