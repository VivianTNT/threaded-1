# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Run linting
npm run lint
```

## Architecture Overview

### Tech Stack
- **Framework**: Next.js 15.2.4 with App Router
- **Language**: TypeScript (strict mode enabled)
- **UI Components**: Radix UI + shadcn/ui components
- **Styling**: Tailwind CSS
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth
- **AI Integration**: OpenAI API
- **Fonts**: Geist Sans and Geist Mono
- **Charts**: Recharts
- **Tables**: TanStack Table

### Project Structure

```
threaded/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   │   ├── auth/         # Authentication endpoints
│   │   └── chat/         # Chat AI endpoints
│   ├── notifications/     # Notifications page
│   ├── login/            # Login page
│   └── signup/           # Signup page
├── components/            # React components
│   ├── fashion-recommendations/ # Fashion product grid and detail views
│   ├── ui/              # Reusable UI components
│   └── fashion-agent-button.tsx # Fashion agent UI component
├── lib/                   # Core utilities
│   ├── types/            # TypeScript definitions
│   │   └── fashion-product.ts # Fashion product types
│   ├── data/             # Mock data
│   │   └── mock-fashion-products.ts # Fashion product data
│   ├── auth-context.tsx  # Authentication provider
│   ├── chat-context.tsx  # Chat UI state
│   └── global-chat-context.tsx # Global chat functionality
└── public/               # Static assets
```

### Key Architectural Patterns

#### Context Providers
The app uses nested React Context providers in `app/layout.tsx`:
1. `AuthProvider` - Manages authentication state
2. `ChatProvider` - Controls chat UI state
3. `GlobalChatProvider` - Handles AI chat functionality

#### Database Access
- Uses Supabase client from `lib/supabase.ts`
- Service role client available in `lib/supabase-service.ts` for admin operations
- Tables include: users, fashion_products, notifications, wardrobe_items, chat_history

#### Fashion Recommendation System
The app provides personalized fashion recommendations:
- **Product Grid**: Displays curated fashion items with images, prices, and recommendations
- **Detail Panel**: Shows comprehensive product information, similar items, and purchase options
- **Fashion Agent**: Simulated AI that "discovers" new items matching user preferences
- **Notifications**: Real-time alerts for new recommendations, price drops, and restocks

#### API Route Pattern
All API routes follow consistent patterns:
- Located in `app/api/*/route.ts`
- Use standard HTTP methods (GET, POST)
- Return consistent JSON responses
- Handle errors gracefully

#### Component Conventions
- All client components must include `"use client"` directive
- Use TypeScript interfaces for component props
- Leverage shadcn/ui components from `components/ui/`
- Follow existing patterns for new components

### Environment Variables
Required environment variables:
```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
OPENAI_API_KEY=
```

### Important Notes

#### Build Configuration
The Next.js config (`next.config.mjs`) has:
- ESLint errors ignored during builds
- TypeScript errors ignored during builds
- Images unoptimized

This suggests focusing on functionality over strict linting during development.

#### Path Aliases
The project uses `@/*` as a path alias for the root directory, configured in `tsconfig.json`.

#### Styling
- Use Tailwind CSS classes exclusively
- Utilize the `cn()` utility from `lib/utils.ts` for conditional classes
- Theme customization via CSS variables in `app/globals.css`
- Minimalist, classy design aesthetic inspired by premium fashion brands

#### Authentication Flow
1. Users authenticate via `/login` or `/signup`
2. Protected routes redirect to login if unauthenticated
3. Auth state managed globally via `AuthProvider`

#### Fashion Product Data
The app focuses on personalized fashion recommendations with:
- Product grid with images, prices, and brand information
- Detailed product views with materials, sizes, colors
- Similar item recommendations
- AI-powered styling suggestions
- Mock data from real luxury and contemporary fashion brands

#### User Workflow
1. **Sign up** - Create account
2. **Upload wardrobe** (optional) - Add photos of existing clothes
3. **Browse recommendations** - View AI-curated fashion items
4. **Explore products** - Click to see details, similar items
5. **Take action** - Add to cart, save, view on brand website
6. **Run fashion agent** - Discover new items matching preferences
7. **Get notifications** - Alerts for new recommendations and updates

#### Mock Data
The app currently uses synthetic mock data for demonstration:
- Fashion products with real brand names (Loro Piana, Max Mara, The Row, Toteme, COS, etc.)
- Product images from Unsplash
- Realistic prices and product descriptions
- Mock notifications and agent runs
