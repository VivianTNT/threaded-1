# Threaded

**AI-Powered Personal Fashion Stylist**

Threaded is a B2C web application that provides personalized fashion recommendations tailored to your unique style. Using AI technology, we curate clothing items from premium brands that match your preferences and existing wardrobe.

## Features

### Personalized Recommendations
- AI-curated fashion items from luxury and contemporary brands
- Smart matching based on your style preferences
- Daily updates with new discoveries

### Wardrobe Integration
- Upload photos of your existing clothes (optional)
- Get recommendations that complement what you already own
- Build a digital closet for better styling suggestions

### Fashion Agent
- Automated discovery of new items matching your style
- Real-time scanning of premium retailers
- Progress tracking and notifications

### Smart Notifications
- Alerts for new recommendations
- Price drop notifications
- Back-in-stock alerts for saved items

### AI Stylist Chat
- Ask questions about any product
- Get styling advice and outfit suggestions
- Receive personalized fashion guidance

### Seamless Shopping
- View detailed product information
- Find similar items instantly
- One-click access to brand websites
- Add items to cart for easy tracking

## Tech Stack

- **Framework**: Next.js 15.2.4 with App Router
- **Language**: TypeScript
- **UI**: Radix UI + shadcn/ui components
- **Styling**: Tailwind CSS
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth
- **AI**: OpenAI API

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Supabase account

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd threaded
```

2. Install dependencies
```bash
npm install
```

3. Set up environment variables
```bash
cp .env.example .env.local
```

Add your environment variables:
```
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
```

4. Run the development server
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## User Workflow

1. **Sign Up** - Create your account to get started
2. **Upload Wardrobe** - Optionally add photos of your existing clothes
3. **Browse Recommendations** - Explore AI-curated fashion items
4. **View Details** - Click any item to see full details and similar products
5. **Take Action** - Save favorites, add to cart, or view on brand website
6. **Run Fashion Agent** - Let AI discover new items for you
7. **Stay Updated** - Receive notifications about new finds and updates

## Project Structure

```
threaded/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── notifications/     # Notifications page
│   ├── login/            # Authentication
│   └── signup/
├── components/            # React components
│   ├── fashion-recommendations/
│   │   ├── fashion-grid.tsx
│   │   └── product-detail-panel.tsx
│   ├── fashion-agent-button.tsx
│   └── ui/               # Reusable UI components
├── lib/                   # Core utilities
│   ├── types/            # TypeScript definitions
│   └── data/             # Mock data
└── public/               # Static assets
```

## Design Philosophy

Threaded embraces a **minimalist, classy aesthetic** inspired by premium fashion brands:

- Clean, spacious layouts
- High-quality product imagery
- Subtle animations and transitions
- Intuitive navigation
- Sophisticated color palette
- Typography-focused design

## Current Status

This is a frontend demonstration with mock data. Key features implemented:

Fashion recommendation grid
Product detail panels
Similar item discovery
Fashion agent UI (simulated)
Notification system
AI stylist chat integration
Responsive design
Authentication flow

## Roadmap

Future enhancements could include:

- Real fashion API integration
- Actual wardrobe upload and analysis
- Advanced filtering and search
- Outfit builder
- Social sharing features
- Wishlist and collections
- Purchase history tracking
- Sustainability scoring

## Contributing

This project is currently in development. Contributions, issues, and feature requests are welcome.

## License

Private and confidential.
