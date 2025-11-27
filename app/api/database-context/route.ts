import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export async function GET(request: NextRequest) {
  console.log("Database context API called");

  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl || !supabaseServiceKey) {
      console.error('Supabase configuration missing');
      return NextResponse.json({ error: 'Configuration error' }, { status: 500 });
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Fetch ALL products from Penn database
    const { data: products, error: productsError } = await supabase
      .from('products')
      .select('*')
      .order('created_at', { ascending: false });

    if (productsError) {
      console.error('Error fetching products:', productsError);
      return NextResponse.json({ error: 'Failed to fetch products' }, { status: 500 });
    }

    // Fetch brands (unique from products)
    const brands = new Set(products?.map(p => p.brand_name).filter(Boolean));
    const brandsList = Array.from(brands).map(name => ({ name }));

    // Calculate summary statistics
    const summaryStats = {
      totalProducts: products?.length || 0,
      totalBrands: brands.size,
      avgPrice: products?.filter(p => p.price > 0).reduce((sum, p) => sum + p.price, 0) / products?.filter(p => p.price > 0).length || 0,
      topBrands: getTopItems(products || [], 'brand_name', 10),
      topCategories: getTopItems(products || [], 'category', 10),
      topDomains: getTopItems(products || [], 'domain', 5)
    };

    // Format context for easy consumption
    const formattedContext = {
      products: products || [],
      brands: brandsList,
      stats: summaryStats,
      formattedText: formatForChat(products || [], brandsList, summaryStats)
    };

    console.log(`Returning database context: ${products?.length || 0} products, ${brands.size} brands`);
    return NextResponse.json(formattedContext);

  } catch (error) {
    console.error('Database context error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

function getTopItems(items: any[], field: string, limit: number) {
  const counts: { [key: string]: number } = {};

  items.forEach(item => {
    const value = item[field];
    if (value) {
      counts[value] = (counts[value] || 0) + 1;
    }
  });

  return Object.entries(counts)
    .map(([key, count]) => ({ name: key, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}

function formatForChat(products: any[], brands: any[], stats: any): string {
  let context = `═══════════════════════════════════════════════════════════════
COMPREHENSIVE DATABASE CONTEXT - FASHION PRODUCT CATALOG
═══════════════════════════════════════════════════════════════

📊 SUMMARY STATISTICS:
────────────────────────────────────────────────────────────────
• Total Fashion Products: ${stats.totalProducts}
• Total Brands: ${stats.totalBrands}
• Average Price: $${stats.avgPrice?.toFixed(2)}

👔 TOP BRANDS (by product count):
────────────────────────────────────────────────────────────────
${stats.topBrands && stats.topBrands.length > 0 ? stats.topBrands.map((b: any) => `   • ${b.name}: ${b.count} products`).join('\n') : '   • No brand data available'}

🏷️  TOP CATEGORIES:
────────────────────────────────────────────────────────────────
${stats.topCategories && stats.topCategories.length > 0 ? stats.topCategories.map((c: any) => `   • ${c.name}: ${c.count} products`).join('\n') : '   • No category data available'}

🌐 TOP RETAILERS:
────────────────────────────────────────────────────────────────
${stats.topDomains && stats.topDomains.length > 0 ? stats.topDomains.map((d: any) => `   • ${d.name}: ${d.count} products`).join('\n') : '   • No retailer data available'}

════════════════════════════════════════════════════════════════
🛍️  FASHION PRODUCTS (First 50 of ${products.length} total)
════════════════════════════════════════════════════════════════
${products.slice(0, 50).map((p: any, i: number) =>
  `${(i+1).toString().padStart(2, '0')}. ${p.name || 'Fashion Item'}
    Brand: ${p.brand_name || p.domain?.replace('.com', '').replace('www.', '') || 'Unknown'}
    Category: ${p.category || 'N/A'} | Price: ${p.price > 0 ? '$' + p.price : 'N/A'}
    ${p.description ? p.description.substring(0, 120) + '...' : 'No description'}`
).join('\n\n')}

════════════════════════════════════════════════════════════════
🔍 COMPLETE PRODUCT CATALOG (All ${products.length} Products)
════════════════════════════════════════════════════════════════
${products.map((p: any) =>
  `• ${p.name || 'N/A'} | Brand: ${p.brand_name || p.domain || 'Unknown'} | Category: ${p.category || 'N/A'} | Price: ${p.price > 0 ? '$' + p.price : 'N/A'} | URL: ${p.product_url || 'N/A'}`
).join('\n')}

═══════════════════════════════════════════════════════════════
END OF DATABASE CONTEXT
═══════════════════════════════════════════════════════════════

INSTRUCTIONS FOR USING THIS DATA:
• You have complete access to ${stats.totalProducts} fashion products from ${stats.totalBrands} brands
• When asked about specific products, brands, or styles, search through the complete catalog above
• Compare products by price, brand, category, or retailer
• Provide personalized fashion recommendations using exact product details from the database
• Reference specific products when making styling suggestions
• Always cite specific product names and brands when making recommendations`;

  return context;
}
