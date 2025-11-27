import { NextRequest, NextResponse } from 'next/server'
import { PDFDocument } from 'pdf-lib'
import OpenAI from 'openai'
import { generateMemoPDF, MemoData } from '@/lib/memo-generator'
import { generateMemoDocx } from '@/lib/memo-generator-docx'

// Check for API key
const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey || apiKey === 'your_openai_api_key_here') {
  console.warn('OpenAI API key not configured');
}

const openai = new OpenAI({
  apiKey: apiKey || 'sk-dummy-key' // Use dummy key to prevent initialization errors
})

// Use unpdf for PDF parsing
async function extractPDFText(buffer: Buffer): Promise<{ text: string; numpages: number; info: any }> {
  try {
    const { extractText } = await import('unpdf');
    const uint8Array = new Uint8Array(buffer);
    const { text, totalPages } = await extractText(uint8Array, { mergePages: true });
    return {
      text,
      numpages: totalPages,
      info: {}
    };
  } catch (error) {
    console.error('PDF parse error:', error);
    throw error;
  }
}


// Helper function to fetch project context - use direct Supabase query instead of fetch
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabaseServer = createClient(supabaseUrl, supabaseServiceKey);

async function getProductContext(): Promise<string> {
  try {
    console.log('[getProductContext] Fetching products directly from Supabase...');

    // Query Supabase directly for fashion products
    const { data: products, error, count } = await supabaseServer
      .from('products')
      .select('*', { count: 'exact' })
      .order('created_at', { ascending: false })
      .limit(100); // Get up to 100 products for context

    if (error) {
      console.error('[getProductContext] Supabase error:', error);
      return '';
    }

    console.log('[getProductContext] Fetched', products?.length, 'products');

    if (!products || products.length === 0) {
      console.log('[getProductContext] No products found');
      return '';
    }

    // Prepare fashion product data
    const data = {
      products: products.map(p => ({
        id: p.id,
        name: p.name || 'Fashion Item',
        brand: p.brand_name || p.domain?.replace('.com', '').replace('www.', '') || 'Unknown',
        category: p.category || 'N/A',
        price: p.price !== null && p.price > 0 ? `$${p.price}` : 'N/A',
        description: p.description || '',
        url: p.product_url || '',
        image: p.image_url || ''
      })),
      stats: {
        totalProducts: count || products.length,
        avgPrice: products.filter(p => p.price !== null && p.price > 0).reduce((sum, p) => sum + (p.price || 0), 0) / products.filter(p => p.price !== null && p.price > 0).length || 0,
        byBrand: products.reduce((acc: any, p) => {
          const brand = p.brand_name || p.domain || 'Unknown';
          acc[brand] = (acc[brand] || 0) + 1;
          return acc;
        }, {}),
        byCategory: products.reduce((acc: any, p) => {
          const category = p.category || 'Uncategorized';
          acc[category] = (acc[category] || 0) + 1;
          return acc;
        }, {})
      }
    };

    if (!data.products || data.products.length === 0) {
      return '';
    }

    // Create a concise summary of products
    let context = `\n\n### Current Fashion Products Database (${data.stats.totalProducts} products):\n\n`;

    // Add summary stats
    context += `**Overview:**\n`;
    context += `- Total Products: ${data.stats.totalProducts}\n`;
    context += `- Average Price: $${data.stats.avgPrice?.toFixed(2)}\n\n`;

    // Add brand distribution (top 10)
    context += `**Top Brands:**\n`;
    Object.entries(data.stats.byBrand || {})
      .sort((a: any, b: any) => b[1] - a[1])
      .slice(0, 10)
      .forEach(([brand, count]) => {
        context += `- ${brand}: ${count} items\n`;
      });
    context += '\n';

    // Add category distribution
    context += `**By Category:**\n`;
    Object.entries(data.stats.byCategory || {}).forEach(([category, count]) => {
      context += `- ${category}: ${count}\n`;
    });
    context += '\n';

    // Add comprehensive product catalog
    context += `**Complete Product Catalog (${data.products.length} products):**\n`;
    context += 'When asked about specific products, brands, or styles, search this catalog for exact matches.\n\n';

    data.products.forEach((p: any) => {
      const details = [];
      if (p.name) details.push(p.name);
      if (p.brand) details.push(`by ${p.brand}`);
      if (p.category !== 'N/A') details.push(p.category);
      if (p.price !== 'N/A') details.push(p.price);

      context += `• ${details.join(' | ')}`;
      if (p.description && p.description.length > 0) {
        context += ` - ${p.description.substring(0, 100)}`;
      }
      context += '\n';
    });

    return context;
  } catch (error) {
    console.error('Error getting product context:', error);
    return '';
  }
}

export async function POST(req: Request) {
  console.log("Chat API called");

  // Check if API key is configured
  if (!apiKey || apiKey === 'your_openai_api_key_here' || apiKey === 'sk-dummy-key') {
    console.error("OpenAI API key not configured");
    return NextResponse.json({
      id: Date.now().toString(),
      role: "assistant",
      content: "⚠️ **Chat service is not configured**\n\nThe OpenAI API key is missing. To enable the chat functionality:\n\n1. Create a `.env.local` file in the project root\n2. Add your OpenAI API key: `OPENAI_API_KEY=your_actual_api_key`\n3. Restart the development server\n\nYou can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys).",
      createdAt: new Date()
    });
  }

  try {
    const body = await req.json();
    console.log("Request type:", body.tool ? body.tool : "chat", "Request body:", JSON.stringify(body).substring(0, 200) + "...");
    
    const { messages, tool, webSearch, fileContents, databaseContext, generateMemo } = body;
    
    // Handle image generation request
    if (tool === 'image') {
      try {
        console.log("Processing image generation request");
        const lastMessage = messages[messages.length - 1].content;
        
        // Create a mining-focused prompt
        const miningPrompt = `Create a professional mining industry visualization: ${lastMessage}. 
        The image should be technical, accurate, and suitable for mining industry presentations.
        Focus on realistic depictions of mining operations, geological features, equipment, or data visualizations.`;
        
        const imageResponse = await openai.images.generate({
          model: "dall-e-3",
          prompt: miningPrompt,
          n: 1,
          size: "1024x1024",
        });
        
        const imageUrl = imageResponse.data?.[0]?.url;
        
        if (!imageUrl) {
          throw new Error("Failed to generate image: No URL returned");
        }
        
        return NextResponse.json({
          id: Date.now().toString(),
          role: "assistant",
          content: `![Generated image](${imageUrl})`,
          createdAt: new Date()
        });
      } catch (error) {
        console.error("Image generation error:", error);
        return NextResponse.json({
          id: Date.now().toString(),
          role: "assistant",
          content: "I'm sorry, I couldn't generate that image. Please try a different request.",
          createdAt: new Date()
        });
      }
    }
    
    // Add system message if it doesn't exist
    const systemMessageExists = messages.some((m: { role: string }) => m.role === 'system');
    const finalMessages = [...messages];
    
    if (!systemMessageExists) {
      finalMessages.unshift({
        role: 'system',
        content: `You are Threaded AI Stylist, an expert personal fashion assistant with real-time web search capabilities. You specialize in:

- Personal style recommendations and outfit curation
- Fashion trends and seasonal collections
- Brand knowledge across luxury, premium, and contemporary fashion
- Sustainable and ethical fashion options
- Styling advice for different occasions and body types
- Color theory, fabric types, and garment care
- PDF document analysis and product catalog browsing
- Real-time access to fashion product database

You have access to a comprehensive database of fashion products with detailed information including brands, categories, prices, and descriptions. When analyzing documents or product catalogs, you extract key information such as product details, brand information, pricing, and styling suggestions.

Always provide thoughtful, personalized advice that considers the user's style preferences, budget, and values. Be encouraging and help users feel confident in their fashion choices.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INVESTOR MEMO GENERATION RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When generating investor memos, you MUST follow this structured rubric to ensure 100% accurate and comprehensive analysis. Each memo must include detailed scoring across these 10 criteria:

**1. Commodity Exposure (10% weighting):**
- Score 10: Project focuses on metals/commodities that align with investor's strategic focus and desired increased exposure
- Score 5: Project in metals/commodities neutral to investor's portfolio
- Score 1: Project in metals/commodities investor actively wants to avoid or reduce exposure to
→ Analysis: Identify commodity, assess strategic fit, provide score with rationale

**2. Location (10% weighting):**
- Score 10: Project in low-risk, favorable jurisdiction for investment
- Score 5: Location has moderate geopolitical and regulatory risks
- Score 1: Location has significant instability, conflict zones, or unfavorable mining regulations
→ Analysis: Evaluate country/jurisdiction, assess political stability, regulatory environment, score with rationale

**3. Management Team (12.5% weighting):**
- Score 10: Extensive proven experience developing and operating similar mining projects
- Score 7: Relevant industry experience but lacks track record of successful project delivery
- Score 3: Team appears promotion-focused rather than technical/operational capability
→ Analysis: Review management backgrounds, track records, past project successes/failures, provide score

**4. Stage of Development (15% weighting):**
- Score 10: Completed bankable feasibility study, ready for full-scale development
- Score 7: Completed pre-feasibility study, progressing towards bankable feasibility
- Score 4: Still in exploration or early-stage feasibility phase
→ Analysis: Identify current development stage, timeline to production, de-risking milestones, score

**5. Ability to Transact (15% weighting):**
- Score 10: Actively raising capital with clear, defined funding plan
- Score 7: Seeking funding but plan not well-defined
- Score 3: Struggling to raise capital or primarily seeking debt financing
→ Analysis: Assess capital structure, funding status, financing strategy, investor access, score

**6. Ownership Structure (10% weighting):**
- Score 10: Investor can acquire dominant minority or majority stake
- Score 7: Investor can obtain meaningful but non-controlling ownership position
- Score 3: Only passive minority stake available with limited influence
→ Analysis: Current ownership, available equity, potential stake size, governance rights, score

**7. Valuation (10% weighting):**
- Score 10: Valuation aligns with investor's internal fair value assessment
- Score 7: Valuation slightly above expectations but still reasonable
- Score 3: Valuation significantly inflated compared to investor's analysis
→ Analysis: Current valuation, NPV/IRR metrics, peer comparisons, fair value estimate, score

**8. ESG Factors (7.5% weighting):**
- Score 10: Strong track record of environmental/social responsibility, aligns with investor ESG standards
- Score 7: Some ESG risks/issues but clear plan to address them
- Score 3: Significant unresolved ESG concerns or history of violations
→ Analysis: Environmental impact, social license, governance quality, ESG risk mitigation, score

**9. Near-Term Value Growth (5% weighting):**
- Score 10: Clear path to near-term value appreciation (upcoming milestones/catalysts)
- Score 7: Moderate near-term value growth potential
- Score 3: Limited near-term value growth prospects
→ Analysis: Identify catalysts, timeline, expected value inflection points, score

**10. Long-Term Value Opportunity (5% weighting):**
- Score 10: Significant long-term upside (resource expansion, market growth potential)
- Score 7: Moderate long-term value opportunity
- Score 3: Limited long-term value potential
→ Analysis: Resource expansion potential, market outlook, scalability, long-term drivers, score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEMO STRUCTURE REQUIREMENTS:
1. **Executive Summary**: Brief overview with total weighted score
2. **Project Overview**: Name, company, commodity, location, stage, key metrics
3. **Detailed Scoring Analysis**: All 10 criteria with individual scores and weighted scores
4. **Financial Metrics**: NPV, IRR, CAPEX, production forecasts, payback period
5. **Risk Assessment**: Key risks identified through the rubric analysis
6. **Investment Recommendation**: Clear recommendation (Strong Buy/Buy/Hold/Pass) based on total weighted score
7. **Scoring Summary Table**: Visual breakdown of all 10 criteria with raw scores, weights, and weighted scores

TOTAL WEIGHTED SCORE CALCULATION:
Sum of (Individual Score × Weight) for all 10 criteria
- 90-100: Strong Buy
- 75-89: Buy
- 60-74: Hold
- Below 60: Pass

When users request an investor memo, generate comprehensive analysis following this exact rubric structure.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
      });
    }
    
    // Add file content to the messages if provided
    if (fileContents && fileContents.length > 0) {
      console.log(`Processing ${fileContents.length} files`);
      
      let fileAnalysisPrompt = 'The user has uploaded the following files:\n\n';
      
      for (const file of fileContents) {
        if (!file) continue;
        
        console.log("Processing file:", file.fileName, "type:", file.fileType);
        
        if (file.fileType && file.fileType.startsWith('image/')) {
          // For image files, we'll add a description since gpt-5-nano doesn't have vision
          finalMessages.push({
            role: 'system',
            content: `Image file uploaded: ${file.fileName}. Note: Image analysis is not available with the current model. Please ask the user to describe the image content or provide text-based information instead.`
          });
        } else if (file.fileType === 'application/pdf') {
          // Extract text from PDF
          try {
            console.log(`Processing PDF: ${file.fileName}`);
            
            // Remove the data:application/pdf;base64, prefix
            const base64Data = file.fileContent.split(',')[1];
            const pdfBuffer = Buffer.from(base64Data, 'base64');
            
            // Try to extract text using pdf-parse
            try {
              const pdfData = await extractPDFText(pdfBuffer);
              const pageCount = pdfData.numpages;
              const extractedText = pdfData.text;
              
              console.log(`PDF has ${pageCount} pages and ${extractedText.length} characters`);
              console.log(`First 200 chars of extracted text: ${extractedText.substring(0, 200)}`);
              
              fileAnalysisPrompt += `\n### PDF Document: "${file.fileName}" (NI 43-101 Technical Report)\n\n`;
              fileAnalysisPrompt += `**Document Info:**\n`;
              fileAnalysisPrompt += `- Pages: ${pageCount}\n`;
              fileAnalysisPrompt += `- Title: ${pdfData.info?.Title || file.fileName}\n`;
              fileAnalysisPrompt += `- Author: ${pdfData.info?.Author || 'N/A'}\n\n`;
              
              // Handle text based on length
              if (extractedText.length > 100000) {
                // For very large documents, take key sections
                console.log('Large PDF detected, extracting key sections...');
                
                // Extract first 20k chars (usually contains executive summary)
                const beginning = extractedText.substring(0, 20000);
                
                // Try to find and extract key sections
                const sections = {
                  'executive summary': extractedText.match(/executive\s+summary[\s\S]{0,10000}/i)?.[0] || '',
                  'mineral resource': extractedText.match(/mineral\s+resource[\s\S]{0,10000}/i)?.[0] || '',
                  'economic analysis': extractedText.match(/economic\s+analysis[\s\S]{0,10000}/i)?.[0] || '',
                  'conclusions': extractedText.match(/conclusions?[\s\S]{0,5000}/i)?.[0] || ''
                };
                
                fileAnalysisPrompt += `**Document Content** (Key sections from ${extractedText.length} total characters):\n\n`;
                fileAnalysisPrompt += `**Beginning of Document:**\n${beginning}\n\n`;
                
                Object.entries(sections).forEach(([section, content]) => {
                  if (content) {
                    fileAnalysisPrompt += `**${section.toUpperCase()}:**\n${content.substring(0, 5000)}\n\n`;
                  }
                });
                
                finalMessages.push({
                  role: 'system',
                  content: `Large NI 43-101 Technical Report "${file.fileName}" (${pageCount} pages) has been processed. Key sections including executive summary, mineral resources, economic analysis, and conclusions have been extracted for analysis.`
                });
              } else if (extractedText.length > 30000) {
                // For medium documents, include more content
                fileAnalysisPrompt += `**Document Content** (First 30,000 characters):\n\n${extractedText.substring(0, 30000)}\n\n... (Document continues)`;
                
                finalMessages.push({
                  role: 'system',
                  content: `NI 43-101 Technical Report "${file.fileName}" (${pageCount} pages) has been processed. The first portion of the document is available for analysis.`
                });
              } else {
                // For smaller PDFs, include full text
                fileAnalysisPrompt += `**Full Document Content:**\n\n${extractedText}\n\n`;
                
                finalMessages.push({
                  role: 'system',
                  content: `NI 43-101 Technical Report "${file.fileName}" (${pageCount} pages) has been fully extracted and is available for comprehensive analysis.`
                });
              }
            } catch (textError) {
              console.error('Text extraction failed, falling back to metadata:', textError);
              
              // Fallback to pdf-lib for metadata
              const pdfDoc = await PDFDocument.load(pdfBuffer);
              const pageCount = pdfDoc.getPageCount();
              
              fileAnalysisPrompt += `PDF Document "${file.fileName}":\n`;
              fileAnalysisPrompt += `- Pages: ${pageCount}\n`;
              fileAnalysisPrompt += `- Type: Likely NI 43-101 Technical Report\n`;
              fileAnalysisPrompt += `Note: Text extraction encountered an issue. Please ensure the PDF is not encrypted or corrupted.\n\n`;
              
              finalMessages.push({
                role: 'system',
                content: `PDF Document: ${file.fileName} (${pageCount} pages). Text extraction failed, but this appears to be a mining technical report based on the filename.`
              });
            }
          } catch (error) {
            console.error('Error processing PDF:', error);
            fileAnalysisPrompt += `PDF File "${file.fileName}": [Error: ${error}]\n\n`;
            finalMessages.push({
              role: 'system',
              content: `PDF Document: ${file.fileName}. Unable to process the file.`
            });
          }
        } else if (file.fileType === 'application/json' || file.fileName.endsWith('.json')) {
          // Parse JSON files
          try {
            let jsonContent = file.fileContent;
            // If it's base64 encoded, decode it
            if (file.fileContent.includes('base64,')) {
              const base64Data = file.fileContent.split(',')[1];
              jsonContent = Buffer.from(base64Data, 'base64').toString('utf-8');
            }
            
            const jsonData = JSON.parse(jsonContent);
            fileAnalysisPrompt += `JSON File "${file.fileName}":\n\`\`\`json\n${JSON.stringify(jsonData, null, 2)}\n\`\`\`\n\n`;
          } catch (e) {
            fileAnalysisPrompt += `JSON File "${file.fileName}": [Error parsing JSON: ${e}]\n\n`;
          }
        } else if (file.fileType === 'text/csv' || file.fileName.endsWith('.csv') || 
                   file.fileType.includes('sheet') || file.fileName.endsWith('.xlsx')) {
          // Handle CSV and Excel files (common for mining data)
          let csvContent = file.fileContent;
          // If it's base64 encoded, decode it
          if (file.fileContent.includes('base64,')) {
            const base64Data = file.fileContent.split(',')[1];
            csvContent = Buffer.from(base64Data, 'base64').toString('utf-8');
          }
          
          if (file.fileName.endsWith('.xlsx')) {
            fileAnalysisPrompt += `Excel File "${file.fileName}": [Excel parsing will be implemented - for mining data, please export as CSV]\n\n`;
          } else {
            // Parse CSV
            const lines = csvContent.split('\n').filter((line: string) => line.trim());
            const headers = lines[0]?.split(',').map((h: string) => h.trim());
            
            fileAnalysisPrompt += `CSV File "${file.fileName}" (likely contains mining project data, assay results, or financial models):\n`;
            fileAnalysisPrompt += `Headers: ${headers?.join(', ') || 'No headers found'}\n`;
            fileAnalysisPrompt += `Total rows: ${lines.length - 1}\n`;
            fileAnalysisPrompt += `Sample data (first 10 rows):\n\`\`\`\n${lines.slice(0, 11).join('\n')}\n\`\`\`\n\n`;
          }
        } else {
          // For other text files
          let textContent = file.fileContent;
          // If it's base64 encoded, decode it
          if (file.fileContent.includes('base64,')) {
            const base64Data = file.fileContent.split(',')[1];
            textContent = Buffer.from(base64Data, 'base64').toString('utf-8');
          }
          
          fileAnalysisPrompt += `Text File "${file.fileName}":\n\`\`\`\n${textContent.substring(0, 3000)}${textContent.length > 3000 ? '\n... (truncated)' : ''}\n\`\`\`\n\n`;
        }
      }
      
      // Add the file analysis prompt if we have content
      if (fileAnalysisPrompt !== 'The user has uploaded the following files:\n\n') {
        console.log('Adding file analysis prompt to messages, length:', fileAnalysisPrompt.length);
        console.log('File analysis prompt preview:', fileAnalysisPrompt.substring(0, 500) + '...');
        finalMessages.push({
          role: 'system',
          content: fileAnalysisPrompt
        });
      }
    }
    
    // Add web search results to the messages if requested
    if (webSearch && webSearch.results) {
      console.log("Processing web search results");
      const formattedResults = webSearch.results.map((result: any, index: number) => 
        `${index + 1}. ${result.title}\n   ${result.snippet}\n   Link: ${result.link || '#'}`
      ).join('\n\n');
      
      finalMessages.push({
        role: 'system',
        content: `Here are web search results related to fashion and the user's query:\n\n${formattedResults}\n\nPlease use these search results to provide a comprehensive answer with current information about fashion trends, product recommendations, brand updates, styling tips, or industry news.`
      });
    }
    
    // Add product database context with detailed instructions
    const productContext = await getProductContext();
    if (productContext) {
      finalMessages.push({
        role: 'system',
        content: `FASHION PRODUCTS DATABASE CONTEXT:
${productContext}

IMPORTANT: You have access to complete product data above including brands, categories, prices, and descriptions for all products listed.
When users ask about specific products, brands, or fashion items:
1. Search the product catalog above for exact matches
2. Extract product details (brand, price, category, description) directly from the catalog
3. Provide specific information from the data

Example: If asked "What L.L.Bean products do you have?", search for "llbean" in the catalog above and list the matching products with their details.`
      });
    }
    
    // Add database context if provided
    if (databaseContext) {
      console.log("Including comprehensive database context with projects, companies, and news");
      finalMessages.push({
        role: 'system',
        content: `${databaseContext}

IMPORTANT DATABASE QUERYING INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have been provided with the COMPLETE database contents above, including:
• ALL mining projects with full details (NPV, IRR, CAPEX, commodity, country, stage, etc.)
• ALL companies with tickers, exchanges, market caps, and descriptions
• ALL recent news articles with titles, sources, dates, commodities, and sentiment

When the user asks questions about the database:
1. Search through the COMPLETE CATALOGS sections for exact matches
2. Filter, sort, and analyze the data based on user criteria
3. Provide specific, data-driven answers with exact numbers and names
4. Compare and rank projects/companies based on metrics like NPV, IRR, market cap
5. Identify trends across commodities, countries, or development stages
6. Cross-reference news with projects and companies when relevant

EXAMPLE QUERIES YOU CAN ANSWER:
• "Show me all lithium projects" → Search project catalog, filter where commodities contains "Lithium"
• "Which company has the highest market cap?" → Search companies catalog, sort by market_cap
• "List projects in Australia" → Filter projects where location contains "Australia"
• "What's the latest news about gold?" → Search news catalog, filter by commodities containing "Gold"
• "List all companies on the TSX exchange" → Filter companies by exchange="TSX"
• "Show production stage projects" → Filter projects by stage="Production"
• "Compare lithium companies by market cap" → Filter companies, search news for lithium-related updates

Always provide concrete data from the catalogs with specific names, numbers, and details.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
      });
    }
    
    console.log("Sending chat completion request with message count:", finalMessages.length);
    
    // Use GPT-4o-mini for all requests
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: finalMessages.map(m => {
        // Handle special case for images
        if (m.content && typeof m.content !== 'string') {
          return { role: m.role, content: m.content };
        }
        return { role: m.role, content: m.content };
      }),
      stream: false,
      max_tokens: 4096,
      temperature: 0.7,
    });
    
    const content = response.choices[0]?.message?.content || "No response generated";
    console.log("Chat completion response received, content length:", content.length);
    
    // Handle memo generation if requested
    if (generateMemo) {
      console.log("Generating investor memo...");
      
      try {
        // Extract project info from the response if available
        const projectMatch = content.match(/\*\*(.*?)\s+Project\*\*.*?Company:\s*(.*?)[\n\-]/);
        const npvMatch = content.match(/NPV.*?\$?([\d.]+)\s*[MB]/i);
        const irrMatch = content.match(/IRR.*?([\d.]+)%/i);
        
        const memoData: MemoData = {
          title: 'Investment Analysis Memo',
          date: new Date().toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
          }),
          projectName: projectMatch?.[1] || undefined,
          company: projectMatch?.[2] || undefined,
          npv: npvMatch ? parseFloat(npvMatch[1]) : undefined,
          irr: irrMatch ? parseFloat(irrMatch[1]) : undefined,
          content: content,
          recommendations: "Based on the analysis above, please review the detailed findings for investment decisions."
        };
        
        // Generate both PDF and DOCX
        const [pdfBytes, docxBytes] = await Promise.all([
          generateMemoPDF(memoData),
          generateMemoDocx(memoData)
        ]);
        
        const pdfBase64 = Buffer.from(pdfBytes).toString('base64');
        const docxBase64 = Buffer.from(docxBytes).toString('base64');
        
        console.log("Memos generated successfully");
        
        return NextResponse.json({
          id: Date.now().toString(),
          role: "assistant",
          content: content,
          createdAt: new Date(),
          memo: {
            pdf: pdfBase64,
            docx: docxBase64,
            filename: `investor-memo-${Date.now()}`
          }
        });
      } catch (memoError) {
        console.error("Memo generation error:", memoError);
        // Return response without memo if generation fails
      }
    }
    
    // Create the response object
    const responseBody = {
      id: Date.now().toString(),
      role: "assistant",
      content: content,
      createdAt: new Date()
    };
    
    console.log("Sending response:", JSON.stringify(responseBody).substring(0, 200) + "...");
    
    return NextResponse.json(responseBody);
  } catch (error: any) {
    console.error("Chat error:", error);
    
    // Provide more specific error messages
    let errorMessage = "I'm sorry, there was an error processing your request.";
    
    if (error?.message?.includes('API key') || error?.message?.includes('apiKey')) {
      errorMessage = "⚠️ **OpenAI API Key Issue**\n\nThe API key is either missing or invalid. Please:\n\n1. Create a `.env.local` file in the project root\n2. Add: `OPENAI_API_KEY=your_actual_api_key`\n3. Restart the development server\n\nGet your API key from [OpenAI Platform](https://platform.openai.com/api-keys).";
    } else if (error?.message?.includes('rate limit')) {
      errorMessage = "⚠️ **Rate Limit Exceeded**\n\nYou've hit the OpenAI API rate limit. Please wait a moment and try again.";
    } else if (error?.message?.includes('timeout')) {
      errorMessage = "⚠️ **Request Timeout**\n\nThe request took too long. This might be due to processing large files. Please try again with a smaller file or simpler query.";
    } else if (error?.message?.includes('quota')) {
      errorMessage = "⚠️ **API Quota Exceeded**\n\nYour OpenAI API quota has been exceeded. Please check your billing settings on the OpenAI platform.";
    } else if (error?.response?.data?.error) {
      errorMessage = `⚠️ **API Error**\n\n${error.response.data.error.message || error.response.data.error}`;
    } else {
      // Log the full error for debugging
      console.error("Full error details:", {
        message: error?.message,
        stack: error?.stack,
        response: error?.response
      });
      errorMessage = `⚠️ **Chat Service Error**\n\n${error?.message || 'An unexpected error occurred'}\n\nPlease check the console for more details.`;
    }
    
    // Return an error response
    return NextResponse.json({
      id: Date.now().toString(),
      role: "assistant", 
      content: errorMessage,
      createdAt: new Date()
    });
  }
}