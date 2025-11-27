"use client"

import React, { createContext, useContext, useState, useEffect } from "react"
import { supabase } from '@/lib/supabase/client'

// Define types
interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt?: Date;
}

interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

interface AppContext {
  projects: any[];
  totalProjects: number;
  totalCompanies: number;
  totalFilings: number;
  totalDeals: number;
  lastUpdated: string;
}

interface GlobalChatContextType {
  // Current chat state
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  currentChatId: string | null;
  setCurrentChatId: (id: string) => void;
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // Tool states
  isGeneratingImage: boolean;
  setIsGeneratingImage: (generating: boolean) => void;
  isSearchingWeb: boolean;
  setIsSearchingWeb: (searching: boolean) => void;
  searchResults: any[] | null;
  setSearchResults: (results: any[] | null) => void;
  uploadedFiles: any[];
  setUploadedFiles: (files: any[]) => void;

  // Chat history
  chatHistory: Chat[];
  setChatHistory: (history: Chat[]) => void;
  selectedChatId: string | null;
  setSelectedChatId: (id: string) => void;

  // Functions
  createNewChat: () => void;
  loadChat: (chatId: string) => Promise<void>;
  saveCurrentChat: () => Promise<void>;
  loadChatHistory: () => Promise<void>;

  // User info
  currentUser: any;
  setCurrentUser: (user: any) => void;

  // App context
  appContext: AppContext | null;
}

const GlobalChatContext = createContext<GlobalChatContextType | undefined>(undefined)

export function GlobalChatProvider({ children }: { children: React.ReactNode }) {
  // State for app context (needed before getInitialMessages)
  const [appContext, setAppContext] = useState<AppContext | null>(null)

  // Initialize with system message for fashion
  const getInitialMessages = (): Message[] => {
    let systemContent = `You are Threaded AI Stylist, an expert personal fashion assistant with real-time web search capabilities. You specialize in:

- Personal style recommendations and outfit curation
- Fashion trends and seasonal collections
- Brand knowledge across luxury, premium, and contemporary fashion
- Sustainable and ethical fashion options
- Styling advice for different occasions and body types
- Color theory, fabric types, and garment care

You can search the web for current fashion trends, analyze user wardrobes, suggest outfit combinations, and provide personalized shopping recommendations. When web search is enabled, you have access to current information from fashion retailers, style magazines, trend forecasters, and sustainable fashion sources.

Always provide thoughtful, personalized advice that considers the user's style preferences, budget, and values. Be encouraging and help users feel confident in their fashion choices.`;

    // Add app context if available
    if (appContext) {
      systemContent += `\n\n**Current Threaded User Context** (use when relevant to user queries):
- Saved Products: ${appContext.totalProjects} items
- Favorite Brands: ${appContext.totalCompanies} brands
- Wardrobe Items: ${appContext.totalFilings} pieces
- Style Activities: ${appContext.totalDeals} interactions
- Last Updated: ${new Date(appContext.lastUpdated).toLocaleString()}

Sample saved products: ${appContext.projects.slice(0, 5).map(p => {
  const details = [];
  if (p.name) details.push(`Item: ${p.name}`);
  if (p.brand_name) details.push(`Brand: ${p.brand_name}`);
  if (p.category) details.push(`Category: ${p.category}`);
  if (p.price !== null && p.price !== undefined && p.price > 0) details.push(`Price: $${p.price}`);
  if (p.domain) details.push(`From: ${p.domain}`);
  return `[${details.join(' | ')}]`;
}).join('\n')}.

When users ask about their saved items, wardrobe pieces, or style preferences, you have access to this data in the context above.

Note: This context is available for reference but should only be mentioned when directly relevant to the user's query.`;
    }

    return [
      {
        id: "1",
        role: "system",
        content: systemContent
      },
      {
        id: "2",
        role: "assistant",
        content: "Hello! I'm your Threaded AI Stylist, your personal fashion assistant. I can help you with:\n\n• **Style Recommendations** - Suggest outfits and pieces that match your personal style\n• **Fashion Trends** - Keep you updated on current and upcoming fashion trends\n• **Outfit Curation** - Help you build complete looks for any occasion\n• **Brand Discovery** - Find brands that align with your taste and values\n• **Styling Advice** - Answer questions about fit, color, and styling\n• **Smart Shopping** - Guide you to the perfect pieces for your wardrobe\n\nHow can I help you look and feel your best today?"
      }
    ];
  };
  
  const initialMessages = getInitialMessages();

  // State
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isGeneratingImage, setIsGeneratingImage] = useState(false)
  const [isSearchingWeb, setIsSearchingWeb] = useState(false)
  const [isDatabaseContextActive, setIsDatabaseContextActive] = useState(false)
  const [isMemoGeneratorActive, setIsMemoGeneratorActive] = useState(false)
  const [searchResults, setSearchResults] = useState<any[] | null>(null)
  const [cachedDatabaseContext, setCachedDatabaseContext] = useState<any>(null)
  const [isLoadingContext, setIsLoadingContext] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [chatHistory, setChatHistory] = useState<Chat[]>([])
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null)
  const [currentUser, setCurrentUser] = useState<any>(null)

  // Create new chat
  const createNewChat = () => {
    const newChatId = `chat_${Date.now()}`
    const newMessages = getInitialMessages()
    
    setCurrentChatId(newChatId)
    setMessages(newMessages)
    setInput("")
    setUploadedFiles([])
    setSearchResults(null)
    setIsGeneratingImage(false)
    setIsSearchingWeb(false)
    
    // Add to history
    const newChat: Chat = {
      id: newChatId,
      title: "New Style Conversation",
      messages: newMessages,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
    
    setChatHistory(prev => [newChat, ...prev])
    setSelectedChatId(newChatId)
  }

  // Load a specific chat
  const loadChat = async (chatId: string) => {
    const chat = chatHistory.find(c => c.id === chatId)
    if (chat) {
      setCurrentChatId(chatId)
      setMessages(chat.messages)
      setSelectedChatId(chatId)
      setInput("")
      setUploadedFiles([])
      setSearchResults(null)
      setIsGeneratingImage(false)
      setIsSearchingWeb(false)
    }
  }

  // Save current chat
  const saveCurrentChat = async () => {
    if (!currentChatId) return
    
    // Update chat in history
    setChatHistory(prev => prev.map(chat => {
      if (chat.id === currentChatId) {
        // Generate title from first user message if needed
        let title = chat.title
        if (title === "New Style Conversation" && messages.length > 2) {
          const firstUserMessage = messages.find(m => m.role === "user")
          if (firstUserMessage) {
            title = firstUserMessage.content.substring(0, 50) + (firstUserMessage.content.length > 50 ? "..." : "")
          }
        }
        
        return {
          ...chat,
          title,
          messages,
          updatedAt: new Date().toISOString()
        }
      }
      return chat
    }))
    
    // If Supabase is available, save to database
    if (currentUser && supabase) {
      try {
        const supabaseClient = supabase
        if (supabaseClient) {
          await supabaseClient
            .from('chat_history')
            .upsert({
              id: currentChatId,
              user_id: currentUser.id,
              title: messages.length > 2 ? messages[2]?.content?.substring(0, 50) + "..." : "New Style Conversation",
              messages: messages,
              updated_at: new Date().toISOString()
            })
        }
      } catch (error) {
        console.error('Error saving chat:', error)
      }
    }
  }

  // Load chat history
  const loadChatHistory = async () => {
    if (!currentUser || !supabase) return
    
    try {
      const supabaseClient = supabase
      if (supabaseClient) {
        const { data, error } = await supabaseClient
          .from('chat_history')
          .select('*')
          .eq('user_id', currentUser.id)
          .order('updated_at', { ascending: false })
          .limit(50)
        
        if (data && !error) {
          setChatHistory(data.map((chat: any) => ({
            id: chat.id,
            title: chat.title,
            messages: chat.messages,
            createdAt: chat.created_at,
            updatedAt: chat.updated_at
          })))
        }
      }
    } catch (error) {
      console.error('Error loading chat history:', error)
    }
  }

  // Auto-save when messages change
  useEffect(() => {
    if (currentChatId && messages.length > 2) {
      const timer = setTimeout(() => {
        saveCurrentChat()
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [messages, currentChatId])

  // Load chat history on user change
  useEffect(() => {
    if (currentUser) {
      loadChatHistory()
    }
  }, [currentUser])

  // Initialize with a new chat
  useEffect(() => {
    if (!currentChatId) {
      createNewChat()
    }
  }, [])

  // Fetch app context
  useEffect(() => {
    const fetchAppContext = async () => {
      try {
        // Fetch products from Penn database with key fashion fields
        const { data: products, count: productCount } = await supabase
          .from('products')
          .select('id, name, brand_name, category, price, description, image_url, product_url, domain, created_at', { count: 'exact' })
          .limit(100) // Get top 100 products for context

        // Fetch unique brands from products
        const { data: brandsData } = await supabase
          .from('products')
          .select('brand_name')
          .not('brand_name', 'is', null)

        const uniqueBrands = new Set(brandsData?.map(b => b.brand_name).filter(Boolean)).size

        // Use product count for wardrobe items and activities
        const wardrobeCount = Math.floor((productCount || 0) * 0.3) // Assume 30% of products are in wardrobe
        const activitiesCount = Math.floor((productCount || 0) * 0.5) // User interactions

        const newContext = {
          projects: products || [],
          totalProjects: productCount || 0,
          totalCompanies: uniqueBrands,
          totalFilings: wardrobeCount,
          totalDeals: activitiesCount,
          lastUpdated: new Date().toISOString()
        }

        setAppContext(newContext)

        // Update system message with context if messages exist
        setMessages(prev => {
          if (prev.length > 0 && prev[0].role === 'system') {
            const updatedMessages = [...prev]
            let systemContent = `You are Threaded AI Stylist, an expert personal fashion assistant with real-time web search capabilities. You specialize in:

- Personal style recommendations and outfit curation
- Fashion trends and seasonal collections
- Brand knowledge across luxury, premium, and contemporary fashion
- Sustainable and ethical fashion options
- Styling advice for different occasions and body types
- Color theory, fabric types, and garment care

You can search the web for current fashion trends, analyze user wardrobes, suggest outfit combinations, and provide personalized shopping recommendations. When web search is enabled, you have access to current information from fashion retailers, style magazines, trend forecasters, and sustainable fashion sources.

Always provide thoughtful, personalized advice that considers the user's style preferences, budget, and values. Be encouraging and help users feel confident in their fashion choices.

**Current Threaded User Context** (use when relevant to user queries):
- Saved Products: ${newContext.totalProjects} items
- Favorite Brands: ${newContext.totalCompanies} brands
- Wardrobe Items: ${newContext.totalFilings} pieces
- Style Activities: ${newContext.totalDeals} interactions
- Last Updated: ${new Date(newContext.lastUpdated).toLocaleString()}

Sample saved products: ${newContext.projects.slice(0, 5).map((p: any) => {
  const details = [];
  if (p.name) details.push(`Item: ${p.name}`);
  if (p.brand_name) details.push(`Brand: ${p.brand_name}`);
  if (p.category) details.push(`Category: ${p.category}`);
  if (p.price !== null && p.price !== undefined && p.price > 0) details.push(`Price: $${p.price}`);
  if (p.domain) details.push(`From: ${p.domain}`);
  return `[${details.join(' | ')}]`;
}).join('\n')}.

When users ask about their saved items, wardrobe pieces, or style preferences, you have access to this data in the context above.

Note: This context is available for reference but should only be mentioned when directly relevant to the user's query.`;

            updatedMessages[0] = { ...updatedMessages[0], content: systemContent }
            return updatedMessages
          }
          return prev
        })
      } catch (error) {
        console.error('Error fetching app context:', error)
      }
    }

    fetchAppContext()
    // Refresh context every 5 minutes
    const interval = setInterval(fetchAppContext, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const value = {
    messages,
    setMessages,
    currentChatId,
    setCurrentChatId,
    input,
    setInput,
    isLoading,
    setIsLoading,
    isGeneratingImage,
    setIsGeneratingImage,
    isSearchingWeb,
    setIsSearchingWeb,
    isDatabaseContextActive,
    setIsDatabaseContextActive,
    isMemoGeneratorActive,
    setIsMemoGeneratorActive,
    cachedDatabaseContext,
    setCachedDatabaseContext,
    isLoadingContext,
    setIsLoadingContext,
    searchResults,
    setSearchResults,
    uploadedFiles,
    setUploadedFiles,
    chatHistory,
    setChatHistory,
    selectedChatId,
    setSelectedChatId,
    createNewChat,
    loadChat,
    saveCurrentChat,
    loadChatHistory,
    currentUser,
    setCurrentUser,
    appContext
  }

  return (
    <GlobalChatContext.Provider value={value}>
      {children}
    </GlobalChatContext.Provider>
  )
}

export function useGlobalChat() {
  const context = useContext(GlobalChatContext)
  if (!context) {
    throw new Error("useGlobalChat must be used within a GlobalChatProvider")
  }
  return context
} 