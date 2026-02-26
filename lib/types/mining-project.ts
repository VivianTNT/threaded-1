export type ProjectStage =
  | 'Exploration'
  | 'Pre-Feasibility'
  | 'Feasibility'
  | 'Development'
  | 'Construction'
  | 'Production'
  | 'On Hold';

export type Commodity = string; // Allow any commodity as string

export type RiskLevel = 'Low' | 'Medium' | 'High' | 'Very High';

export type ESGGrade = 'A' | 'B' | 'C' | 'D' | 'F';

// Qualified Person interface
export interface QualifiedPerson {
  name: string;
  credentials: string;
  company: string;
}

// Database schema (actual columns from Supabase)
export interface MiningProject {
  id: string;
  company_id: string | null;
  name: string;
  location: string | null;
  stage: string | null;
  commodities: string[] | null;
  status: string | null;
  description: string | null;
  urls: string[] | null;
  watchlist: boolean;
  created_at: string;
  updated_at: string;

  // Financial metrics (added in migration 010)
  npv: number | null; // Net Present Value in millions USD
  irr: number | null; // Internal Rate of Return as percentage
  capex: number | null; // Capital Expenditure in millions USD
  aisc: number | null; // All-In Sustaining Cost in USD per unit (oz, lb, tonne)

  // Resource and Reserve estimates
  resource: string | null; // Resource estimate text
  reserve: string | null; // Reserve estimate text

  // Qualified Persons (added in migration 013)
  qualified_persons: QualifiedPerson[] | null; // Array of qualified persons who prepared technical reports

  // Geospatial coordinates (added in migration 014)
  latitude: number | null; // Latitude in decimal degrees (-90 to 90)
  longitude: number | null; // Longitude in decimal degrees (-180 to 180)

  // User upload and privacy fields
  user_id: string | null; // User who uploaded (for private documents)
  is_private: boolean; // Whether this is a user-uploaded private document
  uploaded_at: string | null; // When the document was uploaded
  document_storage_path: string | null; // Path to document in Supabase Storage

  // Computed/display fields (for backward compatibility)
  project?: string; // Alias for name
  company?: string; // Company name (joined)
  primaryCommodity?: string; // First commodity in array
  jurisdiction?: string; // Alias for location
  riskLevel?: RiskLevel;

  // Optional legacy fields
  project_id?: string;
  watchlisted_at?: string;
  generated_image_url?: string;
  technicalReportUrl?: string;
}

export interface ProjectFilter {
  stages?: ProjectStage[];
  commodities?: Commodity[];
  jurisdictions?: string[];
  riskLevels?: RiskLevel[];
  npvRange?: { min: number; max: number };
  irrRange?: { min: number; max: number };
  capexRange?: { min: number; max: number };
  aiscRange?: { min: number; max: number };
  searchQuery?: string;
}

export interface SavedView {
  id: string;
  name: string;
  filters: ProjectFilter;
  visibleColumns: string[];
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
} 