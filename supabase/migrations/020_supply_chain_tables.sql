-- Supply Chain Tables for Critical Minerals Tracking
-- Migration 020: Create supply chain infrastructure

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Supply Chain Nodes Table
-- Represents entities in the supply chain (mines, suppliers, smelters, refineries, manufacturers)
CREATE TABLE IF NOT EXISTS supply_chain_nodes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  node_type TEXT NOT NULL CHECK (node_type IN ('mine', 'supplier', 'smelter', 'refinery', 'manufacturer', 'logistics_provider')),
  location TEXT,
  country TEXT,
  latitude DECIMAL(10, 8),
  longitude DECIMAL(11, 8),
  commodities TEXT[], -- Array of commodities handled
  capacity DECIMAL(15, 2), -- Annual capacity
  capacity_unit TEXT, -- 'tonnes', 'kg', 'oz'
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disrupted', 'at_risk', 'inactive')),
  parent_company TEXT,
  certifications TEXT[], -- ISO, sustainability certifications
  description TEXT,
  website TEXT,
  project_id UUID REFERENCES projects(id) ON DELETE SET NULL, -- Link to mining projects if node_type is 'mine'
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Supply Chain Flows Table
-- Represents material flows between nodes
CREATE TABLE IF NOT EXISTS supply_chain_flows (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_node_id UUID NOT NULL REFERENCES supply_chain_nodes(id) ON DELETE CASCADE,
  target_node_id UUID NOT NULL REFERENCES supply_chain_nodes(id) ON DELETE CASCADE,
  commodity TEXT NOT NULL,
  volume DECIMAL(15, 2), -- Annual volume
  volume_unit TEXT, -- 'tonnes', 'kg', 'oz'
  contract_type TEXT, -- 'offtake', 'spot', 'long_term'
  contract_start_date DATE,
  contract_end_date DATE,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disrupted', 'at_risk', 'inactive')),
  transportation_mode TEXT[], -- ['rail', 'ship', 'truck']
  avg_transit_time_days INTEGER,
  cost_per_unit DECIMAL(15, 2),
  currency TEXT DEFAULT 'USD',
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- Ensure source and target are different
  CONSTRAINT different_nodes CHECK (source_node_id != target_node_id)
);

-- Supply Chain Events Table
-- Tracks disruptions, risks, news, and events
CREATE TABLE IF NOT EXISTS supply_chain_events (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_type TEXT NOT NULL CHECK (event_type IN ('disruption', 'delay', 'price_shock', 'geopolitical', 'esg_incident', 'weather', 'logistics', 'regulatory')),
  title TEXT NOT NULL,
  description TEXT,
  severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'monitoring', 'resolved')),
  affected_node_ids UUID[] NOT NULL, -- Array of affected node IDs
  affected_flow_ids UUID[], -- Array of affected flow IDs
  commodities TEXT[], -- Commodities impacted
  source_url TEXT, -- URL to news article or report
  source_name TEXT, -- Name of the source
  sentiment_score DECIMAL(3, 2), -- -1.0 to 1.0
  event_date DATE NOT NULL,
  resolution_date DATE,
  impact_description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Supply Chain Risk Metrics Table
-- Calculated risk scores for nodes and flows
CREATE TABLE IF NOT EXISTS supply_chain_risk_metrics (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  node_id UUID REFERENCES supply_chain_nodes(id) ON DELETE CASCADE,
  flow_id UUID REFERENCES supply_chain_flows(id) ON DELETE CASCADE,
  commodity TEXT NOT NULL,
  risk_category TEXT NOT NULL, -- 'concentration', 'geopolitical', 'esg', 'logistics', 'financial'
  risk_score INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
  risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
  factors JSONB, -- Contributing factors as JSON
  calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- Ensure either node_id or flow_id is set
  CONSTRAINT node_or_flow CHECK (
    (node_id IS NOT NULL AND flow_id IS NULL) OR
    (node_id IS NULL AND flow_id IS NOT NULL)
  )
);

-- Processing Stages Table
-- Tracks material transformation stages at processing facilities
CREATE TABLE IF NOT EXISTS processing_stages (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  node_id UUID NOT NULL REFERENCES supply_chain_nodes(id) ON DELETE CASCADE,
  input_commodity TEXT NOT NULL,
  output_commodity TEXT NOT NULL,
  conversion_rate DECIMAL(5, 4), -- e.g., 0.95 for 95% yield
  processing_method TEXT,
  energy_consumption_kwh DECIMAL(15, 2),
  water_consumption_m3 DECIMAL(15, 2),
  emissions_co2_tonnes DECIMAL(15, 2),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_supply_chain_nodes_node_type ON supply_chain_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_supply_chain_nodes_country ON supply_chain_nodes(country);
CREATE INDEX IF NOT EXISTS idx_supply_chain_nodes_commodities ON supply_chain_nodes USING GIN(commodities);
CREATE INDEX IF NOT EXISTS idx_supply_chain_nodes_status ON supply_chain_nodes(status);
CREATE INDEX IF NOT EXISTS idx_supply_chain_nodes_project_id ON supply_chain_nodes(project_id);

CREATE INDEX IF NOT EXISTS idx_supply_chain_flows_source ON supply_chain_flows(source_node_id);
CREATE INDEX IF NOT EXISTS idx_supply_chain_flows_target ON supply_chain_flows(target_node_id);
CREATE INDEX IF NOT EXISTS idx_supply_chain_flows_commodity ON supply_chain_flows(commodity);
CREATE INDEX IF NOT EXISTS idx_supply_chain_flows_status ON supply_chain_flows(status);

CREATE INDEX IF NOT EXISTS idx_supply_chain_events_event_type ON supply_chain_events(event_type);
CREATE INDEX IF NOT EXISTS idx_supply_chain_events_severity ON supply_chain_events(severity);
CREATE INDEX IF NOT EXISTS idx_supply_chain_events_status ON supply_chain_events(status);
CREATE INDEX IF NOT EXISTS idx_supply_chain_events_event_date ON supply_chain_events(event_date);
CREATE INDEX IF NOT EXISTS idx_supply_chain_events_affected_nodes ON supply_chain_events USING GIN(affected_node_ids);
CREATE INDEX IF NOT EXISTS idx_supply_chain_events_commodities ON supply_chain_events USING GIN(commodities);

CREATE INDEX IF NOT EXISTS idx_supply_chain_risk_metrics_node ON supply_chain_risk_metrics(node_id);
CREATE INDEX IF NOT EXISTS idx_supply_chain_risk_metrics_flow ON supply_chain_risk_metrics(flow_id);
CREATE INDEX IF NOT EXISTS idx_supply_chain_risk_metrics_commodity ON supply_chain_risk_metrics(commodity);
CREATE INDEX IF NOT EXISTS idx_supply_chain_risk_metrics_risk_level ON supply_chain_risk_metrics(risk_level);

CREATE INDEX IF NOT EXISTS idx_processing_stages_node ON processing_stages(node_id);
CREATE INDEX IF NOT EXISTS idx_processing_stages_commodity ON processing_stages(input_commodity, output_commodity);

-- Row Level Security (RLS) Policies
ALTER TABLE supply_chain_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE supply_chain_flows ENABLE ROW LEVEL SECURITY;
ALTER TABLE supply_chain_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE supply_chain_risk_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_stages ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read all supply chain data
CREATE POLICY "Allow read access to authenticated users" ON supply_chain_nodes
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read access to authenticated users" ON supply_chain_flows
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read access to authenticated users" ON supply_chain_events
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read access to authenticated users" ON supply_chain_risk_metrics
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read access to authenticated users" ON processing_stages
  FOR SELECT TO authenticated USING (true);

-- Allow service role to manage all supply chain data
CREATE POLICY "Allow service role full access" ON supply_chain_nodes
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access" ON supply_chain_flows
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access" ON supply_chain_events
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access" ON supply_chain_risk_metrics
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access" ON processing_stages
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_supply_chain_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_supply_chain_nodes_timestamp
  BEFORE UPDATE ON supply_chain_nodes
  FOR EACH ROW
  EXECUTE FUNCTION update_supply_chain_timestamp();

CREATE TRIGGER update_supply_chain_flows_timestamp
  BEFORE UPDATE ON supply_chain_flows
  FOR EACH ROW
  EXECUTE FUNCTION update_supply_chain_timestamp();

CREATE TRIGGER update_supply_chain_events_timestamp
  BEFORE UPDATE ON supply_chain_events
  FOR EACH ROW
  EXECUTE FUNCTION update_supply_chain_timestamp();

CREATE TRIGGER update_processing_stages_timestamp
  BEFORE UPDATE ON processing_stages
  FOR EACH ROW
  EXECUTE FUNCTION update_supply_chain_timestamp();

-- Insert sample data for demonstration
-- Sample nodes
INSERT INTO supply_chain_nodes (name, node_type, location, country, latitude, longitude, commodities, capacity, capacity_unit, status, parent_company, description)
VALUES
  ('Mt Marion Lithium Mine', 'mine', 'Western Australia', 'Australia', -32.7506, 121.1983, ARRAY['Lithium'], 450000, 'tonnes', 'active', 'Mineral Resources Ltd', 'Major lithium spodumene mine in Western Australia'),
  ('Greenbushes Lithium Mine', 'mine', 'Western Australia', 'Australia', -33.8567, 116.0531, ARRAY['Lithium', 'Tantalum'], 1400000, 'tonnes', 'active', 'Talison Lithium', 'World''s largest hard-rock lithium mine'),
  ('Ganfeng Lithium Refinery', 'refinery', 'Jiangxi Province', 'China', 27.6187, 114.9406, ARRAY['Lithium'], 95000, 'tonnes', 'active', 'Ganfeng Lithium', 'Leading lithium refinery producing lithium carbonate and hydroxide'),
  ('Albemarle Silver Peak', 'mine', 'Nevada', 'USA', 37.7285, -117.8728, ARRAY['Lithium'], 6000, 'tonnes', 'active', 'Albemarle Corporation', 'Only lithium brine operation in the United States'),
  ('BASF Cathode Materials', 'manufacturer', 'Schwarzheide', 'Germany', 51.4830, 13.8703, ARRAY['Lithium'], NULL, NULL, 'active', 'BASF', 'Cathode materials manufacturing for EV batteries'),
  ('Tianqi Lithium', 'refinery', 'Sichuan Province', 'China', 29.3570, 103.0706, ARRAY['Lithium'], 24000, 'tonnes', 'active', 'Tianqi Lithium Industries', 'Major lithium refinery and processor'),
  ('Vale Sudbury Mine', 'mine', 'Ontario', 'Canada', 46.4917, -80.9930, ARRAY['Nickel', 'Copper', 'Cobalt'], 40000, 'tonnes', 'active', 'Vale', 'Integrated nickel-copper mining complex'),
  ('Glencore Murrin Murrin', 'mine', 'Western Australia', 'Australia', -28.7372, 121.2417, ARRAY['Nickel', 'Cobalt'], 35000, 'tonnes', 'active', 'Glencore', 'Nickel-cobalt laterite mine and processing facility'),
  ('Jinchuan Nickel Refinery', 'refinery', 'Gansu Province', 'China', 38.5245, 102.1802, ARRAY['Nickel', 'Copper', 'Cobalt'], 150000, 'tonnes', 'active', 'Jinchuan Group', 'China''s largest nickel producer and refiner'),
  ('Umicore Olen', 'refinery', 'Olen', 'Belgium', 51.1517, 4.8907, ARRAY['Cobalt', 'Nickel'], 20000, 'tonnes', 'active', 'Umicore', 'Leading battery materials and recycling facility');

-- Sample flows
INSERT INTO supply_chain_flows (source_node_id, target_node_id, commodity, volume, volume_unit, contract_type, status, transportation_mode, avg_transit_time_days, cost_per_unit, currency)
SELECT
  s.id, t.id, 'Lithium', 250000, 'tonnes', 'long_term', 'active', ARRAY['ship'], 35, 850, 'USD'
FROM supply_chain_nodes s, supply_chain_nodes t
WHERE s.name = 'Mt Marion Lithium Mine' AND t.name = 'Ganfeng Lithium Refinery';

INSERT INTO supply_chain_flows (source_node_id, target_node_id, commodity, volume, volume_unit, contract_type, status, transportation_mode, avg_transit_time_days, cost_per_unit, currency)
SELECT
  s.id, t.id, 'Lithium', 180000, 'tonnes', 'long_term', 'active', ARRAY['ship'], 38, 875, 'USD'
FROM supply_chain_nodes s, supply_chain_nodes t
WHERE s.name = 'Greenbushes Lithium Mine' AND t.name = 'Tianqi Lithium';

INSERT INTO supply_chain_flows (source_node_id, target_node_id, commodity, volume, volume_unit, contract_type, status, transportation_mode, avg_transit_time_days)
SELECT
  s.id, t.id, 'Lithium', 45000, 'tonnes', 'long_term', 'active', ARRAY['truck', 'rail'], 14
FROM supply_chain_nodes s, supply_chain_nodes t
WHERE s.name = 'Ganfeng Lithium Refinery' AND t.name = 'BASF Cathode Materials';

INSERT INTO supply_chain_flows (source_node_id, target_node_id, commodity, volume, volume_unit, contract_type, status, transportation_mode, avg_transit_time_days)
SELECT
  s.id, t.id, 'Nickel', 25000, 'tonnes', 'offtake', 'active', ARRAY['rail', 'ship'], 42
FROM supply_chain_nodes s, supply_chain_nodes t
WHERE s.name = 'Vale Sudbury Mine' AND t.name = 'Jinchuan Nickel Refinery';

INSERT INTO supply_chain_flows (source_node_id, target_node_id, commodity, volume, volume_unit, contract_type, status, transportation_mode, avg_transit_time_days)
SELECT
  s.id, t.id, 'Cobalt', 3500, 'tonnes', 'long_term', 'active', ARRAY['ship', 'rail'], 55
FROM supply_chain_nodes s, supply_chain_nodes t
WHERE s.name = 'Glencore Murrin Murrin' AND t.name = 'Umicore Olen';

-- Sample events
INSERT INTO supply_chain_events (event_type, title, description, severity, status, affected_node_ids, commodities, source_url, sentiment_score, event_date)
SELECT
  'weather',
  'Tropical Cyclone Disrupts Australian Lithium Exports',
  'Tropical Cyclone Jasper caused temporary closure of Port Hedland, delaying lithium shipments from Western Australia to Asian refineries.',
  'medium',
  'resolved',
  ARRAY[id],
  ARRAY['Lithium'],
  'https://example.com/cyclone-disruption',
  -0.65,
  CURRENT_DATE - INTERVAL '15 days'
FROM supply_chain_nodes
WHERE name IN ('Mt Marion Lithium Mine', 'Greenbushes Lithium Mine')
LIMIT 1;

INSERT INTO supply_chain_events (event_type, title, description, severity, status, affected_node_ids, commodities, sentiment_score, event_date)
SELECT
  'price_shock',
  'Lithium Carbonate Prices Surge 45%',
  'Lithium carbonate prices increased sharply due to strong EV demand and supply constraints in China.',
  'high',
  'active',
  ARRAY_AGG(id),
  ARRAY['Lithium'],
  0.35,
  CURRENT_DATE - INTERVAL '7 days'
FROM supply_chain_nodes
WHERE commodities @> ARRAY['Lithium'] AND node_type IN ('refinery', 'manufacturer');

INSERT INTO supply_chain_events (event_type, title, description, severity, status, affected_node_ids, commodities, event_date)
SELECT
  'esg_incident',
  'Water Usage Concerns at Lithium Brine Operation',
  'Environmental groups raise concerns about water consumption at lithium brine extraction facilities.',
  'medium',
  'monitoring',
  ARRAY[id],
  ARRAY['Lithium'],
  CURRENT_DATE - INTERVAL '3 days'
FROM supply_chain_nodes
WHERE name = 'Albemarle Silver Peak';

-- Sample risk metrics
INSERT INTO supply_chain_risk_metrics (node_id, commodity, risk_category, risk_score, risk_level, factors)
SELECT
  id,
  'Lithium',
  'geopolitical',
  72,
  'high',
  '{"china_dependence": 0.85, "trade_restrictions": 0.6, "export_controls": 0.7}'::jsonb
FROM supply_chain_nodes
WHERE name = 'Ganfeng Lithium Refinery';

INSERT INTO supply_chain_risk_metrics (node_id, commodity, risk_category, risk_score, risk_level, factors)
SELECT
  id,
  'Nickel',
  'esg',
  58,
  'medium',
  '{"environmental_impact": 0.65, "water_usage": 0.55, "community_relations": 0.52}'::jsonb
FROM supply_chain_nodes
WHERE name = 'Vale Sudbury Mine';

-- Sample processing stages
INSERT INTO processing_stages (node_id, input_commodity, output_commodity, conversion_rate, processing_method, energy_consumption_kwh, emissions_co2_tonnes)
SELECT
  id,
  'Lithium Spodumene',
  'Lithium Carbonate',
  0.92,
  'Sulfuric Acid Roasting',
  2500,
  1.8
FROM supply_chain_nodes
WHERE name = 'Ganfeng Lithium Refinery';

INSERT INTO processing_stages (node_id, input_commodity, output_commodity, conversion_rate, processing_method, energy_consumption_kwh, emissions_co2_tonnes)
SELECT
  id,
  'Nickel Ore',
  'Refined Nickel',
  0.88,
  'Hydrometallurgical Processing',
  3800,
  2.5
FROM supply_chain_nodes
WHERE name = 'Jinchuan Nickel Refinery';
