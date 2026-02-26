#!/usr/bin/env python3
"""
Extract financial and project data from mining technical documents
"""

import os
import json
import requests
from typing import Dict, List, Optional
import PyPDF2
from io import BytesIO
from supabase import create_client, Client
from openai import OpenAI
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env.local')

# Initialize clients
supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

supabase: Client = create_client(supabase_url, supabase_key)
openai_client = OpenAI(api_key=openai_api_key)

def extract_text_from_pdf(pdf_content: bytes, max_pages: int = 50) -> str:
    """Extract text from PDF content"""
    try:
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        pages_to_read = min(len(pdf_reader.pages), max_pages)

        for page_num in range(pages_to_read):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_data_with_ai(text: str, file_name: str) -> Optional[Dict]:
    """Use OpenAI to extract structured data from text"""
    try:
        prompt = f"""
        Analyze this mining technical report and extract the following information.
        Return ONLY a valid JSON object with these fields:

        {{
          "project_name": "string",
          "company_name": "string",
          "location": "string (country or region)",
          "commodities": ["array of commodity strings"],
          "npv": number (in millions USD, null if not found),
          "irr": number (percentage, null if not found),
          "capex": number (in millions USD, null if not found),
          "opex": number (per unit cost, null if not found),
          "resource": "string (e.g., '43.7 Mt @ 2.5% Cu')",
          "reserve": "string (e.g., '11.8 Mt @ 3.1% Cu')",
          "mine_life": number (years, null if not found),
          "production_rate": "string (e.g., '100,000 oz/year')",
          "stage": "string (Exploration/Pre-Feasibility/Feasibility/Development/Production)"
        }}

        Document name: {file_name}

        Text excerpt:
        {text[:8000]}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a mining industry analyst. Extract only factual data from technical reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error with AI extraction: {e}")
        return None

def process_document(file_path: str, file_name: str) -> Optional[Dict]:
    """Process a single document from Supabase storage"""
    print(f"\n📄 Processing: {file_name}")

    try:
        # Download file from Supabase storage
        bucket_name = 'technical-documents'
        response = supabase.storage.from_(bucket_name).download(file_path)

        if not response:
            print(f"  ❌ Failed to download file")
            return None

        # Extract text from PDF
        text = extract_text_from_pdf(response)
        if not text or len(text) < 100:
            print(f"  ⚠️ Insufficient text extracted")
            return None

        print(f"  📝 Extracted {len(text)} characters")

        # Extract data using AI
        extracted_data = extract_data_with_ai(text, file_name)
        if not extracted_data:
            print(f"  ❌ Failed to extract structured data")
            return None

        print(f"  ✅ Extracted data successfully")
        return extracted_data

    except Exception as e:
        print(f"  ❌ Error processing document: {e}")
        return None

def save_to_database(data: Dict, file_path: str, file_name: str) -> bool:
    """Save extracted data to Supabase database"""
    try:
        # Find or create company
        company_name = data.get('company_name', 'Unknown')
        company_response = supabase.table('companies').select('id').ilike('name', f'%{company_name}%').limit(1).execute()

        if company_response.data and len(company_response.data) > 0:
            company_id = company_response.data[0]['id']
        else:
            # Create new company
            new_company = {
                'id': str(uuid.uuid4()),
                'name': company_name,
                'ticker': None,
                'description': f'Mining company from {file_name}'
            }
            supabase.table('companies').insert(new_company).execute()
            company_id = new_company['id']

        # Prepare project data
        project_data = {
            'id': str(uuid.uuid4()),
            'company_id': company_id,
            'name': data.get('project_name', file_name.replace('.pdf', '')),
            'location': data.get('location'),
            'commodities': data.get('commodities', []),
            'npv': data.get('npv'),
            'irr': data.get('irr'),
            'capex': data.get('capex'),
            'resource': data.get('resource'),
            'reserve': data.get('reserve'),
            'stage': data.get('stage', 'Unknown'),
            'status': 'Active',
            'description': f'Extracted from technical report: {file_name}',
            'document_storage_path': f'https://{supabase_url.replace("https://", "")}/storage/v1/object/public/technical-documents/{file_path}',
            'urls': [f'https://{supabase_url.replace("https://", "")}/storage/v1/object/public/technical-documents/{file_path}'],
            'watchlist': False,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }

        # Insert project
        supabase.table('projects').insert(project_data).execute()

        # Add highlights for key metrics
        highlights = []
        if data.get('npv'):
            highlights.append({
                'id': str(uuid.uuid4()),
                'project_id': project_data['id'],
                'data_type': 'NPV',
                'value': str(data['npv']),
                'quote': f"NPV: ${data['npv']}M",
                'page': 1
            })

        if data.get('irr'):
            highlights.append({
                'id': str(uuid.uuid4()),
                'project_id': project_data['id'],
                'data_type': 'IRR',
                'value': str(data['irr']),
                'quote': f"IRR: {data['irr']}%",
                'page': 1
            })

        if data.get('capex'):
            highlights.append({
                'id': str(uuid.uuid4()),
                'project_id': project_data['id'],
                'data_type': 'CAPEX',
                'value': str(data['capex']),
                'quote': f"CAPEX: ${data['capex']}M",
                'page': 1
            })

        if highlights:
            supabase.table('pdf_highlights').insert(highlights).execute()

        print(f"  💾 Saved to database: {data.get('project_name', 'Unknown')}")
        return True

    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False

def main():
    """Main extraction process"""
    print("🔍 Mining Document Data Extraction (Python)")
    print("=" * 80)

    # List files in mining-documents folder
    bucket_name = 'technical-documents'
    folder_path = 'mining-documents'

    try:
        # Get list of files
        response = supabase.storage.from_(bucket_name).list(folder_path)

        if not response:
            print("❌ No files found in mining-documents")
            return

        files = [f for f in response if f['name'].endswith('.pdf')]
        print(f"📚 Found {len(files)} PDF files to process\n")

        successful = 0
        failed = 0

        for file in files[:5]:  # Process first 5 for testing
            file_path = f"{folder_path}/{file['name']}"

            # Process document
            extracted_data = process_document(file_path, file['name'])

            if extracted_data:
                # Save to database
                if save_to_database(extracted_data, file_path, file['name']):
                    successful += 1
                else:
                    failed += 1
            else:
                failed += 1

        print("\n" + "=" * 80)
        print(f"✅ Extraction Complete!")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()