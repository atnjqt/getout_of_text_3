#!/usr/bin/env python3
# Script to update HTML metadata for existing entries in the database

import sqlite3
import requests
from bs4 import BeautifulSoup
import sys

def extract_html_metadata(url):
    """
    Extract HTML metadata from the base URL of a PDF
    For URLs like https://supreme.justia.com/cases/federal/us/502/62/case.pdf
    This will extract from https://supreme.justia.com/cases/federal/us/502/62/
    """
    # Convert PDF URL to base URL if needed
    base_url = url
    if base_url.endswith('case.pdf'):
        base_url = base_url[:-8]  # Remove 'case.pdf'
    
    print(f"Fetching metadata from {base_url}")
    
    try:
        response = requests.get(base_url)
        if response.status_code != 200:
            return f"Failed to fetch HTML: HTTP {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the specified div structure
        flex_col_wrapper = soup.find('div', class_='flex-col-wrapper')
        if not flex_col_wrapper:
            return "Metadata wrapper div not found"
        
        # Get the original HTML of the wrapper div
        html_metadata = str(flex_col_wrapper)
        
        return html_metadata
    except Exception as e:
        return f"Error extracting metadata: {str(e)}"

def update_metadata_for_existing_entries(db_name='pdf_texts.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Update the schema to include html_metadata if it doesn't exist
    cursor.execute("PRAGMA table_info(pdf_texts)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'html_metadata' not in columns:
        cursor.execute('ALTER TABLE pdf_texts ADD COLUMN html_metadata TEXT')
        print("Added html_metadata column to database schema")

    # Get all URLs from the database
    cursor.execute('SELECT id, url FROM pdf_texts WHERE html_metadata IS NULL OR html_metadata = ""')
    rows = cursor.fetchall()
    
    if not rows:
        print("No entries without metadata found.")
        conn.close()
        return
    
    print(f"Found {len(rows)} entries without metadata. Starting update...")
    
    for row in rows:
        id_val, url = row
        if not url:
            print(f"Skipping ID {id_val} as URL is NULL")
            continue
            
        print(f"Processing ID {id_val}, URL: {url}")
        metadata = extract_html_metadata(url)
        
        cursor.execute('UPDATE pdf_texts SET html_metadata = ? WHERE id = ?', (metadata, id_val))
        print(f"Updated metadata for ID {id_val}")
    
    conn.commit()
    conn.close()
    print("Metadata update complete!")

def main():
    try:
        db_name = 'pdf_texts.db'
        if len(sys.argv) > 1:
            db_name = sys.argv[1]
        
        update_metadata_for_existing_entries(db_name)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
