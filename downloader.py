# downloader.py
# ARGUMENTS:
# 1. url (required): URL of the PDF to download and analyze
# 2. db_name (optional): Specify a custom database filename (default: pdf_texts.db)

import requests
import sqlite3
import datetime
import re
from bs4 import BeautifulSoup
from io import BytesIO
from PyPDF2 import PdfReader
import sys

def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    elif response.status_code == 404:
        print('PDF not found at the provided URL. Checking for alternative URLs...')
        # Always try to find any .pdf link on the parent page (handles random filenames and subdirectories)
        parent_url = url.rsplit('/', 1)[0]
        print(f"Searching for .pdf links at {parent_url}")
        page = requests.get(parent_url)
        if page.status_code == 200:
            soup = BeautifulSoup(page.text, 'html.parser')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.endswith('.pdf'):
                    # Handle relative and absolute links
                    if href.startswith('http'):
                        pdf_url = href
                    elif href.startswith('/'):
                        pdf_url = 'https://supreme.justia.com' + href
                    else:
                        pdf_url = parent_url + '/' + href
                    print(f"Trying found PDF link: {pdf_url}")
                    resp = requests.get(pdf_url)
                    if resp.status_code == 200:
                        return BytesIO(resp.content)
        raise Exception("Failed to download PDF: 404 and no alternative .pdf found on parent page.")
    else:
        raise Exception(f"Failed to download PDF: {response.status_code}")
    

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def extract_html_metadata(url):
    """
    Extract HTML metadata from the base URL of a PDF
    For URLs like https://supreme.justia.com/cases/federal/us/502/62/case.pdf or ##-####/index.pdf for newer versions
    This will extract from https://supreme.justia.com/cases/federal/us/502/62/
    """
    # Convert PDF URL to base URL if needed
    base_url = url
    if base_url.endswith('case.pdf'):
        base_url = base_url[:-8]  # Remove 'case.pdf'
    elif base_url.endswith('index.pdf'):
        base_url = base_url[:-9]  # Remove 'index.pdf'
    
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
        
        # Extract all the metadata items
        metadata = {}
        for div in flex_col_wrapper.find_all('div', class_='flex-col'):
            label = div.find('strong')
            value = div.find('span')
            if label and value:
                metadata[label.text.strip()] = value.text.strip()
        
        # Convert metadata to a string representation
        metadata_str = '\n'.join([f"{key}: {value}" for key, value in metadata.items()])
        
        # Get the original HTML of the wrapper div
        html_metadata = str(flex_col_wrapper)
        
        return {
            'formatted_metadata': metadata_str,
            'raw_html': html_metadata,
            'metadata_dict': metadata
        }
    except Exception as e:
        return f"Error extracting metadata: {str(e)}"
    
def save_text_to_db(text, url, db_name='pdf_texts.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            url TEXT,
            timestamp TEXT,
            html_metadata TEXT
        )
    ''')
    
    # Update the schema to include html_metadata if it doesn't exist
    cursor.execute("PRAGMA table_info(pdf_texts)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'html_metadata' not in columns:
        cursor.execute('ALTER TABLE pdf_texts ADD COLUMN html_metadata TEXT')
        print("Added html_metadata column to database schema")
    
    # Check if URL already exists in database
    cursor.execute('SELECT id FROM pdf_texts WHERE url = ?', (url,))
    existing = cursor.fetchone()
    
    timestamp = datetime.datetime.now().isoformat()
    metadata_result = extract_html_metadata(url)
    
    if isinstance(metadata_result, dict):
        html_metadata = metadata_result['raw_html']
    else:
        html_metadata = metadata_result  # Error message
    
    if existing:
        # Update the existing entry with HTML metadata
        cursor.execute('UPDATE pdf_texts SET html_metadata = ? WHERE url = ?', 
                      (html_metadata, url))
        print(f"Updated HTML metadata for existing entry with URL: {url}")
    else:
        # Insert new entry with text and HTML metadata
        cursor.execute('INSERT INTO pdf_texts (text, url, timestamp, html_metadata) VALUES (?, ?, ?, ?)', 
                      (text, url, timestamp, html_metadata))
        print(f"Added new PDF from {url} with HTML metadata to database.")
    
    conn.commit()
    conn.close()

def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter the PDF URL: ")

    db_name = 'pdf_texts.db'
    if len(sys.argv) > 2:
        db_name = sys.argv[2]
    print(f"Using database: {db_name}")

    try:
        pdf_file = download_pdf(url)
        text = extract_text_from_pdf(pdf_file)
        save_text_to_db(text, url)
        print("PDF text saved to database successfully.")
        
        conn = sqlite3.connect('pdf_texts.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM pdf_texts')
        row_count = cursor.fetchone()[0]
        print(f"Total PDFs in database: {row_count}")
        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()