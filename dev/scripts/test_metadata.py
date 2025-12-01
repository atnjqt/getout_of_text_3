#!/usr/bin/env python3
# Test script to verify HTML metadata extraction

import requests
from bs4 import BeautifulSoup
import sqlite3

def extract_case_metadata(case_url):
    """
    Extract HTML metadata from a Supreme Court case page
    """
    print(f"Fetching metadata from {case_url}")
    
    try:
        response = requests.get(case_url)
        if response.status_code != 200:
            print(f"Failed to fetch page: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the specified div structure
        flex_col_wrapper = soup.find('div', class_='flex-col-wrapper')
        if not flex_col_wrapper:
            print("Metadata wrapper div not found")
            return None
        
        # Extract all the metadata items
        metadata = {}
        for div in flex_col_wrapper.find_all('div', class_='flex-col'):
            label = div.find('strong')
            value = div.find('span')
            if label and value:
                metadata[label.text.strip()] = value.text.strip()
        
        # Get the original HTML of the wrapper div for storage
        html_metadata = str(flex_col_wrapper)
        
        # Display metadata
        print("Extracted metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Display raw HTML
        print("\nHTML metadata:")
        print(html_metadata[:200] + "..." if len(html_metadata) > 200 else html_metadata)
        
        return html_metadata
        
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return None

def main():
    # Test URL for a Supreme Court case
    test_url = "https://supreme.justia.com/cases/federal/us/502/62/"
    
    print("Testing metadata extraction from a Supreme Court case page")
    metadata = extract_case_metadata(test_url)
    
    if metadata:
        print("\nMetadata extraction successful!")
        
        # Check if we can parse the metadata again to verify it's valid HTML
        try:
            soup = BeautifulSoup(metadata, 'html.parser')
            print("\nHTML validation successful!")
        except Exception as e:
            print(f"\nHTML validation failed: {e}")
    else:
        print("\nMetadata extraction failed!")

if __name__ == "__main__":
    main()
