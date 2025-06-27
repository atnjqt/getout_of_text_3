# This python script will enumerate through each page of https://supreme.justia.com/cases/federal/us/
# there are pages 1 - 606
# I want to start with just 1 - 5 though to avoid scaling up. 
# Let's include a 5 second sleep between requests to avoid overwhelming the server.
# on review it seems like there are only PDFs for 502-606, so we will stick to those for now. 
# Later on we can check for pdf hyperlinks on older ones!
# That's like 1991 - 2006, so we can start with those.

import requests
import re
from bs4 import BeautifulSoup
import sqlite3
import datetime
from io import BytesIO
from PyPDF2 import PdfReader
import sys

# Function to download the page and extract case links
def extract_case_links(url, volume_number):
    print('getting case links from:', url)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page: {response.status_code}")
        return []
        
    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    # Accept any link that starts with the volume path and has more after it
    case_pattern = re.compile(rf'^/cases/federal/us/{volume_number}/[^/]+/?$')
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if case_pattern.match(href):
            links.append(href)
    
    # Get unique links only
    unique_links = list(set(links))
    print(f"Found {len(unique_links)} unique case links")
    return unique_links

# Function to construct PDF URL from case link
def construct_pdf_url(case_link):
    # Convert "/cases/federal/us/502/62/" to "/cases/federal/us/502/62/case.pdf"
    return f"https://supreme.justia.com{case_link}case.pdf"

# Check if volume number is provided as a command line argument
if len(sys.argv) > 1:
    try:
        volume_number = int(sys.argv[1])
        if not (502 <= volume_number <= 605):
            print("Volume number must be between 502 and 605. Exiting.")
            sys.exit(1)
    except ValueError:
        print("Please provide a valid number as the first argument. Exiting.")
        sys.exit(1)
else:
    # Prompt user for the volume number if not provided as argument
    while True:
        try:
            volume_number = int(input("Enter a Supreme Court volume number (502-605): "))
            if 502 <= volume_number <= 605:
                break
            else:
                print("Volume number must be between 502 and 605. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Extract links from the Supreme Court cases page
base_url = f"https://supreme.justia.com/cases/federal/us/{volume_number}/"
print(f"Fetching cases from volume {volume_number}...")
unique_case_links = extract_case_links(base_url, volume_number)

# Display the found links and constructed PDF URLs
case_urls = []
for link in unique_case_links:
    pdf_url = construct_pdf_url(link)
    case_urls.append(pdf_url)
    print(f"Case Link: {link} → PDF URL: {pdf_url}")

# Optional: Save these URLs to our database
print("\nWould you like to download these PDFs and save them to the database?")
auto_accept = False
if len(sys.argv) > 3 and sys.argv[3].lower() == 'auto':
    auto_accept = True
    download_choice = 'yes'
else:
    download_choice = input("Type 'yes' to proceed: ").strip().lower()
if download_choice == 'yes':
    # run the script downloader.py with the URL as the first arg
    import downloader  # Assuming downloader.py is in the same directory
    for pdf_url in case_urls:
        try:
            print(f"Downloading PDF from {pdf_url}...")
            pdf_file = downloader.download_pdf(pdf_url)
            text = downloader.extract_text_from_pdf(pdf_file)
            downloader.save_text_to_db(text, pdf_url)
            print("PDF text saved to database successfully.")
        except Exception as e:
            print(f"An error occurred while processing {pdf_url}: {e}")
        # Sleep for 5 seconds to avoid overwhelming the server
        import time
        time.sleep(5)
    # okay so on the base_url, not the case.pdf, there is info we want to curl and extract
    # Docket No.
    # Granted:
    # Argued:
    # Decided:




else:
    print("Download skipped. Exiting script.")
# End of supreme_justia.py
