# simple python script to download a PDF and read the text into a database

import requests
import sqlite3
from io import BytesIO
from PyPDF2 import PdfReader
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download PDF: {response.status_code}")
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)
def save_text_to_db(text, db_name='pdf_texts.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        )
    ''')
    cursor.execute('INSERT INTO pdf_texts (text) VALUES (?)', (text,))
    conn.commit()
    conn.close()
def main():
    url = input("Enter the PDF URL: ")
    try:
        pdf_file = download_pdf(url)
        text = extract_text_from_pdf(pdf_file)
        save_text_to_db(text)
        print("PDF text saved to database successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
# This script will prompt the user for a PDF URL, download the PDF,
# extract the text, and save it to a SQLite database.
# Make sure to install the required libraries:

