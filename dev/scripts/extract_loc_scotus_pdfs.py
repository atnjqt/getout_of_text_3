#!/usr/bin/env python3
"""
Extract and process PDF content from Supreme Court documents

This script reads all PDFs from the scotus_pdfs directory and extracts their text content
into a pandas DataFrame. It uses multiprocessing for efficient parallel processing.

Based on excel spreadsheet of cases from penn state supreme court database (https://scdb.la.psu.edu/) 1946-2024
Items downloaded from loc.gov 
"""

import pandas as pd
import os
from PyPDF2 import PdfReader
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def read_pdf(file_path):
    """
    Extract and clean text content from a PDF file
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Cleaned text content
    """
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Fix hyphenated line breaks (e.g., "ordinary mean-\ning" -> "ordinary meaning")
        text = re.sub(r'-\n', '', text)
        
        # Clean up other common PDF extraction issues
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with single space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        
        # Additional cleaning for legal documents
        text = re.sub(r'\f', ' ', text)  # Remove form feeds
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', ' ', text)  # Keep only common punctuation
        
        text = text.strip()
        
        return text

def process_pdf(filename):
    """
    Process a single PDF file and return filename and content
    
    Args:
        filename (str): Name of the PDF file
        
    Returns:
        dict: Dictionary with filename and content
    """
    file_path = os.path.join("scotus_pdfs", filename)
    content = read_pdf(file_path)
    return {"filename": filename, "content": content}

def main():
    """Main function to process all PDFs and create DataFrame"""
    
    # Check if scotus_pdfs directory exists
    if not os.path.exists("scotus_pdfs"):
        print("Error: scotus_pdfs directory not found!")
        print("Please make sure you're running this script from the correct directory.")
        return
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir("scotus_pdfs") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in scotus_pdfs directory!")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Use multiprocessing to process PDFs in parallel
    n_cores = min(cpu_count() - 1, len(pdf_files))  # Use all cores except one, or fewer if we have fewer files
    print(f"Using {n_cores} CPU cores for processing")
    
    # Process PDFs in parallel with progress bar
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_pdf, pdf_files), 
            total=len(pdf_files),
            desc="Processing PDFs"
        ))
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate total tokens
    total_tokens = df["content"].apply(lambda x: len(x.split())).sum()
    print(f"\nProcessing complete!")
    print(f"Total documents processed: {len(df)}")
    print(f"Total tokens (approximate by word count): {total_tokens:,}")
    
    # Save to CSV for easy access
    output_file = "scotus_pdfs_content.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Save to pickle for preservation of data types
    pickle_file = "scotus_pdfs_content.pkl"
    df.to_pickle(pickle_file)
    print(f"Results also saved to: {pickle_file}")
    
    return df

if __name__ == "__main__":
    df = main()