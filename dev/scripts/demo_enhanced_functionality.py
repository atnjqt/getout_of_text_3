#!/usr/bin/env python3
"""
Enhanced getout_of_text_3 Functionality Demo
============================================

This script demonstrates the new functionality for legal scholars working with COCA corpora.
The enhancements support open science research in legal scholarship by providing:

1. Easy corpus loading and management
2. Keyword search with context
3. Collocate analysis
4. Frequency analysis across genres

Usage:
    python demo_enhanced_functionality.py
"""

import getout_of_text_3 as got3
import os

def main():
    print("🎯 Enhanced getout_of_text_3 Functionality Demo")
    print("=" * 60)
    print(f"Version: {got3.__version__}")
    print()
    
    # Method 1: Using the convenience functions (as requested in requirements)
    print("📋 METHOD 1: Using convenience functions")
    print("-" * 40)
    
    # Check if sample data exists
    coca_text_dir = "coca-samples-text"
    if os.path.exists(coca_text_dir):
        print(f"✅ Found COCA sample data in {coca_text_dir}")
        
        # 1. Read the database files
        print("\n1️⃣ Reading corpus files...")
        corpus_data = got3.read_corpora(coca_text_dir, "coca_sample")
        
        if corpus_data:
            # 2. Perform keyword search
            print("\n2️⃣ Performing keyword search...")
            search_results = got3.search_keyword_corpus(
                "legal", 
                corpus_data, 
                case_sensitive=False, 
                show_context=True, 
                context_words=3
            )
            
            # 3. Perform collocate analysis
            print("\n3️⃣ Performing collocate analysis...")
            collocate_results = got3.find_collocates(
                "legal", 
                corpus_data, 
                window_size=3, 
                min_freq=1, 
                case_sensitive=False
            )
            
            # 4. Perform frequency analysis
            print("\n4️⃣ Performing frequency analysis...")
            freq_results = got3.keyword_frequency_analysis(
                "legal", 
                corpus_data, 
                case_sensitive=False
            )
    else:
        print(f"❌ COCA sample data not found at {coca_text_dir}")
        print("   Please ensure the coca-samples-text directory exists with sample data.")
    
    print("\n" + "=" * 60)
    
    # Method 2: Using the LegalCorpus class directly
    print("📋 METHOD 2: Using LegalCorpus class (Object-Oriented approach)")
    print("-" * 40)
    
    # Create a corpus instance
    corpus = got3.LegalCorpus()
    
    # Show available methods
    print("🔧 Available methods in LegalCorpus:")
    methods = [method for method in dir(corpus) if not method.startswith('_')]
    for method in methods:
        print(f"   - {method}()")
    
    # Example with the class-based approach
    if os.path.exists(coca_text_dir):
        print(f"\n📂 Loading corpus using LegalCorpus class...")
        corpus_data_oo = corpus.read_corpora(coca_text_dir, "coca_oo_demo")
        
        print(f"\n📊 Corpus summary:")
        corpus.corpus_summary()
        
        print(f"\n📋 Available corpora: {corpus.list_corpora()}")
    
    print("\n🎯 Demo completed!")
    print("\nFor legal scholars: This toolkit now provides comprehensive")
    print("COCA corpus analysis capabilities to support open science research!")

if __name__ == "__main__":
    main()
