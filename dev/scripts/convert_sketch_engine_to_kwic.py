#!/usr/bin/env python3
"""
Convert Sketch Engine CSV concordance exports to KWIC JSON format for Flask annotator app.

Usage:
    python convert_sketch_engine_to_kwic.py

This script:
1. Reads all CSV files from data/sketch-engine/ directory
2. Parses filename to extract: keyword, corpus name, and timestamp
3. Converts concordance format to KWIC format with bold markers
4. Exports to JSON compatible with the Flask annotator app
"""

import pandas as pd
import json
import os
import re
from pathlib import Path
from datetime import datetime

# Configure paths
SKETCH_ENGINE_DIR = Path(__file__).parent.parent / 'data' / 'sketch-engine'
EXPORTS_DIR = Path(__file__).parent.parent / 'examples' / 'public' / 'exports'

# Ensure exports directory exists
EXPORTS_DIR.mkdir(exist_ok=True, parents=True)


def parse_filename(filename):
    """
    Parse Sketch Engine filename to extract metadata.
    
    Format: concordance_{keyword}_{corpus}_{timestamp}.csv
    Examples:
        concordance_best_system_ecolexicon_en_20251115000951.csv
        concordance_best_system_oec_biwec3_20251117072438.csv
    
    Returns:
        dict with 'keyword', 'corpus', 'timestamp' keys
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Pattern: concordance_{keyword}_{corpus}_{timestamp}
    # keyword can be multiple words separated by underscores
    # corpus can be multiple parts separated by underscores
    # timestamp is always 14 digits at the end
    
    match = re.match(r'concordance_(.+)_(\d{14})$', name)
    
    if not match:
        print(f"⚠️  Warning: Could not parse filename: {filename}")
        return None
    
    keywords_and_corpus = match.group(1)
    timestamp = match.group(2)
    
    # Split by underscore and try to identify corpus
    # Common corpus names: ecolexicon_en, oec, biwec3, coca, etc.
    parts = keywords_and_corpus.split('_')
    
    # Known corpus patterns
    corpus_patterns = ['ecolexicon', 'oec', 'biwec', 'coca', 'coha', 'glowbe']
    
    # Find corpus by matching patterns
    corpus_idx = None
    for i, part in enumerate(parts):
        if any(pattern in part.lower() for pattern in corpus_patterns):
            corpus_idx = i
            break
    
    if corpus_idx is not None:
        keyword = '_'.join(parts[:corpus_idx])
        corpus = '_'.join(parts[corpus_idx:])
    else:
        # If no known corpus found, assume last part is corpus
        keyword = '_'.join(parts[:-1])
        corpus = parts[-1]
    
    return {
        'keyword': keyword,
        'corpus': corpus,
        'timestamp': timestamp
    }


def convert_csv_to_kwic(csv_path, metadata):
    """
    Convert Sketch Engine CSV to KWIC JSON format.
    
    Args:
        csv_path: Path to CSV file
        metadata: Dict with 'keyword', 'corpus', 'timestamp'
    
    Returns:
        Dict in KWIC format: {genre_key: [items]}
    """
    # Read CSV, skipping first 4 rows of metadata
    df = pd.read_csv(csv_path, skiprows=4)
    
    # Extract genre from corpus name (e.g., 'ecolexicon_en' -> 'ecolexicon', 'oec' -> 'oec')
    genre = metadata['corpus'].split('_')[0]
    
    # Build KWIC items
    kwic_items = []
    
    for idx, row in df.iterrows():
        # Get concordance components
        left = str(row.get('Left', '')).strip()
        kwic = str(row.get('KWIC', '')).strip()
        right = str(row.get('Right', '')).strip()
        reference = str(row.get('Reference', '')).strip()
        
        # Clean Sketch Engine markup tags (sentence boundaries, etc.)
        # Remove <s>, </s>, <g/>, and other common tags
        left = re.sub(r'</?s>|<g/>', ' ', left).strip()
        kwic = re.sub(r'</?s>|<g/>', ' ', kwic).strip()
        right = re.sub(r'</?s>|<g/>', ' ', right).strip()
        
        # Normalize multiple spaces
        left = re.sub(r'\s+', ' ', left)
        kwic = re.sub(r'\s+', ' ', kwic)
        right = re.sub(r'\s+', ' ', right)
        
        # Create context with bold markers around keyword
        context = f"{left} **{kwic}** {right}"
        
        # Create full text (without markers)
        full_text = f"{left} {kwic} {right}"
        
        # Create KWIC item
        item = {
            'text_id': reference if reference and reference != 'nan' else f"{genre}_{idx}",
            'match': kwic,
            'context': context,
            'full_text': full_text
        }
        
        kwic_items.append(item)
    
    # Create genre key (format: genre_keyword)
    genre_key = f"{genre}_{metadata['keyword']}"
    
    return {genre_key: kwic_items}


def main():
    """Process all Sketch Engine CSV files."""
    
    print("=" * 80)
    print("Sketch Engine CSV to KWIC JSON Converter")
    print("=" * 80)
    print()
    
    if not SKETCH_ENGINE_DIR.exists():
        print(f"❌ Error: Sketch Engine directory not found: {SKETCH_ENGINE_DIR}")
        return
    
    # Find all CSV files
    csv_files = list(SKETCH_ENGINE_DIR.glob('*.csv'))
    
    if not csv_files:
        print(f"⚠️  No CSV files found in: {SKETCH_ENGINE_DIR}")
        return
    
    print(f"📁 Found {len(csv_files)} CSV file(s) in {SKETCH_ENGINE_DIR}")
    print()
    
    # Process each CSV file
    for csv_path in csv_files:
        print(f"Processing: {csv_path.name}")
        
        # Parse filename
        metadata = parse_filename(csv_path.name)
        if not metadata:
            print(f"   ⏭️  Skipping (could not parse filename)")
            print()
            continue
        
        print(f"   Keyword: {metadata['keyword']}")
        print(f"   Corpus: {metadata['corpus']}")
        print(f"   Timestamp: {metadata['timestamp']}")
        
        try:
            # Convert to KWIC format
            kwic_data = convert_csv_to_kwic(csv_path, metadata)
            
            # Count items
            total_items = sum(len(items) for items in kwic_data.values())
            print(f"   Concordance lines: {total_items}")
            
            # Generate output filename
            output_filename = f"kwic_{metadata['keyword']}_{metadata['corpus']}.json"
            output_path = EXPORTS_DIR / output_filename
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(kwic_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Saved to: {output_path.name}")
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
        
        print()
    
    print("=" * 80)
    print("Conversion complete!")
    print(f"📂 Output directory: {EXPORTS_DIR}")
    print()
    
    # List generated files
    kwic_files = list(EXPORTS_DIR.glob('kwic_*.json'))
    if kwic_files:
        print(f"Generated {len(kwic_files)} KWIC JSON file(s):")
        for f in sorted(kwic_files):
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
