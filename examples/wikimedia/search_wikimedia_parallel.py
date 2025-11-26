#!/usr/bin/env python3
"""
Parallel search over Wikimedia datasets using multiprocessing.
This script splits the dataset across CPU cores for faster searching.
"""

import os
import re
import time
import argparse
from multiprocessing import Pool, cpu_count, Manager
from datasets import load_from_disk
from tqdm.auto import tqdm


# Field candidates to search for text content (in priority order)
FIELD_CANDIDATES = ['text', 'title', 'content']


def _extract_text_from_row(row):
    """Extract text content from a dataset row."""
    for f in FIELD_CANDIDATES:
        if f in row and row[f]:
            return row[f]
    parts = []
    for k, v in row.items():
        if isinstance(v, str) and v:
            parts.append(v)
    return "\n".join(parts)


def search_chunk(args):
    """
    Search a chunk of the dataset (worker function for multiprocessing).
    
    Args:
        args: Tuple of (dataset_path, start_idx, end_idx, keyword, context_window, case_insensitive, worker_id)
    
    Returns:
        List of match dictionaries
    """
    dataset_path, start_idx, end_idx, keyword, context_window, case_insensitive, worker_id = args
    
    # Load dataset in this worker process
    dataset_dict = load_from_disk(dataset_path)
    dataset = dataset_dict['train'] if 'train' in dataset_dict else dataset_dict[list(dataset_dict.keys())[0]]
    
    flag = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(re.escape(keyword), flag)
    
    matches = []
    
    # Process only the assigned chunk
    for idx in range(start_idx, min(end_idx, len(dataset))):
        row = dataset[idx]
        text = _extract_text_from_row(row)
        
        if not text:
            continue
            
        m = pattern.search(text)
        if m:
            start, end = m.span()
            
            # Split text into words and find word boundaries around the match
            words = text.split()
            char_count = 0
            match_word_idx = 0
            for word_idx, word in enumerate(words):
                if char_count + len(word) >= start:
                    match_word_idx = word_idx
                    break
                char_count += len(word) + 1  # +1 for space
            
            # Get context window of words
            start_word_idx = max(0, match_word_idx - context_window)
            end_word_idx = min(len(words), match_word_idx + context_window + 1)
            snippet = ' '.join(words[start_word_idx:end_word_idx])
            
            # Highlight the match
            hit_text = text[start:end]
            highlight = snippet.replace(hit_text, f"**{hit_text}**")
            
            # Store match with metadata
            matches.append({
                'id': row.get('id', None),
                'title': row.get('title', ''),
                'url': row.get('url', ''),
                'language': row.get('language', ''),
                'source': row.get('source', ''),
                'highlight': highlight,
                'dataset_idx': idx
            })
    
    return matches


def search_parallel(dataset_path, keyword, max_results=None, context_window=30, 
                   case_insensitive=True, num_workers=None):
    """
    Search dataset in parallel across multiple CPU cores.
    
    Args:
        dataset_path: Path to the saved dataset directory
        keyword: Search keyword
        max_results: Maximum number of results to return (None for all)
        context_window: Number of words before/after match to show
        case_insensitive: Whether search is case insensitive
        num_workers: Number of worker processes (default: cpu_count - 1)
    
    Returns:
        List of match dictionaries
    """
    start_time = time.time()
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Loading dataset from: {dataset_path}")
    dataset_dict = load_from_disk(dataset_path)
    dataset = dataset_dict['train'] if 'train' in dataset_dict else dataset_dict[list(dataset_dict.keys())[0]]
    total_rows = len(dataset)
    
    print(f"Dataset size: {total_rows:,} rows")
    print(f"Using {num_workers} worker processes")
    print(f"Searching for: '{keyword}'\n")
    
    # Calculate chunk sizes for each worker
    chunk_size = (total_rows + num_workers - 1) // num_workers
    chunks = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        if start_idx < total_rows:
            chunks.append((dataset_path, start_idx, end_idx, keyword, 
                          context_window, case_insensitive, i))
    
    # Process chunks in parallel
    all_matches = []
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance and progress tracking
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            for chunk_matches in pool.imap_unordered(search_chunk, chunks):
                all_matches.extend(chunk_matches)
                pbar.set_postfix({"found": len(all_matches)})
                pbar.update(1)
                
                # Early exit if we've found enough results
                if max_results and len(all_matches) >= max_results:
                    pool.terminate()
                    break
    
    # Sort by dataset index to maintain order
    all_matches.sort(key=lambda x: x['dataset_idx'])
    
    # Limit to max_results
    if max_results:
        all_matches = all_matches[:max_results]
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*100}")
    print(f"Search complete!")
    print(f"Found {len(all_matches)} matches in {elapsed:.2f}s")
    print(f"Speed: {total_rows/elapsed:,.0f} rows/sec")
    print(f"{'='*100}\n")
    
    return all_matches


def print_matches(matches):
    """Print matches in a formatted way."""
    for i, match in enumerate(matches, 1):
        print(f"[{i}] id={match['id']} | title={match['title']} | url={match['url']}")
        print(f"    language={match['language']} | source={match['source']}")
        print(f"{match['highlight']}")
        print(f"{'-'*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Search Wikimedia datasets in parallel across CPU cores'
    )
    parser.add_argument('--dataset-dir', required=True, 
                       help='Path to saved dataset directory (e.g., ./wikimedia_en)')
    parser.add_argument('--keyword', required=True,
                       help='Keyword to search for')
    parser.add_argument('--max-results', type=int, default=10,
                       help='Maximum number of results (default: 10)')
    parser.add_argument('--context-window', type=int, default=30,
                       help='Number of words before/after match (default: 30)')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='Make search case sensitive (default: case insensitive)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    # Run parallel search
    matches = search_parallel(
        dataset_path=args.dataset_dir,
        keyword=args.keyword,
        max_results=args.max_results,
        context_window=args.context_window,
        case_insensitive=not args.case_sensitive,
        num_workers=args.workers
    )
    
    # Print results
    print_matches(matches)
    
    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
