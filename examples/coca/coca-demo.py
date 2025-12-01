"""
Example: Loading COCA-style corpus with getout_of_text_3 (got3)

Replicates the notebook steps:
  - import package
  - load corpus from ../coca-text/
  - inspect top-level genres and sample keys
  - safely access a few known IDs (if present)
  - test DataFrame structure and search functionality
"""

import os
import argparse
import json
import getout_of_text_3 as got3
import pandas as pd

def main(coca_path: str, show_samples: bool):
    print(f"getout_of_text_3 version: {got3.__version__}")
    if not os.path.isdir(coca_path):
        raise SystemExit(f"Corpus directory not found: {coca_path}")

    print(f"Loading corpus from: {coca_path}")
    corpus = got3.read_corpus(coca_path)

    # Top-level genres
    genres = list(corpus.keys())
    print(f"Genres ({len(genres)}): {genres}")

    # For each genre show up to 5 second-level keys (years or file nums)
    preview = {}
    for g in genres:
        second_level = list(corpus[g].keys())
        preview[g] = second_level[:5]
    print("Second-level key preview (first 5 each):")
    print(json.dumps(preview, indent=2))

    # Test DataFrame structure
    print("\nTesting DataFrame structure for each genre/year (showing shape):")
    for g in genres[:3]:  # limit output for demo
        for mid_key in list(corpus[g].keys())[:2]:
            df = corpus[g][mid_key]
            print(f"{g} / {mid_key}: type={type(df)}, shape={df.shape}")

    # Test keyword search (if available)
    if hasattr(got3, "search_keyword_corpus"):
        print("\nTesting keyword search for 'law' in first genre/year:")
        first_genre = genres[0]
        first_year = list(corpus[first_genre].keys())[0]
        df = corpus[first_genre][first_year]
        # Use the search function if it supports DataFrame input
        try:
            # If search_keyword_corpus expects the nested dict, pass {genre: {year: df}}
            results = got3.search_keyword_corpus(
                keyword="law",
                db_dict={first_genre: {first_year: df}},
                case_sensitive=False,
                show_context=True,
                context_words=5,
                output='print'
            )
        except Exception as e:
            print(f"Keyword search failed: {e}")
    else:
        print("search_keyword_corpus not found in got3.")

    if show_samples:
        print("\nSample retrieval attempts (only printed if present):")
        samples = [
            ("fic", "2012", "4120126"),
            ("web", "01", "5027201"),
            ("tvm", "2018", "5247077"),
            ("tvm", "2017", "5209288"),
        ]
        for genre, mid_key, text_id in samples:
            try:
                # DataFrame lookup
                df = corpus.get(genre, {}).get(mid_key)
                if isinstance(df, pd.DataFrame):
                    row = df[df['text_id'] == text_id]
                    if not row.empty:
                        text = row.iloc[0]['text']
                        print(f"[FOUND] {genre} / {mid_key} / {text_id}: {text[:80]}...")
                    else:
                        print(f"[MISS ] {genre} / {mid_key} / {text_id}")
                else:
                    print(f"[MISS ] {genre} / {mid_key} / {text_id} (no DataFrame)")
            except Exception as e:
                print(f"[ERROR] {genre} / {mid_key} / {text_id} -> {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect COCA corpus via getout_of_text_3.")
    parser.add_argument(
        "--coca-path",
        default="../coca-text/",
        help="Path to root COCA text directory (default: ../coca-text/)"
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip sample ID lookups."
    )
    args = parser.parse_args()
    main(args.coca_path, show_samples=not args.no_samples)