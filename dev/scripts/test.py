"""
Example: Loading COCA-style corpus with getout_of_text_3 (got3)

Replicates the notebook steps:
  - import package
  - load corpus from ../coca-text/
  - inspect top-level genres and sample keys
  - safely access a few known IDs (if present)
"""

import os
import argparse
import json
import getout_of_text_3 as got3


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

    if show_samples:
        print("\nSample retrieval attempts (only printed if present):")
        samples = [
            ("fic", 2012, "4120126"),
            ("web", "01", "5027201"),
            ("tvm", 2018, "5247077"),
            ("tvm", 2017, "5209288"),
        ]
        for genre, mid_key, text_id in samples:
            try:
                text = corpus.get(genre, {}).get(mid_key, {}).get(text_id)
                if text:
                    print(f"[FOUND] {genre} / {mid_key} / {text_id}: {text[:80]}...")
                else:
                    print(f"[MISS ] {genre} / {mid_key} / {text_id}")
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