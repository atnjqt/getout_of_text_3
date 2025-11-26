"""Search the locally saved OpenLLM-France Wikimedia Arrow shards.

This script streams the dataset using `datasets.load_from_disk` and the
dataset iterator to avoid loading everything into memory. It searches
for a keyword across common text fields and returns contextual snippets.

Usage (CLI):
  python examples/wikimedia/search_wikimedia.py --dataset-dir ./examples/wikimedia/wikimedia_en \
      --keyword bank --max-results 10 --context 200

You can also import `search_dataset` from this module in notebooks.
"""
from __future__ import annotations

import argparse
import re
import threading
import queue
from typing import Dict, Iterable, List, Optional
import os
import multiprocessing
from typing import Tuple

from datasets import load_from_disk, Dataset, DatasetDict


COMMON_TEXT_FIELDS = [
    "text",
    "content",
    "page_content",
    "article",
    "body",
    "document",
    "body_text",
]


def _extract_text_from_row(row: Dict) -> str:
    """Try to extract a single text string from a dataset row dict.

    We attempt common fields and fall back to concatenating all string
    values in the row.
    """
    for f in COMMON_TEXT_FIELDS:
        if f in row and isinstance(row[f], str):
            return row[f]

    # Fallback: join any string fields
    parts: List[str] = []
    for v in row.values():
        if isinstance(v, str):
            parts.append(v)
    return "\n".join(parts)


def search_dataset(
    dataset_dir: str,
    keyword: str,
    max_results: int = 50,
    context: int = 200,
    case_insensitive: bool = True,
    batch_size: int = 1024,
    field: Optional[str] = None,
) -> Iterable[Dict]:
    """Stream through the saved dataset and yield matches as dicts.

    Yields items with keys: index (int), split (str), snippet (str), full_text (optional).
    """
    if case_insensitive:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(keyword))

    ds_obj = load_from_disk(dataset_dir)

    # If the loaded object is a DatasetDict, pick the first split available
    if isinstance(ds_obj, DatasetDict):
        split_name = list(ds_obj.keys())[0]
        ds: Dataset = ds_obj[split_name]
    else:
        ds = ds_obj  # type: ignore
        split_name = getattr(ds, "info", None) and getattr(ds.info, "split", "train") or "train"

    returned = 0
    global_idx = 0

    # Iterate in batches to keep memory usage low
    for batch in ds.iter(batch_size=batch_size):
        # batch is a dict of lists
        length = len(next(iter(batch.values()))) if batch else 0
        for i in range(length):
            # build a row dict for extraction
            row = {k: batch[k][i] for k in batch.keys()}

            text = ""
            if field and field in row and isinstance(row[field], str):
                text = row[field]
            else:
                text = _extract_text_from_row(row)

            if not text:
                global_idx += 1
                continue

            m = pattern.search(text)
            if m:
                start, end = m.span()
                # Build snippet with context, avoid slicing out of bounds
                s0 = max(0, start - context)
                s1 = min(len(text), end + context)
                prefix = "..." if s0 > 0 else ""
                suffix = "..." if s1 < len(text) else ""
                snippet = prefix + text[s0:s1].replace("\n", " ") + suffix

                yield {
                    "index": global_idx,
                    "split": split_name,
                    "snippet": snippet,
                    "match_start": start,
                    "match_end": end,
                }

                returned += 1
                if returned >= max_results:
                    return

            global_idx += 1


def search_dataset_multithreaded(
    dataset_dir: str,
    keyword: str,
    max_results: int = 50,
    context: int = 200,
    case_insensitive: bool = True,
    batch_size: int = 1024,
    field: Optional[str] = None,
    workers: int = 4,
    max_queue_size: int = 32,
) -> Iterable[Dict]:
    """Multithreaded search using a producer-consumer queue.

    The main thread streams batches from the dataset and enqueues them for
    worker threads which perform the search. Results are pushed into an
    output queue and yielded by the caller.
    """
    if case_insensitive:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(keyword))

    ds_obj = load_from_disk(dataset_dir)

    if isinstance(ds_obj, DatasetDict):
        split_name = list(ds_obj.keys())[0]
        ds: Dataset = ds_obj[split_name]
    else:
        ds = ds_obj  # type: ignore
        split_name = getattr(ds, "info", None) and getattr(ds.info, "split", "train") or "train"

    in_q: "queue.Queue" = queue.Queue(maxsize=max_queue_size)
    out_q: "queue.Queue" = queue.Queue()
    stop_event = threading.Event()
    counter_lock = threading.Lock()
    counter = {"value": 0}

    def worker_loop():
        while True:
            item = in_q.get()
            if item is None:
                in_q.task_done()
                break

            batch, start_idx = item
            try:
                length = len(next(iter(batch.values()))) if batch else 0
                for i in range(length):
                    # Early stop check
                    if stop_event.is_set():
                        break

                    row = {k: batch[k][i] for k in batch.keys()}
                    text = ""
                    if field and field in row and isinstance(row[field], str):
                        text = row[field]
                    else:
                        text = _extract_text_from_row(row)

                    if not text:
                        continue

                    m = pattern.search(text)
                    if m:
                        start, end = m.span()
                        s0 = max(0, start - context)
                        s1 = min(len(text), end + context)
                        prefix = "..." if s0 > 0 else ""
                        suffix = "..." if s1 < len(text) else ""
                        snippet = prefix + text[s0:s1].replace("\n", " ") + suffix

                        hit = {
                            "index": start_idx + i,
                            "split": split_name,
                            "snippet": snippet,
                            "match_start": start,
                            "match_end": end,
                        }
                        out_q.put(hit)

                        with counter_lock:
                            counter["value"] += 1
                            if counter["value"] >= max_results:
                                stop_event.set()
                                break
            finally:
                in_q.task_done()

    # Start workers
    threads: List[threading.Thread] = []
    for _ in range(max(1, workers)):
        t = threading.Thread(target=worker_loop, daemon=True)
        t.start()
        threads.append(t)

    # Producer: enqueue batches with starting global index
    global_idx = 0
    try:
        for batch in ds.iter(batch_size=batch_size):
            if stop_event.is_set():
                break

            length = len(next(iter(batch.values()))) if batch else 0
            in_q.put((batch, global_idx))
            global_idx += length

        # Signal workers to stop
        for _ in threads:
            in_q.put(None)

        # Yield results as they appear until we've reached max_results
        yielded = 0
        while True:
            try:
                hit = out_q.get(timeout=0.1)
                yield hit
                yielded += 1
                if yielded >= max_results:
                    break
            except queue.Empty:
                # Check if all workers finished and queues drained
                if all(not t.is_alive() for t in threads) and out_q.empty():
                    break
                if stop_event.is_set() and out_q.empty():
                    break
    finally:
        # Ensure threads are cleaned up
        stop_event.set()
        # Drain in_q to unblock workers
        while not in_q.empty():
            try:
                in_q.get_nowait()
                in_q.task_done()
            except queue.Empty:
                break
        for t in threads:
            t.join(timeout=1.0)


def _process_batch_mp(args: Tuple) -> List[Dict]:
    """Top-level helper for multiprocessing: process one batch and return list of hits.

    args: (batch, start_idx, keyword, context, case_insensitive, field, split_name)
    """
    batch, start_idx, keyword, context, case_insensitive, field, split_name = args
    import re

    flags = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(re.escape(keyword), flags)

    results: List[Dict] = []
    length = len(next(iter(batch.values()))) if batch else 0
    for i in range(length):
        # local extraction (avoids relying on outer functions)
        row = {k: batch[k][i] for k in batch.keys()}

        text = ""
        if field and field in row and isinstance(row[field], str):
            text = row[field]
        else:
            for f in COMMON_TEXT_FIELDS:
                if f in row and isinstance(row[f], str):
                    text = row[f]
                    break
            if not text:
                parts: List[str] = []
                for v in row.values():
                    if isinstance(v, str):
                        parts.append(v)
                text = "\n".join(parts)

        if not text:
            continue

        m = pattern.search(text)
        if m:
            start, end = m.span()
            s0 = max(0, start - context)
            s1 = min(len(text), end + context)
            prefix = "..." if s0 > 0 else ""
            suffix = "..." if s1 < len(text) else ""
            snippet = prefix + text[s0:s1].replace("\n", " ") + suffix

            results.append(
                {
                    "index": start_idx + i,
                    "split": split_name,
                    "snippet": snippet,
                    "match_start": start,
                    "match_end": end,
                }
            )

    return results


def search_dataset_multiprocessed(
    dataset_dir: str,
    keyword: str,
    max_results: int = 50,
    context: int = 200,
    case_insensitive: bool = True,
    batch_size: int = 1024,
    field: Optional[str] = None,
    workers: int = 4,
):
    """Multiprocessed search: stream batches to a Pool and yield hits as they arrive.

    Uses multiprocessing Pool with spawn context (safer on macOS). Stops early when max_results reached.
    """
    ds_obj = load_from_disk(dataset_dir)
    if isinstance(ds_obj, DatasetDict):
        split_name = list(ds_obj.keys())[0]
        ds: Dataset = ds_obj[split_name]
    else:
        ds = ds_obj  # type: ignore
        split_name = getattr(ds, "info", None) and getattr(ds.info, "split", "train") or "train"

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=max(1, workers))

    def arg_generator():
        global_idx = 0
        for batch in ds.iter(batch_size=batch_size):
            length = len(next(iter(batch.values()))) if batch else 0
            yield (batch, global_idx, keyword, context, case_insensitive, field, split_name)
            global_idx += length

    yielded = 0
    try:
        for res_list in pool.imap_unordered(_process_batch_mp, arg_generator()):
            for hit in res_list:
                yield hit
                yielded += 1
                if yielded >= max_results:
                    pool.terminate()
                    pool.join()
                    return
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    finally:
        if pool:
            pool.close()
            pool.join()


def _parse_args():
    p = argparse.ArgumentParser(description="Search saved Wikimedia Arrow shards for a keyword")
    p.add_argument("--dataset-dir", required=True, help="Path to the saved dataset directory (e.g. ./wikimedia_en)")
    p.add_argument("--keyword", required=True, help="Keyword to search for (plain text)")
    p.add_argument("--max-results", type=int, default=20)
    p.add_argument("--context", type=int, default=200, help="Characters of context around the match")
    p.add_argument("--case-insensitive", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--field", default=None, help="If dataset has a known text field, provide it to speed up extraction")
    p.add_argument("--workers", type=int, default=None, nargs="?", help="Number of worker threads to use (default: CPU count - 1)")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        # Compute default workers if not provided: CPU count - 1 (at least 1)
        if args.workers is None:
            cpu = os.cpu_count() or 1
            workers = max(1, cpu - 1)
        else:
            workers = max(1, int(args.workers))

        if workers and workers > 1:
            # Use multiprocessing for better CPU utilization
            iterator = search_dataset_multiprocessed(
                args.dataset_dir,
                args.keyword,
                max_results=args.max_results,
                context=args.context,
                case_insensitive=args.case_insensitive,
                batch_size=args.batch_size,
                field=args.field,
                workers=workers,
            )
        else:
            iterator = search_dataset(
                args.dataset_dir,
                args.keyword,
                max_results=args.max_results,
                context=args.context,
                case_insensitive=args.case_insensitive,
                batch_size=args.batch_size,
                field=args.field,
            )

        for n, hit in enumerate(iterator):
            print(f"[{n+1}] index={hit['index']} split={hit['split']}")
            print(hit["snippet"])
            print("-" * 80)
    except KeyboardInterrupt:
        print("Interrupted, exiting.")


if __name__ == "__main__":
    main()
