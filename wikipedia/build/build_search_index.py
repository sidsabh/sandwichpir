#!/usr/bin/env python3
"""
Step 3: Build a sorted, brotli-compressed TSV search index for the browser.

Reads index.json (from build_wiki_db.py), sorts by title (case-insensitive),
outputs a TSV (title\\trow\\toffset per line), brotli-compresses it.

The browser loads this sorted TSV and uses binary search for instant prefix matching.

Output: data/index.tsv.br

Requires: pip install brotli
"""

import brotli
import json
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "server", "data")
SRC = os.path.join(DATA_DIR, "index.json")
DST = os.path.join(DATA_DIR, "index.tsv.br")

print(f"Reading {SRC}...")
t0 = time.time()
with open(SRC, "r") as f:
    entries = json.load(f)
print(f"  {len(entries):,} entries in {time.time()-t0:.1f}s")

print("Sorting by title (case-insensitive)...")
t0 = time.time()
entries.sort(key=lambda e: e["title"].lower())
print(f"  Sorted in {time.time()-t0:.1f}s")

print("Building TSV...")
t0 = time.time()
tsv = "\n".join(f"{e['title']}\t{e['row']}\t{e['offset']}" for e in entries).encode("utf-8")
print(f"  {len(tsv)/1e6:.1f} MB raw in {time.time()-t0:.1f}s")

print("Compressing (brotli quality=9)...")
t0 = time.time()
compressed = brotli.compress(tsv, quality=9)
print(f"  {len(compressed)/1e6:.1f} MB compressed in {time.time()-t0:.1f}s")

with open(DST, "wb") as f:
    f.write(compressed)
print(f"\nWrote {DST} ({len(compressed)/1e6:.1f} MB)")
