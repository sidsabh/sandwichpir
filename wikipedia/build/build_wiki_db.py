#!/usr/bin/env python3
"""
Step 2: Build a flat binary PIR database from cached raw Wikipedia articles.

Run download_wiki_raw.py first.

Each row is ROW_SIZE bytes. Articles are brotli-5 compressed and bin-packed
sequentially into rows. Each entry: [4-byte u32 length][compressed bytes].
Rows are zero-padded to exactly ROW_SIZE.

Output:
  data/wikipedia.bin  — flat binary DB (num_rows * ROW_SIZE bytes)
  data/index.json     — mapping: [{title, row, offset}, ...]

Requires: pip install brotli tqdm
"""

import os
import json
import pickle
import struct
import brotli
from tqdm import tqdm

ROW_SIZE = 128 * 1024  # 128 KB per row
HEADER_BYTES = 4       # u32 length prefix per article

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "server", "data")
CACHE_FILE = os.path.join(DATA_DIR, "articles_raw.pkl")
DB_FILE = os.path.join(DATA_DIR, "wikipedia.bin")
INDEX_FILE = os.path.join(DATA_DIR, "index.json")

print("Loading cached articles...")
with open(CACHE_FILE, "rb") as f:
    articles = pickle.load(f)
print(f"Loaded {len(articles):,} articles")

index = []
current_row = bytearray(ROW_SIZE)
current_offset = 0
num_rows = 0
num_truncated = 0

f_db = open(DB_FILE, "wb")
for title, text in tqdm(articles, desc="Building DB"):
    compressed = brotli.compress(text.encode("utf-8"), quality=5)
    entry_size = HEADER_BYTES + len(compressed)

    if entry_size > ROW_SIZE:
        orig_size = len(compressed)
        compressed = compressed[:ROW_SIZE - HEADER_BYTES]
        entry_size = ROW_SIZE
        num_truncated += 1

    if current_offset + entry_size > ROW_SIZE:
        f_db.write(current_row)
        num_rows += 1
        current_row = bytearray(ROW_SIZE)
        current_offset = 0

    struct.pack_into("<I", current_row, current_offset, len(compressed))
    current_row[current_offset + HEADER_BYTES:current_offset + entry_size] = compressed
    index.append({"title": title, "row": num_rows, "offset": current_offset})
    current_offset += entry_size

if current_offset > 0:
    f_db.write(current_row)
    num_rows += 1

f_db.close()

with open(INDEX_FILE, "w") as f:
    json.dump(index, f)

file_size = os.path.getsize(DB_FILE)
print(f"\nDone!")
print(f"  Articles:   {len(articles):,}")
print(f"  Truncated:  {num_truncated:,}")
print(f"  Rows:       {num_rows:,}")
print(f"  Row size:   {ROW_SIZE // 1024} KB")
print(f"  DB file:    {file_size / (1024**3):.2f} GB")
print(f"  Index file: {os.path.getsize(INDEX_FILE) / (1024**2):.1f} MB")
