#!/usr/bin/env python3
"""
Step 1: Download all of English Wikipedia as raw text.

Output: data/articles_raw.pkl — list of (title, text) tuples

Requires: pip install datasets tqdm
"""

import os
import pickle
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "server", "data")
CACHE_FILE = os.path.join(OUT_DIR, "articles_raw.pkl")

os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(CACHE_FILE):
    print(f"Cache already exists: {CACHE_FILE} ({os.path.getsize(CACHE_FILE) / (1024**3):.2f} GB)")
    print("Delete it to re-download.")
else:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    articles = []
    for ex in tqdm(ds, desc="Downloading Wikipedia"):
        articles.append((ex["title"], ex["text"]))
    print(f"\nDownloaded {len(articles):,} articles. Saving cache...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(articles, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {CACHE_FILE} ({os.path.getsize(CACHE_FILE) / (1024**3):.2f} GB)")
