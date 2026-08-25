#!/bin/bash
set -e

cd "$(dirname "$0")"

pip install --quiet datasets brotli tqdm

echo "=== Step 1: Download Wikipedia ==="
python download_wiki_raw.py

echo "=== Step 2: Build PIR database ==="
python build_wiki_db.py

echo "=== Step 3: Build search index ==="
python build_search_index.py

echo ""
echo "=== Done ==="
ls -lh ../server/data/wikipedia.bin ../server/data/index.tsv.br
