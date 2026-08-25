#!/bin/bash
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

DATA_DIR="wikipedia/server/data"
NUM_ITEMS="${1:-65536}"
ITEM_BITS="${2:-1048576}"
PORT="${3:-8088}"
# Shift consumed positionals so "$@" carries any extra args verbatim.
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift

MAX_BATCH="${MAX_BATCH:-64}"
BATCH_TIMEOUT_MS="${BATCH_TIMEOUT_MS:-32}"

DB_PATH="$DATA_DIR/wikipedia.bin"

if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    echo ""
    echo "Place wikipedia.bin and index.json.br in $DATA_DIR/"
    echo ""
    echo "Usage: $0 [num_items] [item_size_bits] [port]"
    exit 1
fi

if [ ! -f "$DATA_DIR/index.json.br" ]; then
    echo "Error: index.json.br not found in $DATA_DIR/"
    echo ""
    echo "Place index.json.br in $DATA_DIR/"
    exit 1
fi

ITEM_BYTES=$((ITEM_BITS / 8))
DB_SIZE=$((NUM_ITEMS * ITEM_BYTES))
DB_GB=$(python3 -c "print(f'{$DB_SIZE / 2**30:.1f}')" 2>/dev/null || echo "?")

echo "=== Private Wikipedia ==="
echo "Database: $DB_PATH ($NUM_ITEMS items x $ITEM_BYTES bytes = ${DB_GB} GB)"
echo "Index:    $DATA_DIR/index.json.br"
echo "Port:     $PORT"
echo ""

cd wikipedia/server

# Let trailing args override the env-var defaults without clap
# rejecting duplicates: only emit the env-var flag when the user
# hasn't already provided the same flag via "$@".
EXTRA_FLAGS=()
if ! printf '%s\n' "$@" | grep -q -- '--max-batch'; then
    EXTRA_FLAGS+=(--max-batch "$MAX_BATCH")
fi
if ! printf '%s\n' "$@" | grep -q -- '--batch-timeout-ms'; then
    EXTRA_FLAGS+=(--batch-timeout-ms "$BATCH_TIMEOUT_MS")
fi

exec ../../target/release/wiki-server \
    --db "$ROOT/$DB_PATH" \
    --num-items "$NUM_ITEMS" \
    --item-size-bits "$ITEM_BITS" \
    --listen "0.0.0.0:$PORT" \
    "${EXTRA_FLAGS[@]}" \
    --data-dir data \
    --web-dir ../web \
    -v \
    "$@"
