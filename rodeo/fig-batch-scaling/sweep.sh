#!/usr/bin/env bash
# Batch-scaling sweep: online throughput vs concurrent queries B.
#
# Produces a CSV with the same column convention as the paper's
# data/vs_distpir_online.csv so the new L40S curves drop into the existing
# pgfplots figure.
#
# SandwichPIR is invoked once per batch point (~4 s each; its offline preproc
# re-runs each time but is only ~2 s). DistPIR is invoked ONCE for the whole
# sweep via its -batches flag, because its DB generation and hint computation
# cost ~10 min at 4 GiB and would otherwise be paid per point.
#
# Usage: bash sweep.sh [DB_GIB] [RECORD_BYTES]
#   defaults: 4 GiB DB, 32 KiB records (the paper's reference point)
#
# Env knobs:
#   BATCHES="1 2 4 ..."   batch points to measure
#   TRIALS=5              SandwichPIR trials per point
#   SKIP_DISTPIR=1        SandwichPIR curve only (fast)
set -eu

DB_GIB="${1:-4}"
RECORD_BYTES="${2:-32768}"

BENCH="$(cd "$(dirname "$0")" && pwd)"
RODEO="$BENCH/.."
SANDWICH="$RODEO/.."
DISTPIR="$RODEO/schemes/distpir"

NUM_ITEMS=$(( DB_GIB * 1024 * 1024 * 1024 / RECORD_BYTES ))
ITEM_BITS=$(( RECORD_BYTES * 8 ))
ROWS=$NUM_ITEMS
COLS=$RECORD_BYTES

BATCHES="${BATCHES:-1 2 4 8 16 32 64 128 256 512 1024 2048}"
TRIALS="${TRIALS:-5}"
SKIP_DISTPIR="${SKIP_DISTPIR:-0}"

OUT="$BENCH/batch_scaling_${DB_GIB}gib.csv"
LOGDIR="$BENCH/logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date -u +%H:%M:%S)] $*" >&2; }

# Rewrite the CSV from whatever is currently in SW/DP. Called after the
# SandwichPIR pass and again after DistPIR, so interrupting the (long) DistPIR
# sweep still leaves a usable SandwichPIR-only CSV on disk.
emit_csv() {
    echo "B,sandwichpir_l40s_gib_s,distpir_l40s_gib_s" > "$OUT"
    for b in $BATCHES; do
        echo "$b,${SW[$b]:-},${DP[$b]:-}" >> "$OUT"
    done
}

log "DB=${DB_GIB}GiB  num_items=${NUM_ITEMS}  record=${RECORD_BYTES}B  trials=${TRIALS}"
log "batches: $BATCHES"

export HINT="${HINT:-ntt}"

# ══ SandwichPIR: one invocation per batch point ══
if [ ! -x "$SANDWICH/target/release/run" ]; then
    log "building SandwichPIR"
    ( cd "$SANDWICH" && cargo build --release --features cuda --bin run )
fi

declare -A SW
declare -A DP
for B in $BATCHES; do
    SW_JSON="$LOGDIR/sandwichpir-b${B}.json"
    if ( cd "$SANDWICH" && ./target/release/run \
            "$NUM_ITEMS" "$ITEM_BITS" "$B" "$TRIALS" "$SW_JSON" ) \
            > "$LOGDIR/sandwichpir-b${B}.raw" 2>&1; then
        SW[$B]=$(python3 -c "
import json
d = json.load(open('$SW_JSON'))
print(f\"{($B * $DB_GIB) / (d['online']['serverTimeMs'] / 1000.0):.2f}\")
" 2>/dev/null || echo "")
    else
        SW[$B]=""
    fi
    [ -n "${SW[$B]}" ] && log "  B=$B  SandwichPIR ${SW[$B]} GiB/s" \
                       || log "  B=$B  SandwichPIR FAILED (see $LOGDIR/sandwichpir-b${B}.raw)"
done

emit_csv
log "SandwichPIR pass done — $OUT written (DistPIR column still empty)"

# ══ DistPIR: ONE invocation covering every batch point ══
if [ "$SKIP_DISTPIR" = "1" ]; then
    log "SKIP_DISTPIR=1 — skipping DistPIR curve"
else
    if [ ! -x "$DISTPIR/benches/benches" ]; then
        log "building DistPIR benches"
        export CUDACXX="${CUDACXX:-$(command -v nvcc)}"
        ( cd "$DISTPIR/benches" && go build -o benches . )
    fi

    BATCH_CSV=$(echo "$BATCHES" | tr ' ' ',')
    DP_RAW="$LOGDIR/distpir-sweep.raw"
    log "DistPIR: single invocation, -batches $BATCH_CSV"
    log "  (DB generation + hint dominates; expect ~10 min at 4 GiB before output)"

    if ( cd "$DISTPIR/benches" && ./benches \
            -bench throughput -mode hybrid \
            -rows "$ROWS" -cols "$COLS" -q 32 -p 256 -bits 8 \
            -batches "$BATCH_CSV" ) \
            > "$DP_RAW" 2>&1; then
        # Throughput(rows x cols, k, p, N): X.YZ GB/s  — one line per k
        while IFS='|' read -r k gb; do
            [ -z "$k" ] && continue
            DP[$k]=$(python3 -c "print(f'{$gb * 1e9 / 1024**3:.2f}')")
            log "  B=$k  DistPIR     ${DP[$k]} GiB/s"
        done < <(awk 'match($0, /Throughput\([0-9]+ x [0-9]+, ([0-9]+),[^)]*\):[[:space:]]+([0-9.]+)/, m) { print m[1] "|" m[2] }' "$DP_RAW")
    else
        log "DistPIR sweep FAILED (see $DP_RAW)"
        tail -20 "$DP_RAW" | sed 's/^/    /' >&2
    fi
fi

emit_csv
log "wrote $OUT"
column -s, -t "$OUT"
