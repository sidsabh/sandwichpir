#!/usr/bin/env bash
# DistPIR (CrowdSurf SimplePIR-on-GPU) at the CORRECTED shape: rows pinned to
# the record height (2^15 = 32 KiB records at 1 B/entry), columns grow with DB.
# This is the shape DistPIR would choose itself: hint = rows * n * 4 B = 256 MiB
# constant in DB size, vs the 1 GiB previously reported at 4 GiB by growing
# rows. Scan MACs identical; this measures whether throughput moves.
# 4 GiB: 32768 x 131072. 16 GiB: 32768 x 524288 (host-OOM on 30 GiB boxes;
# uint32 device storage rules 16 GiB out regardless).
cd "$HOME/benchmarking/sandwichpir/rodeo/schemes/distpir/benches" || exit 1
export LD_LIBRARY_PATH="$HOME/benchmarking/sandwichpir/rodeo/schemes/distpir/matrix/gpu/cuda/build:${LD_LIBRARY_PATH:-}"
OUT="$HOME/dist_reshape.txt"
: > "$OUT"

run_one() {
  local ROWS=$1 COLS=$2 BATCH=$3 TAG=$4
  echo "CFG ${TAG} rows=${ROWS} cols=${COLS} batch=${BATCH}" | tee -a "$OUT"
  timeout 3600 ./benches -bench throughput -mode hybrid \
      -rows "$ROWS" -cols "$COLS" -q 32 -p 256 -bits 8 -batch "$BATCH" \
      > "/tmp/dr_${TAG}_${BATCH}.log" 2>&1
  local ec=$?
  if [ $ec -ne 0 ]; then
    echo "  FAIL exit=$ec: $(grep -iE 'out of memory|panic|error' /tmp/dr_${TAG}_${BATCH}.log | head -1)" | tee -a "$OUT"
    return 1
  fi
  grep -E "Throughput" "/tmp/dr_${TAG}_${BATCH}.log" | tee -a "$OUT"
}

echo "DR_START $(date -u)" | tee -a "$OUT"
for B in 1 8 64 128 512 2048; do
  run_one 32768 131072 "$B" 4gib
done
for B in 1 128; do
  run_one 32768 524288 "$B" 16gib
done
echo "DR_DONE $(date -u)" | tee -a "$OUT"
