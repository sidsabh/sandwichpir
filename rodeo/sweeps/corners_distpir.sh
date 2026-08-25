#!/usr/bin/env bash
# Runs ON the g6e.2xlarge (L40S). Sequential: everything contends for the GPU.
# Part 1: the four record-shape corner cells missing from tab:record-shape.
# Part 2: DistPIR (crowdsurf) offline preprocessing, both modes, plus client
#         query-gen latency, at the four square-layout shapes of tab:offline-l40s.
source "$HOME/.cargo/env" 2>/dev/null
source /etc/profile.d/sandwichpir_cuda.sh 2>/dev/null

# ---------- Part 1: record-shape corners ----------
cd "$HOME/benchmarking/sandwichpir" || exit 1
export CUTLASS_DIR="$HOME/benchmarking/cutlass"
TRIALS=5
OUT="$HOME/record_shape_corners.csv"
echo "db_gib,record_kib,items,batch,trials,server_ms_mean,server_ms_std,tput_gibs_mean,tput_gibs_std,upload_bytes,download_bytes,status" > "$OUT"

CONFIGS="
1 128 8192 1048576
1 256 4096 2097152
4 256 16384 2097152
16 8 2097152 65536
"

echo "START_CORNERS $(date -u)"
echo "$CONFIGS" | while read -r DB REC ITEMS BITS; do
  [ -z "$DB" ] && continue
  for B in 64 128 256 512; do
    J="/tmp/rs_${DB}_${REC}_${B}.json"
    L="/tmp/rs_${DB}_${REC}_${B}.log"
    rm -f "$J"
    timeout 900 ./target/release/run "$ITEMS" "$BITS" "$B" "$TRIALS" "$J" > "$L" 2>&1
    ec=$?
    if [ $ec -ne 0 ] || [ ! -s "$J" ]; then
      why=$(grep -ioE 'out of memory|unsupported[^"]*|panicked at .*' "$L" | head -1 | cut -c1-60)
      [ $ec -eq 124 ] && why="timeout"
      [ -z "$why" ] && why="exit=$ec"
      echo "$DB,$REC,$ITEMS,$B,$TRIALS,,,,,,,$why" >> "$OUT"
      echo "  db=${DB}Gi rec=${REC}Ki B=${B}  FAIL($why)"
      continue
    fi
    python3 - "$J" "$DB" "$REC" "$ITEMS" "$B" "$TRIALS" "$OUT" <<'PY'
import json, sys, statistics
j, db, rec, items, batch, trials, out = sys.argv[1:8]
d = json.load(open(j)); o = d.get('online', d)
ts = o.get('allServerTimesMs') or [o.get('serverTimeMs')]
n = int(d.get('numClients', batch)); B = int(d['dbSizeBytes'])
tps = [n * B / (t/1000.0) / 2**30 for t in ts]
tm, tsd = statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)
pm, psd = statistics.mean(tps), (statistics.stdev(tps) if len(tps) > 1 else 0.0)
row = f"{db},{rec},{items},{batch},{len(ts)},{tm:.3f},{tsd:.3f},{pm:.1f},{psd:.1f},{o.get('uploadBytes','')},{o.get('downloadBytes','')},ok"
open(out, 'a').write(row + "\n")
print(f"  db={db}Gi rec={rec}Ki B={batch}  {pm:9.1f} +/- {psd:5.1f} GiB/s  ({tm:.2f} +/- {tsd:.2f} ms)")
PY
  done
done
echo "DONE_CORNERS $(date -u)"

# ---------- Part 2: DistPIR offline + client query ----------
cd "$HOME/benchmarking/sandwichpir/rodeo/schemes/distpir/benches" || exit 1
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/benchmarking/sandwichpir/rodeo/schemes/distpir/matrix/gpu/cuda/build"

# "<label_gib> <rows> <cols>" -- 1 B/element (p=256, bits=8); record = cols bytes
SHAPES="
0.25 16384 16384
1 32768 32768
4 65536 65536
8 65536 131072
"

echo "START_DISTPIR $(date -u)"
echo "$SHAPES" | while read -r GIB ROWS COLS; do
  [ -z "$GIB" ] && continue
  for MODE in none hybrid; do
    echo "== preprocessing db=${GIB}GiB rows=$ROWS cols=$COLS mode=$MODE =="
    timeout 1200 ./benches -rows "$ROWS" -cols "$COLS" -q 32 -p 256 -bits 8 \
      -bench preprocessing -mode "$MODE" 2>&1 | tail -3
  done
  echo "== query db=${GIB}GiB rows=$ROWS cols=$COLS mode=hybrid =="
  timeout 600 ./benches -rows "$ROWS" -cols "$COLS" -q 32 -p 256 -bits 8 \
    -bench query -mode hybrid 2>&1 | tail -3
done
echo "DONE_DISTPIR $(date -u)"
echo "ALL_DONE $(date -u)"
