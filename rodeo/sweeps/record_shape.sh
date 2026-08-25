#!/usr/bin/env bash
# Record-width sweep at fixed database size, on the L40S.
#
# The scan GEMM is C[M x N] = DB[M x K] . Q[K x N] with M the record width,
# K the record count and N = 4B for a batch of B clients; M*K is pinned by the
# database size. Packing cost is linear in rho = M/2048. This sweeps M against
# batch at three database sizes so the peak can be located per size, with the
# best batch chosen per cell rather than held fixed.
#
# Emits one CSV row per (db, record, batch) cell, flushed as it goes.
source "$HOME/.cargo/env" 2>/dev/null
cd "$HOME/benchmarking/sandwichpir" || exit 1
export CUTLASS_DIR="$HOME/benchmarking/cutlass"

TRIALS=5
OUT="$HOME/record_shape_l40s.csv"
echo "db_gib,record_kib,items,batch,trials,server_ms_mean,server_ms_std,tput_gibs_mean,tput_gibs_std,upload_bytes,download_bytes,status" > "$OUT"

# "<db_gib> <record_kib> <items> <bits>" -- items*record = db, bits = record*8*1024
CONFIGS="
1 8 131072 65536
1 16 65536 131072
1 32 32768 262144
1 64 16384 524288
4 8 524288 65536
4 16 262144 131072
4 32 131072 262144
4 64 65536 524288
4 128 32768 1048576
16 16 1048576 131072
16 32 524288 262144
16 64 262144 524288
16 128 131072 1048576
16 256 65536 2097152
"

echo "START $(date -u)"
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
echo "DONE_RECORD_SHAPE $(date -u)"
