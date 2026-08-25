#!/usr/bin/env bash
# (batch x large-DB) grid on g6e.48xlarge, real MULTI_GPU=8 path.
# 1 MiB records at both large sizes (shape-optimal, consistent across the
# table); 32 KiB fixed as the D-003 reference; one 2 MiB shape-check cell;
# one 16 GiB validation cell. B=1/8 latency cells appended by a follow-up run.
source "$HOME/.cargo/env" 2>/dev/null
cd "$HOME/benchmarking/sandwichpir" || exit 1
export CUTLASS_DIR="$HOME/benchmarking/cutlass"
export MULTI_GPU=8
OUT="$HOME/mgpu_grid.csv"
echo "total_gib,record_kib,items,batch,num_gpus,server_ms_mean,server_ms_std,tput_gibs,upload_bytes,download_bytes,status" > "$OUT"

run_cfg() {
  local TOTAL=$1 REC=$2 B=$3
  local ITEMS=$(( TOTAL * 1024 * 1024 / REC ))
  local BITS=$(( REC * 1024 * 8 ))
  echo "CFG total=${TOTAL}GiB rec=${REC}KiB B=${B} items=${ITEMS}"
  local J="/tmp/mgg_${TOTAL}_${REC}_${B}.json" L="/tmp/mgg_${TOTAL}_${REC}_${B}.log"
  rm -f "$J"
  timeout 10800 ./target/release/run "$ITEMS" "$BITS" "$B" 3 "$J" > "$L" 2>&1
  local ec=$?
  if [ $ec -ne 0 ] || [ ! -s "$J" ]; then
    local why=$(grep -ioE 'out of memory|panicked at .[^"]{0,60}|CUDA error[^"]{0,40}' "$L" | head -1)
    [ $ec -eq 124 ] && why="timeout"
    echo "${TOTAL},${REC},${ITEMS},${B},${MULTI_GPU},,,,,,FAIL:${why:-exit=$ec}" >> "$OUT"
    echo "  FAIL ${TOTAL}GiB rec=${REC}KiB B=${B}: ${why:-exit=$ec}"
    return 1
  fi
  python3 - "$J" "$TOTAL" "$REC" "$ITEMS" "$B" "$OUT" <<'PY'
import json, sys, statistics
j, total, rec, items, batch, out = sys.argv[1:7]
d = json.load(open(j)); o = d.get('online', d)
ts = o.get('allServerTimesMs') or [o['serverTimeMs']]
tm = statistics.mean(ts); tsd = statistics.stdev(ts) if len(ts) > 1 else 0.0
tput = int(batch) * int(total) * 2**30 / (tm/1000.0) / 2**30
ng = d.get('numGpus', '?')
row = f"{total},{rec},{items},{batch},{ng},{tm:.3f},{tsd:.3f},{tput:.0f},{o.get('uploadBytes','')},{o.get('downloadBytes','')},ok"
open(out, 'a').write(row + "\n")
print(f"  AGG {total}GiB rec={rec}KiB B={batch} gpus={ng}: server={tm:.2f}+/-{tsd:.2f}ms tput={tput:,.0f} GiB/s")
PY
}

echo "GRID_START $(date -u)"
run_cfg 16  256  256      # validation vs hand-rolled smoke
run_cfg 128 1024 64
run_cfg 128 1024 128
run_cfg 128 1024 256
run_cfg 128 1024 512
run_cfg 128 32   128      # D-003 fixed-record reference
run_cfg 256 1024 64
run_cfg 256 1024 128
run_cfg 256 1024 256
run_cfg 256 1024 512
run_cfg 256 32   128      # reference; OOMs (32 GiB shard + 4 GiB query)
run_cfg 256 2048 256      # shape check; OOMs (rho=1024 response buffers)
echo "GRID_DONE $(date -u)"
