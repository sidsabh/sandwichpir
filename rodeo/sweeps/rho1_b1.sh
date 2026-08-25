#!/usr/bin/env bash
# rho=1 floor (2048 B records, one RLWE output per row) vs the best shape,
# at 1 GiB and 8 GiB. 8 GiB has never been swept, so 64/128 KiB flank its
# predicted optimum (~93 KiB); 1 GiB's best (32 KiB = 14,714) is already in
# record_shape_l40s.csv. Same CSV format as the record sweep.
source "$HOME/.cargo/env" 2>/dev/null
cd "$HOME/benchmarking/sandwichpir" || exit 1
export CUTLASS_DIR="$HOME/benchmarking/cutlass"
OUT="$HOME/rho1_l40s.csv"
true

run_cfg() {
  local DB=$1 REC_B=$2 ITEMS=$3 B=$4
  local BITS=$(( REC_B * 8 ))
  local L="${DB}g_${REC_B}b_${B}"
  timeout 1800 ./target/release/run "$ITEMS" "$BITS" "$B" 3 "/tmp/r1_${L}.json" > "/tmp/r1_${L}.log" 2>&1
  if [ $? -ne 0 ] || [ ! -s "/tmp/r1_${L}.json" ]; then
    local why=$(grep -ioE 'out of memory|panicked at .[^"]{0,50}|CUDA error[^"]{0,40}' "/tmp/r1_${L}.log" | head -1)
    echo "${DB},$(( REC_B / 1024 )),${ITEMS},${B},3,,,,,,,FAIL:${why:-err}" >> "$OUT"
    echo "  ${DB}GiB rec=${REC_B}B B=${B}  FAIL(${why:-err})"
    return 1
  fi
  python3 - "$DB" "$REC_B" "$ITEMS" "$B" "$L" "$OUT" <<'PY'
import json, sys, statistics
db, rec_b, items, batch, tag, out = sys.argv[1:7]
d = json.load(open(f"/tmp/r1_{tag}.json")); o = d.get('online', d)
ts = o.get('allServerTimesMs') or [o['serverTimeMs']]
tm = statistics.mean(ts); tsd = statistics.stdev(ts) if len(ts) > 1 else 0.0
tput = int(batch) * float(db) * 2**30 / (tm/1000.0) / 2**30
row = f"{db},{int(rec_b)/1024:g},{items},{batch},{len(ts)},{tm:.3f},{tsd:.3f},{tput:.0f},{o.get('uploadBytes','')},{o.get('downloadBytes','')},ok"
open(out, 'a').write(row + "\n")
print(f"  {db}GiB rec={rec_b}B B={batch}: server={tm:.2f}+/-{tsd:.2f}ms tput={tput:,.0f} GiB/s")
PY
}

echo "B1_ADDENDUM_START $(date -u)"
run_cfg 1 2048   524288  1
run_cfg 8 2048   4194304 1
run_cfg 8 65536  131072  1
run_cfg 8 131072 65536   1
echo "RHO1_DONE (B=1 addendum) $(date -u)"
