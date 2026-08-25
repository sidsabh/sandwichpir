#!/usr/bin/env bash
# Wikipedia batching sweep on L40S: 8 GiB, 128 KiB records (65536 x 1 MiB-bit rows),
# B in {2,4,8,16,32,64} to complete the {1,128,256} cells already measured.
# 5 trials for tight means; appends to ~/wiki_batch.csv.
source "$HOME/.cargo/env" 2>/dev/null
cd "$HOME/benchmarking/sandwichpir" || exit 1
export CUTLASS_DIR="$HOME/benchmarking/cutlass"
OUT="$HOME/wiki_batch.csv"
echo "db_gib,record_kib,items,batch,trials,server_ms_mean,server_ms_std,tput_gibs_mean,upload_bytes,download_bytes,status" > "$OUT"
for B in 2 4 8 16 32 64; do
  J="/tmp/wb_${B}.json"; L="/tmp/wb_${B}.log"
  timeout 1200 ./target/release/run 65536 1048576 "$B" 5 "$J" > "$L" 2>&1
  if [ $? -ne 0 ] || [ ! -s "$J" ]; then
    echo "8,128,65536,${B},5,,,,,,FAIL" >> "$OUT"; echo "  B=${B} FAIL"; continue
  fi
  python3 - "$B" "$OUT" <<'PY'
import json, sys, statistics
b, out = sys.argv[1:3]
d = json.load(open(f"/tmp/wb_{b}.json")); o = d.get('online', d)
ts = o.get('allServerTimesMs') or [o['serverTimeMs']]
tm = statistics.mean(ts); tsd = statistics.stdev(ts) if len(ts) > 1 else 0.0
tput = int(b) * 8 * 2**30 / (tm/1000.0) / 2**30
open(out,'a').write(f"8,128,65536,{b},{len(ts)},{tm:.3f},{tsd:.3f},{tput:.0f},{o.get('uploadBytes','')},{o.get('downloadBytes','')},ok\n")
print(f"  B={b}: {tm:.2f}+/-{tsd:.2f}ms  {tput:,.0f} GiB/s")
PY
done
echo "WIKI_DONE $(date -u)"
