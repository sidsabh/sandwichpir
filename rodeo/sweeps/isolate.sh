#!/usr/bin/env bash
# Two isolation sweeps for the record-size analysis, both from the
# VERBOSE per-stage timers (matmul(gemm=..)=X, packing(..)=Y):
#
# A) Packing linearity: fixed 4 GiB and B=256, sweep record width M.
#    Claim under test: packing GEMM time is linear in M (one RLWE output per
#    2048 bytes); scan time roughly flat once tiled.
#
# B) Cache residency: fixed M=32 KiB and B=512, sweep DB size so the query
#    operand K*4B crosses L2 (96 MB): 67/134/268/537/1074 MB.
#    Claim under test: scan ms per GiB of database degrades once the operand
#    no longer fits, independent of tiling (M*B fixed).
source "$HOME/.cargo/env" 2>/dev/null
cd "$HOME/benchmarking/sandwichpir" || exit 1
export CUTLASS_DIR="$HOME/benchmarking/cutlass"

run_v() { # items bits batch tag
  local ITEMS=$1 BITS=$2 B=$3 TAG=$4
  VERBOSE=1 timeout 2400 ./target/release/run "$ITEMS" "$BITS" "$B" 3 "/tmp/iso_${TAG}.json" > "/tmp/iso_${TAG}.log" 2>&1
  if [ $? -ne 0 ]; then echo "  ${TAG}: FAIL $(grep -ioE 'out of memory|panicked at .{0,40}' /tmp/iso_${TAG}.log | head -1)"; return 1; fi
  python3 - "$TAG" <<'PY'
import re, sys
tag = sys.argv[1]
txt = open(f"/tmp/iso_{tag}.log").read()
scans = [float(m) for m in re.findall(r"matmul\(gemm=([0-9.]+)", txt)]
packs = [float(m) for m in re.findall(r"packing\([^)]*\)=([0-9.]+)", txt)]
s = scans[len(scans)//2:]; p = packs[len(packs)//2:]
print(f"  {tag}: scan_gemm={sum(s)/len(s):8.2f}ms  packing={sum(p)/len(p):7.2f}ms  (n={len(s)})")
PY
}

echo "ISO_START $(date -u)"
echo "--- A: packing vs record width (4 GiB, B=256) ---"
run_v 262144 131072   256 A_16K
run_v 131072 262144   256 A_32K
run_v 65536  524288   256 A_64K
run_v 32768  1048576  256 A_128K
run_v 16384  2097152  256 A_256K
echo "--- B: cache residency (M=32 KiB, B=512, query operand 67..1074 MB) ---"
run_v 32768  262144 512 B_1G
run_v 65536  262144 512 B_2G
run_v 131072 262144 512 B_4G
run_v 262144 262144 512 B_8G
run_v 524288 262144 512 B_16G
echo "ISO_DONE $(date -u)"
