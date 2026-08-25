#!/bin/bash
# Runs ON r7i. DistPIR CPU: 16 GiB attempt, then native-density measurements:
# (a) their exact 1 GB figure config (29366^2, p=997, bits=9 default),
# (b) 32 KiB-record variant at their validated p=838 (cols=32768 < their
#     validated 59484), rows=29128 so records are 29128*9 bits = 32.8 KB.
export LD_LIBRARY_PATH=~/gpu-noop-build:$LD_LIBRARY_PATH
cd ~/benchmarking/sandwichpir/rodeo/schemes/distpir/benches || exit 1
echo "== 16gib batch=1 p=256 (attempt; may OOM) =="
timeout 3600 ./benches -rows 32768 -cols 524288 -q 32 -p 256 -bits 8 \
  -bench throughput -mode hybrid -batch 1 2>&1 | tail -12
echo "== native 1GB their-config rows=cols=29366 p=997 =="
timeout 1800 ./benches -rows 29366 -cols 29366 -q 32 -p 997 \
  -bench throughput -mode hybrid -batch 1 2>&1
echo "== native 32KiB-record 1GiB rows=29128 cols=32768 p=838 =="
timeout 1800 ./benches -rows 29128 -cols 32768 -q 32 -p 838 \
  -bench throughput -mode hybrid -batch 1 2>&1
echo DISTPIR_NATIVE_DONE
