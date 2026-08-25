#!/bin/bash
# Runs ON r7i. DistPIR CPU backend. The CUDA-built benches binary hard-links
# libgpu.so; inject a noop stub (built from ffi_noop.cpp: cmake with CUDACXX
# unset) via LD_LIBRARY_PATH so use_gpu() is false and the CPU matmul runs.
# NOTE: the CPU path squishes 3 p=256 elements per u32 word, and the bench's
# "DB with size" / "Throughput" prints count squished entries -- both understate
# logical size and throughput by exactly 3x. Derive throughput from the TrialsMs
# lines against logical bytes (rows*cols). The GPU path does not squish.
# Stub build: env -u CUDACXX cmake -S matrix/gpu/cuda -B ~/gpu-noop-build && cmake --build ~/gpu-noop-build
# 16 GiB needs patch 11 plus `sudo sysctl vm.overcommit_memory=1` (the unwritten
# 64 GiB staging mapping is refused by the default overcommit heuristic).
export LD_LIBRARY_PATH=~/gpu-noop-build:$LD_LIBRARY_PATH
cd ~/benchmarking/sandwichpir/rodeo/schemes/distpir/benches || exit 1
echo "== 1gib batch=1 =="
timeout 3600 ./benches -rows 32768 -cols 32768 -q 32 -p 256 -bits 8 \
  -bench throughput -mode hybrid -batch 1 2>&1
echo "== 4gib batches sweep =="
timeout 7200 ./benches -rows 32768 -cols 131072 -q 32 -p 256 -bits 8 \
  -bench throughput -mode hybrid -batches 1,8,64,128,512,2048 2>&1
echo "== 16gib batch=1 (attempt; may OOM) =="
timeout 3600 ./benches -rows 32768 -cols 524288 -q 32 -p 256 -bits 8 \
  -bench throughput -mode hybrid -batch 1 2>&1 | tail -20
echo DISTPIR_CPU_DONE
