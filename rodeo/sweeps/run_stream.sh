#!/bin/bash
# Runs ON the r7i box. STREAM triad: single-core and all-core (8 vCPU).
# 200M doubles/array x 3 arrays = 4.8 GB working set, far beyond the 8488C LLC slice.
set -e
cd ~
if [ ! -f stream.c ]; then
    curl -sO https://www.cs.virginia.edu/stream/FTP/Code/stream.c
fi
gcc -O3 -march=native -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 stream.c -o stream
echo "=== STREAM 1 thread ==="
OMP_NUM_THREADS=1 ./stream | tee stream_1t.txt
echo "=== STREAM 8 threads ==="
OMP_NUM_THREADS=8 ./stream | tee stream_8t.txt
echo "=== DONE ==="
