#!/bin/bash
# Offline compute/upload split: 32 KB shapes for Tables 4/5 + wiki shape (128 KB records).
cd ~/sandwichpir
mkdir -p ~/osplit
for cfg in "8192 262144" "32768 262144" "131072 262144" "262144 262144" "524288 262144" "65536 1048576"; do
    set -- $cfg
    echo "== items=$1 bits=$2 =="
    target/release/run $1 $2 1 5 ~/osplit/n$1_b$2.json > ~/osplit/n$1_b$2.log 2>&1
    echo "rc=$?"
    grep -aE "SandwichPIR offline: compute" ~/osplit/n$1_b$2.log | tail -1
done
echo SWEEP_DONE
