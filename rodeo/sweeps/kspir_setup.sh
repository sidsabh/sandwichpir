#!/bin/bash
# Runs ON the r7i box. Installs clang + HEXL 1.2.5, builds kspir @ 54c2f61 with size-arg patch, smoke-runs.
set -e
cd ~

SUDO="sudo -n"
if ! sudo -n true 2>/dev/null; then
    SUDO="sudo -S"
    export SUDO_PW=1
fi

runsudo() {
    if [ -n "$SUDO_PW" ]; then
        sudo "$@"
    else
        sudo -n "$@"
    fi
}

if ! command -v clang++ >/dev/null; then
    runsudo apt-get update -qq
    runsudo apt-get install -y -qq clang
fi
clang++ --version | head -1

if [ ! -f /usr/local/lib/cmake/hexl-1.2.5/HEXLConfig.cmake ] && [ ! -d /usr/local/include/hexl ]; then
    if [ ! -d ~/hexl ]; then
        git clone --depth 1 --branch v1.2.5 https://github.com/intel/hexl.git ~/hexl
    fi
    cd ~/hexl
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEXL_BENCHMARK=OFF -DHEXL_TESTING=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build build -j8
    runsudo cmake --install build
    cd ~
fi

if [ ! -d ~/kspir ]; then
    git clone https://github.com/mmingluo/kspir.git ~/kspir
fi
cd ~/kspir
git checkout 54c2f61 2>/dev/null || true
git stash 2>/dev/null || true
git apply ~/kspir-size-arg.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --target test-pir

echo "=== SMOKE 256 MB ==="
./build/tests/test-pir 256
echo "=== 1 GiB (r=64) ==="
./build/tests/test-pir 1024
echo "=== DONE ==="
