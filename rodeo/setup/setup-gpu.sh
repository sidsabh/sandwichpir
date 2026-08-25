#!/usr/bin/env bash
# Bootstrap a fresh g6e.2xlarge (L40S, sm_89, 64 GiB DRAM) for the GPU rows of Table 1.
# (g6e.xlarge with 32 GiB DRAM is too small for DistPIR at 4 GiB DB.)
# Assumes Ubuntu 24.04 with nvidia-driver-580+ and CUDA >= 12.4.
# Idempotent — safe to re-run.
set -eu
# NOTE: no `pipefail` — this script uses `... | head -N` patterns that trip pipefail
# via SIGPIPE when the reader closes early. All important commands are checked
# explicitly.

echo "==> Verifying NVIDIA driver + CUDA"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Install NVIDIA driver first (Ubuntu: sudo apt install nvidia-driver-580)." >&2
    exit 1
fi
nvidia-smi | head -10

# Add CUDA to PATH proactively — Ubuntu's cuda-toolkit installs at /usr/local/cuda/bin,
# which isn't on the default PATH.
if [ -d /usr/local/cuda/bin ]; then
    export PATH="$PATH:/usr/local/cuda/bin"
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found on PATH ($PATH) — install CUDA toolkit (12.4+ required)." >&2
    echo "Ubuntu: sudo apt install nvidia-cuda-toolkit (or the NVIDIA installer)." >&2
    exit 1
fi
nvcc --version | tail -3

CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
CC_MINOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f2)
echo "==> GPU compute capability: ${CC_MAJOR}.${CC_MINOR}"
if [ "$CC_MAJOR" -lt 8 ] || { [ "$CC_MAJOR" -eq 8 ] && [ "$CC_MINOR" -lt 9 ]; }; then
    echo "  WARNING: SandwichPIR + DistPIR-SPGPU target sm_89 (L40S). This GPU may not run them."
fi

echo "==> Installing base deps"
sudo apt-get update -y
sudo apt-get install -y \
    build-essential g++-13 clang-18 lld-18 \
    cmake ninja-build pkg-config \
    curl git python3 python3-pip jq bc \
    libssl-dev libboost-all-dev \
    libgmp-dev libgmpxx4ldbl \
    libomp-dev

# Go 1.22.x — CrowdSurf's matrix.go uses `type Elem32 = C.Elem32` with methods,
# which Go 1.23+ rejects ("cannot define new methods on non-local type Elem32").
# Ubuntu 24.04 apt no longer ships golang-1.22, so pull the official tarball.
GO_VERSION="${GO_VERSION:-1.22.10}"
if ! /usr/local/go/bin/go version 2>/dev/null | grep -q "go${GO_VERSION}"; then
    echo "==> Installing Go ${GO_VERSION} from official tarball"
    sudo rm -rf /usr/local/go
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | sudo tar -xz -C /usr/local
fi
sudo ln -sf /usr/local/go/bin/go /usr/local/bin/go
hash -r
command -v go >/dev/null || { echo "ERROR: go install failed"; exit 1; }
go version

echo "==> Setting CUDA env (persistent in /etc/profile.d/, in-shell for this script)"
# Resolve nvcc's actual location; on Ubuntu with apt cuda-toolkit it's /usr/bin/nvcc,
# on NVIDIA installers it's /usr/local/cuda/bin/nvcc.
NVCC_PATH="$(command -v nvcc)"
CUDA_HOME_DETECTED="$(dirname "$(dirname "$NVCC_PATH")")"
[ -d "$CUDA_HOME_DETECTED/lib64" ] || CUDA_HOME_DETECTED=/usr/local/cuda

sudo tee /etc/profile.d/sandwichpir_cuda.sh > /dev/null <<EOF
export CUDA_HOME=$CUDA_HOME_DETECTED
export CUDAHOSTCXX=/usr/bin/g++-13
export CUDACXX=$NVCC_PATH
export PATH=\$PATH:$CUDA_HOME_DETECTED/bin
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:$CUDA_HOME_DETECTED/lib64
EOF
export CUDA_HOME="$CUDA_HOME_DETECTED"
export CUDAHOSTCXX=/usr/bin/g++-13
export CUDACXX="$NVCC_PATH"
export PATH="$PATH:$CUDA_HOME/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib64"

echo "==> Installing CUTLASS v3.5.1 (needed by both DistPIR + SandwichPIR)"
if [ ! -d "$HOME/cutlass" ]; then
    git clone --branch v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
fi
# SandwichPIR accumulates u8xu8 products mod 2^32; upstream CUTLASS emits
# saturating (.satfinite) int8 MMA PTX, which silently clamps partial sums
# once the scan dimension exceeds 2^16 rows. Strip .satfinite so the
# accumulator wraps. Without this, decode fails on databases >= 4 GB.
PATCH_FILE="$(cd "$(dirname "$0")/../patches" && pwd)/12-cutlass-no-satfinite.patch"
if grep -q satfinite "$HOME/cutlass/include/cutlass/arch/mma_sm80.h"; then
    (cd "$HOME/cutlass" && git apply "$PATCH_FILE")
fi
if grep -q satfinite "$HOME/cutlass/include/cutlass/arch/mma_sm80.h"; then
    echo "ERROR: CUTLASS satfinite patch did not apply; see rodeo/patches/12-cutlass-no-satfinite.patch" >&2
    exit 1
fi

echo "==> Installing Rustup + toolchains"
if ! command -v rustup >/dev/null 2>&1; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    . "$HOME/.cargo/env"
fi
# SandwichPIR itself pins 1.95.0 via rust-toolchain.toml (matches Cargo.lock).
# The pinned nightly is for baseline submodules that carry their own pins.
rustup toolchain install 1.95.0 --profile minimal --target wasm32-unknown-unknown
rustup toolchain install nightly-2024-02-07 --profile minimal
rustup default 1.95.0

echo "==> Go 1.22 (for CrowdSurf / DistPIR)"
go version || { echo "Go install failed."; exit 1; }

echo "==> setup-gpu.sh DONE"
