#!/usr/bin/env bash
# Bootstrap a fresh r7i.2xlarge for the CPU baseline rows.
# Assumes Ubuntu 24.04 (default AL2023 or Ubuntu on EC2).
# Idempotent — safe to re-run.
set -eu
# NOTE: no `pipefail` — this script uses `... | head` patterns that trip pipefail
# via SIGPIPE when the reader closes early. All important commands are checked
# explicitly.

# Detect distro
if command -v apt-get >/dev/null 2>&1; then
    PKG="apt"
elif command -v dnf >/dev/null 2>&1; then
    PKG="dnf"
else
    echo "Unsupported package manager." >&2; exit 1
fi

echo "==> Installing base build deps"
if [ "$PKG" = "apt" ]; then
    sudo apt-get update -y
    sudo apt-get install -y \
        build-essential g++-13 g++-14 clang-18 lld-18 \
        cmake ninja-build pkg-config \
        libssl-dev libboost-all-dev libgoogle-perftools-dev \
        curl git python3 python3-pip jq bc \
        libomp-dev
else
    sudo dnf install -y \
        gcc gcc-c++ clang lld cmake ninja-build pkgconfig \
        openssl-devel boost-devel google-perftools-devel \
        curl git python3 python3-pip jq bc libomp-devel
fi

echo "==> Installing Bazelisk (for HintlessPIR)"
if ! command -v bazelisk >/dev/null 2>&1; then
    sudo curl -fsSL -o /usr/local/bin/bazelisk \
        https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    sudo chmod +x /usr/local/bin/bazelisk
    sudo ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel
fi

echo "==> Installing Rustup + pinned nightly (for YPIR/InsPIRe)"
if ! command -v rustup >/dev/null 2>&1; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    . "$HOME/.cargo/env"
fi
rustup toolchain install nightly-2024-02-07 --profile minimal --component rustfmt
rustup default nightly-2024-02-07

echo "==> Sanity"
gcc --version | head -1
g++ --version | head -1
cmake --version | head -1
bazel --version || true
rustc --version
python3 --version
jq --version

echo "==> CPU capability check"
grep -oE '(avx|avx2|avx512[a-z_]*)' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
if grep -q avx512 /proc/cpuinfo; then
    echo "  AVX-512 present — YPIR/InsPIRe fast paths will fire."
else
    echo "  WARNING: no AVX-512 — YPIR/InsPIRe will not hit their peak."
fi

echo "==> setup-cpu.sh DONE"
