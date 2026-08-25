#!/bin/bash
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "=== Building SandwichPIR ==="

# Detect CUDA
if command -v nvcc &>/dev/null; then
    echo "CUDA detected — building with GPU support"
    FEATURES="--features cuda"
else
    echo "No CUDA — building CPU only"
    FEATURES=""
fi

# Core library + benchmark
echo "Building core library..."
cargo build --release $FEATURES

# PIR server
echo "Building PIR server..."
cargo build --release -p pir_server $FEATURES

# Wiki server
echo "Building wiki-server..."
cargo build --release -p wiki-server $FEATURES

# Native PIR client
echo "Building native PIR client..."
cargo build --release -p pir-client --features native

# WASM PIR client
echo "Building WASM PIR client..."
rustup target add wasm32-unknown-unknown 2>/dev/null || true
cargo build --release -p pir-client --features wasm --target wasm32-unknown-unknown

# wasm-bindgen JS glue
WASM_BINDGEN=""
if command -v wasm-bindgen &>/dev/null; then
    WASM_BINDGEN="wasm-bindgen"
elif [ -f /tmp/wasm-bindgen-0.2.117-x86_64-unknown-linux-musl/wasm-bindgen ]; then
    WASM_BINDGEN="/tmp/wasm-bindgen-0.2.117-x86_64-unknown-linux-musl/wasm-bindgen"
else
    echo "Downloading wasm-bindgen CLI..."
    curl -sL https://github.com/rustwasm/wasm-bindgen/releases/download/0.2.117/wasm-bindgen-0.2.117-x86_64-unknown-linux-musl.tar.gz | tar xz -C /tmp/
    WASM_BINDGEN="/tmp/wasm-bindgen-0.2.117-x86_64-unknown-linux-musl/wasm-bindgen"
fi

echo "Generating WASM JS glue..."
mkdir -p wikipedia/web/pkg
$WASM_BINDGEN --target web --out-dir wikipedia/web/pkg \
    target/wasm32-unknown-unknown/release/pir_client.wasm

echo ""
echo "=== Build complete ==="
echo "Binaries:"
ls -1 target/release/run target/release/pir-serve target/release/wiki-server target/release/pir-query 2>/dev/null
