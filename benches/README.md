# Benchmarks

## End-to-End Sweep

`sweep.py` drives the full SandwichPIR binary (`cargo run --bin run`) across the
DB sizes and batch sizes used in the paper. For each `(config, batch)` it
captures the `Measurement` JSON (offline + online times, byte counts, per-phase
breakdown, per-trial variance) into one aggregate file.

### DB configs

| Name | num_items | item_size_bits | $\ell_1 \times \ell_2$ | $\rho$ | DB |
|---|---|---|---|---|---|
| `0.25gib` | 16384 | 131072 | 16384 x 16384 | 8 | 0.25 GiB (square, DistPIR fig3 low end) |
| `1gib` | 32768 | 262144 | 32768 x 32768 | 16 | 1 GiB (square) |
| `4gib` | 65536 | 524288 | 65536 x 65536 | 32 | 4 GiB (square, DistPIR 4 GiB point) |
| `8gib` | 65536 | 1048576 | 65536 x 131072 | 64 | 8 GiB (Wikipedia) |

Batch sizes default to `1, 2, 4, 8, ..., 2048` (powers of two). All are
dispatched by the binary's `dispatch_const!` macro.

### Run

```bash
cd sandwichpir

# Full sweep, single GPU
python benches/sweep.py

# Multi-GPU column sharding (passes through as MULTI_GPU env var)
MULTI_GPU=3 python benches/sweep.py --out benches/sweep_results/a100_mg3.json

# Subset: only 4 GiB, a few batches
python benches/sweep.py --configs 4gib --batches 1,64,1024

# Stream the binary's JSON to stdout as it runs
python benches/sweep.py --verbose-binary
```

### Output

`benches/sweep_results/<gpu-slug>_mg<MULTI_GPU>.json`:

```json
{
  "meta": {
    "gpu": "NVIDIA A100-PCIE-40GB",
    "hostname": "anonymized",
    "git_commit": "eb2c699",
    "multi_gpu": 3,
    "trials_per_config": 5,
    "timestamp": "2026-04-14T10:32:00+00:00",
    ...
  },
  "results": [
    {
      "config": "4gib", "db_gib": 4.0,
      "num_items": 65536, "item_size_bits": 524288,
      "ell_1": 65536, "ell_2": 65536, "rho": 32,
      "batch": 64,
      "measurement": { "offline": {...}, "online": {...} }
    },
    ...
  ]
}
```

Each row is appended and the file is rewritten after every run, so an
interrupted sweep still yields usable partial data.

The binary is always invoked with `VERBOSE=1`, so each result also carries a
`breakdown` dict parsed from stdout (one entry per GPU shard):

```json
"breakdown": {
  "hint_shards": [{"ntt_mul_acc_ms": 221.3, "barrett_intt_ms": 8.6, "total_ms": 229.9}, ...],
  "precomp_shards": [{"prep_ms": 8.5, "phase1_ms": 843.7, "phase2_ms": 124.7, "total_ms": 976.9}, ...],
  "tc_packing_init_shards": [{"rows": 8192, "cols": 65536, "mib": 2048.0, "init_ms": 126.4}, ...],
  "online_shards": [
    {"clients": 1,
     "packing": {"y_decomp_ms": 0.1, "gemm_ms": 2.1, "accum_ms": 0.0, "z_final_ms": 0.0, "total_ms": 2.3},
     "matmul":  {"gemm_ms": 3.6, "accum_ms": 0.0, "total_ms": 3.6},
     "post_ms": 0.0, "compute_ms": 5.9, "download_ms": 0.0, "total_ms": 5.9},
    ...
  ],
  "memory_mib": {"monomials": 64.0, "a_ct": 1024.0, "w_all": 127.9, "r_all": 1024.0, "bold_t": 4092.0},
  "throughput": {"hint_gib_s": 34.78, "offline_gib_s": 4.67, "online_gib_s": 955.0}
}
```

Raw stdout is also stored as `stdout` (pass `--no-raw-stdout` to omit on long
sweeps where size matters).

## Brotli Decompression Microbench

`brotli_bench.rs` times the exact per-article decompression the browser runs
in the Wikipedia client ([`wiki.js`](../wikipedia/web/js/wiki.js)):

1. Read one 128 KB row from `wikipedia.bin`.
2. Parse `[4B u32 len][compressed bytes]` entries out of the row.
3. Brotli-decompress each article's compressed slice, time N iterations.

Used by the paper's end-to-end Wikipedia latency breakdown. Engine is the
native Rust `brotli` crate; the browser uses `brotli-wasm`, which is typically
1.5-2x slower — note that gap when quoting numbers.

```bash
cargo build --release --bin brotli_bench

# Default: 30 rows evenly spaced through the file, 50 iters per article.
# Yields ~3000 articles x 50 = ~150k decompress samples across ~30 topics.
./target/release/brotli_bench \
    --wiki-bin wikipedia/server/data/wikipedia.bin \
    --json benches/sweep_results/brotli_m6a_xlarge.json

# More rows / more iterations for tighter tails
./target/release/brotli_bench \
    --wiki-bin wikipedia/server/data/wikipedia.bin \
    --num-rows 100 --iters 200
```

Output (JSON, single object):
```json
{
  "sample_source": "wikipedia/server/data/wikipedia.bin (30 rows sampled, 3100 articles)",
  "row_size_bytes": 131072,
  "total_rows_in_file": 65536,
  "rows_sampled": 30,
  "sampled_row_indices": [0, 2184, 4368, ..., 63352],
  "total_articles": 3100,
  "median_compressed_bytes": 2800,
  "median_uncompressed_bytes": 8900,
  "iters_per_article": 50,
  "total_samples": 155000,
  "median_decompress_us": 64.2,
  "mean_decompress_us": 70.1,
  "p95_decompress_us": 112.5,
  "min_decompress_us": 61.3,
  "throughput_mib_s": 132.0,
  "engine": "rust brotli crate (native)"
}
```

## PIR GEMM Benchmark

GPU throughput sweep for the SimplePIR-style database multiply (`DB × Q`) across
SIMT and tensor-core kernels. This is the source of the roofline data in the paper.

Modes (all database shape `2^log_dim × 2^log_dim`, batch swept):

| Mode | Description | Role |
|---|---|---|
| `db8_q32` | SIMT `uint8 × uint32 → uint32` | SandwichPIR SIMT baseline |
| `db8_q32_tc_cutlass` | CUTLASS INT8 TC, 1× wide GEMM (query split into 4 byte lanes) | SandwichPIR TC operating point |
| `db16_q64` | SIMT `uint16 × uint64 → uint64` | legacy YPIR word path |
| `db16_q64_tc_cutlass` | CUTLASS INT8 TC, stacked GEMM for `uint16 × uint64` | future work (larger plaintext) |
| `thesis` (default) | runs the four modes above in order | roofline figure data |

Other modes (`db32_q32`, `db32_q64`, `db16_q32`, `db16_crt`, `db32_crt_i64`,
`db8_q32_tc`, `db16_q64_tc`) sweep cuBLAS and coarser-precision variants; see
`--help`.

### Prerequisites

- CUDA Toolkit >= 11.4
- NVIDIA GPU with SM >= 75 (T4, L4, A100, ...). CUTLASS-TC modes need SM >= 70.
- CUTLASS headers (header-only):

```bash
git clone https://github.com/NVIDIA/cutlass.git ../cutlass
```

### Build

```bash
cd sandwichpir
mkdir -p benches/build
nvcc -O3 -arch=sm_80 --expt-relaxed-constexpr \
  -I ../cutlass/include -I ../cutlass/tools/util/include \
  benches/pir_bench.cu \
  -lcublas \
  -o benches/build/pir_bench
```

GPU architectures: `sm_75` (T4/L4-Turing), `sm_80` (A100), `sm_89` (L4-Ada), `sm_90` (H100).

### Run

```bash
# Default "thesis" sweep: four modes, log_dim 15, batches 1..2048
./benches/build/pir_bench

# Single mode at log_dim 16 (4 GiB uint8 DB)
./benches/build/pir_bench --mode db8_q32_tc_cutlass --log_dim 16

# Custom batch list
./benches/build/pir_bench --mode db8_q32 --batches 1,8,64,512,2048
```

### Output columns

- **Eff Tput (GB/s)** — `(DB_size * batch) / time`; amortized metric reported in PIR papers. Exceeds HBM BW at large batch because the database is read once per sweep.
- **HW BW (GB/s)** — approximate true memory traffic `(db + queries + output + accum) / comp_time`; bounded by the GPU's theoretical HBM bandwidth.

## NTT Benchmark: SandwichPIR vs GPU-NTT

Side-by-side comparison of our NTT (spiral-rs derived, lazy Barrett) against [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) (Ozcan, Apache 2.0, strict Barrett).

Parameters: d=2048, Q=4294955009, negacyclic (x^n+1).

### Prerequisites

Clone GPU-NTT and build for your GPU architecture:

```bash
cd ..
git clone https://github.com/Alisah-Ozcan/GPU-NTT.git
cd GPU-NTT && mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80"   # A100
# cmake .. -DCMAKE_CUDA_ARCHITECTURES="61" # 1080
make -j
```

### Build

```bash
cd sandwichpir
nvcc -O3 -arch=sm_80 \
  -I src/cuda -I ../GPU-NTT/src/include \
  benches/ntt_bench.cu \
  ../GPU-NTT/build/src/libntt-1.0.a \
  -o benches/build/ntt_bench
```

### Run

```bash
./benches/build/ntt_bench [max_batch] [iters]
./benches/build/ntt_bench 32768 5
```

### Output (A100 example)

```
         |        Ours (spiral-rs alt, lazy)      |     GPU-NTT (Ozcan, strict Barrett)    | round-trip
 batch   |  fwd ms     inv ms     per/poly us     |  fwd ms     inv ms     per/poly us     | ours/ gpu
---------+----------------------------------------+----------------------------------------+-----------
 1       |      0.01       0.02            13.90  |      0.03       0.03            30.50  |   OK/  OK
 1024    |      0.52       0.78             1.27  |      0.54       0.58             1.09  |   OK/  OK
 32768   |     14.25      21.72             1.10  |     15.44      16.28             0.97  |   OK/  OK
```

- **fwd / inv**: total wall-clock time (ms) of one kernel launch processing the entire batch (forward / inverse NTT respectively).
- **per/poly**: amortized per-polynomial round-trip latency in microseconds, `(fwd + inv) * 1000 / batch`. Lower is higher throughput.
- **round-trip**: forward∘inverse equality check, run independently for each implementation. The bench finds a primitive 2N-th root mod Q, builds spiral-rs alt-form tables (Harvey/Shoup) for our NTT and bit-reversed tables for GPU-NTT. Cross-implementation output equivalence is *not* checked (different butterfly indexing).

### Notes

- Our forward NTT is 5-8% faster (lazy Barrett: fewer ops per butterfly)
- GPU-NTT's inverse is ~1.3x faster (strict reduction avoids heavier final pass)
- Both hit ~10M NTTs/sec at large batch — confirming our implementation matches SOTA
- GPU-NTT's butterfly cannot be used inside our fused kernels (lazy vs strict reduction semantics are incompatible)
