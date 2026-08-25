# Evaluation shapes and conventions

Fixed conventions behind the paper's comparison tables: records are 32 KiB
and databases are 1 / 4 / 16 GiB unless a table states otherwise, so
throughput = database size / server time is comparable across schemes and
sizes. Storage columns follow exactly from the shape.

## SimplePIR-family shapes (record = ℓ₂ = 2^15 bytes)

| DB      | ℓ₁      | ℓ₂     | Bytes per row | Notes                                  |
|---------|--------:|-------:|--------------:|----------------------------------------|
| 1 GiB   | 2^15    | 2^15   | 32 KiB        | Fits every scheme                      |

HintlessPIR is column-shaped at every size (`num_rows = record_bytes =
2^15`, `num_cols = DB/2^15`, the runner enforces `rows × cols = DB`);
YPIR-SP and InsPIRe (compute-optimal) use the table's shapes directly.

## Variant selection

Where a scheme offers both a SimplePIR-like and a DoublePIR-like mode, the
SimplePIR-like variant is measured: YPIR as YPIR-SP (`-i`), InsPIRe in its
compute-optimal configuration, HintlessPIR column-shaped, DistPIR in
its native record size.

## DistPIR shape exception

DistPIR holds ℓ₁ = 2^15 (its record is a column; the hint is ℓ₁ · n · 4 B)
and grows ℓ₂ with the database: 2^15 × 2^17 at 4 GiB. This keeps the
per-client hint at 256 MiB independent of database size; growing ℓ₁
instead quadruples the hint at 4 GiB. The throughput-optimal layout
(ℓ₁ = 2^17) peaks 2.4× higher (5,072 vs 2,119 GiB/s) and is reported as a
footnote only, since the hint dominates DistPIR's cost either way. Raw
data: paper repo `data/raw_distpir/dist_reshape.txt`. 16 GiB does not fit
on the GPU at any shape: the uint32 device image is 64 GB against 45 GB of
VRAM.

## Key accounting

SimplePIR-family schemes fix the LWE matrix `A` across queries (shipped as
a seed), so the client's LWE secret must be fresh per query. Schemes that
additionally fix an RLWE CRS (HintlessPIR, YPIR, InsPIRe, SandwichPIR)
need a fresh RLWE secret per query as well; every derived key-switching or
packing key is therefore per-query and counted in the upload. Only
Client-cacheable downloads (SandwichPIR's response mask, DistPIR's hint)
are charged to the offline/onboarding column.

## Per-scheme runner CLI

| Scheme              | Runner arg style               | Notes                                                                |
|---------------------|--------------------------------|----------------------------------------------------------------------|
| YPIR-SP             | `<N> <record_bytes> <batch> <trials> -i` | `N = ℓ₁`; `-i` selects the SP variant.                     |
| HintlessPIR         | `--num_rows=<ℓ₁> --record_bytes=<32768>` | Column-shaped; must pass `--benchmark_filter`.             |
| InsPIRe compute-opt | (fork of the YPIR CLI)         | Same `N` / `record_bytes`; compute-opt via config flag.              |
| DistPIR (GPU)       | `-mode hybrid`, `p ≤ 10 bits`  | Scan-only mode; sm_89 patch required.                                |
| SandwichPIR         | `cargo run --release --features cuda -- <N> <record_bytes> <batch>` | Standard shapes on L40S.        |


## Table scope

The headline table fixes batch = 1 and charges hint egress per client at
one query; cross-client batching and hint amortization appear in separate
tables. Offline preprocessing is its own column rather than being folded
into amortized cost. No 8 or 32 GiB rows: 1/4/16 keeps power-of-4 spacing,
and 32 GiB would force YPIR-SP off the 32 KiB record convention.

## Sanity anchors (published single-thread numbers, 1 GiB)

| Scheme        | Reported anchor         |
|---------------|-------------------------|
| SimplePIR     | ~1.2 s                  |
| HintlessPIR   | ~613 ms                 |
| YPIR          | ~129 ms                 |
| InsPIRe       | 280 ms – 4.3 s (dim0 range) |
| DistPIR (GPU) | ~10 ms (V100, batch 1)  |

Measured runs should land within ~2× of these; they validate the harness
and are never cited as measurements.
