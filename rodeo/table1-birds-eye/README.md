# Birds-eye PIR comparison

Regenerates the paper's PIR comparison table: HintlessPIR, YPIR, InsPIRe,
KsPIR, DistPIR (both engines), and SandwichPIR at DB in {1, 4, 16} GB,
records fixed at 32 KB, batch = 1, single query per client.

## Rows

| Scheme               | Instance                   | Notes                              |
|----------------------|----------------------------|------------------------------------|
| HintlessPIR          | r7i.2xlarge (\$0.5292/hr)  | LinPIR rows_per_block=2048         |
| YPIR                 | r7i.2xlarge                | silent-preprocessing variant       |
| InsPIRe              | r7i.2xlarge                | dim0 = record count, auto interp   |
| KsPIR                | r7i.2xlarge                | packing tree terminated two early  |
| DistPIR (CPU)        | r7i.2xlarge                | scan only, p = 256                 |
| DistPIR (GPU)        | g6e.xlarge (\$1.861/hr)    | SimplePIR-on-GPU scan-only         |
| SandwichPIR          | g6e.xlarge                 | INT8 tensor cores, sm_89           |

## Shape sweep

| DB      | ℓ₁ (num_items) | ℓ₂ (record_bytes) |
|---------|---------------:|------------------:|
| 1 GiB   | 32,768   (2^15)| 32,768  (2^15)    |
| 4 GiB   | 131,072  (2^17)| 32,768  (2^15)    |
| 16 GiB  | 524,288  (2^19)| 32,768  (2^15)    |

## Usage

```bash
# From this directory, after `make -C .. setup-cpu` (or setup-gpu):
make measure-cpu      # or measure-gpu
make table-latex      # → out/table1.tex (paste into Overleaf)
```

Results land in `results/*.json`; the renderer picks the latest per (scheme, DB).

## Cost model

Uses shared `../configs/cost_model.toml`:
- r7i.2xlarge: \$0.5292/hr
- g6e.xlarge: \$1.861/hr
- Egress: \$0.09/GB (ingress free)

`$/Mq = compute + egress` where
`compute = server_seconds × instance_hourly × 10⁶ / 3600`
and `egress = (response + hint) × egress_rate × 10⁶ / 10⁹`.
