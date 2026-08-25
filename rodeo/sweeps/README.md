# Ad-hoc sweep drivers (L40S evaluation)

One-shot drivers behind the evaluation tables; each writes a CSV that is
committed in the paper repo under `data/`. All assume the box layout from
`rodeo/setup` (`~/benchmarking/sandwichpir`, CUTLASS at `~/benchmarking/cutlass`).

- `wiki_batch.sh` — 8 GiB x 128 KiB batch sweep (tab:wiki-batching)
- `rho1.sh` + `rho1_b1.sh` — rho=1 floor vs best shape at 1/8 GiB
- `isolate.sh` — packing-vs-M and cache-vs-KB isolation (data/isolation_l40s.csv)
- `dist_reshape.sh` — DistPIR at its hint-minimal layout (data/raw_distpir)
- `mgpu_grid.sh` — (batch x large-DB) grid on 8x L40S (data/mgpu_grid_l40s.csv)
- `record_shape.sh` — record-width sweep (data/record_shape_l40s.csv); recover
  from the benchmarking box home dir on next boot.
