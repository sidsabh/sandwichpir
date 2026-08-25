# rodeo — SandwichPIR benchmarking harness

Reproducibility target: **`git clone --recursive && make -C table1-birds-eye table`**
regenerates the paper's PIR comparison table end-to-end.

## Layout

Shared infrastructure at the top; each benchmark gets its own subdirectory with
its own Makefile, sweep spec, results, and rendered outputs.

```
rodeo/
├── configs/
│   └── cost_model.toml    # AWS $/hr + egress rates (shared across benches)
│
├── schemes/               # git submodules — pinned upstream commits
│   ├── hintlesspir/       # google/hintless_pir
│   ├── ypir/              # menonsamir/ypir
│   ├── inspire/           # google/private-membership (subdir research/InsPIRe)
│   ├── kspir/             # mmingluo/kspir
│   └── distpir/           # ryanleh/crowdsurf
│
├── patches/               # Baseline patches; see patches/README.md for the inventory
│
├── runners/               # Per-scheme parameterized runners (reused across benches)
│   ├── lib.sh
│   ├── run_hintlesspir.py
│   ├── run_ypir_sp.py
│   ├── run_inspire.py
│   ├── run_distpir.py
│   └── run_sandwichpir.py
│
├── harness/schema/
│   └── result.schema.json # runner-to-renderer contract
│
├── setup/                 # Bootstrap a fresh instance
│   ├── setup-cpu.sh       # r7i.2xlarge: apt + rust + bazel
│   └── setup-gpu.sh       # g6e.2xlarge: + CUDA + CUTLASS + Go
│
├── table/
│   └── make_table.py      # Renderer; takes --results DIR --format {md,tex,csv}
│
├── Makefile               # Top-level: setup-cpu | setup-gpu | schemes
│
└── table1-birds-eye/      # Per-benchmark directory
    ├── Makefile           # measure-cpu | measure-gpu | table | table-latex | table-csv
    ├── README.md          # what this table measures
    ├── sweep.toml         # (scheme, DB) → shape parameters
    ├── results/           # runner JSONs land here
    └── out/               # rendered table1.{md,tex,csv}
```

## Usage

```bash
git clone --recursive <repository-url>
cd sandwichpir/rodeo

# One-time bootstrap
make setup-cpu           # (or setup-gpu depending on instance)
make schemes             # inits submodules + applies patches

# Run + render Table 1
make -C table1-birds-eye measure-cpu    # (or measure-gpu)
make -C table1-birds-eye table-latex    # → table1-birds-eye/out/table1.tex
```

## Cost model

See `configs/cost_model.toml`. Every benchmark computes `$/Mq` uniformly:
`$/Mq = compute + egress` where compute uses the row's instance and egress uses
`$0.09/GB` on response + downloaded-once bytes. Ingress is free.
