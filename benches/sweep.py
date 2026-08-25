#!/usr/bin/env python3
"""
Sweep SandwichPIR end-to-end throughput across DB sizes and batch sizes.

For each (DB config, batch size) pair, invokes:
  cargo run --release --features cuda --bin run -- NUM_ITEMS ITEM_SIZE_BITS NUM_CLIENTS TRIALS OUT.json

The binary emits the full Measurement struct (offline + online times, bytes,
per-phase breakdown, per-trial variance) as JSON. We collect them into one
aggregate file with sweep metadata.

Usage:
    # full sweep, single GPU, default output
    python scripts/sweep.py

    # A100 cluster, 3-GPU column sharding
    MULTI_GPU=3 python scripts/sweep.py --out sweep_a100_3gpu.json

    # subset: only 4 GiB, batches 1/64/1024
    python scripts/sweep.py --configs 4gib --batches 1,64,1024

Env vars forwarded to the binary:
    MULTI_GPU        column-shard across N GPUs (default 1)
    YPIR_VERBOSE     1 to enable CUDA-event timing inside the binary
    HINT             "ntt" (default) or "gemm" for the offline hint kernel

The binary's own JSON output is not shape-matched to other sweeps; we wrap it
with sweep-level metadata so downstream plotting can filter by (db_gib, batch).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
CONFIGS = {
    # name: (num_items, item_size_bits, db_gib, ell_1, ell_2, rho)
    "0.25gib": (16384,  131072,  0.25, 16384, 16384,   8),  # square, matches DistPIR fig3 low end
    "1gib":    (32768,  262144,  1.0,  32768, 32768,  16),  # square
    "4gib":    (65536,  524288,  4.0,  65536, 65536,  32),  # square, matches DistPIR 4 GiB
    "8gib":    (65536, 1048576,  8.0,  65536, 131072, 64),  # Wikipedia shape
}

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_TRIALS = 10

# Per-GPU per-config max batch. Combinations exceeding the cap are skipped
# entirely (no failed-row left in the JSON). Caps were calibrated empirically
# from observed OOMs (single-GPU only — multi-GPU sharding lifts these).
# Match GPU by substring of `meta.gpu`. Order = priority; first match wins.
MAX_BATCH_SINGLE_GPU = [
    # (gpu_substr, {config_name: max_batch})
    ("L4",   {"4gib": 1024, "8gib": 512}),
    ("A100", {"8gib": 1024}),
]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------
def gpu_label() -> str:
    """Best-effort GPU identification via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not names:
            return "unknown"
        # Usually homogeneous on a single host; prefix count if not.
        if len(set(names)) == 1:
            return f"{len(names)}x {names[0]}" if len(names) > 1 else names[0]
        return "; ".join(names)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "no-nvidia-smi"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def parse_verbose_stdout(text: str) -> dict:
    """Pull per-phase breakdowns out of VERBOSE=1 stdout.

    The binary prints one block per GPU shard. Fields we extract:
      - hint_shards:   [{ntt_mul_acc_ms, barrett_intt_ms, total_ms}]
      - precomp_shards:[{prep_ms, phase1_ms, phase2_ms, total_ms}]
      - tc_packing_init_shards: [{rows, cols, mib, init_ms}]
      - online_shards: [{packing: {...}, matmul: {...}, post_ms,
                         compute_ms, download_ms, total_ms}]
      - memory_mib:    {monomials, a_ct, w_all, r_all, bold_t} (first shard)
      - throughput:    {hint_gib_s, offline_gib_s, online_gib_s}
    """
    out: dict = {}

    # Hint
    hint_rows = []
    for m in re.finditer(
        r"ntt\+mul\+acc=([\d.]+)\s+barrett\+intt=([\d.]+)\s+total=([\d.]+)\s*ms",
        text,
    ):
        hint_rows.append({
            "ntt_mul_acc_ms": float(m.group(1)),
            "barrett_intt_ms": float(m.group(2)),
            "total_ms": float(m.group(3)),
        })
    if hint_rows:
        out["hint_shards"] = hint_rows

    # InspiRING precomp summary line
    precomp_rows = []
    for m in re.finditer(
        r"InspiRING GPU precomp: prep=([\d.]+), phase1=([\d.]+), "
        r"phase2=([\d.]+), total=([\d.]+)\s*ms",
        text,
    ):
        precomp_rows.append({
            "prep_ms": float(m.group(1)),
            "phase1_ms": float(m.group(2)),
            "phase2_ms": float(m.group(3)),
            "total_ms": float(m.group(4)),
        })
    if precomp_rows:
        out["precomp_shards"] = precomp_rows

    # Tensor-core packing init: "M [R × C] = S MiB" and "done in X ms"
    tc_rows = []
    init_re = re.compile(
        r"SwTcPacking init: M \[(\d+)\s*[\u00d7x]\s*(\d+)\]\s*=\s*([\d.]+)\s*MiB"
    )
    done_re = re.compile(r"SwTcPacking init done in ([\d.]+)\s*ms")
    inits = init_re.findall(text)
    dones = done_re.findall(text)
    for (rows, cols, mib), init_ms in zip(inits, dones):
        tc_rows.append({
            "rows": int(rows), "cols": int(cols),
            "mib": float(mib), "init_ms": float(init_ms),
        })
    if tc_rows:
        out["tc_packing_init_shards"] = tc_rows

    # Memory line (may be printed twice in multi-GPU; capture first)
    mem_m = re.search(
        r"monomials=([\d.]+)\s*MiB,\s*a_ct=([\d.]+)\s*MiB,\s*"
        r"w_all=([\d.]+)\s*MiB,\s*r_all=([\d.]+)\s*MiB,\s*"
        r"bold_t=([\d.]+)\s*MiB",
        text,
    )
    if mem_m:
        out["memory_mib"] = {
            "monomials": float(mem_m.group(1)),
            "a_ct": float(mem_m.group(2)),
            "w_all": float(mem_m.group(3)),
            "r_all": float(mem_m.group(4)),
            "bold_t": float(mem_m.group(5)),
        }

    # Online breakdown: multi-line block per shard.
    online_rows = []
    block_re = re.compile(
        r"SW online breakdown \((\d+) clients\):\s*\n"
        r"\s*packing\(Ydecomp=([\d.]+)\s+gemm=([\d.]+)\s+accum=([\d.]+)\s+Zfinal=([\d.]+)\)=([\d.]+)\s*\n"
        r"\s*matmul\(gemm=([\d.]+)\s+accum=([\d.]+)\)=([\d.]+)\s+post=([\d.]+)\s*ms\s*\n"
        r"\s*wall:\s*compute=([\d.]+)\s+download=([\d.]+)\s+total=([\d.]+)\s*ms",
    )
    for m in block_re.finditer(text):
        online_rows.append({
            "clients": int(m.group(1)),
            "packing": {
                "y_decomp_ms": float(m.group(2)),
                "gemm_ms": float(m.group(3)),
                "accum_ms": float(m.group(4)),
                "z_final_ms": float(m.group(5)),
                "total_ms": float(m.group(6)),
            },
            "matmul": {
                "gemm_ms": float(m.group(7)),
                "accum_ms": float(m.group(8)),
                "total_ms": float(m.group(9)),
            },
            "post_ms": float(m.group(10)),
            "compute_ms": float(m.group(11)),
            "download_ms": float(m.group(12)),
            "total_ms": float(m.group(13)),
        })
    if online_rows:
        out["online_shards"] = online_rows

    # Aggregate throughputs printed at the end (GiB/sec per the binary).
    tput = {}
    hm = re.search(r"Hint Throughput:\s*([\d.]+)\s*GiB/sec", text)
    if hm:
        tput["hint_gib_s"] = float(hm.group(1))
    om = re.search(r"Offline Throughput:\s*([\d.]+)\s*GiB/sec", text)
    if om:
        tput["offline_gib_s"] = float(om.group(1))
    on = re.search(r"Online Throughput:\s*([\d.]+)\s*GiB/sec", text)
    if on:
        tput["online_gib_s"] = float(on.group(1))
    if tput:
        out["throughput"] = tput

    return out


def run_one(num_items: int, item_size_bits: int, batch: int, trials: int,
            multi_gpu: int, echo_stdout: bool) -> dict:
    """Run the binary once; return {measurement, stdout, breakdown}."""
    with tempfile.NamedTemporaryFile(
        mode="r", suffix=".json", delete=False, dir=str(REPO / "target"),
    ) as tmp:
        out_path = tmp.name

    cmd = [
        "cargo", "run", "--release", "--features", "cuda", "--bin", "run", "--",
        str(num_items), str(item_size_bits),
        str(batch), str(trials), out_path,
    ]
    env = os.environ.copy()
    env["MULTI_GPU"] = str(multi_gpu)
    env["VERBOSE"] = "1"
    env.setdefault("YPIR_VERBOSE", "1")  # belt + suspenders

    print(f"    $ MULTI_GPU={multi_gpu} VERBOSE=1 {shlex.join(cmd)}", flush=True)
    # Verbose breakdown (`SW_LOG` in src/cuda/common/log.cuh) writes to stderr;
    # merge it into stdout so the parser sees it.
    proc = subprocess.run(
        cmd, cwd=str(REPO), env=env,
        check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    if echo_stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)

    with open(out_path) as f:
        measurement = json.load(f)
    os.unlink(out_path)

    breakdown = parse_verbose_stdout(proc.stdout)
    return {
        "measurement": measurement,
        "breakdown": breakdown,
        "stdout": proc.stdout,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--configs", default="0.25gib,1gib,4gib,8gib",
        help="comma-separated subset of {%s}" % ",".join(CONFIGS),
    )
    ap.add_argument(
        "--batches", default=",".join(map(str, DEFAULT_BATCHES)),
        help="comma-separated batch sizes (powers of two up to 2048)",
    )
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument(
        "--out", default=None,
        help=("output JSON path (default: benches/sweep_results/"
              "<gpu-slug>_mg<MULTI_GPU>.json)"),
    )
    ap.add_argument(
        "--skip-build", action="store_true",
        help="skip the initial `cargo build` warmup (per-run cargo still "
             "invokes the cached binary)",
    )
    ap.add_argument(
        "--verbose-binary", action="store_true",
        help="echo the binary's VERBOSE=1 stdout as it runs (always captured "
             "and stored either way)",
    )
    ap.add_argument(
        "--no-raw-stdout", action="store_true",
        help="don't store raw stdout in the output JSON (breakdown still "
             "stored; saves space on long sweeps)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="if --out already exists, load it and skip (config, batch) "
             "pairs that already completed; append remaining to the same file",
    )
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            ap.error(f"unknown config {c!r}; known: {list(CONFIGS)}")
    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    multi_gpu = int(os.environ.get("MULTI_GPU", "1"))

    # Build per-config max-batch cap. Multi-GPU lifts single-GPU memory limits,
    # so the cap only applies when MULTI_GPU=1.
    cap: dict[str, int] = {}
    if multi_gpu == 1:
        gpu_str = gpu_label()
        for substr, c in MAX_BATCH_SINGLE_GPU:
            if substr in gpu_str:
                cap = c
                break

    # Default output path — slug-safe and records MULTI_GPU.
    if args.out is None:
        gpu_slug = (
            gpu_label()
            .lower()
            .replace(" ", "_")
            .replace("/", "-")
            .replace(";", "-")
        )
        out_dir = REPO / "benches" / "sweep_results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{gpu_slug}_mg{multi_gpu}.json"
    else:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prebuild so per-run cargo invocations are near-instant.
    if not args.skip_build:
        print("# cargo build --release --features cuda --bin run", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "--features", "cuda", "--bin", "run"],
            cwd=str(REPO), check=True,
        )

    meta = {
        "gpu": gpu_label(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "multi_gpu": multi_gpu,
        "trials_per_config": args.trials,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "configs_requested": configs,
        "batches_requested": batches,
    }
    results: list[dict] = []
    completed: set[tuple[str, int]] = set()

    if args.resume and out_path.exists():
        with open(out_path) as f:
            prior = json.load(f)
        results = list(prior.get("results", []))
        completed = {
            (r["config"], r["batch"])
            for r in results
            if "error" not in r and "measurement" in r
        }
        print(f"# --resume: loaded {len(results)} prior rows from {out_path}, "
              f"{len(completed)} completed; skipping those.", flush=True)

    total = len(configs) * len(batches)
    done = 0
    print(f"# sweep: {len(configs)} configs x {len(batches)} batches "
          f"= {total} runs, MULTI_GPU={multi_gpu}")
    for cname in configs:
        num_items, item_size_bits, db_gib, ell_1, ell_2, rho = CONFIGS[cname]
        cname_cap = cap.get(cname)
        for batch in batches:
            done += 1
            if (cname, batch) in completed:
                print(f"[{done}/{total}] {cname} (B={batch}) — skip (resume)", flush=True)
                continue
            if cname_cap is not None and batch > cname_cap:
                print(f"[{done}/{total}] {cname} (B={batch}) — skip "
                      f"(over cap B<={cname_cap} for this GPU at MULTI_GPU=1)", flush=True)
                continue
            print(f"[{done}/{total}] {cname} (B={batch})", flush=True)
            try:
                got = run_one(
                    num_items, item_size_bits, batch, args.trials,
                    multi_gpu, args.verbose_binary,
                )
                row = {
                    "config": cname,
                    "num_items": num_items,
                    "item_size_bits": item_size_bits,
                    "db_gib": db_gib,
                    "ell_1": ell_1,
                    "ell_2": ell_2,
                    "rho": rho,
                    "batch": batch,
                    "measurement": got["measurement"],
                    "breakdown": got["breakdown"],
                }
                if not args.no_raw_stdout:
                    row["stdout"] = got["stdout"]
                results.append(row)
            except subprocess.CalledProcessError as e:
                print(f"    FAILED (exit {e.returncode}); continuing", file=sys.stderr)
                results.append({
                    "config": cname,
                    "num_items": num_items,
                    "item_size_bits": item_size_bits,
                    "db_gib": db_gib,
                    "ell_1": ell_1,
                    "ell_2": ell_2,
                    "rho": rho,
                    "batch": batch,
                    "error": f"exit {e.returncode}",
                })

            # Flush after every run — long sweeps can get preempted on HPC.
            with open(out_path, "w") as f:
                json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"# wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
