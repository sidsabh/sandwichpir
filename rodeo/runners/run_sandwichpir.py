#!/usr/bin/env python3
"""Runner: SandwichPIR (ours; sandwichpir/ parent repo).

Actual CLI (src/bin/run.rs): positional <num_items> <item_size_bits> [num_clients] [trials]
with -v for verbose. Prints pretty JSON to stdout on completion.

CLI: run_sandwichpir.py <db_gib> <num_items> <item_size_bytes> [trials] [warmup]
"""

import json
import os
import sys

from lib import (RODEO_ROOT, RESULTS_DIR, die, emit_env_json,
                 extract_last_json_block, log, parse_num, run_cmd, sink_result,
                 stats, timestamp)


def main():
    if len(sys.argv) < 4:
        die("usage: run_sandwichpir.py <db_gib> <num_items> <record_bytes> [trials] [warmup]")
    db_gib = parse_num(sys.argv[1])
    num_items = int(sys.argv[2])
    record_bytes = int(sys.argv[3])
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    # SandwichPIR is the parent repo (rodeo/.. == sandwichpir/).
    scheme_dir = RODEO_ROOT.parent
    if not (scheme_dir / "Cargo.toml").is_file():
        die(f"SandwichPIR Cargo.toml not found at {scheme_dir}")

    item_size_bits = record_bytes * 8
    total_trials = trials + warmup

    # Explicit sane defaults — scheme.rs reads HINT env var; "gemm" triggers the
    # symmetric TC-GEMM hint which has a known correctness collapse (see eval.tex).
    # Force NTT unless the caller explicitly overrides.
    hint = os.environ.get("HINT", "ntt")
    env = {"HINT": hint}

    log(f"SandwichPIR: DB={db_gib}GiB, num_items={num_items}, "
        f"item_bits={item_size_bits}, trials={trials}, HINT={hint}")

    bin_path = scheme_dir / "target" / "release" / "run"
    if not os.access(bin_path, os.X_OK):
        log("Building SandwichPIR (release, cuda feature, sm_89)")
        run_cmd(["cargo", "build", "--release", "--features", "cuda", "--bin", "run"],
                RESULTS_DIR / "sandwichpir-build.log", cwd=scheme_dir, env=env,
                what="SandwichPIR build")

    # Preserve raw log next to the result for post-hoc debug.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    raw = RESULTS_DIR / f"sandwichpir-{db_gib}gib-{ts}.raw"
    json_out = RESULTS_DIR / f"sandwichpir-{db_gib}gib-{ts}.parsed.json"

    run_cmd([bin_path, num_items, item_size_bits, 1, total_trials, json_out],
            raw, cwd=scheme_dir, env=env, what="SandwichPIR bench")

    # Parse the JSON output file. Fields per Measurement struct: allServerTimesMs,
    # uploadBytes, downloadBytes (camelCase serde), nested under `online`.
    if not json_out.exists() or json_out.stat().st_size == 0:
        # Fallback: JSON was also printed to stdout, extract last {...} block.
        extract_last_json_block(raw, json_out)

    d = json.loads(json_out.read_text())
    online = d.get("online") or {}
    times_ms = online.get("allServerTimesMs") or []
    # Drop warmup trials.
    times_s = [t / 1000.0 for t in times_ms[max(0, len(times_ms) - trials):]]
    comm_up = online.get("uploadBytes", 500000)
    comm_down = online.get("downloadBytes", 50000)

    result = {
        "scheme": "sandwichpir",
        "instance": "gpu",
        "db_gib": db_gib,
        "ell1": num_items,
        "record_bytes": record_bytes,
        "batch": 1,
        "trials": trials,
        "warmup": warmup,
        "server_seconds": stats(times_s),
        "upload_bytes_per_query": comm_up,
        "download_bytes_per_query": comm_down,
        "upload_bytes_once": 0,
        "download_bytes_once": 0,
        "env": emit_env_json(scheme_dir),
        "config_notes": "CUDA sm_89 INT8 TC; batch=1; positional CLI",
    }
    sink_result("sandwichpir", db_gib, result)


if __name__ == "__main__":
    main()
