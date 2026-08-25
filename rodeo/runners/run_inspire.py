#!/usr/bin/env python3
"""Runner: InsPIRe compute-optimal (google/private-membership @ 037b451, subdir research/InsPIRe).

CLI: run_inspire.py <db_gib> <num_items> <record_bytes> [trials] [warmup] [dim0] [interp_degree]

Verified against the repo's own eval script (evaluation/run-inspire.sh):
  * The InsPIRe benchmark is the `inspire` binary (NOT `run`, which is the
    YPIR-style CLI). Flags are hyphenated:
      --num-items --item-size-bits --dim0 --interpolate-degree
      --trials --out-report-json --label
  * `--dim0` is the SimplePIR first-dimension size and `--interpolate-degree`
    is the RGSW Horner-eval degree. Together they are the compute-opt <-> comm-opt
    knob (D-001). Compute-optimal = degree 1 (no extra interpolation, least
    server compute). An empirical dim0 sweep at 1 GiB was flat (<2% across
    2^12..2^15), so we default dim0 = num_items (the natural largest).
  * Measurement JSON nests under `online` (rename_all="camelCase").

Requires AVX-512; RUSTFLAGS forces target-cpu=native (see run_ypir_sp.py for why).
"""

import json
import os
import sys

from lib import (RODEO_ROOT, RESULTS_DIR, die, dram_guard, emit_env_json, log,
                 parse_num, run_cmd, sink_result, stats, tail_file, timestamp)


def main():
    if len(sys.argv) < 4:
        die("usage: run_inspire.py <db_gib> <num_items> <record_bytes> "
            "[trials] [warmup] [dim0] [interp_degree]")
    db_gib = parse_num(sys.argv[1])
    num_items = int(sys.argv[2])
    record_bytes = int(sys.argv[3])
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    dim0 = int(sys.argv[6]) if len(sys.argv) > 6 else num_items  # default: largest (compute-optimal)
    interp = int(sys.argv[7]) if len(sys.argv) > 7 else 1        # default: degree 1 (compute-optimal)

    rustflags = {"RUSTFLAGS": os.environ.get("RUSTFLAGS", "") + " -C target-cpu=native"}

    scheme_dir = RODEO_ROOT / "schemes" / "inspire" / "research" / "InsPIRe"
    if not scheme_dir.is_dir():
        die(f"{scheme_dir} missing")

    item_size_bits = record_bytes * 8
    total_trials = trials + warmup

    log(f"InsPIRe c-opt: DB={db_gib}GiB, num_items={num_items}, "
        f"item_bits={item_size_bits}, dim0={dim0}, interp={interp}, trials={trials}")
    # Peak is the encoded database buffer: db_rows*db_cols coefficients at 2 bytes
    # each (the implementation stores each R_p coefficient in a u16 regardless of the
    # plaintext modulus width), i.e. 2x the logical database. Patch 09 removes the
    # second copy the benchmark used to materialise before handing it to the
    # constructor, which had made peak 4x and put 16 GiB out of reach on a 64 GiB host.
    dram_guard(db_gib * 2.5)

    bin_path = scheme_dir / "target" / "release" / "inspire"
    if not os.access(bin_path, os.X_OK):
        log("Building InsPIRe (release --bin inspire, target-cpu=native)")
        run_cmd(["cargo", "build", "--release", "--bin", "inspire"],
                RESULTS_DIR / "inspire-build.log", cwd=scheme_dir, env=rustflags,
                what="InsPIRe build")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_out = RESULTS_DIR / f"inspire_copt-{db_gib}gib-{timestamp()}.raw.json"
    raw = RESULTS_DIR / f"inspire_copt-{db_gib}gib-{timestamp()}.raw"

    run_cmd([bin_path,
             "--num-items", num_items,
             "--item-size-bits", item_size_bits,
             "--dim0", dim0,
             "--interpolate-degree", interp,
             "--trials", total_trials,
             "--out-report-json", json_out],
            raw, cwd=scheme_dir, env=rustflags, what="InsPIRe bench")

    # online nests everything; upload may be split uploadKeys+uploadQuery or a single
    # uploadBytes depending on build. allServerTimesMs may be absent -> fall back to
    # the scalar serverTimeMs. Strict on the field we cannot do without (a server time).
    try:
        d = json.loads(json_out.read_text())
        o = d["online"]
    except (OSError, ValueError, KeyError) as e:
        log(f"Failed to parse InsPIRe JSON at {json_out}: {e}")
        tail_file(raw, 40)
        sys.exit(1)
    times = o.get("allServerTimesMs")
    if times:
        times = times[-trials:]
    else:
        st = o.get("serverTimeMs")
        if st is None:
            die("no server time in InsPIRe JSON")
        times = [st]
    times_s = [t / 1000.0 for t in times]
    # Report the one-time key material separately from the per-query upload: the
    # keys are uploaded once per client and amortized, so folding them into the
    # per-query figure (as an earlier version did) overstates online communication.
    keys = o.get("uploadKeys", 0)
    query = o.get("uploadQuery", 0)
    if not (keys or query):
        keys, query = 0, o.get("uploadBytes", 0)
    download = o["downloadBytes"]

    log(f"  {len(times_s)} trials, up/q={query}B down/q={download}B keys(once)={keys}B")

    result = {
        "scheme": "inspire_copt",
        "instance": "cpu",
        "db_gib": db_gib,
        "ell1": num_items,
        "record_bytes": record_bytes,
        "trials": trials,
        "warmup": warmup,
        "server_seconds": stats(times_s),
        "upload_bytes_per_query": query,
        "download_bytes_per_query": download,
        "upload_bytes_once": keys,
        "download_bytes_once": 0,
        "env": emit_env_json(scheme_dir),
        "config_notes": (f"inspire binary; compute-optimal (interpolate-degree={interp}, "
                         f"dim0={dim0}); batch=1"),
    }
    sink_result("inspire_copt", db_gib, result)


if __name__ == "__main__":
    main()
