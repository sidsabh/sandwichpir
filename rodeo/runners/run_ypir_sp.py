#!/usr/bin/env python3
"""Runner: YPIR-SP (menonsamir/ypir @ a73e550).

Actual CLI (src/bin/run.rs): positional <num_items> <item_size_bits> [num_clients] [trials]
with -i for SP (SimplePIR) variant. Prints JSON on stdout.

CLI: run_ypir_sp.py <db_gib> <num_items> <item_size_bytes> [trials] [warmup]
"""

import json
import os
import sys
import tempfile

from lib import (RODEO_ROOT, RESULTS_DIR, die, dram_guard, emit_env_json,
                 extract_last_json_block, log, parse_num, run_cmd, sink_result,
                 stats, timestamp)


def main():
    if len(sys.argv) < 4:
        die("usage: run_ypir_sp.py <db_gib> <num_items> <record_bytes> [trials] [warmup]")
    db_gib = parse_num(sys.argv[1])
    num_items = int(sys.argv[2])
    record_bytes = int(sys.argv[3])
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    # YPIR requires AVX-512 (its scan kernel calls _mm512_* intrinsics and gates the
    # packing behind cfg(target_feature="avx512f")). Its own .cargo/config.toml asks
    # for target-cpu=native, but vendoring inside the sandwichpir tree means cargo
    # finds the parent's [target.x86_64].rustflags and REPLACES (not merges) ypir's
    # [build].rustflags, silently dropping target-cpu=native -> the non-AVX512
    # fallback path compiles and fails. Setting RUSTFLAGS in the env overrides both
    # config files. No change to upstream source or its spiral-rs pin.
    rustflags = {"RUSTFLAGS": os.environ.get("RUSTFLAGS", "") + " -C target-cpu=native"}

    scheme_dir = RODEO_ROOT / "schemes" / "ypir"
    if not scheme_dir.is_dir():
        die(f"{scheme_dir} missing")

    item_size_bits = record_bytes * 8
    total_trials = trials + warmup

    log(f"YPIR-SP: DB={db_gib}GiB, num_items={num_items}, item_bits={item_size_bits}, trials={trials}")
    # YPIR holds the DB plus the silent-preprocessing hint; budget 2x DB.
    dram_guard(db_gib * 2.0)

    bin_path = scheme_dir / "target" / "release" / "run"
    if not os.access(bin_path, os.X_OK):
        log("Building YPIR-SP (release, requires AVX-512 for perf)")
        run_cmd(["cargo", "build", "--release"], tempfile.mktemp(), cwd=scheme_dir,
                env=rustflags, what="YPIR-SP build")

    # Preserve the raw YPIR measurement JSON — it carries the phase split
    # (firstPassTimeMs = the SimplePIR* scan; ringPackingTimeMs = CDKS packing) and
    # the SimplePIR* comm fields, which we use to derive a SimplePIR* baseline row.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_out = RESULTS_DIR / f"ypir_sp-{db_gib}gib-{timestamp()}.raw.json"
    raw = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
    run_cmd([bin_path, num_items, item_size_bits, 1, total_trials, json_out, "-i"],
            raw.name, cwd=scheme_dir, env=rustflags, what="YPIR-SP bench")

    if not json_out.exists() or json_out.stat().st_size == 0:
        extract_last_json_block(raw.name, json_out)

    # YPIR's Measurement serialises as {offline: {...}, online: {...}} with
    # rename_all="camelCase" (src/measurement.rs); fields nest under `online`/`offline`.
    # Critical fields indexed strictly (fail loud); SimplePIR*/phase fields are
    # optional (.get) because they are a bonus and may be absent in some builds.
    try:
        d = json.loads(json_out.read_text())
        o = d["online"]
        off = d.get("offline", {})
        times_s = [t / 1000.0 for t in o["allServerTimesMs"][-trials:]]
        comm_up = o["uploadBytes"]
        comm_down = o["downloadBytes"]
    except (KeyError, ValueError) as e:
        die(f"Failed to parse YPIR JSON at {json_out}: {e}")
    first_pass_ms = o.get("firstPassTimeMs", 0)
    pack_ms = o.get("ringPackingTimeMs", 0)
    sp_query = o.get("simplepirQueryBytes", 0) or 0
    sp_resp = o.get("simplepirRespBytes", 0) or 0
    sp_hint = off.get("simplepirHintBytes", 0) or 0

    log(f"  {len(times_s)} trials, up={comm_up}B down={comm_down}B; "
        f"firstPass={first_pass_ms}ms pack={pack_ms}ms; "
        f"SimplePIR* hint={sp_hint}B resp={sp_resp}B")

    result = {
        "scheme": "ypir_sp",
        "instance": "cpu",
        "db_gib": db_gib,
        "ell1": num_items,
        "record_bytes": record_bytes,
        "trials": trials,
        "warmup": warmup,
        "server_seconds": stats(times_s),
        "upload_bytes_per_query": comm_up,
        "download_bytes_per_query": comm_down,
        "upload_bytes_once": 0,
        "download_bytes_once": 0,
        "env": emit_env_json(scheme_dir),
        "config_notes": "-i (SP variant); batch=1; positional CLI",
        # Phase split (firstPass = SimplePIR* scan, packing = CDKS): documents that
        # YPIR-SP server time is scan + CDKS packing, the latter dominating.
        "phase_ms": {"first_pass_scan": first_pass_ms, "ring_packing": pack_ms},
        # SimplePIR* baseline (the optimized SimplePIR YPIR reimplemented, README:137):
        # same scan, but raw hint egress instead of YPIR packing. Derive its row from here.
        "simplepir_star": {
            "server_ms": first_pass_ms,
            "upload_bytes_per_query": sp_query,
            "download_bytes_per_query": sp_resp,
            "download_bytes_once": sp_hint,
        },
    }
    sink_result("ypir_sp", db_gib, result)


if __name__ == "__main__":
    main()
