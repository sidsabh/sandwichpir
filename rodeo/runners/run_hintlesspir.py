#!/usr/bin/env python3
"""Runner: HintlessPIR (google/hintless_pir @ 49434e0).

CLI: run_hintlesspir.py <db_gib> <num_items> <record_bytes> [trials] [warmup]
  (same argument order as the other SimplePIR-family runners)

Shape mapping — verified against hintless_simplepir/hintless_simplepir_benchmarks.cc:
  * Records are hardcoded to 8 bits (`db_record_bit_size = 8`), and the DB holds
    `num_rows * num_cols` of them, so DB bytes = num_rows * num_cols.
  * A SimplePIR response is one full column, i.e. `num_rows` entries. Their client
    extracts a single byte from it, but the *server* work is identical to that of a
    scheme with `num_rows`-byte records. So to time a 32 KiB-record configuration we
    set num_rows = record_bytes and let num_cols carry the database size.
  => num_rows = record_bytes,  num_cols = num_items

Two traps in the upstream benchmark:
  * main() sets FLAGS_benchmark_filter = "" and only runs when it is non-empty, so
    the binary EXITS SILENTLY WITH STATUS 0 if --benchmark_filter is missing or does
    not match. The one benchmark is BM_HintlessPirRlwe64.
  * server->Preprocess() sits outside the timed loop, so the reported time is online
    only. Preprocessing is ~3-4 min/GiB single-threaded and is not captured here.
"""

import json
import os
import sys

from lib import (RODEO_ROOT, RESULTS_DIR, die, dram_guard, emit_env_json, log,
                 parse_num, run_cmd, sink_result, stats, tail_file, timestamp)


def main():
    if len(sys.argv) < 4:
        die("usage: run_hintlesspir.py <db_gib> <num_items> <record_bytes> [trials] [warmup]")
    db_gib = parse_num(sys.argv[1])
    num_items = int(sys.argv[2])
    record_bytes = int(sys.argv[3])
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    scheme_dir = RODEO_ROOT / "schemes" / "hintlesspir"
    if not scheme_dir.is_dir():
        die(f"{scheme_dir} missing")

    num_rows = record_bytes
    num_cols = num_items

    # Sanity: rows*cols must equal the requested database size.
    expected = int(db_gib * 1024 ** 3)
    if num_rows * num_cols != expected:
        die(f"shape mismatch: rows*cols={num_rows * num_cols} but {db_gib}GiB={expected}")

    log(f"HintlessPIR: DB={db_gib}GiB, num_rows={num_rows} (=record bytes), "
        f"num_cols={num_cols}, trials={trials}")
    # HintlessPIR holds the DB (~1x) plus the LinPIR hint + NTT working set during
    # preprocessing; empirically peaks near 1.6x DB, and the 16 GiB row swap-killed
    # a 15 GiB box. Budget 2.5x for safety.
    dram_guard(db_gib * 2.5)

    bazel_target = "//hintless_simplepir:hintless_simplepir_benchmarks"
    bin_path = scheme_dir / "bazel-bin" / "hintless_simplepir" / "hintless_simplepir_benchmarks"

    if not os.access(bin_path, os.X_OK):
        log("Building HintlessPIR (bazel, -c opt -march=native; first build pulls deps, ~10 min)")
        run_cmd(["bazel", "build", "-c", "opt", "--copt=-march=native", bazel_target],
                RESULTS_DIR / "hintlesspir-build.log", cwd=scheme_dir,
                what="bazel build")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = RESULTS_DIR / f"hintlesspir-{db_gib}gib-{timestamp()}.raw"

    # LinPIR block size. The response carries one RLWE ciphertext per block, where
    # blocks = ceil(db_rows / rows_per_block), so this is a communication/computation
    # knob: doubling it halves the download but doubles the rotations (and their
    # key-switching noise). Upstream defaults to 1024, which is what the HintlessPIR
    # paper's evaluation uses and what google/hintless_pir's README prescribes -- but
    # that leaves the download at 5.78 MiB here, whereas 2048 measures 2.97 MiB for
    # +41% server time, a strictly lower total $/Mq and a match for the response size
    # the paper itself reports. 2048 is the structural maximum (n/2 at log_n=12);
    # 4096 and above fail. Override with ROWS_PER_BLOCK to reproduce either point.
    rows_per_block = int(os.environ.get("ROWS_PER_BLOCK", "2048"))
    blocks = (num_rows + rows_per_block - 1) // rows_per_block
    log(f"  LinPIR rows_per_block={rows_per_block} (blocks={blocks})")
    log("  running (preprocessing is untimed and slow: ~3-4 min/GiB)")
    run_cmd([bin_path,
             "--benchmark_filter=BM_HintlessPirRlwe64",
             f"--benchmark_repetitions={trials}",
             "--benchmark_format=json",
             f"--num_rows={num_rows}",
             f"--num_cols={num_cols}",
             f"--rows_per_block={rows_per_block}"],
            raw, what="HintlessPIR bench", tail=30)

    # google-benchmark JSON: benchmarks[] holds one entry per repetition plus aggregate
    # rows (mean/median/stddev) tagged with aggregate_name. Keep only the raw repetitions.
    try:
        d = json.loads(raw.read_text())
    except ValueError:
        log(f"Failed to parse benchmark JSON. Raw log at {raw}")
        tail_file(raw, 30)
        sys.exit(1)
    runs = [b for b in d.get("benchmarks", []) if "aggregate_name" not in b]
    if not runs:
        log(f"no benchmark runs in output. Raw log at {raw}")
        tail_file(raw, 30)
        sys.exit(1)
    scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    times_s = [b["real_time"] * scale[b.get("time_unit", "ns")] for b in runs]
    log(f"  extracted {len(times_s)} per-repetition times")

    # Communication is measured, not cited: patches/08 adds ByteSizeLong() counters to
    # upstream's benchmark (it emits no byte counts of its own). The request bundles
    # the one-time LinPIR Galois key with the per-query ciphertexts, so the patch sizes
    # that field separately -- hence a real offline/online split. Fail loudly if the
    # counters are absent, which means the binary was built without the patch.
    have = [b for b in runs if "online_down_bytes" in b]
    if not have:
        die("no communication counters in benchmark output: rebuild with "
            "rodeo/patches/08-hintlesspir-measure-comm.patch applied")
    b = have[0]
    upload_once = int(b["offline_up_bytes"])
    download_once = int(b["offline_down_bytes"])
    upload = int(b["online_up_bytes"])
    download = int(b["online_down_bytes"])
    log(f"  comm: offline up={upload_once}B down={download_once}B; "
        f"online up={upload}B down={download}B")

    st = stats(times_s)
    result = {
        "scheme": "hintlesspir",
        "instance": "cpu",
        "db_gib": db_gib,
        "ell1": num_rows,
        "ell2": num_cols,
        "record_bytes": record_bytes,
        "trials": len(st["raw"]),
        "warmup": warmup,
        "server_seconds": st,
        "upload_bytes_per_query": upload,
        "download_bytes_per_query": download,
        "upload_bytes_once": upload_once,
        "download_bytes_once": download_once,
        "env": emit_env_json(scheme_dir),
        "config_notes": (
            "BM_HintlessPirRlwe64; num_rows=record_bytes column shaping; "
            f"rows_per_block={rows_per_block}; online only (Preprocess untimed); "
            "comm measured via patch 08 counters"
        ),
    }
    sink_result("hintlesspir", db_gib, result)


if __name__ == "__main__":
    main()
