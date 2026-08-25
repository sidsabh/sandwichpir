#!/usr/bin/env python3
"""Runner: KsPIR (mmingluo/kspir @ 54c2f61, CCS'24 "Faster FHE-Based PIR").

Upstream's only end-to-end driver is tests/test-pir.cpp with the database size
hardcoded (r=16, 256 MB); patch 10 adds `test-pir <db_mb>` (r = db_mb/16),
mirroring the CLI that InsPIRe's evaluation harness (run-kspir.sh) drives.
Each invocation preprocesses the DB once and answers one query, so trials are
separate invocations; the preprocessing time doubles as the offline compute
measurement.

Communication is analytic, ported from InsPIRe's parse_kspir.py and matching
their published methodology: the artifact packs the response all the way to a
single ciphertext (8 KiB payload), and the 32 KiB record point (factor=4) stops
the packing tree log2(4)=2 levels early — 4 response ciphertexts, packing keys
ceil(log2(r/4)) instead of ceil(log2(r)), query and scan unchanged. The skipped
levels operate on <=4 ciphertexts, so measured server time is unaffected.
Stateless accounting (keys shipped with every query, no server-held client
state) counts keys in the per-query upload; offline comm is 0.

Memory: the CRT/NTT database image is 8 B per 16-bit plaintext coefficient
(datacrt in test-pir.cpp) = 4x the raw DB. 16 GiB DB -> 64 GiB resident: needs
r7i.4xlarge, not 2xlarge.

CLI: run_kspir.py <db_gib> [trials] [warmup]
"""

import re
import sys
import tempfile
from math import ceil, log2
from pathlib import Path

from lib import (RODEO_ROOT, die, dram_guard, emit_env_json, log, parse_num,
                 run_cmd, sink_result, stats)

# Scheme constants (src/params.h and InsPIRe's parse_kspir.py).
N = 4096
LOG_Q = 56
LOG_P = 16
K_KS = 4
K_RGSW = 2
N1 = 128
N2 = N // 2 // N1
FACTOR = 4  # response ciphertexts; record = FACTOR * 8 KiB = 32 KiB


def analytic_comm_bits(r: int) -> dict:
    num_keys = ceil(log2(r / FACTOR))
    key_packing = num_keys * K_KS * LOG_Q * N
    key_bsgs = N2 * K_KS * LOG_Q * N
    key_rgsw = K_RGSW * 2 * LOG_Q * N
    keys = key_packing + key_bsgs + key_rgsw
    query = N // 2 * LOG_Q
    response = FACTOR * 2 * N * LOG_Q
    return {"keys": keys, "query": query, "response": response}


def parse_times(raw_path: Path) -> dict:
    text = Path(raw_path).read_text(errors="replace")
    out = {}
    for key, pat in [
        ("prep_ms", r"server preprocessing costs (\d+) ms"),
        ("querygen_us", r"query costs (\d+) us"),
        ("online_us", r"online server response costs (\d+) us"),
        ("decrypt_us", r"decrypt costs (\d+) us"),
    ]:
        m = re.search(pat, text)
        if not m:
            die(f"missing '{pat}' in {raw_path}")
        out[key] = int(m.group(1))
    return out


def main():
    if len(sys.argv) < 2:
        die("usage: run_kspir.py <db_gib> [trials] [warmup]")
    db_gib = parse_num(sys.argv[1])
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    db_mb = int(db_gib * 1024)
    if db_mb % 16 != 0:
        die(f"db_mb={db_mb} must be a multiple of 16 (r = db_mb/16 blocks)")
    r = db_mb // 16

    scheme_dir = RODEO_ROOT / "schemes" / "kspir"
    if not scheme_dir.is_dir():
        die(f"{scheme_dir} missing")

    log(f"KsPIR: DB={db_gib}GiB (r={r}), record=32KiB (factor={FACTOR}), trials={trials}")
    # datacrt is 4x the raw DB (8 B/coeff), plus per-block staging buffers.
    dram_guard(db_gib * 4.0 + 0.5)

    bin_path = scheme_dir / "build" / "tests" / "test-pir"
    if not bin_path.exists():
        log("Building KsPIR (Release, clang++ + HEXL 1.2.5)")
        run_cmd(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
                tempfile.mktemp(), cwd=scheme_dir, what="KsPIR configure")
        run_cmd(["cmake", "--build", "build", "-j8", "--target", "test-pir"],
                tempfile.mktemp(), cwd=scheme_dir, what="KsPIR build")

    online_s, prep_s, querygen_ms, decrypt_ms = [], [], [], []
    for t in range(trials + warmup):
        raw = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
        run_cmd([bin_path, db_mb], raw.name, cwd=scheme_dir,
                what=f"KsPIR trial {t}")
        times = parse_times(raw.name)
        log(f"  trial {t}: prep={times['prep_ms']}ms online={times['online_us']}us")
        if t < warmup:
            continue
        online_s.append(times["online_us"] / 1e6)
        prep_s.append(times["prep_ms"] / 1e3)
        querygen_ms.append(times["querygen_us"] / 1e3)
        decrypt_ms.append(times["decrypt_us"] / 1e3)

    comm = analytic_comm_bits(r)
    keys_b = ceil(comm["keys"] / 8)
    query_b = ceil(comm["query"] / 8)
    response_b = ceil(comm["response"] / 8)

    result = {
        "scheme": "kspir",
        "instance": "cpu",
        "db_gib": db_gib,
        "record_bytes": FACTOR * N * LOG_P // 8,
        "trials": trials,
        "warmup": warmup,
        "server_seconds": stats(online_s),
        # Stateless accounting: keys travel with every query, no server-held state.
        "upload_bytes_per_query": keys_b + query_b,
        "download_bytes_per_query": response_b,
        "upload_bytes_once": 0,
        "download_bytes_once": 0,
        "upload_split_bytes": {"keys": keys_b, "query": query_b},
        # Per-DATABASE preprocessing (client-independent), not per-client work.
        "offline_seconds": stats(prep_s),
        "client_ms": {"querygen": stats(querygen_ms), "decrypt": stats(decrypt_ms)},
        "env": emit_env_json(scheme_dir),
        "config_notes": (
            f"factor={FACTOR} (32 KiB record, comm analytic per InsPIRe parse_kspir.py); "
            "THREADS_NUM=16 (upstream default); one query per invocation"
        ),
    }
    sink_result("kspir", db_gib, result)


if __name__ == "__main__":
    main()
