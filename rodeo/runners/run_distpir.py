#!/usr/bin/env python3
"""Runner: DistPIR SimplePIR-on-GPU mode (ryanleh/crowdsurf @ v0.2).

Actual CLI (benches/benches.go):
  -rows <N> -cols <N> -q <32|64> -p <plaintext_mod> -bits <bits_per_element>
  -bench throughput -mode hybrid -batch <K>
Prints: "Throughput(rows x cols, k, p, N): X.YZ GB/s"

CUDACXX must be set at build time to enable GPU path (else CMake compiles ffi_noop).
Patch 00 (sm_89) applied via `make schemes`.

CLI: run_distpir.py <db_gib> <rows> <cols> [trials] [warmup]
Server time derived from throughput: server_s = db_gb / tput_gb_s.
"""

import os
import re
import shutil
import sys
from pathlib import Path

from lib import (RODEO_ROOT, RESULTS_DIR, die, emit_env_json, log, parse_num,
                 run_cmd, sink_result, stats, tail_file, timestamp)


def find_cudacxx() -> str:
    # /etc/profile.d/sandwichpir_cuda.sh may set CUDACXX to a stale path.
    # Re-derive if the current value doesn't point at a real file.
    cudacxx = os.environ.get("CUDACXX", "")
    if cudacxx and Path(cudacxx).exists():
        return cudacxx
    nvcc = shutil.which("nvcc")
    if not nvcc:
        for candidate in ("/usr/bin/nvcc", "/usr/local/cuda/bin/nvcc",
                          "/usr/lib/nvidia-cuda-toolkit/bin/nvcc"):
            if Path(candidate).exists():
                nvcc = candidate
                break
    if not nvcc:
        die("nvcc not found. Install CUDA toolkit (e.g. sudo apt install nvidia-cuda-toolkit).")
    assert nvcc is not None
    return nvcc


def main():
    if len(sys.argv) < 4:
        die("usage: run_distpir.py <db_gib> <rows> <cols> [trials] [warmup]")
    db_gib = parse_num(sys.argv[1])
    rows = int(sys.argv[2])
    cols = int(sys.argv[3])
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    # warmup (argv[5]) accepted for CLI parity; patch 03's fixed loop handles it.

    scheme_dir = RODEO_ROOT / "schemes" / "distpir"
    if not scheme_dir.is_dir():
        die(f"{scheme_dir} missing")

    log(f"DistPIR-SPGPU: DB={db_gib}GiB, rows={rows}, cols={cols}, trials={trials}")

    cudacxx = find_cudacxx()
    log(f"Using CUDACXX={cudacxx}")
    cuda_home = str(Path(cudacxx).parent.parent)
    env = {
        "CUDACXX": cudacxx,
        "CUDAHOSTCXX": os.environ.get("CUDAHOSTCXX") or shutil.which("g++-13")
                       or shutil.which("g++") or "",
        "CUDA_HOME": os.environ.get("CUDA_HOME", cuda_home),
        "LD_LIBRARY_PATH": f"{os.environ.get('LD_LIBRARY_PATH', '')}:{cuda_home}/lib64",
    }

    bin_path = scheme_dir / "benches" / "benches"

    # CrowdSurf's Go bench depends on:
    #   - SEAL (external/SEAL) — cgo #include <seal/seal.h>
    #   - CUDA GEMM lib (matrix/gpu/cuda) — dlopen'd
    # Both must be built + installed before `go build` will succeed.
    seal_header = (scheme_dir / "external" / "SEAL" / "build" / "include"
                   / "SEAL-4.1" / "seal" / "seal.h")
    if not seal_header.is_file():
        log("Building Microsoft SEAL (one-time, ~2 min)")
        build = scheme_dir / "external" / "SEAL" / "build"
        blog = RESULTS_DIR / "distpir-seal-build.log"
        run_cmd(["cmake", "-S", "external/SEAL", "-B", build,
                 f"-DCMAKE_INSTALL_PREFIX={build}",
                 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                 "-DSEAL_USE_INTEL_HEXL=ON"],
                blog, cwd=scheme_dir, env=env, what="SEAL configure")
        run_cmd(["cmake", "--build", build, "-j", str(os.cpu_count())],
                blog, cwd=scheme_dir, env=env, what="SEAL build")
        run_cmd(["cmake", "--install", build],
                blog, cwd=scheme_dir, env=env, what="SEAL install")

    if not (scheme_dir / "matrix" / "gpu" / "cuda" / "build" / "libgpu.so").is_file():
        log("Building CrowdSurf CUDA GEMM lib (one-time)")
        blog = RESULTS_DIR / "distpir-gpu-build.log"
        run_cmd(["cmake", "-S", "matrix/gpu/cuda", "-B", "matrix/gpu/cuda/build",
                 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
                blog, cwd=scheme_dir, env=env, what="CUDA GEMM configure")
        run_cmd(["cmake", "--build", "matrix/gpu/cuda/build", "-j", str(os.cpu_count())],
                blog, cwd=scheme_dir, env=env, what="CUDA GEMM build")

    if not os.access(bin_path, os.X_OK):
        log("Building CrowdSurf benches (Go + cgo + CUDA)")
        run_cmd(["go", "build", "-o", "benches", "."],
                RESULTS_DIR / "distpir-go-build.log", cwd=scheme_dir / "benches",
                env=env, what="Go build")

    # Patch 03 makes the bench run a fixed 10-trial loop and print each trial's time.
    # bits = 8, p = 256 → each Z_p element is 1 byte; DB = rows × cols bytes.
    bits, pmod = 8, 256

    expected_bytes = rows * cols * bits // 8
    log(f"  (expected DB = {expected_bytes} bytes = {expected_bytes // 1024**3} GiB)")

    # Save raw log next to the result for post-hoc debug (not cleaned up).
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = RESULTS_DIR / f"distpir_spgpu-{db_gib}gib-{timestamp()}.raw"

    run_cmd(["./benches", "-bench", "throughput", "-mode", "hybrid",
             "-rows", rows, "-cols", cols, "-q", 32, "-p", pmod,
             "-bits", bits, "-batch", 1],
            raw, cwd=scheme_dir / "benches", env=env, what="CrowdSurf bench")

    # Patch 03 emits: "TrialsMs(rows x cols, k, p, N): t1, t2, ..., tN"
    # giving per-trial answer times in ms (10 trials by default).
    text = raw.read_text(errors="replace")
    m = re.search(r"TrialsMs\([^)]+\):\s+(.+)", text)
    if m:
        server_s = [float(x) / 1000.0
                    for x in m.group(1).replace(",", " ").split() if x.strip()]
        log(f"  extracted {len(server_s)} per-trial times")
    else:
        log("Failed to extract per-trial times. Falling back to Throughput line.")
        mt = re.search(r"Throughput\([^)]+\):\s+([0-9.]+)", text)
        if not mt or float(mt.group(1)) <= 0:
            tail_file(raw, 30)
            die("Cannot parse DistPIR output")
        assert mt is not None
        db_gb = db_gib * 1024 ** 3 / 1e9
        server_s = [db_gb / float(mt.group(1))]
        log(f"  derived server time (single) = {server_s[0]} s")

    # SimplePIR shape: query = cols × 4 B, response = rows × 4 B, hint = rows × n_lwe × 4 B.
    # CrowdSurf's secretDims[logq=32] = 2048 (crypto/constants.go), NOT 1024.
    # Earlier draft used n=1024 which halved the hint size and understated DistPIR's egress cost.
    lwe_dim = 2048
    upload = cols * 4
    download = rows * 4
    hint = rows * lwe_dim * 4

    st = stats(server_s)
    result = {
        "scheme": "distpir_spgpu",
        "instance": "gpu",
        "db_gib": db_gib,
        "ell1": rows,
        "ell2": cols,
        "record_bytes": cols,
        "trials": len(st["raw"]),
        "warmup": 1,
        "server_seconds": st,
        "upload_bytes_per_query": upload,
        "download_bytes_per_query": download,
        "upload_bytes_once": 0,
        "download_bytes_once": hint,
        "env": emit_env_json(scheme_dir),
        "config_notes": ("bench=throughput mode=hybrid batch=1 bits=8 p=256; "
                         "patches 00 (sm_89), 02 (db dedup), 03 (batch flag + 10-trial loop)"),
    }
    sink_result("distpir_spgpu", db_gib, result)


if __name__ == "__main__":
    main()
