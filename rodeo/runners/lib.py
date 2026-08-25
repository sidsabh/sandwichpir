"""Common helpers imported by every runner.

Python port of lib.sh. Same contracts: results land in RESULTS_DIR as
{scheme}-{db}gib-{timestamp}.json, logs go to stderr, and dram_guard dies
BEFORE allocating.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

RODEO_ROOT = Path(os.environ.get("RODEO_ROOT", Path(__file__).resolve().parent.parent))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", RODEO_ROOT / "results"))


def parse_num(s):
    """Parse a CLI number, keeping integers integral so result filenames match
    the shell runners' ({scheme}-1gib-..., not 1.0gib)."""
    f = float(s)
    return int(f) if f == int(f) else f


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(*args) -> None:
    print(f"[{utcnow()}]", *args, file=sys.stderr)


def die(*args) -> NoReturn:
    log("ERROR:", *args)
    sys.exit(1)


def total_ram_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def dram_guard(needed_gib: float) -> None:
    """Refuse to run if the estimated peak footprint exceeds a safe fraction of
    physical RAM, and die BEFORE allocating.

    Rationale: a scheme whose working set exceeds RAM does not fail cleanly -- it
    drives the box into swap and hangs the whole instance (this killed the c7i
    overnight at the 16 GiB HintlessPIR row). Exiting here means the overnight
    harness records a clean FAIL and the box survives to run the next scheme.
    Override the ceiling with DRAM_GUARD_FRACTION=0.95 (or DRAM_GUARD_OFF=1 to
    disable entirely) if you know the box has swap headroom you're willing to use.
    """
    if os.environ.get("DRAM_GUARD_OFF", "0") == "1":
        return
    frac = float(os.environ.get("DRAM_GUARD_FRACTION", "0.85"))
    total = total_ram_gib()
    if needed_gib > frac * total:
        die(
            f"estimated peak {needed_gib} GiB exceeds {frac} x {total} GiB RAM. "
            "Refusing to run (would swap-kill the box). Use a bigger instance, "
            "or set DRAM_GUARD_OFF=1 to override."
        )
    log(f"  dram_guard: {needed_gib} GiB estimated peak vs {total} GiB RAM — OK")


def run_cmd(cmd, raw_path: Path, cwd=None, env=None, what: str = "bench", tail: int = 40):
    """Run a command with stdout+stderr captured to raw_path; die with the log
    tail on failure. env, when given, is merged over os.environ."""
    full_env = dict(os.environ, **{k: str(v) for k, v in env.items()}) if env else None
    with open(raw_path, "w") as raw:
        rc = subprocess.call(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=full_env,
            stdout=raw, stderr=subprocess.STDOUT,
        )
    if rc != 0:
        log(f"{what} failed (exit {rc}). Raw log at {raw_path}")
        tail_file(raw_path, tail)
        sys.exit(1)


def tail_file(path: Path, n: int = 40) -> None:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()[-n:]
        print("\n".join(lines), file=sys.stderr)
    except OSError:
        pass


def extract_last_json_block(raw_path: Path, out_path: Path) -> None:
    """Fallback when a scheme prints its report JSON to stdout instead of the
    requested output file: extract the last {...} block from the raw log."""
    raw = Path(raw_path).read_text(errors="replace")
    blocks = re.findall(r"\{.*?\}", raw, re.DOTALL)
    Path(out_path).write_text(blocks[-1] if blocks else "{}")


def which(name: str):
    return shutil.which(name)


def _run_out(cmd, **kw) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=kw.pop("timeout", 5), **kw
        ).stdout.strip()
    except Exception:
        return ""


def emit_env_json(scheme_dir) -> dict:
    hostname = _run_out(["hostname"]) or "unknown"
    instance_type = _run_out(
        ["curl", "-sf", "--max-time", "1",
         "http://169.254.169.254/latest/meta-data/instance-type"]
    ) or "local"
    cpu_model, avx512 = "unknown", False
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        if m:
            cpu_model = m.group(1).strip()
        avx512 = "avx512" in cpuinfo
    except OSError:
        pass
    gpu_model, compute_cap, vram_gib = "", "", 0.0
    if which("nvidia-smi"):
        gpu_model = _run_out(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).splitlines()[:1]
        gpu_model = gpu_model[0] if gpu_model else ""
        compute_cap = _run_out(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        ).splitlines()[:1]
        compute_cap = compute_cap[0] if compute_cap else ""
        if gpu_model:
            mem = _run_out(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"]
            ).splitlines()[:1]
            try:
                vram_gib = round(float(mem[0]) / 1024, 1)
            except (IndexError, ValueError):
                pass

    def git_head(path) -> str:
        return _run_out(["git", "-C", str(path), "rev-parse", "HEAD"]) or "unknown"

    return {
        "hostname": hostname,
        "instance_type": instance_type,
        "cpu_model": cpu_model,
        "cpu_avx512": avx512,
        "gpu_model": gpu_model,
        "compute_cap": compute_cap,
        "ram_gib": round(total_ram_gib(), 1),
        "vram_gib": vram_gib,
        "commit_scheme": git_head(scheme_dir),
        "commit_rodeo": git_head(RODEO_ROOT / ".."),
        "timestamp_utc": utcnow(),
    }


def stats(values) -> dict:
    """Mean/median/stddev over per-trial values. (The shell version's stdin-vs-
    heredoc trap that silently zeroed measurements does not exist here: values
    are passed as a list, never piped.)"""
    raw = [float(x) for x in values]
    if not raw:
        return {"mean": 0, "median": 0, "stddev": 0, "raw": []}
    return {
        "mean": statistics.mean(raw),
        "median": statistics.median(raw),
        "stddev": statistics.stdev(raw) if len(raw) > 1 else 0.0,
        "raw": raw,
    }


def sink_result(scheme: str, db_gib, body: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{scheme}-{db_gib}gib-{timestamp()}.json"
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    log(f"Wrote {out}")
    print(out)
    return out
