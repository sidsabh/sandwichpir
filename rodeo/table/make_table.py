#!/usr/bin/env python3
"""
Render Table 1 from result JSONs in ../results/.

Usage:
    make_table.py [--format latex|markdown|csv] [--out FILE] [--results DIR]

Reads:
    - configs/cost_model.toml — instance $/hr, egress $/GB
    - configs/sweep.toml      — expected rows (used to detect gaps)
    - harness/schema/result.schema.json — for validation
    - results/*.json          — one file per (scheme, db_gib)

Emits Table 1 in the requested format. Missing measurements show up as "—".
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import tomllib  # Requires Python 3.11+

RODEO_ROOT = Path(__file__).resolve().parent.parent

# ---- Load configs -----------------------------------------------------------

def load_cost_model():
    with open(RODEO_ROOT / "configs/cost_model.toml", "rb") as f:
        return tomllib.load(f)

def load_sweep():
    with open(RODEO_ROOT / "configs/sweep.toml", "rb") as f:
        return tomllib.load(f)

# ---- Load results -----------------------------------------------------------

def load_results(results_dir: Path):
    """
    Returns a dict keyed by (scheme, db_gib) → most-recent result JSON.
    Multiple files per key take the latest by mtime.
    """
    latest = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue
        key = (data["scheme"], float(data["db_gib"]))
        if key not in latest or path.stat().st_mtime > latest[key][0]:
            latest[key] = (path.stat().st_mtime, data)
    return {k: v[1] for k, v in latest.items()}

# ---- Derived columns --------------------------------------------------------

def throughput_gib_per_s(db_gib: float, server_seconds: float) -> float:
    if server_seconds <= 0:
        return 0.0
    return db_gib / server_seconds

def cost_per_mq(row: dict, instance_hourly: float, egress_usd_per_gb: float) -> dict:
    """
    $/Mq = compute + egress.
      compute = server_seconds * hourly * 1e6 / 3600
      egress  = (download_per_query + download_once) * egress * 1e6 / 1e9
    """
    server = row["server_seconds"]["mean"]
    dl_q = row["download_bytes_per_query"]
    dl_1 = row["download_bytes_once"]
    compute = server * instance_hourly * 1e6 / 3600
    egress = (dl_q + dl_1) * egress_usd_per_gb * 1e6 / 1e9
    return {"compute": compute, "egress": egress, "total": compute + egress}

# ---- Formatting -------------------------------------------------------------

SCHEMES_ORDER = [
    ("hintlesspir",   "HintlessPIR",       "cpu"),
    ("ypir_sp",       "YPIR-SP",           "cpu"),
    ("inspire_copt",  "InsPIRe c-opt",     "cpu"),
    ("onionpirv2",    "OnionPIRv2",        "cpu"),
    ("distpir_spgpu", "DistPIR (SPGPU)",   "gpu"),
    ("sandwichpir",   "SandwichPIR",       "gpu"),
]
DB_ORDER = [1, 4, 16]

def fmt_bytes(n: int | None) -> str:
    if n is None or n == 0:
        return "0"
    if n < 1024:                  return f"{n} B"
    if n < 1024**2:               return f"{n/1024:.1f} KB"
    if n < 1024**3:               return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GiB"

def fmt_seconds(s: float | None, na: str = "—") -> str:
    if s is None or s <= 0:
        return na
    if s < 1e-3:                  return f"{s*1e6:.0f} µs"
    if s < 1:                     return f"{s*1e3:.1f} ms"
    if s < 60:                    return f"{s:.2f} s"
    return f"{s/60:.1f} min"

def fmt_tput(g: float | None) -> str:
    if g is None or g <= 0:
        return "—"
    return f"{g:.1f} GiB/s"

def fmt_cost(c: float | None) -> str:
    if c is None:
        return "—"
    return f"${c:,.2f}"

# ---- Renderers --------------------------------------------------------------

def render_markdown(rows: list[dict]) -> str:
    header = "| Scheme | DB | Upload/q | Download/q | 1× hint | Server | Throughput | \\$/Mq |"
    sep    = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['db_gib']} GiB | {r['upload']} | {r['download']} | {r['hint']} |"
            f" {r['server']} | {r['tput']} | {r['cost']} |"
        )
    return "\n".join(lines)

def render_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Scheme & DB & Upload/q & Download/q & $1\times$ hint & Server & Throughput & \$/Mq \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['db_gib']}\\,GiB & {r['upload']} & {r['download']} & {r['hint']} & "
            f"{r['server']} & {r['tput']} & {r['cost']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)

def render_csv(rows: list[dict]) -> str:
    header = "scheme,db_gib,upload_bytes,download_bytes,hint_bytes,server_seconds,throughput_gib_s,cost_usd_per_mq"
    lines = [header]
    for r in rows:
        lines.append(",".join([
            r["scheme"], f"{r['db_gib']}",
            f"{r['upload_raw']}", f"{r['download_raw']}", f"{r['hint_raw']}",
            f"{r['server_raw']:.6f}", f"{r['tput_raw']:.3f}",
            f"{r['cost_raw']:.2f}",
        ]))
    return "\n".join(lines)

# ---- Assembly ---------------------------------------------------------------

def build_rows(results, cost_model):
    rows = []
    for scheme, label, instance_kind in SCHEMES_ORDER:
        inst = cost_model["instances"][instance_kind]
        hourly = inst["hourly_usd"]
        egress = cost_model["bandwidth"]["egress_usd_per_gb"]
        for db in DB_ORDER:
            r = results.get((scheme, float(db)))
            if r is None:
                rows.append({
                    "scheme": scheme, "label": label, "db_gib": db,
                    "upload": "—", "download": "—", "hint": "—",
                    "server": "—", "tput": "—", "cost": "—",
                    "upload_raw": 0, "download_raw": 0, "hint_raw": 0,
                    "server_raw": 0, "tput_raw": 0, "cost_raw": 0,
                })
                continue
            server_s = r["server_seconds"]["mean"]
            tput = throughput_gib_per_s(db, server_s)
            cost = cost_per_mq(r, hourly, egress)
            rows.append({
                "scheme": scheme, "label": label, "db_gib": db,
                "upload":   fmt_bytes(r["upload_bytes_per_query"]),
                "download": fmt_bytes(r["download_bytes_per_query"]),
                "hint":     fmt_bytes(r["download_bytes_once"]),
                "server":   fmt_seconds(server_s),
                "tput":     fmt_tput(tput),
                "cost":     fmt_cost(cost["total"]),
                "upload_raw":   r["upload_bytes_per_query"],
                "download_raw": r["download_bytes_per_query"],
                "hint_raw":     r["download_bytes_once"],
                "server_raw":   server_s,
                "tput_raw":     tput,
                "cost_raw":     cost["total"],
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["latex", "markdown", "csv"], default="markdown")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RODEO_ROOT / "results")
    args = ap.parse_args()

    cost_model = load_cost_model()
    results = load_results(args.results)
    rows = build_rows(results, cost_model)

    renderer = {"markdown": render_markdown, "latex": render_latex, "csv": render_csv}[args.format]
    out = renderer(rows)

    if args.out:
        args.out.write_text(out + "\n")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(out)

if __name__ == "__main__":
    main()
