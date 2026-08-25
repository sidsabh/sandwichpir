#!/usr/bin/env python3
"""
SandwichPIR correctness analysis — noise bounds and failure probability.

Usage:
    python scripts/noise_bounds.py                          # paper-shape sweep
    python scripts/noise_bounds.py 65536 1048576            # 8 GB Wikipedia
    python scripts/noise_bounds.py 65536 1048576 --verbose  # term breakdown
    python scripts/noise_bounds.py 131072 262144 --symmetric # GEMM-hint variant

Sub-Gaussian tail: Pr[|X| > t] ≤ 2 · exp(-πt²/σ²)
Per entry (correctness theorem):   log2(delta) = 1 - π·τ²/(σ²·ln2),
with the packing variance bounded by t·d².
Per record (corollary, union bound over the rho·d coefficients of a record):
log2(delta_record) = log2(delta) + log2(rho·d).

Security is assessed separately by scripts/estimator_wrapper.py.
"""

import math
import sys

# ═══════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════

Q     = 4294955009
W     = 1 << 32
d     = 2048
sigma_disc = 0.5
sigma_s = sigma_disc * math.sqrt(2 * math.pi)   # D(0.5) secret
sigma_e = sigma_s                                 # D(0.5) error
sigma_round = 0.5 * math.sqrt(2 * math.pi)       # Hoeffding for |ε| ≤ 1/2
z     = 256
t     = 4
q21   = 1 << 18
q22   = 1 << 10
p     = 256
pt_bits = 8

def sp_dims(num_items, item_bits):
    """Derive (l1, l2) from user-facing DB parameters."""
    instances = math.ceil(item_bits / (d * pt_bits))
    nu_1 = max(0, int(math.ceil(math.log2(max(num_items, 1)))) - 11)
    return 1 << (nu_1 + 11), instances * d


# ═══════════════════════════════════════════════════════════════
# NTT hint noise (5 terms)
# ═══════════════════════════════════════════════════════════════

def tau():
    margin = (q22 - (q22 % p)) / (2 * p)
    det = (2 + (q22 % p) + (q22 / Q) * (Q % p)) / 2
    return margin - det

def sigma_sq_ntt(l1):
    a = (q22 / Q) ** 2
    QW2 = (Q / W) ** 2
    B = p / 2
    sr2 = sigma_round ** 2
    return {
        '(1) final MS: s×mask_round':       (q22/q21)**2 * d * sigma_s**2 / 4,
        '(2) RLWE error through scan':       a * B**2 * l1 * sigma_e**2,
        '(3) packing: s×gadget_round':       a * z**2 * sigma_s**2 * t * d * d / 4,
        '(4) body ε_b Q→W through scan':     a * QW2 * B**2 * l1 * 2 * math.pi / 4,
        '(5) body ε_r W→Q':                  a * sr2,
    }

def sigma_sq_symmetric_extra(l1):
    # symmetric (GEMM) hint: mask rounding through the scan x secret, plus
    # hint rounding x secret, both absent from the asymmetric NTT hint
    a = (q22 / Q) ** 2
    B = p / 2
    return {
        '(s3) mask ε_a through scan × s':    a * l1 * B**2 * d * sigma_s**2 / 4,
        '(s5) hint ε_H × s':                 a * d * sigma_s**2 / 4,
    }

def log2_delta_record(l1, l2, symmetric=False):
    # union over the rho*d coefficients of one record (paper corollary)
    rho = l2 // d
    return log2_delta(l1, l2, symmetric) + math.log2(rho * d)

def log2_delta(l1, l2, symmetric=False):
    t_val = tau()
    terms = sigma_sq_ntt(l1)
    if symmetric:
        terms = {**terms, **sigma_sq_symmetric_extra(l1)}
    s2 = sum(terms.values())
    if t_val <= 0 or s2 <= 0:
        return float('inf')
    return 1 - math.pi * t_val**2 / (s2 * math.log(2))


# ═══════════════════════════════════════════════════════════════
# Communication
# ═══════════════════════════════════════════════════════════════

def communication(l1, l2):
    up_q = l1 * 4
    up_k = 2 * t * d * 4
    dl_mask = l2 * int(math.log2(q21)) // 8
    dl_body = l2 * int(math.log2(q22)) // 8
    pt = l2
    return up_q, up_k, dl_mask, dl_body, pt


# ═══════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════

def fmt(nbytes):
    if nbytes >= 1 << 40: return f"{nbytes / (1 << 40):.0f} TB"
    if nbytes >= 1 << 30: return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20: return f"{nbytes / (1 << 20):.0f} MB"
    if nbytes >= 1 << 10: return f"{nbytes / (1 << 10):.0f} KB"
    return f"{nbytes} B"

def print_single(l1, l2, verbose=False, symmetric=False):
    rho = l2 // d
    db_bytes = l1 * l2
    ld = log2_delta(l1, l2, symmetric)
    t_val = tau()
    terms = sigma_sq_ntt(l1)
    if symmetric:
        terms = {**terms, **sigma_sq_symmetric_extra(l1)}
    s2 = sum(terms.values())

    ldr = log2_delta_record(l1, l2, symmetric)
    print(f"Database: {fmt(db_bytes)}  (l1 = {l1} = 2^{int(math.log2(l1))}, l2 = {l2}, rho = {rho})")
    print(f"log2(delta) per entry  = {ld:.1f}   {'PASS' if ld < -40 else 'FAIL'} (threshold: -40)")
    print(f"log2(delta) per record = {ldr:.1f}   {'PASS' if ldr < -40 else 'FAIL'} (corollary: + log2(rho*d))")

    if verbose:
        print(f"\ntau = {t_val:.6f}, sigma^2 = {s2:.6e}")
        print()
        for name, val in terms.items():
            pct = val / s2 * 100 if s2 > 0 else 0
            print(f"  {name:40s}: {val:.2e}  ({pct:5.1f}%)")
        print()

    up_q, up_k, dl_m, dl_b, pt = communication(l1, l2)
    total_1st = up_q + up_k + dl_m + dl_b
    total_sub = up_q + up_k + dl_b
    print(f"Upload: {fmt(up_q + up_k)}  Download (1st): {fmt(dl_m + dl_b)}  Download (sub): {fmt(dl_b)}  Rate: {pt/total_1st*100:.0f}% / {pt/total_sub*100:.0f}%")

def print_select(l1=1 << 20, l2=1 << 20):
    """Output-moduli selection: the (q21, q22) tradeoff at the most aggressive
    configuration we contemplate (1 TB: l1 = 2^20 rows of 1 MB). Rounding to
    q22 must leave a decoding margin over p = 256, and the mask modulus q21
    controls both the final modulus-switching noise and the (cacheable) mask
    size. We require log2(delta_record) <= -40 at this stress shape; the paper
    uses (2^18, 2^10)."""
    global q21, q22
    rho = l2 // d
    saved = (q21, q22)
    print(f"Selection grid at l1 = 2^{int(math.log2(l1))}, l2 = {l2} "
          f"(record delta shown; mask/body per response; * = paper choice)")
    print(f"{'q21 \\ q22':>10s}" + "".join(f"{'2^' + str(b):>22s}" for b in (9, 10, 11, 12)))
    for a in range(14, 21):
        row = [f"{'2^' + str(a):>10s}"]
        for b in (9, 10, 11, 12):
            q21, q22 = 1 << a, 1 << b
            ldr = log2_delta_record(l1, l2)
            _, _, dl_m, dl_b, _ = communication(l1, l2)
            mark = "*" if (a, b) == (18, 10) else " "
            val = f"{ldr:7.0f}" if ldr < 0 else "   FAIL"
            row.append(f"{val} {fmt(dl_m):>6s}/{fmt(dl_b):>6s}{mark}")
        print("".join(row))
    q21, q22 = saved
    print()
    print("Criterion: smallest moduli with log2(delta_record) <= -40 at this")
    print("stress shape; q22 = 2^10 is the smallest power of two that decodes")
    print("p = 256 with positive margin, and q21 = 2^18 keeps the final")
    print("modulus-switching noise negligible at 4.5 KB of cacheable mask per")
    print("packed output (72 KB per response at 32 KB records).")
    print()

    global z, t
    saved_zt = (z, t)
    print(f"Gadget (z, t) with z^t = 2^32, same stress shape (* = paper choice):")
    print(f"{'z':>6s} {'t':>3s} {'record delta':>13s} {'ks keys':>8s}  note")
    for zz, tt in [(1 << 16, 2), (1 << 8, 4), (1 << 4, 8)]:
        z, t = zz, tt
        ldr = log2_delta_record(l1, l2)
        keys = 2 * t * d * 4
        mark = "*" if (zz, tt) == (256, 4) else " "
        note = ("digits exceed one byte: packing GEMM cannot use int8 tensor cores"
                if zz > 256 else
                "byte digits; minimal t (keys and packing work scale with t)" if zz == 256 else
                "byte digits, but 2x the keys and packing work")
        val = f"{ldr:8.0f}" if ldr < 0 else "    FAIL"
        print(f"2^{int(math.log2(zz)):<4d} {tt:>3d} {val:>13s}{mark} {fmt(keys):>7s}  {note}")
    z, t = saved_zt
    print()
    print("z = 2^8 is forced by the int8 tensor-core datapath (gadget digits")
    print("must fit a byte), which fixes t = 32/8 = 4.")


def print_sweep():
    scenarios = [
        (1 << 15, 1 << 15),   # 1 GB, 32 KB records (paper tab:birds-eye)
        (1 << 17, 1 << 15),   # 4 GB, 32 KB records
        (1 << 18, 1 << 15),   # 8 GB, 32 KB records
        (1 << 19, 1 << 15),   # 16 GB, 32 KB records
        (1 << 16, 1 << 17),   # 8 GB, 128 KB records (Wikipedia)
        (1 << 18, 1 << 18),   # 64 GB
        (1 << 20, 1 << 20),   # 1 TB
    ]

    print(f"{'DB':>8s}  {'l1':>8s}  {'rho':>5s}  {'entry':>7s}  {'record':>7s}  {'Upload':>8s}  {'Dl(1st)':>9s}  {'Dl(sub)':>9s}  {'Rate':>10s}")
    print("-" * 88)
    for l1, l2 in scenarios:
        rho = l2 // d
        ld = log2_delta(l1, l2)
        ldr = log2_delta_record(l1, l2)
        up_q, up_k, dl_m, dl_b, pt = communication(l1, l2)
        total_1st = up_q + up_k + dl_m + dl_b
        total_sub = up_q + up_k + dl_b
        ld_str = f"{ld:.0f}" if ld < 0 else "FAIL"
        ldr_str = f"{ldr:.0f}" if ldr < 0 else "FAIL"
        print(f"{fmt(l1*l2):>8s}  2^{int(math.log2(l1)):<5d}  {rho:>5d}  {ld_str:>7s}  {ldr_str:>7s}  {fmt(up_q+up_k):>8s}  {fmt(dl_m+dl_b):>9s}  {fmt(dl_b):>9s}  {pt/total_1st*100:.0f}%/{pt/total_sub*100:.0f}%")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    symmetric = '--symmetric' in sys.argv
    if '--select' in sys.argv:
        print_select()
        sys.exit(0)

    print(f"Q={Q}, W=2^32, d={d}, p={p}, sigma=D(0.5), z={z}, t={t}, q21=2^{int(math.log2(q21))}, q22=2^{int(math.log2(q22))}")
    print()

    if len(args) >= 2:
        num_items = int(args[0])
        item_size_bits = int(args[1])
        l1, l2 = sp_dims(num_items, item_size_bits)
        print_single(l1, l2, verbose, symmetric)
    else:
        print_sweep()
