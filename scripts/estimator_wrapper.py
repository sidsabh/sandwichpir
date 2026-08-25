#!/usr/bin/env python3
"""
SandwichPIR security estimation — drives the lattice estimator on the paper's
LWE instance.

The client encrypts under RLWE with ring dimension d = 2048 and modulus
Q = 4294955009, with secret and error drawn from a discrete gaussian of width
sigma = 0.5. Viewed as LWE, one ciphertext exposes m = d samples, giving the
instance:

    LWE.Parameters(n=2048, q=4294955009,
                   Xs=DiscreteGaussian(0.5), Xe=DiscreteGaussian(0.5),
                   m=2048)

The paper's security claim uses lattice-estimator commit 53da598; this script
checks the clone out at that commit before importing it (pass --no-checkout to
use the clone as-is).

Requires SageMath (the estimator depends on it). Clone the estimator next to
this repository, then:

    cd sandwichpir
    sage -python scripts/estimator_wrapper.py [--le-path ../lattice-estimator]

Expected output at the pinned commit:
    usvp 2^201.0, bdd 2^199.2, dual_hybrid 2^192.7 (minimum), bkw 2^372.0
i.e., at least 128 bits of classical security, with margin.
"""

import argparse
import math
import subprocess
import sys

LE_COMMIT = "53da5982597709ba0fdf94ea37a84d822310fd84"

# SandwichPIR LWE instance (tab:params in the paper)
N = 2048
Q = 4294955009
SIGMA = 0.5
M = 2048


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--le-path", default="../lattice-estimator",
                    help="path to a clone of malb/lattice-estimator")
    ap.add_argument("--no-checkout", action="store_true",
                    help="do not git-checkout the pinned commit")
    args = ap.parse_args()

    if not args.no_checkout:
        r = subprocess.run(["git", "checkout", LE_COMMIT],
                           cwd=args.le_path, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"warning: could not checkout {LE_COMMIT[:7]} in {args.le_path}:"
                  f" {r.stderr.strip()}", file=sys.stderr)

    sys.path.insert(0, args.le_path)
    from estimator import LWE
    from estimator.lwe_parameters import LWEParameters
    from estimator.nd import DiscreteGaussian as D

    params = LWEParameters(n=N, q=Q, Xs=D(SIGMA), Xe=D(SIGMA), m=M,
                           tag="SandwichPIR")
    print(params)
    results = LWE.estimate(params)

    best = min(math.log(v["rop"], 2) for v in results.values())
    print()
    print(f"minimum attack cost: 2^{best:.1f}  "
          f"({'PASS' if best >= 128 else 'FAIL'}: target 128-bit classical)")


if __name__ == "__main__":
    main()
