# Patches applied to baseline schemes

**Policy**: never edit `schemes/<name>/` in place. All modifications are unified
diffs in this directory, applied by `make schemes` from the parent directory.
This keeps the pinned submodules pristine and the diffs review-friendly.

## Inventory

| ID | Target | Purpose | Blocking? |
|---:|--------|---------|-----------|
| 00 | `distpir/matrix/gpu/cuda/` | Add sm_89 (L40S) to the sm_70-only whitelist, in both the CMake arch list and the runtime `props.major != 7` guard | YES — no GPU run without it |
| 02 | `distpir/lhe/database.go` | Fix a 2× RSS blowup: bench-mode `NewDB` allocates a fresh `L×M` `uint32` matrix while ignoring the same-size one the caller already passed | YES at ≥ 4 GiB DB (OoMs otherwise) |
| 03 | `distpir/benches/{lhe_throughput,benches}.go` | (a) respect `-batch N` instead of sweeping the hardcoded `ks[]`; (b) add `-batches 1,2,4,...` so a sweep pays DB setup once; (c) replace `testing.Benchmark` with an explicit 10-trial loop emitting per-trial times | YES for per-trial stddev; (a)/(b) cut sweep runtime ~8× |
| 05 | `ypir/Cargo.toml` | Empty `[workspace]` so cargo does not treat the vendored crate as a member of the sandwichpir workspace | YES — cargo refuses to build otherwise |
| 06 | `inspire/research/InsPIRe/Cargo.toml` | Same as 05, for the InsPIRe fork | YES — cargo refuses to build otherwise |
| 08 | `hintlesspir/hintless_simplepir/` | Emit measured comm (request/response `ByteSizeLong`, Galois key split) and expose LinPIR `rows_per_block` | YES for measured comm |
| 09 | `inspire/research/InsPIRe/src/bin/inspire.rs` | Generate the synthetic benchmark DB on demand (pure function of index) instead of materialising a second copy; cuts peak from ~4x to ~2.5x DB | YES at 16 GiB (OoMs otherwise) |
| 10 | `kspir/tests/test-pir.cpp` | Add `test-pir <db_mb>` (r = db_mb/16); upstream hardcodes r=16 (256 MB) and ignores argv | YES — no size sweep without it |
| 11 | `distpir/benches/lhe_throughput.go` | Stage the bench DB with an unwritten allocation (`m.New`; NewDB's bench path regenerates content anyway) so staging commits no RSS; needs `vm.overcommit_memory=1` for the 64 GiB mapping | YES at 16 GiB CPU (OOMs otherwise) |

## Applying

`make schemes` from `rodeo/` initializes each submodule and applies its patches.
It is idempotent and self-repairing: patches run with `-N --batch` (never
interactive, skips already-applied hunks), their exit status is ignored, and
correctness is decided afterwards by grepping a marker in **every file the patch
touches**. A missing marker fails the build naming the specific patch.

Exit status is deliberately not trusted: `patch -N` returns failure when it
correctly skips hunks that were already applied -- which happens routinely,
because `git reset --hard` in the superproject does not clean submodule working
trees.

When editing a patch, update its marker string in the Makefile too. A stale
marker makes verification fail on a correctly-patched tree.

To force a re-patch after changing a patch file:

```bash
find schemes -name '.ready' -delete
make schemes
```

Note that patched Go/C++ sources still need their build artifact removed to
actually rebuild — e.g. `rm -rf schemes/distpir/benches/benches`.

## Adding a patch

Always generate from a real diff; hand-written hunks drift out of context and
apply with fuzz (or silently half-apply, which is worse).

1. Edit under `schemes/<name>/` directly to prototype.
2. `git -C schemes/<name> diff > /tmp/x.diff`
3. Prepend a purpose header, then append `tail -n +3 /tmp/x.diff` (dropping the
   `diff --git` and `index` lines) into `patches/<ID>-<slug>.patch`.
4. `git -C schemes/<name> checkout .` to revert the working copy.
5. `patch --dry-run -l -p1 -d schemes/<name> < patches/<ID>-<slug>.patch` to verify.
6. Add a row to the table above and a recipe line to the parent `Makefile`.

If two patches touch the same region of one file, **merge them into a single
patch** rather than chaining — the second one's context will not match after the
first applies.

## 10-kspir-size-arg.patch

`tests/test-pir.cpp` hardcodes `r = 16` (256 MB) and ignores argv. This adds
`test-pir <db_mb>` with `r = db_mb/16`, mirroring the CLI that InsPIRe's
evaluation harness (`run-kspir.sh`) drives; upstream `test-larger-database.cpp`
is an abandoned stub that does not compile. Applies to mmingluo/kspir @ 54c2f61.
