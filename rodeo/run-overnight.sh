#!/usr/bin/env bash
# Unattended CPU measurement sweep for Table 1.
#
#   bash rodeo/run-overnight.sh 2>&1 | tee /tmp/overnight.log
#
# Design notes:
#   * NOTHING here aborts the run. Each measurement is isolated so one scheme
#     failing (build error, OoM, parse failure) never costs you the others.
#   * Ordered cheapest-first (1 GiB -> 4 GiB -> 16 GiB) so that if the machine
#     dies or you stop it early, the results you do have are the useful ones.
#   * A per-step status line is appended to STATUS so the morning triage is one
#     `cat` rather than scrolling a 10k-line log.
#
# 16 GiB rows need more DRAM than c7i.2xlarge's 16 GiB (HintlessPIR alone peaks
# at ~13-14 GiB RSS for an *8* GiB database). They are attempted last and are
# expected to OoM on that instance; that is recorded, not fatal.
set -u

RODEO="$(cd "$(dirname "$0")" && pwd)"
BENCH="$RODEO/table1-birds-eye"
RUNNERS="$RODEO/runners"
export RESULTS_DIR="$BENCH/results"

LOGDIR="$BENCH/logs"
STATUS="$BENCH/overnight-status.txt"
mkdir -p "$RESULTS_DIR" "$LOGDIR"

: > "$STATUS"
START=$(date +%s)

say()  { echo "[$(date -u +%H:%M:%S)] $*"; }
note() { echo "$*" >> "$STATUS"; }

# step <label> <command...>
# Runs the command, tees to its own log, records PASS/FAIL with elapsed time.
step() {
    local label="$1"; shift
    local log="$LOGDIR/${label}.log"
    local t0 t1 rc mins
    say "START  $label"
    t0=$(date +%s)
    "$@" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    mins=$(( (t1 - t0) / 60 ))
    if [ $rc -eq 0 ]; then
        say "PASS   $label  (${mins}m)"
        note "PASS  $label  ${mins}m"
    else
        say "FAIL   $label  (${mins}m, rc=$rc) -- see $log"
        note "FAIL  $label  ${mins}m  rc=$rc  tail: $(tail -3 "$log" | tr '\n' ' ' | cut -c1-160)"
    fi
    return 0   # never propagate failure
}

say "===== overnight sweep starting ====="
say "results  -> $RESULTS_DIR"
say "per-step -> $LOGDIR"
say "summary  -> $STATUS"
free -g 2>/dev/null | head -2
note "host=$(hostname) instance=$(curl -sf --max-time 1 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo local)"
note "started $(date -u)"
note ""

# ── Prepare submodules (idempotent, self-repairing) ──
step "00-schemes" make -C "$RODEO" schemes

# ── 1 GiB: all three SimplePIR-family schemes ──
step "01-hintlesspir-1gib" python3 "$RUNNERS/run_hintlesspir.py" 1 32768 32768 5 1
step "02-ypir-1gib"        python3 "$RUNNERS/run_ypir_sp.py"     1 32768 32768 5 1
step "03-inspire-1gib"     python3 "$RUNNERS/run_inspire.py"     1 32768 32768 5 1

# ── 4 GiB ──
step "04-hintlesspir-4gib" python3 "$RUNNERS/run_hintlesspir.py" 4 131072 32768 5 1
step "05-ypir-4gib"        python3 "$RUNNERS/run_ypir_sp.py"     4 131072 32768 5 1
step "06-inspire-4gib"     python3 "$RUNNERS/run_inspire.py"     4 131072 32768 5 1

# ── 16 GiB: expected to OoM on a 16 GiB instance. Attempted last. ──
step "08-hintlesspir-16gib" python3 "$RUNNERS/run_hintlesspir.py" 16 524288 32768 3 1
step "09-ypir-16gib"        python3 "$RUNNERS/run_ypir_sp.py"     16 524288 32768 3 1
step "10-inspire-16gib"     python3 "$RUNNERS/run_inspire.py"     16 524288 32768 3 1

# ── Render whatever landed ──
step "11-table-md"    make -C "$BENCH" table
step "12-table-latex" make -C "$BENCH" table-latex

ELAPSED=$(( ($(date +%s) - START) / 60 ))
note ""
note "finished $(date -u), ${ELAPSED}m total"
note "result JSONs: $(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)"

say "===== done in ${ELAPSED}m ====="
echo
echo "================ SUMMARY ================"
cat "$STATUS"
echo "========================================="
echo
echo "Measured rows:"
for f in "$RESULTS_DIR"/*.json; do
    [ -e "$f" ] || continue
    python3 -c "
import json,sys
d=json.load(open('$f'))
s=d['server_seconds']
n=len(s['raw'])
print(f\"  {d['scheme']:<14} {d['db_gib']:>3} GiB  mean={s['mean']:.4f}s  trials={n}\"
      + ('   <-- ZERO, CHECK PARSE' if s['mean']==0 else ''))
" 2>/dev/null
done
