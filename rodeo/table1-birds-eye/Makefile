# Table 1 (birds-eye PIR comparison) — per-benchmark Makefile.
#
# Shared infra lives at ../ (schemes/, runners/, patches/, setup/, table/, configs/).
# Only this benchmark's Makefile, sweep.toml, results, and rendered outputs live here.
#
# PREREQUISITE (run once per fresh instance):
#   make -C ../ setup-cpu    # OR setup-gpu
#   make -C ../ schemes      # inits submodules + applies patches
#
# Then:
#   make measure-cpu         # runs the 4 CPU rows of this table
#   make measure-gpu         # runs the 2 GPU rows of this table
#   make table-latex         # renders results/*.json → out/table1.tex

BENCH := $(CURDIR)
RODEO := $(BENCH)/..
RUNNERS := $(RODEO)/runners
TABLE   := $(RODEO)/table

# Export so the runners' lib.sh drops results here instead of $RODEO/results.
export RESULTS_DIR := $(BENCH)/results

SHELL := /bin/bash

.PHONY: measure-cpu measure-gpu measure-all table table-latex table-csv clean

# ---- Measure --------------------------------------------------------------

# The three SimplePIR-family CPU baselines. All take
#   <db_gib> <num_items> <record_bytes> <trials> <warmup>
# with records fixed at 32 KiB, so num_items carries the database size.
#
# Ordered smallest-DB-first so partial results are useful if a later size OoMs.
# See MEMORY note in the README: the 16 GiB rows need more DRAM than
# c7i.2xlarge provides.
measure-cpu: measure-cpu-1gib measure-cpu-4gib measure-cpu-16gib

measure-cpu-1gib:
	@mkdir -p $(RESULTS_DIR)
	@echo "==> 1 GiB"
	python3 $(RUNNERS)/run_hintlesspir.py 1 32768 32768 5 1
	python3 $(RUNNERS)/run_ypir_sp.py     1 32768 32768 5 1
	python3 $(RUNNERS)/run_inspire.py     1 32768 32768 5 1

measure-cpu-4gib:
	@mkdir -p $(RESULTS_DIR)
	@echo "==> 4 GiB"
	python3 $(RUNNERS)/run_hintlesspir.py 4 131072 32768 5 1
	python3 $(RUNNERS)/run_ypir_sp.py     4 131072 32768 5 1
	python3 $(RUNNERS)/run_inspire.py     4 131072 32768 5 1

measure-cpu-16gib:
	@mkdir -p $(RESULTS_DIR)
	@echo "==> 16 GiB (needs >16 GiB DRAM; see README)"
	python3 $(RUNNERS)/run_hintlesspir.py 16 524288 32768 3 1
	python3 $(RUNNERS)/run_ypir_sp.py     16 524288 32768 3 1
	python3 $(RUNNERS)/run_inspire.py     16 524288 32768 3 1

# OnionPIRv2 is off-family (poly-HE, 3 KiB native records) and its runner is
# still unverified: DB size is a compile-time constant requiring sed+rebuild,
# and it needs an Intel HEXL install. Kept out of measure-cpu so it cannot
# block the three comparable baselines.
measure-onionpir:
	@mkdir -p $(RESULTS_DIR)
	python3 $(RUNNERS)/run_onionpirv2.py 1 k1_comp 5 1

measure-gpu:
	@mkdir -p $(RESULTS_DIR)
	@echo "==> DistPIR SPGPU (1 & 4 GiB — 16 GiB OoM: needs ~128 GiB host, exceeds g6e.2xlarge's 64 GiB)"
	python3 $(RUNNERS)/run_distpir.py    1  32768  32768 10 2
	python3 $(RUNNERS)/run_distpir.py    4  131072 32768 10 2
	@echo "==> SandwichPIR"
	python3 $(RUNNERS)/run_sandwichpir.py 1  32768  32768 10 2
	python3 $(RUNNERS)/run_sandwichpir.py 4  131072 32768 10 2
	python3 $(RUNNERS)/run_sandwichpir.py 16 524288 32768 10 2

measure-all: measure-cpu measure-gpu

# ---- Render ---------------------------------------------------------------

table:
	python3 $(TABLE)/make_table.py --results $(RESULTS_DIR) --format markdown

table-latex: | $(BENCH)/out
	python3 $(TABLE)/make_table.py --results $(RESULTS_DIR) --format latex --out $(BENCH)/out/table1.tex
	@echo "==> Wrote $(BENCH)/out/table1.tex"

table-csv: | $(BENCH)/out
	python3 $(TABLE)/make_table.py --results $(RESULTS_DIR) --format csv --out $(BENCH)/out/table1.csv
	@echo "==> Wrote $(BENCH)/out/table1.csv"

$(BENCH)/out:
	@mkdir -p $@

clean:
	rm -f $(RESULTS_DIR)/*.json $(BENCH)/out/*
