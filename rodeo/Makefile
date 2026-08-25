# Top-level rodeo Makefile — one-time bootstrap + shared scheme management.
#
# Per-benchmark measurement + rendering lives in each benchmark subdir
# (e.g., table1-birds-eye/Makefile).

RODEO := $(CURDIR)
SCHEMES := $(RODEO)/schemes
PATCHES := $(RODEO)/patches

SHELL := /bin/bash

.PHONY: setup-cpu setup-gpu schemes clean-schemes

setup-cpu:
	bash setup/setup-cpu.sh

setup-gpu:
	bash setup/setup-gpu.sh

# Initialize + patch all baseline submodules. Idempotent.
schemes: $(SCHEMES)/hintlesspir/.ready \
         $(SCHEMES)/ypir/.ready \
         $(SCHEMES)/inspire/.ready \
         $(SCHEMES)/distpir/.ready \
         $(SCHEMES)/kspir/.ready

$(SCHEMES)/%/.ready:
	@echo "==> Preparing $*"
	git -C .. submodule update --init --recursive rodeo/schemes/$*
	@$(MAKE) -s patch-$*
	@touch $@

.PHONY: patch-hintlesspir patch-ypir patch-inspire patch-distpir patch-kspir
patch-hintlesspir:
	@# Upstream's benchmark times the server but emits no byte counts, which is why
	@# the comparison table previously cited the paper's figures instead of measuring.
	@patch -l -N --batch -p1 -d $(SCHEMES)/hintlesspir < $(PATCHES)/08-hintlesspir-measure-comm.patch >/dev/null 2>&1 || true
	@grep -q "offline_down_bytes" $(SCHEMES)/hintlesspir/hintless_simplepir/hintless_simplepir_benchmarks.cc || { echo "ERROR: patch 08 (hintlesspir comm counters) not applied"; exit 1; }
	@find $(SCHEMES)/hintlesspir -name '*.rej' -o -name '*.orig' | xargs -r rm -f
	@echo "    hintlesspir: patch 08 verified"
patch-ypir:
	@patch -l -N --batch -p1 -d $(SCHEMES)/ypir < $(PATCHES)/05-ypir-standalone-workspace.patch >/dev/null 2>&1 || true
	@grep -q "^\[workspace\]" $(SCHEMES)/ypir/Cargo.toml || { echo "ERROR: patch 05 (ypir workspace) not applied"; exit 1; }
	@find $(SCHEMES)/ypir -name '*.rej' -o -name '*.orig' | xargs -r rm -f
	@echo "    ypir: patch 05 verified"
patch-inspire:
	@patch -l -N --batch -p1 -d $(SCHEMES)/inspire < $(PATCHES)/06-inspire-standalone-workspace.patch >/dev/null 2>&1 || true
	@grep -q "^\[workspace\]" $(SCHEMES)/inspire/research/InsPIRe/Cargo.toml || { echo "ERROR: patch 06 (inspire workspace) not applied"; exit 1; }
	@# 09: generate the synthetic DB on demand instead of materialising it twice
	@# (u16 per coefficient, held in both the caller's Vec and the encoder's buffer
	@# = 4x the logical database; 16 GiB needed 64 GiB of DRAM). Deterministic
	@# generation keeps the benchmark's correctness check intact.
	@patch -l -N --batch -p1 -d $(SCHEMES)/inspire < $(PATCHES)/09-inspire-stream-db.patch >/dev/null 2>&1 || true
	@grep -q "synth_pt" $(SCHEMES)/inspire/research/InsPIRe/src/bin/inspire.rs || { echo "ERROR: patch 09 (inspire stream db) not applied"; exit 1; }
	@find $(SCHEMES)/inspire -name '*.rej' -o -name '*.orig' | xargs -r rm -f
	@echo "    inspire: patch 06 verified"
patch-distpir:
	@# Apply, tolerating a nonzero exit: with -N, patch returns failure when it
	@# skips hunks that were already applied (e.g. a submodule tree left dirty by
	@# an earlier run, which `git reset --hard` in the superproject does not
	@# clean). Correctness is checked by verifying the end state below instead.
	@patch -l -N --batch -p1 -d $(SCHEMES)/distpir < $(PATCHES)/00-distpir-sm89.patch          >/dev/null 2>&1 || true
	@patch -l -N --batch -p1 -d $(SCHEMES)/distpir < $(PATCHES)/02-distpir-bench-db-dedup.patch >/dev/null 2>&1 || true
	@patch -l -N --batch -p1 -d $(SCHEMES)/distpir < $(PATCHES)/03-distpir-bench-harness.patch  >/dev/null 2>&1 || true
	@grep -q "CMAKE_CUDA_ARCHITECTURES 70 89"            $(SCHEMES)/distpir/matrix/gpu/cuda/CMakeLists.txt || { echo "ERROR: patch 00 (sm_89) not applied"; exit 1; }
	@grep -q "PATCH: reuse caller's data slice"          $(SCHEMES)/distpir/lhe/database.go              || { echo "ERROR: patch 02 (db dedup) not applied"; exit 1; }
	@grep -q "PATCH: respect the -batch"        $(SCHEMES)/distpir/benches/lhe_throughput.go    || { echo "ERROR: patch 03 (lhe_throughput) not applied"; exit 1; }
	@grep -q "PATCH: optional comma-separated batch list" $(SCHEMES)/distpir/benches/benches.go          || { echo "ERROR: patch 03 (benches.go) not applied"; exit 1; }
	@find $(SCHEMES)/distpir -name '*.rej' -o -name '*.orig' | xargs -r rm -f
	@echo "    distpir: patches 00, 02, 03 verified"

clean-schemes:
	find $(SCHEMES) -name '.ready' -delete


patch-kspir:
	@patch -l -N --batch -p1 -d $(SCHEMES)/kspir < $(PATCHES)/10-kspir-size-arg.patch >/dev/null 2>&1 || true
	@grep -q "db_mb" $(SCHEMES)/kspir/tests/test-pir.cpp || { echo "ERROR: patch 10 (kspir size arg) not applied"; exit 1; }
	@find $(SCHEMES)/kspir -name '*.rej' -o -name '*.orig' | xargs -r rm -f
	@echo "    kspir: patch 10 verified"
