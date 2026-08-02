# SPDX-License-Identifier: GPL-2.0-or-later
#
# nguo-3d developer gates. CI runs THESE targets, not its own copies of the
# commands, so local and CI cannot drift. If you change a gate, change it here.
#
#   make ci          everything a PR must pass (fmt, clippy, test, wgpu, licences)
#   make gpu         Metal smoke test + the 10x determinism gate (needs a GPU)
#   make help        list targets
#
# Two cargo workspaces live in this tree and neither contains the other, so
# every target loops over both.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

WORKSPACES := core-engine ui

# Portability: macOS has `shasum` and no `sha256sum`; Linux and Git-Bash-on-
# Windows have `sha256sum`. Windows binaries carry `.exe`.
SHA256 := $(shell command -v sha256sum >/dev/null 2>&1 && echo sha256sum || echo "shasum -a 256")
EXE    := $(if $(findstring Windows,$(OS)),.exe,)

# The engine CLI. Override to reuse a debug build: `make gpu MGUSO=core-engine/target/debug/mguso`
MGUSO ?= core-engine/target/release/mguso$(EXE)

# 10x determinism gate knobs (see `determinism` below).
DET_SCENE  ?= core-engine/examples/fabric_drape.json
DET_FRAMES ?= 5
DET_RUNS   ?= 10
DET_SOLVER ?= implicit
DET_DIR    ?= .determinism-tmp

.PHONY: help ci fmt fmt-fix clippy test check-wgpu licenses licenses-sync gpu gpu-smoke determinism \
        build-release build-engine build-ui bench clean-det

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t22

## ---------------------------------------------------------------------------
## The PR gate
## ---------------------------------------------------------------------------

ci: fmt clippy check-wgpu test licenses ## Everything a PR must pass (no GPU needed)
	@echo "== make ci: all gates passed =="

fmt: ## cargo fmt --all --check, both workspaces
	@for ws in $(WORKSPACES); do echo "== fmt: $$ws"; (cd $$ws && cargo fmt --all --check); done

fmt-fix: ## rewrite files to satisfy `make fmt`
	@for ws in $(WORKSPACES); do (cd $$ws && cargo fmt --all); done

clippy: ## cargo clippy --workspace --all-targets -- -D warnings, both workspaces
	@for ws in $(WORKSPACES); do echo "== clippy: $$ws"; (cd $$ws && cargo clippy --workspace --all-targets -- -D warnings); done

test: ## cargo test --workspace --no-fail-fast, both workspaces
	@for ws in $(WORKSPACES); do echo "== test: $$ws"; (cd $$ws && cargo test --workspace --no-fail-fast); done

licenses: licenses-sync ## cargo-deny: licence allow-list + wgpu ban + source allow-list
	@command -v cargo-deny >/dev/null || { echo "cargo-deny not installed: brew install cargo-deny (or cargo install --locked cargo-deny)"; exit 1; }
	@for ws in $(WORKSPACES); do \
	  echo "== cargo-deny: $$ws"; \
	  cargo deny --manifest-path $$ws/Cargo.toml --config deny.toml check licenses bans sources; \
	done

## The two allow-lists (deny.toml = enforcement, core-engine/about.toml =
## attribution) are documented as identical. They had drifted, because nothing
## checked. This checks. It needs no tools, no build and no network — it is a
## sorted diff of the two arrays.
##
## It does NOT run cargo-about: attribution regeneration is still manual and
## there is no cargo-about job anywhere. This target guarantees the *lists*
## agree, not that THIRD_PARTY.md is current.
licenses-sync: ## Fail if deny.toml and core-engine/about.toml allow-lists differ
	@a=$$(sed -n '/^allow = \[$$/,/^\]$$/p'    deny.toml              | grep -o '"[^"]*"' | sort); \
	 b=$$(sed -n '/^accepted = \[$$/,/^\]$$/p' core-engine/about.toml | grep -o '"[^"]*"' | sort); \
	 if [ -z "$$a" ] || [ -z "$$b" ]; then echo "ERROR: could not extract an allow-list — did the array formatting change?"; exit 1; fi; \
	 if [ "$$a" != "$$b" ]; then \
	   echo "ERROR: licence allow-lists have drifted."; \
	   echo "  '<' = only in deny.toml   '>' = only in core-engine/about.toml"; \
	   diff <(echo "$$a") <(echo "$$b") || true; \
	   exit 1; \
	 fi; \
	 echo "== licences-sync: OK — deny.toml and core-engine/about.toml agree ($$(echo "$$a" | wc -l | tr -d ' ') licences)"

## ---------------------------------------------------------------------------
## The single-wgpu-version check (wgpu version drift: the #1 integration
## hazard). iced, Vello, egui and the 3D viewport must share ONE wgpu::Device;
## two wgpu majors in one lockfile makes that impossible. Reads Cargo.lock, so
## it needs no build and no network.
## ---------------------------------------------------------------------------

check-wgpu: ## Fail if a lockfile contains more than one wgpu major
	@fail=0; \
	for ws in $(WORKSPACES); do \
	  lock=$$ws/Cargo.lock; \
	  if [ ! -f "$$lock" ]; then echo "== wgpu: $$ws has no Cargo.lock yet — skipped"; continue; fi; \
	  vers=$$(awk '/^name = "wgpu"$$/{getline; gsub(/[",]/,"",$$3); print $$3}' "$$lock" | sort -u); \
	  if [ -z "$$vers" ]; then echo "== wgpu: $$ws does not depend on wgpu"; continue; fi; \
	  majors=$$(echo "$$vers" | cut -d. -f1 | sort -u | tr '\n' ' '); \
	  n=$$(echo "$$majors" | wc -w | tr -d ' '); \
	  if [ "$$n" -gt 1 ]; then \
	    echo "ERROR: $$ws/Cargo.lock has $$n wgpu majors: $$majors"; \
	    for v in $$vers; do \
	      echo "--- who pulls wgpu $$v ---"; \
	      (cd $$ws && cargo tree --offline -i wgpu@$$v 2>&1 || cargo tree -i wgpu@$$v); \
	    done; \
	    fail=1; \
	  else \
	    echo "== wgpu: $$ws OK — single major ($$vers)"; \
	  fi; \
	done; \
	if [ "$$fail" != 0 ]; then \
	  echo "One wgpu major per workspace. See ui/Cargo.toml's header and core-engine/docs/WGPU_PIN.md."; \
	  exit 1; \
	fi

## ---------------------------------------------------------------------------
## GPU gates — need a real adapter, so they are their own job in CI.
## ---------------------------------------------------------------------------

gpu: gpu-smoke determinism ## Metal/Vulkan/DX12 adapter smoke test + 10x determinism

build-release: build-engine build-ui ## Release-build both workspaces' binaries

build-engine: ## Release-build the `mguso` engine CLI
	cd core-engine && cargo build --release -p engine-cli
	@test -x core-engine/target/release/mguso$(EXE) && echo "built core-engine/target/release/mguso$(EXE)"

build-ui: ## Release-build the UI workspace's binaries (`nguo` once it exists)
	cd ui && cargo build --release --workspace --bins
	@if [ -x ui/target/release/nguo$(EXE) ]; then \
	  echo "built ui/target/release/nguo$(EXE)"; \
	else \
	  echo "NOTE: no 'nguo' binary was produced — no crate in ui/crates/ declares a [[bin]] named nguo yet."; \
	  echo "      This is reported, not skipped silently."; \
	fi

# Set EXPECT_BACKEND to assert the wgpu backend as well (Metal / Vulkan / Dx12).
# Empty (the default) just requires *some* working adapter.
EXPECT_BACKEND ?=

gpu-smoke: ## `mguso gpu-info` must bring up an adapter and pass the SAXPY probe
	@test -x $(MGUSO) || { echo "$(MGUSO) missing — run 'make build-release' (or pass MGUSO=...)"; exit 1; }
	@$(MGUSO) gpu-info 2>/dev/null > $(DET_DIR).gpu.json || { echo "gpu-info FAILED — no usable adapter"; exit 1; }
	@cat $(DET_DIR).gpu.json
	@jq -e '.noop_saxpy_ok == true' $(DET_DIR).gpu.json >/dev/null || { echo "SAXPY probe wrong result — the GPU ran but computed garbage"; exit 1; }
	@if [ -n "$(EXPECT_BACKEND)" ]; then \
	  got=$$(jq -r .backend $(DET_DIR).gpu.json); \
	  [ "$$got" = "$(EXPECT_BACKEND)" ] || { echo "expected backend $(EXPECT_BACKEND), got $$got"; exit 1; }; \
	fi
	@jq -r '"gpu-smoke OK: adapter=\(.adapter) backend=\(.backend) device_type=\(.device_type)"' $(DET_DIR).gpu.json
	@rm -f $(DET_DIR).gpu.json

## The 10x determinism gate ("run each scenario 10x; a
## single failure fails the suite"). Drives the real CLI, not a stand-in: each
## run writes a per-frame OBJ sequence plus the stdout JSON summary, and all
## $(DET_RUNS) runs must hash byte-identically. Defaults to the implicit/GPU
## solver because that is the path where nondeterminism (unordered atomics,
## nondeterministic reductions) can actually appear.
determinism: ## Run $(DET_SCENE) $(DET_RUNS)x and require byte-identical output
	@test -x $(MGUSO) || { echo "$(MGUSO) missing — run 'make build-release' (or pass MGUSO=...)"; exit 1; }
	@rm -rf $(DET_DIR); mkdir -p $(DET_DIR)
	@echo "== determinism: $(DET_RUNS)x $(DET_SCENE) ($(DET_SOLVER), $(DET_FRAMES) frames)"
	@for i in $$(seq 1 $(DET_RUNS)); do \
	  $(MGUSO) run $(DET_SCENE) --frames $(DET_FRAMES) --solver $(DET_SOLVER) \
	    --out $(DET_DIR)/run$$i > $(DET_DIR)/run$$i.json 2>$(DET_DIR)/run$$i.log \
	    || { echo "run $$i FAILED:"; cat $(DET_DIR)/run$$i.log; exit 1; }; \
	done
	@cd $(DET_DIR) && for i in $$(seq 1 $(DET_RUNS)); do \
	  printf '%s  run%s\n' "$$(cat run$$i.json run$$i/*.obj | $(SHA256) | cut -d' ' -f1)" "$$i"; \
	done > hashes.txt
	@cat $(DET_DIR)/hashes.txt
	@n=$$(cut -d' ' -f1 $(DET_DIR)/hashes.txt | sort -u | wc -l | tr -d ' '); \
	if [ "$$n" != 1 ]; then \
	  echo "DETERMINISM FAILURE: $(DET_RUNS) runs produced $$n distinct outputs (expected 1)."; \
	  exit 1; \
	fi
	@echo "== determinism OK: $(DET_RUNS)/$(DET_RUNS) runs byte-identical"
	@rm -rf $(DET_DIR)

bench: ## Solver throughput on the canonical scene (ms/step, JSON on stdout)
	@test -x $(MGUSO) || { echo "$(MGUSO) missing — run 'make build-release' (or pass MGUSO=...)"; exit 1; }
	@$(MGUSO) bench $(DET_SCENE) --steps 200

clean-det:
	@rm -rf $(DET_DIR)
