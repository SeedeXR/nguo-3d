<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# PROCESS_CONTRACT.md

How the engine behaves as a process, so it embeds anywhere (CI, render farm,
another app) with predictable, machine-readable behavior.

## Binary
`mguso` (crate `engine-cli`). Headless, no UI, no interactive prompts.

## Exit codes
| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Failure: solver diverged (non-finite positions), I/O error, unsupported input, or GPU required but unavailable |
| 2 | `mguso verify` only: the run completed but a solver invariant failed |

`1` and `2` are deliberately distinct — "the tool broke" and "the simulation is
wrong" are different things to a CI job. `1` always accompanies a logged
`ERROR`. The engine never exits 0 on a failed simulation.

## Streams
- **stdout** — machine-readable results only: a single pretty JSON object per
  command (GPU report, or simulation summary with `frames`, `verts`,
  `final_time`, `final_kinetic_energy`, `bbox_min/max`). Parseable; nothing else
  is written to stdout.
- **stderr** — human/structured logs (`tracing`). Never mixed into stdout.

## Logging
- Levels via `RUST_LOG` (e.g. `RUST_LOG=engine_core=debug`) or `-v`/`-vv`.
- `--log-format json` emits one JSON log record per line (ingestible by log
  pipelines). Default is human-readable text.
- Every step logs phase progress at `debug`; scene build and sim start/failure
  at `info`/`error`.

## Determinism contract
- Same input + same backend → identical output. No wall-clock, no RNG, no atomic
  scatter in the CPU tier; fixed iteration order.
- The regression suite (`cargo test -p engine-api --test regression`) enforces
  this against committed golden oracles.
- `make determinism` (repository root) runs a scene 10× through this binary and
  requires all ten outputs to hash identically. One differing run fails it.

## Commands
- `mguso gpu-info` — prints selected backend/adapter + runs the no-op GPU probe.
- `mguso run <scene.json|mesh.obj> [--frames N] [--out DIR] [--solver xpbd|implicit]`
  — simulate; write a per-frame OBJ sequence to `--out` if given.
- `mguso bench <scene> [--steps N]` — solver throughput, JSON on stdout.
- `mguso verify <scene> [--steps N]` — run and check the solver invariants
  (penetration-free, strain-limit, CG-converged, zero-steady-state-alloc,
  all-finite). Exit 2 on invariant failure. See `GAUGE_FINDINGS.md` §1.
- `mguso golden <dir>` / `mguso gauge-scene <dir>` — drive a captured reference
  scene and report the divergence gates. The capture is an internal benchmark
  corpus **not distributed with this source** (`GAUGE_FINDINGS.md`).

## Cross-platform
Pure Rust + wgpu; one binary builds and runs on macOS (Metal), Linux (Vulkan),
Windows (DX12). `WGPU_BACKEND` env selects the backend explicitly for CI.
No CUDA, no Python, no platform-specific source. **Only Metal has actually been
exercised** — see the status section in `../README.md`.
