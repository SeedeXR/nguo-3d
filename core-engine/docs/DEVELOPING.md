<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# DEVELOPING.md — working on the solver

Start here if you are about to change engine code. Everything below is
checkable against this tree; if a claim and the tree disagree, the tree wins
and this file is a bug.

## 1. What this is

A cross-platform **cloth / shell / rod contact solver**, pure Rust + wgpu.

- **One backend abstraction, three platforms.** Metal on macOS, Vulkan on
  Linux, DX12 on Windows — the same WGSL. **No CUDA, no Python, no
  platform-specific source.** `WGPU_BACKEND` picks the backend explicitly.
- **Headless.** No UI, no windowing, no interactive prompts. It is a Rust
  library and a binary (`mguso`) that reads files and writes files.
- **Deterministic by construction** (§6).

The solver is a port of ZOZO's [`st-tech/ppf-contact-solver`][ppf]
(Apache-2.0), whose reference implementation is Python + Rust + C++/CUDA and
NVIDIA-only. See `../LICENSES/README.md` and `../README.md` for the licensing
position — the attribution is an obligation, not a courtesy.

## 2. Crate map

| Crate | Owns |
|---|---|
| `engine-core` | Solver core: sim state, step state machine, CPU physics for both tiers, assemblers, linear solve, contact, strain limiting. No GPU, no I/O. |
| `engine-gpu` | The `wgpu` device and every compute dispatch. WGSL lives in `crates/engine-gpu/src/shaders/`. Each Rust module owns its shader and its GPU-vs-CPU oracle test. |
| `engine-api` | Authoring surface: `Scene`, `Object`, `Material`, `SolverConfig`, `FabricPreset`, `auto_drape`, and `avatar` (body mesh → static collider → drapeable `Scene`). This is what an embedder calls. |
| `engine-io` | Mesh and sequence I/O: OBJ, STL, glTF/GLB, COLLADA, USDA, FBX, plus the binary golden-capture reader. |
| `engine-cli` | `mguso`, the headless runner. Thin — argument parsing, logging setup, JSON on stdout. |

Dependency direction, read off the manifests: `engine-cli` → all four;
`engine-io` → `engine-api` → `engine-core`; `engine-gpu` → `engine-core`.
`engine-core` depends on no other engine crate, and nothing depends on
`engine-gpu` except the CLI — the GPU is an accelerator the core does not know
about.

**Two physics tiers.** `engine-core/src/implicit.rs` is the implicit
backward-Euler tier and is where all reference-parity work happens.
`engine-core/src/solver.rs` is an XPBD preview tier kept for fast previews — it
has **no self-collision**, so anything involving contact must use the implicit
tier (`Scene::build_implicit()`, `--solver implicit`).

## 3. Build and run

```sh
cargo build --workspace
cargo test  --workspace

# Which adapter did we get, and does it compute correctly?
cargo run -p engine-cli -- gpu-info

# Simulate. A .obj is treated as a draping cloth; a .json is a full Scene.
cargo run -p engine-cli -- run examples/cloth_grid.obj --frames 60 --out /tmp/drape
cargo run -p engine-cli -- run examples/fabric_drape.json --solver implicit

# Throughput, and the invariant gate.
cargo run -p engine-cli -- bench  examples/fabric_drape.json --steps 200
cargo run -p engine-cli -- verify examples/fabric_drape.json
```

The toolchain is pinned in `../rust-toolchain.toml` (stable, with `rustfmt` and
`clippy`). Runnable scenes live in `examples/`.

`mguso` subcommands: `gpu-info`, `run`, `bench`, `verify`, `golden`,
`gauge-scene`. The last two drive a captured reference corpus that is **not
distributed with the source** — see §7.

## 4. The process contract

The full contract is `PROCESS_CONTRACT.md`. The three rules you must not break:

- **stdout is data.** One pretty JSON object per command. Nothing else — no
  progress, no banners, no warnings.
- **stderr is logs.** `tracing`, human text by default, one JSON record per
  line with `--log-format json`. Level via `RUST_LOG` or `-v`/`-vv`.
- **Exit code is the verdict.** `0` success, `1` failure (diverged, I/O error,
  unsupported input, GPU required but unavailable), `2` `verify` ran fine and
  an invariant failed. `2` is deliberately distinct from `1`: "the tool broke"
  and "the simulation is wrong" are different things to a CI job.

If you add a subcommand, it obeys all three or it does not merge.

## 5. Test tiers

| Tier | Where | What it is for |
|---|---|---|
| Unit | `#[cfg(test)]` inside each `src/*.rs` | The maths. Analytic cases, finite-difference gradient checks, symmetry/PSD properties. |
| GPU oracle | `#[cfg(test)]` inside `engine-gpu/src/*.rs` | Every kernel is asserted equal to its CPU implementation (`gpu_*_matches_cpu`). These **skip** when no adapter is available, so a green run on a headless box proves less than one on a GPU box. |
| Integration | `crates/*/tests/*.rs` | Whole readers and whole scenes through the public API. |
| Regression / golden | `engine-api/tests/regression.rs` + `engine-api/tests/golden/` | Committed golden oracles. This is the tier that catches "the physics changed and nobody noticed". |

Three integration tests are **gated on data outside this workspace** and skip
loudly when it is absent: `engine-io/tests/golden_file.rs` (the reference
capture, `PPF_GAUGE_DIR`), `engine-io/tests/samples.rs` (the garment sample
corpus, `NGUO_SAMPLES_DIR`) and `engine-api/tests/avatar_bundle.rs` (the avatar
bundle at `../../../assets/avatars`, 227 MB behind git-LFS). A clean clone runs
none of them. They still *report* as passing, because skipping is the defined
behaviour — so a green suite in a clean clone covers strictly fewer assertions
than the same suite with the data present. Read the skip messages on stderr.

**`avatar_bundle.rs` has a switch that removes the ambiguity:**
`MGUSO_REQUIRE_AVATAR_BUNDLE=1` turns its skip into a failure, so a silently
vanished asset cannot quietly disable the file. Set it on any machine that has
the bundle. The repo-root `make test-avatars` sets it and additionally refuses to
run without a bundle; that is the gate to use, and CI has a dedicated job for it
(`ui/docs/CI.md` §8b). Its skip notice is written to fd 2 directly rather than
through `eprintln!`, because libtest's capture swallows the print macros unless
the test fails.

A handful of tests are `#[ignore]`d with a reason string naming the open
problem. Read the reason before assuming a test is dead weight. One of them is
load-bearing documentation: `avatar_bundle.rs::collision_proxy_step_cost` is the
benchmark that measured **9.410 s/step** (male) and **12.533 s/step** (female)
for a whole 15 000-triangle body proxy against **0.068 s/step** with
self-collision off — i.e. the body's own triangles are the entire cost. Run it
with `cargo test -p engine-api --test avatar_bundle --release -- --ignored
--nocapture`. See `engine_api::avatar`'s module docs for what that implies (a
static mesh collider in `engine-core`/`engine-gpu` is the fix and is not written)
and `ui/docs/AVATARS.md` §5.

## 6. Determinism is a feature, not a side effect

Same input + same backend → **byte-identical** output. This is enforced, not
aspirational:

- No wall-clock, no RNG, and **no float atomics** anywhere that affects a
  result. The GPU assemblers are per-row *gather* kernels — worker `i` is the
  only writer of row `i` — precisely so that a result never depends on the
  order in which workgroups happen to retire. Core WebGPU has no float atomics
  anyway, but the CPU tier follows the same rule so the two tiers agree.
- Reductions use a fixed-shape Kahan tree, not an atomic accumulate.
- `make determinism` (repo root) runs a scene 10× through the real CLI and
  requires all ten outputs to hash identically. One differing run fails it.

Why it is worth the cost: the CUDA reference this engine is ported from is
**not reproducible against itself** — re-running the same input on the same
GPU self-diverges, because its assembly uses `atomicAdd` and float addition is
not associative. That makes tight regression gates impossible upstream. Here
they are possible. Do not trade this away for a small speed win; if you think
you must, the standing rule is that you bring the measured number.

## 7. The reference capture

`GAUGE_FINDINGS.md` analyses a recorded capture of the CUDA reference (22
scenes, inputs + per-frame checkpoints + performance traces). It is an
**internal benchmark corpus, not distributed with this source** — ~950 MB of
machine-specific output rather than code. Everything the capture
taught us that matters for porting is written down in `GAUGE_FINDINGS.md`,
`CONFIG.md` and `KERNELS.md`, so those documents are usable without it.

The code paths that read it (`engine-io/src/golden.rs`, `mguso golden`,
`mguso gauge-scene`) are still built and still tested for their parsing logic;
they simply find nothing to read in a clean clone.

## 8. The wgpu pin — 27, process-wide

`wgpu` is pinned at **27** and the pin is a product-wide decision, not an
engine preference. The renderers that share this process create **one**
`wgpu::Device`; wgpu majors are mutually incompatible, so every consumer of
that device must agree on the major. 27 is the only published version all of
them satisfy today.

Consequences for you:

- **Do not bump `wgpu` in `Cargo.toml` alone.** The move happens across every
  consumer in one change, or not at all.
- `make check-wgpu` (repo root) reads the lockfiles and fails if a workspace
  resolves more than one wgpu major. It needs no build and no network.
- When the engine is handed a device it does not own (`GpuContext::from_device`),
  the device must have been created with `GpuContext::required_limits(&adapter)`.
  The solver's SpMV binds more storage buffers than `Limits::downlevel_defaults()`
  allows, and the failure surfaces at *pipeline creation*, far from the cause.

`WGPU_PIN.md` has the version table, the full 30 → 27 API delta, and the
device-sharing recipe.

## 9. Gates

The GitHub Actions workflows call the `Makefile` targets at the repository root
rather than inlining cargo commands, so local and CI cannot drift. Do not
hand-roll the commands. (Note: the workflows exist but have not executed yet —
see the status section of `../README.md`.)

```sh
make ci            # fmt + clippy + check-wgpu + test + licenses. No GPU needed.
make gpu           # adapter smoke test + the 10x determinism gate. Needs a real GPU.
make test-avatars  # the avatar suites with skips turned into failures. Needs the bundle.
make help          # every target
```

`test-avatars` is deliberately outside `make ci`: the avatar bundle is 227 MB
behind git-LFS and `make ci` must stay runnable on a clone made without it.
`make test` prints whether the bundle is present before running, so a
bundle-less run cannot hide its skips.

`make ci` must be clean before you call anything done: `cargo fmt --all
--check`, `cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo test --workspace --no-fail-fast` all green, a single wgpu major in each
lockfile, and `cargo-deny` happy with the licence allow-list.

Two more standing rules, because they are the ones that actually get broken:

- **Put `SPDX-License-Identifier: GPL-2.0-or-later` on every new source file.**
  `cargo-deny` will not catch a missing header; review will.
- **Measure, do not estimate.** Any performance or cost claim in a comment, a
  commit message or a doc is a number you produced on hardware you name. If it
  cannot be measured here, label it unmeasured.

## 10. The other documents

| File | What it answers |
|---|---|
| `PROCESS_CONTRACT.md` | How `mguso` behaves as a process: streams, exit codes, logging, commands. |
| `CONFIG.md` | Every solver parameter and its reference default, with the f64-vs-f32 map. Read before touching `SolverConfig` or `Material`. |
| `KERNELS.md` | The GPU porting backlog: which kernels exist, which do not. |
| `WGPU_PIN.md` | Why 27, the 30 → 27 API deltas, and how to share a device. |
| `AUTO_DRAPE.md` | The one-click drape layer and its licensing position. |
| `GAUGE_FINDINGS.md` | What the reference capture measured: acceptance criteria, phase costs, preconditioner findings. |

[ppf]: https://github.com/st-tech/ppf-contact-solver
