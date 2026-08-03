<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# nguo-3d

**Open-source 2D/3D garment design.** Draft a pattern, sew it, drape it on a
body, export it to Blender or Unity — natively, on your own machine, with no
subscription and no cloud.

*nguo* is Swahili for cloth.

Created by **Seede XR**. First developer: **Alex Mkwizu**
<a.mkwizu@seedexr.com>. Code is **GPL-2.0-or-later**; the avatar assets in
`assets/avatars/` are **CC BY 4.0** — [`LICENSING.md`](LICENSING.md) §0 has the
split and why.

> **Status: early. Under active construction.** The simulation engine is mature
> and tested; the application on top of it is being built now. Nothing here has
> been released, and the cross-platform claim is *untested* — see
> [What is actually true today](#what-is-actually-true-today). We would rather
> say that than imply otherwise.

---

## What it is meant to be

An industrial garment CAD suite, informed by established practice in the field
and built around five things:

- **Parametric.** Patterns are driven by constraints and measurements, so one
  block generates every size and every made-to-measure fit. Drafting is not a
  manual redraw per size; it is a model you re-solve. This is the
  differentiator.
- **Robust.** The solver is a port of an implicit barrier (IPC) contact solver.
  It starts from tangled garments and untangles them instead of exploding.
- **Scriptable and agent-drivable.** A headless CLI and an **MCP server** mean
  Claude Code — or any agentic tool — can drive the whole application. That is a
  first-class product surface, not a debug hatch.
- **Interoperable.** Direct, headless, tested round-trips with Blender and
  Unity, with the axis conventions written down instead of guessed at.
- **Bodied out of the box.** Two rigged, measured human bodies ship with it,
  CC BY 4.0 and ours — so drafting to a real stature and draping on a real torso
  needs no third-party model with a research-only licence attached.
- **Yours.** GPL, no telemetry, no account, no network dependency.

## Layout

```
core-engine/   Rust + wgpu cloth/shell/rod contact solver. Headless. `mguso`.
ui/            The desktop application (iced + Vello + egui on one wgpu device).
bridges/       Blender extension and Unity headless integration.
assets/        Shared runtime data used by both workspaces — the avatar bundle.
third_party/   Vendored third-party sources (planegcs, Eigen) with provenance.
tools/         Repeatable developer tooling (knowledge-graph builds, …).
```

`assets/avatars/` is **227 MB** (70 files) and its `.glb`/`.vrm`/`.png` files are
tracked with **Git LFS**. Clone with `git lfs` installed, or set
`NGUO_AVATAR_BUNDLE` to a copy elsewhere — with no bundle the application still
runs, it just shows no body. The **code is GPL-2.0-or-later; the avatar assets
are CC BY 4.0** — see [`LICENSING.md`](LICENSING.md) §0.

`instructions/` and `memory/` also exist in a working tree but are gitignored:
they are specifications and engineering memory, i.e. working state rather than
published documentation. Start from `ui/docs/` and `core-engine/docs/`.

Two cargo workspaces: `core-engine/` and `ui/`. Both build with the pinned
stable toolchain in their `rust-toolchain.toml`.

## Build

```sh
# Engine
cd core-engine
cargo build --workspace
cargo test  --workspace --no-fail-fast
cargo run -p engine-cli -- gpu-info      # confirm a real GPU backend

# Application
cd ../ui
cargo run -p nguo-app
```

## What is actually true today

Measured on 2026-08-03, Apple M4 / macOS, not recalled:

| | |
|---|---|
| Engine tests | **393 pass, 0 fail** (3 ignored) |
| Application tests | **400 pass, 0 fail** |
| Gates | `cargo fmt --check` and `cargo clippy --all-targets -D warnings` clean, both workspaces |
| GPU backend | **Metal / Apple M4 verified**, compute probe green |
| wgpu version | **27.0.1**, single version across the whole tree — [why](core-engine/docs/WGPU_PIN.md) |
| Application | **runs** — splash → workspace at **8.35–8.74 ms** with an avatar on screen |
| Avatars | **2 bodies load, display and measure.** Licence resolved: **CC BY 4.0, ours** |
| Draping on a body | **not interactive.** A full-body collision proxy costs **9.4–12.5 s/step** — see below |
| Linux / Windows | **unverified.** The repository has 1 commit and no push, so CI has never executed |

The frame time has a floor worth naming: present mode is `Fifo` on a 120 Hz
display, so **8.33 ms is the vsync cadence**, not a measurement of headroom.
That figure says the application keeps up; it does not say by how much. Without
the avatar the same capture reads 8.35–8.56 ms — the 50 000-triangle body costs
nothing measurable to draw.

The engine models every element type in its 22-scene reference capture — cloth
membrane/shell, tets, hinges, rods, pins, colliders, strain limiting,
plasticity, air drag and seams — after 118 development sessions. Details in
[`core-engine/docs/`](core-engine/docs/).

### Avatars: they load and display; draping on one is not interactive yet

`assets/avatars/` holds **two** rigged bodies — `male` and `female` — on a shared
65-bone Mixamo skeleton, in four geometry tiers (archival 800 k–843 k vertices
down to a 7 500-vertex collision proxy; the default runtime tier is **LOD2, 25 k
vertices / 50 k triangles**), with ten Monk skin tones and nine measured girths
per body. They render in the 3D viewport, ghost behind the 2D pattern canvas,
feed the solver as a collision proxy, and both pickers are on the command
palette.

**The licence question is settled.** The bodies, rigs, weights and LODs are
**Seede XR's own work, licensed CC BY 4.0** — so are the skin textures
(texturecan.com, cgbookcase.com) and the Monk Skin Tone Scale (Ellis Monk /
Google). **Attribution is required for all three**, and one consolidated notice
discharges it. The earlier release blocker over unverified body provenance was
closed by *replacing* that bundle, not by re-licensing it. Full position:
[`LICENSING.md`](LICENSING.md) §0, §4, §5.

**What is honestly not true yet:** you cannot drape against a whole body at
interactive speed. Because the body enters the solver as static geometry inside
the cloth self-collision pass, a 15 000-triangle full-body proxy measures
**9.410 s/step** (male) and **12.533 s/step** (female) at dt = 1/120 — the same
scene with self-collision off is 0.068 s/step, so the cost is the body's own
triangles being tested against each other, not the body in the linear system.
Decimation does not rescue it: from two measured points the cost exponent is
≈1.35, which puts an interactive whole body at ~520 triangles. The fix is a
static mesh collider inside the solver, and it is **not written**. What ships
today is an AABB region crop (`engine_api::avatar::avatar_collider_near`), which
lands in the same ~0.05 s/step bracket as everything else because a garment only
ever touches part of a body. Detail and the full table:
[`ui/docs/AVATARS.md`](ui/docs/AVATARS.md) §5.

## Documentation

Read in this order:

| Document | What it answers |
|---|---|
| [`ui/docs/DEVELOPING.md`](ui/docs/DEVELOPING.md) | **Contributors start here** — build, run, test the application |
| [`core-engine/docs/DEVELOPING.md`](core-engine/docs/DEVELOPING.md) | Build, run and test the solver |
| [`LICENSING.md`](LICENSING.md) | The licence policy, the commercial-use rule, and per-component attribution |
| [`ui/docs/AGENTIC.md`](ui/docs/AGENTIC.md) | Driving the application from the CLI and the MCP server |
| [`ui/docs/INTEROP.md`](ui/docs/INTEROP.md) | Blender and Unity round-trips, units and axis conventions |
| [`ui/docs/AVATARS.md`](ui/docs/AVATARS.md) | The body bundle: tiers, loading, skin tone, **why a full-body collision proxy is unusable today**, and the licence position |
| [`core-engine/docs/PROCESS_CONTRACT.md`](core-engine/docs/PROCESS_CONTRACT.md) | The app↔engine process boundary |
| [`core-engine/docs/`](core-engine/docs/) | Solver internals, kernels, config parity |

Project-level engineering memory (`memory/`) is deliberately not published —
it is working state, not documentation.

## Contributing

Two rules above all others, both taken seriously:

1. **Zero hallucination.** Every claim about code, an API or system state is
   observed and cited (`file:line`, or the command and its output), or is
   explicitly labelled unverified. This project's own docs have claimed green
   tests that were not green and a CI workflow that did not exist; both were
   caught by running the commands.
2. **Code that has not been run is a draft.** Non-trivial logic leaves a
   runnable check behind. Compiling is not evidence.

Beyond that: SPDX header on every file, no dependency outside the allowed
licence set, and `cargo fmt` / `cargo clippy -D warnings` / the test suite green
before you hand anything over — `make ci` runs all of it. Dependency licences
are policy, not preference: read [`LICENSING.md`](LICENSING.md) before adding
one.

## Licence

**Code: GPL-2.0-or-later** — deliberately Blender's licence, because that is the
ecosystem this lives in. The Blender extension under `bridges/blender/` is
GPL-3.0-or-later; see its own header.

**Avatar assets: CC BY 4.0**, licensor Seede XR. A licence written to keep
software source open is the wrong instrument for a mesh, which is why Blender
splits the same way. Attribution is required — including to us, and to
texturecan.com / cgbookcase.com for the skin textures and to Ellis Monk / Google
for the Monk Skin Tone Scale. The consolidated notice is in
`assets/avatars/LICENSE.md`.

**Every dependency must permit commercial use.** Copyleft is not the same thing
as non-commercial — GPL, LGPL and MPL all permit commercial use, they constrain
distribution terms. What is rejected is the non-commercial family (CC BY-NC,
SSPL, BUSL, Elastic, research-only), plus GPL-2.0-**only** because it blocks
this tree's GPLv3 upgrade path. Where a capability exists only under a
non-commercial licence, we build it ourselves. The allow-list is machine-checked
in `deny.toml` and blocking in CI (`make licenses`).

Full policy and per-component attribution: [`LICENSING.md`](LICENSING.md).

The solver is a port of ZOZO's
[`ppf-contact-solver`](https://github.com/st-tech/ppf-contact-solver)
(Apache-2.0), conveyed here under GPL-2.0-or-later. See `core-engine/LICENSES/`.
