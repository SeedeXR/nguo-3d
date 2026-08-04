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

Apple M4 / macOS, not recalled. The test counts, the gates and every drape
figure were re-run on 2026-08-04; the GPU, wgpu and avatar rows were verified
earlier and are unchanged since.

| | |
|---|---|
| Engine tests | **399 pass, 0 fail** (3 ignored) |
| Application tests | **542 pass, 2 fail** (3 ignored) — the two need a built `mguso`; see below |
| Gates | `cargo fmt --check` and `cargo clippy --all-targets -D warnings` clean, both workspaces |
| GPU backend | **Metal / Apple M4 verified**, compute probe green |
| wgpu version | **27.0.1**, single version across the whole tree — [why](core-engine/docs/WGPU_PIN.md) |
| Application | **runs** — splash → workspace with an avatar on screen |
| Avatars | **2 bodies load, display and measure.** Licence resolved: **CC BY 4.0, ours** |
| **Sew → drape on a body** | **works, and was watched on screen.** A 36-step preview streams in **4.68 s** |
| **Drafting with the mouse** | **not wired.** The canvas draws the document; no pointer gesture edits it |
| Linux / Windows | **advisory only.** The `cross-platform` job (ubuntu/lavapipe, windows/DX12) **passed** on the last push; it is not a blocking check |
| CI on `main` | **RED.** The last push failed `CI → test` and `Licenses` — see below |

The two failing application tests are `nguo-cli`'s end-to-end drape checks. They
hard-fail rather than skip when `core-engine/target/release/mguso` is absent,
which is deliberate — this project does not ship tests that pass by doing
nothing. Their own failure message names both fixes: build it with
`cargo build -p engine-cli` in `core-engine/`, or point `$NGUO_MGUSO` at one.

### CI went red on `main`, and the reason is instructive

The repository has **2 commits, both pushed** (`5077aef` 2026-08-02, `7ce7827`
2026-08-03), and CI has executed on both. The first push was green across all
three workflows. The second — the one that added the 227 MB avatar bundle —
went red on two of them, verified with `gh run view`. **Both are fixed in the
working tree as of 2026-08-04 and await a commit:**

- **`CI → cargo test (both workspaces)` failed.** The `test` job checks out with
  `lfs: false`, so `assets/avatars/**/*.glb` arrive as ~130-byte LFS **pointer
  text**. Pointer files are not *missing*, so the bundle-gated tests did not skip
  — they failed inside the glTF reader. Four `engine-api` avatar tests panicked at
  `avatar_bundle.rs:168` (and three more in `ui` behind them, never reached
  because `make test` stops at the first workspace). `ui/docs/CI.md` §8b
  predicted this exact failure before the bundle was committed, which is the one
  consolation. **Fixed** by teaching the shared loader what a pointer is:
  `engine_io::is_lfs_pointer` makes an unfetched asset count as an *absent* one,
  so the gates skip — and `MGUSO_REQUIRE_AVATAR_BUNDLE=1` still turns that skip
  into a hard failure in the `avatars` job, which is the one that pays for LFS.
  It also turns a fresh clone's confusing glTF error into "run `git lfs pull`".
- **`Licenses → attribution-check` failed**: the generated
  `LICENSES/THIRD_PARTY.md` was stale (`Makefile:189`) — permanently, and through
  no one's neglect. `approx 0.5.1` ships its Apache-2.0 licence with **CRLF**;
  `cargo-about` copies it verbatim; `.gitattributes` normalises `*.md` to LF on
  commit. So the committed file could never again equal the generator's output,
  while a local check run against the un-normalised working tree passed happily.
  **Fixed** by normalising the generator's output (`tr -d '\r'` in `ABOUT_GEN`),
  so generated bytes equal committed bytes on every platform.
- **What passed on that same push:** the Metal smoke + 10× determinism gate,
  fmt/clippy/wgpu-pin, the release build, the perf job, **and the dedicated
  `avatars` job with `lfs: true`** — 13m34s, green. The LFS design works; the
  job that does not opt into it is the one that broke.
- **One thing the maintainer must check:** the `SeedeXR` org is on GitHub's
  **free** plan, so LFS bandwidth is **1 GiB/month** — about **four** runs of the
  `avatars` job at 227 MB each. See `ui/docs/CI.md` §8b.

### The honest shape of "it works"

You can open the application, see a pattern in the 2D canvas and a body in the
3D viewport, press one key, and watch two panels sew into a tube and drape onto
that body — the seams close from **212.3 mm** of gap to **31.3 mm**. That whole
path is real and tested end to end.

What you **cannot** do yet is draw the pattern. `nguo-canvas` has the pen, the
snapping, the hit-testing and the spatial index, all tested; the shell renders
the canvas and never routes a pointer event into it. The document you drape is
the one the application opens with, plus whatever a command edits. So the
accurate phrase is **sew → drape**, not draft → sew → drape, and it stays that
way until the tool dispatcher lands.

The engine models every element type in its 22-scene reference capture — cloth
membrane/shell, tets, hinges, rods, pins, colliders, strain limiting,
plasticity, air drag and seams — after 118 development sessions. Details in
[`core-engine/docs/`](core-engine/docs/).

### Avatars: they load, display, and get draped on

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

**What changed on 2026-08-04.** Until then a full-body collision proxy cost
**9.4–12.5 s/step** and nothing could drape on one. The body enters the solver
as static geometry inside the cloth self-collision pass, so its own 15 000
triangles were being tested against **each other**. The fix was a filter, not a
new collider type: a candidate contact pair whose participants are all immovable
is provably inert — the solver drops all four of its rows anyway — so the
broadphase stops emitting them. Measured on the same benchmark, `--release`,
dt = 1/120, medians of 3 runs:

| 15 050-triangle proxy | before | after |
|---|---|---|
| male, self-collision on | 9.724 s/step | **0.235 s/step** |
| female, self-collision on | 15.951 s/step | **0.245 s/step** |
| either, self-collision off | 0.068 s/step | 0.068 s/step |

**41×**, and the remaining 3.5× over the self-collision-off case is a normal
ratio for a cloth solver, not a pathology. A 300-frame reference drape went from
9.4 hours to 14 minutes.

**What is still not true:** a drape is not interactive. The shipped preview tier
runs at **0.051 s/step** and takes about four seconds for its 36 steps; the
application streams those frames and stays responsive, but you are watching a
simulation finish, not dragging cloth in real time. And there is a **known,
unfixed defect**: the mesh contact path uses an unsigned distance, so a garment
that starts even 10 µm *inside* the body falls straight through it, silently.
The default arrangement clears the body with margin, so you have to go out of
your way to hit it — but nothing validates it before the solve. Detail and the
full table: [`ui/docs/AVATARS.md`](ui/docs/AVATARS.md) §5.

## Performance

Every number below was produced by `tools/profile.sh` (`make profile`) or by a
benchmark it calls. None is an estimate. `tools/perf-baseline.tsv` carries each
one with the hardware, the toolchain, the date, the machine load at the time,
and the reasoning behind its tolerance; `ui/docs/CI.md` §9 explains how the
`perf` job uses it.

**Hardware:** Apple M4, macOS 26.5.2 (build 25F84), arm64, unified memory, no
discrete GPU. `rustc 1.96.0`, `--release`. **Re-measure on new hardware; never
copy these.**

### Guarded baselines

| Metric | Baseline | Tolerance | What it is |
|---|---|---|---|
| `solver_ms_per_step` | 20.952 ms/step | 2× | `mguso bench` on `fabric_drape.json`, 200 steps. Only 100 vertices, so this is **GPU dispatch latency**, not cloth maths |
| `registry_ns_per_op` | 65.4 ns/op | 4× | Observability overhead per `start`+`done` with recording on |
| `handoff_blocking_ms_100k` | 2.885 ms | 3× | Blocking engine→UI readback + upload round trip, 100 k vertices |
| `handoff_pipelined_ms_100k` | 0.325 ms | 3× | The same transfer triple-buffered — ~9× better, and this row is what would catch pipelining being lost |
| `proxy_step_s_selfcol_off` | 0.066 s/step | 3× | Full collision proxy (10.3 k v / 15.05 k tris) + a garment, self-collision off |
| `proxy_step_s_selfcol_on` | 0.253 s/step | 4× | The same scene with self-collision on — the metric D-17 moved by 41× |

**The tolerances are wide because the noise is wide, and that was measured too.**
Identical binaries, back to back on bare metal: `mguso bench` spread 1.39×, the
registry 2.01× (and in one run the strictly-*more*-work variant came out
faster), the blocking handoff 1.34×, the collision proxy 3.76×. A ±20 % gate on
any of that would fire daily and be muted within a week. These guards catch
**structural** regressions — pipelining lost, a device sync added, a copy that
used to be a borrow. Nothing here catches a 30 % slowdown, and we would rather
say so than imply a precision we do not have.

### The drape path, measured end to end

Sample garment on the shipped collision proxy, dev profile, medians:

| Stage | Cost |
|---|---|
| Collision proxy load (10 531 v / 15 000 t) | 25.9 ms, once, cached |
| Sew — validate + triangulate + arrange + correspond | **0.73 ms** → 346 v / 608 t / 22 stitches |
| Submit → first frame on screen | 69 ms |
| Solver step, preview tier (3 717 v / 5 979 t, self-collision on) | **0.051 s** idle, 0.106 s under load |
| Solver step, same garment on the **whole** body (10 877 v / 15 608 t) | 0.119 s |
| Whole 36-step preview | **4.68 s**, streamed frame by frame |

Two levers came out of that table and both are counter-intuitive. Clipping the
body to the band the garment can reach is worth **2.3×** for twenty lines of
code. Halving the triangle edge costs **3.2×** for only 27 % more vertices,
because the self-collision grid is indexed by the moving primitives — so the
preview tier is coarse on purpose.

### Frame time, and the caveat that must travel with it

The application renders at **8.35–8.74 ms/frame** with an avatar on screen,
8.35–8.56 ms without, and 8.53 ms with a drape animating. **All three numbers
are vsync-floored and none of them measures headroom.** Present mode is `Fifo`
on a 120 Hz display, so 8.33 ms is the cadence, not the cost: a GPU four times
too slow would still read 8.33 ms right up until it started dropping frames, at
which point the figure jumps to 16.6 ms. What these numbers honestly say is that
the application keeps its cadence — including with a solve in flight, because
the solver runs on a blocking pool off the render thread. They do not say by how
much. `make profile` stays headless and deliberately refuses to guard them.

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
