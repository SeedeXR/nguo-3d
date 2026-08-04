<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# DEVELOPING.md — start here

The entry point for anyone about to change the nguo-3d desktop application.
Read it once end to end; it is short, and most of it is a rule that will
otherwise bite you.

Every command below is one you can run. Every architectural rule names the file
that enforces it, so when the rule and the code disagree you can tell which one
moved.

---

## 1. The two workspaces

```
nguo-3d/
  core-engine/   Rust + wgpu cloth / shell / rod contact solver. Headless.
                 Ships the `mguso` binary. Its own cargo workspace.
  ui/            The desktop application. Its own cargo workspace.
  bridges/       Blender extension and Unity headless integration.
  third_party/   Vendored sources we are obliged to ship (planegcs).
  tools/         Repeatable developer tooling.
  Makefile       The gates. CI runs these targets, not its own copies.
```

**Neither workspace contains the other**, and that is deliberate: the engine is
usable, testable and shippable without a UI. The consequence is that every
tree-wide command has to loop over both, which is what the `Makefile` does
(`WORKSPACES := core-engine ui`).

Both pin their toolchain in `rust-toolchain.toml` (`channel = "stable"`, with
`rustfmt` and `clippy`). `ui` sets `rust-version = "1.85"`.

---

## 2. Build and run

```sh
cd ui
cargo run -p nguo-app --features debug-overlay
```

That is the whole thing: splash screen, then the shell — the 2D pattern canvas,
the 3D viewport, and the egui debug overlay on top. Without the feature you get
the same app minus the overlay; `debug-overlay` is off by default and is the
only feature `nguo-app` declares.

The overlay is not a hidden dev switch. Enabling it also flips
`FEATURES["ui.debug_overlay"].enabled` in `nguo-registry`, so the application
can tell you truthfully whether the overlay was compiled in.

Other useful entry points:

```sh
cargo run -p nguo-cli -- --help          # `nguo`, the headless driver
cargo run -p nguo-mcp                    # the MCP server (same operations)
cd ../core-engine && cargo run -p engine-cli -- gpu-info    # confirm a real GPU
```

**A first build is cold and expensive.** Expect several GB in `ui/target`. Use
`cargo check` while iterating and do not run `cargo clean` casually.

---

## 3. The crate map

**22 crates** in `ui/crates` (`ls ui/crates | wc -l`, 2026-08-04). What each one
owns:

**Document and model** — no GPU, no UI framework, fast tests.

| Crate | Owns |
|---|---|
| `nguo-doc` | The authoritative document: pieces, seams, parameters, materials, avatar. Serialisation and schema migration. |
| `nguo-doc-commands` | Commands, transactions, gesture coalescing, the undo/redo stack. |
| `nguo-pattern` | Pattern geometry: stable entity ids, the vector network, seam-allowance offset. |
| `nguo-measure` | Units, length parse/format, and the numeric-field expression evaluator (`waist/4 + 1cm`). |
| `nguo-constraints` | The 2D geometric constraint solver — safe Rust over FreeCAD's `planegcs`. |
| `nguo-avatar` | The avatar bundle as typed data: the two-body catalogue, the four geometry tiers, path resolution, mesh loading, the skin library (Monk tone → linear `baseColorFactor`), and the bundle's measured girths converted to mm. It does **not** measure meshes — see [`AVATARS.md`](AVATARS.md) §6. |
| `nguo-triangulate` | Pattern → sim mesh. Curve flattening, snap-round sanitising, constrained Delaunay, refinement, **the mm → m conversion**, and the pattern ↔ mesh map (`pattern_mm`, `sources`, `chains`). One entry point, one resolution knob (§4.5). |
| `nguo-sewing` | Seam topology, segment correspondence, 3D arrangement, and the assembly that becomes a drape: `validate` → `triangulate` → `arrange` → `correspond` → `Garment`. Knows nothing about the solver — a `Stitch6` is six indices and six weights. |

**Rendering and GPU.**

| Crate | Owns |
|---|---|
| `nguo-gpu` | The only crate that knows iced's wgpu renderer, Vello and `egui-wgpu` at once. Device adoption, the frame graph, the engine→UI handoff, frame stats. |
| `nguo-canvas` | The 2D pattern canvas: a Vello scene with pan/zoom, a derived grid, a Bézier pen, snapping, hit-testing through a spatial index, and a per-piece scene cache. **Fully tested and not yet wired to the mouse** — the shell calls `frame` and nothing else. |
| `nguo-viewport3d` | The 3D drape viewport as an iced custom primitive, plus the turntable camera. |

**Shell and interaction.**

| Crate | Owns |
|---|---|
| `nguo-app` | The desktop shell: window, dock layout, the `CommandId` → handler table, the render graph. |
| `nguo-app-debug` | What the egui overlay *says* (`nguo-gpu` owns how it is drawn). Feature-gated. |
| `nguo-ui-theme` | Design tokens. The **only** place a colour literal may appear. Framework-free on purpose. |
| `nguo-ui-widgets` | Reusable iced widgets: docking, panels, numeric fields, tree, toolbar, toasts. |
| `nguo-splash` | The splash screen: a pure dismissal state machine plus its iced view. |
| `nguo-keymap` | The keyboard map — data, contexts, conflict detection, display. No input-library dependency. |

**Outside the window.**

| Crate | Owns |
|---|---|
| `nguo-engine-bridge` | Async bridge from the app to the solver: submission, epoch cancellation, frame sink, and **two transports** — `process.rs` (the `mguso` child, used by the CLI) and `inproc.rs` (in-process, used by the app, so a preview needs no built binary). |
| `nguo-cli` | `nguo`, the headless driver. `ops::SPECS` + `ops::dispatch` are the operation surface. |
| `nguo-mcp` | An MCP server whose tool list is *generated* from `nguo_cli::ops::SPECS` and which calls the same `dispatch`. One implementation, two front doors. |
| `nguo-bridge-dcc` | Headless Blender / Unity interop: executable discovery, argv construction, output parsing, axis conversion. |
| `nguo-registry` | Feature inventory (what the app can do) and the live process registry (what it is doing right now). Backs the status bar. |

### The one path that crosses most of them: sew → drape

`nguo-app/src/drape.rs` is worth reading before touching any of the above. It is
the only place the whole chain is visible at once, and its module comment
carries the measurements that justify every choice in it (tier resolution, body
clipping, why nothing is pinned, why `"self-collision"` is mandatory and
kebab-case).

```text
nguo_doc::Document  →  nguo_sewing::build  →  Garment  →  drape.rs::scene()
                       (validate · triangulate · arrange · correspond)
                                                       ↓
    nguo_viewport3d  ←  Drape::poll on the vsync tick  ←  InProcessTransport
```

---

## 4. The rules that are load-bearing

Break one of these and the failure is not local — it is a rewrite. They are
listed in the order they will bite you.

### 4.1 One wgpu major, process-wide

wgpu majors are mutually incompatible. Renderers on different majors cannot
share a device, a texture or a buffer — the types are simply different types.
This application runs **four** renderers in one process (iced, Vello, egui, and
our own 3D pass), so they must all resolve to the same wgpu major or the shell
cannot exist.

The only published set that unifies is **wgpu 27**: `iced_wgpu 0.14`,
`vello 0.7`, `egui-wgpu 0.33.3`. `core-engine` was ported from wgpu 30 to 27 to
join them. Both lockfiles resolve `wgpu 27.0.1`.

`make check-wgpu` reads both `Cargo.lock`s and fails if either contains more
than one wgpu major, printing `cargo tree -i` for each culprit. It needs no
build and no network, so it is cheap enough to run before every push. Do not
bump one renderer without the rest.

### 4.2 Two devices, one adapter

The natural design is one device shared by everyone. **iced 0.14 does not allow
it**, and this is verified against the installed sources rather than assumed:
`iced_winit` builds the compositor itself with no parameter and no setter, and
`iced_wgpu`'s `Compositor::request()` calls `request_adapter` and
`request_device` unconditionally with private fields. There is no seam to
inject a device into.

So ownership is inverted: **iced creates the UI device, and every other UI
renderer borrows it.** Everyone receives it through the two callbacks iced hands
out (`Pipeline::new`, `Primitive::prepare`) and routes through
`nguo_gpu::adopt()`, which asserts it really is the same device.

The *engine* gets its own device. `iced_wgpu` hardcodes `Limits::default()`,
which allows 8 storage buffers; the solver needs 17. One adapter, two devices —
and therefore a real seam between them.

That seam costs something, and the cost is measured, not guessed:
`ui/docs/GPU_HANDOFF.md` has the full table from
`cargo test -p nguo-gpu --release --test handoff_bench -- --ignored --nocapture`.
The short version is that the shipped pipelined `Handoff` keeps the UI thread
well inside the 8.3 ms frame budget, while the naive blocking round trip eats
all of it at 500 k vertices. If you change the handoff, re-run the bench and
update the table — do not reason about it.

### 4.3 The document is the single source of truth

`nguo-doc::Document` holds the model. Views do not. A widget that caches
document state, or holds a value the document does not have, is a bug waiting
for the next undo.

### 4.4 Mutation only through commands

`Document`'s fields are `pub(crate)`. The only way to change one is
`Document::apply_edit`, which takes a primitive `Edit` from a closed enum and
**returns the edit that undoes it**. Undo is therefore not a second
implementation that has to be kept in step — it is the same code path run on the
returned value.

`nguo-doc-commands` composes those primitives into transactions and the undo
stack, and it cannot do anything `Edit` does not allow. `Edit` is serializable
on purpose: the same values feed the undo stack, the autosave journal, and any
later collaboration work.

Practically: if you find yourself wanting a `&mut` into the document, you want
a new `Edit` variant instead.

### 4.5 Millimetres internally, metres in the solver, converted once

The UI and the document are **millimetres, always**.
`nguo_measure::Length` is the canonical type and `Unit` exists only to parse and
format — a user typing inches or centimetres changes the display, never the
storage. `Metadata::display_unit` is a presentation setting.

The engine works in **metres** (its gravity is `-9.8`, not `-9800`). **Landed
2026-08-04, and not where this section originally said it would go:** the
conversion lives in **`nguo-triangulate`**, as `MM_TO_M`
(`crates/nguo-triangulate/src/lib.rs:59`). The bridge was the wrong home because
the boundary is not one call site — the arranger, the stitch builder and the
viewport all read the same mesh, so converting "at the boundary" would have
meant converting three times. Instead `SimMesh` carries **both**:
`positions` in metres, `pattern_mm` in millimetres, and `SimMesh::check()` pins
the ratio exactly, so drift is a test failure rather than a garment 1 000× too
big.

One qualification, stated because it is a second site: `nguo_sewing::arrange`
applies `1e-3` itself to coordinates it places directly, and
`arrange::tests::wrapping_a_panel_preserves_its_width_as_arc_length` is what
keeps the two in agreement. Do not add a **third**.

**The avatar is the exception that proves the rule.** `assets/avatars/` is
authored in **metres**, i.e. already solver units, so the body → collision
proxy → `Scene` path converts **nothing** — converting there would be the bug.
`nguo_avatar::Measurements::from_avatar` converts to mm exactly once, on its way
out of the crate; the canvas ghost's `MM_PER_M` composes with the zoom rather
than converting a second time. See [`AVATARS.md`](AVATARS.md) §6.

### 4.6 Determinism is a feature

No wall-clock reads, no RNG, no unordered atomics in anything that affects a
result. This is not fastidiousness: it is what makes a drape state reproducible
from its inputs, which is what lets a simulation live in the undo history at all
(`ui/docs/UX_SPEC.md` §3.4). It is enforced by `make determinism`, which runs
the same scene ten times and requires byte-identical output.

### 4.7 Nothing outside `nguo-ui-theme` hardcodes a colour

Every widget, canvas, viewport and chrome element reads a `Tokens` value. That
is the entire reason "restyle later without breaking functionality" is true
rather than aspirational.

---

## 5. Tests

**Where the suite stands, 2026-08-04, re-run rather than recalled:**
`ui` **542 pass / 2 fail / 3 ignored**, `core-engine` **399 pass / 0 fail /
3 ignored** (both `cargo test --workspace --no-fail-fast`, debug).

The two failures are `nguo-cli/tests/cli_end_to_end.rs:141,183`, which need the
`mguso` binary and **hard-fail rather than skip** when it is missing. Build it —
`cargo build -p engine-cli` in `core-engine/`, or point `$NGUO_MGUSO` at one —
and they pass. That is the standard working as intended: a test that quietly
passes by doing nothing is worse than one that fails loudly. If you gate a test
here, write the skip notice to **fd 2** (libtest swallows `eprintln!` under
capture), give it a require-env escape hatch, and assert a floor so it cannot
pass vacuously.

One caveat on that baseline: it is a **debug** run.
`cargo test -p engine-core --release` currently fails one timing test
(`tet.rs:1045`, 4.6× measured against a 5× threshold) — a pre-existing,
tracked issue, not a regression.

Four tiers, all run by plain `cargo test`. There is no separate runner.

1. **Unit** — in-crate `#[cfg(test)]` modules. The bulk of the suite. Pure
   logic, no GPU, no window.
2. **Integration** — `crates/*/tests/*.rs`, exercising a crate through its
   public API only: `nguo-keymap/tests/defaults.rs`,
   `nguo-constraints/tests/solver.rs`, `nguo-cli/tests/cli_end_to_end.rs`,
   `nguo-mcp/tests/mcp_server.rs`, `nguo-app/tests/render_graph.rs`,
   `nguo-registry/tests/concurrency.rs`.
3. **Property** — `nguo-measure/tests/prop.rs`, `nguo-pattern/tests/prop.rs`,
   `nguo-doc-commands/tests/prop.rs`. Where an invariant is worth more than a
   handful of examples (undo really inverts, expressions are bit-exact).
4. **Golden / regression** — `nguo-doc/tests/golden`,
   `nguo-doc-commands/tests/golden`, plus `nguo-doc/tests/migration.rs`. These
   are what stop a serialisation change from silently orphaning saved files.

Two suites are **`#[ignore]`d by default** because they need hardware or
software that a machine may not have. Run them explicitly:

```sh
cargo test -p nguo-gpu --release --test handoff_bench -- --ignored --nocapture
cargo test -p nguo-bridge-dcc --test dcc_installed -- --ignored
```

**Benchmarks are `#[ignore]`d tests too, and there is one script that runs all
of them.** `make profile` (`tools/profile.sh`) collects every performance number
this repo measures and prints it against `tools/perf-baseline.tsv`;
`make profile-check` fails on a regression past that row's tolerance. It adds no
benchmark of its own, so nothing rots. Read the baseline file before quoting any
number from it — it records the hardware, the load at measurement time and why
each tolerance is as wide as it is. [`CI.md`](CI.md) §9.

A third kind of gating: **the avatar suites are bundle-gated.** With no
`assets/avatars/` — 227 MB behind git-LFS — they print a skip notice on stderr
and **pass**, which is right for a laptop and wrong for a gate.

**A clone without `git lfs pull` counts as "no bundle" too.** The directory is
tracked, so it is there, but every mesh and texture is ~130 bytes of pointer
text. `engine_io::is_lfs_pointer` recognises that and the gates skip with a
notice that says `git lfs pull`, instead of the glTF parse error this used to
produce (which is exactly how CI went red on `7ce7827`).

**25 tests are behind that gate**, counted 2026-08-04 rather than recalled:
8 in `nguo-avatar/tests/bundle.rs`, 11 in `nguo-app`'s `avatar.rs`, and 6
(1 `#[ignore]` benchmark) in `core-engine`'s `engine-api/tests/avatar_bundle.rs`.
Older figures of 13, 15 and 20 are all in circulation in this repo's docs and
comments; none of them is right. Each test that runs asserts a **floor** — the
tier vertex/triangle counts, all 10 tones, all 7 shared skin maps — so it cannot
pass vacuously with the bundle present.

`$NGUO_AVATAR_BUNDLE` relocates the bundle for **all three** suites (the
`engine-api` one honoured it only from 2026-08-04; before that it hardcoded a
crate-relative path and silently ignored the variable `make test-avatars` was
already exporting).

```sh
make test-avatars                                    # THE GATE: a skip is a FAILURE
make avatar-bundle                                   # is the bundle here? never fails
cargo test -p nguo-avatar                            # with the bundle
NGUO_AVATAR_BUNDLE=/nope cargo test -p nguo-avatar   # exercise the skip path
```

`make test` depends on `make avatar-bundle`, so a bundle-less run says
`ABSENT` at the top of its log rather than hiding the skips among ~800 test
lines. `make test-avatars` is what asserts the coverage: it refuses to run
without a bundle and sets `MGUSO_REQUIRE_AVATAR_BUNDLE=1`, `engine-api`'s own
switch that turns a skip into a failure. **The UI suite has no equivalent
in-crate switch** — the Makefile's presence check stands in for it. That gap, and
`nguo-app`'s test gate swallowing *every* `AvatarError` rather than only a
missing bundle, are tracked findings, not designs. CI runs `test-avatars` in a
dedicated job because `actions/checkout` does not fetch LFS by default; see
[`CI.md`](CI.md) §8b.

The rule the whole tree is held to: **a change that has not been run is a
draft.** Non-trivial logic ships with the smallest check that fails if the logic
breaks.

---

## 6. The gates

```sh
make ci          # everything a PR must pass — no GPU needed
make gpu         # adapter smoke test + the 10x determinism gate — needs a GPU
make help        # list every target
```

`make ci` is `fmt`, `clippy`, `check-wgpu`, `test`, `licenses`, in that order,
over both workspaces:

| Target | What it runs |
|---|---|
| `fmt` | `cargo fmt --all --check` |
| `clippy` | `cargo clippy --workspace --all-targets -- -D warnings` |
| `check-wgpu` | one wgpu major per lockfile (§4.1) |
| `test` | `cargo test --workspace --no-fail-fast` |
| `licenses` | `cargo deny check licenses bans sources` against `deny.toml` |

`--no-fail-fast` is set so one broken target never masks the others. `licenses`
needs `cargo-deny` installed (`cargo install --locked cargo-deny`) and it says
so rather than passing quietly.

**CI runs these Makefile targets, not its own copies of the commands.** That is
the only reason local and CI cannot drift. If you change a gate, change it in
the `Makefile`. `ui/docs/CI.md` records which gates have actually executed and
which have never run.

Three more targets worth knowing:

```sh
make determinism   # runs one scene 10x, requires byte-identical output
make bench         # solver throughput on the canonical scene, JSON on stdout
make test-avatars  # the avatar suites with skips turned into FAILURES
```

`test-avatars` is **not** part of `make ci`, on purpose: the avatar bundle is
227 MB behind git-LFS and `make ci` has to stay runnable on a clone made without
LFS. Run it whenever you touch the avatar path. CI runs it in a dedicated job
([`CI.md`](CI.md) §8b).

Licensing rule, non-negotiable: put
`SPDX-License-Identifier: GPL-2.0-or-later` on the first line of every new
source file. Allowed dependency licences are MIT / Apache-2.0 / BSD / MPL-2.0 /
LGPL / GPLv2+ / GPLv3; `deny.toml` is the enforcement and `make licenses` is the
gate.

---

## 7. Where to read next

| | |
|---|---|
| [`UX_SPEC.md`](UX_SPEC.md) | The interface: layout, interaction model, and why |
| [`KEYMAP.md`](KEYMAP.md) | Every binding, the contexts, the conflict model, remapping |
| [`CONSTRAINTS.md`](CONSTRAINTS.md) | The 2D constraint solver, its licence position, worked examples |
| [`GPU_HANDOFF.md`](GPU_HANDOFF.md) | Two devices, one adapter, and the measured cost of the seam |
| [`AGENTIC.md`](AGENTIC.md) | Driving the application from a script, the CLI, or an agent |
| [`INTEROP.md`](INTEROP.md) | Blender and Unity round trips, with the axis conventions written down |
| [`AVATARS.md`](AVATARS.md) | The body bundle: the four tiers, loading, the skin-tone trap, **why a full-body collision proxy costs 9.4 s/step**, and the licence position |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | The feature inventory, the process registry, logs and errors |
| [`SPLASH.md`](SPLASH.md) | The splash screen and its dismissal state machine |
| [`CI.md`](CI.md) | What the gates are, and which of them have actually run |
| [`SAMPLES.md`](SAMPLES.md) | The garment corpus used for import and solver testing |

Third-party attribution lives with the assets it covers:
[`../assets/icons/LICENSE.md`](../assets/icons/LICENSE.md) (Lucide, ISC;
Tabler, MIT), [`../assets/fonts/LICENSE.md`](../assets/fonts/LICENSE.md)
(Inter, SIL OFL 1.1), and `third_party/planegcs/PROVENANCE.md` for planegcs
(LGPL-2.1-or-later), whose obligations are set out in
[`CONSTRAINTS.md`](CONSTRAINTS.md) §1.
