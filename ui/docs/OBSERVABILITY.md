<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# OBSERVABILITY.md — the feature inventory, the live process registry, logs and errors

Two questions this is here to answer, at 3am, without reading source:

1. **What can this application do, and is that bit even switched on?**
2. **What is it doing right now, what did it just finish, and what just broke?**

`nguo-registry` answers both. 1206 lines of implementation (most of it the
feature table and doc comments), 787 lines of tests, **no new dependencies**,
and nothing that `tracing` already does.

Everything below was observed on **2026-08-02** — a file read, or a command run
with its output pasted — not recalled.

## 0. Status

| | |
|---|---|
| Feature inventory | ✅ 18 entries, `const` table, duplicate/unsorted id is a **compile error** |
| Live process registry | ✅ queued/running/done/failed/cancelled, epoch-aware, bounded history |
| Epoch model | ✅ reuses `nguo-engine-bridge`'s "newest wins, stale is silent" — no second convention |
| Error chains | ✅ `source()` walked to the root, capped at 32 |
| Snapshot API | ✅ one short lock, nothing borrowed, safe to hold across a frame |
| Determinism | ✅ nothing readable by the solver or the document; snapshot order is by id |
| Overhead | ✅ measured, §7 — 69 ns recorded, 3.3 ns disabled, 112 ns per snapshot |
| Tests | ✅ 20 unit + 4 concurrency + 4 doctests, all green; 1 `#[ignore]`d benchmark |
| Telemetry | ✅ **none**, by construction — §8 |
| Not done | no `serde` on the snapshot types (nothing needs to serialise them yet); no wiring into `nguo-app-debug` or `nguo-engine-bridge` — those crates belong to other agents, §9 |

---

## 1. What this is not

`tracing` is already a dependency of both workspaces and already used
throughout the engine. **This crate does not wrap it, replace it, or re-export
it.** If what you want is "a log line", write `tracing::info!` and stop reading.

What `tracing` structurally cannot give you, and what this crate adds:

| Need | Why `tracing` can't | What `nguo-registry` does |
|---|---|---|
| "List every feature the app has" | Spans only exist while code is running | `FEATURES`, a `const` table |
| "What is in flight *right now*" | A subscriber sees events pass; it keeps no live set | `Registry::snapshot()` |
| "What did the last 64 solves do" | Same | bounded completion + failure rings |
| "Which `RUST_LOG` string turns on the drape pipeline" | Tribal knowledge | `Feature::target` + `Feature::rust_log()` |

Anything else observability-shaped — filtering, formatting, JSON output, file
rotation, span timing — is `tracing` + `tracing-subscriber`'s job and stays
there.

---

## 2. The model

### 2.1 A **feature** is a capability, and it is compile-time data

```rust
pub struct Feature {
    pub id:        &'static str,   // "drape.preview" — stable forever
    pub name:      &'static str,   // "Drape preview"
    pub subsystem: Subsystem,      // Engine
    pub enabled:   bool,           // false = in the tree, NOT reachable from the app
    pub owner:     &'static str,   // "nguo-engine-bridge"
    pub target:    &'static str,   // "nguo_engine_bridge" — its tracing target
    pub process:   bool,           // true = instances appear in the live list
    pub summary:   &'static str,
}
```

**Static, not dynamic — and why.** A runtime registry (`inventory`, `linkme`, or
a `register()` call in every crate's init) can lie in three ways a `const` table
cannot: it depends on init order, it depends on the linker not discarding a
section it believes is dead, and it silently reports *nothing* when the
registering crate was never linked into this binary. None of those failures are
loud. A `const` table has none of them, costs zero at runtime, needs no
dependency (so no licence question), and is greppable.

It is also **the convention this tree already uses**: `nguo_cli::ops::SPECS` is
exactly this shape — a `const` slice of `&'static str`-described specs, shared
by the CLI and the MCP server. Two encodings for "a table of things the app can
do" would be one too many. (`SPECS` is the *agent-facing operation* surface;
`FEATURES` is the *capability* surface. They intentionally do not overlap.)

The one real risk with a `const` table is drift — someone adds a feature and
forgets the entry. No scheme prevents that. What this one *does* prevent, at
**compile time**, is a duplicate or out-of-order id:

```
$ # (temporarily duplicating an id, to prove the check fires)
$ cargo check -p nguo-registry
error[E0080]: evaluation panicked: nguo_registry::FEATURES must be sorted by `id`,
              ascending, with no duplicates
   --> crates/nguo-registry/src/features.rs:363:15
    |
363 |   const _: () = assert!(
    |  _______________^
```

Sortedness is not cosmetic: it is what makes the duplicate check one linear
pass, and what makes `feature()` a binary search.

`feature_const("…")` resolves an id **at compile time**, so a typo is a build
error too. That is why `Registry::start` takes a `&'static Feature` and not a
string: a process can never be registered against a feature that is not in the
inventory.

### 2.2 A **process** is one long-running, cancellable unit of work

Concretely: a drape solve (preview or bake), an import, an export, a
triangulation, a constraint solve, a DCC round-trip, the GPU handoff, the MCP
server. Not a thread, not an OS process.

Short pure calls — a seam-allowance offset, an expression evaluation — are
*features* but not *processes*. They already have `tracing` spans and putting
them in the live list would drown it. That is what `Feature::process` records.

```text
  Queued ──► Running ──► Done
    │           │
    └───────────┴──────► Failed
    └───────────┴──────► Cancelled
```

Every other transition is rejected with
`RegistryError::IllegalTransition { from, to }`. In particular **`Done → Running`
is an error**, and so is `Queued → Done` — you cannot finish work that never
started; call `run()` first.

Dropping a `Process` handle without a terminal call records **Cancelled**. That
is the whole reason the handle exists: a solver thread that panics leaves a
cancelled process behind, not one that reads as running forever.

### 2.3 Epochs — reused, not reinvented

`nguo-engine-bridge` already owns the generation/cancellation model
(`crates/nguo-engine-bridge/src/bridge.rs:13-115`): a monotonic `Epoch(u64)`,
newest wins, anything older is stale and **emits nothing at all — not even a
failure**.

The registry applies exactly that rule, per feature:

* `epoch < highest seen for this feature` → the registration is **dropped**. You
  get an inert handle (`ProcessId(0)`); nothing is recorded, nothing surfaces.
* `epoch > highest seen` → every still-active process of that feature with an
  older epoch is retired as **Cancelled**. The bridge's coalescing, made visible.
* `epoch == 0` → "not epoch-managed" (an import has no generation counter) and
  never stale.

**The registry takes the epoch as a raw `u64`, not as a type.** That is
deliberate: there is exactly one `Epoch` *type* in this tree,
`nguo_engine_bridge::Epoch`, and the registry declining to define a second one
is the only way to keep it that way — the dependency also points the wrong way
(the bridge should be free to depend on the registry, not vice-versa). Call
sites pass `epoch.0`.

### 2.4 Snapshots

```rust
pub struct Snapshot {
    pub active:   Vec<ActiveProcess>,    // ascending by ProcessId
    pub recent:   Vec<CompletedProcess>, // newest first, ≤ RECENT_CAPACITY  (64)
    pub failures: Vec<FailureRecord>,    // newest first, ≤ FAILURE_CAPACITY (32)
}
```

`Registry::snapshot()` takes the lock once, copies everything out, and releases
it before returning. The result borrows nothing, so the egui overlay can hold it
for a whole frame. No user code runs under the lock, so it cannot deadlock; a
poisoned lock is recovered with `into_inner()` rather than propagated —
observability is the last thing that should escalate a failure.

### 2.5 Determinism

The registry cannot perturb a result:

* It is **write-only** from the solver's and the document's point of view.
  `start` / `progress` / `done` return nothing to branch on except a transition
  error, and timings are readable only through `snapshot()`, which only the
  debug overlay and (eventually) the CLI call.
* Snapshot ordering is **by `ProcessId`, ascending** — never hash order — and
  ids are assigned under the lock, so what the registry *reports* is
  deterministic even when what it *observes* is racing. `tests/concurrency.rs`
  asserts this from 8 threads.
* Nothing here touches the wall clock. `Instant` is monotonic and its values
  never leave the registry.

---

## 3. The inventory today

`FEATURES` in `ui/crates/nguo-registry/src/features.rs` **is** the inventory —
read it there. It was duplicated here as an 18-row table claiming to be
"generated from `features.rs`"; there is no generator, so it was a hand-copied
second source of truth that could only ever drift.

`enabled: false` means **the code is not reachable from the running app** — for
`nguo-io`, `nguo-sewing` and `nguo-triangulate` the crate is in
`ui/Cargo.toml [workspace.dependencies]` but is not on disk yet. `enabled` for
`ui.debug_overlay` follows `nguo-app`'s `debug-overlay` cargo feature, which
`nguo-app` forwards to `nguo-registry` — the overlay is a compile-time feature,
not a debug-build guess.

---

## 4. Recipes — copy, paste, edit

### 4.1 Add a feature

One edit. Append to `FEATURES` in
`ui/crates/nguo-registry/src/features.rs`, **keeping the list sorted by `id`**:

```rust
Feature {
    id: "grading.apply",                 // stable forever; it ends up in logs and bug reports
    name: "Apply grading rule",
    subsystem: Subsystem::Pattern,
    enabled: false,                      // flip to true when the app can actually reach it
    owner: "nguo-grading",
    target: "nguo_grading",              // == owner with '-' → '_'; a test enforces this
    process: true,                       // does it belong in the live list?
    summary: "Scale a pattern piece across a size run.",  // one sentence, ends with '.'
},
```

If the list is now unsorted or has a duplicate id, `cargo check` fails with the
message in §2.1. `cargo test -p nguo-registry` additionally enforces: dotted
lower-case ascii id, non-empty name, `owner` starts with `nguo-`, `target` is
the owner's module path, and a summary that is a sentence.

Nothing else. There is no registration call and no init order. The one cargo
feature this crate has (`debug-overlay`) exists only so `enabled` can report a
compile-time fact it would otherwise have to guess at; do not add another
without the same justification.

### 4.2 Instrument a process

```rust
use nguo_registry::{feature_const, Feature};

const DRAPE: &Feature = feature_const("drape.preview");   // typo == build error

// `epoch.0` — the bridge's Epoch, raw. Use 0 if this work has no generation.
let job = nguo_registry::start(DRAPE, epoch.0);
let _span = job.span().clone().entered();                 // optional; attributes tracing events

job.progress(frame as f32 / total as f32);

match solve() {
    Ok(_)  => job.done()?,
    Err(e) => job.failed(&e)?,   // records the whole source() chain
}
// Or just drop `job` — that records Cancelled.
```

Queued-then-started work (the bridge's one-slot queue):

```rust
let job = nguo_registry::queue(DRAPE, epoch.0);
// … later, on the worker …
job.run()?;
job.done()?;
```

Use your own `Registry::new()` in tests so they do not share `GLOBAL`.

### 4.3 Read it back

```rust
// Runtime lookup by string, e.g. from a CLI flag:
if let Some(f) = nguo_registry::feature("drape.preview") {
    println!("{} ({}) — {}", f.name, f.subsystem, f.summary);
    println!("RUST_LOG={}", f.rust_log("debug"));
}

// Live state, for the debug overlay — nothing borrowed, safe across a frame:
let snap = nguo_registry::snapshot();
for p in &snap.active {
    println!("{} {} {:?} {:?}", p.id, p.feature.id, p.state, p.elapsed);
}
```

---

## 5. Logging conventions

### 5.1 Levels

| Level | For | Examples |
|---|---|---|
| `error` | Something the **user** must know about; always accompanied by a `FailureRecord` | solve diverged, export failed, DCC exited non-zero |
| `warn` | Degraded but continuing | fell back to the CPU tier, dropped preview frames, unknown key in a manifest |
| `info` | Lifecycle a user would recognise | app start, adapter chosen, project opened, drape started/finished |
| `debug` | Per-operation detail | process registered/finished, epoch superseded, phase progress |
| `trace` | Per-frame / per-iteration | solver iterations, canvas scene rebuilds |

Nothing in a per-frame path may log above `trace`. `engine`'s own contract
(`core-engine/docs/PROCESS_CONTRACT.md`) says the same thing — phase progress at
`debug`, scene build and sim start/failure at `info`/`error` — and this is the
same rule, not a second one.

### 5.2 Targets

Every crate's target is its module path (`nguo_engine_bridge`,
`nguo_constraints`, …). `Feature::target` is the **single place** that mapping
is written down.

Process spans and events use one fixed target:

```
nguo::process
```

It is fixed because it has to be: `tracing` builds its `Metadata` in a `static`
callsite (`tracing-0.1.44/src/macros.rs:2788`), so a target must be a constant
and a per-feature target is not expressible from one macro site. The feature id
travels as the **`feature` field** instead, which `EnvFilter` can match on
(`tracing-subscriber-0.3.23/src/filter/env/mod.rs:55` documents
`target[span{field=value}]=level`).

Fields on every process span: `feature`, `subsystem`, `id`, `epoch`. Completion
events add `elapsed_ms` and `state`; failures add `error` (the chain joined with
`": "`).

### 5.3 `RUST_LOG` recipes

The app defaults to `info,nguo_app=info,nguo_gpu=info`
(`crates/nguo-app/src/main.rs:208-213`); the CLI defaults to `info` and takes
`-v`/`-vv` (`crates/nguo-cli/src/cli.rs:138-152`). `RUST_LOG` always wins.

```sh
# Everything a process does, app-wide. Start here.
RUST_LOG='info,nguo::process=debug' cargo run -p nguo-app

# One feature only — this is what Feature::rust_log("debug") prints:
RUST_LOG='nguo_engine_bridge=debug,nguo::process[{feature="drape.preview"}]=debug' \
  cargo run -p nguo-app

# Per subsystem
RUST_LOG='nguo_engine_bridge=debug,nguo_gpu=debug'   # engine + handoff
RUST_LOG='nguo_constraints=debug'                    # constraint solver
RUST_LOG='nguo_canvas=debug,nguo_viewport3d=debug'   # rendering
RUST_LOG='nguo_doc=debug,nguo_doc_commands=debug'    # document + undo
RUST_LOG='nguo_bridge_dcc=debug'                     # Blender / Unity
RUST_LOG='nguo_mcp=debug,nguo_cli=debug'             # headless surfaces

# Failures only, from anywhere
RUST_LOG='warn,nguo::process=error'

# The engine binary is separate — it has its own process (PROCESS_CONTRACT.md):
RUST_LOG='engine_core=debug' mguso run scene.json
nguo drape --scene scene.json -vv --log-format json     # one JSON record per line
```

---

## 6. Error-handling policy

The project standard, unchanged:

1. **Typed errors in core crates.** `thiserror`, one enum per crate, with
   `#[source]` on the wrapped cause so the chain is walkable. `anyhow` only at
   binary boundaries.
2. **User-facing errors are actionable and non-fatal where possible.** A failed
   constraint solve shows a diagnostic; it does not kill the app.
3. **The registry never escalates.** `Process::failed(&err)` *records* — it
   returns `Ok(())` and does nothing else. Whether to show a toast is the
   caller's decision.

`FailureRecord` keeps the whole chain: `Display` of the error, then of each
`source()` in turn, down to the root (capped at 32 links so a cyclic
`source()` cannot hang the logger). Three accessors map it onto
`nguo_ui_widgets::toast(tokens, ToastLevel::Error, title, detail, on_dismiss)`:

| accessor | value | goes to |
|---|---|---|
| `title()` | the human feature name, e.g. `"Drape preview"` | toast title |
| `detail()` | the top of the chain, e.g. `"could not drape \`bodice\`"` | toast body |
| `root_cause()` | the bottom, e.g. `"solver diverged at frame 12"` | toast body, or the "details" expander |

The full chain is in the log at `error` regardless of what the toast shows. The
UI gets the short actionable form; the log keeps everything.

**Panics.** A panicking worker unwinds through `Process::drop`, which records
**Cancelled** — so the overlay never shows a process that ran forever.
`tests/concurrency.rs::a_panicking_worker_leaves_a_cancelled_process_not_a_zombie`
proves it. Crash reporting itself stays local and out of this crate
(the project collects no telemetry at all — §8).

---

## 7. Measured overhead

The maintainer's standing rule is **measure before deciding**, so these are
numbers, not adjectives.

```
$ cargo test -p nguo-registry --release --test overhead -- --ignored --nocapture

nguo-registry overhead (200000 iterations each)
--------------------------------------------------------
start + done, recording                           68.7 ns/op
start + progress + done, recording                72.6 ns/op
start + failed (2-deep chain), recording          99.3 ns/op
start + done, registry disabled                    3.3 ns/op
snapshot (16 active, full history)               111.1 ns/op
--------------------------------------------------------
recording overhead over the disabled path: 65.4 ns/op
snapshot as a share of a 8.3 ms frame budget: 0.0014 %
```

Apple M4, macOS 26.5.2, `rustc 1.96.0`, `--release`, no subscriber installed.
Three consecutive runs agreed to within 1 ns on every line except the failure
path (99–115 ns; it allocates the chain `Vec` and two `String`s). The very first
run after a rebuild reads ~148 ns on whichever benchmark goes first — that is
CPU ramp, not the code.

What this means in practice:

* **A drape solve costs 69 ns to record.** A preview solve is measured in tens
  of milliseconds. This is 6 orders of magnitude below the thing it observes.
* **The disabled path is 3.3 ns** — one relaxed atomic load and a struct
  construction, no lock, no allocation. `Registry::set_enabled(false)` is a real
  off switch, but at 65 ns of difference there is no reason to reach for it.
* **A snapshot with 16 active processes and a full 64-entry history costs
  112 ns**, i.e. **0.0014 %** of the 8.3 ms frame budget.
  Per-frame snapshotting from the egui overlay is free. There is
  consequently **no** `snapshot_into(&mut Snapshot)` buffer-reuse API — it would
  be optimising 112 ns.

Caveats, stated rather than hidden: these are **uncontended** single-thread
numbers with **no `tracing` subscriber installed**. With a subscriber the
`debug!` events dominate — that cost is `tracing`'s and is controlled by
`RUST_LOG`, not by this crate. Contended cost was not measured; the deadlock and
correctness behaviour under 6 concurrent writers plus a snapshotting reader is
covered by `tests/concurrency.rs`, not by a number.

Re-measure with the command above. Do not copy these numbers to another machine.

---

## 8. No telemetry

Project policy is **no telemetry: none.** Not opt-in analytics — none.

This crate is the place someone will eventually mistake for an analytics hook,
so, explicitly: everything here is **in-process, in-memory, bounded, and dies
with the process**. It writes no file, opens no socket, spawns nothing, and
reads no environment variable. The history rings are 64 and 32 entries and are
overwritten in place. Nothing leaves the machine, and there is no code path that
could be flipped to make it.

A tool designers run locally on unreleased collections has no business phoning
home. Crash reporting stays local too: a panic hook writes a backtrace next to
the autosave and the *user* decides whether to attach it to an issue.

If you are here to add a metrics exporter: the answer is no, and it was no
before you asked.

---

## 9. What is deliberately not built

* **No `serde` on the snapshot types.** Nothing needs to serialise them yet. Add
  `#[derive(Serialize)]` when the CLI or MCP grows a `processes` op — it is one
  line and `serde` is already a workspace dependency.
* **No `snapshot_into` / buffer reuse.** Measured at 112 ns per frame (§7).
* **No per-feature runtime enable/disable.** `Feature::enabled` is a `const`
  describing this build. A runtime toggle would be a config for a value that
  never changes at runtime.
* **No sharded lock.** One `Mutex`; every critical section is a `Vec` push or a
  scan of <100 elements with no user code inside it.
* **No log-file rotation, no JSON formatter, no panic hook.** All of that is
  `tracing-subscriber` + the binary's `main`, and the CLI already does the JSON
  half (`--log-format json`).
* **Not wired into anything yet.** `nguo-engine-bridge` (drape processes),
  `nguo-gpu` (handoff), `nguo-bridge-dcc` (round-trip) and `nguo-app-debug` (the
  overlay panel) each need three lines from §4.2. Those crates belong to other
  agents; this one owns only `nguo-registry` and this document.

Each of these has a `// ponytail:` comment at the site naming the ceiling and
the upgrade path.
