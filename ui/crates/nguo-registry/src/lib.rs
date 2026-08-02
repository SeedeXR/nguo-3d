// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-registry` — what the application can do, and what it is doing right
//! now.
//!
//! Two things, and deliberately nothing else:
//!
//! 1. [`FEATURES`] — a compile-time inventory of every feature and every
//!    long-running process, as data: stable id, name, subsystem, whether it is
//!    wired up, the owning crate, and its `tracing` target.
//!    Queryable at runtime with [`feature`], and at compile time with
//!    [`feature_const`].
//! 2. [`Registry`] — the live process list. Which processes are active, with
//!    state, progress, elapsed time and epoch; the last
//!    [`RECENT_CAPACITY`] completions with their outcome and duration; the
//!    last [`FAILURE_CAPACITY`] failures with their full error chain.
//!    [`Registry::snapshot`] copies all of that out under one short lock, for
//!    the egui debug overlay to render without holding anything across a
//!    frame.
//!
//! # This is not a logging framework
//!
//! `tracing` already is one, it is already a dependency of both workspaces,
//! and it is already used throughout the engine. This crate does not wrap it,
//! replace it or re-export it. It adds the two things `tracing` structurally
//! cannot give you: a *queryable inventory* of capabilities, and a *live view*
//! of in-flight work. Every process here also opens a `tracing` span under the
//! target [`PROCESS_TARGET`], so existing `RUST_LOG` tooling keeps working —
//! see `ui/docs/OBSERVABILITY.md`.
//!
//! # No telemetry
//!
//! project policy is **no telemetry: none**. Not opt-in analytics, none. Everything
//! in this crate is in-process, in-memory, bounded, and dies with the process.
//! Nothing here writes a file, opens a socket, or leaves the machine. If you
//! are reading this because you want to add an analytics hook: the answer is
//! no, and it was no before you asked.
//!
//! # Epochs
//!
//! `nguo-engine-bridge` already owns the generation/cancellation model: a
//! monotonic `Epoch(u64)`, newest wins, anything older is stale and its output
//! is dropped ([`nguo_engine_bridge::EngineBridge::submit`]). This crate takes
//! the epoch as a **raw `u64`** precisely so it does not become a second
//! `Epoch` type in the tree — pass `epoch.0`. Semantics are identical: a
//! process registered with an epoch lower than the highest one already seen
//! *for the same feature* is dropped, and a higher one cancels the older
//! active processes of that feature.
//!
//! `0` means "not epoch-managed" and is never stale — an import has no
//! generation counter.
//!
//! [`nguo_engine_bridge::EngineBridge::submit`]: https://docs.rs/nguo-engine-bridge
//!
//! # Determinism
//!
//! Nothing here can perturb a result. The registry is write-only from the
//! solver's and the document's point of view: `start`/`progress`/`done` return
//! nothing a caller can branch on except a transition error, timings are read
//! back only through [`Registry::snapshot`], and no code outside the debug
//! overlay and the CLI is expected to call that. Snapshot ordering is by
//! `ProcessId`, ascending — never hash order — so what the registry *reports*
//! is deterministic even when what it *observes* is racing.
//!
//! # Example
//!
//! ```
//! use nguo_registry::{feature_const, Feature, Registry};
//!
//! const DRAPE: &Feature = feature_const("drape.preview");
//!
//! let registry = Registry::new();
//! let job = registry.start(DRAPE, 7);
//! job.progress(0.5);
//! job.done().unwrap();
//!
//! let snap = registry.snapshot();
//! assert!(snap.active.is_empty());
//! assert_eq!(snap.recent[0].feature.id, "drape.preview");
//! ```

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

mod features;
mod process;

pub use features::{feature, feature_const, Feature, Subsystem, FEATURES};
pub use process::{ActiveProcess, CompletedProcess, FailureRecord, ProcessId, ProcessState};

/// The `tracing` target every process span and event is emitted under.
///
/// It is a fixed string on purpose: `tracing`'s target lives in a `static`
/// callsite `Metadata`, so it must be a constant — a per-feature target is not
/// expressible from one macro site. The feature id travels as the `feature`
/// **field** instead, which `EnvFilter` can match:
/// `RUST_LOG='nguo::process[{feature="drape.preview"}]=debug'`.
/// [`Feature::rust_log`] writes that string for you.
pub const PROCESS_TARGET: &str = "nguo::process";

/// How many completed processes a registry remembers.
pub const RECENT_CAPACITY: usize = 64;

/// How many failures a registry remembers, with their error chains.
pub const FAILURE_CAPACITY: usize = 32;

/// The only thing that can go wrong in here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    /// A caller asked for a lifecycle step that does not exist.
    #[error("illegal process transition: {from} -> {to}")]
    IllegalTransition {
        from: ProcessState,
        to: ProcessState,
    },
}

/// Everything the debug overlay needs, copied out in one lock acquisition.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    /// Queued and running, ascending by [`ProcessId`].
    pub active: Vec<ActiveProcess>,
    /// Most recently completed **first**, up to [`RECENT_CAPACITY`].
    pub recent: Vec<CompletedProcess>,
    /// Most recent failure **first**, up to [`FAILURE_CAPACITY`].
    pub failures: Vec<FailureRecord>,
}

struct ActiveRecord {
    id: ProcessId,
    feature: &'static Feature,
    state: ProcessState,
    epoch: u64,
    progress: Option<f32>,
    started: Instant,
}

struct Inner {
    next_id: u64,
    active: Vec<ActiveRecord>,
    recent: Vec<CompletedProcess>,
    failures: Vec<FailureRecord>,
    /// Highest epoch seen per feature id — the same "newest wins" rule
    /// `nguo-engine-bridge` applies with its `latest` atomic, kept per feature
    /// because two subsystems have independent generation counters.
    // ponytail: a `BTreeMap` for a table that will hold ~10 entries. It is
    // deterministic to iterate and never shows in a profile; make it a fixed
    // `[u64; FEATURES.len()]` if it ever does.
    max_epoch: BTreeMap<&'static str, u64>,
}

/// The live process registry.
///
/// Use [`GLOBAL`] in the application; construct your own in tests so they do
/// not share state.
pub struct Registry {
    enabled: AtomicBool,
    // ponytail: one global mutex. Every critical section is a `Vec` push or a
    // linear scan of <100 elements with no user code inside it, so it cannot
    // deadlock and does not contend measurably (see the overhead numbers in
    // `ui/docs/OBSERVABILITY.md`). Shard per subsystem if that ever changes.
    inner: Mutex<Inner>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// An empty, enabled registry.
    pub const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            inner: Mutex::new(Inner {
                next_id: 0,
                active: Vec::new(),
                recent: Vec::new(),
                failures: Vec::new(),
                max_epoch: BTreeMap::new(),
            }),
        }
    }

    /// Turn recording off. Handles handed out afterwards are inert: no lock is
    /// taken, no allocation happens, and every method on them is a no-op that
    /// still returns `Ok`.
    ///
    /// `tracing` is unaffected — turn logging off with `RUST_LOG`, not here.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    /// Is recording on? On by default.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Register work that is starting **now**.
    ///
    /// `epoch` is `nguo_engine_bridge::Epoch::0`, or `0` if this work has no
    /// generation counter.
    pub fn start(&self, feature: &'static Feature, epoch: u64) -> Process<'_> {
        self.insert(feature, epoch, ProcessState::Running)
    }

    /// Register work that has been submitted but not started. Promote it with
    /// [`Process::run`].
    pub fn queue(&self, feature: &'static Feature, epoch: u64) -> Process<'_> {
        self.insert(feature, epoch, ProcessState::Queued)
    }

    /// Copy the whole live state out. Takes the lock once, releases it before
    /// returning, and never calls back into user code while holding it — so a
    /// caller can hold the result across a frame with nothing borrowed.
    pub fn snapshot(&self) -> Snapshot {
        let inner = self.lock();
        let now = Instant::now();
        Snapshot {
            active: inner
                .active
                .iter()
                .map(|r| ActiveProcess {
                    id: r.id,
                    feature: r.feature,
                    state: r.state,
                    epoch: r.epoch,
                    progress: r.progress,
                    elapsed: now.saturating_duration_since(r.started),
                })
                .collect(),
            recent: inner.recent.iter().rev().cloned().collect(),
            failures: inner.failures.iter().rev().cloned().collect(),
        }
    }

    /// Forget every completed process and failure, and reset the epoch
    /// watermarks. Active processes are left alone. For tests and for a
    /// "clear" button in the overlay.
    pub fn clear_history(&self) {
        let mut inner = self.lock();
        inner.recent.clear();
        inner.failures.clear();
        inner.max_epoch.clear();
    }

    /// A poisoned registry lock must never take the app down with it —
    /// observability is the last thing that should escalate a failure
    /// (ui/docs/OBSERVABILITY.md §6). The data behind it is plain records with no
    /// invariant a panicking writer could have half-broken.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn insert(&self, feature: &'static Feature, epoch: u64, state: ProcessState) -> Process<'_> {
        if !self.is_enabled() {
            return Process::inert(self, feature, state);
        }

        let id = {
            let mut inner = self.lock();
            let max = inner.max_epoch.entry(feature.id).or_insert(0);
            if epoch != 0 && epoch < *max {
                // Same rule as the bridge: a stale generation produces
                // nothing at all, not even a failure.
                let max = *max;
                drop(inner);
                tracing::debug!(
                    target: PROCESS_TARGET,
                    feature = feature.id,
                    epoch,
                    newest = max,
                    "dropped a stale process registration"
                );
                return Process::inert(self, feature, state);
            }
            if epoch > *max {
                *max = epoch;
                inner.supersede(feature, epoch);
            }

            inner.next_id += 1;
            let id = ProcessId(inner.next_id);
            inner.active.push(ActiveRecord {
                id,
                feature,
                state,
                epoch,
                progress: None,
                started: Instant::now(),
            });
            id
        };

        let span = tracing::span!(
            target: PROCESS_TARGET,
            tracing::Level::DEBUG,
            "process",
            feature = feature.id,
            subsystem = feature.subsystem.as_str(),
            id = id.0,
            epoch,
        );
        tracing::debug!(
            target: PROCESS_TARGET,
            feature = feature.id,
            id = id.0,
            epoch,
            state = state.as_str(),
            "process registered"
        );

        Process {
            registry: self,
            id,
            feature,
            span,
            state: Cell::new(state),
        }
    }

    /// Apply a validated transition to the stored record. The handle has
    /// already checked legality; if the record is gone (superseded by a newer
    /// epoch) this is a no-op, which is correct — the registry's view wins.
    fn commit(&self, id: ProcessId, to: ProcessState, chain: Option<Vec<String>>) {
        if id.0 == 0 {
            return;
        }
        let mut inner = self.lock();
        let Some(pos) = inner.active.iter().position(|r| r.id == id) else {
            return;
        };
        if !to.is_terminal() {
            inner.active[pos].state = to;
            return;
        }
        inner.finish(pos, to, chain);
    }
}

impl Inner {
    /// A newer epoch retires every still-active process of the same feature
    /// that belongs to an older one — `nguo-engine-bridge`'s coalescing, made
    /// visible.
    fn supersede(&mut self, feature: &'static Feature, epoch: u64) {
        loop {
            let Some(pos) = self
                .active
                .iter()
                .position(|r| r.feature.id == feature.id && r.epoch != 0 && r.epoch < epoch)
            else {
                return;
            };
            tracing::debug!(
                target: PROCESS_TARGET,
                feature = feature.id,
                id = self.active[pos].id.0,
                epoch = self.active[pos].epoch,
                superseded_by = epoch,
                "process superseded by a newer epoch"
            );
            self.finish(pos, ProcessState::Cancelled, None);
        }
    }

    fn finish(&mut self, pos: usize, outcome: ProcessState, chain: Option<Vec<String>>) {
        let rec = self.active.remove(pos);
        let duration = Instant::now().saturating_duration_since(rec.started);

        if let Some(chain) = chain {
            tracing::error!(
                target: PROCESS_TARGET,
                feature = rec.feature.id,
                id = rec.id.0,
                epoch = rec.epoch,
                elapsed_ms = duration.as_secs_f64() * 1e3,
                error = %chain.join(": "),
                "process failed"
            );
            push_capped(
                &mut self.failures,
                FailureRecord {
                    id: rec.id,
                    feature: rec.feature,
                    epoch: rec.epoch,
                    chain,
                    duration,
                },
                FAILURE_CAPACITY,
            );
        } else {
            tracing::debug!(
                target: PROCESS_TARGET,
                feature = rec.feature.id,
                id = rec.id.0,
                epoch = rec.epoch,
                elapsed_ms = duration.as_secs_f64() * 1e3,
                state = outcome.as_str(),
                "process finished"
            );
        }

        push_capped(
            &mut self.recent,
            CompletedProcess {
                id: rec.id,
                feature: rec.feature,
                outcome,
                epoch: rec.epoch,
                duration,
            },
            RECENT_CAPACITY,
        );
    }
}

/// Bounded ring, oldest first.
// ponytail: `remove(0)` is O(n) on a 64-element `Vec` of 48-byte records. A
// `VecDeque` would be O(1) but is not `const`-constructible, which `GLOBAL`
// needs. Measured cost is in the noise; revisit if the capacity grows.
fn push_capped<T>(buf: &mut Vec<T>, item: T, cap: usize) {
    if buf.len() >= cap {
        buf.remove(0);
    }
    buf.push(item);
}

/// A handle to one registered process.
///
/// The handle owns the lifecycle: it validates transitions against its own
/// state with no lock at all, and only then touches the registry. Dropping it
/// without a terminal call records `Cancelled`, so a panicking solver leaves a
/// cancelled process behind instead of one that runs forever.
///
/// `Send`, not `Sync` — move it into the worker that does the work; do not
/// share it. That is also the only shape in which its state can be tracked
/// without a second lock.
pub struct Process<'r> {
    registry: &'r Registry,
    id: ProcessId,
    feature: &'static Feature,
    span: tracing::Span,
    state: Cell<ProcessState>,
}

impl<'r> Process<'r> {
    /// A handle that records nothing. It still tracks and validates its own
    /// lifecycle, so a caller sees exactly the same `Ok`/`Err` sequence
    /// whether or not the registry is on — an observability switch that
    /// changes control flow is worse than no switch.
    fn inert(registry: &'r Registry, feature: &'static Feature, state: ProcessState) -> Self {
        Self {
            registry,
            id: ProcessId(0),
            feature,
            span: tracing::Span::none(),
            state: Cell::new(state),
        }
    }

    /// `ProcessId(0)` if this handle is inert (registry disabled, or a stale
    /// epoch).
    pub fn id(&self) -> ProcessId {
        self.id
    }

    /// The inventory entry this process is an instance of.
    pub fn feature(&self) -> &'static Feature {
        self.feature
    }

    /// Current lifecycle state, as this handle sees it.
    pub fn state(&self) -> ProcessState {
        self.state.get()
    }

    /// The `tracing` span for this process. `enter()` it around the work so
    /// existing `tracing` tooling attributes events to it:
    ///
    /// ```
    /// # let r = nguo_registry::Registry::new();
    /// let job = r.start(nguo_registry::feature_const("io.open"), 0);
    /// let _guard = job.span().clone().entered();
    /// tracing::info!("reading manifest");
    /// ```
    pub fn span(&self) -> &tracing::Span {
        &self.span
    }

    /// `Queued -> Running`.
    pub fn run(&self) -> Result<(), RegistryError> {
        self.transition(ProcessState::Running, None)
    }

    /// `Running -> Done`.
    pub fn done(&self) -> Result<(), RegistryError> {
        self.transition(ProcessState::Done, None)
    }

    /// `Queued | Running -> Cancelled`.
    pub fn cancelled(&self) -> Result<(), RegistryError> {
        self.transition(ProcessState::Cancelled, None)
    }

    /// `Queued | Running -> Failed`, recording `err` and every `source()`
    /// under it, to the root.
    ///
    /// This is the whole error-handling contract of the crate: a failure is
    /// *recorded*, never escalated. The caller decides whether to show
    /// [`FailureRecord::title`]/[`detail`] as a toast; the log already has the
    /// full chain.
    ///
    /// [`detail`]: FailureRecord::detail
    pub fn failed(&self, err: &(dyn std::error::Error + 'static)) -> Result<(), RegistryError> {
        self.transition(ProcessState::Failed, Some(process::error_chain(err)))
    }

    /// Report progress, `0.0..=1.0` (clamped). Ignored unless running.
    pub fn progress(&self, fraction: f32) {
        if self.id.0 == 0 || self.state.get() != ProcessState::Running {
            return;
        }
        let fraction = fraction.clamp(0.0, 1.0);
        let mut inner = self.registry.lock();
        if let Some(rec) = inner.active.iter_mut().find(|r| r.id == self.id) {
            rec.progress = Some(fraction);
        }
    }

    fn transition(
        &self,
        to: ProcessState,
        chain: Option<Vec<String>>,
    ) -> Result<(), RegistryError> {
        let from = self.state.get();
        if !from.can_transition_to(to) {
            return Err(RegistryError::IllegalTransition { from, to });
        }
        self.state.set(to);
        self.registry.commit(self.id, to, chain);
        Ok(())
    }
}

impl Drop for Process<'_> {
    fn drop(&mut self) {
        if !self.state.get().is_terminal() {
            let _ = self.transition(ProcessState::Cancelled, None);
        }
    }
}

impl std::fmt::Debug for Process<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Process")
            .field("id", &self.id)
            .field("feature", &self.feature.id)
            .field("state", &self.state.get())
            .finish()
    }
}

/// The application-wide registry. Const-initialised, so there is no init order
/// and no `OnceLock`.
pub static GLOBAL: Registry = Registry::new();

/// [`Registry::start`] on [`GLOBAL`].
pub fn start(feature: &'static Feature, epoch: u64) -> Process<'static> {
    GLOBAL.start(feature, epoch)
}

/// [`Registry::queue`] on [`GLOBAL`].
pub fn queue(feature: &'static Feature, epoch: u64) -> Process<'static> {
    GLOBAL.queue(feature, epoch)
}

/// [`Registry::snapshot`] on [`GLOBAL`].
pub fn snapshot() -> Snapshot {
    GLOBAL.snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAPE: &Feature = feature_const("drape.preview");
    const BAKE: &Feature = feature_const("drape.bake");
    const OPEN: &Feature = feature_const("io.open");

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct Leaf(&'static str);

    #[derive(Debug, thiserror::Error)]
    #[error("could not drape `bodice`")]
    struct Wrapped(#[source] Leaf);

    #[test]
    fn a_running_process_shows_up_then_completes() {
        let r = Registry::new();
        let job = r.start(DRAPE, 1);
        job.progress(0.25);

        let snap = r.snapshot();
        assert_eq!(snap.active.len(), 1);
        assert_eq!(snap.active[0].feature.id, "drape.preview");
        assert_eq!(snap.active[0].state, ProcessState::Running);
        assert_eq!(snap.active[0].epoch, 1);
        assert_eq!(snap.active[0].progress, Some(0.25));
        assert!(snap.recent.is_empty());

        job.done().unwrap();
        let snap = r.snapshot();
        assert!(snap.active.is_empty());
        assert_eq!(snap.recent.len(), 1);
        assert_eq!(snap.recent[0].outcome, ProcessState::Done);
    }

    #[test]
    fn queued_promotes_to_running() {
        let r = Registry::new();
        let job = r.queue(OPEN, 0);
        assert_eq!(r.snapshot().active[0].state, ProcessState::Queued);
        job.run().unwrap();
        assert_eq!(r.snapshot().active[0].state, ProcessState::Running);
        job.done().unwrap();
    }

    #[test]
    fn illegal_transitions_are_rejected() {
        let r = Registry::new();
        let job = r.start(DRAPE, 0);
        job.done().unwrap();
        assert_eq!(
            job.run(),
            Err(RegistryError::IllegalTransition {
                from: ProcessState::Done,
                to: ProcessState::Running,
            }),
            "done -> running must be rejected"
        );
        assert!(job.done().is_err(), "done -> done must be rejected");
        assert!(job.cancelled().is_err());
        // The rejected calls left exactly one completed record behind.
        assert_eq!(r.snapshot().recent.len(), 1);

        let queued = r.queue(OPEN, 0);
        assert_eq!(
            queued.done(),
            Err(RegistryError::IllegalTransition {
                from: ProcessState::Queued,
                to: ProcessState::Done,
            }),
            "you cannot finish work that never ran"
        );
    }

    #[test]
    fn dropping_a_handle_cancels_rather_than_leaking_a_runner() {
        let r = Registry::new();
        {
            let _job = r.start(DRAPE, 0);
            assert_eq!(r.snapshot().active.len(), 1);
        }
        let snap = r.snapshot();
        assert!(snap.active.is_empty(), "a dropped process kept running");
        assert_eq!(snap.recent[0].outcome, ProcessState::Cancelled);
    }

    #[test]
    fn a_stale_epoch_is_dropped_entirely() {
        let r = Registry::new();
        let newest = r.start(DRAPE, 5);
        let stale = r.start(DRAPE, 4);
        assert_eq!(stale.id(), ProcessId(0), "a stale epoch must not register");

        let snap = r.snapshot();
        assert_eq!(snap.active.len(), 1);
        assert_eq!(snap.active[0].epoch, 5);
        assert!(snap.recent.is_empty(), "a stale epoch must not even fail");

        // The stale handle behaves exactly like a live one — it just records
        // nothing anywhere.
        stale.done().unwrap();
        assert!(stale.failed(&Leaf("nope")).is_err(), "done -> failed");
        assert!(r.snapshot().failures.is_empty());
        assert!(r.snapshot().recent.is_empty());
        newest.done().unwrap();
    }

    #[test]
    fn a_newer_epoch_supersedes_the_older_one_of_the_same_feature() {
        let r = Registry::new();
        let old = r.start(DRAPE, 1);
        let other = r.start(BAKE, 1);
        let new = r.start(DRAPE, 2);

        let snap = r.snapshot();
        let active: Vec<_> = snap
            .active
            .iter()
            .map(|a| (a.feature.id, a.epoch))
            .collect();
        assert_eq!(
            active,
            vec![("drape.bake", 1), ("drape.preview", 2)],
            "only the older process of the *same* feature is superseded"
        );
        assert_eq!(snap.recent.len(), 1);
        assert_eq!(snap.recent[0].outcome, ProcessState::Cancelled);
        assert_eq!(snap.recent[0].id, old.id());

        // The superseded handle can still be driven; it just changes nothing.
        old.done().unwrap();
        assert_eq!(r.snapshot().recent.len(), 1);
        new.done().unwrap();
        other.done().unwrap();
    }

    #[test]
    fn epoch_zero_is_never_stale() {
        let r = Registry::new();
        let a = r.start(OPEN, 0);
        let b = r.start(OPEN, 0);
        assert_ne!(a.id(), ProcessId(0));
        assert_ne!(b.id(), ProcessId(0));
        assert_eq!(r.snapshot().active.len(), 2);
    }

    #[test]
    fn a_failure_keeps_the_whole_error_chain() {
        let r = Registry::new();
        let job = r.start(DRAPE, 0);
        job.failed(&Wrapped(Leaf("solver diverged at frame 12")))
            .unwrap();

        let snap = r.snapshot();
        assert_eq!(snap.failures.len(), 1);
        let f = &snap.failures[0];
        assert_eq!(
            f.chain,
            vec!["could not drape `bodice`", "solver diverged at frame 12"]
        );
        assert_eq!(f.title(), "Drape preview");
        assert_eq!(f.detail(), "could not drape `bodice`");
        assert_eq!(f.root_cause(), "solver diverged at frame 12");
        assert_eq!(snap.recent[0].outcome, ProcessState::Failed);
    }

    #[test]
    fn history_is_bounded_and_newest_first() {
        let r = Registry::new();
        for _ in 0..(RECENT_CAPACITY + 10) {
            r.start(OPEN, 0).done().unwrap();
        }
        for _ in 0..(FAILURE_CAPACITY + 10) {
            r.start(OPEN, 0).failed(&Leaf("boom")).unwrap();
        }
        let snap = r.snapshot();
        assert_eq!(snap.recent.len(), RECENT_CAPACITY);
        assert_eq!(snap.failures.len(), FAILURE_CAPACITY);
        // Newest first, in both rings.
        assert!(snap.recent.windows(2).all(|w| w[0].id > w[1].id));
        assert!(snap.failures.windows(2).all(|w| w[0].id > w[1].id));
    }

    #[test]
    fn snapshot_is_ordered_by_id_not_by_insertion_luck() {
        let r = Registry::new();
        let jobs: Vec<_> = (0..8).map(|_| r.start(OPEN, 0)).collect();
        let snap = r.snapshot();
        assert!(snap.active.windows(2).all(|w| w[0].id < w[1].id));
        assert_eq!(
            snap.active.iter().map(|a| a.id).collect::<Vec<_>>(),
            jobs.iter().map(|j| j.id()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn disabled_registry_records_nothing_and_still_behaves() {
        let r = Registry::new();
        r.set_enabled(false);
        let job = r.start(DRAPE, 3);
        assert_eq!(job.id(), ProcessId(0));
        assert_eq!(job.state(), ProcessState::Running, "same state as enabled");
        job.progress(0.5);
        assert!(job.run().is_err(), "same rejection as an enabled registry");
        job.done().unwrap();
        let snap = r.snapshot();
        assert!(snap.active.is_empty() && snap.recent.is_empty());

        // Turning it back on resumes recording; nothing is retroactive.
        r.set_enabled(true);
        r.start(DRAPE, 4).done().unwrap();
        assert_eq!(r.snapshot().recent.len(), 1);
    }

    #[test]
    fn clear_history_leaves_active_work_alone() {
        let r = Registry::new();
        let live = r.start(DRAPE, 1);
        r.start(OPEN, 0).failed(&Leaf("boom")).unwrap();
        r.clear_history();
        let snap = r.snapshot();
        assert_eq!(snap.active.len(), 1);
        assert!(snap.recent.is_empty() && snap.failures.is_empty());
        live.done().unwrap();
    }

    #[test]
    fn every_feature_marked_as_a_process_can_actually_be_started() {
        let r = Registry::new();
        for f in FEATURES.iter().filter(|f| f.process) {
            let job = r.start(f, 0);
            assert_eq!(job.feature().id, f.id);
            job.done().unwrap();
        }
        assert_eq!(
            r.snapshot().recent.len(),
            FEATURES.iter().filter(|f| f.process).count()
        );
    }
}
