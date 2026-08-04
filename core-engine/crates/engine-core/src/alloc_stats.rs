// SPDX-License-Identifier: GPL-2.0-or-later
//! Allocation counters, so peak LIVE bytes can be attributed to a PHASE of the step.
//!
//! The counters live here rather than in the binary because attributing memory needs probes
//! inside `implicit::step`, and `engine-core` cannot see the binary's global allocator. The
//! binary installs a `GlobalAlloc` that feeds [`on_alloc`] / [`on_dealloc`]; when no such
//! allocator is installed every counter simply stays at zero and [`checkpoint`] prints
//! nothing, so linking this costs two relaxed atomics per allocation and nothing else.
//!
//! Why this exists: peak RSS cannot tell "we hold N bytes live" from "we allocated and freed
//! N bytes and the system allocator kept the pages". Three earlier sessions ruled things out
//! without finding the cause of B-1 because RSS was the only instrument. Measured with this:
//! `fishingknot` goes from 22.3 MB live after loading to **1057.2 MB during its first step**,
//! with 0.0 MB still live at exit — no leak, a large transient.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

pub static LIVE: AtomicUsize = AtomicUsize::new(0);
pub static PEAK_LIVE: AtomicUsize = AtomicUsize::new(0);
pub static TOTAL: AtomicUsize = AtomicUsize::new(0);
pub static COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// `(live, peak)` for THIS thread. The process-wide `PEAK_LIVE` above cannot be
    /// asserted on from a test, because `cargo test` runs tests concurrently and any other
    /// test's allocation lands in it. A solver step is single-threaded, so a per-thread
    /// watermark is exact for it and immune to that noise — which is what makes
    /// `one_step_footprint_is_bounded` a guard rather than a flake.
    ///
    /// `const`-initialised on a `Copy` type deliberately: that compiles to a plain
    /// `#[thread_local]` static with no lazy init and no destructor registration, so it
    /// cannot allocate, which would be re-entrant from inside a `GlobalAlloc`.
    static TL: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[inline]
fn tl_add(size: usize) {
    TL.with(|t| {
        let (live, peak) = t.get();
        let live = live + size;
        t.set((live, peak.max(live)));
    });
}

#[inline]
fn tl_sub(size: usize) {
    TL.with(|t| {
        let (live, peak) = t.get();
        // Saturating: a buffer allocated on one thread and freed on another would otherwise
        // underflow. Only the watermark is asserted on, and that stays valid.
        t.set((live.saturating_sub(size), peak));
    });
}

/// Peak live bytes allocated by the CALLING thread since [`reset_thread_peak`].
pub fn thread_peak_bytes() -> usize {
    TL.with(|t| t.get().1)
}

/// Drop this thread's watermark to its current live level.
pub fn reset_thread_peak() {
    TL.with(|t| {
        let (live, _) = t.get();
        t.set((live, live));
    });
}

/// Call from a `GlobalAlloc::alloc` that succeeded.
#[inline]
pub fn on_alloc(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK_LIVE.fetch_max(live, Relaxed);
    TOTAL.fetch_add(size, Relaxed);
    COUNT.fetch_add(1, Relaxed);
    tl_add(size);
}

/// Call from a `GlobalAlloc::dealloc`.
#[inline]
pub fn on_dealloc(size: usize) {
    LIVE.fetch_sub(size, Relaxed);
    tl_sub(size);
}

/// Call from a `GlobalAlloc::realloc` that succeeded.
#[inline]
pub fn on_realloc(old: usize, new: usize) {
    if new >= old {
        let d = new - old;
        let live = LIVE.fetch_add(d, Relaxed) + d;
        PEAK_LIVE.fetch_max(live, Relaxed);
        TOTAL.fetch_add(d, Relaxed);
        tl_add(d);
    } else {
        LIVE.fetch_sub(old - new, Relaxed);
        tl_sub(old - new);
    }
    COUNT.fetch_add(1, Relaxed);
}

pub fn live_bytes() -> usize {
    LIVE.load(Relaxed)
}

pub fn peak_bytes() -> usize {
    PEAK_LIVE.load(Relaxed)
}

/// `MGUSO_MEM_TRACE=1` prints live/peak bytes at a named point. Read once — these sit on the
/// step path.
pub fn tracing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MGUSO_MEM_TRACE").is_ok())
}

/// Print live and peak bytes tagged with `label`, if tracing is on and an allocator is
/// feeding the counters.
pub fn checkpoint(label: &str) {
    if !tracing_enabled() || LIVE.load(Relaxed) == 0 {
        return;
    }
    let mb = |b: usize| b as f64 / 1048576.0;
    // COUNT is here because peak LIVE bytes does not predict RSS: measured, 5 041 cloth
    // vertices over the male proxy, 12 steps, peak live 149.5 MB against `/usr/bin/time -l`
    // maximum RSS 1 309 474 816 B — 8.4x. The gap is macOS libmalloc retaining spans it has
    // been handed back.
    //
    // COUNT is a LEAD, not the explanatory variable, and an earlier version of this comment
    // claimed otherwise. A/B measured over the streaming-candidates change: count moved
    // 975 351 -> 968 622 (0.7 %) while max RSS fell 1.58x — what tracked RSS was churned
    // BYTES (2.17x). Count did explain the `accd_toi` per-pair `to_vec`, which is why it is
    // printed; bytes churned is the quantity to reach for when count does not move.
    eprintln!(
        "  mem[{label}]: live {:.1} MB  peak {:.1} MB  allocs {}",
        mb(LIVE.load(Relaxed)),
        mb(PEAK_LIVE.load(Relaxed)),
        COUNT.load(Relaxed)
    );
}

/// Reset the PEAK watermark to the current live level, so the next phase's peak is measured
/// against where it started rather than against the whole run.
pub fn reset_peak() {
    PEAK_LIVE.store(LIVE.load(Relaxed), Relaxed);
}
