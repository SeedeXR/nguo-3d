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

/// Call from a `GlobalAlloc::alloc` that succeeded.
#[inline]
pub fn on_alloc(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK_LIVE.fetch_max(live, Relaxed);
    TOTAL.fetch_add(size, Relaxed);
    COUNT.fetch_add(1, Relaxed);
}

/// Call from a `GlobalAlloc::dealloc`.
#[inline]
pub fn on_dealloc(size: usize) {
    LIVE.fetch_sub(size, Relaxed);
}

/// Call from a `GlobalAlloc::realloc` that succeeded.
#[inline]
pub fn on_realloc(old: usize, new: usize) {
    if new >= old {
        let d = new - old;
        let live = LIVE.fetch_add(d, Relaxed) + d;
        PEAK_LIVE.fetch_max(live, Relaxed);
        TOTAL.fetch_add(d, Relaxed);
    } else {
        LIVE.fetch_sub(old - new, Relaxed);
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
    eprintln!(
        "  mem[{label}]: live {:.1} MB  peak {:.1} MB",
        mb(LIVE.load(Relaxed)),
        mb(PEAK_LIVE.load(Relaxed))
    );
}

/// Reset the PEAK watermark to the current live level, so the next phase's peak is measured
/// against where it started rather than against the whole run.
pub fn reset_peak() {
    PEAK_LIVE.store(LIVE.load(Relaxed), Relaxed);
}
