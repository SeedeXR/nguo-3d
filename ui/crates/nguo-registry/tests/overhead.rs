// SPDX-License-Identifier: GPL-2.0-or-later
//! Overhead measurement. The maintainer's standing rule is *measure before
//! deciding*, so "negligible" is not a claim this crate is allowed to make
//! without these numbers.
//!
//! Run it:
//!
//! ```text
//! cargo test -p nguo-registry --release --test overhead -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d because it is a measurement, not an assertion: it prints
//! numbers and only fails if the overhead has become absurd (the guards below
//! are three orders of magnitude above what was measured, so they catch a
//! structural regression — an accidental syscall, an unbounded ring — not
//! machine-to-machine noise).
//!
//! Numbers from a run are recorded in `ui/docs/OBSERVABILITY.md`. Re-measure
//! there, do not copy from here.

use std::hint::black_box;
use std::time::Instant;

use nguo_registry::{feature_const, Feature, Registry};

const DRAPE: &Feature = feature_const("drape.preview");

#[derive(Debug, thiserror::Error)]
#[error("measurement failure")]
struct Boom;

fn bench(label: &str, iters: u32, mut body: impl FnMut()) -> f64 {
    // Warm up: first-touch allocation of the rings, and the CPU.
    for _ in 0..(iters / 10).max(1) {
        body();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        body();
    }
    let ns = t0.elapsed().as_secs_f64() * 1e9 / f64::from(iters);
    println!("{label:<44} {ns:>9.1} ns/op");
    ns
}

#[test]
#[ignore = "measurement, not an assertion — run with --ignored --nocapture"]
fn overhead() {
    const ITERS: u32 = 200_000;
    println!();
    println!("nguo-registry overhead ({ITERS} iterations each)");
    println!("{:-<56}", "");

    let r = Registry::new();
    let enabled = bench("start + done, recording", ITERS, || {
        let job = r.start(DRAPE, 0);
        black_box(&job).done().unwrap();
    });

    let enabled_progress = bench("start + progress + done, recording", ITERS, || {
        let job = r.start(DRAPE, 0);
        job.progress(0.5);
        black_box(&job).done().unwrap();
    });

    let enabled_failed = bench("start + failed (2-deep chain), recording", ITERS, || {
        let job = r.start(DRAPE, 0);
        black_box(&job).failed(&Boom).unwrap();
    });

    r.set_enabled(false);
    let disabled = bench("start + done, registry disabled", ITERS, || {
        let job = r.start(DRAPE, 0);
        black_box(&job).done().unwrap();
    });
    r.set_enabled(true);

    // Snapshot cost at a realistic live-process count, since this is what the
    // egui overlay pays every frame.
    r.clear_history();
    let held: Vec<_> = (0..16).map(|_| r.start(DRAPE, 0)).collect();
    for _ in 0..nguo_registry::RECENT_CAPACITY {
        r.start(DRAPE, 0).done().unwrap();
    }
    let snap = bench("snapshot (16 active, full history)", ITERS / 10, || {
        black_box(r.snapshot());
    });
    drop(held);

    println!("{:-<56}", "");
    println!(
        "recording overhead over the disabled path: {:.1} ns/op",
        enabled - disabled
    );
    println!(
        "snapshot as a share of a 8.3 ms frame budget: {:.4} %",
        snap / 8.3e6 * 100.0
    );
    println!();

    // Structural guards only — three orders of magnitude of headroom.
    assert!(
        enabled < 100_000.0,
        "start+done got absurdly slow: {enabled} ns"
    );
    assert!(
        enabled_progress < 100_000.0,
        "progress got absurdly slow: {enabled_progress} ns"
    );
    assert!(
        enabled_failed < 100_000.0,
        "failure recording got absurdly slow: {enabled_failed} ns"
    );
    assert!(
        disabled < 10_000.0,
        "the disabled path is supposed to be free: {disabled} ns"
    );
    assert!(
        snap < 1_000_000.0,
        "snapshot no longer fits in a frame: {snap} ns"
    );
}
