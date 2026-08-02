// SPDX-License-Identifier: GPL-2.0-or-later
//! The registry is written from solver threads and read from the render
//! thread every frame. These tests hammer that shape.
//!
//! No sleeps and no timeouts-as-assertions: every test here either completes
//! or hangs, and a hang *is* the failure signal (the harness kills it). That
//! is deliberate — a "no deadlock" test that passes by timing out proves
//! nothing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;
use std::thread;

use nguo_registry::{feature_const, Feature, ProcessState, Registry};

const DRAPE: &Feature = feature_const("drape.preview");
const OPEN: &Feature = feature_const("io.open");
const TRI: &Feature = feature_const("triangulate.piece");

#[derive(Debug, thiserror::Error)]
#[error("worker {0} gave up")]
struct Boom(usize);

/// Nothing leaks out of the active list under contention, the bounded rings
/// never record a process twice, and the snapshot's ordering is a pure function
/// of the state, not of who won which race.
#[test]
fn concurrent_registration_loses_nothing_and_reports_deterministically() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 500;
    const TOTAL: u64 = (THREADS * PER_THREAD) as u64;

    let registry = Registry::new();
    // One feature for everybody, so the threads also contend on the same
    // epoch-watermark entry and the same active vector.
    let gate = Barrier::new(THREADS);

    thread::scope(|s| {
        for t in 0..THREADS {
            let registry = &registry;
            let gate = &gate;
            s.spawn(move || {
                gate.wait();
                for i in 0..PER_THREAD {
                    let job = registry.start(OPEN, 0);
                    job.progress(i as f32 / PER_THREAD as f32);
                    if i % 3 == 0 {
                        job.failed(&Boom(t)).unwrap();
                    } else {
                        job.done().unwrap();
                    }
                }
            });
        }
    });

    let snap = registry.snapshot();
    assert!(snap.active.is_empty(), "a process outlived its thread");
    assert_eq!(snap.recent.len(), nguo_registry::RECENT_CAPACITY);
    assert_eq!(snap.failures.len(), nguo_registry::FAILURE_CAPACITY);
    // Bounded rings never repeat a process.
    let mut recent: Vec<u64> = snap.recent.iter().map(|c| c.id.0).collect();
    let unique = {
        let mut v = recent.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert_eq!(recent.len(), unique, "a completion was recorded twice");
    recent.sort_unstable();
    assert!(recent.iter().all(|id| (1..=TOTAL).contains(id)));
    assert!(snap
        .failures
        .iter()
        .all(|f| f.chain.len() == 1 && f.chain[0].starts_with("worker ")));

    // Deterministic: with no writers running, the same state snapshots the
    // same way, every time.
    let again = registry.snapshot();
    assert_eq!(
        snap.recent.iter().map(|c| c.id).collect::<Vec<_>>(),
        again.recent.iter().map(|c| c.id).collect::<Vec<_>>()
    );
}

/// The explicit maintainer requirement: snapshotting from one thread while
/// several others write must never deadlock, and must never observe a torn or
/// contradictory view.
#[test]
fn snapshot_never_deadlocks_against_hammering_writers() {
    const WRITERS: usize = 6;
    const SNAPSHOTS: usize = 5_000;

    let registry = Registry::new();
    let stop = AtomicBool::new(false);
    let gate = Barrier::new(WRITERS + 1);

    thread::scope(|s| {
        for t in 0..WRITERS {
            let registry = &registry;
            let stop = &stop;
            let gate = &gate;
            s.spawn(move || {
                gate.wait();
                let mut epoch = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    epoch += 1;
                    // Mix every exit path — done, failed, cancelled and the
                    // Drop path — so the reader sees every state there is.
                    let a = registry.start(DRAPE, epoch);
                    let b = registry.queue(TRI, 0);
                    let c = registry.start(OPEN, 0);
                    a.progress(0.5);
                    b.run().unwrap();
                    match t % 3 {
                        0 => a.done().unwrap(),
                        1 => a.failed(&Boom(t)).unwrap(),
                        _ => a.cancelled().unwrap(),
                    }
                    drop(b); // cancelled by Drop
                    c.done().unwrap();
                }
            });
        }

        gate.wait();
        for _ in 0..SNAPSHOTS {
            let snap = registry.snapshot();
            assert!(
                snap.active.windows(2).all(|w| w[0].id < w[1].id),
                "the active list must always be ascending by id"
            );
            assert!(
                snap.active.iter().all(|a| !a.state.is_terminal()),
                "a terminal record surfaced as active"
            );
            assert!(
                snap.active
                    .iter()
                    .all(|a| a.progress.is_none_or(|p| (0.0..=1.0).contains(&p))),
                "progress outside 0..=1"
            );
            assert!(
                snap.recent.iter().all(|c| c.outcome.is_terminal()),
                "a non-terminal record surfaced as completed"
            );
            assert!(
                snap.failures.iter().all(|f| !f.chain.is_empty()),
                "a failure with no error chain"
            );
        }
        stop.store(true, Ordering::Relaxed);
    });

    // Still works afterwards: no lock left held, nothing wedged.
    let job = registry.start(DRAPE, u64::MAX);
    job.done().unwrap();
    assert_eq!(
        registry.snapshot().recent[0].outcome,
        ProcessState::Done,
        "the registry stopped working after the hammering"
    );
}

/// A worker that panics must leave a *cancelled* process behind, not one that
/// reads as running forever. This is the 3am case the `Drop` impl exists for.
#[test]
fn a_panicking_worker_leaves_a_cancelled_process_not_a_zombie() {
    let registry = Registry::new();

    let panicked = thread::scope(|s| {
        let registry = &registry;
        s.spawn(move || {
            let _job = registry.start(DRAPE, 1);
            assert_eq!(registry.snapshot().active.len(), 1);
            panic!("solver thread died");
        })
        .join()
    });
    assert!(panicked.is_err(), "the test worker was supposed to panic");

    let snap = registry.snapshot();
    assert!(snap.active.is_empty(), "a zombie process survived a panic");
    assert_eq!(snap.recent.len(), 1);
    assert_eq!(snap.recent[0].outcome, ProcessState::Cancelled);

    // And the registry is still usable.
    registry.start(OPEN, 0).done().unwrap();
    assert_eq!(registry.snapshot().recent[0].outcome, ProcessState::Done);
}

/// Newest-epoch-wins holds under contention, exactly as in
/// `nguo-engine-bridge`: after a burst of overlapping submissions the only
/// surviving active process is the one with the highest epoch, and every
/// superseded one reads as cancelled — never failed.
#[test]
fn concurrent_epochs_leave_only_the_newest_alive() {
    const THREADS: u64 = 8;
    const EACH: u64 = 200;

    let registry = Registry::new();
    let gate = Barrier::new(THREADS as usize);

    // The handles are kept alive for the whole scope, so nothing is retired by
    // `Drop` — only supersession can remove a process from the active list.
    thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let registry = &registry;
                let gate = &gate;
                s.spawn(move || {
                    gate.wait();
                    let mut mine = Vec::with_capacity(EACH as usize);
                    for i in 0..EACH {
                        // Overlapping epoch ranges across threads.
                        mine.push(registry.start(DRAPE, t * EACH + i + 1));
                    }
                    mine
                })
            })
            .collect();
        let held: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let snap = registry.snapshot();
        assert_eq!(
            snap.active.len(),
            1,
            "every superseded epoch should have been retired, saw {:?}",
            snap.active.iter().map(|a| a.epoch).collect::<Vec<_>>()
        );
        assert_eq!(snap.active[0].epoch, THREADS * EACH);
        assert!(
            snap.recent
                .iter()
                .all(|c| c.outcome == ProcessState::Cancelled),
            "superseded work must read as cancelled, not failed"
        );
        assert!(snap.failures.is_empty());
        drop(held);
    });
}
