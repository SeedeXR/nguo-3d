// SPDX-License-Identifier: GPL-2.0-or-later
//! Epoch/generation bookkeeping, coalescing, and the frame channel.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tokio::sync::{mpsc, watch};

use crate::transport::{Flow, FrameSink, Submission, Summary, Transport};

/// Generation id. Monotonic; a job is stale the moment a larger one exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct Epoch(pub u64);

/// Frame channel depth. Three slots = triple buffering: the solver can be
/// filling one while the consumer holds one and one waits, and because each
/// frame is an owned `Vec` moved through the channel, a consumer can never see
/// a partially written frame (guide §3.1).
pub const FRAME_CHANNEL_CAPACITY: usize = 3;

/// Everything the bridge emits. Only ever for the newest epoch — stale results
/// are dropped inside the bridge and never reach this channel.
#[derive(Debug)]
pub enum Update {
    Frame {
        epoch: Epoch,
        index: u32,
        positions: Vec<[f32; 3]>,
    },
    Done {
        epoch: Epoch,
        summary: Summary,
    },
    Failed {
        epoch: Epoch,
        error: String,
    },
}

impl Update {
    pub fn epoch(&self) -> Epoch {
        match self {
            Update::Frame { epoch, .. }
            | Update::Done { epoch, .. }
            | Update::Failed { epoch, .. } => *epoch,
        }
    }
}

#[derive(Clone)]
struct PendingJob {
    epoch: Epoch,
    sub: Submission,
}

/// The app-facing handle. Cheap to clone-free share behind an `Arc`.
///
/// Dropping the bridge shuts the worker down and closes the update channel.
pub struct EngineBridge {
    pending: watch::Sender<Option<PendingJob>>,
    /// The newest epoch handed out. A solve whose epoch != this is stale.
    latest: Arc<AtomicU64>,
    next: AtomicU64,
}

impl EngineBridge {
    /// Spawn the solver worker. Must be called from inside a tokio runtime.
    pub fn new(transport: Arc<dyn Transport>) -> (Self, mpsc::Receiver<Update>) {
        let (pending, rx_pending) = watch::channel(None);
        let (tx_update, rx_update) = mpsc::channel(FRAME_CHANNEL_CAPACITY);
        let latest = Arc::new(AtomicU64::new(0));
        tokio::spawn(worker(transport, rx_pending, tx_update, latest.clone()));
        (
            Self {
                pending,
                latest,
                next: AtomicU64::new(0),
            },
            rx_update,
        )
    }

    /// Queue a drape. Returns its epoch.
    ///
    /// Coalescing (guide §3.3): the queue holds exactly one job, so N rapid
    /// submissions leave at most one solve in flight plus the newest queued;
    /// the ones in between are never started. Any in-flight solve is
    /// superseded and its output discarded.
    pub fn submit(&self, sub: Submission) -> Epoch {
        let epoch = Epoch(self.next.fetch_add(1, Ordering::Relaxed) + 1);
        // Store BEFORE queueing, so a solve that starts immediately already
        // sees itself as current.
        self.latest.store(epoch.0, Ordering::Release);
        // Fails only if the worker task is gone; the closed update channel is
        // how the consumer learns that.
        let _ = self.pending.send(Some(PendingJob { epoch, sub }));
        epoch
    }

    /// The newest epoch handed out (0 before the first submission).
    pub fn latest_epoch(&self) -> Epoch {
        Epoch(self.latest.load(Ordering::Acquire))
    }

    /// Supersede everything in flight without queueing new work.
    pub fn cancel_all(&self) {
        self.latest.store(
            self.next.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Release,
        );
        let _ = self.pending.send(None);
    }
}

async fn worker(
    transport: Arc<dyn Transport>,
    mut pending: watch::Receiver<Option<PendingJob>>,
    updates: mpsc::Sender<Update>,
    latest: Arc<AtomicU64>,
) {
    while pending.changed().await.is_ok() {
        let job = pending.borrow_and_update().clone();
        let Some(job) = job else { continue };
        // Already superseded before we even started? Skip it entirely.
        if latest.load(Ordering::Acquire) != job.epoch.0 {
            continue;
        }
        let transport = transport.clone();
        let updates = updates.clone();
        let latest = latest.clone();
        // The solver must not run on the caller's (UI) thread; the blocking
        // pool is where a child process wait / long CPU solve belongs.
        let done = tokio::task::spawn_blocking(move || {
            let mut sink = ChannelSink {
                epoch: job.epoch,
                tier: job.sub.tier,
                tx: updates.clone(),
                latest: latest.clone(),
            };
            let result = transport.solve(&job.sub, &mut sink);
            // Stale results are dropped here — a superseded epoch must never
            // surface, not even as a failure.
            if latest.load(Ordering::Acquire) != job.epoch.0 {
                return;
            }
            let update = match result {
                Ok(summary) => Update::Done {
                    epoch: job.epoch,
                    summary,
                },
                Err(crate::BridgeError::Cancelled) => return,
                Err(e) => Update::Failed {
                    epoch: job.epoch,
                    error: format!("{e}"),
                },
            };
            let _ = updates.blocking_send(update);
        })
        .await;
        if done.is_err() {
            // The blocking task panicked; the runtime already logged it. Keep
            // the worker alive so the next submission still runs.
            tracing::error!("drape worker task panicked");
        }
    }
}

struct ChannelSink {
    epoch: Epoch,
    tier: crate::Tier,
    tx: mpsc::Sender<Update>,
    latest: Arc<AtomicU64>,
}

impl FrameSink for ChannelSink {
    fn push(&mut self, index: u32, positions: Vec<[f32; 3]>) -> Flow {
        if self.is_cancelled() {
            return Flow::Cancel;
        }
        let update = Update::Frame {
            epoch: self.epoch,
            index,
            positions,
        };
        match self.tier {
            // Preview: never stall the solver for a slow consumer. A dropped
            // preview frame is invisible — the next one supersedes it anyway.
            crate::Tier::Preview => match self.tx.try_send(update) {
                Ok(()) => Flow::Continue,
                Err(mpsc::error::TrySendError::Full(_)) => Flow::Continue,
                Err(mpsc::error::TrySendError::Closed(_)) => Flow::Cancel,
            },
            // Bake: every frame matters (it is the export), so back-pressure
            // the solver instead of dropping.
            crate::Tier::Bake => {
                if self.tx.blocking_send(update).is_err() {
                    Flow::Cancel
                } else {
                    Flow::Continue
                }
            }
        }
    }

    fn is_cancelled(&self) -> bool {
        self.latest.load(Ordering::Acquire) != self.epoch.0 || self.tx.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{BridgeError, Tier};
    use std::sync::mpsc as std_mpsc;
    use std::sync::Mutex;

    fn submission(tier: Tier) -> Submission {
        Submission {
            scene: serde_json::json!({}),
            stitches: Vec::new(),
            tier,
            frames: None,
            out_dir: None,
        }
    }

    fn summary(tier: Tier, frames: u32) -> Summary {
        Summary {
            tier,
            frames_emitted: frames,
            report: serde_json::json!({"fake": true}),
        }
    }

    /// A transport whose progress the test controls exactly — no sleeps, so no
    /// flakes. Each solve announces itself on `started`, then waits for one
    /// `gate` token before emitting each frame.
    struct GatedTransport {
        started: std_mpsc::Sender<()>,
        gate: Mutex<std_mpsc::Receiver<()>>,
        frames: u32,
        verts: usize,
    }

    impl Transport for GatedTransport {
        fn solve(
            &self,
            sub: &Submission,
            sink: &mut dyn FrameSink,
        ) -> Result<Summary, BridgeError> {
            self.started.send(()).expect("test holds the receiver");
            let mut emitted = 0;
            for index in 0..self.frames {
                self.gate
                    .lock()
                    .expect("gate")
                    .recv()
                    .expect("test holds the sender");
                if sink.is_cancelled() {
                    return Err(BridgeError::Cancelled);
                }
                if sink.push(index, vec![[index as f32; 3]; self.verts]) == Flow::Cancel {
                    return Err(BridgeError::Cancelled);
                }
                emitted += 1;
            }
            Ok(summary(sub.tier, emitted))
        }
    }

    /// A transport that emits every frame as fast as it can, no gating, and
    /// announces on `finished` (a side channel, not the update channel) the
    /// moment the last frame has been pushed.
    struct FloodTransport {
        frames: u32,
        verts: usize,
        finished: Option<std_mpsc::Sender<u32>>,
    }

    impl Transport for FloodTransport {
        fn solve(
            &self,
            sub: &Submission,
            sink: &mut dyn FrameSink,
        ) -> Result<Summary, BridgeError> {
            let mut emitted = 0;
            for index in 0..self.frames {
                if sink.push(index, vec![[index as f32; 3]; self.verts]) == Flow::Cancel {
                    return Err(BridgeError::Cancelled);
                }
                emitted += 1;
            }
            if let Some(tx) = &self.finished {
                tx.send(emitted).expect("test holds the receiver");
            }
            Ok(summary(sub.tier, emitted))
        }
    }

    /// Guide §3.3: fire N submissions rapidly; only the newest may be
    /// delivered, and no stale epoch may surface in any form.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rapid_submissions_coalesce_to_the_newest_epoch() {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (gate_tx, gate_rx) = std_mpsc::channel();
        let transport = Arc::new(GatedTransport {
            started: started_tx,
            gate: Mutex::new(gate_rx),
            frames: 2,
            verts: 4,
        });
        let (bridge, mut updates) = EngineBridge::new(transport);

        let first = bridge.submit(submission(Tier::Preview));
        // Deterministic: solve #1 is provably in flight before we supersede it.
        let started_rx = tokio::task::spawn_blocking(move || {
            started_rx.recv().expect("solve 1 started");
            started_rx
        })
        .await
        .unwrap();

        let mut newest = first;
        for _ in 0..4 {
            newest = bridge.submit(submission(Tier::Preview));
        }
        assert_ne!(first, newest);
        assert_eq!(bridge.latest_epoch(), newest);

        // Let solve #1 reach its first frame: it must observe cancellation.
        gate_tx.send(()).unwrap();
        // Wait until the winning solve is provably running before feeding it.
        tokio::task::spawn_blocking(move || started_rx.recv().expect("the newest solve started"))
            .await
            .unwrap();
        // Then let it run both its frames.
        gate_tx.send(()).unwrap();
        gate_tx.send(()).unwrap();

        let mut seen_frames = 0;
        let mut done = false;
        while let Some(update) = updates.recv().await {
            assert_eq!(
                update.epoch(),
                newest,
                "a stale epoch surfaced: {update:?} (newest = {newest:?})"
            );
            match update {
                Update::Frame { .. } => seen_frames += 1,
                Update::Done { summary, .. } => {
                    assert_eq!(summary.frames_emitted, 2);
                    done = true;
                    break;
                }
                Update::Failed { error, .. } => panic!("unexpected failure: {error}"),
            }
        }
        assert!(done, "the winning epoch never completed");
        assert_eq!(seen_frames, 2);
    }

    /// A cancelled solve must not emit a `Failed` either — silence, not noise.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn superseded_solve_emits_nothing_at_all() {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (gate_tx, gate_rx) = std_mpsc::channel();
        let transport = Arc::new(GatedTransport {
            started: started_tx,
            gate: Mutex::new(gate_rx),
            frames: 1,
            verts: 2,
        });
        let (bridge, mut updates) = EngineBridge::new(transport);

        bridge.submit(submission(Tier::Preview));
        let started_rx = tokio::task::spawn_blocking(move || {
            started_rx.recv().expect("solve 1 started");
            started_rx
        })
        .await
        .unwrap();

        bridge.cancel_all();
        gate_tx.send(()).unwrap();

        // Nothing from the cancelled epoch. Prove it by making a fresh solve
        // start afterwards: if the cancelled one had emitted, we would have
        // read it before this point.
        let winner = bridge.submit(submission(Tier::Preview));
        tokio::task::spawn_blocking(move || started_rx.recv().expect("solve 2 started"))
            .await
            .unwrap();
        gate_tx.send(()).unwrap();

        let first = updates.recv().await.expect("an update");
        assert_eq!(
            first.epoch(),
            winner,
            "got {first:?} from a cancelled epoch"
        );
    }

    /// Bake back-pressures rather than dropping: every frame arrives, in order,
    /// full length — no gaps, no torn buffers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bake_delivers_every_frame_in_order() {
        const FRAMES: u32 = 10;
        const VERTS: usize = 8;
        let (bridge, mut updates) = EngineBridge::new(Arc::new(FloodTransport {
            frames: FRAMES,
            verts: VERTS,
            finished: None,
        }));
        bridge.submit(submission(Tier::Bake));

        let mut expect = 0u32;
        loop {
            match updates.recv().await.expect("channel open") {
                Update::Frame {
                    index, positions, ..
                } => {
                    assert_eq!(index, expect, "bake dropped or reordered a frame");
                    assert_eq!(positions.len(), VERTS, "torn frame");
                    assert!(positions.iter().all(|p| *p == [index as f32; 3]));
                    expect += 1;
                }
                Update::Done { summary, .. } => {
                    assert_eq!(summary.frames_emitted, FRAMES);
                    break;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(expect, FRAMES);
    }

    /// Preview never stalls the solver. The consumer reads nothing until the
    /// solver has pushed all 64 frames (signalled out-of-band), so exactly the
    /// `FRAME_CHANNEL_CAPACITY` oldest frames can be buffered and the rest are
    /// dropped — deterministically, with no sleeps.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn preview_drops_frames_instead_of_blocking() {
        const FRAMES: u32 = 64;
        const VERTS: usize = 8;
        let (finished_tx, finished_rx) = std_mpsc::channel();
        let (bridge, mut updates) = EngineBridge::new(Arc::new(FloodTransport {
            frames: FRAMES,
            verts: VERTS,
            finished: Some(finished_tx),
        }));
        bridge.submit(submission(Tier::Preview));

        // The solver got through every frame without the consumer reading one.
        let emitted = tokio::task::spawn_blocking(move || finished_rx.recv().expect("solve ran"))
            .await
            .unwrap();
        assert_eq!(emitted, FRAMES, "the solver was stalled by the consumer");

        let mut frames = Vec::new();
        let mut done = None;
        while let Some(update) = updates.recv().await {
            match update {
                Update::Frame {
                    index, positions, ..
                } => {
                    assert_eq!(positions.len(), VERTS, "torn frame");
                    assert!(positions.iter().all(|p| *p == [index as f32; 3]));
                    frames.push(index);
                }
                Update::Done { summary, .. } => {
                    done = Some(summary);
                    break;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(done.expect("Done delivered").frames_emitted, FRAMES);
        assert_eq!(
            frames,
            (0..FRAME_CHANNEL_CAPACITY as u32).collect::<Vec<_>>(),
            "preview should buffer exactly the {FRAME_CHANNEL_CAPACITY} oldest frames and drop the rest"
        );
    }
}
