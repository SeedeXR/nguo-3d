// SPDX-License-Identifier: GPL-2.0-or-later
//! The transport seam: what "run one drape" means, independent of *where* the
//! solver lives. See the crate docs for why this trait exists.

use std::path::PathBuf;

/// Solve tier — interactive preview vs on-demand high-fidelity bake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// XPBD — fast, interactive. Used while arranging/dragging; frames may be
    /// dropped if the consumer falls behind (the next one supersedes them).
    Preview,
    /// Implicit backward Euler — high fidelity, on demand. Every frame is
    /// delivered; the solver is back-pressured rather than dropping frames.
    Bake,
}

impl Tier {
    /// The `mguso --solver` value for this tier.
    pub fn solver_flag(self) -> &'static str {
        match self {
            Tier::Preview => "xpbd",
            Tier::Bake => "implicit",
        }
    }
}

/// One garment submission: the scene to drape and how to drape it.
///
/// `scene` is an `engine-api::Scene` JSON document (the same thing
/// `mguso run scene.json` eats). It is carried as [`serde_json::Value`]
/// rather than a typed struct because typing it would mean depending on
/// `engine-api`, which is on the wrong wgpu major today (crate docs).
#[derive(Debug, Clone, PartialEq)]
pub struct Submission {
    pub scene: serde_json::Value,
    pub tier: Tier,
    /// Frames to simulate. `None` = whatever the scene's `config.frames` says.
    pub frames: Option<u32>,
    /// Where to keep the per-frame OBJ sequence. `None` = a temp dir that is
    /// deleted when the solve ends (frames still stream to the consumer).
    pub out_dir: Option<PathBuf>,
}

/// What the engine reported when the solve finished. `report` is the engine's
/// own stdout JSON, verbatim — we do not re-derive or embellish it.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Summary {
    pub tier: Tier,
    pub frames_emitted: u32,
    pub report: serde_json::Value,
}

/// Whether the transport should keep solving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// The result is no longer wanted (superseded, or the consumer is gone).
    /// Stop the solve; anything further would be thrown away.
    Cancel,
}

/// Where a transport hands finished frames. Implemented by the bridge.
pub trait FrameSink {
    /// Publish frame `index`. Ownership of `positions` moves to the sink, so a
    /// consumer can never observe a half-written frame.
    fn push(&mut self, index: u32, positions: Vec<[f32; 3]>) -> Flow;
    /// Cheap check for cancellation between frames (a long implicit step can
    /// take a while; do not wait for the next `push` to notice).
    fn is_cancelled(&self) -> bool;
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// The engine ran and failed. `stderr_tail` is the engine's own log output,
    /// never a paraphrase.
    #[error("engine exited with {code} ({context}){}", tail(.stderr_tail))]
    Engine {
        code: i32,
        context: String,
        stderr_tail: String,
    },
    #[error("engine binary not usable: {0}")]
    Binary(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("could not parse engine output as JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Superseded by a newer epoch, or the consumer dropped the channel.
    #[error("solve cancelled")]
    Cancelled,
}

fn tail(s: &str) -> String {
    if s.trim().is_empty() {
        String::new()
    } else {
        format!(": {}", s.trim())
    }
}

/// One way of running a drape.
///
/// Deliberately **blocking**: the only implementation today spawns a child
/// process and waits on it, and the bridge already runs transports on tokio's
/// blocking pool. A blocking trait is also object-safe, which `async fn` in
/// trait is not on the pinned toolchain.
pub trait Transport: Send + Sync + 'static {
    /// Run one drape to completion, pushing frames to `sink` as they are
    /// produced. Must return [`BridgeError::Cancelled`] promptly once
    /// `sink.is_cancelled()` or a `push` returns [`Flow::Cancel`].
    fn solve(&self, sub: &Submission, sink: &mut dyn FrameSink) -> Result<Summary, BridgeError>;
}
