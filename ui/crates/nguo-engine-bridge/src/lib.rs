// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-engine-bridge` — the async interface from the app to the ppf-mguso
//! cloth solver (ui/docs/AGENTIC.md §7).
//!
//! # What this gives you
//!
//! * [`EngineBridge::submit`] takes a [`Submission`] (a scene + a solve tier)
//!   and returns an [`Epoch`]. Drape frames and the final summary arrive on the
//!   [`Update`] channel handed back by [`EngineBridge::new`].
//! * Two tiers ([`Tier::Preview`] = XPBD, [`Tier::Bake`] = implicit backward
//!   Euler), per guide §14 / engine §7.
//! * The solve never runs on the caller's thread: it is driven by a worker task
//!   on tokio's blocking pool.
//! * Every job carries an epoch. A newer submission supersedes an older one:
//!   the in-flight solve is cancelled at its next frame boundary and **nothing**
//!   from a stale epoch is ever delivered (guide §3.3). At most one solve is in
//!   flight and one is queued — rapid input is coalesced to the newest.
//!
//! # Why there is a [`Transport`] trait
//!
//! Two implementations exist, and the choice between them is not about speed.
//!
//! * [`InProcessTransport`] calls `engine-api` directly. **Use it to drape a
//!   sewn garment.** `engine_api::Scene` has no stitch field
//!   (`engine-api/src/scene.rs:71-78`) — stitches reach the solver only through
//!   `ImplicitSolver::set_stitches` — so this is the only transport that can
//!   close a seam.
//! * [`ProcessTransport`] spawns the `mguso` binary over its documented process
//!   contract (`core-engine/docs/PROCESS_CONTRACT.md`). It writes a per-frame OBJ
//!   sequence, which is what an export wants, and it isolates a solver crash in
//!   another process. It **rejects** a submission carrying stitches rather than
//!   quietly draping the garment unsewn.
//!
//! The original reason for the seam — `core-engine` on `wgpu 30` against this
//! workspace's `wgpu 27` — is **gone**: both lockfiles now resolve `wgpu 27.0.1`,
//! which is what made [`InProcessTransport`] possible at all. The trait stays
//! because two live implementations now use it.

mod bridge;
mod inproc;
mod process;
mod transport;

pub use bridge::{EngineBridge, Epoch, Update, FRAME_CHANNEL_CAPACITY};
pub use inproc::InProcessTransport;
pub use process::{resolve_engine_binary, ProcessTransport};
pub use transport::{BridgeError, Flow, FrameSink, Stitch, Submission, Summary, Tier, Transport};
