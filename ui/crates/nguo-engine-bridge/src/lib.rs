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
//! This is a deliberate, load-bearing seam, not speculative abstraction.
//! `core-engine` is on `wgpu 30` today while this workspace is pinned to
//! `wgpu 27` (`core-engine/docs/WGPU_PIN.md`); a path dependency on
//! `engine-api`/`engine-core`
//! right now would pull two incompatible wgpu majors into one binary. So the
//! only implementation that exists today is [`ProcessTransport`], which drives
//! the `mguso` binary over its documented process contract
//! (`core-engine/docs/PROCESS_CONTRACT.md`). Once the wgpu pin unifies, an
//! in-process transport calling `engine_core` directly is a drop-in second
//! impl behind the same trait — nothing above [`Transport`] changes.

mod bridge;
mod process;
mod transport;

pub use bridge::{EngineBridge, Epoch, Update, FRAME_CHANNEL_CAPACITY};
pub use process::{resolve_engine_binary, ProcessTransport};
pub use transport::{BridgeError, Flow, FrameSink, Submission, Summary, Tier, Transport};
