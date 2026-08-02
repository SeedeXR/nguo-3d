// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-cli` — the headless operation surface for the nguo-3d cloth suite.
//!
//! The binary (`nguo`) is a thin shell over [`ops`]. `nguo-mcp` exposes the
//! same [`ops::SPECS`] as MCP tools and calls the same [`ops::dispatch`], so
//! there is exactly one implementation of every operation.

pub mod cli;
pub mod ops;
