// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-sewing` — seam topology, segment correspondence, 3D arrangement, and
//! the 2D → 3D assembly that becomes a drape (`frontend-guide.md` §10–§11,
//! milestone M6).
//!
//! # The path
//!
//! ```text
//!   nguo-doc (pieces + seams, mm)
//!        │  validate            maintained constraint, per revision
//!        │  triangulate         nguo-triangulate, mm -> m
//!        │  arrange             flat panels wrapped around the body
//!        │  correspond          seam sides -> weighted stitches
//!        ▼
//!   Garment { positions, tris, uvs, stitches }      metres, glTF space
//! ```
//!
//! # What this crate deliberately does not know
//!
//! The solver. [`Garment`] is plain geometry plus a stitch list in
//! `engine_core::stitch`'s *shape* — six indices, six weights — and nothing
//! here depends on the engine. `nguo-engine-bridge` owns that boundary, which
//! is why every test in this crate runs in milliseconds.
//!
//! # Units and axes
//!
//! Pattern space is **millimetres, `+y` downward**. World space is **metres,
//! `+Y` up, `+Z` forward, right-handed** — identical to glTF and to the avatar
//! bundle, so there is no axis conversion anywhere in this crate. The one
//! mm → m conversion on the path lives in `nguo-triangulate`; the arranger uses
//! `1e-3` on pattern coordinates it places directly, and
//! `arrange::tests::wrapping_a_panel_preserves_its_width_as_arc_length` is what
//! keeps the two agreeing.

#![forbid(unsafe_code)]

pub mod arrange;
pub mod correspond;
pub mod garment;
pub mod sample;
pub mod validate;

pub use arrange::{
    arrange, hang, seam_order, support_arc_mm, Aabb, Archetype, ArrangeOptions, Arrangement, Hang,
    PieceExtent, Placement3d, DEFAULT_GAP_M,
};
pub use correspond::{
    check_chain, correspond, pairing_distance_m, side_length_mm, side_samples, Correspondence,
    SeamError, SewnSeam, SideSamples, Stitch6, EASE_LIMIT,
};
pub use garment::{build, Garment, PieceMesh, SewError, SewOptions, Timings};
pub use validate::{counts, is_sewable, validate, Diagnostic, Kind, Severity};
