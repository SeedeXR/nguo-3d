// SPDX-License-Identifier: GPL-2.0-or-later
//! The nguo-3d document model: the single source of authored truth.
//!
//! UI-framework-free — `kurbo`, `serde`, `nguo-measure` and
//! `nguo-pattern` only. Fully testable headless.
//!
//! # The shape of it
//! - [`Document`] holds pieces, seams, materials, parameters, the avatar
//!   reference, the camera and metadata. Its fields are private.
//! - [`Edit`] is the *only* way to change one. [`Document::apply_edit`] returns
//!   the exact inverse edit, which is what `nguo-doc-commands` stacks for undo.
//! - [`Derived`] holds anything computed from the document. It has no
//!   `Serialize` impl on purpose: derived state can never be saved as truth
//!   (guide §4.3).
//! - [`migrate`] carries old manifests forward; [`SCHEMA_VERSION`] is what this
//!   build writes.
//!
//! # Units
//! Millimetres, everywhere, always (guide §20). [`Metadata::display_unit`] is
//! for the UI edge and nothing else.

#![forbid(unsafe_code)]

mod derived;
mod doc;
mod edit;
mod error;
pub mod migrate;
pub mod model;

pub use derived::Derived;
pub use doc::Document;
pub use edit::Edit;
pub use error::DocError;
pub use migrate::{migrate, MigrationError, SCHEMA_VERSION};
pub use model::{
    AvatarRef, Camera, Feature, FeatureId, Material, MaterialId, Metadata, NotchKind, Piece,
    PieceId, Placement, Seam, SeamId, SeamSide, Symmetry,
};

/// Re-exported so every crate downstream agrees on one `kurbo`, one
/// `nguo_pattern` and one `nguo_measure`.
pub use kurbo;
pub use nguo_measure as measure;
pub use nguo_pattern as pattern;
