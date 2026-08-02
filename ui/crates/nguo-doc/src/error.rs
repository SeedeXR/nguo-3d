// SPDX-License-Identifier: GPL-2.0-or-later
//! Typed, actionable document errors (ui/docs/OBSERVABILITY.md §6).

use nguo_measure::EvalError;
use nguo_pattern::NetworkError;

use crate::migrate::MigrationError;
use crate::model::{FeatureId, MaterialId, PieceId, SeamId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DocError {
    #[error("no such piece {0:?}")]
    NoSuchPiece(PieceId),
    #[error("no such seam {0:?}")]
    NoSuchSeam(SeamId),
    #[error("no such material {0:?}")]
    NoSuchMaterial(MaterialId),
    #[error("no such feature {0:?}")]
    NoSuchFeature(FeatureId),
    #[error("{0} already exists")]
    IdInUse(String),
    #[error("{what} is still referenced by {by}")]
    StillReferenced { what: String, by: String },
    #[error(transparent)]
    Network(#[from] NetworkError),
    #[error(transparent)]
    Expr(#[from] EvalError),
    #[error(transparent)]
    Migration(#[from] MigrationError),
    #[error("serialization failed: {0}")]
    Serialize(String),
}
