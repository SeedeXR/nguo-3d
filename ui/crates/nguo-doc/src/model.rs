// SPDX-License-Identifier: GPL-2.0-or-later
//! The document's data types.
//!
//! Everything here is plain, serializable data with public fields. It is not
//! the mutation surface: values only enter a [`crate::Document`] through
//! [`crate::Edit`], because the document's own fields are private.

use std::collections::BTreeMap;

use kurbo::{Affine, Point};
use nguo_measure::Unit;
use nguo_pattern::{Arena, EdgeId, Id, VectorNetwork, VertexId};
use serde::{Deserialize, Serialize};

pub type PieceId = Id<Piece>;
pub type SeamId = Id<Seam>;
pub type MaterialId = Id<Material>;
pub type FeatureId = Id<Feature>;

/// Project-level settings (guide §4.1 "Metadata").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    pub name: String,
    pub author: String,
    /// The unit shown to the user. Geometry is always millimetres (guide §20).
    pub display_unit: Unit,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

impl Default for Metadata {
    fn default() -> Self {
        Metadata {
            name: "Untitled".into(),
            author: String::new(),
            display_unit: Unit::Millimeter,
            notes: String::new(),
        }
    }
}

/// One pattern piece: a vector network plus the things drawn on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    pub name: String,
    pub network: VectorNetwork,
    /// Placement on the 2D drafting canvas. Never bake the view zoom in here
    /// (guide §7.1).
    pub transform: Affine,
    pub material: Option<MaterialId>,
    pub features: Arena<Feature>,
    pub symmetry: Option<Symmetry>,
    /// Fallback seam allowance in mm for seam edges with no per-seam width.
    pub default_allowance_mm: f64,
    /// Where this piece sits in the 3D arrangement before draping (guide §11).
    pub placement: Option<Placement>,
}

impl Piece {
    pub fn new(name: impl Into<String>) -> Piece {
        Piece {
            name: name.into(),
            network: VectorNetwork::new(),
            transform: Affine::IDENTITY,
            material: None,
            features: Arena::new(),
            symmetry: None,
            default_allowance_mm: 10.0,
            placement: None,
        }
    }
}

/// A mirror line, optionally linked to the piece it generates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Symmetry {
    pub axis_from: Point,
    pub axis_to: Point,
    pub mirrored_piece: Option<PieceId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotchKind {
    Slit,
    V,
    Castle,
}

/// Things drawn on a piece that are not themselves network topology
/// (guide §4.1 "Entities").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Feature {
    /// A registration mark at parameter `t` along `edge`.
    Notch {
        edge: EdgeId,
        t: f64,
        kind: NotchKind,
    },
    /// Groups the two dart legs and their apex. The legs themselves are
    /// `EdgeSemantic::DartLeg` edges in the network.
    Dart { apex: VertexId, legs: [EdgeId; 2] },
    /// Fabric grain direction, in piece-local mm.
    GrainLine { from: Point, to: Point },
    /// A named run of `EdgeSemantic::Internal` edges (pocket line, style line).
    InternalLine { name: String, edges: Vec<EdgeId> },
    /// Text printed on the piece.
    Label {
        at: Point,
        angle_rad: f64,
        text: String,
    },
    /// The edge this piece **hangs from**, and the body landmark it hangs at: a
    /// waistband at the waist, a bodice at the underbust. `landmark` is the
    /// avatar manifest's own key (`"waist"`, `"hip"`, `"underbust"`, …) as a
    /// plain string, so `nguo-doc` needs no dependency on `nguo-avatar` —
    /// `nguo_avatar::Landmark::from_key` resolves it.
    ///
    /// This is **placement only**. It says where the garment goes, never how big
    /// it is: nothing downstream may scale a piece to make a girth match, because
    /// re-sizing a drafted pattern to a body is *grading* and grading re-evaluates
    /// the parameter table (guide §19), it is not a transform.
    ///
    /// No schema bump: an additive enum variant, and a document without one is
    /// still a valid schema-2 document (the garment then falls back to the
    /// arranger's default height).
    Support { edge: EdgeId, landmark: String },
}

/// One side of a seam: an ordered chain of edges within one piece, and the
/// sub-span of that chain that actually sews (guide §10 segment correspondence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeamSide {
    pub piece: PieceId,
    /// `(edge, reversed)` — `reversed` means traverse the edge end-to-start.
    pub edges: Vec<(EdgeId, bool)>,
    /// Normalised span along the chain, `0.0..=1.0`. `(0.0, 1.0)` is the whole
    /// chain; a partial span is how a short edge sews to part of a long one.
    pub span: (f64, f64),
}

impl SeamSide {
    pub fn whole(piece: PieceId, edges: Vec<(EdgeId, bool)>) -> SeamSide {
        SeamSide {
            piece,
            edges,
            span: (0.0, 1.0),
        }
    }
}

/// A seam lives outside the geometry and references edge ids, so editing an
/// edge cannot orphan it (guide §10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Seam {
    pub a: SeamSide,
    pub b: SeamSide,
    /// Seam allowance for this seam, in mm. Overrides the pieces' defaults.
    pub allowance_mm: f64,
    /// `len(b) / len(a)` the seam is built for. `1.0` is flat; `> 1.0` gathers
    /// side `b` into side `a`.
    pub ease_ratio: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

impl Seam {
    pub fn new(a: SeamSide, b: SeamSide) -> Seam {
        Seam {
            a,
            b,
            allowance_mm: 10.0,
            ease_ratio: 1.0,
            note: String::new(),
        }
    }

    pub fn references(&self, piece: PieceId) -> bool {
        self.a.piece == piece || self.b.piece == piece
    }

    pub fn references_edge(&self, piece: PieceId, edge: EdgeId) -> bool {
        [&self.a, &self.b]
            .iter()
            .any(|s| s.piece == piece && s.edges.iter().any(|&(e, _)| e == edge))
    }
}

/// Fabric parameters. These map onto the engine's material inputs (guide §14).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    pub weight_gsm: f64,
    pub thickness_mm: f64,
    pub stretch_warp: f64,
    pub stretch_weft: f64,
    pub bend: f64,
    pub friction: f64,
    pub color_srgb: [f32; 3],
}

impl Default for Material {
    fn default() -> Self {
        Material {
            name: "Cotton poplin".into(),
            weight_gsm: 120.0,
            thickness_mm: 0.3,
            stretch_warp: 0.02,
            stretch_weft: 0.03,
            bend: 1.0e-5,
            friction: 0.4,
            color_srgb: [0.85, 0.85, 0.85],
        }
    }
}

/// The body the garment is drafted for (guide §12). Measurements are in mm and
/// are visible to expressions as bare names or as `body.<name>`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AvatarRef {
    pub name: String,
    pub measurements: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pose: String,
    /// Path inside the project bundle, resolved by `nguo-io`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset: Option<String>,
}

/// 3D placement of a piece before draping.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub position_mm: [f64; 3],
    /// `[x, y, z, w]`.
    pub rotation_quat: [f64; 4],
}

impl Default for Placement {
    fn default() -> Self {
        Placement {
            position_mm: [0.0; 3],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub eye_mm: [f64; 3],
    pub target_mm: [f64; 3],
    pub fov_y_rad: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            eye_mm: [0.0, 900.0, 2500.0],
            target_mm: [0.0, 900.0, 0.0],
            fov_y_rad: 0.7,
        }
    }
}
