// SPDX-License-Identifier: GPL-2.0-or-later
//! The document itself.

use nguo_measure::{EvalError, ParamTable, Unit};
use nguo_pattern::Arena;
use serde::{Deserialize, Serialize};

use crate::error::DocError;
use crate::migrate::{self, SCHEMA_VERSION};
use crate::model::{
    AvatarRef, Camera, Material, MaterialId, Metadata, Piece, PieceId, Seam, SeamId,
};

/// The single source of truth (guide §4).
///
/// # Why every field is `pub(crate)`
/// Nothing outside this crate can reach in and change a value. The only way to
/// mutate a document is [`Document::apply_edit`], which takes a closed
/// [`crate::Edit`] and hands back the exact inverse — which is what makes undo
/// correct by construction rather than by discipline. `nguo-doc-commands` is
/// built on that single entry point; it has no other way in either.
///
/// # Derived state is not here
/// Tessellations, spatial indices, sim meshes and thumbnails are rebuildable
/// and must never be saved as truth (guide §4.3). They live in
/// [`crate::Derived`], which is deliberately not serializable, keyed on
/// [`Document::revision`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// First field so a hand-edited or partially-read manifest still surfaces
    /// its version before anything else.
    pub(crate) schema_version: u32,
    pub(crate) meta: Metadata,
    pub(crate) pieces: Arena<Piece>,
    pub(crate) seams: Arena<Seam>,
    pub(crate) materials: Arena<Material>,
    pub(crate) params: ParamTable,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) avatar: Option<AvatarRef>,
    pub(crate) camera: Camera,
    /// Bumped by every applied edit. Not content: skipped by serde and ignored
    /// by `PartialEq`, so "edit then undo" is byte-identical to the original.
    #[serde(skip)]
    pub(crate) revision: u64,
}

impl Default for Document {
    fn default() -> Self {
        Document {
            schema_version: SCHEMA_VERSION,
            meta: Metadata::default(),
            pieces: Arena::new(),
            seams: Arena::new(),
            materials: Arena::new(),
            params: ParamTable::new(),
            avatar: None,
            camera: Camera::default(),
            revision: 0,
        }
    }
}

impl PartialEq for Document {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.meta == other.meta
            && self.pieces == other.pieces
            && self.seams == other.seams
            && self.materials == other.materials
            && self.params == other.params
            && self.avatar == other.avatar
            && self.camera == other.camera
    }
}

impl Document {
    pub fn new() -> Document {
        Document::default()
    }

    // --- read -------------------------------------------------------------

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Monotone edit counter. Derived caches key off this (guide §4.3).
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn meta(&self) -> &Metadata {
        &self.meta
    }

    pub fn display_unit(&self) -> Unit {
        self.meta.display_unit
    }

    pub fn pieces(&self) -> &Arena<Piece> {
        &self.pieces
    }

    pub fn piece(&self, id: PieceId) -> Result<&Piece, DocError> {
        self.pieces.get(id).ok_or(DocError::NoSuchPiece(id))
    }

    pub fn seams(&self) -> &Arena<Seam> {
        &self.seams
    }

    pub fn seam(&self, id: SeamId) -> Result<&Seam, DocError> {
        self.seams.get(id).ok_or(DocError::NoSuchSeam(id))
    }

    pub fn materials(&self) -> &Arena<Material> {
        &self.materials
    }

    pub fn material(&self, id: MaterialId) -> Result<&Material, DocError> {
        self.materials.get(id).ok_or(DocError::NoSuchMaterial(id))
    }

    pub fn params(&self) -> &ParamTable {
        &self.params
    }

    pub fn avatar(&self) -> Option<&AvatarRef> {
        self.avatar.as_ref()
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Ids the next `Add*` edit should use. Pure — planning an edit does not
    /// change the document.
    pub fn next_piece_id(&self) -> PieceId {
        self.pieces.next_id()
    }

    pub fn next_seam_id(&self) -> SeamId {
        self.seams.next_id()
    }

    pub fn next_material_id(&self) -> MaterialId {
        self.materials.next_id()
    }

    // --- expressions ------------------------------------------------------

    /// Evaluate a numeric-field expression in the document's context: the
    /// parameter table first, then avatar measurements (bare or `body.`-
    /// prefixed). Returns millimetres.
    pub fn eval(&self, src: &str) -> Result<f64, EvalError> {
        self.eval_in(src, self.meta.display_unit)
    }

    pub fn eval_in(&self, src: &str, unit: Unit) -> Result<f64, EvalError> {
        self.params.eval_with(src, unit, &|name| {
            let a = self.avatar.as_ref()?;
            let key = name.strip_prefix("body.").unwrap_or(name);
            a.measurements.get(key).copied()
        })
    }

    // --- serialization ----------------------------------------------------

    pub fn to_json(&self) -> Result<String, DocError> {
        serde_json::to_string(self).map_err(|e| DocError::Serialize(e.to_string()))
    }

    pub fn to_json_pretty(&self) -> Result<String, DocError> {
        serde_json::to_string_pretty(self).map_err(|e| DocError::Serialize(e.to_string()))
    }

    /// Load a manifest, migrating it up to [`SCHEMA_VERSION`] first.
    ///
    /// `nguo-io` owns the zip; this owns the schema. Always go through here
    /// rather than `serde_json::from_str::<Document>` so old projects keep
    /// opening (guide §4.4).
    pub fn from_json(src: &str) -> Result<Document, DocError> {
        let value: serde_json::Value =
            serde_json::from_str(src).map_err(|e| DocError::Serialize(e.to_string()))?;
        Document::from_value(value)
    }

    pub fn from_value(value: serde_json::Value) -> Result<Document, DocError> {
        let migrated = migrate::migrate(value)?;
        serde_json::from_value(migrated).map_err(|e| DocError::Serialize(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AvatarRef;

    #[test]
    fn a_new_document_is_at_the_current_schema() {
        assert_eq!(Document::new().schema_version(), SCHEMA_VERSION);
    }

    #[test]
    fn round_trips_through_json() {
        let d = Document::new();
        let back = Document::from_json(&d.to_json().unwrap()).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn expressions_see_params_then_avatar() {
        let mut d = Document::new();
        d.params.set("ease", "2cm");
        let mut a = AvatarRef::default();
        a.measurements.insert("waist".into(), 800.0);
        d.avatar = Some(a);
        assert_eq!(d.eval("waist/4 + ease").unwrap(), 220.0);
        assert_eq!(d.eval("body.waist/4").unwrap(), 200.0);
        assert!(d.eval("collar").is_err());
    }

    #[test]
    fn revision_is_not_serialized() {
        let mut d = Document::new();
        d.revision = 42;
        let json = d.to_json().unwrap();
        assert!(!json.contains("revision"));
        assert_eq!(Document::from_json(&json).unwrap().revision(), 0);
    }
}
