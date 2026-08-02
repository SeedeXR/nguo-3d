// SPDX-License-Identifier: GPL-2.0-or-later
//! The one and only way to change a [`Document`].
//!
//! [`Edit`] is a closed set of primitive, individually invertible mutations.
//! [`Document::apply_edit`] applies one and returns the edit that undoes it, so
//! undo is not a second implementation that has to be kept in step — it is the
//! same code path run on the returned value.
//!
//! `nguo-doc-commands` composes these into transactions and a stack. It cannot
//! do anything this enum does not allow, because `Document`'s fields are
//! `pub(crate)`.

use kurbo::{Affine, Point};
use nguo_measure::Param;
use nguo_pattern::{Edge, EdgeGeom, EdgeId, EdgeSemantic, VertexId};
use serde::{Deserialize, Serialize};

use crate::doc::Document;
use crate::error::DocError;
use crate::model::{
    AvatarRef, Camera, Feature, FeatureId, Material, MaterialId, Metadata, Piece, PieceId,
    Placement, Seam, SeamId, Symmetry,
};

/// A primitive document mutation.
///
/// Serializable on purpose: the same values feed the undo stack, the autosave
/// journal (guide §5.4) and, later, collaboration (guide §24).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Edit {
    // --- pieces -----------------------------------------------------------
    AddPiece {
        id: PieceId,
        piece: Box<Piece>,
    },
    RemovePiece {
        id: PieceId,
    },
    SetPieceName {
        id: PieceId,
        name: String,
    },
    SetPieceTransform {
        id: PieceId,
        transform: Affine,
    },
    SetPieceMaterial {
        id: PieceId,
        material: Option<MaterialId>,
    },
    SetPieceAllowance {
        id: PieceId,
        mm: f64,
    },
    SetPieceSymmetry {
        id: PieceId,
        symmetry: Option<Symmetry>,
    },
    SetPiecePlacement {
        id: PieceId,
        placement: Option<Placement>,
    },

    // --- geometry ---------------------------------------------------------
    AddVertex {
        piece: PieceId,
        id: VertexId,
        pos: Point,
    },
    RemoveVertex {
        piece: PieceId,
        id: VertexId,
    },
    MoveVertex {
        piece: PieceId,
        id: VertexId,
        pos: Point,
    },
    AddEdge {
        piece: PieceId,
        id: EdgeId,
        edge: Edge,
    },
    RemoveEdge {
        piece: PieceId,
        id: EdgeId,
    },
    SetEdgeGeom {
        piece: PieceId,
        id: EdgeId,
        geom: EdgeGeom,
    },
    SetEdgeSemantic {
        piece: PieceId,
        id: EdgeId,
        semantic: EdgeSemantic,
    },
    SetEdgeEnds {
        piece: PieceId,
        id: EdgeId,
        start: VertexId,
        end: VertexId,
    },

    // --- features ---------------------------------------------------------
    AddFeature {
        piece: PieceId,
        id: FeatureId,
        feature: Box<Feature>,
    },
    RemoveFeature {
        piece: PieceId,
        id: FeatureId,
    },
    SetFeature {
        piece: PieceId,
        id: FeatureId,
        feature: Box<Feature>,
    },

    // --- seams ------------------------------------------------------------
    AddSeam {
        id: SeamId,
        seam: Box<Seam>,
    },
    RemoveSeam {
        id: SeamId,
    },
    SetSeam {
        id: SeamId,
        seam: Box<Seam>,
    },

    // --- materials --------------------------------------------------------
    AddMaterial {
        id: MaterialId,
        material: Box<Material>,
    },
    RemoveMaterial {
        id: MaterialId,
    },
    SetMaterial {
        id: MaterialId,
        material: Box<Material>,
    },

    // --- everything else --------------------------------------------------
    /// `None` removes the parameter.
    SetParam {
        name: String,
        param: Option<Param>,
    },
    SetMetadata {
        meta: Box<Metadata>,
    },
    SetAvatar {
        avatar: Option<Box<AvatarRef>>,
    },
    SetCamera {
        camera: Camera,
    },
}

impl Edit {
    /// A short, human-readable name for menus and the undo history.
    pub fn label(&self) -> &'static str {
        match self {
            Edit::AddPiece { .. } => "Add piece",
            Edit::RemovePiece { .. } => "Delete piece",
            Edit::SetPieceName { .. } => "Rename piece",
            Edit::SetPieceTransform { .. } => "Move piece",
            Edit::SetPieceMaterial { .. } => "Assign fabric",
            Edit::SetPieceAllowance { .. } => "Set seam allowance",
            Edit::SetPieceSymmetry { .. } => "Set symmetry",
            Edit::SetPiecePlacement { .. } => "Arrange piece",
            Edit::AddVertex { .. } => "Add point",
            Edit::RemoveVertex { .. } => "Delete point",
            Edit::MoveVertex { .. } => "Move point",
            Edit::AddEdge { .. } => "Add edge",
            Edit::RemoveEdge { .. } => "Delete edge",
            Edit::SetEdgeGeom { .. } => "Reshape edge",
            Edit::SetEdgeSemantic { .. } => "Set edge type",
            Edit::SetEdgeEnds { .. } => "Reconnect edge",
            Edit::AddFeature { .. } => "Add feature",
            Edit::RemoveFeature { .. } => "Delete feature",
            Edit::SetFeature { .. } => "Edit feature",
            Edit::AddSeam { .. } => "Add seam",
            Edit::RemoveSeam { .. } => "Delete seam",
            Edit::SetSeam { .. } => "Edit seam",
            Edit::AddMaterial { .. } => "Add fabric",
            Edit::RemoveMaterial { .. } => "Delete fabric",
            Edit::SetMaterial { .. } => "Edit fabric",
            Edit::SetParam { .. } => "Set parameter",
            Edit::SetMetadata { .. } => "Project settings",
            Edit::SetAvatar { .. } => "Set avatar",
            Edit::SetCamera { .. } => "Move camera",
        }
    }
}

impl Document {
    /// Apply one primitive edit and return the edit that undoes it.
    ///
    /// On error the document is untouched: every variant validates before it
    /// writes.
    pub fn apply_edit(&mut self, edit: Edit) -> Result<Edit, DocError> {
        let inverse = self.apply_inner(edit)?;
        self.revision += 1;
        Ok(inverse)
    }

    fn apply_inner(&mut self, edit: Edit) -> Result<Edit, DocError> {
        match edit {
            // --- pieces ---------------------------------------------------
            Edit::AddPiece { id, piece } => {
                if self.pieces.contains(id) {
                    return Err(DocError::IdInUse(format!("piece {id:?}")));
                }
                self.pieces.insert_at(id, *piece);
                Ok(Edit::RemovePiece { id })
            }
            Edit::RemovePiece { id } => {
                if !self.pieces.contains(id) {
                    return Err(DocError::NoSuchPiece(id));
                }
                if let Some((sid, _)) = self.seams.iter().find(|(_, s)| s.references(id)) {
                    return Err(DocError::StillReferenced {
                        what: format!("piece {id:?}"),
                        by: format!("seam {sid:?}"),
                    });
                }
                let piece = self.pieces.remove(id).expect("checked above");
                Ok(Edit::AddPiece {
                    id,
                    piece: Box::new(piece),
                })
            }
            Edit::SetPieceName { id, name } => {
                let p = self.piece_mut(id)?;
                let old = std::mem::replace(&mut p.name, name);
                Ok(Edit::SetPieceName { id, name: old })
            }
            Edit::SetPieceTransform { id, transform } => {
                let p = self.piece_mut(id)?;
                let old = std::mem::replace(&mut p.transform, transform);
                Ok(Edit::SetPieceTransform { id, transform: old })
            }
            Edit::SetPieceMaterial { id, material } => {
                if let Some(m) = material {
                    if !self.materials.contains(m) {
                        return Err(DocError::NoSuchMaterial(m));
                    }
                }
                let p = self.piece_mut(id)?;
                let old = std::mem::replace(&mut p.material, material);
                Ok(Edit::SetPieceMaterial { id, material: old })
            }
            Edit::SetPieceAllowance { id, mm } => {
                let p = self.piece_mut(id)?;
                let old = std::mem::replace(&mut p.default_allowance_mm, mm);
                Ok(Edit::SetPieceAllowance { id, mm: old })
            }
            Edit::SetPieceSymmetry { id, symmetry } => {
                let p = self.piece_mut(id)?;
                let old = std::mem::replace(&mut p.symmetry, symmetry);
                Ok(Edit::SetPieceSymmetry { id, symmetry: old })
            }
            Edit::SetPiecePlacement { id, placement } => {
                let p = self.piece_mut(id)?;
                let old = std::mem::replace(&mut p.placement, placement);
                Ok(Edit::SetPiecePlacement { id, placement: old })
            }

            // --- geometry -------------------------------------------------
            Edit::AddVertex { piece, id, pos } => {
                let p = self.piece_mut(piece)?;
                if p.network.vertices.contains(id) {
                    return Err(DocError::IdInUse(format!("vertex {id:?}")));
                }
                p.network
                    .vertices
                    .insert_at(id, nguo_pattern::Vertex { pos });
                Ok(Edit::RemoveVertex { piece, id })
            }
            Edit::RemoveVertex { piece, id } => {
                let p = self.piece_mut(piece)?;
                let pos = p.network.pos(id)?;
                p.network.remove_vertex(id)?;
                Ok(Edit::AddVertex { piece, id, pos })
            }
            Edit::MoveVertex { piece, id, pos } => {
                let p = self.piece_mut(piece)?;
                let old = p.network.set_vertex_pos(id, pos)?;
                Ok(Edit::MoveVertex {
                    piece,
                    id,
                    pos: old,
                })
            }
            Edit::AddEdge { piece, id, edge } => {
                let p = self.piece_mut(piece)?;
                if p.network.edges.contains(id) {
                    return Err(DocError::IdInUse(format!("edge {id:?}")));
                }
                // Reject dangling references up front.
                p.network.pos(edge.start)?;
                p.network.pos(edge.end)?;
                p.network.edges.insert_at(id, edge);
                Ok(Edit::RemoveEdge { piece, id })
            }
            Edit::RemoveEdge { piece, id } => {
                if let Some((sid, _)) = self
                    .seams
                    .iter()
                    .find(|(_, s)| s.references_edge(piece, id))
                {
                    return Err(DocError::StillReferenced {
                        what: format!("edge {id:?}"),
                        by: format!("seam {sid:?}"),
                    });
                }
                let p = self.piece_mut(piece)?;
                let edge = p.network.remove_edge(id)?;
                Ok(Edit::AddEdge { piece, id, edge })
            }
            Edit::SetEdgeGeom { piece, id, geom } => {
                let p = self.piece_mut(piece)?;
                let e = p
                    .network
                    .edges
                    .get_mut(id)
                    .ok_or(nguo_pattern::NetworkError::NoSuchEdge(id))?;
                let old = std::mem::replace(&mut e.geom, geom);
                Ok(Edit::SetEdgeGeom {
                    piece,
                    id,
                    geom: old,
                })
            }
            Edit::SetEdgeSemantic {
                piece,
                id,
                semantic,
            } => {
                let p = self.piece_mut(piece)?;
                let e = p
                    .network
                    .edges
                    .get_mut(id)
                    .ok_or(nguo_pattern::NetworkError::NoSuchEdge(id))?;
                let old = std::mem::replace(&mut e.semantic, semantic);
                Ok(Edit::SetEdgeSemantic {
                    piece,
                    id,
                    semantic: old,
                })
            }
            Edit::SetEdgeEnds {
                piece,
                id,
                start,
                end,
            } => {
                let p = self.piece_mut(piece)?;
                p.network.pos(start)?;
                p.network.pos(end)?;
                let e = p
                    .network
                    .edges
                    .get_mut(id)
                    .ok_or(nguo_pattern::NetworkError::NoSuchEdge(id))?;
                let old = (e.start, e.end);
                e.start = start;
                e.end = end;
                Ok(Edit::SetEdgeEnds {
                    piece,
                    id,
                    start: old.0,
                    end: old.1,
                })
            }

            // --- features -------------------------------------------------
            Edit::AddFeature { piece, id, feature } => {
                let p = self.piece_mut(piece)?;
                if p.features.contains(id) {
                    return Err(DocError::IdInUse(format!("feature {id:?}")));
                }
                p.features.insert_at(id, *feature);
                Ok(Edit::RemoveFeature { piece, id })
            }
            Edit::RemoveFeature { piece, id } => {
                let p = self.piece_mut(piece)?;
                let feature = p.features.remove(id).ok_or(DocError::NoSuchFeature(id))?;
                Ok(Edit::AddFeature {
                    piece,
                    id,
                    feature: Box::new(feature),
                })
            }
            Edit::SetFeature { piece, id, feature } => {
                let p = self.piece_mut(piece)?;
                let slot = p.features.get_mut(id).ok_or(DocError::NoSuchFeature(id))?;
                let old = std::mem::replace(slot, *feature);
                Ok(Edit::SetFeature {
                    piece,
                    id,
                    feature: Box::new(old),
                })
            }

            // --- seams ----------------------------------------------------
            Edit::AddSeam { id, seam } => {
                if self.seams.contains(id) {
                    return Err(DocError::IdInUse(format!("seam {id:?}")));
                }
                self.check_seam(&seam)?;
                self.seams.insert_at(id, *seam);
                Ok(Edit::RemoveSeam { id })
            }
            Edit::RemoveSeam { id } => {
                let seam = self.seams.remove(id).ok_or(DocError::NoSuchSeam(id))?;
                Ok(Edit::AddSeam {
                    id,
                    seam: Box::new(seam),
                })
            }
            Edit::SetSeam { id, seam } => {
                if !self.seams.contains(id) {
                    return Err(DocError::NoSuchSeam(id));
                }
                self.check_seam(&seam)?;
                let old = self.seams.insert_at(id, *seam).expect("checked above");
                Ok(Edit::SetSeam {
                    id,
                    seam: Box::new(old),
                })
            }

            // --- materials ------------------------------------------------
            Edit::AddMaterial { id, material } => {
                if self.materials.contains(id) {
                    return Err(DocError::IdInUse(format!("material {id:?}")));
                }
                self.materials.insert_at(id, *material);
                Ok(Edit::RemoveMaterial { id })
            }
            Edit::RemoveMaterial { id } => {
                if let Some((pid, _)) = self.pieces.iter().find(|(_, p)| p.material == Some(id)) {
                    return Err(DocError::StillReferenced {
                        what: format!("material {id:?}"),
                        by: format!("piece {pid:?}"),
                    });
                }
                let material = self
                    .materials
                    .remove(id)
                    .ok_or(DocError::NoSuchMaterial(id))?;
                Ok(Edit::AddMaterial {
                    id,
                    material: Box::new(material),
                })
            }
            Edit::SetMaterial { id, material } => {
                if !self.materials.contains(id) {
                    return Err(DocError::NoSuchMaterial(id));
                }
                let old = self.materials.insert_at(id, *material).expect("checked");
                Ok(Edit::SetMaterial {
                    id,
                    material: Box::new(old),
                })
            }

            // --- everything else ------------------------------------------
            Edit::SetParam { name, param } => {
                let old = match param {
                    Some(p) => self.params.insert(name.clone(), p),
                    None => self.params.remove(&name),
                };
                Ok(Edit::SetParam { name, param: old })
            }
            Edit::SetMetadata { meta } => {
                let old = std::mem::replace(&mut self.meta, *meta);
                Ok(Edit::SetMetadata {
                    meta: Box::new(old),
                })
            }
            Edit::SetAvatar { avatar } => {
                let old = std::mem::replace(&mut self.avatar, avatar.map(|a| *a));
                Ok(Edit::SetAvatar {
                    avatar: old.map(Box::new),
                })
            }
            Edit::SetCamera { camera } => {
                let old = std::mem::replace(&mut self.camera, camera);
                Ok(Edit::SetCamera { camera: old })
            }
        }
    }

    fn piece_mut(&mut self, id: PieceId) -> Result<&mut Piece, DocError> {
        self.pieces.get_mut(id).ok_or(DocError::NoSuchPiece(id))
    }

    /// A seam may only reference live pieces and live edges of those pieces.
    fn check_seam(&self, seam: &Seam) -> Result<(), DocError> {
        for side in [&seam.a, &seam.b] {
            let p = self.piece(side.piece)?;
            for &(e, _) in &side.edges {
                if !p.network.edges.contains(e) {
                    return Err(nguo_pattern::NetworkError::NoSuchEdge(e).into());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SeamSide;
    use nguo_pattern::EdgeSemantic;

    fn doc_with_square() -> (Document, PieceId, Vec<EdgeId>) {
        let mut d = Document::new();
        let pid = d.next_piece_id();
        let mut piece = Piece::new("Front");
        let v: Vec<VertexId> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            .iter()
            .map(|&(x, y)| piece.network.add_vertex(Point::new(x, y)))
            .collect();
        let e: Vec<EdgeId> = (0..4)
            .map(|i| piece.network.add_line(v[i], v[(i + 1) % 4]).unwrap())
            .collect();
        d.apply_edit(Edit::AddPiece {
            id: pid,
            piece: Box::new(piece),
        })
        .unwrap();
        (d, pid, e)
    }

    #[test]
    fn every_edit_is_exactly_invertible() {
        let (mut d, pid, e) = doc_with_square();
        let vid = d.piece(pid).unwrap().network.vertices.next_id();
        // Kept applied: the later geometry edits need something to act on.
        d.apply_edit(Edit::AddVertex {
            piece: pid,
            id: vid,
            pos: Point::new(50.0, 50.0),
        })
        .unwrap();
        let edits = vec![
            Edit::SetPieceName {
                id: pid,
                name: "Back".into(),
            },
            Edit::SetPieceAllowance { id: pid, mm: 15.0 },
            Edit::RemoveVertex {
                piece: pid,
                id: vid,
            },
            Edit::MoveVertex {
                piece: pid,
                id: vid,
                pos: Point::new(51.0, 49.0),
            },
            Edit::SetEdgeSemantic {
                piece: pid,
                id: e[0],
                semantic: EdgeSemantic::Fold,
            },
            Edit::SetParam {
                name: "waist".into(),
                param: Some(Param::new("80cm")),
            }
            .clone(),
            Edit::SetCamera {
                camera: Camera {
                    fov_y_rad: 0.9,
                    ..Camera::default()
                },
            },
        ];
        for edit in edits {
            let before = d.to_json().unwrap();
            let inv = d.apply_edit(edit.clone()).unwrap();
            assert_ne!(
                d.to_json().unwrap(),
                before,
                "{edit:?} did not change anything"
            );
            d.apply_edit(inv).unwrap();
            assert_eq!(d.to_json().unwrap(), before, "{edit:?} did not invert");
        }
    }

    #[test]
    fn removing_a_piece_restores_it_byte_for_byte() {
        let (mut d, pid, _) = doc_with_square();
        let before = d.to_json().unwrap();
        let inv = d.apply_edit(Edit::RemovePiece { id: pid }).unwrap();
        assert!(d.pieces().is_empty());
        d.apply_edit(inv).unwrap();
        assert_eq!(d.to_json().unwrap(), before);
    }

    #[test]
    fn referential_integrity_is_enforced() {
        let (mut d, pid, e) = doc_with_square();
        let sid = d.next_seam_id();
        d.apply_edit(Edit::AddSeam {
            id: sid,
            seam: Box::new(Seam::new(
                SeamSide::whole(pid, vec![(e[0], false)]),
                SeamSide::whole(pid, vec![(e[2], true)]),
            )),
        })
        .unwrap();

        assert!(matches!(
            d.apply_edit(Edit::RemoveEdge {
                piece: pid,
                id: e[0]
            }),
            Err(DocError::StillReferenced { .. })
        ));
        assert!(matches!(
            d.apply_edit(Edit::RemovePiece { id: pid }),
            Err(DocError::StillReferenced { .. })
        ));
        // Removing an unreferenced edge is fine.
        d.apply_edit(Edit::RemoveEdge {
            piece: pid,
            id: e[1],
        })
        .unwrap();
    }

    #[test]
    fn a_seam_cannot_reference_a_dead_edge() {
        let (mut d, pid, _) = doc_with_square();
        let sid = d.next_seam_id();
        let ghost = nguo_pattern::Id::from_raw(999);
        assert!(d
            .apply_edit(Edit::AddSeam {
                id: sid,
                seam: Box::new(Seam::new(
                    SeamSide::whole(pid, vec![(ghost, false)]),
                    SeamSide::whole(pid, vec![(ghost, false)]),
                )),
            })
            .is_err());
    }

    #[test]
    fn a_failed_edit_leaves_the_document_untouched() {
        let (mut d, pid, _) = doc_with_square();
        let before = d.to_json().unwrap();
        let rev = d.revision();
        assert!(d
            .apply_edit(Edit::SetPieceMaterial {
                id: pid,
                material: Some(nguo_pattern::Id::from_raw(7)),
            })
            .is_err());
        assert_eq!(d.to_json().unwrap(), before);
        assert_eq!(
            d.revision(),
            rev,
            "a failed edit must not bump the revision"
        );
    }

    #[test]
    fn revision_advances_on_every_applied_edit() {
        let (mut d, pid, _) = doc_with_square();
        let r = d.revision();
        let inv = d
            .apply_edit(Edit::SetPieceName {
                id: pid,
                name: "x".into(),
            })
            .unwrap();
        assert_eq!(d.revision(), r + 1);
        d.apply_edit(inv).unwrap();
        assert_eq!(d.revision(), r + 2, "undo is an edit too");
    }
}
