// SPDX-License-Identifier: GPL-2.0-or-later
//! Composite operations: multi-edit transactions that read the document to
//! work out what edits to emit.
//!
//! These are pure planners — they take `&Document` and return a
//! [`Transaction`]. Nothing here mutates; only [`crate::CommandStack::apply`]
//! does.

use kurbo::{ParamCurve, ParamCurveArclen, Point};
use nguo_doc::{DocError, Document, Edit, Feature, PieceId};
use nguo_pattern::{Edge, EdgeGeom, EdgeId, EdgeSemantic};

use crate::Transaction;

/// Split `edge` at curve parameter `t`, as one undo entry.
///
/// The original edge keeps its id and becomes the `[0, t]` half, so seams and
/// features pointing at it stay valid; a new edge covers `[t, 1]`.
///
/// This is the command-level counterpart of
/// [`nguo_pattern::VectorNetwork::split_edge`], which mutates a network
/// directly and therefore cannot be used on a document.
pub fn split_edge(
    doc: &Document,
    piece: PieceId,
    edge: EdgeId,
    t: f64,
) -> Result<Transaction, DocError> {
    let p = doc.piece(piece)?;
    if !(t > 0.0 && t < 1.0) {
        return Err(nguo_pattern::NetworkError::ParamOutOfRange(t).into());
    }
    let seg = p.network.seg(edge)?;
    let old = *p
        .network
        .edges
        .get(edge)
        .ok_or(nguo_pattern::NetworkError::NoSuchEdge(edge))?;

    let new_vertex = p.network.vertices.next_id();
    let new_edge = p.network.edges.next_id();
    let (first, second) = (seg.subsegment(0.0..t), seg.subsegment(t..1.0));

    Ok(Transaction::new("Split edge")
        .edit(Edit::AddVertex {
            piece,
            id: new_vertex,
            pos: seg.eval(t),
        })
        .edit(Edit::SetEdgeEnds {
            piece,
            id: edge,
            start: old.start,
            end: new_vertex,
        })
        .edit(Edit::SetEdgeGeom {
            piece,
            id: edge,
            geom: EdgeGeom::of_seg(first),
        })
        .edit(Edit::AddEdge {
            piece,
            id: new_edge,
            edge: Edge {
                start: new_vertex,
                end: old.end,
                geom: EdgeGeom::of_seg(second),
                semantic: old.semantic,
            },
        }))
}

/// Add a dart on `edge`: split the edge twice to open the dart mouth, add the
/// apex, and connect two dart legs — the guide's own example of why
/// transactions exist (§5.2). One undo entry.
///
/// `t` is where the dart's centre lands on the edge and `width` is the mouth
/// opening in mm, split evenly either side. `apex` is in piece-local mm.
///
/// ponytail: the mouth is opened by splitting at two parameters offset from `t`
/// by half the width measured along the *chord*, which is exact for a straight
/// edge and within a fraction of a millimetre on a normal waist curve. Solve
/// for arc length instead if a user ever complains about a dart on a tight
/// curve.
pub fn add_dart(
    doc: &Document,
    piece: PieceId,
    edge: EdgeId,
    t: f64,
    width: f64,
    apex: Point,
) -> Result<Transaction, DocError> {
    let p = doc.piece(piece)?;
    if width <= 0.0 {
        return Err(nguo_pattern::NetworkError::ParamOutOfRange(width).into());
    }
    let seg = p.network.seg(edge)?;
    let len = seg.arclen(nguo_pattern::GEOM_ACCURACY);
    if len <= width {
        return Err(nguo_pattern::NetworkError::ParamOutOfRange(width).into());
    }
    let half = 0.5 * width / len;
    let (t0, t1) = (t - half, t + half);
    if !(t0 > 0.0 && t1 < 1.0) {
        return Err(nguo_pattern::NetworkError::ParamOutOfRange(t).into());
    }

    // Split at the far side first: splitting at `t0` leaves the second half as
    // a *new* edge, and we would have to re-parametrise to find `t1` on it.
    let net = &p.network;
    let v_right = net.vertices.next_id();
    let e_right = net.edges.next_id();
    let v_left = nguo_pattern::Id::from_raw(v_right.raw() + 1);
    let e_left = nguo_pattern::Id::from_raw(e_right.raw() + 1);
    let v_apex = nguo_pattern::Id::from_raw(v_right.raw() + 2);
    let leg_a = nguo_pattern::Id::from_raw(e_right.raw() + 2);
    let leg_b = nguo_pattern::Id::from_raw(e_right.raw() + 3);
    let feature = p.features.next_id();

    let old = *net
        .edges
        .get(edge)
        .ok_or(nguo_pattern::NetworkError::NoSuchEdge(edge))?;

    // [0, t1] stays on `edge`; [t1, 1] becomes `e_right`.
    let head = seg.subsegment(0.0..t1);
    let tail = seg.subsegment(t1..1.0);
    // Split the head again at the left parameter, rescaled into the head.
    let t0_in_head = t0 / t1;
    let (head_a, head_b) = (
        head.subsegment(0.0..t0_in_head),
        head.subsegment(t0_in_head..1.0),
    );

    let mut tx = Transaction::new("Add dart");
    // Right split.
    tx.push(Edit::AddVertex {
        piece,
        id: v_right,
        pos: seg.eval(t1),
    });
    tx.push(Edit::AddEdge {
        piece,
        id: e_right,
        edge: Edge {
            start: v_right,
            end: old.end,
            geom: EdgeGeom::of_seg(tail),
            semantic: old.semantic,
        },
    });
    // Left split, carving `edge` down to [0, t0].
    tx.push(Edit::AddVertex {
        piece,
        id: v_left,
        pos: seg.eval(t0),
    });
    tx.push(Edit::AddEdge {
        piece,
        id: e_left,
        edge: Edge {
            start: v_left,
            end: v_right,
            geom: EdgeGeom::of_seg(head_b),
            semantic: old.semantic,
        },
    });
    tx.push(Edit::SetEdgeEnds {
        piece,
        id: edge,
        start: old.start,
        end: v_left,
    });
    tx.push(Edit::SetEdgeGeom {
        piece,
        id: edge,
        geom: EdgeGeom::of_seg(head_a),
    });
    // The dart itself: an apex and two legs. The mouth edge between the legs is
    // the piece's boundary and stays there; it is the legs that get sewn.
    tx.push(Edit::AddVertex {
        piece,
        id: v_apex,
        pos: apex,
    });
    for (id, from) in [(leg_a, v_left), (leg_b, v_right)] {
        tx.push(Edit::AddEdge {
            piece,
            id,
            edge: Edge {
                start: from,
                end: v_apex,
                geom: EdgeGeom::Line,
                semantic: EdgeSemantic::DartLeg,
            },
        });
    }
    tx.push(Edit::AddFeature {
        piece,
        id: feature,
        feature: Box::new(Feature::Dart {
            apex: v_apex,
            legs: [leg_a, leg_b],
        }),
    });
    Ok(tx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommandStack;
    use nguo_doc::Piece;
    use nguo_pattern::VertexId;

    fn square() -> (Document, CommandStack, PieceId, Vec<EdgeId>) {
        let mut doc = Document::new();
        let mut stack = CommandStack::new();
        let pid = doc.next_piece_id();
        let mut piece = Piece::new("Front");
        let v: Vec<VertexId> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            .iter()
            .map(|&(x, y)| piece.network.add_vertex(Point::new(x, y)))
            .collect();
        let e: Vec<EdgeId> = (0..4)
            .map(|i| piece.network.add_line(v[i], v[(i + 1) % 4]).unwrap())
            .collect();
        stack
            .apply(
                &mut doc,
                Edit::AddPiece {
                    id: pid,
                    piece: Box::new(piece),
                },
            )
            .unwrap();
        (doc, stack, pid, e)
    }

    #[test]
    fn split_edge_matches_the_network_operation() {
        let (mut doc, mut stack, pid, e) = square();
        let tx = split_edge(&doc, pid, e[0], 0.25).unwrap();
        stack.apply(&mut doc, tx).unwrap();

        let net = &doc.piece(pid).unwrap().network;
        assert_eq!(net.edges.len(), 5);
        assert_eq!(net.vertices.len(), 5);
        assert!((net.arc_length(e[0]).unwrap() - 25.0).abs() < 1e-9);
        // The two halves still add up.
        let new_edge = net.edges.ids().next_back().unwrap();
        assert!((net.arc_length(new_edge).unwrap() - 75.0).abs() < 1e-9);
    }

    #[test]
    fn split_edge_is_one_undo_entry() {
        let (mut doc, mut stack, pid, e) = square();
        let before = doc.to_json().unwrap();
        let depth = stack.undo_depth();
        let tx = split_edge(&doc, pid, e[1], 0.5).unwrap();
        stack.apply(&mut doc, tx).unwrap();
        assert_eq!(stack.undo_depth(), depth + 1);
        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.to_json().unwrap(), before);
    }

    #[test]
    fn split_rejects_endpoints_and_dead_edges() {
        let (doc, _, pid, e) = square();
        assert!(split_edge(&doc, pid, e[0], 0.0).is_err());
        assert!(split_edge(&doc, pid, e[0], 1.0).is_err());
        assert!(split_edge(&doc, pid, nguo_pattern::Id::from_raw(42), 0.5).is_err());
    }

    #[test]
    fn add_dart_is_one_undo_entry_and_builds_the_whole_dart() {
        let (mut doc, mut stack, pid, e) = square();
        let before = doc.to_json().unwrap();
        let depth = stack.undo_depth();

        let tx = add_dart(&doc, pid, e[0], 0.5, 20.0, Point::new(50.0, 60.0)).unwrap();
        assert!(tx.len() > 5, "a dart is genuinely multi-step");
        stack.apply(&mut doc, tx).unwrap();
        assert_eq!(stack.undo_depth(), depth + 1);

        let p = doc.piece(pid).unwrap();
        assert_eq!(p.features.len(), 1);
        let legs: Vec<_> = p
            .network
            .edges
            .iter()
            .filter(|(_, x)| x.semantic == EdgeSemantic::DartLeg)
            .collect();
        assert_eq!(legs.len(), 2);
        // The mouth between the two splits is the dart width.
        let total: f64 = p
            .network
            .edges
            .iter()
            .filter(|(_, x)| x.semantic != EdgeSemantic::DartLeg)
            .map(|(id, _)| p.network.arc_length(id).unwrap())
            .sum();
        assert!((total - 400.0).abs() < 1e-6, "perimeter changed: {total}");

        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.to_json().unwrap(), before, "one undo removes the dart");
    }

    #[test]
    fn add_dart_refuses_impossible_geometry() {
        let (doc, _, pid, e) = square();
        assert!(add_dart(&doc, pid, e[0], 0.5, 0.0, Point::ORIGIN).is_err());
        assert!(add_dart(&doc, pid, e[0], 0.5, 500.0, Point::ORIGIN).is_err());
        assert!(add_dart(&doc, pid, e[0], 0.01, 20.0, Point::ORIGIN).is_err());
    }

    #[test]
    fn dart_mouth_is_the_requested_width() {
        let (mut doc, mut stack, pid, e) = square();
        let tx = add_dart(&doc, pid, e[0], 0.5, 24.0, Point::new(50.0, 70.0)).unwrap();
        stack.apply(&mut doc, tx).unwrap();
        let p = doc.piece(pid).unwrap();
        let Feature::Dart { legs, .. } = p.features.iter().next().unwrap().1 else {
            panic!("expected a dart")
        };
        let a = p
            .network
            .pos(p.network.edges.get(legs[0]).unwrap().start)
            .unwrap();
        let b = p
            .network
            .pos(p.network.edges.get(legs[1]).unwrap().start)
            .unwrap();
        assert!(((b - a).hypot() - 24.0).abs() < 1e-6, "{a:?} {b:?}");
    }
}
