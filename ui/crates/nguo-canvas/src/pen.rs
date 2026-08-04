// SPDX-License-Identifier: GPL-2.0-or-later
//! The Bézier pen tool and point/handle editing (guide §6.1).
//!
//! # It never mutates the document
//! Every method here returns a [`Transaction`] for the caller to hand to a
//! `CommandStack`. That is not a convention: `Document`'s fields are private to
//! `nguo-doc` and its only `&mut self` method takes an `Edit`, so this module
//! *cannot* mutate even by accident. Undo therefore covers the pen for free.
//!
//! # The state machine
//! ```text
//!   Idle ──click──▶ Drawing{first,last} ──click──▶ Drawing{first,last'}
//!                        │  │
//!                        │  └─drag──▶ curves the segment just placed
//!                        ├─close───▶ Idle (edge back to `first`)
//!                        └─cancel──▶ Idle (no transaction; the placed points
//!                                    stay, undo removes them)
//! ```

use nguo_doc::{DocError, Document, Edit, Piece, PieceId};
use nguo_doc_commands::Transaction;
use nguo_pattern::{Edge, EdgeGeom, EdgeId, EdgeSemantic, VertexId};
use vello::kurbo::Point;

/// Where the pen is in its gesture. Public so the renderer can draw the
/// rubber-band and the app can show the right cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenState {
    Idle,
    Drawing {
        piece: PieceId,
        /// The first vertex of the run, so `close` knows where to come back to.
        first: VertexId,
        /// The vertex the next segment starts from.
        last: VertexId,
        /// The edge just placed, if any — what a drag curves.
        last_edge: Option<EdgeId>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pen {
    pub state: PenState,
    /// What new edges mean. A pen run draws piece outline by default; the app
    /// switches this for the internal-line and dart tools.
    pub semantic: EdgeSemantic,
}

impl Default for Pen {
    fn default() -> Self {
        Pen {
            state: PenState::Idle,
            semantic: EdgeSemantic::Seam,
        }
    }
}

impl Pen {
    pub fn new() -> Pen {
        Pen::default()
    }

    pub fn is_drawing(&self) -> bool {
        matches!(self.state, PenState::Drawing { .. })
    }

    /// The vertex the next segment will start from.
    pub fn anchor(&self, doc: &Document) -> Option<Point> {
        let PenState::Drawing { piece, last, .. } = self.state else {
            return None;
        };
        doc.piece(piece).ok()?.network.pos(last).ok()
    }

    /// Places a point at `world` (mm).
    ///
    /// `into` is the piece to draw on; `None` starts a new piece, which is what
    /// the first click on an empty document does. One transaction, so one undo
    /// step per click.
    pub fn click(
        &mut self,
        doc: &Document,
        into: Option<PieceId>,
        world: Point,
    ) -> Result<Transaction, DocError> {
        match self.state {
            PenState::Idle => {
                // Ids always come from the arena that will hold them — the
                // document's own watermark for an existing piece, the fresh
                // piece's for a new one. Never invented, never a literal.
                let (mut tx, piece, vertex) = match into {
                    Some(piece) => {
                        // Fail now if the target does not exist, rather than
                        // emitting edits that cannot apply.
                        let vertex = doc.piece(piece)?.network.vertices.next_id();
                        (Transaction::new("Draw point"), piece, vertex)
                    }
                    None => {
                        let piece = doc.next_piece_id();
                        let fresh = Piece::new("Piece");
                        let vertex = fresh.network.vertices.next_id();
                        (
                            Transaction::new("Draw piece").edit(Edit::AddPiece {
                                id: piece,
                                piece: Box::new(fresh),
                            }),
                            piece,
                            vertex,
                        )
                    }
                };
                tx.push(Edit::AddVertex {
                    piece,
                    id: vertex,
                    pos: world,
                });
                self.state = PenState::Drawing {
                    piece,
                    first: vertex,
                    last: vertex,
                    last_edge: None,
                };
                Ok(tx)
            }
            PenState::Drawing {
                piece, first, last, ..
            } => {
                let net = &doc.piece(piece)?.network;
                let vertex = net.vertices.next_id();
                let edge = net.edges.next_id();
                self.state = PenState::Drawing {
                    piece,
                    first,
                    last: vertex,
                    last_edge: Some(edge),
                };
                Ok(Transaction::new("Draw segment")
                    .edit(Edit::AddVertex {
                        piece,
                        id: vertex,
                        pos: world,
                    })
                    .edit(Edit::AddEdge {
                        piece,
                        id: edge,
                        edge: Edge {
                            start: last,
                            end: vertex,
                            geom: EdgeGeom::Line,
                            semantic: self.semantic,
                        },
                    }))
            }
        }
    }

    /// Dragging after a click curves the segment just placed: the drag position
    /// is the outgoing handle, and the incoming control point is its mirror
    /// through the point — which is what makes a pen draw smooth curves.
    ///
    /// Returns `None` when there is no placed segment to curve (the very first
    /// click of a run).
    pub fn drag_handle(&self, doc: &Document, world: Point) -> Option<Transaction> {
        let PenState::Drawing {
            piece,
            last,
            last_edge: Some(edge),
            ..
        } = self.state
        else {
            return None;
        };
        let net = &doc.piece(piece).ok()?.network;
        let e = net.edges.get(edge)?;
        let start = net.pos(e.start).ok()?;
        let end = net.pos(last).ok()?;
        let out = world - end;

        Some(Transaction::new("Curve segment").edit(Edit::SetEdgeGeom {
            piece,
            id: edge,
            geom: EdgeGeom::Cubic {
                // The start handle keeps its default third; only the end
                // one is being dragged.
                c1: start + (end - start) / 3.0,
                c2: end - out,
            },
        }))
    }

    /// Closes the run: one edge from the current point back to the first, and
    /// back to `Idle`. `None` if there is nothing to close (fewer than three
    /// points, or the run is already closed).
    pub fn close(&mut self, doc: &Document) -> Option<Transaction> {
        let PenState::Drawing {
            piece, first, last, ..
        } = self.state
        else {
            return None;
        };
        if first == last {
            return None;
        }
        let net = &doc.piece(piece).ok()?.network;
        // Already joined? Closing twice would leave a duplicate edge.
        if net
            .edges_at(first)
            .any(|(_, e)| e.start == last || e.end == last)
        {
            self.state = PenState::Idle;
            return None;
        }
        let edge = net.edges.next_id();
        self.state = PenState::Idle;
        Some(Transaction::new("Close path").edit(Edit::AddEdge {
            piece,
            id: edge,
            edge: Edge {
                start: last,
                end: first,
                geom: EdgeGeom::Line,
                semantic: self.semantic,
            },
        }))
    }

    /// Ends the run, leaving it open. Emits nothing: the points are already in
    /// the document, and undo is what removes them.
    pub fn finish(&mut self) {
        self.state = PenState::Idle;
    }
}

/// Moves an existing point. Tag the transaction with a [`GestureId`] to make a
/// whole drag one undo entry (`nguo-doc-commands` §5.2).
///
/// [`GestureId`]: nguo_doc_commands::GestureId
pub fn move_point(
    doc: &Document,
    piece: PieceId,
    vertex: VertexId,
    to: Point,
) -> Result<Transaction, DocError> {
    // Read first, so a bad id is an error and not a no-op edit in the history.
    doc.piece(piece)?.network.pos(vertex)?;
    Ok(Transaction::new("Move point").edit(Edit::MoveVertex {
        piece,
        id: vertex,
        pos: to,
    }))
}

/// Moves one Bézier handle of an edge, promoting a straight edge to a curve.
///
/// A `Line` has no stored control points, so the untouched side is materialised
/// at its implicit third — the position where the curve is still exactly the
/// straight line it replaced.
pub fn move_handle(
    doc: &Document,
    piece: PieceId,
    edge: EdgeId,
    end: crate::hit::HandleEnd,
    to: Point,
) -> Result<Transaction, DocError> {
    let net = &doc.piece(piece)?.network;
    let e = *net
        .edges
        .get(edge)
        .ok_or(nguo_pattern::NetworkError::NoSuchEdge(edge))?;
    let p0 = net.pos(e.start)?;
    let p3 = net.pos(e.end)?;
    let (c1, c2) = match e.geom {
        EdgeGeom::Cubic { c1, c2 } => (c1, c2),
        EdgeGeom::Line => (p0 + (p3 - p0) / 3.0, p3 - (p3 - p0) / 3.0),
    };
    let geom = match end {
        crate::hit::HandleEnd::Start => EdgeGeom::Cubic { c1: to, c2 },
        crate::hit::HandleEnd::End => EdgeGeom::Cubic { c1, c2: to },
    };
    Ok(Transaction::new("Move handle").edit(Edit::SetEdgeGeom {
        piece,
        id: edge,
        geom,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::square_doc;
    use crate::hit::HandleEnd;
    use nguo_doc_commands::{CommandStack, GestureId};
    use vello::kurbo::ParamCurve;

    fn drawn(points: &[(f64, f64)]) -> (Document, CommandStack, Pen, PieceId) {
        let mut doc = Document::new();
        let mut stack = CommandStack::new();
        let mut pen = Pen::new();
        let mut piece = None;
        for &(x, y) in points {
            let tx = pen.click(&doc, piece, Point::new(x, y)).unwrap();
            stack.apply(&mut doc, tx).unwrap();
            if piece.is_none() {
                piece = Some(match pen.state {
                    PenState::Drawing { piece, .. } => piece,
                    PenState::Idle => unreachable!("a click leaves the pen drawing"),
                });
            }
        }
        let piece = piece.expect("at least one point");
        (doc, stack, pen, piece)
    }

    #[test]
    fn the_first_click_on_an_empty_document_creates_a_piece_and_a_point() {
        let (doc, stack, pen, piece) = drawn(&[(10.0, 20.0)]);
        assert_eq!(doc.pieces().len(), 1);
        let net = &doc.piece(piece).unwrap().network;
        assert_eq!(net.vertices.len(), 1);
        assert_eq!(net.edges.len(), 0);
        assert!(pen.is_drawing());
        assert_eq!(stack.undo_depth(), 1, "one click, one undo step");
        assert_eq!(pen.anchor(&doc), Some(Point::new(10.0, 20.0)));
    }

    #[test]
    fn each_further_click_adds_a_point_and_a_segment() {
        let (doc, stack, _, piece) = drawn(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]);
        let net = &doc.piece(piece).unwrap().network;
        assert_eq!(net.vertices.len(), 3);
        assert_eq!(net.edges.len(), 2);
        assert_eq!(stack.undo_depth(), 3, "three clicks, three undo steps");
    }

    #[test]
    fn closing_the_run_joins_back_to_the_first_point_and_ends_the_gesture() {
        let (mut doc, mut stack, mut pen, piece) =
            drawn(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]);
        let tx = pen.close(&doc).expect("three points can close");
        stack.apply(&mut doc, tx).unwrap();

        let net = &doc.piece(piece).unwrap().network;
        assert_eq!(net.edges.len(), 3);
        assert!(!pen.is_drawing());
        // Every vertex has degree 2: that is what "closed" means, and it is what
        // `boundary_loop` needs to find a fill.
        for v in net.vertices.ids() {
            assert_eq!(net.degree(v), 2, "vertex {v:?}");
        }
        assert!(pen.close(&doc).is_none(), "closing twice does nothing");
    }

    #[test]
    fn a_drag_after_a_click_curves_that_segment_only() {
        let (mut doc, mut stack, pen, piece) = drawn(&[(0.0, 0.0), (100.0, 0.0)]);
        let tx = pen
            .drag_handle(&doc, Point::new(140.0, -40.0))
            .expect("a placed segment can be curved");
        stack.apply(&mut doc, tx).unwrap();

        let net = &doc.piece(piece).unwrap().network;
        let edge = net.edges.ids().next().unwrap();
        match net.edges.get(edge).unwrap().geom {
            EdgeGeom::Cubic { c2, .. } => {
                // Mirror of the drag through the point just placed.
                assert_eq!(c2, Point::new(60.0, 40.0));
            }
            EdgeGeom::Line => panic!("the drag did not curve the segment"),
        }
        // The curve still starts and ends at the two clicked points.
        let seg = net.seg(edge).unwrap();
        assert_eq!(seg.eval(0.0), Point::new(0.0, 0.0));
        assert_eq!(seg.eval(1.0), Point::new(100.0, 0.0));
    }

    #[test]
    fn a_drag_with_nothing_placed_yet_emits_nothing() {
        let (doc, _, pen, _) = drawn(&[(0.0, 0.0)]);
        assert!(pen.drag_handle(&doc, Point::new(5.0, 5.0)).is_none());
        let mut idle = Pen::new();
        assert!(idle.drag_handle(&doc, Point::new(5.0, 5.0)).is_none());
        assert!(idle.close(&doc).is_none());
    }

    #[test]
    fn the_pen_draws_into_an_existing_piece_when_told_to() {
        let (mut doc, piece, _) = square_doc();
        let mut stack = CommandStack::new();
        let before = doc.piece(piece).unwrap().network.vertices.len();
        let mut pen = Pen {
            semantic: EdgeSemantic::Internal,
            ..Pen::new()
        };
        for p in [(20.0, 20.0), (80.0, 80.0)] {
            let tx = pen.click(&doc, Some(piece), Point::new(p.0, p.1)).unwrap();
            stack.apply(&mut doc, tx).unwrap();
        }
        let net = &doc.piece(piece).unwrap().network;
        assert_eq!(doc.pieces().len(), 1, "no second piece was created");
        assert_eq!(net.vertices.len(), before + 2);
        let new_edge = net.edges.ids().next_back().unwrap();
        assert_eq!(
            net.edges.get(new_edge).unwrap().semantic,
            EdgeSemantic::Internal,
            "the pen's semantic reaches the edge"
        );
    }

    #[test]
    fn drawing_into_a_missing_piece_is_an_error_not_a_panic() {
        let doc = Document::new();
        let mut pen = Pen::new();
        assert!(pen
            .click(&doc, Some(nguo_pattern::Id::from_raw(9)), Point::ORIGIN)
            .is_err());
        assert!(!pen.is_drawing(), "a failed click does not start a run");
    }

    #[test]
    fn a_drag_of_a_point_coalesces_into_one_undo_step() {
        let (mut doc, piece, _) = square_doc();
        let mut stack = CommandStack::new();
        let v = doc
            .piece(piece)
            .unwrap()
            .network
            .vertices
            .ids()
            .next()
            .unwrap();
        let depth = stack.undo_depth();
        let g = GestureId(1);
        for i in 1..=10 {
            let tx = move_point(&doc, piece, v, Point::new(i as f64, i as f64))
                .unwrap()
                .with_gesture(g);
            stack.apply(&mut doc, tx).unwrap();
        }
        assert_eq!(
            stack.undo_depth(),
            depth + 1,
            "a tagged drag is one undo entry"
        );
        assert_eq!(
            doc.piece(piece).unwrap().network.pos(v).unwrap(),
            Point::new(10.0, 10.0)
        );
        stack.undo(&mut doc).unwrap();
        assert_eq!(
            doc.piece(piece).unwrap().network.pos(v).unwrap(),
            Point::new(0.0, 0.0),
            "one undo returns to where the drag started"
        );
    }

    #[test]
    fn moving_a_handle_promotes_a_straight_edge_without_changing_its_shape() {
        let (mut doc, piece, edges) = square_doc();
        let mut stack = CommandStack::new();
        let before = doc.piece(piece).unwrap().network.seg(edges[0]).unwrap();

        // Move the start handle to exactly where the promotion puts it: the
        // curve must be the same line it was.
        let implicit = before.eval(1.0 / 3.0);
        let tx = move_handle(&doc, piece, edges[0], HandleEnd::Start, implicit).unwrap();
        stack.apply(&mut doc, tx).unwrap();
        let after = doc.piece(piece).unwrap().network.seg(edges[0]).unwrap();
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            assert!(
                (after.eval(t) - before.eval(t)).hypot() < 1e-9,
                "promotion moved the curve at t={t}"
            );
        }

        // Now move it somewhere else: the shape changes, the endpoints do not.
        let tx = move_handle(
            &doc,
            piece,
            edges[0],
            HandleEnd::End,
            Point::new(70.0, -50.0),
        )
        .unwrap();
        stack.apply(&mut doc, tx).unwrap();
        let curved = doc.piece(piece).unwrap().network.seg(edges[0]).unwrap();
        assert_eq!(curved.eval(0.0), before.eval(0.0));
        assert_eq!(curved.eval(1.0), before.eval(1.0));
        assert!((curved.eval(0.5) - before.eval(0.5)).hypot() > 1.0);
    }

    #[test]
    fn editing_a_missing_point_or_edge_is_an_error() {
        let (doc, piece, _) = square_doc();
        let ghost_v = nguo_pattern::Id::from_raw(99);
        let ghost_e = nguo_pattern::Id::from_raw(99);
        assert!(move_point(&doc, piece, ghost_v, Point::ORIGIN).is_err());
        assert!(move_handle(&doc, piece, ghost_e, HandleEnd::Start, Point::ORIGIN).is_err());
    }
}
