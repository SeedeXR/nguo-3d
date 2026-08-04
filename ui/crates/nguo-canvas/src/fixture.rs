// SPDX-License-Identifier: GPL-2.0-or-later
//! Document fixtures for tests and profiling.
//!
//! Public on purpose: the unit tests in this crate, the integration tests in
//! `tests/`, and the perf harness all need the same shapes, and a fixture
//! duplicated three ways is a fixture that drifts. Nothing in the shipped code
//! path calls these.

use nguo_doc::{Document, Edit, Feature, Piece, PieceId};
use nguo_doc_commands::CommandStack;
use nguo_pattern::{EdgeGeom, EdgeId, EdgeSemantic, VertexId};
use vello::kurbo::{Affine, Point, Vec2};

/// A 100 mm square of `Seam` edges, four straight lines.
pub fn square_piece() -> Piece {
    let mut piece = Piece::new("Square");
    let v: Vec<VertexId> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        .iter()
        .map(|&(x, y)| piece.network.add_vertex(Point::new(x, y)))
        .collect();
    for i in 0..4 {
        piece
            .network
            .add_line(v[i], v[(i + 1) % 4])
            .expect("fixture vertices exist");
    }
    piece
}

/// A document holding one [`square_piece`], plus its edge ids in order.
pub fn square_doc() -> (Document, PieceId, Vec<EdgeId>) {
    let mut doc = Document::new();
    let mut stack = CommandStack::new();
    let id = doc.next_piece_id();
    let piece = square_piece();
    let edges: Vec<EdgeId> = piece.network.edges.ids().collect();
    stack
        .apply(
            &mut doc,
            Edit::AddPiece {
                id,
                piece: Box::new(piece),
            },
        )
        .expect("adding a piece to an empty document");
    (doc, id, edges)
}

/// A piece of realistic drafting complexity: a closed curved outline of
/// `segments` cubic edges, a grain line and a notch.
///
/// `segments` is what makes this useful for profiling — a real bodice front is
/// 20-40 boundary edges once darts and notches have split it.
pub fn curved_piece(name: &str, segments: usize, radius: f64) -> Piece {
    let segments = segments.max(3);
    let mut piece = Piece::new(name);

    let v: Vec<VertexId> = (0..segments)
        .map(|i| {
            let a = i as f64 / segments as f64 * std::f64::consts::TAU;
            // Not a circle: a lobed outline, so the curvature varies the way a
            // pattern outline does and the offset/flatten work is realistic.
            let r = radius * (1.0 + 0.18 * (a * 3.0).sin());
            piece
                .network
                .add_vertex(Point::new(r * a.cos(), r * a.sin()))
        })
        .collect();

    for i in 0..segments {
        let (a, b) = (v[i], v[(i + 1) % segments]);
        let pa = piece.network.pos(a).expect("fixture vertex");
        let pb = piece.network.pos(b).expect("fixture vertex");
        let d = (pb - pa) / 3.0;
        let bulge = Vec2::new(-d.y, d.x) * 0.25;
        piece
            .network
            .add_edge(
                a,
                b,
                EdgeGeom::Cubic {
                    c1: pa + d + bulge,
                    c2: pb - d + bulge,
                },
                EdgeSemantic::Seam,
            )
            .expect("fixture vertices exist");
    }

    let first = piece.network.edges.ids().next().expect("at least one edge");
    piece.features.insert(Feature::GrainLine {
        from: Point::new(0.0, -radius * 0.8),
        to: Point::new(0.0, radius * 0.8),
    });
    piece.features.insert(Feature::Notch {
        edge: first,
        t: 0.5,
        kind: nguo_doc::NotchKind::V,
    });
    piece
}

/// `count` curved pieces laid out on a marker, 4 per row.
pub fn marker(count: usize, segments: usize) -> Document {
    let mut doc = Document::new();
    let mut stack = CommandStack::new();
    for i in 0..count {
        let id = doc.next_piece_id();
        let mut piece = curved_piece(&format!("Piece {i}"), segments, 150.0);
        piece.transform = Affine::translate(((i % 4) as f64 * 400.0, (i / 4) as f64 * 400.0));
        stack
            .apply(
                &mut doc,
                Edit::AddPiece {
                    id,
                    piece: Box::new(piece),
                },
            )
            .expect("adding a piece to a document");
    }
    doc
}
