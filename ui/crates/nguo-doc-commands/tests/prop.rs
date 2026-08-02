// SPDX-License-Identifier: GPL-2.0-or-later
//! Property tests for undo/redo and id stability (guide §18).
//!
//! A random script of high-level operations is interpreted against a live
//! document. Operations whose targets do not exist are simply skipped, so the
//! generator stays trivial and every generated script is meaningful.

use kurbo::Point;
use nguo_doc::{Document, Edit, Piece, PieceId};
use nguo_doc_commands::{split_edge, CommandStack, GestureId, Transaction};
use nguo_measure::Param;
use nguo_pattern::{Edge, EdgeGeom, EdgeSemantic};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    AddPiece,
    RemovePiece(usize),
    RenamePiece(usize, u8),
    SetAllowance(usize, f64),
    AddVertex(usize, f64, f64),
    MoveVertex(usize, usize, f64, f64),
    RemoveVertex(usize, usize),
    AddEdge(usize, usize, usize),
    RemoveEdge(usize, usize),
    SplitEdge(usize, usize, f64),
    SetParam(u8, u8),
    ClearParam(u8),
    Drag(usize, usize, Vec<(f64, f64)>),
}

fn op() -> impl Strategy<Value = Op> {
    let ix = 0usize..6;
    let co = -500.0f64..500.0;
    prop_oneof![
        2 => Just(Op::AddPiece),
        1 => ix.clone().prop_map(Op::RemovePiece),
        1 => (ix.clone(), 0u8..8).prop_map(|(a, b)| Op::RenamePiece(a, b)),
        1 => (ix.clone(), 0.0f64..30.0).prop_map(|(a, b)| Op::SetAllowance(a, b)),
        4 => (ix.clone(), co.clone(), co.clone()).prop_map(|(a, x, y)| Op::AddVertex(a, x, y)),
        3 => (ix.clone(), ix.clone(), co.clone(), co.clone())
            .prop_map(|(a, b, x, y)| Op::MoveVertex(a, b, x, y)),
        1 => (ix.clone(), ix.clone()).prop_map(|(a, b)| Op::RemoveVertex(a, b)),
        3 => (ix.clone(), ix.clone(), ix.clone()).prop_map(|(a, b, c)| Op::AddEdge(a, b, c)),
        1 => (ix.clone(), ix.clone()).prop_map(|(a, b)| Op::RemoveEdge(a, b)),
        2 => (ix.clone(), ix.clone(), 0.1f64..0.9).prop_map(|(a, b, t)| Op::SplitEdge(a, b, t)),
        2 => (0u8..5, 0u8..5).prop_map(|(a, b)| Op::SetParam(a, b)),
        1 => (0u8..5).prop_map(Op::ClearParam),
        2 => (
            ix.clone(),
            ix,
            proptest::collection::vec((co.clone(), co), 1..8)
        )
            .prop_map(|(a, b, ps)| Op::Drag(a, b, ps)),
    ]
}

const PARAM_NAMES: [&str; 5] = ["waist", "hip", "ease", "hem", "rise"];
const PARAM_EXPRS: [&str; 5] = [
    "80cm",
    "waist / 4",
    "2cm",
    "waist / 4 + ease",
    "hip - waist",
];

fn nth_piece(doc: &Document, i: usize) -> Option<PieceId> {
    doc.pieces().ids().nth(i % doc.pieces().len().max(1))
}

/// Run one op, returning the transaction it produced (if any). Failures from
/// referential-integrity checks are expected and swallowed: the point is that
/// whatever *does* get applied undoes cleanly.
fn run_op(doc: &mut Document, stack: &mut CommandStack, gesture: &mut u64, op: &Op) {
    let tx: Option<Transaction> = match op {
        Op::AddPiece => Some(
            Edit::AddPiece {
                id: doc.next_piece_id(),
                piece: Box::new(Piece::new("P")),
            }
            .into(),
        ),
        Op::RemovePiece(i) => nth_piece(doc, *i).map(|id| Edit::RemovePiece { id }.into()),
        Op::RenamePiece(i, n) => nth_piece(doc, *i).map(|id| {
            Edit::SetPieceName {
                id,
                name: format!("piece-{n}"),
            }
            .into()
        }),
        Op::SetAllowance(i, mm) => {
            nth_piece(doc, *i).map(|id| Edit::SetPieceAllowance { id, mm: *mm }.into())
        }
        Op::AddVertex(i, x, y) => nth_piece(doc, *i).map(|p| {
            Edit::AddVertex {
                piece: p,
                id: doc.piece(p).unwrap().network.vertices.next_id(),
                pos: Point::new(*x, *y),
            }
            .into()
        }),
        Op::MoveVertex(i, v, x, y) => nth_piece(doc, *i).and_then(|p| {
            let net = &doc.piece(p).unwrap().network;
            let id = net.vertices.ids().nth(v % net.vertices.len().max(1))?;
            Some(
                Edit::MoveVertex {
                    piece: p,
                    id,
                    pos: Point::new(*x, *y),
                }
                .into(),
            )
        }),
        Op::RemoveVertex(i, v) => nth_piece(doc, *i).and_then(|p| {
            let net = &doc.piece(p).unwrap().network;
            let id = net.vertices.ids().nth(v % net.vertices.len().max(1))?;
            Some(Edit::RemoveVertex { piece: p, id }.into())
        }),
        Op::AddEdge(i, a, b) => nth_piece(doc, *i).and_then(|p| {
            let net = &doc.piece(p).unwrap().network;
            let n = net.vertices.len();
            if n < 2 {
                return None;
            }
            let (sa, sb) = (
                net.vertices.ids().nth(a % n)?,
                net.vertices.ids().nth(b % n)?,
            );
            if sa == sb {
                return None;
            }
            Some(
                Edit::AddEdge {
                    piece: p,
                    id: net.edges.next_id(),
                    edge: Edge {
                        start: sa,
                        end: sb,
                        geom: EdgeGeom::Line,
                        semantic: EdgeSemantic::Seam,
                    },
                }
                .into(),
            )
        }),
        Op::RemoveEdge(i, e) => nth_piece(doc, *i).and_then(|p| {
            let net = &doc.piece(p).unwrap().network;
            let id = net.edges.ids().nth(e % net.edges.len().max(1))?;
            Some(Edit::RemoveEdge { piece: p, id }.into())
        }),
        Op::SplitEdge(i, e, t) => nth_piece(doc, *i).and_then(|p| {
            let net = &doc.piece(p).unwrap().network;
            let id = net.edges.ids().nth(e % net.edges.len().max(1))?;
            split_edge(doc, p, id, *t).ok()
        }),
        Op::SetParam(n, x) => Some(
            Edit::SetParam {
                name: PARAM_NAMES[*n as usize % PARAM_NAMES.len()].into(),
                param: Some(Param::new(PARAM_EXPRS[*x as usize % PARAM_EXPRS.len()])),
            }
            .into(),
        ),
        Op::ClearParam(n) => Some(
            Edit::SetParam {
                name: PARAM_NAMES[*n as usize % PARAM_NAMES.len()].into(),
                param: None,
            }
            .into(),
        ),
        Op::Drag(i, v, points) => {
            let Some(p) = nth_piece(doc, *i) else {
                return;
            };
            let net = &doc.piece(p).unwrap().network;
            let Some(id) = net.vertices.ids().nth(v % net.vertices.len().max(1)) else {
                return;
            };
            *gesture += 1;
            let g = GestureId(*gesture);
            for (x, y) in points {
                let _ = stack.apply(
                    doc,
                    Transaction::new("Move point")
                        .with_gesture(g)
                        .edit(Edit::MoveVertex {
                            piece: p,
                            id,
                            pos: Point::new(*x, *y),
                        }),
                );
            }
            None
        }
    };
    if let Some(tx) = tx {
        // Integrity violations are legitimate outcomes here; they just do not
        // get recorded, which is itself part of what we are asserting.
        let _ = stack.apply(doc, tx);
    }
}

/// A document with something in it to edit.
fn seeded() -> (Document, CommandStack) {
    let mut doc = Document::new();
    let mut stack = CommandStack::new();
    let id = doc.next_piece_id();
    let mut piece = Piece::new("Front");
    let v: Vec<_> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        .iter()
        .map(|&(x, y)| piece.network.add_vertex(Point::new(x, y)))
        .collect();
    for i in 0..4 {
        piece.network.add_line(v[i], v[(i + 1) % 4]).unwrap();
    }
    stack
        .apply(
            &mut doc,
            Edit::AddPiece {
                id,
                piece: Box::new(piece),
            },
        )
        .unwrap();
    stack.clear();
    (doc, stack)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Apply N commands, undo N, and the document must be byte-identical to
    /// what it was. Then redo N and it must be byte-identical to the edited
    /// state. This is the whole undo contract in one assertion.
    #[test]
    fn undo_redo_round_trips(script in proptest::collection::vec(op(), 1..40)) {
        let (mut doc, mut stack) = seeded();
        let start = doc.to_json().unwrap();
        let mut gesture = 0u64;
        for o in &script {
            run_op(&mut doc, &mut stack, &mut gesture, o);
        }
        let edited = doc.to_json().unwrap();

        let mut undone = 0;
        while stack.undo(&mut doc).unwrap() {
            undone += 1;
            prop_assert!(undone < 10_000, "undo did not terminate");
        }
        prop_assert_eq!(
            doc.to_json().unwrap(),
            start.clone(),
            "undoing everything must restore the original bytes"
        );

        let mut redone = 0;
        while stack.redo(&mut doc).unwrap() {
            redone += 1;
        }
        prop_assert_eq!(redone, undone);
        prop_assert_eq!(doc.to_json().unwrap(), edited, "redo must restore the edited bytes");
    }

    /// Undo/redo halfway through, repeatedly, and the document still tracks the
    /// history exactly.
    #[test]
    fn partial_undo_redo_is_consistent(
        script in proptest::collection::vec(op(), 1..25),
        cuts in proptest::collection::vec(1usize..6, 1..6),
    ) {
        let (mut doc, mut stack) = seeded();
        let mut gesture = 0u64;
        for o in &script {
            run_op(&mut doc, &mut stack, &mut gesture, o);
        }
        let edited = doc.to_json().unwrap();

        for k in cuts {
            let mut n = 0;
            for _ in 0..k {
                if stack.undo(&mut doc).unwrap() {
                    n += 1;
                }
            }
            for _ in 0..n {
                prop_assert!(stack.redo(&mut doc).unwrap());
            }
            prop_assert_eq!(doc.to_json().unwrap(), edited.clone());
        }
    }

    /// Every id that exists before a command still means the same entity after
    /// an undo/redo cycle — this is what seams, features and selections rely on.
    #[test]
    fn ids_survive_undo_and_redo(script in proptest::collection::vec(op(), 1..30)) {
        let (mut doc, mut stack) = seeded();
        let mut gesture = 0u64;
        for o in &script {
            run_op(&mut doc, &mut stack, &mut gesture, o);
        }
        let census: Vec<(u64, String, Vec<u64>, Vec<u64>)> = doc
            .pieces()
            .iter()
            .map(|(id, p)| {
                (
                    id.raw(),
                    p.name.clone(),
                    p.network.vertices.ids().map(|v| v.raw()).collect(),
                    p.network.edges.ids().map(|e| e.raw()).collect(),
                )
            })
            .collect();

        let mut n = 0;
        while stack.undo(&mut doc).unwrap() {
            n += 1;
        }
        for _ in 0..n {
            stack.redo(&mut doc).unwrap();
        }

        let after: Vec<(u64, String, Vec<u64>, Vec<u64>)> = doc
            .pieces()
            .iter()
            .map(|(id, p)| {
                (
                    id.raw(),
                    p.name.clone(),
                    p.network.vertices.ids().map(|v| v.raw()).collect(),
                    p.network.edges.ids().map(|e| e.raw()).collect(),
                )
            })
            .collect();
        prop_assert_eq!(census, after, "ids drifted across an undo/redo cycle");
    }

    /// Coalescing changes the shape of the history, never the document.
    #[test]
    fn coalescing_does_not_change_the_result(
        moves in proptest::collection::vec((-200.0f64..200.0, -200.0f64..200.0), 2..20),
    ) {
        let run = |gesture: Option<GestureId>| {
            let (mut doc, mut stack) = seeded();
            let pid = doc.pieces().ids().next().unwrap();
            let vid = doc.piece(pid).unwrap().network.vertices.ids().next().unwrap();
            for (x, y) in &moves {
                let mut tx = Transaction::new("Move point").edit(Edit::MoveVertex {
                    piece: pid,
                    id: vid,
                    pos: Point::new(*x, *y),
                });
                if let Some(g) = gesture {
                    tx = tx.with_gesture(g);
                }
                stack.apply(&mut doc, tx).unwrap();
            }
            (doc.to_json().unwrap(), stack.undo_depth())
        };
        let (loose, loose_depth) = run(None);
        let (merged, merged_depth) = run(Some(GestureId(1)));
        prop_assert_eq!(loose, merged, "coalescing changed the document");
        prop_assert_eq!(merged_depth, 1);
        prop_assert_eq!(loose_depth, moves.len());
    }

    /// A document that has been through arbitrary editing still reloads to
    /// itself. Guards the whole serialize / migrate / deserialize loop.
    #[test]
    fn any_edited_document_round_trips_through_json(
        script in proptest::collection::vec(op(), 1..30),
    ) {
        let (mut doc, mut stack) = seeded();
        let mut gesture = 0u64;
        for o in &script {
            run_op(&mut doc, &mut stack, &mut gesture, o);
        }
        let text = doc.to_json().unwrap();
        let back = Document::from_json(&text).unwrap();
        prop_assert_eq!(back.to_json().unwrap(), text);
    }
}
