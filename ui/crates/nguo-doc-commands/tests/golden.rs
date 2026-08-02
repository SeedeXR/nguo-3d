// SPDX-License-Identifier: GPL-2.0-or-later
//! Golden/snapshot test (guide §18): run a scripted edit sequence, serialise,
//! diff against a committed file.
//!
//! Regenerate with `NGUO_BLESS=1 cargo test -p nguo-doc-commands --test golden`,
//! matching the `MGUSO_BLESS` idiom in
//! `core-engine/crates/engine-api/tests/regression.rs`.

use std::path::PathBuf;

use kurbo::{Affine, Point};
use nguo_doc::{
    AvatarRef, Camera, Document, Edit, Material, Metadata, Piece, PieceId, Seam, SeamSide,
};
use nguo_doc_commands::{add_dart, split_edge, CommandStack, GestureId, Transaction};
use nguo_measure::{Param, Unit};
use nguo_pattern::{EdgeGeom, EdgeId, EdgeSemantic, VertexId};

/// The scripted session. Deterministic: no clock, no RNG, no hashing order.
///
/// Drafts a two-piece A-line skirt, darts the front, sews the side seam,
/// nudges a point through a coalescing drag, then undoes and redoes the drag —
/// so the golden also pins down that undo/redo lands exactly where it started.
fn scripted_session() -> (Document, CommandStack) {
    let mut doc = Document::new();
    let mut stack = CommandStack::new();

    stack
        .apply(
            &mut doc,
            Edit::SetMetadata {
                meta: Box::new(Metadata {
                    name: "A-line skirt".into(),
                    author: "Seede XR".into(),
                    display_unit: Unit::Centimeter,
                    notes: String::new(),
                }),
            },
        )
        .unwrap();

    // Fabric.
    let fabric = doc.next_material_id();
    stack
        .apply(
            &mut doc,
            Edit::AddMaterial {
                id: fabric,
                material: Box::new(Material {
                    name: "Wool gabardine".into(),
                    weight_gsm: 260.0,
                    ..Material::default()
                }),
            },
        )
        .unwrap();

    // Parameters and the body they draw from.
    let mut avatar = AvatarRef {
        name: "Standard 38".into(),
        pose: "a-pose".into(),
        ..AvatarRef::default()
    };
    avatar.measurements.insert("waist".into(), 720.0);
    avatar.measurements.insert("hip".into(), 980.0);
    stack
        .apply(
            &mut doc,
            Edit::SetAvatar {
                avatar: Some(Box::new(avatar)),
            },
        )
        .unwrap();
    for (name, expr) in [("ease", "2cm"), ("quarter", "(waist + ease) / 4")] {
        stack
            .apply(
                &mut doc,
                Edit::SetParam {
                    name: name.into(),
                    param: Some(Param::new(expr)),
                },
            )
            .unwrap();
    }

    // Two pieces, drafted as squares off the parameter table.
    // Through `Document::eval`, not the bare table: `quarter` reaches through to
    // the avatar's waist measurement.
    let quarter = doc.eval("quarter").unwrap();
    let front = draft_panel(&mut doc, &mut stack, "Front", quarter, 600.0, 0.0, fabric);
    let back = draft_panel(&mut doc, &mut stack, "Back", quarter, 600.0, 400.0, fabric);

    // A waist dart on the front's top edge — one undo entry, many edits.
    let top = doc
        .piece(front)
        .unwrap()
        .network
        .edges
        .ids()
        .next()
        .unwrap();
    let tx = add_dart(
        &doc,
        front,
        top,
        0.5,
        24.0,
        Point::new(quarter * 0.5, 140.0),
    )
    .unwrap();
    stack.apply(&mut doc, tx).unwrap();

    // Split the back's side seam so it can be eased in two spans.
    let back_side = doc.piece(back).unwrap().network.edges.ids().nth(1).unwrap();
    let tx = split_edge(&doc, back, back_side, 0.4).unwrap();
    stack.apply(&mut doc, tx).unwrap();

    // Sew the side seam.
    let front_side = doc
        .piece(front)
        .unwrap()
        .network
        .edges
        .ids()
        .nth(1)
        .unwrap();
    let seam = doc.next_seam_id();
    stack
        .apply(
            &mut doc,
            Edit::AddSeam {
                id: seam,
                seam: Box::new(Seam {
                    ease_ratio: 1.02,
                    note: "side seam".into(),
                    ..Seam::new(
                        SeamSide::whole(front, vec![(front_side, false)]),
                        SeamSide::whole(back, vec![(back_side, true)]),
                    )
                }),
            },
        )
        .unwrap();

    // Arrange in 3D.
    stack
        .apply(
            &mut doc,
            Edit::SetPieceTransform {
                id: back,
                transform: Affine::translate((400.0, 0.0)),
            },
        )
        .unwrap();
    stack
        .apply(
            &mut doc,
            Edit::SetCamera {
                camera: Camera {
                    fov_y_rad: 0.6,
                    ..Camera::default()
                },
            },
        )
        .unwrap();

    // A coalescing drag on the front hem, then undo and redo it. The document
    // must land exactly where the drag left it.
    let hem_vertex = doc
        .piece(front)
        .unwrap()
        .network
        .vertices
        .ids()
        .nth(2)
        .unwrap();
    let g = GestureId(1);
    for i in 1..=8 {
        stack
            .apply(
                &mut doc,
                Transaction::new("Move point")
                    .with_gesture(g)
                    .edit(Edit::MoveVertex {
                        piece: front,
                        id: hem_vertex,
                        pos: Point::new(quarter + i as f64 * 5.0, 600.0),
                    }),
            )
            .unwrap();
    }
    stack.undo(&mut doc).unwrap();
    stack.redo(&mut doc).unwrap();

    (doc, stack)
}

fn draft_panel(
    doc: &mut Document,
    stack: &mut CommandStack,
    name: &str,
    w: f64,
    h: f64,
    x: f64,
    fabric: nguo_doc::MaterialId,
) -> PieceId {
    let id = doc.next_piece_id();
    let mut piece = Piece::new(name);
    piece.material = Some(fabric);
    piece.default_allowance_mm = 12.5;
    let v: Vec<VertexId> = [(x, 0.0), (x + w, 0.0), (x + w, h), (x, h)]
        .iter()
        .map(|&(px, py)| piece.network.add_vertex(Point::new(px, py)))
        .collect();
    let semantics = [
        EdgeSemantic::Seam,
        EdgeSemantic::Seam,
        EdgeSemantic::Cut,
        EdgeSemantic::Fold,
    ];
    let _: Vec<EdgeId> = (0..4)
        .map(|i| {
            piece
                .network
                .add_edge(v[i], v[(i + 1) % 4], EdgeGeom::Line, semantics[i])
                .unwrap()
        })
        .collect();
    // A curved side seam, so the golden covers cubic geometry too.
    let side = piece.network.edges.ids().nth(1).unwrap();
    piece.network.edges.get_mut(side).unwrap().geom = EdgeGeom::Cubic {
        c1: Point::new(x + w + 20.0, 200.0),
        c2: Point::new(x + w + 30.0, 420.0),
    };
    stack
        .apply(
            doc,
            Edit::AddPiece {
                id,
                piece: Box::new(piece),
            },
        )
        .unwrap();
    id
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/scripted_session.json")
}

#[test]
fn scripted_session_matches_the_golden() {
    let (doc, _) = scripted_session();
    let text = doc.to_json_pretty().unwrap() + "\n";
    let path = golden_path();

    if std::env::var("NGUO_BLESS").is_ok() || !path.exists() {
        std::fs::write(&path, &text).unwrap();
        eprintln!("blessed golden at {}", path.display());
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        text, want,
        "the scripted session drifted; re-bless with NGUO_BLESS=1 if intended"
    );
}

#[test]
fn the_scripted_session_is_deterministic() {
    let a = scripted_session().0.to_json().unwrap();
    let b = scripted_session().0.to_json().unwrap();
    assert_eq!(a, b);
}

#[test]
fn the_golden_document_reloads() {
    let (doc, _) = scripted_session();
    let reloaded = Document::from_json(&doc.to_json().unwrap()).unwrap();
    assert_eq!(reloaded, doc);
    assert_eq!(reloaded.eval("quarter").unwrap(), 185.0);
}

#[test]
fn undoing_the_whole_session_empties_the_document() {
    let (mut doc, mut stack) = scripted_session();
    let fresh = Document::new().to_json().unwrap();
    while stack.undo(&mut doc).unwrap() {}
    assert_eq!(
        doc.to_json().unwrap(),
        fresh,
        "undoing everything must return the document it started from"
    );

    let (target, _) = scripted_session();
    while stack.redo(&mut doc).unwrap() {}
    assert_eq!(doc.to_json().unwrap(), target.to_json().unwrap());
}
