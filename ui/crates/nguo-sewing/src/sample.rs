// SPDX-License-Identifier: GPL-2.0-or-later
//! The one garment three different callers need: the shell's opening document,
//! the end-to-end integration test, and the golden drape.
//!
//! It lives here rather than in `nguo-app` because a document with *seams* in it
//! is the thing this crate is about, and because a golden regression that
//! authors its own geometry is testing the test.
//!
//! # What it is
//!
//! Two rectangular panels sewn at both side seams: the simplest closed garment
//! there is. The waist and hem are [`EdgeSemantic::Cut`] — finished edges, not
//! unsewn seam edges — so [`crate::validate`] reports nothing, which is what
//! makes it usable as a golden.
//!
//! Wrapped by [`crate::arrange`] it becomes a tube around the body, and the two
//! side seams are what close it. That is the whole of milestone M6 in one
//! document.

use nguo_doc::pattern::{EdgeGeom, EdgeSemantic};
use nguo_doc::{Document, Edit, Piece, Seam, SeamSide};

/// Panel width, mm. Two of them give a **1 120 mm** circumference.
///
/// Sized off the shipped bodies, not picked: the **hip is 1 087.3 mm female,
/// 1 053.7 mm male** in `assets/avatars/data/manifest.json`, and 1 120 mm is 3.0 %
/// ease over the larger — a fitted tube rather than a sack.
///
/// Because it is sized for the hip it is *annotated* for the hip
/// ([`SUPPORT_LANDMARK`]), so the shell hangs it at the measured hip height and
/// the ease is 3.0 % by construction. It used to hang from the underbust and be
/// relied upon to slide down and catch, which is a garment held up by luck.
///
/// Both bounds are load-bearing:
///
/// * Narrower than the hip and the garment simply cannot close around the body.
///   The arranger would leave the whole difference as slack for the stitches to
///   pull shut *through* the body.
/// * Much wider and it never touches the body at all — a frictionless tube
///   hanging from its pinned waistband, clear of everything. Measured: at
///   1 300 mm the cloth settled at a mean radius of 0.2108 m with collision on
///   against 0.2107 m with it **off**, i.e. the body was doing nothing.
pub const PANEL_W_MM: f64 = 560.0;
/// Panel length, mm — measured hip (932.2 mm female) down to mid-calf.
pub const PANEL_H_MM: f64 = 560.0;
/// The body landmark both panels hang from: the girth [`PANEL_W_MM`] was sized
/// for. A manifest key, resolved by `nguo_avatar::Landmark::from_key`.
pub const SUPPORT_LANDMARK: &str = "hip";
/// Pattern-space gap between the two panels. Only affects the flat layout.
const PANEL_GAP_MM: f64 = 80.0;

/// Corner index of each edge's start, in the order [`piece`] adds them.
/// `e0` waist (left→right), `e1` right side, `e2` hem (right→left), `e3` left side.
const E_WAIST: usize = 0;
const E_RIGHT: usize = 1;
const E_HEM: usize = 2;
const E_LEFT: usize = 3;

/// A two-panel bodice block, sewn at both side seams.
///
/// Deterministic: same ids, same geometry, every call. That is what lets the
/// golden drape key on it.
pub fn bodice_block() -> Document {
    let mut doc = Document::new();
    doc.apply_edit(Edit::SetMetadata {
        meta: Box::new(nguo_doc::Metadata {
            name: "Bodice block".into(),
            ..doc.meta().clone()
        }),
    })
    .expect("metadata is always valid");

    let mut added = Vec::new();
    for (name, x) in [
        ("Front bodice", 0.0),
        ("Back bodice", PANEL_W_MM + PANEL_GAP_MM),
    ] {
        let (piece, edges) = panel(name, x);
        let id = doc.next_piece_id();
        doc.apply_edit(Edit::AddPiece {
            id,
            piece: Box::new(piece),
        })
        .expect("AddPiece at the planned id is always valid");
        added.push((id, edges));
    }

    let (front, fe) = &added[0];
    let (back, be) = &added[1];

    // Both sides are authored **top-down**, so the two chains already run the
    // same way and neither seam is twisted. Reversal is not cosmetic: the back
    // panel's own outline runs clockwise, so its left edge points *up*.
    //
    // Round the body the order is front-left, front-right, back-left,
    // back-right; front's right meets back's left, and front's left meets
    // back's right the long way round, closing the tube.
    for (a, b) in [
        (
            SeamSide::whole(*front, vec![(fe[E_RIGHT], false)]),
            SeamSide::whole(*back, vec![(be[E_LEFT], true)]),
        ),
        (
            SeamSide::whole(*front, vec![(fe[E_LEFT], true)]),
            SeamSide::whole(*back, vec![(be[E_RIGHT], false)]),
        ),
    ] {
        let id = doc.next_seam_id();
        doc.apply_edit(Edit::AddSeam {
            id,
            seam: Box::new(Seam::new(a, b)),
        })
        .expect("both sides name edges that were just added");
    }

    doc
}

/// One rectangular panel with its four boundary edges, in `E_*` order.
fn panel(name: &str, x: f64) -> (Piece, Vec<nguo_pattern::EdgeId>) {
    let mut piece = Piece::new(name);
    let corners = [
        (x, 0.0),
        (x + PANEL_W_MM, 0.0),
        (x + PANEL_W_MM, PANEL_H_MM),
        (x, PANEL_H_MM),
    ];
    let vertices: Vec<_> = corners
        .iter()
        .map(|&(vx, vy)| {
            piece
                .network
                .add_vertex(nguo_doc::kurbo::Point::new(vx, vy))
        })
        .collect();

    // Waist and hem are finished edges; only the sides are sewn. An edge marked
    // `Seam` that nothing sews is a warning, and a sample that ships four of
    // them would train the user to ignore the panel.
    let mut semantics = [EdgeSemantic::Seam; 4];
    semantics[E_WAIST] = EdgeSemantic::Cut;
    semantics[E_HEM] = EdgeSemantic::Cut;
    let edges: Vec<_> = (0..4)
        .map(|i| {
            piece
                .network
                .add_edge(
                    vertices[i],
                    vertices[(i + 1) % 4],
                    EdgeGeom::Line,
                    semantics[i],
                )
                .expect("both vertices were just added")
        })
        .collect();

    // The waist edge is what the garment hangs by. Both panels carry it, because
    // the supporting girth is the sum over the panels round the body.
    piece.features.insert(nguo_doc::Feature::Support {
        edge: edges[E_WAIST],
        landmark: SUPPORT_LANDMARK.into(),
    });

    (piece, edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build, is_sewable, validate, Severity, SewOptions};

    #[test]
    fn the_sample_is_two_sewn_panels_with_nothing_to_report() {
        let doc = bodice_block();
        assert_eq!(doc.pieces().len(), 2);
        assert_eq!(doc.seams().len(), 2);

        let diagnostics = validate(&doc);
        assert!(is_sewable(&diagnostics), "{diagnostics:?}");
        // Equal-length sides: no ease, no unsewn edge, no warning at all. The
        // `Info` ease rows are expected; anything above them is not.
        assert!(
            diagnostics.iter().all(|d| d.severity == Severity::Info),
            "the shipped sample must be clean: {:?}",
            diagnostics
                .iter()
                .filter(|d| d.severity != Severity::Info)
                .map(|d| d.message())
                .collect::<Vec<_>>(),
        );
    }

    /// Both seams must actually resolve onto the mesh and produce stitches, and
    /// the wrap must leave them nearly shut. A sample that sews to nothing would
    /// make every downstream test vacuous.
    #[test]
    fn the_sample_sews_and_arranges_into_a_closed_tube() {
        let garment = build(&bodice_block(), &SewOptions::default()).expect("the sample sews");
        assert_eq!(garment.seams.len(), 2);
        assert!(
            garment.seams.iter().all(|s| s.stitches > 4),
            "{:?}",
            garment.seams
        );
        assert!(
            garment.seams.iter().all(|s| !s.flipped),
            "{:?}",
            garment.seams
        );

        // Wrapped, so the whole garment sits in one shell around the origin: it
        // must not be a flat plane and it must not be inside out.
        let bounds = garment.bounds().expect("geometry");
        let extent = bounds.extent();
        assert!(
            extent[0] > 0.15 && extent[2] > 0.15,
            "not a tube: {extent:?}"
        );
        assert!(
            (extent[1] - (PANEL_H_MM * 1e-3) as f32).abs() < 1e-3,
            "the panel changed length in 3D: {extent:?}",
        );

        // The seams start open by the arranger's slack, shared between them,
        // and nothing else. That is the number the solver has to close.
        let slack = garment.arrangement.slack_m;
        assert!(
            garment.max_seam_gap_m() < slack.max(1e-3) as f64 + 1e-3,
            "seam gap {} m against {slack} m of slack",
            garment.max_seam_gap_m(),
        );
    }

    /// The sample is sized for the hip, so it must *say* hip — and its support
    /// arc must be the panel width it was sized as, or the arranger sizes the
    /// cylinder off something other than the girth in the doc comment above.
    #[test]
    fn the_sample_hangs_from_the_girth_it_was_sized_for() {
        let doc = bodice_block();
        let h = crate::hang(&doc);
        assert_eq!(h.landmark.as_deref(), Some(SUPPORT_LANDMARK));
        assert!((h.drop_mm - PANEL_H_MM).abs() < 1e-9, "{h:?}");

        let mut arcs = Vec::new();
        for (_, piece) in doc.pieces().iter() {
            arcs.push(crate::support_arc_mm(piece).expect("every panel is annotated"));
        }
        assert_eq!(arcs, vec![PANEL_W_MM, PANEL_W_MM]);
        // 1 120 mm over the female hip's 1 087.3 mm: the 3.0 % ease the constant
        // claims. A regression here means the sample no longer fits the body.
        let ease = arcs.iter().sum::<f64>() / 1087.3 - 1.0;
        assert!((0.025..0.035).contains(&ease), "{:.4} ease", ease);
    }

    /// Determinism, at the document level: the ids and the geometry a golden
    /// keys on cannot drift between calls.
    #[test]
    fn the_sample_is_byte_identical_call_to_call() {
        let a = serde_json::to_string(&bodice_block()).expect("serialises");
        let b = serde_json::to_string(&bodice_block()).expect("serialises");
        assert_eq!(a, b);
    }
}
