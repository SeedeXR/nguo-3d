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
//!
//! # The two documents here are not interchangeable
//!
//! [`bodice_block`] is sized off **this repository's own avatars** — it fits the
//! shipped bodies by construction, which is what makes it a good golden and a bad
//! example of anything. [`skirt_size_12`] is sized off a **published body chart**
//! and knows nothing about our avatars, which is what a real graded pattern is:
//! it does not fit, and something has to make it fit. That is the drop-and-fit
//! path (`engine_api::resize`).

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
    rect_panel(name, x, PANEL_W_MM, PANEL_H_MM, SUPPORT_LANDMARK)
}

/// One rectangular panel `w` × `h` at pattern-space `x`, hanging from `landmark`.
fn rect_panel(
    name: &str,
    x: f64,
    w: f64,
    h: f64,
    landmark: &str,
) -> (Piece, Vec<nguo_pattern::EdgeId>) {
    let mut piece = Piece::new(name);
    let corners = [(x, 0.0), (x + w, 0.0), (x + w, h), (x, h)];
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
        landmark: landmark.into(),
    });

    (piece, edges)
}

// ---------------------------------------------------------------------------
// The industrial example: one graded garment, in real millimetres
// ---------------------------------------------------------------------------

/// UK ready-to-wear **size 12** hip girth, millimetres.
///
/// From the standard UK size-12 body chart — bust 880 / waist 700 / **hip 960** —
/// the sizing published by essentially every UK high-street size guide. It is a
/// *body* measurement, not a garment one: the pattern adds ease to it below. This
/// file deliberately does not cite a particular pattern-cutting edition, because
/// nobody here has one on the desk to check against; what it does claim is that
/// these are the numbers a size-12 skirt is drafted to, and they are not invented.
///
/// The point of using a chart at all is that **it is not our avatars**. The shipped
/// hips are 1 053.7 mm (male) and 1 087.3 mm (female), so a correctly drafted
/// size-12 skirt is 8–12 % too small for both, and something has to close that gap.
pub const SKIRT_HIP_BODY_MM: f64 = 960.0;
/// Wearing ease at the hip: 40 mm, the usual allowance for a woven straight skirt.
/// A knit or a stretch woven would take less; nothing here is stretch.
pub const SKIRT_HIP_EASE_MM: f64 = 40.0;
/// Finished hip girth: body + ease = **1 000 mm**.
pub const SKIRT_HIP_MM: f64 = SKIRT_HIP_BODY_MM + SKIRT_HIP_EASE_MM;
/// Half of it, i.e. one panel — front and back are the same rectangle.
pub const SKIRT_PANEL_W_MM: f64 = SKIRT_HIP_MM / 2.0;
/// Hip line to hem, millimetres: just above the knee on a size-12 figure.
pub const SKIRT_LENGTH_MM: f64 = 430.0;
/// Side-seam allowance, millimetres — 15 mm, the industry standard (5/8").
///
/// **Not part of the simulated outline.** The mesh is the *net* pattern, i.e. the
/// stitch line, because that is the garment; the allowance is the fabric outside
/// it that the overlocker eats. Cutting instructions add it back.
pub const SKIRT_SEAM_ALLOWANCE_MM: f64 = 15.0;
/// The landmark the finished girth was drafted to, so it hangs where it was sized.
pub const SKIRT_SUPPORT_LANDMARK: &str = "hip";
const SKIRT_GAP_MM: f64 = 80.0;

/// A **size-12 straight skirt**, two panels, drafted to a published body chart.
/// Every dimension is one of the `SKIRT_*` constants above; read them there
/// rather than from a table beside them that has to be kept in step by hand.
///
/// Grain is the straight grain, running down the centre of each panel parallel to
/// the side seams — carried as a real [`nguo_doc::Feature::GrainLine`], not as a
/// comment, so an exporter can read it.
///
/// # Why straight and not A-line
///
/// Measured, not preferred. [`crate::arrange`] wraps every piece on **one**
/// cylinder whose radius comes from the *support* arc
/// ([`crate::arrange::Arrangement::radius_m`]), so a panel wider at the hem than at
/// the hip overruns the circle and the hem self-intersects at t = 0 — the one
/// initial condition a barrier method cannot recover from. Widening the cylinder
/// instead leaves the hip ring **open**, and then `engine_api::resize::ring` reads
/// the convex hull *including the closing chord*: a 1 000 mm hip arc on a 180 mm
/// radius measures 1 128 mm, 13 % over, and the resize factor is wrong by exactly
/// that. A straight panel closes the tube exactly, so the ring reads true.
///
/// Flare needs a per-piece radius (a cone, not a cylinder) in `Placement3d`. That
/// is a real feature and it is not this file's.
// ponytail: rectangles. A cone archetype in `arrange` unlocks A-line, peplum and
// circle skirts in one change; add it when a flared garment needs to drape, not
// before.
pub fn skirt_size_12() -> Document {
    let mut doc = Document::new();
    doc.apply_edit(Edit::SetMetadata {
        meta: Box::new(nguo_doc::Metadata {
            name: "Straight skirt, UK size 12".into(),
            ..doc.meta().clone()
        }),
    })
    .expect("metadata is always valid");

    let mut added = Vec::new();
    for (name, x) in [
        ("Skirt front", 0.0),
        ("Skirt back", SKIRT_PANEL_W_MM + SKIRT_GAP_MM),
    ] {
        let (mut piece, edges) = rect_panel(
            name,
            x,
            SKIRT_PANEL_W_MM,
            SKIRT_LENGTH_MM,
            SKIRT_SUPPORT_LANDMARK,
        );
        // Straight grain: down the panel centre, hip line to hem.
        piece.features.insert(nguo_doc::Feature::GrainLine {
            from: nguo_doc::kurbo::Point::new(x + SKIRT_PANEL_W_MM * 0.5, 0.0),
            to: nguo_doc::kurbo::Point::new(x + SKIRT_PANEL_W_MM * 0.5, SKIRT_LENGTH_MM),
        });
        piece.features.insert(nguo_doc::Feature::Label {
            at: nguo_doc::kurbo::Point::new(x + SKIRT_PANEL_W_MM * 0.5, SKIRT_LENGTH_MM * 0.5),
            angle_rad: 0.0,
            text: format!("{name} — UK 12 — cut 1 — SA {SKIRT_SEAM_ALLOWANCE_MM:.0} mm"),
        });
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
    // Same chain order as `bodice_block`: front-right to back-left, front-left to
    // back-right the long way round. See that function for why the reversal flags
    // are what they are — the back panel's outline runs clockwise.
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

    /// The graded example sews clean, carries its grain and its size, and — the
    /// part that makes it an example rather than a second golden — **does not fit
    /// either shipped body**. If it ever does, the drop-and-fit chain it exists to
    /// exercise becomes a no-op and every downstream assertion goes vacuous.
    #[test]
    fn the_graded_skirt_is_a_clean_size_12_that_fits_neither_shipped_body() {
        let doc = skirt_size_12();
        assert_eq!(doc.pieces().len(), 2);
        assert_eq!(doc.seams().len(), 2);
        let diagnostics = validate(&doc);
        assert!(is_sewable(&diagnostics), "{diagnostics:?}");
        assert!(
            diagnostics.iter().all(|d| d.severity == Severity::Info),
            "{:?}",
            diagnostics
                .iter()
                .filter(|d| d.severity != Severity::Info)
                .map(|d| d.message())
                .collect::<Vec<_>>(),
        );

        // Drafted to the chart, hung at the girth it was drafted to.
        let h = crate::hang(&doc);
        assert_eq!(h.landmark.as_deref(), Some(SKIRT_SUPPORT_LANDMARK));
        assert!((h.drop_mm - SKIRT_LENGTH_MM).abs() < 1e-9, "{h:?}");
        let arc: f64 = doc
            .pieces()
            .iter()
            .map(|(_, p)| crate::support_arc_mm(p).expect("both panels are annotated"))
            .sum();
        assert!((arc - SKIRT_HIP_MM).abs() < 1e-9, "{arc} mm of hip");

        // Every panel carries a grain line and a cutting label — this is a pattern,
        // not a mesh generator.
        for (_, p) in doc.pieces().iter() {
            assert!(
                p.features
                    .iter()
                    .any(|(_, f)| matches!(f, nguo_doc::Feature::GrainLine { .. })),
                "no grain line on {}",
                p.name,
            );
            assert!(p
                .features
                .iter()
                .any(|(_, f)| matches!(f, nguo_doc::Feature::Label { .. })));
        }

        // The load-bearing negative. Shipped hips, from assets/avatars/data/
        // manifest.json: 1 053.7 mm male, 1 087.3 mm female. A 1 000 mm skirt is
        // 5.1 % and 8.0 % too small — both outside `engine_api::resize`'s 3 %
        // deadband, so the resize genuinely fires on both bodies.
        for (body, hip_mm) in [("male", 1053.7), ("female", 1087.3)] {
            let short = hip_mm / SKIRT_HIP_MM - 1.0;
            assert!(
                short > 0.03,
                "the skirt is only {:.1} % under the {body} hip — inside the resize \
                 deadband, so drop-and-fit would do nothing",
                short * 100.0,
            );
        }
    }
}
