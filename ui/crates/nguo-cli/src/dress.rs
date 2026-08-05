// SPDX-License-Identifier: GPL-2.0-or-later
//! The industrial example: **one sheath dress**, drafted in millimetres to the
//! shipped `female` avatar, sewn, draped, and saved as a `.nguo`.
//!
//! This module is only the *pattern* — a [`nguo_doc::Document`]. The pipeline
//! that meshes it, hangs it on the body and drapes it is
//! [`crate::ops::dress_example`].
//!
//! # Where every number comes from
//!
//! **Body girths and heights: `assets/avatars/data/manifest.json`,
//! `avatars.female.measurements`.** Read from the file, not measured off the
//! collision proxy — a horizontal slab through a T-posed body catches both arms
//! and reads the chest ~10 % high. The nine girths the bundle ships are the only
//! body numbers this file uses, and they are quoted to the millimetre below so a
//! `git diff` of the manifest breaks the test in `tests/dress.rs`.
//!
//! **The hem height is a landmark, not a derivation, and that is a correction.**
//! The manifest has no knee landmark, so the obvious move is to derive one from
//! `avatars.female.skeleton`: `inseam_m − femur_m` = 925.717 − 438.774 =
//! 486.943 mm. **That number is not usable and the file says why itself** — the
//! `measurements.calf` landmark is at 571.43 mm, i.e. the derived knee would sit
//! 84 mm *below* the calf, which no leg does. One of the two blocks is wrong and
//! this module cannot tell which, so it uses the one in `measurements`: the hem
//! finishes at **`calf`, 571.43 mm**, which is 30.0 % of this body's stature
//! against the ~28.5 % that standard anthropometry puts a knee at — a just-below-
//! the-knee sheath length. The derivation is recorded here as a *finding about
//! the manifest*, and deliberately not used.
//!
//! **Ease and seam allowances are pattern-cutting convention, and they are
//! stated as convention rather than dressed up as a citation** — nobody here has
//! an edition of Aldrich or Armstrong on the desk to check a page number
//! against, and a fabricated page number is worse than none. What is claimed:
//! 40 mm bust ease, 20 mm waist ease and 40 mm hip ease are the allowances a
//! close-fitting woven bodice-and-skirt block is drafted with, 15 mm (5/8″) is
//! the standard industrial seam allowance, and a 40 mm turn-up is a standard
//! dress hem. Nothing here is stretch fabric; a knit block would take less.
//!
//! # Which size is this?
//!
//! **None.** It is drafted *to measure*, and that is a finding rather than a
//! shortcut. Against the UK ready-to-wear chart (`nguo_sewing::sample`'s
//! `SKIRT_HIP_BODY_MM` cites UK 12 = bust 880 / waist 700 / hip 960, and the
//! grade is +40 mm per size per girth):
//!
//! | girth | avatar   | nearest UK size |
//! |-------|----------|-----------------|
//! | waist | 745.0 mm | **UK 14** (740) |
//! | hip   | 1087.3 mm| **UK 18** (1080)|
//! | bust  | 1116.0 mm| **off the chart** — UK 20 is 1040, UK 22 is 1080 |
//!
//! No single nominal size fits this body: the waist is a 14, the hip an 18, and
//! the bust is larger than a 22 while the waist is a 14 — a bust-to-waist drop
//! of **371 mm** against the 180–200 mm every published chart uses. Nothing
//! human has that drop, so the manifest's `chest` girth is almost certainly
//! measured on a T-posed body as a horizontal hull and therefore **includes both
//! arms** (`upper_chest` = 1114.7 mm and `chest` = 1116.0 mm, 1.4 mm apart
//! across 16 mm of height, is the tell). It is still what this file drafts to,
//! because a loose bust is the safe direction — the arranger must never start
//! cloth *inside* the body — and because inventing a corrected girth would be
//! exactly the fabrication this module refuses. The consequence is measured, not
//! hidden: the dress is loose at the bust, and what holds it up is the waist,
//! not the bust. See [`WAIST_HALF_MM`].
//!
//! # Why a two-panel sheath and not a bodice-plus-skirt
//!
//! Measured against the code, not preferred. `nguo_sewing::arrange`'s
//! [`Archetype::Tube`](nguo_sewing::Archetype) lays **every** piece on **one**
//! cylinder with **one** `top_y_m`, and sums every piece's support arc into that
//! one circumference. A bodice/skirt pair therefore arranges as four panels
//! round a circle of (bust + hip)/2π, all four starting at the bust line and
//! overlapping each other — the one initial condition a barrier method cannot
//! recover from. The crate's own `hang()` says so: *"A two-tier garment (bodice
//! at the underbust, skirt at the waist) needs a placement per piece, and that is
//! a different arrangement, not a different string."* Shoulder seams fail for the
//! same reason: on a wrapped cylinder the front and back panels meet only along
//! the two side seams, so a shoulder seam would fold the cloth at the *side*
//! rather than over the shoulder.
//!
//! So the honest dress that actually closes is the one whose seams a tube can
//! close: a **sleeveless sheath**, cut front and back, shaped through the waist,
//! joined by two side seams. That is a real garment, not a rectangle — the side
//! seam is a three-segment curve through the bust, waist and hip lines.
//!
// ponytail: two panels and two seams. A `Placement3d::Cone` variant plus a
// per-piece `top_y_m` in `nguo_sewing::arrange` unlocks the bodice/skirt/waist-
// seam dress, the A-line and the peplum in one change; build it when there is a
// two-tier garment to measure it on, and not before.

use nguo_doc::kurbo::Point;
use nguo_doc::measure::{Param, Unit};
use nguo_doc::pattern::{EdgeGeom, EdgeId, EdgeSemantic};
use nguo_doc::{Document, Edit, Feature, Material, Piece, Seam, SeamSide};

// ---------------------------------------------------------------------------
// The body — `assets/avatars/data/manifest.json`, `avatars.female`
// ---------------------------------------------------------------------------

/// `measurements.chest.circumference_m` × 1000. See the module docs: this is
/// almost certainly arm-contaminated, it is used anyway, and the report says so.
pub const BODY_BUST_MM: f64 = 1116.03;
/// `measurements.chest.height_m` × 1000 — where the dress's top edge sits.
pub const BODY_BUST_Y_MM: f64 = 1471.87;
/// `measurements.waist.circumference_m` × 1000.
pub const BODY_WAIST_MM: f64 = 745.00;
/// `measurements.waist.height_m` × 1000.
pub const BODY_WAIST_Y_MM: f64 = 1208.66;
/// `measurements.hip.circumference_m` × 1000.
pub const BODY_HIP_MM: f64 = 1087.27;
/// `measurements.hip.height_m` × 1000.
pub const BODY_HIP_Y_MM: f64 = 932.18;
/// `measurements.calf.height_m` × 1000 — where the hem finishes. See the module
/// docs for why this landmark and not a knee derived from the skeleton block.
pub const BODY_CALF_Y_MM: f64 = 571.43;
/// `stature_m` × 1000, for the report only.
pub const BODY_STATURE_MM: f64 = 1904.77;

// ---------------------------------------------------------------------------
// The draft — ease, allowances, and the four horizontal lines
// ---------------------------------------------------------------------------

/// Bust ease, mm: a close-fitting woven bodice block.
pub const BUST_EASE_MM: f64 = 40.0;
/// Waist ease, mm. Least of the three — a sheath is fitted at the waist, and on
/// this body the waist is the only thing holding the dress up.
pub const WAIST_EASE_MM: f64 = 20.0;
/// Hip ease, mm: a straight woven skirt block, the same allowance
/// `nguo_sewing::sample::SKIRT_HIP_EASE_MM` uses.
pub const HIP_EASE_MM: f64 = 40.0;
/// Side-seam allowance, mm — 15 mm, 5/8″, the industrial standard.
///
/// **Not part of the simulated outline.** The mesh is the *net* pattern, i.e.
/// the stitch line, because that is the garment. The allowance is the fabric
/// outside it; cutting instructions add it back.
pub const SEAM_ALLOWANCE_MM: f64 = 15.0;
/// Hem turn-up, mm. Also outside the net pattern, for the same reason.
pub const HEM_ALLOWANCE_MM: f64 = 40.0;
/// Neckline/top-edge finish: a shaped facing, so the top edge is `Cut` and takes
/// no seam allowance in the outline. Width recorded for the cutting list.
pub const FACING_WIDTH_MM: f64 = 25.0;

/// Half the finished bust girth = one panel's top edge. **1156.03 / 2.**
pub const BUST_HALF_MM: f64 = (BODY_BUST_MM + BUST_EASE_MM) / 2.0;
/// Half the finished waist girth. **765.0 / 2.**
///
/// This is the load-bearing dimension of the whole garment. The dress is loose
/// at the bust (see the module docs), so it will try to slide down — and it
/// cannot, because a 765 mm waist section cannot pass a 1087 mm hip. The slide
/// stops where the body girth reaches 765 mm, which by linear interpolation
/// between `waist` (745.0 mm @ 1208.66) and `high_hip` (1005.7 mm @ 1038.10) is
/// y ≈ 1195.6 mm, i.e. **~13 mm below where it was placed**. The garment is held
/// up by its own fit, not by a pin and not by luck.
pub const WAIST_HALF_MM: f64 = (BODY_WAIST_MM + WAIST_EASE_MM) / 2.0;
/// Half the finished hip girth. **1127.27 / 2**, and it must stay below
/// [`BUST_HALF_MM`] — the arranger sizes one cylinder from the support edge, so a
/// panel wider anywhere than at its support edge overruns the circle and
/// self-intersects at t = 0.
pub const HIP_HALF_MM: f64 = (BODY_HIP_MM + HIP_EASE_MM) / 2.0;

/// Bust line to waist line, mm: 1471.87 − 1208.66.
pub const WAIST_DROP_MM: f64 = BODY_BUST_Y_MM - BODY_WAIST_Y_MM;
/// Bust line to hip line, mm: 1471.87 − 932.18.
pub const HIP_DROP_MM: f64 = BODY_BUST_Y_MM - BODY_HIP_Y_MM;
/// Bust line to hem, mm: 1471.87 − 571.43, i.e. just below the knee.
pub const LENGTH_MM: f64 = BODY_BUST_Y_MM - BODY_CALF_Y_MM;

/// The avatar landmark the top edge hangs at — a manifest key, resolved by
/// `nguo_avatar::Landmark::from_key`.
pub const SUPPORT_LANDMARK: &str = "chest";
/// The avatar this is drafted for: the manifest key, and the `.nguo`'s
/// `AvatarRef::name`.
pub const AVATAR_KEY: &str = "female";
/// Pattern-space gap between the two panels on the cutting layout. Affects the
/// flat lay only; the tube arrangement re-spaces them round the body.
const PANEL_GAP_MM: f64 = 100.0;

/// The four boundary chains of a panel, as indices into what [`panel`] returns.
/// `top` and `hem` are one edge each; each side is a chain of three.
pub const E_TOP: usize = 0;
const E_RIGHT: std::ops::Range<usize> = 1..4;
pub const E_HEM: usize = 4;
const E_LEFT: std::ops::Range<usize> = 5..8;

/// The dress: two pieces, two side seams, one fabric, three named eases.
///
/// Deterministic — same ids, same geometry, byte-identical JSON, every call.
/// That is what lets the `.nguo` in `assets/examples/` be a fixture.
pub fn dress() -> Document {
    let mut doc = Document::new();
    doc.apply_edit(Edit::SetMetadata {
        meta: Box::new(nguo_doc::Metadata {
            name: "Sheath dress — drafted to Nguo Female".into(),
            author: "Seede XR".into(),
            display_unit: Unit::Millimeter,
            notes: format!(
                "Sleeveless sheath dress, front and back panel, two shaped side seams. \
                 Drafted to measure for the `{AVATAR_KEY}` avatar of assets/avatars \
                 (stature {BODY_STATURE_MM:.0} mm), NOT to a nominal size: waist {BODY_WAIST_MM:.0} mm \
                 is a UK 14, hip {BODY_HIP_MM:.0} mm a UK 18, bust {BODY_BUST_MM:.0} mm off the chart. \
                 Body girths from assets/avatars/data/manifest.json. Ease: bust \
                 {BUST_EASE_MM:.0} mm, waist {WAIST_EASE_MM:.0} mm, hip {HIP_EASE_MM:.0} mm. \
                 Seam allowance {SEAM_ALLOWANCE_MM:.0} mm on the side seams, {HEM_ALLOWANCE_MM:.0} mm hem \
                 turn-up, {FACING_WIDTH_MM:.0} mm faced top edge — none of the three is in the \
                 outline, which is the net (stitch-line) pattern. Grain: straight \
                 grain down the centre of each panel, parallel to the side seams."
            ),
        }),
    })
    .expect("metadata is always valid");

    // The eases live in the document as parameters, so the file explains its own
    // draft to whoever opens it — and so a re-grade edits three numbers.
    for (name, mm, what) in [
        (
            "bust_ease",
            BUST_EASE_MM,
            "close-fitting woven bodice block",
        ),
        ("waist_ease", WAIST_EASE_MM, "fitted at the waist"),
        ("hip_ease", HIP_EASE_MM, "straight woven skirt block"),
        (
            "seam_allowance",
            SEAM_ALLOWANCE_MM,
            "5/8in, side seams; outside the net outline",
        ),
    ] {
        doc.apply_edit(Edit::SetParam {
            name: name.into(),
            param: Some(Param {
                expr: format!("{mm}"),
                unit: Unit::Millimeter,
                description: what.into(),
            }),
        })
        .expect("a literal millimetre expression is always valid");
    }

    let fabric = doc.next_material_id();
    doc.apply_edit(Edit::AddMaterial {
        id: fabric,
        material: Box::new(Material {
            name: "Cotton poplin, 120 g/m²".into(),
            ..Material::default()
        }),
    })
    .expect("AddMaterial at the planned id is always valid");

    let mut added = Vec::new();
    for (name, x) in [
        ("Dress front", 0.0),
        ("Dress back", BUST_HALF_MM + PANEL_GAP_MM),
    ] {
        let (mut piece, edges) = panel(name, x);
        piece.material = Some(fabric);
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
    // Both panels are authored with the same outline direction — top edge left to
    // right, right side downward, hem right to left, left side *upward* — so the
    // left chain has to be walked in reverse to run top-down like its partner.
    // Round the body the order is front-left, front-right, back-left, back-right:
    // front's right meets back's left, and front's left meets back's right the
    // long way round, which is what closes the tube. Same wiring as
    // `nguo_sewing::sample::bodice_block`; see there for why.
    let down = |edges: &[EdgeId], range: std::ops::Range<usize>| -> Vec<(EdgeId, bool)> {
        edges[range].iter().map(|&e| (e, false)).collect()
    };
    let up = |edges: &[EdgeId], range: std::ops::Range<usize>| -> Vec<(EdgeId, bool)> {
        edges[range].iter().rev().map(|&e| (e, true)).collect()
    };
    for (a, b, note) in [
        (
            SeamSide::whole(*front, down(fe, E_RIGHT)),
            SeamSide::whole(*back, up(be, E_LEFT)),
            "Right side seam (wearer's left)",
        ),
        (
            SeamSide::whole(*front, up(fe, E_LEFT)),
            SeamSide::whole(*back, down(be, E_RIGHT)),
            "Left side seam (wearer's right)",
        ),
    ] {
        let id = doc.next_seam_id();
        doc.apply_edit(Edit::AddSeam {
            id,
            seam: Box::new(Seam {
                allowance_mm: SEAM_ALLOWANCE_MM,
                note: note.into(),
                ..Seam::new(a, b)
            }),
        })
        .expect("both sides name edges that were just added");
    }

    // The avatar is REFERENCED, never embedded: the bundle is 227 MB, Git-LFS and
    // CC BY 4.0, licensed separately from this GPL code. Tier and bundle version
    // are filled in by `ops::dress_example` from the manifest it actually opened —
    // this is the draft's own record of the body it was drafted to.
    doc.apply_edit(Edit::SetAvatar {
        avatar: Some(Box::new(nguo_doc::AvatarRef {
            name: AVATAR_KEY.into(),
            measurements: [
                ("bust".to_owned(), BODY_BUST_MM),
                ("waist".to_owned(), BODY_WAIST_MM),
                ("hip".to_owned(), BODY_HIP_MM),
                ("stature".to_owned(), BODY_STATURE_MM),
            ]
            .into_iter()
            .collect(),
            pose: String::new(),
            asset: None,
            tier: None,
            bundle_version: None,
        })),
    })
    .expect("an avatar reference is always valid");

    doc
}

/// One panel: eight boundary edges, in [`E_TOP`] / [`E_RIGHT`] / [`E_HEM`] /
/// [`E_LEFT`] order, at pattern-space `x`.
///
/// Pattern space is **millimetres, `+y` downward**, so `y = 0` is the bust line
/// and `y = LENGTH_MM` is the hem. Front and back are the same outline: a sheath
/// with no dart and no centre-back shaping is symmetric, which is also what makes
/// the two seam sides equal in length and the ease ratio exactly 1.0.
fn panel(name: &str, x: f64) -> (Piece, Vec<EdgeId>) {
    let mut piece = Piece::new(name);
    piece.default_allowance_mm = SEAM_ALLOWANCE_MM;
    let cx = x + BUST_HALF_MM / 2.0;
    // (half-width, y) at the four horizontal lines of the draft. Straight from
    // the hip line down: the hem must not be wider than the support edge.
    let lines = [
        (BUST_HALF_MM / 2.0, 0.0),
        (WAIST_HALF_MM / 2.0, WAIST_DROP_MM),
        (HIP_HALF_MM / 2.0, HIP_DROP_MM),
        (HIP_HALF_MM / 2.0, LENGTH_MM),
    ];
    // Right side top-down, then left side bottom-up: one closed loop.
    let right: Vec<_> = lines.iter().map(|&(w, y)| Point::new(cx + w, y)).collect();
    let left: Vec<_> = lines
        .iter()
        .rev()
        .map(|&(w, y)| Point::new(cx - w, y))
        .collect();
    let ring: Vec<_> = right
        .iter()
        .chain(left.iter())
        .map(|&p| piece.network.add_vertex(p))
        .collect();
    // `ring` is [bustR, waistR, hipR, hemR, hemL, hipL, waistL, bustL].
    let (bust_r, hem_r, hem_l, bust_l) = (ring[0], ring[3], ring[4], ring[7]);

    let mut edges = Vec::with_capacity(8);
    let mut add = |piece: &mut Piece, a, b, geom, semantic| {
        edges.push(
            piece
                .network
                .add_edge(a, b, geom, semantic)
                .expect("both vertices were just added"),
        );
    };

    // E_TOP: the bust line, faced rather than seamed.
    add(
        &mut piece,
        bust_l,
        bust_r,
        EdgeGeom::Line,
        EdgeSemantic::Cut,
    );
    // E_RIGHT: bust -> waist -> hip -> hem, downward.
    for i in 0..3 {
        add(
            &mut piece,
            ring[i],
            ring[i + 1],
            side_geom(right[i], right[i + 1]),
            EdgeSemantic::Seam,
        );
    }
    // E_HEM.
    add(&mut piece, hem_r, hem_l, EdgeGeom::Line, EdgeSemantic::Cut);
    // E_LEFT: hem -> hip -> waist -> bust, upward — the mirror of E_RIGHT.
    for i in 4..7 {
        add(
            &mut piece,
            ring[i],
            ring[i + 1],
            side_geom(left[i - 4], left[i - 3]),
            EdgeSemantic::Seam,
        );
    }

    // The top edge is what the garment hangs by, and it is the widest line in the
    // draft, which is what makes it the right thing for the arranger to size the
    // start cylinder from.
    piece.features.insert(Feature::Support {
        edge: edges[E_TOP],
        landmark: SUPPORT_LANDMARK.into(),
    });
    // Straight grain: down the panel centre, bust line to hem, parallel to the
    // side seams. A real feature an exporter can read, not a comment.
    piece.features.insert(Feature::GrainLine {
        from: Point::new(cx, 0.0),
        to: Point::new(cx, LENGTH_MM),
    });
    // Balance notches at the waist, on both side seams: the registration marks
    // that stop a shaped seam being sewn 20 mm out of true.
    for e in [edges[E_RIGHT.start], edges[E_LEFT.end - 1]] {
        piece.features.insert(Feature::Notch {
            edge: e,
            t: 1.0,
            kind: nguo_doc::NotchKind::V,
        });
    }
    piece.features.insert(Feature::Label {
        at: Point::new(cx, LENGTH_MM * 0.5),
        angle_rad: 0.0,
        text: format!(
            "{name} — cut 1 on the straight grain — SA {SEAM_ALLOWANCE_MM:.0} mm sides, \
             hem {HEM_ALLOWANCE_MM:.0} mm, top edge faced {FACING_WIDTH_MM:.0} mm"
        ),
    });

    (piece, edges)
}

/// The shaped side seam between two draft lines: a cubic with **vertical
/// tangents at both ends**.
///
/// Two properties, both load-bearing rather than decorative:
///
/// * The x-coordinate control polygon is `[x0, x0, x1, x1]`, so the curve's x is
///   monotone between its endpoints. A side seam that bulged past its own bust
///   line would overrun the arranger's cylinder — which is the failure
///   `nguo_sewing::sample::skirt_size_12` documents.
/// * Vertical at the waist from both sides means the two segments meet smoothly
///   (G1), which is what a pattern cutter's curve does and what stops a visible
///   corner in the drape.
fn side_geom(a: Point, b: Point) -> EdgeGeom {
    // 0.45, not 1/3: a fuller curve through the waist, still monotone in x.
    let dy = (b.y - a.y) * 0.45;
    EdgeGeom::Cubic {
        c1: Point::new(a.x, a.y + dy),
        c2: Point::new(b.x, b.y - dy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The draft's own arithmetic, checked against the manifest numbers quoted in
    /// this file. If someone re-measures the avatars, this is what fails first.
    #[test]
    fn the_draft_lines_are_the_manifest_girths_plus_the_stated_ease() {
        assert_eq!(BUST_HALF_MM, 578.015);
        assert_eq!(WAIST_HALF_MM, 382.5);
        assert_eq!(HIP_HALF_MM, 563.635);
        // The support edge must be the widest line, or the arranger's cylinder is
        // too small for the hip and the panel overruns it. A `const` block, so a
        // re-grade that breaks it fails to *compile* rather than failing a test.
        const {
            assert!(
                HIP_HALF_MM < BUST_HALF_MM,
                "the hip is wider than the support edge: the panel will overrun \
                 the arranger's cylinder and self-intersect at t = 0",
            )
        };
        const { assert!(WAIST_HALF_MM < HIP_HALF_MM) };
        // Vertical: bust line to waist, hip, hem.
        assert!((WAIST_DROP_MM - 263.21).abs() < 1e-9, "{WAIST_DROP_MM}");
        assert!((HIP_DROP_MM - 539.69).abs() < 1e-9, "{HIP_DROP_MM}");
        assert!((LENGTH_MM - 900.44).abs() < 1e-9, "{LENGTH_MM}");
        // The hem is below the hip line and above the floor, and the length is a
        // dress length rather than a tunic or a maxi: 47 % of stature.
        const { assert!(BODY_CALF_Y_MM < BODY_HIP_Y_MM) };
        let of_stature = LENGTH_MM / BODY_STATURE_MM;
        assert!((0.40..0.55).contains(&of_stature), "{of_stature:.3}");
    }

    #[test]
    fn the_dress_is_two_panels_two_seams_and_carries_its_own_draft() {
        let doc = dress();
        assert_eq!(doc.pieces().len(), 2);
        assert_eq!(doc.seams().len(), 2);
        assert_eq!(doc.materials().len(), 1);
        assert_eq!(doc.params().len(), 4);
        assert_eq!(doc.avatar().map(|a| a.name.as_str()), Some(AVATAR_KEY));

        for (_, p) in doc.pieces().iter() {
            assert_eq!(p.network.edges.len(), 8, "{}", p.name);
            assert_eq!(p.network.vertices.len(), 8, "{}", p.name);
            // A pattern, not a mesh generator: grain, support, notches, label.
            let has = |f: fn(&Feature) -> bool| p.features.values().filter(|x| f(x)).count();
            assert_eq!(has(|f| matches!(f, Feature::GrainLine { .. })), 1);
            assert_eq!(has(|f| matches!(f, Feature::Support { .. })), 1);
            assert_eq!(has(|f| matches!(f, Feature::Notch { .. })), 2);
            assert_eq!(has(|f| matches!(f, Feature::Label { .. })), 1);
            // The support edge's arc is the top edge, which is the bust half.
            let arc = nguo_sewing::support_arc_mm(p).expect("annotated");
            assert!((arc - BUST_HALF_MM).abs() < 1e-9, "{arc}");
            // ...and the panel is never wider than it, at any height.
            let b = p.network.bounds().expect("bounds");
            assert!(b.width() <= BUST_HALF_MM + 1e-9, "{} wide", b.width());
            assert!((b.height() - LENGTH_MM).abs() < 1e-9, "{} tall", b.height());
        }

        // Both side seams carry the industrial allowance, not the 10 mm default.
        for (_, s) in doc.seams().iter() {
            assert_eq!(s.allowance_mm, SEAM_ALLOWANCE_MM);
            assert_eq!(s.ease_ratio, 1.0);
            assert_eq!(s.a.edges.len(), 3);
            assert_eq!(s.b.edges.len(), 3);
        }
    }

    /// Determinism at the document level: the `.nguo` in `assets/examples/` is a
    /// fixture, so the geometry and the ids cannot drift between calls.
    #[test]
    fn the_dress_is_byte_identical_call_to_call() {
        assert_eq!(
            dress().to_json().expect("serialises"),
            dress().to_json().expect("serialises"),
        );
    }
}
