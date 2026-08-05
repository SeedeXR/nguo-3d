// SPDX-License-Identifier: GPL-2.0-or-later
//! Import-time **uniform** resize — make a third-party garment the right SIZE for a
//! body before anything tries to fit or drape it.
//!
//! This is the first half of *drop and fit*. The second half is
//! [`avatar::resolve_penetration`](crate::avatar::resolve_penetration), which cleans up
//! whatever overlap is left. Resize does not replace that pass; it makes the pass's job
//! legitimate instead of a compensation for a garment that is simply the wrong size.
//!
//! # The rule, and why it is this rule
//!
//! Scale by the girth of the ring the garment is **worn at** — a skirt's waistband
//! against the body's waist — matched with an ease allowance. Not stature, not an
//! average, not a bounding box. Measured on `women-skirt`/female: the waistband's girth
//! is 0.6357 m against a 0.7306 m waist, so the waistband is 12.9 % too small to go on
//! the body **at all**, while the hem is 32 % loose. One factor cannot fit both ends, so
//! the factor has to come from the end that bears the load.
//!
//! # What a uniform scale is, and what it cannot do
//!
//! A uniform scale is a similarity: every angle and every length *ratio* survives, so it
//! cannot distort the garment's design — it changes absolute size only. [`apply_resize`]
//! returns that as a number ([`ResizeCheck`]), not a claim. **A per-axis scale is not
//! offered here and must not be added**: it changes length ratios, which shears the
//! fabric's grain.
//!
//! What it cannot fix: if a garment matches the hip but not the chest, no single factor
//! fits both. [`plan_resize`] takes the **largest** demanded factor, so the garment can
//! always physically go on, and reports the residual slack at every other landmark in
//! [`ResizePlan::fits`] for the caller to show. The honest alternatives when that slack
//! is unacceptable are to let the solver take it up within the fabric's strain limit, or
//! to *grade* — a per-region change to the pattern, which is a different operation, is
//! not a scale, and is deliberately not implemented here.
//!
//! # Measured on the whole sample corpus, re-measured 2026-08-05
//!
//! Garment vertices starting inside the body, before → after this resize, over the five
//! drapeable samples × both shipped proxies. `inside %` is counted the way the project counts
//! it: pass 1 of [`avatar::resolve_penetration`](crate::avatar::resolve_penetration).
//! `depth left` is what that whole pass fails to clear, run once on the authored mesh and once
//! on the resized one — the payoff of resizing, not a measurement of the resize.
//!
//! | garment (units) | body | factor | landmark | inside % | depth left mm |
//! |---|---|---|---|---|---|
//! | women-skirt (cm) | male | **1.3456** | waist | **49.01 → 3.01** | 0.0 → 0.0 |
//! | women-skirt (cm) | female | 1.1409 | high_hip | 50.84 → 27.60 | 0.5 → 0.6 |
//! | dress-a (cm) | female | 1.1385 | high_hip | 12.33 → 6.17 | 3.2 → 46.7 |
//! | dress-b (cm) | male | 1.1348 | underbust | 14.21 → 11.01 | 58.9 → 11.5 |
//! | men-shirt (cm) | female | 1.1264 | hip | 17.25 → 14.72 | 46.9 → 43.0 |
//! | dress-a (cm) | male | 1.0475 | high_hip | 14.49 → 13.41 | 58.3 → 57.9 |
//! | men-shirt (cm) | male | 1.0 | — refused 0.9260 | 26.55 → 26.55 | 55.2 → 55.2 |
//! | dress-b (cm) | female | 1.0 | — refused 0.9618 | 12.48 → 12.48 | 6.0 → 6.0 |
//! | bike-jacket (m) | male | 1.0 | — no usable ring | 19.56 → 19.56 | 55.9 → 55.9 |
//! | bike-jacket (m) | female | 1.0 | — no usable ring | 15.67 → 15.67 | 47.9 → 47.9 |
//!
//! **Six improved, four are exact identities**, and the identities are the point: this
//! operation declines rather than guesses. *Exact* is meant bit-for-bit and is verified on the
//! corpus — the four identity pairs come back with `positions` equal to the authored array and
//! an `edge_spread()` of **0.0**. They did not before; see [`apply_resize`].
//!
//! **On `inside %` nothing got worse. On `depth left` one pair did**: `dress-a`/female goes
//! 3.2 → 46.7 mm even as its overlap halves, because the enlarged dress reaches further down
//! the leg into a deeper pocket. Said out loud because a table that printed only the improving
//! column would be exactly the flattering measurement this module exists to stop.
//!
//! Edge-length ratio spread across the ten: exactly **0.0** on the four identities and
//! 4.9e-5 … 2.5e-4 on the six real scales — f32 rounding, as a similarity must be. **An
//! earlier version of this paragraph read "1.2e-5 … 2.5e-4 — zero to f32 rounding", and that
//! floor was not rounding, it was a bug**: the identity arithmetic in [`apply_resize`]
//! perturbing meshes it was supposed to leave untouched. A number in a doc comment that was
//! true once is this project's most common defect.
//!
//! The whole 10-pair table, two full [`avatar::resolve_penetration`](crate::avatar::resolve_penetration)
//! runs per pair included, is **182 s at a peak RSS of 194 MB** (`/usr/bin/time -l`, release,
//! Apple M4). The resize itself is a fraction of a second per pair; the seconds are the fit.
//!
//! # Do not tune the factor to minimise penetration
//!
//! Measured sweep on `women-skirt`/female: total factor 1.184 → 27.5 % of vertices
//! inside; 1.420 → 13.6 %; 1.894 → 2.5 %; 2.367 → 0.15 %. The deepest overlap barely
//! moves across that whole range (74 → 35 mm) because the residual is the *pleat
//! valleys* folding inward, which a convex-hull girth cannot see and which the solver
//! settles outward as the cloth drapes. The factor that zeroes penetration turns a
//! knee-length pleated skirt into a maxi skirt: at 2.367 the hem lands at the ankle.
//! Chasing zero here is the trap.
//!
//! # Body girths come from the manifest, garment girths from the mesh
//!
//! [`plan_resize`] takes the body's girths as [`BodyGirth`] values rather than measuring
//! them, because measuring the *collision proxy* is wrong for the upper body: a
//! horizontal slab at chest height also contains both T-pose arms, and re-measuring
//! reads the male chest **+9.7 %** and the thigh **+51.9 %** high with no window width
//! that escapes it. The bundle's `manifest.json` measurements were taken on the master
//! sculpt with the torso isolated. The caller reads them (`nguo_avatar::Measurements`)
//! and passes them in; this crate never learns a bundle layout.
//!
//! Known bias, stated because it is real: the manifest is 0.45–1.94 % larger than the
//! proxy geometry the solver actually collides against (hip 1.0873 vs 1.0766 measured).
//! The bias is toward clearance and is absorbed into `ease`.
//!
//! # Units
//!
//! Everything here is metres, like the rest of the engine. A garment authored in
//! centimetres must be loaded with `engine_io::load_mesh_in` — that is the one and only
//! unit conversion site (D-18) and this module is not a second one. The scale factor
//! computed here is a *fit* factor and is never a unit conversion; if you find yourself
//! reaching for 100.0 or 0.001, the mesh was loaded in the wrong units.

use crate::avatar::AvatarError;
use crate::{Aabb, Object};

/// Default ease: the garment ends up 3 % larger than the body girth it is matched to.
///
/// Not zero, deliberately — cloth sitting exactly on a body starts every contact in the
/// barrier. 3 % is what `women-skirt` was authored with over its own mannequin's hip.
pub const FIT_EASE: f32 = 0.03;

/// Half-thickness of the horizontal slab a ring girth is measured over (±10 mm).
pub const RING_SLAB_M: f32 = 0.010;

/// A slab wider than this (x-extent ÷ z-extent) is not a ring around the torso — it is a
/// cross-section that has caught the sleeves.
///
/// Measured: `women-skirt`'s rings are all ≤ 1.59; sleeved garments read 2.2–11.4, and
/// `bike-jacket` has no ring under the gate at all, which is the correct answer for it.
///
/// **The ratio is x ÷ z and so is deliberately asymmetric**: it detects a section stretched
/// along **x**, because the engine's frame is +Y up / +Z forward and a T-pose's arms run
/// along x. A garment whose sleeves ran along z would read ≪ 1 and sail through. That is a
/// frame convention, not a general shape test, and a garment that arrives rotated must be
/// brought into the avatar's frame before it gets here — which
/// [`plan_resize`] already requires for every other reason.
pub const RING_MAX_ASPECT: f32 = 1.8;

/// A slab with fewer vertices than this is not a cross-section, it is a sliver caught on
/// an edge of the mesh, and its hull perimeter is meaningless.
///
/// **This gate was found by running the corpus, not reasoned about.** `bike-jacket`'s hem
/// clips the female avatar's hip height with **5** vertices, whose hull reads a girth of
/// 0.1185 m against a 1.0873 m hip and so demanded a factor of **9.45** — clamped to
/// [`RESIZE_LIMIT`] and still enough to lengthen the jacket by 37 cm and drive the deepest
/// overlap from 53 mm to 107 mm.
///
/// The margin is wide. Re-measured 2026-08-05 over the whole corpus — five drapeable
/// garments × both proxies × every manifest landmark — the fewest **distinct** positions in
/// any *usable* ring is **462** (`men-shirt`/female's hip), so 16 is nowhere near a
/// legitimate ring while still admitting a genuinely low-poly garment. That sweep also
/// re-confirms the sliver above at 5 distinct positions and a 0.1185 m hull, and finds **no
/// corpus ring whose verdict changes** between counting slab entries and counting distinct
/// positions — the seam-split case is a hazard the gate must be robust to, not something the
/// corpus happens to exhibit.
///
/// It counts [`Ring::verts`], which is **distinct** positions: counting slab entries let a
/// UV/normal-seam-split copy of the very sliver above clear this gate.
pub const RING_MIN_VERTS: usize = 16;

/// Factors this close to 1 are reported and not applied. The garment is the right size;
/// scaling it by 1.02 buys nothing and costs the caller a resized asset to keep track of.
pub const RESIZE_DEADBAND: f32 = 0.03;

/// Hard upper clamp on the factor.
///
/// A garment needing more than this is not the same size class as the body — a child's
/// dress on an adult avatar. **This clamps and then applies; it does not refuse.** The raw
/// demand is in the [`LandmarkFit`] table so a caller can refuse, and
/// `plan_resize`'s contract is that resize is a *proposal a user confirms*. Note the clamp
/// can itself do damage when a bogus ring wins the `max` — see [`RING_MIN_VERTS`], where a
/// clamped 9.45 lengthened `bike-jacket` by 37 cm and drove the deepest overlap 53 → 107 mm.
/// The gate that stops that is the ring's own usability, not this number.
///
/// **There is no lower clamp, because this operation never shrinks a garment.** Resize
/// exists because a waistband could not physically go on; a garment that is already loose
/// everywhere does not need fixing, and shrinking it toward the body is measured to make
/// things strictly worse. Both shrinks the corpus produced were regressions:
/// `men-shirt`/male proposed 0.9260 off its hem ring and drove overlap **26.55 % →
/// 55.51 %**, and `dress-b`/female proposed 0.9618 and drove it 12.48 % → 14.39 %. A hem
/// or a skirt sweep is loose *by design*, and no ring measurement can tell that apart
/// from "too big". A factor below 1 is therefore reported in
/// [`ResizePlan::fits`] and not applied; a user who really has an oversized import can set
/// [`ResizePlan::factor`] by hand.
pub const RESIZE_LIMIT: f32 = 1.5;

/// Landmarks that are never used to size a garment, and why.
///
/// `neck` only: measured, `men-shirt`'s collar ring is 0.3963 m against a 0.53095 m
/// neck-base girth, which alone would demand 1.42 — a collar is cut to stand away from
/// the neck, not to grip it. Every other landmark that is a poor match loses the `max`
/// on its own without needing a rule.
pub const EXCLUDED_LANDMARKS: [&str; 1] = ["neck"];

/// One body girth, as the bundle manifest measured it.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyGirth {
    /// The manifest's own key — `"waist"`, `"hip"`, … Compared against
    /// [`EXCLUDED_LANDMARKS`] and echoed back in [`LandmarkFit`].
    pub landmark: String,
    pub circumference_m: f32,
    /// Height above the ground plane; the avatar's feet are at y = 0.
    pub height_m: f32,
}

/// A horizontal cross-section of a mesh: what a tape measure would read there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    pub height_m: f32,
    /// Convex-hull perimeter of the slab's vertices projected to the xz plane — a taut
    /// tape, so it is blind to pleats and folds. That is intentional: a pleat valley is
    /// not a smaller body.
    pub girth_m: f32,
    /// Centre of the slab's bounding box. The anchor point for the scale.
    pub centre: [f32; 3],
    /// x-extent ÷ z-extent, against [`RING_MAX_ASPECT`].
    pub aspect: f32,
    /// **Distinct** xz positions in the slab, against [`RING_MIN_VERTS`] — the same point
    /// set `girth_m` was measured from, not the raw slab entry count. glTF splits vertices
    /// on UV and normal seams (this repo's own male proxy is 10 230 glTF vertices for 7 498
    /// unique positions), so counting entries would let a seam-split sliver clear a gate
    /// that exists to reject it: 5 positions in 4 copies each is the same meaningless hull
    /// reading 20.
    pub verts: usize,
}

/// What one landmark demands of the factor, usable or not, so the caller can show the
/// whole table and the user can see which landmark won.
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkFit {
    pub landmark: String,
    pub body_girth_m: f32,
    pub height_m: f32,
    /// The garment's own ring at that height. `None`: the garment does not reach there.
    pub garment: Option<Ring>,
    /// `body × (1 + ease) ÷ ring`. `None` when there is no ring, when the ring failed
    /// [`RING_MAX_ASPECT`] or [`RING_MIN_VERTS`], or when the landmark is in
    /// [`EXCLUDED_LANDMARKS`].
    pub factor: Option<f32>,
}

impl LandmarkFit {
    /// Girth slack at this landmark once `factor` is applied: `+0.05` is 5 % looser than
    /// the body, `-0.05` is 5 % too tight. This is the residual mismatch a single uniform
    /// factor leaves behind, and it is the number that says whether grading is needed.
    pub fn slack_at(&self, factor: f32) -> Option<f32> {
        let g = self.garment?;
        (self.body_girth_m > 0.0).then(|| g.girth_m * factor / self.body_girth_m - 1.0)
    }
}

/// A named, explicit, reversible resize: the factor, where it came from, and what it
/// costs. Nothing is applied until [`apply_resize`].
///
/// `factor` is public so a user can override the proposal — that is the point of
/// proposing rather than acting. The anchor is independent of the factor, so an override
/// stays anchored at the same landmark.
#[derive(Clone, Debug, PartialEq)]
pub struct ResizePlan {
    /// The uniform factor to apply. 1.0 means "leave it alone".
    pub factor: f32,
    /// Ease the factor was computed with, echoed for reproducibility.
    pub ease: f32,
    /// The landmark whose girth set the factor. `None` when no ring was usable, in which
    /// case `factor` is 1.0 and the garment is left exactly as authored.
    pub supporting: Option<String>,
    /// Fixed point of the scale: the supporting ring's centre on the garment.
    pub anchor_from: [f32; 3],
    /// Where that point ends up: the body's cross-section centre at the landmark height.
    /// Equal to `anchor_from` when there is no supporting landmark.
    pub anchor_to: [f32; 3],
    /// Every landmark that was considered, in the order given.
    pub fits: Vec<LandmarkFit>,
    /// Authored height of the garment, so [`ResizePlan::length_delta_m`] survives an
    /// override of `factor`.
    pub garment_height_m: f32,
}

impl ResizePlan {
    /// A no-op plan: the honest answer when nothing on the body matches the garment.
    fn identity(anchor: [f32; 3], height: f32, ease: f32, fits: Vec<LandmarkFit>) -> ResizePlan {
        ResizePlan {
            factor: 1.0,
            ease,
            supporting: None,
            anchor_from: anchor,
            anchor_to: anchor,
            fits,
            garment_height_m: height,
        }
    }

    /// How much longer (or shorter) the garment gets. **Always report this**: a
    /// similarity that fixes a girth changes the length by the same factor, and
    /// +16.3 cm on a skirt is the honest cost of not grading.
    pub fn length_delta_m(&self) -> f32 {
        (self.factor - 1.0) * self.garment_height_m
    }

    /// Whether [`apply_resize`] would move anything at all.
    pub fn is_identity(&self) -> bool {
        self.factor == 1.0 && self.anchor_from == self.anchor_to
    }
}

/// An **authored** edge shorter than this cannot have its length ratio measured to better
/// than 1e-4 in an f32 mesh at human scale, so it is counted
/// ([`ResizeCheck::degenerate_edges`], and [`ResizeCheck::degenerate_tris`] for the area
/// spread's ½·`EDGE_MEASURABLE_M`² counterpart) and excluded from the ratio spread. **Both
/// exclusions are also measured without any threshold at all** —
/// [`ResizeCheck::edge_ratio_min_all`] and [`ResizeCheck::area_ratio_min_all`] — so nothing
/// here can flatter a result by not looking.
///
/// **This is a representation floor, not distortion, and the arithmetic is exact.** f32's
/// spacing near a 1.5 m coordinate is 1.19e-7 m, so a length L carries an absolute error of
/// about that and a *relative* error of 1.19e-7 / L: 1.2e-4 at L = 1 mm, 1.2e-3 at 100 µm,
/// 1.2e-2 at 10 µm. Measured, that is exactly what `men-shirt` shows — 100 of its edges are
/// under 10 µm and a pure uniform scale *reads* as a 1.93e-2 spread, against 6e-5…2e-4 on
/// every other sample in the corpus. A cloth edge is millimetre-scale (measured median
/// 5.85 mm), so a 1 mm floor excludes only degenerate triangles.
///
/// The threshold is on the **authored** length, so this cannot hide a collapse: an edge
/// that starts at 5 mm and is crushed to 10 µm is still measured, and still reads 0.002.
pub const EDGE_MEASURABLE_M: f32 = 1.0e-3;

/// Proof that a transform preserved the garment. Ratios are **normalised by the factor**,
/// so 1.0 means undistorted whatever the factor was.
///
/// A uniform scale must show `edge_ratio_min == edge_ratio_max == 1.0` to floating-point
/// rounding. If it does not, the transform was not a similarity and something is wrong.
///
/// [`measure`] produces one for **any** position-only transform, not just a resize:
/// [`avatar::resolve_penetration`](crate::avatar::resolve_penetration) reports one at
/// `factor` 1.0, where the ratios are the raw distortion its projection caused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResizeCheck {
    pub factor: f32,
    /// Edge **incidences** the ratio spread was measured over — a shared edge is visited
    /// once per adjacent triangle, so this is ≈ 3 × `tris` and not the edge count. Only
    /// min/max are reported, and those are unaffected by counting an edge twice.
    pub edges: usize,
    /// Edges shorter than [`EDGE_MEASURABLE_M`], excluded from the spread and reported so
    /// the exclusion is visible rather than convenient. Same incidence counting.
    pub degenerate_edges: usize,
    /// Shortest authored edge, so a reader can judge the exclusion for themselves.
    pub edge_min_m: f32,
    /// Triangles the area ratio was measured over; the rest are counted in
    /// [`ResizeCheck::degenerate_tris`] and excluded for the same reason.
    pub tris: usize,
    /// Triangles excluded from `area_ratio_min`/`max`: authored area under
    /// ½·[`EDGE_MEASURABLE_M`]², or exactly zero.
    ///
    /// **Reported for the same reason `degenerate_edges` is, and its absence was the same
    /// defect.** `men-shirt` has **39 369** of these — 30.7 % of its 128 191 triangles — and
    /// while they were dropped silently a reader had no way to see that `tris` was not the
    /// triangle count. Measured, the exclusion flattered the result: on `bike-jacket`/female
    /// the filtered `area_ratio_max` read **2.2379** where every triangle read **4.4453**,
    /// understating the worst area blow-up by 2×.
    pub degenerate_tris: usize,
    /// `min over edges of (new_len / old_len) / factor`, over edges at or above
    /// [`EDGE_MEASURABLE_M`] only.
    pub edge_ratio_min: f32,
    pub edge_ratio_max: f32,
    /// The same ratio over **every** edge, including the sub-[`EDGE_MEASURABLE_M`] ones.
    ///
    /// **Read this one for a strain claim; read the filtered pair for a similarity claim.**
    /// One threshold cannot serve both purposes and pretending it can was a real defect:
    /// the filtered pair read 0.479…1.518 on `men-shirt`/male where all **384 573** of its
    /// edge incidences read **0.347…1.796**, because 108 682 of that mesh's edges — 28 % —
    /// are under 1 mm and were excluded by construction. A strain bound of 1.5 is 30 000×
    /// f32's relative error at 1 mm, so the exclusion has no justification at that scale — it
    /// exists only so a *uniform scale* can still be asserted to spread < 1e-3, where f32's
    /// 1.2e-2 relative error on a 10 µm edge genuinely would break the assertion.
    ///
    /// Those two numbers are the **pre-fix** state, kept because they are why this field
    /// exists. Re-measured 2026-08-05 over all 384 573, `men-shirt`/male reads 0.4990…1.5008
    /// and the worst of the ten pairs is 0.4902…1.5128.
    pub edge_ratio_min_all: f32,
    pub edge_ratio_max_all: f32,
    /// `min over triangles of (new_area / old_area) / factor²`, over triangles above the
    /// [`ResizeCheck::degenerate_tris`] floor only.
    pub area_ratio_min: f32,
    pub area_ratio_max: f32,
    /// The same ratio over **every** triangle with a non-zero authored area.
    ///
    /// **Read this one for a strain claim; read the filtered pair for a similarity claim** —
    /// exactly the split [`ResizeCheck::edge_ratio_min_all`] documents, for exactly the same
    /// reason. Measured on the corpus 2026-08-05, the two differ by up to 2×
    /// (`men-shirt`/male 0.1252 vs a filtered 0.2731; `bike-jacket`/female 4.4453 vs 2.2379).
    pub area_ratio_min_all: f32,
    pub area_ratio_max_all: f32,
}

impl ResizeCheck {
    /// Edge-length ratio spread over the **measurable** edges. **Exactly the maintainer's
    /// hard requirement as a number**: for a uniform scale this is zero to rounding (measured
    /// 4.9e-5 … 2.5e-4 over the corpus's six real scales), and **exactly 0.0** when the plan
    /// is the identity — all four of the corpus's identity pairs, verified 2026-08-05.
    pub fn edge_spread(&self) -> f32 {
        self.edge_ratio_max - self.edge_ratio_min
    }
}

/// "Nothing was transformed": every ratio exactly 1, nothing measured. The neutral element
/// a caller that did no work reports, so `Default` never has to mean "undistorted because
/// it was never checked" ambiguously — `edges == 0` says it was not checked.
impl Default for ResizeCheck {
    fn default() -> Self {
        Self {
            factor: 1.0,
            edges: 0,
            degenerate_edges: 0,
            edge_min_m: f32::INFINITY,
            tris: 0,
            degenerate_tris: 0,
            edge_ratio_min: 1.0,
            edge_ratio_max: 1.0,
            edge_ratio_min_all: 1.0,
            edge_ratio_max_all: 1.0,
            area_ratio_min: 1.0,
            area_ratio_max: 1.0,
            area_ratio_min_all: 1.0,
            area_ratio_max_all: 1.0,
        }
    }
}

/// Edge-length and triangle-area distortion of `after` against `before`, normalised by
/// `factor` (use 1.0 for a transform that is not a scale).
///
/// `before` must be parallel to `after.positions` and `after.tris` must index both — i.e.
/// the transform moved vertices and changed no topology, which is the only case in which
/// this comparison is exact.
///
/// This is the shared measurement behind both halves of drop-and-fit, so "did not destroy
/// the clothing" is the same number computed the same way whether it was a resize or a
/// projection.
pub fn measure(before: &[[f32; 3]], after: &Object, factor: f32) -> ResizeCheck {
    let len = |p: [f32; 3], q: [f32; 3]| {
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    // Twice the triangle area — the factor of 2 cancels in the ratio.
    let area2 = |p: [f32; 3], q: [f32; 3], r: [f32; 3]| {
        let (u, v) = (
            [q[0] - p[0], q[1] - p[1], q[2] - p[2]],
            [r[0] - p[0], r[1] - p[1], r[2] - p[2]],
        );
        len(
            [0.0; 3],
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ],
        )
    };
    let mut c = ResizeCheck {
        factor,
        edges: 0,
        degenerate_edges: 0,
        edge_min_m: f32::INFINITY,
        tris: 0,
        degenerate_tris: 0,
        edge_ratio_min: f32::INFINITY,
        edge_ratio_max: f32::NEG_INFINITY,
        edge_ratio_min_all: f32::INFINITY,
        edge_ratio_max_all: f32::NEG_INFINITY,
        area_ratio_min: f32::INFINITY,
        area_ratio_max: f32::NEG_INFINITY,
        area_ratio_min_all: f32::INFINITY,
        area_ratio_max_all: f32::NEG_INFINITY,
    };
    for t in &after.tris {
        let (i, j, k) = (t[0] as usize, t[1] as usize, t[2] as usize);
        for (u, v) in [(i, j), (j, k), (k, i)] {
            let old = len(before[u], before[v]);
            c.edge_min_m = c.edge_min_m.min(old);
            if old <= 0.0 {
                c.degenerate_edges += 1;
                continue; // a genuinely zero-length authored edge has no ratio at all
            }
            let r = len(after.positions[u], after.positions[v]) / old / factor;
            c.edge_ratio_min_all = c.edge_ratio_min_all.min(r);
            c.edge_ratio_max_all = c.edge_ratio_max_all.max(r);
            if old >= EDGE_MEASURABLE_M {
                c.edges += 1;
                c.edge_ratio_min = c.edge_ratio_min.min(r);
                c.edge_ratio_max = c.edge_ratio_max.max(r);
            } else {
                c.degenerate_edges += 1;
            }
        }
        // `area2` is TWICE the area, so the floor below is ½·EDGE_MEASURABLE_M² of real area.
        // Stated because a threshold whose documented value is 2× its effective one is the
        // same class of defect as the exclusion this reports.
        let old = area2(before[i], before[j], before[k]);
        if old <= 0.0 {
            c.degenerate_tris += 1; // a zero-area authored triangle has no ratio at all
            continue;
        }
        let r = area2(after.positions[i], after.positions[j], after.positions[k])
            / old
            / (factor * factor);
        c.area_ratio_min_all = c.area_ratio_min_all.min(r);
        c.area_ratio_max_all = c.area_ratio_max_all.max(r);
        if old >= EDGE_MEASURABLE_M * EDGE_MEASURABLE_M {
            c.tris += 1;
            c.area_ratio_min = c.area_ratio_min.min(r);
            c.area_ratio_max = c.area_ratio_max.max(r);
        } else {
            c.degenerate_tris += 1;
        }
    }
    // Nothing measured in a bucket: report the neutral element rather than ±inf sentinels.
    let d = ResizeCheck::default();
    if c.edges == 0 {
        (c.edge_ratio_min, c.edge_ratio_max) = (d.edge_ratio_min, d.edge_ratio_max);
    }
    if c.tris == 0 {
        (c.area_ratio_min, c.area_ratio_max) = (d.area_ratio_min, d.area_ratio_max);
    }
    if !c.edge_ratio_min_all.is_finite() {
        (c.edge_ratio_min_all, c.edge_ratio_max_all) = (d.edge_ratio_min_all, d.edge_ratio_max_all);
    }
    if !c.area_ratio_min_all.is_finite() {
        (c.area_ratio_min_all, c.area_ratio_max_all) = (d.area_ratio_min_all, d.area_ratio_max_all);
    }
    c
}

/// Convex-hull perimeter of a 2D point set (Andrew's monotone chain).
///
/// Degenerate sets need no special case: two points give a hull walked out and back, so
/// twice the distance, and one point gives zero.
fn hull_perimeter(pts: &mut Vec<[f32; 2]>) -> f32 {
    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let half = |seq: &mut dyn Iterator<Item = [f32; 2]>| -> f32 {
        let mut h: Vec<[f32; 2]> = Vec::new();
        for q in seq {
            while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.windows(2)
            .map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt())
            .sum()
    };
    half(&mut pts.iter().copied()) + half(&mut pts.iter().rev().copied())
}

/// The mesh's cross-section at `height_m`, over a slab of ±`slab_m`. `None` when the slab
/// is empty — the mesh does not reach that height.
pub fn ring(mesh: &Object, height_m: f32, slab_m: f32) -> Option<Ring> {
    let slab: Vec<[f32; 3]> = mesh
        .positions
        .iter()
        .copied()
        .filter(|p| (p[1] - height_m).abs() <= slab_m && p.iter().all(|c| c.is_finite()))
        .collect();
    let bb = Aabb::of(&slab)?;
    let mut xz: Vec<[f32; 2]> = slab.iter().map(|p| [p[0], p[2]]).collect();
    let e = bb.extent();
    // `hull_perimeter` sorts and dedups in place, so read `verts` AFTER it: the gate must
    // count the DISTINCT positions the girth was measured from. See `Ring::verts`.
    let girth_m = hull_perimeter(&mut xz);
    Some(Ring {
        height_m,
        girth_m,
        // Report the slab's real mid-height in y, not the requested one, so the anchor
        // lands on geometry that exists.
        centre: bb.center(),
        aspect: if e[2] > 0.0 {
            e[0] / e[2]
        } else {
            f32::INFINITY
        },
        verts: xz.len(),
    })
}

/// Propose a uniform resize of `garment` onto `body`.
///
/// `girths` are the body's measured girths (see the module docs on why they are passed in
/// rather than measured). `ease` is the fractional allowance — [`FIT_EASE`] is the
/// default; 0.0 puts the garment exactly on the body, which is not what you want.
///
/// The factor is the **largest** demanded by any usable landmark, so the garment can
/// physically go on; every other landmark's residual slack is in the returned
/// [`ResizePlan::fits`]. Nothing is modified — this only reads.
///
/// # Ceiling
///
/// The supporting landmark is chosen by geometry, not by garment type, so it can be
/// wrong in a way no amount of code fixes: measured, `men-shirt`/female's only usable
/// ring is its hem, which is not load-bearing on a shirt. **Show the plan and let the
/// user confirm or override `factor`** rather than applying it silently.
pub fn plan_resize(
    body: &Object,
    garment: &Object,
    girths: &[BodyGirth],
    ease: f32,
) -> Result<ResizePlan, AvatarError> {
    let gbb = crate::avatar::checked_bounds("garment", garment)?;
    crate::avatar::checked_bounds("avatar", body)?;
    let height = gbb.extent()[1];
    let centre = gbb.center();

    let fits: Vec<LandmarkFit> = girths
        .iter()
        .map(|b| {
            let r = ring(garment, b.height_m, RING_SLAB_M);
            let usable = r.filter(|r| {
                r.aspect <= RING_MAX_ASPECT
                    && r.verts >= RING_MIN_VERTS
                    && r.girth_m > 0.0
                    && b.circumference_m > 0.0
                    && !EXCLUDED_LANDMARKS.contains(&b.landmark.as_str())
            });
            LandmarkFit {
                landmark: b.landmark.clone(),
                body_girth_m: b.circumference_m,
                height_m: b.height_m,
                garment: r,
                factor: usable.map(|r| b.circumference_m * (1.0 + ease) / r.girth_m),
            }
        })
        .collect();

    let Some(win) = fits
        .iter()
        .filter(|f| f.factor.is_some())
        .max_by(|a, b| a.factor.unwrap().total_cmp(&b.factor.unwrap()))
    else {
        tracing::info!(
            garment = %garment.name, avatar = %body.name,
            "no usable ring: garment left exactly as authored"
        );
        return Ok(ResizePlan::identity(centre, height, ease, fits));
    };

    let (raw, ring_g, landmark, at) = (
        win.factor.unwrap(),
        win.garment.unwrap(),
        win.landmark.clone(),
        win.height_m,
    );
    let body_girth_m = win.body_girth_m;
    // Enlarge only, and only past the deadband. See RESIZE_LIMIT for the measurements.
    let factor = if raw <= 1.0 + RESIZE_DEADBAND {
        1.0
    } else {
        raw.min(RESIZE_LIMIT)
    };
    let anchor_from = ring_g.centre;
    // Where the ring should end up: the body's own cross-section centre at that height,
    // so the waistband lands on the waist instead of wherever the vendor's mannequin
    // happened to put it. Measuring the body's CENTRE is safe even where measuring its
    // girth is not — T-pose arms are symmetric in x and do not reach in z.
    //
    // **Not when the factor deadbanded to 1.** A resize that scales by nothing must move
    // nothing: the garment is the right size, so where it sits is a placement question and
    // placement is a translation this operation does not own (D-18, D-20). Measured, doing
    // it anyway is a coin flip that can lose badly — on `men-shirt`/male the anchor alone,
    // at factor 1.0000, drove overlap **26.55 % → 39.07 %**.
    let anchor_to = if factor == 1.0 {
        anchor_from
    } else {
        ring(body, at, RING_SLAB_M)
            .map(|r| r.centre)
            .unwrap_or(anchor_from)
    };

    let plan = ResizePlan {
        factor,
        ease,
        supporting: Some(landmark.clone()),
        anchor_from,
        anchor_to,
        fits,
        garment_height_m: height,
    };
    tracing::info!(
        garment = %garment.name, avatar = %body.name,
        factor = plan.factor, raw, ease,
        supporting = %landmark,
        body_girth_m,
        garment_girth_m = ring_g.girth_m,
        length_delta_m = plan.length_delta_m(),
        "garment resize planned"
    );
    Ok(plan)
}

/// Apply `plan` to `garment`'s positions: a uniform scale about
/// [`ResizePlan::anchor_from`], landing that point on [`ResizePlan::anchor_to`].
///
/// Positions only. Normals, UVs, pins, skin and triangles are untouched, which is correct
/// for a similarity — a uniform scale does not change a normal direction.
///
/// The returned [`ResizeCheck`] is measured against the mesh as it was on entry, and is
/// the proof that the garment was not destroyed. Keep the authored mesh if you want to
/// undo: this is in-place and the plan is the record of what was done.
pub fn apply_resize(garment: &mut Object, plan: &ResizePlan) -> ResizeCheck {
    let before = garment.positions.clone();
    let (s, a, b) = (plan.factor, plan.anchor_from, plan.anchor_to);
    // An identity plan must leave the mesh BIT-identical, and the arithmetic below does not:
    // `(p − a) · 1.0 + a` is not the f32 identity when `a ≠ 0`. Measured, it perturbed
    // 17 325 of `men-shirt`'s 70 312 vertices by up to 30 nm — which is where the "identity"
    // rows' 1.2e-5…2.2e-5 edge spread came from, and 30 nm is enough to move
    // `resolve_penetration`'s pass tail by a vertex or two. A uniform scale must show a
    // spread of exactly zero; anything else is a bug, and this was it.
    if !plan.is_identity() {
        for p in &mut garment.positions {
            for k in 0..3 {
                p[k] = (p[k] - a[k]) * s + b[k];
            }
        }
    }

    let c = measure(&before, garment, s);
    // A uniform scale is a similarity or this function has a bug. Not a release
    // assertion: it is O(tris) either way, and the caller gets the numbers regardless.
    debug_assert!(
        c.edges == 0 || c.edge_spread() < 1.0e-3,
        "uniform resize distorted the garment: edge ratio {}..{}",
        c.edge_ratio_min,
        c.edge_ratio_max
    );
    tracing::info!(
        garment = %garment.name,
        factor = s, edges = c.edges,
        degenerate_edges = c.degenerate_edges,
        edge_min_m = c.edge_min_m,
        tris = c.tris,
        degenerate_tris = c.degenerate_tris,
        edge_ratio = ?(c.edge_ratio_min, c.edge_ratio_max),
        edge_ratio_all = ?(c.edge_ratio_min_all, c.edge_ratio_max_all),
        area_ratio = ?(c.area_ratio_min, c.area_ratio_max),
        area_ratio_all = ?(c.area_ratio_min_all, c.area_ratio_max_all),
        "garment resized"
    );
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avatar::tests::body_box;

    /// A cylinder of `n` sides, `h` tall, radius `r`, centred on the y axis at
    /// `(cx, 0, cz)`. Stands in for a tube garment: its hull girth is the polygon's
    /// perimeter, which is analytically `2 n r sin(π/n)`.
    fn tube(name: &str, n: usize, r: f32, y0: f32, h: f32, cx: f32, cz: f32) -> Object {
        let mut o = body_box();
        o.name = name.into();
        o.is_static = false;
        o.positions.clear();
        o.tris.clear();
        for row in 0..2 {
            for i in 0..n {
                let a = std::f32::consts::TAU * i as f32 / n as f32;
                o.positions
                    .push([cx + r * a.cos(), y0 + row as f32 * h, cz + r * a.sin()]);
            }
        }
        for i in 0..n {
            let (a, b) = (i as u32, ((i + 1) % n) as u32);
            o.tris.push([a, b, a + n as u32]);
            o.tris.push([b, b + n as u32, a + n as u32]);
        }
        o
    }

    fn polygon_perimeter(n: usize, r: f32) -> f32 {
        2.0 * n as f32 * r * (std::f32::consts::PI / n as f32).sin()
    }

    fn girth(landmark: &str, c: f32, h: f32) -> BodyGirth {
        BodyGirth {
            landmark: landmark.into(),
            circumference_m: c,
            height_m: h,
        }
    }

    #[test]
    fn hull_perimeter_matches_the_analytic_polygon() {
        for n in [3usize, 4, 8, 64] {
            let mut p: Vec<[f32; 2]> = (0..n)
                .map(|i| {
                    let a = std::f32::consts::TAU * i as f32 / n as f32;
                    [0.3 * a.cos(), 0.3 * a.sin()]
                })
                .collect();
            let got = hull_perimeter(&mut p);
            let want = polygon_perimeter(n, 0.3);
            assert!((got - want).abs() < 1e-4 * want, "n={n}: {got} vs {want}");
        }
        // Interior points must not change the hull — a pleat valley is not a smaller body.
        let mut with_inner = vec![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]];
        let mut without = with_inner[..4].to_vec();
        assert_eq!(
            hull_perimeter(&mut with_inner),
            hull_perimeter(&mut without)
        );
        // Degenerate sets: a segment is walked out and back, a point is zero.
        assert!((hull_perimeter(&mut vec![[0.0, 0.0], [2.0, 0.0]]) - 4.0).abs() < 1e-6);
        assert_eq!(hull_perimeter(&mut vec![[1.0, 1.0]]), 0.0);
    }

    #[test]
    fn ring_reads_the_cross_section_and_its_aspect() {
        let t = tube("tube", 32, 0.1, 0.0, 1.0, 0.0, 0.0);
        let r = ring(&t, 0.0, RING_SLAB_M).expect("bottom ring");
        assert_eq!(r.verts, 32);
        let want = polygon_perimeter(32, 0.1);
        assert!((r.girth_m - want).abs() < 1e-4, "{} vs {want}", r.girth_m);
        assert!((r.aspect - 1.0).abs() < 1e-3, "round: {}", r.aspect);
        // Nothing at half height: the tube has vertices only at its two ends.
        assert!(ring(&t, 0.5, RING_SLAB_M).is_none());
        // A sleeved cross-section reads a wide aspect and is gated out.
        let mut winged = t.clone();
        winged.positions.push([0.9, 0.0, 0.0]);
        winged.positions.push([-0.9, 0.0, 0.0]);
        assert!(ring(&winged, 0.0, RING_SLAB_M).unwrap().aspect > RING_MAX_ASPECT);
    }

    #[test]
    fn factor_matches_the_supporting_girth_with_ease() {
        // Waistband ring at y=1.0 with girth 0.6, body waist 0.75 → 0.75*1.03/0.6.
        let g = tube(
            "skirt",
            64,
            0.6 / polygon_perimeter(64, 1.0),
            0.4,
            0.6,
            0.0,
            0.0,
        );
        let waist = ring(&g, 1.0, RING_SLAB_M).unwrap().girth_m;
        let plan = plan_resize(&body_box(), &g, &[girth("waist", 0.75, 1.0)], FIT_EASE).unwrap();
        let want = 0.75 * (1.0 + FIT_EASE) / waist;
        assert_eq!(plan.supporting.as_deref(), Some("waist"));
        assert!(
            (plan.factor - want).abs() < 1e-5,
            "{} vs {want}",
            plan.factor
        );
        // Ease is a parameter, not a magic number: zero ease lands exactly on the body.
        let tight = plan_resize(&body_box(), &g, &[girth("waist", 0.75, 1.0)], 0.0).unwrap();
        assert!(
            (tight.factor * waist - 0.75).abs() < 1e-5,
            "{}",
            tight.factor
        );
        assert!(tight.factor < plan.factor, "ease must enlarge");
        // The garment gets longer by the same factor. That cost is always reported.
        assert!(
            (plan.length_delta_m() - (plan.factor - 1.0) * 0.6).abs() < 1e-5,
            "{}",
            plan.length_delta_m()
        );
    }

    #[test]
    fn the_tightest_landmark_wins_and_the_rest_report_their_slack() {
        // A cone-ish garment: narrow ring at y=1.0 (girth 0.6), wide at y=0.4 (girth 1.2).
        let mut g = tube("skirt", 64, 0.0, 0.4, 0.6, 0.0, 0.0);
        let n = 64usize;
        let r_top = 0.6 / polygon_perimeter(n, 1.0);
        let r_hem = 1.2 / polygon_perimeter(n, 1.0);
        for (i, p) in g.positions.iter_mut().enumerate() {
            let a = std::f32::consts::TAU * (i % n) as f32 / n as f32;
            let r = if i < n { r_hem } else { r_top };
            *p = [r * a.cos(), p[1], r * a.sin()];
        }
        // Waist demands 0.75*1.03/0.6 = 1.2875; hip demands 1.0*1.03/1.2 = 0.858.
        let plan = plan_resize(
            &body_box(),
            &g,
            &[girth("hip", 1.0, 0.4), girth("waist", 0.75, 1.0)],
            FIT_EASE,
        )
        .unwrap();
        assert_eq!(plan.supporting.as_deref(), Some("waist"));
        assert!((plan.factor - 1.2875).abs() < 2e-3, "{}", plan.factor);
        // The residual at the loose landmark is reported, not silently corrected: the hem
        // ends up ~50 % looser than the hip. That is the honest cost of one factor.
        let hip = plan.fits.iter().find(|f| f.landmark == "hip").unwrap();
        let slack = hip.slack_at(plan.factor).unwrap();
        assert!((slack - 0.545).abs() < 0.01, "hip slack {slack}");
        assert!(plan.fits.iter().all(|f| f.factor.is_some()));
    }

    #[test]
    fn a_garment_already_the_right_size_is_left_alone() {
        // Deadband: a 2 % mismatch is reported and not applied.
        let g = tube(
            "tube",
            64,
            0.75 / polygon_perimeter(64, 1.0),
            0.4,
            0.6,
            0.0,
            0.0,
        );
        let near = ring(&g, 1.0, RING_SLAB_M).unwrap().girth_m / (1.0 + FIT_EASE);
        let plan = plan_resize(
            &body_box(),
            &g,
            &[girth("waist", near * 1.02, 1.0)],
            FIT_EASE,
        )
        .unwrap();
        assert_eq!(plan.factor, 1.0, "deadband");
        assert!(plan.supporting.is_some(), "still reports which landmark");
        assert_eq!(plan.length_delta_m(), 0.0);

        // Out of the deadband but past the clamp: a child's dress is refused, not stretched.
        let huge = plan_resize(&body_box(), &g, &[girth("waist", 10.0, 1.0)], FIT_EASE).unwrap();
        assert_eq!(huge.factor, RESIZE_LIMIT);
        assert_eq!(
            huge.fits[0].factor.map(|f| f > 2.0),
            Some(true),
            "raw demand kept"
        );

        // A garment that is LOOSE everywhere is never shrunk onto the body: measured, both
        // shrinks the corpus proposed made overlap worse (see RESIZE_LIMIT). The demand is
        // still reported so a user can override `factor` by hand.
        let loose = plan_resize(&body_box(), &g, &[girth("waist", 0.05, 1.0)], FIT_EASE).unwrap();
        assert_eq!(loose.factor, 1.0);
        assert!(loose.fits[0].factor.unwrap() < 0.2, "raw demand kept");
        assert_eq!(loose.length_delta_m(), 0.0);
    }

    #[test]
    fn unusable_landmarks_never_set_the_factor() {
        let g = tube("shirt", 64, 0.2, 0.4, 0.6, 0.0, 0.0);
        // A landmark the garment does not reach; a collar; and a sleeve-wide slab. The
        // sleeves carry plenty of vertices, so RING_MIN_VERTS cannot stand in for the
        // aspect gate here — the aspect gate has to be the thing that rejects them.
        let mut winged = g.clone();
        for i in 0..12 {
            let x = 0.4 + 0.15 * i as f32;
            winged.positions.push([x, 0.5, 0.02 * (i % 3) as f32]);
            winged.positions.push([-x, 0.5, 0.02 * (i % 3) as f32]);
        }
        let plan = plan_resize(
            &body_box(),
            &winged,
            &[
                girth("hip", 1.0, 0.05),  // below the garment: no ring
                girth("neck", 0.53, 1.0), // excluded by name
                girth("chest", 1.2, 0.5), // ring exists but the slab caught the sleeves
            ],
            FIT_EASE,
        )
        .unwrap();
        assert!(plan.is_identity(), "{plan:?}");
        assert!(plan.supporting.is_none());
        assert!(plan.fits.iter().all(|f| f.factor.is_none()));
        assert!(plan.fits[0].garment.is_none(), "no ring below the hem");
        assert!(
            plan.fits[1].garment.is_some(),
            "neck ring measured, not used"
        );
        assert!(plan.fits[2].garment.unwrap().aspect > RING_MAX_ASPECT);

        // The sliver: `bike-jacket`'s hem clips the female hip height with 5 vertices whose
        // hull reads 0.1185 m against a 1.0873 m hip. Round, so the aspect gate lets it
        // through; nonsense, so RING_MIN_VERTS must not. Without this gate the plan asks
        // for 9.45, clamps to 1.5, and lengthens the jacket by 37 cm.
        //
        // `copies` is the seam split: a glTF exporter emits one entry per UV/normal corner,
        // so the SAME 5 positions can arrive 4× over. The hull, the girth and the nonsense
        // demand are identical either way, so the gate must count distinct positions — at
        // `copies = 4` a slab-entry count reads 20, clears RING_MIN_VERTS, and applies a
        // clamped 1.5 off a 5-point sliver.
        for copies in [1usize, 4] {
            let mut sliver = g.clone();
            for i in 0..5 {
                let a = std::f32::consts::TAU * i as f32 / 5.0;
                let p = [0.019 * a.cos(), 0.05, 0.019 * a.sin()];
                sliver.positions.extend(std::iter::repeat_n(p, copies));
            }
            let s = ring(&sliver, 0.05, RING_SLAB_M).expect("the sliver is measured");
            assert_eq!(s.verts, 5, "copies={copies} counted duplicate positions");
            assert!(s.aspect < RING_MAX_ASPECT, "sliver aspect {}", s.aspect);
            let sp = plan_resize(
                &body_box(),
                &sliver,
                &[girth("hip", 1.0873, 0.05)],
                FIT_EASE,
            )
            .unwrap();
            assert!(
                sp.is_identity(),
                "a 5-vertex sliver in {copies} copies set the factor: {sp:?}"
            );
            assert_eq!(sp.fits[0].factor, None);
        }
        // `plan` is still the winged identity plan above: applying an identity changes nothing.
        let mut same = winged.clone();
        let c = apply_resize(&mut same, &plan);
        assert_eq!(same.positions, winged.positions);
        assert_eq!(c.edge_ratio_min, c.edge_ratio_max);
    }

    /// An identity plan must leave the mesh **bit**-identical, on a garment nowhere near the
    /// origin and with coordinates that do not round-trip.
    ///
    /// This is not pedantry. `(p − a) · 1.0 + a` is not the f32 identity, and it perturbed
    /// 17 325 of `men-shirt`'s 70 312 vertices by up to 30 nm — which is exactly where the
    /// corpus table's four "identity" rows got their 1.2e-5…2.2e-5 edge spread, reported as
    /// f32 rounding when it was a bug. The maintainer's rule is that a uniform scale shows a
    /// spread of *exactly* zero, and this is that rule as a test.
    ///
    /// Two things have to be pinned separately and the earlier attempt pinned neither: the
    /// early-out in `apply_resize` (mutation: run the arithmetic anyway), and the
    /// `factor == 1.0` guard on `anchor_to` in `plan_resize` (mutation: always take the
    /// body's centre). The second needs a body with a real cross-section at the landmark
    /// height, which `body_box` does not have — its vertices are all at y = 0 or y = 1, so
    /// `ring` returns `None` and the guard is unreachable through it.
    #[test]
    fn an_identity_plan_leaves_the_mesh_bit_identical() {
        // `men-shirt`'s own measured ring centre, and three coordinates that provably do NOT
        // round-trip through (p − a) + a at it. Found by search, not by guessing — the first
        // three sets of round numbers I tried were all exact, which is precisely why an
        // origin-centred fixture proves nothing.
        const ANCHOR: [f32; 3] = [-0.001_572_996_4, 1.301_962_5, -0.003_092_497_6];
        let awkward = Object {
            positions: vec![
                [-0.338_533_94, 1.060_978_4, 0.015_605_058],
                [0.334_613_92, 1.011_011_8, 0.006_685_952],
                [-0.118_373_394, 1.117_036_7, 0.002_764_006_3],
            ],
            tris: vec![[0, 1, 2]],
            ..body_box()
        };
        let mut m = awkward.clone();
        let plan = ResizePlan {
            factor: 1.0,
            anchor_from: ANCHOR,
            anchor_to: ANCHOR,
            ..ResizePlan::identity([0.0; 3], 1.0, FIT_EASE, vec![])
        };
        assert!(plan.is_identity());
        let c = apply_resize(&mut m, &plan);
        assert_eq!(
            m.positions, awkward.positions,
            "an identity plan moved the mesh"
        );
        assert_eq!(c.edge_spread(), 0.0, "{c:?}");
        assert_eq!(c.edge_ratio_min_all, c.edge_ratio_max_all, "{c:?}");
        // Control: the same numbers WITH the arithmetic run are not bit-identical, so the
        // fixture really does exercise the case and the assertion above is not vacuous.
        let mut ctrl = awkward.clone();
        let (a, s) = (plan.anchor_from, 1.0f32);
        for p in &mut ctrl.positions {
            for k in 0..3 {
                p[k] = (p[k] - a[k]) * s + a[k];
            }
        }
        assert_ne!(
            ctrl.positions, awkward.positions,
            "the fixture round-trips exactly — pick nastier coordinates"
        );

        // And the `anchor_to` guard, which needs a body that HAS a ring at the landmark.
        // r = 0.20 is loose against a 1.0873 m hip, so the factor deadbands to 1.0; the
        // garment sits at (0.31, −0.17) and the body at the origin, so dropping the guard
        // translates it by 0.35 m.
        // The body must carry vertices INSIDE the ±RING_SLAB_M slab at the landmark, or
        // `ring` returns `None`, `anchor_to` falls back to `anchor_from`, and the guard is
        // unreachable — which is how two earlier attempts at this test passed under mutation.
        let body = tube("body", 64, 0.30, 0.045, 0.010, 0.0, 0.0);
        assert!(
            ring(&body, 0.05, RING_SLAB_M).is_some(),
            "the body fixture has no ring at the landmark; the guard below is unreachable"
        );
        let mut g = tube("off-centre", 64, 0.20, 0.05, 0.4, 0.31, -0.17);
        let authored = g.positions.clone();
        let plan = plan_resize(&body, &g, &[girth("hip", 1.0873, 0.05)], FIT_EASE).unwrap();
        assert_eq!(plan.factor, 1.0, "{plan:?}");
        assert_eq!(
            plan.anchor_to, plan.anchor_from,
            "a factor of 1 must not translate: {plan:?}"
        );
        apply_resize(&mut g, &plan);
        assert_eq!(g.positions, authored);
    }

    #[test]
    fn apply_resize_is_a_similarity_with_zero_ratio_spread() {
        // r = 0.10 gives a waistband girth of 0.6279 against 0.7725 wanted, i.e. factor
        // 1.2303 — comfortably outside RESIZE_DEADBAND, so the transform really happens.
        let mut g = tube("skirt", 48, 0.10, 0.4, 0.6, 0.31, -0.17);
        let authored = g.clone();
        let plan = plan_resize(&body_box(), &g, &[girth("waist", 0.75, 1.0)], FIT_EASE).unwrap();
        let c = apply_resize(&mut g, &plan);

        // THE hard requirement, as a number: a uniform scale changes no length ratio.
        assert!(c.edges > 0 && c.tris > 0);
        assert!(
            c.edge_spread() < 1e-4,
            "edge ratio {}..{}",
            c.edge_ratio_min,
            c.edge_ratio_max
        );
        assert!((c.edge_ratio_min - 1.0).abs() < 1e-4);
        assert!((c.area_ratio_min - 1.0).abs() < 1e-3 && (c.area_ratio_max - 1.0).abs() < 1e-3);
        assert_eq!(c.degenerate_edges, 0, "nothing degenerate in this fixture");
        assert_eq!(c.degenerate_tris, 0, "nothing degenerate in this fixture");
        // With nothing excluded the unfiltered pair must equal the filtered one exactly, or
        // the two are not measuring the same thing.
        assert_eq!(
            (c.edge_ratio_min_all, c.edge_ratio_max_all),
            (c.edge_ratio_min, c.edge_ratio_max)
        );
        assert_eq!(
            (c.area_ratio_min_all, c.area_ratio_max_all),
            (c.area_ratio_min, c.area_ratio_max)
        );
        assert_eq!(c.tris, authored.tris.len(), "every triangle was measured");

        // A degenerate edge is excluded from the spread and COUNTED, not hidden: without
        // that, f32 cancellation on a 1 µm edge reads as 1 % of fake distortion (measured
        // on men-shirt, 0.988..1.013 against 1e-4 everywhere else).
        let mut hairy = authored.clone();
        let n = hairy.positions.len() as u32;
        let mut p = hairy.positions[0];
        p[0] += 1.0e-6;
        hairy.positions.push(p);
        hairy.tris.push([0, 1, n]);
        let mut scaled = hairy.clone();
        let d = apply_resize(&mut scaled, &plan);
        assert_eq!(d.degenerate_edges, 1, "the 1 µm edge, and only it");
        assert_eq!(d.edges, c.edges + 2, "its two long neighbours still count");
        assert!(d.edge_min_m < EDGE_MEASURABLE_M, "{}", d.edge_min_m);
        assert!(d.edge_spread() < 1e-4, "spread {}", d.edge_spread());
        // The area half of the same exclusion, which used to be silent: the sliver's area is
        // 1.3e-8 m², so it is dropped from `area_ratio_min/max` — and `tris` alone cannot say
        // so. `degenerate_tris` is the number that makes it visible, and the unfiltered pair
        // is what an acceptance criterion has to be asserted on.
        assert_eq!(d.degenerate_tris, 1, "the sliver triangle, and only it");
        assert_eq!(d.tris, c.tris, "the sliver is NOT in the area spread");
        assert_eq!(
            d.tris + d.degenerate_tris,
            hairy.tris.len(),
            "every triangle is either measured or counted"
        );
        // Every edge and every triangle is either measured or counted. This is the property
        // that makes a silent exclusion impossible, and it is the one that was missing.
        assert_eq!(d.edges + d.degenerate_edges, 3 * hairy.tris.len(), "{d:?}");
        // Girth actually lands on the target, which is the whole point.
        let after = ring(&g, plan.anchor_to[1], RING_SLAB_M).unwrap();
        assert!(
            (after.girth_m - 0.75 * (1.0 + FIT_EASE)).abs() < 1e-3,
            "girth {} vs {}",
            after.girth_m,
            0.75 * (1.0 + FIT_EASE)
        );
        // The supporting ring is anchored on the body's own cross-section centre.
        let body_c = ring(&body_box(), 1.0, RING_SLAB_M).unwrap().centre;
        assert!((after.centre[0] - body_c[0]).abs() < 1e-5);
        assert!((after.centre[2] - body_c[2]).abs() < 1e-5);
        // Only positions changed; a uniform scale leaves normals and topology alone.
        assert_eq!(g.tris, authored.tris);
        assert_eq!(g.normals, authored.normals);
        assert_eq!(g.uvs, authored.uvs);

        // Per-axis scaling is what this guards against: the same check on a sheared mesh
        // must fail loudly, or the "spread is zero" assertion above proves nothing.
        let mut sheared = authored.clone();
        for p in &mut sheared.positions {
            p[0] *= 1.4;
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for t in &authored.tris {
            for (u, v) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let d = |m: &Object, i: u32, j: u32| {
                    let (p, q) = (m.positions[i as usize], m.positions[j as usize]);
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
                };
                let r = d(&sheared, u, v) / d(&authored, u, v);
                lo = lo.min(r);
                hi = hi.max(r);
            }
        }
        assert!(
            hi - lo > 0.3,
            "a per-axis scale must show spread: {lo}..{hi}"
        );
    }

    /// The exclusion must never be able to report an undamaged mesh. A collapse confined to
    /// the geometry `EDGE_MEASURABLE_M` filters out is invisible to the filtered pair — which
    /// is the whole defect, stated as one fixture: a sliver whose 100 µm edge is crushed to
    /// 1 % of its length and whose area goes with it, on which `edge_ratio_min` and
    /// `area_ratio_min` both read a perfect 1.0.
    ///
    /// The acceptance criterion ("no edge under 1 % of authored length, no triangle under 1 %
    /// of authored area") must therefore be asserted on the `_all` pair, and the excluded
    /// counts must be reported so `edges`/`tris` can never be mistaken for the totals.
    #[test]
    fn a_collapse_among_the_excluded_geometry_is_still_reported() {
        // A sliver: 0.5 m long, 1 µm of height, so its area is 2.5e-7 m² — under the
        // ½·EDGE_MEASURABLE_M² floor — while two of its three edges are half a metre.
        let before = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0e-4, 1.0e-6, 0.0]];
        let after = Object {
            // The 100 µm edge crushed to 1 µm: ratio 0.01, area ratio 0.01.
            positions: vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0e-6, 1.0e-8, 0.0]],
            // Plus a triangle of exactly zero area, the other branch that must be counted
            // rather than dropped: two of its corners are the same vertex.
            tris: vec![[0, 1, 2], [0, 1, 1]],
            ..body_box()
        };
        let m = measure(&before, &after, 1.0);

        // What was excluded, and it is exactly the collapsed and degenerate geometry.
        assert_eq!((m.edges, m.degenerate_edges), (4, 2), "{m:?}");
        assert_eq!((m.tris, m.degenerate_tris), (0, 2), "{m:?}");
        assert_eq!(m.edges + m.degenerate_edges, 6, "{m:?}");
        assert_eq!(m.tris + m.degenerate_tris, 2, "{m:?}");

        // The filtered pair reports a mesh in perfect health. `area_ratio_min` is 1.0 because
        // NOTHING was measurable, which `Default` documents and `tris == 0` is the only tell.
        assert!(
            m.edge_ratio_min > 0.9,
            "filtered edges saw the collapse: {m:?}"
        );
        assert_eq!(m.area_ratio_min, 1.0, "{m:?}");
        assert_eq!(m.area_ratio_max, 1.0, "{m:?}");

        // The unfiltered pair reports the truth, and it fails the acceptance criterion.
        assert!(
            (m.edge_ratio_min_all - 0.01).abs() < 1.0e-3,
            "the edge collapse is invisible: {m:?}"
        );
        assert!(
            (m.area_ratio_min_all - 0.01).abs() < 1.0e-3,
            "the area collapse is invisible: {m:?}"
        );
        assert!(m.edge_min_m < EDGE_MEASURABLE_M, "{}", m.edge_min_m);
    }

    #[test]
    fn regression_the_measured_skirt_factor_does_not_drift() {
        // Pins the numbers from the maintainer's Blender session on women-skirt/female,
        // reproduced synthetically: a waistband ring of girth 0.6357 against a waist of
        // 0.7306 with 3 % ease was measured to give factor 1.1836, landing the waistband
        // at 0.7526 against a 0.7525 target. samples/ is gitignored, so this stands in
        // for it in CI — if the rule changes, this number moves and the test says so.
        let n = 96usize;
        let g = tube(
            "skirt",
            n,
            0.6357 / polygon_perimeter(n, 1.0),
            0.6,
            0.5409,
            0.0,
            0.0,
        );
        let band = ring(&g, 1.1409, RING_SLAB_M).unwrap();
        assert!((band.girth_m - 0.6357).abs() < 1e-4, "{}", band.girth_m);
        let plan =
            plan_resize(&body_box(), &g, &[girth("waist", 0.7306, 1.1409)], FIT_EASE).unwrap();
        assert!(
            (plan.factor - 1.1836).abs() < 5e-4,
            "factor drifted: {}",
            plan.factor
        );
        let mut resized = g.clone();
        apply_resize(&mut resized, &plan);
        let after = ring(&resized, plan.anchor_to[1], RING_SLAB_M).unwrap();
        assert!((after.girth_m - 0.7525).abs() < 1e-3, "{}", after.girth_m);
        // +16 % on a 0.54 m skirt is 8.7 cm of extra length. Reported, not hidden.
        assert!(
            (plan.length_delta_m() - 0.0993).abs() < 1e-3,
            "{}",
            plan.length_delta_m()
        );
    }

    #[test]
    fn degenerate_inputs_error_cleanly() {
        let body = body_box();
        let mut empty = tube("t", 8, 0.1, 0.0, 1.0, 0.0, 0.0);
        empty.positions.clear();
        empty.tris.clear();
        assert_eq!(
            plan_resize(&body, &empty, &[], FIT_EASE).unwrap_err(),
            AvatarError::Empty {
                role: "garment",
                name: "t".into()
            }
        );
        let mut bad_body = body.clone();
        bad_body.tris[1] = [0, 1, 99];
        let g = tube("t", 8, 0.1, 0.0, 1.0, 0.0, 0.0);
        assert!(matches!(
            plan_resize(&bad_body, &g, &[], FIT_EASE).unwrap_err(),
            AvatarError::BadIndex { index: 99, .. }
        ));
        // No girths at all is not an error — it is an identity plan.
        assert!(plan_resize(&body, &g, &[], FIT_EASE).unwrap().is_identity());
    }
}
