// SPDX-License-Identifier: GPL-2.0-or-later
//! Snapping (guide §6.3).
//!
//! **Tolerance in screen space, resolution in world space.** A candidate is
//! accepted on its distance in logical px, so the snap feels the same at every
//! zoom; the value handed back is world millimetres, so nothing about the view
//! ever reaches stored geometry.
//!
//! Candidates come from the same [`Index`] that hit-testing uses — one spatial
//! structure, not two.

use nguo_pattern::GEOM_ACCURACY;
use vello::kurbo::{Line, ParamCurve, PathEl, PathSeg, Point, Shape, Vec2};

use crate::hit::{Index, Target};
use crate::View;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapKind {
    /// Highest priority first: these are the tie-breaks when two candidates
    /// are the same distance away.
    Endpoint,
    Intersection,
    Midpoint,
    Symmetry,
    /// Aligned in x with a vertex elsewhere in the document — the guide line is
    /// vertical. The axis is in the kind so the overlay knows which way to draw
    /// it without carrying the source position around.
    AlignX,
    /// Aligned in y — a horizontal guide line.
    AlignY,
    Angle,
    Grid,
}

/// Per-type toggles (guide §6.3: "Toggle per-type").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapToggles {
    pub grid: bool,
    pub endpoint: bool,
    pub midpoint: bool,
    pub intersection: bool,
    pub angle: bool,
    pub symmetry: bool,
    /// Covers both [`SnapKind::AlignX`] and [`SnapKind::AlignY`] — one control,
    /// because nobody wants horizontal guides without vertical ones.
    pub align: bool,
}

impl Default for SnapToggles {
    fn default() -> Self {
        SnapToggles {
            grid: true,
            endpoint: true,
            midpoint: true,
            intersection: true,
            angle: true,
            symmetry: true,
            align: true,
        }
    }
}

impl SnapToggles {
    pub const NONE: SnapToggles = SnapToggles {
        grid: false,
        endpoint: false,
        midpoint: false,
        intersection: false,
        angle: false,
        symmetry: false,
        align: false,
    };

    pub fn enabled(&self, kind: SnapKind) -> bool {
        match kind {
            SnapKind::Grid => self.grid,
            SnapKind::Endpoint => self.endpoint,
            SnapKind::Midpoint => self.midpoint,
            SnapKind::Intersection => self.intersection,
            SnapKind::Angle => self.angle,
            SnapKind::Symmetry => self.symmetry,
            SnapKind::AlignX | SnapKind::AlignY => self.align,
        }
    }

    pub fn set(&mut self, kind: SnapKind, on: bool) {
        match kind {
            SnapKind::Grid => self.grid = on,
            SnapKind::Endpoint => self.endpoint = on,
            SnapKind::Midpoint => self.midpoint = on,
            SnapKind::Intersection => self.intersection = on,
            SnapKind::Angle => self.angle = on,
            SnapKind::Symmetry => self.symmetry = on,
            SnapKind::AlignX | SnapKind::AlignY => self.align = on,
        }
    }
}

/// A resolved snap. `world` is what a command must store; `screen_distance` is
/// what the indicator uses to explain itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap {
    pub world: Point,
    pub kind: SnapKind,
    pub screen_distance: f64,
    /// The entity that produced the snap, when there was one.
    pub source: Option<Target>,
}

/// The snap resolver. Cheap to copy; the app owns one and edits it from the UI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snapper {
    pub toggles: SnapToggles,
    /// Screen-space capture radius, logical px.
    pub tolerance_px: f64,
    /// Grid pitch in world mm. `0.0` disables the grid candidate.
    pub grid_mm: f64,
    /// Rubber-band origin for angle snapping — the previous pen point, or the
    /// start of a drag. Without it there is no angle to constrain.
    pub from: Option<Point>,
    /// Mirror line in world mm, from the piece being drawn.
    pub axis: Option<(Point, Point)>,
}

impl Default for Snapper {
    fn default() -> Self {
        Snapper {
            toggles: SnapToggles::default(),
            tolerance_px: 8.0,
            grid_mm: 10.0,
            from: None,
            axis: None,
        }
    }
}

/// The angles an angle-snap constrains to, in radians: 0/45/90 and their
/// negatives, which is every multiple of 45°.
const ANGLE_STEP: f64 = std::f64::consts::FRAC_PI_4;

/// Two candidates within this many screen px of each other are a tie, decided
/// by [`SnapKind`] priority. A tenth of a pixel: below what anyone can aim at,
/// above the float noise of a projection.
const TIE_PX: f64 = 0.1;

impl Snapper {
    /// Resolves `raw` (world mm) against everything enabled.
    ///
    /// `zoom` is logical px per world mm — the only thing the resolver needs
    /// from the view, and the reason the tolerance is a pixel count.
    pub fn snap(&self, raw: Point, zoom: f64, index: &Index) -> Option<Snap> {
        let tolerance_world = self.tolerance_px / zoom;
        // A `Cell` and not a plain local: the alignment block below has to read
        // whether anything has been accepted yet, and it cannot do that while a
        // `FnMut` closure holds `best` borrowed. `Snap` is `Copy`, so this costs
        // nothing.
        let best: std::cell::Cell<Option<Snap>> = std::cell::Cell::new(None);
        let consider = |kind: SnapKind, world: Point, source: Option<Target>| {
            if !self.toggles.enabled(kind) {
                return;
            }
            let screen_distance = (world - raw).hypot() * zoom;
            if screen_distance > self.tolerance_px {
                return;
            }
            let candidate = Snap {
                world,
                kind,
                screen_distance,
                source,
            };
            // Closest wins; equal distance falls back to the `SnapKind` order,
            // so a vertex beats the grid node it happens to sit on.
            let better = match &best.get() {
                None => true,
                Some(b) => {
                    let d = candidate.screen_distance - b.screen_distance;
                    if d < -TIE_PX {
                        true
                    } else if d > TIE_PX {
                        false
                    } else {
                        candidate.kind < b.kind
                    }
                }
            };
            if better {
                best.set(Some(candidate));
            }
        };

        // --- entity candidates --------------------------------------------
        if self.toggles.endpoint {
            for (target, p) in index.vertices_near(raw, tolerance_world) {
                consider(SnapKind::Endpoint, p, Some(target));
            }
        }

        let segments = if self.toggles.midpoint || self.toggles.intersection {
            index.segments_near(raw, tolerance_world)
        } else {
            Vec::new()
        };

        if self.toggles.midpoint {
            for (target, seg) in &segments {
                consider(SnapKind::Midpoint, seg.eval(0.5), Some(*target));
            }
        }

        if self.toggles.intersection {
            for (i, (target, a)) in segments.iter().enumerate() {
                for (_, b) in &segments[i + 1..] {
                    for p in intersections(*a, *b) {
                        consider(SnapKind::Intersection, p, Some(*target));
                    }
                }
            }
        }

        // Alignment guides: share one axis with a vertex anywhere in the
        // document, however far away in the other. Only the perpendicular error
        // is what the tolerance judges, which is what makes a guide catch — and
        // is also why this is a **fallback** rather than another distance-ranked
        // candidate. An alignment candidate's distance is *only* that
        // perpendicular error, so it is nearly always the smallest number in the
        // set and would otherwise beat the corner you were actually reaching
        // for. A real entity in range wins; alignment fills the space between.
        if self.toggles.align && best.get().is_none() {
            for (target, v) in index.vertices_aligned(raw, tolerance_world) {
                consider(SnapKind::AlignX, Point::new(v.x, raw.y), Some(target));
                consider(SnapKind::AlignY, Point::new(raw.x, v.y), Some(target));
            }
        }

        // --- construction candidates --------------------------------------
        if let (true, Some(from)) = (self.toggles.angle, self.from) {
            if let Some(p) = angle_snap(from, raw) {
                consider(SnapKind::Angle, p, None);
            }
        }

        if let (true, Some((a, b))) = (self.toggles.symmetry, self.axis) {
            if a != b {
                let line = Line::new(a, b);
                let t = project_on_line(line, raw);
                consider(SnapKind::Symmetry, line.eval(t.clamp(0.0, 1.0)), None);
            }
        }

        if self.toggles.grid && self.grid_mm > 0.0 {
            let g = self.grid_mm;
            consider(
                SnapKind::Grid,
                Point::new((raw.x / g).round() * g, (raw.y / g).round() * g),
                None,
            );
        }

        best.get()
    }

    /// Convenience: snap a pointer position given in logical widget px.
    pub fn snap_screen(
        &self,
        view: &View,
        screen: Point,
        size: Vec2,
        index: &Index,
    ) -> Option<Snap> {
        self.snap(view.screen_to_world(screen, size), view.zoom, index)
    }
}

/// The projection of `raw` onto whichever multiple-of-45° ray leaves `from`.
///
/// Returns `None` only for a zero-length drag, where there is no direction to
/// constrain.
fn angle_snap(from: Point, raw: Point) -> Option<Point> {
    let d = raw - from;
    let len = d.hypot();
    if len < 1e-12 {
        return None;
    }
    let snapped = (d.atan2() / ANGLE_STEP).round() * ANGLE_STEP;
    let dir = Vec2::new(snapped.cos(), snapped.sin());
    // Project onto the ray, so dragging along it slides freely and only the
    // perpendicular error is what the tolerance judges.
    Some(from + dir * d.dot(dir))
}

/// Parameter of the closest point on `line` to `p`, unclamped.
fn project_on_line(line: Line, p: Point) -> f64 {
    let d = line.p1 - line.p0;
    let len2 = d.dot(d);
    if len2 < 1e-24 {
        0.0
    } else {
        (p - line.p0).dot(d) / len2
    }
}

/// Intersection points of two segments.
///
/// ponytail: one side is flattened to lines and fed to kurbo's exact
/// `PathSeg::intersect_line`, which is exact for the line/line case (every
/// straight pattern edge) and within the flattening tolerance on curve/curve.
/// A true curve/curve solver is a lot of code for sub-`GEOM_ACCURACY` gain on a
/// snap the user is looking at.
fn intersections(a: PathSeg, b: PathSeg) -> Vec<Point> {
    if let (PathSeg::Line(_), PathSeg::Line(l)) = (a, b) {
        return a
            .intersect_line(l)
            .iter()
            .map(|x| a.eval(x.segment_t))
            .collect();
    }
    let mut out = Vec::new();
    let mut prev: Option<Point> = None;
    vello::kurbo::flatten(b.to_path(GEOM_ACCURACY), GEOM_ACCURACY, |el| {
        prev = match el {
            PathEl::MoveTo(p) => Some(p),
            PathEl::LineTo(p) => {
                if let Some(from) = prev {
                    for x in a.intersect_line(Line::new(from, p)) {
                        out.push(a.eval(x.segment_t));
                    }
                }
                Some(p)
            }
            _ => prev,
        };
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::square_doc;
    use crate::hit::Index;

    /// Zoom 1 means 1 px == 1 mm, so a tolerance in px reads directly as mm and
    /// the tests say what they mean.
    const ZOOM: f64 = 1.0;

    fn snapper() -> Snapper {
        Snapper {
            toggles: SnapToggles::NONE,
            tolerance_px: 5.0,
            grid_mm: 10.0,
            from: None,
            axis: None,
        }
    }

    fn index() -> Index {
        let (doc, _, _) = square_doc();
        Index::of_document(&doc)
    }

    #[test]
    fn endpoint_snaps_inside_the_tolerance_and_not_outside() {
        let index = index();
        let mut s = snapper();
        s.toggles.endpoint = true;

        let near = s
            .snap(Point::new(97.0, 2.0), ZOOM, &index)
            .expect("3.6 mm from the corner is inside 5 px at zoom 1");
        assert_eq!(near.kind, SnapKind::Endpoint);
        assert_eq!(near.world, Point::new(100.0, 0.0));

        assert!(
            s.snap(Point::new(90.0, 10.0), ZOOM, &index).is_none(),
            "14 mm away must not snap"
        );
    }

    #[test]
    fn midpoint_snaps_to_the_middle_of_the_edge() {
        let index = index();
        let mut s = snapper();
        s.toggles.midpoint = true;
        let got = s
            .snap(Point::new(51.0, 2.0), ZOOM, &index)
            .expect("near the midpoint of the bottom edge");
        assert_eq!(got.kind, SnapKind::Midpoint);
        assert_eq!(got.world, Point::new(50.0, 0.0));
        assert!(s.snap(Point::new(30.0, 0.0), ZOOM, &index).is_none());
    }

    #[test]
    fn intersection_snaps_where_two_edges_cross() {
        // A square's edges only meet at its corners, so build a genuine
        // crossing: an internal line straight through the middle.
        let (mut doc, piece, _) = square_doc();
        let mut stack = nguo_doc_commands::CommandStack::new();
        let a = doc.piece(piece).unwrap().network.vertices.next_id();
        let b = nguo_pattern::Id::from_raw(a.raw() + 1);
        let e = doc.piece(piece).unwrap().network.edges.next_id();
        stack
            .apply(
                &mut doc,
                nguo_doc_commands::Transaction::new("crossing")
                    .edit(nguo_doc::Edit::AddVertex {
                        piece,
                        id: a,
                        pos: Point::new(-20.0, 50.0),
                    })
                    .edit(nguo_doc::Edit::AddVertex {
                        piece,
                        id: b,
                        pos: Point::new(120.0, 50.0),
                    })
                    .edit(nguo_doc::Edit::AddEdge {
                        piece,
                        id: e,
                        edge: nguo_pattern::Edge {
                            start: a,
                            end: b,
                            geom: nguo_pattern::EdgeGeom::Line,
                            semantic: nguo_pattern::EdgeSemantic::Internal,
                        },
                    }),
            )
            .unwrap();
        let index = Index::of_document(&doc);

        let mut s = snapper();
        s.toggles.intersection = true;
        let got = s
            .snap(Point::new(2.0, 52.0), ZOOM, &index)
            .expect("near where the internal line crosses the left edge");
        assert_eq!(got.kind, SnapKind::Intersection);
        assert!(
            (got.world - Point::new(0.0, 50.0)).hypot() < 1e-6,
            "{got:?}"
        );
    }

    #[test]
    fn angle_snaps_to_multiples_of_45_from_the_origin_point() {
        let index = index();
        let mut s = snapper();
        s.toggles.angle = true;
        s.from = Some(Point::new(0.0, 0.0));

        // 2 mm off a horizontal 100 mm drag: the perpendicular error is inside
        // the tolerance, so it snaps onto the axis.
        let got = s
            .snap(Point::new(100.0, 2.0), ZOOM, &index)
            .expect("2 mm off horizontal");
        assert_eq!(got.kind, SnapKind::Angle);
        assert!(got.world.y.abs() < 1e-9, "{:?}", got.world);

        // 30° is not a multiple of 45 and is far from one at this radius.
        assert!(s.snap(Point::new(86.6, 50.0), ZOOM, &index).is_none());

        // 45° itself is a snap target.
        let diag = s
            .snap(Point::new(70.0, 72.0), ZOOM, &index)
            .expect("near the 45° ray");
        assert!(
            (diag.world.x - diag.world.y).abs() < 1e-9,
            "{:?}",
            diag.world
        );
    }

    #[test]
    fn symmetry_snaps_onto_the_axis() {
        let index = index();
        let mut s = snapper();
        s.toggles.symmetry = true;
        s.axis = Some((Point::new(50.0, -100.0), Point::new(50.0, 200.0)));
        let got = s
            .snap(Point::new(52.0, 30.0), ZOOM, &index)
            .expect("2 mm off the mirror line");
        assert_eq!(got.kind, SnapKind::Symmetry);
        assert_eq!(got.world, Point::new(50.0, 30.0));
        assert!(s.snap(Point::new(70.0, 30.0), ZOOM, &index).is_none());
    }

    #[test]
    fn grid_snaps_to_the_pitch_and_the_result_is_world_space() {
        let index = index();
        let mut s = snapper();
        s.toggles.grid = true;
        let got = s
            .snap(Point::new(203.0, -97.0), ZOOM, &index)
            .expect("3 mm from a 10 mm grid node");
        assert_eq!(got.kind, SnapKind::Grid);
        assert_eq!(got.world, Point::new(200.0, -100.0));
        // World space: nowhere near the widget's pixel range, and unaffected by
        // pan. This is the "zoom is never baked in" invariant, in a snap.
        assert!(s.snap(Point::new(205.0, -95.0), ZOOM, &index).is_none());
    }

    #[test]
    fn every_toggle_actually_disables_its_type() {
        let index = index();
        let all = Snapper {
            toggles: SnapToggles::default(),
            tolerance_px: 5.0,
            grid_mm: 10.0,
            from: Some(Point::new(0.0, 0.0)),
            axis: Some((Point::new(50.0, -100.0), Point::new(50.0, 200.0))),
        };
        // A probe per kind that only that kind can answer.
        let probes = [
            (SnapKind::Endpoint, Point::new(98.5, 1.0)),
            (SnapKind::Midpoint, Point::new(50.0, 1.5)),
            (SnapKind::Symmetry, Point::new(51.5, 33.0)),
            (SnapKind::Grid, Point::new(201.5, -98.5)),
        ];
        for (kind, probe) in probes {
            let mut only = all;
            only.toggles = SnapToggles::NONE;
            only.toggles.set(kind, true);
            assert_eq!(
                only.snap(probe, ZOOM, &index).map(|s| s.kind),
                Some(kind),
                "{kind:?} must snap at {probe:?} when it is the only type on"
            );

            let mut without = all;
            without.toggles.set(kind, false);
            assert!(
                without
                    .snap(probe, ZOOM, &index)
                    .is_none_or(|s| s.kind != kind),
                "{kind:?} still snapped with its toggle off"
            );
        }
        // And with everything off, nothing snaps anywhere.
        let mut none = all;
        none.toggles = SnapToggles::NONE;
        for (_, probe) in probes {
            assert!(none.snap(probe, ZOOM, &index).is_none());
        }
    }

    #[test]
    fn the_tolerance_is_screen_space_so_zoom_changes_what_captures() {
        let index = index();
        let mut s = snapper(); // 5 px
        s.toggles.endpoint = true;
        let probe = Point::new(108.0, 0.0); // 8 mm from the corner at (100, 0)

        assert!(
            s.snap(probe, 1.0, &index).is_none(),
            "8 mm is 8 px at zoom 1: outside 5 px"
        );
        let out = s
            .snap(probe, 0.5, &index)
            .expect("8 mm is 4 px at zoom 0.5: inside 5 px");
        assert_eq!(out.kind, SnapKind::Endpoint);
        // Resolved in WORLD units — the same corner whatever the zoom was.
        assert_eq!(out.world, Point::new(100.0, 0.0));
        assert!((out.screen_distance - 4.0).abs() < 1e-9);
    }

    #[test]
    fn a_closer_candidate_beats_a_higher_priority_one_further_away() {
        let index = index();
        let mut s = snapper();
        s.toggles.endpoint = true;
        s.toggles.grid = true;
        s.tolerance_px = 20.0;
        // (100, 0) is a vertex and also a grid node, so make the grid node
        // strictly closer: probe near (110, 0), 10 mm from the vertex.
        let got = s.snap(Point::new(109.0, 1.0), ZOOM, &index).unwrap();
        assert_eq!(got.kind, SnapKind::Grid);
        assert_eq!(got.world, Point::new(110.0, 0.0));
    }

    #[test]
    fn an_exact_tie_goes_to_the_higher_priority_kind() {
        let index = index();
        let mut s = snapper();
        s.toggles.endpoint = true;
        s.toggles.grid = true;
        // (100, 0) is both a vertex and a 10 mm grid node: same point, same
        // distance. The endpoint must win, or a vertex becomes unsnappable
        // wherever it sits on the grid.
        let got = s.snap(Point::new(101.0, 1.0), ZOOM, &index).unwrap();
        assert_eq!(got.kind, SnapKind::Endpoint);
    }
}
