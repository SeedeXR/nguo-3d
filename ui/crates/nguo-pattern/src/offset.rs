// SPDX-License-Identifier: GPL-2.0-or-later
//! Seam-allowance offset.
//!
//! Offsetting a piece outline outward by a **per-edge** width, with miter,
//! round or bevel joins, and with the self-intersections that concave corners
//! produce removed.
//!
//! # How it works
//! 1. The loop is oriented counter-clockwise so "outward" is unambiguous, then
//!    flipped back at the end to preserve the caller's winding.
//! 2. Each segment is offset on its own. Lines are translated; cubics go
//!    through [`kurbo::offset::CubicOffset`] + [`kurbo::fit_to_bezpath`], which
//!    is the part we emphatically do not reimplement.
//! 3. Consecutive offsets are joined: an arc, a miter point or a bevel at
//!    convex corners; a bevel at concave ones, whose spurious loop step 4
//!    removes.
//! 4. **Cleanup.** The raw offset is flattened and tested for
//!    self-intersections. If there are none — the common case — the *curved*
//!    raw offset is returned untouched. If there are, the flattened path is cut
//!    at every crossing and each resulting arc is kept only when it lies
//!    outside every source segment's band, i.e. when `dist(p, seg_j) >= w_j`
//!    for all `j`. That is the definition of the offset boundary, so the result
//!    cannot self-intersect through a notch.
//!
//! Offsetting by zero is the identity, exactly.

use kurbo::{
    Arc, BezPath, CubicBez, Line, ParamCurve, ParamCurveNearest, PathEl, PathSeg, Point, Shape,
    Vec2,
};

/// How the outer corner between two offset segments is closed.
///
/// `kurbo::Join`, not a parallel enum of our own: the canvas hands the same
/// value to Vello's stroker, and one type means they cannot disagree.
pub use kurbo::Join;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetOptions {
    pub join: Join,
    /// Bevel a miter once it would stick out further than `miter_limit` times
    /// the allowance (the SVG/CSS rule).
    pub miter_limit: f64,
    /// Flattening / curve-fitting accuracy in millimetres.
    pub tolerance: f64,
}

impl Default for OffsetOptions {
    fn default() -> Self {
        OffsetOptions {
            // What a rotary cutter traces around a corner.
            join: Join::Round,
            miter_limit: 4.0,
            // 0.01 mm: an order below any cutter's repeatability.
            tolerance: 1e-2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OffsetError {
    #[error("cannot offset an empty loop")]
    Empty,
    #[error("{segs} segments but {widths} widths")]
    WidthCountMismatch { segs: usize, widths: usize },
    #[error("seam allowance must not be negative, got {0}")]
    NegativeWidth(f64),
    #[error("the loop is not closed: {0} mm between the last end and the first start")]
    NotClosed(f64),
    #[error("the input contains a non-finite coordinate")]
    NonFinite,
}

/// Offset a closed loop outward by `width` on every segment.
pub fn offset_uniform(
    segs: &[PathSeg],
    width: f64,
    opts: &OffsetOptions,
) -> Result<BezPath, OffsetError> {
    offset_closed(segs, &vec![width; segs.len()], opts)
}

/// Offset a closed loop outward, one width per segment.
///
/// `segs` must form a closed chain (`segs[i].end() == segs[i+1].start()`), in
/// either winding. Widths are in millimetres and must be `>= 0`.
pub fn offset_closed(
    segs: &[PathSeg],
    widths: &[f64],
    opts: &OffsetOptions,
) -> Result<BezPath, OffsetError> {
    if segs.is_empty() {
        return Err(OffsetError::Empty);
    }
    if segs.len() != widths.len() {
        return Err(OffsetError::WidthCountMismatch {
            segs: segs.len(),
            widths: widths.len(),
        });
    }
    for &w in widths {
        if !w.is_finite() {
            return Err(OffsetError::NonFinite);
        }
        if w < 0.0 {
            return Err(OffsetError::NegativeWidth(w));
        }
    }
    if segs.iter().any(|s| !s.is_finite()) {
        return Err(OffsetError::NonFinite);
    }
    let gap = (segs[segs.len() - 1].end() - segs[0].start()).hypot();
    if gap > opts.tolerance.max(1e-6) {
        return Err(OffsetError::NotClosed(gap));
    }
    let tol = opts.tolerance.max(1e-9);

    // Offsetting by nothing is the identity, bit for bit.
    if widths.iter().all(|&w| w == 0.0) {
        return Ok(path_of(segs));
    }

    // --- 1. orient counter-clockwise so "outward" means "right normal" ------
    let flipped = path_of(segs).area() < 0.0;
    let (segs, widths): (Vec<PathSeg>, Vec<f64>) = if flipped {
        (
            segs.iter().rev().map(|s| s.reverse()).collect(),
            widths.iter().rev().copied().collect(),
        )
    } else {
        (segs.to_vec(), widths.to_vec())
    };

    // --- 2 + 3. per-segment offsets and joins -------------------------------
    let mut raw: Vec<PathSeg> = Vec::new();
    let n = segs.len();
    for i in 0..n {
        let piece = offset_seg(segs[i], widths[i], tol);
        if let (Some(prev_end), Some(first)) =
            (raw.last().map(|s: &PathSeg| s.end()), piece.first())
        {
            add_join(
                &mut raw,
                segs[i].start(),
                prev_end,
                first.start(),
                tangent_end(segs[(i + n - 1) % n]),
                tangent_start(segs[i]),
                widths[(i + n - 1) % n],
                widths[i],
                opts,
                tol,
            );
        }
        raw.extend(piece);
    }
    if raw.is_empty() {
        return Ok(BezPath::new());
    }
    // Close the ring with the join at the seam between the last and first segment.
    let (last_end, first_start) = (raw[raw.len() - 1].end(), raw[0].start());
    let mut closing = Vec::new();
    add_join(
        &mut closing,
        segs[0].start(),
        last_end,
        first_start,
        tangent_end(segs[n - 1]),
        tangent_start(segs[0]),
        widths[n - 1],
        widths[0],
        opts,
        tol,
    );
    raw.extend(closing);

    // --- 4. cleanup ---------------------------------------------------------
    let out = clean(&raw, &segs, &widths, tol);
    Ok(if flipped { out.reverse_subpaths() } else { out })
}

// --- helpers ---------------------------------------------------------------

fn path_of(segs: &[PathSeg]) -> BezPath {
    let mut p = BezPath::new();
    if let Some(first) = segs.first() {
        p.move_to(first.start());
        for s in segs {
            p.push(s.as_path_el());
        }
        p.close_path();
    }
    p
}

/// Right-hand normal of `t`, unit length. For a counter-clockwise loop this
/// points out of the shape.
fn right_normal(t: Vec2) -> Vec2 {
    let h = t.hypot();
    if h == 0.0 {
        Vec2::ZERO
    } else {
        Vec2::new(t.y / h, -t.x / h)
    }
}

fn tangent_start(s: PathSeg) -> Vec2 {
    match s {
        PathSeg::Line(l) => l.p1 - l.p0,
        PathSeg::Quad(q) => nonzero(q.p1 - q.p0, q.p2 - q.p0),
        PathSeg::Cubic(c) => nonzero(c.p1 - c.p0, nonzero(c.p2 - c.p0, c.p3 - c.p0)),
    }
}

fn tangent_end(s: PathSeg) -> Vec2 {
    match s {
        PathSeg::Line(l) => l.p1 - l.p0,
        PathSeg::Quad(q) => nonzero(q.p2 - q.p1, q.p2 - q.p0),
        PathSeg::Cubic(c) => nonzero(c.p3 - c.p2, nonzero(c.p3 - c.p1, c.p3 - c.p0)),
    }
}

fn nonzero(a: Vec2, b: Vec2) -> Vec2 {
    if a.hypot2() > 0.0 {
        a
    } else {
        b
    }
}

/// One source segment, offset by `w` to its right.
fn offset_seg(s: PathSeg, w: f64, tol: f64) -> Vec<PathSeg> {
    if w == 0.0 {
        return vec![s];
    }
    match s {
        PathSeg::Line(l) => {
            let n = right_normal(l.p1 - l.p0);
            if n == Vec2::ZERO {
                return Vec::new();
            }
            vec![PathSeg::Line(Line::new(l.p0 + n * w, l.p1 + n * w))]
        }
        PathSeg::Quad(q) => offset_one_cubic(q.raise(), w, tol),
        PathSeg::Cubic(c) => offset_one_cubic(c, w, tol),
    }
}

fn offset_one_cubic(c: CubicBez, w: f64, tol: f64) -> Vec<PathSeg> {
    // `kurbo::offset::offset_cubic` offsets along `turn_90()`, the *left*
    // normal, so the outward direction of a counter-clockwise loop is negative.
    let mut out = BezPath::new();
    kurbo::offset::offset_cubic(c, -w, tol, &mut out);
    out.segments().collect()
}

/// Bridge the gap between the end of one offset and the start of the next.
#[allow(clippy::too_many_arguments)]
fn add_join(
    out: &mut Vec<PathSeg>,
    corner: Point,
    from: Point,
    to: Point,
    t_in: Vec2,
    t_out: Vec2,
    w_in: f64,
    w_out: f64,
    opts: &OffsetOptions,
    tol: f64,
) {
    if (to - from).hypot() <= tol * 0.5 {
        return;
    }
    let convex = t_in.cross(t_out) > 0.0;
    // A concave corner's offsets already overlap; any connector we draw here is
    // spurious and gets cut out by `clean`, so the cheapest one will do.
    if !convex {
        out.push(PathSeg::Line(Line::new(from, to)));
        return;
    }
    match opts.join {
        Join::Bevel => out.push(PathSeg::Line(Line::new(from, to))),
        Join::Miter => {
            let apex = Line::new(from, from + t_in).crossing_point(Line::new(to, to - t_out));
            let w = w_in.max(w_out);
            match apex {
                Some(p) if (p - corner).hypot() <= opts.miter_limit.max(1.0) * w => {
                    out.push(PathSeg::Line(Line::new(from, p)));
                    out.push(PathSeg::Line(Line::new(p, to)));
                }
                _ => out.push(PathSeg::Line(Line::new(from, to))),
            }
        }
        Join::Round => {
            // An arc only exists when both sides are the same distance away;
            // otherwise the "corner" is a width step and a chord is correct.
            if (w_in - w_out).abs() > tol || w_in <= 0.0 {
                out.push(PathSeg::Line(Line::new(from, to)));
                return;
            }
            let a0 = (from - corner).atan2();
            let a1 = (to - corner).atan2();
            let mut sweep = a1 - a0;
            while sweep > std::f64::consts::PI {
                sweep -= 2.0 * std::f64::consts::PI;
            }
            while sweep < -std::f64::consts::PI {
                sweep += 2.0 * std::f64::consts::PI;
            }
            let arc = Arc::new(corner, (w_in, w_in), a0, sweep, 0.0);
            let mut p = from;
            arc.to_cubic_beziers(tol, |p1, p2, p3| {
                out.push(PathSeg::Cubic(CubicBez::new(p, p1, p2, p3)));
                p = p3;
            });
            if (p - to).hypot() > tol * 0.5 {
                out.push(PathSeg::Line(Line::new(p, to)));
            }
        }
    }
}

/// Flatten, cut at self-intersections, drop everything that is inside the
/// source's offset band, and stitch the survivors back together.
///
/// Returns the untouched curved input when it does not self-intersect.
fn clean(raw: &[PathSeg], src: &[PathSeg], widths: &[f64], tol: f64) -> BezPath {
    let pts = flatten_closed(raw, tol);
    if pts.len() < 3 {
        return path_of(raw);
    }
    let cuts = self_intersections(&pts);
    if cuts.is_empty() {
        return path_of(raw);
    }

    // Splice the crossings into the polyline, remembering where the cuts land.
    let n = pts.len();
    let mut aug: Vec<Point> = Vec::with_capacity(n + cuts.len() * 2);
    let mut is_cut: Vec<bool> = Vec::with_capacity(aug.capacity());
    for (i, &pt) in pts.iter().enumerate() {
        aug.push(pt);
        is_cut.push(false);
        let mut here: Vec<(f64, Point)> = cuts
            .iter()
            .filter(|c| c.0 == i)
            .map(|c| (c.1, c.2))
            .collect();
        here.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, p) in here {
            aug.push(p);
            is_cut.push(true);
        }
    }

    let cut_idx: Vec<usize> = (0..aug.len()).filter(|&i| is_cut[i]).collect();
    if cut_idx.is_empty() {
        return path_of(raw);
    }

    // Arcs run from one crossing to the next, wrapping around.
    let m = aug.len();
    let mut arcs: Vec<Vec<Point>> = Vec::with_capacity(cut_idx.len());
    for k in 0..cut_idx.len() {
        let start = cut_idx[k];
        let end = cut_idx[(k + 1) % cut_idx.len()];
        let mut arc = vec![aug[start]];
        let mut i = (start + 1) % m;
        loop {
            arc.push(aug[i]);
            if i == end {
                break;
            }
            i = (i + 1) % m;
        }
        arcs.push(arc);
    }

    let keep: Vec<bool> = arcs
        .iter()
        .map(|a| is_outside(polyline_midpoint(a), src, widths, tol))
        .collect();

    // Start just after a discarded arc so a surviving run is never split
    // across the wrap-around.
    let rotate = keep.iter().position(|k| !k).map_or(0, |d| d + 1);
    let mut out = BezPath::new();
    let mut cur: Vec<Point> = Vec::new();
    let eps = tol.max(1e-9);
    for step in 0..arcs.len() {
        let k = (rotate + step) % arcs.len();
        if !keep[k] {
            // Do NOT break the subpath here: removing a loop is exactly the
            // case where the arcs on either side meet at the same crossing and
            // must be stitched together. Whether they do is decided below by
            // comparing endpoints.
            continue;
        }
        match cur.last() {
            Some(&last) if (last - arcs[k][0]).hypot() <= eps => {
                cur.extend_from_slice(&arcs[k][1..])
            }
            Some(_) => {
                flush(&mut out, &mut cur);
                cur.extend_from_slice(&arcs[k]);
            }
            None => cur.extend_from_slice(&arcs[k]),
        }
    }
    flush(&mut out, &mut cur);
    out
}

fn flush(out: &mut BezPath, cur: &mut Vec<Point>) {
    if cur.len() >= 3 {
        out.move_to(cur[0]);
        for p in &cur[1..] {
            out.line_to(*p);
        }
        out.close_path();
    }
    cur.clear();
}

/// Does `path` cross itself?
///
/// Flattened test at `tolerance`, per closed subpath. Exposed because "is this
/// outline simple?" is a validation question the sewing and triangulation
/// stages ask too (guide §10, §11) — and because it is what pins the offset's
/// own cleanup down in tests.
pub fn is_simple(path: &BezPath, tolerance: f64) -> bool {
    closed_subpaths(path, tolerance)
        .iter()
        .all(|s| self_intersections(s).is_empty())
}

/// Flatten `path` into one point list per closed subpath, first point not
/// repeated at the end.
fn closed_subpaths(path: &BezPath, tol: f64) -> Vec<Vec<Point>> {
    let eps = tol.max(1e-9) * 1e-3;
    let mut subs: Vec<Vec<Point>> = Vec::new();
    let mut cur: Vec<Point> = Vec::new();
    let finish = |cur: &mut Vec<Point>, subs: &mut Vec<Vec<Point>>| {
        while cur.len() > 1 && (cur[0] - cur[cur.len() - 1]).hypot() <= eps {
            cur.pop();
        }
        if cur.len() > 2 {
            subs.push(std::mem::take(cur));
        }
        cur.clear();
    };
    kurbo::flatten(path, tol, |el| match el {
        PathEl::MoveTo(p) => {
            finish(&mut cur, &mut subs);
            cur.push(p);
        }
        PathEl::LineTo(p) => cur.push(p),
        PathEl::ClosePath => finish(&mut cur, &mut subs),
        _ => {}
    });
    finish(&mut cur, &mut subs);
    subs
}

fn flatten_closed(raw: &[PathSeg], tol: f64) -> Vec<Point> {
    let mut pts: Vec<Point> = Vec::new();
    kurbo::flatten(path_of(raw), tol, |el| match el {
        PathEl::MoveTo(p) | PathEl::LineTo(p) => pts.push(p),
        _ => {}
    });
    let eps = tol.max(1e-9) * 1e-3;
    pts.dedup_by(|a, b| (*a - *b).hypot() <= eps);
    while pts.len() > 1 && (pts[0] - pts[pts.len() - 1]).hypot() <= eps {
        pts.pop();
    }
    pts
}

/// Crossings of a closed polyline with itself, as `(segment index, t, point)`.
///
/// ponytail: O(n^2) over the flattened offset. At pattern scale (a few thousand
/// flattened segments) that is a handful of milliseconds and it is only reached
/// when the offset actually self-intersects. Swap in a Bentley-Ottmann sweep if
/// a profile ever says otherwise.
fn self_intersections(pts: &[Point]) -> Vec<(usize, f64, Point)> {
    let n = pts.len();
    let mut out = Vec::new();
    for i in 0..n {
        let (a0, a1) = (pts[i], pts[(i + 1) % n]);
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue; // adjacent across the closing edge
            }
            let (b0, b1) = (pts[j], pts[(j + 1) % n]);
            if let Some((t, u, p)) = cross(a0, a1, b0, b1) {
                out.push((i, t, p));
                out.push((j, u, p));
            }
        }
    }
    out
}

fn cross(a0: Point, a1: Point, b0: Point, b1: Point) -> Option<(f64, f64, Point)> {
    let r = a1 - a0;
    let s = b1 - b0;
    let denom = r.cross(s);
    if denom.abs() < 1e-12 {
        return None;
    }
    let d = b0 - a0;
    let t = d.cross(s) / denom;
    let u = d.cross(r) / denom;
    const E: f64 = 1e-9;
    (t > E && t < 1.0 - E && u > E && u < 1.0 - E).then(|| (t, u, a0 + r * t))
}

fn polyline_midpoint(pts: &[Point]) -> Point {
    let total: f64 = pts.windows(2).map(|w| (w[1] - w[0]).hypot()).sum();
    if total == 0.0 {
        return pts[0];
    }
    let mut walked = 0.0;
    for w in pts.windows(2) {
        let len = (w[1] - w[0]).hypot();
        if walked + len >= total * 0.5 {
            let f = if len == 0.0 {
                0.0
            } else {
                (total * 0.5 - walked) / len
            };
            return w[0] + (w[1] - w[0]) * f;
        }
        walked += len;
    }
    pts[pts.len() - 1]
}

/// Is `p` on or outside every source segment's offset band? That is precisely
/// the condition for lying on the offset boundary rather than inside the swept
/// region, and it is what makes a concave notch come out clean.
fn is_outside(p: Point, src: &[PathSeg], widths: &[f64], tol: f64) -> bool {
    let slack = tol * 2.0 + 1e-9;
    src.iter().zip(widths).all(|(s, &w)| {
        let limit = w - slack;
        limit <= 0.0 || s.nearest(p, tol * 0.1).distance_sq >= limit * limit
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::Rect;

    fn segs_of(p: &BezPath) -> Vec<PathSeg> {
        p.segments().collect()
    }

    fn rect_loop(w: f64, h: f64) -> Vec<PathSeg> {
        segs_of(&Rect::new(0.0, 0.0, w, h).to_path(1e-9))
    }

    #[test]
    fn offset_by_zero_is_the_identity() {
        let s = rect_loop(100.0, 50.0);
        let out = offset_uniform(&s, 0.0, &OffsetOptions::default()).unwrap();
        assert_eq!(segs_of(&out), s);
    }

    /// The invariant that actually pins the algorithm down: growing a convex
    /// shape by `w` with round joins adds `perimeter*w + pi*w^2` of area.
    #[test]
    fn convex_round_offset_area_matches_theory() {
        for &(w, h) in &[(100.0, 50.0), (30.0, 30.0), (200.0, 7.0)] {
            for &d in &[1.0, 5.0, 12.5] {
                let s = rect_loop(w, h);
                let base = path_of(&s).area().abs();
                let perim = 2.0 * (w + h);
                let out = offset_uniform(&s, d, &OffsetOptions::default()).unwrap();
                let got = out.area().abs();
                let want = base + perim * d + std::f64::consts::PI * d * d;
                assert!(
                    (got - want).abs() < want * 1e-3,
                    "{w}x{h} by {d}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn convex_offset_area_holds_for_a_polygon_too() {
        // Regular hexagon, circumradius 50.
        let mut p = BezPath::new();
        let r = 50.0;
        for i in 0..6 {
            let a = i as f64 * std::f64::consts::PI / 3.0;
            let pt = Point::new(r * a.cos(), r * a.sin());
            if i == 0 {
                p.move_to(pt)
            } else {
                p.line_to(pt)
            }
        }
        p.close_path();
        let s = segs_of(&p);
        let base = p.area().abs();
        let perim: f64 = s.iter().map(|x| (x.end() - x.start()).hypot()).sum();
        let d = 8.0;
        let out = offset_uniform(&s, d, &OffsetOptions::default()).unwrap();
        let want = base + perim * d + std::f64::consts::PI * d * d;
        assert!(
            (out.area().abs() - want).abs() < want * 1e-3,
            "got {}, want {want}",
            out.area().abs()
        );
    }

    #[test]
    fn winding_does_not_change_the_result() {
        let s = rect_loop(80.0, 40.0);
        let rev: Vec<PathSeg> = s.iter().rev().map(|x| x.reverse()).collect();
        let a = offset_uniform(&s, 6.0, &OffsetOptions::default()).unwrap();
        let b = offset_uniform(&rev, 6.0, &OffsetOptions::default()).unwrap();
        assert!((a.area().abs() - b.area().abs()).abs() < 1e-6);
        // ... but the caller's winding is preserved.
        assert!(a.area() * b.area() < 0.0, "winding should be mirrored");
    }

    /// A notch cut into the outline: the naive offset crosses itself here.
    fn notched() -> Vec<PathSeg> {
        let mut p = BezPath::new();
        p.move_to((0.0, 0.0));
        p.line_to((100.0, 0.0));
        p.line_to((100.0, 100.0));
        p.line_to((60.0, 100.0));
        p.line_to((50.0, 20.0)); // deep, narrow notch
        p.line_to((40.0, 100.0));
        p.line_to((0.0, 100.0));
        p.close_path();
        segs_of(&p)
    }

    #[test]
    fn a_concave_notch_does_not_produce_self_intersections() {
        let s = notched();
        assert!(
            is_simple(&path_of(&s), 1e-3),
            "the source itself must be simple"
        );
        for &d in &[1.0, 3.0, 6.0, 10.0] {
            let out = offset_uniform(&s, d, &OffsetOptions::default()).unwrap();
            assert!(is_simple(&out, 1e-3), "offset by {d} self-intersects");
            assert!(out.area().abs() > 0.0, "offset by {d} vanished");
        }
    }

    #[test]
    fn offsetting_a_notch_grows_the_area() {
        let s = notched();
        let base = path_of(&s).area().abs();
        let mut prev = base;
        for &d in &[1.0, 2.0, 4.0] {
            let a = offset_uniform(&s, d, &OffsetOptions::default())
                .unwrap()
                .area()
                .abs();
            assert!(a > prev, "offset by {d} shrank: {a} <= {prev}");
            prev = a;
        }
    }

    #[test]
    fn per_edge_widths_are_honoured() {
        let s = rect_loop(100.0, 100.0);
        // Allowance only on the bottom edge.
        let out = offset_closed(&s, &[10.0, 0.0, 0.0, 0.0], &OffsetOptions::default()).unwrap();
        let bb = out.bounding_box();
        assert!(
            bb.y1 - bb.y0 > 100.0,
            "the offset edge should push the box out"
        );
        assert!(bb.y1 - bb.y0 < 100.0 + 10.0 + 1e-6);
    }

    #[test]
    fn curved_edges_stay_curved_when_nothing_crosses() {
        let mut p = BezPath::new();
        p.move_to((0.0, 0.0));
        p.curve_to((30.0, 40.0), (70.0, 40.0), (100.0, 0.0));
        p.line_to((100.0, -50.0));
        p.line_to((0.0, -50.0));
        p.close_path();
        let out = offset_uniform(&segs_of(&p), 5.0, &OffsetOptions::default()).unwrap();
        assert!(
            out.segments().any(|s| matches!(s, PathSeg::Cubic(_))),
            "a clean offset must not be flattened"
        );
    }

    #[test]
    fn miter_and_bevel_bracket_round() {
        let s = rect_loop(100.0, 100.0);
        let d = 10.0;
        let a = |join| {
            offset_uniform(
                &s,
                d,
                &OffsetOptions {
                    join,
                    tolerance: 1e-3,
                    ..OffsetOptions::default()
                },
            )
            .unwrap()
            .area()
            .abs()
        };
        let bevel = a(Join::Bevel);
        let round = a(Join::Round);
        let miter = a(Join::Miter);
        assert!(bevel < round && round < miter, "{bevel} {round} {miter}");
        // Square corners: miter adds the full w^2 per corner, bevel half of it.
        let base = 100.0 * 100.0 + 4.0 * 100.0 * d;
        assert!((miter - (base + 4.0 * d * d)).abs() < 1e-3);
        assert!((bevel - (base + 2.0 * d * d)).abs() < 1e-3);
    }

    #[test]
    fn rejects_bad_input() {
        let s = rect_loop(10.0, 10.0);
        assert!(matches!(
            offset_closed(&[], &[], &OffsetOptions::default()),
            Err(OffsetError::Empty)
        ));
        assert!(matches!(
            offset_closed(&s, &[1.0], &OffsetOptions::default()),
            Err(OffsetError::WidthCountMismatch { .. })
        ));
        assert!(matches!(
            offset_uniform(&s, -1.0, &OffsetOptions::default()),
            Err(OffsetError::NegativeWidth(_))
        ));
        assert!(matches!(
            offset_closed(&s[..2], &[1.0, 1.0], &OffsetOptions::default()),
            Err(OffsetError::NotClosed(_))
        ));
    }
}
