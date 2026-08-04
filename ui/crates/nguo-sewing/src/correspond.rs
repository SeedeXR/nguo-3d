// SPDX-License-Identifier: GPL-2.0-or-later
//! Segment correspondence: which sub-span of one seam side sews to which
//! sub-span of the other (`frontend-guide.md` §10, the 🔸 CAN EXPAND item).
//!
//! # The whole idea in one paragraph
//!
//! A [`SeamSide`] is an ordered chain of pattern edges plus the normalised
//! sub-span of that chain that actually sews. Resolve it against a
//! [`SimMesh`] and you get the mesh vertices lying on it, each with its
//! position along the sewn span in **normalised arc length**. Two sides
//! resolved that way are two monotone lists over the same `0..=1` parameter,
//! so pairing them is a merge, not a search — and because the parameter is
//! *normalised* arc length, two sides of unequal length pair proportionally.
//! That is precisely what makes a 600 mm edge sew to a 500 mm one as **ease**
//! (or gathering) instead of an error.
//!
//! # Why the stitch is barycentric
//!
//! The two sides never have the same vertex count: they are triangulated
//! independently. `engine_core::stitch` sews a *weighted barycentric point* to
//! a weighted barycentric point, so the denser side's vertices are sewn to
//! **interpolated** points on the sparser side rather than snapped to its
//! nearest vertex. Snapping would quantise the seam to the coarser mesh and
//! visibly scallop it; interpolation costs one lerp.
//!
//! ponytail: stitches are emitted from the denser side only, one per vertex.
//! Emitting both directions would double the seam's stiffness for no extra
//! coverage — every point of the sparser side already lies between two
//! stitches. Upgrade if a seam ever needs asymmetric stiffness per side.

use kurbo::{ParamCurve, ParamCurveArclen};
use nguo_doc::{SeamId, SeamSide};
use nguo_pattern::{EdgeId, VectorNetwork, GEOM_ACCURACY};
use nguo_triangulate::SimMesh;

/// Ease ratio (`len_b / len_a`) beyond which a length mismatch stops being ease
/// and becomes an actionable error.
///
/// 1.25 = 25 % of gathering. Chosen because it is the widest ratio a plain
/// seam absorbs without a gather/ruffle spec of its own (an authored gather is
/// 1.5–3.0 and needs stitch density, not a tolerance), and because the sewn
/// result at 1.25 is still a seam rather than a pleat.
pub const EASE_LIMIT: f64 = 1.25;

/// Anything below this is one point, not a span: a seam side whose sewn span is
/// shorter than the pattern accuracy has no direction and cannot be corresponded.
pub const MIN_SPAN_MM: f64 = 10.0 * GEOM_ACCURACY;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SeamError {
    #[error(transparent)]
    Network(#[from] nguo_pattern::NetworkError),

    #[error(
        "seam side on piece {piece} references edge {edge:?}, which contributed no mesh vertices \
         — it is a Grain marker or was not part of the triangulated piece"
    )]
    NoChain { piece: String, edge: EdgeId },

    #[error("seam side on piece {piece} has no edges; a side must name at least one edge")]
    EmptySide { piece: String },

    #[error(
        "seam side on piece {piece} is discontinuous: edge {edge:?} does not start where the \
         previous edge ended — the (edge, reversed) flags are wrong, so the side has no single \
         direction to sew along"
    )]
    BrokenChain { piece: String, edge: EdgeId },

    #[error(
        "seam side on piece {piece} sews a span of {span_mm:.3} mm, which is not a span — check \
         SeamSide::span"
    )]
    DegenerateSpan { piece: String, span_mm: f64 },
}

/// One seam side resolved onto mesh vertices.
#[derive(Debug, Clone, PartialEq)]
pub struct SideSamples {
    /// `(mesh vertex, normalised position along the sewn span)`, ascending.
    /// `0.0` and `1.0` are present whenever the span reaches the chain's ends.
    pub vertices: Vec<(u32, f64)>,
    /// Arc length of the **sewn span**, millimetres.
    pub len_mm: f64,
}

/// One sewing constraint in `engine_core::stitch`'s own shape: a weighted
/// barycentric point on one piece sewn to a weighted barycentric point on
/// another.
///
/// `index[0..3]`/`weight[0..3]` is the source and `index[3..6]`/`weight[3..6]`
/// the target; each half's weights sum to 1. Indices are **garment-local** —
/// `crate::Garment` has already offset them into its concatenated vertex space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stitch6 {
    pub index: [u32; 6],
    pub weight: [f32; 6],
}

impl Stitch6 {
    /// Sew a single vertex to a point `w` of the way from `i0` to `i1`.
    pub fn vertex_to_edge(v: u32, i0: u32, i1: u32, w: f32) -> Stitch6 {
        Stitch6 {
            index: [v, v, v, i0, i1, i1],
            weight: [1.0, 0.0, 0.0, 1.0 - w, w, 0.0],
        }
    }

    /// Every vertex this stitch touches, source side first.
    pub fn vertices(&self) -> [u32; 6] {
        self.index
    }
}

/// The correspondence between two resolved sides.
#[derive(Debug, Clone, PartialEq)]
pub struct Correspondence {
    pub stitches: Vec<Stitch6>,
    pub len_a_mm: f64,
    pub len_b_mm: f64,
}

impl Correspondence {
    /// `len(b) / len(a)`: what [`nguo_doc::Seam::ease_ratio`] declares the seam
    /// is built for. `> 1.0` means side `b` is gathered into side `a`.
    pub fn ease(&self) -> f64 {
        if self.len_a_mm > 0.0 {
            self.len_b_mm / self.len_a_mm
        } else {
            f64::INFINITY
        }
    }

    /// Ease as a signed fraction, e.g. `0.032` for 3.2 % of gathering.
    pub fn ease_fraction(&self) -> f64 {
        self.ease() - 1.0
    }
}

/// Arc length of a seam side's **sewn span**, in pattern millimetres.
///
/// Pure document geometry — no mesh — so `crate::validate` can surface a length
/// mismatch the moment an edge moves, without triangulating anything.
pub fn side_length_mm(net: &VectorNetwork, side: &SeamSide) -> Result<f64, SeamError> {
    let piece = format!("{:?}", side.piece);
    if side.edges.is_empty() {
        return Err(SeamError::EmptySide { piece });
    }
    let mut total = 0.0;
    for &(edge, _) in &side.edges {
        total += net.arc_length(edge)?;
    }
    let (lo, hi) = span(side);
    Ok(total * (hi - lo))
}

/// Is the chain contiguous under its own `reversed` flags?
///
/// This *is* the direction check at document level: a side whose edge `i` does
/// not end where edge `i+1` starts has no single direction to sew along, and
/// the usual cause is a flag that should have been flipped.
pub fn check_chain(net: &VectorNetwork, side: &SeamSide) -> Result<(), SeamError> {
    let piece = format!("{:?}", side.piece);
    let mut prev_end = None;
    for &(edge, reversed) in &side.edges {
        let e = net
            .edges
            .get(edge)
            .ok_or(nguo_pattern::NetworkError::NoSuchEdge(edge))?;
        let (start, end) = if reversed {
            (e.end, e.start)
        } else {
            (e.start, e.end)
        };
        if let Some(prev) = prev_end {
            if prev != start {
                return Err(SeamError::BrokenChain { piece, edge });
            }
        }
        prev_end = Some(end);
    }
    Ok(())
}

/// Resolve a seam side onto the mesh vertices lying along it.
///
/// `mesh` must be the triangulation of `side.piece`'s own network, and the
/// returned indices are that mesh's own — the caller offsets them.
pub fn side_samples(
    net: &VectorNetwork,
    mesh: &SimMesh,
    side: &SeamSide,
) -> Result<SideSamples, SeamError> {
    let piece = format!("{:?}", side.piece);
    if side.edges.is_empty() {
        return Err(SeamError::EmptySide { piece });
    }
    check_chain(net, side)?;

    // Cumulative arc length along the whole chain, and every mesh vertex on it.
    let mut walked = 0.0f64;
    let mut along: Vec<(u32, f64)> = Vec::new();
    for &(edge, reversed) in &side.edges {
        let seg = net.seg(edge)?;
        let edge_len = seg.arclen(GEOM_ACCURACY);
        let chain = mesh.chain(edge).ok_or_else(|| SeamError::NoChain {
            piece: piece.clone(),
            edge,
        })?;

        // Curve parameter -> arc length. For a line these are proportional; for
        // a cubic they are not, and correspondence is defined on arc length
        // because that is what "the same distance along the seam" means to a
        // person sewing it.
        let mut points: Vec<(u32, f64)> = chain
            .points
            .iter()
            .map(|&(i, t)| {
                let s = if t <= 0.0 {
                    0.0
                } else if t >= 1.0 {
                    edge_len
                } else {
                    seg.subsegment(0.0..t).arclen(GEOM_ACCURACY)
                };
                (i, s)
            })
            .collect();
        if reversed {
            points.reverse();
            for p in &mut points {
                p.1 = edge_len - p.1;
            }
        }

        for (i, s) in points {
            // Consecutive edges share their meeting vertex; it arrives twice at
            // the same arc position, so keep the first.
            if along.last().is_some_and(|&(last, _)| last == i) {
                continue;
            }
            along.push((i, walked + s));
        }
        walked += edge_len;
    }

    let (lo, hi) = span(side);
    let (a, b) = (lo * walked, hi * walked);
    let len_mm = b - a;
    if len_mm.partial_cmp(&MIN_SPAN_MM) != Some(std::cmp::Ordering::Greater) {
        return Err(SeamError::DegenerateSpan {
            piece,
            span_mm: len_mm,
        });
    }
    // Half a weld tolerance of slack at each end, so the chain's own endpoints
    // are not dropped by floating-point noise on a full-span side.
    let eps = 1e-9 * walked.max(1.0);
    let vertices: Vec<(u32, f64)> = along
        .into_iter()
        .filter(|&(_, s)| s >= a - eps && s <= b + eps)
        .map(|(i, s)| (i, ((s - a) / len_mm).clamp(0.0, 1.0)))
        .collect();

    Ok(SideSamples { vertices, len_mm })
}

/// Pair two resolved sides. Stitches are emitted from the denser side onto
/// interpolated points of the other; `flip` reverses side `b`'s parameter,
/// which is how a direction mismatch is corrected once detected.
pub fn correspond(a: &SideSamples, b: &SideSamples, flip: bool) -> Correspondence {
    let b_at =
        |u: f64| -> (u32, u32, f32) { interpolate(&b.vertices, if flip { 1.0 - u } else { u }) };
    let a_at = |u: f64| -> (u32, u32, f32) { interpolate(&a.vertices, u) };

    let stitches = if a.vertices.len() >= b.vertices.len() {
        a.vertices
            .iter()
            .map(|&(v, u)| {
                let (i0, i1, w) = b_at(u);
                Stitch6::vertex_to_edge(v, i0, i1, w)
            })
            .collect()
    } else {
        b.vertices
            .iter()
            .map(|&(v, u)| {
                let u = if flip { 1.0 - u } else { u };
                let (i0, i1, w) = a_at(u);
                Stitch6::vertex_to_edge(v, i0, i1, w)
            })
            .collect()
    };

    Correspondence {
        stitches,
        len_a_mm: a.len_mm,
        len_b_mm: b.len_mm,
    }
}

/// Total 3D distance the stitches of this pairing have to close, in metres.
/// Comparing it against the flipped pairing is how a twisted seam is caught:
/// a mismatch reads as a seam whose reversed pairing is dramatically shorter.
pub fn pairing_distance_m(
    a: &SideSamples,
    b: &SideSamples,
    positions: &[[f32; 3]],
    flip: bool,
) -> f64 {
    let mut total = 0.0;
    for &(v, u) in &a.vertices {
        let (i0, i1, w) = interpolate(&b.vertices, if flip { 1.0 - u } else { u });
        let (Some(&p), Some(&q0), Some(&q1)) = (
            positions.get(v as usize),
            positions.get(i0 as usize),
            positions.get(i1 as usize),
        ) else {
            continue;
        };
        let q = [
            q0[0] + (q1[0] - q0[0]) * w,
            q0[1] + (q1[1] - q0[1]) * w,
            q0[2] + (q1[2] - q0[2]) * w,
        ];
        let d = (0..3)
            .map(|k| (p[k] - q[k]) as f64)
            .map(|d| d * d)
            .sum::<f64>();
        total += d.sqrt();
    }
    total
}

/// `(lo, hi)` of a side's sewn span, ordered and clamped. A reversed span
/// `(0.7, 0.3)` is the same span as `(0.3, 0.7)` — direction lives in the edge
/// flags, not in the span.
fn span(side: &SeamSide) -> (f64, f64) {
    let (a, b) = side.span;
    let (a, b) = (a.clamp(0.0, 1.0), b.clamp(0.0, 1.0));
    (a.min(b), a.max(b))
}

/// The point `u` of the way along `samples`, as `(i0, i1, w)` with
/// `p = (1-w)·p[i0] + w·p[i1]`.
fn interpolate(samples: &[(u32, f64)], u: f64) -> (u32, u32, f32) {
    match samples.len() {
        0 => (0, 0, 0.0),
        1 => (samples[0].0, samples[0].0, 0.0),
        _ => {
            let k = samples
                .iter()
                .position(|&(_, s)| s >= u)
                .unwrap_or(samples.len() - 1)
                .max(1);
            let (i0, u0) = samples[k - 1];
            let (i1, u1) = samples[k];
            let w = if u1 > u0 {
                ((u - u0) / (u1 - u0)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (i0, i1, w as f32)
        }
    }
}

/// A seam and the correspondence built for it, for reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct SewnSeam {
    pub seam: SeamId,
    pub stitches: usize,
    pub ease: f64,
    /// `true` if side `b` had to be reversed to pair with side `a`.
    pub flipped: bool,
    /// Distance the seam has to travel to close, metres, at arrangement time.
    pub open_m: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_doc::PieceId;
    use nguo_pattern::{EdgeGeom, EdgeSemantic};
    use nguo_triangulate::{triangulate, TriangulateOptions};

    /// A `w × h` rectangle, edges 0=bottom 1=right 2=top 3=left, all `Seam`.
    fn rect(w: f64, h: f64) -> (VectorNetwork, Vec<EdgeId>) {
        let mut net = VectorNetwork::new();
        let v: Vec<_> = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
            .iter()
            .map(|&(x, y)| net.add_vertex(kurbo::Point::new(x, y)))
            .collect();
        let edges = (0..4)
            .map(|i| {
                net.add_edge(v[i], v[(i + 1) % 4], EdgeGeom::Line, EdgeSemantic::Seam)
                    .unwrap()
            })
            .collect();
        (net, edges)
    }

    fn mesh_of(net: &VectorNetwork, target: f64) -> SimMesh {
        triangulate(
            net,
            &[],
            &TriangulateOptions::default().with_target_edge_mm(target),
        )
        .expect("a rectangle triangulates")
    }

    fn side(edges: Vec<(EdgeId, bool)>) -> SeamSide {
        SeamSide::whole(PieceId::from_raw(0), edges)
    }

    #[test]
    fn a_full_side_is_sampled_end_to_end_in_normalised_arc_length() {
        let (net, e) = rect(100.0, 200.0);
        let mesh = mesh_of(&net, 20.0);
        let s = side_samples(&net, &mesh, &side(vec![(e[1], false)])).unwrap();

        assert!((s.len_mm - 200.0).abs() < 1e-6, "right edge is 200 mm");
        assert!(s.vertices.len() >= 3, "20 mm target on 200 mm: {:?}", s);
        assert_eq!(s.vertices.first().unwrap().1, 0.0);
        assert_eq!(s.vertices.last().unwrap().1, 1.0);
        assert!(
            s.vertices.windows(2).all(|w| w[0].1 <= w[1].1),
            "samples must be monotone: {:?}",
            s.vertices
        );
    }

    /// The load-bearing case: a two-edge side is one parameter space, and the
    /// vertex where the two edges meet appears exactly once.
    #[test]
    fn a_two_edge_side_is_one_parameter_space_with_no_duplicate_join() {
        let (net, e) = rect(100.0, 200.0);
        let mesh = mesh_of(&net, 25.0);
        // bottom then right: (0,0) -> (100,0) -> (100,200). 300 mm total.
        let s = side_samples(&net, &mesh, &side(vec![(e[0], false), (e[1], false)])).unwrap();

        assert!((s.len_mm - 300.0).abs() < 1e-6);
        let corner = mesh
            .chain(e[0])
            .unwrap()
            .points
            .last()
            .expect("bottom edge has points")
            .0;
        assert_eq!(
            s.vertices.iter().filter(|&&(i, _)| i == corner).count(),
            1,
            "the corner vertex is shared, not duplicated"
        );
        // The corner sits 100/300 of the way along.
        let (_, u) = s.vertices.iter().find(|&&(i, _)| i == corner).unwrap();
        assert!((u - 1.0 / 3.0).abs() < 1e-6, "corner at u = {u}");
    }

    #[test]
    fn reversing_an_edge_reverses_its_parameter() {
        let (net, e) = rect(100.0, 200.0);
        let mesh = mesh_of(&net, 20.0);
        let fwd = side_samples(&net, &mesh, &side(vec![(e[1], false)])).unwrap();
        let rev = side_samples(&net, &mesh, &side(vec![(e[1], true)])).unwrap();

        assert_eq!(fwd.vertices.len(), rev.vertices.len());
        for (&(vf, uf), &(vr, ur)) in fwd.vertices.iter().zip(rev.vertices.iter().rev()) {
            assert_eq!(vf, vr, "same vertices, opposite order");
            assert!((uf - (1.0 - ur)).abs() < 1e-9, "{uf} vs {ur}");
        }
    }

    #[test]
    fn a_partial_span_keeps_only_its_own_vertices_and_renormalises() {
        let (net, e) = rect(100.0, 200.0);
        let mesh = mesh_of(&net, 10.0);
        let mut half = side(vec![(e[1], false)]);
        half.span = (0.5, 1.0);
        let s = side_samples(&net, &mesh, &half).unwrap();

        assert!((s.len_mm - 100.0).abs() < 1e-6, "half of 200 mm");
        assert_eq!(s.vertices.first().unwrap().1, 0.0);
        assert_eq!(s.vertices.last().unwrap().1, 1.0);
        let full = side_samples(&net, &mesh, &side(vec![(e[1], false)])).unwrap();
        assert!(
            s.vertices.len() < full.vertices.len(),
            "a half span must sample fewer vertices than the whole edge"
        );
    }

    #[test]
    fn a_discontinuous_side_is_an_error_naming_the_edge() {
        let (net, e) = rect(100.0, 200.0);
        let mesh = mesh_of(&net, 25.0);
        // bottom forward then right REVERSED: (100,0) is the end of one and the
        // end of the other, so the chain does not join.
        let err = side_samples(&net, &mesh, &side(vec![(e[0], false), (e[1], true)])).unwrap_err();
        assert!(
            matches!(err, SeamError::BrokenChain { edge, .. } if edge == e[1]),
            "{err}"
        );
    }

    #[test]
    fn a_zero_length_span_is_an_error_not_a_division_by_zero() {
        let (net, e) = rect(100.0, 200.0);
        let mesh = mesh_of(&net, 25.0);
        let mut point = side(vec![(e[1], false)]);
        point.span = (0.4, 0.4);
        assert!(matches!(
            side_samples(&net, &mesh, &point).unwrap_err(),
            SeamError::DegenerateSpan { .. }
        ));
    }

    /// Equal-length sides at *different* resolutions: every stitch on the
    /// denser side must land on the geometrically matching point of the
    /// sparser one, and the ease must be exactly 1.
    #[test]
    fn unequal_resolutions_pair_by_parameter_not_by_vertex_count() {
        let (net, e) = rect(100.0, 200.0);
        let fine = mesh_of(&net, 8.0);
        let coarse = mesh_of(&net, 40.0);
        let a = side_samples(&net, &fine, &side(vec![(e[1], false)])).unwrap();
        let b = side_samples(&net, &coarse, &side(vec![(e[1], false)])).unwrap();
        assert!(a.vertices.len() > b.vertices.len(), "{a:?} vs {b:?}");

        let c = correspond(&a, &b, false);
        assert_eq!(c.stitches.len(), a.vertices.len(), "one per dense vertex");
        assert!((c.ease() - 1.0).abs() < 1e-9);

        // Every stitch's target is the parameter-matched point: check the
        // interpolated y against the source y in the coarse mesh's own space.
        for (&(v, u), st) in a.vertices.iter().zip(&c.stitches) {
            let want = fine.pattern_mm[v as usize][1];
            let (i0, i1) = (st.index[3] as usize, st.index[4] as usize);
            let w = st.weight[4] as f64;
            let got = coarse.pattern_mm[i0][1] * (1.0 - w) + coarse.pattern_mm[i1][1] * w;
            assert!(
                (want - got).abs() < 1e-6,
                "u = {u}: wanted y {want}, interpolated {got}"
            );
            assert!((st.weight[0..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!((st.weight[3..6].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    /// Unequal *lengths* become ease, proportionally: the midpoint of the long
    /// side sews to the midpoint of the short one.
    #[test]
    fn unequal_lengths_become_ease_and_pair_proportionally() {
        let (long_net, le) = rect(100.0, 250.0);
        let (short_net, se) = rect(100.0, 200.0);
        let long_mesh = mesh_of(&long_net, 10.0);
        let short_mesh = mesh_of(&short_net, 10.0);

        let a = side_samples(&short_net, &short_mesh, &side(vec![(se[1], false)])).unwrap();
        let b = side_samples(&long_net, &long_mesh, &side(vec![(le[1], false)])).unwrap();
        let c = correspond(&a, &b, false);

        assert!((c.ease() - 1.25).abs() < 1e-9, "250/200 = 1.25");
        assert!((c.ease_fraction() - 0.25).abs() < 1e-9);
        assert_eq!(c.stitches.len(), b.vertices.len(), "b is denser (250 mm)");

        // Take the stitch nearest the middle of the long side and check its
        // target is the middle of the short side: 125 mm up sews to 100 mm up.
        let (v, _) = *b
            .vertices
            .iter()
            .min_by(|x, y| (x.1 - 0.5).abs().total_cmp(&(y.1 - 0.5).abs()))
            .unwrap();
        let st = c
            .stitches
            .iter()
            .find(|s| s.index[0] == v)
            .expect("the middle vertex is stitched");
        let (i0, i1) = (st.index[3] as usize, st.index[4] as usize);
        let w = st.weight[4] as f64;
        let target_y = short_mesh.pattern_mm[i0][1] * (1.0 - w) + short_mesh.pattern_mm[i1][1] * w;
        let source_y = long_mesh.pattern_mm[v as usize][1];
        assert!(
            (target_y / 200.0 - source_y / 250.0).abs() < 1e-6,
            "proportional: {source_y}/250 vs {target_y}/200"
        );
    }

    /// A flipped pairing must be *detectable*: with two sides arranged nose to
    /// tail, the flipped pairing is much shorter than the straight one.
    #[test]
    fn a_flipped_pairing_measures_shorter_when_the_seam_is_twisted() {
        let (net, e) = rect(100.0, 200.0);
        let mesh = mesh_of(&net, 20.0);
        let a = side_samples(&net, &mesh, &side(vec![(e[1], false)])).unwrap();
        let b = side_samples(&net, &mesh, &side(vec![(e[1], false)])).unwrap();

        // Side b lifted 1 m and mirrored in y, so b's u=1 sits next to a's u=0.
        let mut positions: Vec<[f32; 3]> = mesh
            .pattern_mm
            .iter()
            .map(|&[x, y]| [x as f32 * 1e-3, y as f32 * 1e-3, 0.0])
            .collect();
        let flipped: Vec<[f32; 3]> = positions.iter().map(|p| [p[0], 0.2 - p[1], 0.0]).collect();
        let base = positions.len() as u32;
        positions.extend(flipped);
        let b = SideSamples {
            vertices: b.vertices.iter().map(|&(i, u)| (i + base, u)).collect(),
            len_mm: b.len_mm,
        };

        let straight = pairing_distance_m(&a, &b, &positions, false);
        let flipped = pairing_distance_m(&a, &b, &positions, true);
        assert!(
            flipped < 0.5 * straight,
            "flipped {flipped} should be far shorter than straight {straight}"
        );
    }
}
