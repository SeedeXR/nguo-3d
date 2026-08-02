// SPDX-License-Identifier: GPL-2.0-or-later
//! Property tests for the vector network and the seam-allowance offset
//! (guide §18).

use kurbo::{BezPath, ParamCurve, PathSeg, Point, Shape};
use nguo_pattern::network::{EdgeGeom, EdgeSemantic, VectorNetwork};
use nguo_pattern::{is_simple, offset_uniform, OffsetOptions};
use proptest::prelude::*;

/// A star-shaped polygon: vertices at evenly spaced angles with random radii.
/// Always simple, frequently very concave — exactly the family that breaks a
/// naive offset.
fn star(radii: &[f64]) -> Vec<PathSeg> {
    let n = radii.len();
    let mut p = BezPath::new();
    for (i, r) in radii.iter().enumerate() {
        let a = i as f64 * std::f64::consts::TAU / n as f64;
        let pt = Point::new(r * a.cos(), r * a.sin());
        if i == 0 {
            p.move_to(pt)
        } else {
            p.line_to(pt)
        }
    }
    p.close_path();
    p.segments().collect()
}

fn path_of(segs: &[PathSeg]) -> BezPath {
    let mut p = BezPath::new();
    p.move_to(segs[0].start());
    for s in segs {
        p.push(s.as_path_el());
    }
    p.close_path();
    p
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// Growing a convex shape by `w` with round joins adds exactly
    /// `perimeter*w + pi*w^2`. This is the invariant that pins the joins down.
    #[test]
    fn convex_offset_area_is_exact(
        sides in 3usize..9,
        radius in 20.0f64..120.0,
        w in 0.5f64..15.0,
    ) {
        let radii = vec![radius; sides];
        let s = star(&radii);
        let base = path_of(&s).area().abs();
        let perim: f64 = s.iter().map(|x| (x.end() - x.start()).hypot()).sum();
        let out = offset_uniform(&s, w, &OffsetOptions::default()).unwrap();
        let want = base + perim * w + std::f64::consts::PI * w * w;
        prop_assert!(
            (out.area().abs() - want).abs() < want * 2e-3,
            "{sides}-gon r={radius} w={w}: got {}, want {want}",
            out.area().abs()
        );
    }

    /// Concave shapes: the output must stay simple, must grow, and can never
    /// exceed the convex bound.
    #[test]
    fn concave_offset_stays_simple(
        radii in proptest::collection::vec(15.0f64..80.0, 5..12),
        w in 0.5f64..14.0,
    ) {
        let s = star(&radii);
        let src = path_of(&s);
        prop_assume!(is_simple(&src, 1e-3));
        let base = src.area().abs();
        let perim: f64 = s.iter().map(|x| (x.end() - x.start()).hypot()).sum();

        let out = offset_uniform(&s, w, &OffsetOptions::default()).unwrap();
        prop_assert!(is_simple(&out, 1e-3), "offset by {w} self-intersects");
        let a = out.area().abs();
        prop_assert!(a >= base - 1e-6, "offset by {w} shrank: {a} < {base}");
        let bound = base + perim * w + std::f64::consts::PI * w * w;
        prop_assert!(a <= bound * (1.0 + 2e-3), "offset by {w} exceeded {bound}: {a}");
    }

    /// Offsetting by nothing changes nothing, bit for bit.
    #[test]
    fn zero_offset_is_identity(radii in proptest::collection::vec(15.0f64..80.0, 3..10)) {
        let s = star(&radii);
        let out = offset_uniform(&s, 0.0, &OffsetOptions::default()).unwrap();
        prop_assert_eq!(out.segments().collect::<Vec<_>>(), s);
    }

    /// Area grows monotonically with the allowance.
    #[test]
    fn offset_area_is_monotone(
        radii in proptest::collection::vec(20.0f64..80.0, 4..10),
        a in 0.5f64..5.0,
        b in 5.5f64..12.0,
    ) {
        let s = star(&radii);
        prop_assume!(is_simple(&path_of(&s), 1e-3));
        let o = OffsetOptions::default();
        let (ra, rb) = (
            offset_uniform(&s, a, &o).unwrap().area().abs(),
            offset_uniform(&s, b, &o).unwrap().area().abs(),
        );
        prop_assert!(rb > ra, "w={a} -> {ra} but w={b} -> {rb}");
    }
}

proptest! {
    /// Split then join gives back the same edge id and the same length.
    #[test]
    fn split_join_round_trips(t in 0.05f64..0.95, len in 1.0f64..500.0) {
        let mut n = VectorNetwork::new();
        let a = n.add_vertex(Point::new(0.0, 0.0));
        let b = n.add_vertex(Point::new(len, 0.0));
        let e = n.add_line(a, b).unwrap();
        let before = n.clone();
        let s = n.split_edge(e, t).unwrap();
        prop_assert_eq!(s.first, e);
        let joined = n.join_edges(s.first, s.second).unwrap();
        prop_assert_eq!(joined, e);
        prop_assert!((n.arc_length(e).unwrap() - len).abs() < 1e-9);
        prop_assert_eq!(n.edges.len(), before.edges.len());
        prop_assert_eq!(n.vertices.len(), before.vertices.len());
    }

    /// Splitting never changes the total length of the edge, curved or not.
    #[test]
    fn split_preserves_length(
        t in 0.05f64..0.95,
        cx in -100.0f64..200.0,
        cy in -100.0f64..100.0,
    ) {
        let mut n = VectorNetwork::new();
        let a = n.add_vertex(Point::new(0.0, 0.0));
        let b = n.add_vertex(Point::new(100.0, 0.0));
        let e = n
            .add_edge(
                a,
                b,
                EdgeGeom::Cubic { c1: Point::new(cx, cy), c2: Point::new(100.0 - cx, cy) },
                EdgeSemantic::Seam,
            )
            .unwrap();
        let before = n.arc_length(e).unwrap();
        let s = n.split_edge(e, t).unwrap();
        let after = n.arc_length(s.first).unwrap() + n.arc_length(s.second).unwrap();
        // Two arclen calls, each accurate to GEOM_ACCURACY (0.01 mm).
        prop_assert!((before - after).abs() < 3e-2, "{before} vs {after}");
    }

    /// The reported nearest distance really is the distance to the point at the
    /// reported parameter.
    #[test]
    fn nearest_point_is_consistent(
        px in -200.0f64..300.0,
        py in -200.0f64..300.0,
        cy in -80.0f64..80.0,
    ) {
        let mut n = VectorNetwork::new();
        let a = n.add_vertex(Point::new(0.0, 0.0));
        let b = n.add_vertex(Point::new(100.0, 0.0));
        let e = n
            .add_edge(
                a,
                b,
                EdgeGeom::Cubic { c1: Point::new(20.0, cy), c2: Point::new(80.0, cy) },
                EdgeSemantic::Seam,
            )
            .unwrap();
        let p = Point::new(px, py);
        let (t, d) = n.nearest_point(e, p).unwrap();
        prop_assert!((0.0..=1.0).contains(&t));
        let on_curve = n.seg(e).unwrap().eval(t);
        prop_assert!(((on_curve - p).hypot() - d).abs() < 1e-9);
        // And nothing sampled along the curve is meaningfully closer.
        for k in 0..=40 {
            let q = n.seg(e).unwrap().eval(k as f64 / 40.0);
            prop_assert!((q - p).hypot() >= d - 1e-2);
        }
    }

    /// Ids survive an arbitrary sequence of unrelated edits.
    #[test]
    fn ids_are_stable_across_edits(splits in proptest::collection::vec(0.1f64..0.9, 0..12)) {
        let mut n = VectorNetwork::new();
        let v: Vec<_> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        let e: Vec<_> = (0..4).map(|i| n.add_line(v[i], v[(i + 1) % 4]).unwrap()).collect();
        let watched = e[2];
        let watched_len = n.arc_length(watched).unwrap();
        for (k, t) in splits.iter().enumerate() {
            n.split_edge(e[k % 2], *t).ok();
        }
        prop_assert!(n.edges.contains(watched));
        prop_assert!((n.arc_length(watched).unwrap() - watched_len).abs() < 1e-9);
        for id in &v {
            prop_assert!(n.vertices.contains(*id));
        }
    }
}
