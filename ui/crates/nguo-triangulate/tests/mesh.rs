// SPDX-License-Identifier: GPL-2.0-or-later
//! Integration, regression and property tests for `nguo-triangulate`, plus the
//! profiling harness.
//!
//! The golden mesh is regenerated with
//! `NGUO_BLESS=1 cargo test -p nguo-triangulate --test mesh`, the same idiom as
//! `nguo-doc/tests/migration.rs`.
//!
//! The profiling test prints its numbers to **fd 2** and asserts floors on the
//! work it did, so it cannot report `ok` while measuring nothing.
//!
//! `clippy::explicit_write` is allowed throughout this file on purpose.
//! `eprintln!` goes through libtest's capture shim and is swallowed unless the
//! test fails or `--nocapture` is passed — which is exactly how this codebase
//! previously shipped 15 tests that reported `ok` while covering nothing
//! (`memory/AGENT_BRIEF.md`). `writeln!(std::io::stderr(), ..)` writes to fd 2
//! directly and is always visible.
#![allow(clippy::explicit_write)]

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use kurbo::{ParamCurve, Point};
use nguo_pattern::{EdgeGeom, EdgeId, EdgeSemantic, VectorNetwork, VertexId};
use nguo_triangulate::{triangulate, SimMesh, TriangulateOptions};

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/// A half bodice front panel in millimetres: straight waist and side seam,
/// cubic armhole and neckline, a centre-front fold, one waist dart, two notches.
/// Roughly a size-12 block, which is what makes the timings below mean
/// something.
fn bodice() -> (VectorNetwork, Vec<(EdgeId, f64)>) {
    let mut n = VectorNetwork::new();
    let v = |n: &mut VectorNetwork, x: f64, y: f64| n.add_vertex(Point::new(x, y));

    let cf_waist = v(&mut n, 0.0, 0.0);
    let dart_l = v(&mut n, 45.0, 0.0);
    let dart_r = v(&mut n, 65.0, 0.0);
    let side_waist = v(&mut n, 110.0, 0.0);
    let underarm = v(&mut n, 118.0, 180.0);
    let shoulder_out = v(&mut n, 78.0, 330.0);
    let shoulder_in = v(&mut n, 28.0, 356.0);
    let cf_neck = v(&mut n, 0.0, 300.0);

    let line = |n: &mut VectorNetwork, a, b, s| n.add_edge(a, b, EdgeGeom::Line, s).unwrap();

    // Waist, in three pieces so the dart legs land on existing vertices.
    let waist_a = line(&mut n, cf_waist, dart_l, EdgeSemantic::Seam);
    line(&mut n, dart_l, dart_r, EdgeSemantic::Seam);
    let waist_c = line(&mut n, dart_r, side_waist, EdgeSemantic::Seam);
    // Side seam, gently hollowed.
    let side = n
        .add_edge(
            side_waist,
            underarm,
            EdgeGeom::Cubic {
                c1: Point::new(106.0, 60.0),
                c2: Point::new(108.0, 120.0),
            },
            EdgeSemantic::Seam,
        )
        .unwrap();
    // Armhole: the scoop that makes this a real pattern piece.
    let armhole = n
        .add_edge(
            underarm,
            shoulder_out,
            EdgeGeom::Cubic {
                c1: Point::new(128.0, 250.0),
                c2: Point::new(108.0, 320.0),
            },
            EdgeSemantic::Seam,
        )
        .unwrap();
    line(&mut n, shoulder_out, shoulder_in, EdgeSemantic::Seam);
    // Neckline.
    let neck = n
        .add_edge(
            shoulder_in,
            cf_neck,
            EdgeGeom::Cubic {
                c1: Point::new(14.0, 348.0),
                c2: Point::new(0.0, 330.0),
            },
            EdgeSemantic::Seam,
        )
        .unwrap();
    line(&mut n, cf_neck, cf_waist, EdgeSemantic::Fold);

    // Waist dart, apex at bust level.
    let apex = v(&mut n, 55.0, 150.0);
    n.add_edge(dart_l, apex, EdgeGeom::Line, EdgeSemantic::DartLeg)
        .unwrap();
    n.add_edge(apex, dart_r, EdgeGeom::Line, EdgeSemantic::DartLeg)
        .unwrap();

    // Grain line, as a marker edge that must not reach the mesh.
    let g0 = v(&mut n, 60.0, 40.0);
    let g1 = v(&mut n, 60.0, 260.0);
    n.add_edge(g0, g1, EdgeGeom::Line, EdgeSemantic::Grain)
        .unwrap();

    // Balance notches, where a sewer matches the front to the back.
    let notches = vec![(armhole, 0.5), (side, 0.5), (waist_a, 0.5), (waist_c, 0.25)];
    let _ = (neck, armhole);
    (n, notches)
}

/// A star-shaped polygon: `n` vertices at sorted distinct angles. Star-shaped
/// polygons are simple by construction, which is what makes them the right
/// generator for a "no flipped triangles" property test.
fn star(radii: &[(f64, f64)]) -> VectorNetwork {
    let mut n = VectorNetwork::new();
    let v: Vec<VertexId> = radii
        .iter()
        .map(|&(angle, r)| n.add_vertex(Point::new(r * angle.cos(), r * angle.sin())))
        .collect();
    for i in 0..v.len() {
        n.add_edge(
            v[i],
            v[(i + 1) % v.len()],
            EdgeGeom::Line,
            EdgeSemantic::Cut,
        )
        .unwrap();
    }
    n
}

/// The golden fixture: a straight-edged 200 x 300 mm panel with a dart, a
/// pocket-placement internal line, a cut-out and three notches.
///
/// **Straight edges only, on purpose.** Every coordinate the mesher then touches
/// comes out of `+ - * /` and `sqrt`, which IEEE-754 makes bit-identical on every
/// platform. Put a cubic in here and the golden starts depending on `libm`'s
/// `acos`/`cos` inside `kurbo`'s root finder, and the file stops being portable
/// between the macOS, Linux and Windows CI runners. The curved bodice is covered
/// by `the_bodice_triangulates_and_maps_back_to_the_pattern` instead, which
/// asserts properties rather than bytes.
fn golden_panel() -> (VectorNetwork, Vec<(EdgeId, f64)>) {
    let mut n = VectorNetwork::new();
    let v = |n: &mut VectorNetwork, x: f64, y: f64| n.add_vertex(Point::new(x, y));

    // Outline, counter-clockwise, split at the dart legs' feet.
    let outline: Vec<VertexId> = [
        (0.0, 0.0),
        (70.0, 0.0),
        (110.0, 0.0),
        (200.0, 0.0),
        (200.0, 300.0),
        (0.0, 300.0),
    ]
    .iter()
    .map(|&(x, y)| v(&mut n, x, y))
    .collect();
    let edges: Vec<EdgeId> = (0..6)
        .map(|i| {
            n.add_edge(
                outline[i],
                outline[(i + 1) % 6],
                EdgeGeom::Line,
                EdgeSemantic::Seam,
            )
            .unwrap()
        })
        .collect();

    // Waist dart.
    let apex = v(&mut n, 90.0, 160.0);
    n.add_edge(outline[1], apex, EdgeGeom::Line, EdgeSemantic::DartLeg)
        .unwrap();
    n.add_edge(apex, outline[2], EdgeGeom::Line, EdgeSemantic::DartLeg)
        .unwrap();

    // Pocket placement line, T-junction to T-junction across the panel.
    let pl = v(&mut n, 40.0, 220.0);
    let pr = v(&mut n, 160.0, 220.0);
    n.add_edge(pl, pr, EdgeGeom::Line, EdgeSemantic::Internal)
        .unwrap();

    // A cut-out, which parity must exclude.
    let hole: Vec<VertexId> = [(150.0, 60.0), (180.0, 60.0), (180.0, 100.0)]
        .iter()
        .map(|&(x, y)| v(&mut n, x, y))
        .collect();
    for i in 0..3 {
        n.add_edge(
            hole[i],
            hole[(i + 1) % 3],
            EdgeGeom::Line,
            EdgeSemantic::Cut,
        )
        .unwrap();
    }

    let notches = vec![(edges[0], 0.5), (edges[3], 0.25), (edges[3], 0.75)];
    (n, notches)
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/panel_mesh.json")
}

// ---------------------------------------------------------------------------
// integration
// ---------------------------------------------------------------------------

#[test]
fn the_bodice_triangulates_and_maps_back_to_the_pattern() {
    let (net, notches) = bodice();
    let m = triangulate(&net, &notches, &TriangulateOptions::default()).unwrap();
    m.check().unwrap();

    // Every notch is a mesh vertex at the right place on the right edge.
    for &(edge, t) in &notches {
        let chain = m.chain(edge).expect("every notched edge must be mapped");
        let hit = chain
            .points
            .iter()
            .find(|&&(_, ct)| (ct - t).abs() < 1e-12)
            .unwrap_or_else(|| panic!("notch t={t} missing from {edge:?}"));
        let want = net.seg(edge).unwrap().eval(t);
        let got = m.pattern_mm[hit.0 as usize];
        assert!(
            (got[0] - want.x).hypot(got[1] - want.y) < 1e-3,
            "notch at ({}, {}) landed at {got:?}",
            want.x,
            want.y
        );
    }

    // The dart wedge is gone. Cross-check against the same piece with the dart
    // legs demoted to internal lines, which leaves the wedge in: the difference
    // must be the wedge's area, 20 mm base x 150 mm high / 2 = 1500 mm².
    let mut no_dart = net.clone();
    for (_, e) in no_dart.edges.iter_mut() {
        if e.semantic == EdgeSemantic::DartLeg {
            e.semantic = EdgeSemantic::Internal;
        }
    }
    let whole = triangulate(&no_dart, &notches, &TriangulateOptions::default()).unwrap();
    assert!(
        ((whole.area_mm2() - m.area_mm2()) - 1_500.0).abs() < 1e-3,
        "the dart wedge should be 1500 mm², measured {}",
        whole.area_mm2() - m.area_mm2()
    );

    // Grain markers contribute nothing.
    assert!(m.chains.iter().all(|c| c.semantic != EdgeSemantic::Grain));
    // Sewable outline, ready for M6.
    assert!(m.outline_vertices().len() > 80);

    // Mesh quality. The dart apex is a 7.6° wedge of the *input*, so Ruppert
    // provably cannot open it up and the global minimum sits near 3°. Judge the
    // mesh on the 1% quantile instead — everything but the apex slivers.
    assert!(
        m.angle_quantile_deg(0.01) > 20.0,
        "1% quantile {} (min {})",
        m.angle_quantile_deg(0.01),
        m.min_angle_deg()
    );
    // MEASURED: 9 triangles below 15° out of ~900, all of them at an acute or
    // tangential *input* corner — the two at the dart apex (7.6° of input
    // angle) and the rest where the neckline meets the centre-front fold
    // tangentially (0° of input angle). Both are properties of the pattern, not
    // of the mesher. 2% is the ceiling; if it is ever exceeded, refinement
    // regressed.
    let slivers = m.min_angles_deg().iter().filter(|&&a| a < 15.0).count();
    assert!(
        (slivers as f64) < 0.02 * m.triangle_count() as f64,
        "{slivers} of {} triangles below 15°: {:?}",
        m.triangle_count(),
        m.min_angles_deg()
            .iter()
            .filter(|&&a| a < 15.0)
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// regression: a golden mesh
// ---------------------------------------------------------------------------

#[test]
fn the_panel_mesh_matches_its_golden() {
    let (net, notches) = golden_panel();
    let m = triangulate(
        &net,
        &notches,
        &TriangulateOptions::default().with_target_edge_mm(40.0),
    )
    .unwrap();
    // A coarse resolution keeps the fixture readable in a diff while still
    // exercising the dart, the cut-out, the internal line and the notches.
    assert!(m.triangle_count() > 80 && m.triangle_count() < 400);
    // Hand-checkable: 200 x 300 panel, minus a 40 x 160 / 2 dart wedge, minus a
    // 30 x 40 / 2 cut-out. If the golden ever needs re-blessing, this is the
    // number that says whether the new mesh is still the right shape.
    assert!(
        (m.area_mm2() - (60_000.0 - 3_200.0 - 600.0)).abs() < 1e-6,
        "area {} mm², expected 56200",
        m.area_mm2()
    );
    let text = serde_json::to_string_pretty(&m).unwrap() + "\n";
    let path = golden_path();

    if std::env::var("NGUO_BLESS").is_ok() || !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
        writeln!(std::io::stderr(), "blessed golden at {}", path.display()).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        text, want,
        "the panel mesh drifted; re-bless with NGUO_BLESS=1 if intended"
    );
    // And the golden is itself a valid mesh, so the format is closed under
    // save/load.
    let back: SimMesh = serde_json::from_str(&want).unwrap();
    back.check().unwrap();
    assert_eq!(back, m);
}

// ---------------------------------------------------------------------------
// property: no flipped, no degenerate triangles, ever
// ---------------------------------------------------------------------------

mod prop {
    use super::*;
    use proptest::prelude::*;

    /// A star-shaped polygon: `n` vertices at strictly increasing angles with
    /// random radii. Simple by construction, and the radius spread makes plenty
    /// of very acute corners and near-collinear runs.
    ///
    /// The angular jitter is capped at ±20% of the even spacing, which pins two
    /// things down:
    ///  - every angular gap lands in `[0.6, 1.4] x step`, so no gap reaches 180°
    ///    and the polygon is therefore star-shaped **about the origin** — which
    ///    tests below rely on when they run a spur to the centre.
    ///  - two vertices are never at *arbitrarily* close angles. Two long edges
    ///    arbitrarily close together get snap-rounded onto a 1 µm grid and can
    ///    then genuinely cross, at which point `triangulate` correctly returns
    ///    `CrossingConstraint` and "must succeed" stops being a fair demand.
    ///
    /// The radii still span 80:1, so corners down to a couple of degrees are
    /// generated. `crossing_pattern_edges_are_refused_with_a_fixable_error` and
    /// `nothing_degenerate_ever_panics_or_lies` cover the other side.
    fn star_polygon() -> impl Strategy<Value = Vec<(f64, f64)>> {
        (3usize..40).prop_flat_map(|n| {
            (
                proptest::collection::vec(-0.2f64..0.2, n),
                proptest::collection::vec(5.0f64..400.0, n),
            )
                .prop_map(move |(jitter, radii)| {
                    let step = std::f64::consts::TAU / n as f64;
                    jitter
                        .iter()
                        .zip(radii)
                        .enumerate()
                        .map(|(i, (j, r))| ((i as f64 + j) * step, r))
                        .collect()
                })
        })
    }

    /// Area tolerance for a snap-rounded polygon: snapping moves every vertex by
    /// up to `weld/2`, which can change the area by up to about
    /// `weld * perimeter`. Asserting tighter than the sanitiser's own error is
    /// asserting that the sanitiser does not exist.
    fn area_tolerance(poly: &[(f64, f64)], weld_mm: f64) -> f64 {
        let perimeter: f64 = (0..poly.len())
            .map(|i| {
                let (a, r) = poly[i];
                let (b, s) = poly[(i + 1) % poly.len()];
                (r * a.cos() - s * b.cos()).hypot(r * a.sin() - s * b.sin())
            })
            .sum();
        weld_mm * perimeter + 1e-9
    }

    fn shoelace(poly: &[(f64, f64)]) -> f64 {
        let mut acc = 0.0;
        for i in 0..poly.len() {
            let (a, r) = poly[i];
            let (b, s) = poly[(i + 1) % poly.len()];
            acc += (r * a.cos()) * (s * b.sin()) - (s * b.cos()) * (r * a.sin());
        }
        0.5 * acc.abs()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        /// The M5 acceptance criterion.
        #[test]
        fn no_flipped_or_degenerate_triangles(poly in star_polygon()) {
            let net = star(&poly);
            let m = triangulate(&net, &[], &TriangulateOptions::default().with_target_edge_mm(25.0))
                .expect("a simple polygon must triangulate");
            // `check` asserts strictly positive signed area for every triangle,
            // which is exactly "no flipped and no degenerate".
            prop_assert_eq!(m.check(), Ok(()));
            prop_assert_eq!(m.degenerate_dropped, 0);

            // And the mesh covers the polygon: shoelace over the input must
            // equal the sum of triangle areas. This is what catches an overlap
            // or a hole that per-triangle checks cannot see.
            let want = shoelace(&poly);
            let tol = area_tolerance(&poly, TriangulateOptions::default().weld_mm);
            prop_assert!(
                (m.area_mm2() - want).abs() < tol,
                "mesh area {} vs polygon area {want} (tolerance {tol})",
                m.area_mm2()
            );
        }

        /// Snap-rounding is the sanitiser, and this is its contract: duplicated
        /// input vertices must produce **the same mesh** as the network with the
        /// duplicates already merged. Not merely "not crash" — bit-identical.
        ///
        /// The duplicates here are exact, which is what makes the demand
        /// provable: grid snap-rounding guarantees that points in the same cell
        /// merge, and it guarantees nothing about points merely *close* to each
        /// other — two points 0.26 µm apart still land in different cells if a
        /// grid line falls between them. Sub-grid neighbours are covered
        /// deterministically by the crate's
        /// `near_duplicate_input_is_welded_not_fatal`, and the adversarial regime
        /// (where snapping can flip a point across a 2° corner and
        /// `CrossingConstraint` is the right answer) by
        /// `nothing_degenerate_ever_panics_or_lies`.
        #[test]
        fn duplicate_vertices_produce_an_identical_mesh(poly in star_polygon()) {
            // Two networks with matching edge ids, differing only in whether the
            // internal spurs start at the shared corner vertex or at a duplicate
            // of it. The spurs run to the origin, which a star-shaped polygon
            // always contains, so they are unambiguously interior.
            let build = |duplicate: bool| {
                let mut net = star(&poly);
                let hub = net.add_vertex(Point::ZERO);
                let corners: Vec<VertexId> = net.vertices.ids().take(poly.len()).collect();
                for c in corners {
                    let start = if duplicate {
                        net.add_vertex(net.pos(c).unwrap())
                    } else {
                        c
                    };
                    net.add_edge(start, hub, EdgeGeom::Line, EdgeSemantic::Internal).unwrap();
                }
                net
            };
            let opts = TriangulateOptions::default().with_target_edge_mm(25.0);
            let want = triangulate(&build(false), &[], &opts)
                .expect("the already-welded network must triangulate");
            let got = triangulate(&build(true), &[], &opts)
                .expect("duplicates must be welded, not fatal");
            prop_assert_eq!(got.check(), Ok(()));
            prop_assert_eq!(&got.pattern_mm, &want.pattern_mm);
            prop_assert_eq!(&got.indices, &want.indices);
            prop_assert_eq!(&got.chains, &want.chains);
        }

        /// The robustness contract, on the nastiest input the generator can
        /// make: slivers, 80:1 radius ratios, ghost vertices, notches at both
        /// ends of every edge. `triangulate` must either return a mesh that
        /// passes every invariant, or a typed error. Never a panic, never a
        /// silently broken mesh.
        ///
        /// This is the test that found the reachable `panic!` inside
        /// `spade::refine` (see the catch in `triangulate`).
        #[test]
        fn nothing_degenerate_ever_panics_or_lies(
            poly in star_polygon(),
            jitter in 0.0f64..1e-3,
            t in 0.0f64..1.0,
        ) {
            let mut net = star(&poly);
            let hub = net.add_vertex(Point::new(0.0, 0.0));
            for &(a, r) in &poly {
                let p = Point::new(r * a.cos() + jitter, r * a.sin() - jitter);
                let ghost = net.add_vertex(p);
                net.add_edge(ghost, hub, EdgeGeom::Line, EdgeSemantic::DartLeg).unwrap();
            }
            let notches: Vec<(EdgeId, f64)> =
                net.edges.ids().flat_map(|e| [(e, t), (e, 1.0 - t)]).collect();
            let opts = TriangulateOptions::default().with_target_edge_mm(15.0);
            match triangulate(&net, &notches, &opts) {
                Ok(m) => prop_assert_eq!(m.check(), Ok(())),
                Err(e) => {
                    // Every error must carry something the user can act on.
                    prop_assert!(!e.to_string().is_empty());
                }
            }
        }

        /// Notches never move the geometry, only add vertices to it.
        #[test]
        fn notches_add_vertices_without_changing_the_area(
            poly in star_polygon(),
            t in 0.01f64..0.99,
        ) {
            let net = star(&poly);
            let opts = TriangulateOptions::default().with_target_edge_mm(25.0);
            let plain = triangulate(&net, &[], &opts).unwrap();
            let edges: Vec<EdgeId> = net.edges.ids().collect();
            let notches: Vec<(EdgeId, f64)> = edges.iter().map(|&e| (e, t)).collect();
            let notched = triangulate(&net, &notches, &opts).unwrap();
            prop_assert_eq!(notched.check(), Ok(()));
            let tol = area_tolerance(&poly, opts.weld_mm);
            prop_assert!(
                (notched.area_mm2() - plain.area_mm2()).abs() < tol,
                "{} vs {} (tolerance {tol})", notched.area_mm2(), plain.area_mm2()
            );
            for &(e, _) in &notches {
                let chain = notched.chain(e).unwrap();
                prop_assert!(chain.points.iter().any(|&(_, ct)| (ct - t).abs() < 1e-12));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// profiling
// ---------------------------------------------------------------------------

/// Median of `runs` timings, in milliseconds, plus the last mesh.
fn time_it(
    label: &str,
    runs: usize,
    net: &VectorNetwork,
    notches: &[(EdgeId, f64)],
    opts: &TriangulateOptions,
) -> (f64, SimMesh) {
    // One warm-up so we are not timing first-touch allocation.
    let mut mesh = triangulate(net, notches, opts).expect(label);
    let mut ms: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        mesh = triangulate(net, notches, opts).expect(label);
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    let median = ms[ms.len() / 2];
    let boundary = mesh.outline_vertices().len();
    writeln!(
        std::io::stderr(),
        "{label:<34} {median:>8.3} ms   min {:>7.3}  max {:>7.3}   \
         {boundary:>4} boundary v, {:>6} v, {:>6} tri, angle min {:>5.2}° p1 {:>5.2}° p50 {:>5.2}°",
        ms[0],
        ms[ms.len() - 1],
        mesh.vertex_count(),
        mesh.triangle_count(),
        mesh.min_angle_deg(),
        mesh.angle_quantile_deg(0.01),
        mesh.angle_quantile_deg(0.5),
    )
    .unwrap();
    (median, mesh)
}

/// Real numbers for a real piece, and the cost curve of refinement.
///
/// Run with `-- --nocapture` to read the table. The assertions are floors on
/// the work actually done, so this test cannot pass vacuously; set
/// `NGUO_PROFILE_STRICT=1` to also fail if a piece misses the frame budget.
#[test]
fn profile_triangulation() {
    let strict = std::env::var("NGUO_PROFILE_STRICT").is_ok();
    let (net, notches) = bodice();

    writeln!(
        std::io::stderr(),
        "\n--- nguo-triangulate: bodice front panel, {} pattern edges ---",
        net.edges.len()
    )
    .unwrap();

    // The realistic case first: default options, ~1 cm triangles.
    let (bodice_ms, m) = time_it(
        "bodice, 10 mm target edge",
        20,
        &net,
        &notches,
        &TriangulateOptions::default(),
    );
    assert!(
        m.triangle_count() > 500,
        "the profile measured almost nothing: {} triangles",
        m.triangle_count()
    );
    assert!(m.angle_quantile_deg(0.01) > 20.0);

    // What the always-on `SimMesh::check` inside `triangulate` costs. It is a
    // fraction of the total or it would not be always-on.
    let mut check_ms: Vec<f64> = (0..40)
        .map(|_| {
            let t = Instant::now();
            assert_eq!(m.check(), Ok(()));
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    check_ms.sort_by(f64::total_cmp);
    let check_ms = check_ms[check_ms.len() / 2];
    writeln!(
        std::io::stderr(),
        "SimMesh::check on that mesh          {check_ms:>8.3} ms  = {:.1}% of the triangulation",
        100.0 * check_ms / bodice_ms
    )
    .unwrap();
    assert!(
        check_ms < 0.25 * bodice_ms,
        "the always-on validity check costs {check_ms} ms against {bodice_ms} ms of triangulation"
    );

    // The ~200-boundary-point case the brief asks about.
    let (dense_ms, dense) = time_it(
        "bodice, 5 mm target edge",
        20,
        &net,
        &notches,
        &TriangulateOptions::default().with_target_edge_mm(5.0),
    );
    assert!(
        dense.outline_vertices().len() > 150,
        "expected ~200 boundary points, got {}",
        dense.outline_vertices().len()
    );

    // Refinement on and off, same resolution: what the quality pass costs.
    let (no_refine_ms, no_refine) = time_it(
        "bodice, 10 mm, no refinement",
        20,
        &net,
        &notches,
        &TriangulateOptions::default().without_refinement(),
    );

    // The cost curve. Halving the target edge quadruples the triangle count, so
    // the interesting question is whether time is linear in output size.
    writeln!(std::io::stderr(), "\n--- cost curve in target_edge_mm ---").unwrap();
    let mut curve: Vec<(f64, f64, usize)> = Vec::new();
    for edge_mm in [40.0, 20.0, 10.0, 5.0, 2.5, 1.25] {
        let (ms, mesh) = time_it(
            &format!("bodice, {edge_mm:>5} mm target edge"),
            if edge_mm > 2.0 { 10 } else { 3 },
            &net,
            &notches,
            &TriangulateOptions::default().with_target_edge_mm(edge_mm),
        );
        curve.push((edge_mm, ms, mesh.triangle_count()));
    }

    // Stress case: a 400-gon with an internal line, at a fine resolution.
    writeln!(std::io::stderr(), "\n--- stress ---").unwrap();
    let poly: Vec<(f64, f64)> = (0..400)
        .map(|i| {
            let a = i as f64 / 400.0 * std::f64::consts::TAU;
            (a, 400.0 + 60.0 * (7.0 * a).sin())
        })
        .collect();
    let stress_net = star(&poly);
    let (stress_ms, stress) = time_it(
        "400-gon, 5 mm target edge",
        3,
        &stress_net,
        &[],
        &TriangulateOptions::default().with_target_edge_mm(5.0),
    );
    assert!(
        stress.triangle_count() > 50_000,
        "the stress case is not stressful: {} triangles",
        stress.triangle_count()
    );

    writeln!(
        std::io::stderr(),
        "\nrefinement overhead on the bodice: {:.3} ms -> {:.3} ms ({:.1}x); \
         throughput at the stress point: {:.0} tri/ms\n",
        no_refine_ms,
        bodice_ms,
        bodice_ms / no_refine_ms.max(1e-9),
        stress.triangle_count() as f64 / stress_ms,
    )
    .unwrap();

    let _ = (dense_ms, no_refine);
    if strict {
        assert!(
            bodice_ms < 8.3,
            "a bodice panel must triangulate inside one 120 fps frame, took {bodice_ms:.3} ms"
        );
    }
}
