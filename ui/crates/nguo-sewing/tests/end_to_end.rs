// SPDX-License-Identifier: GPL-2.0-or-later
//! Draft → triangulate → arrange → sew, end to end (milestone M6).
//!
//! The *drape* half of the loop is tested in `nguo-engine-bridge`
//! (`inproc.rs::stitches_pull_two_sheets_together_and_their_absence_does_not`) and
//! in `nguo-app` (`src/drape.rs`, module `solve_tests` — unit tests rather than
//! `tests/*.rs` because `nguo-app` is a binary crate), because it needs a solver
//! and this crate deliberately has none. What is tested here is everything the solver is handed:
//! that it is finite, that it indexes only itself, that the seams line up, and
//! that it does not change from run to run.

use nguo_sewing::{arrange::Archetype, build, sample, Garment, SewOptions};

/// `TriangulateOptions::default().with_target_edge_mm(30.0)` is what the shell
/// ships. Everything here uses the shipped settings, so a regression in the
/// default is a regression in a test.
fn garment() -> Garment {
    build(&sample::bodice_block(), &SewOptions::default()).expect("the sample sews")
}

/// The whole contract of a mesh handed to a solver, on the garment the shell
/// opens with.
#[test]
fn the_sample_garment_is_a_valid_simulation_mesh() {
    let g = garment();

    // `Garment::check` is what `build` already ran, so re-running it here would
    // assert nothing new. These are the things it does *not* cover.
    assert_eq!(g.pieces.len(), 2);
    assert_eq!(g.seams.len(), 2);
    assert!(g.vertex_count() > 200, "{} vertices", g.vertex_count());
    assert!(g.triangle_count() > 300, "{} triangles", g.triangle_count());
    assert_eq!(g.uvs.len(), g.vertex_count());

    // Every triangle has real area. A zero-area triangle divides by zero when its
    // membrane element is built, which poisons the whole solve with NaN — and it
    // is the classic output of a bad constrained triangulation.
    for piece in &g.pieces {
        let worst = piece.mesh.min_angle_deg();
        assert!(
            worst > 5.0,
            "{}: a {worst:.2}° sliver triangle survived refinement",
            piece.name,
        );
    }

    // No degenerate triangle in 3D either — the placement could still collapse
    // one even if the flat mesh is clean.
    let area = |t: &[u32; 3]| {
        let [a, b, c] = t.map(|i| g.positions[i as usize]);
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
    };
    let smallest = g.tris.iter().map(area).fold(f32::INFINITY, f32::min);
    assert!(smallest > 1e-9, "a 3D triangle collapsed: {smallest} m²");

    // Total area is the pattern's own, because a wrap is isometric. This is the
    // assertion that catches an arranger that *scales* a panel — which would bake
    // the scale in as the fabric's rest state and the garment would never spring
    // back (`arrange.rs` module docs).
    let total: f32 = g.tris.iter().map(area).sum();
    let flat_m2 = (2.0 * sample::PANEL_W_MM * sample::PANEL_H_MM * 1e-6) as f32;
    assert!(
        (total / flat_m2 - 1.0).abs() < 0.01,
        "3D area {total:.4} m² against a flat {flat_m2:.4} m²: the wrap is not isometric",
    );
}

/// Seams close: every stitch's two barycentric points must start within the arc
/// the arranger left open, and the two sides must be *paired*, not merely near.
#[test]
fn every_seam_is_resolved_paired_and_within_the_arrangers_slack() {
    let g = garment();
    let slack = g.arrangement.slack_m as f64;

    for seam in &g.seams {
        assert!(
            seam.stitches >= 10,
            "seam {:?} has {} stitches",
            seam.seam,
            seam.stitches
        );
        assert!(!seam.flipped, "seam {:?} came out twisted", seam.seam);
        assert!(
            (seam.ease - 1.0).abs() < 1e-6,
            "equal-length sides reported {} ease",
            seam.ease,
        );
    }
    assert_eq!(
        g.stitches.len(),
        g.seams.iter().map(|s| s.stitches).sum::<usize>(),
    );

    // The per-stitch gap, not the total: a total can hide one wildly wrong pair.
    let worst = g.max_seam_gap_m();
    assert!(
        worst <= slack.max(1e-3) + 1e-3,
        "a stitch spans {worst:.4} m against {slack:.4} m of slack — the two sides are not paired",
    );

    // Both halves of every stitch must live in *different* pieces, or a seam has
    // sewn a panel to itself.
    let piece_of = |v: u32| {
        g.pieces
            .iter()
            .rev()
            .find(|p| v >= p.vertex_offset)
            .map(|p| p.piece)
            .expect("every vertex belongs to a piece")
    };
    for s in &g.stitches {
        assert_ne!(
            piece_of(s.index[0]),
            piece_of(s.index[3]),
            "stitch {:?} sews a piece to itself",
            s.index,
        );
    }
}

/// The arrangement must start the panels **outside** the body. A barrier method
/// cannot recover from geometry that starts inside its own collision volume, and
/// that is the one initial condition the "the solver untangles it" argument does
/// not cover.
#[test]
fn no_panel_starts_inside_the_body_cylinder() {
    let mut opts = SewOptions::default();
    opts.arrange.body_radius_m = 0.16;
    opts.arrange.center_xz = [0.0, 0.0];
    let g = build(&sample::bodice_block(), &opts).expect("sews");

    let closest = g
        .positions
        .iter()
        .map(|p| (p[0] * p[0] + p[2] * p[2]).sqrt())
        .fold(f32::INFINITY, f32::min);
    assert!(
        closest >= opts.arrange.body_radius_m,
        "a vertex sits {closest:.4} m from the axis, inside the {:.4} m body",
        opts.arrange.body_radius_m,
    );
    assert!(
        closest >= opts.arrange.body_radius_m + opts.arrange.gap_m - 1e-6,
        "the {} m clearance was not honoured: {closest:.4} m",
        opts.arrange.gap_m,
    );
}

/// Determinism. The same document must produce a bit-identical garment, because
/// the golden below is worthless otherwise — and because floating-point
/// non-determinism in a triangulator is usually hash iteration order leaking into
/// vertex order.
#[test]
fn the_same_document_produces_a_bit_identical_garment() {
    let a = garment();
    let b = garment();
    assert_eq!(a.positions, b.positions, "positions drifted");
    assert_eq!(a.tris, b.tris, "triangle order drifted");
    assert_eq!(a.uvs, b.uvs);
    assert_eq!(
        a.stitches.iter().map(|s| s.index).collect::<Vec<_>>(),
        b.stitches.iter().map(|s| s.index).collect::<Vec<_>>(),
        "stitch order drifted",
    );
    assert_eq!(
        a.stitches.iter().map(|s| s.weight).collect::<Vec<_>>(),
        b.stitches.iter().map(|s| s.weight).collect::<Vec<_>>(),
    );
}

/// The golden: a fixed garment's shape, to four decimal places of a metre
/// (0.1 mm — finer than any tolerance in the pipeline).
///
/// Bless with `NGUO_BLESS=1 cargo test -p nguo-sewing --test end_to_end`.
/// It records counts, bounds and a checksum rather than every vertex: a
/// 250-vertex dump is unreadable in a diff, whereas these five numbers move if
/// **any** vertex does and say which way.
#[test]
fn a_fixed_garment_matches_its_golden() {
    let g = garment();
    let bounds = g.bounds().expect("geometry");
    let actual = serde_json::json!({
        "vertices": g.vertex_count(),
        "triangles": g.triangle_count(),
        "seams": g.seams.len(),
        "stitches": g.stitches.len(),
        "bounds_min_mm": bounds.min.map(|c| round(c as f64 * 1000.0)),
        "bounds_max_mm": bounds.max.map(|c| round(c as f64 * 1000.0)),
        "radius_mm": g.arrangement.radius_m.map(|r| round(r as f64 * 1000.0)),
        "circumference_mm": round(g.arrangement.circumference_m as f64 * 1000.0),
        "slack_mm": round(g.arrangement.slack_m as f64 * 1000.0),
        "max_seam_gap_mm": round(g.max_seam_gap_m() * 1000.0),
        // Every position and every stitch index, in one order-sensitive number.
        "checksum": checksum(&g),
    });

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/bodice_drape.json");
    if std::env::var_os("NGUO_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().expect("has a parent")).expect("mkdir");
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string_pretty(&actual).expect("json")),
        )
        .expect("write the golden");
        // Not a silent pass: a blessing run must be visible in the output, and
        // `eprintln!` is swallowed by libtest's capture.
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "BLESSED {}", path.display());
        return;
    }

    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — bless it with NGUO_BLESS=1 if this is a new golden",
            path.display()
        )
    });
    let expected: serde_json::Value = serde_json::from_str(&raw).expect("the golden is json");

    // A floor, so a golden of `{}` (or of a garment with no geometry) cannot pass
    // by matching an equally empty run.
    assert!(
        expected["vertices"].as_u64().unwrap_or(0) > 200,
        "the golden itself is empty: {expected}",
    );
    assert!(expected["stitches"].as_u64().unwrap_or(0) > 20);

    if actual != expected {
        // Name the fields that moved rather than dumping two blobs.
        let mut moved = Vec::new();
        for (k, v) in expected.as_object().expect("object") {
            if actual.get(k) != Some(v) {
                moved.push(format!(
                    "{k}: {v} -> {}",
                    actual.get(k).unwrap_or(&serde_json::Value::Null)
                ));
            }
        }
        panic!("the garment changed shape:\n  {}", moved.join("\n  "));
    }
}

fn round(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// Order-sensitive FNV-1a over every position bit pattern and every stitch index.
/// Bit patterns, not rounded decimals: the point of a determinism checksum is to
/// catch a 1-ULP drift, which rounding would hide.
fn checksum(g: &Garment) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    for p in &g.positions {
        for c in p {
            eat(&c.to_bits().to_le_bytes());
        }
    }
    for t in &g.tris {
        for i in t {
            eat(&i.to_le_bytes());
        }
    }
    for s in &g.stitches {
        for i in &s.index {
            eat(&i.to_le_bytes());
        }
        for w in &s.weight {
            eat(&w.to_bits().to_le_bytes());
        }
    }
    format!("{h:016x}")
}

/// The resolution/cost trade-off the shipped `SewOptions` default is chosen from.
/// Prints real numbers; asserts only the monotone relationship, because the
/// absolute cost is this machine's.
#[test]
fn mesh_resolution_cost() {
    use std::io::Write;
    let doc = sample::bodice_block();
    let mut rows = Vec::new();
    for mm in [60.0, 30.0, 15.0, 10.0] {
        let opts = SewOptions {
            triangulate: nguo_triangulate::TriangulateOptions::default().with_target_edge_mm(mm),
            ..SewOptions::default()
        };
        let started = std::time::Instant::now();
        let g = build(&doc, &opts).expect("sews at every resolution");
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let _ = writeln!(
            std::io::stderr(),
            "target {mm:>5.1} mm -> {:>6} verts, {:>6} tris, {:>5} stitches, build {ms:>7.2} ms \
             (validate {:.2} / triangulate {:.2} / arrange {:.2} / sew {:.2})",
            g.vertex_count(),
            g.triangle_count(),
            g.stitches.len(),
            g.timings.validate_ms,
            g.timings.triangulate_ms,
            g.timings.arrange_ms,
            g.timings.sew_ms,
        );
        rows.push((mm, g.vertex_count(), g.triangle_count()));
    }

    // Monotone: halving the target edge must not *reduce* the mesh.
    for w in rows.windows(2) {
        assert!(
            w[1].1 > w[0].1,
            "target {} mm produced {} verts, {} mm produced {}",
            w[0].0,
            w[0].1,
            w[1].0,
            w[1].1,
        );
    }
    // The shipped default must be in the middle of that range, not at an end.
    let shipped = build(&doc, &SewOptions::default()).expect("sews");
    assert!(shipped.vertex_count() > rows[0].1);
    assert!(shipped.vertex_count() < rows[3].1);
}

/// A garment with no seams at all still has to produce something drapeable —
/// this is the "drape under gravity without a body" fallback, one level up.
#[test]
fn an_unsewn_document_still_builds_and_says_it_has_no_seams() {
    let mut doc = sample::bodice_block();
    let ids: Vec<_> = doc.seams().ids().collect();
    for id in ids {
        doc.apply_edit(nguo_doc::Edit::RemoveSeam { id })
            .expect("a seam can always be removed");
    }
    let g = build(&doc, &SewOptions::default()).expect("an unsewn draft still builds");
    assert!(g.stitches.is_empty());
    assert!(g.seams.is_empty());
    // ...and it is still a valid mesh, so the solver gets loose panels rather
    // than a refusal.
    assert!(g.check().is_ok(), "{:?}", g.check());
    assert_eq!(g.max_seam_gap_m(), 0.0);
}

/// `Archetype::Flat` is the fallback for a draft with no seam graph to walk. It
/// must still produce finite, non-overlapping geometry in front of the body.
#[test]
fn the_flat_archetype_stands_the_panels_up_side_by_side() {
    let mut opts = SewOptions::default();
    opts.arrange.archetype = Archetype::Flat;
    let g = build(&sample::bodice_block(), &opts).expect("sews flat");

    assert!(g.arrangement.radius_m.is_none());
    assert_eq!(g.arrangement.slack_m, 0.0);
    let bounds = g.bounds().expect("geometry");
    // One plane: every vertex at the same Z, in front of the body.
    assert!(
        (bounds.max[2] - bounds.min[2]).abs() < 1e-6,
        "the flat layout is not flat: {bounds:?}",
    );
    assert!(
        bounds.min[2] > 0.0,
        "the panels are behind the body: {bounds:?}"
    );
    // Two 480 mm panels side by side.
    assert!(
        (bounds.max[0] - bounds.min[0] - (2.0 * sample::PANEL_W_MM * 1e-3) as f32).abs() < 1e-3,
        "{bounds:?}",
    );
}
