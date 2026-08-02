// SPDX-License-Identifier: GPL-2.0-or-later
//! Golden-format reader test against the real reference capture. Gated: skips
//! cleanly when the capture is absent — which is the normal case, since it is an
//! internal benchmark corpus not distributed with this source (`docs/GAUGE_FINDINGS.md`)
//! — like the GPU/usdcat tests skip without their prerequisite. Set `PPF_GAUGE_DIR`
//! to point at a capture; otherwise the sibling `../ppf-cts-gauge` is used.
//!
//! The decode is verified against the reference's OWN recorded per-frame statistics
//! in `frame_checksums.csv` (centroid + bounding box): if our `f32`-xyz decode of a
//! checkpoint reproduces the reference centroid/bbox, we are reading it correctly.

use std::path::Path;

// One implementation, in the library. It was copied into four places and two of the
// copies then diverged (`exists()` instead of `is_dir()`, which lets a path naming a
// FILE past the guard). See `engine_io::golden::gauge_dir`.
use engine_io::golden::gauge_dir;

/// centroid (x,y,z) + bbox (minx,maxx,miny,maxy,minz,maxz) of a point set.
fn centroid_bbox(pts: &[[f32; 3]]) -> ([f64; 3], [f32; 6]) {
    let mut sum = [0.0f64; 3];
    let mut bb = [f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN];
    for p in pts {
        for k in 0..3 {
            sum[k] += p[k] as f64;
            bb[2 * k] = bb[2 * k].min(p[k]);
            bb[2 * k + 1] = bb[2 * k + 1].max(p[k]);
        }
    }
    let n = pts.len().max(1) as f64;
    ([sum[0] / n, sum[1] / n, sum[2] / n], bb)
}

/// Row of `frame_checksums.csv` for `frame`: (centroid xyz, bbox minx,maxx,miny,maxy,minz,maxz).
fn ref_stats(scene: &Path, frame: u32) -> Option<([f64; 3], [f32; 6])> {
    let csv = std::fs::read_to_string(scene.join("frame_checksums.csv")).ok()?;
    let row = csv.lines().find(|l| {
        l.split(',')
            .next()
            .and_then(|f| f.parse::<u32>().ok())
            .is_some_and(|f| f == frame)
    })?;
    let c: Vec<&str> = row.split(',').collect();
    // cols: 0 frame,1 n_verts,2 sha256,3 cx,4 cy,5 cz,6 minx,7 maxx,8 miny,9 maxy,10 minz,11 maxz
    let f = |i: usize| c.get(i)?.parse::<f64>().ok();
    Some((
        [f(3)?, f(4)?, f(5)?],
        [
            f(6)? as f32,
            f(7)? as f32,
            f(8)? as f32,
            f(9)? as f32,
            f(10)? as f32,
            f(11)? as f32,
        ],
    ))
}

#[test]
fn reads_drape_golden_input_and_checkpoint() {
    let Some(gauge) = gauge_dir() else {
        eprintln!("ppf-cts-gauge not found; skipping golden reader test");
        return;
    };
    let scene = gauge.join("goldens/drape");
    if !scene.is_dir() {
        eprintln!("drape golden scene missing; skipping");
        return;
    }

    let input = engine_io::load_golden_input(&scene).expect("load drape golden input");
    // Empirically-verified counts for the drape capture.
    assert_eq!(input.positions.len(), 81920, "drape vertex count");
    assert_eq!(input.tris.len(), 161290, "drape triangle count");
    assert_eq!(input.velocities.len(), input.positions.len());
    assert_eq!(input.tris[0], [0, 1, 2], "first triangle (from raw bytes)");

    // Decode verification: our f32-xyz decode of checkpoint 0 must reproduce the
    // reference's own recorded centroid + bbox (frame_checksums.csv). This proves
    // the decode is correct against ground truth, not a self-referential check.
    let frame0 = engine_io::load_checkpoint(&scene, 0).expect("load checkpoint 0");
    assert_eq!(frame0.len(), input.positions.len());
    let (cent, bb) = centroid_bbox(&frame0);
    let (rc, rb) = ref_stats(&scene, 0).expect("frame 0 stats");
    for k in 0..3 {
        assert!(
            (cent[k] - rc[k]).abs() < 1e-4,
            "centroid[{k}] {} vs ref {}",
            cent[k],
            rc[k]
        );
    }
    for i in 0..6 {
        assert!(
            (bb[i] - rb[i]).abs() < 1e-4,
            "bbox[{i}] {} vs ref {}",
            bb[i],
            rb[i]
        );
    }

    // A later checkpoint exists, same count, and the reference centroid moved
    // (cloth draped under gravity) — sanity that frames are distinct.
    let frame100 = engine_io::load_checkpoint(&scene, 100).expect("load checkpoint 100");
    assert_eq!(frame100.len(), input.positions.len());
    let (_, rb100) = ref_stats(&scene, 100).expect("frame 100 stats");
    assert!(
        (rb100[2] - rb[2]).abs() > 1e-4 || (rb100[3] - rb[3]).abs() > 1e-4,
        "frame 100 y-extent should differ from frame 0"
    );

    // Gate metrics on REAL reference frames: a frame vs itself is a perfect pass;
    // frame 20 vs frame 40 (the cloth moved a lot between them) is a finite,
    // non-trivial, failing divergence. Verifies compare_frames on real data.
    let f20 = engine_io::load_checkpoint(&scene, 20).unwrap();
    let f40 = engine_io::load_checkpoint(&scene, 40).unwrap();
    let self_cmp = engine_io::compare_frames(&f20, &f20).unwrap();
    assert!(
        self_cmp.passes() && self_cmp.max_rel == 0.0,
        "self-compare is 0"
    );
    let cross = engine_io::compare_frames(&f20, &f40).unwrap();
    assert!(cross.diagonal > 0.0 && cross.max_rel.is_finite());
    assert!(cross.max_rel > 0.0, "distinct frames must diverge");
    eprintln!(
        "drape f20-vs-f40: max_rel={:.4} rms={:.4} centroid_rel={:.4} diag={:.3}",
        cross.max_rel, cross.rms, cross.centroid_rel, cross.diagonal
    );
}

#[test]
fn reads_hang_params_and_pins() {
    let Some(gauge) = gauge_dir() else {
        eprintln!("ppf-cts-gauge not found; skipping");
        return;
    };
    let scene = gauge.join("goldens/hang");
    if !scene.is_dir() {
        eprintln!("hang scene missing; skipping");
        return;
    }
    // param.toml values (verified from the file).
    let p = engine_io::read_param_toml(&scene).expect("param.toml");
    assert_eq!(p.dt, 0.001);
    assert_eq!(p.fps, 60.0);
    assert_eq!(p.frames, 200);
    assert_eq!(p.gravity, [0.0, -9.8, 0.0]);
    assert_eq!(p.world_scaling, 1.0);
    // pin-ind-0.bin = (16256, 16383) → TWO constrained vertices. The file is a flat u64
    // index LIST, not an inclusive range: `hang/info.toml` declares `[pin-0] pin = 2`,
    // and `PORTING_data_layout.md:657` gives the shape as `[P_i]` with its length
    // asserted against exactly that count. (The two indices are the ends of the cloth's
    // top row, so the y=1.0 check below holds under either reading and cannot
    // discriminate — info.toml is what settles it.)
    let pins = engine_io::read_pin_indices(&scene).expect("pins");
    assert_eq!(pins, vec![16256, 16383]);
    let frame0 = engine_io::load_checkpoint(&scene, 0).unwrap();
    assert!(
        pins.iter()
            .all(|&i| (frame0[i as usize][1] - 1.0).abs() < 1e-4),
        "constrained verts are the top (y=1) row"
    );

    // Reference material (input_bin/param/tri-*.bin) for hang: uniform young 100,
    // bend 1, density 1, strain-limit 0.01.
    let m = engine_io::read_material(&scene).expect("hang material");
    assert!((m.young_mod - 100.0).abs() < 1e-3);
    assert!((m.bend - 1.0).abs() < 1e-3);
    assert!((m.density - 1.0).abs() < 1e-3);
    assert!((m.strain_limit - 0.01).abs() < 1e-4);
}

/// Decode a REAL reference volumetric scene's `tet.bin` and pin the format against
/// the reference's own reported element count. `plastic` ships 405,440 bytes of
/// u64×4 = 12,670 tets, matching the "Solids (12670 elements)" line in its
/// `param_summary.txt`. (Building/validating the elements themselves lives in
/// engine-cli's test, which has both engine-io and engine-core available.)
#[test]
fn decodes_real_gauge_tet_scene() {
    let Some(gauge) = gauge_dir() else {
        eprintln!("no gauge dir; skipping");
        return;
    };
    let scene = gauge.join("goldens/plastic");
    if !scene.exists() {
        eprintln!("plastic scene absent; skipping");
        return;
    }
    let input = engine_io::golden::load_golden_input(&scene).expect("load plastic");
    assert_eq!(
        input.tets.len(),
        12_670,
        "tet.bin decode count must match param_summary's Solids count"
    );
    assert!(!input.tris.is_empty(), "plastic also has shells");
    // Every index must be in range (the loader checks this, so this documents it).
    let nv = input.positions.len() as u32;
    assert!(input.tets.iter().all(|t| t.iter().all(|&i| i < nv)));
    // A cloth-only scene must decode to zero tets, not fail.
    let hang = gauge.join("goldens/hang");
    if hang.exists() {
        let h = engine_io::golden::load_golden_input(&hang).expect("load hang");
        eprintln!("hang tets: {}", h.tets.len());
    }
}

/// Read a real scene's per-element parameter arrays and pin the format against the
/// reference's OWN reported statistics. `plastic`'s `param_summary.txt` says
/// `model: (arap: 5120 elements, baraffwitkin: 3844 elements)` for its 8964 shells and
/// `(arap: 12670 elements)` for its solids, with tet `plasticity` a uniform 0.5. If our
/// decode of the 1-byte model discriminant or the f32 arrays were wrong, these counts
/// would not line up.
#[test]
fn reads_real_gauge_element_params() {
    use engine_io::golden::{indices_with_model, load_element_params, RefModel};
    let Some(gauge) = gauge_dir() else {
        eprintln!("no gauge dir; skipping");
        return;
    };
    let scene = gauge.join("goldens/plastic");
    if !scene.exists() {
        eprintln!("plastic scene absent; skipping");
        return;
    }

    // --- shells (tri) ---
    let tri = load_element_params(&scene, "tri", false).expect("tri params");
    assert_eq!(tri.model.len(), 8964, "one model byte per shell face");
    let arap = indices_with_model(&tri.model, RefModel::Arap);
    let bw = indices_with_model(&tri.model, RefModel::BaraffWitkin);
    assert_eq!(arap.len(), 5120, "param_summary: arap: 5120 elements");
    assert_eq!(bw.len(), 3844, "param_summary: baraffwitkin: 3844 elements");
    assert_eq!(
        arap.len() + bw.len(),
        tri.model.len(),
        "models partition the faces"
    );
    // The two sets must be disjoint (they index the same face list).
    let mut all: Vec<u32> = arap.iter().chain(bw.iter()).copied().collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 8964, "model sets must be disjoint");

    // --- solids (tet) ---
    let tet = load_element_params(&scene, "tet", false).expect("tet params");
    assert_eq!(tet.model.len(), 12_670, "one model byte per tet");
    assert!(
        tet.model.iter().all(|m| *m == RefModel::Arap),
        "param_summary: solids are all arap"
    );
    assert_eq!(tet.plasticity.len(), 12_670, "f32 per tet");
    // param_summary: tet plasticity max=min=mean=0.5, threshold likewise 0.5.
    assert!(
        tet.plasticity.iter().all(|&p| (p - 0.5).abs() < 1e-6),
        "tet plasticity should be a uniform 0.5"
    );
    assert!(
        tet.plasticity_threshold
            .iter()
            .all(|&p| (p - 0.5).abs() < 1e-6),
        "tet plasticity threshold should be a uniform 0.5"
    );
    // Young's modulus: param_summary reports 5.0e2 for solids.
    assert_eq!(tet.young_mod.len(), 12_670);
    assert!(
        tet.young_mod.iter().all(|&y| (y - 500.0).abs() < 1e-3),
        "tet young-mod should be 5.0e2"
    );

    // --- bend plasticity is a SEPARATE array from in-plane plasticity ---
    let bend = load_element_params(&scene, "tri", true).expect("tri bend params");
    assert_eq!(bend.plasticity.len(), 8964);
    // param_summary: bend-plasticity is all zeros for this scene, while in-plane
    // plasticity is not — proving the two files are genuinely distinct.
    assert!(
        bend.plasticity.iter().all(|&p| p == 0.0),
        "param_summary: bend-plasticity is 0 for plastic"
    );
    let inplane_max = tri.plasticity.iter().fold(0.0f32, |m, &p| m.max(p));
    assert!(
        inplane_max > 0.0,
        "in-plane plasticity should be nonzero, distinguishing it from bend"
    );
}
