// SPDX-License-Identifier: GPL-2.0-or-later
//! Regression + integration tests.
//!
//! The regression-oracle pattern (`docs/DEVELOPING.md` §5): a canonical
//! scene is stepped deterministically and its final vertex positions are
//! compared against a committed golden file within tolerance. Regenerate the
//! golden with `MGUSO_BLESS=1 cargo test -p engine-api --test regression`.
//! CI runs each canonical scene and fails on drift.

use engine_api::{Material, Object, Scene, SolverConfig};
use std::path::PathBuf;

/// A closed axis-aligned box mesh (8 corners, 12 triangles) — a realistic
/// auto-drape target with actual surface geometry, not just a point cloud.
fn box_mesh(min: [f32; 3], max: [f32; 3]) -> Object {
    let positions = vec![
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [max[0], min[1], max[2]],
        [min[0], min[1], max[2]],
        [min[0], max[1], min[2]],
        [max[0], max[1], min[2]],
        [max[0], max[1], max[2]],
        [min[0], max[1], max[2]],
    ];
    let tris = vec![
        [0, 2, 1],
        [0, 3, 2], // bottom
        [4, 5, 6],
        [4, 6, 7], // top
        [0, 1, 5],
        [0, 5, 4], // -z
        [2, 3, 7],
        [2, 7, 6], // +z
        [3, 0, 4],
        [3, 4, 7], // -x
        [1, 2, 6],
        [1, 6, 5], // +x
    ];
    Object {
        name: "table".into(),
        positions,
        tris,
        material: Material::default(),
        preset: None,
        pins: Vec::new(),
        is_static: false,
        normals: Vec::new(),
        uvs: Vec::new(),
        skin: None,
        animations: Vec::new(),
    }
}

/// Auto-drape end-to-end: a fabric preset plus a target mesh produce a scene that
/// builds, steps, and stays physical. Asserts the invariants this file already
/// uses elsewhere — all-finite, no floor tunneling — plus the two the auto-drape
/// layer is specifically responsible for: the cloth starts fully covering the
/// target from above, and it does not end up inside it.
#[test]
fn auto_drape_preset_scene_steps_and_stays_out_of_the_target() {
    use engine_api::{auto_drape, Aabb, DrapePreset, FabricPreset};

    let target = box_mesh([-0.3, 0.0, -0.3], [0.3, 0.6, 0.3]);
    let target_bb = Aabb::of(&target.positions).expect("target has vertices");
    // `Hard` is the coarsest preset (8x8) — the cheapest resolution that still has
    // a ring of overhang outside the target's footprint. Silk is the drapiest
    // fabric, so the droop is unambiguous at this ~0.9 m scale.
    let mut scene = auto_drape(&target, DrapePreset::Hard, FabricPreset::Silk)
        .expect("auto_drape builds a scene");
    // Larger steps than the 1e-3 default so the drape actually settles in-test.
    scene.config.dt = 1.0 / 120.0;
    scene.config.iterations = 5;

    let n_target = scene.objects[0].positions.len();
    let cloth0 = scene.objects[1].positions.clone();

    // Placement precondition: every cloth vertex starts above the target's top and
    // the sheet fully covers the target's XZ footprint (that is what makes it a
    // drape rather than a miss).
    let cloth_bb = Aabb::of(&cloth0).expect("cloth has vertices");
    assert!(
        cloth_bb.min[1] > target_bb.max[1],
        "cloth must start above the target: {} vs {}",
        cloth_bb.min[1],
        target_bb.max[1]
    );
    assert!(cloth_bb.min[0] < target_bb.min[0] && cloth_bb.max[0] > target_bb.max[0]);
    assert!(cloth_bb.min[2] < target_bb.min[2] && cloth_bb.max[2] > target_bb.max[2]);

    // Split the sheet by its STARTING footprint: vertices over the target are
    // supported, the ring outside it is free to fall. Comparing these two groups is
    // what makes "it draped" observable — the contact barrier lifts the whole sheet
    // by ~dhat, so an absolute drop below the start height measures the barrier,
    // not the drape.
    let overhang: Vec<usize> = (0..cloth0.len())
        .filter(|&i| {
            cloth0[i][0] < target_bb.min[0]
                || cloth0[i][0] > target_bb.max[0]
                || cloth0[i][2] < target_bb.min[2]
                || cloth0[i][2] > target_bb.max[2]
        })
        .collect();
    let supported: Vec<usize> = (0..cloth0.len())
        .filter(|i| !overhang.contains(i))
        .collect();
    assert!(
        !overhang.is_empty() && !supported.is_empty(),
        "auto-drape must produce both an overhang ({}) and a supported region ({})",
        overhang.len(),
        supported.len()
    );

    let mut solver = scene.build_implicit();
    solver.params.contact_kappa = 1.0e3;
    solver.params.contact_dhat = 5.0e-3;
    for _ in 0..300 {
        solver.step(&engine_core::linsolve::CgCpu::new());
    }

    let ps = &solver.state.positions;
    assert!(ps.iter().all(|p| p.is_finite()), "auto-drape sim blew up");

    // The static target is kinematic: its vertices must not have moved at all.
    for (i, p) in ps.iter().take(n_target).enumerate() {
        let want = target.positions[i];
        assert!(
            (p.x - want[0]).abs() < 1e-6
                && (p.y - want[1]).abs() < 1e-6
                && (p.z - want[2]).abs() < 1e-6,
            "static target vertex {i} moved: {p:?} vs {want:?}"
        );
    }

    // Cloth invariants. `tol` matches the contact barrier slack used by the
    // sphere-drape tests in this crate.
    let tol = 2.0e-2;
    let floor = solver.params.floor_y;
    for p in ps.iter().skip(n_target) {
        assert!(p.y >= floor - tol, "cloth fell through the floor: {p:?}");
        let inside = p.x > target_bb.min[0] + tol
            && p.x < target_bb.max[0] - tol
            && p.y > target_bb.min[1] + tol
            && p.y < target_bb.max[1] - tol
            && p.z > target_bb.min[2] + tol
            && p.z < target_bb.max[2] - tol;
        assert!(!inside, "cloth vertex penetrated the target: {p:?}");
    }

    // It actually draped: the unsupported ring hangs below the supported centre.
    let mean_y = |idx: &[usize]| -> f32 {
        idx.iter().map(|&i| ps[n_target + i].y).sum::<f32>() / idx.len() as f32
    };
    let (over, sup) = (mean_y(&overhang), mean_y(&supported));
    assert!(
        over < sup - 1.0e-3,
        "cloth did not drape: overhang mean y={over}, supported mean y={sup}"
    );
}

/// A pinned cloth grid draping under gravity — the canonical parity scene.
fn drape_scene() -> Scene {
    let n = 10usize;
    let mut positions = Vec::new();
    for j in 0..n {
        for i in 0..n {
            positions.push([i as f32 / (n - 1) as f32, 1.0, j as f32 / (n - 1) as f32]);
        }
    }
    let mut tris = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let v = (j * n + i) as u32;
            let r = v + n as u32;
            tris.push([v, v + 1, r]);
            tris.push([r, v + 1, r + 1]);
        }
    }
    Scene {
        config: SolverConfig {
            dt: 1.0e-3,
            substeps: 8,
            iterations: 6,
            frames: 60,
            floor_y: Some(-1.0),
            ..Default::default()
        },
        objects: vec![Object {
            name: "drape".into(),
            positions,
            tris,
            material: Material::default(),
            preset: None,
            pins: vec![0, (n - 1) as u32], // two top corners
            is_static: false,
            normals: Vec::new(),
            uvs: Vec::new(),
            skin: None,
            animations: Vec::new(),
        }],
        colliders: vec![],
    }
}

fn run_drape(steps: u32) -> Vec<[f32; 3]> {
    let mut solver = drape_scene().build();
    for _ in 0..steps {
        solver.step();
    }
    solver
        .state
        .positions
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect()
}

#[test]
fn drape_invariants_hold() {
    let mut solver = drape_scene().build();
    for _ in 0..500 {
        solver.step();
    }
    // No blow-up, no tunneling through the floor, pinned verts stay pinned.
    for p in &solver.state.positions {
        assert!(p.is_finite(), "non-finite position {p:?}");
        assert!(p.y >= -1.0 - 1e-3, "fell through floor: {p:?}");
    }
    assert_eq!(solver.state.positions[0], glam::Vec3::new(0.0, 1.0, 0.0));
}

#[test]
fn drape_is_deterministic() {
    assert_eq!(
        run_drape(120),
        run_drape(120),
        "same input must give same output"
    );
}

#[test]
fn drape_matches_golden() {
    let steps = 120;
    let positions = run_drape(steps);
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/drape.json");

    if std::env::var("MGUSO_BLESS").is_ok() || !golden_path.exists() {
        std::fs::write(
            &golden_path,
            serde_json::to_string_pretty(&positions).unwrap(),
        )
        .unwrap();
        eprintln!("blessed golden at {}", golden_path.display());
        return;
    }

    let golden: Vec<[f32; 3]> =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).unwrap()).unwrap();
    assert_eq!(golden.len(), positions.len(), "vertex count drift");
    let tol = 1e-4;
    let mut max_err = 0.0f32;
    for (g, p) in golden.iter().zip(&positions) {
        for k in 0..3 {
            max_err = max_err.max((g[k] - p[k]).abs());
        }
    }
    assert!(
        max_err <= tol,
        "regression drift: max_err={max_err} > tol={tol}"
    );
}

/// The `sphere_drape.json` example: a flat cloth falls past a sphere collider. With
/// the XPBD tier now honoring `Scene.colliders` (Session 49), the cloth is kept out
/// of the sphere. Rigorous WITH-vs-WITHOUT-collider contrast: without the collider
/// the cloth passes THROUGH the sphere (penetrates); with it, it does not. Exercises
/// the collider path end-to-end (JSON → Scene → build() → solve). (The cloth is
/// unpinned + frictionless, so it slides off the sphere rather than draping — hence
/// we test penetration, not a stable rest-on-sphere.)
#[test]
fn sphere_drape_example_collider_keeps_cloth_out() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/sphere_drape.json");
    let scene: Scene =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read sphere_drape.json"))
            .expect("parse sphere_drape.json");
    let (center, radius) = scene
        .colliders
        .iter()
        .find_map(|c| match *c {
            engine_api::ColliderSpec::Sphere { center, radius } => Some((center, radius)),
            _ => None,
        })
        .expect("example has a sphere collider");

    // Closest the cloth EVER gets to inside the sphere over the whole fall (the
    // cloth falls PAST the sphere to the floor, so penetration — if any — is
    // transient; the final frame alone would miss it).
    let min_signed_over_run = |scene: &Scene| -> f32 {
        let mut solver = scene.build();
        let mut m = f32::MAX;
        for _ in 0..600 {
            solver.step();
            for p in &solver.state.positions {
                assert!(p.is_finite(), "non-finite vertex {p:?}");
                let d = ((p.x - center[0]).powi(2)
                    + (p.y - center[1]).powi(2)
                    + (p.z - center[2]).powi(2))
                .sqrt()
                    - radius;
                m = m.min(d);
            }
        }
        m
    };

    let with = min_signed_over_run(&scene);
    let mut no_collider = scene.clone();
    no_collider.colliders.clear();
    let without = min_signed_over_run(&no_collider);

    // With the collider: no vertex penetrates (small numerical slack).
    assert!(
        with > -2.0e-2,
        "cloth penetrated the sphere with collider on: {with}"
    );
    // Without it: the cloth passes through the sphere (penetrates) — proving the
    // collider is what keeps it out, not the geometry happening to miss.
    assert!(
        without < -0.05,
        "collider off should let the cloth pass through the sphere, got {without}"
    );
}
