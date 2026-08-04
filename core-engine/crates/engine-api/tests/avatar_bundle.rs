// SPDX-License-Identifier: GPL-2.0-or-later
//! Avatar collider integration — loads the **real** `*_collision.glb` proxies from
//! the avatar bundle, builds drape scenes, steps them, and asserts the invariants
//! this crate already uses (`regression.rs`): all-finite, the kinematic body does
//! not move, and the cloth does not end up inside it.
//!
//! Everything is driven off `assets/avatars/data/manifest.json` — no avatar id, no
//! file path and no vertex count is hardcoded here. The manifest keys `avatars` by
//! sex (`male`, `female`), and every test runs over **all** of them, so adding or
//! renaming a body changes the coverage without touching this file.
//!
//! # Gating
//!
//! The bundle is committed (`7ce7827`) but its meshes are **git-LFS**, so a plain
//! `git clone` or an `actions/checkout` without `lfs: true` still has no usable
//! bundle. These tests **skip** in both cases — no `data/manifest.json`, *or* a
//! collision proxy that is a git-LFS pointer stub rather than a GLB — loudly, on
//! stderr. `$NGUO_AVATAR_BUNDLE` relocates the root. Set
//! `MGUSO_REQUIRE_AVATAR_BUNDLE=1` (CI, or any machine that has the bundle) to turn
//! the skip into a failure, so a silently vanished asset cannot quietly disable the
//! whole file. Every test that does run asserts a **floor** on what it loaded and on
//! what the simulation did, so a stub proxy or a no-op step fails rather than passes.
//!
//! The manifest declares 18 file paths per bundle and only 8 exist (both `.blend`,
//! both `glb.lod0` and all six `unity_fbx` are absent). Declared-but-absent is
//! treated as **optional**: only the file a test actually opens is required.
//!
//! # Why the stepped tests simulate only the crown
//!
//! Measured by [`collision_proxy_step_cost`] below (Apple Silicon, `--release`,
//! dt = 1/120, one 36-vertex patch on the body). D-17 — the static mesh collider —
//! has since landed in `engine_core::broadphase`, so BOTH columns are given:
//!
//! | body in the scene | tris | self-collision | before D-17 | after D-17 |
//! |---|---|---|---|---|
//! | whole proxy, male | 15 050 | on | **9.724** | **0.235** |
//! | whole proxy, female | 15 050 | on | **15.951** | **0.245** |
//! | whole proxy, male | 15 050 | off | 0.068 | 0.068 |
//! | whole proxy, female | 15 050 | off | 0.154 | 0.069 |
//! | crown region crop | 306–333 | on | 0.065–0.087 | 0.022–0.026 |
//!
//! After-column figures are the median of three consecutive runs on an otherwise idle
//! machine; a single run under load reads up to 2.4x higher on BOTH columns, which is
//! why the guard below asserts the on/off RATIO and not an absolute.
//!
//! The old 143× was the body's own 15 000 triangles going through the cloth
//! self-collision broad phase as if they could collide with each other. The fix is a
//! movability filter on the candidate pairs (`engine_core::broadphase::any_movable`):
//! a pair whose every participant is immovable is inert, so it is never emitted.
//! What remains is ~4× the self-collision-off cost and it is honest work — every
//! query still SCANS the static geometry to find what is near the cloth, because the
//! grid is rebuilt per call rather than cached across steps.
//!
//! The stepped tests still simulate only the crown: it is ~12× cheaper again, and
//! keeping CI cost down is worth more than re-covering the same solver path. The full
//! proxy is loaded, built and lowered in
//! [`full_proxy_lowers_into_a_static_collider_scene`] and now also STEPPED, with the
//! regression guard, in [`collision_proxy_step_cost`].

use engine_api::avatar::{
    avatar_collider, avatar_collider_near, drape_on_avatar, GarmentPlacement,
};
use engine_api::{Aabb, FabricPreset, Material, Object, Scene};
use engine_core::linsolve::CgCpu;
use std::collections::HashSet;
use std::path::PathBuf;

/// Loud skip. `None` so a test can `else { return; }`, but a hard failure when
/// `MGUSO_REQUIRE_AVATAR_BUNDLE` says this run must really cover the avatar path.
///
/// Written to fd 2 directly, not via `eprintln!`: libtest's output capture only
/// intercepts the print macros, so a macro'd skip notice is swallowed unless the
/// test fails or `--nocapture` is passed — i.e. not loud at all.
fn skip<T>(msg: String) -> Option<T> {
    assert!(
        std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none(),
        "MGUSO_REQUIRE_AVATAR_BUNDLE is set but {msg}"
    );
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {msg}");
    None
}

/// The bundle root: `$NGUO_AVATAR_BUNDLE`, else `assets/avatars/`, holding
/// `data/manifest.json`. Same override `nguo_avatar::default_root` and
/// `make test-avatars` use, so one variable relocates the bundle for every
/// consumer.
fn bundle_root() -> Option<PathBuf> {
    let root = match std::env::var_os("NGUO_AVATAR_BUNDLE") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../assets/avatars"),
    };
    if root.join("data/manifest.json").is_file() {
        return Some(root);
    }
    skip(format!(
        "avatar bundle not present at {} (no data/manifest.json) — the avatar collider \
         layer is NOT covered by this run",
        root.display()
    ))
}

/// One avatar's collision tier, exactly as the bundle declares it. The counts and
/// deviations are the bundle's own published numbers — an oracle produced by a
/// different toolchain (Blender) than the one under test, so a bundle swap that
/// silently changes the mesh fails instead of passing.
struct Col {
    sex: String,
    path: PathBuf,
    /// `tiers[COL].verts` — Blender's count, i.e. **distinct positions**, which is
    /// lower than the loaded glTF count (seam splitting). See
    /// [`collision_mesh_matches_the_manifest_col_tier`].
    verts: usize,
    tris: usize,
    stature_m: f64,
    /// `verification.collision_proxy_posed_deviation_mm`.
    deviation_mm: f64,
    /// `verification.silhouette_mismatch_fraction.COL`.
    silhouette: f64,
}

fn f64_at(v: &serde_json::Value, path: &[&str]) -> f64 {
    let mut cur = v;
    for k in path {
        cur = &cur[k];
    }
    cur.as_f64()
        .unwrap_or_else(|| panic!("manifest {path:?} is not a number: {cur}"))
}

/// Every collision proxy the manifest declares. Asserts the bundle shape itself:
/// metres, 30 fps, and a `male`/`female` dict rather than a list.
fn collision_tiers() -> Option<Vec<Col>> {
    let root = bundle_root()?;
    let text = std::fs::read_to_string(root.join("data/manifest.json")).expect("read manifest");
    let m: serde_json::Value = serde_json::from_str(&text).expect("parse manifest");

    // Units are the one manifest field this crate's maths depends on: every
    // placement number, contact margin and the stature floors below are metres.
    assert_eq!(
        m["conventions"]["units"], "metres",
        "the solver works in metres; a rescaled bundle breaks every placement number"
    );
    assert_eq!(m["conventions"]["fps"], 30, "animation rate changed");
    assert_eq!(m["conventions"]["gltf_vrm"]["up"], "+Y");
    assert_eq!(m["conventions"]["gltf_vrm"]["forward"], "+Z");

    let avatars = m["avatars"]
        .as_object()
        .expect("manifest `avatars` must be a dict keyed by sex");
    let mut out = Vec::new();
    for (sex, a) in avatars {
        let col = a["tiers"]
            .as_array()
            .expect("tiers array")
            .iter()
            .find(|t| t["tier"] == "COL")
            .unwrap_or_else(|| panic!("{sex} has no COL tier"));
        assert_eq!(
            col["role"], "cloth collision body",
            "{sex} COL tier is no longer the cloth collision body"
        );
        let rel = a["files"]["glb"]["collision"]
            .as_str()
            .unwrap_or_else(|| panic!("{sex} declares no files.glb.collision"));
        out.push(Col {
            sex: sex.clone(),
            path: root.join(rel),
            verts: f64_at(col, &["verts"]) as usize,
            tris: f64_at(col, &["tris"]) as usize,
            stature_m: f64_at(a, &["stature_m"]),
            deviation_mm: f64_at(a, &["verification", "collision_proxy_posed_deviation_mm"]),
            silhouette: f64_at(a, &["verification", "silhouette_mismatch_fraction", "COL"]),
        });
    }
    out.sort_by(|a, b| a.sex.cmp(&b.sex));
    assert_eq!(
        out.len(),
        2,
        "expected two avatars (male, female), got {:?}",
        out.iter().map(|c| &c.sex).collect::<Vec<_>>()
    );
    // The GLBs are git-LFS; the manifest is not. A checkout without LFS therefore
    // parses the manifest fine and leaves ~130 bytes of pointer text where each
    // proxy should be. That file EXISTS, so the presence gate above is satisfied
    // and the tests used to fail inside the glTF reader instead of skipping — the
    // exact way `main` went red on 7ce7827. A pointer is an absent bundle.
    if let Some(stub) = out.iter().find(|c| engine_io::is_lfs_pointer(&c.path)) {
        return skip(format!(
            "{} is a git-LFS pointer, not the collision proxy — this checkout has no \
             avatar bundle hydrated. Run `git lfs pull` (in CI, check out with \
             `lfs: true`). The avatar collider layer is NOT covered by this run",
            stub.path.display()
        ));
    }
    Some(out)
}

/// Load a proxy through `engine_io::load_mesh` and assert the floor: a real body
/// mesh in metres, standing at the manifest's stature.
fn load(col: &Col) -> Object {
    assert!(
        col.path.is_file(),
        "manifest declares {} but it is absent — the collision proxy is required",
        col.path.display()
    );
    let body = engine_io::load_mesh(&col.path)
        .unwrap_or_else(|e| panic!("load {}: {e}", col.path.display()));
    assert!(
        body.tris.len() >= 1000 && body.positions.len() >= 1000,
        "{} is {} verts / {} tris — that is not a body mesh",
        col.sex,
        body.positions.len(),
        body.tris.len()
    );
    let stature = Aabb::of(&body.positions).unwrap().extent()[1];
    assert!(
        ((col.stature_m - 0.02)..=(col.stature_m + 0.02)).contains(&(stature as f64)),
        "{} proxy stature {stature} m ≠ manifest {} m — wrong file or wrong units",
        col.sex,
        col.stature_m
    );
    body
}

/// An n×n cloth patch spanning `over` in XZ grown by `margin`, authored at y = 0.
/// The placement layer lifts it into position. Reuses `auto_drape::cloth_plane`.
fn patch(over: Aabb, n: usize, margin: f32) -> Object {
    let (positions, tris, uvs) = engine_api::auto_drape::cloth_plane(over, n, margin, 0.0);
    Object {
        name: "hat".into(),
        positions,
        tris,
        material: Material::default(),
        preset: None,
        pins: Vec::new(),
        is_static: false,
        normals: Vec::new(),
        uvs,
        skin: None,
        animations: Vec::new(),
    }
}

/// 6 steps at dt = 1/120 is 0.05 s of simulated time: free fall travels 12.3 mm,
/// which is what makes "did not sink into the head" discriminating, and the
/// contact barrier is 10 mm so the cloth never outruns it.
const STEPS: usize = 6;
const DT: f32 = 1.0 / 120.0;
const CONTACT_DHAT: f32 = 1.0e-2;
const CONTACT_KAPPA: f32 = 1.0e4;
/// Crown crop depth. 0.05 m keeps ~235–261 triangles of the 15 000 — the bracket the
/// old (2.9× coarser) bundle's 0.10 m crop sat in. See the module docs.
const CROWN: f32 = 0.05;

/// The XZ footprint of the topmost `depth` of `positions` — the crown of the head.
/// Cloth over this patch is what the body has to hold up, so `depth` must stay well
/// inside [`CROWN`]: a cap as wide as the whole crop puts patch vertices over the
/// *sides* of the head, where the body legitimately cannot hold them up and the
/// "did not sink" invariant stops discriminating.
fn head_cap(positions: &[[f32; 3]], depth: f32) -> Aabb {
    let top_y = Aabb::of(positions).unwrap().max[1] - depth;
    let top: Vec<[f32; 3]> = positions.iter().copied().filter(|p| p[1] > top_y).collect();
    assert!(top.len() >= 8, "only {} crown vertices", top.len());
    Aabb::of(&top).unwrap()
}

/// The canonical stepped scene: a 6×6 cloth patch dropped onto the crown of a real
/// proxy, cropped for CI cost through the shipped [`avatar_collider_near`] (see the
/// module docs).
fn stepped_scene(body: &Object) -> (Scene, Object, Aabb) {
    let bb = Aabb::of(&body.positions).unwrap();
    let crown = Aabb {
        min: [bb.min[0], bb.max[1] - CROWN, bb.min[2]],
        max: bb.max,
    };
    let head = avatar_collider_near(body, crown, 0.0).expect("crown region");
    assert!(
        (50..=500).contains(&head.tris.len()),
        "crown crop kept {} triangles — the proxy is not what we think",
        head.tris.len()
    );
    let cap = head_cap(&head.positions, CROWN * 0.5);
    let mut scene = drape_on_avatar(
        &head,
        &patch(cap, 6, 0.03),
        FabricPreset::Cotton,
        GarmentPlacement::DropFromAbove,
    )
    .expect("avatar drape scene");
    scene.config.dt = DT;
    scene.config.iterations = 3;
    (scene, head, cap)
}

fn run(scene: &Scene) -> Vec<[f32; 3]> {
    let mut solver = scene.build_implicit();
    solver.params.contact_kappa = CONTACT_KAPPA;
    solver.params.contact_dhat = CONTACT_DHAT;
    for _ in 0..STEPS {
        solver.step(&CgCpu::new());
    }
    assert!(
        solver.state.positions.iter().all(|p| p.is_finite()),
        "avatar drape blew up"
    );
    solver
        .state
        .positions
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect()
}

/// Regression against the bundle's **own** numbers — an oracle from a different
/// toolchain (Blender) than the one under test.
///
/// **Triangles are the invariant that holds exactly**: 15 000, both avatars, loaded
/// and declared. Vertices are *not* — `tiers[COL].verts` is the Blender mesh count
/// (7 498 male / 7 500 female) and glTF splits vertices along UV and normal seams, so
/// `engine_io::load_mesh` legitimately returns more (10 230 / 10 531). Asserting that
/// equality would be asserting a false number.
///
/// What is checked instead is a *derived* relationship: the loaded count is at least
/// the declared one, and the count of **distinct positions** is exactly the declared
/// one — a seam split duplicates a vertex's position bit-for-bit, so deduplicating
/// positions undoes it. Verified: 7 498 / 7 500, matching the manifest. If a future
/// re-export perturbs coordinates rather than copying them, this fails, which is the
/// correct outcome for "the mesh changed".
#[test]
fn collision_mesh_matches_the_manifest_col_tier() {
    let Some(cols) = collision_tiers() else {
        return;
    };
    for col in &cols {
        let body = load(col);
        assert_eq!(
            body.tris.len(),
            col.tris,
            "{} collision tris {} ≠ manifest {}",
            col.sex,
            body.tris.len(),
            col.tris
        );
        assert!(
            body.positions.len() >= col.verts,
            "{} loaded {} verts, fewer than the {} the manifest declares — seam \
             splitting can only ever add vertices",
            col.sex,
            body.positions.len(),
            col.verts
        );
        let unique: HashSet<[u32; 3]> = body
            .positions
            .iter()
            .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
            .collect();
        assert_eq!(
            unique.len(),
            col.verts,
            "{} has {} distinct positions ({} loaded, seams included) ≠ manifest {}",
            col.sex,
            unique.len(),
            body.positions.len(),
            col.verts
        );
        // The bundle's verified accuracy is what justifies a millimetre contact
        // margin. If a swap makes the proxy cruder, the contact tuning in
        // `auto_drape` (1 mm gap/offset) stops being defensible — fail loudly.
        assert!(
            col.deviation_mm <= 6.0,
            "{} posed deviation {} mm exceeds the 1 mm contact margin budget",
            col.sex,
            col.deviation_mm
        );
        assert!(
            col.silhouette <= 0.005,
            "{} COL silhouette mismatch {} is worse than the shipped 0.0023",
            col.sex,
            col.silhouette
        );
    }
}

/// The whole proxy, loaded for real, lowered into a runnable scene. Building only —
/// stepping the full 15 000-triangle body costs 50.7 s/step in release (module docs).
#[test]
fn full_proxy_lowers_into_a_static_collider_scene() {
    let Some(cols) = collision_tiers() else {
        return;
    };
    for col in &cols {
        let body = load(col);

        let collider = avatar_collider(&body).expect("real proxy is a valid collider");
        assert!(collider.is_static);
        assert!(collider.pins.is_empty());
        assert_eq!(collider.positions.len(), body.positions.len());
        // The proxy is skinned (65-bone Mixamo skeleton, shared with the render
        // mesh); the collider carries the rig through untouched so a glTF
        // round-trip does not lose it and one pose can drive both.
        assert!(collider.skin.is_some(), "collision proxy lost its skin");
        // UVs parallel to positions, so `Scene::lower` keeps the garment's UVs too.
        assert_eq!(collider.uvs.len(), collider.positions.len());

        let bb = Aabb::of(&body.positions).unwrap();
        // Feet on the ground, midline at X = 0 (INTEGRATION.md §2) — the reason no
        // fix-up transform exists in `avatar.rs`.
        assert!(
            bb.min[1].abs() < 0.01,
            "{} feet not on y=0: {bb:?}",
            col.sex
        );
        assert!(
            bb.center()[0].abs() < 0.02,
            "{} midline not at x=0: {bb:?}",
            col.sex
        );

        let garment = patch(head_cap(&body.positions, CROWN), 6, 0.03);
        let scene = drape_on_avatar(
            &body,
            &garment,
            FabricPreset::Cotton,
            GarmentPlacement::DropFromAbove,
        )
        .expect("drape scene from the real proxy");

        assert!(
            scene.config.self_collision,
            "the body is only seen via self-collision"
        );
        assert_eq!(scene.config.floor_y, Some(bb.min[1]));
        assert!(scene.objects[0].is_static && !scene.objects[1].is_static);

        let solver = scene.build_implicit();
        let n_body = body.positions.len();
        assert_eq!(
            solver.state.vertex_count(),
            n_body + garment.positions.len()
        );
        // Every body vertex is pinned (infinite mass) and no garment vertex is.
        assert!((0..n_body).all(|i| solver.state.is_pinned(i)));
        assert!(
            (n_body..solver.state.vertex_count()).all(|i| !solver.state.is_pinned(i)),
            "the garment must stay dynamic"
        );
        // UVs survived the concatenation, so the anisotropic membrane is live.
        assert_eq!(solver.state.uvs.len(), solver.state.vertex_count());
    }
}

#[test]
fn cloth_dropped_on_a_real_proxy_does_not_sink_into_the_body() {
    let Some(cols) = collision_tiers() else {
        return;
    };
    for col in &cols {
        let body = load(col);
        let (scene, head, cap) = stepped_scene(&body);
        let head_bb = Aabb::of(&head.positions).unwrap();
        let n_body = head.positions.len();
        let cloth0 = scene.objects[1].positions.clone();

        // Placement precondition: the patch starts above the crown.
        assert!(
            Aabb::of(&cloth0).unwrap().min[1] > head_bb.max[1],
            "patch must start above the crown"
        );
        // The vertices the head is actually under.
        let over_head: Vec<usize> = (0..cloth0.len())
            .filter(|&i| {
                let p = cloth0[i];
                p[0] >= cap.min[0] && p[0] <= cap.max[0] && p[2] >= cap.min[2] && p[2] <= cap.max[2]
            })
            .collect();
        assert!(
            over_head.len() >= 8,
            "only {} patch vertices over the crown — the patch missed the head",
            over_head.len()
        );

        let held = run(&scene);

        // The body is kinematic: not one of its vertices may have moved.
        for (i, p) in held.iter().take(n_body).enumerate() {
            let want = head.positions[i];
            assert!(
                (0..3).all(|k| (p[k] - want[k]).abs() < 1e-6),
                "{} static body vertex {i} moved: {p:?} vs {want:?}",
                col.sex
            );
        }

        // Control: the same scene with the body invisible to the cloth. This is the
        // checked floor — it proves the invariant below is measuring contact and not
        // a patch that simply had no time to fall.
        let mut no_contact = scene.clone();
        no_contact.config.self_collision = false;
        let free = run(&no_contact);
        let lowest = |ps: &[[f32; 3]]| {
            over_head
                .iter()
                .map(|&i| ps[n_body + i][1])
                .fold(f32::INFINITY, f32::min)
        };
        let (crown, y_held, y_free) = (head_bb.max[1], lowest(&held), lowest(&free));
        assert!(
            y_free < crown - 0.010,
            "{} control run did not fall ({y_free} vs crown {crown}) — the test is \
             not discriminating; free fall over {STEPS} steps of {DT} s should reach {}",
            col.sex,
            crown - 0.5 * 9.8 * (STEPS as f32 * DT).powi(2)
        );
        assert!(
            y_held > crown - 0.006,
            "{} cloth sank into the head: {y_held} vs crown {crown} (control fell to \
             {y_free})",
            col.sex
        );
        assert!(
            y_held > y_free + 0.005,
            "{} body did not hold the cloth up: held {y_held} vs free {y_free}",
            col.sex
        );

        // Nothing tunneled through the ground plane either.
        for p in held.iter().skip(n_body) {
            assert!(
                p[1] >= scene.config.floor_y.unwrap() - 2.0e-2,
                "cloth fell through the floor: {p:?}"
            );
        }
    }
}

/// Determinism is a property of the solver path, not of which body is in the scene,
/// so this runs on **one** avatar to keep CI cost down — the ungated twin in
/// `avatar.rs` covers the same invariant on a synthetic body.
#[test]
fn real_proxy_scene_is_deterministic() {
    let Some(cols) = collision_tiers() else {
        return;
    };
    for col in cols.iter().take(1) {
        let body = load(col);
        let (scene, head, _) = stepped_scene(&body);
        let (a, b) = (run(&scene), run(&scene));
        assert!(a.len() > head.positions.len(), "degenerate scene");
        // Floor: the cloth must actually have moved, or "identical" is trivially true.
        let moved = a
            .iter()
            .skip(head.positions.len())
            .zip(scene.objects[1].positions.iter())
            .any(|(p, q)| (p[1] - q[1]).abs() > 1e-6);
        assert!(
            moved,
            "{} no cloth vertex moved — determinism would be vacuous",
            col.sex
        );
        assert_eq!(
            a, b,
            "{} same inputs must give identical positions",
            col.sex
        );
    }
}

/// The measurement behind the module docs and behind `engine_api::avatar`'s
/// "build_implicit, but not with the whole proxy" note. Ignored: it is a benchmark,
/// it uses the wall clock, and the full-proxy case takes minutes.
///
/// ```text
/// cargo test --release -p engine-api --test avatar_bundle -- --ignored --nocapture
/// ```
#[test]
#[ignore = "benchmark: minutes, wall-clock"]
fn collision_proxy_step_cost() {
    let Some(cols) = collision_tiers() else {
        return;
    };
    let time = |label: &str, scene: &Scene, steps: usize| -> f64 {
        let mut solver = scene.build_implicit();
        solver.params.contact_kappa = CONTACT_KAPPA;
        solver.params.contact_dhat = CONTACT_DHAT;
        let t = std::time::Instant::now();
        for _ in 0..steps {
            solver.step(&CgCpu::new());
        }
        let per = t.elapsed().as_secs_f64() / steps as f64;
        println!(
            "{label}: {} verts / {} tris, {steps} steps, {per:.3} s/step",
            solver.state.vertex_count(),
            scene.objects.iter().map(|o| o.tris.len()).sum::<usize>(),
        );
        per
    };
    for col in &cols {
        let body = load(col);
        let (crown, _, _) = stepped_scene(&body);
        time(&format!("{} crown, self-collision on", col.sex), &crown, 6);

        let mut full = drape_on_avatar(
            &body,
            &patch(head_cap(&body.positions, CROWN), 6, 0.03),
            FabricPreset::Cotton,
            GarmentPlacement::DropFromAbove,
        )
        .expect("full proxy scene");
        full.config.dt = DT;
        full.config.iterations = 3;
        let mut off = full.clone();
        off.config.self_collision = false;
        let off_cost = time(
            &format!("{} full proxy, self-collision off", col.sex),
            &off,
            3,
        );
        // 3 steps, not 1. This was the named upgrade path in `tools/perf-baseline.tsv` for
        // `proxy_step_s_selfcol_on`: timing exactly one step gave the row NO averaging, which
        // is why it carried a 3.67x observed spread and a tolerance of 4.0 — a tolerance that
        // cannot distinguish a real regression from a busy machine. Averaged over 3 (matching
        // the OFF row) the spread measured 1.004x over five runs, so the tolerance came down
        // to 2.0. Cost of the change: ~0.5 s.
        let on_cost = time(
            &format!("{} full proxy, self-collision on", col.sex),
            &full,
            3,
        );

        // ---- REGRESSION GUARD (D-17) ----
        //
        // The static mesh collider's whole claim is that a body mesh costs roughly what it
        // costs with self-collision switched off, because cloth-vs-body contact no longer
        // runs the body's own 15 000 triangles against each other.
        //
        // The guard is a RATIO against the self-collision-off run measured on the same
        // machine in the same process, not an absolute s/step: absolutes are hardware and
        // are the one thing guaranteed to make this flaky. It is set an order of magnitude
        // loose. Measured on Apple Silicon at the time of writing the ratio is ~3.5x
        // (male 0.235 / 0.068, female 0.245 / 0.069) against a pre-fix ratio of **143x**
        // (9.724 / 0.068), so 40x catches the regression that matters — the reintroduction
        // of body-vs-body pairs — while leaving 10x of headroom for a slow or noisy box.
        // Verified to bite: reverting the filter makes this fail at 171x.
        let ratio = on_cost / off_cost;
        assert!(
            ratio < 40.0,
            "{} full proxy costs {ratio:.1}x the self-collision-off run \
             ({on_cost:.3} vs {off_cost:.3} s/step). D-17 regressed: the body's own \
             triangles are being tested against each other again.",
            col.sex
        );
        // Floor, so the guard cannot pass by measuring nothing: the off run must have done
        // real work, and self-collision must cost SOMETHING (if `on` were free, the contact
        // pass would have been switched off rather than made cheap).
        assert!(
            off_cost > 1.0e-4 && ratio > 1.0,
            "{} degenerate measurement: on {on_cost:.4} off {off_cost:.4} s/step",
            col.sex
        );
    }
}

/// **The precondition of the signed mesh contact** (`engine_core::contact::
/// signed_point_tri_distance_grad`): a body proxy's triangles must be wound so their
/// geometric normals point OUT of the body. That winding is what tells the contact barrier
/// which side is inside, and it is the difference between a garment that recovers from a
/// 10 µm initial overlap and one that silently falls through the body. It was an
/// assumption until this test; the shipped bundle is the thing that has to satisfy it.
///
/// Two independent oracles, neither of them the solver:
///
/// 1. **Signed volume** (divergence theorem, `V = ⅙ Σ (a×b)·c`). Positive ⇒ outward
///    winding. Nothing in the loader or the crop can fake it — it is a property of the
///    authored index buffer — and comparing it to the bounding box also proves the proxy is
///    closed enough to have a volume at all. An inverted proxy gives a negative number.
/// 2. **The authored glTF NORMAL attribute**, which the exporter writes outward for
///    rendering and which the solver never reads.
///
/// Oracle 2 is asserted as a FRACTION, not as zero, because it is not zero: measured on the
/// shipped bundle, **7 of the female proxy's 15 000 triangles** disagree with their own
/// shading normal, every one of them a sliver of 1.0e-6 to 4.4e-5 m² (typical facet
/// 1.3e-4 m²) in a crease where the averaged vertex normal is not the facet's. That is
/// 0.05 % by count and 0.003 % by area — nothing a contact stencil can act on alone, since
/// a cloth vertex within `dhat` of a sub-millimetre sliver is within `dhat` of a dozen
/// correctly wound neighbours. A genuinely inverted or half-inverted export reads as tens
/// of percent here and fails.
#[test]
fn collision_proxies_are_outward_wound() {
    let Some(cols) = collision_tiers() else {
        return;
    };
    for col in &cols {
        let body = load(col);
        let p = |i: u32| glam::Vec3::from(body.positions[i as usize]);
        let geo = |t: &[u32; 3]| (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));

        let (mut vol, mut area) = (0.0f64, 0.0f64);
        for t in &body.tris {
            vol += p(t[0]).cross(p(t[1])).dot(p(t[2])) as f64 / 6.0;
            area += 0.5 * geo(t).length() as f64;
        }
        let e = Aabb::of(&body.positions).unwrap().extent();
        let box_vol = (e[0] * e[1] * e[2]) as f64;
        assert!(
            vol > 0.05 * box_vol,
            "{}: signed volume {vol:.4} m³ against a bounding box of {box_vol:.4} m³ — a body \
             fills a good fraction of its box, so this is either an INWARD winding (negative) \
             or an unclosed surface, and the signed mesh collider would push cloth INTO it",
            col.sex
        );
        assert!(area > 0.1, "{}: degenerate proxy, area {area} m²", col.sex);

        // Cross-check against the exporter's own normals, if the proxy ships them.
        if body.normals.len() != body.positions.len() {
            continue;
        }
        let bad: Vec<&[u32; 3]> = body
            .tris
            .iter()
            .filter(|t| {
                let shading: glam::Vec3 = t
                    .iter()
                    .map(|&i| glam::Vec3::from(body.normals[i as usize]))
                    .sum();
                geo(t).dot(shading) < 0.0
            })
            .collect();
        let bad_area: f64 = bad.iter().map(|t| 0.5 * geo(t).length() as f64).sum();
        assert!(
            bad.len() * 500 < body.tris.len() && bad_area < 0.001 * area,
            "{}: {} of {} triangles ({:.3} % by count, {:.4} % of {area:.3} m² by area) are \
             wound against their own authored glTF normal — beyond the sliver-in-a-crease \
             noise floor, so the proxy's winding is no longer consistently outward and the \
             signed mesh collider cannot be trusted on it",
            col.sex,
            bad.len(),
            body.tris.len(),
            100.0 * bad.len() as f64 / body.tris.len() as f64,
            100.0 * bad_area / area,
        );
    }
}
