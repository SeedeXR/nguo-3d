// SPDX-License-Identifier: GPL-2.0-or-later
//! **Animated collider** — the body moves and the cloth follows it.
//!
//! Two halves, tested separately because they fail separately:
//!
//! 1. [`pose::pose_at`] samples a clip and skins the mesh. Its oracle is the shipped
//!    bundle itself: skinning the BIND pose must reproduce the GLB's own `POSITION`
//!    accessor. This is the one check that cannot be satisfied by agreeing with our own
//!    arithmetic — the accessor came from Blender, the rig from the glTF exporter.
//! 2. `ImplicitSolver::set_kinematic_collider_positions` hands those positions to the
//!    solver between steps. Its observable is a garment that rides a moving body
//!    without entering it, measured against a RAY-CAST point-in-mesh oracle rather
//!    than against the solver's own signed gap.
//!
//! # Why the stepped test is synthetic
//!
//! An animated run is many steps, and the 10 230-vertex proxy costs ~0.26 s/step with
//! self-collision on. The shipped test drives a 92-vertex rigged body of revolution with
//! a 50-vertex skirt so the whole walk is ~36 s in a debug build, and it needs no
//! `samples/` and no bundle. The real proxy is used as a local oracle only.
//!
//! # The ceiling, and it is not hidden
//!
//! Per-step teleportation, no swept collider CCD. Body motion must stay under
//! `contact_dhat` per step or the body crosses the cloth between steps and no barrier
//! sees it. The synthetic clip below moves 2.5 mm/step against `dhat = 10 mm`. The
//! shipped `m_Catwalk_Walk_Forward_02` moves 97.4 mm/step at dt = 1/60 and does NOT
//! fit; `m_Catwalk_Idle_Twist_L` moves 1.67 mm/step and does.

use engine_api::pose;
use engine_api::rig::{AnimationChannel, AnimationClip, AnimationPath, Joint, Skin};
use engine_api::{Material, Object, Scene, SolverConfig};
use engine_core::linsolve::CgCpu;
use glam::{Mat4, Quat, Vec3};
use std::path::PathBuf;

// ---------------------------------------------------------------- bundle oracle

/// Loud skip — written to fd 2 directly, because libtest's capture swallows
/// `eprintln!` on a passing test and a silent skip is not a skip at all
/// (`avatar_bundle.rs` uses the same shape and the same env vars).
fn skip<T>(msg: String) -> Option<T> {
    assert!(
        std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none(),
        "MGUSO_REQUIRE_AVATAR_BUNDLE is set but {msg}"
    );
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {msg}");
    None
}

fn collision_glb() -> Option<PathBuf> {
    let root = match std::env::var_os("NGUO_AVATAR_BUNDLE") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../assets/avatars"),
    };
    let p = root.join("avatars/glb/nguo_male_collision.glb");
    // git-LFS without `lfs: true` leaves a ~130-byte pointer stub in place of the GLB.
    match std::fs::metadata(&p) {
        Ok(m) if m.len() > 100_000 => Some(p),
        _ => skip(format!(
            "no usable collision GLB at {} — the LBS oracle is NOT covered by this run",
            p.display()
        )),
    }
}

/// **The oracle.** `POSITION` in the file is the BIND pose, and skinning the bind pose
/// through `global(j) * inverse_bind(j)` is the identity by construction — so running
/// LBS with no channels applied must return the mesh the file already stores. Nothing
/// in `pose.rs` can satisfy this by agreeing with itself: the `POSITION` accessor was
/// produced by Blender, `inverse_bind` and the joint hierarchy by the glTF exporter,
/// and the blend by us.
///
/// **Not** "any clip at t = 0". A Mixamo clip's first key is the first frame of the
/// MOTION, not the rest pose — measured on this bundle, sampling the shipped clips at
/// t = 0 displaces a vertex by up to 1.0967 m. Treating t = 0 as the bind pose is a
/// plausible-sounding oracle that is simply false here.
#[test]
fn bind_pose_lbs_reproduces_the_shipped_rest_mesh() {
    let Some(path) = collision_glb() else { return };
    let obj = engine_io::load_gltf(&path).expect("load collision GLB");
    let skin = obj.skin.as_ref().expect("the collision proxy is rigged");
    assert!(
        obj.animations.len() >= 10,
        "expected the bundle's clip set, found {}",
        obj.animations.len()
    );
    assert!(
        skin.joints.len() >= 60 && skin.vertex_weights.len() == obj.positions.len(),
        "{} joints, {} weight sets for {} vertices",
        skin.joints.len(),
        skin.vertex_weights.len(),
        obj.positions.len()
    );

    let bind = pose::pose_at(skin, &AnimationClip::default(), 0.0, &obj.positions);
    let worst = bind
        .iter()
        .zip(&obj.positions)
        .map(|(p, r)| (*p - Vec3::from(*r)).length())
        .fold(0.0f32, f32::max);
    // Measured 4.9157e-7 m over 10 230 vertices; the bound is the next round number up,
    // not a reading re-baselined to whatever the code happened to do.
    assert!(
        worst < 1.0e-6,
        "bind-pose LBS deviates from the file's own POSITION by {worst} m"
    );

    // The control the whole file needs: every shipped clip must actually MOVE the mesh
    // at both ends of its range. Without it, `worst` above is satisfied by a `pose_at`
    // that ignores its arguments and returns `rest`.
    for clip in &obj.animations {
        let travel = |t: f32| {
            pose::pose_at(skin, clip, t, &obj.positions)
                .iter()
                .zip(&obj.positions)
                .map(|(p, r)| (*p - Vec3::from(*r)).length())
                .fold(0.0f32, f32::max)
        };
        assert!(
            travel(0.0) > 1.0e-3 && travel(clip.duration * 0.5) > 1.0e-3,
            "clip '{}' ({} s) is inert: {} m at t=0, {} m at mid",
            clip.name,
            clip.duration,
            travel(0.0),
            travel(clip.duration * 0.5)
        );
    }
}

// ------------------------------------------------------- synthetic rigged body

const BODY_H: f32 = 1.0;
const SEG: usize = 10;
const RINGS: usize = 8;

/// Torso profile: `(height fraction, radius)`, piecewise linear. A STRAIGHT tube is
/// useless here — nothing holds a skirt on a cylinder, and a waistband cut undersize to
/// grip one starts inside the body, which is an initial overlap the test then cannot
/// distinguish from a tunnelling failure (measured: the whole 16-vertex waistband ring
/// read as "inside" at the worst step). A hip flare holds the garment the way a real one
/// does, with everything starting cleanly outside.
const PROFILE: [(f32, f32); 6] = [
    (0.00, 0.11),
    (0.25, 0.15),
    (0.45, 0.18), // hip — the widest point, and what the skirt jams on
    (0.65, 0.13), // waist
    (0.85, 0.17), // chest
    (1.00, 0.15),
];

fn body_radius(y: f32) -> f32 {
    let u = (y / BODY_H).clamp(0.0, 1.0);
    let i = PROFILE
        .partition_point(|(h, _)| *h <= u)
        .clamp(1, PROFILE.len() - 1);
    let (h0, r0) = PROFILE[i - 1];
    let (h1, r1) = PROFILE[i];
    r0 + (r1 - r0) * (u - h0) / (h1 - h0)
}

/// A closed, OUTWARD-wound body of revolution from `y = 0` to `y = BODY_H`: the smallest
/// thing that is recognisably a torso for a garment to hang on. Outward winding is not
/// decoration — `set_static_collider_verts` rejects an inward-wound collider and falls
/// back to the unsigned distance, which is a silent loss of the whole signed-gap path.
fn tube_body() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (mut pos, mut tris) = (Vec::new(), Vec::new());
    for r in 0..=RINGS {
        let y = BODY_H * r as f32 / RINGS as f32;
        let rad = body_radius(y);
        for s in 0..SEG {
            let th = std::f32::consts::TAU * s as f32 / SEG as f32;
            pos.push([rad * th.cos(), y, rad * th.sin()]);
        }
    }
    for r in 0..RINGS {
        for s in 0..SEG {
            let a = (r * SEG + s) as u32;
            let b = (r * SEG + (s + 1) % SEG) as u32;
            let (c, d) = (a + SEG as u32, b + SEG as u32);
            // Outward for a +Y tube with counter-clockwise-in-xz rings.
            tris.push([a, c, b]);
            tris.push([b, c, d]);
        }
    }
    // Caps, as fans about their own centre, wound outward (bottom −Y, top +Y).
    let bot = pos.len() as u32;
    pos.push([0.0, 0.0, 0.0]);
    let top = pos.len() as u32;
    pos.push([0.0, BODY_H, 0.0]);
    for s in 0..SEG as u32 {
        let n = (s + 1) % SEG as u32;
        tris.push([bot, n, s]);
        let r0 = (RINGS * SEG) as u32;
        tris.push([top, r0 + s, r0 + n]);
    }
    (pos, tris)
}

/// Two joints: `hips` at the origin drives the whole body, `chest` at mid-height owns
/// the upper half. Weights blend over the middle ring so the rig genuinely exercises
/// multi-influence LBS rather than a rigid transform in disguise.
fn tube_skin(pos: &[[f32; 3]]) -> Skin {
    let chest_bind = Mat4::from_translation(Vec3::new(0.0, 0.5 * BODY_H, 0.0));
    let (vertex_joints, vertex_weights) = pos
        .iter()
        .map(|p| {
            // 0 at y ≤ 0.3·H, 1 at y ≥ 0.7·H, linear between.
            let t = ((p[1] / BODY_H - 0.3) / 0.4).clamp(0.0, 1.0);
            ([0u32, 1, 0, 0], [1.0 - t, t, 0.0, 0.0])
        })
        .unzip();
    Skin {
        joints: vec![
            Joint {
                name: "hips".into(),
                parent: None,
                local: Mat4::IDENTITY.to_cols_array_2d(),
            },
            Joint {
                name: "chest".into(),
                parent: Some(0),
                local: chest_bind.to_cols_array_2d(),
            },
        ],
        inverse_bind: vec![
            Mat4::IDENTITY.to_cols_array_2d(),
            chest_bind.inverse().to_cols_array_2d(),
        ],
        vertex_joints,
        vertex_weights,
    }
}

/// A walk-shaped clip: the hips travel `dist` metres in `+x` while swaying in `y`, and
/// the chest counter-rotates about `y`. Deliberately not a straight translation — a
/// rigid slide would be passed by a `pose_at` that ignored `inverse_bind` entirely.
fn walk_clip(dist: f32, dur: f32) -> AnimationClip {
    let keys = 9;
    let (mut times, mut tvals, mut rvals) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..keys {
        let u = k as f32 / (keys - 1) as f32;
        times.push(u * dur);
        // Two full stride cycles over the clip: a 15 mm bob and a 12° chest twist.
        let phase = u * 4.0 * std::f32::consts::PI;
        tvals.extend_from_slice(&[u * dist, 0.015 * phase.sin(), 0.0]);
        let q = Quat::from_rotation_y(0.21 * phase.cos());
        rvals.extend_from_slice(&[q.x, q.y, q.z, q.w]);
    }
    AnimationClip {
        name: "synthetic_walk".into(),
        duration: dur,
        channels: vec![
            AnimationChannel {
                target_joint: 0,
                path: AnimationPath::Translation,
                times: times.clone(),
                values: tvals,
            },
            AnimationChannel {
                target_joint: 1,
                path: AnimationPath::Rotation,
                times,
                values: rvals,
            },
        ],
    }
}

/// An A-line skirt: `rows` rings from `y_top` down, flaring from `r_top` to `r_bot`.
/// Authored strictly OUTSIDE the body everywhere, so a vertex reading "inside" at any
/// step is the simulation's doing and not the fixture's.
fn skirt(
    y_top: f32,
    height: f32,
    rows: usize,
    r_top: f32,
    r_bot: f32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (mut pos, mut tris) = (Vec::new(), Vec::new());
    for r in 0..=rows {
        let u = r as f32 / rows as f32;
        let y = y_top - height * u;
        let rad = r_top + (r_bot - r_top) * u;
        for s in 0..SEG {
            let th = std::f32::consts::TAU * s as f32 / SEG as f32;
            pos.push([rad * th.cos(), y, rad * th.sin()]);
        }
    }
    for r in 0..rows {
        for s in 0..SEG {
            let a = (r * SEG + s) as u32;
            let b = (r * SEG + (s + 1) % SEG) as u32;
            let (c, d) = (a + SEG as u32, b + SEG as u32);
            tris.push([a, c, b]);
            tris.push([b, c, d]);
        }
    }
    (pos, tris)
}

/// **The independent oracle**: is `p` inside the closed mesh? Parity of the crossings
/// of a `+x` ray. Deliberately NOT the solver's pseudonormal criterion — an instrument
/// that shares the code under test would flatter it, which is the failure mode this
/// project has hit before.
///
/// The ray direction is irrational-ish so it does not graze the tube's own axis-aligned
/// cap fan; a grazing hit is the only way parity goes wrong on a closed mesh.
fn inside_mesh(p: Vec3, verts: &[Vec3], tris: &[[u32; 3]]) -> bool {
    let dir = Vec3::new(0.6431, 0.1132277, 0.6979461).normalize();
    let mut hits = 0usize;
    for t in tris {
        let (a, b, c) = (
            verts[t[0] as usize],
            verts[t[1] as usize],
            verts[t[2] as usize],
        );
        // Möller-Trumbore, front and back faces both counted (parity, not culling).
        let (e1, e2) = (b - a, c - a);
        let pv = dir.cross(e2);
        let det = e1.dot(pv);
        if det.abs() < 1.0e-12 {
            continue;
        }
        let inv = 1.0 / det;
        let tv = p - a;
        let u = tv.dot(pv) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let qv = tv.cross(e1);
        let v = dir.dot(qv) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        if e2.dot(qv) * inv > 1.0e-9 {
            hits += 1;
        }
    }
    hits % 2 == 1
}

struct Run {
    /// Cloth vertices inside the body once the drape has settled and BEFORE the body
    /// moves. Non-zero here means the fixture or the drape is wrong, not the animation.
    settled_inside: usize,
    /// Cloth vertices inside the body, worst over all walk steps (ray-cast oracle).
    worst_inside: usize,
    /// Hem centroid displacement in `+x` from where it started.
    hem_travel: f32,
    /// Hem centroid `y` at the end — did the garment stay up or fall off?
    hem_y: f32,
    /// Distance the body's own hips actually moved in `+x`.
    body_travel: f32,
}

/// Drape the skirt on the tube, then play `clip` at `dt` for `steps`, re-posing the
/// collider every step when `drive` is set. `drive = false` is the control: the clip is
/// still sampled and the body still *would* move, but the solver is never told — so the
/// garment has nothing to follow.
fn walk(drive: bool, steps: usize, dt: f32, mu: f32) -> Run {
    let (bpos, btris) = tube_body();
    let skin = tube_skin(&bpos);
    // 2.5 mm/step against the 10 mm barrier reach: inside what per-step
    // teleportation can resolve, and the report says what happens above it.
    let clip = walk_clip(0.0025 * steps as f32, steps as f32 * dt);
    // Waistband at y = 0.60 with 7.5 mm of clearance over the 0.1425 m waist; it slides
    // down a few centimetres and jams on the 0.18 m hip, which is how a real skirt stays
    // up. The hem flares clear of the hip by 15 mm so only the band is ever in contact.
    let (spos, stris) = skirt(0.60, 0.30, 4, 0.150, 0.240);
    let ncloth = spos.len();

    let material = Material {
        friction: mu,
        ..Material::default()
    };
    let scene = Scene {
        config: SolverConfig {
            dt,
            // 10 mm barrier reach against 2.5 mm of body motion per step.
            self_collision: true,
            ..Default::default()
        },
        objects: vec![
            Object {
                name: "skirt".into(),
                positions: spos,
                tris: stris,
                material: Material {
                    contact_gap: 0.01,
                    ..material
                },
                ..default_object()
            },
            Object {
                name: "body".into(),
                positions: bpos.clone(),
                tris: btris.clone(),
                is_static: true,
                ..default_object()
            },
        ],
        colliders: Vec::new(),
    };
    let mut solver = scene.build_implicit();
    let lin = CgCpu::new();

    // Settle onto the body before anything moves.
    for _ in 0..40 {
        solver.step(&lin);
    }
    let mut body_now: Vec<Vec3> = bpos.iter().map(|p| Vec3::from(*p)).collect();
    let settled_inside = solver.state().positions[..ncloth]
        .iter()
        .filter(|p| inside_mesh(**p, &body_now, &btris))
        .count();

    let hem_x = |s: &engine_core::ImplicitSolver| {
        let p = &s.state().positions[..ncloth];
        let lo = p.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
        let hem: Vec<Vec3> = p.iter().filter(|q| q.y < lo + 0.02).copied().collect();
        (
            hem.iter().map(|q| q.x).sum::<f32>() / hem.len() as f32,
            hem.iter().map(|q| q.y).sum::<f32>() / hem.len() as f32,
        )
    };
    let (x0, _) = hem_x(&solver);

    let mut worst_inside = 0usize;
    for k in 0..steps {
        let t = (k + 1) as f32 * dt;
        let posed = pose::pose_at(&skin, &clip, t, &bpos);
        if drive {
            let mut target = solver.state().positions.clone();
            target[ncloth..].copy_from_slice(&posed);
            assert!(
                solver.set_kinematic_collider_positions(&target),
                "collider move rejected at step {k}"
            );
            body_now = posed;
        }
        solver.step(&lin);
        let n = solver.state().vertex_count();
        assert!(
            solver.state().positions.iter().all(|p| p.is_finite()),
            "non-finite at step {k} of {n} vertices"
        );
        let inside = solver.state().positions[..ncloth]
            .iter()
            .filter(|p| inside_mesh(**p, &body_now, &btris))
            .count();
        worst_inside = worst_inside.max(inside);
    }
    let (x1, y1) = hem_x(&solver);
    Run {
        settled_inside,
        worst_inside,
        hem_travel: x1 - x0,
        hem_y: y1,
        body_travel: body_now
            .iter()
            .zip(&bpos)
            .map(|(p, r)| p.x - r[0])
            .fold(f32::NEG_INFINITY, f32::max),
    }
}

fn default_object() -> Object {
    Object {
        name: String::new(),
        positions: Vec::new(),
        tris: Vec::new(),
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

/// **The measurable.** A garment on a body playing a walk clip stays on, its hem travels
/// with the body, and no cloth vertex is ever inside the body.
///
/// The `drive = false` run is the control, and it is what makes the numbers mean
/// something: same clip, same garment, same steps, but the solver never hears about the
/// motion, so the hem must stay put. If both runs travelled, the test would be measuring
/// gravity down a slope rather than a moving collider.
#[test]
fn a_garment_rides_a_walking_body_without_entering_it() {
    let driven = walk(true, 60, 1.0 / 60.0, 0.35);
    let frozen = walk(false, 60, 1.0 / 60.0, 0.35);

    assert_eq!(
        driven.settled_inside, 0,
        "the drape was already broken before the body moved: {} vertices inside",
        driven.settled_inside
    );
    assert_eq!(
        driven.worst_inside, 0,
        "cloth entered the moving body: {} vertices inside at the worst step",
        driven.worst_inside
    );
    assert!(
        driven.hem_travel > 0.5 * driven.body_travel,
        "the hem did not ride the body: hem {} m against body {} m",
        driven.hem_travel,
        driven.body_travel
    );
    assert!(
        frozen.hem_travel.abs() < 0.1 * driven.hem_travel,
        "the control travelled {} m with the body never told to move — this test is \
         not measuring the collider",
        frozen.hem_travel
    );
    // Still a garment on a body, not a puddle: the hem started at y = 0.28 and must not
    // have slid off the bottom of the 1 m tube.
    assert!(
        driven.hem_y > 0.15,
        "the garment fell off during the walk: hem y {}",
        driven.hem_y
    );
}
