// SPDX-License-Identifier: GPL-2.0-or-later
//! Authoring scene — the front-end data model (replaces the Python frontend).
//! Distinct from the solver-side `engine_core::SimState`; `build()` is the
//! one-way lowering, mirroring the reference's frontend→solver handoff.

use crate::material::Material;
use crate::params::SolverConfig;
use engine_core::contact::Collider;
use engine_core::implicit::{ImplicitParams, ImplicitSolver};
use engine_core::{geometry::Mesh as CoreMesh, SimState, Solver, SolverParams};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Authoring spec for an analytic collider (lowered to `engine_core::Collider`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ColliderSpec {
    Plane { point: [f32; 3], normal: [f32; 3] },
    Sphere { center: [f32; 3], radius: f32 },
}

impl ColliderSpec {
    fn to_core(self) -> Collider {
        match self {
            ColliderSpec::Plane { point, normal } => Collider::HalfSpace {
                point: Vec3::from(point),
                normal: Vec3::from(normal).normalize_or_zero(),
            },
            ColliderSpec::Sphere { center, radius } => Collider::Sphere {
                center: Vec3::from(center),
                radius,
            },
        }
    }
}

/// An authored triangle mesh with material and pinned vertices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Object {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
    #[serde(default)]
    pub material: Material,
    /// Optional named fabric preset; when set it overrides `material`.
    #[serde(default)]
    pub preset: Option<crate::FabricPreset>,
    /// Vertex indices held kinematic.
    #[serde(default)]
    pub pins: Vec<u32>,
    /// Static (kinematic) collider — infinite mass, never integrated.
    #[serde(default)]
    pub is_static: bool,
    /// Optional per-vertex normals (parallel to `positions`). Carried through
    /// glTF I/O; the cloth solver recomputes normals, so these are pass-through.
    #[serde(default)]
    pub normals: Vec<[f32; 3]>,
    /// Optional per-vertex texture coordinates (parallel to `positions`).
    #[serde(default)]
    pub uvs: Vec<[f32; 2]>,
    /// Optional skin (joints + inverse-bind + per-vertex weights) for a rigged
    /// mesh. Carried through glTF I/O; the cloth solver does not consume it yet.
    #[serde(default)]
    pub skin: Option<crate::rig::Skin>,
    /// Optional animation clips targeting this object's skin joints.
    #[serde(default)]
    pub animations: Vec<crate::rig::AnimationClip>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Scene {
    #[serde(default)]
    pub config: SolverConfig,
    pub objects: Vec<Object>,
    /// Analytic colliders the cloth must not penetrate (implicit solver only).
    #[serde(default)]
    pub colliders: Vec<ColliderSpec>,
}

impl Scene {
    /// Concatenate all objects into one vertex space and produce the solver-side
    /// [`SimState`] plus the representative dynamic [`Material`]. Per-vertex mass
    /// = areal_density * total_area / vertex_count (uniform lumped mass).
    /// The third return is the **static mesh collider mask**, parallel to the vertices:
    /// true exactly for vertices of an `is_static` object. Deliberately not the same
    /// thing as `pin_flags`, which also carries authored pins on dynamic cloth — a pinned
    /// cloth vertex is not body geometry, and signing contact against a pinned sheet
    /// pushes cloth through it (`ImplicitSolver::set_static_collider_verts`).
    fn lower(&self) -> (SimState, Material, Vec<bool>) {
        let mut positions: Vec<Vec3> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut pin_flags: Vec<bool> = Vec::new();
        let mut static_obj: Vec<bool> = Vec::new();
        let mut material = Material::default();
        // Material-space UVs, concatenated parallel to positions. Kept only if EVERY
        // object supplies them (parallel-or-empty), so downstream `SimState::uvs` is
        // either fully populated or empty — never partial.
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut have_uvs = true;

        for obj in &self.objects {
            let base = positions.len() as u32;
            for p in &obj.positions {
                positions.push(Vec3::from(*p));
                pin_flags.push(obj.is_static);
                static_obj.push(obj.is_static);
            }
            for t in &obj.tris {
                tris.push([t[0] + base, t[1] + base, t[2] + base]);
            }
            if obj.uvs.len() == obj.positions.len() {
                uvs.extend_from_slice(&obj.uvs);
            } else {
                have_uvs = false;
            }
            if !obj.is_static {
                // ponytail: single global material until per-element params land.
                // A named preset overrides the inline material.
                material = obj.preset.map(|p| p.material()).unwrap_or(obj.material);
            }
            for &pin in &obj.pins {
                let idx = (base + pin) as usize;
                if idx < pin_flags.len() {
                    pin_flags[idx] = true;
                }
            }
        }

        let mesh = CoreMesh::new(positions, tris);
        let area = total_area(&mesh);
        let n = mesh.vertex_count().max(1);
        let vertex_mass = (area * material.density / n as f32).max(1.0e-6);

        let mut state = SimState::from_mesh(&mesh, vertex_mass);
        if have_uvs && uvs.len() == state.vertex_count() {
            state.uvs = uvs;
        }
        for (i, &pinned) in pin_flags.iter().enumerate() {
            if pinned {
                state.pin(i);
            }
        }
        tracing::info!(
            objects = self.objects.len(),
            verts = state.vertex_count(),
            tris = state.tris.len(),
            springs = state.stretch.len(),
            vertex_mass,
            "scene lowered"
        );
        (state, material, static_obj)
    }

    /// Lower to a ready-to-run XPBD preview-tier [`Solver`].
    pub fn build(&self) -> Solver {
        let (state, material, _) = self.lower();
        let mut solver = Solver::new(state, self.solver_params(material.stretch_compliance()));
        solver.set_colliders(self.colliders.iter().map(|c| c.to_core()).collect());
        solver
    }

    /// Lower to a ready-to-run implicit backward-Euler [`ImplicitSolver`]. The
    /// caller supplies the linear solver (`engine_core::CgCpu` or the GPU PCG)
    /// per `step`. Spring stiffness comes from the material's Young's modulus.
    pub fn build_implicit(&self) -> ImplicitSolver {
        let (state, material, static_obj) = self.lower();
        let c = &self.config;
        let params = ImplicitParams {
            dt: c.dt,
            gravity: Vec3::from(c.gravity),
            newton_iters: c.iterations.max(1),
            stiffness: material.young_mod.max(1.0),
            bend_stiffness: material.bend,
            // Membrane if the material sets it (e.g. a fabric preset); else the
            // spring model (young_mod) carries stretch.
            membrane_warp: material.membrane_warp,
            membrane_weft: material.membrane_weft,
            membrane_shear: material.membrane_shear,
            contact_dhat: material.contact_gap.max(1.0e-3),
            contact_kappa: material.young_mod.max(1.0),
            contact_dynamic: material.contact_dynamic,
            // Coulomb friction. A static body is an `Object`, not a `ColliderSpec`, so
            // cloth-body contact runs through the SELF-COLLISION stencil and reads this
            // field via `ImplicitSolver::stencil_mu` — `collider_friction` and
            // `friction_mode` never enter that path and are deliberately left at their
            // defaults (the reference's `default:` is `Min`, and its INFINITY sentinel is
            // skipped explicitly, so both are inert here). Without this line IPC barriers
            // are purely repulsive and nothing can hold a garment on a torso.
            friction: material.friction,
            // The engine's own default is 1e-3; the scene config declares 1e-5 and was
            // never read, so the looser value silently won. Inert while `friction` is 0.
            friction_eps: c.friction_eps,
            self_collision: c.self_collision,
            // Strain limiting: `material.strain_limit` is the max tensile strain
            // fraction (0 disables); the limiter's rest ratio is `1 + that`. Stiffness
            // reuses the material's `young_mod` scale so it holds against the springs.
            strain_limit_ratio: if material.strain_limit > 0.0 {
                1.0 + material.strain_limit
            } else {
                0.0
            },
            strain_limit_k: if material.strain_limit > 0.0 {
                material.young_mod.max(1.0)
            } else {
                0.0
            },
            // Anisotropic weft ratio (0 = isotropic). Needs mesh UVs, which
            // `lower` now carries into `SimState::uvs`.
            strain_limit_ratio_weft: if material.strain_limit_weft > 0.0 {
                1.0 + material.strain_limit_weft
            } else {
                0.0
            },
            cg_tol: c.cg_tol,
            cg_max_iter: c.cg_max_iter,
            floor_y: c.floor_y.unwrap_or(f32::NEG_INFINITY),
            // Reference cubic barrier / Rayleigh κ stay opt-in: the scene API keeps
            // the engine's documented defaults (log barrier + multiplicative κ).
            ..Default::default()
        };
        let mut solver = ImplicitSolver::new(state, params);
        solver.set_colliders(self.colliders.iter().map(|c| c.to_core()).collect());
        // A static object is a body proxy: closed, outward-wound, and the one geometry
        // whose contact gap can carry a sign. Without this the mesh collider uses the
        // unsigned point-triangle distance and a garment authored even micrometres inside
        // the body falls straight through it.
        solver.set_static_collider_verts(&static_obj);
        solver
    }

    fn solver_params(&self, compliance: f32) -> SolverParams {
        let c = &self.config;
        SolverParams {
            dt: c.dt,
            substeps: c.substeps.max(1),
            iterations: c.iterations,
            gravity: Vec3::from(c.gravity),
            stretch_compliance: compliance,
            // Preview-tier bending stays OFF by default (config `bend_compliance` is
            // None) so the golden drape is unchanged; a scene opts in by setting it.
            bend_compliance: c.bend_compliance.unwrap_or(f32::INFINITY),
            // Aero drag OFF by default (air_density 0) unless `preview_aero` is set.
            air_density: if c.preview_aero { c.air_density } else { 0.0 },
            air_friction: c.air_friction,
            wind: Vec3::from(c.wind),
            // ponytail: scalar velocity damping stands in for per-triangle
            // aerodynamic drag until the anisotropic damper is wired here.
            damping: c.damping,
            floor_y: c.floor_y.unwrap_or(f32::NEG_INFINITY),
            // ponytail: NOT `material.friction`. This is the preview tier's floor
            // half-space and it is a different quantity — the FRACTION of tangential
            // slip removed per substep, `clamp(0,1)` (`solver.rs:242`), not a Coulomb
            // ratio. Feeding μ = 1.0 into it would read as "delete all floor slip",
            // which is a pun on the name rather than a wiring. Upgrade path: give the
            // XPBD `collide()` a real Coulomb cone, then map the material through.
            floor_friction: 0.0,
        }
    }
}

fn total_area(mesh: &CoreMesh) -> f32 {
    mesh.tris
        .iter()
        .map(|t| {
            let a = mesh.positions[t[0] as usize];
            let b = mesh.positions[t[1] as usize];
            let c = mesh.positions[t[2] as usize];
            0.5 * (b - a).cross(c - a).length()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_scene() -> Scene {
        Scene {
            config: SolverConfig::default(),
            objects: vec![Object {
                name: "quad".into(),
                positions: vec![
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                tris: vec![[0, 1, 2], [2, 1, 3]],
                material: Material::default(),
                preset: None,
                pins: vec![0, 1],
                is_static: false,
                normals: Vec::new(),
                uvs: Vec::new(),
                skin: None,
                animations: Vec::new(),
            }],
            colliders: vec![],
        }
    }

    #[test]
    fn build_produces_runnable_solver() {
        let mut solver = quad_scene().build();
        assert_eq!(solver.state.vertex_count(), 4);
        assert!(solver.state.is_pinned(0));
        assert!(solver.state.is_pinned(1));
        solver.step();
        assert!(solver.state.positions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn bend_compliance_config_flows_to_solver() {
        // Default: bending OFF (INFINITY) so preview goldens are unchanged.
        let off = quad_scene().build();
        assert!(off.params.bend_compliance.is_infinite());
        // Opt-in: config.bend_compliance flows through solver_params.
        let mut scene = quad_scene();
        scene.config.bend_compliance = Some(0.01);
        let on = scene.build();
        assert!((on.params.bend_compliance - 0.01).abs() < 1e-9);
    }

    #[test]
    fn scene_round_trips_json() {
        let s = quad_scene();
        let json = serde_json::to_string(&s).unwrap();
        let back: Scene = serde_json::from_str(&json).unwrap();
        assert_eq!(back.objects.len(), 1);
        assert_eq!(back.objects[0].tris, s.objects[0].tris);
    }

    #[test]
    fn material_deserializes_strain_limit_kebab() {
        // A scene author sets `strain-limit` (kebab) on a material.
        let m: Material = serde_json::from_str(r#"{"strain-limit": 0.05}"#).unwrap();
        assert!((m.strain_limit - 0.05).abs() < 1e-6);
    }

    #[test]
    fn lower_carries_object_uvs_into_state() {
        // A quad with per-vertex UVs → state.uvs populated (parallel to positions).
        let mut scene = quad_scene();
        scene.objects[0].uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let solver = scene.build_implicit();
        assert_eq!(solver.state.uvs.len(), 4, "UVs carried into SimState");
        assert_eq!(solver.state.uvs[3], [1.0, 1.0]);
        // No UVs → empty (parallel-or-empty).
        let bare = quad_scene().build_implicit();
        assert!(bare.state.uvs.is_empty());
    }

    #[test]
    fn build_implicit_wires_strain_limit() {
        // material.strain_limit (max tensile strain) → params ratio = 1 + limit.
        let mut scene = quad_scene();
        scene.objects[0].material.strain_limit = 0.1;
        let solver = scene.build_implicit();
        assert!((solver.params.strain_limit_ratio - 1.1).abs() < 1e-6);
        assert!(solver.params.strain_limit_k > 0.0);
        // Isotropic by default (no weft limit set).
        assert_eq!(solver.params.strain_limit_ratio_weft, 0.0);
        // Disabled by default (strain_limit == 0).
        let off = quad_scene().build_implicit();
        assert_eq!(off.params.strain_limit_ratio, 0.0);
        assert_eq!(off.params.strain_limit_k, 0.0);
    }

    /// The cheap guard against the `..Default::default()` swallow recurring: eleven
    /// material fields were mapped by hand and `friction` was simply not among them,
    /// so it read the engine default 0.0 and `auto_drape`'s `HIGH_FRICTION_MU` did
    /// nothing. `friction_eps` was worse — declared on `SolverConfig`, never read, so
    /// the engine's looser 1e-3 won over the config's own 1e-5.
    #[test]
    fn build_implicit_wires_friction() {
        let mut scene = quad_scene();
        scene.objects[0].material.friction = 0.42;
        scene.config.friction_eps = 2.0e-5;
        let solver = scene.build_implicit();
        assert!((solver.params.friction - 0.42).abs() < 1e-9);
        assert!((solver.params.friction_eps - 2.0e-5).abs() < 1e-12);
        // Frictionless is still the default, so no existing scene changes.
        assert_eq!(quad_scene().build_implicit().params.friction, 0.0);
    }

    #[test]
    fn build_implicit_wires_anisotropic_strain_limit() {
        // Warp 10% + weft 40% → params ratios 1.1 (warp) and 1.4 (weft).
        let mut scene = quad_scene();
        scene.objects[0].material.strain_limit = 0.1;
        scene.objects[0].material.strain_limit_weft = 0.4;
        let solver = scene.build_implicit();
        assert!((solver.params.strain_limit_ratio - 1.1).abs() < 1e-6);
        assert!((solver.params.strain_limit_ratio_weft - 1.4).abs() < 1e-6);
    }

    /// Cloth resting on a **15° incline**, run for 0.5 s; returns how far it slid
    /// down-slope, in millimetres.
    ///
    /// The slab is [`crate::avatar::tests::body_box`] — closed and outward-wound, and
    /// it asserts its own winding, unlike `box_mesh` — with every position rotated
    /// about Z so the surface tilts while gravity stays `-Y`. tan 15° = 0.268, so a
    /// Coulomb μ of 0.35 must hold the patch and μ = 0 must let it run.
    ///
    /// This exercises the path a garment on a torso actually takes: a static `Object`
    /// is not a `ColliderSpec`, so cloth-body contact goes through the self-collision
    /// stencil, whose coefficient is `ImplicitParams::friction` via
    /// `ImplicitSolver::stencil_mu` — not `collider_friction`, which only ever reaches
    /// the analytic-collider path.
    ///
    /// `young_mod` carries `contact_kappa` and `contact_gap` carries `contact_dhat`
    /// (see [`Scene::build_implicit`]); they are set to the 1e4 / 1e-2 the avatar
    /// bench uses rather than poking `solver.params`, so the test goes through the
    /// same wiring it is guarding.
    fn incline_slide_mm(mu: f32) -> f32 {
        let theta = 15.0f32.to_radians();
        let (s, c) = theta.sin_cos();
        let rot = |p: [f32; 3]| [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]];
        // Down-slope unit direction after the rotation: gravity with its normal
        // component removed, normalised. Derivation: n = (-sin, cos, 0), and
        // g - (g·n)n = -sin θ·(cos θ, sin θ, 0).
        let downhill = [-c, -s, 0.0f32];

        let mut slab = crate::avatar::tests::body_box();
        let bb = crate::auto_drape::Aabb::of(&slab.positions).expect("slab bounds");
        // A quarter-width patch (margin -0.25 shrinks the footprint) 3 mm above the
        // slab's top face, so there is 0.25 m of slab left to slide across.
        let (cloth_pos, cloth_tris, uvs) =
            crate::auto_drape::cloth_plane(bb, 5, -0.25, bb.max[1] + 0.003);
        let start: Vec<[f32; 3]> = cloth_pos.iter().map(|p| rot(*p)).collect();
        for p in &mut slab.positions {
            *p = rot(*p);
        }

        let scene = Scene {
            config: SolverConfig {
                dt: 1.0 / 60.0,
                self_collision: true,
                ..Default::default()
            },
            // Static target FIRST: `lower` takes the material from the last dynamic
            // object, so the cloth has to come second.
            objects: vec![
                slab,
                Object {
                    name: "cloth".into(),
                    positions: start.clone(),
                    tris: cloth_tris,
                    material: Material {
                        young_mod: 1.0e4,
                        contact_gap: 1.0e-2,
                        friction: mu,
                        ..Default::default()
                    },
                    preset: None,
                    pins: Vec::new(),
                    is_static: false,
                    normals: Vec::new(),
                    uvs,
                    skin: None,
                    animations: Vec::new(),
                },
            ],
            colliders: vec![],
        };

        let mut solver = scene.build_implicit();
        let base = solver.state.vertex_count() - start.len();
        for _ in 0..30 {
            solver.step(&engine_core::linsolve::CgCpu::new());
        }
        let slid = |acc: f32, i: usize| {
            let d = solver.state.positions[base + i] - Vec3::from(start[i]);
            acc + d.dot(Vec3::from(downhill))
        };
        (0..start.len()).fold(0.0, slid) / start.len() as f32 * 1000.0
    }

    /// **Friction is the only thing that can hold cloth on a body.** IPC barriers are
    /// purely repulsive, so with μ = 0 a garment on any non-horizontal surface slides
    /// off, however good the contact is. Differential, with the frictionless run as
    /// its own control — a single "it stayed put" number proves nothing, because a
    /// solve that does nothing at all also stays put. Before `build_implicit` read
    /// `material.friction` both runs measured 327.5 mm, bit-identical.
    ///
    /// The 15 mm bound is taken from a sweep, not from whatever the code happened to
    /// do (measured `--release`, this machine, 2026-08-05):
    ///
    /// ```text
    ///   μ      0.0    0.10    0.20   0.268   0.30   0.35   0.45   0.60   1.00
    ///   slid  327.5  151.8    66.1    25.1   15.8   10.1    6.8    5.3    4.5  mm
    /// ```
    ///
    /// Monotone, and it asymptotes at ~4.5 mm rather than 0: that floor is the 3 mm
    /// placement drop plus membrane relaxation, projected onto the slope, and it is
    /// not sliding. 15 mm therefore sits between the μ = tan θ point (25.1 mm, where
    /// theory says holding only just begins) and the wired 0.35 (10.1 mm), so the
    /// assertion still fails for every coefficient at or below the physical threshold.
    /// If it starts reading 20 mm, diagnose — do not raise the bound.
    #[test]
    fn friction_holds_cloth_on_an_incline_and_zero_does_not() {
        let free = incline_slide_mm(0.0);
        let held = incline_slide_mm(0.35);
        eprintln!("  15° incline, 0.5 s: μ=0 slid {free:.1} mm, μ=0.35 slid {held:.1} mm");
        assert!(
            free > 100.0,
            "frictionless cloth on a 15° slope only slid {free:.1} mm — the control \
             did not move, so this test proves nothing",
        );
        assert!(
            held < 15.0,
            "μ = 0.35 (> tan 15° = 0.268) must hold the patch, but it slid {held:.1} mm",
        );
    }

    /// Horizontal cantilever pinned along its first two columns; returns the mean
    /// height of the free far edge after settling. Lower = droops more.
    fn cantilever_far_edge_y(preset: crate::FabricPreset) -> f32 {
        let n = 7usize;
        let (dx, dz) = (0.1f32, 0.1f32);
        let mut positions = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push([i as f32 * dx, 1.0, j as f32 * dz]);
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
        let mut pins = Vec::new();
        for j in 0..n {
            pins.push((j * n) as u32);
            pins.push((j * n + 1) as u32);
        }
        let scene = Scene {
            config: SolverConfig {
                dt: 1.0 / 120.0,
                iterations: 5,
                gravity: [0.0, -9.8, 0.0],
                ..Default::default()
            },
            objects: vec![Object {
                name: "cloth".into(),
                positions,
                tris,
                material: preset.material(),
                preset: None,
                pins,
                is_static: false,
                normals: Vec::new(),
                uvs: Vec::new(),
                skin: None,
                animations: Vec::new(),
            }],
            colliders: vec![],
        };
        let mut solver = scene.build_implicit();
        for _ in 0..300 {
            solver.step(&engine_core::linsolve::CgCpu::new());
        }
        let far: Vec<f32> = (0..n)
            .map(|j| solver.state.positions[j * n + (n - 1)].y)
            .collect();
        assert!(far.iter().all(|y| y.is_finite()));
        far.iter().sum::<f32>() / n as f32
    }

    #[test]
    fn full_model_drape_over_sphere_is_stable_and_intersection_free() {
        // The whole engine together: a named fabric (cotton → anisotropic membrane
        // + bending) draping over a sphere and settling toward a floor, solved
        // implicitly with contact. Verifies integration + non-penetration.
        let n = 9;
        let s = 0.12;
        let shift = -(n as f32 - 1.0) * s / 2.0;
        let mut positions = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push([i as f32 * s + shift, 1.1, j as f32 * s + shift]);
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
        let (center, radius) = (Vec3::new(0.0, 0.5, 0.0), 0.5f32);
        let scene = Scene {
            config: SolverConfig {
                dt: 1.0 / 120.0,
                iterations: 5,
                gravity: [0.0, -9.8, 0.0],
                floor_y: Some(-1.0),
                ..Default::default()
            },
            objects: vec![Object {
                name: "cotton".into(),
                positions,
                tris,
                material: Material::default(),
                preset: Some(crate::FabricPreset::Cotton),
                pins: vec![],
                is_static: false,
                normals: Vec::new(),
                uvs: Vec::new(),
                skin: None,
                animations: Vec::new(),
            }],
            colliders: vec![
                crate::ColliderSpec::Sphere {
                    center: [center.x, center.y, center.z],
                    radius,
                },
                crate::ColliderSpec::Plane {
                    point: [0.0, -1.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                },
            ],
        };
        let mut solver = scene.build_implicit();
        // Give contact headroom to resolve gradually.
        solver.params.contact_kappa = 1.0e3;
        solver.params.contact_dhat = 0.02;
        for _ in 0..500 {
            solver.step(&engine_core::linsolve::CgCpu::new());
        }
        let ps = &solver.state.positions;
        assert!(ps.iter().all(|p| p.is_finite()), "full-model sim blew up");
        // No vertex inside the sphere, none below the floor (small barrier tol).
        for p in ps {
            assert!(
                (*p - center).length() > radius - 0.02,
                "penetrated sphere: {}",
                (*p - center).length()
            );
            assert!(p.y > -1.0 - 0.02, "penetrated floor: y={}", p.y);
        }
        // The cloth fell out of the air (started at y = 1.1) and came to rest ON the
        // sphere (crown y = 1.0), rather than sliding off it.
        //
        // **This bound is two-sided now, and that is a strengthening, not a
        // re-baseline.** It used to read `min_y < 0.9` with the comment "the cloth
        // actually fell and draped". Measured 2026-08-05 while wiring
        // `material.friction`: what it was really passing on was the sheet sliding off
        // the ball entirely and pooling on the floor — μ = 0 ends at min_y −0.9803,
        // i.e. flat on the plane at y = −1.0, with every vertex at the same height. A
        // cotton square 0.96 m across on a 1.0 m sphere is a cap, and a cap that slid
        // off is the one outcome a drape test must not accept.
        //
        // With cotton's μ = 0.35 it rests at 0.9953 (centre 1.0171, the barrier
        // standoff at dhat = 0.02); the corners overhang the silhouette by only
        // 0.179 m, which at this bending stiffness sags a few mm. Measured across μ:
        // 0 → −0.9803, then 0.10 / 0.20 / 0.35 all → 0.995, and `friction_eps`
        // 1e-3 / 1e-4 / 1e-5 move it by 0.2 mm. So the transition is the presence of
        // friction, not its magnitude, and neither bound is near an edge.
        let min_y = ps.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        assert!(
            min_y < 1.02,
            "cloth never came down onto the sphere: min_y={min_y} (started at 1.1)",
        );
        assert!(
            min_y > 0.9,
            "cloth slid off the sphere to min_y={min_y} — friction is not reaching \
             the contact (μ = 0 measures −0.98, on the floor)",
        );
    }

    #[test]
    fn silk_drapes_more_than_denim() {
        // Silk (low bend) curls far more than denim (high bend) → its far edge
        // hangs much lower. The classic drape discriminator.
        let silk = cantilever_far_edge_y(crate::FabricPreset::Silk);
        let denim = cantilever_far_edge_y(crate::FabricPreset::Denim);
        // Correct-direction + comfortably-above-noise margin. The absolute droop
        // is modest at this ~0.6 m scale and grows with cloth size / true KES
        // calibration; the tested claim is the ordering (silk drapes more).
        assert!(
            silk < denim - 0.01,
            "silk should drape lower than denim: silk={silk}, denim={denim}"
        );
    }
}
