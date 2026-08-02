// SPDX-License-Identifier: GPL-2.0-or-later
//! Scene→solver driver for the reference (gauge) scenes, including the **mixed
//! tri+tet volumetric** ones.
//!
//! This is the assembly layer that turns a captured reference scene into a configured
//! [`ImplicitSolver`]: it reads the per-element parameter arrays, splits the faces by
//! constitutive model into the Baraff-Witkin set and the isotropic-shell (ARAP/StVK)
//! set, attaches the tets, and couples all of them in one solve.
//!
//! **What it honors and what it does not** is reported explicitly in
//! [`DriverReport::unsupported`] rather than silently approximated — the whole point of
//! this layer is that a parity attempt should state its own fidelity.

use anyhow::{bail, Result};
use engine_core::contact::Collider;
use engine_core::implicit::{ImplicitParams, ImplicitSolver, Pin, Pull};
use engine_core::state::SimState;
use engine_core::tet::TetModel;
use engine_core::Mesh;
use engine_io::golden::{
    indices_with_model, load_element_params, load_golden_input, read_param_toml, RefModel,
};
use glam::Vec3;
use std::path::Path;

/// What the driver was able to reproduce from a scene, and what it could not.
#[derive(Clone, Debug, Default)]
pub struct DriverReport {
    pub n_verts: usize,
    pub n_tris: usize,
    pub n_tets: usize,
    /// Stitches declared by the scene (`stitch_ind.bin`, `u64[S][6]`). Not modeled.
    pub n_stitches: usize,
    /// Rod segments attached from `rod.bin`.
    pub n_rods: usize,
    /// Rod bend elements with NON-ZERO stiffness (interior rod vertices on a scene whose
    /// `rod-bend` is > 0). `codim`/`yarn` ship `rod-bend = 0` so they get none.
    pub n_rod_hinges: usize,
    /// Faces assigned to the Baraff-Witkin membrane model.
    pub n_baraff_witkin: usize,
    /// Faces assigned to the isotropic shell (ARAP/StVK) model.
    pub n_shell: usize,
    /// Faces carrying the reference's strain-limit barrier.
    pub n_strain_limited: usize,
    /// Hinges given a scene-derived per-hinge bending stiffness.
    pub n_hinges: usize,
    /// Of the applied pins, how many are KINEMATIC at the evaluated time (at least one of
    /// their operations has begun).
    pub n_pins_kinematic: usize,
    /// Analytic colliders attached from the scene (walls + spheres).
    pub n_colliders: usize,
    /// Of those, how many animate over the run.
    pub n_colliders_animated: usize,
    /// Vertices driven as weighted `PullPair`s (blocks with `pull > 0`).
    pub n_pulls_applied: usize,
    /// Pins applied from the scene's `FixPair` blocks.
    pub n_pins_applied: usize,
    /// Faces that are a SOLID's surface (the tet boundary): contact geometry only,
    /// carrying no in-plane material — see the classification note in `build_from_scene`.
    pub n_solid_surface: usize,
    /// Features present in the scene that this build does NOT model. Each entry is a
    /// concrete statement of a fidelity gap, so a parity number can be interpreted.
    pub unsupported: Vec<String>,
}

/// A loaded scene ready to be stepped, with its pin schedule re-evaluated every step.
///
/// The reference re-derives every pin target from the current time at each step
/// (`scene.rs:2053-2100`), so a scene with animated pins must be advanced through this
/// rather than by calling [`ImplicitSolver::step`] directly — otherwise the targets stay
/// frozen at whatever time they were last built for.
pub struct SceneRun {
    pub solver: ImplicitSolver,
    schedule: Vec<engine_io::golden::PinBlockSchedule>,
    /// The scene's analytic colliders, re-sampled each step so an ANIMATED one
    /// (`trapped`'s sphere moves and resizes over 4 keyframes) actually moves.
    colliders: Vec<engine_io::golden::SceneCollider>,
    /// The INITIAL positions every pin operation composes from. Deliberately not the
    /// current positions: each op is defined relative to where the vertex started.
    base: Vec<Vec3>,
    ghat: f32,
}

impl SceneRun {
    /// Advance one step, refreshing the pin targets FIRST, evaluated at the END-OF-STEP
    /// time `state.time + dt`.
    ///
    /// End-of-step, not start-of-step, is what the reference uses and it matters: the
    /// implicit integrator pulls a pinned vertex TOWARD the constraint during the step, so
    /// the target must be where the pin should be once the step lands. `backend.rs:664-668`
    /// spells this out, including that its own earlier wiring used `state.time` and "left
    /// every pin one step behind by the end of the step" — reproduced here before the
    /// reference settled it, so the note is load-bearing.
    pub fn step(&mut self, linear: &dyn engine_core::linsolve::LinearSolver) {
        if !self.schedule.is_empty() {
            let t = self.solver.state().time + self.solver.params.dt as f64;
            let pins = pins_at(&self.schedule, &self.base, self.ghat, t);
            // Cannot fail: `pins_at` skips out-of-range indices and `ghat` was validated
            // when this `SceneRun` was built, and neither can change between steps.
            self.solver
                .set_pins(pins)
                .expect("pin invariants are established at construction");
            // Pull targets animate on the same schedule as pins.
            let pulls = pulls_at(&self.schedule, &self.base, t);
            if !pulls.is_empty() {
                self.solver
                    .set_pulls(pulls)
                    .expect("pull invariants are established at construction");
            }
        }
        // Re-sample colliders only when at least one actually animates — a static wall must
        // not be rebuilt every step.
        if self.colliders.iter().any(|c| c.is_animated()) {
            let t = self.solver.state().time + self.solver.params.dt as f64;
            self.solver
                .set_colliders(colliders_at(&self.colliders, t as f32));
        }
        self.solver.step(linear);
    }

    /// Whether this scene has any animated pin target at all.
    pub fn has_animated_pins(&self) -> bool {
        self.schedule.iter().any(|b| {
            b.ops
                .iter()
                .any(|o| !matches!(o.kind, engine_io::golden::PinOpKind::Unsupported(_)))
        })
    }
}

/// Load a scene as a steppable [`SceneRun`] — [`build_from_scene`] plus the pin schedule
/// needed to animate it.
pub fn scene_run(scene_dir: impl AsRef<Path>) -> Result<(SceneRun, DriverReport)> {
    let scene = scene_dir.as_ref();
    let (solver, rep) = build_from_scene(scene)?;
    let ghat = read_param_toml(scene)
        .map(|t| t.constraint_ghat)
        .unwrap_or(1.0e-3);
    if ghat.is_nan() || ghat <= 0.0 {
        bail!("scene constraint_ghat must be positive, got {ghat}");
    }
    // `build_from_scene` already set the t=0 pins; re-read the schedule so the run can
    // advance them. Cheap: it is startup I/O, once.
    let schedule = engine_io::golden::read_pin_schedule(scene).unwrap_or_default();
    let colliders = engine_io::golden::read_colliders(scene).unwrap_or_default();
    let base = solver.state().positions.clone();
    Ok((
        SceneRun {
            solver,
            schedule,
            colliders,
            base,
            ghat,
        },
        rep,
    ))
}

/// Convert the scene's analytic colliders to engine colliders at `time`.
///
/// A wall becomes a `HalfSpace` through its sampled point with the declared outward normal;
/// a sphere becomes a `Sphere` at its sampled centre and radius. Static colliders (every
/// wall in the gauge, and 2 of its 3 spheres) are time-invariant; `trapped`'s sphere
/// animates both centre and radius across 4 keyframes.
pub fn colliders_at(scene: &[engine_io::golden::SceneCollider], time: f32) -> Vec<Collider> {
    scene
        .iter()
        .filter_map(|c| {
            let (p, r) = c.sample(time);
            match c.normal {
                Some(n) => {
                    let normal = Vec3::new(n[0], n[1], n[2]);
                    // A zero/denormal normal would make the half-space meaningless; skip it
                    // rather than feed a NaN direction into the barrier.
                    (normal.length() > 1.0e-6).then(|| Collider::HalfSpace {
                        point: Vec3::new(p[0], p[1], p[2]),
                        normal: normal.normalize(),
                    })
                }
                None => (r > 0.0).then(|| Collider::Sphere {
                    center: Vec3::new(p[0], p[1], p[2]),
                    radius: r,
                }),
            }
        })
        .collect()
}

/// Build the PULL set a schedule implies at `time` — the complement of [`pins_at`]: blocks
/// with `pull > 0` go here, blocks with `pull == 0` go there, and the two are disjoint
/// exactly as the reference splits them (`scene.rs:2088-2104`).
pub fn pulls_at(
    schedule: &[engine_io::golden::PinBlockSchedule],
    base: &[Vec3],
    time: f64,
) -> Vec<Pull> {
    let mut out = Vec::new();
    for b in schedule {
        if b.pull <= 0.0 {
            continue;
        }
        for (slot, &vi) in b.indices.iter().enumerate() {
            let Some(&p0) = base.get(vi as usize) else {
                continue;
            };
            let (t, _kinematic) = b.target_at(slot, [p0.x, p0.y, p0.z], time);
            out.push(Pull {
                index: vi,
                target: Vec3::new(t[0], t[1], t[2]),
                weight: b.pull,
            });
        }
    }
    out
}

/// Build the pin set a schedule implies at `time`, starting each pin from its vertex's
/// INITIAL position in `base`.
///
/// Call this again with a later `time` (and the same `base`) to animate the targets over a
/// run — the reference re-evaluates them every step, and `base` must stay the initial
/// positions rather than the current ones, since each operation composes from the template.
pub fn pins_at(
    schedule: &[engine_io::golden::PinBlockSchedule],
    base: &[Vec3],
    ghat: f32,
    time: f64,
) -> Vec<Pin> {
    let mut pins = Vec::new();
    for b in schedule {
        if b.pull > 0.0 {
            continue;
        }
        for (slot, &vi) in b.indices.iter().enumerate() {
            let Some(&p0) = base.get(vi as usize) else {
                continue;
            };
            let (t, kinematic) = b.target_at(slot, [p0.x, p0.y, p0.z], time);
            pins.push(Pin {
                index: vi,
                target: Vec3::new(t[0], t[1], t[2]),
                ghat,
                kinematic,
                stiffness: b.stiffness,
            });
        }
    }
    pins
}

/// Mean of a parameter array, or `fallback` when the scene ships none.
fn mean_or(v: &[f32], fallback: f32) -> f32 {
    if v.is_empty() {
        fallback
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// Does an array vary meaningfully across elements?
fn varies(v: &[f32]) -> bool {
    if v.len() < 2 {
        return false;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo > 1e-6 * hi.abs().max(1.0)
}

/// Poisson's ratio for element `i`, falling back to the array's mean and then to the
/// reference's own assertion range. Every gauge scene ships a complete `*-poiss-rat.bin`
/// (0.25, 0.35 or 0.49), so the fallbacks exist only so a malformed scene degrades instead of
/// silently getting a wrong material.
fn poisson_for(arr: &[f32], i: usize) -> f32 {
    match arr.get(i) {
        Some(&v) if v.is_finite() && v > 0.0 && v < 0.5 => v,
        _ => poisson_representative(arr),
    }
}

/// The scene-wide Poisson ratio: the mean of the VALID entries, or the value 26 of the 30
/// gauge param sets use. Used for the scene-wide Lamé fallbacks, where there is no element
/// index to look up — previously expressed by passing `usize::MAX` to `poisson_for`, which
/// worked only because the index was guaranteed out of range.
fn poisson_representative(arr: &[f32]) -> f32 {
    let good: Vec<f32> = arr
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0 && *v < 0.5)
        .collect();
    if good.is_empty() {
        0.35
    } else {
        good.iter().sum::<f32>() / good.len() as f32
    }
}

/// Investigation-only ablation knobs for A-4. Read from the environment so an ablation is
/// reproducible from the shell without editing code, and INERT unless set — no default
/// behaviour changes.
///
/// * `MGUSO_ABLATE_NO_STRAIN_LIMIT=1` — attach no strain-limit barrier.
/// * `MGUSO_ABLATE_YOUNG_SCALE=<f>` — multiply every per-element elastic stiffness by `<f>`.
fn ablate_no_strain_limit() -> bool {
    std::env::var("MGUSO_ABLATE_NO_STRAIN_LIMIT").is_ok()
}

fn ablate_young_scale() -> f32 {
    std::env::var("MGUSO_ABLATE_YOUNG_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or(1.0)
}

/// Load a reference scene and build a solver that couples its shells and solids.
///
/// Returns the solver plus a [`DriverReport`] enumerating the fidelity gaps. Per-element
/// stiffness/plasticity VARIATION is one such gap: our `ImplicitParams` carries
/// scene-wide scalars, so where a scene's array varies the driver uses its mean and says
/// so. Uniform arrays (which is how the gauge ships tet plasticity, for instance) are
/// reproduced exactly.
pub fn build_from_scene(scene_dir: impl AsRef<Path>) -> Result<(ImplicitSolver, DriverReport)> {
    let scene = scene_dir.as_ref();
    let input = load_golden_input(scene)?;
    if input.positions.is_empty() {
        bail!("scene has no vertices");
    }
    let tri_p = load_element_params(scene, "tri", false)?;
    let bend_p = load_element_params(scene, "tri", true)?;
    let tet_p = load_element_params(scene, "tet", false)?;
    // Rods ship only the BEND flavour of the plasticity files (`rod-bend-plasticity*`);
    // there is no `rod-plasticity.bin` in any gauge scene.
    let rod_p = load_element_params(scene, "rod", true)?;
    let toml = read_param_toml(scene).ok();

    let mut rep = DriverReport {
        n_verts: input.positions.len(),
        n_tris: input.tris.len(),
        n_tets: input.tets.len(),
        ..Default::default()
    };

    // ---- SOLID-SURFACE faces vs true shells ----
    // The reference dispatches face elastic energy, plasticity, and strain limiting over
    // `shell_face_count` ONLY (`energy.cu:655`, `plasticity.cu:30`,
    // `strainlimiting.cu:57`). Faces at or beyond that index are a solid's surface: they
    // exist for contact and for wind/air area, but carry NO in-plane material — the
    // volumetric response comes from the tets underneath.
    //
    // The capture does not ship `shell_face_count`, but those faces are recoverable
    // exactly: they are the boundary of the tet mesh. Verified on the gauge's `plastic`
    // scene, where the 12670 tets have 5120 boundary faces and all 5120 appear in
    // `tri.bin` carrying the ARAP model, ordered after the 3844 Baraff-Witkin cloth
    // faces. Driving them with a shell model would stack a spurious membrane on top of
    // the tets, stiffening the solid.
    let solid_face: Vec<bool> = {
        let mut count: std::collections::HashMap<(u32, u32, u32), u32> =
            std::collections::HashMap::new();
        for t in &input.tets {
            for f in engine_core::tet::tet_faces(t) {
                let mut k = f;
                k.sort_unstable();
                *count.entry((k[0], k[1], k[2])).or_insert(0) += 1;
            }
        }
        input
            .tris
            .iter()
            .map(|f| {
                let mut k = *f;
                k.sort_unstable();
                // A face shared by two tets is interior; one occurrence means boundary.
                count.get(&(k[0], k[1], k[2])) == Some(&1)
            })
            .collect()
    };
    rep.n_solid_surface = solid_face.iter().filter(|&&b| b).count();

    // Split faces by constitutive model.
    let (mut bw_idx, mut shell_idx) = if tri_p.model.len() == input.tris.len() {
        (indices_with_model(&tri_p.model, RefModel::BaraffWitkin), {
            let mut s = indices_with_model(&tri_p.model, RefModel::Arap);
            s.extend(indices_with_model(&tri_p.model, RefModel::StVk));
            s.sort_unstable();
            s
        })
    } else {
        // No per-face model array: treat every face as Baraff-Witkin (our default).
        rep.unsupported
            .push("no per-face model array; all faces treated as Baraff-Witkin".into());
        ((0..input.tris.len() as u32).collect(), Vec::new())
    };
    // Face models we do not implement — counted over ELASTIC faces only. A solid-surface
    // face carries no in-plane material in the reference either (its `shell_face_count`
    // dispatch excludes it), so its constitutive model is irrelevant and reporting it
    // overstates the gap. `friction` is exactly this case: all 4296 of its SNHk faces are
    // tet boundary, so the model never acts on them.
    for (name, m) in [("SNHk", RefModel::SnHk), ("Pdrd", RefModel::Pdrd)] {
        let n = indices_with_model(&tri_p.model, m)
            .into_iter()
            .filter(|&i| !solid_face[i as usize])
            .count();
        if n > 0 {
            rep.unsupported.push(format!(
                "{n} ELASTIC faces use the {name} model (not implemented)"
            ));
        }
    }
    // TET model comes FROM THE SCENE (`tet-model.bin`). ARAP/StVK/SNHk are implemented; a
    // scene assigning anything else would be silently given the WRONG material, which is
    // worse than an omission, so it is reported. The whole batch shares one model — the
    // gauge never mixes tet models within a scene (checked: every scene's `tet-model.bin`
    // is uniform), and a mixed scene would need per-element dispatch.
    let tet_model = {
        let first = tet_p.model.first().copied();
        let uniform = tet_p.model.windows(2).all(|w| w[0] == w[1]);
        if !uniform {
            rep.unsupported.push(
                "per-tet model VARIES within the scene; the driver applies one model to the \
                 whole batch, so the first tet's model is used"
                    .into(),
            );
        }
        match first {
            Some(RefModel::Arap) | None => TetModel::Arap,
            Some(RefModel::StVk) => TetModel::StVK,
            Some(RefModel::SnHk) => TetModel::SnHk,
            Some(other) => {
                let n = tet_p.model.len();
                rep.unsupported.push(format!(
                    "{n} tets use the {other:?} model (not implemented); they are driven as \
                     ARAP instead, which is a substituted material rather than a missing one"
                ));
                TetModel::Arap
            }
        }
    };
    // A solid's surface face gets no in-plane material from either path.
    bw_idx.retain(|&i| !solid_face[i as usize]);
    shell_idx.retain(|&i| !solid_face[i as usize]);
    rep.n_baraff_witkin = bw_idx.len();
    rep.n_shell = shell_idx.len();

    let positions: Vec<Vec3> = input
        .positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    // Mass lumping needs the full triangle set.
    let mut state = SimState::from_mesh(&Mesh::new(positions, input.tris.clone()), 1.0);

    // ---- per-vertex mass from the scene's densities ----
    // The reference forms element mass as `density * area` for a face and
    // `density * volume` for a tet (`scene.rs:1665,1759`), then lumps a face's mass 1/3
    // to each of its vertices and a tet's 1/4 (`builder.rs:492,529`). Solid-surface
    // faces are EXCLUDED from the face sum unless `include_face_mass` is set
    // (`builder.rs:485`), which defaults to false — their mass comes from the tets.
    //
    // This replaces a uniform per-vertex mass of 1.0, which on `plastic` is wrong by up
    // to two orders of magnitude (its face density spans 1..100).
    let include_face_mass = toml.as_ref().is_some_and(|t| t.include_face_mass);
    let have_tri_rho = tri_p.density.len() == input.tris.len();
    let have_tet_rho = tet_p.density.len() == input.tets.len();
    let have_rod_rho = rod_p.density.len() == input.rods.len();
    // Kept after the block so rod BENDING can use the reference's `stiff_k = bend * mass`
    // (`energy.cu:511`), which needs the true lumped mass rather than `1/inv_mass` (pins
    // zero that out). Empty when the scene ships no densities.
    let mut lumped_mass: Vec<f32> = Vec::new();
    if have_tri_rho || have_tet_rho || (have_rod_rho && !input.rods.is_empty()) {
        let mut mass = vec![0.0f32; state.positions.len()];
        if have_tri_rho {
            for (i, f) in input.tris.iter().enumerate() {
                if solid_face[i] && !include_face_mass {
                    continue;
                }
                let (a, b, c) = (
                    state.positions[f[0] as usize],
                    state.positions[f[1] as usize],
                    state.positions[f[2] as usize],
                );
                let area = 0.5 * (b - a).cross(c - a).length();
                let m = tri_p.density[i] * area / 3.0;
                for &v in f {
                    mass[v as usize] += m;
                }
            }
        } else if !input.tris.is_empty() {
            rep.unsupported
                .push("no per-face density array; face mass omitted".into());
        }
        if have_tet_rho {
            for (i, t) in input.tets.iter().enumerate() {
                let (p0, p1, p2, p3) = (
                    state.positions[t[0] as usize],
                    state.positions[t[1] as usize],
                    state.positions[t[2] as usize],
                    state.positions[t[3] as usize],
                );
                let vol = (p1 - p0).cross(p2 - p0).dot(p3 - p0).abs() / 6.0;
                let m = tet_p.density[i] * vol / 4.0;
                for &v in t {
                    mass[v as usize] += m;
                }
            }
        } else if !input.tets.is_empty() {
            rep.unsupported
                .push("no per-tet density array; tet mass omitted".into());
        }
        // A rod's mass is `density * GEOMETRIC length` — the reference computes it before
        // scaling the rest length by `length_factor` (`scene.rs:1498-1499`) — lumped half
        // to each endpoint (`builder.rs:511`, gated on `ei < rod_count`).
        if have_rod_rho {
            for (i, r) in input.rods.iter().enumerate() {
                let l = (state.positions[r[1] as usize] - state.positions[r[0] as usize]).length();
                let m = rod_p.density[i] * l / 2.0;
                mass[r[0] as usize] += m;
                mass[r[1] as usize] += m;
            }
        } else if !input.rods.is_empty() {
            rep.unsupported
                .push("no per-rod density array; rod mass omitted".into());
        }
        // A vertex that accumulated no mass would get inv_mass = 0, which THIS solver
        // reads as an immovable pin — silently freezing geometry the reference moves.
        // Give it the mean positive mass instead, and say so.
        let pos: Vec<f32> = mass.iter().copied().filter(|&m| m > 0.0).collect();
        let fallback = if pos.is_empty() {
            1.0
        } else {
            pos.iter().sum::<f32>() / pos.len() as f32
        };
        let n_zero = mass.iter().filter(|&&m| m <= 0.0).count();
        if n_zero > 0 {
            rep.unsupported.push(format!(
                "{n_zero} vertices carry no element mass; each was given the mean mass                  {fallback:.4e} so it stays dynamic rather than acting as a pin"
            ));
        }
        for (i, &m) in mass.iter().enumerate() {
            let m = if m > 0.0 { m } else { fallback };
            state.inv_mass[i] = 1.0 / m;
        }
        lumped_mass = mass;
    } else {
        rep.unsupported
            .push("scene ships no density arrays; a uniform vertex mass of 1.0 is used".into());
    }
    for (i, v) in input.velocities.iter().enumerate() {
        if i < state.velocities.len() {
            state.velocities[i] = Vec3::new(v[0], v[1], v[2]);
        }
    }

    // Stiffnesses: Young's modulus drives both paths. PER-ELEMENT arrays are passed
    // through when the scene ships them, so a varying material is reproduced rather
    // than averaged.
    let tri_young = mean_or(&tri_p.young_mod, 1.0e3);
    // Scene-wide Lamé fallbacks, from the reference's own conversion (A-4). Used only when a
    // scene ships no complete per-element array; the per-element paths below override them.
    let (tri_mu, tri_lambda) = engine_core::material::lame_from_young_poisson(
        tri_young,
        poisson_representative(&tri_p.poiss_rat),
    );
    let (tet_mu_s, tet_lambda_s) = engine_core::material::lame_from_young_poisson(
        mean_or(&tet_p.young_mod, 1.0e3),
        poisson_representative(&tet_p.poiss_rat),
    );
    let per_face_young = tri_p.young_mod.len() == input.tris.len();
    let per_tet_young = tet_p.young_mod.len() == input.tets.len();
    if varies(&tri_p.young_mod) && !per_face_young {
        rep.unsupported
            .push("per-face young-mod varies but its array length does not match the face count; using the mean".into());
    }
    if varies(&tet_p.young_mod) && !per_tet_young {
        rep.unsupported
            .push("per-tet young-mod varies but its array length does not match the tet count; using the mean".into());
    }

    let tet_plast = mean_or(&tet_p.plasticity, 0.0);
    let tet_plast_thr = mean_or(&tet_p.plasticity_threshold, 0.0);
    let tri_plast = mean_or(&tri_p.plasticity, 0.0);
    let tri_plast_thr = mean_or(&tri_p.plasticity_threshold, 0.0);
    let bend_plast = mean_or(&bend_p.plasticity, 0.0);
    let bend_plast_thr = mean_or(&bend_p.plasticity_threshold, 0.0);
    // Per-element plasticity is passed through as arrays below when the scene's arrays
    // align with the element counts; only a length mismatch forces the mean.
    let per_tet_plast = tet_p.plasticity.len() == input.tets.len()
        && tet_p.plasticity_threshold.len() == input.tets.len();
    let per_tri_plast = tri_p.plasticity.len() == input.tris.len()
        && tri_p.plasticity_threshold.len() == input.tris.len();
    let per_bend_plast = bend_p.plasticity.len() == input.tris.len()
        && bend_p.plasticity_threshold.len() == input.tris.len();
    for (label, ok, arr) in [
        ("tet plasticity", per_tet_plast, &tet_p.plasticity),
        ("face plasticity", per_tri_plast, &tri_p.plasticity),
    ] {
        if varies(arr) && !ok {
            rep.unsupported.push(format!(
                "per-element {label} varies but its array length does not match the element \
                 count; using the mean"
            ));
        }
    }

    // Strain limiting: the reference ships a per-face allowed-stretch FRACTION, and our
    // `strain_limit_ratio` is the ceiling `1 + fraction` (confirmed against the gauge's
    // own INVARIANTS line, e.g. belt's "limit 0.05 (ceiling 1.0500)"). A uniform array is
    // reproduced exactly; a varying one is reported and the mean used, since our strain
    // limit is an edge-wise scalar and per-edge values would need a face→edge rule.
    let sl_vals = &tri_p.strain_limit;
    let sl_active = sl_vals.iter().any(|&v| v > 0.0) && !ablate_no_strain_limit();
    let sl_mean = mean_or(sl_vals, 0.0);
    // The gap here is the MODEL, not just the averaging. The reference limits strain with a
    // per-FACE barrier on the principal-stretch excess of the deformation gradient
    // (`strainlimiting.cu:46-92`): `y = limit − (σᵢ−1)` fed through the same cubic/log
    // barrier as contact, with a Rayleigh dynamic stiffness — so the limit is ENFORCED
    // (energy diverges at it). Ours is a one-sided quadratic spring on EDGE lengths with a
    // fixed stiffness, which only resists. It also ignores the reference's shrink coupling
    // `limit = (strainlimit + 1)/min(shrink_x, shrink_y) − 1`.
    //
    // The faithful element is implemented and verified against finite differences
    // (`shell_model::strain_limit_{derivs,energy,gradient,hessian}`) but is NOT yet wired
    // into the assembly, which needs a face-element list covering every elastic face
    // regardless of the membrane/shell model split.
    // Strain limiting now matches the reference in both model and stiffness (barrier on the
    // principal-stretch excess, `elastic-Rayleigh + mass/d²`, with the step-limiter
    // bisection that makes the limit enforced), so it reports no gap of its own.
    let (sl_ratio, sl_k) = if sl_active {
        (1.0 + sl_mean, tri_young.max(1.0))
    } else {
        (0.0, 0.0)
    };

    // Contact activation distance: the scene ships it per face. Uniform ⇒ exact.
    let gap_mean = mean_or(&tri_p.contact_gap, 1.0e-2);
    if varies(&tri_p.contact_gap) {
        rep.unsupported.push(format!(
            "per-face contact-gap varies; our d̂ is scene-wide so the mean {gap_mean:.3e} is used"
        ));
    }

    // Bending stiffness comes from the scene per FACE and is mapped to hinges below with
    // the reference's own area-weighted rule, so variation is reproduced rather than
    // averaged. This scalar is only the fallback for a scene that ships no array.
    let bend_k = mean_or(&bend_p.bend, 1.0);
    let per_face_bend = bend_p.bend.len() == input.tris.len();

    let dt = toml.as_ref().map(|t| t.dt).unwrap_or(0.01);
    let params = ImplicitParams {
        dt,
        // Springs off: the shells carry in-plane response via membrane/shell models.
        stiffness: 0.0,
        // Gravity from the SCENE, not a constant. Every gauge scene happens to ship
        // [0, −9.8, 0], so a hardcoded value looked correct while being a latent bug.
        gravity: toml
            .as_ref()
            .map(|t| Vec3::new(t.gravity[0], t.gravity[1], t.gravity[2]))
            .unwrap_or(Vec3::new(0.0, -9.8, 0.0)),
        // Scene-wide fallbacks; overridden per element below when the scene ships arrays.
        membrane_warp: if bw_idx.is_empty() { 0.0 } else { tri_mu },
        membrane_weft: if bw_idx.is_empty() { 0.0 } else { tri_mu },
        membrane_shear: if bw_idx.is_empty() { 0.0 } else { tri_lambda },
        bend_stiffness: bend_k,
        contact_dhat: gap_mean,
        constraint_tol: toml.as_ref().map(|t| t.constraint_tol).unwrap_or(1.0e-2),
        // Contact: the reference uses the CUBIC barrier (finite through zero gap) and its
        // elasticity-Rayleigh dynamic stiffness, and every gauge scene leaves contact on.
        barrier_kind: if toml.as_ref().is_some_and(|t| t.cubic_barrier) {
            engine_core::contact::BarrierKind::Cubic
        } else {
            engine_core::contact::BarrierKind::Log
        },
        contact_rayleigh: true,
        // How friction combines at a contact, FROM THE SCENE. 14 of the 22 gauge scenes ship
        // "max" and only 8 ship "min"; hardwiring `min` under-frictioned the majority.
        friction_mode: match toml.as_ref().map(|t| t.friction_mode) {
            Some(engine_io::golden::GoldenFrictionMode::Max) => {
                engine_core::contact::FrictionMode::Max
            }
            Some(engine_io::golden::GoldenFrictionMode::Mean) => {
                engine_core::contact::FrictionMode::Mean
            }
            _ => engine_core::contact::FrictionMode::Min,
        },
        eiganalysis_eps: toml.as_ref().map(|t| t.eiganalysis_eps).unwrap_or(1.0e-6),
        // Linear-solve controls come from the scene too.
        cg_tol: toml.as_ref().map(|t| t.cg_tol).unwrap_or(1.0e-4),
        cg_max_iter: toml.as_ref().map(|t| t.cg_max_iter).unwrap_or(2000),
        // Newton termination and the CCD cap, from the scene. `newton_iters` becomes the
        // upper bound rather than a fixed count once `target_toi` is set.
        target_toi: toml.as_ref().map(|t| t.target_toi).unwrap_or(0.0),
        min_newton_steps: toml.as_ref().map(|t| t.min_newton_steps).unwrap_or(0),
        ccd_max_iter: toml.as_ref().map(|t| t.ccd_max_iter).unwrap_or(256),
        newton_iters: 32,
        max_dx: toml.as_ref().map(|t| t.max_dx).unwrap_or(f32::INFINITY),
        friction_eps: toml.as_ref().map(|t| t.friction_eps).unwrap_or(1.0e-5),
        self_collision: !toml.as_ref().is_some_and(|t| t.disable_contact),
        // Aerodynamic damper, straight from the scene.
        air_density: toml.as_ref().map(|t| t.air_density).unwrap_or(0.0),
        air_friction: toml.as_ref().map(|t| t.air_friction).unwrap_or(0.0),
        isotropic_air_friction: toml
            .as_ref()
            .map(|t| t.isotropic_air_friction)
            .unwrap_or(0.0),
        wind: toml
            .as_ref()
            .map(|t| Vec3::new(t.wind[0], t.wind[1], t.wind[2]))
            .unwrap_or(Vec3::ZERO),
        strain_limit_ratio: sl_ratio,
        strain_limit_k: sl_k,
        shell_mu: if shell_idx.is_empty() { 0.0 } else { tri_mu },
        shell_lambda: if shell_idx.is_empty() {
            0.0
        } else {
            tri_lambda
        },
        shell_model: TetModel::Arap,
        tet_model,
        tet_mu: if input.tets.is_empty() { 0.0 } else { tet_mu_s },
        tet_lambda: if input.tets.is_empty() {
            0.0
        } else {
            tet_lambda_s
        },
        tet_plasticity: tet_plast,
        tet_plasticity_threshold: tet_plast_thr,
        shell_plasticity: tri_plast,
        shell_plasticity_threshold: tri_plast_thr,
        bend_plasticity: bend_plast,
        bend_plasticity_threshold: bend_plast_thr,
        ..Default::default()
    };

    let mut solver = ImplicitSolver::new(state, params);

    // TRUE per-face partition: restrict the membrane to the Baraff-Witkin faces so the
    // isotropic-shell faces are not driven by both models.
    let bw_faces: Vec<[u32; 3]> = bw_idx.iter().map(|&i| input.tris[i as usize]).collect();
    solver.set_membrane_faces(&bw_faces);
    if per_face_young && solver.membrane_len() == bw_faces.len() {
        let k: Vec<[f32; 3]> = bw_idx
            .iter()
            .map(|&i| {
                // A-4: the reference's own conversion (`builder.rs:1273` `convert_prop`),
                // not the inferred `(y, y, 0.5y)`. `mu` scales BOTH stretch directions and
                // `lambda` the shear, exactly as `BaraffWitkin::stretch_*`/`shear_*` consume
                // them (`energy.cu:317-320`).
                let (mu, lambda) = engine_core::material::lame_from_young_poisson(
                    tri_p.young_mod[i as usize] * ablate_young_scale(),
                    poisson_for(&tri_p.poiss_rat, i as usize),
                );
                [mu, mu, lambda]
            })
            .collect();
        solver
            .set_membrane_stiffness(k)
            .map_err(|e| anyhow::anyhow!(e))?;
    } else if per_face_young {
        rep.unsupported.push(
            "membrane element count differs from the Baraff-Witkin face count; per-face \
             stiffness not applied"
                .into(),
        );
    }

    if !input.tets.is_empty() {
        solver.set_tets(&input.tets);
        if per_tet_young && solver.tets().len() == input.tets.len() {
            let sc = ablate_young_scale();
            let k: Vec<[f32; 2]> = tet_p
                .young_mod
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let (mu, lambda) = engine_core::material::lame_from_young_poisson(
                        y * sc,
                        poisson_for(&tet_p.poiss_rat, i),
                    );
                    [mu, lambda]
                })
                .collect();
            solver
                .set_tet_stiffness(k)
                .map_err(|e| anyhow::anyhow!(e))?;
        } else if per_tet_young {
            rep.unsupported.push(
                "some tets were dropped as degenerate, so the per-tet stiffness array no longer \
                 aligns; using the scene-wide mean"
                    .into(),
            );
        }
    }
    if !shell_idx.is_empty() {
        let faces: Vec<[u32; 3]> = shell_idx.iter().map(|&i| input.tris[i as usize]).collect();
        solver.set_shell_faces(&faces);
        if per_face_young && solver.shells().len() == faces.len() {
            let k: Vec<[f32; 2]> = shell_idx
                .iter()
                .map(|&i| {
                    let (mu, lambda) = engine_core::material::lame_from_young_poisson(
                        tri_p.young_mod[i as usize] * ablate_young_scale(),
                        poisson_for(&tri_p.poiss_rat, i as usize),
                    );
                    [mu, lambda]
                })
                .collect();
            solver
                .set_shell_stiffness(k)
                .map_err(|e| anyhow::anyhow!(e))?;
        } else if per_face_young {
            rep.unsupported.push(
                "some shell faces were dropped as degenerate, so the per-face stiffness array no \
                 longer aligns; using the scene-wide mean"
                    .into(),
            );
        }
    }

    // ---- per-vertex friction, averaged from the scene's per-FACE array ----
    if tri_p.friction.len() == input.tris.len() {
        let mu = solver.face_scalars_to_vertices(&input.tris, &tri_p.friction, 0.0);
        solver
            .set_vertex_friction(mu)
            .map_err(|e| anyhow::anyhow!(e))?;
    } else if !tri_p.friction.is_empty() {
        rep.unsupported.push(
            "per-face friction array length does not match the face count; friction disabled"
                .into(),
        );
    }

    // ---- per-element plasticity ----
    if per_tet_plast && solver.tets().len() == input.tets.len() {
        let k: Vec<[f32; 2]> = tet_p
            .plasticity
            .iter()
            .zip(&tet_p.plasticity_threshold)
            .map(|(&r, &t)| [r, t])
            .collect();
        solver
            .set_tet_plasticity(k)
            .map_err(|e| anyhow::anyhow!(e))?;
    }
    if per_tri_plast && solver.shells().len() == shell_idx.len() {
        let k: Vec<[f32; 2]> = shell_idx
            .iter()
            .map(|&i| {
                [
                    tri_p.plasticity[i as usize],
                    tri_p.plasticity_threshold[i as usize],
                ]
            })
            .collect();
        solver
            .set_shell_plasticity(k)
            .map_err(|e| anyhow::anyhow!(e))?;
    }
    // Hinges are built from the FULL triangle set, so a per-hinge array can only be
    // derived when every face shares the same bend plasticity; otherwise a hinge would
    // need a rule for combining its two incident faces' values, which the reference
    // resolves via its own hinge param indices (not exposed in the captured arrays).
    if per_bend_plast && !varies(&bend_p.plasticity) && !varies(&bend_p.plasticity_threshold) {
        let n = solver.hinge_count();
        let (r, t) = (
            bend_p.plasticity.first().copied().unwrap_or(0.0),
            bend_p.plasticity_threshold.first().copied().unwrap_or(0.0),
        );
        solver
            .set_hinge_plasticity(vec![[r, t]; n])
            .map_err(|e| anyhow::anyhow!(e))?;
    } else if per_bend_plast && varies(&bend_p.plasticity) {
        rep.unsupported.push(
            "per-face bend plasticity varies; hinges span two faces and the capture does not \
             expose the reference's hinge param mapping, so the scene-wide mean is used"
                .into(),
        );
    }
    // Report an unmodeled feature ONLY if the scene actually activates it. Listing a
    // feature the scene leaves at its identity value overstates the fidelity gap — e.g.
    // `plastic` ships shrink = 1.0 and pressure/damping/offset = 0, so none of them
    // affect it and none belong in this list.
    for (name, arr, identity) in [
        ("shrink-x", &tri_p.shrink_x, 1.0f32),
        ("shrink-y", &tri_p.shrink_y, 1.0),
        ("pressure", &tri_p.pressure, 0.0),
        ("deformation-damping", &tri_p.deformation_damping, 0.0),
        ("contact-offset", &tri_p.contact_offset, 0.0),
    ] {
        if arr.iter().any(|v| (v - identity).abs() > 1e-6) {
            rep.unsupported
                .push(format!("{name} is ACTIVE in this scene but is not modeled"));
        }
    }

    // ---- strain-limit BARRIER from the scene's per-face limits ----
    // The reference applies strain limiting to every shell face whatever its constitutive
    // model, so this covers all ELASTIC faces (solid-surface faces are excluded, matching
    // its `shell_face_count` dispatch). Per-face limits are used directly — no averaging —
    // and a face whose limit is 0 is simply not constrained.
    if sl_active && have_tri_rho {
        let elastic: Vec<[u32; 3]> = (0..input.tris.len())
            .filter(|&i| !solid_face[i])
            .map(|i| input.tris[i])
            .collect();
        let limits: Vec<f32> = (0..input.tris.len())
            .filter(|&i| !solid_face[i])
            .map(|i| sl_vals.get(i).copied().unwrap_or(0.0))
            .collect();
        // Face mass is the reference's `density * rest_area` (`scene.rs:1665`).
        let masses: Vec<f32> = (0..input.tris.len())
            .filter(|&i| !solid_face[i])
            .map(|i| {
                let f = input.tris[i];
                let (a, b, c) = (
                    solver.state().positions[f[0] as usize],
                    solver.state().positions[f[1] as usize],
                    solver.state().positions[f[2] as usize],
                );
                let area = 0.5 * (b - a).cross(c - a).length();
                tri_p.density[i] * area
            })
            .collect();
        solver
            .set_strain_limit_faces(&elastic, &limits, &masses)
            .map_err(|e| anyhow::anyhow!(e))?;
        rep.n_strain_limited = solver.strain_limit_face_count();
        // The edge-spring path is superseded for these faces.
        solver.params.strain_limit_ratio = 0.0;
        solver.params.strain_limit_k = 0.0;
    }

    // ---- per-hinge bending stiffness from the scene's per-face `bend` ----
    // Uses the reference's area-weighted face→hinge mapping and its resolution-independent
    // coefficient (see `set_hinge_bend_from_faces`). Supersedes `bend_stiffness`.
    if per_face_bend {
        solver
            .set_hinge_bend_from_faces(&input.tris, &bend_p.bend)
            .map_err(|e| anyhow::anyhow!(e))?;
        rep.n_hinges = solver.hinge_count();
    } else if !bend_p.bend.is_empty() {
        rep.unsupported.push(format!(
            "per-face bend array length {} does not match the face count {}; the scene-wide \
             mean {bend_k:.3} is used with our own hinge coefficient",
            bend_p.bend.len(),
            input.tris.len()
        ));
    }

    // Self-collision barrier stiffness now uses the reference's full `compute_stiffness`
    // (`barrier.cu:48-86`): the Rayleigh quotient of `H_elastic + δ_ij·(m_ii/g²)·I` along
    // the weighted contact direction, with the proximity weights recovered from the
    // distance gradient. No gap of its own; the per-element `ghat`/offset variation it
    // would still miss is reported by the contact-gap and contact-offset checks above.

    // ---- rod segments ----
    // The reference's rod stretch is `hook::make_diff_table` with
    // `weight = young_mod * (density * l_geom)` (`energy.cu:286`, `scene.rs:1498`), whose
    // gradient integrates to `E = w(l − l₀)²/(2 l₀)` with `l₀ = l_geom * length_factor`.
    // That is a spring of stiffness `w / l₀ = young_mod * density / length_factor`,
    // independent of the segment's own length. See `implicit::Rod`.
    if !input.rods.is_empty() {
        let n = input.rods.len();
        let have = |v: &Vec<f32>| v.len() == n;
        if have(&rod_p.young_mod) && have(&rod_p.density) && have(&rod_p.length_factor) {
            let rods: Vec<engine_core::implicit::Rod> = input
                .rods
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let l = (solver.state().positions[r[1] as usize]
                        - solver.state().positions[r[0] as usize])
                        .length();
                    let lf = rod_p.length_factor[i];
                    engine_core::implicit::Rod {
                        a: r[0],
                        b: r[1],
                        rest_len: (l * lf).max(1e-9),
                        k: rod_p.young_mod[i] * rod_p.density[i] / lf.max(1e-9),
                        // The segment's contact radius. Absent array ⇒ a zero-thickness
                        // curve, which is what the reference's default offset is.
                        offset: rod_p.contact_offset.get(i).copied().unwrap_or(0.0),
                        // Per-segment activation gap; the reference composes a pair's as the
                        // MEAN of the two (`contact.cu:721`). 0 => the scene-wide dhat.
                        ghat: rod_p.contact_gap.get(i).copied().unwrap_or(0.0),
                    }
                })
                .collect();
            rep.n_rods = rods.len();
            solver.set_rods(rods).map_err(|e| anyhow::anyhow!(e))?;
        } else {
            rep.unsupported.push(format!(
                "{n} rod segments ship no complete young-mod/density/length-factor \
                 triple; rod stretch omitted"
            ));
        }
        // Rod BENDING — the reference's `embed_rod_bend_force_hessian` (`energy.cu:495`):
        // `½·stiff_k·(θ − θ₀)²` at each vertex with exactly 2 incident rod segments and 0
        // incident faces, where `stiff_k = mean(bend of the two segments) * vertex_mass`
        // (`energy.cu:508-511`) and `θ₀` defaults to π = straight (`builder.rs:385-388`).
        // `codim` and `yarn` ship `rod-bend = 0`, which the reference itself skips
        // (`if (mass > 0 && stiff_k > 0)`), so they get no hinges and lose nothing.
        if rod_p.bend.len() == n && !lumped_mass.is_empty() {
            // Per-vertex mean segment bend: every hinge vertex has degree exactly 2, so the
            // mean is the accumulated sum over 2.
            let mut bend_sum = vec![0.0f32; solver.state().positions.len()];
            for (i, r) in input.rods.iter().enumerate() {
                bend_sum[r[0] as usize] += rod_p.bend[i];
                bend_sum[r[1] as usize] += rod_p.bend[i];
            }
            let hinges: Vec<engine_core::rod_bend::RodHinge> =
                engine_core::rod_bend::build_rod_hinges(
                    solver.state().positions.len(),
                    &input.rods,
                    &input.tris,
                )
                .into_iter()
                .map(|h| {
                    let c = h[1] as usize;
                    engine_core::rod_bend::RodHinge {
                        j: h[0],
                        i: h[1],
                        k: h[2],
                        // No gauge scene sets `rod-bend-rest-from-geometry`, so this is the
                        // reference's default; a scene that ever does is reported below.
                        rest_angle: std::f32::consts::PI,
                        stiff: 0.5 * bend_sum[c] * lumped_mass[c],
                    }
                })
                .collect();
            rep.n_rod_hinges = hinges.iter().filter(|h| h.stiff > 0.0).count();
            solver
                .set_rod_hinges(hinges)
                .map_err(|e| anyhow::anyhow!(e))?;
        } else if rod_p.bend.iter().any(|&b| b > 0.0) {
            rep.unsupported.push(
                "rod segments carry bending but the bend/density arrays are incomplete; \
                 rod bending omitted"
                    .into(),
            );
        }
        // Two rod-bend refinements the reference has and we do not. Reported only when a
        // scene actually turns them on — every gauge scene ships both as 0.
        let geo = rod_p
            .bend_rest_from_geometry
            .iter()
            .filter(|&&g| g != 0.0)
            .count();
        if geo > 0 {
            rep.unsupported.push(format!(
                "{geo} rod segments request bend-rest-from-geometry; the straight rest \
                 angle π is used instead"
            ));
        }
        let damp = rod_p.bending_damping.iter().filter(|&&d| d > 0.0).count();
        if damp > 0 {
            rep.unsupported.push(format!(
                "{damp} rod segments set bending-damping; the reference's lagged bending \
                 damping is not modeled"
            ));
        }
        // Rod CONTACT: rod segments are now part of the collision edge set, so they take
        // part in edge-edge CCD and the barrier with their own radius (`rod-contact-offset`)
        // subtracted from the measured distance. What is still approximated is the
        // ACTIVATION GAP: the reference's `ghat` is per-element (`rod-contact-gap`) while
        // ours is the scene-wide `contact_dhat`, so report it whenever a rod's gap differs
        // from what we drive.
        if rod_p.contact_offset.len() != n && rod_p.contact_offset.iter().any(|&o| o > 0.0) {
            rep.unsupported.push(
                "rod contact-offset array is incomplete; strands treated as zero-thickness".into(),
            );
        }
        if rod_p.contact_gap.len() != n && rod_p.contact_gap.iter().any(|&g| g > 0.0) {
            rep.unsupported.push(
                "rod contact-gap array is incomplete; the scene-wide contact_dhat is used".into(),
            );
        }
        for (name, v) in [
            ("strain-limit", &rod_p.strain_limit),
            ("bend-plasticity", &rod_p.plasticity),
            ("deformation-damping", &rod_p.deformation_damping),
            ("pressure", &rod_p.pressure),
        ] {
            let on = v.iter().filter(|&&x| x > 0.0).count();
            if on > 0 {
                rep.unsupported
                    .push(format!("{on} rod segments set {name}; not modeled"));
            }
        }
    }

    // ---- stitches ----
    // UNREPORTED GAP found in S118 while surveying the gap tally: `fitting` ships 851
    // stitches and was claiming ZERO gaps. `stitch_ind.bin` is `u64[S][6]` (48 B/stitch,
    // width 6 is an ABI contract), with `stitch_w.bin` `f32[S][6]` barycentric weights and
    // an optional `stitch_stiffness.bin` `f32[S]` defaulting to 1.0
    // (`PORTING_data_layout.md:676-678`, `scene.rs:616-617`). Not modeled — the seams stay
    // unsewn — so say so rather than let the scene read as fully reproduced.
    let scene_stitches = engine_io::golden::read_stitches(scene).unwrap_or_default();
    if !scene_stitches.is_empty() {
        rep.n_stitches = scene_stitches.len();
        // `l0` is the weighted mean of the six vertices' contact `ghat` — the reference sums
        // `wᵢ·ghatᵢ` over all six and halves it, and since each side's weights sum to 1 the
        // six weights sum to 2, making it a weighted MEAN (`energy.cu:726-728`). Each side's
        // `offset` is the MAX over its three vertices, and
        // `l_cap = stitch_length_factor·l0 + source_offset + target_offset`.
        //
        // A vertex's `ghat`/`offset` come from its incident FACES: `fitting` ships only
        // `tri-*` params, no per-vertex ones. Both of its arrays are uniform (gap 0.001,
        // offset 0.0), so the per-vertex value is that constant however the reference
        // aggregates it — but a scene with VARYING face gaps would need the real rule, so
        // that case is reported rather than guessed.
        let nv = solver.state().positions.len();
        let mut v_gap = vec![f32::NAN; nv];
        let mut v_off = vec![0.0f32; nv];
        let have_gap = tri_p.contact_gap.len() == input.tris.len();
        let have_off = tri_p.contact_offset.len() == input.tris.len();
        if have_gap || have_off {
            for (i, t) in input.tris.iter().enumerate() {
                for &vi in t {
                    let vi = vi as usize;
                    if have_gap {
                        let g = tri_p.contact_gap[i];
                        v_gap[vi] = if v_gap[vi].is_nan() {
                            g
                        } else {
                            v_gap[vi].max(g)
                        };
                    }
                    if have_off {
                        v_off[vi] = v_off[vi].max(tri_p.contact_offset[i]);
                    }
                }
            }
        }
        // Falls back to the scene-wide contact gap the driver already derived.
        let gap_fallback = solver.params.contact_dhat;
        let uniform_gap = have_gap
            && tri_p
                .contact_gap
                .windows(2)
                .all(|w| (w[0] - w[1]).abs() < 1.0e-9);
        if have_gap && !uniform_gap {
            rep.unsupported.push(
                "face contact gaps VARY, so a stitched vertex's ghat depends on the \
                 reference's face->vertex aggregation, which is not modeled; the max over \
                 incident faces is used"
                    .into(),
            );
        }
        let slf = toml
            .as_ref()
            .map(|t| t.stitch_length_factor)
            .unwrap_or(10.0);
        let stitches: Vec<engine_core::implicit::Stitch> = scene_stitches
            .iter()
            .map(|st| {
                let gap_of = |v: u32| {
                    let g = v_gap[v as usize];
                    if g.is_nan() || g <= 0.0 {
                        gap_fallback
                    } else {
                        g
                    }
                };
                let l0 = 0.5
                    * (0..6)
                        .map(|k| st.weight[k] * gap_of(st.index[k]))
                        .sum::<f32>();
                let l0 = if l0 > 0.0 { l0 } else { gap_fallback };
                let src_off = (0..3)
                    .map(|k| v_off[st.index[k] as usize])
                    .fold(0.0f32, f32::max);
                let tgt_off = (3..6)
                    .map(|k| v_off[st.index[k] as usize])
                    .fold(0.0f32, f32::max);
                engine_core::implicit::Stitch {
                    index: st.index,
                    weight: st.weight,
                    stiffness: st.stiffness,
                    l0,
                    l_cap: slf * l0 + src_off + tgt_off,
                }
            })
            .collect();
        solver
            .set_stitches(stitches)
            .map_err(|e| anyhow::anyhow!(e))?;
    }

    // ---- analytic colliders ----
    // The driver attaches NONE, so a scene that ships walls or spheres is being driven
    // without geometry it is supposed to collide against. This was silently unreported
    // until S112: `domino` (a solid pushed onto a wall) claimed zero gaps while ignoring
    // the wall entirely.
    let scene_colliders = engine_io::golden::read_colliders(scene).unwrap_or_default();
    if !scene_colliders.is_empty() {
        let cols = colliders_at(&scene_colliders, 0.0);
        rep.n_colliders = cols.len();
        rep.n_colliders_animated = scene_colliders.iter().filter(|c| c.is_animated()).count();
        solver.set_colliders(cols);
        // What the geometry still does NOT carry. Per-collider friction is combined with the
        // vertex's by the reference (`contact.cu:1162`); ours is per-vertex only, so a
        // collider with its own friction is reported. `thickness` is a pass-through depth
        // that never fires at the gauge's 1.0, and `active-duration < 0` is always-active
        // everywhere — both decoded and checked rather than assumed.
        // The collider coefficient is combined with the per-vertex one as `min`, matching
        // `combine_friction(..., Min)`. A scene mixing DIFFERENT per-collider coefficients
        // cannot be expressed by one scene-wide value, so that case is reported.
        let fr: Vec<f32> = scene_colliders.iter().map(|c| c.friction).collect();
        let uniform = fr.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);
        if uniform {
            if let Some(&f) = fr.first() {
                solver.params.collider_friction = f;
            }
        } else {
            let lo = fr.iter().copied().fold(f32::INFINITY, f32::min);
            solver.params.collider_friction = lo;
            rep.unsupported.push(format!(
                "colliders declare DIFFERENT frictions ({fr:?}); our coefficient is \
                 scene-wide, so the smallest ({lo}) is used for all of them"
            ));
        }
        if scene_colliders.iter().any(|c| c.active_duration >= 0.0) {
            rep.unsupported
                .push("a collider has a finite active-duration; not modeled".into());
        }
    }

    // Newton termination, CCD cap, and the linear-solve controls all come from the scene
    // now (`target_toi` + `min_newton_steps` reproduce `main.cu:1038-1046`), so the
    // iteration strategy is no longer a gap. `newton_iters` is set well above the expected
    // count so it acts purely as a safety bound.

    // Aerodynamic damper is modeled now; only the wind TIME WEIGHT is not.
    if let Some(t) = toml.as_ref() {
        if t.wind.iter().any(|w| w.abs() > 1e-9) {
            rep.unsupported.push(
                "wind is nonzero; the constant vector is applied but the reference's \
                 time-varying wind weight (`get_wind_weight`) is not modeled"
                    .into(),
            );
        }
    }

    // ---- constrained vertices (the reference's FixPair) ----
    // Targets come from the scene's pin SCHEDULE: each block's ordered operations, read
    // from `info.toml` and its `pin-<i>-op-<j>.bin` payloads, composed over time exactly
    // as the reference's `apply_op` loop does. `move_by` and `move_to` are modeled; any
    // other op kind contributes no motion and is named in the report.
    //
    // A read error must NOT be swallowed: an earlier version used `if let Ok(..)` and a
    // decode bug silently produced a report claiming no constrained vertices.
    let ghat = toml.as_ref().map(|t| t.constraint_ghat).unwrap_or(1.0e-3);
    match engine_io::golden::read_pin_schedule(scene) {
        Ok(schedule) => {
            let base: Vec<Vec3> = solver.state().positions.clone();
            let mut unsupported_ops: Vec<String> = Vec::new();
            let mut pulled = 0usize;
            let mut bezier = 0usize;
            for b in &schedule {
                if b.pull > 0.0 {
                    // Driven as weighted PullPairs below; still scan their ops so an
                    // unmodeled op kind in a pulled block is reported too.
                    pulled += b.indices.len();
                }
                for op in &b.ops {
                    if let engine_io::golden::PinOpKind::Unsupported(name) = &op.kind {
                        if !unsupported_ops.contains(name) {
                            unsupported_ops.push(name.clone());
                        }
                    }
                    if op.transition == engine_io::golden::PinTransition::Bezier {
                        bezier += 1;
                    }
                }
            }
            // `pull > 0` blocks are driven as weighted PullPairs (the reference's own
            // split), so they are no longer a gap.
            let pulls = pulls_at(&schedule, &base, 0.0);
            rep.n_pulls_applied = pulls.len();
            debug_assert_eq!(pulls.len(), pulled, "every pulled vertex must get a Pull");
            if !pulls.is_empty() {
                solver.set_pulls(pulls).map_err(|e| anyhow::anyhow!(e))?;
            }
            if !unsupported_ops.is_empty() {
                rep.unsupported.push(format!(
                    "pin operations not modeled: {}; affected pins hold their \
                     last modeled target",
                    unsupported_ops.join(", ")
                ));
            }
            if bezier > 0 {
                rep.unsupported.push(format!(
                    "{bezier} pin operations use bezier easing; the linear ramp is used"
                ));
            }
            let pins = pins_at(&schedule, &base, ghat, 0.0);
            rep.n_pins_applied = pins.len();
            rep.n_pins_kinematic = pins.iter().filter(|p| p.kinematic).count();
            // Static pins are contained by the time-of-impact clamp
            // (`ImplicitSolver::ccd_max_step`), matching `contact.cu:1996-2012`.
            if !pins.is_empty() {
                solver.set_pins(pins).map_err(|e| anyhow::anyhow!(e))?;
            }
        }
        Err(e) => rep.unsupported.push(format!(
            "pin schedule could not be decoded ({e}); no pins applied"
        )),
    }

    Ok((solver, rep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::linsolve::CgCpu;

    // One implementation, in engine-io. This copy used `exists()` for the env-var
    // override, which is true for a FILE too and so let a mistyped `PPF_GAUGE_DIR`
    // past the guard to fail deep inside the assertions. See
    // `engine_io::golden::gauge_dir`.
    use engine_io::golden::gauge_dir;

    /// The mixed tri+tet milestone: the reference's `plastic` scene (5654 verts, 8964
    /// shells split 3844 Baraff-Witkin / 5120 ARAP, 12670 ARAP tets) must LOAD and STEP
    /// in our solver, coupling all three element types in one solve, staying finite with
    /// no inverted tets.
    ///
    /// This is a stability/integration check, NOT a parity claim — `DriverReport`
    /// enumerates the fidelity gaps that still make a frame-checksum comparison
    /// meaningless.
    /// A scene's wall must actually STOP its geometry. `domino` is a solid standing on a
    /// floor at y = −0.1255 with no gravity support otherwise: before colliders were attached
    /// it simply fell through, so this is the test that would have caught that.
    #[test]
    fn domino_rests_on_its_wall_instead_of_falling_through() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let scene = gauge.join("goldens/domino");
        if !scene.exists() {
            eprintln!("domino absent; skipping");
            return;
        }
        let (mut run, rep) = scene_run(&scene).expect("domino");
        assert_eq!(rep.n_colliders, 1, "domino ships one wall");
        // Self-collision is off for affordability here; the wall is what is under test.
        run.solver.params.self_collision = false;
        let floor_y = -0.1255f32;
        let solve = CgCpu::new();
        let mut worst = f32::INFINITY;
        // 20 steps, not 60: a free fall would already be far below the floor by then, and
        // this runs in a debug build where each domino step is ~5 s.
        for _ in 0..20 {
            run.step(&solve);
            let m = run
                .solver
                .state
                .positions
                .iter()
                .map(|p| p.y)
                .fold(f32::INFINITY, f32::min);
            worst = worst.min(m);
        }
        assert!(
            run.solver.state.positions.iter().all(|p| p.is_finite()),
            "state must stay finite"
        );
        // The barrier permits a small penetration on the order of the contact gap (1e-3);
        // free-falling through would be orders of magnitude below the floor.
        assert!(
            worst > floor_y - 0.02,
            "geometry fell through its wall: min y {worst:.5} vs floor {floor_y}"
        );
    }

    /// `trapped`'s sphere is the gauge's ONE animated collider (4 keyframes moving centre and
    /// radius). Sampling must actually change it over time — a static read would leave the
    /// sphere parked at its frame-0 pose while the reference sweeps it through the cloth.
    #[test]
    fn trapped_collider_actually_animates() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let scene = gauge.join("goldens/trapped");
        if !scene.exists() {
            eprintln!("trapped absent; skipping");
            return;
        }
        let cols = engine_io::golden::read_colliders(&scene).expect("colliders");
        assert_eq!(cols.len(), 1);
        assert!(cols[0].is_animated(), "trapped's sphere must be animated");
        let last = *cols[0].times.last().unwrap();
        let a = colliders_at(&cols, 0.0);
        let b = colliders_at(&cols, last);
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        let unpack = |c: &engine_core::contact::Collider| match *c {
            engine_core::contact::Collider::Sphere { center, radius } => (center, radius),
            _ => panic!("expected a sphere"),
        };
        let (ca, ra) = unpack(&a[0]);
        let (cb, rb) = unpack(&b[0]);
        assert!(
            (ca - cb).length() > 1.0e-4 || (ra - rb).abs() > 1.0e-4,
            "the animated sphere must differ between t=0 and t={last}: \
             {ca:?}/{ra} vs {cb:?}/{rb}"
        );
    }

    /// SYSTEMIC guard: every scene's `friction_mode` must reach the solver.
    ///
    /// Written after finding that the driver hardwired `Min` while **14 of the 22 scenes ship
    /// `"max"`** — a claim in the S117 audit entry ("`friction_mode = \"min\"` in every scene")
    /// that was simply wrong and that nothing tested. Under `Min` a collider declaring friction
    /// 0 makes its contacts frictionless; under `Max` the cloth keeps its own coefficient, so
    /// this is a visible physical difference on the majority of the suite.
    #[test]
    fn every_scene_friction_mode_reaches_the_solver() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let Ok(entries) = std::fs::read_dir(gauge.join("goldens")) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut seen_max = 0usize;
        let mut seen_min = 0usize;
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            let toml_path = scene.join("param.toml");
            if !toml_path.exists() {
                continue;
            }
            let name = scene.file_name().unwrap().to_string_lossy().to_string();
            // Read the raw declaration straight from the file, independently of our parser.
            let raw = std::fs::read_to_string(&toml_path).unwrap_or_default();
            let declared = raw
                .lines()
                .find_map(|l| l.strip_prefix("friction_mode"))
                .and_then(|r| r.split('=').nth(1))
                .map(|v| v.trim().trim_matches('"').to_string())
                .unwrap_or_else(|| "min".to_string());
            let want = match declared.as_str() {
                "max" => engine_core::contact::FrictionMode::Max,
                "mean" => engine_core::contact::FrictionMode::Mean,
                _ => engine_core::contact::FrictionMode::Min,
            };
            let Ok((solver, _)) = build_from_scene(&scene) else {
                continue;
            };
            assert_eq!(
                solver.params.friction_mode, want,
                "{name}: param.toml declares friction_mode = {declared:?} but the solver got \
                 {:?}",
                solver.params.friction_mode
            );
            match want {
                engine_core::contact::FrictionMode::Max => seen_max += 1,
                engine_core::contact::FrictionMode::Min => seen_min += 1,
                engine_core::contact::FrictionMode::Mean => {}
            }
        }
        // The suite is majority-`max`; if that ever flips, the note above needs rewriting.
        assert!(
            seen_max > seen_min,
            "expected most scenes to ship friction_mode = max, saw {seen_max} max / {seen_min} min"
        );
        eprintln!("friction_mode wired for {seen_max} max + {seen_min} min scenes");
    }

    /// SYSTEMIC guard: a scene whose per-face `tri-young-mod` VARIES must reach the solver
    /// as varying element stiffness, not collapsed to a scene-wide mean.
    ///
    /// Written after an audit wrongly concluded this was an unreported gap: the wiring
    /// existed (`set_membrane_stiffness` / `set_shell_stiffness`) but NOTHING asserted it,
    /// so the only check available was reading the code — and the wrong symbol name got
    /// grepped. A guard is worth more than a re-read.
    ///
    /// The first version of this test then failed on `needle` for a second wrong reason: it
    /// measured variation over ALL faces, including a solid's surface faces, which carry a
    /// young-mod but no in-plane material.
    #[test]
    fn per_face_young_modulus_reaches_the_solver_unaveraged() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let Ok(entries) = std::fs::read_dir(gauge.join("goldens")) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut checked = 0usize;
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            if !scene.join("param.toml").exists() {
                continue;
            }
            let name = scene.file_name().unwrap().to_string_lossy().to_string();
            let tp = match load_element_params(&scene, "tri", false) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let Ok(input) = engine_io::golden::load_golden_input(&scene) else {
                continue;
            };
            if tp.young_mod.len() != input.tris.len() {
                continue;
            }
            // Only ELASTIC faces count. A solid's surface faces DO appear in `tri.bin` and
            // carry a young-mod, but the reference dispatches in-plane energy over
            // `shell_face_count` only, so they are contact geometry with no material — and
            // they are exactly where `needle`'s 500 lives (5120 ARAP tet-boundary faces
            // against 163840 cloth faces at 100). Counting them made this test claim a gap
            // that is not one.
            let mut face_count: std::collections::HashMap<(u32, u32, u32), u32> =
                std::collections::HashMap::new();
            for t in &input.tets {
                for f in engine_core::tet::tet_faces(t) {
                    let mut k = f;
                    k.sort_unstable();
                    *face_count.entry((k[0], k[1], k[2])).or_insert(0) += 1;
                }
            }
            let elastic: Vec<f32> = input
                .tris
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    let mut k = **t;
                    k.sort_unstable();
                    !face_count.contains_key(&(k[0], k[1], k[2]))
                })
                .map(|(i, _)| tp.young_mod[i])
                .collect();
            let lo = elastic.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = elastic.iter().copied().fold(0.0f32, f32::max);
            if elastic.is_empty() || hi <= lo * 1.001 {
                continue; // uniform over elastic faces: the scene-wide value IS exact
            }
            let Ok((solver, _)) = build_from_scene(&scene) else {
                continue;
            };
            // Every elastic face carries SOME model, so the variation must show up in the
            // membrane list, the shell list, or both.
            let mut spread = 0.0f32;
            for k in solver.membrane_stiffness() {
                spread = spread.max(k[0]);
            }
            let mut mlo = f32::INFINITY;
            for k in solver.membrane_stiffness() {
                mlo = mlo.min(k[0]);
            }
            let mut slo = f32::INFINITY;
            let mut shi = 0.0f32;
            for k in solver.shell_stiffness() {
                slo = slo.min(k[0]);
                shi = shi.max(k[0]);
            }
            let membrane_varies = !solver.membrane_stiffness().is_empty() && spread > mlo * 1.001;
            let shell_varies = !solver.shell_stiffness().is_empty() && shi > slo * 1.001;
            assert!(
                membrane_varies || shell_varies,
                "{name}: tri-young-mod varies {lo}..{hi} but the solver's element \
                 stiffness does not — it was collapsed to a scene-wide value. \
                 membrane {} entries ({mlo}..{spread}), shell {} entries ({slo}..{shi})",
                solver.membrane_stiffness().len(),
                solver.shell_stiffness().len()
            );
            checked += 1;
        }
        // Measured 2026-07-28: of the 8 scenes whose `tri-young-mod` array holds two
        // values, only `fitting` varies it across ELASTIC faces (100 and 1000 over all
        // 54928). In `cards`, `needle`, `plastic`, `ribbon`, `roller`, `stack` and
        // `trampoline` the second value belongs entirely to solid-surface faces, which carry
        // no in-plane material — so per-face stiffness is load-bearing for exactly one scene
        // and this guard covers it.
        assert_eq!(
            checked, 1,
            "expected exactly `fitting` to vary young-mod across elastic faces, saw \
             {checked} scene(s); if a scene was added or reclassified, re-derive the note \
             above rather than bumping this number"
        );
    }

    /// SYSTEMIC guard: a scene shipping `stitch_ind.bin` must ATTACH that many stitches,
    /// with the reference's rest length and saturation cap.
    ///
    /// This replaces the S118 version, which asserted stitches implied a reported GAP. That
    /// was the right invariant while they were unimplemented (it caught `fitting` claiming
    /// full fidelity while ignoring 851 seams) and became wrong the moment they landed — the
    /// test failing on that change is the guard doing its job, the same way the S112
    /// collider guard did.
    #[test]
    fn every_scene_attaches_the_stitches_it_ships() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let Ok(entries) = std::fs::read_dir(gauge.join("goldens")) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut with = 0usize;
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            if !scene.join("param.toml").exists() {
                continue;
            }
            let name = scene.file_name().unwrap().to_string_lossy().to_string();
            // `u64[S][6]` — 48 bytes per stitch; the f32[S][6] weights must agree on S,
            // which is what pins the width-6 layout down.
            let n = std::fs::metadata(scene.join("input_bin/stitch_ind.bin"))
                .map(|m| m.len() as usize / 48)
                .unwrap_or(0);
            let Ok((solver, rep)) = build_from_scene(&scene) else {
                continue;
            };
            assert_eq!(rep.n_stitches, n, "{name}: stitch count");
            assert_eq!(
                solver.stitches().len(),
                n,
                "{name}: report and solver disagree on stitch count"
            );
            if n == 0 {
                continue;
            }
            with += 1;
            let w = std::fs::metadata(scene.join("input_bin/stitch_w.bin"))
                .map(|m| m.len() as usize / 24)
                .unwrap_or(0);
            assert_eq!(
                w, n,
                "{name}: stitch_w implies {w} stitches but stitch_ind implies {n}"
            );
            assert!(
                !rep.unsupported.iter().any(|u| u.contains("stitches")),
                "{name}: stitches are implemented but still reported as a gap: {:?}",
                rep.unsupported
            );

            // Recompute the reference's parameters straight from disk. `fitting` ships a
            // uniform face contact gap of 0.001 and zero offset, and
            // `stitch_length_factor = 10`, so every stitch must get l0 = 0.001, cap = 0.01.
            let raw = engine_io::golden::read_stitches(&scene).expect("stitches");
            let toml = read_param_toml(&scene).ok();
            let slf = toml
                .as_ref()
                .map(|t| t.stitch_length_factor)
                .unwrap_or(10.0);
            let tp = load_element_params(&scene, "tri", false).expect("tri params");
            let gaps: Vec<f32> = tp.contact_gap.clone();
            let uniform = !gaps.is_empty()
                && gaps.windows(2).all(|p| (p[0] - p[1]).abs() < 1.0e-9)
                && tp.contact_offset.iter().all(|&o| o == 0.0);
            for (i, (st, got)) in raw.iter().zip(solver.stitches()).enumerate() {
                assert_eq!(got.index, st.index, "{name}: stitch {i} indices");
                assert_eq!(got.weight, st.weight, "{name}: stitch {i} weights");
                assert!(
                    (got.stiffness - st.stiffness).abs() < 1.0e-9,
                    "{name}: stitch {i} stiffness {} != {}",
                    got.stiffness,
                    st.stiffness
                );
                // Each side's barycentric weights must sum to 1 — the property that makes
                // the reference's centre-of-gravity terms cancel.
                for (half, r) in [("source", 0..3), ("target", 3..6)] {
                    let sum: f32 = r.map(|k| st.weight[k]).sum();
                    assert!(
                        (sum - 1.0).abs() < 1.0e-3,
                        "{name}: stitch {i} {half} weights sum to {sum}"
                    );
                }
                if uniform {
                    let want_l0 = gaps[0];
                    assert!(
                        (got.l0 - want_l0).abs() <= 1.0e-6 * want_l0,
                        "{name}: stitch {i} l0 {} != the uniform face gap {want_l0}",
                        got.l0
                    );
                    assert!(
                        (got.l_cap - slf * want_l0).abs() <= 1.0e-6 * slf * want_l0,
                        "{name}: stitch {i} cap {} != stitch_length_factor {slf} * l0",
                        got.l_cap
                    );
                }
            }
            // The feature only matters if the seams are actually OPEN at frame 0 — a stitch
            // list that is already satisfied would make any parity claim vacuous.
            let p = &solver.state().positions;
            let open = solver
                .stitches()
                .iter()
                .filter(|st| {
                    let xs: [Vec3; 6] = std::array::from_fn(|k| p[st.index[k] as usize]);
                    engine_core::stitch::seam_vector(&xs, &st.weight).length() > 2.0 * st.l0
                })
                .count();
            assert!(
                open > 0,
                "{name}: every seam is already closed at frame 0, so the stitch energy \
                 would be doing nothing"
            );
            eprintln!("{name}: {n} stitches attached, {open} seams open at frame 0");
        }
        assert!(
            with >= 1,
            "expected at least one stitched scene, saw {with}"
        );
    }

    /// SYSTEMIC guard: every scene that ships `rod.bin` must attach exactly that many rod
    /// segments, with the reference's rest length and stiffness. Walking every scene means a
    /// rod scene added later is covered without anyone remembering.
    ///
    /// The numbers are recomputed here straight from the on-disk params rather than from the
    /// driver's own arithmetic, so this fails if the `length_factor` ordering regresses —
    /// the reference takes rod MASS from the geometric length and only then scales the REST
    /// length (`scene.rs:1498-1499`), which is what pre-tensions `yarn` (0.85) and `woven`
    /// (0.8).
    #[test]
    fn every_scene_attaches_the_rods_it_ships() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let Ok(entries) = std::fs::read_dir(gauge.join("goldens")) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut with = 0usize;
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            if !scene.join("param.toml").exists() {
                continue;
            }
            let name = scene.file_name().unwrap().to_string_lossy().to_string();
            let n_disk = std::fs::metadata(scene.join("input_bin/rod.bin"))
                .map(|m| m.len() as usize / 16)
                .unwrap_or(0);
            let Ok((solver, rep)) = build_from_scene(&scene) else {
                continue;
            };
            assert_eq!(
                rep.n_rods, n_disk,
                "{name}: rod.bin holds {n_disk} segments but {} were attached",
                rep.n_rods
            );
            assert_eq!(
                solver.rods().len(),
                rep.n_rods,
                "{name}: report and solver disagree on rod count"
            );
            if n_disk == 0 {
                continue;
            }
            with += 1;

            let input = engine_io::golden::load_golden_input(&scene).expect("input");
            let rp = load_element_params(&scene, "rod", true).expect("rod params");
            for (i, rd) in solver.rods().iter().enumerate() {
                let (a, b) = (input.rods[i][0], input.rods[i][1]);
                assert_eq!((rd.a, rd.b), (a, b), "{name}: rod {i} endpoints");
                let l = (Vec3::from(input.positions[b as usize])
                    - Vec3::from(input.positions[a as usize]))
                .length();
                let lf = rp.length_factor[i];
                let want_rest = l * lf;
                assert!(
                    (rd.rest_len - want_rest).abs() <= 1.0e-6 * want_rest.max(1.0e-6),
                    "{name}: rod {i} rest length {} != geometric {l} * length_factor {lf}",
                    rd.rest_len
                );
                // Contact radius and activation gap must come from the scene, not a default:
                // the reference subtracts the SUM of the two radii from the distance and
                // uses the MEAN of the two gaps (`contact.cu:558,721`).
                assert!(
                    (rd.offset - rp.contact_offset[i]).abs() <= 1.0e-9,
                    "{name}: rod {i} offset {} != rod-contact-offset {}",
                    rd.offset,
                    rp.contact_offset[i]
                );
                assert!(
                    (rd.ghat - rp.contact_gap[i]).abs() <= 1.0e-9,
                    "{name}: rod {i} ghat {} != rod-contact-gap {}",
                    rd.ghat,
                    rp.contact_gap[i]
                );
                let want_k = rp.young_mod[i] * rp.density[i] / lf;
                assert!(
                    (rd.k - want_k).abs() <= 1.0e-3 * want_k,
                    "{name}: rod {i} stiffness {} != young_mod * density / length_factor \
                     = {want_k}",
                    rd.k
                );
            }

            // Rod vertices must carry the reference's lumped rod mass, `density * l / 2`
            // per endpoint — this is what closed `codim`'s 1365 zero-mass vertices.
            let mut want_mass = vec![0.0f32; input.positions.len()];
            for (i, r) in input.rods.iter().enumerate() {
                let l = (Vec3::from(input.positions[r[1] as usize])
                    - Vec3::from(input.positions[r[0] as usize]))
                .length();
                let m = rp.density[i] * l / 2.0;
                want_mass[r[0] as usize] += m;
                want_mass[r[1] as usize] += m;
            }
            // Only vertices touched by rods ALONE can be checked exactly; in `codim` the rod
            // vertices are disjoint from every tri and tet (verified: rod∩tri = rod∩tet = 0).
            let mut touched = vec![false; input.positions.len()];
            for t in &input.tris {
                for &v in t {
                    touched[v as usize] = true;
                }
            }
            for t in &input.tets {
                for &v in t {
                    touched[v as usize] = true;
                }
            }
            let inv = &solver.state().inv_mass;
            let mut checked = 0usize;
            for (v, &m) in want_mass.iter().enumerate() {
                if m <= 0.0 || touched[v] {
                    continue;
                }
                // A vertex pinned by the scene legitimately has inv_mass 0.
                if inv[v] == 0.0 {
                    continue;
                }
                let got = 1.0 / inv[v];
                assert!(
                    (got - m).abs() <= 1.0e-3 * m,
                    "{name}: rod vertex {v} mass {got} != density * length / 2 = {m}"
                );
                checked += 1;
            }
            assert!(
                checked > 0,
                "{name}: no rod-only vertex mass could be checked"
            );

            // Rod segments must be IN the collision edge set — without that they are
            // invisible to contact and strands pass through each other.
            assert!(
                solver.collision_edge_count() >= input.rods.len(),
                "{name}: {} collision edges is fewer than its {} rod segments, so some \
                 rod cannot be contacting anything",
                solver.collision_edge_count(),
                input.rods.len()
            );

            // And the gap this closed must not come back.
            assert!(
                !rep.unsupported
                    .iter()
                    .any(|u| u.contains("no element mass")),
                "{name}: still reports vertices with no element mass: {:?}",
                rep.unsupported
            );

            // ---- rod bending ----
            // Interior rod vertices, recomputed from the topology: degree exactly 2 in the
            // rod graph and on no triangle. A scene whose `rod-bend` is 0 must get NO
            // hinges, because the reference skips them too (its `stiff_k > 0` gate).
            let mut deg = vec![0u32; input.positions.len()];
            for r in &input.rods {
                deg[r[0] as usize] += 1;
                deg[r[1] as usize] += 1;
            }
            let mut on_face = vec![false; input.positions.len()];
            for t in &input.tris {
                for &v in t {
                    on_face[v as usize] = true;
                }
            }
            let interior = (0..input.positions.len())
                .filter(|&v| deg[v] == 2 && !on_face[v])
                .count();
            let any_bend = rp.bend.iter().any(|&b| b > 0.0);
            let want_hinges = if any_bend { interior } else { 0 };
            assert_eq!(
                rep.n_rod_hinges,
                want_hinges,
                "{name}: {interior} interior rod vertices, rod-bend {} => expected \
                 {want_hinges} stiff hinges, attached {}",
                if any_bend { "> 0" } else { "== 0" },
                rep.n_rod_hinges
            );
            if any_bend {
                assert!(
                    !rep.unsupported.iter().any(|u| u.contains("rod bending")),
                    "{name}: rod bending is implemented but still reported: {:?}",
                    rep.unsupported
                );
            }
            // `stiff_k = mean(bend of the two incident segments) * vertex_mass`
            // (`energy.cu:508-511`), recomputed from disk.
            let mut bend_sum = vec![0.0f32; input.positions.len()];
            for (i, r) in input.rods.iter().enumerate() {
                bend_sum[r[0] as usize] += rp.bend[i];
                bend_sum[r[1] as usize] += rp.bend[i];
            }
            for h in solver.rod_hinges() {
                let c = h.i as usize;
                assert_eq!(deg[c], 2, "{name}: hinge centre {c} must have rod degree 2");
                assert!(!on_face[c], "{name}: hinge centre {c} must be on no face");
                assert!(
                    (h.rest_angle - std::f32::consts::PI).abs() < 1.0e-6,
                    "{name}: rest angle must be the straight default pi, got {}",
                    h.rest_angle
                );
                let want = 0.5 * bend_sum[c] * want_mass[c];
                assert!(
                    (h.stiff - want).abs() <= 1.0e-3 * want.max(1.0e-12),
                    "{name}: hinge {c} stiff {} != mean(bend) * vertex_mass = {want}",
                    h.stiff
                );
            }
        }
        assert!(with >= 4, "expected >= 4 rod scenes, saw {with}");
    }

    /// SYSTEMIC guard: for EVERY gauge scene, the collider inputs it ships must be ATTACHED
    /// — one engine collider per declared wall/sphere — and a scene that ships none must get
    /// none.
    ///
    /// This replaces the S112 version, which asserted collider inputs implied a REPORTED GAP.
    /// That was the right invariant while the driver ignored them (it caught `domino`
    /// silently claiming full fidelity while dropping the wall it stands on) and became wrong
    /// the moment they were implemented — the test failing on that change is the guard doing
    /// its job, not a regression. Walking every scene means a scene added later is covered
    /// without anyone remembering.
    #[test]
    fn every_scene_attaches_the_colliders_it_ships() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let Ok(entries) = std::fs::read_dir(gauge.join("goldens")) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut with = 0usize;
        let mut without = 0usize;
        let mut animated = 0usize;
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            if !scene.join("param.toml").exists() {
                continue;
            }
            let (walls, spheres) = engine_io::golden::count_collider_inputs(&scene);
            let Ok((solver, rep)) = build_from_scene(&scene) else {
                continue;
            };
            let name = scene.file_name().unwrap().to_string_lossy().to_string();
            assert_eq!(
                rep.n_colliders,
                walls + spheres,
                "{name}: ships {walls} wall(s) + {spheres} sphere(s) but attached \
                 {} collider(s)",
                rep.n_colliders
            );
            // The report must agree with the solver, not just with itself.
            assert_eq!(
                solver.colliders_len(),
                rep.n_colliders,
                "{name}: report and solver disagree on collider count"
            );
            // Colliders are attached now, so none may be reported as absent geometry.
            assert!(
                !rep.unsupported.iter().any(|u| u.contains("attaches none")),
                "{name}: colliders are implemented; nothing should claim they are absent"
            );
            if walls + spheres > 0 {
                with += 1;
                animated += rep.n_colliders_animated;
            } else {
                without += 1;
            }
        }
        assert!(with > 0, "no collider-bearing scene was checked");
        assert!(without > 0, "no collider-free scene was checked");
        assert_eq!(
            animated, 1,
            "exactly one gauge collider animates (trapped's sphere); found {animated}"
        );
        eprintln!("colliders verified: {with} scenes with, {without} without, {animated} animated");
    }

    /// `hang` ships ONE pin block with no `-op-` files, so its target is static and the
    /// driver must apply it — the complement of `plastic`, whose blocks are all animated
    /// and must therefore be reported rather than pinned at frame 0.
    #[test]
    fn hang_static_pins_are_applied() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let scene = gauge.join("goldens/hang");
        if !scene.exists() {
            eprintln!("hang scene absent; skipping");
            return;
        }
        let (solver, rep) = build_from_scene(&scene).expect("build hang");
        eprintln!("{rep:#?}");
        // info.toml declares `[pin-0] pin = 2`, and the block is static.
        assert_eq!(
            rep.n_pins_applied, 2,
            "hang's two static pins must be applied"
        );
        assert_eq!(solver.pins().len(), 2);
        assert!(
            !rep.unsupported.iter().any(|u| u.contains("ANIMATED pin")),
            "hang has no animated pin block, got {:?}",
            rep.unsupported
        );
        // Each pin's target must be the vertex's own initial position, and ghat must come
        // from the scene (`param.toml: constraint_ghat`), not a hardcoded default.
        let want_ghat = engine_io::golden::read_param_toml(&scene)
            .expect("param.toml")
            .constraint_ghat;
        for pin in solver.pins() {
            assert_eq!(pin.target, solver.state().positions[pin.index as usize]);
            assert_eq!(pin.ghat, want_ghat);
            assert!(!pin.kinematic);
        }
    }

    /// Stepping a `SceneRun` must ADVANCE the animated pin targets, not hold the t=0 ones.
    /// `plastic`'s block-0 pins ramp down 0.74 per unit t over t∈[0,1], so their targets
    /// must sit `0.74·t` below where they started. Kept to a few steps deliberately: each
    /// `plastic` step is ~20 s (5654 verts + 12670 tets), and the property under test is
    /// the schedule wiring, which one step already exercises.
    #[test]
    fn scene_run_advances_animated_pin_targets() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let scene = gauge.join("goldens/plastic");
        if !scene.exists() {
            eprintln!("plastic absent; skipping");
            return;
        }
        let (mut run, _rep) = scene_run(&scene).expect("scene run");
        // As above: this test is about the pin SCHEDULE being re-evaluated per step, so
        // self-collision is disabled to keep it affordable.
        run.solver.params.self_collision = false;
        assert!(run.has_animated_pins(), "plastic has move_by pin ops");
        let base_y: Vec<f32> = run.solver.pins().iter().map(|p| p.target.y).collect();
        assert_eq!(base_y.len(), 2048);

        let solve = CgCpu::new();
        let steps = 3;
        for _ in 0..steps {
            run.step(&solve);
        }
        let dt = engine_io::golden::read_param_toml(&scene)
            .expect("param.toml")
            .dt;
        let t = run.solver.state().time;
        assert!(
            (t - (dt as f64) * steps as f64).abs() < 1e-9,
            "clock should be {} but is {t}",
            dt as f64 * steps as f64
        );

        // Block 0 (the first 1024 pins) ramps at −0.74 per unit t over [0,1].
        let want = -0.74 * t as f32;
        let got = run.solver.pins()[0].target.y - base_y[0];
        assert!(
            (got - want).abs() < 1e-4,
            "block-0 target should have moved {want} in y, moved {got}"
        );
        // Block 1's op has a zero delta, so its targets must NOT drift.
        let b1 = run.solver.pins()[1024].target.y - base_y[1024];
        assert!(b1.abs() < 1e-6, "block-1 target must stay put, moved {b1}");
        assert!(
            run.solver.state.positions.iter().all(|p| p.is_finite()),
            "state must stay finite while pins are driven"
        );
    }

    /// The driver must produce time-correct pin targets for `plastic`: its block-0 pins
    /// move down 0.74 over t∈[0,1], hold, then rise 1.0 over t∈[1.5,2] — a net +0.26.
    /// This is the driver-level counterpart to the schedule test in `engine-io`: it checks
    /// that `pins_at` uses each vertex's own initial position as the base and carries the
    /// block's `ghat`/`stiffness` through.
    #[test]
    fn plastic_pin_targets_animate_over_time() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let scene = gauge.join("goldens/plastic");
        if !scene.exists() {
            eprintln!("plastic absent; skipping");
            return;
        }
        let input = engine_io::golden::load_golden_input(&scene).expect("input");
        let base: Vec<Vec3> = input
            .positions
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();
        let sched = engine_io::golden::read_pin_schedule(&scene).expect("schedule");
        let ghat = engine_io::golden::read_param_toml(&scene)
            .expect("param.toml")
            .constraint_ghat;

        let dy_at = |t: f64| {
            let pins = pins_at(&sched, &base, ghat, t);
            // Block 0 is the first 1024 pins; sample its first vertex.
            let p = pins[0];
            p.target.y - base[p.index as usize].y
        };
        let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
        assert!(near(dy_at(0.0), 0.0), "t=0: {}", dy_at(0.0));
        assert!(near(dy_at(0.5), -0.37), "t=0.5: {}", dy_at(0.5));
        assert!(near(dy_at(1.0), -0.74), "t=1.0: {}", dy_at(1.0));
        assert!(
            near(dy_at(1.25), -0.74),
            "t=1.25 (between ops): {}",
            dy_at(1.25)
        );
        assert!(near(dy_at(2.0), 0.26), "t=2.0: {}", dy_at(2.0));

        // Every pin carries the scene's ghat, and all 2048 are present at t=0.
        let pins = pins_at(&sched, &base, ghat, 0.0);
        assert_eq!(pins.len(), 2048);
        assert!(pins.iter().all(|p| p.ghat == ghat && p.stiffness == 1.0));
        // Block 0's op starts at t=0, so those pins are kinematic immediately.
        assert!(pins[0].kinematic);
    }

    #[test]
    fn plastic_scene_loads_and_steps() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        let scene = gauge.join("goldens/plastic");
        if !scene.exists() {
            eprintln!("plastic scene absent; skipping");
            return;
        }
        let (mut solver, rep) = build_from_scene(&scene).expect("build plastic");
        eprintln!("{rep:#?}");
        // This test checks ASSEMBLY stability (all three element types coupled, finite,
        // no inverted tets), not contact. The driver faithfully enables self-collision
        // from the scene, but on 8964 collision triangles that costs ~40 s per step in
        // release and far more in debug, so it is switched off here — deliberately, for
        // this check only. Contact itself is covered by the engine-core suite.
        solver.params.self_collision = false;
        assert_eq!(rep.n_verts, 5654);
        assert_eq!(rep.n_tris, 8964);
        assert_eq!(rep.n_tets, 12_670);
        // The face split must match the reference's STRUCTURE, which is not the naive
        // by-model split: the reference runs face elastic energy over `shell_face_count`
        // only (energy.cu:655), and this scene's 5120 ARAP faces are exactly the boundary
        // of its 12670 tets — the solid's surface, carrying contact but no in-plane
        // material. So shell_face_count is 3844 and every ARAP face is solid surface.
        assert_eq!(rep.n_baraff_witkin, 3844);
        assert_eq!(
            rep.n_solid_surface, 5120,
            "the ARAP faces are the tet boundary, not shells"
        );
        assert_eq!(
            rep.n_solid_surface + rep.n_baraff_witkin,
            8964,
            "cloth faces and solid-surface faces must partition the face list"
        );
        assert_eq!(rep.n_shell, 0);
        assert_eq!(solver.tets().len(), 12_670, "all tets non-degenerate");
        assert_eq!(
            solver.shells().len(),
            0,
            "a solid's surface must not be driven by a shell model"
        );
        // TRUE partition: the membrane covers ONLY the Baraff-Witkin cloth faces.
        assert_eq!(
            solver.membrane_len(),
            3844,
            "membrane must be restricted to the Baraff-Witkin faces"
        );
        // The solid's surface must still be COLLISION geometry, and must not be
        // duplicated there (it is already present in the scene's own face list).
        assert_eq!(
            solver.collision_tri_count(),
            8964,
            "collision geometry is the full face list, with no duplicated tet boundary"
        );
        // Mass must come from the scene's densities, not the old uniform 1.0. `plastic`
        // ships face density 1..100 and tet density 100, so a uniform unit mass is wrong
        // by orders of magnitude; assert the lumped masses actually vary.
        let inv = &solver.state().inv_mass;
        let (lo, hi) = inv
            .iter()
            .fold((f32::INFINITY, 0.0f32), |(a, b), &m| (a.min(m), b.max(m)));
        assert!(
            inv.iter().all(|m| m.is_finite() && *m > 0.0),
            "every vertex must have positive finite inverse mass"
        );
        assert!(
            hi / lo > 2.0,
            "lumped mass must vary with the scene's densities, got inv_mass {lo:.4e}..{hi:.4e}"
        );
        // Gaps that have since been closed must NOT still be reported. These assertions
        // exist because an earlier edit silently failed to apply and the test still
        // passed — checking the report's CONTENT is what caught it.
        assert!(
            !rep.unsupported
                .iter()
                .any(|u| u.contains("membrane path runs over ALL faces")),
            "the mixed-model double-drive gap should be closed, got {:?}",
            rep.unsupported
        );
        // Gap 2: per-element plasticity must no longer be averaged either.
        assert!(
            !rep.unsupported
                .iter()
                .any(|u| u.contains("plasticity") && u.contains("using the mean")),
            "per-element plasticity should no longer be averaged, got {:?}",
            rep.unsupported
        );
        assert!(
            !rep.unsupported
                .iter()
                .any(|u| u.contains("young-mod") && u.contains("mean")),
            "per-element stiffness should no longer be averaged, got {:?}",
            rep.unsupported
        );
        // This guard used to require `unsupported` to be NON-EMPTY, on the reasoning that a
        // parity attempt must state its own gaps. It earned its keep — it caught a falsely
        // clean report three times (S98, S104, S105). But every material model `plastic`
        // exercises is now ported, so "non-empty" is no longer a true invariant and keeping
        // it would mean inventing a gap to satisfy a test.
        //
        // What replaces it is the property the guard was really protecting: an empty gap
        // list must mean the driver CONSUMED the scene, not that it failed to read it. Each
        // assertion below names a scene input that would silently go missing on a decode or
        // wiring failure — which is exactly how the earlier false-clean reports arose.
        assert_eq!(rep.n_pins_applied, 2048, "pins must be consumed");
        assert_eq!(rep.n_hinges, solver.hinge_count(), "bend must be consumed");
        assert!(rep.n_hinges > 0, "plastic has hinges");
        assert!(
            solver.hinge_k().iter().any(|&k| k > 0.0),
            "per-hinge bend stiffness must be populated from the scene"
        );
        assert!(
            solver
                .state()
                .inv_mass
                .iter()
                .all(|m| m.is_finite() && *m > 0.0),
            "per-vertex mass must be populated from the scene's densities"
        );
        assert!(
            (solver.params.gravity.y + 9.8).abs() < 1e-6,
            "gravity must come from the scene, got {:?}",
            solver.params.gravity
        );
        assert!(
            solver.params.air_density > 0.0 && solver.params.target_toi > 0.0,
            "scene-wide params must be wired: air_density={} target_toi={}",
            solver.params.air_density,
            solver.params.target_toi
        );

        // A few steps of the coupled system: it must remain finite and un-inverted.
        for step in 0..3 {
            solver.step(&CgCpu::new());
            assert!(
                solver.state.positions.iter().all(|p| p.is_finite()),
                "non-finite state at step {step}"
            );
        }
        let inverted = solver
            .tets()
            .iter()
            .filter(|el| {
                let p = [
                    solver.state.positions[el.v[0] as usize],
                    solver.state.positions[el.v[1] as usize],
                    solver.state.positions[el.v[2] as usize],
                    solver.state.positions[el.v[3] as usize],
                ];
                engine_core::tet::deformation_gradient(&p, &el.dm_inv).determinant() <= 0.0
            })
            .count();
        assert_eq!(
            inverted, 0,
            "{inverted} tets inverted during the coupled solve"
        );
    }

    /// Strain limiting must be driven BY THE SCENE, not by a hardcoded default, and it now
    /// runs through the reference's per-FACE BARRIER rather than the edge spring. `plastic`
    /// ships all-zero limits (nothing constrained), `hang` a uniform 0.01 (every face
    /// constrained), and `belt` a MIXED 0..0.05 — where only the faces with a positive limit
    /// may be constrained, which is the per-face fidelity the old averaging path lost.
    #[test]
    fn strain_limit_comes_from_the_scene() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge dir; skipping");
            return;
        };
        // plastic: strain-limit is all zeros ⇒ no face is constrained.
        let plastic = gauge.join("goldens/plastic");
        if plastic.exists() {
            let (s, r) = build_from_scene(&plastic).expect("plastic");
            assert_eq!(
                s.strain_limit_face_count(),
                0,
                "plastic ships strain-limit 0 ⇒ nothing constrained"
            );
            assert_eq!(r.n_strain_limited, 0);
            // With nothing constrained there is no strain-limit gap to report either.
            assert!(
                !r.unsupported.iter().any(|u| u.contains("strain-limit")),
                "no strain limiting ⇒ no strain-limit gap: {:?}",
                r.unsupported
            );
        }
        // hang: uniform 0.01 ⇒ EVERY face carries the barrier, and the superseded
        // edge-spring path must be switched off so the two cannot both act.
        let hang = gauge.join("goldens/hang");
        if hang.exists() {
            let (s, r) = build_from_scene(&hang).expect("hang");
            assert_eq!(
                s.strain_limit_face_count(),
                r.n_tris,
                "hang's uniform limit must constrain every face"
            );
            assert_eq!(
                s.params.strain_limit_ratio, 0.0,
                "the edge-spring path must be superseded by the barrier"
            );
            assert!(
                !r.unsupported
                    .iter()
                    .any(|u| u.contains("EDGE spring") || u.contains("strain-limit varies")),
                "the edge-spring and averaging gaps are closed: {:?}",
                r.unsupported
            );
        }
        // belt: limits vary 0..0.05, so SOME faces are constrained and some are not —
        // strictly between "none" and "all". Averaging would have constrained all of them.
        let belt = gauge.join("goldens/belt");
        if belt.exists() {
            let (s, r) = build_from_scene(&belt).expect("belt");
            let n = s.strain_limit_face_count();
            assert!(
                n > 0 && n < r.n_tris,
                "belt's mixed limits must constrain a strict subset: {n} of {}",
                r.n_tris
            );
        }
    }
}
