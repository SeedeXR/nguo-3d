// SPDX-License-Identifier: GPL-2.0-or-later
//! CPU physics — the **XPBD preview tier**.
//!
//! This is the fast, real, deterministic v0 integrator. It is NOT the ported
//! implicit barrier solver. The engine runs two tiers by design: a fast XPBD
//! preview (this) and a high-fidelity implicit-barrier bake
//! ([`crate::implicit`], plus its GPU port in `engine-gpu`). The step-loop
//! phases below deliberately mirror the reference Newton step
//! (predict → constrain → collide → finalize) so the control flow ports cleanly.
//!
//! References: Macklin et al., "XPBD: Position-Based Simulation of Compliant
//! Constrained Dynamics" (2016). Compliance `alpha` relates to stiffness as
//! `alpha = 1/k`; `alpha == 0` is an infinitely stiff (inextensible) constraint.

use crate::bending::{self, Hinge};
use crate::contact::Collider;
use crate::state::SimState;
use glam::Vec3;

/// Solver tunables. Defaults chosen to be stable for cloth; physical fabric
/// defaults (young's modulus → compliance mapping, fabric presets) come later in
/// engine-api per docs/CONFIG.md.
#[derive(Clone, Copy, Debug)]
pub struct SolverParams {
    /// Full step size (seconds). Reference default dt = 1e-3 (docs/CONFIG.md).
    pub dt: f32,
    /// Substeps per step. XPBD accuracy scales with substeps far more cheaply
    /// than with constraint iterations (Macklin 2019, "small steps").
    pub substeps: u32,
    /// Constraint (Gauss-Seidel) iterations per substep.
    pub iterations: u32,
    pub gravity: Vec3,
    /// Stretch compliance (1/stiffness). 0.0 = inextensible.
    pub stretch_compliance: f32,
    /// Dihedral bending compliance (1/stiffness) for the discrete-shell hinges.
    /// `INFINITY` (the default) = zero bending stiffness = disabled; a small value
    /// = stiff bending (believable drape). 0.0 = perfectly rigid hinges.
    pub bend_compliance: f32,
    /// Aerodynamic air density (kg/m²). 0 (default) = aero drag OFF. The reference's
    /// anisotropic air damper (PORTING_energy_models.md): per triangle, drag opposes
    /// broadside (normal) motion fully and tangential motion by `air_friction`.
    pub air_density: f32,
    /// Tangential fraction of the aero drag (0..1); the normal drag is full. 0.2 ref.
    pub air_friction: f32,
    /// Ambient wind velocity; aero drag acts on `v − wind`.
    pub wind: Vec3,
    /// Velocity damping per step in [0,1). 0 = no damping.
    pub damping: f32,
    /// Ground plane height on Y; vertices are clamped to `y >= floor_y`.
    pub floor_y: f32,
    /// Coulomb-ish tangential friction at the floor in [0,1]. 1 = full stick.
    pub floor_friction: f32,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            dt: 1.0e-3,
            substeps: 8,
            iterations: 4,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            stretch_compliance: 0.0,
            bend_compliance: f32::INFINITY, // bending disabled by default
            air_density: 0.0,               // aero drag disabled by default
            air_friction: 0.2,
            wind: Vec3::ZERO,
            damping: 0.0,
            floor_y: f32::NEG_INFINITY, // no floor unless set
            floor_friction: 0.0,
        }
    }
}

/// Which phase of a step we are in — mirrors the reference Newton step stages so
/// logs read the same across the CPU tier and the future GPU implicit tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Predict,
    Constrain,
    Collide,
    Finalize,
}

pub struct Solver {
    pub state: SimState,
    pub params: SolverParams,
    /// Per-stretch-constraint Lagrange multiplier scratch (reset each substep).
    lambda: Vec<f32>,
    /// Discrete-shell bending hinges (interior edges), with rest dihedral angles.
    hinges: Vec<Hinge>,
    /// Per-hinge Lagrange multiplier scratch (reset each substep).
    bend_lambda: Vec<f32>,
    /// Analytic colliders (spheres / half-spaces) the cloth is projected out of.
    colliders: Vec<Collider>,
}

impl Solver {
    pub fn new(state: SimState, params: SolverParams) -> Self {
        let lambda = vec![0.0f32; state.stretch.len()];
        let hinges = bending::build_hinges(&state.tris, &state.positions);
        let bend_lambda = vec![0.0f32; hinges.len()];
        Self {
            state,
            params,
            lambda,
            hinges,
            bend_lambda,
            colliders: Vec::new(),
        }
    }

    /// Set the analytic colliders (spheres / half-spaces) the cloth must not
    /// penetrate. Projected out position-wise in the collide phase each substep.
    pub fn set_colliders(&mut self, colliders: Vec<Collider>) {
        self.colliders = colliders;
    }

    /// Advance one full step (`dt`), running `substeps` semi-implicit XPBD
    /// substeps. Deterministic: fixed iteration order, no atomics, no threads.
    #[tracing::instrument(level = "debug", skip(self), fields(dt = self.params.dt, substeps = self.params.substeps))]
    pub fn step(&mut self) {
        let p = self.params;
        let h = p.dt / p.substeps.max(1) as f32;
        // f32 robustness guard (Phase 6/9): snapshot so a non-finite substep result
        // (NaN/Inf) can be rejected instead of corrupting the sim.
        let snapshot = self.state.positions.clone();
        for _ in 0..p.substeps.max(1) {
            self.predict(h);
            self.constrain(h);
            self.collide();
            self.finalize(h);
            self.apply_aero();
        }
        if self.state.positions.iter().any(|q| !q.is_finite()) {
            // Reject the whole step: restore the snapshot, kill momentum.
            self.state.positions.copy_from_slice(&snapshot);
            self.state.prev_positions.copy_from_slice(&snapshot);
            for v in self.state.velocities.iter_mut() {
                *v = Vec3::ZERO;
            }
        }
        self.state.time += p.dt as f64;
        tracing::debug!(
            time = self.state.time,
            ke = self.state.kinetic_energy(),
            "step complete"
        );
    }

    fn predict(&mut self, h: f32) {
        let g = self.params.gravity;
        let s = &mut self.state;
        for i in 0..s.positions.len() {
            s.prev_positions[i] = s.positions[i];
            if s.inv_mass[i] > 0.0 {
                s.velocities[i] += g * h;
                s.positions[i] += s.velocities[i] * h;
            }
        }
    }

    fn constrain(&mut self, h: f32) {
        for l in self.lambda.iter_mut() {
            *l = 0.0;
        }
        for l in self.bend_lambda.iter_mut() {
            *l = 0.0;
        }
        let alpha_tilde = self.params.stretch_compliance / (h * h);
        // Bending disabled when compliance is INFINITY (zero stiffness).
        let bend_enabled = self.params.bend_compliance.is_finite() && !self.hinges.is_empty();
        let bend_alpha = self.params.bend_compliance / (h * h);
        for _ in 0..self.params.iterations {
            self.solve_stretch(alpha_tilde);
            if bend_enabled {
                self.solve_bending(bend_alpha);
            }
        }
    }

    /// XPBD dihedral bending: for each hinge, project the scalar constraint
    /// `C = θ(x) − θ_rest` using the exact angle gradient (`bending::dihedral_grad`).
    /// Standard XPBD scalar-constraint update:
    /// `Δλ = (−C − α̃λ) / (Σ wᵢ‖gᵢ‖² + α̃)`, `xᵢ += wᵢ Δλ gᵢ`.
    fn solve_bending(&mut self, alpha_tilde: f32) {
        let s = &mut self.state;
        for (h_idx, hinge) in self.hinges.iter().enumerate() {
            let v = hinge.v.map(|i| i as usize);
            let p = [
                s.positions[v[0]],
                s.positions[v[1]],
                s.positions[v[2]],
                s.positions[v[3]],
            ];
            let g = bending::dihedral_grad_analytic(&p);
            let mut wsum = 0.0f32;
            for k in 0..4 {
                wsum += s.inv_mass[v[k]] * g[k].length_squared();
            }
            if wsum <= 1.0e-12 {
                continue;
            }
            let c_val = bending::dihedral_angle(&p) - hinge.rest;
            let denom = wsum + alpha_tilde;
            let dlambda = (-c_val - alpha_tilde * self.bend_lambda[h_idx]) / denom;
            self.bend_lambda[h_idx] += dlambda;
            for k in 0..4 {
                s.positions[v[k]] += g[k] * (s.inv_mass[v[k]] * dlambda);
            }
        }
    }

    fn solve_stretch(&mut self, alpha_tilde: f32) {
        let s = &mut self.state;
        for (c_idx, c) in s.stretch.iter().enumerate() {
            let a = c.a as usize;
            let b = c.b as usize;
            let wa = s.inv_mass[a];
            let wb = s.inv_mass[b];
            let w = wa + wb;
            if w == 0.0 {
                continue;
            }
            let delta = s.positions[a] - s.positions[b];
            let len = delta.length();
            if len < 1.0e-12 {
                continue;
            }
            let dir = delta / len;
            let c_val = len - c.rest_len;
            let denom = w + alpha_tilde;
            let dlambda = (-c_val - alpha_tilde * self.lambda[c_idx]) / denom;
            self.lambda[c_idx] += dlambda;
            let corr = dir * dlambda;
            s.positions[a] += corr * wa;
            s.positions[b] -= corr * wb;
        }
    }

    fn collide(&mut self) {
        let floor = self.params.floor_y;
        let friction = self.params.floor_friction.clamp(0.0, 1.0);
        // Colliders are Copy + few (1–4); clone to split the borrow from `state`.
        let colliders = self.colliders.clone();
        let s = &mut self.state;
        for i in 0..s.positions.len() {
            if s.inv_mass[i] <= 0.0 {
                continue;
            }
            // Floor half-space, special-cased for its tangential friction.
            if floor.is_finite() && s.positions[i].y < floor {
                let slip = s.positions[i] - s.prev_positions[i];
                s.positions[i].y = floor;
                let tangential = Vec3::new(slip.x, 0.0, slip.z);
                s.positions[i] -= tangential * friction;
            }
            // Analytic colliders: push a penetrating vertex (signed distance < 0)
            // back to the surface along the outward gradient (position projection).
            for c in &colliders {
                let (d, n) = c.distance(s.positions[i]);
                if d < 0.0 {
                    s.positions[i] -= n * d; // d<0 → move outward by |d|
                }
            }
        }
    }

    fn finalize(&mut self, h: f32) {
        let damp = 1.0 - self.params.damping.clamp(0.0, 1.0);
        let inv_h = 1.0 / h;
        let s = &mut self.state;
        for i in 0..s.positions.len() {
            if s.inv_mass[i] > 0.0 {
                s.velocities[i] = (s.positions[i] - s.prev_positions[i]) * inv_h * damp;
            } else {
                s.velocities[i] = Vec3::ZERO;
            }
        }
    }

    /// Anisotropic aerodynamic drag (the reference air damper): per triangle, the
    /// relative velocity `v − wind` is damped fully along the triangle normal and by
    /// `air_friction` along the tangent, scaled by triangle area × air density. Each
    /// triangle's drag impulse is lumped 1/3 to its vertices (semi-implicit velocity
    /// update). No-op when `air_density == 0` (default). This kills broadside swing
    /// while barely touching in-plane tension oscillations — what simple isotropic
    /// velocity damping cannot do.
    fn apply_aero(&mut self) {
        let rho = self.params.air_density;
        if rho <= 0.0 {
            return;
        }
        let af = self.params.air_friction.clamp(0.0, 1.0);
        let wind = self.params.wind;
        let s = &mut self.state;
        let mut dv = vec![Vec3::ZERO; s.positions.len()];
        for t in &s.tris {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let cross = (s.positions[b] - s.positions[a]).cross(s.positions[c] - s.positions[a]);
            let len = cross.length();
            if len < 1.0e-12 {
                continue;
            }
            let normal = cross / len;
            let area = 0.5 * len;
            let v_rel = (s.velocities[a] + s.velocities[b] + s.velocities[c]) / 3.0 - wind;
            let vn = v_rel.dot(normal);
            let vt = v_rel - normal * vn;
            // Damping "force" direction: full normal + af·tangential, opposing v_rel.
            let d = normal * vn + vt * af;
            // Per-vertex Δv = −(area·ρ/3)·d·inv_mass (lumped third of the triangle).
            let k = area * rho / 3.0;
            for &i in &[a, b, c] {
                if s.inv_mass[i] > 0.0 {
                    dv[i] -= d * (k * s.inv_mass[i]);
                }
            }
        }
        for (i, d) in dv.into_iter().enumerate() {
            if s.inv_mass[i] > 0.0 {
                s.velocities[i] += d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Mesh;
    use crate::state::SimState;
    use approx::assert_relative_eq;

    fn single_particle() -> SimState {
        let mesh = Mesh::new(vec![Vec3::ZERO], vec![]);
        SimState::from_mesh(&mesh, 1.0)
    }

    #[test]
    fn free_fall_matches_semi_implicit_euler() {
        // One substep, no constraints, no floor: after one step a particle at
        // rest must fall exactly dt^2 * g (v starts at 0). Analytic ground truth.
        let mut solver = Solver::new(
            single_particle(),
            SolverParams {
                dt: 0.01,
                substeps: 1,
                iterations: 0,
                gravity: Vec3::new(0.0, -10.0, 0.0),
                ..Default::default()
            },
        );
        solver.step();
        let expected_y = 0.01 * 0.01 * -10.0;
        assert_relative_eq!(solver.state.positions[0].y, expected_y, epsilon = 1e-6);
        assert_relative_eq!(solver.state.velocities[0].y, -10.0 * 0.01, epsilon = 1e-6);
    }

    #[test]
    fn floor_stops_the_fall() {
        let mut s = single_particle();
        s.positions[0].y = 0.05;
        let mut solver = Solver::new(
            s,
            SolverParams {
                dt: 1.0,
                substeps: 4,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                floor_y: 0.0,
                ..Default::default()
            },
        );
        for _ in 0..50 {
            solver.step();
        }
        assert!(solver.state.positions[0].y >= -1e-4, "fell through floor");
    }

    #[test]
    fn deterministic_across_runs() {
        // Same input → identical output (regression/determinism requirement).
        let run = || {
            let verts = vec![
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ];
            let mut st = SimState::from_mesh(&Mesh::new(verts, vec![[0, 1, 2], [2, 1, 3]]), 0.25);
            st.pin(0);
            let mut solver = Solver::new(st, SolverParams::default());
            for _ in 0..20 {
                solver.step();
            }
            solver.state.positions
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
    }

    #[test]
    fn xpbd_bending_relaxes_fold_toward_rest() {
        use crate::bending::dihedral_angle;
        // Two triangles sharing edge (v1,v2) along X, built FLAT (rest angle 0),
        // then folded 90° about the shared edge (isometric: edge lengths preserved,
        // so stretch is already satisfied and only bending acts).
        let flat = vec![
            Vec3::new(0.5, -1.0, 0.0), // wing A
            Vec3::new(0.0, 0.0, 0.0),  // shared 0
            Vec3::new(1.0, 0.0, 0.0),  // shared 1
            Vec3::new(0.5, 1.0, 0.0),  // wing B
        ];
        let tris = vec![[0u32, 1, 2], [1, 2, 3]];

        let run = |bend_compliance: f32| {
            let st = SimState::from_mesh(&Mesh::new(flat.clone(), tris.clone()), 1.0);
            let mut solver = Solver::new(
                st,
                SolverParams {
                    dt: 1.0 / 60.0,
                    substeps: 8,
                    iterations: 10,
                    gravity: Vec3::ZERO,
                    stretch_compliance: 0.0, // inextensible; the fold is isometric
                    bend_compliance,
                    ..Default::default()
                },
            );
            // Fold wing B 90° about the X axis (shared edge): (0.5,1,0) → (0.5,0,1).
            solver.state.positions[3] = Vec3::new(0.5, 0.0, 1.0);
            for _ in 0..60 {
                solver.step();
            }
            let p = [
                solver.state.positions[0],
                solver.state.positions[1],
                solver.state.positions[2],
                solver.state.positions[3],
            ];
            dihedral_angle(&p).abs()
        };

        // Bending ON (stiff): the ~90° (≈1.57 rad) fold relaxes well toward flat (0).
        let bent = run(1.0e-6);
        assert!(
            bent < 0.5,
            "bending should flatten the fold, got {bent} rad"
        );
        // Bending OFF (INFINITY): nothing resists → the fold persists near 90°.
        let free = run(f32::INFINITY);
        assert!(
            free > 1.3,
            "no bending → fold should persist, got {free} rad"
        );
    }

    #[test]
    fn nan_guard_rejects_bad_step() {
        // A NaN velocity propagates into positions during predict(); the robustness
        // guard must reject the step and restore the finite pre-step state.
        let verts = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let st = SimState::from_mesh(&Mesh::new(verts, vec![[0, 1, 2], [2, 1, 3]]), 0.05);
        let mut solver = Solver::new(st, SolverParams::default());
        solver.state.velocities[0] = Vec3::new(f32::NAN, 0.0, 0.0);
        solver.step();
        assert!(
            solver.state.positions.iter().all(|p| p.is_finite()),
            "guard must reject the NaN step and keep positions finite"
        );
        // Velocities are zeroed by the reject path.
        assert!(solver.state.velocities.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn aero_drag_damps_normal_more_than_tangential() {
        // A single triangle in the xz-plane (normal along y). A velocity with equal
        // normal (y) and tangential (x) components must be damped far more along the
        // normal (full drag) than along the tangent (air_friction·drag).
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let mut st = SimState::from_mesh(&Mesh::new(verts, vec![[0, 1, 2]]), 1.0);
        for v in st.velocities.iter_mut() {
            *v = Vec3::new(1.0, 1.0, 0.0); // x tangential, y normal
        }
        let params = SolverParams {
            dt: 1.0 / 60.0,
            substeps: 1,
            iterations: 0,
            gravity: Vec3::ZERO,
            air_density: 1.0,  // strong, for a clear one-step effect
            air_friction: 0.2, // tangential fraction
            ..Default::default()
        };
        let mut solver = Solver::new(st, params);
        solver.step();
        let v = solver.state.velocities[0];
        // Normal (y) clearly damped; tangential (x) barely; normal damped much more.
        assert!(v.y < 0.9, "normal velocity should be damped, got {}", v.y);
        assert!(v.x > 0.95, "tangential barely damped, got {}", v.x);
        assert!(v.y < v.x, "normal must be damped more than tangential");
        // Aero OFF (air_density 0) leaves velocity untouched.
        let mut st2 = SimState::from_mesh(
            &Mesh::new(
                vec![
                    Vec3::ZERO,
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                ],
                vec![[0, 1, 2]],
            ),
            1.0,
        );
        for v in st2.velocities.iter_mut() {
            *v = Vec3::new(1.0, 1.0, 0.0);
        }
        let mut off = Solver::new(
            st2,
            SolverParams {
                substeps: 1,
                iterations: 0,
                gravity: Vec3::ZERO,
                ..Default::default()
            },
        );
        off.step();
        assert!(
            (off.state.velocities[0].y - 1.0).abs() < 1e-4,
            "aero off must not damp"
        );
    }

    #[test]
    fn sphere_collider_projects_cloth_out() {
        use crate::contact::Collider;
        // A flat cloth initialized INSIDE a unit sphere: every vertex starts
        // penetrating; the collider must project them all back to (or outside) the
        // surface. Directly exercises the projection, independent of settling.
        let n = 10;
        let mut verts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                verts.push(Vec3::new(
                    (i as f32 / (n - 1) as f32) * 1.6 - 0.8, // x ∈ [-0.8, 0.8]
                    0.3,
                    (j as f32 / (n - 1) as f32) * 1.6 - 0.8, // z ∈ [-0.8, 0.8]
                ));
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
        let sphere = Collider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let st = SimState::from_mesh(&Mesh::new(verts, tris), 0.02);
        // Sanity: the cloth really starts inside the sphere.
        let start_min = st
            .positions
            .iter()
            .map(|p| sphere.distance(*p).0)
            .fold(f32::MAX, f32::min);
        assert!(start_min < -0.1, "cloth should start inside: {start_min}");

        let mut solver = Solver::new(
            st,
            SolverParams {
                gravity: Vec3::ZERO, // isolate the projection from gravity
                ..Default::default()
            },
        );
        solver.set_colliders(vec![sphere]);
        for _ in 0..10 {
            solver.step();
        }
        let min_d = solver
            .state
            .positions
            .iter()
            .map(|p| sphere.distance(*p).0)
            .fold(f32::MAX, f32::min);
        assert!(
            min_d > -1.0e-3,
            "collider failed to project cloth out: min signed distance {min_d}"
        );
    }

    #[test]
    fn hanging_cloth_stays_bounded() {
        // A pinned cloth under gravity must settle, not explode.
        let mut verts = Vec::new();
        let n = 8;
        for j in 0..n {
            for i in 0..n {
                verts.push(Vec3::new(i as f32 * 0.1, 1.0, j as f32 * 0.1));
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
        let mut st = SimState::from_mesh(&Mesh::new(verts, tris), 0.01);
        st.pin(0);
        st.pin((n - 1) as usize);
        let mut solver = Solver::new(st, SolverParams::default());
        for _ in 0..200 {
            solver.step();
        }
        for pos in &solver.state.positions {
            assert!(pos.is_finite(), "cloth blew up: {pos:?}");
            assert!(pos.y > -100.0, "cloth fell unbounded");
        }
    }
}
