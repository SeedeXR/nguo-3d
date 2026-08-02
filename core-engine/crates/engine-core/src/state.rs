// SPDX-License-Identifier: GPL-2.0-or-later
//! Simulation state: SoA vertex buffers + stretch constraints. This is the data
//! the step loop mutates. Kept backend-agnostic so the same state can later be
//! uploaded to GPU buffers (engine-gpu) for the implicit solver milestone.

use crate::geometry::Mesh;
use glam::Vec3;

/// A distance (stretch) constraint between two vertices, XPBD-style.
#[derive(Clone, Copy, Debug)]
pub struct StretchConstraint {
    pub a: u32,
    pub b: u32,
    pub rest_len: f32,
}

/// Mutable simulation state. `inv_mass == 0.0` marks a pinned (kinematic) vertex.
#[derive(Clone, Debug, Default)]
pub struct SimState {
    pub positions: Vec<Vec3>,
    /// Position at the start of the current step (for velocity update / CCD ref).
    pub prev_positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub inv_mass: Vec<f32>,
    pub tris: Vec<[u32; 3]>,
    pub stretch: Vec<StretchConstraint>,
    /// Optional per-vertex material-space (UV) coordinates, parallel to
    /// `positions` (empty = none). Used to classify stretch edges as warp (U) vs
    /// weft (V) for anisotropic strain limiting; carried from the loaders' UVs.
    pub uvs: Vec<[f32; 2]>,
    /// Simulated time in seconds (f64 — matches the reference's time accumulator,
    /// see docs/CONFIG.md: ParamSet.time is f64 while spatial math is f32).
    pub time: f64,
}

impl SimState {
    /// Build state from a mesh with a uniform per-vertex mass. Stretch
    /// constraints are seeded from the mesh edges at their initial length.
    pub fn from_mesh(mesh: &Mesh, vertex_mass: f32) -> Self {
        let n = mesh.vertex_count();
        let inv = if vertex_mass > 0.0 {
            1.0 / vertex_mass
        } else {
            0.0
        };
        let stretch = mesh
            .edges
            .iter()
            .map(|&[a, b]| StretchConstraint {
                a,
                b,
                rest_len: mesh.positions[a as usize].distance(mesh.positions[b as usize]),
            })
            .collect();
        Self {
            positions: mesh.positions.clone(),
            prev_positions: mesh.positions.clone(),
            velocities: vec![Vec3::ZERO; n],
            inv_mass: vec![inv; n],
            tris: mesh.tris.clone(),
            stretch,
            uvs: Vec::new(),
            time: 0.0,
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Pin a vertex in place (make it kinematic). No-op if out of range.
    pub fn pin(&mut self, index: usize) {
        if let Some(m) = self.inv_mass.get_mut(index) {
            *m = 0.0;
        }
    }

    pub fn is_pinned(&self, index: usize) -> bool {
        self.inv_mass.get(index).copied().unwrap_or(0.0) == 0.0
    }

    /// Total kinetic + gravitational energy — a cheap invariant probe for tests
    /// (a stable drape must not gain energy without bound).
    pub fn kinetic_energy(&self) -> f32 {
        self.velocities
            .iter()
            .zip(&self.inv_mass)
            .filter(|(_, &im)| im > 0.0)
            .map(|(v, &im)| 0.5 * (1.0 / im) * v.length_squared())
            .sum()
    }
}
