// SPDX-License-Identifier: GPL-2.0-or-later
//! Solver parameters — the authoring surface, mirroring the reference
//! ppf-contact-solver config (see docs/CONFIG.md for the source-of-truth table
//! with file:line citations). Defaults here are taken verbatim from the
//! reference `app_param()` / `object_param()` so a ported scene reproduces the
//! same behavior.
//!
//! Note on tiers: the current CPU solver (engine-core XPBD tier) consumes a
//! subset of these; contact-barrier / PCG / CCD fields are carried now so scene
//! files are forward-compatible with the implicit-barrier GPU milestone. Fields
//! not yet consumed are marked `// implicit-tier`.

use serde::{Deserialize, Serialize};

/// Constitutive model id — order matches the reference `MODEL_NAMES`
/// (elastic_model.rs:23): arap=0, stvk=1, baraff-witkin=2, snhk=3, pdrd=4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Model {
    Arap,
    Stvk,
    #[default]
    BaraffWitkin,
    Snhk,
    Pdrd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Barrier {
    #[default]
    Cubic,
    Quad,
    Log,
}

/// App-level solver parameters. Defaults from `params.rs` app_param()
/// (docs/CONFIG.md). All spatial quantities are f32 on device; `fps` is f64 to
/// match the reference time accumulator.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct SolverConfig {
    pub dt: f32,           // 1e-3
    pub fps: f64,          // 60.0
    pub frames: i32,       // 300
    pub gravity: [f32; 3], // [0,-9.8,0]
    pub wind: [f32; 3],    // [0,0,0]
    pub air_friction: f32, // 0.2
    pub air_density: f32,  // 1e-3

    // Newton / step control (implicit-tier)
    pub min_newton_steps: u32, // 0
    pub target_toi: f32,       // 0.25
    pub max_dx: f32,           // 1.0

    // Linear solver PCG (implicit-tier)
    pub cg_max_iter: u32, // 10000
    pub cg_tol: f32,      // 1e-3

    // Contact barrier (implicit-tier)
    pub disable_contact: bool, // false
    pub barrier: Barrier,      // cubic

    // Friction (implicit-tier). `friction_eps` is the reference's eps_v.
    pub friction_eps: f32, // 1e-5

    // CCD / ACCD (implicit-tier)
    pub ccd_reduction: f32,     // 0.01
    pub ccd_max_iter: u32,      // 4096
    pub line_search_max_t: f32, // 1.25

    // XPBD preview-tier knobs (engine-core). Not present in the reference; used
    // by the current CPU solver until the implicit port lands.
    pub substeps: u32,   // 8
    pub iterations: u32, // 4
    /// Enable vertex–triangle self-collision (implicit solver).
    pub self_collision: bool, // false
    /// Per-step velocity damping in [0,1). 0 = none (default, keeps goldens
    /// energy-conserving). A small value (~0.02) lets a drape settle instead of
    /// swinging forever; superseded by real aerodynamic drag later.
    pub damping: f32, // 0.0
    pub floor_y: Option<f32>, // None = no ground plane
    /// XPBD preview-tier dihedral bending compliance (1/stiffness). `None` (default)
    /// = bending OFF (preserves existing preview goldens); `Some(c)` enables it.
    /// The implicit tier gets bending from `material.bend` regardless.
    pub bend_compliance: Option<f32>, // None
    /// Enable the XPBD preview-tier anisotropic aero drag (uses `air_density`/
    /// `air_friction`/`wind`). `false` (default) keeps aero OFF so existing preview
    /// goldens are unchanged; the golden parity driver turns it on.
    pub preview_aero: bool, // false
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            dt: 1.0e-3,
            fps: 60.0,
            frames: 300,
            gravity: [0.0, -9.8, 0.0],
            wind: [0.0, 0.0, 0.0],
            air_friction: 0.2,
            air_density: 1.0e-3,
            min_newton_steps: 0,
            target_toi: 0.25,
            max_dx: 1.0,
            cg_max_iter: 10_000,
            cg_tol: 1.0e-3,
            disable_contact: false,
            barrier: Barrier::Cubic,
            friction_eps: 1.0e-5,
            ccd_reduction: 0.01,
            ccd_max_iter: 4096,
            line_search_max_t: 1.25,
            substeps: 8,
            iterations: 4,
            self_collision: false,
            damping: 0.0,
            floor_y: None,
            bend_compliance: None,
            preview_aero: false,
        }
    }
}
