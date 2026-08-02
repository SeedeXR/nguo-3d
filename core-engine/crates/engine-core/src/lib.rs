// SPDX-License-Identifier: GPL-2.0-or-later
//! engine-core — backend-agnostic solver core for ppf-mguso-solver.
//!
//! Owns the simulation state, the step state machine, and the CPU physics for
//! both tiers: the XPBD preview tier ([`Solver`]) and the implicit
//! backward-Euler tier ([`implicit::ImplicitSolver`]). `engine-gpu` accelerates
//! the implicit tier through the same [`SimState`] seam and is asserted equal to
//! this crate by its oracle tests. Start at `docs/DEVELOPING.md`.

pub mod alloc_stats;
pub mod assembler;
pub mod bending;
pub mod bending_assembler;
pub mod broadphase;
pub mod contact;
pub mod csr;
pub mod error;
pub mod gather3;
pub mod gather4;
pub mod geometry;
pub mod implicit;
pub mod linsolve;
pub mod material;
pub mod membrane;
pub mod membrane_assembler;
pub mod metrics;
pub mod rod_bend;
pub mod shell_assembler;
pub mod shell_model;
pub mod solver;
pub mod state;
pub mod stitch;
pub mod strain_limit;
pub mod sym_eig12;
pub mod tet;
pub mod tet_assembler;
pub mod unified_assembler;

pub use assembler::SpringAssembler;
pub use csr::CsrMatrix;
pub use error::{CoreError, Result};
pub use geometry::Mesh;
pub use implicit::ImplicitSolver;
pub use linsolve::{
    cg_cpu, cg_cpu_precond, solve_cg_cpu, LinearSolver, LinearStats, Precond, SchwarzCpu,
    SchwarzGpuBuilder, SchwarzGpuFactors, SchwarzGpuPlan, SchwarzMlCpu,
};
pub use metrics::{check_invariants, Invariant, InvariantReport, Metrics, StepMetrics};
pub use solver::{Phase, Solver, SolverParams};
pub use state::{SimState, StretchConstraint};
