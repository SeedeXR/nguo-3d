// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU membrane cloth step: assembly on the GPU ([`GpuUnified`], inertia+springs+
//! membrane → the union CSR) fed into the GPU PCG ([`GpuLinearSolver`]). The
//! backward-Euler Newton loop mirrors `engine_core::ImplicitSolver::step` for the
//! contact-free case (no colliders / self-collision → the CCD line search is
//! trivially `t = 1`). Verified to match the CPU `ImplicitSolver` over many steps.
//!
//! Caveat (perf, not correctness): the assembled `(values, grad)` are read back to
//! the host and re-uploaded by `GpuLinearSolver` each Newton iteration — a true
//! zero-readback resident loop (PCG binding `GpuUnified`'s values buffer directly)
//! is the remaining optimization. Correctness of the GPU assemble→solve→update is
//! what this verifies.

use crate::{GpuContext, GpuLinearSolver, GpuUnified};
use engine_core::implicit::ImplicitParams;
use engine_core::membrane::MembraneElement;
use engine_core::unified_assembler::UnifiedAssembler;
use engine_core::{CsrMatrix, LinearSolver, SimState};
use glam::Vec3;

/// A membrane cloth solver whose Newton system is assembled AND solved on the GPU.
pub struct GpuUnifiedSolver<'a> {
    pub state: SimState,
    params: ImplicitParams,
    unified: GpuUnified,
    // Union CSR sparsity template (values filled per Newton iter by `unified`).
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    ndof: usize,
    linear: GpuLinearSolver<'a>,
}

impl<'a> GpuUnifiedSolver<'a> {
    /// Build from a cloth state + its membrane elements + params. Intended for the
    /// contact-free membrane case (colliders/self-collision are NOT applied here).
    pub fn new(
        ctx: &'a GpuContext,
        state: SimState,
        elements: &[MembraneElement],
        params: ImplicitParams,
    ) -> Self {
        let asm = UnifiedAssembler::new(&state, elements);
        let unified = GpuUnified::new(
            ctx,
            &asm,
            params.stiffness,
            params.membrane_warp,
            params.membrane_weft,
            params.membrane_shear,
        );
        Self {
            row_ptr: asm.structure.row_ptr.clone(),
            col_idx: asm.structure.col_idx.clone(),
            ndof: asm.ndof,
            state,
            params,
            unified,
            linear: GpuLinearSolver::new(ctx),
        }
    }

    /// One backward-Euler step (GPU assemble + GPU solve + update).
    pub fn step(&mut self) {
        let n = self.state.vertex_count();
        let h = self.params.dt;
        let inv_h2 = 1.0 / (h * h);
        let g = self.params.gravity;

        let x0: Vec<Vec3> = self.state.positions.clone();
        let x_tilde: Vec<Vec3> = (0..n)
            .map(|i| {
                if self.state.inv_mass[i] > 0.0 {
                    x0[i] + self.state.velocities[i] * h + g * (h * h)
                } else {
                    x0[i]
                }
            })
            .collect();

        let mut x = x0.clone();
        for _ in 0..self.params.newton_iters.max(1) {
            let (values, grad) = self.unified.assemble(&x, &x_tilde, inv_h2);
            let a = CsrMatrix {
                n: self.ndof,
                row_ptr: self.row_ptr.clone(),
                col_idx: self.col_idx.clone(),
                values,
            };
            let rhs: Vec<f32> = grad.iter().map(|v| -v).collect();
            let dxflat = self
                .linear
                .solve(&a, &rhs, self.params.cg_tol, self.params.cg_max_iter);
            let mut max_step = 0.0f32;
            for i in 0..n {
                // Contact-free membrane: CCD is trivially t = 1.
                let d = Vec3::new(dxflat[3 * i], dxflat[3 * i + 1], dxflat[3 * i + 2]);
                x[i] += d;
                max_step = max_step.max(d.length());
            }
            if max_step < 1.0e-7 {
                break;
            }
        }

        let inv_h = 1.0 / h;
        for i in 0..n {
            self.state.velocities[i] = if self.state.inv_mass[i] > 0.0 {
                (x[i] - x0[i]) * inv_h
            } else {
                Vec3::ZERO
            };
            self.state.prev_positions[i] = x0[i];
            self.state.positions[i] = x[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::linsolve::CgCpu;
    use engine_core::membrane::build_membrane;
    use engine_core::{ImplicitSolver, Mesh};

    fn ctx() -> Option<GpuContext> {
        GpuContext::new().ok()
    }

    fn cloth() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let nn = 5u32;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                verts.push(Vec3::new(i as f32 * 0.1, 1.0, j as f32 * 0.1));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nn - 1 {
            for i in 0..nn - 1 {
                let v = j * nn + i;
                tris.push([v, v + 1, v + nn]);
                tris.push([v + nn, v + 1, v + nn + 1]);
            }
        }
        (verts, tris)
    }

    #[test]
    fn gpu_unified_step_matches_cpu() {
        let Some(ctx) = ctx() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (verts, tris) = cloth();
        let els = build_membrane(&tris, &verts);
        let params = ImplicitParams {
            dt: 1.0 / 120.0,
            stiffness: 500.0,
            membrane_warp: 40.0,
            membrane_weft: 30.0,
            membrane_shear: 10.0,
            cg_tol: 1.0e-7,
            ..Default::default()
        };
        let mkstate = || {
            let mut st = SimState::from_mesh(&Mesh::new(verts.clone(), tris.clone()), 0.05);
            st.pin(0);
            st.pin(4);
            st
        };

        let mut cpu = ImplicitSolver::new(mkstate(), params);
        let mut gpu = GpuUnifiedSolver::new(&ctx, mkstate(), &els, params);
        for _ in 0..20 {
            cpu.step(&CgCpu::new());
            gpu.step();
        }
        let max_err = cpu
            .state
            .positions
            .iter()
            .zip(&gpu.state.positions)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 2.0e-3,
            "GPU-resident vs CPU membrane step drift {max_err}"
        );
    }

    #[test]
    #[ignore = "perf timing, run explicitly"]
    fn bench_unified_step() {
        let Some(ctx) = ctx() else { return };
        let nn = 24u32;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                verts.push(Vec3::new(i as f32 * 0.03, 1.0, j as f32 * 0.03));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nn - 1 {
            for i in 0..nn - 1 {
                let v = j * nn + i;
                tris.push([v, v + 1, v + nn]);
                tris.push([v + nn, v + 1, v + nn + 1]);
            }
        }
        let els = build_membrane(&tris, &verts);
        let params = ImplicitParams {
            dt: 1.0 / 120.0,
            stiffness: 500.0,
            membrane_warp: 40.0,
            membrane_weft: 30.0,
            membrane_shear: 10.0,
            cg_tol: 1.0e-5,
            ..Default::default()
        };
        let mkstate = || {
            let mut st = SimState::from_mesh(&Mesh::new(verts.clone(), tris.clone()), 0.05);
            st.pin(0);
            st.pin((nn - 1) as usize);
            st
        };
        let n = (nn * nn) as usize;
        // GPU-resident
        let mut gpu = GpuUnifiedSolver::new(&ctx, mkstate(), &els, params);
        for _ in 0..5 {
            gpu.step();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..40 {
            gpu.step();
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / 40.0;
        // CPU assemble + CPU solve
        let mut cpu = ImplicitSolver::new(mkstate(), params);
        let lin = CgCpu::new();
        for _ in 0..5 {
            cpu.step(&lin);
        }
        let t1 = std::time::Instant::now();
        for _ in 0..40 {
            cpu.step(&lin);
        }
        let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / 40.0;
        // CPU assemble + GPU solve (existing path)
        let mut cpu2 = ImplicitSolver::new(mkstate(), params);
        let glin = crate::GpuLinearSolver::new(&ctx);
        for _ in 0..5 {
            cpu2.step(&glin);
        }
        let t2 = std::time::Instant::now();
        for _ in 0..40 {
            cpu2.step(&glin);
        }
        let cpugpu_ms = t2.elapsed().as_secs_f64() * 1000.0 / 40.0;
        eprintln!("BENCH n={n}: gpu-resident={gpu_ms:.2} ms/step  cpu-assemble+cpu-solve={cpu_ms:.2}  cpu-assemble+gpu-solve={cpugpu_ms:.2}");
    }
}
