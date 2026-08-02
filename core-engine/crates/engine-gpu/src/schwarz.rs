// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU apply of the 2-level Galerkin additive Schwarz preconditioner
//! (`z = S₀r + w·C₁ᵀ S₁(C₁ r)`). The CPU (`engine_core::SchwarzGpuBuilder`) owns
//! the aggregation + dense Cholesky **factorization** (per solve); this runs only
//! the **apply** on-device via three WGSL kernels — `solve` (per-domain dense
//! triangular solve, reused for both levels), `restrict_r` (gather-sum to coarse),
//! `prolong` (weighted scatter-add). Domains are disjoint, so the apply is
//! atomic-free (see docs/DEVELOPING.md §6).
//!
//! The CPU 2-level Schwarz beats block-Jacobi ~6–9% on cloth; this is the GPU
//! port. It is verified correct three ways (apply == CPU apply to <1e-3; the
//! spliced PCG solves a known system; GPU-Schwarz physics == GPU-block physics),
//! but **measured SLOWER on GPU** — 491 vs 354 ms/step on the 900-vert cloth
//! (`mguso bench`). Why: block-Jacobi's apply is ONE massively-parallel dispatch
//! over all 900 vertices, whereas the Schwarz apply is 4 dispatches whose
//! per-domain dense triangular solve is *serial* with only ~56 active threads
//! (one per domain). On GPU each CG iteration is already cheap, so the heavier
//! apply outweighs the iteration-count savings that made it win on CPU. So
//! block-Jacobi stays the PCG default; this is opt-in (`--precond schwarz-ml`).
//! Future perf path (if revisited): parallelize the domain solve across the m
//! dimension (a workgroup per domain, not a thread), or the spec's Gram-form.

use crate::GpuContext;
use engine_core::{SchwarzGpuFactors, SchwarzGpuPlan};
use wgpu::util::DeviceExt;

const MAX_M: u32 = 96; // must match schwarz.wgsl; domain_size ≤ 32

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Dims {
    count: u32,
    weight: f32,
}

/// Total length of concatenated `m×m` factors implied by a domain-offset array
/// (`m = 3·|domain|`). Lets us size the factor buffers from the plan alone.
fn fact_len(off: &[u32]) -> usize {
    off.windows(2)
        .map(|w| {
            let m = 3 * (w[1] - w[0]) as usize;
            m * m
        })
        .sum()
}

fn max_m(off: &[u32]) -> u32 {
    off.windows(2).map(|w| 3 * (w[1] - w[0])).max().unwrap_or(0)
}

/// A reusable GPU Schwarz-apply operator bound to one topology. Upload the plan
/// once (`new`), re-upload factors each solve (`set_factors`), `apply` per PCG
/// iteration.
pub struct SchwarzApply {
    device: wgpu::Device,
    queue: wgpu::Queue,
    n: usize,
    n_coarse: u32,
    n_coarse_dom: u32,
    nv: u32,

    solve: wgpu::ComputePipeline,
    restrict: wgpu::ComputePipeline,
    prolong: wgpu::ComputePipeline,

    // plan (fixed)
    l0_vert: wgpu::Buffer,
    l0_off: wgpu::Buffer,
    l1_vert: wgpu::Buffer,
    l1_off: wgpu::Buffer,
    agg0: wgpu::Buffer,
    // factors (re-uploaded per solve)
    l0_fact: wgpu::Buffer,
    l0_fact_off: wgpu::Buffer,
    l0_m: wgpu::Buffer,
    l1_fact: wgpu::Buffer,
    l1_fact_off: wgpu::Buffer,
    l1_m: wgpu::Buffer,
    // vectors
    r: wgpu::Buffer,
    z: wgpu::Buffer,
    rc: wgpu::Buffer,
    ec: wgpu::Buffer,
    // uniforms
    u_nc: wgpu::Buffer,      // count = n_coarse (l0 solve + restrict)
    u_ncd: wgpu::Buffer,     // count = n_coarse_dom (l1 solve)
    u_prolong: wgpu::Buffer, // count = nv, weight
}

impl SchwarzApply {
    /// Whether a plan is representable on the GPU (domain m within `MAX_M`).
    pub fn supported(plan: &SchwarzGpuPlan) -> bool {
        max_m(&plan.l0_off) <= MAX_M && max_m(&plan.l1_off) <= MAX_M
    }

    pub fn new(ctx: &GpuContext, plan: &SchwarzGpuPlan) -> Self {
        assert!(
            Self::supported(plan),
            "Schwarz domain too large for GPU (m > {MAX_M}); use a smaller --schwarz domain"
        );
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("schwarz"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/schwarz.wgsl").into()),
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let store = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let store_src = store | wgpu::BufferUsages::COPY_SRC;
        let init_u32 = |label: &str, data: &[u32]| {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 4] } else { bytes },
                usage: store,
            })
        };
        let zeros = |label: &str, len: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) as u64) * 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let mkdims = |label: &str, count: u32, weight: f32| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&Dims { count, weight }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };

        let n = plan.n as usize;
        let nc_dof = 3 * plan.n_coarse as usize;

        Self {
            n,
            n_coarse: plan.n_coarse,
            n_coarse_dom: plan.n_coarse_dom,
            nv: plan.nv,
            solve: pipe("solve"),
            restrict: pipe("restrict_r"),
            prolong: pipe("prolong"),
            l0_vert: init_u32("l0_vert", &plan.l0_vert),
            l0_off: init_u32("l0_off", &plan.l0_off),
            l1_vert: init_u32("l1_vert", &plan.l1_vert),
            l1_off: init_u32("l1_off", &plan.l1_off),
            agg0: init_u32("agg0", &plan.agg0),
            l0_fact: zeros("l0_fact", fact_len(&plan.l0_off), store),
            l0_fact_off: zeros("l0_fact_off", plan.n_coarse as usize, store),
            l0_m: zeros("l0_m", plan.n_coarse as usize, store),
            l1_fact: zeros("l1_fact", fact_len(&plan.l1_off), store),
            l1_fact_off: zeros("l1_fact_off", plan.n_coarse_dom as usize, store),
            l1_m: zeros("l1_m", plan.n_coarse_dom as usize, store),
            r: zeros("schwarz.r", n, store),
            z: zeros("schwarz.z", n, store_src),
            rc: zeros("schwarz.rc", nc_dof, store),
            ec: zeros("schwarz.ec", nc_dof, store),
            u_nc: mkdims("u_nc", plan.n_coarse, 0.0),
            u_ncd: mkdims("u_ncd", plan.n_coarse_dom, 0.0),
            u_prolong: mkdims("u_prolong", plan.nv, plan.weight),
            device,
            queue,
        }
    }

    /// Re-upload the dense factors for the current matrix values (per solve).
    pub fn set_factors(&self, f: &SchwarzGpuFactors) {
        let w = |buf: &wgpu::Buffer, data: &[u8]| {
            if !data.is_empty() {
                self.queue.write_buffer(buf, 0, data);
            }
        };
        w(&self.l0_fact, bytemuck::cast_slice(&f.l0_fact));
        w(&self.l0_fact_off, bytemuck::cast_slice(&f.l0_fact_off));
        w(&self.l0_m, bytemuck::cast_slice(&f.l0_m));
        w(&self.l1_fact, bytemuck::cast_slice(&f.l1_fact));
        w(&self.l1_fact_off, bytemuck::cast_slice(&f.l1_fact_off));
        w(&self.l1_m, bytemuck::cast_slice(&f.l1_m));
    }

    fn bind(&self, pipe: &wgpu::ComputePipeline, e: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = e
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Build the four apply dispatches bound to EXTERNAL `rhs`→`out` buffers, for
    /// splicing into a caller's command batch (e.g. the PCG loop) fully resident —
    /// no readback. Internal `rc`/`ec` coarse scratch + plan/factor buffers are
    /// used automatically. The returned handle owns the bind groups and must
    /// outlive the dispatch.
    pub fn precond<'s>(&'s self, rhs: &wgpu::Buffer, out: &wgpu::Buffer) -> SchwarzPrecond<'s> {
        // Level-0 solve: out = S₀ rhs.
        let bg_l0 = self.bind(
            &self.solve,
            &[
                (0, &self.u_nc),
                (1, &self.l0_vert),
                (2, &self.l0_off),
                (3, &self.l0_fact),
                (4, &self.l0_fact_off),
                (5, &self.l0_m),
                (6, rhs),
                (7, out),
            ],
        );
        // Restrict: rc = C₁ rhs.
        let bg_res = self.bind(
            &self.restrict,
            &[
                (0, &self.u_nc),
                (1, &self.l0_vert),
                (2, &self.l0_off),
                (6, rhs),
                (7, &self.rc),
            ],
        );
        // Level-1 solve: ec = S₁ rc  (over n_coarse_dom coarse domains).
        let bg_l1 = self.bind(
            &self.solve,
            &[
                (0, &self.u_ncd),
                (1, &self.l1_vert),
                (2, &self.l1_off),
                (3, &self.l1_fact),
                (4, &self.l1_fact_off),
                (5, &self.l1_m),
                (6, &self.rc),
                (7, &self.ec),
            ],
        );
        // Prolong: out += w · C₁ᵀ ec.
        let bg_pro = self.bind(
            &self.prolong,
            &[
                (0, &self.u_prolong),
                (8, &self.agg0),
                (9, &self.ec),
                (7, out),
            ],
        );
        let g = |count: u32| count.max(1).div_ceil(64);
        SchwarzPrecond {
            bind_groups: [bg_l0, bg_res, bg_l1, bg_pro],
            pipes: [&self.solve, &self.restrict, &self.solve, &self.prolong],
            groups: [
                g(self.n_coarse),
                g(self.n_coarse),
                g(self.n_coarse_dom),
                g(self.nv),
            ],
        }
    }

    /// Apply `z = M⁻¹ r` standalone (writes `r`, runs the kernels, reads back `z`).
    /// Used by the equivalence test; the PCG path uses [`precond`](Self::precond)
    /// to stay fully resident.
    pub fn apply(&self, r: &[f32]) -> Vec<f32> {
        assert_eq!(r.len(), self.n, "Schwarz apply: rhs length mismatch");
        self.queue.write_buffer(&self.r, 0, bytemuck::cast_slice(r));
        let sp = self.precond(&self.r, &self.z);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (pipe, bg, groups) in sp.ops() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.readback(&self.z, self.n)
    }

    fn readback(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len as u64) * 4;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("schwarz.readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &out, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = out.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        out.unmap();
        v
    }
}

/// The four Schwarz-apply dispatches bound to a rhs→out pair, ready to splice
/// into a caller's command batch. Owns its bind groups; borrows the pipelines.
pub struct SchwarzPrecond<'a> {
    bind_groups: [wgpu::BindGroup; 4],
    pipes: [&'a wgpu::ComputePipeline; 4],
    groups: [u32; 4],
}

impl<'a> SchwarzPrecond<'a> {
    /// `(pipeline, bind_group, workgroups)` tuples in apply order
    /// (level-0 solve → restrict → level-1 solve → prolong).
    pub fn ops(&self) -> Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, u32)> {
        (0..4)
            .map(|i| (self.pipes[i], &self.bind_groups[i], self.groups[i]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{CsrMatrix, SchwarzGpuBuilder};

    fn ctx() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping GPU Schwarz test, no adapter: {e}");
                None
            }
        }
    }

    /// A 3-DOF SPD cloth-like Hessian: a 2D grid graph Laplacian, expanded to 3
    /// dofs/vertex with a diagonal shift so each vertex block is SPD.
    fn grid_hessian(w: usize) -> CsrMatrix {
        let nv = w * w;
        let mut trips: Vec<engine_core::csr::Triplet> = Vec::new();
        let idx = |i: usize, j: usize| j * w + i;
        for j in 0..w {
            for i in 0..w {
                let v = idx(i, j);
                let mut deg = 0.0f32;
                let mut nbrs = Vec::new();
                if i > 0 {
                    nbrs.push(idx(i - 1, j));
                }
                if i + 1 < w {
                    nbrs.push(idx(i + 1, j));
                }
                if j > 0 {
                    nbrs.push(idx(i, j - 1));
                }
                if j + 1 < w {
                    nbrs.push(idx(i, j + 1));
                }
                for &nb in &nbrs {
                    deg += 1.0;
                    for c in 0..3 {
                        trips.push(((3 * v + c) as u32, (3 * nb + c) as u32, -1.0));
                    }
                }
                for c in 0..3 {
                    // SPD diagonal with a little intra-vertex coupling.
                    trips.push(((3 * v + c) as u32, (3 * v + c) as u32, deg + 2.0));
                    for c2 in 0..3 {
                        if c2 != c {
                            trips.push(((3 * v + c) as u32, (3 * v + c2) as u32, 0.1));
                        }
                    }
                }
            }
        }
        CsrMatrix::from_triplets(3 * nv, &trips)
    }

    #[test]
    fn gpu_schwarz_apply_matches_cpu() {
        let Some(ctx) = ctx() else { return };
        let a = grid_hessian(8); // 64 verts, 192 dofs
        let builder = SchwarzGpuBuilder::new(&a, 16, 0.7);
        let plan = builder.plan();
        assert!(SchwarzApply::supported(&plan));

        let op = SchwarzApply::new(&ctx, &plan);
        op.set_factors(&builder.factors(&a));

        // A representative residual.
        let r: Vec<f32> = (0..a.n).map(|i| ((i as f32) * 0.37).sin() + 0.5).collect();
        let gpu = op.apply(&r);
        let cpu = builder.apply_cpu(&a, &r);

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (g, c) in gpu.iter().zip(&cpu) {
            max_abs = max_abs.max((g - c).abs());
            max_rel = max_rel.max((g - c).abs() / (c.abs() + 1e-3));
        }
        assert!(
            max_abs < 1e-3,
            "GPU Schwarz apply vs CPU: max_abs={max_abs} max_rel={max_rel}"
        );
    }

    // The full GPU PCG with the Schwarz preconditioner spliced into the resident
    // loop must solve a known SPD system — proves the r→z splicing is correct.
    #[test]
    fn gpu_schwarz_pcg_solves_known_system() {
        use crate::pcg::{GpuPrecond, Pcg};
        let Some(ctx) = ctx() else { return };
        let a = grid_hessian(10); // 100 verts, 300 dofs
        let x_star: Vec<f32> = (0..a.n).map(|i| ((i as f32) * 0.11).cos() + 1.0).collect();
        let b = a.spmv_cpu(&x_star);

        let pcg = Pcg::with_precond(
            &ctx,
            &a,
            GpuPrecond::SchwarzMl {
                domain: 16,
                weight: 0.7,
            },
        );
        let res = pcg.solve(&b, 1e-6, 2000);
        assert!(res.converged, "Schwarz-PCG did not converge: {res:?}");
        let max_err = res
            .x
            .iter()
            .zip(&x_star)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-2,
            "Schwarz-PCG solution error {max_err} (iters={})",
            res.iters
        );
    }
}
