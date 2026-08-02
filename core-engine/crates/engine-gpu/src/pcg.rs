// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU matrix-free* preconditioned conjugate gradient (Jacobi precond).
//!
//! *matrix-free in the reference sense: no external sparse library — the CSR is
//! applied by a custom WGSL SpMV kernel. This is the linear-solver backbone the
//! implicit-barrier Newton step reuses, and the phase where GPU performance is
//! won or lost. CPU-orchestrated, GPU-executed loop. Dot products are
//! reduced fully on-device and alpha/beta computed on the GPU (see
//! `shaders/pcg_reduce.wgsl`); the host reads back only the residual every 16
//! iterations for the early-out. The whole iteration body is also encoded as a
//! single command buffer (`dispatch_batch`) — barriers between compute passes
//! preserve the SpMV→dot→reduce→α→axpy→emul→dot→reduce dependency chain — so a
//! step costs ~2 submits, not ~11. Together these took Metal from ~103 to
//! ~53 ms/step at 144 verts (`mguso bench`), ~2×. Self-verified against a known
//! solution + CPU CG — no reference goldens required.
//!
//! Preconditioner: 3×3 per-vertex block-Jacobi for 3-DOF systems (`bmul`, block
//! inverses precomputed on the CPU and uploaded), scalar diagonal otherwise.
//!
//! Perf upgrades still open: a multigrid / additive-Schwarz preconditioner (the
//! reference default — captures inter-vertex coupling, the real iteration-count
//! win for cloth), subgroup reductions, and GPU-resident membrane/bending
//! assembly (currently CPU-assembled).

use crate::schwarz::SchwarzApply;
use crate::GpuContext;
use engine_core::{CsrMatrix, SchwarzGpuBuilder};
use wgpu::util::DeviceExt;

/// GPU PCG preconditioner choice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum GpuPrecond {
    /// 3×3 per-vertex block-Jacobi (default) / scalar diagonal fallback.
    #[default]
    Block,
    /// 2-level Galerkin additive Schwarz (`domain` verts/subdomain, coarse
    /// correction `weight`). Falls back to Block for non-3-DOF or oversized domains.
    SchwarzMl { domain: usize, weight: f32 },
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    n: u32,
    alpha: f32,
    beta: f32,
    _pad: f32,
}

/// Per-vertex 3×3 block-Jacobi inverses (row-major, `nv × 9` f32) from the CSR
/// diagonal blocks. Singular blocks fall back to the scalar diagonal inverse.
/// Requires a 3-DOF-per-vertex layout (`n % 3 == 0`).
fn block_inverses(a: &CsrMatrix) -> Vec<f32> {
    let nv = a.n / 3;
    let mut out = vec![0.0f32; nv * 9];
    for i in 0..nv {
        // Gather the 3×3 diagonal block (f64).
        let mut m = [[0.0f64; 3]; 3];
        for (r, mr) in m.iter_mut().enumerate() {
            let row = 3 * i + r;
            let (s, e) = (a.row_ptr[row] as usize, a.row_ptr[row + 1] as usize);
            for k in s..e {
                let c = a.col_idx[k] as usize;
                if (3 * i..3 * i + 3).contains(&c) {
                    mr[c - 3 * i] = a.values[k] as f64;
                }
            }
        }
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let o = i * 9;
        if det.abs() < 1e-20 {
            // Fallback: scalar diagonal inverse (diagonal 3×3).
            for c in 0..3 {
                let d = m[c][c];
                out[o + c * 3 + c] = if d != 0.0 { (1.0 / d) as f32 } else { 1.0 };
            }
        } else {
            let id = 1.0 / det;
            let inv = [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
            ];
            for (k, v) in inv.iter().enumerate() {
                out[o + k] = *v as f32;
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct PcgResult {
    pub x: Vec<f32>,
    pub iters: u32,
    pub residual: f64,
    pub converged: bool,
}

/// A reusable Jacobi-PCG solver bound to one CSR matrix (and its sparsity).
/// Build once, `solve` many times (e.g. across Newton iterations).
pub struct Pcg {
    device: wgpu::Device,
    queue: wgpu::Queue,
    n: usize,
    nnz: usize,
    npart: u32,
    wg64: u32,
    wg256: u32,
    wg_nv: u32,      // workgroups over n/3 vertices (block-Jacobi apply)
    use_block: bool, // 3×3 block-Jacobi (3-DOF layout) vs scalar diagonal

    emul: wgpu::ComputePipeline,
    bmul: wgpu::ComputePipeline,
    dot: wgpu::ComputePipeline,
    spmv: wgpu::ComputePipeline,
    // GPU-resident scalar machinery (removes per-iteration readback).
    reduce_final: wgpu::ComputePipeline,
    compute_alpha: wgpu::ComputePipeline,
    compute_beta: wgpu::ComputePipeline,
    axpy_s: wgpu::ComputePipeline,
    xpby_s: wgpu::ComputePipeline,

    // CSR (fixed for this matrix)
    row_ptr: wgpu::Buffer,
    col_idx: wgpu::Buffer,
    values: wgpu::Buffer,

    // vectors
    x: wgpu::Buffer,
    r: wgpu::Buffer,
    p: wgpu::Buffer,
    ap: wgpu::Buffer,
    z: wgpu::Buffer,
    inv_diag: wgpu::Buffer, // scalar Jacobi
    binv: wgpu::Buffer,     // block Jacobi (nv × 9)
    u_bm: wgpu::Buffer,     // bmul {nv}

    // uniforms
    u_emul: wgpu::Buffer,
    u_dot: wgpu::Buffer,
    u_spmv: wgpu::Buffer,
    // Slot-fixed uniforms (set once) so the PCG loop needs no per-dispatch writes
    // and can be encoded as a single command buffer per iteration.
    u_rf0: wgpu::Buffer,   // reduce_final → scalars[0] (rz, setup)
    u_rf1: wgpu::Buffer,   // → scalars[1] (pAp)
    u_rf2: wgpu::Buffer,   // → scalars[2] (rz_new)
    u_rf6: wgpu::Buffer,   // → scalars[6] (rr)
    u_axs_p: wgpu::Buffer, // axpy_s slot 3 (+alpha)
    u_axs_n: wgpu::Buffer, // axpy_s slot 4 (−alpha)
    u_xps: wgpu::Buffer,   // xpby_s slot 5 (beta)
    scalars: wgpu::Buffer,

    partials: wgpu::Buffer,
    staging: wgpu::Buffer,

    // 2-level Schwarz preconditioner (opt-in). `sbuilder` (CPU) owns the cached
    // topology + does the per-solve factorization; `schwarz` runs the apply on GPU.
    schwarz: Option<SchwarzApply>,
    sbuilder: Option<SchwarzGpuBuilder>,
}

impl Pcg {
    /// Build with the default block-Jacobi preconditioner.
    pub fn new(ctx: &GpuContext, a: &CsrMatrix) -> Self {
        Self::with_precond(ctx, a, GpuPrecond::Block)
    }

    /// Build with an explicit preconditioner. `SchwarzMl` falls back to block if
    /// the system is not 3-DOF or a domain is too large for the GPU kernel.
    pub fn with_precond(ctx: &GpuContext, a: &CsrMatrix, precond: GpuPrecond) -> Self {
        let mut pcg = Self::build_block(ctx, a);
        if let GpuPrecond::SchwarzMl { domain, weight } = precond {
            if a.n.is_multiple_of(3) {
                let builder = SchwarzGpuBuilder::new(a, domain, weight);
                let plan = builder.plan();
                if SchwarzApply::supported(&plan) {
                    let apply = SchwarzApply::new(ctx, &plan);
                    apply.set_factors(&builder.factors(a));
                    pcg.schwarz = Some(apply);
                    pcg.sbuilder = Some(builder);
                }
            }
        }
        pcg
    }

    fn build_block(ctx: &GpuContext, a: &CsrMatrix) -> Self {
        assert!(a.n > 0, "PCG: empty system");
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let n = a.n;
        let wg64 = (n as u32).div_ceil(64);
        let wg256 = (n as u32).div_ceil(256);
        let npart = wg256;
        let use_block = n.is_multiple_of(3);
        let nv = if use_block { n / 3 } else { 0 };
        let wg_nv = (nv.max(1) as u32).div_ceil(64);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pcg_ops"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pcg_ops.wgsl").into()),
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
        let reduce_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pcg_reduce"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pcg_reduce.wgsl").into()),
        });
        let rpipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &reduce_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let store = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let vec_zeros = vec![0.0f32; n];
        let mkvec = |label: &str| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&vec_zeros),
                usage: store,
            })
        };
        let mkuniform = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let inv_diag_cpu: Vec<f32> = a
            .diagonal()
            .iter()
            .map(|&d| if d != 0.0 { 1.0 / d } else { 1.0 })
            .collect();
        let binv_cpu: Vec<f32> = if use_block {
            block_inverses(a)
        } else {
            vec![0.0f32; 9] // dummy (unused)
        };

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pcg.staging"),
            size: (npart.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // CSR buffers are re-uploadable (COPY_DST) so a cached Pcg can be reused
        // across Newton solves of the same sparsity — see `reload`.
        let csr_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Self {
            n,
            nnz: a.nnz(),
            npart,
            wg64,
            wg256,
            wg_nv,
            use_block,
            emul: pipe("emul"),
            bmul: rpipe("bmul"),
            dot: pipe("dot"),
            spmv: pipe("spmv"),
            reduce_final: rpipe("reduce_final"),
            compute_alpha: rpipe("compute_alpha"),
            compute_beta: rpipe("compute_beta"),
            axpy_s: rpipe("axpy_s"),
            xpby_s: rpipe("xpby_s"),
            row_ptr: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("csr.row_ptr"),
                contents: bytemuck::cast_slice(&a.row_ptr),
                usage: csr_usage,
            }),
            col_idx: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("csr.col_idx"),
                contents: bytemuck::cast_slice(&a.col_idx),
                usage: csr_usage,
            }),
            values: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("csr.values"),
                contents: bytemuck::cast_slice(&a.values),
                usage: csr_usage,
            }),
            x: mkvec("pcg.x"),
            r: mkvec("pcg.r"),
            p: mkvec("pcg.p"),
            ap: mkvec("pcg.ap"),
            z: mkvec("pcg.z"),
            inv_diag: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pcg.inv_diag"),
                contents: bytemuck::cast_slice(&inv_diag_cpu),
                usage: store,
            }),
            binv: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pcg.binv"),
                contents: bytemuck::cast_slice(&binv_cpu),
                usage: store,
            }),
            u_bm: mkuniform("u.bm"),
            u_emul: mkuniform("u.emul"),
            u_dot: mkuniform("u.dot"),
            u_spmv: mkuniform("u.spmv"),
            u_rf0: mkuniform("u.rf0"),
            u_rf1: mkuniform("u.rf1"),
            u_rf2: mkuniform("u.rf2"),
            u_rf6: mkuniform("u.rf6"),
            u_axs_p: mkuniform("u.axs_p"),
            u_axs_n: mkuniform("u.axs_n"),
            u_xps: mkuniform("u.xps"),
            scalars: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pcg.scalars"),
                size: 8 * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            partials: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pcg.partials"),
                size: (npart.max(1) as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging,
            device,
            queue,
            schwarz: None,
            sbuilder: None,
        }
    }

    /// Whether this solver's buffers fit matrix `a` (same size + sparsity count),
    /// so it can be reused via [`reload`](Self::reload) instead of rebuilt.
    pub fn matches(&self, a: &CsrMatrix) -> bool {
        self.n == a.n && self.nnz == a.nnz()
    }

    /// Re-upload a new matrix of the SAME size/nnz into the existing buffers and
    /// refresh the Jacobi preconditioner — avoids recompiling shaders / realloc.
    pub fn reload(&self, a: &CsrMatrix) {
        debug_assert!(self.matches(a));
        self.queue
            .write_buffer(&self.row_ptr, 0, bytemuck::cast_slice(&a.row_ptr));
        self.queue
            .write_buffer(&self.col_idx, 0, bytemuck::cast_slice(&a.col_idx));
        self.queue
            .write_buffer(&self.values, 0, bytemuck::cast_slice(&a.values));
        if self.use_block {
            self.queue
                .write_buffer(&self.binv, 0, bytemuck::cast_slice(&block_inverses(a)));
        } else {
            let inv_diag: Vec<f32> = a
                .diagonal()
                .iter()
                .map(|&d| if d != 0.0 { 1.0 / d } else { 1.0 })
                .collect();
            self.queue
                .write_buffer(&self.inv_diag, 0, bytemuck::cast_slice(&inv_diag));
        }
        // Schwarz topology is sparsity-fixed (same nnz, guaranteed by `matches`);
        // only the dense factors change with the values → re-factorize + upload.
        if let (Some(apply), Some(builder)) = (&self.schwarz, &self.sbuilder) {
            apply.set_factors(&builder.factors(a));
        }
    }

    fn set_params(&self, buf: &wgpu::Buffer, alpha: f32, beta: f32) {
        let p = Params {
            n: self.n as u32,
            alpha,
            beta,
            _pad: 0.0,
        };
        self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
    }

    fn bind(
        &self,
        pipe: &wgpu::ComputePipeline,
        entries: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let e: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &e,
        })
    }

    fn dispatch(&self, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, groups: u32) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn set_u32x2(&self, buf: &wgpu::Buffer, a: u32, b: u32) {
        self.queue
            .write_buffer(buf, 0, bytemuck::cast_slice(&[a, b, 0u32, 0u32]));
    }

    /// Encode a list of (pipeline, bind-group, workgroups) into ONE command
    /// buffer — each op as its own compute pass so wgpu inserts the dependency
    /// barriers — and submit once. Collapses ~11 submits/iteration into 1.
    fn dispatch_batch(&self, ops: &[(&wgpu::ComputePipeline, &wgpu::BindGroup, u32)]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (pipe, bg, groups) in ops {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, *bg, &[]);
            pass.dispatch_workgroups(*groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Read one `scalars[slot]` value back to the host (the only periodic sync).
    fn read_scalar(&self, slot: u32) -> f32 {
        let off = (slot as u64) * 4;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.scalars, off, &self.staging, 0, 4);
        self.queue.submit(Some(enc.finish()));
        let slice = self.staging.slice(0..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v = bytemuck::cast_slice::<u8, f32>(&data)[0];
        drop(data);
        self.staging.unmap();
        v
    }

    /// Solve `A x = b` for x. `tol` is relative to ‖b‖. Fully GPU-resident:
    /// alpha/beta are computed on the device and the residual is read back only
    /// every `CHECK` iterations (each readback is a GPU sync — the old
    /// per-iteration readback dominated runtime, per `mguso bench`).
    pub fn solve(&self, b: &[f32], tol: f32, max_iter: u32) -> PcgResult {
        assert_eq!(b.len(), self.n, "PCG: rhs length mismatch");
        const CHECK: u32 = 16; // residual check / readback cadence
        let bnorm2 = b
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .max(1e-30);
        let threshold = (tol as f64) * (tol as f64) * bnorm2;

        // x = 0, r = b. Preset all slot-fixed uniforms once (constant per solve),
        // so the loop below issues NO per-dispatch writes and can be batched.
        let n = self.n as u32;
        self.queue
            .write_buffer(&self.x, 0, bytemuck::cast_slice(&vec![0.0f32; self.n]));
        self.queue.write_buffer(&self.r, 0, bytemuck::cast_slice(b));
        self.set_params(&self.u_emul, 0.0, 0.0);
        self.set_params(&self.u_dot, 0.0, 0.0);
        self.set_params(&self.u_spmv, 0.0, 0.0);
        self.set_u32x2(&self.u_rf0, self.npart, 0);
        self.set_u32x2(&self.u_rf1, self.npart, 1);
        self.set_u32x2(&self.u_rf2, self.npart, 2);
        self.set_u32x2(&self.u_rf6, self.npart, 6);
        self.set_u32x2(&self.u_axs_p, n, 3);
        self.set_u32x2(&self.u_axs_n, n, 4);
        self.set_u32x2(&self.u_xps, n, 5);
        self.set_u32x2(&self.u_bm, (self.n / 3).max(1) as u32, 0);

        // Preconditioner apply z = M⁻¹ r: 3×3 block-Jacobi (bmul) for 3-DOF
        // systems, else scalar Jacobi (emul).
        let (precond_pipe, bg_precond, precond_groups) = if self.use_block {
            (
                &self.bmul,
                self.bind(
                    &self.bmul,
                    &[
                        (13, &self.u_bm),
                        (14, &self.binv),
                        (15, &self.r),
                        (16, &self.z),
                    ],
                ),
                self.wg_nv,
            )
        } else {
            (
                &self.emul,
                self.bind(
                    &self.emul,
                    &[
                        (6, &self.u_emul),
                        (7, &self.r),
                        (8, &self.inv_diag),
                        (9, &self.z),
                    ],
                ),
                self.wg64,
            )
        };
        let bg_spmv = self.bind(
            &self.spmv,
            &[
                (14, &self.u_spmv),
                (15, &self.row_ptr),
                (16, &self.col_idx),
                (17, &self.values),
                (18, &self.p),
                (19, &self.ap),
            ],
        );
        let dotbg = |a: &wgpu::Buffer, b: &wgpu::Buffer| {
            self.bind(
                &self.dot,
                &[(10, &self.u_dot), (11, a), (12, b), (13, &self.partials)],
            )
        };
        let bg_dot_pap = dotbg(&self.p, &self.ap);
        let bg_dot_rz = dotbg(&self.r, &self.z);
        let bg_dot_rr = dotbg(&self.r, &self.r);
        let rfbg = |u: &wgpu::Buffer| {
            self.bind(
                &self.reduce_final,
                &[(0, u), (1, &self.partials), (2, &self.scalars)],
            )
        };
        let bg_rf0 = rfbg(&self.u_rf0);
        let bg_rf1 = rfbg(&self.u_rf1);
        let bg_rf2 = rfbg(&self.u_rf2);
        let bg_rf6 = rfbg(&self.u_rf6);
        let bg_alpha = self.bind(&self.compute_alpha, &[(3, &self.scalars)]);
        let bg_beta = self.bind(&self.compute_beta, &[(4, &self.scalars)]);
        let bg_axpy_x = self.bind(
            &self.axpy_s,
            &[
                (5, &self.u_axs_p),
                (6, &self.scalars),
                (7, &self.p),
                (8, &self.x),
            ],
        );
        let bg_axpy_r = self.bind(
            &self.axpy_s,
            &[
                (5, &self.u_axs_n),
                (6, &self.scalars),
                (7, &self.ap),
                (8, &self.r),
            ],
        );
        let bg_xpby = self.bind(
            &self.xpby_s,
            &[
                (9, &self.u_xps),
                (10, &self.scalars),
                (11, &self.z),
                (12, &self.p),
            ],
        );

        // Preconditioner apply z = M⁻¹ r as spliceable ops: the 4 Schwarz
        // dispatches (bound r→z, fully resident — no readback) when enabled, else
        // the single block-Jacobi / scalar dispatch.
        let schwarz_precond = self.schwarz.as_ref().map(|s| s.precond(&self.r, &self.z));
        let precond_ops: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, u32)> =
            match &schwarz_precond {
                Some(sp) => sp.ops(),
                None => vec![(precond_pipe, &bg_precond, precond_groups)],
            };

        // Setup: z = M⁻¹ r ; p = z ; rz = r·z → scalars[0]  (one submit)
        self.dispatch_batch(&precond_ops);
        self.copy(&self.z, &self.p);
        self.dispatch_batch(&[
            (&self.dot, &bg_dot_rz, self.wg256),
            (&self.reduce_final, &bg_rf0, 1),
        ]);

        // Whole iteration body as one command buffer (barriers between passes
        // preserve the SpMV→dot→reduce→α→axpy→precond→dot→reduce chain). Built once;
        // the precond is 1 op (block) or 4 (Schwarz), spliced at its position.
        let mut main_ops: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, u32)> = vec![
            (&self.spmv, &bg_spmv, self.wg64),
            (&self.dot, &bg_dot_pap, self.wg256),
            (&self.reduce_final, &bg_rf1, 1), // pAp → scalars[1]
            (&self.compute_alpha, &bg_alpha, 1),
            (&self.axpy_s, &bg_axpy_x, self.wg64), // x += α p
            (&self.axpy_s, &bg_axpy_r, self.wg64), // r += −α ap
        ];
        main_ops.extend(precond_ops.iter().copied()); // z = M⁻¹ r
        main_ops.push((&self.dot, &bg_dot_rz, self.wg256));
        main_ops.push((&self.reduce_final, &bg_rf2, 1)); // rz_new → scalars[2]

        let mut iters = 0u32;
        let mut residual = f64::INFINITY;
        let mut converged = false;

        while iters < max_iter {
            self.dispatch_batch(&main_ops);
            iters += 1;

            if iters.is_multiple_of(CHECK) || iters == max_iter {
                self.dispatch_batch(&[
                    (&self.dot, &bg_dot_rr, self.wg256),
                    (&self.reduce_final, &bg_rf6, 1), // rr → scalars[6]
                ]);
                let rr = self.read_scalar(6) as f64; // the only sync
                residual = rr.max(0.0).sqrt();
                if rr <= threshold {
                    converged = true;
                    break;
                }
            }
            // β = rz_new/rz ; roll rz ; p = z + β p  (one submit)
            self.dispatch_batch(&[
                (&self.compute_beta, &bg_beta, 1),
                (&self.xpby_s, &bg_xpby, self.wg64),
            ]);
        }

        PcgResult {
            x: self.readback(&self.x),
            iters,
            residual,
            converged,
        }
    }

    fn copy(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, dst, 0, (self.n as u64) * 4);
        self.queue.submit(Some(enc.finish()));
    }

    fn readback(&self, buf: &wgpu::Buffer) -> Vec<f32> {
        let bytes = (self.n as u64) * 4;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pcg.readback"),
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

    /// GPU SpMV `y = A x`, exposed for unit-testing against `CsrMatrix::spmv_cpu`.
    pub fn spmv_debug(&self, x: &[f32]) -> Vec<f32> {
        self.queue.write_buffer(&self.p, 0, bytemuck::cast_slice(x));
        let bg = self.bind(
            &self.spmv,
            &[
                (14, &self.u_spmv),
                (15, &self.row_ptr),
                (16, &self.col_idx),
                (17, &self.values),
                (18, &self.p),
                (19, &self.ap),
            ],
        );
        self.set_params(&self.u_spmv, 0.0, 0.0);
        self.dispatch(&self.spmv, &bg, self.wg64);
        self.readback(&self.ap)
    }
}

/// The GPU PCG exposed as an `engine_core::LinearSolver`, so the implicit
/// stepper can run its Newton solves on the GPU.
///
/// Caches a [`Pcg`] and reuses it whenever the sparsity is unchanged (the common
/// case across Newton iterations / steps), re-uploading only the matrix values —
/// this avoids recompiling the WGSL pipelines and reallocating buffers every
/// solve, which dominated the previous per-solve cost.
pub struct GpuLinearSolver<'a> {
    pub ctx: &'a GpuContext,
    precond: GpuPrecond,
    cache: std::cell::RefCell<Option<Pcg>>,
}

impl<'a> GpuLinearSolver<'a> {
    /// Block-Jacobi preconditioner (default).
    pub fn new(ctx: &'a GpuContext) -> Self {
        Self::with_precond(ctx, GpuPrecond::Block)
    }

    /// Choose the GPU PCG preconditioner (e.g. 2-level Schwarz).
    pub fn with_precond(ctx: &'a GpuContext, precond: GpuPrecond) -> Self {
        Self {
            ctx,
            precond,
            cache: std::cell::RefCell::new(None),
        }
    }
}

impl engine_core::LinearSolver for GpuLinearSolver<'_> {
    fn solve(&self, a: &CsrMatrix, b: &[f32], tol: f32, max_iter: u32) -> Vec<f32> {
        let mut cache = self.cache.borrow_mut();
        let reuse = cache.as_ref().is_some_and(|p| p.matches(a));
        if reuse {
            let pcg = cache.as_ref().unwrap();
            pcg.reload(a);
            pcg.solve(b, tol, max_iter).x
        } else {
            let pcg = Pcg::with_precond(self.ctx, a, self.precond);
            let x = pcg.solve(b, tol, max_iter).x;
            *cache = Some(pcg);
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::CsrMatrix;

    fn ctx() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping GPU PCG test, no adapter: {e}");
                None
            }
        }
    }

    #[test]
    fn gpu_spmv_matches_cpu() {
        let Some(ctx) = ctx() else { return };
        let a = CsrMatrix::laplacian_1d(64);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let solver = Pcg::new(&ctx, &a);
        let gpu = solver.spmv_debug(&x);
        let cpu = a.spmv_cpu(&x);
        for i in 0..64 {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4,
                "row {i}: {} vs {}",
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn pcg_solves_known_system() {
        let Some(ctx) = ctx() else { return };
        // Known x*, b = A x*, solve, assert x ≈ x*. Self-verifying oracle.
        let n = 128;
        let a = CsrMatrix::laplacian_1d(n);
        let x_star: Vec<f32> = (0..n)
            .map(|i| ((i as f32) / n as f32 * std::f32::consts::PI).sin() + 1.0)
            .collect();
        let b = a.spmv_cpu(&x_star);

        let solver = Pcg::new(&ctx, &a);
        let res = solver.solve(&b, 1e-6, 1000);
        assert!(res.converged, "PCG did not converge: {res:?}");
        // 1D Laplacian CG converges well within n iterations.
        assert!(res.iters <= n as u32, "too many iters: {}", res.iters);
        let max_err = res
            .x
            .iter()
            .zip(&x_star)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "solution error {max_err} too large (iters={})",
            res.iters
        );
    }

    #[test]
    fn pcg_zero_rhs_is_zero() {
        let Some(ctx) = ctx() else { return };
        let a = CsrMatrix::laplacian_1d(16);
        let solver = Pcg::new(&ctx, &a);
        let res = solver.solve(&[0.0f32; 16], 1e-6, 100);
        assert!(res.x.iter().all(|&v| v.abs() < 1e-6));
    }

    // The decisive cross-check: an implicit cloth step driven by the GPU PCG must
    // produce the same physics as the CPU CG. Same assembly, two linear backends.
    #[test]
    fn gpu_backed_implicit_step_matches_cpu() {
        use engine_core::implicit::{ImplicitParams, ImplicitSolver};
        use engine_core::linsolve::CgCpu;
        use engine_core::{Mesh, SimState};
        use glam::Vec3;

        let Some(ctx) = ctx() else { return };
        let build = || {
            let n = 5;
            let mut verts = Vec::new();
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
            let mut st = SimState::from_mesh(&Mesh::new(verts, tris), 0.05);
            st.pin(0);
            st.pin((n - 1) as usize);
            ImplicitSolver::new(
                st,
                ImplicitParams {
                    cg_tol: 1e-7,
                    ..Default::default()
                },
            )
        };

        let mut cpu = build();
        let mut gpu = build();
        let gpu_solver = GpuLinearSolver::new(&ctx);
        for _ in 0..20 {
            cpu.step(&CgCpu::new());
            gpu.step(&gpu_solver);
        }
        let max_err = cpu
            .state
            .positions
            .iter()
            .zip(&gpu.state.positions)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "GPU vs CPU implicit drift: {max_err}");
    }

    // The GPU Schwarz preconditioner must produce the SAME physics as GPU
    // block-Jacobi (a preconditioner changes convergence, not the solution).
    #[test]
    fn gpu_schwarz_implicit_matches_block() {
        use engine_core::implicit::{ImplicitParams, ImplicitSolver};
        use engine_core::{Mesh, SimState};
        use glam::Vec3;

        let Some(ctx) = ctx() else { return };
        let build = || {
            let n = 6;
            let mut verts = Vec::new();
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
            let mut st = SimState::from_mesh(&Mesh::new(verts, tris), 0.05);
            st.pin(0);
            st.pin((n - 1) as usize);
            ImplicitSolver::new(
                st,
                ImplicitParams {
                    cg_tol: 1e-7,
                    ..Default::default()
                },
            )
        };
        let mut block = build();
        let mut schwarz = build();
        let block_solver = GpuLinearSolver::new(&ctx);
        let schwarz_solver = GpuLinearSolver::with_precond(
            &ctx,
            GpuPrecond::SchwarzMl {
                domain: 16,
                weight: 0.7,
            },
        );
        for _ in 0..20 {
            block.step(&block_solver);
            schwarz.step(&schwarz_solver);
        }
        let max_err = block
            .state
            .positions
            .iter()
            .zip(&schwarz.state.positions)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "GPU Schwarz vs block-Jacobi physics drift: {max_err}"
        );
    }
}
