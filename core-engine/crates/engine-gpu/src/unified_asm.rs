// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU unified assembly (inertia + springs + membrane → one union CSR). Runs the
//! membrane per-triangle compute pass (`membrane.wgsl` → egrad/ehess) then the
//! atomic-free per-vertex gather (`unified_gather.wgsl`) over the union offsets
//! from `engine_core::UnifiedAssembler`, into one pre-zeroed `values`/`grad`
//! buffer. Verified bit-close to `UnifiedAssembler::assemble` (the CPU oracle) —
//! the assembly half of a fully GPU-resident membrane step.

use crate::GpuContext;
use engine_core::membrane::stretch_coeffs;
use engine_core::unified_assembler::UnifiedAssembler;
use glam::Vec3;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Mp {
    ntri: u32,
    _p: [u32; 3],
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Ap {
    n: u32,
    inv_h2: f32,
    k: f32,
    _pad: f32,
}

/// GPU unified assembler bound to a fixed topology (from `UnifiedAssembler`) +
/// stiffnesses. `assemble(x, x_tilde, inv_h2)` → `(values, grad)`.
pub struct GpuUnified {
    device: wgpu::Device,
    queue: wgpu::Queue,
    n: usize,
    nnz: usize,
    ndof: usize,
    wg_tri: u32,
    wg_vert: u32,
    k: f32,
    compute: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    // shared
    x: wgpu::Buffer,
    x_tilde: wgpu::Buffer,
    egrad: wgpu::Buffer,
    ehess: wgpu::Buffer,
    /// Shell pass: writes into `egrad`/`ehess` AFTER the membrane elements, so both kinds
    /// share one element-index space and `unified_gather.wgsl` is untouched (see
    /// `UnifiedAssembler::shells`). None when the scene has no shell elements.
    shell: Option<ShellPass>,
    values: wgpu::Buffer,
    grad: wgpu::Buffer,
    ap: wgpu::Buffer,
    // membrane compute
    mp: wgpu::Buffer,
    tri: wgpu::Buffer,
    coef: wgpu::Buffer,
    // gather topology
    mass: wgpu::Buffer,
    pinned: wgpu::Buffer,
    diag_off: wgpu::Buffer,
    s_ptr: wgpu::Buffer,
    s_j: wgpu::Buffer,
    s_rest: wgpu::Buffer,
    s_off: wgpu::Buffer,
    m_ptr: wgpu::Buffer,
    m_elem: wgpu::Buffer,
    m_local: wgpu::Buffer,
    m_off: wgpu::Buffer,
}

/// Buffers and pipeline for the shell half of the unified assembly.
struct ShellPass {
    compute: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    idx: wgpu::Buffer,
    dm: wgpu::Buffer,
    area: wgpu::Buffer,
    energy: wgpu::Buffer,
    wg: u32,
}

impl GpuUnified {
    pub fn new(
        ctx: &GpuContext,
        asm: &UnifiedAssembler,
        k: f32,
        kw: f32,
        kf: f32,
        ks: f32,
    ) -> Self {
        Self::with_shells(ctx, asm, k, kw, kf, ks, 0.0, 0.0, 1.0e-6)
    }

    /// As [`Self::new`], also assembling the assembler's SHELL elements. Zero
    /// `shell_mu`/`shell_lambda` leaves them inert, so `new` delegating here is
    /// behaviour-preserving for a membrane-only scene.
    #[allow(clippy::too_many_arguments)]
    pub fn with_shells(
        ctx: &GpuContext,
        asm: &UnifiedAssembler,
        k: f32,
        kw: f32,
        kf: f32,
        ks: f32,
        shell_mu: f32,
        shell_lambda: f32,
        eps: f32,
    ) -> Self {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let n = asm.n;
        let ndof = asm.ndof;
        let nnz = asm.nnz();
        let elements = asm.elements();
        let ntri = elements.len();

        let mut tri = Vec::with_capacity(ntri * 3);
        let mut coef = Vec::with_capacity(ntri * 9);
        for el in elements {
            tri.extend_from_slice(&el.tri);
            let (cu, cv) = stretch_coeffs(&el.dm_inv);
            coef.extend_from_slice(&cu);
            coef.extend_from_slice(&cv);
            coef.push(kw * el.area);
            coef.push(kf * el.area);
            coef.push(ks * el.area);
        }

        let shells = asm.shells();
        let nshell = shells.len();
        let nelem = ntri + nshell;

        let pipe = |src: &str, entry: &str| {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let storage = wgpu::BufferUsages::STORAGE;
        let store_dst = storage | wgpu::BufferUsages::COPY_DST;
        let iu = |l: &str, d: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: bytemuck::cast_slice(d),
                usage: storage,
            })
        };
        let ii = |l: &str, d: &[i32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: bytemuck::cast_slice(d),
                usage: storage,
            })
        };
        let iff = |l: &str, d: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: bytemuck::cast_slice(d),
                usage: storage,
            })
        };
        let dynb = |l: &str, len: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(l),
                size: (len.max(1) * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let uni = |l: &str, b: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: b,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let out = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

        // Shell pass, only when the assembler carries shell elements. Its outputs start at
        // element index `ntri`, so the membrane pass fills [0, ntri) and this fills
        // [ntri, nelem) of the SAME egrad/ehess buffers.
        let shell_pass = if nshell > 0 {
            let mut sidx = Vec::with_capacity(nshell * 3);
            let mut sdm = Vec::with_capacity(nshell * 4);
            let mut sarea = Vec::with_capacity(nshell);
            for el in shells {
                sidx.extend_from_slice(&el.v);
                // Column-major 2x2, matching shell.wgsl's vec4 layout.
                sdm.extend_from_slice(&[
                    el.dm_inv.x_axis.x,
                    el.dm_inv.x_axis.y,
                    el.dm_inv.y_axis.x,
                    el.dm_inv.y_axis.y,
                ]);
                sarea.push(el.area);
            }
            // All shells in one batch share a model (`build_shells` assigns one), so the
            // first element's model is the batch's.
            let model_id: u32 = match shells[0].model {
                engine_core::tet::TetModel::Arap => 0,
                engine_core::tet::TetModel::StVK => 1,
                engine_core::tet::TetModel::SnHk => 2,
            };
            let sp = [
                (nshell as u32).to_le_bytes(),
                shell_mu.to_le_bytes(),
                shell_lambda.to_le_bytes(),
                eps.to_le_bytes(),
                model_id.to_le_bytes(),
                (ntri as u32).to_le_bytes(),
                0u32.to_le_bytes(),
                0u32.to_le_bytes(),
            ]
            .concat();
            Some(ShellPass {
                compute: pipe(include_str!("shaders/shell.wgsl"), "shell"),
                params: uni("u.sparams", &sp),
                idx: iu("u.sidx", &sidx),
                dm: iff("u.sdm", &sdm),
                area: iff("u.sarea", &sarea),
                energy: dynb("u.senergy", nelem, storage),
                wg: (nshell as u32).div_ceil(64),
            })
        } else {
            None
        };

        Self {
            n,
            nnz,
            ndof,
            wg_tri: (ntri.max(1) as u32).div_ceil(64),
            wg_vert: (n.max(1) as u32).div_ceil(64),
            k,
            compute: pipe(include_str!("shaders/membrane.wgsl"), "membrane"),
            gather: pipe(include_str!("shaders/unified_gather.wgsl"), "gather"),
            x: dynb("u.x", 3 * n, store_dst),
            x_tilde: dynb("u.xtilde", 3 * n, store_dst),
            egrad: dynb("u.egrad", 9 * nelem, storage),
            ehess: dynb("u.ehess", 81 * nelem, storage),
            shell: shell_pass,
            values: dynb("u.values", nnz, out),
            grad: dynb("u.grad", ndof, out),
            ap: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("u.ap"),
                size: std::mem::size_of::<Ap>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            mp: uni(
                "u.mp",
                bytemuck::bytes_of(&Mp {
                    ntri: ntri as u32,
                    _p: [0; 3],
                }),
            ),
            tri: iu("u.tri", &tri),
            coef: iff("u.coef", &coef),
            mass: iff("u.mass", asm.mass()),
            pinned: iu("u.pinned", &asm.pinned_u32()),
            diag_off: ii("u.diag_off", asm.diag_off()),
            s_ptr: iu("u.s_ptr", asm.s_ptr()),
            s_j: iu("u.s_j", asm.s_j()),
            s_rest: iff("u.s_rest", asm.s_rest()),
            s_off: ii("u.s_off", asm.s_off()),
            m_ptr: iu("u.m_ptr", asm.m_ptr()),
            m_elem: iu("u.m_elem", asm.m_elem()),
            m_local: iu("u.m_local", asm.m_local()),
            m_off: ii("u.m_off", asm.m_off()),
            device,
            queue,
        }
    }

    /// Assemble `(values matching the union CSR, grad)` at `(x, x_tilde)`.
    pub fn assemble(&self, x: &[Vec3], x_tilde: &[Vec3], inv_h2: f32) -> (Vec<f32>, Vec<f32>) {
        let flat = |v: &[Vec3]| -> Vec<f32> { v.iter().flat_map(|p| [p.x, p.y, p.z]).collect() };
        self.queue
            .write_buffer(&self.x, 0, bytemuck::cast_slice(&flat(x)));
        self.queue
            .write_buffer(&self.x_tilde, 0, bytemuck::cast_slice(&flat(x_tilde)));
        self.queue.write_buffer(
            &self.ap,
            0,
            bytemuck::bytes_of(&Ap {
                n: self.n as u32,
                inv_h2,
                k: self.k,
                _pad: 0.0,
            }),
        );
        // Pass 1: per-triangle membrane element blocks.
        let bg_c = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.compute.get_bind_group_layout(0),
            entries: &[
                bge(0, &self.mp),
                bge(1, &self.x),
                bge(2, &self.tri),
                bge(3, &self.coef),
                bge(4, &self.egrad),
                bge(5, &self.ehess),
            ],
        });
        self.run(&self.compute, &bg_c, self.wg_tri);
        // Zero accumulators (gather uses += and rows are disjoint per worker).
        self.queue.write_buffer(
            &self.values,
            0,
            bytemuck::cast_slice(&vec![0.0f32; self.nnz]),
        );
        self.queue.write_buffer(
            &self.grad,
            0,
            bytemuck::cast_slice(&vec![0.0f32; self.ndof]),
        );
        // Pass 2: unified per-vertex gather.
        let bg_g = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.gather.get_bind_group_layout(0),
            entries: &[
                bge(0, &self.ap),
                bge(1, &self.x),
                bge(2, &self.x_tilde),
                bge(3, &self.mass),
                bge(4, &self.pinned),
                bge(5, &self.diag_off),
                bge(6, &self.s_ptr),
                bge(7, &self.s_j),
                bge(8, &self.s_rest),
                bge(9, &self.s_off),
                bge(10, &self.m_ptr),
                bge(11, &self.m_elem),
                bge(12, &self.m_local),
                bge(13, &self.m_off),
                bge(14, &self.egrad),
                bge(15, &self.ehess),
                bge(16, &self.values),
                bge(17, &self.grad),
            ],
        });
        // Shell elements fill the tail of egrad/ehess before the gather consumes both.
        if let Some(sp) = &self.shell {
            let bgl = sp.compute.get_bind_group_layout(0);
            let bg_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("u.bg_shell"),
                layout: &bgl,
                entries: &[
                    bge(0, &sp.params),
                    bge(1, &sp.idx),
                    bge(2, &sp.dm),
                    bge(3, &sp.area),
                    bge(4, &self.x),
                    bge(5, &sp.energy),
                    bge(6, &self.egrad),
                    bge(7, &self.ehess),
                ],
            });
            self.run(&sp.compute, &bg_s, sp.wg);
        }
        self.run(&self.gather, &bg_g, self.wg_vert);
        (
            self.read(&self.values, self.nnz),
            self.read(&self.grad, self.ndof),
        )
    }

    fn run(&self, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, groups: u32) {
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

    fn read(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len.max(1) * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u.read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v = bytemuck::cast_slice::<u8, f32>(&data)[..len].to_vec();
        drop(data);
        staging.unmap();
        v
    }
}

fn bge(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::membrane::build_membrane;
    use engine_core::{Mesh, SimState};

    fn ctx() -> Option<GpuContext> {
        GpuContext::new().ok()
    }

    #[test]
    fn gpu_unified_matches_cpu() {
        let Some(ctx) = ctx() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let nn = 4u32;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                verts.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
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
        let mut state = SimState::from_mesh(&Mesh::new(verts.clone(), tris.clone()), 0.1);
        state.pin(0);
        state.pin((nn - 1) as usize);
        let els = build_membrane(&tris, &verts);
        let x: Vec<Vec3> = verts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                *p + Vec3::new(
                    (i as f32 * 0.3).sin() * 0.03,
                    -0.02,
                    0.04 * ((i % 3) as f32 - 1.0),
                )
            })
            .collect();
        let x_tilde: Vec<Vec3> = x.iter().map(|p| *p + Vec3::new(0.0, -0.001, 0.0)).collect();
        let (inv_h2, k, kw, kf, ks) = (1.0e4, 1000.0, 0.8, 0.5, 0.3);

        let asm = UnifiedAssembler::new(&state, &els);
        let (cval, cgrad) = asm.assemble(&x, &x_tilde, inv_h2, k, kw, kf, ks);

        let gpu = GpuUnified::new(&ctx, &asm, k, kw, kf, ks);
        let (gval, ggrad) = gpu.assemble(&x, &x_tilde, inv_h2);

        let vmax = gval
            .iter()
            .zip(&cval)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let gmax = ggrad
            .iter()
            .zip(&cgrad)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(vmax < 1e-2, "GPU vs CPU unified values drift {vmax}");
        assert!(gmax < 1e-3, "GPU vs CPU unified grad drift {gmax}");
    }
}

#[cfg(test)]
mod shell_stream_tests {
    use super::*;
    use engine_core::membrane::build_membrane;
    use engine_core::shell_model::build_shells;
    use engine_core::tet::TetModel;
    use engine_core::{Mesh, SimState};

    /// The GPU unified assembly must match the CPU oracle when the element-index space
    /// carries BOTH membrane and shell elements — the whole point of sharing one space is
    /// that two compute passes fill one buffer and the gather stays unchanged, so the
    /// failure mode this guards is a shell block landing at a membrane element's offset.
    ///
    /// Membrane and shell cover DISJOINT face subsets (as the gauge driver splits them by
    /// constitutive model), and the sets are interleaved rather than split down the middle
    /// so a systematic off-by-`ntri` cannot pass.
    #[test]
    fn gpu_unified_with_shells_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let nn = 5usize;
        let mut rest = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                rest.push(glam::Vec3::new(i as f32 * 0.3, 0.0, j as f32 * 0.3));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nn - 1 {
            for i in 0..nn - 1 {
                let v = (j * nn + i) as u32;
                let r = v + nn as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        // Interleaved split: even faces membrane, odd faces shell.
        let mem_tris: Vec<[u32; 3]> = tris.iter().step_by(2).copied().collect();
        let sh_tris: Vec<[u32; 3]> = tris.iter().skip(1).step_by(2).copied().collect();
        assert!(mem_tris.len() >= 8 && sh_tris.len() >= 8);

        let mem = build_membrane(&mem_tris, &rest);
        let shells = build_shells(&sh_tris, &rest, 1.0e-12, TetModel::Arap);
        let state = SimState::from_mesh(&Mesh::new(rest.clone(), tris.clone()), 0.2);
        let asm =
            engine_core::unified_assembler::UnifiedAssembler::with_shells(&state, &mem, &shells);
        assert_eq!(asm.element_count(), mem.len() + shells.len());

        let (k, kw, kf, ks) = (12.0f32, 60.0f32, 55.0f32, 30.0f32);
        let (smu, slam, eps) = (140.0f32, 80.0f32, 1.0e-6f32);
        let x: Vec<glam::Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let f = i as f32;
                glam::Vec3::new(
                    p.x * 1.10 + 0.03 * (f * 0.4).sin(),
                    0.07 * (f * 0.5).sin(),
                    p.z * 0.94 + 0.02 * (f * 0.9).cos(),
                )
            })
            .collect();
        let x_tilde: Vec<glam::Vec3> = x
            .iter()
            .map(|p| *p + glam::Vec3::new(0.0, -0.015, 0.0))
            .collect();
        let inv_h2 = 3600.0f32;

        let (cv, cg) =
            asm.assemble_with_shells(&x, &x_tilde, inv_h2, k, kw, kf, ks, smu, slam, eps);
        let gpu = GpuUnified::with_shells(&ctx, &asm, k, kw, kf, ks, smu, slam, eps);
        let (gv, gg) = gpu.assemble(&x, &x_tilde, inv_h2);

        assert_eq!(gv.len(), cv.len());
        assert_eq!(gg.len(), cg.len());
        let rel = |a: &[f32], b: &[f32]| {
            let mut worst = 0.0f32;
            let mut scale = 0.0f32;
            for (p, q) in a.iter().zip(b) {
                worst = worst.max((p - q).abs());
                scale = scale.max(q.abs());
            }
            (worst, scale)
        };
        let (wv, sv) = rel(&gv, &cv);
        let (wg, sg) = rel(&gg, &cg);
        assert!(sv > 0.0 && sg > 0.0, "degenerate test");
        assert!(
            wv / sv < 1.0e-3,
            "values mismatch: worst {wv:.3e} on scale {sv:.3e}"
        );
        assert!(
            wg / sg < 1.0e-3,
            "grad mismatch: worst {wg:.3e} on scale {sg:.3e}"
        );
        eprintln!(
            "unified+shells: {} membrane + {} shell elements, rel err values {:.2e} grad {:.2e}",
            mem.len(),
            shells.len(),
            wv / sv,
            wg / sg
        );
    }
}
