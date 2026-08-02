// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU Baraff-Witkin membrane assembly in two passes. Pass 1 (`membrane.wgsl`)
//! computes, per triangle, the 3-vertex gradient + 9×9 element Hessian from the
//! eigen-projected stretch + shear materials (the exact port of
//! `engine_core::membrane`). Pass 2 (`membrane_gather.wgsl`) is the atomic-free
//! per-vertex gather of those element blocks into the global CSR `values`/`grad`
//! via the precomputed offsets (`engine_core::MembraneAssembler`).
//!
//! `assemble_full` runs both and is verified bit-close to the CPU
//! `MembraneAssembler::assemble` (values + grad); `assemble` runs only pass 1 (the
//! per-triangle blocks) for the compute-stage test.

use crate::GpuContext;
use engine_core::membrane::stretch_coeffs;
use engine_core::membrane_assembler::MembraneAssembler;
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
struct Gp {
    n: u32,
    nnz: u32,
    ndof: u32,
    _p: u32,
}

/// GPU membrane assembler bound to a fixed topology (from a [`MembraneAssembler`])
/// + stiffnesses. Runs the per-triangle compute + the per-vertex CSR gather.
pub struct GpuMembrane {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ntri: usize,
    nverts: usize,
    nnz: usize,
    ndof: usize,
    wg_tri: u32,
    wg_vert: u32,
    compute: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    // pass-1
    mp: wgpu::Buffer,
    tri: wgpu::Buffer,
    coef: wgpu::Buffer,
    x: wgpu::Buffer,
    out_grad: wgpu::Buffer,
    out_hess: wgpu::Buffer,
    // pass-2 topology
    gp: wgpu::Buffer,
    vt_ptr: wgpu::Buffer,
    vt_elem: wgpu::Buffer,
    vt_local: wgpu::Buffer,
    vt_off: wgpu::Buffer,
    diag_off: wgpu::Buffer,
    pinned: wgpu::Buffer,
    values: wgpu::Buffer,
    grad: wgpu::Buffer,
}

impl GpuMembrane {
    pub fn new(ctx: &GpuContext, asm: &MembraneAssembler, kw: f32, kf: f32, ks: f32) -> Self {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let elements = asm.elements();
        let ntri = elements.len();
        let nverts = asm.n;
        let nnz = asm.nnz();
        let ndof = asm.ndof;

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

        let pipe = |src: &str, entry: &str, label: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let compute = pipe(
            include_str!("shaders/membrane.wgsl"),
            "membrane",
            "membrane",
        );
        let gather = pipe(
            include_str!("shaders/membrane_gather.wgsl"),
            "gather",
            "membrane_gather",
        );

        let storage = wgpu::BufferUsages::STORAGE;
        let store_dst = storage | wgpu::BufferUsages::COPY_DST;
        let store_src = storage | wgpu::BufferUsages::COPY_SRC;
        let init_u32 = |label: &str, d: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(d),
                usage: storage,
            })
        };
        let init_i32 = |label: &str, d: &[i32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(d),
                usage: storage,
            })
        };
        let dyn_buf = |label: &str, len: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let uniform = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };

        Self {
            ntri,
            nverts,
            nnz,
            ndof,
            wg_tri: (ntri.max(1) as u32).div_ceil(64),
            wg_vert: (nverts.max(1) as u32).div_ceil(64),
            compute,
            gather,
            mp: uniform(
                "mem.mp",
                bytemuck::bytes_of(&Mp {
                    ntri: ntri as u32,
                    _p: [0; 3],
                }),
            ),
            tri: init_u32("mem.tri", &tri),
            coef: {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mem.coef"),
                    contents: bytemuck::cast_slice(&coef),
                    usage: storage,
                })
            },
            x: dyn_buf("mem.x", 3 * nverts, store_dst),
            out_grad: dyn_buf("mem.egrad", 9 * ntri, store_src),
            out_hess: dyn_buf("mem.ehess", 81 * ntri, store_src),
            gp: uniform(
                "mem.gp",
                bytemuck::bytes_of(&Gp {
                    n: nverts as u32,
                    nnz: nnz as u32,
                    ndof: ndof as u32,
                    _p: 0,
                }),
            ),
            vt_ptr: init_u32("mem.vt_ptr", asm.vt_ptr()),
            vt_elem: init_u32("mem.vt_elem", asm.vt_elem()),
            vt_local: init_u32("mem.vt_local", asm.vt_local()),
            vt_off: init_i32("mem.vt_off", asm.vt_off()),
            diag_off: init_i32("mem.diag_off", asm.diag_off()),
            pinned: init_u32("mem.pinned", &asm.pinned_u32()),
            values: dyn_buf("mem.values", nnz, store_dst | wgpu::BufferUsages::COPY_SRC),
            grad: dyn_buf("mem.grad", ndof, store_dst | wgpu::BufferUsages::COPY_SRC),
            device,
            queue,
        }
    }

    fn upload_positions(&self, positions: &[Vec3]) {
        assert_eq!(
            positions.len(),
            self.nverts,
            "membrane: vertex count mismatch"
        );
        let flat: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        self.queue
            .write_buffer(&self.x, 0, bytemuck::cast_slice(&flat));
    }

    fn dispatch_compute(&self) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.compute.get_bind_group_layout(0),
            entries: &[
                bge(0, &self.mp),
                bge(1, &self.x),
                bge(2, &self.tri),
                bge(3, &self.coef),
                bge(4, &self.out_grad),
                bge(5, &self.out_hess),
            ],
        });
        self.run(&self.compute, &bg, self.wg_tri);
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

    /// Pass 1 only: per-triangle `(grad[9·ntri], hess[81·ntri])`.
    pub fn assemble(&self, positions: &[Vec3]) -> (Vec<f32>, Vec<f32>) {
        self.upload_positions(positions);
        self.dispatch_compute();
        (
            self.read(&self.out_grad, 9 * self.ntri),
            self.read(&self.out_hess, 81 * self.ntri),
        )
    }

    /// Both passes: the global membrane `(values matching the CSR, grad)`.
    pub fn assemble_full(&self, positions: &[Vec3]) -> (Vec<f32>, Vec<f32>) {
        self.upload_positions(positions);
        self.dispatch_compute();
        // Zero the accumulators (the gather uses += and rows are disjoint per worker).
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
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.gather.get_bind_group_layout(0),
            entries: &[
                bge(0, &self.gp),
                bge(1, &self.vt_ptr),
                bge(2, &self.vt_elem),
                bge(3, &self.vt_local),
                bge(4, &self.vt_off),
                bge(5, &self.diag_off),
                bge(6, &self.pinned),
                bge(7, &self.out_grad),
                bge(8, &self.out_hess),
                bge(9, &self.values),
                bge(10, &self.grad),
            ],
        });
        self.run(&self.gather, &bg, self.wg_vert);
        (
            self.read(&self.values, self.nnz),
            self.read(&self.grad, self.ndof),
        )
    }

    fn read(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len.max(1) * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mem.read"),
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
    use engine_core::membrane::{build_membrane, element_assembly};

    fn ctx() -> Option<GpuContext> {
        GpuContext::new().ok()
    }

    fn scene() -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<bool>, Vec<Vec3>) {
        let nn = 4u32;
        let mut rest = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                rest.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
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
        let n = rest.len();
        let mut pinned = vec![false; n];
        pinned[0] = true;
        pinned[(nn - 1) as usize] = true;
        let pos: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| {
                *p + Vec3::new(
                    (i as f32 * 0.3).sin() * 0.03,
                    (i as f32 * 0.2).cos() * 0.02,
                    0.04 * ((i % 3) as f32 - 1.0),
                )
            })
            .collect();
        (rest, tris, pinned, pos)
    }

    #[test]
    fn gpu_membrane_matches_cpu() {
        // Pass-1 (per-triangle) equivalence, on a deformed patch.
        let Some(ctx) = ctx() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (rest, tris, pinned, pos) = scene();
        let els = build_membrane(&tris, &rest);
        let (kw, kf, ks) = (0.8f32, 0.5, 0.3);
        let asm = MembraneAssembler::new(&els, rest.len(), &pinned);
        let gpu = GpuMembrane::new(&ctx, &asm, kw, kf, ks);
        let (ggrad, ghess) = gpu.assemble(&pos);
        let mut gmax = 0.0f32;
        let mut hmax = 0.0f32;
        for (ti, el) in els.iter().enumerate() {
            let p = [
                pos[el.tri[0] as usize],
                pos[el.tri[1] as usize],
                pos[el.tri[2] as usize],
            ];
            let (cg, ch) = element_assembly(&p, &el.dm_inv, el.area, kw, kf, ks);
            for k in 0..9 {
                gmax = gmax.max((ggrad[9 * ti + k] - cg[k]).abs());
            }
            for k in 0..81 {
                hmax = hmax.max((ghess[81 * ti + k] - ch[k]).abs());
            }
        }
        assert!(gmax < 1e-4, "GPU vs CPU element grad drift {gmax}");
        assert!(hmax < 1e-4, "GPU vs CPU element hess drift {hmax}");
    }

    #[test]
    fn gpu_membrane_gather_matches_cpu() {
        // Full assembly (compute + gather): GPU (values, grad) == CPU assembler.
        let Some(ctx) = ctx() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (rest, tris, pinned, pos) = scene();
        let els = build_membrane(&tris, &rest);
        let (kw, kf, ks) = (0.8f32, 0.5, 0.3);
        let asm = MembraneAssembler::new(&els, rest.len(), &pinned);
        let (cval, cgrad) = asm.assemble(&pos, kw, kf, ks);

        let gpu = GpuMembrane::new(&ctx, &asm, kw, kf, ks);
        let (gval, ggrad) = gpu.assemble_full(&pos);

        let mut vmax = 0.0f32;
        for (a, b) in gval.iter().zip(&cval) {
            vmax = vmax.max((a - b).abs());
        }
        let mut gmax = 0.0f32;
        for (a, b) in ggrad.iter().zip(&cgrad) {
            gmax = gmax.max((a - b).abs());
        }
        assert!(vmax < 1e-3, "GPU vs CPU gather values drift {vmax}");
        assert!(gmax < 1e-4, "GPU vs CPU gather grad drift {gmax}");
    }
}
