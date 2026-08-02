// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU-resident volumetric (tet) assembler: runs the tet element kernel
//! (`shaders/tet.wgsl` → per-element egrad/ehess) then the **shared** atomic-free
//! 4-vertex gather (`shaders/gather4.wgsl`) to scatter the 12×12 blocks into the
//! global CSR. That gather shader is the same one the bending assembler uses — tets
//! and hinges have the identical stencil layout, so it is reused rather than copied,
//! mirroring the CPU `Gather4` extraction. Verified GPU==CPU on Metal.

use crate::read_f32;
use crate::{GpuContext, GpuError};
use engine_core::tet_assembler::TetAssembler;
use glam::Vec3;
use wgpu::util::DeviceExt;

/// GPU tet assembler bound to a fixed topology.
pub struct GpuTetAssembler<'a> {
    ctx: &'a GpuContext,
    n: usize,
    ndof: usize,
    nnz: usize,
    n_tets: usize,
    // static buffers
    idx: wgpu::Buffer,
    dm: wgpu::Buffer,
    vol: wgpu::Buffer,
    vt_ptr: wgpu::Buffer,
    vt_elem: wgpu::Buffer,
    vt_local: wgpu::Buffer,
    vt_off: wgpu::Buffer,
    diag_off: wgpu::Buffer,
    pinned: wgpu::Buffer,
    gp: wgpu::Buffer,
    elem_layout: wgpu::BindGroupLayout,
    elem_pipe: wgpu::ComputePipeline,
    gather_layout: wgpu::BindGroupLayout,
    gather_pipe: wgpu::ComputePipeline,
}

impl<'a> GpuTetAssembler<'a> {
    pub fn new(ctx: &'a GpuContext, asm: &TetAssembler) -> Self {
        let dev = &ctx.device;
        let (n, ndof, nnz) = (asm.n, asm.ndof, asm.nnz());
        let tets = asm.tets();
        let n_tets = tets.len();

        let stor = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let idx_flat: Vec<u32> = tets.iter().flat_map(|t| t.v).collect();
        // Row-major Dm⁻¹ to match the shader's convention (glam is column-major).
        let dm_flat: Vec<f32> = tets
            .iter()
            .flat_map(|t| {
                let m = t.dm_inv;
                [
                    m.x_axis.x, m.y_axis.x, m.z_axis.x, m.x_axis.y, m.y_axis.y, m.z_axis.y,
                    m.x_axis.z, m.y_axis.z, m.z_axis.z,
                ]
            })
            .collect();
        let vol_flat: Vec<f32> = tets.iter().map(|t| t.volume).collect();

        let idx = stor("tetasm.idx", bytemuck::cast_slice(&idx_flat));
        let dm = stor("tetasm.dm", bytemuck::cast_slice(&dm_flat));
        let vol = stor("tetasm.vol", bytemuck::cast_slice(&vol_flat));
        let vt_ptr = stor("tetasm.vt_ptr", bytemuck::cast_slice(asm.vt_ptr()));
        let vt_elem = stor("tetasm.vt_elem", bytemuck::cast_slice(asm.vt_elem()));
        let vt_local = stor("tetasm.vt_local", bytemuck::cast_slice(asm.vt_local()));
        let vt_off = stor("tetasm.vt_off", bytemuck::cast_slice(asm.vt_off()));
        let diag_off = stor("tetasm.diag_off", bytemuck::cast_slice(asm.diag_off()));
        let pinned = stor("tetasm.pinned", bytemuck::cast_slice(&asm.pinned_u32()));
        let gp = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetasm.gp"),
            contents: bytemuck::cast_slice(&[n as u32, nnz as u32, ndof as u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let entry = |binding, ro: bool, uniform: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if uniform {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: ro }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // tet element layout: 0 params(u) 1 idx 2 dm 3 vol 4 pos | 5 energy 6 grad 7 hess
        let elem_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tetasm.elem.bgl"),
            entries: &[
                entry(0, true, true),
                entry(1, true, false),
                entry(2, true, false),
                entry(3, true, false),
                entry(4, true, false),
                entry(5, false, false),
                entry(6, false, false),
                entry(7, false, false),
            ],
        });
        // shared gather layout: 0 gp(u) 1..8 read-only | 9 values(rw) 10 grad(rw)
        let mut ge = vec![entry(0, true, true)];
        for b in 1..=8 {
            ge.push(entry(b, true, false));
        }
        ge.push(entry(9, false, false));
        ge.push(entry(10, false, false));
        let gather_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tetasm.gather.bgl"),
            entries: &ge,
        });

        let mk = |bgl: &wgpu::BindGroupLayout, src: &str, ep: &str, label: &str| {
            let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let elem_pipe = mk(
            &elem_layout,
            include_str!("shaders/tet.wgsl"),
            "tet",
            "tetasm.elem",
        );
        // The SAME gather kernel the bending assembler uses.
        let gather_pipe = mk(
            &gather_layout,
            include_str!("shaders/gather4.wgsl"),
            "gather",
            "tetasm.gather",
        );

        Self {
            ctx,
            n,
            ndof,
            nnz,
            n_tets,
            idx,
            dm,
            vol,
            vt_ptr,
            vt_elem,
            vt_local,
            vt_off,
            diag_off,
            pinned,
            gp,
            elem_layout,
            elem_pipe,
            gather_layout,
            gather_pipe,
        }
    }

    /// Assemble the tet CSR values + gradient at `positions`, GPU-resident between
    /// the element and gather passes.
    pub fn assemble(
        &self,
        positions: &[Vec3],
        mu: f32,
        lambda: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let dev = &self.ctx.device;
        if self.n_tets == 0 {
            return Ok((vec![0.0; self.nnz], vec![0.0; self.ndof]));
        }
        let pos: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let pos_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetasm.pos"),
            contents: bytemuck::cast_slice(&pos),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = [
            (self.n_tets as u32).to_le_bytes(),
            mu.to_le_bytes(),
            lambda.to_le_bytes(),
            1.0e-6f32.to_le_bytes(),
        ]
        .concat();
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetasm.elem.params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mk_rw = |label, len: usize, copy_src: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if copy_src {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let energy = mk_rw("tetasm.energy", self.n_tets, false);
        let egrad = mk_rw("tetasm.egrad", self.n_tets * 12, false);
        let ehess = mk_rw("tetasm.ehess", self.n_tets * 144, false);
        let values = mk_rw("tetasm.values", self.nnz, true);
        let grad = mk_rw("tetasm.grad", self.ndof, true);
        // The gather accumulates, so the targets must start zeroed.
        self.ctx
            .queue
            .write_buffer(&values, 0, bytemuck::cast_slice(&vec![0.0f32; self.nnz]));
        self.ctx
            .queue
            .write_buffer(&grad, 0, bytemuck::cast_slice(&vec![0.0f32; self.ndof]));

        let bind = |layout: &wgpu::BindGroupLayout, res: &[&wgpu::Buffer]| {
            let entries: Vec<wgpu::BindGroupEntry> = res
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tetasm.bind"),
                layout,
                entries: &entries,
            })
        };
        let elem_bind = bind(
            &self.elem_layout,
            &[
                &params_buf,
                &self.idx,
                &self.dm,
                &self.vol,
                &pos_buf,
                &energy,
                &egrad,
                &ehess,
            ],
        );
        let gather_bind = bind(
            &self.gather_layout,
            &[
                &self.gp,
                &self.vt_ptr,
                &self.vt_elem,
                &self.vt_local,
                &self.vt_off,
                &self.diag_off,
                &self.pinned,
                &egrad,
                &ehess,
                &values,
                &grad,
            ],
        );

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tetasm"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tetasm.cp"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.elem_pipe);
            cp.set_bind_group(0, &elem_bind, &[]);
            cp.dispatch_workgroups(self.n_tets.div_ceil(64) as u32, 1, 1);
            cp.set_pipeline(&self.gather_pipe);
            cp.set_bind_group(0, &gather_bind, &[]);
            cp.dispatch_workgroups(self.n.div_ceil(64) as u32, 1, 1);
        }
        let vs = crate::stage_buffer(&self.ctx.device, &mut enc, &values, (self.nnz * 4) as u64);
        let gs = crate::stage_buffer(&self.ctx.device, &mut enc, &grad, (self.ndof * 4) as u64);
        self.ctx.queue.submit(Some(enc.finish()));
        Ok((read_f32(self.ctx, &vs)?, read_f32(self.ctx, &gs)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::tet::{build_tets, TetModel};

    #[test]
    fn gpu_tet_assembly_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let nn = 3usize;
        let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
        let mut rest = Vec::new();
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    rest.push(Vec3::new(i as f32 * 0.5, j as f32 * 0.5, k as f32 * 0.5));
                }
            }
        }
        let mut tet_idx = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
                    let (a, b, c, d) = (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i, j + 1, k),
                        idx(i + 1, j + 1, k),
                    );
                    let (e, f, g, h) = (
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i, j + 1, k + 1),
                        idx(i + 1, j + 1, k + 1),
                    );
                    tet_idx.push([a, b, c, e]);
                    tet_idx.push([b, d, c, h]);
                    tet_idx.push([b, c, e, h]);
                    tet_idx.push([c, e, g, h]);
                    tet_idx.push([b, e, f, h]);
                }
            }
        }
        let n = rest.len();
        let els = build_tets(&tet_idx, &rest, 1e-9, TetModel::StVK);
        // Pin one face so the Dirichlet path is exercised too.
        let mut pinned = vec![false; n];
        for (i, p) in rest.iter().enumerate() {
            if p.y == 0.0 {
                pinned[i] = true;
            }
        }
        let x: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| {
                Vec3::new(
                    p.x * 1.12 + 0.03 * (i as f32 * 0.4).sin(),
                    p.y * 0.94 + 0.02 * (i as f32 * 0.6).cos(),
                    p.z * 1.06 + 0.02 * (i as f32 * 0.8).sin(),
                )
            })
            .collect();
        let (mu, lambda) = (60.0f32, 120.0f32);

        let cpu = TetAssembler::new(&els, n, &pinned);
        let (cv, cg) = cpu.assemble(&x, mu, lambda);
        let gpu = GpuTetAssembler::new(&ctx, &cpu);
        let (gv, gg) = gpu.assemble(&x, mu, lambda).expect("gpu assemble");

        assert_eq!(gv.len(), cv.len());
        let vscale = cv.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let mut vmax = 0.0f32;
        for (a, b) in gv.iter().zip(&cv) {
            vmax = vmax.max((a - b).abs());
        }
        assert!(
            vmax < 5e-3 * vscale,
            "GPU vs CPU tet CSR values drift {vmax} (scale {vscale})"
        );
        let gscale = cg.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let mut gmax = 0.0f32;
        for (a, b) in gg.iter().zip(&cg) {
            gmax = gmax.max((a - b).abs());
        }
        assert!(
            gmax < 5e-3 * gscale,
            "GPU vs CPU tet grad drift {gmax} (scale {gscale})"
        );
    }
}
