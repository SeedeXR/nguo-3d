// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU-resident bending assembler: runs the bending embed kernel
//! (`shaders/bending.wgsl` → per-hinge egrad/ehess) then the atomic-free per-vertex
//! gather (`shaders/gather4.wgsl`) to scatter the 12×12 hinge Hessian blocks
//! into the global CSR `values` + `grad`. The static sparsity/offset arrays come
//! from [`engine_core::bending_assembler::BendingAssembler`] (built once). Mirrors
//! the membrane GPU gather assembler. Verified GPU==CPU on Metal.

use crate::read_f32;
use crate::{GpuContext, GpuError};
use engine_core::bending_assembler::BendingAssembler;
use glam::Vec3;
use wgpu::util::DeviceExt;

/// GPU bending assembler bound to a fixed topology.
pub struct GpuBendingAssembler<'a> {
    ctx: &'a GpuContext,
    n: usize,
    ndof: usize,
    nnz: usize,
    n_hinges: usize,
    // static (topology) buffers
    idx: wgpu::Buffer,
    rest: wgpu::Buffer,
    wt: wgpu::Buffer,
    vt_ptr: wgpu::Buffer,
    vt_elem: wgpu::Buffer,
    vt_local: wgpu::Buffer,
    vt_off: wgpu::Buffer,
    diag_off: wgpu::Buffer,
    pinned: wgpu::Buffer,
    gp: wgpu::Buffer,
    // pipelines
    embed_layout: wgpu::BindGroupLayout,
    embed_pipe: wgpu::ComputePipeline,
    gather_layout: wgpu::BindGroupLayout,
    gather_pipe: wgpu::ComputePipeline,
}

impl<'a> GpuBendingAssembler<'a> {
    pub fn new(ctx: &'a GpuContext, asm: &BendingAssembler) -> Self {
        let dev = &ctx.device;
        let n = asm.n;
        let ndof = asm.ndof;
        let nnz = asm.nnz();
        let hinges = asm.hinges();
        let n_hinges = hinges.len();

        let stor = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let idx_flat: Vec<u32> = hinges.iter().flat_map(|h| h.v).collect();
        let rest_flat: Vec<f32> = hinges.iter().map(|h| h.rest).collect();
        let wt_flat: Vec<f32> = hinges.iter().map(|h| h.w).collect();

        let idx = stor("bend.idx", bytemuck::cast_slice(&idx_flat));
        let rest = stor("bend.rest", bytemuck::cast_slice(&rest_flat));
        let wt = stor("bend.wt", bytemuck::cast_slice(&wt_flat));
        let vt_ptr = stor("bend.vt_ptr", bytemuck::cast_slice(asm.vt_ptr()));
        let vt_elem = stor("bend.vt_elem", bytemuck::cast_slice(asm.vt_elem()));
        let vt_local = stor("bend.vt_local", bytemuck::cast_slice(asm.vt_local()));
        let vt_off = stor("bend.vt_off", bytemuck::cast_slice(asm.vt_off()));
        let diag_off = stor("bend.diag_off", bytemuck::cast_slice(asm.diag_off()));
        let pinned = stor("bend.pinned", bytemuck::cast_slice(&asm.pinned_u32()));
        let gp = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bend.gp"),
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
        // embed layout: 0 params(u) 1 idx 2 rest 3 wt 4 pos | 5 energy(rw) 6 grad(rw) 7 hess(rw)
        let embed_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bend.embed.bgl"),
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
        // gather layout: 0 gp(u) 1..6 topology(ro) 7 egrad 8 ehess | 9 values(rw) 10 grad(rw)
        let mut ge = vec![entry(0, true, true)];
        for b in 1..=8 {
            ge.push(entry(b, true, false));
        }
        ge.push(entry(9, false, false));
        ge.push(entry(10, false, false));
        let gather_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bend.gather.bgl"),
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
        let embed_pipe = mk(
            &embed_layout,
            include_str!("shaders/bending.wgsl"),
            "bending",
            "bend.embed",
        );
        let gather_pipe = mk(
            &gather_layout,
            include_str!("shaders/gather4.wgsl"),
            "gather",
            "bend.gather",
        );

        Self {
            ctx,
            n,
            ndof,
            nnz,
            n_hinges,
            idx,
            rest,
            wt,
            vt_ptr,
            vt_elem,
            vt_local,
            vt_off,
            diag_off,
            pinned,
            gp,
            embed_layout,
            embed_pipe,
            gather_layout,
            gather_pipe,
        }
    }

    /// Assemble bending CSR values + gradient at `positions` for stiffness `kb`.
    /// Returns `(values matching the assembler's structure, grad)`.
    pub fn assemble(&self, positions: &[Vec3], kb: f32) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let dev = &self.ctx.device;
        if self.n_hinges == 0 {
            return Ok((vec![0.0; self.nnz], vec![0.0; self.ndof]));
        }
        let pos: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let pos_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bend.pos"),
            contents: bytemuck::cast_slice(&pos),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bend.embed.params"),
            contents: bytemuck::cast_slice(&[self.n_hinges as u32, kb.to_bits(), 0u32, 0u32]),
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
        let energy = mk_rw("bend.energy", self.n_hinges, false);
        let egrad = mk_rw("bend.egrad", self.n_hinges * 12, false);
        let ehess = mk_rw("bend.ehess", self.n_hinges * 144, false);
        let values = mk_rw("bend.values", self.nnz, true);
        let grad = mk_rw("bend.grad", self.ndof, true);
        // Zero the accumulators (gather assumes pre-zeroed).
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
                label: Some("bend.bind"),
                layout,
                entries: &entries,
            })
        };
        let embed_bind = bind(
            &self.embed_layout,
            &[
                &params, &self.idx, &self.rest, &self.wt, &pos_buf, &energy, &egrad, &ehess,
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
            label: Some("bend.asm"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bend.asm.cp"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.embed_pipe);
            cp.set_bind_group(0, &embed_bind, &[]);
            cp.dispatch_workgroups(self.n_hinges.div_ceil(64) as u32, 1, 1);
            cp.set_pipeline(&self.gather_pipe);
            cp.set_bind_group(0, &gather_bind, &[]);
            cp.dispatch_workgroups(self.n.div_ceil(64) as u32, 1, 1);
        }
        let vbytes = (self.nnz * 4) as u64;
        let gbytes = (self.ndof * 4) as u64;
        let v_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &values, vbytes);
        let g_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &grad, gbytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let values_out = read_f32(self.ctx, &v_stage)?;
        let grad_out = read_f32(self.ctx, &g_stage)?;
        Ok((values_out, grad_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::bending::build_hinges;

    #[test]
    fn gpu_bending_assembly_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let nn = 5u32;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                let a = 0.25 * j as f32;
                verts.push(Vec3::new(
                    i as f32 * 0.1,
                    a.cos() * j as f32 * 0.1,
                    a.sin() * j as f32 * 0.1,
                ));
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
        let n = verts.len();
        let hinges = build_hinges(&tris, &verts);
        let mut pinned = vec![false; n];
        pinned[0] = true;
        pinned[(nn - 1) as usize] = true;
        let x: Vec<Vec3> = verts
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::new(0.0, 0.0, 0.02 * (i as f32).sin()))
            .collect();
        let kb = 0.7f32;

        let asm = BendingAssembler::new(&hinges, n, &pinned);
        let (cpu_v, cpu_g) = asm.assemble(&x, kb);
        let gasm = GpuBendingAssembler::new(&ctx, &asm);
        let (gpu_v, gpu_g) = gasm.assemble(&x, kb).expect("gpu assemble");

        assert_eq!(gpu_v.len(), cpu_v.len());
        // Tolerance 1e-3: values are O(70) here (∇θ ∝ 1/flap-height is large), so
        // GPU-gather vs CPU-triplet accumulation-order differences land at ~1e-4
        // absolute (~1e-6 relative) — f32 rounding, not a bug. Same tolerance the
        // membrane GPU-gather test uses.
        let mut vmax = 0.0f32;
        for (a, b) in gpu_v.iter().zip(&cpu_v) {
            vmax = vmax.max((a - b).abs());
        }
        assert!(vmax < 1e-3, "GPU vs CPU bending values drift {vmax}");
        let mut gmax = 0.0f32;
        for (a, b) in gpu_g.iter().zip(&cpu_g) {
            gmax = gmax.max((a - b).abs());
        }
        assert!(gmax < 1e-3, "GPU vs CPU bending grad drift {gmax}");
    }
}
