// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU volumetric (tet) FEM element kernel (`shaders/tet.wgsl`) — per-element StVK
//! energy, analytic 12-dof gradient (Piola stress) and analytic 12×12 Hessian
//! (isotropic eigensystem), mirroring `engine_core::tet`. One thread per tet, one
//! SVD each, no finite differences. Verified GPU≈CPU on Metal.

use crate::read_f32;
use crate::{GpuContext, GpuError};
use engine_core::tet::TetElement;
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Per-tet element output: energy, 12-dof gradient, row-major 12×12 Hessian.
pub struct TetOut {
    pub energy: Vec<f32>,
    pub grad: Vec<[f32; 12]>,
    pub hess: Vec<[f32; 144]>,
}

/// GPU tet element engine.
pub struct GpuTet<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuTet<'a> {
    pub fn new(ctx: &'a GpuContext) -> Self {
        let dev = &ctx.device;
        let buf = |binding, ro: bool, uniform: bool| wgpu::BindGroupLayoutEntry {
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
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tet.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, true, false),
                buf(3, true, false),
                buf(4, true, false),
                buf(5, false, false),
                buf(6, false, false),
                buf(7, false, false),
            ],
        });
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tet"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tet.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tet.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tet"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("tet"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// Per-element energy, gradient and Hessian at `positions`.
    pub fn elements(
        &self,
        tets: &[TetElement],
        positions: &[Vec3],
        mu: f32,
        lambda: f32,
    ) -> Result<TetOut, GpuError> {
        let n = tets.len();
        if n == 0 {
            return Ok(TetOut {
                energy: Vec::new(),
                grad: Vec::new(),
                hess: Vec::new(),
            });
        }
        let dev = &self.ctx.device;
        let idx: Vec<u32> = tets.iter().flat_map(|t| t.v).collect();
        // Row-major Dm⁻¹ (glam Mat3 is column-major).
        let dm: Vec<f32> = tets
            .iter()
            .flat_map(|t| {
                let m = t.dm_inv;
                [
                    m.x_axis.x, m.y_axis.x, m.z_axis.x, m.x_axis.y, m.y_axis.y, m.z_axis.y,
                    m.x_axis.z, m.y_axis.z, m.z_axis.z,
                ]
            })
            .collect();
        let vol: Vec<f32> = tets.iter().map(|t| t.volume).collect();
        let pos: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();

        let params = [
            (n as u32).to_le_bytes(),
            mu.to_le_bytes(),
            lambda.to_le_bytes(),
            1.0e-6f32.to_le_bytes(),
        ]
        .concat();
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet.params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let ro = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let idx_buf = ro("tet.idx", bytemuck::cast_slice(&idx));
        let dm_buf = ro("tet.dm", bytemuck::cast_slice(&dm));
        let vol_buf = ro("tet.vol", bytemuck::cast_slice(&vol));
        let pos_buf = ro("tet.pos", bytemuck::cast_slice(&pos));
        let mk = |label, len: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let e_buf = mk("tet.energy", n);
        let g_buf = mk("tet.grad", n * 12);
        let h_buf = mk("tet.hess", n * 144);
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tet.bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: idx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vol_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: e_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: h_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc =
            dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tet") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tet.cp"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }
        let es = crate::stage_buffer(&self.ctx.device, &mut enc, &e_buf, (n * 4) as u64);
        let gs = crate::stage_buffer(&self.ctx.device, &mut enc, &g_buf, (n * 12 * 4) as u64);
        let hs = crate::stage_buffer(&self.ctx.device, &mut enc, &h_buf, (n * 144 * 4) as u64);
        self.ctx.queue.submit(Some(enc.finish()));

        let energy = read_f32(self.ctx, &es)?;
        let gflat = read_f32(self.ctx, &gs)?;
        let hflat = read_f32(self.ctx, &hs)?;
        let grad = (0..n)
            .map(|i| {
                let mut g = [0.0f32; 12];
                g.copy_from_slice(&gflat[12 * i..12 * i + 12]);
                g
            })
            .collect();
        let hess = (0..n)
            .map(|i| {
                let mut h = [0.0f32; 144];
                h.copy_from_slice(&hflat[144 * i..144 * i + 144]);
                h
            })
            .collect();
        Ok(TetOut { energy, grad, hess })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::tet::{build_tets, tet_element_assembly, TetModel};

    #[test]
    fn gpu_tet_elements_match_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // A deformed 3×3×3 block → 40 tets with a spread of deformation states.
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
        let els = build_tets(&tet_idx, &rest, 1e-9, TetModel::StVK);
        assert!(els.len() >= 30);
        // Deform: stretch + shear + twist so elements see varied σ.
        let x: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| {
                Vec3::new(
                    p.x * 1.15 + 0.05 * (i as f32 * 0.3).sin(),
                    p.y * 0.92 + 0.04 * (i as f32 * 0.7).cos(),
                    p.z * 1.05 + 0.03 * (i as f32 * 0.5).sin(),
                )
            })
            .collect();
        let (mu, lambda) = (100.0f32, 200.0f32);
        let out = GpuTet::new(&ctx)
            .elements(&els, &x, mu, lambda)
            .expect("gpu tet");

        for (i, el) in els.iter().enumerate() {
            let p = [
                x[el.v[0] as usize],
                x[el.v[1] as usize],
                x[el.v[2] as usize],
                x[el.v[3] as usize],
            ];
            let (ce, cg, ch) = tet_element_assembly(&p, el, mu, lambda);
            // Energy.
            assert!(
                (out.energy[i] - ce).abs() < 1e-3 * ce.abs().max(1.0),
                "energy mismatch tet {i}: gpu {} cpu {ce}",
                out.energy[i]
            );
            // Gradient.
            let gscale = cg.iter().map(|v| v.length()).fold(1.0f32, f32::max);
            for (a, cga) in cg.iter().enumerate() {
                for c in 0..3 {
                    let g = out.grad[i][3 * a + c];
                    assert!(
                        (g - cga[c]).abs() < 5e-3 * gscale,
                        "grad mismatch tet {i} vert {a} comp {c}: gpu {g} cpu {}",
                        cga[c]
                    );
                }
            }
            // Hessian.
            let hscale = ch.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            for (k, chk) in ch.iter().enumerate() {
                assert!(
                    (out.hess[i][k] - chk).abs() < 5e-3 * hscale,
                    "hess mismatch tet {i} entry {k}: gpu {} cpu {chk}",
                    out.hess[i][k]
                );
            }
        }
    }
}
