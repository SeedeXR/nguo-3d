// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU discrete-shell bending embed kernel (`shaders/bending.wgsl`) — per-hinge
//! dihedral angle, analytic ∂θ/∂x, bending energy and its per-vertex gradient.
//! Mirrors `engine_core::bending` (the CPU reference). The hinge-stencil half of
//! the GPU bending assembler; the Gauss-Newton Hessian `kb·w·∇θ∇θᵀ` is assembled
//! from the returned ∇θ downstream. Verified GPU≈CPU on Metal.

use crate::read_f32;
use crate::{GpuContext, GpuError};
use engine_core::bending::Hinge;
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Per-hinge bending result: total `energy`, the 4 per-vertex energy gradients
/// `∇E` (order `[opp_a, s0, s1, opp_b]`), and the 12×12 Gauss-Newton Hessian block
/// (row-major, `kb·w·∇θ∇θᵀ`), which the CSR gather assembler scatters.
pub struct BendingOut {
    pub energy: Vec<f32>,
    pub grad: Vec<[Vec3; 4]>,
    pub hess: Vec<[f32; 144]>,
}

/// GPU bending embed engine.
pub struct GpuBending<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuBending<'a> {
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
            label: Some("bending.bgl"),
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
            label: Some("bending"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bending.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bending.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bending"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("bending"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// Per-hinge bending energy + gradient for the given `positions` and stiffness
    /// `kb`. Uses each hinge's own `rest`/`w`.
    pub fn embed(
        &self,
        hinges: &[Hinge],
        positions: &[Vec3],
        kb: f32,
    ) -> Result<BendingOut, GpuError> {
        let n = hinges.len();
        if n == 0 {
            return Ok(BendingOut {
                energy: Vec::new(),
                grad: Vec::new(),
                hess: Vec::new(),
            });
        }
        let dev = &self.ctx.device;
        let idx: Vec<u32> = hinges.iter().flat_map(|h| h.v).collect();
        let rest: Vec<f32> = hinges.iter().map(|h| h.rest).collect();
        let wt: Vec<f32> = hinges.iter().map(|h| h.w).collect();
        let pos: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();

        let params = [n as u32, kb.to_bits(), 0u32, 0u32];
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bending.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let ro = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let idx_buf = ro("bending.idx", bytemuck::cast_slice(&idx));
        let rest_buf = ro("bending.rest", bytemuck::cast_slice(&rest));
        let wt_buf = ro("bending.wt", bytemuck::cast_slice(&wt));
        let pos_buf = ro("bending.pos", bytemuck::cast_slice(&pos));
        let e_bytes = (n * 4) as u64;
        let g_bytes = (n * 12 * 4) as u64;
        let mk = |label, bytes| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let h_bytes = (n * 144 * 4) as u64;
        let e_buf = mk("bending.energy", e_bytes);
        let g_buf = mk("bending.grad", g_bytes);
        let h_buf = mk("bending.hess", h_bytes);
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bending.bind"),
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
                    resource: rest_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wt_buf.as_entire_binding(),
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

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bending"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bending.cp"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }
        let e_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &e_buf, e_bytes);
        let g_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &g_buf, g_bytes);
        let h_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &h_buf, h_bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let energy = read_f32(self.ctx, &e_stage)?;
        let gflat = read_f32(self.ctx, &g_stage)?;
        let hflat = read_f32(self.ctx, &h_stage)?;
        let grad = (0..n)
            .map(|i| {
                let b = 12 * i;
                std::array::from_fn(|k| {
                    Vec3::new(gflat[b + 3 * k], gflat[b + 3 * k + 1], gflat[b + 3 * k + 2])
                })
            })
            .collect();
        let hess = (0..n)
            .map(|i| {
                let mut m = [0.0f32; 144];
                m.copy_from_slice(&hflat[144 * i..144 * i + 144]);
                m
            })
            .collect();
        Ok(BendingOut { energy, grad, hess })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::bending::{dihedral_angle, dihedral_grad_analytic};

    // A folded, skewed cloth strip: a row of quads folded along the shared edges,
    // giving several interior hinges with nonzero, varied dihedral angles.
    fn folded_strip(n: usize, fold: f32) -> (Vec<Vec3>, Vec<Hinge>) {
        let mut pos = Vec::new();
        for j in 0..=n {
            let a = fold * (j as f32);
            // Each successive quad row tilts, so hinges bend by ~fold.
            pos.push(Vec3::new(0.0, (a).cos() * j as f32, (a).sin() * j as f32));
            pos.push(Vec3::new(1.0, (a).cos() * j as f32, (a).sin() * j as f32));
        }
        let mut tris = Vec::new();
        for j in 0..n {
            let b = (2 * j) as u32;
            tris.push([b, b + 1, b + 2]);
            tris.push([b + 2, b + 1, b + 3]);
        }
        let hinges = engine_core::bending::build_hinges(&tris, &pos);
        // Re-fold slightly so hinges are away from rest (nonzero gradient).
        let moved: Vec<Vec3> = pos
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::new(0.0, 0.0, 0.02 * (i as f32).sin()))
            .collect();
        (moved, hinges)
    }

    #[test]
    fn gpu_bending_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (pos, hinges) = folded_strip(12, 0.3);
        assert!(hinges.len() > 3, "expected several interior hinges");
        let kb = 0.7f32;
        let out = GpuBending::new(&ctx)
            .embed(&hinges, &pos, kb)
            .expect("gpu bending");

        for (hi, h) in hinges.iter().enumerate() {
            let p = [
                pos[h.v[0] as usize],
                pos[h.v[1] as usize],
                pos[h.v[2] as usize],
                pos[h.v[3] as usize],
            ];
            let theta = dihedral_angle(&p);
            let gtheta = dihedral_grad_analytic(&p);
            let d = theta - h.rest;
            let e_cpu = 0.5 * kb * h.w * d * d;
            assert!((out.energy[hi] - e_cpu).abs() < 1e-4, "energy hinge {hi}");
            for (k, gt) in gtheta.iter().enumerate() {
                let g_cpu = *gt * (kb * h.w * d);
                assert!(
                    (out.grad[hi][k] - g_cpu).length() < 1e-4,
                    "grad hinge {hi} vert {k}: gpu {:?} cpu {:?}",
                    out.grad[hi][k],
                    g_cpu
                );
            }
            // Translation invariance carries to the GPU: Σ ∇E = 0.
            let sum: Vec3 = out.grad[hi].iter().copied().sum();
            assert!(sum.length() < 1e-4, "grad sum not zero, hinge {hi}");

            // 12×12 Gauss-Newton Hessian block = kb·w·∇θ∇θᵀ.
            let hb = &out.hess[hi];
            let kw = kb * h.w;
            for a in 0..4 {
                for b in 0..4 {
                    for r in 0..3 {
                        for c in 0..3 {
                            let expect = kw * gtheta[a][r] * gtheta[b][c];
                            let got = hb[(3 * a + r) * 12 + (3 * b + c)];
                            assert!(
                                (got - expect).abs() < 1e-4,
                                "hess hinge {hi} [{a}{r}][{b}{c}]"
                            );
                        }
                    }
                }
            }
            // Symmetric.
            for i in 0..12 {
                for j in 0..12 {
                    assert!(
                        (hb[i * 12 + j] - hb[j * 12 + i]).abs() < 1e-5,
                        "H not symmetric"
                    );
                }
            }
            // PSD (rank-1 nonneg-scaled outer product): xᵀHx ≥ 0 for random x.
            for t in 0..8u64 {
                let x: [f32; 12] = std::array::from_fn(|k| {
                    (((t * 2654435761 + k as u64 * 40503) % 200) as f32) / 100.0 - 1.0
                });
                let mut q = 0.0f32;
                for i in 0..12 {
                    for j in 0..12 {
                        q += x[i] * hb[i * 12 + j] * x[j];
                    }
                }
                assert!(q >= -1e-4, "Hessian not PSD, hinge {hi}: xᵀHx={q}");
            }
        }
    }
}
