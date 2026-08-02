// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU isotropic **shell** element kernel (`shaders/shell.wgsl`) — per-element energy,
//! analytic 9-dof gradient (Piola stress) and analytic 9×9 Hessian (the six-eigenpair
//! isotropic eigensystem), mirroring `engine_core::shell_model`. One thread per triangle,
//! one 3×2 SVD each, no finite differences.
//!
//! This closes the last element type without a GPU path: membrane, bending and tets
//! already had one, so the isotropic shell energy was CPU-only — which is what made
//! trajectory validation of the shell path impractical (~25 s/step on this host).
//!
//! The per-element output layout (9 gradient + 81 Hessian per triangle) is deliberately
//! identical to the membrane kernel's, so `membrane_gather.wgsl` scatters shell elements
//! into the CSR without a new gather kernel.
//!
//! This type is the per-element VERIFICATION harness (it owns the GPU==CPU test). The
//! production path is `unified_asm::ShellPass`, which drives the same `shaders/shell.wgsl`
//! with the shared `egrad`/`ehess` buffers and its own `out_base`. One shader, two callers
//! — the shader stays the single source of truth for the math.

use crate::read_f32;
use crate::{GpuContext, GpuError};
use engine_core::shell_model::ShellElement;
use engine_core::tet::TetModel;
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Per-shell element output: energy, 9-dof gradient, row-major 9×9 Hessian.
pub struct ShellOut {
    pub energy: Vec<f32>,
    pub grad: Vec<[f32; 9]>,
    pub hess: Vec<[f32; 81]>,
}

/// GPU shell element engine.
pub struct GpuShell<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuShell<'a> {
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
            label: Some("shell.bgl"),
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
            label: Some("shell"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shell.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shell.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shell"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("shell"),
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
    ///
    /// `eps` is the degenerate-eigenvalue guard (`ImplicitParams::eiganalysis_eps`); it
    /// must match the CPU value or the flip mode's limit form kicks in at a different
    /// point and the two paths diverge on near-equal singular values.
    ///
    /// All elements are evaluated with the SAME model: the kernel takes one `model` wire
    /// value, so a mixed-model set must be dispatched per model. (`build_shells` assigns
    /// one model to the whole batch, so this matches how they are built.)
    pub fn elements(
        &self,
        shells: &[ShellElement],
        positions: &[Vec3],
        mu: f32,
        lambda: f32,
        model: TetModel,
        eps: f32,
    ) -> Result<ShellOut, GpuError> {
        let n = shells.len();
        if n == 0 {
            return Ok(ShellOut {
                energy: Vec::new(),
                grad: Vec::new(),
                hess: Vec::new(),
            });
        }
        let dev = &self.ctx.device;
        let idx: Vec<u32> = shells.iter().flat_map(|s| s.v).collect();
        // Column-major 2×2 (glam `Mat2::from_cols`), matching the shader's vec4 layout.
        let dm: Vec<f32> = shells
            .iter()
            .flat_map(|s| {
                let m = s.dm_inv;
                [m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y]
            })
            .collect();
        let area: Vec<f32> = shells.iter().map(|s| s.area).collect();
        let pos: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();

        // 0 = ARAP, 1 = StVK — an explicit wire value, not a Rust discriminant, so the
        // shader cannot silently drift if the enum is reordered.
        let model_id: u32 = match model {
            TetModel::Arap => 0,
            TetModel::StVK => 1,
            TetModel::SnHk => 2,
        };
        let params = [
            (n as u32).to_le_bytes(),
            mu.to_le_bytes(),
            lambda.to_le_bytes(),
            eps.to_le_bytes(),
            model_id.to_le_bytes(),
            0u32.to_le_bytes(), // out_base: this harness always writes from 0
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
        ]
        .concat();
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shell.params"),
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
        let idx_buf = ro("shell.idx", bytemuck::cast_slice(&idx));
        let dm_buf = ro("shell.dm", bytemuck::cast_slice(&dm));
        let area_buf = ro("shell.area", bytemuck::cast_slice(&area));
        let pos_buf = ro("shell.pos", bytemuck::cast_slice(&pos));
        let mk = |label, len: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let e_buf = mk("shell.energy", n);
        let g_buf = mk("shell.grad", n * 9);
        let h_buf = mk("shell.hess", n * 81);
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shell.bind"),
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
                    resource: area_buf.as_entire_binding(),
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
            label: Some("shell"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shell.cp"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }
        let es = crate::stage_buffer(&self.ctx.device, &mut enc, &e_buf, (n * 4) as u64);
        let gs = crate::stage_buffer(&self.ctx.device, &mut enc, &g_buf, (n * 9 * 4) as u64);
        let hs = crate::stage_buffer(&self.ctx.device, &mut enc, &h_buf, (n * 81 * 4) as u64);
        self.ctx.queue.submit(Some(enc.finish()));

        let energy = read_f32(self.ctx, &es)?;
        let gflat = read_f32(self.ctx, &gs)?;
        let hflat = read_f32(self.ctx, &hs)?;
        let grad = (0..n)
            .map(|i| {
                let mut g = [0.0f32; 9];
                g.copy_from_slice(&gflat[9 * i..9 * i + 9]);
                g
            })
            .collect();
        let hess = (0..n)
            .map(|i| {
                let mut h = [0.0f32; 81];
                h.copy_from_slice(&hflat[81 * i..81 * i + 81]);
                h
            })
            .collect();
        Ok(ShellOut { energy, grad, hess })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::shell_model::{
        build_shells, shell_energy, shell_gradient_analytic, shell_hessian_analytic,
    };

    /// A deformed sheet, so elements see a spread of principal stretches (stretched,
    /// compressed, sheared) rather than one easy state.
    fn scene() -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<Vec3>) {
        let nn = 6usize;
        let mut rest = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                rest.push(Vec3::new(i as f32 * 0.4, 0.0, j as f32 * 0.4));
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
        // Stretch in x, compress in z, shear, and lift out of plane so the third
        // (out-of-plane) eigen-directions are genuinely exercised.
        let x: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let f = i as f32;
                Vec3::new(
                    p.x * 1.18 + 0.05 * (f * 0.3).sin() + 0.06 * p.z,
                    0.10 * (f * 0.45).sin() + 0.04 * (f * 0.11).cos(),
                    p.z * 0.88 + 0.04 * (f * 0.7).cos(),
                )
            })
            .collect();
        (rest, tris, x)
    }

    /// The GPU shell kernel must reproduce the CPU element math — energy, 9-dof gradient
    /// AND the full 9×9 Hessian — for both constitutive models. This is the standing
    /// GPU==CPU requirement; the Hessian is the part most likely to diverge, since it
    /// carries the eigensystem, the PSD clamp and two layout conventions.
    #[test]
    fn gpu_shell_elements_match_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (rest, tris, x) = scene();
        let eps = 1.0e-6f32;
        for model in [TetModel::Arap, TetModel::StVK, TetModel::SnHk] {
            let els = build_shells(&tris, &rest, 1.0e-12, model);
            assert!(els.len() >= 40, "need a real batch, got {}", els.len());
            let (mu, lambda) = (100.0f32, 200.0f32);
            let out = GpuShell::new(&ctx)
                .elements(&els, &x, mu, lambda, model, eps)
                .expect("gpu shell");
            assert_eq!(out.grad.len(), els.len());

            let mut worst_e = 0.0f32;
            let mut worst_g = 0.0f32;
            let mut worst_h = 0.0f32;
            let mut scale_g = 0.0f32;
            let mut scale_h = 0.0f32;
            for (i, el) in els.iter().enumerate() {
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                ];
                let ce = shell_energy(&p, el, mu, lambda);
                let cg = shell_gradient_analytic(&p, el, mu, lambda);
                let ch = shell_hessian_analytic(&p, el, mu, lambda, eps);
                worst_e = worst_e.max((ce - out.energy[i]).abs() / ce.abs().max(1.0));
                for (a, gvec) in cg.iter().enumerate() {
                    for c in 0..3 {
                        let gv = gvec[c];
                        worst_g = worst_g.max((gv - out.grad[i][3 * a + c]).abs());
                        scale_g = scale_g.max(gv.abs());
                    }
                }
                for (k, &hv) in ch.iter().enumerate() {
                    worst_h = worst_h.max((hv - out.hess[i][k]).abs());
                    scale_h = scale_h.max(hv.abs());
                }
            }
            assert!(scale_g > 0.0 && scale_h > 0.0, "{model:?}: degenerate test");
            // f32 on two different execution paths: relative agreement, not bitwise.
            assert!(
                worst_e < 1.0e-4,
                "{model:?}: energy mismatch, worst rel {worst_e:.3e}"
            );
            assert!(
                worst_g / scale_g < 1.0e-3,
                "{model:?}: gradient mismatch, worst {worst_g:.3e} on scale {scale_g:.3e}"
            );
            assert!(
                worst_h / scale_h < 1.0e-3,
                "{model:?}: Hessian mismatch, worst {worst_h:.3e} on scale {scale_h:.3e}"
            );
            eprintln!(
                "{model:?}: {} elements, rel err energy {:.2e} grad {:.2e} hess {:.2e}",
                els.len(),
                worst_e,
                worst_g / scale_g,
                worst_h / scale_h
            );
        }
    }

    /// An empty batch must be a no-op rather than a dispatch of zero workgroups.
    #[test]
    fn gpu_shell_empty_batch_is_a_noop() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let out = GpuShell::new(&ctx)
            .elements(&[], &[], 1.0, 1.0, TetModel::Arap, 1.0e-6)
            .expect("empty");
        assert!(out.energy.is_empty() && out.grad.is_empty() && out.hess.is_empty());
    }
}
