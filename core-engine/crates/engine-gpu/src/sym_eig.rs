// SPDX-License-Identifier: GPL-2.0-or-later
//! 3×3 symmetric eigenanalysis + PSD projection (`shaders/sym_eig.wgsl`) — the GPU
//! device function the reference uses to filter Hessian blocks to positive-
//! semidefinite via eigenanalysis. Cyclic Jacobi (fixed sweeps). CPU oracle
//! [`sym_eig3_cpu`] mirrors the WGSL exactly. Verified GPU≈CPU on Metal (Jacobi
//! uses `sqrt`/`atan2`, so agreement is to float tolerance, not bit-exact).

use crate::read_f32;
use crate::{GpuContext, GpuError};
use wgpu::util::DeviceExt;

const SWEEPS: usize = 12;
type M3 = [f32; 9]; // row-major

fn at(a: &M3, r: usize, c: usize) -> f32 {
    a[3 * r + c]
}

/// Symmetric-3×3 eigendecomposition via cyclic Jacobi. Returns
/// `(eigenvalues[3], eigenvectors)` where the eigenvectors are the columns of the
/// returned row-major matrix: `A = V·diag(λ)·Vᵀ`. Mirrors the WGSL kernel.
pub fn sym_eig3_cpu(input: &M3) -> ([f32; 3], M3) {
    let mut a = *input;
    let mut v: M3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let rotate = |a: &mut M3, v: &mut M3, p: usize, q: usize| {
        let apq = a[3 * p + q];
        if apq.abs() < 1e-20 {
            return;
        }
        let app = a[3 * p + p];
        let aqq = a[3 * q + q];
        let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (c, s) = (phi.cos(), phi.sin());
        for k in 0..3 {
            let akp = a[3 * k + p];
            let akq = a[3 * k + q];
            a[3 * k + p] = c * akp - s * akq;
            a[3 * k + q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let apk = a[3 * p + k];
            let aqk = a[3 * q + k];
            a[3 * p + k] = c * apk - s * aqk;
            a[3 * q + k] = s * apk + c * aqk;
        }
        for k in 0..3 {
            let vkp = v[3 * k + p];
            let vkq = v[3 * k + q];
            v[3 * k + p] = c * vkp - s * vkq;
            v[3 * k + q] = s * vkp + c * vkq;
        }
    };
    for _ in 0..SWEEPS {
        rotate(&mut a, &mut v, 0, 1);
        rotate(&mut a, &mut v, 0, 2);
        rotate(&mut a, &mut v, 1, 2);
    }
    ([a[0], a[4], a[8]], v)
}

/// Nearest PSD matrix (clamp negative eigenvalues to 0, reconstruct). The
/// Hessian-filtering operation.
pub fn project_psd3_cpu(input: &M3) -> M3 {
    let (l, v) = sym_eig3_cpu(input);
    let c = [l[0].max(0.0), l[1].max(0.0), l[2].max(0.0)];
    let mut out = [0.0f32; 9];
    for r in 0..3 {
        for col in 0..3 {
            out[3 * r + col] = at(&v, r, 0) * c[0] * at(&v, col, 0)
                + at(&v, r, 1) * c[1] * at(&v, col, 1)
                + at(&v, r, 2) * c[2] * at(&v, col, 2);
        }
    }
    out
}

/// GPU batch symmetric-eigen engine.
pub struct GpuSymEig<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuSymEig<'a> {
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
            label: Some("sym_eig.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, false, false),
                buf(3, false, false),
            ],
        });
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sym_eig"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sym_eig.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sym_eig.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sym_eig"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("eig_project"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// For each symmetric 3×3 (row-major, 9 floats): returns its eigenvalues and
    /// its PSD projection. `mats.len()` must be a multiple of 9.
    pub fn eig_project(&self, mats: &[f32]) -> Result<(Vec<[f32; 3]>, Vec<M3>), GpuError> {
        assert_eq!(
            mats.len() % 9,
            0,
            "eig_project: length must be a multiple of 9"
        );
        let count = mats.len() / 9;
        if count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let dev = &self.ctx.device;
        let params = [count as u32, 0u32, 0u32, 0u32];
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sym_eig.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let in_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sym_eig.in"),
            contents: bytemuck::cast_slice(mats),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let eig_bytes = (count * 3 * 4) as u64;
        let proj_bytes = (count * 9 * 4) as u64;
        let eig_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sym_eig.eig"),
            size: eig_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let proj_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sym_eig.proj"),
            size: proj_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sym_eig.bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: eig_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: proj_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sym_eig"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sym_eig.cp"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        }
        let eig_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &eig_buf, eig_bytes);
        let proj_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &proj_buf, proj_bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let eig_raw = read_f32(self.ctx, &eig_stage)?;
        let proj_raw = read_f32(self.ctx, &proj_stage)?;
        let eigs = (0..count)
            .map(|i| [eig_raw[3 * i], eig_raw[3 * i + 1], eig_raw[3 * i + 2]])
            .collect();
        let projs = (0..count)
            .map(|i| {
                let mut m = [0.0f32; 9];
                m.copy_from_slice(&proj_raw[9 * i..9 * i + 9]);
                m
            })
            .collect();
        Ok((eigs, projs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: [f32; 3]) -> [f32; 3] {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    // Symmetric matrix from a deterministic seed, spanning indefinite cases.
    fn sym(seed: u64) -> M3 {
        let r = |k: u64| {
            let x = (seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(k * 1442695040888963407))
                >> 33;
            (x as f32 / u32::MAX as f32) * 4.0 - 2.0 // in [-2, 2]
        };
        let (d0, d1, d2) = (r(1), r(2), r(3));
        let (o01, o02, o12) = (r(4), r(5), r(6));
        [d0, o01, o02, o01, d1, o12, o02, o12, d2]
    }

    fn recon(l: &[f32; 3], v: &M3) -> M3 {
        let mut out = [0.0f32; 9];
        for row in 0..3 {
            for col in 0..3 {
                out[3 * row + col] = at(v, row, 0) * l[0] * at(v, col, 0)
                    + at(v, row, 1) * l[1] * at(v, col, 1)
                    + at(v, row, 2) * l[2] * at(v, col, 2);
            }
        }
        out
    }

    #[test]
    fn cpu_eig_reconstructs_and_is_orthonormal() {
        for seed in 0..200u64 {
            let a = sym(seed);
            let (l, v) = sym_eig3_cpu(&a);
            let rc = recon(&l, &v);
            for i in 0..9 {
                assert!(
                    (rc[i] - a[i]).abs() < 1e-3,
                    "recon fail seed={seed}: {rc:?} vs {a:?}"
                );
            }
            // Columns orthonormal: VᵀV ≈ I.
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f32 = (0..3).map(|k| at(&v, k, i) * at(&v, k, j)).sum();
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expect).abs() < 1e-3, "not orthonormal seed={seed}");
                }
            }
        }
    }

    #[test]
    fn cpu_psd_projection_is_psd_and_idempotent_on_psd() {
        for seed in 0..200u64 {
            let a = sym(seed);
            let proj = project_psd3_cpu(&a);
            // Projected matrix must have all eigenvalues ≥ 0.
            let (lp, _) = sym_eig3_cpu(&proj);
            for &e in &lp {
                assert!(e > -1e-3, "projection not PSD seed={seed}: {lp:?}");
            }
            // Idempotent: projecting an already-PSD matrix is (near) a no-op.
            let proj2 = project_psd3_cpu(&proj);
            for i in 0..9 {
                assert!(
                    (proj2[i] - proj[i]).abs() < 1e-3,
                    "not idempotent seed={seed}"
                );
            }
        }
    }

    #[test]
    fn gpu_sym_eig_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mats: Vec<f32> = (0..300u64).flat_map(sym).collect();
        let (geig, gproj) = GpuSymEig::new(&ctx).eig_project(&mats).expect("gpu eig");
        for (i, chunk) in mats.chunks(9).enumerate() {
            let mut a = [0.0f32; 9];
            a.copy_from_slice(chunk);
            let (cl, _) = sym_eig3_cpu(&a);
            // Eigenvalues (order-independent) match to float tolerance.
            let (gs, cs) = (sorted(geig[i]), sorted(cl));
            for k in 0..3 {
                assert!(
                    (gs[k] - cs[k]).abs() < 1e-3,
                    "eig mismatch i={i}: {gs:?} vs {cs:?}"
                );
            }
            // GPU projection reconstructs A's PSD part: symmetric + all eigenvalues ≥ 0.
            let cproj = project_psd3_cpu(&a);
            for k in 0..9 {
                assert!((gproj[i][k] - cproj[k]).abs() < 1e-3, "proj mismatch i={i}");
            }
        }
    }
}
