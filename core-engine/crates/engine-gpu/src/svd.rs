// SPDX-License-Identifier: GPL-2.0-or-later
//! 3×3 SVD + polar decomposition (`shaders/svd.wgsl`) — for corotational/ARAP
//! elasticity. `F = U·diag(σ)·Vᵀ` and the polar rotation `R = U·Vᵀ` forced to a
//! proper rotation (det +1) when `F` reflects. Built on the cyclic-Jacobi
//! eigensolver ([`crate::sym_eig3_cpu`], eig of `FᵀF`). CPU oracle [`svd3_cpu`]
//! mirrors the WGSL; verified GPU≈CPU on Metal (float tolerance — Jacobi + sqrt).

use crate::read_f32;
use crate::sym_eig::sym_eig3_cpu;
use crate::{GpuContext, GpuError};
use glam::Vec3;
use wgpu::util::DeviceExt;

type M3 = [f32; 9]; // row-major

/// Full 3×3 SVD/polar result. `F = U·diag(sigma)·Vᵀ`; `r` is the proper-rotation
/// polar factor (det +1). Columns of `u`/`v` are the singular vectors.
pub struct Svd3 {
    pub u: M3,
    pub sigma: [f32; 3],
    pub v: M3,
    pub r: M3,
}

fn col(m: &M3, c: usize) -> Vec3 {
    Vec3::new(m[c], m[3 + c], m[6 + c])
}
fn set_col(m: &mut M3, c: usize, v: Vec3) {
    m[c] = v.x;
    m[3 + c] = v.y;
    m[6 + c] = v.z;
}
fn matvec(m: &M3, v: Vec3) -> Vec3 {
    Vec3::new(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    )
}
fn det3(m: &M3) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}
/// `a · bᵀ` for two row-major 3×3 matrices.
fn mul_abt(a: &M3, b: &M3) -> M3 {
    let mut out = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += a[3 * i + k] * b[3 * j + k];
            }
            out[3 * i + j] = acc;
        }
    }
    out
}
/// `aᵀ · a` (symmetric).
fn ata(a: &M3) -> M3 {
    let mut out = [0.0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += a[3 * k + r] * a[3 * k + c];
            }
            out[3 * r + c] = acc;
        }
    }
    out
}

/// SVD + polar decomposition of `f`. Mirrors the WGSL kernel exactly.
pub fn svd3_cpu(f: &M3) -> Svd3 {
    let s = ata(f);
    let (lam, v) = sym_eig3_cpu(&s);
    // Sort eigenpairs descending by eigenvalue.
    let mut ord = [0usize, 1, 2];
    ord.sort_by(|&a, &b| lam[b].partial_cmp(&lam[a]).unwrap());

    let mut vs = [0.0f32; 9];
    for (c, &o) in ord.iter().enumerate() {
        set_col(&mut vs, c, col(&v, o).normalize());
    }
    let sig = Vec3::new(
        lam[ord[0]].max(0.0).sqrt(),
        lam[ord[1]].max(0.0).sqrt(),
        lam[ord[2]].max(0.0).sqrt(),
    );

    let eps = 1e-6 * sig.x.max(1.0);
    let b0 = matvec(f, col(&vs, 0));
    let b1 = matvec(f, col(&vs, 1));
    let u0 = if sig.x > eps { b0 / sig.x } else { Vec3::X }.normalize();
    let u1 = {
        let raw = if sig.y > eps { b1 / sig.y } else { Vec3::Y };
        (raw - raw.dot(u0) * u0).normalize()
    };
    // Sign-align the completed basis vector with F·v2 so reconstruction is exact.
    let u2 = {
        let c = u0.cross(u1);
        if c.dot(matvec(f, col(&vs, 2))) < 0.0 {
            -c
        } else {
            c
        }
    };
    let mut u = [0.0f32; 9];
    set_col(&mut u, 0, u0);
    set_col(&mut u, 1, u1);
    set_col(&mut u, 2, u2);

    // Proper-rotation polar factor: flip last U column if U·V reflects.
    let mut us = u;
    if det3(&u) * det3(&vs) < 0.0 {
        set_col(&mut us, 2, -u2);
    }
    let r = mul_abt(&us, &vs);
    Svd3 {
        u,
        sigma: [sig.x, sig.y, sig.z],
        v: vs,
        r,
    }
}

/// GPU batch 3×3 SVD engine.
pub struct GpuSvd<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuSvd<'a> {
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
            label: Some("svd.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, false, false),
                buf(3, false, false),
                buf(4, false, false),
                buf(5, false, false),
            ],
        });
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svd"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/svd.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("svd.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svd"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("svd"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// For each 3×3 (row-major, 9 floats): returns `(U, sigma, V, R)` per matrix.
    /// `mats.len()` must be a multiple of 9.
    #[allow(clippy::type_complexity)]
    pub fn svd(&self, mats: &[f32]) -> Result<Vec<Svd3>, GpuError> {
        assert_eq!(mats.len() % 9, 0, "svd: length must be a multiple of 9");
        let count = mats.len() / 9;
        if count == 0 {
            return Ok(Vec::new());
        }
        let dev = &self.ctx.device;
        let params = [count as u32, 0u32, 0u32, 0u32];
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("svd.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let in_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("svd.in"),
            contents: bytemuck::cast_slice(mats),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let m9 = (count * 9 * 4) as u64;
        let m3 = (count * 3 * 4) as u64;
        let out = |label, bytes| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let u_buf = out("svd.u", m9);
        let s_buf = out("svd.s", m3);
        let v_buf = out("svd.v", m9);
        let r_buf = out("svd.r", m9);
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("svd.bind"),
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
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: r_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc =
            dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("svd") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("svd.cp"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        }
        let us = crate::stage_buffer(&self.ctx.device, &mut enc, &u_buf, m9);
        let ss = crate::stage_buffer(&self.ctx.device, &mut enc, &s_buf, m3);
        let vsx = crate::stage_buffer(&self.ctx.device, &mut enc, &v_buf, m9);
        let rs = crate::stage_buffer(&self.ctx.device, &mut enc, &r_buf, m9);
        self.ctx.queue.submit(Some(enc.finish()));

        let ud = read_f32(self.ctx, &us)?;
        let sd = read_f32(self.ctx, &ss)?;
        let vd = read_f32(self.ctx, &vsx)?;
        let rd = read_f32(self.ctx, &rs)?;
        let grab = |src: &[f32], i: usize| {
            let mut m = [0.0f32; 9];
            m.copy_from_slice(&src[9 * i..9 * i + 9]);
            m
        };
        Ok((0..count)
            .map(|i| Svd3 {
                u: grab(&ud, i),
                sigma: [sd[3 * i], sd[3 * i + 1], sd[3 * i + 2]],
                v: grab(&vd, i),
                r: grab(&rd, i),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reconstruct U·diag(σ)·Vᵀ.
    fn recon(u: &M3, s: &[f32; 3], v: &M3) -> M3 {
        let mut out = [0.0f32; 9];
        for row in 0..3 {
            for c in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += u[3 * row + k] * s[k] * v[3 * c + k];
                }
                out[3 * row + c] = acc;
            }
        }
        out
    }
    fn mul(a: &M3, b: &M3) -> M3 {
        let mut out = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += a[3 * i + k] * b[3 * k + j];
                }
                out[3 * i + j] = acc;
            }
        }
        out
    }

    // Deterministic 3×3 with a controllable determinant sign.
    fn mat(seed: u64, reflect: bool) -> M3 {
        let r = |k: u64| {
            let x = (seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(k * 1442695040888963407))
                >> 33;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0 + 1.2 // biased away from 0 → invertible
        };
        let mut m = [
            r(1) + 1.5,
            r(2),
            r(3),
            r(4),
            r(5) + 1.5,
            r(6),
            r(7),
            r(8),
            r(9) + 1.5,
        ];
        if reflect {
            // Negate a column → det flips sign.
            m[2] = -m[2];
            m[5] = -m[5];
            m[8] = -m[8];
        }
        m
    }

    #[test]
    fn cpu_svd_reconstructs_and_polar_is_rotation() {
        for seed in 0..300u64 {
            for reflect in [false, true] {
                let f = mat(seed, reflect);
                let s = svd3_cpu(&f);
                // F ≈ U diag(σ) Vᵀ.
                let rc = recon(&s.u, &s.sigma, &s.v);
                for i in 0..9 {
                    assert!(
                        (rc[i] - f[i]).abs() < 2e-3,
                        "recon seed={seed} refl={reflect}"
                    );
                }
                // R is a proper rotation: RᵀR = I, det = +1.
                let rtr = mul_abt(&s.r, &s.r); // r · rᵀ
                for i in 0..3 {
                    for j in 0..3 {
                        let e = if i == j { 1.0 } else { 0.0 };
                        assert!((rtr[3 * i + j] - e).abs() < 2e-3, "R not orthonormal");
                    }
                }
                assert!((det3(&s.r) - 1.0).abs() < 2e-3, "det(R) != 1 seed={seed}");
            }
        }
    }

    #[test]
    fn cpu_polar_factor_reconstructs_f() {
        // F = R · S, S = Rᵀ F must be symmetric.
        for seed in 0..300u64 {
            for reflect in [false, true] {
                let f = mat(seed, reflect);
                let s = svd3_cpu(&f);
                // S = Rᵀ F.
                let mut stretch = [0.0f32; 9];
                for i in 0..3 {
                    for j in 0..3 {
                        let mut acc = 0.0;
                        for k in 0..3 {
                            acc += s.r[3 * k + i] * f[3 * k + j]; // (Rᵀ)_{ik} F_{kj}
                        }
                        stretch[3 * i + j] = acc;
                    }
                }
                // stretch must be symmetric.
                assert!(
                    (stretch[1] - stretch[3]).abs() < 3e-3,
                    "S not symmetric seed={seed}"
                );
                assert!((stretch[2] - stretch[6]).abs() < 3e-3);
                assert!((stretch[5] - stretch[7]).abs() < 3e-3);
                // R · S == F.
                let back = mul(&s.r, &stretch);
                for i in 0..9 {
                    assert!(
                        (back[i] - f[i]).abs() < 3e-3,
                        "R·S != F seed={seed} refl={reflect}"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_svd_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut mats = Vec::new();
        for seed in 0..200u64 {
            mats.extend_from_slice(&mat(seed, seed % 2 == 0));
        }
        let g = GpuSvd::new(&ctx).svd(&mats).expect("gpu svd");
        for (i, chunk) in mats.chunks(9).enumerate() {
            let mut f = [0.0f32; 9];
            f.copy_from_slice(chunk);
            let c = svd3_cpu(&f);
            // Singular values match (both sorted descending by construction).
            for k in 0..3 {
                assert!((g[i].sigma[k] - c.sigma[k]).abs() < 2e-3, "sigma i={i}");
            }
            // GPU R reconstructs F and is a proper rotation.
            let mut stretch = [0.0f32; 9];
            for a in 0..3 {
                for b in 0..3 {
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += g[i].r[3 * k + a] * f[3 * k + b];
                    }
                    stretch[3 * a + b] = acc;
                }
            }
            let back = mul(&g[i].r, &stretch);
            for k in 0..9 {
                assert!((back[k] - f[k]).abs() < 3e-3, "GPU R·S != F i={i}");
            }
            assert!((det3(&g[i].r) - 1.0).abs() < 3e-3, "GPU det(R)!=1 i={i}");
        }
    }
}
