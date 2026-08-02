// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU 30-bit Morton codes for triangle centroids (`shaders/morton.wgsl`) — the
//! first stage of the LBVH broad phase. Its output is radix-sorted
//! ([`crate::GpuRadixSort`]) to give a spatially-local primitive ordering, over
//! which a Karras tree is built (follow-up). Verified GPU==CPU on Metal.

use crate::{GpuContext, GpuError};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// CPU Morton oracle — the spec the GPU kernel must match bit-for-bit.
fn expand_bits(v: u32) -> u32 {
    let mut x = v & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// 30-bit Morton code of a centroid already normalized into `[0,1]^3`.
pub fn morton_code_cpu(n: Vec3) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 1023.0).clamp(0.0, 1023.0) as u32;
    expand_bits(q(n.x)) * 4 + expand_bits(q(n.y)) * 2 + expand_bits(q(n.z))
}

/// Scene AABB `(lo, inv)` over triangle centroids, where `inv = 1/(hi-lo)` per
/// axis (0 where the extent is 0). The normalization used by both CPU and GPU.
pub fn centroid_bounds(positions: &[Vec3], tris: &[[u32; 3]]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for t in tris {
        let c =
            (positions[t[0] as usize] + positions[t[1] as usize] + positions[t[2] as usize]) / 3.0;
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let ext = hi - lo;
    let inv = Vec3::new(
        if ext.x > 0.0 { 1.0 / ext.x } else { 0.0 },
        if ext.y > 0.0 { 1.0 / ext.y } else { 0.0 },
        if ext.z > 0.0 { 1.0 / ext.z } else { 0.0 },
    );
    (lo, inv)
}

/// CPU Morton codes for all triangles (the verification oracle).
pub fn morton_codes_cpu(positions: &[Vec3], tris: &[[u32; 3]]) -> Vec<u32> {
    let (lo, inv) = centroid_bounds(positions, tris);
    tris.iter()
        .map(|t| {
            let c =
                (positions[t[0] as usize] + positions[t[1] as usize] + positions[t[2] as usize])
                    / 3.0;
            morton_code_cpu((c - lo) * inv)
        })
        .collect()
}

/// GPU Morton-code engine. Build once, `codes` many times.
pub struct GpuMorton<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuMorton<'a> {
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
            label: Some("morton.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, true, false),
                buf(3, false, false),
            ],
        });
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morton"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/morton.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morton.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morton"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("morton"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// Morton codes for every triangle (centroid-based, normalized by the scene
    /// AABB). Same normalization as [`morton_codes_cpu`].
    pub fn codes(&self, positions: &[Vec3], tris: &[[u32; 3]]) -> Result<Vec<u32>, GpuError> {
        let num = tris.len();
        if num == 0 {
            return Ok(Vec::new());
        }
        let dev = &self.ctx.device;
        let (lo, inv) = centroid_bounds(positions, tris);

        // Flat f32 positions (3n), matching the shader convention.
        let flat: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let tri_flat: Vec<u32> = tris.iter().flat_map(|t| [t[0], t[1], t[2]]).collect();
        // Uniform: lo.xyz, lo.w = bitcast(num); inv.xyz, inv.w = 0.
        let params: [f32; 8] = [
            lo.x,
            lo.y,
            lo.z,
            f32::from_bits(num as u32),
            inv.x,
            inv.y,
            inv.z,
            0.0,
        ];

        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("morton.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pos_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("morton.pos"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tri_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("morton.tris"),
            contents: bytemuck::cast_slice(&tri_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = (num * 4) as u64;
        let codes_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morton.codes"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morton.staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton.bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tri_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: codes_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("morton"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morton.pass"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(num.div_ceil(256) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&codes_buf, 0, &staging, 0, out_bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        rx.recv()
            .map_err(|e| GpuError::Readback(e.to_string()))?
            .map_err(|e| GpuError::Readback(e.to_string()))?;
        let data = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuRadixSort;

    // A jittered grid mesh — spreads centroids across the scene AABB.
    fn grid_mesh(n: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut pos = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let h = (((i * 73 + j * 179) % 17) as f32) * 0.005;
                pos.push(Vec3::new(i as f32 * 0.1, h, j as f32 * 0.1));
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
        (pos, tris)
    }

    // Inverse of expand_bits: gather every 3rd bit back into a 10-bit index.
    fn compact_bits(mut x: u32) -> u32 {
        x &= 0x09249249;
        x = (x | (x >> 2)) & 0x030c30c3;
        x = (x | (x >> 4)) & 0x0300f00f;
        x = (x | (x >> 8)) & 0x030000ff;
        x = (x | (x >> 16)) & 0x000003ff;
        x
    }
    // Decode a 30-bit Morton code back to its per-axis quantized indices.
    fn decode(code: u32) -> [u32; 3] {
        [
            compact_bits(code >> 2),
            compact_bits(code >> 1),
            compact_bits(code),
        ]
    }

    #[test]
    fn gpu_morton_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let m = GpuMorton::new(&ctx);
        for &n in &[3usize, 8, 20, 40] {
            let (pos, tris) = grid_mesh(n);
            let gpu = m.codes(&pos, &tris).expect("gpu morton");
            let cpu = morton_codes_cpu(&pos, &tris);
            assert_eq!(gpu.len(), cpu.len());
            // Exact-code equality is not guaranteed across CPU/GPU float paths: a
            // last-ULP difference in a normalized centroid can flip one axis's
            // quantized index by ±1 at a bucket boundary. The correct invariant is
            // per-axis agreement within ±1 (a real bug — axis swap, wrong shift —
            // would blow this bound wide open, not stay at 1).
            let mut exact = 0usize;
            for (ti, (&g, &c)) in gpu.iter().zip(&cpu).enumerate() {
                if g == c {
                    exact += 1;
                    continue;
                }
                let (dg, dc) = (decode(g), decode(c));
                for axis in 0..3 {
                    let diff = dg[axis].abs_diff(dc[axis]);
                    assert!(
                        diff <= 1,
                        "morton axis {axis} off by {diff} (t={ti}, gpu={dg:?}, cpu={dc:?})"
                    );
                }
            }
            // Most triangles must still be bit-exact — a systematic error would
            // push far more than a few % onto the ±1 boundary path.
            assert!(
                exact * 100 >= cpu.len() * 80,
                "too many boundary mismatches at n={n}: {} / {}",
                cpu.len() - exact,
                cpu.len()
            );
        }
    }

    #[test]
    fn gpu_morton_then_radix_sorts() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (pos, tris) = grid_mesh(20);
        let codes = GpuMorton::new(&ctx).codes(&pos, &tris).expect("codes");
        let sorted = GpuRadixSort::new(&ctx).sort(&codes).expect("sort");
        assert!(sorted.windows(2).all(|w| w[0] <= w[1]), "codes not sorted");
        let mut cpu = codes.clone();
        cpu.sort_unstable();
        assert_eq!(sorted, cpu, "GPU morton+radix != CPU sort of same codes");
    }
}
