// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU f32 sum reduction with a `subgroupAdd` fast-path (`shaders/subgroup_reduce.wgsl`)
//! and a shared-memory tree fallback (`shaders/tree_reduce.wgsl`), chosen at build
//! time from [`GpuContext::has_subgroups`]. Both use Kahan-compensated per-thread
//! accumulation. Two dispatches (data → per-workgroup partials → scalar). Verified
//! GPU≈CPU on Metal.

use crate::{GpuContext, GpuError};
use wgpu::util::DeviceExt;

const WG: usize = 256;
const MAX_GROUPS: usize = 256;

/// The `subgroupAdd` fast-path (`shaders/subgroup_reduce.wgsl`) is written and the
/// hardware `SUBGROUP` feature is detected/requested, but Naga (wgpu 27's WGSL
/// frontend) does NOT yet implement the `enable subgroups;` extension
/// (gfx-rs/wgpu#5555) — the shader fails to compile. Until Naga lands WGSL
/// subgroup ops, we run the Kahan tree-reduce fallback. Flip this to `true` (and
/// the shader compiles) once the toolchain supports it. Verified honestly: this
/// host reports SUBGROUP=true but the WGSL path is toolchain-blocked, not shipped.
const SUBGROUP_WGSL_SUPPORTED: bool = false;

/// GPU sum-reduction engine. Uses subgroups when available, else a tree reduce.
pub struct GpuReduce<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    /// True when the subgroup fast-path shader is in use.
    pub subgroup_path: bool,
}

impl<'a> GpuReduce<'a> {
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
            label: Some("reduce.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, false, false),
            ],
        });
        let use_subgroup = ctx.has_subgroups && SUBGROUP_WGSL_SUPPORTED;
        let src = if use_subgroup {
            include_str!("shaders/subgroup_reduce.wgsl")
        } else {
            include_str!("shaders/tree_reduce.wgsl")
        };
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduce"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reduce.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reduce"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("reduce"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            subgroup_path: use_subgroup,
            ctx,
        }
    }

    /// Sum of `data` (two-level GPU reduction).
    pub fn sum(&self, data: &[f32]) -> Result<f32, GpuError> {
        let n = data.len();
        if n == 0 {
            return Ok(0.0);
        }
        let dev = &self.ctx.device;
        let n_groups = n.div_ceil(WG).clamp(1, MAX_GROUPS);

        let input = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reduce.input"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mk = |label, len: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let partials = mk("reduce.partials", n_groups);
        let final_buf = mk("reduce.final", 1);

        // Pass 1: data (n elements, total_threads = n_groups*256) → partials[n_groups].
        let p1 = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reduce.p1"),
            contents: bytemuck::cast_slice(&[n as u32, (n_groups * WG) as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Pass 2: partials (n_groups elements, total_threads = 256) → final[1].
        let p2 = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reduce.p2"),
            contents: bytemuck::cast_slice(&[n_groups as u32, WG as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = |params: &wgpu::Buffer, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reduce.bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                ],
            })
        };
        let b1 = bind(&p1, &input, &partials);
        let b2 = bind(&p2, &partials, &final_buf);

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduce"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce.cp"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline);
            cp.set_bind_group(0, &b1, &[]);
            cp.dispatch_workgroups(n_groups as u32, 1, 1);
            cp.set_bind_group(0, &b2, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce.staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&final_buf, 0, &staging, 0, 4);
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
        let out: f32 = bytemuck::cast_slice(&data)[0];
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_reduce_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let reduce = GpuReduce::new(&ctx);
        eprintln!("subgroup_path = {}", reduce.subgroup_path);
        for &n in &[1usize, 7, 256, 257, 1000, 65536, 200000] {
            // Mixed-magnitude values that stress f32 summation.
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    let s = if i % 2 == 0 { 1.0 } else { -1.0 };
                    s * (((i * 2654435761) % 1000) as f32 / 7.0 + 0.5)
                })
                .collect();
            let gpu = reduce.sum(&data).expect("gpu reduce");
            let cpu: f64 = data.iter().map(|&x| x as f64).sum();
            let tol = 1e-3 * (cpu.abs() as f32).max(1.0);
            assert!(
                (gpu - cpu as f32).abs() < tol,
                "n={n}: gpu={gpu} cpu={cpu} tol={tol}"
            );
        }
    }
}
