// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU exclusive prefix-sum (scan) over `u32` — a self-contained, verifiable GPU
//! primitive (three-pass multi-block Hillis-Steele, `shaders/scan.wgsl`). Building
//! block for the LBVH broad phase (compaction / histogram offsets). Verified
//! GPU==CPU on Metal. Supports `n ≤ 65536` (one block-sum level, WG=256); larger
//! inputs need a recursive block-sum pass (documented follow-up).

use crate::{GpuContext, GpuError};
use wgpu::util::DeviceExt;

const WG: usize = 256;

/// GPU exclusive-scan engine. Build once, `exclusive_scan` many times.
pub struct GpuScan<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    scan_blocks: wgpu::ComputePipeline,
    scan_block_totals: wgpu::ComputePipeline,
    add_block_offsets: wgpu::ComputePipeline,
}

impl<'a> GpuScan<'a> {
    pub fn new(ctx: &'a GpuContext) -> Self {
        let dev = &ctx.device;
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scan.wgsl").into()),
        });
        // 5 bindings shared by all three passes (uniform + 4 storage).
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
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scan.bgl"),
            entries: &[
                entry(0, true, true),   // params
                entry(1, true, false),  // input (read)
                entry(2, false, false), // output
                entry(3, false, false), // block_total
                entry(4, false, false), // block_offset
            ],
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let mk = |ep: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            scan_blocks: mk("scan_blocks"),
            scan_block_totals: mk("scan_block_totals"),
            add_block_offsets: mk("add_block_offsets"),
            layout,
            ctx,
        }
    }

    /// Exclusive prefix sum: `out[i] = sum(input[0..i])`, `out[0] = 0`. Errors if
    /// `input.len() > 65536` (the single-block-sum-level limit).
    pub fn exclusive_scan(&self, input: &[u32]) -> Result<Vec<u32>, GpuError> {
        let n = input.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let num_blocks = n.div_ceil(WG);
        if num_blocks > WG {
            return Err(GpuError::Readback(format!(
                "scan supports n ≤ {} (got {n}); recursive block-sum is a follow-up",
                WG * WG
            )));
        }
        let dev = &self.ctx.device;
        let params = [n as u32, num_blocks as u32, 0u32, 0u32];
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scan.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let input_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scan.input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = (n * 4) as u64;
        let output_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan.output"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bt_bytes = (num_blocks.max(1) * 4) as u64;
        let block_total = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan.block_total"),
            size: bt_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let block_offset = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan.block_offset"),
            size: bt_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan.staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan.bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: block_total.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: block_offset.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scan"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan.pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind, &[]);
            pass.set_pipeline(&self.scan_blocks);
            pass.dispatch_workgroups(num_blocks as u32, 1, 1);
            pass.set_pipeline(&self.scan_block_totals);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.add_block_offsets);
            pass.dispatch_workgroups(num_blocks as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, out_bytes);
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

/// CPU exclusive scan — the verification oracle.
pub fn exclusive_scan_cpu(input: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    input
        .iter()
        .map(|&x| {
            let prev = acc;
            acc = acc.wrapping_add(x);
            prev
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_exclusive_scan_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let scan = GpuScan::new(&ctx);
        // Sizes crossing block boundaries + non-multiples of the workgroup size.
        for &n in &[1usize, 7, 256, 257, 1000, 4096, 5000, 65536] {
            let input: Vec<u32> = (0..n).map(|i| ((i * 2654435761) % 97) as u32).collect();
            let gpu = scan.exclusive_scan(&input).expect("gpu scan");
            let cpu = exclusive_scan_cpu(&input);
            assert_eq!(gpu, cpu, "scan mismatch at n={n}");
        }
        // Over the supported size → clean error, not a wrong answer.
        assert!(scan.exclusive_scan(&vec![1u32; 65537]).is_err());
    }

    #[test]
    fn cpu_scan_is_exclusive() {
        assert_eq!(exclusive_scan_cpu(&[3, 1, 4, 1, 5]), vec![0, 3, 4, 8, 9]);
        assert_eq!(exclusive_scan_cpu(&[]), Vec::<u32>::new());
    }
}
