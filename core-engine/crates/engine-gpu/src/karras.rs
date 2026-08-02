// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU Karras binary-radix tree over sorted Morton codes (`shaders/karras.wgsl`) —
//! the LBVH topology stage. `n` leaves + `n-1` internal nodes; each internal node's
//! two children are computed independently (Karras 2012). Children encode leaf vs
//! internal in the top bit ([`LEAF`]). Integer-only, so GPU output is bit-identical
//! to [`karras_tree_cpu`] (the reference/oracle). Verified on Metal.

use crate::{GpuContext, GpuError};
use wgpu::util::DeviceExt;

/// Top-bit flag: a child index with this bit set refers to a **leaf**, else an
/// internal node. `child & !LEAF` is the array index in either case.
pub const LEAF: u32 = 0x8000_0000;

/// Karras tree topology: `left[i]`/`right[i]` are internal node `i`'s children
/// (both length `n-1`). Empty for `n < 2` (a single leaf has no internal nodes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KarrasTree {
    pub left: Vec<u32>,
    pub right: Vec<u32>,
}

/// CPU reference tree — mirrors the WGSL exactly (i32 / `leading_zeros`).
pub fn karras_tree_cpu(codes: &[u32]) -> KarrasTree {
    let n = codes.len();
    let m = n.saturating_sub(1);
    let mut left = vec![0u32; m];
    let mut right = vec![0u32; m];
    let delta = |i: i32, j: i32| -> i32 {
        if j < 0 || j >= n as i32 {
            return -1;
        }
        let x = codes[i as usize] ^ codes[j as usize];
        if x == 0 {
            32 + ((i as u32) ^ (j as u32)).leading_zeros() as i32
        } else {
            x.leading_zeros() as i32
        }
    };
    for i in 0..m as i32 {
        let d = if delta(i, i + 1) - delta(i, i - 1) < 0 {
            -1
        } else {
            1
        };
        let delta_min = delta(i, i - d);
        let mut l_max = 2;
        while delta(i, i + l_max * d) > delta_min {
            l_max *= 2;
        }
        let mut l = 0;
        let mut t = l_max / 2;
        while t >= 1 {
            if delta(i, i + (l + t) * d) > delta_min {
                l += t;
            }
            t /= 2;
        }
        let j = i + l * d;
        let first = i.min(j);
        let last = i.max(j);
        let node_delta = delta(first, last);
        let mut split = first;
        let mut step = last - first;
        loop {
            step = (step + 1) / 2;
            let new_split = split + step;
            if new_split < last && delta(first, new_split) > node_delta {
                split = new_split;
            }
            if step <= 1 {
                break;
            }
        }
        let gamma = split;
        left[i as usize] = if first == gamma {
            gamma as u32 | LEAF
        } else {
            gamma as u32
        };
        right[i as usize] = if last == gamma + 1 {
            (gamma + 1) as u32 | LEAF
        } else {
            (gamma + 1) as u32
        };
    }
    KarrasTree { left, right }
}

/// GPU Karras-tree engine. Build once, `build_tree` many times.
pub struct GpuKarras<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuKarras<'a> {
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
            label: Some("karras.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, false, false),
                buf(3, false, false),
            ],
        });
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("karras"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/karras.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("karras.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("karras"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("build"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// Build the tree from `codes` (must be sorted ascending, as from the Morton +
    /// radix stages). Returns empty vecs for `n < 2`.
    pub fn build_tree(&self, codes: &[u32]) -> Result<KarrasTree, GpuError> {
        let n = codes.len();
        if n < 2 {
            return Ok(KarrasTree {
                left: Vec::new(),
                right: Vec::new(),
            });
        }
        let m = n - 1;
        let dev = &self.ctx.device;
        let params = [n as u32, 0u32, 0u32, 0u32];
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("karras.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let codes_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("karras.codes"),
            contents: bytemuck::cast_slice(codes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let child_bytes = (m * 4) as u64;
        let mk = |label| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: child_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let left_buf = mk("karras.left");
        let right_buf = mk("karras.right");
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras.bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: codes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: left_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: right_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("karras"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("karras.pass"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(m.div_ceil(256) as u32, 1, 1);
        }
        let left = self.readback(&mut enc, &left_buf, child_bytes);
        let right = self.readback(&mut enc, &right_buf, child_bytes);
        self.ctx.queue.submit(Some(enc.finish()));
        Ok(KarrasTree {
            left: left()?,
            right: right()?,
        })
    }

    // Copy `src` into a fresh staging buffer; returns a closure that maps+reads it
    // after the encoder is submitted (so both readbacks share one submit).
    fn readback(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        bytes: u64,
    ) -> impl FnOnce() -> Result<Vec<u32>, GpuError> + '_ {
        let dev = &self.ctx.device;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("karras.staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        move || {
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
}

/// In-order (left-first) leaf sequence of a Karras tree. For a well-formed tree
/// this is exactly `0, 1, …, n-1` — every leaf reached once, in sorted order.
pub fn inorder_leaves(tree: &KarrasTree, n: usize) -> Vec<u32> {
    if n < 2 {
        return (0..n as u32).collect();
    }
    let mut out = Vec::with_capacity(n);
    // Recursive DFS from the root (internal node 0), left child before right.
    fn visit(tree: &KarrasTree, node: u32, is_leaf: bool, out: &mut Vec<u32>) {
        if is_leaf {
            out.push(node);
            return;
        }
        let l = tree.left[node as usize];
        let r = tree.right[node as usize];
        visit(tree, l & !LEAF, l & LEAF != 0, out);
        visit(tree, r & !LEAF, r & LEAF != 0, out);
    }
    visit(tree, 0, false, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuMorton, GpuRadixSort};
    use glam::Vec3;

    fn grid_codes(ctx: &GpuContext, n: usize) -> Vec<u32> {
        let mut pos = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let h = (((i * 31 + j * 97) % 23) as f32) * 0.004;
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
        let codes = GpuMorton::new(ctx).codes(&pos, &tris).expect("codes");
        GpuRadixSort::new(ctx).sort(&codes).expect("sort")
    }

    #[test]
    fn cpu_tree_is_well_formed() {
        // Distinct + duplicate-heavy sorted inputs both must give a valid tree.
        let cases: Vec<Vec<u32>> = vec![
            (0..16u32).collect(),
            (0..1000u32).map(|i| i / 3).collect(), // many duplicates
            vec![5, 5, 5, 5, 5, 5],                // all equal
            vec![0, 0, 1, 2, 2, 2, 9, 9],
        ];
        for codes in cases {
            let n = codes.len();
            let tree = karras_tree_cpu(&codes);
            assert_eq!(tree.left.len(), n - 1);
            // In-order leaves must be exactly 0..n (each leaf once, sorted order).
            assert_eq!(
                inorder_leaves(&tree, n),
                (0..n as u32).collect::<Vec<_>>(),
                "malformed tree for {codes:?}"
            );
        }
    }

    #[test]
    fn gpu_karras_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let k = GpuKarras::new(&ctx);
        for &n in &[8usize, 20, 40] {
            let codes = grid_codes(&ctx, n);
            let gpu = k.build_tree(&codes).expect("gpu tree");
            let cpu = karras_tree_cpu(&codes);
            // Integer-only construction → bit-exact GPU == CPU.
            assert_eq!(gpu, cpu, "GPU/CPU tree differ at n={n}");
            // And the GPU tree is itself well-formed.
            assert_eq!(
                inorder_leaves(&gpu, codes.len()),
                (0..codes.len() as u32).collect::<Vec<_>>()
            );
        }
    }
}
