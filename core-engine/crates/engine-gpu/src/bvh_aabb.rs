// SPDX-License-Identifier: GPL-2.0-or-later
//! LBVH bottom-up node-AABB propagation (`shaders/bvh_aabb.wgsl`) — given a Karras
//! tree and per-leaf AABBs, compute each internal node's AABB as the union of its
//! subtree. Level-pass propagation (see the shader header for why, not the atomic
//! single-climb). Merge is `min`/`max` of identical leaf floats, so GPU output is
//! bit-exact to [`propagate_aabb_cpu`] (the reference). Verified on Metal.

use crate::karras::{KarrasTree, LEAF};
use crate::{GpuContext, GpuError};
use wgpu::util::DeviceExt;

/// Per-node axis-aligned box as `[min.xyz, max.xyz]`.
pub type Aabb = [f32; 6];

fn merge(a: &Aabb, b: &Aabb) -> Aabb {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].max(b[3]),
        a[4].max(b[4]),
        a[5].max(b[5]),
    ]
}

/// CPU reference: internal node `i`'s AABB = union of its subtree's leaf AABBs.
/// Post-order recursion from the root (internal 0). `leaf_aabb.len()` == n.
pub fn propagate_aabb_cpu(tree: &KarrasTree, leaf_aabb: &[Aabb]) -> Vec<Aabb> {
    let n = leaf_aabb.len();
    let mut node = vec![[0.0f32; 6]; n.saturating_sub(1)];
    if n < 2 {
        return node;
    }
    // Returns the AABB of the subtree rooted at (node, is_leaf), filling `node`.
    fn go(
        tree: &KarrasTree,
        leaf_aabb: &[Aabb],
        node: &mut [Aabb],
        idx: u32,
        is_leaf: bool,
    ) -> Aabb {
        if is_leaf {
            return leaf_aabb[idx as usize];
        }
        let l = tree.left[idx as usize];
        let r = tree.right[idx as usize];
        let la = go(tree, leaf_aabb, node, l & !LEAF, l & LEAF != 0);
        let ra = go(tree, leaf_aabb, node, r & !LEAF, r & LEAF != 0);
        let m = merge(&la, &ra);
        node[idx as usize] = m;
        m
    }
    go(tree, leaf_aabb, &mut node, 0, false);
    node
}

/// GPU node-AABB propagation engine.
pub struct GpuBvhAabb<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuBvhAabb<'a> {
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
            label: Some("bvh_aabb.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, true, false),
                buf(3, true, false),
                buf(4, true, false),
                buf(5, false, false),
                buf(6, false, false),
                buf(7, true, false),
                buf(8, false, false),
                buf(9, false, false),
            ],
        });
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bvh_aabb"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bvh_aabb.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bvh_aabb.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bvh_aabb"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("merge"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// Compute internal-node AABBs. Loops level passes until no node changes (a
    /// separate submit per pass → cross-workgroup memory sync). Errors if it fails
    /// to converge within `n` passes (would indicate a malformed tree).
    pub fn propagate(&self, tree: &KarrasTree, leaf_aabb: &[Aabb]) -> Result<Vec<Aabb>, GpuError> {
        let n = leaf_aabb.len();
        if n < 2 {
            return Ok(Vec::new());
        }
        let m = n - 1;
        let dev = &self.ctx.device;

        let flat = |aabbs: &[Aabb], sel: std::ops::Range<usize>| -> Vec<f32> {
            aabbs.iter().flat_map(|a| a[sel.clone()].to_vec()).collect()
        };
        let leaf_min: Vec<f32> = flat(leaf_aabb, 0..3);
        let leaf_max: Vec<f32> = flat(leaf_aabb, 3..6);

        let params = [n as u32, 0u32, 0u32, 0u32];
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_aabb.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let ro_storage = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let left_buf = ro_storage("bvh_aabb.left", bytemuck::cast_slice(&tree.left));
        let right_buf = ro_storage("bvh_aabb.right", bytemuck::cast_slice(&tree.right));
        let leaf_min_buf = ro_storage("bvh_aabb.leaf_min", bytemuck::cast_slice(&leaf_min));
        let leaf_max_buf = ro_storage("bvh_aabb.leaf_max", bytemuck::cast_slice(&leaf_max));

        let rw = |label, bytes: u64, copy_src: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if copy_src {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let node_bytes = (m * 3 * 4) as u64;
        let node_min = rw("bvh_aabb.node_min", node_bytes, true);
        let node_max = rw("bvh_aabb.node_max", node_bytes, true);
        // Two ready buffers (ping-pong), both start all-zero.
        let ready_bytes = (m * 4) as u64;
        let ready_a = rw("bvh_aabb.ready_a", ready_bytes, false);
        let ready_b = rw("bvh_aabb.ready_b", ready_bytes, false);
        let changed = rw("bvh_aabb.changed", 4, true);
        let changed_stage = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_aabb.changed_stage"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-init ready buffers and node aabbs.
        let zeros_m = vec![0u32; m];
        let zeros_node = vec![0f32; m * 3];
        self.ctx
            .queue
            .write_buffer(&ready_a, 0, bytemuck::cast_slice(&zeros_m));
        self.ctx
            .queue
            .write_buffer(&ready_b, 0, bytemuck::cast_slice(&zeros_m));
        self.ctx
            .queue
            .write_buffer(&node_min, 0, bytemuck::cast_slice(&zeros_node));
        self.ctx
            .queue
            .write_buffer(&node_max, 0, bytemuck::cast_slice(&zeros_node));

        let make_bind = |ready_in: &wgpu::Buffer, ready_out: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bvh_aabb.bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: left_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: right_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: leaf_min_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: leaf_max_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: node_min.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: node_max.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: ready_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: ready_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: changed.as_entire_binding(),
                    },
                ],
            })
        };

        let groups = m.div_ceil(256) as u32;
        let mut in_is_a = true;
        for _pass in 0..=n {
            self.ctx
                .queue
                .write_buffer(&changed, 0, bytemuck::cast_slice(&[0u32]));
            let bind = if in_is_a {
                make_bind(&ready_a, &ready_b)
            } else {
                make_bind(&ready_b, &ready_a)
            };
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bvh_aabb.pass"),
            });
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bvh_aabb.cp"),
                    timestamp_writes: None,
                });
                cp.set_bind_group(0, &bind, &[]);
                cp.set_pipeline(&self.pipeline);
                cp.dispatch_workgroups(groups, 1, 1);
            }
            enc.copy_buffer_to_buffer(&changed, 0, &changed_stage, 0, 4);
            self.ctx.queue.submit(Some(enc.finish()));
            let c = read_u32(self.ctx, &changed_stage)?;
            in_is_a = !in_is_a;
            if c == 0 {
                // Converged (this pass changed nothing).
                let mins = read_f32(self.ctx, &node_min, node_bytes)?;
                let maxs = read_f32(self.ctx, &node_max, node_bytes)?;
                return Ok((0..m)
                    .map(|i| {
                        [
                            mins[3 * i],
                            mins[3 * i + 1],
                            mins[3 * i + 2],
                            maxs[3 * i],
                            maxs[3 * i + 1],
                            maxs[3 * i + 2],
                        ]
                    })
                    .collect());
            }
        }
        Err(GpuError::Readback(format!(
            "node-AABB propagation did not converge in {n} passes (malformed tree?)"
        )))
    }
}

fn read_u32(ctx: &GpuContext, staging: &wgpu::Buffer) -> Result<u32, GpuError> {
    let v = read_f32_bits(ctx, staging, 4)?;
    Ok(v[0])
}

fn read_f32(ctx: &GpuContext, src: &wgpu::Buffer, bytes: u64) -> Result<Vec<f32>, GpuError> {
    let dev = &ctx.device;
    let staging = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bvh_aabb.read"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));
    let bits = read_f32_bits(ctx, &staging, bytes)?;
    Ok(bits.iter().map(|&b| f32::from_bits(b)).collect())
}

// Map a MAP_READ|COPY_DST staging buffer already holding the data and read its u32 words.
fn read_f32_bits(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    _bytes: u64,
) -> Result<Vec<u32>, GpuError> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv()
        .map_err(|e| GpuError::Readback(e.to_string()))?
        .map_err(|e| GpuError::Readback(e.to_string()))?;
    let data = slice.get_mapped_range();
    let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{karras_tree_cpu, GpuKarras, GpuMorton, GpuRadixSort};
    use glam::Vec3;

    // Build (sorted codes, per-leaf triangle AABBs) for an n×n grid mesh. Leaf i is
    // mapped to triangle i (identity) — propagation correctness is permutation-
    // independent; the leaf→primitive permutation (radix payload) is the traversal
    // stage's concern.
    fn scene(ctx: &GpuContext, n: usize) -> (Vec<u32>, Vec<Aabb>) {
        let mut pos = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let h = (((i * 41 + j * 13) % 19) as f32) * 0.006;
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
        let sorted = GpuRadixSort::new(ctx).sort(&codes).expect("sort");
        let leaf_aabb: Vec<Aabb> = tris
            .iter()
            .map(|t| {
                let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
                let lo = a.min(b).min(c);
                let hi = a.max(b).max(c);
                [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z]
            })
            .collect();
        (sorted, leaf_aabb)
    }

    #[test]
    fn gpu_node_aabb_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let engine = GpuBvhAabb::new(&ctx);
        for &n in &[8usize, 16, 32] {
            let (codes, leaf_aabb) = scene(&ctx, n);
            let tree = GpuKarras::new(&ctx).build_tree(&codes).expect("tree");
            debug_assert_eq!(tree, karras_tree_cpu(&codes));
            let gpu = engine.propagate(&tree, &leaf_aabb).expect("gpu propagate");
            let cpu = propagate_aabb_cpu(&tree, &leaf_aabb);
            // min/max of identical leaf floats → bit-exact.
            assert_eq!(gpu, cpu, "node AABBs differ at n={n}");
            // Root AABB must enclose every leaf (sanity beyond CPU==GPU).
            let root = gpu[0];
            for a in &leaf_aabb {
                assert!(a[0] >= root[0] - 1e-6 && a[3] <= root[3] + 1e-6);
                assert!(a[1] >= root[1] - 1e-6 && a[4] <= root[4] + 1e-6);
                assert!(a[2] >= root[2] - 1e-6 && a[5] <= root[5] + 1e-6);
            }
        }
    }
}
