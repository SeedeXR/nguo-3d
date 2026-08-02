// SPDX-License-Identifier: GPL-2.0-or-later
//! LBVH traversal (`shaders/bvh_traverse.wgsl`) — the final broad-phase stage.
//! Per-vertex descent of the Karras tree + node AABBs, emitting (vertex, triangle)
//! candidate pairs. Emits the geometric superset (1-ring exclusion is a downstream
//! filter), so the output must contain every truly-close pair — the same contract
//! the CPU broad phase is verified against. Closes the GPU LBVH broad phase.

use crate::bvh_aabb::Aabb;
use crate::karras::KarrasTree;
use crate::{GpuContext, GpuError};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// GPU BVH traversal engine.
pub struct GpuBvhTraverse<'a> {
    ctx: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuBvhTraverse<'a> {
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
        let mut entries = vec![buf(0, true, true)];
        for b in 1..=8 {
            entries.push(buf(b, true, false)); // read-only inputs
        }
        entries.push(buf(9, false, false)); // out_pairs
        entries.push(buf(10, false, false)); // counter
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bvh_traverse.bgl"),
            entries: &entries,
        });
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bvh_traverse"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bvh_traverse.wgsl").into()),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bvh_traverse.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bvh_traverse"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("traverse"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            ctx,
        }
    }

    /// Emit candidate `(vertex, triangle)` pairs whose separation may be below
    /// `dhat`. `leaf_aabb`/`prim` are in sorted (leaf) order — `prim[leaf]` is the
    /// real triangle index. Errors if the pair buffer or the fixed traversal stack
    /// overflows (would violate the superset guarantee).
    #[allow(clippy::too_many_arguments)]
    pub fn traverse(
        &self,
        tree: &KarrasTree,
        node_aabb: &[Aabb],
        leaf_aabb: &[Aabb],
        prim: &[u32],
        positions: &[Vec3],
        dhat: f32,
        max_pairs: usize,
    ) -> Result<Vec<(u32, u32)>, GpuError> {
        let n_tris = leaf_aabb.len();
        let n_verts = positions.len();
        if n_tris == 0 || n_verts == 0 {
            return Ok(Vec::new());
        }
        let dev = &self.ctx.device;

        let split = |aabbs: &[Aabb]| -> (Vec<f32>, Vec<f32>) {
            let lo = aabbs.iter().flat_map(|a| [a[0], a[1], a[2]]).collect();
            let hi = aabbs.iter().flat_map(|a| [a[3], a[4], a[5]]).collect();
            (lo, hi)
        };
        let (node_lo, node_hi) = split(node_aabb);
        let (leaf_lo, leaf_hi) = split(leaf_aabb);
        let pos_flat: Vec<f32> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let params = [
            n_tris as u32,
            n_verts as u32,
            dhat.to_bits(),
            max_pairs as u32,
        ];

        let ro = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        // Node buffers may be empty (n_tris==1); pad to 4 bytes so the binding is valid.
        let ro_pad = |label, v: &[f32]| {
            let bytes: Vec<u8> = if v.is_empty() {
                vec![0u8; 4]
            } else {
                bytemuck::cast_slice(v).to_vec()
            };
            ro(label, &bytes)
        };
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_traverse.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let left_buf = ro_pad("bvh_traverse.left", &to_f32(&tree.left));
        let right_buf = ro_pad("bvh_traverse.right", &to_f32(&tree.right));
        let node_lo_buf = ro_pad("bvh_traverse.node_lo", &node_lo);
        let node_hi_buf = ro_pad("bvh_traverse.node_hi", &node_hi);
        let leaf_lo_buf = ro("bvh_traverse.leaf_lo", bytemuck::cast_slice(&leaf_lo));
        let leaf_hi_buf = ro("bvh_traverse.leaf_hi", bytemuck::cast_slice(&leaf_hi));
        let prim_buf = ro("bvh_traverse.prim", bytemuck::cast_slice(prim));
        let pos_buf = ro("bvh_traverse.pos", bytemuck::cast_slice(&pos_flat));

        let pairs_bytes = (max_pairs.max(1) * 2 * 4) as u64;
        let out_pairs = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_traverse.out_pairs"),
            size: pairs_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counter = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_traverse.counter"),
            contents: bytemuck::cast_slice(&[0u32, 0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bvh_traverse.bind"),
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
                    resource: node_lo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: node_hi_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: leaf_lo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: leaf_hi_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: prim_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: out_pairs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: counter.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bvh_traverse"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bvh_traverse.cp"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bind, &[]);
            cp.set_pipeline(&self.pipeline);
            cp.dispatch_workgroups(n_verts.div_ceil(256) as u32, 1, 1);
        }
        let counter_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &counter, 8);
        let pairs_stage = crate::stage_buffer(&self.ctx.device, &mut enc, &out_pairs, pairs_bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let cnt = read_u32(self.ctx, &counter_stage)?;
        let count = cnt[0] as usize;
        let overflow = cnt[1];
        if overflow != 0 {
            return Err(GpuError::Readback(format!(
                "BVH traversal overflow (flag={overflow}: 1=pair-buffer, 2=stack); \
                 emitted {count} > max_pairs {max_pairs} or tree too deep"
            )));
        }
        let raw = read_u32(self.ctx, &pairs_stage)?;
        Ok((0..count).map(|i| (raw[2 * i], raw[2 * i + 1])).collect())
    }
}

fn to_f32(v: &[u32]) -> Vec<f32> {
    // Reinterpret u32 bits as f32 so the padded-storage helper can carry indices.
    v.iter().map(|&x| f32::from_bits(x)).collect()
}

fn read_u32(ctx: &GpuContext, staging: &wgpu::Buffer) -> Result<Vec<u32>, GpuError> {
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
    use crate::{GpuBvhAabb, GpuKarras, GpuMorton, GpuRadixSort};
    use engine_core::contact::point_triangle_distance;
    use std::collections::HashSet;

    // Two close horizontal cloth layers → real cross-layer vertex-triangle pairs.
    fn two_layer_mesh() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        let n = 6;
        for (_layer, y) in [(0usize, 0.0f32), (1, 0.03)] {
            let base = positions.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.1, y, j as f32 * 0.1));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = base + (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([r, v + 1, r + 1]);
                }
            }
        }
        (positions, tris)
    }

    #[test]
    fn gpu_lbvh_is_superset_of_brute_force() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (positions, tris) = two_layer_mesh();
        let dhat = 0.05f32;

        // Full GPU LBVH pipeline: Morton -> radix(pairs) -> Karras -> node AABBs -> traverse.
        let codes = GpuMorton::new(&ctx)
            .codes(&positions, &tris)
            .expect("morton");
        let ids: Vec<u32> = (0..tris.len() as u32).collect();
        let (sorted, prim) = GpuRadixSort::new(&ctx)
            .sort_pairs(&codes, &ids)
            .expect("sort");
        let tree = GpuKarras::new(&ctx).build_tree(&sorted).expect("tree");
        // Leaf AABBs in sorted order: leaf i wraps triangle prim[i].
        let leaf_aabb: Vec<Aabb> = prim
            .iter()
            .map(|&t| {
                let tr = tris[t as usize];
                let (a, b, c) = (
                    positions[tr[0] as usize],
                    positions[tr[1] as usize],
                    positions[tr[2] as usize],
                );
                let lo = a.min(b).min(c);
                let hi = a.max(b).max(c);
                [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z]
            })
            .collect();
        let node_aabb = GpuBvhAabb::new(&ctx)
            .propagate(&tree, &leaf_aabb)
            .expect("aabb");

        let pairs = GpuBvhTraverse::new(&ctx)
            .traverse(
                &tree,
                &node_aabb,
                &leaf_aabb,
                &prim,
                &positions,
                dhat,
                1 << 16,
            )
            .expect("traverse");
        let set: HashSet<(u32, u32)> = pairs.iter().copied().collect();

        // Superset property: every truly-close (v, triangle) pair (distance < dhat,
        // v not a vertex of the triangle) MUST be emitted by the GPU broad phase.
        let mut close_pairs = 0usize;
        for (vi, pt) in positions.iter().enumerate() {
            for (ti, t) in tris.iter().enumerate() {
                if t.contains(&(vi as u32)) {
                    continue; // v is a corner of t — self, excluded downstream
                }
                let d = point_triangle_distance(
                    *pt,
                    positions[t[0] as usize],
                    positions[t[1] as usize],
                    positions[t[2] as usize],
                );
                if d < dhat {
                    close_pairs += 1;
                    assert!(
                        set.contains(&(vi as u32, ti as u32)),
                        "LBVH missed close pair (v={vi}, t={ti}, d={d})"
                    );
                }
            }
        }
        assert!(
            close_pairs > 0,
            "test mesh should have cross-layer close pairs"
        );
    }
}
