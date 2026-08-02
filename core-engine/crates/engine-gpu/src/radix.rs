// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU LSD radix sort over `u32` keys (`shaders/radix.wgsl` + `shaders/scan.wgsl`),
//! 8-bit digit, 4 passes. Everything stays GPU-resident between passes — only the
//! final result is read back. Building block for the LBVH broad phase (sorting
//! Morton codes). Verified GPU==CPU on Metal. Supports `n ≤ 65536` (the scan limit).

use crate::{GpuContext, GpuError};
use wgpu::util::DeviceExt;

const WG: usize = 256;
const RADIX: usize = 256;

/// GPU radix-sort engine. Build once, `sort` many times.
pub struct GpuRadixSort<'a> {
    ctx: &'a GpuContext,
    radix_bgl: wgpu::BindGroupLayout,
    scan_bgl: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    scan_blocks: wgpu::ComputePipeline,
    scan_block_totals: wgpu::ComputePipeline,
    add_block_offsets: wgpu::ComputePipeline,
}

impl<'a> GpuRadixSort<'a> {
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
        // radix: 0=params(uniform) 1=keys_in(read) 2=keys_out(rw) 3=aux(rw)
        //        4=vals_in(read) 5=vals_out(rw)  — payload carried through scatter
        let radix_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, false, false),
                buf(3, false, false),
                buf(4, true, false),
                buf(5, false, false),
            ],
        });
        // scan: 0=params(uniform) 1=input(read) 2=output(rw) 3=block_total(rw) 4=block_offset(rw)
        let scan_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix.scan.bgl"),
            entries: &[
                buf(0, true, true),
                buf(1, true, false),
                buf(2, false, false),
                buf(3, false, false),
                buf(4, false, false),
            ],
        });
        let radix_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/radix.wgsl").into()),
        });
        let scan_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix.scan"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scan.wgsl").into()),
        });
        let mk = |bgl: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, ep: &str| {
            let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(ep),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl),
                module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            histogram: mk(&radix_bgl, &radix_shader, "histogram"),
            scatter: mk(&radix_bgl, &radix_shader, "scatter"),
            scan_blocks: mk(&scan_bgl, &scan_shader, "scan_blocks"),
            scan_block_totals: mk(&scan_bgl, &scan_shader, "scan_block_totals"),
            add_block_offsets: mk(&scan_bgl, &scan_shader, "add_block_offsets"),
            radix_bgl,
            scan_bgl,
            ctx,
        }
    }

    /// Ascending stable radix sort of `keys`. Errors if `keys.len() > 65536`.
    pub fn sort(&self, keys: &[u32]) -> Result<Vec<u32>, GpuError> {
        // Sort with a throwaway identity payload; the sorted keys are all we need.
        let vals: Vec<u32> = (0..keys.len() as u32).collect();
        Ok(self.sort_pairs(keys, &vals)?.0)
    }

    /// Ascending stable radix sort of `(key, value)` pairs: returns
    /// `(sorted_keys, values_permuted_the_same_way)`. For the LBVH, `keys` are
    /// Morton codes and `vals` are primitive indices, so the result gives each
    /// sorted leaf position its source primitive. Errors if `len > 65536`.
    pub fn sort_pairs(&self, keys: &[u32], vals: &[u32]) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
        assert_eq!(
            keys.len(),
            vals.len(),
            "sort_pairs: keys/vals length mismatch"
        );
        let n = keys.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let num_blocks = n.div_ceil(WG);
        if num_blocks > WG {
            return Err(GpuError::Readback(format!(
                "radix sort supports n ≤ {} (got {n})",
                WG * WG
            )));
        }
        let dev = &self.ctx.device;
        let storage = wgpu::BufferUsages::STORAGE;

        // Two ping-pong key buffers and two ping-pong value buffers (in lockstep).
        let key_a = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radix.key_a"),
            contents: bytemuck::cast_slice(keys),
            usage: storage | wgpu::BufferUsages::COPY_SRC,
        });
        let key_b = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix.key_b"),
            size: (n * 4) as u64,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let val_a = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radix.val_a"),
            contents: bytemuck::cast_slice(vals),
            usage: storage | wgpu::BufferUsages::COPY_SRC,
        });
        let val_b = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix.val_b"),
            size: (n * 4) as u64,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Histogram / offsets, length RADIX * num_blocks (bucket-major).
        let hist_len = RADIX * num_blocks;
        let mk_storage = |label, len: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * 4) as u64,
                usage: storage,
                mapped_at_creation: false,
            })
        };
        let hist = mk_storage("radix.hist", hist_len);
        let offsets = mk_storage("radix.offsets", hist_len);
        // Scan-of-histogram scratch: hist_len elements → scan_blocks = ceil(hist_len/256).
        let scan_blocks_n = hist_len.div_ceil(WG);
        let block_total = mk_storage("radix.scan.block_total", scan_blocks_n);
        let block_offset = mk_storage("radix.scan.block_offset", scan_blocks_n);
        let scan_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radix.scan.params"),
            contents: bytemuck::cast_slice(&[hist_len as u32, scan_blocks_n as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // One radix params uniform per digit pass (differ only by shift).
        let rparams: Vec<wgpu::Buffer> = (0..4)
            .map(|pass| {
                dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("radix.params"),
                    contents: bytemuck::cast_slice(&[
                        n as u32,
                        num_blocks as u32,
                        (pass * 8) as u32,
                        0u32,
                    ]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();

        let scan_bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix.scan.bind"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scan_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: offsets.as_entire_binding(),
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

        // Each pass reads `src`, writes `dst`, then swaps.
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("radix"),
        });
        let (mut src, mut dst) = (&key_a, &key_b);
        let (mut vsrc, mut vdst) = (&val_a, &val_b);
        for rparam in &rparams {
            // Histogram: aux = hist. (vals unused here but the layout requires them.)
            let hist_bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix.hist.bind"),
                layout: &self.radix_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: rparam.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: hist.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: vsrc.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: vdst.as_entire_binding(),
                    },
                ],
            });
            // Scatter: aux = offsets.
            let scatter_bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix.scatter.bind"),
                layout: &self.radix_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: rparam.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: vsrc.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: vdst.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix.pass"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &hist_bind, &[]);
            cp.set_pipeline(&self.histogram);
            cp.dispatch_workgroups(num_blocks as u32, 1, 1);
            // Exclusive scan of the bucket-major histogram → offsets.
            cp.set_bind_group(0, &scan_bind, &[]);
            cp.set_pipeline(&self.scan_blocks);
            cp.dispatch_workgroups(scan_blocks_n as u32, 1, 1);
            cp.set_pipeline(&self.scan_block_totals);
            cp.dispatch_workgroups(1, 1, 1);
            cp.set_pipeline(&self.add_block_offsets);
            cp.dispatch_workgroups(scan_blocks_n as u32, 1, 1);
            // Stable scatter into dst.
            cp.set_bind_group(0, &scatter_bind, &[]);
            cp.set_pipeline(&self.scatter);
            cp.dispatch_workgroups(num_blocks as u32, 1, 1);
            drop(cp);
            std::mem::swap(&mut src, &mut dst);
            std::mem::swap(&mut vsrc, &mut vdst);
        }
        // After 4 swaps, sorted keys/vals are back in key_a/val_a (== src/vsrc).
        let out_bytes = (n * 4) as u64;
        let key_stage = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix.key_stage"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let val_stage = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix.val_stage"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(src, 0, &key_stage, 0, out_bytes);
        enc.copy_buffer_to_buffer(vsrc, 0, &val_stage, 0, out_bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let read = |staging: &wgpu::Buffer| -> Result<Vec<u32>, GpuError> {
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
        };
        let keys_out = read(&key_stage)?;
        let vals_out = read(&val_stage)?;
        Ok((keys_out, vals_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_radix_sort_matches_cpu() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let rs = GpuRadixSort::new(&ctx);
        // Deterministic pseudo-random keys spanning all 4 byte-lanes, various sizes.
        for &n in &[1usize, 3, 256, 257, 1000, 4096, 5000, 65536] {
            let keys: Vec<u32> = (0..n)
                .map(|i| ((i as u64 * 2654435761 + 1013904223) & 0xffff_ffff) as u32)
                .collect();
            let gpu = rs.sort(&keys).expect("gpu sort");
            let mut cpu = keys.clone();
            cpu.sort_unstable();
            assert_eq!(gpu, cpu, "radix mismatch at n={n}");
        }
        // Over the supported size → clean error.
        assert!(rs.sort(&vec![0u32; 65537]).is_err());
    }

    #[test]
    fn gpu_radix_sort_pairs_carries_payload() {
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let rs = GpuRadixSort::new(&ctx);
        for &n in &[1usize, 3, 257, 1000, 5000] {
            let keys: Vec<u32> = (0..n)
                .map(|i| ((i as u64 * 40503 + 12345) % 500) as u32) // dup-heavy → tests stability
                .collect();
            let vals: Vec<u32> = (0..n as u32).collect(); // identity payload = the permutation
            let (gk, gv) = rs.sort_pairs(&keys, &vals).expect("sort_pairs");

            // Keys sorted ascending; each returned value points at a source key equal
            // to the sorted key (payload followed its key).
            assert!(
                gk.windows(2).all(|w| w[0] <= w[1]),
                "keys not sorted at n={n}"
            );
            for (k, &v) in gk.iter().zip(&gv) {
                assert_eq!(keys[v as usize], *k, "payload/key mismatch at n={n}");
            }
            // Stable: within equal-key runs the permutation is ascending (LSD radix
            // with a stable scatter preserves original order).
            for w in gk.windows(2).enumerate() {
                let (i, pair) = w;
                if pair[0] == pair[1] {
                    assert!(gv[i] < gv[i + 1], "not stable within equal keys at n={n}");
                }
            }
        }
    }
}
