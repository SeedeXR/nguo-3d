// SPDX-License-Identifier: GPL-2.0-or-later
//! GPU assembly of the implicit mass-spring Newton system, driving the
//! atomic-free gather kernel (`shaders/assemble.wgsl`). Consumes an
//! `engine_core::SpringAssembler` (the precomputed, topology-fixed sparsity +
//! offsets) and produces the CSR `values` + `grad` on the GPU. Verified
//! bit-close to the CPU `SpringAssembler::assemble` — the last piece needed to
//! make the implicit Newton step GPU-resident (assemble → PCG solve on GPU).

use crate::{GpuContext, GpuError};
use engine_core::SpringAssembler;
use glam::Vec3;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct AsmParams {
    n: u32,
    inv_h2: f32,
    k: f32,
    _pad: f32,
}

pub struct GpuAssembler {
    device: wgpu::Device,
    queue: wgpu::Queue,
    n: usize,
    nnz: usize,
    wg: u32,
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    x: wgpu::Buffer,
    x_tilde: wgpu::Buffer,
    mass: wgpu::Buffer,
    pinned: wgpu::Buffer,
    diag_off: wgpu::Buffer,
    adj_ptr: wgpu::Buffer,
    adj_j: wgpu::Buffer,
    adj_rest: wgpu::Buffer,
    adj_off: wgpu::Buffer,
    values: wgpu::Buffer,
    grad: wgpu::Buffer,
}

impl GpuAssembler {
    pub fn new(ctx: &GpuContext, plan: &SpringAssembler) -> Self {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let n = plan.n;
        let nnz = plan.structure.nnz();
        let wg = (n as u32).max(1).div_ceil(64);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("assemble"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/assemble.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("assemble"),
            layout: None,
            module: &module,
            entry_point: Some("assemble"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let init_u32 = |label: &str, data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let init_f32 = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let init_i32 = |label: &str, data: &[i32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let dyn_f32 = |label: &str, len: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * 4) as u64,
                usage: ro,
                mapped_at_creation: false,
            })
        };
        let out_f32 = |label: &str, len: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        Self {
            n,
            nnz,
            wg,
            pipeline,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("asm.params"),
                size: std::mem::size_of::<AsmParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            x: dyn_f32("asm.x", 3 * n),
            x_tilde: dyn_f32("asm.x_tilde", 3 * n),
            mass: init_f32("asm.mass", &plan.mass),
            pinned: init_u32("asm.pinned", &plan.pinned_u32()),
            diag_off: init_i32("asm.diag_off", plan.diag_off()),
            adj_ptr: init_u32("asm.adj_ptr", plan.adj_ptr()),
            adj_j: init_u32("asm.adj_j", plan.adj_j()),
            adj_rest: init_f32("asm.adj_rest", plan.adj_rest()),
            adj_off: init_i32("asm.adj_off", plan.adj_off()),
            values: out_f32("asm.values", nnz),
            grad: out_f32("asm.grad", 3 * n),
            device,
            queue,
        }
    }

    /// Assemble on the GPU and read back `(values, grad)`.
    pub fn assemble(
        &self,
        x: &[Vec3],
        x_tilde: &[Vec3],
        inv_h2: f32,
        k: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let flat = |v: &[Vec3]| -> Vec<f32> { v.iter().flat_map(|p| [p.x, p.y, p.z]).collect() };
        self.queue
            .write_buffer(&self.x, 0, bytemuck::cast_slice(&flat(x)));
        self.queue
            .write_buffer(&self.x_tilde, 0, bytemuck::cast_slice(&flat(x_tilde)));
        let p = AsmParams {
            n: self.n as u32,
            inv_h2,
            k,
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&p));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("asm.bind"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                bge(0, &self.params),
                bge(1, &self.x),
                bge(2, &self.x_tilde),
                bge(3, &self.mass),
                bge(4, &self.pinned),
                bge(5, &self.diag_off),
                bge(6, &self.adj_ptr),
                bge(7, &self.adj_j),
                bge(8, &self.adj_rest),
                bge(9, &self.adj_off),
                bge(10, &self.values),
                bge(11, &self.grad),
            ],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("asm.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.wg, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        Ok((
            self.read(&self.values, self.nnz)?,
            self.read(&self.grad, 3 * self.n)?,
        ))
    }

    /// Copy a device buffer back as `f32`. Fallible: a lost/failed device must surface as
    /// [`GpuError`] rather than panic, and must never be papered over with zeros — silently
    /// returning wrong numbers would be worse than failing.
    fn read(&self, buf: &wgpu::Buffer, len: usize) -> Result<Vec<f32>, GpuError> {
        let bytes = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asm.read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv()
            .map_err(|e| GpuError::Readback(e.to_string()))?
            .map_err(|e| GpuError::Readback(e.to_string()))?;
        let data = slice.get_mapped_range();
        let v = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(v)
    }
}

fn bge(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{Mesh, SimState, SpringAssembler};

    #[test]
    fn gpu_assembly_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU assembly test, no adapter: {e}");
                return;
            }
        };
        let nn = 5;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                verts.push(Vec3::new(i as f32 * 0.1, 1.0, j as f32 * 0.1));
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
        let mut state = SimState::from_mesh(&Mesh::new(verts, tris), 0.1);
        state.pin(0);
        state.pin(nn - 1);
        let x: Vec<Vec3> = state
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::new((i as f32 * 0.3).sin() * 0.02, -0.05, 0.01))
            .collect();
        let x_tilde: Vec<Vec3> = x.iter().map(|p| *p + Vec3::new(0.0, -0.001, 0.0)).collect();
        let (inv_h2, k) = (1.0e4, 1000.0);

        let plan = SpringAssembler::new(&state);
        let (cpu_v, cpu_g) = plan.assemble(&x, &x_tilde, inv_h2, k);
        let gpu = GpuAssembler::new(&ctx, &plan);
        let (gpu_v, gpu_g) = gpu.assemble(&x, &x_tilde, inv_h2, k).expect("gpu assemble");

        let vmax = cpu_v
            .iter()
            .zip(&gpu_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let gmax = cpu_g
            .iter()
            .zip(&gpu_g)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(vmax < 1e-3, "GPU/CPU values drift {vmax}");
        assert!(gmax < 1e-4, "GPU/CPU grad drift {gmax}");
    }
}
