// SPDX-License-Identifier: GPL-2.0-or-later
//! engine-gpu — cross-platform GPU context (wgpu) and compute dispatch.
//!
//! This crate owns the single `wgpu::Device` + `wgpu::Queue` the engine runs on
//! (Metal on Apple Silicon, Vulkan on Linux, DX12 on Windows): device bring-up,
//! a SAXPY probe used as the adapter smoke test, and every WGSL compute kernel
//! in `src/shaders/`. Each module owns its shader and a `gpu_*_matches_cpu`
//! oracle test asserting it agrees with `engine-core`. Porting status for the
//! kernels that do not exist yet is in `docs/KERNELS.md`.

pub mod assemble;
pub mod bending;
pub mod bending_asm;
pub mod bvh_aabb;
pub mod bvh_traverse;
pub mod gpu_implicit;
pub mod karras;
pub mod membrane_asm;
pub mod morton;
pub mod pcg;
pub mod radix;
pub mod reduce;
pub mod scan;
pub mod schwarz;
pub mod shell_asm;
pub mod svd;
pub mod sym_eig;
pub mod tet_asm;
pub mod tet_assembler;
pub mod unified_asm;
pub mod unified_solver;

pub use assemble::GpuAssembler;
pub use bending::{BendingOut, GpuBending};
pub use bending_asm::GpuBendingAssembler;
pub use bvh_aabb::{propagate_aabb_cpu, Aabb, GpuBvhAabb};
pub use bvh_traverse::GpuBvhTraverse;
pub use engine_core::CsrMatrix;
pub use gpu_implicit::GpuImplicitSolver;
pub use karras::{inorder_leaves, karras_tree_cpu, GpuKarras, KarrasTree, LEAF};
pub use membrane_asm::GpuMembrane;
pub use morton::{morton_code_cpu, morton_codes_cpu, GpuMorton};
pub use pcg::{GpuLinearSolver, GpuPrecond, Pcg, PcgResult};
pub use radix::GpuRadixSort;
pub use reduce::GpuReduce;
pub use scan::{exclusive_scan_cpu, GpuScan};
pub use schwarz::{SchwarzApply, SchwarzPrecond};
pub use svd::{svd3_cpu, GpuSvd, Svd3};
pub use sym_eig::{project_psd3_cpu, sym_eig3_cpu, GpuSymEig};
pub use tet_asm::{GpuTet, TetOut};
pub use tet_assembler::GpuTetAssembler;
pub use unified_asm::GpuUnified;
pub use unified_solver::GpuUnifiedSolver;

use thiserror::Error;
use wgpu::util::DeviceExt;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("no suitable GPU adapter found: {0}")]
    NoAdapter(String),
    #[error("device request failed: {0}")]
    Device(String),
    #[error("buffer readback failed: {0}")]
    Readback(String),
}

/// The engine's GPU context. Constructed once, shared everywhere.
///
/// Two ways in: [`GpuContext::new`] creates its own instance/adapter/device (the
/// headless CLI and test path), or [`GpuContext::from_device`] joins a device
/// somebody else already made (the desktop app path — the product requires one
/// device shared by the viewport, iced, Vello and egui, and the UI's `nguo-gpu`
/// crate owns it; see `docs/WGPU_PIN.md`).
///
/// No `instance` field: `Adapter` and `Device` each hold their own `Arc` to the
/// wgpu context (`wgpu-27.0.1/src/backend/wgpu_core.rs:41,458-469`), so the
/// `Instance` does not need to outlive them and the engine has no reason to hold one.
pub struct GpuContext {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Whether the `SUBGROUP` feature was available and requested — gates the
    /// `subgroupAdd` reduction fast-path (falls back to the workgroup-tree reduce).
    pub has_subgroups: bool,
}

impl GpuContext {
    /// Bring up a headless compute device on the best available backend. Logs
    /// the chosen backend/adapter; errors loudly if none is available.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self, GpuError> {
        // Honors WGPU_BACKEND for the cross-backend CI matrix (Metal/Vulkan/DX12);
        // defaults to all enabled backends, adapter selection picks the best.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::NoAdapter(e.to_string()))?;

        let info = adapter.get_info();
        tracing::info!(
            backend = ?info.backend,
            adapter = %info.name,
            device_type = ?info.device_type,
            "GPU adapter selected"
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ppf-mguso engine device"),
                required_features: Self::required_features(&adapter),
                required_limits: Self::required_limits(&adapter),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        Ok(Self::from_device(&adapter, &device, &queue))
    }

    /// Limits a device must be created with to run the engine's kernels.
    ///
    /// The solver's SpMV binds **more than 4 storage buffers**, and
    /// `Limits::downlevel_defaults()` caps `max_storage_buffers_per_shader_stage`
    /// at 4 — a device built with the downlevel defaults fails bind-group
    /// validation at pipeline creation, not at startup, which is miserable to
    /// debug. `adapter.limits()` is the adapter's real capability and never
    /// over-requests, so this can't fail on a device the adapter supports.
    ///
    /// Call this when creating the shared device rather than re-deriving it —
    /// two copies of this rule in two crates will drift.
    pub fn required_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
        adapter.limits()
    }

    /// Features a device must be created with to run the engine's kernels.
    ///
    /// `SUBGROUP` when the adapter exposes it, so the `subgroupAdd` reduction
    /// fast-path can run; masked against the adapter so it is always requestable.
    /// A device created with *fewer* features than this still works but silently
    /// drops to the slower workgroup-tree reduce (see [`Self::has_subgroups`]).
    ///
    /// Requesting the mask rather than a hardcoded flag means an adapter without
    /// subgroups yields `Features::empty()`, which is always valid.
    pub fn required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::SUBGROUP
    }

    /// Join a `wgpu::Device` that somebody else created, instead of making one.
    ///
    /// This is the desktop-app entry point: the product mandates a
    /// single `wgpu::Device` shared by the 3D viewport, iced, Vello and egui, and
    /// the UI's `nguo-gpu` crate is its owner (it must outlive the renderers built
    /// against it). All three handles are `Arc`-backed and cloned cheaply here, so
    /// the caller keeps its own and the engine does not take ownership.
    ///
    /// # The caller MUST create the device with
    ///
    /// - `required_limits: GpuContext::required_limits(&adapter)` — **not**
    ///   `Limits::downlevel_defaults()`, or the solver's >4 storage-buffer bind
    ///   groups fail validation when a pipeline is built.
    /// - `required_features: GpuContext::required_features(&adapter)` — at minimum.
    ///   Fewer is not a hard error but costs the subgroup reduction fast-path.
    ///
    /// Use those two helpers rather than restating the rules; that is the whole
    /// reason they are public.
    pub fn from_device(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        Self {
            adapter: adapter.clone(),
            device: device.clone(),
            queue: queue.clone(),
            // Read from the *device*, not the adapter: what matters is the feature
            // actually enabled on this device, which for a caller-built device may
            // be less than the adapter can do.
            has_subgroups: device.features().contains(wgpu::Features::SUBGROUP),
        }
    }

    /// Human-readable backend name for logs / process contract output.
    pub fn backend_name(&self) -> String {
        format!("{:?}", self.adapter.get_info().backend)
    }

    /// Run `y = a*x + y` on the GPU and read `y` back. The end-to-end no-op step:
    /// upload -> dispatch a WGSL compute shader -> read back. Returns the result.
    pub fn saxpy(&self, a: f32, x: &[f32], y: &[f32]) -> Result<Vec<f32>, GpuError> {
        assert_eq!(x.len(), y.len(), "saxpy: x and y must match length");
        let n = x.len() as u32;
        if n == 0 {
            return Ok(Vec::new());
        }
        let dev = &self.device;

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("saxpy"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/saxpy.wgsl").into()),
        });

        // Uniform params: a (f32) + n (u32), padded to 16 bytes (std140/std430).
        let params = [a.to_bits(), n, 0u32, 0u32];
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("saxpy.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let x_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("saxpy.x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let y_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("saxpy.y"),
            contents: bytemuck::cast_slice(y),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bytes = (n as usize * std::mem::size_of::<f32>()) as u64;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("saxpy.staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("saxpy.pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("saxpy.bind"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("saxpy"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("saxpy.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = n.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

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
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

/// Map a staging buffer and read it back as `f32`. One copy shared by every element
/// kernel — this body was duplicated verbatim in 7 modules (md5-identical).
pub(crate) fn read_f32(ctx: &GpuContext, staging: &wgpu::Buffer) -> Result<Vec<f32>, GpuError> {
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
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

/// Allocate a MAP_READ staging buffer and queue a copy of `bytes` from `src` into it.
///
/// One copy shared by every element kernel — this was duplicated as an inherent `stage`
/// method in 8 modules, identical apart from the debug label.
pub(crate) fn stage_buffer(
    device: &wgpu::Device,
    enc: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    bytes: u64,
) -> wgpu::Buffer {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stage"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    staging
}

#[cfg(test)]
mod tests {
    use super::*;

    // Gated: CI runners without a GPU/software adapter skip this. Locally on
    // Apple Silicon (Metal) it must pass — that IS the Phase 2 exit criterion.
    #[test]
    fn saxpy_on_gpu_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU test, no adapter: {e}");
                return;
            }
        };
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let out = ctx.saxpy(2.0, &x, &y).unwrap();
        for i in 0..x.len() {
            assert!((out[i] - (2.0 * x[i] + y[i])).abs() < 1e-5);
        }
    }

    /// The shared-device path the desktop app uses: `nguo-gpu` owns the device,
    /// the engine joins it. Builds the device by hand exactly as the UI will —
    /// through `required_limits`/`required_features` — then runs a real dispatch
    /// through `from_device`.
    ///
    /// Uses PCG rather than saxpy on purpose: PCG's SpMV binds >4 storage buffers,
    /// so this fails at pipeline creation if the limits helper is ever wrong or the
    /// caller used `downlevel_defaults()`. Saxpy binds 3 and would pass regardless.
    #[test]
    fn from_device_runs_on_a_caller_owned_device() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("skipping GPU test, no adapter");
            return;
        };

        // Verbatim what nguo-gpu does when it creates the one shared device.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ui-owned shared device"),
            required_features: GpuContext::required_features(&adapter),
            required_limits: GpuContext::required_limits(&adapter),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("caller-owned device must be creatable from the engine's own helpers");

        let n = 64;
        let a = CsrMatrix::laplacian_1d(n);
        let x_star: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() + 1.0).collect();
        let b = a.spmv_cpu(&x_star);

        let ctx = GpuContext::from_device(&adapter, &device, &queue);
        let res = Pcg::new(&ctx, &a).solve(&b, 1e-6, 1000);
        assert!(
            res.converged,
            "PCG on a shared device did not converge: {res:?}"
        );
        for (i, (got, want)) in res.x.iter().zip(&x_star).enumerate() {
            assert!((got - want).abs() < 1e-3, "row {i}: {got} vs {want}");
        }

        // The engine only borrowed the device: dropping the context must leave the
        // caller's handles usable, since the UI keeps rendering after teardown.
        drop(ctx);
        assert!(device
            .features()
            .contains(GpuContext::required_features(&adapter)));
        queue.submit(None);
    }
}
