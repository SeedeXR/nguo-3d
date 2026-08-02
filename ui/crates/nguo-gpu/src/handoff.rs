// SPDX-License-Identifier: GPL-2.0-or-later
//! Engine-device → UI-device drape handoff.
//!
//! # Why there are two devices
//!
//! `iced_wgpu-0.14.0/src/window/compositor.rs:151` hardcodes its device's
//! limits (`wgpu::Limits::default()` with `max_bind_groups: 2`) and features
//! (`Features::empty()`), takes no parameter for either, and keeps its
//! `Adapter`/`Device` fields private (`:11-18`). There is no injection point.
//!
//! `wgpu::Limits::default().max_storage_buffers_per_shader_stage` is **8**
//! (`wgpu-types-27.0.1/src/lib.rs:794`); the solver's
//! `engine-gpu/src/shaders/unified_gather.wgsl` declares **17** storage
//! buffers. The solver's pipelines therefore cannot be created on iced's
//! device — not "would be slow", cannot be created.
//!
//! The fix is *not* to fork iced. Those 17 buffers are solver guts (PCG
//! scratch, contact pairs, Hessian-apply intermediates) that the UI never
//! reads. Coupling a UI framework's device policy to an internal solver
//! detail is the wrong dependency regardless of what a benchmark says. So:
//! **two devices on one `Instance` and one `Adapter`** — an engine device with
//! `adapter.limits()`, and iced's UI device that also backs Vello, egui and
//! the 3D viewport (all four renderers verified sharing it, see
//! [`crate::adopt`]). What crosses the boundary is small and output-only:
//! [`DrapeVertex`].
//!
//! Full reasoning and the measured numbers: `ui/docs/GPU_HANDOFF.md`.
//!
//! # Invariants
//!
//! 1. **Never render a half-written frame.** A slot's staging buffer is
//!    written by exactly one in-flight copy, and the UI-side vertex buffer is
//!    filled by one `write_buffer` that is queue-ordered before the draw that
//!    reads it (`DATAFLOW.md` §8 pass order: uploads are step 1).
//! 2. **A slot in flight is never handed out again** — [`Handoff::begin`]
//!    only takes a `Free` slot, so a readback can never overwrite a buffer the
//!    GPU is still copying into or the UI is still uploading from.
//! 3. **Stale epochs are dropped, never displayed.** Checked twice, mirroring
//!    `nguo-engine-bridge`'s worker (`bridge.rs:127` before starting,
//!    `bridge.rs:145` before emitting): once in `begin`, once in `pump`.
//! 4. **The UI thread never blocks on the GPU.** `pump` polls; it never waits.

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use nguo_engine_bridge::Epoch;

/// Readback slots in flight. Three = the solver can be copying into one while
/// the UI holds one and one waits, which is the same depth (and the same
/// reason) as `nguo_engine_bridge::FRAME_CHANNEL_CAPACITY`.
pub const SLOTS: usize = 3;

/// One vertex of a drape frame as it crosses the device boundary: 28 bytes.
///
/// `strain` is the slot for the strain-map display mode
/// (`ui/docs/GPU_HANDOFF.md` §3). The solver fills it when the mode is on and
/// leaves it `0.0` otherwise — one payload shape, no second code path. The
/// shader that colours it is not built yet.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrapeVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub strain: f32,
}

/// 28 bytes. Asserted below so a field added without thought fails the build.
pub const VERTEX_SIZE: u64 = std::mem::size_of::<DrapeVertex>() as u64;
const _: () = assert!(VERTEX_SIZE == 28);

const FREE: u8 = 0;
const COPYING: u8 = 1;
const READY: u8 = 2;
const FAILED: u8 = 3;

struct Slot {
    /// Engine-device staging buffer: `COPY_DST | MAP_READ`.
    staging: wgpu::Buffer,
    /// Shared with the `map_async` callback, which is why it is atomic.
    state: Arc<AtomicU8>,
    epoch: Epoch,
    vertices: u32,
}

/// Triple-buffered, non-blocking engine → UI frame transfer.
///
/// One per drape mesh. Drive it from the UI thread: [`Handoff::begin`] when
/// the solver reports a finished frame, [`Handoff::pump`] once per UI frame
/// before rendering.
pub struct Handoff {
    engine: wgpu::Device,
    engine_queue: wgpu::Queue,
    ui_queue: wgpu::Queue,
    slots: Vec<Slot>,
    /// UI-device buffer the viewport draws from: `VERTEX | COPY_DST`.
    vertices: wgpu::Buffer,
    capacity: u32,
    /// Newest epoch a readback was started for. `Epoch(0)` = none, matching
    /// `EngineBridge::latest_epoch`, whose first handed-out epoch is 1.
    submitted: Epoch,
    /// Newest epoch actually sitting in `vertices`.
    displayed: Epoch,
    displayed_vertices: u32,
}

impl Handoff {
    /// `capacity_vertices` sizes the three staging buffers and the UI vertex
    /// buffer up front — a drape's vertex count does not change mid-solve, and
    /// reallocating under an in-flight map is exactly the bug this avoids.
    ///
    /// # Panics
    /// If the two devices are not the same device (they must not be) or
    /// `capacity_vertices` is 0.
    pub fn new(
        engine_device: &wgpu::Device,
        engine_queue: &wgpu::Queue,
        ui_device: &wgpu::Device,
        ui_queue: &wgpu::Queue,
        capacity_vertices: u32,
    ) -> Self {
        assert!(capacity_vertices > 0, "a drape with no vertices");
        assert!(
            engine_device != ui_device,
            "engine and UI device are the same device — either iced grew an \
             injection point (delete this crate's reason to exist) or the \
             wiring is wrong"
        );
        let bytes = u64::from(capacity_vertices) * VERTEX_SIZE;

        let slots = (0..SLOTS)
            .map(|i| Slot {
                staging: engine_device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("nguo drape readback {i}")),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
                epoch: Epoch(0),
                vertices: 0,
            })
            .collect();

        Self {
            engine: engine_device.clone(),
            engine_queue: engine_queue.clone(),
            ui_queue: ui_queue.clone(),
            slots,
            vertices: ui_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nguo drape vertices (UI device)"),
                size: bytes,
                // COPY_SRC is not needed to draw; it is what lets the
                // round-trip test read the UI side back and prove the bytes
                // arrived. Free on every backend, and a usage flag is a worse
                // thing to make conditional than to leave on.
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            capacity: capacity_vertices,
            submitted: Epoch(0),
            displayed: Epoch(0),
            displayed_vertices: 0,
        }
    }

    /// Start an async readback of `src` (an engine-device buffer with
    /// `COPY_SRC`, laid out as `[DrapeVertex; vertices]`).
    ///
    /// Returns `false` — having done nothing — when the frame is stale, or
    /// when all [`SLOTS`] readbacks are still in flight. Both are drops, not
    /// errors: a preview frame the UI could not keep up with is superseded by
    /// the next one anyway (`bridge.rs:190`, the same choice for the same
    /// reason). Never blocks.
    ///
    /// # Panics
    /// If `vertices` exceeds the capacity this `Handoff` was built for.
    pub fn begin(&mut self, epoch: Epoch, src: &wgpu::Buffer, vertices: u32) -> bool {
        assert!(
            vertices <= self.capacity,
            "drape grew to {vertices} vertices, handoff was sized for {}",
            self.capacity
        );
        // Stale before we even start (bridge.rs:127).
        if epoch <= self.submitted {
            tracing::debug!(target: "nguo_gpu", ?epoch, newest = ?self.submitted, "stale drape frame dropped at submit");
            return false;
        }
        let Some(slot) = self
            .slots
            .iter_mut()
            .find(|s| s.state.load(Ordering::Acquire) == FREE)
        else {
            tracing::debug!(target: "nguo_gpu", ?epoch, "all {SLOTS} readbacks in flight, frame dropped");
            return false;
        };

        let bytes = u64::from(vertices) * VERTEX_SIZE;
        slot.epoch = epoch;
        slot.vertices = vertices;
        slot.state.store(COPYING, Ordering::Release);

        let mut encoder = self
            .engine
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nguo drape readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &slot.staging, 0, bytes);
        self.engine_queue.submit([encoder.finish()]);

        // Queued behind the copy above; the callback fires on a later poll.
        let state = slot.state.clone();
        slot.staging
            .slice(0..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                state.store(if r.is_ok() { READY } else { FAILED }, Ordering::Release);
            });

        self.submitted = epoch;
        true
    }

    /// Non-blocking. Uploads the newest completed readback to the UI-device
    /// vertex buffer and frees its slot; drops any older completed frame
    /// without displaying it. Returns the epoch newly uploaded, if any.
    ///
    /// Call once per UI frame, before recording the viewport pass —
    /// `write_buffer` is ordered against later submissions on the same queue,
    /// which is what makes invariant 1 hold without a fence.
    pub fn pump(&mut self) -> Option<Epoch> {
        // Poll, never Wait: a readback that is not done yet just stays in
        // flight and the UI redraws the previous drape.
        let _ = self.engine.poll(wgpu::PollType::Poll);

        let newest = self
            .slots
            .iter()
            .filter(|s| s.state.load(Ordering::Acquire) == READY)
            .map(|s| s.epoch)
            .max()?;

        let mut uploaded = None;
        for slot in &self.slots {
            if slot.state.load(Ordering::Acquire) != READY {
                continue;
            }
            let bytes = u64::from(slot.vertices) * VERTEX_SIZE;
            // Newest wins; anything else completed out of order or behind, and
            // must never reach the screen (bridge.rs:145).
            if slot.epoch == newest && slot.epoch > self.displayed {
                let view = slot.staging.slice(0..bytes).get_mapped_range();
                // Straight from the mapped range — no intermediate `Vec`.
                self.ui_queue.write_buffer(&self.vertices, 0, &view);
                drop(view);
                uploaded = Some((slot.epoch, slot.vertices));
            } else {
                tracing::debug!(target: "nguo_gpu", epoch = ?slot.epoch, newest = ?self.displayed, "stale drape frame dropped at upload");
            }
            slot.staging.unmap();
            slot.state.store(FREE, Ordering::Release);
        }

        if let Some((epoch, vertices)) = uploaded {
            self.displayed = epoch;
            self.displayed_vertices = vertices;
        }
        uploaded.map(|(epoch, _)| epoch)
    }

    /// The UI-device vertex buffer. Contains a whole frame or nothing.
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertices
    }

    /// `(epoch, vertex count)` of what `vertex_buffer` holds; `None` before
    /// the first completed frame.
    pub fn displayed(&self) -> Option<(Epoch, u32)> {
        (self.displayed.0 != 0).then_some((self.displayed, self.displayed_vertices))
    }

    /// Slots not currently in flight. `SLOTS` when idle.
    pub fn free_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state.load(Ordering::Acquire) == FREE)
            .count()
    }

    /// Block until every in-flight readback has landed. Tests and the shutdown
    /// path only — calling this from the UI thread defeats the whole design.
    pub fn drain(&mut self) -> Option<Epoch> {
        let mut last = None;
        while self.free_slots() < SLOTS {
            let _ = self.engine.poll(wgpu::PollType::wait_indefinitely());
            last = self.pump().or(last);
            if self
                .slots
                .iter()
                .any(|s| s.state.load(Ordering::Acquire) == FAILED)
            {
                break;
            }
        }
        last
    }
}

/// The limits `iced_wgpu` builds its device with, verified at
/// `iced_wgpu-0.14.0/src/window/compositor.rs:155-158`. Tests and benchmarks
/// use it to reproduce iced's device exactly on our own adapter.
pub fn iced_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_bind_groups: 2,
        max_non_sampler_bindings: 2048,
        ..wgpu::Limits::default()
    }
}

/// Limits the engine device must have. Identical to
/// `engine_gpu::GpuContext::required_limits` (`engine-gpu/src/lib.rs:141-143`),
/// restated because `engine-gpu` is in the other workspace and is not a
/// dependency of the UI.
pub fn engine_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    adapter.limits()
}

/// Bring up the engine device on the adapter iced is using.
///
/// The one-`Instance`-one-`Adapter` rule is enforced here, not hoped for: the
/// request options are byte-for-byte iced's
/// (`iced_wgpu-0.14.0/src/window/compositor.rs:85`,
/// `PowerPreference::HighPerformance`), and the resulting adapter is asserted
/// equal to [`crate::adapter_description`] — the same enumeration iced's own
/// pick resolves to. Two *adapters* would be a bug; two devices is the design.
///
/// # Panics
/// If the adapter obtained is not the one the UI is on.
pub fn engine_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    assert_one_adapter(&adapter);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("nguo engine device"),
        required_features: adapter.features() & wgpu::Features::SUBGROUP,
        required_limits: engine_limits(&adapter),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    Some((adapter, device, queue))
}

/// Asserts `adapter` is the same physical adapter the UI renders on.
///
// ponytail: compared by `AdapterInfo` string, because iced 0.14 keeps its
// `Adapter` private (compositor.rs:13) and `wgpu::Device` exposes no adapter
// handle, so there is nothing to compare by identity. Exact on a single-GPU
// machine; on a dual-GPU laptop, take the real handle from a patched
// compositor instead.
///
/// # Panics
/// If it is a different adapter.
pub fn assert_one_adapter(adapter: &wgpu::Adapter) {
    let info = adapter.get_info();
    let engine = format!(
        "{} · {:?} · {:?}",
        info.name, info.backend, info.device_type
    );
    let ui = crate::adapter_description();
    assert_eq!(
        engine, ui,
        "engine and UI are on different adapters — the design is two devices \
         on ONE adapter; two adapters means cross-GPU copies"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two devices, one instance, one adapter — the architecture of record.
    /// `None` when the machine has no GPU, in which case the GPU tests skip.
    struct Rig {
        engine: wgpu::Device,
        engine_queue: wgpu::Queue,
        ui: wgpu::Device,
        ui_queue: wgpu::Queue,
    }

    impl Rig {
        fn new() -> Option<Self> {
            let (adapter, engine, engine_queue) = engine_device()?;
            // Same `adapter` value, so "one adapter" holds by construction —
            // there is only one to be had.
            let (ui, ui_queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("iced-like UI device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: iced_limits(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                }))
                .ok()?;
            Some(Self {
                engine,
                engine_queue,
                ui,
                ui_queue,
            })
        }

        fn handoff(&self, capacity: u32) -> Handoff {
            Handoff::new(
                &self.engine,
                &self.engine_queue,
                &self.ui,
                &self.ui_queue,
                capacity,
            )
        }

        /// An engine-device buffer holding `verts`, ready to be read back.
        fn source(&self, verts: &[DrapeVertex]) -> wgpu::Buffer {
            let buffer = self.engine.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test drape"),
                size: std::mem::size_of_val(verts) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.engine_queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(verts));
            self.engine_queue.submit([]);
            buffer
        }

        /// Reads the UI-device vertex buffer back. Test-only: it needs a
        /// second staging buffer *on the UI device*, which production never
        /// does — the whole point is that data goes one way.
        fn read_ui(&self, handoff: &Handoff, verts: u32) -> Vec<DrapeVertex> {
            let bytes = u64::from(verts) * VERTEX_SIZE;
            let staging = self.ui.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test UI readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .ui
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(handoff.vertex_buffer(), 0, &staging, 0, bytes);
            self.ui_queue.submit([encoder.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.ui
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            let out =
                bytemuck::cast_slice::<u8, DrapeVertex>(&staging.slice(..).get_mapped_range())
                    .to_vec();
            staging.unmap();
            out
        }
    }

    fn pattern(n: u32, seed: f32) -> Vec<DrapeVertex> {
        (0..n)
            .map(|i| {
                let f = i as f32;
                DrapeVertex {
                    position: [f + seed, f * 2.0, -f],
                    normal: [0.0, 1.0, 0.0],
                    strain: f * 0.001 + seed,
                }
            })
            .collect()
    }

    macro_rules! rig {
        () => {
            match Rig::new() {
                Some(rig) => rig,
                None => {
                    eprintln!("no GPU adapter; skipping");
                    return;
                }
            }
        };
    }

    /// The correctness half: a known pattern must arrive on the UI device
    /// byte-identical. The benchmark only proves speed.
    #[test]
    fn round_trip_preserves_every_byte() {
        let rig = rig!();
        const N: u32 = 4096;
        let verts = pattern(N, 0.5);
        let src = rig.source(&verts);
        let mut handoff = rig.handoff(N);

        assert!(handoff.begin(Epoch(1), &src, N));
        assert_eq!(handoff.drain(), Some(Epoch(1)));
        assert_eq!(handoff.displayed(), Some((Epoch(1), N)));
        assert_eq!(rig.read_ui(&handoff, N), verts);
    }

    /// Same input twice → identical bytes. No wall-clock, no ordering luck.
    #[test]
    fn determinism_same_input_same_bytes() {
        let rig = rig!();
        const N: u32 = 2048;
        let verts = pattern(N, -3.25);
        let src = rig.source(&verts);
        let mut handoff = rig.handoff(N);

        let mut runs = Vec::new();
        for epoch in 1..=3u64 {
            assert!(handoff.begin(Epoch(epoch), &src, N));
            handoff.drain();
            runs.push(rig.read_ui(&handoff, N));
        }
        assert_eq!(runs[0], verts);
        assert!(
            runs.iter().all(|r| *r == runs[0]),
            "handoff is not deterministic"
        );
    }

    /// Invariant 2: with all three slots in flight, `begin` refuses rather
    /// than reusing a buffer that is still being written.
    #[test]
    fn rotation_never_reuses_a_slot_in_flight() {
        let rig = rig!();
        const N: u32 = 1024;
        let src = rig.source(&pattern(N, 0.0));
        let mut handoff = rig.handoff(N);

        for epoch in 1..=SLOTS as u64 {
            assert!(handoff.begin(Epoch(epoch), &src, N), "slot {epoch} refused");
            assert_eq!(handoff.free_slots(), SLOTS - epoch as usize);
        }
        assert_eq!(handoff.free_slots(), 0);
        assert!(
            !handoff.begin(Epoch(99), &src, N),
            "handed out a fourth slot with three readbacks in flight"
        );

        // And they all come back, so the refusal was backpressure, not a leak.
        handoff.drain();
        assert_eq!(handoff.free_slots(), SLOTS);
    }

    /// Invariant 3, at submit: an epoch older than one already started is
    /// dropped before any GPU work happens.
    #[test]
    fn stale_epoch_is_refused_at_submit() {
        let rig = rig!();
        const N: u32 = 512;
        let newer = pattern(N, 7.0);
        let src_new = rig.source(&newer);
        let src_old = rig.source(&pattern(N, -7.0));
        let mut handoff = rig.handoff(N);

        assert!(handoff.begin(Epoch(5), &src_new, N));
        assert!(
            !handoff.begin(Epoch(3), &src_old, N),
            "stale epoch accepted"
        );
        assert!(
            !handoff.begin(Epoch(5), &src_old, N),
            "repeat epoch accepted"
        );
        handoff.drain();

        assert_eq!(handoff.displayed(), Some((Epoch(5), N)));
        assert_eq!(
            rig.read_ui(&handoff, N),
            newer,
            "a stale frame reached the UI"
        );
    }

    /// Invariant 3, at upload: when several readbacks land in the same pump,
    /// only the newest is displayed and the older ones are dropped — even
    /// though every one of them completed successfully.
    #[test]
    fn only_the_newest_completed_frame_is_displayed() {
        let rig = rig!();
        const N: u32 = 512;
        let last = pattern(N, 3.0);
        let sources = [
            rig.source(&pattern(N, 1.0)),
            rig.source(&pattern(N, 2.0)),
            rig.source(&last),
        ];
        let mut handoff = rig.handoff(N);

        for (i, src) in sources.iter().enumerate() {
            assert!(handoff.begin(Epoch(i as u64 + 1), src, N));
        }
        // All three are in flight; one drain resolves them together.
        assert_eq!(handoff.free_slots(), 0);
        assert_eq!(handoff.drain(), Some(Epoch(3)));
        assert_eq!(handoff.displayed(), Some((Epoch(3), N)));
        assert_eq!(
            rig.read_ui(&handoff, N),
            last,
            "an older completed frame overwrote the newest"
        );
    }

    /// `pump` must never block, and must report nothing when nothing is ready.
    #[test]
    fn pump_is_a_no_op_when_idle() {
        let rig = rig!();
        let mut handoff = rig.handoff(64);
        assert_eq!(handoff.pump(), None);
        assert_eq!(handoff.displayed(), None);
        assert_eq!(handoff.free_slots(), SLOTS);
    }

    /// The premise of the whole crate, checked rather than asserted in prose:
    /// iced's limits cannot host the solver's 17-storage-buffer shaders.
    #[test]
    fn iced_limits_cannot_host_the_solver() {
        assert_eq!(
            wgpu::Limits::default().max_storage_buffers_per_shader_stage,
            8
        );
        assert_eq!(iced_limits().max_storage_buffers_per_shader_stage, 8);
        // unified_gather.wgsl declares 17.
        assert!(iced_limits().max_storage_buffers_per_shader_stage < 17);
    }
}
