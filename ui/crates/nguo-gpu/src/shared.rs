// SPDX-License-Identifier: GPL-2.0-or-later
//! The single shared `wgpu` device — see the crate docs for why iced owns it.

use std::sync::OnceLock;

/// The one device/queue every renderer in the process uses.
#[derive(Debug, Clone)]
pub struct Shared {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// The surface format iced picked for the window.
    pub format: wgpu::TextureFormat,
}

static SHARED: OnceLock<Shared> = OnceLock::new();

/// Registers the device iced handed us, or asserts it is the one we already
/// have. Every renderer (viewport, Vello canvas, egui overlay) calls this from
/// its `Pipeline::new`, which is what makes "one device" a checked invariant
/// rather than a hope.
///
/// # Panics
/// If two different devices show up. That is the bug this exists to catch.
pub fn adopt(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    who: &str,
) -> &'static Shared {
    let shared = SHARED.get_or_init(|| {
        tracing::info!(
            target: "nguo_gpu",
            %who, ?format,
            "shared wgpu device adopted from iced"
        );
        Shared {
            device: device.clone(),
            queue: queue.clone(),
            format,
        }
    });

    assert!(
        &shared.device == device && &shared.queue == queue,
        "{who} was handed a different wgpu device — the one-device invariant \
         is broken"
    );
    tracing::debug!(target: "nguo_gpu", %who, "joined the shared device");

    shared
}

/// The shared device, if any renderer has been initialised yet.
pub fn shared() -> Option<&'static Shared> {
    SHARED.get()
}

/// Last frame's window geometry, recorded by whichever primitive prepared
/// first. iced gives this to primitives but not to `view()`, and the debug
/// overlay wants to show it, so we stash it here rather than plumbing an
/// `iced::window::scale_factor` task round trip through the application state.
mod viewport {
    use std::sync::atomic::{AtomicU32, Ordering};

    pub(super) static WIDTH: AtomicU32 = AtomicU32::new(0);
    pub(super) static HEIGHT: AtomicU32 = AtomicU32::new(0);
    pub(super) static SCALE_BITS: AtomicU32 = AtomicU32::new(0);

    pub(super) fn store(width: u32, height: u32, scale: f32) {
        WIDTH.store(width, Ordering::Relaxed);
        HEIGHT.store(height, Ordering::Relaxed);
        SCALE_BITS.store(scale.to_bits(), Ordering::Relaxed);
    }

    pub(super) fn load() -> (u32, u32, f32) {
        (
            WIDTH.load(Ordering::Relaxed),
            HEIGHT.load(Ordering::Relaxed),
            f32::from_bits(SCALE_BITS.load(Ordering::Relaxed)),
        )
    }
}

/// Records the frame geometry. Called from every primitive's `prepare`.
pub fn note_viewport(viewport: &iced::widget::shader::Viewport) {
    let size = viewport.physical_size();
    viewport::store(size.width, size.height, viewport.scale_factor());
}

/// `(physical width, physical height, scale factor)` of the last frame.
pub fn window_geometry() -> (u32, u32, f32) {
    viewport::load()
}

/// A human-readable name for the adapter iced is most likely using.
///
// ponytail: iced 0.14 keeps its `Adapter` private
// (`iced_wgpu-0.14.0/src/window/compositor.rs:13`) and `wgpu::Device` exposes
// no adapter info, so we enumerate and report what iced's own request options
// (`PowerPreference::HighPerformance`, compositor.rs:85) would pick. Exact on a
// single-GPU machine; if we ever ship on a dual-GPU laptop, take the name from
// a patched compositor instead of guessing.
pub fn adapter_description() -> String {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    let pick = adapters
        .iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .or_else(|| adapters.first());

    match pick {
        Some(adapter) => {
            let info = adapter.get_info();
            format!(
                "{} · {:?} · {:?}",
                info.name, info.backend, info.device_type
            )
        }
        None => "<no adapter>".to_owned(),
    }
}
