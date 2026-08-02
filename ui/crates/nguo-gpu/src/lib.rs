// SPDX-License-Identifier: GPL-2.0-or-later
//! Shared GPU layer for the nguo-3d desktop shell.
//!
//! This is the one crate that knows iced's `wgpu` renderer, Vello and
//! `egui-wgpu` at the same time.
//!
//! # Device ownership — READ THIS
//!
//! The original design asked for `nguo-gpu` to create the `Instance`/`Adapter`/`Device`/
//! `Queue` and hand borrows to iced. **iced 0.14 does not allow that.** Verified
//! in the installed sources:
//!
//! * `iced_winit-0.14.0/src/lib.rs:593` — the runtime builds the compositor
//!   itself: `<P::Renderer as compositor::Default>::Compositor::new(..)`. The
//!   compositor type is an associated type of the renderer; there is no
//!   parameter and no setter.
//! * `iced_wgpu-0.14.0/src/window/compositor.rs:50-95` —
//!   `Compositor::request()` calls `new_instance_with_webgpu_detection`,
//!   `instance.request_adapter(..)` and (`:165`) `adapter.request_device(..)`
//!   unconditionally. Its fields (`:11-18`) are private, so the struct cannot be
//!   assembled from outside either.
//!
//! So the ownership is inverted, which satisfies "exactly one device" equally
//! well: **iced creates the device, and every other renderer borrows it.**
//! Everyone gets it through the two callbacks iced hands out —
//! `Pipeline::new(device, queue, format)` and
//! `Primitive::prepare(.., device, queue, ..)`
//! (`iced_wgpu-0.14.0/src/primitive.rs:25-32,76-82`) — and every one of them
//! routes through [`adopt`], which asserts it really is the same device.
//!
//! See the crate README section in the task report for the two ways to flip
//! ownership back (custom `graphics::Compositor` impl, or a `[patch]`ed
//! `iced_wgpu`), and for why the *engine* cannot use iced's device as-is.
//!
//! # Frame graph
//!
//! iced owns the surface and the frame `CommandEncoder`; our passes are
//! sequenced inside it. Per frame, `iced_wgpu-0.14.0/src/lib.rs:417-640`
//! walks layers in order and, within a layer, runs
//! `quads -> triangles -> custom primitives -> images -> text`. Custom
//! primitives that return `false` from `draw` are handed the encoder in
//! `render` **in widget order**. Both of our primitives use the `render` path,
//! so the order the guide asks for falls out of the widget order in
//! [`compose`]-ing the view:
//!
//! ```text
//! 3D viewport pass -> Vello canvas blit -> iced UI pass -> egui overlay -> present
//! ```
//!
//! Surface acquire / resize / lost is handled centrally by
//! `iced_wgpu-0.14.0/src/window/compositor.rs:218-256` (`present`) plus
//! `iced_winit`'s `Outdated | Lost => compositor.configure_surface(..)`
//! recovery; we do not duplicate it.

pub mod blit;
pub mod canvas;
pub mod handoff;
pub mod scale;
pub mod stats;

#[cfg(feature = "debug-overlay")]
pub mod overlay;

mod shared;

pub use shared::{adapter_description, adopt, note_viewport, shared, window_geometry, Shared};
