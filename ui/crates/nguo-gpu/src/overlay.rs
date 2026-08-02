// SPDX-License-Identifier: GPL-2.0-or-later
//! The egui debug overlay, running inside iced's frame on iced's device.
//!
//! Plumbing only — `nguo-app-debug` decides what the overlay says.

use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use iced::widget::shader::{self, Primitive};

/// The egui context + renderer. Persistent across frames.
pub struct OverlayPipeline {
    context: egui::Context,
    renderer: egui_wgpu::Renderer,
    jobs: Vec<egui::ClippedPrimitive>,
    screen: ScreenDescriptor,
}

impl std::fmt::Debug for OverlayPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayPipeline")
            .field("jobs", &self.jobs.len())
            .finish_non_exhaustive()
    }
}

impl shader::Pipeline for OverlayPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        crate::adopt(device, queue, format, "egui-overlay");

        Self {
            context: egui::Context::default(),
            // egui wants a gamma-space target, which is exactly what iced
            // gives us with its default `web-colors` feature.
            renderer: egui_wgpu::Renderer::new(
                device,
                format,
                egui_wgpu::RendererOptions::default(),
            ),
            jobs: Vec::new(),
            screen: ScreenDescriptor {
                size_in_pixels: [1, 1],
                pixels_per_point: 1.0,
            },
        }
    }
}

/// A frame of overlay UI.
///
// ponytail: the overlay is display-only — we feed egui a screen rect and no
// pointer events, so nothing in it is clickable. Wire `RawInput.events` from
// iced's `mouse::Event` when the profiler grows knobs.
#[derive(Clone)]
pub struct OverlayPrimitive {
    ui: Arc<dyn Fn(&egui::Context) + Send + Sync>,
}

impl OverlayPrimitive {
    pub fn new(ui: impl Fn(&egui::Context) + Send + Sync + 'static) -> Self {
        Self { ui: Arc::new(ui) }
    }
}

impl std::fmt::Debug for OverlayPrimitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OverlayPrimitive")
    }
}

impl Primitive for OverlayPrimitive {
    type Pipeline = OverlayPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &iced::Rectangle,
        viewport: &shader::Viewport,
    ) {
        // egui's renderer resets the pass viewport to the full target
        // (`egui-wgpu-0.33.3/src/renderer.rs:489`), so the overlay is always
        // the whole window and its bounds are ignored.
        crate::note_viewport(viewport);

        let physical = viewport.physical_size();
        let logical = viewport.logical_size();

        // Tell egui the DPI *before* laying out, and then use the value it
        // actually settled on for both tessellation and the screen descriptor.
        // Any disagreement between the two makes epaint warn on every glyph.
        pipeline
            .context
            .set_pixels_per_point(viewport.scale_factor());
        let pixels_per_point = pipeline.context.pixels_per_point();

        pipeline.screen = ScreenDescriptor {
            size_in_pixels: [physical.width, physical.height],
            pixels_per_point,
        };

        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(logical.width, logical.height),
            )),
            ..Default::default()
        };

        let output = pipeline.context.run(input, |ctx| (self.ui)(ctx));

        pipeline.jobs = pipeline.context.tessellate(output.shapes, pixels_per_point);

        for (id, delta) in &output.textures_delta.set {
            pipeline.renderer.update_texture(device, queue, *id, delta);
        }

        // `update_buffers` needs an encoder and `prepare` is not given one, so
        // we submit our own. iced's frame encoder is submitted after all
        // `prepare` calls (`iced_wgpu-0.14.0/src/lib.rs:165`), so this lands
        // first — which is the order we want.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nguo egui upload"),
        });
        let buffers = pipeline.renderer.update_buffers(
            device,
            queue,
            &mut encoder,
            &pipeline.jobs,
            &pipeline.screen,
        );
        queue.submit(buffers.into_iter().chain([encoder.finish()]));

        for id in &output.textures_delta.free {
            pipeline.renderer.free_texture(id);
        }
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        _clip_bounds: &iced::Rectangle<u32>,
    ) {
        if pipeline.jobs.is_empty() {
            return;
        }

        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nguo egui overlay pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pipeline.renderer.render(
            &mut pass.forget_lifetime(),
            &pipeline.jobs,
            &pipeline.screen,
        );
    }
}
