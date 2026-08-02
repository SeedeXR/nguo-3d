// SPDX-License-Identifier: GPL-2.0-or-later
//! The Vello 2D canvas as an iced custom primitive.
//!
//! `nguo-canvas` builds a [`vello::Scene`]; this module gets it onto the
//! screen through iced's device.

use std::num::NonZeroUsize;
use std::sync::Arc;

use vello::peniko;

use iced::widget::shader::{self, Primitive};

use crate::blit::Blit;
use crate::scale::{self, PhysicalRect};

/// Vello renderer + its offscreen target, shared by every canvas widget.
pub struct VelloPipeline {
    // `vello::Renderer` is `Send` but not `Sync` (it holds a
    // `RefCell<Vec<u8>>` in its `BindMap`, `vello-0.7.0/src/wgpu_engine.rs:108`)
    // and iced requires `Pipeline: MaybeSync`
    // (`iced_wgpu-0.14.0/src/primitive.rs:22`). One uncontended lock per frame.
    renderer: std::sync::Mutex<vello::Renderer>,
    blit: Blit,
    target: Option<Target>,
    /// Where the current frame's scene goes, in physical pixels.
    rect: PhysicalRect,
}

struct Target {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl std::fmt::Debug for VelloPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelloPipeline")
            .field("rect", &self.rect)
            .finish()
    }
}

impl VelloPipeline {
    fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if matches!(&self.target, Some(t) if t.width == width && t.height == height) {
            return;
        }

        tracing::debug!(target: "nguo_gpu", width, height, "(re)allocating Vello target");

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nguo vello target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Rgba8Unorm + STORAGE_BINDING is what `render_to_texture`
            // requires (`vello-0.7.0/src/lib.rs:459-462`).
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.blit.bind(device, &view);

        self.target = Some(Target {
            view,
            bind_group,
            width,
            height,
        });
    }

    /// Rasterises `scene` into the offscreen target. Vello submits its own
    /// command buffer here; iced's frame encoder is submitted afterwards, so
    /// the ordering is correct without any explicit sync.
    pub fn rasterize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &vello::Scene,
        base_color: peniko::Color,
        width: u32,
        height: u32,
    ) {
        self.ensure_target(device, width, height);
        // Clone the view (wgpu handles are Arc-backed) so the borrow of
        // `self.target` ends before we take `self.renderer` mutably.
        let Some(view) = self.target.as_ref().map(|target| target.view.clone()) else {
            return;
        };

        let renderer = self
            .renderer
            .get_mut()
            .expect("the Vello renderer lock is never poisoned");

        if let Err(error) = renderer.render_to_texture(
            device,
            queue,
            scene,
            &view,
            &vello::RenderParams {
                base_color,
                width,
                height,
                antialiasing_method: vello::AaConfig::Area,
            },
        ) {
            tracing::error!(target: "nguo_gpu", %error, "Vello render failed");
        }
    }

    /// Fills the pass's current viewport with the rasterised scene.
    pub fn blit_into(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(target) = &self.target {
            self.blit.draw(pass, &target.bind_group);
        }
    }
}

impl shader::Pipeline for VelloPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        crate::adopt(device, queue, format, "vello-canvas");

        let renderer = vello::Renderer::new(
            device,
            vello::RendererOptions {
                use_cpu: false,
                // One AA mode = one set of `fine` permutations to compile.
                antialiasing_support: vello::AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )
        .expect("Vello renderer should build on iced's device");

        Self {
            renderer: std::sync::Mutex::new(renderer),
            blit: Blit::new(device, format),
            target: None,
            rect: PhysicalRect::default(),
        }
    }
}

/// One frame of 2D canvas content.
#[derive(Clone)]
pub struct ScenePrimitive {
    pub scene: Arc<vello::Scene>,
    pub base_color: peniko::Color,
}

impl std::fmt::Debug for ScenePrimitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScenePrimitive")
            .field("base_color", &self.base_color)
            .finish_non_exhaustive()
    }
}

impl Primitive for ScenePrimitive {
    type Pipeline = VelloPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &iced::Rectangle,
        viewport: &shader::Viewport,
    ) {
        let scale_factor = viewport.scale_factor();
        crate::note_viewport(viewport);

        // ponytail: one canvas widget, so the pipeline can hold this frame's
        // rect. Key it by widget id when we grow a second canvas.
        pipeline.rect = scale::physical_rect(
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
            scale_factor,
        );

        if pipeline.rect.is_empty() {
            return;
        }

        // The scene arrives in LOGICAL units (`Program::draw` is never told the
        // scale factor) and the target is PHYSICAL, so the HiDPI scale is
        // applied here — one place, at the boundary (guide §2.4).
        //
        // ponytail: this costs one encoding copy per frame. If it ever shows up
        // in a profile, push the scale factor into scene building instead
        // (`iced::window::scale_factor(id) -> Task<f32>`,
        // `iced_runtime-0.14.0/src/window.rs:376`).
        let scene = if (scale_factor - 1.0).abs() < f32::EPSILON {
            None
        } else {
            let mut scaled = vello::Scene::new();
            scaled.append(
                &self.scene,
                Some(vello::kurbo::Affine::scale(scale_factor as f64)),
            );
            Some(scaled)
        };

        pipeline.rasterize(
            device,
            queue,
            scene.as_ref().unwrap_or(&self.scene),
            self.base_color,
            pipeline.rect.width,
            pipeline.rect.height,
        );
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &iced::Rectangle<u32>,
    ) {
        if pipeline.target.is_none()
            || pipeline.rect.is_empty()
            || clip_bounds.width == 0
            || clip_bounds.height == 0
        {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nguo canvas blit pass"),
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

        pass.set_viewport(
            pipeline.rect.x as f32,
            pipeline.rect.y as f32,
            pipeline.rect.width as f32,
            pipeline.rect.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pipeline.blit_into(&mut pass);
    }
}
