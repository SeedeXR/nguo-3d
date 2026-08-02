// SPDX-License-Identifier: GPL-2.0-or-later
//! Headless end-to-end frame: runs the whole render graph (viewport pass ->
//! Vello canvas pass) into an offscreen texture and reads the pixels back.
//!
//! This is the regression net for "the pass never ran" — the failure mode that
//! a window-less CI can otherwise never see. It deliberately builds the device
//! with the same limits iced asks for
//! (`iced_wgpu-0.14.0/src/window/compositor.rs:151-159`), so it also proves
//! Vello and the viewport fit inside iced's device envelope.
//!
//! Everything a window would add — surface acquire, present, resize, input —
//! is NOT covered here. See the task report.

use iced::widget::shader::Pipeline as _;
use nguo_canvas::View;
use nguo_gpu::canvas::VelloPipeline;
use nguo_gpu::scale::PhysicalRect;
use nguo_viewport3d::{Camera, ClothRenderer};
use vello::kurbo::Vec2;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 384;
/// What iced's compositor picks with its default `web-colors` feature.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
/// Nothing else in the frame is this colour, so any pixel still holding it was
/// never painted.
const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 1.0,
    b: 0.0,
    a: 1.0,
};

struct Pixels {
    data: Vec<u8>,
}

impl Pixels {
    /// `(r, g, b)` — the texture is BGRA.
    fn at(&self, x: u32, y: u32) -> (u8, u8, u8) {
        let i = ((y * WIDTH + x) * 4) as usize;
        (self.data[i + 2], self.data[i + 1], self.data[i])
    }

    fn is_untouched(&self, x: u32, y: u32) -> bool {
        let (r, g, b) = self.at(x, y);
        r < 8 && g > 247 && b < 8
    }
}

#[test]
fn one_frame_of_the_render_graph_paints_both_widgets() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("a wgpu adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("nguo headless test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            max_bind_groups: 2,
            max_non_sampler_bindings: 2048,
            ..wgpu::Limits::default()
        },
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .expect("a wgpu device with iced's limits");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless frame"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let left = PhysicalRect {
        x: 0,
        y: 0,
        width: WIDTH / 2,
        height: HEIGHT,
    };
    let right = PhysicalRect {
        x: WIDTH / 2,
        y: 0,
        width: WIDTH / 2,
        height: HEIGHT,
    };

    let mut cloth = ClothRenderer::new(&device, FORMAT);
    cloth.ensure_depth(&device, WIDTH, HEIGHT);
    cloth.upload(
        &queue,
        nguo_viewport3d::view_projection(&Camera::default(), left.width, left.height),
        0.0,
    );

    let mut canvas = VelloPipeline::new(&device, &queue, FORMAT);
    let scene = nguo_canvas::build_scene(
        &View::default(),
        Vec2::new(right.width as f64, right.height as f64),
        None,
    );
    // Vello submits its own command buffer, exactly as it does in `prepare`.
    canvas.rasterize(
        &device,
        &queue,
        &scene,
        nguo_canvas::BACKGROUND,
        right.width,
        right.height,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless frame"),
    });

    // Step 0: the clear iced's compositor would do.
    drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(CLEAR),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    }));

    // Step 1: the 3D viewport pass.
    cloth.draw(&mut encoder, &view, left, left);

    // Step 2: the Vello canvas blit.
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("canvas blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
            right.x as f32,
            right.y as f32,
            right.width as f32,
            right.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(right.x, right.y, right.width, right.height);
        canvas.blit_into(&mut pass);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (WIDTH * HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 4), // already 256-aligned
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("readback should map");
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("the queue should drain");

    let pixels = Pixels {
        data: readback.slice(..).get_mapped_range().to_vec(),
    };

    // The canvas covers its whole rectangle with the Vello base colour, so a
    // single untouched pixel there means the blit never happened.
    let untouched_right = (right.x..right.x + right.width)
        .flat_map(|x| (0..HEIGHT).map(move |y| (x, y)))
        .filter(|&(x, y)| pixels.is_untouched(x, y))
        .count();
    assert_eq!(
        untouched_right, 0,
        "the Vello canvas pass did not cover its widget"
    );

    // The blue petal fill must actually be there — not just the background.
    let blue = (right.x..right.x + right.width)
        .flat_map(|x| (0..HEIGHT).map(move |y| (x, y)))
        .filter(|&(x, y)| {
            let (r, _, b) = pixels.at(x, y);
            b > 128 && b as i32 - r as i32 > 40
        })
        .count();
    assert!(
        blue > 1_000,
        "expected the filled Bezier shape in the canvas, found {blue} blue pixels"
    );

    // The cloth only covers part of its rectangle, so count instead.
    let painted_left = (0..left.width)
        .flat_map(|x| (0..HEIGHT).map(move |y| (x, y)))
        .filter(|&(x, y)| !pixels.is_untouched(x, y))
        .count();
    let total_left = (left.width * HEIGHT) as usize;
    assert!(
        painted_left * 100 / total_left > 10,
        "expected the lit cloth to cover a good part of the viewport, got \
         {painted_left}/{total_left} pixels"
    );

    // ...and it must be the cloth's warm colour, not stray background.
    let warm = (0..left.width)
        .flat_map(|x| (0..HEIGHT).map(move |y| (x, y)))
        .filter(|&(x, y)| {
            let (r, g, b) = pixels.at(x, y);
            r > 100 && r > g + 30 && r > b + 30
        })
        .count();
    assert!(
        warm > 1_000,
        "expected lit cloth pixels in the viewport, found {warm}"
    );
}
