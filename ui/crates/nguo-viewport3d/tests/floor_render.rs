// SPDX-License-Identifier: GPL-2.0-or-later
//! The ground grid, rendered on a real GPU and read back.
//!
//! `shaders.rs` proves the WGSL compiles; nothing there proves it *draws*. A
//! grid shader that discards every pixel, unprojects onto the wrong plane, or
//! puts the axis lines in the wrong colour validates perfectly clean. This
//! renders it and looks at the pixels.
//!
//! Skips (and passes) with no adapter, which is what a CI box without a GPU
//! has — the same shape as the bundle-gated tests, loud skip included. Set
//! `NGUO_REQUIRE_GPU=1` to make the skip a failure instead. Run locally on
//! Metal, which is where the numbers in the report came from.

use glam::Vec3;
use nguo_gpu::scale::PhysicalRect;
use nguo_viewport3d::{camera::Camera, grid::Floor, GridRenderer};

const SIZE: u32 = 256;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Loud skip — written to fd 2 directly, because libtest's capture swallows
/// `eprintln!` on a *passing* test, and a skip nobody can see is how three GPU
/// tests quietly stop testing anything (`posed_collider.rs` uses the same shape).
/// Set `NGUO_REQUIRE_GPU=1` to turn the skip into a failure; CI is `macos-15`,
/// which has an adapter, so nothing there should ever take this path.
fn skip(msg: &str) -> Option<Gpu> {
    assert!(
        std::env::var_os("NGUO_REQUIRE_GPU").is_none(),
        "NGUO_REQUIRE_GPU is set but {msg}"
    );
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {msg}");
    None
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        ..Default::default()
    })) else {
        return skip("no wgpu adapter");
    };
    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("nguo grid test device"),
        ..Default::default()
    })) else {
        return skip("the wgpu adapter refused a device");
    };
    Some(Gpu { device, queue })
}

/// Render one frame of floor for `camera` and return it as RGBA8 rows.
fn render(gpu: &Gpu, camera: &Camera) -> Vec<[u8; 4]> {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("grid test target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // 256 * 4 = 1024, already a multiple of COPY_BYTES_PER_ROW_ALIGNMENT.
    let bytes = (SIZE * SIZE * 4) as u64;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let renderer = GridRenderer::new(&gpu.device, FORMAT);
    renderer.upload(
        &gpu.queue,
        camera.view_projection(1.0),
        camera,
        Floor::for_camera(camera, SIZE as f32),
        &nguo_ui_theme::Tokens::dark(),
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Clear to black first: the floor pass loads rather than clears, exactly
    // as it does inside the iced frame.
    encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("grid test clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        })
        .forget_lifetime();

    let whole = PhysicalRect {
        x: 0,
        y: 0,
        width: SIZE,
        height: SIZE,
    };
    renderer.draw(&mut encoder, &view, whole, whole);

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = slice.get_mapped_range();
    data.chunks_exact(4)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect()
}

/// Looking down at the origin from above: the whole plane is in view.
fn looking_down() -> Camera {
    Camera {
        yaw: 0.0,
        pitch: 1.2,
        distance: 4.0,
        target: Vec3::ZERO,
        ..Camera::default()
    }
}

fn lit(pixels: &[[u8; 4]]) -> usize {
    pixels
        .iter()
        .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
        .count()
}

#[test]
fn the_floor_draws_lines_and_both_axes() {
    let Some(gpu) = gpu() else { return };
    let pixels = render(&gpu, &looking_down());

    let lines = lit(&pixels);
    assert!(
        lines > 200,
        "only {lines} of {} pixels drew — the floor is not there",
        pixels.len()
    );
    assert!(
        lines < pixels.len() / 2,
        "{lines} pixels drew: that is a fill, not a grid"
    );

    // The X axis is `error` red and the Z axis is `accent` blue (Blender's
    // X-red / Z-blue, overlays.rst:47). Both must be on screen, and neither
    // may be the grey the minor lines are.
    //
    // Tested as channel *ratios*, not differences: the distance fade scales
    // all three channels together, so an absolute threshold measures how far
    // down the line the fade bit rather than what colour it is.
    let ratio = |p: &[u8; 4], hi: usize, a: usize, b: usize| {
        p[hi] > 20 && p[hi] as f32 > p[a] as f32 * 1.5 && p[hi] as f32 > p[b] as f32 * 1.2
    };
    let red = pixels.iter().filter(|p| ratio(p, 0, 1, 2)).count();
    let blue = pixels.iter().filter(|p| ratio(p, 2, 0, 1)).count();
    assert!(red > 20, "no red X axis: {red} px");
    assert!(blue > 20, "no blue Z axis: {blue} px");
}

/// The control. Point the camera at the sky and every ray misses the plane, so
/// the pass must draw **nothing**. Without this, a shader that painted the
/// whole viewport a flat grey would satisfy the test above.
#[test]
fn a_camera_looking_at_the_sky_draws_no_floor_at_all() {
    let Some(gpu) = gpu() else { return };
    let camera = Camera {
        // Below the plane, looking further down — the floor is behind the eye.
        pitch: -1.4,
        target: Vec3::new(0.0, 40.0, 0.0),
        distance: 4.0,
        yaw: 0.0,
        ..Camera::default()
    };
    let pixels = render(&gpu, &camera);
    assert_eq!(lit(&pixels), 0, "the floor drew where there is no floor");
}

/// Zooming out coarsens the cell, so the *same* viewport shows a comparable
/// number of lines instead of a solid wash. This is the adaptive rule visible
/// in pixels rather than in a unit test's arithmetic.
#[test]
fn pulling_the_camera_back_does_not_turn_the_floor_into_a_wash() {
    let Some(gpu) = gpu() else { return };
    let near = lit(&render(
        &gpu,
        &Camera {
            distance: 1.0,
            ..looking_down()
        },
    ));
    let far = lit(&render(
        &gpu,
        &Camera {
            distance: 40.0,
            ..looking_down()
        },
    ));
    let ratio = far as f32 / near as f32;
    assert!(
        (0.25..4.0).contains(&ratio),
        "40x the distance changed the lit-pixel count {near} -> {far} ({ratio:.2}x); \
         the cell is not adapting"
    );
}
