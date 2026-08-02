// SPDX-License-Identifier: GPL-2.0-or-later
// Bring-up geometry: a lit, waving grid standing in for a draped panel.
// The mesh is generated from the vertex index — no vertex buffers to keep in
// sync while the real cloth pipeline is still being built.

struct Uniforms {
    view_projection: mat4x4<f32>,
    // WGSL rounds the struct to its 16-byte alignment: 64 + 4 -> 80 bytes,
    // which is exactly `Uniforms` in `renderer.rs` with its explicit padding.
    time: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

const CELLS: u32 = 32u;
const AMPLITUDE: f32 = 0.11;

fn height(p: vec2<f32>) -> f32 {
    return AMPLITUDE * sin(p.x * 3.1 + u.time * 1.7)
         + AMPLITUDE * 0.6 * cos(p.y * 4.3 - u.time * 1.1);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    let cell = index / 6u;
    let corner = index % 6u;

    var offsets = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u),
        vec2<u32>(0u, 1u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u),
    );

    let grid = vec2<u32>(cell % CELLS, cell / CELLS) + offsets[corner];
    let uv = vec2<f32>(f32(grid.x), f32(grid.y)) / f32(CELLS);
    let p = (uv - 0.5) * 2.0;

    let world = vec3<f32>(p.x, height(p), p.y);

    // Central difference normal — cheap and exact enough for a bring-up.
    let e = 0.01;
    let dx = (height(p + vec2<f32>(e, 0.0)) - height(p - vec2<f32>(e, 0.0))) / (2.0 * e);
    let dz = (height(p + vec2<f32>(0.0, e)) - height(p - vec2<f32>(0.0, e))) / (2.0 * e);

    var out: VsOut;
    out.clip = u.view_projection * vec4<f32>(world, 1.0);
    out.normal = normalize(vec3<f32>(-dx, 1.0, -dz));
    out.uv = uv;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light = normalize(vec3<f32>(0.4, 0.8, 0.45));

    // Two-sided: we see the back of the sheet as it waves.
    let lambert = abs(dot(n, light));
    let ambient = 0.18;

    // A weave check so the surface reads as fabric, and so a wrong UV or a
    // missing pass is obvious on screen rather than subtle.
    let checks = vec2<f32>(floor(in.uv * 8.0));
    let weave = select(0.88, 1.0, (checks.x + checks.y) % 2.0 < 0.5);

    let base = vec3<f32>(0.86, 0.33, 0.38) * weave;
    let linear = base * (ambient + 0.9 * lambert);

    // iced's default surface is gamma-space (its `web-colors` feature is on),
    // so encode here rather than relying on an sRGB attachment.
    return vec4<f32>(pow(linear, vec3<f32>(1.0 / 2.2)), 1.0);
}
