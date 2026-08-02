// SPDX-License-Identifier: GPL-2.0-or-later
// Full-viewport triangle that copies an offscreen texture into the frame.

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    // 0 -> (-1,-1), 1 -> (-1, 3), 2 -> ( 3,-1): one triangle covering the
    // whole viewport rectangle set by the caller.
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) % 2) * 4.0 - 1.0;

    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src, src_sampler, in.uv);
}
