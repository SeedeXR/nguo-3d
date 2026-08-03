// SPDX-License-Identifier: GPL-2.0-or-later
// The avatar body: an indexed triangle mesh, two-light Lambert.
//
// `base_color` is a LINEAR baseColorFactor (nguo_avatar::MonkTone::factor),
// never a swatch colour — feeding sRGB in here washes the skin out.
//
// There is no albedo texture bound, so the factor IS the albedo here. The
// bundle's factor is pre-divided by the neutral map's 0.9337 mean, which makes
// the flat body read ~7 % brighter than the published Monk tone.
// ponytail: within the noise for a body you drape cloth on. Bind
// `SkinMap::AlbedoNeutral` with the tiling uv0 set (AddressMode::Repeat, one
// tile = 0.20 m) and the factor becomes exactly right — that is also what buys
// pore detail, so do both at once or neither.

struct Uniforms {
    view_projection: mat4x4<f32>,
    base_color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
}

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VsOut {
    var out: VsOut;
    // The bundle is metres, +Y up, +Z forward, right-handed — the same space
    // the camera works in, so there is no conversion here on purpose.
    out.clip = u.view_projection * vec4<f32>(position, 1.0);
    out.normal = normal;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let key = normalize(vec3<f32>(0.35, 0.75, 0.55));
    let fill = normalize(vec3<f32>(-0.55, 0.15, -0.60));

    // Two-sided: the mesh is not back-face culled, so a triangle seen from
    // behind must still light rather than go black.
    let lambert = abs(dot(n, key)) + 0.28 * abs(dot(n, fill));
    let linear = u.base_color.rgb * (0.14 + 0.78 * lambert);

    // iced's surface is gamma-space (its `web-colors` feature is on), so
    // encode here rather than relying on an sRGB attachment — same as
    // cloth.wgsl.
    return vec4<f32>(pow(linear, vec3<f32>(1.0 / 2.2)), 1.0);
}
