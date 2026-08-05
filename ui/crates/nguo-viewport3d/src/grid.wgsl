// SPDX-License-Identifier: GPL-2.0-or-later
// The ground-plane floor, as one full-viewport triangle.
//
// There is no grid mesh: every pixel unprojects itself onto y == 0 and asks
// how far it is from the nearest line. That is why the floor is infinite, why
// it never tessellates, and why it costs one draw call of three vertices no
// matter how far you zoom out.
//
// Anti-aliasing is `fwidth` on the *world* coordinate, which is also the
// grazing-angle fade Blender's floor has: as the plane tips towards edge-on, a
// pixel spans more and more world space, `fwidth` grows, and the line fades
// out instead of aliasing into a moire. Nothing special-cases the horizon.
//
// What to draw is decided in `grid.rs`; this file only draws it.

struct Uniforms {
    inverse_view_projection: mat4x4<f32>,
    // xyz: eye in world space. w: distance at which the floor has faded out.
    eye_fade: vec4<f32>,
    // x: minor cell (m). y: major cell (m). zw: unused.
    steps: vec4<f32>,
    // Theme tokens, already gamma-space (see the encode note at the bottom).
    minor: vec4<f32>,
    major: vec4<f32>,
    axis_x: vec4<f32>,
    axis_z: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

// One oversized triangle covering the whole viewport: (-1,-1), (3,-1), (-1,3).
@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    let ndc = vec2<f32>(
        f32((index << 1u) & 2u) * 2.0 - 1.0,
        f32(index & 2u) * 2.0 - 1.0,
    );
    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.ndc = ndc;
    return out;
}

/// Coverage of the nearest line of a grid of pitch `cell`, in 0..1.
///
/// `abs(fract(c - 0.5) - 0.5)` is the distance to the nearest integer in cell
/// units; dividing by `fwidth(c)` turns it into a distance in *pixels*, which
/// is what makes the line one pixel wide at every zoom and every angle.
fn cell_coverage(xz: vec2<f32>, cell: f32) -> f32 {
    let c = xz / cell;
    let d = abs(fract(c - vec2<f32>(0.5)) - vec2<f32>(0.5)) / max(fwidth(c), vec2<f32>(1e-8));
    return 1.0 - min(min(d.x, d.y), 1.0);
}

/// Coverage of the single line `v == 0`.
fn axis_coverage(v: f32) -> f32 {
    return 1.0 - min(abs(v) / max(fwidth(v), 1e-8), 1.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Unproject the pixel into a world-space ray. wgpu clip space is z in
    // 0..1, so z = 0 is the near plane and z = 1 the far one.
    let near4 = u.inverse_view_projection * vec4<f32>(in.ndc, 0.0, 1.0);
    let far4 = u.inverse_view_projection * vec4<f32>(in.ndc, 1.0, 1.0);
    let near = near4.xyz / near4.w;
    let far = far4.xyz / far4.w;
    let direction = far - near;

    // t is in near..far units, so `t > 1` is "past the far plane" and `t < 0`
    // is "behind the eye" — i.e. above the horizon. Both are misses.
    //
    // Held OUTSIDE any branch, because `fwidth` below needs uniform control
    // flow: `discard` demotes the invocation without leaving the block, so the
    // derivatives are still well formed for the quad.
    let parallel = abs(direction.y) < 1e-6;
    let t = select(-near.y / direction.y, 2.0, parallel);
    let world = near + direction * clamp(t, 0.0, 1.0);

    let minor = cell_coverage(world.xz, u.steps.x) * u.minor.a;
    let major = cell_coverage(world.xz, u.steps.y) * u.major.a;
    // The X axis is the line running along x, which is where z == 0.
    let ax = axis_coverage(world.z) * u.axis_x.a;
    let az = axis_coverage(world.x) * u.axis_z.a;

    var rgb = u.minor.rgb;
    var alpha = minor;
    rgb = mix(rgb, u.major.rgb, select(0.0, 1.0, major > alpha));
    alpha = max(alpha, major);
    rgb = mix(rgb, u.axis_x.rgb, select(0.0, 1.0, ax > 0.0));
    alpha = max(alpha, ax);
    rgb = mix(rgb, u.axis_z.rgb, select(0.0, 1.0, az > 0.0));
    alpha = max(alpha, az);

    // Distance fade, so the floor dissolves instead of ending on an edge.
    let distance = length(world - u.eye_fade.xyz);
    alpha = alpha * (1.0 - smoothstep(u.eye_fade.w * 0.3, u.eye_fade.w, distance));

    if (t < 0.0 || t > 1.0 || parallel || alpha < 0.002) {
        discard;
    }

    // The token colours arrive gamma-space and iced's surface is gamma-space
    // (`web-colors` is on), so — unlike cloth.wgsl and avatar.wgsl, which
    // shade in linear and encode — there is nothing to encode here.
    return vec4<f32>(rgb, alpha);
}
