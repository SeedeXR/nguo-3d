// SPDX-License-Identifier: GPL-2.0-or-later
// GPU 30-bit Morton codes for triangle centroids — the first stage of the LBVH
// broad phase (its output is radix-sorted, then a Karras tree is built over it).
// 10 bits/axis, magic-number bit interleave (Terdiman/NVIDIA). Centroids are
// normalized into [0,1] by the scene AABB passed in `p.lo`/`p.inv`.

struct MParams { lo: vec4<f32>, inv: vec4<f32> }; // lo.w = bitcast(num_tris)
@group(0) @binding(0) var<uniform> p: MParams;
@group(0) @binding(1) var<storage, read> pos: array<f32>;   // 3n
@group(0) @binding(2) var<storage, read> tris: array<u32>;  // 3*num_tris
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;

// Insert two 0 bits after each of the low 10 bits of v.
fn expand_bits(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

fn vpos(i: u32) -> vec3<f32> {
    return vec3<f32>(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u]);
}

@compute @workgroup_size(256)
fn morton(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let num = bitcast<u32>(p.lo.w);
    if (t >= num) { return; }
    let a = vpos(tris[3u * t]);
    let b = vpos(tris[3u * t + 1u]);
    let c = vpos(tris[3u * t + 2u]);
    let ctr = (a + b + c) * (1.0 / 3.0);
    let n = clamp((ctr - p.lo.xyz) * p.inv.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let xi = u32(clamp(n.x * 1023.0, 0.0, 1023.0));
    let yi = u32(clamp(n.y * 1023.0, 0.0, 1023.0));
    let zi = u32(clamp(n.z * 1023.0, 0.0, 1023.0));
    codes[t] = expand_bits(xi) * 4u + expand_bits(yi) * 2u + expand_bits(zi);
}
