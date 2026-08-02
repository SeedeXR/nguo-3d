// SPDX-License-Identifier: GPL-2.0-or-later
// Fallback f32 sum reduction (no SUBGROUP feature): Kahan-compensated grid-stride
// per thread, then a shared-memory tree reduction. Same bindings + entry point as
// subgroup_reduce.wgsl, so the host code is identical regardless of which is used.
struct P { n: u32, total_threads: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;

var<workgroup> scr: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    var s = 0.0;
    var c = 0.0;
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        let y = inp[i] - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
        i = i + p.total_threads;
    }
    scr[li] = s;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (li < stride) { scr[li] = scr[li] + scr[li + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (li == 0u) { outp[wid.x] = scr[0]; }
}
