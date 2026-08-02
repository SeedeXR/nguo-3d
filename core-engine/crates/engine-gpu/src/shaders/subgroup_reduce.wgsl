// SPDX-License-Identifier: GPL-2.0-or-later
// Subgroup-accelerated f32 sum reduction. Each thread grid-strides its slice with
// a Kahan-compensated accumulator (item 580), the subgroup sums its lanes with
// subgroupAdd (one hardware op, no shared-memory tree), and one thread combines the
// per-subgroup partials. Two dispatches of this same kernel reduce an array to a
// scalar (data → per-workgroup partials → final). Requires the SUBGROUP feature.
enable subgroups;

struct P { n: u32, total_threads: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;

var<workgroup> sg: array<f32, 16>; // per-subgroup partials (≤ 256/min_subgroup)

@compute @workgroup_size(256)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(subgroup_size) ss: u32,
    @builtin(subgroup_invocation_id) sl: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // Kahan-compensated grid-stride accumulation.
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
    // Subgroup reduction → one partial per subgroup.
    let red = subgroupAdd(s);
    let sgid = li / ss;
    if (sl == 0u) { sg[sgid] = red; }
    workgroupBarrier();
    // One thread combines the per-subgroup partials for this workgroup.
    if (li == 0u) {
        let nsg = 256u / ss;
        var acc = 0.0;
        for (var k = 0u; k < nsg; k = k + 1u) { acc = acc + sg[k]; }
        outp[wid.x] = acc;
    }
}
