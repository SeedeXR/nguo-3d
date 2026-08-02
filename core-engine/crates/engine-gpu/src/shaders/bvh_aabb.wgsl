// SPDX-License-Identifier: GPL-2.0-or-later
// LBVH bottom-up node-AABB propagation over a Karras tree. Each internal node's
// AABB is the union of its two children's. Done as level passes: one dispatch per
// pass, gated on a readiness SNAPSHOT (`ready_in`) from before the pass so a node
// only ever reads children finalized in an EARLIER pass (whose writes are visible
// across workgroups because each pass is a separate submit). `ready_out` is this
// pass's readiness; the host ping-pongs the two and repeats until nothing changes.
//
// ponytail: level passes are the obviously-correct variant; the single-dispatch
// atomic-climb (Karras/Apetrei) is the perf variant, but it depends on a
// cross-workgroup non-atomic-write visibility guarantee WGSL doesn't cleanly make
// — deferred to the GPU perf pass, where it can be validated against this.

struct AParams { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> p: AParams;
@group(0) @binding(1) var<storage, read> left: array<u32>;   // n-1
@group(0) @binding(2) var<storage, read> right: array<u32>;  // n-1
@group(0) @binding(3) var<storage, read> leaf_min: array<f32>;  // 3n
@group(0) @binding(4) var<storage, read> leaf_max: array<f32>;  // 3n
@group(0) @binding(5) var<storage, read_write> node_min: array<f32>; // 3(n-1)
@group(0) @binding(6) var<storage, read_write> node_max: array<f32>; // 3(n-1)
@group(0) @binding(7) var<storage, read> ready_in: array<u32>;       // n-1
@group(0) @binding(8) var<storage, read_write> ready_out: array<u32>; // n-1
@group(0) @binding(9) var<storage, read_write> changed: array<atomic<u32>>; // 1

const LEAF: u32 = 0x80000000u;

fn child_ready(c: u32) -> bool {
    if ((c & LEAF) != 0u) { return true; }
    return ready_in[c & 0x7fffffffu] == 1u;
}
fn child_min(c: u32, k: u32) -> f32 {
    let idx = c & 0x7fffffffu;
    if ((c & LEAF) != 0u) { return leaf_min[3u * idx + k]; }
    return node_min[3u * idx + k];
}
fn child_max(c: u32, k: u32) -> f32 {
    let idx = c & 0x7fffffffu;
    if ((c & LEAF) != 0u) { return leaf_max[3u * idx + k]; }
    return node_max[3u * idx + k];
}

@compute @workgroup_size(256)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n - 1u) { return; }
    if (ready_in[i] == 1u) { ready_out[i] = 1u; return; } // already done, carry forward
    let l = left[i];
    let r = right[i];
    if (!child_ready(l) || !child_ready(r)) { ready_out[i] = 0u; return; }
    for (var k = 0u; k < 3u; k = k + 1u) {
        node_min[3u * i + k] = min(child_min(l, k), child_min(r, k));
        node_max[3u * i + k] = max(child_max(l, k), child_max(r, k));
    }
    ready_out[i] = 1u;
    atomicAdd(&changed[0], 1u);
}
