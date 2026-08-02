// SPDX-License-Identifier: GPL-2.0-or-later
// LBVH traversal — the final broad-phase stage. One thread per vertex descends the
// Karras tree, testing the vertex's dhat-expanded AABB against node/leaf boxes, and
// emits (vertex, triangle) candidate pairs into a global append buffer. Emits the
// geometric superset (no 1-ring exclusion — that's a downstream narrow-phase
// filter), which is exactly the "must contain every truly-close pair" contract the
// CPU broad phase (`broadphase.rs`) is checked against.
//
// ponytail: fixed 64-deep per-thread stack — ample for Morton-balanced trees. A
// degenerate all-equal-code mesh could exceed it; the host checks the overflow flag
// and the test fails loudly rather than silently dropping a subtree.

struct TParams { n_tris: u32, n_verts: u32, dhat: f32, max_pairs: u32 };
@group(0) @binding(0) var<uniform> p: TParams;
@group(0) @binding(1) var<storage, read> left: array<u32>;      // n-1
@group(0) @binding(2) var<storage, read> right: array<u32>;     // n-1
@group(0) @binding(3) var<storage, read> node_min: array<f32>;  // 3(n-1)
@group(0) @binding(4) var<storage, read> node_max: array<f32>;  // 3(n-1)
@group(0) @binding(5) var<storage, read> leaf_min: array<f32>;  // 3n (sorted order)
@group(0) @binding(6) var<storage, read> leaf_max: array<f32>;  // 3n
@group(0) @binding(7) var<storage, read> prim: array<u32>;      // n: leaf -> triangle index
@group(0) @binding(8) var<storage, read> pos: array<f32>;       // 3 * n_verts
@group(0) @binding(9) var<storage, read_write> out_pairs: array<u32>; // 2 * max_pairs
@group(0) @binding(10) var<storage, read_write> counter: array<atomic<u32>>; // [0]=count [1]=overflow

const LEAF: u32 = 0x80000000u;

fn overlaps(qlo: vec3<f32>, qhi: vec3<f32>, blo: vec3<f32>, bhi: vec3<f32>) -> bool {
    return all(qlo <= bhi) && all(blo <= qhi);
}
fn node_box(i: u32) -> array<vec3<f32>, 2> {
    return array<vec3<f32>, 2>(
        vec3<f32>(node_min[3u * i], node_min[3u * i + 1u], node_min[3u * i + 2u]),
        vec3<f32>(node_max[3u * i], node_max[3u * i + 1u], node_max[3u * i + 2u]),
    );
}
fn leaf_box(i: u32) -> array<vec3<f32>, 2> {
    return array<vec3<f32>, 2>(
        vec3<f32>(leaf_min[3u * i], leaf_min[3u * i + 1u], leaf_min[3u * i + 2u]),
        vec3<f32>(leaf_max[3u * i], leaf_max[3u * i + 1u], leaf_max[3u * i + 2u]),
    );
}
fn emit(v: u32, tri: u32) {
    let idx = atomicAdd(&counter[0], 1u);
    if (idx < p.max_pairs) {
        out_pairs[2u * idx] = v;
        out_pairs[2u * idx + 1u] = tri;
    } else {
        atomicOr(&counter[1], 1u); // pair-buffer overflow
    }
}

@compute @workgroup_size(256)
fn traverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= p.n_verts || p.n_tris == 0u) { return; }
    let pt = vec3<f32>(pos[3u * v], pos[3u * v + 1u], pos[3u * v + 2u]);
    let qlo = pt - vec3<f32>(p.dhat);
    let qhi = pt + vec3<f32>(p.dhat);

    // Degenerate tree: a single leaf (no internal nodes).
    if (p.n_tris == 1u) {
        let b = leaf_box(0u);
        if (overlaps(qlo, qhi, b[0], b[1])) { emit(v, prim[0]); }
        return;
    }

    var stack: array<u32, 64>;
    var sp = 0u;
    stack[sp] = 0u; // root = internal node 0 (no LEAF bit)
    sp = sp + 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = stack[sp];
        if ((node & LEAF) != 0u) {
            let leaf = node & 0x7fffffffu;
            let b = leaf_box(leaf);
            if (overlaps(qlo, qhi, b[0], b[1])) { emit(v, prim[leaf]); }
        } else {
            let b = node_box(node);
            if (overlaps(qlo, qhi, b[0], b[1])) {
                if (sp + 2u > 64u) {
                    atomicOr(&counter[1], 2u); // stack overflow
                } else {
                    stack[sp] = left[node]; sp = sp + 1u;
                    stack[sp] = right[node]; sp = sp + 1u;
                }
            }
        }
    }
}
