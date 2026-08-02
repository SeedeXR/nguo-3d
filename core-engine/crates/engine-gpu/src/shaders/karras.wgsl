// SPDX-License-Identifier: GPL-2.0-or-later
// GPU Karras binary-radix tree over sorted Morton codes (Karras 2012, "Maximizing
// Parallelism in the Construction of BVHs"). One thread per internal node computes
// that node's covered range and split, then its two children. n leaves + (n-1)
// internal nodes. Children encode leaf-vs-internal in the top bit (LEAF). All
// integer math — GPU output is bit-identical to the CPU reference.

struct KParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: KParams;
@group(0) @binding(1) var<storage, read> codes: array<u32>;      // n, sorted ascending
@group(0) @binding(2) var<storage, read_write> left: array<u32>;  // n-1
@group(0) @binding(3) var<storage, read_write> right: array<u32>; // n-1

const LEAF: u32 = 0x80000000u;

// Longest-common-prefix length of codes[i], codes[j]; -1 if j out of range.
// Equal codes are disambiguated by index (append the index bits), so the keys are
// effectively unique and monotonic — required for a well-formed tree.
fn delta(i: i32, j: i32) -> i32 {
    let n = i32(p.n);
    if (j < 0 || j >= n) { return -1; }
    let x = codes[u32(i)] ^ codes[u32(j)];
    if (x == 0u) {
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(x));
}

@compute @workgroup_size(256)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= i32(p.n) - 1) { return; }

    // Direction of the range (+1 grows right, -1 grows left).
    var d = 1;
    if (delta(i, i + 1) - delta(i, i - 1) < 0) { d = -1; }
    let delta_min = delta(i, i - d);

    // Upper bound on range length, then binary-search the exact far end.
    var l_max = 2;
    while (delta(i, i + l_max * d) > delta_min) { l_max = l_max * 2; }
    var l = 0;
    var t = l_max / 2;
    while (t >= 1) {
        if (delta(i, i + (l + t) * d) > delta_min) { l = l + t; }
        t = t / 2;
    }
    let j = i + l * d;
    let first = min(i, j);
    let last = max(i, j);

    // Split position: highest index in (first,last) whose prefix with `first`
    // exceeds the node's shared prefix.
    let node_delta = delta(first, last);
    var split = first;
    var step = last - first;
    loop {
        step = (step + 1) / 2;
        let new_split = split + step;
        if (new_split < last && delta(first, new_split) > node_delta) {
            split = new_split;
        }
        if (step <= 1) { break; }
    }
    let gamma = split;

    var lc = u32(gamma);
    if (first == gamma) { lc = u32(gamma) | LEAF; }
    var rc = u32(gamma + 1);
    if (last == gamma + 1) { rc = u32(gamma + 1) | LEAF; }
    left[u32(i)] = lc;
    right[u32(i)] = rc;
}
