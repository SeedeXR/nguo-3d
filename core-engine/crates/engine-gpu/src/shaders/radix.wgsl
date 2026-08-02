// SPDX-License-Identifier: GPL-2.0-or-later
// GPU LSD radix sort over u32 keys, 8-bit digit (4 passes for 32-bit keys).
// Multi-block, blocks of 256; supports n up to 65536 (num_blocks <= 256, matching
// the scan primitive that computes the per-(bucket,block) global offsets).
// Building block for the LBVH broad phase (sorting Morton codes).
//
// Per digit pass: histogram -> exclusive scan of the bucket-major histogram
// (done by scan.wgsl) -> stable scatter. The histogram is laid out bucket-major
// (hist[bucket*num_blocks + block]) so a single flat exclusive scan yields, for
// each (bucket, block), the count of all keys that sort strictly before it:
// earlier buckets in full, plus earlier blocks of the same bucket. That value is
// exactly the destination of that (block, bucket) group's first key.

struct RParams { n: u32, num_blocks: u32, shift: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: RParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> aux: array<u32>; // histogram (write) or offsets (read)
@group(0) @binding(4) var<storage, read> vals_in: array<u32>;   // payload, permuted with the keys
@group(0) @binding(5) var<storage, read_write> vals_out: array<u32>;

const RADIX: u32 = 256u;

var<workgroup> lhist: array<atomic<u32>, 256>;

// Pass A: per-block histogram of the current 8-bit digit, written bucket-major.
@compute @workgroup_size(256)
fn histogram(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    atomicStore(&lhist[tid], 0u);
    workgroupBarrier();
    let gid = wid.x * 256u + tid;
    if (gid < p.n) {
        let d = (keys_in[gid] >> p.shift) & 0xffu;
        atomicAdd(&lhist[d], 1u);
    }
    workgroupBarrier();
    aux[tid * p.num_blocks + wid.x] = atomicLoad(&lhist[tid]);
}

// Pass C: stable scatter. One thread per block walks its 256 elements in order,
// placing each at the running global offset for its digit. Serial within a block
// (preserves stability), parallel across blocks. `aux` here holds the scanned
// offsets. Correctness-first; a warp-cooperative scatter is the perf follow-up.
var<workgroup> ctr: array<u32, 256>;

@compute @workgroup_size(1)
fn scatter(@builtin(workgroup_id) wid: vec3<u32>) {
    let block = wid.x;
    for (var b = 0u; b < RADIX; b = b + 1u) {
        ctr[b] = aux[b * p.num_blocks + block];
    }
    let base = block * 256u;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let gid = base + i;
        if (gid < p.n) {
            let k = keys_in[gid];
            let d = (k >> p.shift) & 0xffu;
            keys_out[ctr[d]] = k;
            vals_out[ctr[d]] = vals_in[gid]; // move the payload to the same slot
            ctr[d] = ctr[d] + 1u;
        }
    }
}
