// SPDX-License-Identifier: GPL-2.0-or-later
// GPU exclusive prefix-sum (scan) over u32, three-pass multi-block. WG = 256, so
// one level of block-sum scan supports n up to 256*256 = 65536. Building block for
// the LBVH broad phase (compaction / histogram offsets).

struct Params { n: u32, num_blocks: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_total: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_offset: array<u32>;

var<workgroup> tile: array<u32, 256>;

// In-place Hillis-Steele inclusive scan of `tile` (256 elements). Barriers: one
// before each read so the previous iteration's writes are visible, one after so all
// reads finish before the write.
fn hillis_steele_inclusive(tid: u32) {
    var offset = 1u;
    for (var d = 0u; d < 8u; d = d + 1u) {
        workgroupBarrier();
        var add = 0u;
        if (tid >= offset) { add = tile[tid - offset]; }
        workgroupBarrier();
        tile[tid] = tile[tid] + add;
        offset = offset * 2u;
    }
}

// Pass 1: per-block exclusive scan → output; each block's total → block_total[wid].
@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;
    let orig = select(0u, input[gid], gid < params.n);
    tile[tid] = orig;
    hillis_steele_inclusive(tid);
    if (gid < params.n) { output[gid] = tile[tid] - orig; } // inclusive − self = exclusive
    if (tid == 255u) { block_total[wid.x] = tile[tid]; }
}

// Pass 2: exclusive scan of the block totals (single workgroup; num_blocks ≤ 256).
@compute @workgroup_size(256)
fn scan_block_totals(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let orig = select(0u, block_total[tid], tid < params.num_blocks);
    tile[tid] = orig;
    hillis_steele_inclusive(tid);
    if (tid < params.num_blocks) { block_offset[tid] = tile[tid] - orig; }
}

// Pass 3: add each block's scanned offset back to its elements.
@compute @workgroup_size(256)
fn add_block_offsets(@builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wid: vec3<u32>) {
    let gid = wid.x * 256u + lid.x;
    if (gid < params.n) { output[gid] = output[gid] + block_offset[wid.x]; }
}
