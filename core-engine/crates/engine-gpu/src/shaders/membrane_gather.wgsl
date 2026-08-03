// SPDX-License-Identifier: GPL-2.0-or-later
// Atomic-free per-vertex GATHER of the membrane element blocks into the global
// CSR `values` + `grad`. Worker `i` owns vertex i's rows and accumulates the
// contributions of its incident triangles (from the per-triangle compute pass,
// `membrane.wgsl` → egrad/ehess) into its own entries — a membrane EDGE is shared
// by up to two triangles, so off-diagonal blocks accumulate, but only worker i
// writes row i, so no atomics. Mirrors engine_core::MembraneAssembler::assemble.
// The `values`/`grad` buffers must be pre-zeroed (the host clears them each call).
//
// BOUND BY: engine-gpu/src/{membrane_asm,shell_asm}.rs, fed from
// engine_core::{MembraneAssembler,ShellAssembler} (both wrap a Gather3) — NOT the
// SpringAssembler that binds assemble.wgsl.
// SENTINELS: `vt_off` CAN be -1 (block dropped, pinned partner vertex) — the live
// sentinel, guarded at every use. `diag_off` can NOT be -1 (Gather3 pushes a
// diagonal block for every vertex unconditionally); the `off >= 0` test on it
// below is belt-and-braces, and assemble.wgsl omitting it is correct, not a bug.

struct GP { n: u32, nnz: u32, ndof: u32, _p: u32 };

@group(0) @binding(0) var<uniform> gp: GP;
@group(0) @binding(1) var<storage, read> vt_ptr: array<u32>;    // n+1
@group(0) @binding(2) var<storage, read> vt_elem: array<u32>;   // m
@group(0) @binding(3) var<storage, read> vt_local: array<u32>;  // m
@group(0) @binding(4) var<storage, read> vt_off: array<i32>;    // 9m
@group(0) @binding(5) var<storage, read> diag_off: array<i32>;  // 3n
@group(0) @binding(6) var<storage, read> pinned: array<u32>;    // n
@group(0) @binding(7) var<storage, read> egrad: array<f32>;     // 9 per triangle
@group(0) @binding(8) var<storage, read> ehess: array<f32>;     // 81 per triangle
@group(0) @binding(9) var<storage, read_write> values: array<f32>;
@group(0) @binding(10) var<storage, read_write> grad: array<f32>;

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gp.n) { return; }
    if (pinned[i] == 1u) {
        for (var r = 0u; r < 3u; r = r + 1u) {
            let off = diag_off[3u * i + r];
            if (off >= 0) { values[u32(off)] = 1.0; }
        }
        return;
    }
    let s = vt_ptr[i];
    let e = vt_ptr[i + 1u];
    for (var k = s; k < e; k = k + 1u) {
        let t = vt_elem[k];
        let a = vt_local[k];
        for (var r = 0u; r < 3u; r = r + 1u) {
            grad[3u * i + r] += egrad[9u * t + 3u * a + r];
        }
        for (var b = 0u; b < 3u; b = b + 1u) {
            for (var r = 0u; r < 3u; r = r + 1u) {
                let off = vt_off[9u * k + 3u * b + r];
                if (off >= 0) {
                    let base = u32(off);
                    for (var c = 0u; c < 3u; c = c + 1u) {
                        values[base + c] += ehess[81u * t + (3u * a + r) * 9u + (3u * b + c)];
                    }
                }
            }
        }
    }
}
