// SPDX-License-Identifier: GPL-2.0-or-later
// Atomic-free per-vertex GATHER of ANY 4-vertex element's blocks into the global CSR
// `values` + `grad` — used by BOTH bending hinges and volumetric tets, which share
// the identical stencil layout (12 offsets per incidence, 144-entry Hessian blocks).
// Worker `i` owns vertex i's rows and accumulates the contributions of its incident
// elements (egrad/ehess, produced by whichever element kernel ran first) into its own
// entries. Only worker i writes row i → no atomics. Mirrors
// engine_core::gather4::Gather4::assemble. `values`/`grad` must be pre-zeroed.
//
// BOUND BY: engine-gpu/src/{bending_asm,tet_assembler}.rs, fed from
// engine_core::Gather4 — NOT the SpringAssembler that binds assemble.wgsl.
// SENTINELS: `vt_off` CAN be -1 (block dropped, pinned partner vertex) — that is
// the live sentinel and every use is guarded. `diag_off` can NOT be -1 (Gather4
// pushes a diagonal block for every vertex unconditionally); the `off >= 0` test
// on it below is belt-and-braces, and assemble.wgsl omitting the same test is
// correct, not a bug.

struct GP { n: u32, nnz: u32, ndof: u32, _p: u32 };

@group(0) @binding(0) var<uniform> gp: GP;
@group(0) @binding(1) var<storage, read> vt_ptr: array<u32>;    // n+1
@group(0) @binding(2) var<storage, read> vt_elem: array<u32>;   // m
@group(0) @binding(3) var<storage, read> vt_local: array<u32>;  // m
@group(0) @binding(4) var<storage, read> vt_off: array<i32>;    // 12m
@group(0) @binding(5) var<storage, read> diag_off: array<i32>;  // 3n
@group(0) @binding(6) var<storage, read> pinned: array<u32>;    // n
@group(0) @binding(7) var<storage, read> egrad: array<f32>;     // 12 per hinge
@group(0) @binding(8) var<storage, read> ehess: array<f32>;     // 144 per hinge
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
            grad[3u * i + r] += egrad[12u * t + 3u * a + r];
        }
        for (var b = 0u; b < 4u; b = b + 1u) {
            for (var r = 0u; r < 3u; r = r + 1u) {
                let off = vt_off[12u * k + 3u * b + r];
                if (off >= 0) {
                    let base = u32(off);
                    for (var c = 0u; c < 3u; c = c + 1u) {
                        values[base + c] += ehess[144u * t + (3u * a + r) * 12u + (3u * b + c)];
                    }
                }
            }
        }
    }
}
