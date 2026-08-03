// SPDX-License-Identifier: GPL-2.0-or-later
// Atomic-free per-vertex GATHER assembly of the implicit mass-spring Hessian +
// gradient. One thread owns vertex i's CSR rows and gathers its incident
// springs' contributions into its own entries — no thread writes another's
// rows, so NO atomics (core WebGPU has no float atomics; see docs/DEVELOPING.md §6). This
// mirrors engine_core::assembler::SpringAssembler::assemble one-to-one; the CPU
// version is the verification oracle. Offsets are absolute indices into `values`.
//
// BOUND BY: engine-gpu/src/assemble.rs + gpu_implicit.rs, fed from
// engine_core::SpringAssembler (NOT Gather3/Gather4 — the near-identical
// gather4.wgsl / membrane_gather.wgsl / unified_gather.wgsl serve a different
// assembler; read the header of each before assuming they share a contract).
// SENTINELS: `adj_off` CAN be -1 (off-diagonal block dropped because the
// neighbour is pinned) and IS guarded below. `diag_off` can NOT: SpringAssembler
// pushes a diagonal block for every vertex unconditionally, so the two casts
// `u32(diag_off[...])` are safe by construction, asserted (not guarded) on the
// CPU side in engine-core/src/assembler.rs. Do not "fix" them with a guard —
// that would mask a structure regression instead of tripping the assert.

struct AP { n: u32, inv_h2: f32, k: f32, _pad: f32 };

@group(0) @binding(0) var<uniform> ap: AP;
@group(0) @binding(1) var<storage, read> x: array<f32>;         // 3n
@group(0) @binding(2) var<storage, read> x_tilde: array<f32>;   // 3n
@group(0) @binding(3) var<storage, read> mass: array<f32>;      // n
@group(0) @binding(4) var<storage, read> pinned: array<u32>;    // n
@group(0) @binding(5) var<storage, read> diag_off: array<i32>;  // 3n
@group(0) @binding(6) var<storage, read> adj_ptr: array<u32>;   // n+1
@group(0) @binding(7) var<storage, read> adj_j: array<u32>;     // m
@group(0) @binding(8) var<storage, read> adj_rest: array<f32>;  // m
@group(0) @binding(9) var<storage, read> adj_off: array<i32>;   // 3m
@group(0) @binding(10) var<storage, read_write> values: array<f32>;
@group(0) @binding(11) var<storage, read_write> grad: array<f32>;

fn load3(base: u32) -> vec3<f32> {
    return vec3<f32>(x[3u * base], x[3u * base + 1u], x[3u * base + 2u]);
}

struct SB { h: array<f32, 9>, g: vec3<f32> };

fn spring_block(d: vec3<f32>, rest: f32, k: f32) -> SB {
    var sb: SB; // arrays/vecs zero-initialized
    let l = length(d);
    if (l < 1e-9) {
        sb.h[0] = k; sb.h[4] = k; sb.h[8] = k;
        return sb;
    }
    let na = array<f32, 3>(d.x / l, d.y / l, d.z / l);
    let coeff_t = max(1.0 - rest / l, 0.0);
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            let nnt = na[r] * na[c];
            let iden = select(0.0, 1.0, r == c);
            sb.h[r * 3u + c] = k * (nnt + coeff_t * (iden - nnt));
        }
    }
    sb.g = vec3<f32>(na[0], na[1], na[2]) * (k * (l - rest));
    return sb;
}

@compute @workgroup_size(64)
fn assemble(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ap.n) { return; }

    if (pinned[i] != 0u) {
        for (var r = 0u; r < 3u; r = r + 1u) {
            // Unguarded by design: diag_off is never -1 here (see header).
            values[u32(diag_off[3u * i + r])] = 1.0;
        }
        return;
    }

    let mh2 = mass[i] * ap.inv_h2;
    var diag: array<f32, 9>;
    diag[0] = mh2; diag[4] = mh2; diag[8] = mh2;

    let xi = load3(i);
    let xti = vec3<f32>(x_tilde[3u * i], x_tilde[3u * i + 1u], x_tilde[3u * i + 2u]);
    var gi = (xi - xti) * mh2;

    let s = adj_ptr[i];
    let e = adj_ptr[i + 1u];
    for (var idx = s; idx < e; idx = idx + 1u) {
        let j = adj_j[idx];
        let d = xi - load3(j);
        let sb = spring_block(d, adj_rest[idx], ap.k);
        for (var t = 0u; t < 9u; t = t + 1u) {
            diag[t] = diag[t] + sb.h[t];
        }
        gi = gi + sb.g;
        if (adj_off[3u * idx] >= 0) {
            for (var r = 0u; r < 3u; r = r + 1u) {
                let base = u32(adj_off[3u * idx + r]);
                values[base] = -sb.h[r * 3u];
                values[base + 1u] = -sb.h[r * 3u + 1u];
                values[base + 2u] = -sb.h[r * 3u + 2u];
            }
        }
    }

    for (var r = 0u; r < 3u; r = r + 1u) {
        // Unguarded by design: diag_off is never -1 here (see header). For a free
        // vertex it points at the start of the 3-run of cols 3i..3i+2 in row 3i+r.
        let base = u32(diag_off[3u * i + r]);
        values[base] = diag[r * 3u];
        values[base + 1u] = diag[r * 3u + 1u];
        values[base + 2u] = diag[r * 3u + 2u];
    }
    grad[3u * i] = gi.x;
    grad[3u * i + 1u] = gi.y;
    grad[3u * i + 2u] = gi.z;
}
