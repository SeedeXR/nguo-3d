// SPDX-License-Identifier: GPL-2.0-or-later
// Atomic-free per-vertex GATHER of inertia + springs + membrane into ONE union
// CSR (engine_core::UnifiedAssembler). Worker i owns row i and accumulates: the
// inertia diagonal + incident spring blocks (spring_block) + incident membrane
// element blocks (from the per-triangle compute pass, membrane.wgsl → egrad/ehess)
// into row i's entries. Off-diagonals use += — a SHARED edge (spring == triangle
// edge) is ONE entry that both add into; the diagonal is accumulated in a register
// and written once. `values`/`grad` must be pre-zeroed (the host clears them). No
// two workers write the same entry → no atomics (see docs/DEVELOPING.md §6). Mirrors
// UnifiedAssembler::assemble exactly (the CPU oracle).

struct AP { n: u32, inv_h2: f32, k: f32, _pad: f32 };

@group(0) @binding(0) var<uniform> ap: AP;
@group(0) @binding(1) var<storage, read> x: array<f32>;         // 3n
@group(0) @binding(2) var<storage, read> x_tilde: array<f32>;   // 3n
@group(0) @binding(3) var<storage, read> mass: array<f32>;      // n
@group(0) @binding(4) var<storage, read> pinned: array<u32>;    // n
@group(0) @binding(5) var<storage, read> diag_off: array<i32>;  // 3n
@group(0) @binding(6) var<storage, read> s_ptr: array<u32>;     // n+1
@group(0) @binding(7) var<storage, read> s_j: array<u32>;       // ms
@group(0) @binding(8) var<storage, read> s_rest: array<f32>;    // ms
@group(0) @binding(9) var<storage, read> s_off: array<i32>;     // 3 ms
@group(0) @binding(10) var<storage, read> m_ptr: array<u32>;    // n+1
@group(0) @binding(11) var<storage, read> m_elem: array<u32>;   // mt
@group(0) @binding(12) var<storage, read> m_local: array<u32>;  // mt
@group(0) @binding(13) var<storage, read> m_off: array<i32>;    // 9 mt
@group(0) @binding(14) var<storage, read> egrad: array<f32>;    // 9 per triangle
@group(0) @binding(15) var<storage, read> ehess: array<f32>;    // 81 per triangle
@group(0) @binding(16) var<storage, read_write> values: array<f32>;
@group(0) @binding(17) var<storage, read_write> grad: array<f32>;

fn load3(base: u32) -> vec3<f32> {
    return vec3<f32>(x[3u * base], x[3u * base + 1u], x[3u * base + 2u]);
}

struct SB { h: array<f32, 9>, g: vec3<f32> };

fn spring_block(d: vec3<f32>, rest: f32, k: f32) -> SB {
    var sb: SB;
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
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ap.n) { return; }
    if (pinned[i] != 0u) {
        for (var r = 0u; r < 3u; r = r + 1u) {
            let off = diag_off[3u * i + r];
            if (off >= 0) { values[u32(off)] = 1.0; }
        }
        return;
    }

    // Inertia.
    let mh2 = mass[i] * ap.inv_h2;
    var diag: array<f32, 9>;
    diag[0] = mh2; diag[4] = mh2; diag[8] = mh2;
    let xi = load3(i);
    let xti = vec3<f32>(x_tilde[3u * i], x_tilde[3u * i + 1u], x_tilde[3u * i + 2u]);
    var gi = (xi - xti) * mh2;

    // Springs (off-diagonal += so a shared edge accumulates with membrane).
    let ss = s_ptr[i];
    let se = s_ptr[i + 1u];
    for (var e = ss; e < se; e = e + 1u) {
        let j = s_j[e];
        let sb = spring_block(xi - load3(j), s_rest[e], ap.k);
        for (var t = 0u; t < 9u; t = t + 1u) { diag[t] = diag[t] + sb.h[t]; }
        gi = gi + sb.g;
        if (s_off[3u * e] >= 0) {
            for (var r = 0u; r < 3u; r = r + 1u) {
                let base = u32(s_off[3u * e + r]);
                values[base] = values[base] - sb.h[r * 3u];
                values[base + 1u] = values[base + 1u] - sb.h[r * 3u + 1u];
                values[base + 2u] = values[base + 2u] - sb.h[r * 3u + 2u];
            }
        }
    }

    // Membrane: incident triangles (element blocks from the compute pass).
    let ms = m_ptr[i];
    let me = m_ptr[i + 1u];
    for (var kk = ms; kk < me; kk = kk + 1u) {
        let t = m_elem[kk];
        let a = m_local[kk];
        for (var r = 0u; r < 3u; r = r + 1u) {
            gi[r] = gi[r] + egrad[9u * t + 3u * a + r];
        }
        for (var b = 0u; b < 3u; b = b + 1u) {
            for (var r = 0u; r < 3u; r = r + 1u) {
                for (var c = 0u; c < 3u; c = c + 1u) {
                    let v = ehess[81u * t + (3u * a + r) * 9u + (3u * b + c)];
                    if (b == a) {
                        diag[r * 3u + c] = diag[r * 3u + c] + v; // own block → diagonal register
                    } else {
                        let off = m_off[9u * kk + 3u * b + r];
                        if (off >= 0) {
                            values[u32(off) + c] = values[u32(off) + c] + v;
                        }
                    }
                }
            }
        }
    }

    // Write the diagonal block (only this worker touches col 3i) + gradient.
    for (var r = 0u; r < 3u; r = r + 1u) {
        let base = u32(diag_off[3u * i + r]);
        values[base] = diag[r * 3u];
        values[base + 1u] = diag[r * 3u + 1u];
        values[base + 2u] = diag[r * 3u + 2u];
    }
    grad[3u * i] = gi.x;
    grad[3u * i + 1u] = gi.y;
    grad[3u * i + 2u] = gi.z;
}
