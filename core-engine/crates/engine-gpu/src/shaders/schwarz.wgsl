// SPDX-License-Identifier: GPL-2.0-or-later
// GPU 2-level Galerkin additive Schwarz preconditioner apply:
//   z = S0 r + w * C1^T S1(C1 r)
// The dense per-domain Cholesky factors are computed on the CPU (per solve) and
// uploaded; this shader only APPLIES them. Domains are disjoint (aggregation
// partitions the vertices), so every invocation owns distinct output dofs — no
// atomics (see docs/DEVELOPING.md §6: no float atomics in core WebGPU). `solve` is used for
// BOTH levels (level-1 vertices are coarse nodes; the math is identical).

const MAX_M: u32 = 96u; // domain_size ≤ 32 (m = 3·|domain|); asserted on the CPU

struct Dims {
    count: u32,   // number of domains (solve/restrict) or vertices (prolong)
    weight: f32,  // coarse-correction weight (prolong only)
};

// ---- solve: per-domain L Lᵀ x = r_d, scatter x → out. One invocation / domain.
@group(0) @binding(0) var<uniform>       s_dims: Dims;
@group(0) @binding(1) var<storage, read> s_vert: array<u32>;     // domain vertex ids
@group(0) @binding(2) var<storage, read> s_off: array<u32>;      // domain boundaries
@group(0) @binding(3) var<storage, read> s_fact: array<f32>;     // L factors (row-major mxm)
@group(0) @binding(4) var<storage, read> s_fact_off: array<u32>; // factor start per domain
@group(0) @binding(5) var<storage, read> s_m: array<u32>;        // m per domain
@group(0) @binding(6) var<storage, read> s_rhs: array<f32>;      // input r
@group(0) @binding(7) var<storage, read_write> s_out: array<f32>; // output z

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= s_dims.count) { return; }
    let base = s_off[d];
    let cnt = s_off[d + 1u] - base;
    let m = s_m[d];
    let fo = s_fact_off[d];

    var rr: array<f32, MAX_M>;
    var y: array<f32, MAX_M>;
    var x: array<f32, MAX_M>;

    // Gather the domain's residual (3 dofs per vertex).
    for (var p = 0u; p < cnt; p = p + 1u) {
        let v = s_vert[base + p];
        rr[3u * p] = s_rhs[3u * v];
        rr[3u * p + 1u] = s_rhs[3u * v + 1u];
        rr[3u * p + 2u] = s_rhs[3u * v + 2u];
    }
    // Forward solve L y = rr:  y[i] = (rr[i] - Σ_{k<i} L[i,k] y[k]) / L[i,i]
    for (var i = 0u; i < m; i = i + 1u) {
        var s = rr[i];
        for (var k = 0u; k < i; k = k + 1u) {
            s = s - s_fact[fo + i * m + k] * y[k];
        }
        y[i] = s / s_fact[fo + i * m + i];
    }
    // Back solve Lᵀ x = y:  x[i] = (y[i] - Σ_{k>i} L[k,i] x[k]) / L[i,i]
    for (var ii = 0u; ii < m; ii = ii + 1u) {
        let i = m - 1u - ii;
        var s = y[i];
        for (var k = i + 1u; k < m; k = k + 1u) {
            s = s - s_fact[fo + k * m + i] * x[k];
        }
        x[i] = s / s_fact[fo + i * m + i];
    }
    // Scatter to output.
    for (var p = 0u; p < cnt; p = p + 1u) {
        let v = s_vert[base + p];
        s_out[3u * v] = x[3u * p];
        s_out[3u * v + 1u] = x[3u * p + 1u];
        s_out[3u * v + 2u] = x[3u * p + 2u];
    }
}

// ---- restrict: rc[g] = Σ_{i∈domain g} r[i]  (sum, matching the CPU reference).
@group(0) @binding(0) var<uniform>       r_dims: Dims;
@group(0) @binding(1) var<storage, read> r_vert: array<u32>;
@group(0) @binding(2) var<storage, read> r_off: array<u32>;
@group(0) @binding(6) var<storage, read> r_rhs: array<f32>;       // fine r
@group(0) @binding(7) var<storage, read_write> r_out: array<f32>; // coarse rc

@compute @workgroup_size(64)
fn restrict_r(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= r_dims.count) { return; }
    let base = r_off[g];
    let cnt = r_off[g + 1u] - base;
    var s0 = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var p = 0u; p < cnt; p = p + 1u) {
        let v = r_vert[base + p];
        s0 = s0 + r_rhs[3u * v];
        s1 = s1 + r_rhs[3u * v + 1u];
        s2 = s2 + r_rhs[3u * v + 2u];
    }
    r_out[3u * g] = s0;
    r_out[3u * g + 1u] = s1;
    r_out[3u * g + 2u] = s2;
}

// ---- prolong: z[i] += w · ec[agg0[i]]  (per fine vertex; disjoint writes).
@group(0) @binding(0) var<uniform>       p_dims: Dims;
@group(0) @binding(8) var<storage, read> p_agg0: array<u32>;      // fine vertex → coarse node
@group(0) @binding(9) var<storage, read> p_ec: array<f32>;        // coarse correction
@group(0) @binding(7) var<storage, read_write> p_out: array<f32>; // z (accumulate)

@compute @workgroup_size(64)
fn prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p_dims.count) { return; }
    let g = p_agg0[i];
    let w = p_dims.weight;
    p_out[3u * i] = p_out[3u * i] + w * p_ec[3u * g];
    p_out[3u * i + 1u] = p_out[3u * i + 1u] + w * p_ec[3u * g + 1u];
    p_out[3u * i + 2u] = p_out[3u * i + 2u] + w * p_ec[3u * g + 2u];
}
