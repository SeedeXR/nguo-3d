// SPDX-License-Identifier: GPL-2.0-or-later
// Discrete-shell dihedral bending embed kernel (Grinspun 2003). One thread per
// hinge computes the dihedral angle θ (exact, atan2), the analytic gradient ∂θ/∂x
// (Tamstorf-Grinspun tip/edge split), the bending energy E = ½·kb·w·(θ−rest)², and
// its per-vertex gradient ∇E = kb·w·(θ−rest)·∇θ. Mirrors engine-core/bending.rs
// (the CPU reference/oracle). The Gauss-Newton Hessian kb·w·∇θ∇θᵀ is PSD by
// construction — assembled from ∇θ downstream. Hinge = [opp_a, s0, s1, opp_b].

struct BParams { n_hinges: u32, kb: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> pr: BParams;
@group(0) @binding(1) var<storage, read> idx: array<u32>;   // 4 * n_hinges
@group(0) @binding(2) var<storage, read> rest: array<f32>;  // n_hinges
@group(0) @binding(3) var<storage, read> wt: array<f32>;    // n_hinges (hinge weight)
@group(0) @binding(4) var<storage, read> pos: array<f32>;   // 3 * n_verts
@group(0) @binding(5) var<storage, read_write> energy: array<f32>; // n_hinges
@group(0) @binding(6) var<storage, read_write> grad: array<f32>;   // 12 * n_hinges
@group(0) @binding(7) var<storage, read_write> hess: array<f32>;   // 144 * n_hinges (12×12 GN block)

fn vpos(i: u32) -> vec3<f32> {
    return vec3<f32>(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u]);
}

fn dihedral_angle(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> f32 {
    let n1 = normalize(cross(p1 - p0, p2 - p0));
    let n2 = normalize(cross(p1 - p2, p3 - p2));
    let e = normalize(p2 - p1);
    let sn = dot(cross(n1, n2), e);
    let cs = clamp(dot(n1, n2), -1.0, 1.0);
    return atan2(sn, cs);
}

@compute @workgroup_size(64)
fn bending(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= pr.n_hinges) { return; }
    let i0 = idx[4u * h];
    let i1 = idx[4u * h + 1u];
    let i2 = idx[4u * h + 2u];
    let i3 = idx[4u * h + 3u];
    let p0 = vpos(i0);
    let p1 = vpos(i1);
    let p2 = vpos(i2);
    let p3 = vpos(i3);

    let theta = dihedral_angle(p0, p1, p2, p3);

    // Analytic ∂θ/∂x.
    let m1 = cross(p1 - p0, p2 - p0);
    let m2 = cross(p1 - p2, p3 - p2);
    let e = p2 - p1;
    let a1 = length(m1);
    let a2 = length(m2);
    let l2 = dot(e, e);
    let l = sqrt(l2);
    let n1 = m1 / a1;
    let n2 = m2 / a2;
    let g0 = n1 * (-l / a1);
    let g3 = n2 * (-l / a2);
    let ta = dot(p0 - p1, e) / l2;
    let tb = dot(p3 - p1, e) / l2;
    let g1 = g0 * (ta - 1.0) + g3 * (tb - 1.0);
    let g2 = g0 * (-ta) + g3 * (-tb);

    let w = wt[h];
    let d = theta - rest[h];
    energy[h] = 0.5 * pr.kb * w * d * d;
    let coef = pr.kb * w * d;

    let base = 12u * h;
    let gth = array<vec3<f32>, 4>(g0, g1, g2, g3); // raw ∇θ per vertex
    for (var k = 0u; k < 4u; k = k + 1u) {
        grad[base + 3u * k] = gth[k].x * coef;
        grad[base + 3u * k + 1u] = gth[k].y * coef;
        grad[base + 3u * k + 2u] = gth[k].z * coef;
    }

    // Gauss-Newton Hessian block: H[3a+r][3b+c] = kb·w·∇θ[a]_r·∇θ[b]_c (12×12,
    // rank-1 outer product scaled by kb·w ≥ 0 → PSD by construction).
    let kw = pr.kb * w;
    let hbase = 144u * h;
    for (var a = 0u; a < 4u; a = a + 1u) {
        for (var b = 0u; b < 4u; b = b + 1u) {
            for (var r = 0u; r < 3u; r = r + 1u) {
                for (var c = 0u; c < 3u; c = c + 1u) {
                    hess[hbase + (3u * a + r) * 12u + (3u * b + c)] = kw * gth[a][r] * gth[b][c];
                }
            }
        }
    }
}
