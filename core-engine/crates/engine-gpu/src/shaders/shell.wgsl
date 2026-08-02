// SPDX-License-Identifier: GPL-2.0-or-later
// Isotropic SHELL element kernel — the 3x2 twin of tet.wgsl. One thread per triangle:
// deformation gradient (3x2) -> thin SVD -> principal-stretch derivatives -> the six
// known eigenpairs -> analytic 9-dof gradient and 9x9 Hessian. Mirrors
// `engine_core::shell_model` exactly; verified GPU==CPU by test.
//
// Layout conventions, both deliberate:
//  * A 2x2 is a vec4 laid out (x_axis.x, x_axis.y, y_axis.x, y_axis.y) — COLUMN-major,
//    matching glam's `Mat2::from_cols`, so the CPU layout carries over untouched. WGSL's
//    own mat2x2 is avoided to keep that mapping explicit rather than implied.
//  * The 3x2 `vec6` is ROW-major, `v[2*r + c]`, as on the CPU. The convention cancels in
//    the 6x6 assembly, but it must be the SAME convention on both sides.
//
// `model`: 0 = ARAP, 1 = StVK, 2 = SNHk (an explicit wire value, not a Rust discriminant).

struct SParams {
    n: u32,
    mu: f32,
    lambda: f32,
    eps: f32,
    model: u32,
    // Element offset for the OUTPUT buffers. Lets the shell pass write into a shared
    // egrad/ehess buffer after the membrane elements, so both element kinds occupy one
    // element-index space and the union gather never learns there are two kinds.
    out_base: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> pr: SParams;
@group(0) @binding(1) var<storage, read> idx: array<u32>;            // 3 * n
@group(0) @binding(2) var<storage, read> dm_inv: array<f32>;         // 4 * n
@group(0) @binding(3) var<storage, read> area: array<f32>;           // n
@group(0) @binding(4) var<storage, read> pos: array<f32>;            // 3 * n_verts
@group(0) @binding(5) var<storage, read_write> energy: array<f32>;   // n
@group(0) @binding(6) var<storage, read_write> grad: array<f32>;     // 9 * n
@group(0) @binding(7) var<storage, read_write> hess: array<f32>;     // 81 * n

// A 3x2 matrix as its two 3-D columns.
struct M32 {
    c0: vec3<f32>,
    c1: vec3<f32>,
};

fn m2_transpose(m: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(m.x, m.z, m.y, m.w);
}

// self * m, for a 2x2 m.
fn m32_mul_m2(a: M32, m: vec4<f32>) -> M32 {
    return M32(a.c0 * m.x + a.c1 * m.y, a.c0 * m.z + a.c1 * m.w);
}

// M^T * M (2x2, symmetric).
fn m32_ata(a: M32) -> vec4<f32> {
    return vec4<f32>(dot(a.c0, a.c0), dot(a.c1, a.c0), dot(a.c0, a.c1), dot(a.c1, a.c1));
}

struct Eig2 {
    lam: vec2<f32>,
    v: vec4<f32>,
};

// Symmetric 2x2 eigendecomposition, closed form, eigenvalues descending.
fn sym_eig2(a: vec4<f32>) -> Eig2 {
    let p = a.x;
    let q = a.z;
    let r = a.w;
    let tr = p + r;
    let diff = sqrt(max((p - r) * (p - r) + 4.0 * q * q, 0.0));
    let l0 = 0.5 * (tr + diff);
    let l1 = 0.5 * (tr - diff);
    var v0 = vec2<f32>(1.0, 0.0);
    if abs(q) > 1.0e-20 {
        let t = vec2<f32>(q, l0 - p);
        let n = length(t);
        if n > 0.0 {
            v0 = t / n;
        }
    } else if p < r {
        v0 = vec2<f32>(0.0, 1.0);
    }
    let v1 = vec2<f32>(-v0.y, v0.x);
    return Eig2(vec2<f32>(l0, l1), vec4<f32>(v0.x, v0.y, v1.x, v1.y));
}

struct Svd32 {
    u: M32,
    sigma: vec2<f32>,
    v: vec4<f32>,
};

// Thin SVD of a 3x2: F = U*diag(sigma)*V^T. No orientation to preserve (a triangle has
// no signed volume), so the singular values stay non-negative.
fn svd3x2(f: M32) -> Svd32 {
    let e = sym_eig2(m32_ata(f));
    let sigma = vec2<f32>(sqrt(max(e.lam.x, 0.0)), sqrt(max(e.lam.y, 0.0)));
    let eps = 1.0e-9 * max(sigma.x, 1.0);
    let v0 = vec2<f32>(e.v.x, e.v.y);
    let v1 = vec2<f32>(e.v.z, e.v.w);

    var u0 = vec3<f32>(1.0, 0.0, 0.0);
    if sigma.x > eps {
        u0 = (f.c0 * v0.x + f.c1 * v0.y) / sigma.x;
    }
    let n0 = length(u0);
    if n0 > 0.0 {
        u0 = u0 / n0;
    } else {
        u0 = vec3<f32>(1.0, 0.0, 0.0);
    }

    var u1r: vec3<f32>;
    if sigma.y > eps {
        u1r = (f.c0 * v1.x + f.c1 * v1.y) / sigma.y;
    } else {
        // Any unit vector orthogonal to u0.
        var t = vec3<f32>(1.0, 0.0, 0.0);
        if abs(u0.x) >= 0.9 {
            t = vec3<f32>(0.0, 1.0, 0.0);
        }
        u1r = t - dot(t, u0) * u0;
    }
    var u1 = u1r - dot(u1r, u0) * u0;
    let n1 = length(u1);
    if n1 > 0.0 {
        u1 = u1 / n1;
    } else {
        u1 = vec3<f32>(0.0, 1.0, 0.0);
    }
    return Svd32(M32(u0, u1), sigma, e.v);
}

fn dev_dpsi(s: f32) -> f32 {
    if pr.model == 1u {
        let d = s - 1.0;
        return pr.mu * d * d * d;
    }
    return pr.mu * (s - 1.0);
}

fn dev_d2psi(s: f32) -> f32 {
    if pr.model == 1u {
        let d = s - 1.0;
        return 3.0 * pr.mu * d * d;
    }
    return pr.mu;
}

fn shell_density(sigma: vec2<f32>) -> f32 {
    var dev: f32;
    let a = sigma.x - 1.0;
    let b = sigma.y - 1.0;
    let j = sigma.x * sigma.y;
    if pr.model == 1u {
        dev = pr.mu * (a * a * a * a + b * b * b * b) / 4.0;
    } else if pr.model == 2u {
        // 2D Stable Neo-Hookean: 0.5*mu*(I2-2) - mu*(J-1).
        dev = 0.5 * pr.mu * (sigma.x * sigma.x + sigma.y * sigma.y - 2.0) - pr.mu * (j - 1.0);
    } else {
        dev = 0.5 * pr.mu * (a * a + b * b);
    }
    return dev + 0.5 * pr.lambda * (j - 1.0) * (j - 1.0);
}

fn dpsi_dsigma(sigma: vec2<f32>) -> vec2<f32> {
    let j = sigma.x * sigma.y;
    if pr.model == 2u {
        // snhk.hpp::make_diff_table2 — the deviatoric half COUPLES the stretches.
        return vec2<f32>(
            pr.lambda * sigma.y * (j - 1.0) + pr.mu * (sigma.x - sigma.y),
            pr.lambda * sigma.x * (j - 1.0) + pr.mu * (sigma.y - sigma.x),
        );
    }
    return vec2<f32>(
        dev_dpsi(sigma.x) + pr.lambda * sigma.y * (j - 1.0),
        dev_dpsi(sigma.y) + pr.lambda * sigma.x * (j - 1.0),
    );
}

// Row-major 2x2: [d00, d01, d10, d11].
fn d2psi_dsigma2(sigma: vec2<f32>) -> vec4<f32> {
    let a = sigma.x;
    let b = sigma.y;
    if pr.model == 2u {
        let off = 2.0 * pr.lambda * a * b - pr.lambda - pr.mu;
        return vec4<f32>(pr.lambda * b * b + pr.mu, off, off, pr.lambda * a * a + pr.mu);
    }
    let cross = pr.lambda * (2.0 * a * b - 1.0);
    return vec4<f32>(
        dev_d2psi(a) + pr.lambda * b * b,
        cross,
        cross,
        dev_d2psi(b) + pr.lambda * a * a,
    );
}

@compute @workgroup_size(64)
fn shell(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= pr.n {
        return;
    }
    let i0 = idx[3u * e + 0u];
    let i1 = idx[3u * e + 1u];
    let i2 = idx[3u * e + 2u];
    let p0 = vec3<f32>(pos[3u * i0], pos[3u * i0 + 1u], pos[3u * i0 + 2u]);
    let p1 = vec3<f32>(pos[3u * i1], pos[3u * i1 + 1u], pos[3u * i1 + 2u]);
    let p2 = vec3<f32>(pos[3u * i2], pos[3u * i2 + 1u], pos[3u * i2 + 2u]);
    let dm = vec4<f32>(
        dm_inv[4u * e + 0u],
        dm_inv[4u * e + 1u],
        dm_inv[4u * e + 2u],
        dm_inv[4u * e + 3u],
    );
    let ar = area[e];

    // F = Ds * Dm^-1, Ds = [p1-p0 | p2-p0].
    let f = m32_mul_m2(M32(p1 - p0, p2 - p0), dm);
    let s = svd3x2(f);
    let deda = dpsi_dsigma(s.sigma);
    let d2 = d2psi_dsigma2(s.sigma);

    let o = pr.out_base + e;
    energy[o] = ar * shell_density(s.sigma);

    // ---- gradient: P = U*diag(deda)*V^T, then dE/dDs = area * P * Dm^-T ----
    let pk1 = m32_mul_m2(M32(s.u.c0 * deda.x, s.u.c1 * deda.y), m2_transpose(s.v));
    let hg = m32_mul_m2(pk1, m2_transpose(dm));
    let g1 = hg.c0 * ar;
    let g2 = hg.c1 * ar;
    let g0 = -(g1 + g2);
    grad[9u * o + 0u] = g0.x;
    grad[9u * o + 1u] = g0.y;
    grad[9u * o + 2u] = g0.z;
    grad[9u * o + 3u] = g1.x;
    grad[9u * o + 4u] = g1.y;
    grad[9u * o + 5u] = g1.z;
    grad[9u * o + 6u] = g2.x;
    grad[9u * o + 7u] = g2.y;
    grad[9u * o + 8u] = g2.z;

    // ---- the six eigenpairs of d2Psi/dF2 ----
    let ev = sym_eig2(vec4<f32>(d2.x, d2.z, d2.y, d2.w));
    let a = s.sigma.x;
    let b = s.sigma.y;
    let denom = a - b;
    let fallback = 0.5 * (d2.x + d2.w) - 0.5 * (d2.y + d2.z);
    var lam: array<f32, 6>;
    lam[0] = (deda.x + deda.y) / (a + b);
    if abs(denom) > pr.eps {
        lam[1] = (deda.x - deda.y) / denom;
    } else {
        lam[1] = fallback;
    }
    lam[2] = deda.x / a;
    lam[3] = deda.y / b;
    lam[4] = ev.lam.x;
    lam[5] = ev.lam.y;

    // Mode matrices Q_k, given by their two columns as coefficients over (u0, u1, u2).
    // Q[1] (the flip mode) is the SYMMETRIC off-diagonal matrix, NOT the antisymmetric
    // twist: using twist there makes vec(U*Q0*V^T) == vec(U*Q1*V^T) and silently
    // collapses the flip subspace whenever the singular values differ. Same trap the
    // CPU port documents.
    let r2 = 0.70710678118654752440;
    let u2 = cross(s.u.c0, s.u.c1);
    let vt = m2_transpose(s.v);

    var a6: array<f32, 36>;
    for (var k = 0u; k < 6u; k = k + 1u) {
        let l = max(lam[k], 0.0);
        if l == 0.0 {
            continue;
        }
        // Columns of Q_k as (coef_u0, coef_u1, coef_u2).
        var qc0: vec3<f32>;
        var qc1: vec3<f32>;
        if k == 0u {
            qc0 = vec3<f32>(0.0, -r2, 0.0);
            qc1 = vec3<f32>(r2, 0.0, 0.0);
        } else if k == 1u {
            qc0 = vec3<f32>(0.0, r2, 0.0);
            qc1 = vec3<f32>(r2, 0.0, 0.0);
        } else if k == 2u {
            qc0 = vec3<f32>(0.0, 0.0, 1.0);
            qc1 = vec3<f32>(0.0, 0.0, 0.0);
        } else if k == 3u {
            qc0 = vec3<f32>(0.0, 0.0, 0.0);
            qc1 = vec3<f32>(0.0, 0.0, 1.0);
        } else if k == 4u {
            qc0 = vec3<f32>(ev.v.x, 0.0, 0.0);
            qc1 = vec3<f32>(0.0, ev.v.y, 0.0);
        } else {
            qc0 = vec3<f32>(ev.v.z, 0.0, 0.0);
            qc1 = vec3<f32>(0.0, ev.v.w, 0.0);
        }
        // U3 * Q_k, then * V^T.
        let uq = M32(
            s.u.c0 * qc0.x + s.u.c1 * qc0.y + u2 * qc0.z,
            s.u.c0 * qc1.x + s.u.c1 * qc1.y + u2 * qc1.z,
        );
        let qv = m32_mul_m2(uq, vt);
        // Row-major vec6 of the 3x2: v[2r + c].
        var q: array<f32, 6>;
        q[0] = qv.c0.x;
        q[1] = qv.c1.x;
        q[2] = qv.c0.y;
        q[3] = qv.c1.y;
        q[4] = qv.c0.z;
        q[5] = qv.c1.z;
        for (var r = 0u; r < 6u; r = r + 1u) {
            for (var c = 0u; c < 6u; c = c + 1u) {
                a6[r * 6u + c] = a6[r * 6u + c] + l * q[r] * q[c];
            }
        }
    }

    // ---- dF/dx as a row-major 6x9, then hess = area * B^T * A6 * B ----
    var bm: array<f32, 54>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 2u; j = j + 1u) {
            let row = 2u * i + j;
            var sum = 0.0;
            for (var kk = 0u; kk < 2u; kk = kk + 1u) {
                // dm_inv[kk][j] with the column-major vec4 layout.
                var d: f32;
                if j == 0u {
                    d = select(dm.y, dm.x, kk == 0u);
                } else {
                    d = select(dm.w, dm.z, kk == 0u);
                }
                bm[row * 9u + 3u * (kk + 1u) + i] = d;
                sum = sum + d;
            }
            bm[row * 9u + i] = -sum;
        }
    }
    var ab: array<f32, 54>;
    for (var r = 0u; r < 6u; r = r + 1u) {
        for (var c = 0u; c < 9u; c = c + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 6u; k = k + 1u) {
                acc = acc + a6[r * 6u + k] * bm[k * 9u + c];
            }
            ab[r * 9u + c] = acc;
        }
    }
    for (var r = 0u; r < 9u; r = r + 1u) {
        for (var c = 0u; c < 9u; c = c + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 6u; k = k + 1u) {
                acc = acc + bm[k * 9u + r] * ab[k * 9u + c];
            }
            hess[81u * o + r * 9u + c] = ar * acc;
        }
    }
}
