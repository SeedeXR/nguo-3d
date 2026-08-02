// SPDX-License-Identifier: GPL-2.0-or-later
// 3×3 SVD + polar decomposition as a GPU device function, for corotational/ARAP
// elasticity. F = U·diag(σ)·Vᵀ (σ ≥ 0), and the polar rotation R = U·Vᵀ made a
// PROPER rotation (det +1) by flipping the smallest singular value when F contains
// a reflection (det F < 0). Then F = R·S with S = V·diag(σ_signed)·Vᵀ symmetric.
// Built on the same cyclic-Jacobi eigensolver as sym_eig.wgsl (eig of FᵀF).
// One thread per matrix; row-major 9-float blocks.

struct SParams { count: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> pp: SParams;
@group(0) @binding(1) var<storage, read> fin: array<f32>;        // 9*count
@group(0) @binding(2) var<storage, read_write> uout: array<f32>; // 9*count
@group(0) @binding(3) var<storage, read_write> sout: array<f32>; // 3*count (unsigned σ, descending)
@group(0) @binding(4) var<storage, read_write> vout: array<f32>; // 9*count
@group(0) @binding(5) var<storage, read_write> rout: array<f32>; // 9*count (proper rotation)

const SWEEPS: u32 = 12u;

fn jrot(a: ptr<function, array<f32, 9>>, v: ptr<function, array<f32, 9>>, pi: u32, qi: u32) {
    let apq = (*a)[3u * pi + qi];
    if (abs(apq) < 1e-20) { return; }
    let app = (*a)[3u * pi + pi];
    let aqq = (*a)[3u * qi + qi];
    let phi = 0.5 * atan2(2.0 * apq, aqq - app);
    let c = cos(phi);
    let s = sin(phi);
    for (var k = 0u; k < 3u; k = k + 1u) {
        let akp = (*a)[3u * k + pi]; let akq = (*a)[3u * k + qi];
        (*a)[3u * k + pi] = c * akp - s * akq;
        (*a)[3u * k + qi] = s * akp + c * akq;
    }
    for (var k = 0u; k < 3u; k = k + 1u) {
        let apk = (*a)[3u * pi + k]; let aqk = (*a)[3u * qi + k];
        (*a)[3u * pi + k] = c * apk - s * aqk;
        (*a)[3u * qi + k] = s * apk + c * aqk;
    }
    for (var k = 0u; k < 3u; k = k + 1u) {
        let vkp = (*v)[3u * k + pi]; let vkq = (*v)[3u * k + qi];
        (*v)[3u * k + pi] = c * vkp - s * vkq;
        (*v)[3u * k + qi] = s * vkp + c * vkq;
    }
}

fn col(m: array<f32, 9>, c: u32) -> vec3<f32> {
    return vec3<f32>(m[c], m[3u + c], m[6u + c]);
}
fn set_col(m: ptr<function, array<f32, 9>>, c: u32, v: vec3<f32>) {
    (*m)[c] = v.x; (*m)[3u + c] = v.y; (*m)[6u + c] = v.z;
}
fn matvec(m: array<f32, 9>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    );
}
fn det3(m: array<f32, 9>) -> f32 {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

@compute @workgroup_size(64)
fn svd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= pp.count) { return; }
    var f: array<f32, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) { f[i] = fin[9u * m + i]; }

    // S = Fᵀ F (symmetric).
    var a: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 3u; k = k + 1u) { acc = acc + f[3u * k + r] * f[3u * k + c]; }
            a[3u * r + c] = acc;
        }
    }
    var v: array<f32, 9> = array<f32, 9>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    for (var sweep = 0u; sweep < SWEEPS; sweep = sweep + 1u) {
        jrot(&a, &v, 0u, 1u);
        jrot(&a, &v, 0u, 2u);
        jrot(&a, &v, 1u, 2u);
    }
    var lam = vec3<f32>(a[0], a[4], a[8]);

    // Sort eigenpairs by eigenvalue, descending (3-element network).
    var ord = vec3<u32>(0u, 1u, 2u);
    if (lam[ord.x] < lam[ord.y]) { let t = ord.x; ord.x = ord.y; ord.y = t; }
    if (lam[ord.x] < lam[ord.z]) { let t = ord.x; ord.x = ord.z; ord.z = t; }
    if (lam[ord.y] < lam[ord.z]) { let t = ord.y; ord.y = ord.z; ord.z = t; }

    var vs: array<f32, 9>;
    set_col(&vs, 0u, normalize(col(v, ord.x)));
    set_col(&vs, 1u, normalize(col(v, ord.y)));
    set_col(&vs, 2u, normalize(col(v, ord.z)));
    var sig = vec3<f32>(sqrt(max(lam[ord.x], 0.0)), sqrt(max(lam[ord.y], 0.0)), sqrt(max(lam[ord.z], 0.0)));

    // U columns: u_i = F v_i / σ_i, with cross-product completion for tiny σ.
    let eps = 1e-6 * max(sig.x, 1.0);
    let b0 = matvec(f, col(vs, 0u));
    let b1 = matvec(f, col(vs, 1u));
    var u0 = select(vec3<f32>(1.0, 0.0, 0.0), b0 / sig.x, sig.x > eps);
    u0 = normalize(u0);
    var u1 = select(vec3<f32>(0.0, 1.0, 0.0), b1 / sig.y, sig.y > eps);
    u1 = normalize(u1 - dot(u1, u0) * u0); // re-orthonormalize
    // Complete the basis; align its sign with F·v2 so U·diag(σ)·Vᵀ reconstructs F
    // exactly (cross alone could pick the opposite sign).
    var u2 = cross(u0, u1);
    if (dot(u2, matvec(f, col(vs, 2u))) < 0.0) { u2 = -u2; }
    var u: array<f32, 9>;
    set_col(&u, 0u, u0); set_col(&u, 1u, u1); set_col(&u, 2u, u2);

    // Proper-rotation polar factor: R = U Vᵀ; if it reflects (det U·V < 0), flip the
    // last U column so R is a proper rotation (det +1). Reconstruction uses U as-is.
    var us = u;
    if (det3(u) * det3(vs) < 0.0) { set_col(&us, 2u, -u2); }
    var r: array<f32, 9>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 3u; k = k + 1u) { acc = acc + us[3u * i + k] * vs[3u * j + k]; }
            r[3u * i + j] = acc;
        }
    }

    for (var i = 0u; i < 9u; i = i + 1u) {
        uout[9u * m + i] = u[i];
        vout[9u * m + i] = vs[i];
        rout[9u * m + i] = r[i];
    }
    sout[3u * m] = sig.x; sout[3u * m + 1u] = sig.y; sout[3u * m + 2u] = sig.z;
}
