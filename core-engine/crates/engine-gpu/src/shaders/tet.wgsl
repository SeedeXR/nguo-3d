// SPDX-License-Identifier: GPL-2.0-or-later
// Volumetric (tet) FEM element kernel: one thread per tetrahedron computes the StVK
// energy, the 12-dof analytic gradient (Piola stress) and the 12×12 analytic Hessian
// (isotropic eigensystem), mirroring engine_core::tet exactly.
//
// All 3×3 matrices are row-major array<f32,9> (m[3*r + c]) so the CPU/GPU vec()
// convention matches by construction — WGSL's mat3x3 is column-major and mixing the
// two is an easy silent bug.

struct TParams { n_tets: u32, mu: f32, lambda: f32, eps: f32 };
@group(0) @binding(0) var<uniform> pr: TParams;
@group(0) @binding(1) var<storage, read> idx: array<u32>;      // 4 * n_tets
@group(0) @binding(2) var<storage, read> dm_inv: array<f32>;   // 9 * n_tets (row-major)
@group(0) @binding(3) var<storage, read> volume: array<f32>;   // n_tets
@group(0) @binding(4) var<storage, read> pos: array<f32>;      // 3 * n_verts
@group(0) @binding(5) var<storage, read_write> energy: array<f32>; // n_tets
@group(0) @binding(6) var<storage, read_write> grad: array<f32>;   // 12 * n_tets
@group(0) @binding(7) var<storage, read_write> hess: array<f32>;   // 144 * n_tets

fn mm(a: array<f32, 9>, b: array<f32, 9>) -> array<f32, 9> {
    var o: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 3u; k = k + 1u) { acc = acc + a[3u * r + k] * b[3u * k + c]; }
            o[3u * r + c] = acc;
        }
    }
    return o;
}
fn mt(a: array<f32, 9>) -> array<f32, 9> {
    return array<f32, 9>(a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]);
}
fn det3(m: array<f32, 9>) -> f32 {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
fn mcol(m: array<f32, 9>, c: u32) -> vec3<f32> {
    return vec3<f32>(m[c], m[3u + c], m[6u + c]);
}
fn matvec(m: array<f32, 9>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

// Cyclic-Jacobi symmetric eigendecomposition; eigenvectors returned as COLUMNS.
fn sym_eig3(a_in: array<f32, 9>) -> array<f32, 12> { // [0..8]=V(cols), [9..11]=lambda
    var a = a_in;
    var v = array<f32, 9>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    for (var sweep = 0u; sweep < 12u; sweep = sweep + 1u) {
        for (var pq = 0u; pq < 3u; pq = pq + 1u) {
            var p = 0u; var q = 1u;
            if (pq == 1u) { p = 0u; q = 2u; } else if (pq == 2u) { p = 1u; q = 2u; }
            let apq = a[3u * p + q];
            if (abs(apq) < 1e-20) { continue; }
            let app = a[3u * p + p];
            let aqq = a[3u * q + q];
            let phi = 0.5 * atan2(2.0 * apq, aqq - app);
            let c = cos(phi);
            let s = sin(phi);
            for (var k = 0u; k < 3u; k = k + 1u) {
                let akp = a[3u * k + p]; let akq = a[3u * k + q];
                a[3u * k + p] = c * akp - s * akq;
                a[3u * k + q] = s * akp + c * akq;
            }
            for (var k = 0u; k < 3u; k = k + 1u) {
                let apk = a[3u * p + k]; let aqk = a[3u * q + k];
                a[3u * p + k] = c * apk - s * aqk;
                a[3u * q + k] = s * apk + c * aqk;
            }
            for (var k = 0u; k < 3u; k = k + 1u) {
                let vkp = v[3u * k + p]; let vkq = v[3u * k + q];
                v[3u * k + p] = c * vkp - s * vkq;
                v[3u * k + q] = s * vkp + c * vkq;
            }
        }
    }
    return array<f32, 12>(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                          a[0], a[4], a[8]);
}

@compute @workgroup_size(64)
fn tet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= pr.n_tets) { return; }
    let i0 = idx[4u * t]; let i1 = idx[4u * t + 1u];
    let i2 = idx[4u * t + 2u]; let i3 = idx[4u * t + 3u];
    let p0 = vec3<f32>(pos[3u * i0], pos[3u * i0 + 1u], pos[3u * i0 + 2u]);
    let p1 = vec3<f32>(pos[3u * i1], pos[3u * i1 + 1u], pos[3u * i1 + 2u]);
    let p2 = vec3<f32>(pos[3u * i2], pos[3u * i2 + 1u], pos[3u * i2 + 2u]);
    let p3 = vec3<f32>(pos[3u * i3], pos[3u * i3 + 1u], pos[3u * i3 + 2u]);

    var dmi: array<f32, 9>;
    for (var k = 0u; k < 9u; k = k + 1u) { dmi[k] = dm_inv[9u * t + k]; }
    // Ds has COLUMNS (p1-p0, p2-p0, p3-p0); store row-major.
    let e1 = p1 - p0; let e2 = p2 - p0; let e3 = p3 - p0;
    let ds = array<f32, 9>(e1.x, e2.x, e3.x, e1.y, e2.y, e3.y, e1.z, e2.z, e3.z);
    let f = mm(ds, dmi);

    // ---- SVD: F = U diag(sigma) V^T, both proper rotations, sigma[2] signed.
    let ata = mm(mt(f), f);
    let ev = sym_eig3(ata);
    var vm = array<f32, 9>(ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8]);
    var lam = vec3<f32>(ev[9], ev[10], ev[11]);
    // Sort descending (3-element network) tracking columns.
    var ord = vec3<u32>(0u, 1u, 2u);
    if (lam[ord.x] < lam[ord.y]) { let tmp = ord.x; ord.x = ord.y; ord.y = tmp; }
    if (lam[ord.x] < lam[ord.z]) { let tmp = ord.x; ord.x = ord.z; ord.z = tmp; }
    if (lam[ord.y] < lam[ord.z]) { let tmp = ord.y; ord.y = ord.z; ord.z = tmp; }
    let v0 = normalize(mcol(vm, ord.x));
    let v1 = normalize(mcol(vm, ord.y));
    var v2 = normalize(mcol(vm, ord.z));
    var vs = array<f32, 9>(v0.x, v1.x, v2.x, v0.y, v1.y, v2.y, v0.z, v1.z, v2.z);
    if (det3(vs) < 0.0) {
        v2 = -v2;
        vs = array<f32, 9>(v0.x, v1.x, v2.x, v0.y, v1.y, v2.y, v0.z, v1.z, v2.z);
    }
    var sig = vec3<f32>(sqrt(max(lam[ord.x], 0.0)), sqrt(max(lam[ord.y], 0.0)), sqrt(max(lam[ord.z], 0.0)));
    let epsu = 1e-9 * max(sig.x, 1.0);
    var u0 = matvec(f, v0);
    u0 = select(vec3<f32>(1.0, 0.0, 0.0), u0 / sig.x, sig.x > epsu);
    u0 = normalize(u0);
    var u1 = matvec(f, v1);
    if (sig.y > epsu) { u1 = u1 / sig.y; } else {
        var tv = vec3<f32>(1.0, 0.0, 0.0);
        if (abs(u0.x) >= 0.9) { tv = vec3<f32>(0.0, 1.0, 0.0); }
        u1 = tv - dot(tv, u0) * u0;
    }
    u1 = normalize(u1 - dot(u1, u0) * u0);
    let u2 = cross(u0, u1);
    let us = array<f32, 9>(u0.x, u1.x, u2.x, u0.y, u1.y, u2.y, u0.z, u1.z, u2.z);
    if (dot(matvec(f, v2), u2) < 0.0) { sig.z = -sig.z; }

    // ---- StVK derivatives on stretches.
    let mu = pr.mu; let lm = pr.lambda;
    let a = sig.x; let b = sig.y; let c = sig.z;
    let jj = a * b * c;
    let deda = vec3<f32>(
        mu * pow(a - 1.0, 3.0) + lm * (jj - 1.0) * (b * c),
        mu * pow(b - 1.0, 3.0) + lm * (jj - 1.0) * (a * c),
        mu * pow(c - 1.0, 3.0) + lm * (jj - 1.0) * (a * b));
    let tt = 2.0 * jj - 1.0;
    var d2 = array<f32, 9>(
        3.0 * mu * (a - 1.0) * (a - 1.0) + lm * (b * c) * (b * c), lm * c * tt, lm * b * tt,
        lm * c * tt, 3.0 * mu * (b - 1.0) * (b - 1.0) + lm * (a * c) * (a * c), lm * a * tt,
        lm * b * tt, lm * a * tt, 3.0 * mu * (c - 1.0) * (c - 1.0) + lm * (a * b) * (a * b));

    // Energy.
    let dev = pow(a - 1.0, 4.0) + pow(b - 1.0, 4.0) + pow(c - 1.0, 4.0);
    energy[t] = volume[t] * (mu * dev / 4.0 + 0.5 * lm * (jj - 1.0) * (jj - 1.0));

    // ---- Gradient: P = U diag(deda) V^T ; dE/dDs = vol * P * Dm^-T.
    let ud = array<f32, 9>(
        us[0] * deda.x, us[1] * deda.y, us[2] * deda.z,
        us[3] * deda.x, us[4] * deda.y, us[5] * deda.z,
        us[6] * deda.x, us[7] * deda.y, us[8] * deda.z);
    let pk1 = mm(ud, mt(vs));
    let hmat = mm(pk1, mt(dmi));
    let g1 = vec3<f32>(hmat[0], hmat[3], hmat[6]) * volume[t];
    let g2 = vec3<f32>(hmat[1], hmat[4], hmat[7]) * volume[t];
    let g3 = vec3<f32>(hmat[2], hmat[5], hmat[8]) * volume[t];
    let g0 = -(g1 + g2 + g3);
    let gb = 12u * t;
    grad[gb] = g0.x; grad[gb + 1u] = g0.y; grad[gb + 2u] = g0.z;
    grad[gb + 3u] = g1.x; grad[gb + 4u] = g1.y; grad[gb + 5u] = g1.z;
    grad[gb + 6u] = g2.x; grad[gb + 7u] = g2.y; grad[gb + 8u] = g2.z;
    grad[gb + 9u] = g3.x; grad[gb + 10u] = g3.y; grad[gb + 11u] = g3.z;

    // ---- Hessian: 9 eigenpairs of d2Psi/dF2, clamped at 0 (reference filter).
    let ev2 = sym_eig3(d2);
    let evec = array<f32, 9>(ev2[0], ev2[1], ev2[2], ev2[3], ev2[4], ev2[5], ev2[6], ev2[7], ev2[8]);
    let eval3 = vec3<f32>(ev2[9], ev2[10], ev2[11]);
    let r2 = 0.70710678118654752;
    var qs: array<array<f32, 9>, 9>;
    qs[0] = array<f32, 9>(0.0, r2, 0.0, -r2, 0.0, 0.0, 0.0, 0.0, 0.0);
    qs[1] = array<f32, 9>(0.0, 0.0, r2, 0.0, 0.0, 0.0, -r2, 0.0, 0.0);
    qs[2] = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, r2, 0.0, -r2, 0.0);
    qs[3] = array<f32, 9>(0.0, r2, 0.0, r2, 0.0, 0.0, 0.0, 0.0, 0.0);
    qs[4] = array<f32, 9>(0.0, 0.0, r2, 0.0, 0.0, 0.0, r2, 0.0, 0.0);
    qs[5] = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, r2, 0.0, r2, 0.0);
    for (var k = 6u; k < 9u; k = k + 1u) {
        let e = mcol(evec, k - 6u);
        qs[k] = array<f32, 9>(e.x, 0.0, 0.0, 0.0, e.y, 0.0, 0.0, 0.0, e.z);
    }
    var lamv: array<f32, 9>;
    lamv[0] = (deda.x + deda.y) / (a + b);
    lamv[1] = (deda.x + deda.z) / (a + c);
    lamv[2] = (deda.y + deda.z) / (b + c);
    let fb01 = 0.5 * (d2[0] + d2[4]) - 0.5 * (d2[1] + d2[3]);
    let fb02 = 0.5 * (d2[0] + d2[8]) - 0.5 * (d2[2] + d2[6]);
    let fb12 = 0.5 * (d2[4] + d2[8]) - 0.5 * (d2[5] + d2[7]);
    lamv[3] = select(fb01, (deda.x - deda.y) / (a - b), abs(a - b) > pr.eps);
    lamv[4] = select(fb02, (deda.x - deda.z) / (a - c), abs(a - c) > pr.eps);
    lamv[5] = select(fb12, (deda.y - deda.z) / (b - c), abs(b - c) > pr.eps);
    lamv[6] = eval3.x; lamv[7] = eval3.y; lamv[8] = eval3.z;

    var a9: array<f32, 81>;
    for (var k = 0u; k < 81u; k = k + 1u) { a9[k] = 0.0; }
    for (var k = 0u; k < 9u; k = k + 1u) {
        let l = max(lamv[k], 0.0);
        if (l <= 0.0) { continue; }
        let qm = mm(mm(us, qs[k]), mt(vs));
        for (var r = 0u; r < 9u; r = r + 1u) {
            for (var cc = 0u; cc < 9u; cc = cc + 1u) {
                a9[r * 9u + cc] = a9[r * 9u + cc] + l * qm[r] * qm[cc];
            }
        }
    }

    // dF/dx (9×12), constant: F = Ds Dm^-1 is linear in x.
    var bmat: array<f32, 108>;
    for (var k = 0u; k < 108u; k = k + 1u) { bmat[k] = 0.0; }
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            let row = 3u * i + j;
            var sum = 0.0;
            for (var k = 0u; k < 3u; k = k + 1u) {
                let d = dmi[3u * k + j];
                bmat[row * 12u + 3u * (k + 1u) + i] = d;
                sum = sum + d;
            }
            bmat[row * 12u + i] = -sum;
        }
    }
    // H = vol * B^T A B
    let hb = 144u * t;
    for (var r = 0u; r < 12u; r = r + 1u) {
        for (var cc = 0u; cc < 12u; cc = cc + 1u) {
            var acc = 0.0;
            for (var k = 0u; k < 9u; k = k + 1u) {
                var inner = 0.0;
                for (var l2 = 0u; l2 < 9u; l2 = l2 + 1u) {
                    inner = inner + a9[k * 9u + l2] * bmat[l2 * 12u + cc];
                }
                acc = acc + bmat[k * 12u + r] * inner;
            }
            hess[hb + r * 12u + cc] = volume[t] * acc;
        }
    }
}
