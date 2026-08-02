// SPDX-License-Identifier: GPL-2.0-or-later
// Per-triangle Baraff-Witkin membrane assembly on the GPU: one invocation per
// triangle computes the 3-vertex gradient (9 floats) and the 9x9 element Hessian
// (81 floats), using the EIGEN-projected stretch + shear materials — the exact
// port of engine_core::membrane (the CPU version is the verification oracle).
// wu = Σ cuⱼ xⱼ, wv = Σ cvⱼ xⱼ, so only the (cu, cv) coefficients + positions are
// needed (no dm_inv). Blocks are row-major 3x3 (`b[3*r+c]`), matching the CPU
// `[[f32;3];3]`. This is the atomic-free per-triangle stage; the CSR gather that
// scatters these element blocks into the global matrix is the follow-up.

struct MP { ntri: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<uniform> mp: MP;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // 3n positions
@group(0) @binding(2) var<storage, read> tri: array<u32>;     // 3 per triangle
@group(0) @binding(3) var<storage, read> coef: array<f32>;    // 9 per tri: cu0..2,cv0..2,kwA,kfA,ksA
@group(0) @binding(4) var<storage, read_write> out_grad: array<f32>; // 9 per tri
@group(0) @binding(5) var<storage, read_write> out_hess: array<f32>; // 81 per tri

fn load3(base: u32) -> vec3<f32> {
    return vec3<f32>(x[3u * base], x[3u * base + 1u], x[3u * base + 2u]);
}

fn norm0(w: vec3<f32>) -> vec3<f32> {
    let l = length(w);
    if (l > 1e-12) { return w / l; }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// I − n̂n̂ᵀ, row-major 3x3.
fn proj_perp(w: vec3<f32>) -> array<f32, 9> {
    let n = norm0(w);
    var q: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            var ident = 0.0;
            if (r == c) { ident = 1.0; }
            q[3u * r + c] = ident - n[r] * n[c];
        }
    }
    return q;
}

// Eigen-projected stretch material: n̂n̂ᵀ + max(0,1−1/‖w‖)·(I−n̂n̂ᵀ), row-major.
fn stretch_mat(w: vec3<f32>) -> array<f32, 9> {
    let s = length(w);
    var n = vec3<f32>(0.0, 0.0, 0.0);
    var perp = 0.0;
    if (s > 1e-6) {
        n = w / s;
        perp = max(1.0 - 1.0 / s, 0.0);
    }
    var m: array<f32, 9>;
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            let nnt = n[r] * n[c];
            var ident = 0.0;
            if (r == c) { ident = 1.0; }
            m[3u * r + c] = nnt + perp * (ident - nnt);
        }
    }
    return m;
}

// Eigen-projected (clamped) shear material: four row-major 3x3 blocks packed as
// [uu(0..8), uv(9..17), vu(18..26), vv(27..35)].
fn shear_blocks(wu: vec3<f32>, wv: vec3<f32>) -> array<f32, 36> {
    let f = dot(wu, wv);
    let a = dot(wu, wu) + dot(wv, wv);
    let disc = sqrt(max(a * a + 12.0 * f * f, 0.0));
    let mu = array<f32, 2>(0.5 * (a + disc), 0.5 * (a - disc));
    var m: array<f32, 36>;
    // Mixing eigenvectors v = (wv·μ + wu·f ; wu·μ + wv·f), eigenvalue max(0,μ).
    for (var e = 0u; e < 2u; e = e + 1u) {
        let mui = mu[e];
        let top = wv * mui + wu * f;
        let bot = wu * mui + wv * f;
        let nn = dot(top, top) + dot(bot, bot);
        if (nn > 1e-20) {
            let s = max(mui, 0.0) / nn;
            for (var r = 0u; r < 3u; r = r + 1u) {
                for (var c = 0u; c < 3u; c = c + 1u) {
                    m[3u * r + c] += s * top[r] * top[c];
                    m[9u + 3u * r + c] += s * top[r] * bot[c];
                    m[18u + 3u * r + c] += s * bot[r] * top[c];
                    m[27u + 3u * r + c] += s * bot[r] * bot[c];
                }
            }
        }
    }
    // +Csh pair: ½·max(0,Csh)·[[Q+,Q+],[Q+,Q+]], Q+ = proj_perp(wu+wv).
    let fp = 0.5 * max(f, 0.0);
    if (fp != 0.0) {
        let qp = proj_perp(wu + wv);
        for (var i = 0u; i < 9u; i = i + 1u) {
            m[i] += fp * qp[i];
            m[9u + i] += fp * qp[i];
            m[18u + i] += fp * qp[i];
            m[27u + i] += fp * qp[i];
        }
    }
    // −Csh pair: ½·max(0,−Csh)·[[Q−,−Q−],[−Q−,Q−]], Q− = proj_perp(wv−wu).
    let fm = 0.5 * max(-f, 0.0);
    if (fm != 0.0) {
        let qm = proj_perp(wv - wu);
        for (var i = 0u; i < 9u; i = i + 1u) {
            m[i] += fm * qm[i];
            m[9u + i] += -fm * qm[i];
            m[18u + i] += -fm * qm[i];
            m[27u + i] += fm * qm[i];
        }
    }
    return m;
}

@compute @workgroup_size(64)
fn membrane(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= mp.ntri) { return; }
    let v0 = tri[3u * t];
    let v1 = tri[3u * t + 1u];
    let v2 = tri[3u * t + 2u];
    let x0 = load3(v0);
    let x1 = load3(v1);
    let x2 = load3(v2);

    let cu = array<f32, 3>(coef[9u * t], coef[9u * t + 1u], coef[9u * t + 2u]);
    let cv = array<f32, 3>(coef[9u * t + 3u], coef[9u * t + 4u], coef[9u * t + 5u]);
    let kwa = coef[9u * t + 6u];
    let kfa = coef[9u * t + 7u];
    let ksa = coef[9u * t + 8u];

    let wu = x0 * cu[0] + x1 * cu[1] + x2 * cu[2];
    let wv = x0 * cv[0] + x1 * cv[1] + x2 * cv[2];
    let cuu = length(wu) - 1.0;
    let cvv = length(wv) - 1.0;
    let csh = dot(wu, wv);
    let nu = norm0(wu);
    let nv = norm0(wv);

    // Gradient per vertex (analytic condition gradients × condition × k·A).
    for (var av = 0u; av < 3u; av = av + 1u) {
        let ga = nu * (kwa * cuu * cu[av])
            + nv * (kfa * cvv * cv[av])
            + (wv * cu[av] + wu * cv[av]) * (ksa * csh);
        out_grad[9u * t + 3u * av] = ga.x;
        out_grad[9u * t + 3u * av + 1u] = ga.y;
        out_grad[9u * t + 3u * av + 2u] = ga.z;
    }

    // Eigen materials.
    let mmu = stretch_mat(wu);
    let mmv = stretch_mat(wv);
    let sm = shear_blocks(wu, wv);

    // Element Hessian: block(a,b)[r][c] at out_hess[81t + (3a+r)*9 + (3b+c)].
    for (var av = 0u; av < 3u; av = av + 1u) {
        for (var bv = 0u; bv < 3u; bv = bv + 1u) {
            for (var r = 0u; r < 3u; r = r + 1u) {
                for (var c = 0u; c < 3u; c = c + 1u) {
                    let idx = 3u * r + c;
                    let val = kwa * cu[av] * cu[bv] * mmu[idx]
                        + kfa * cv[av] * cv[bv] * mmv[idx]
                        + ksa * (cu[av] * cu[bv] * sm[idx]
                            + cu[av] * cv[bv] * sm[9u + idx]
                            + cv[av] * cu[bv] * sm[18u + idx]
                            + cv[av] * cv[bv] * sm[27u + idx]);
                    out_hess[81u * t + (3u * av + r) * 9u + (3u * bv + c)] = val;
                }
            }
        }
    }
}
