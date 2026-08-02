// SPDX-License-Identifier: GPL-2.0-or-later
// 3×3 symmetric eigenanalysis as a GPU device function (cyclic Jacobi, fixed
// sweeps) + PSD projection. The reference filters Hessian blocks to positive-
// semidefinite via eigenanalysis (docs/KERNELS.md energy/eigenanalysis); this is
// the GPU form. One thread per matrix. Matrices are row-major 9-float blocks.
//
// A = V·diag(λ)·Vᵀ. `project` clamps λ→max(λ,0) and reconstructs → the nearest
// PSD matrix (Frobenius), which is the Hessian-filtering operation.

struct EParams { count: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> p: EParams;
@group(0) @binding(1) var<storage, read> ain: array<f32>;        // 9 * count (row-major symmetric)
@group(0) @binding(2) var<storage, read_write> eig: array<f32>;  // 3 * count (eigenvalues)
@group(0) @binding(3) var<storage, read_write> proj: array<f32>; // 9 * count (PSD projection)

const SWEEPS: u32 = 12u;

fn jacobi_rotate(a: ptr<function, array<f32, 9>>, v: ptr<function, array<f32, 9>>, pp: u32, qq: u32) {
    let apq = (*a)[3u * pp + qq];
    if (abs(apq) < 1e-20) { return; }
    let app = (*a)[3u * pp + pp];
    let aqq = (*a)[3u * qq + qq];
    let phi = 0.5 * atan2(2.0 * apq, aqq - app);
    let c = cos(phi);
    let s = sin(phi);
    // Column rotation: a[:,p], a[:,q].
    for (var k = 0u; k < 3u; k = k + 1u) {
        let akp = (*a)[3u * k + pp];
        let akq = (*a)[3u * k + qq];
        (*a)[3u * k + pp] = c * akp - s * akq;
        (*a)[3u * k + qq] = s * akp + c * akq;
    }
    // Row rotation: a[p,:], a[q,:].
    for (var k = 0u; k < 3u; k = k + 1u) {
        let apk = (*a)[3u * pp + k];
        let aqk = (*a)[3u * qq + k];
        (*a)[3u * pp + k] = c * apk - s * aqk;
        (*a)[3u * qq + k] = s * apk + c * aqk;
    }
    // Accumulate eigenvectors: V = V·R.
    for (var k = 0u; k < 3u; k = k + 1u) {
        let vkp = (*v)[3u * k + pp];
        let vkq = (*v)[3u * k + qq];
        (*v)[3u * k + pp] = c * vkp - s * vkq;
        (*v)[3u * k + qq] = s * vkp + c * vkq;
    }
}

@compute @workgroup_size(64)
fn eig_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= p.count) { return; }
    var a: array<f32, 9>;
    var v: array<f32, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        a[i] = ain[9u * m + i];
        v[i] = 0.0;
    }
    v[0] = 1.0; v[4] = 1.0; v[8] = 1.0; // identity

    for (var sweep = 0u; sweep < SWEEPS; sweep = sweep + 1u) {
        jacobi_rotate(&a, &v, 0u, 1u);
        jacobi_rotate(&a, &v, 0u, 2u);
        jacobi_rotate(&a, &v, 1u, 2u);
    }

    let l0 = a[0];
    let l1 = a[4];
    let l2 = a[8];
    eig[3u * m] = l0;
    eig[3u * m + 1u] = l1;
    eig[3u * m + 2u] = l2;

    // proj = V · diag(max(λ,0)) · Vᵀ
    let c0 = max(l0, 0.0);
    let c1 = max(l1, 0.0);
    let c2 = max(l2, 0.0);
    for (var r = 0u; r < 3u; r = r + 1u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            let val = v[3u * r] * c0 * v[3u * c]
                + v[3u * r + 1u] * c1 * v[3u * c + 1u]
                + v[3u * r + 2u] * c2 * v[3u * c + 2u];
            proj[9u * m + 3u * r + c] = val;
        }
    }
}
