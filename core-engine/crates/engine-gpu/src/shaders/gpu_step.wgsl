// SPDX-License-Identifier: GPL-2.0-or-later
// Physics kernels for the GPU-resident implicit step: inertial prediction,
// Jacobi diagonal extraction, and velocity update + floor projection. Combined
// with assemble.wgsl (Hessian) and pcg_ops.wgsl (linear solve), the whole
// backward-Euler step runs on-device — the host only reads back PCG scalars and
// (per frame) positions. Unique binding indices per entry so one module's auto
// layouts never alias.

struct SP {
    n: u32,
    h: f32,
    inv_h2: f32,
    floor: f32,
    gx: f32,
    gy: f32,
    gz: f32,
    has_floor: u32,
};

// ---- predict: x0 = pos ; x_tilde = pos + h*vel + h²*g (free) or pos (pinned) --
@group(0) @binding(0) var<uniform> pr_sp: SP;
@group(0) @binding(1) var<storage, read> pr_pos: array<f32>;
@group(0) @binding(2) var<storage, read> pr_vel: array<f32>;
@group(0) @binding(3) var<storage, read_write> pr_x0: array<f32>;
@group(0) @binding(4) var<storage, read_write> pr_xt: array<f32>;
@group(0) @binding(5) var<storage, read> pr_pinned: array<u32>;
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pr_sp.n) { return; }
    let b = 3u * i;
    for (var c = 0u; c < 3u; c = c + 1u) {
        pr_x0[b + c] = pr_pos[b + c];
    }
    if (pr_pinned[i] != 0u) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            pr_xt[b + c] = pr_pos[b + c];
        }
    } else {
        let hh = pr_sp.h * pr_sp.h;
        pr_xt[b] = pr_pos[b] + pr_sp.h * pr_vel[b] + hh * pr_sp.gx;
        pr_xt[b + 1u] = pr_pos[b + 1u] + pr_sp.h * pr_vel[b + 1u] + hh * pr_sp.gy;
        pr_xt[b + 2u] = pr_pos[b + 2u] + pr_sp.h * pr_vel[b + 2u] + hh * pr_sp.gz;
    }
}

// ---- jacobi: inv_diag[d] = 1 / A[d,d], read via the assembler's diag offsets --
@group(0) @binding(6) var<uniform> jc_sp: SP;
@group(0) @binding(7) var<storage, read> jc_values: array<f32>;
@group(0) @binding(8) var<storage, read> jc_diag_off: array<i32>;
@group(0) @binding(9) var<storage, read> jc_pinned: array<u32>;
@group(0) @binding(10) var<storage, read_write> jc_inv_diag: array<f32>;
@compute @workgroup_size(64)
fn jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= 3u * jc_sp.n) { return; }
    let i = d / 3u;
    let r = d % 3u;
    // Free rows store a 3-run at diag_off[d] (col 3i); the (d,d) entry is +r in.
    // Pinned rows store the single (d,d) entry at diag_off[d].
    var entry = jc_diag_off[d];
    if (jc_pinned[i] == 0u) {
        entry = entry + i32(r);
    }
    let v = jc_values[u32(entry)];
    jc_inv_diag[d] = select(1.0, 1.0 / v, v != 0.0);
}

// ---- velupdate: floor-clamp pos, then vel = (pos - x0)/h (free); vel=0 pinned -
@group(0) @binding(11) var<uniform> vu_sp: SP;
@group(0) @binding(12) var<storage, read_write> vu_pos: array<f32>;
@group(0) @binding(13) var<storage, read> vu_x0: array<f32>;
@group(0) @binding(14) var<storage, read_write> vu_vel: array<f32>;
@group(0) @binding(15) var<storage, read> vu_pinned: array<u32>;
@compute @workgroup_size(64)
fn velupdate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vu_sp.n) { return; }
    let b = 3u * i;
    if (vu_pinned[i] != 0u) {
        vu_vel[b] = 0.0;
        vu_vel[b + 1u] = 0.0;
        vu_vel[b + 2u] = 0.0;
        return;
    }
    if (vu_sp.has_floor != 0u && vu_pos[b + 1u] < vu_sp.floor) {
        vu_pos[b + 1u] = vu_sp.floor;
    }
    let inv_h = 1.0 / vu_sp.h;
    vu_vel[b] = (vu_pos[b] - vu_x0[b]) * inv_h;
    vu_vel[b + 1u] = (vu_pos[b + 1u] - vu_x0[b + 1u]) * inv_h;
    vu_vel[b + 2u] = (vu_pos[b + 2u] - vu_x0[b + 2u]) * inv_h;
}
