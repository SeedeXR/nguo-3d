// SPDX-License-Identifier: GPL-2.0-or-later
// GPU-resident PCG scalar machinery: final reduction of dot-product partials
// into a device `scalars` buffer, and on-device alpha/beta computation, so the
// CPU no longer reads back a scalar every iteration (the measured bottleneck —
// each readback is a map/poll GPU sync). The host only reads the residual
// every K iterations for the early-out. Unique bindings per entry (one module).
//
// scalars layout: [0]=rz [1]=pAp [2]=rz_new [3]=alpha [4]=-alpha [5]=beta [6]=rr

struct RF { npart: u32, slot: u32, _a: u32, _b: u32 };
struct SL { n: u32, slot: u32, _a: u32, _b: u32 };

// ---- reduce_final: sum `partials` (grid-stride) into scalars[slot] -----------
@group(0) @binding(0) var<uniform> rf: RF;
@group(0) @binding(1) var<storage, read> rf_partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> rf_scalars: array<f32>;
var<workgroup> rf_scratch: array<f32, 256>;
@compute @workgroup_size(256)
fn reduce_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    var s = 0.0;
    var i = lid.x;
    loop {
        if (i >= rf.npart) { break; }
        s = s + rf_partials[i];
        i = i + 256u;
    }
    rf_scratch[lid.x] = s;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { rf_scratch[lid.x] = rf_scratch[lid.x] + rf_scratch[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { rf_scalars[rf.slot] = rf_scratch[0]; }
}

// ---- compute_alpha: alpha = rz/pAp (0 on non-SPD), and -alpha ----------------
@group(0) @binding(3) var<storage, read_write> a_scalars: array<f32>;
@compute @workgroup_size(1)
fn compute_alpha() {
    let pap = a_scalars[1];
    let alpha = select(0.0, a_scalars[0] / pap, pap > 0.0);
    a_scalars[3] = alpha;
    a_scalars[4] = -alpha;
}

// ---- compute_beta: beta = rz_new/rz, then roll rz := rz_new ------------------
@group(0) @binding(4) var<storage, read_write> b_scalars: array<f32>;
@compute @workgroup_size(1)
fn compute_beta() {
    let rz = b_scalars[0];
    let rznew = b_scalars[2];
    b_scalars[5] = select(0.0, rznew / rz, rz != 0.0);
    b_scalars[0] = rznew;
}

// ---- axpy_s: y += scalars[slot] * x ------------------------------------------
@group(0) @binding(5) var<uniform> ax_p: SL;
@group(0) @binding(6) var<storage, read> ax_scalars: array<f32>;
@group(0) @binding(7) var<storage, read> ax_x: array<f32>;
@group(0) @binding(8) var<storage, read_write> ax_y: array<f32>;
@compute @workgroup_size(64)
fn axpy_s(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ax_p.n) { return; }
    ax_y[i] = ax_y[i] + ax_scalars[ax_p.slot] * ax_x[i];
}

// ---- bmul: z_i = Binv_i · r_i  (3×3 per-vertex block-Jacobi precond apply) ----
struct BM { nv: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(13) var<uniform> bm: BM;
@group(0) @binding(14) var<storage, read> bm_binv: array<f32>; // nv × 9, row-major
@group(0) @binding(15) var<storage, read> bm_r: array<f32>;
@group(0) @binding(16) var<storage, read_write> bm_z: array<f32>;
@compute @workgroup_size(64)
fn bmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bm.nv) { return; }
    let o = 9u * i;
    let d = 3u * i;
    let r0 = bm_r[d];
    let r1 = bm_r[d + 1u];
    let r2 = bm_r[d + 2u];
    bm_z[d] = bm_binv[o] * r0 + bm_binv[o + 1u] * r1 + bm_binv[o + 2u] * r2;
    bm_z[d + 1u] = bm_binv[o + 3u] * r0 + bm_binv[o + 4u] * r1 + bm_binv[o + 5u] * r2;
    bm_z[d + 2u] = bm_binv[o + 6u] * r0 + bm_binv[o + 7u] * r1 + bm_binv[o + 8u] * r2;
}

// ---- xpby_s: y = x + scalars[slot] * y ---------------------------------------
@group(0) @binding(9) var<uniform> xp_p: SL;
@group(0) @binding(10) var<storage, read> xp_scalars: array<f32>;
@group(0) @binding(11) var<storage, read> xp_x: array<f32>;
@group(0) @binding(12) var<storage, read_write> xp_y: array<f32>;
@compute @workgroup_size(64)
fn xpby_s(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= xp_p.n) { return; }
    xp_y[i] = xp_x[i] + xp_scalars[xp_p.slot] * xp_y[i];
}
