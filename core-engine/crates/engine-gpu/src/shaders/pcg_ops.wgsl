// SPDX-License-Identifier: GPL-2.0-or-later
// PCG building blocks. One entry point per op, dispatched as its own pipeline
// (auto bind-group layout). Every global has a UNIQUE binding index so no two
// resource variables alias the same @group/@binding — each pipeline's auto
// layout picks up only the bindings its entry point actually uses.
//
// The dot reduction uses a portable workgroup shared-memory tree (no subgroup
// ops) — correct on every backend; the subgroupAdd fast path is a later,
// profiled upgrade, and is blocked today by naga (see docs/WGPU_PIN.md).

struct P { n: u32, alpha: f32, beta: f32, _pad: f32 };

// ---- axpy: y = y + alpha * x -------------------------------------------------
@group(0) @binding(0) var<uniform> axpy_p: P;
@group(0) @binding(1) var<storage, read> axpy_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> axpy_y: array<f32>;
@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= axpy_p.n) { return; }
    axpy_y[i] = axpy_y[i] + axpy_p.alpha * axpy_x[i];
}

// ---- xpby: y = x + beta * y  (used for p = z + beta*p) -----------------------
@group(0) @binding(3) var<uniform> xpby_p: P;
@group(0) @binding(4) var<storage, read> xpby_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> xpby_y: array<f32>;
@compute @workgroup_size(64)
fn xpby(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= xpby_p.n) { return; }
    xpby_y[i] = xpby_x[i] + xpby_p.beta * xpby_y[i];
}

// ---- emul: out = a .* b  (z = r .* inv_diag, Jacobi precond) ------------------
@group(0) @binding(6) var<uniform> emul_p: P;
@group(0) @binding(7) var<storage, read> emul_a: array<f32>;
@group(0) @binding(8) var<storage, read> emul_b: array<f32>;
@group(0) @binding(9) var<storage, read_write> emul_out: array<f32>;
@compute @workgroup_size(64)
fn emul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= emul_p.n) { return; }
    emul_out[i] = emul_a[i] * emul_b[i];
}

// ---- dot: partials[wg] = sum over workgroup of a[i]*b[i] ----------------------
const WG: u32 = 256u;
@group(0) @binding(10) var<uniform> dot_p: P;
@group(0) @binding(11) var<storage, read> dot_a: array<f32>;
@group(0) @binding(12) var<storage, read> dot_b: array<f32>;
@group(0) @binding(13) var<storage, read_write> dot_partials: array<f32>;
var<workgroup> scratch: array<f32, 256>;
@compute @workgroup_size(256)
fn dot(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    var v = 0.0;
    if (i < dot_p.n) { v = dot_a[i] * dot_b[i]; }
    scratch[lid.x] = v;
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { dot_partials[wid.x] = scratch[0]; }
}

// ---- spmv: y = A x, A in CSR, one thread per row -----------------------------
@group(0) @binding(14) var<uniform> spmv_p: P;
@group(0) @binding(15) var<storage, read> spmv_row_ptr: array<u32>;
@group(0) @binding(16) var<storage, read> spmv_col_idx: array<u32>;
@group(0) @binding(17) var<storage, read> spmv_values: array<f32>;
@group(0) @binding(18) var<storage, read> spmv_x: array<f32>;
@group(0) @binding(19) var<storage, read_write> spmv_y: array<f32>;
@compute @workgroup_size(64)
fn spmv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= spmv_p.n) { return; }
    let start = spmv_row_ptr[row];
    let end = spmv_row_ptr[row + 1u];
    var sum = 0.0;
    for (var k = start; k < end; k = k + 1u) {
        sum = sum + spmv_values[k] * spmv_x[spmv_col_idx[k]];
    }
    spmv_y[row] = sum;
}
