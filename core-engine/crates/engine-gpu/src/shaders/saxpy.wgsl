// SPDX-License-Identifier: GPL-2.0-or-later
// Trivial device-bring-up kernel: y = a*x + y (SAXPY). Proves the full compute
// path (upload -> dispatch -> readback) works on the selected backend. The real
// PCG vector kernels (axpy, dot, apply_H) replace/extend this at the implicit
// milestone — see docs/KERNELS.md.

struct Params { a: f32, n: u32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    y[i] = params.a * x[i] + y[i];
}
