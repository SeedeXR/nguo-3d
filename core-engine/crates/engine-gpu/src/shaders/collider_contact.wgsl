// SPDX-License-Identifier: GPL-2.0-or-later
// Analytic-collider IPC barrier as a post-pass on the assembled system: per free
// vertex, add the collider barrier gradient + PSD Hessian into `grad` and the
// vertex's diagonal block of `values` (via the assembler's diag offsets). Runs
// after assemble.wgsl; each vertex owns its own rows → no atomics. Mirrors
// engine_core::contact::vertex_contact exactly (verified GPU==CPU).
//
// BOUND BY: engine-gpu/src/gpu_implicit.rs, sharing assemble.wgsl's `diag_off`
// from engine_core::SpringAssembler, where -1 is impossible — so the bare
// `u32(diag_off[...])` below is unguarded by design (see assemble.wgsl's header).

struct CP { n: u32, ncol: u32, dhat: f32, kappa: f32 };

struct Collider {
    kind: u32,   // 0 = half-space, 1 = sphere
    _p0: u32,
    _p1: u32,
    _p2: u32,
    a: vec4<f32>, // half-space: point.xyz ; sphere: center.xyz, radius=w
    b: vec4<f32>, // half-space: normal.xyz
};

@group(0) @binding(0) var<uniform> cp: CP;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> pinned: array<u32>;
@group(0) @binding(3) var<storage, read> diag_off: array<i32>;
@group(0) @binding(4) var<storage, read> colliders: array<Collider>;
@group(0) @binding(5) var<storage, read_write> values: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad: array<f32>;

@compute @workgroup_size(64)
fn collider_contact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cp.n || pinned[i] != 0u) { return; }
    let xi = vec3<f32>(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u]);

    for (var ci = 0u; ci < cp.ncol; ci = ci + 1u) {
        let c = colliders[ci];
        var d: f32;
        var na = array<f32, 3>(0.0, 0.0, 0.0);
        if (c.kind == 0u) {
            let nrm = c.b.xyz;
            d = dot(xi - c.a.xyz, nrm);
            na = array<f32, 3>(nrm.x, nrm.y, nrm.z);
        } else {
            let v = xi - c.a.xyz;
            let l = length(v);
            if (l < 1e-9) {
                d = -c.a.w;
                na = array<f32, 3>(0.0, 1.0, 0.0);
            } else {
                d = l - c.a.w;
                na = array<f32, 3>(v.x / l, v.y / l, v.z / l);
            }
        }
        if (d < cp.dhat) {
            let dd = max(d, 1e-6);
            let t = dd - cp.dhat;
            let ld = log(dd / cp.dhat);
            let g = -cp.kappa * (2.0 * t * ld + t * t / dd);
            let h = -cp.kappa * (2.0 * ld + 4.0 * t / dd - (t * t) / (dd * dd));
            let hp = max(h, 0.0);
            grad[3u * i] = grad[3u * i] + g * na[0];
            grad[3u * i + 1u] = grad[3u * i + 1u] + g * na[1];
            grad[3u * i + 2u] = grad[3u * i + 2u] + g * na[2];
            for (var r = 0u; r < 3u; r = r + 1u) {
                let base = u32(diag_off[3u * i + r]);
                values[base] = values[base] + hp * na[r] * na[0];
                values[base + 1u] = values[base + 1u] + hp * na[r] * na[1];
                values[base + 2u] = values[base + 2u] + hp * na[r] * na[2];
            }
        }
    }
}
