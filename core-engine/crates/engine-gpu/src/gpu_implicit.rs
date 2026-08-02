// SPDX-License-Identifier: GPL-2.0-or-later
//! Fully GPU-resident implicit backward-Euler step.
//!
//! Positions, velocities, the assembled CSR values, the gradient and all PCG
//! scratch stay in device buffers across `assemble → Jacobi → PCG solve →
//! position update → velocity/floor`. The host only reads back the small PCG
//! decision scalars (alpha/beta/residual, in f64 — core WebGPU has no f64, so
//! the handful of scalars that need it come back to the host; see
//! `docs/CONFIG.md`) and, on demand, the positions. The division of labour:
//! CPU orchestrates control flow, the GPU owns the math and the data.
//!
//! Physics matches `engine_core::ImplicitSolver` exactly (same assembler, same
//! spring energy, same Newton form) and is verified against it on Metal.
//!
//! ponytail: the PCG loop is re-expressed here on resident buffers rather than
//! reusing `Pcg` (which owns its own values buffer and re-uploads a CsrMatrix).
//! The shared piece is the WGSL — `pcg_ops.wgsl` is reused verbatim. Fusing the
//! two solvers behind one buffer-owning core is a later cleanup.

use crate::GpuContext;
use engine_core::contact::Collider;
use engine_core::implicit::ImplicitParams;
use engine_core::{SimState, SpringAssembler};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GpuCollider {
    kind: u32, // 0 = half-space, 1 = sphere
    _pad: [u32; 3],
    a: [f32; 4], // half-space point.xyz ; sphere center.xyz + radius.w
    b: [f32; 4], // half-space normal.xyz
}

impl GpuCollider {
    fn from(c: &Collider) -> Self {
        match *c {
            Collider::HalfSpace { point, normal } => GpuCollider {
                kind: 0,
                _pad: [0; 3],
                a: [point.x, point.y, point.z, 0.0],
                b: [normal.x, normal.y, normal.z, 0.0],
            },
            Collider::Sphere { center, radius } => GpuCollider {
                kind: 1,
                _pad: [0; 3],
                a: [center.x, center.y, center.z, radius],
                b: [0.0; 4],
            },
        }
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Cp {
    n: u32,
    ncol: u32,
    dhat: f32,
    kappa: f32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Pcgp {
    n: u32,
    alpha: f32,
    beta: f32,
    _pad: f32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Asmp {
    n: u32,
    inv_h2: f32,
    k: f32,
    _pad: f32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Stepp {
    n: u32,
    h: f32,
    inv_h2: f32,
    floor: f32,
    gx: f32,
    gy: f32,
    gz: f32,
    has_floor: u32,
}

pub struct GpuImplicitSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: ImplicitParams,
    n: usize,
    wg_v: u32,   // workgroups over n vertices (@64)
    wg_d: u32,   // workgroups over 3n dofs (@64)
    wg_dot: u32, // workgroups over 3n dofs (@256)
    npart: u32,

    // pipelines
    assemble: wgpu::ComputePipeline,
    predict: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
    velupdate: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    xpby: wgpu::ComputePipeline,
    emul: wgpu::ComputePipeline,
    dot: wgpu::ComputePipeline,
    spmv: wgpu::ComputePipeline,
    collider: wgpu::ComputePipeline,

    // plan buffers (static)
    row_ptr: wgpu::Buffer,
    col_idx: wgpu::Buffer,
    mass: wgpu::Buffer,
    pinned: wgpu::Buffer,
    diag_off: wgpu::Buffer,
    adj_ptr: wgpu::Buffer,
    adj_j: wgpu::Buffer,
    adj_rest: wgpu::Buffer,
    adj_off: wgpu::Buffer,

    // state (resident)
    pos: wgpu::Buffer,
    vel: wgpu::Buffer,
    x0: wgpu::Buffer,
    x_tilde: wgpu::Buffer,

    // system + PCG scratch (resident)
    values: wgpu::Buffer,
    grad: wgpu::Buffer,
    inv_diag: wgpu::Buffer,
    sol: wgpu::Buffer,
    r: wgpu::Buffer,
    p: wgpu::Buffer,
    ap: wgpu::Buffer,
    zz: wgpu::Buffer,

    // uniforms
    u_asm: wgpu::Buffer,
    u_step: wgpu::Buffer,
    u_axpy: wgpu::Buffer,
    u_xpby: wgpu::Buffer,
    u_n: wgpu::Buffer, // emul/dot/spmv (only n used)
    u_cp: wgpu::Buffer,

    colliders: wgpu::Buffer,
    ncol: u32,

    partials: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl GpuImplicitSolver {
    pub fn new(ctx: &GpuContext, state: &SimState, params: ImplicitParams) -> Self {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let plan = SpringAssembler::new(state);
        let n = plan.n;
        let ndof = 3 * n;
        let nnz = plan.structure.nnz();
        let wg_v = (n as u32).max(1).div_ceil(64);
        let wg_d = (ndof as u32).max(1).div_ceil(64);
        let wg_dot = (ndof as u32).max(1).div_ceil(256);
        let npart = wg_dot;

        let pcg_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pcg_ops"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pcg_ops.wgsl").into()),
        });
        let asm_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("assemble"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/assemble.wgsl").into()),
        });
        let step_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_step.wgsl").into()),
        });
        let col_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collider_contact"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/collider_contact.wgsl").into()),
        });
        let pipe = |m: &wgpu::ShaderModule, e: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: None,
                module: m,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let stor = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let iu32 = |l: &str, d: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let if32 = |l: &str, d: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let ii32 = |l: &str, d: &[i32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let zeros = |l: &str, len: usize| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(l),
                contents: bytemuck::cast_slice(&vec![0.0f32; len.max(1)]),
                usage: stor,
            })
        };
        let uni = |l: &str, sz: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(l),
                size: sz as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Initial state upload.
        let flat =
            |v: &[glam::Vec3]| -> Vec<f32> { v.iter().flat_map(|p| [p.x, p.y, p.z]).collect() };
        let pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos"),
            contents: bytemuck::cast_slice(&flat(&state.positions)),
            usage: stor,
        });
        let vel = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vel"),
            contents: bytemuck::cast_slice(&flat(&state.velocities)),
            usage: stor,
        });

        Self {
            n,
            wg_v,
            wg_d,
            wg_dot,
            npart,
            params,
            assemble: pipe(&asm_mod, "assemble"),
            predict: pipe(&step_mod, "predict"),
            jacobi: pipe(&step_mod, "jacobi"),
            velupdate: pipe(&step_mod, "velupdate"),
            axpy: pipe(&pcg_mod, "axpy"),
            xpby: pipe(&pcg_mod, "xpby"),
            emul: pipe(&pcg_mod, "emul"),
            dot: pipe(&pcg_mod, "dot"),
            spmv: pipe(&pcg_mod, "spmv"),
            collider: pipe(&col_mod, "collider_contact"),
            row_ptr: iu32("row_ptr", plan.row_ptr()),
            col_idx: iu32("col_idx", plan.col_idx()),
            mass: if32("mass", &plan.mass),
            pinned: iu32("pinned", &plan.pinned_u32()),
            diag_off: ii32("diag_off", plan.diag_off()),
            adj_ptr: iu32("adj_ptr", plan.adj_ptr()),
            adj_j: iu32("adj_j", plan.adj_j()),
            adj_rest: if32("adj_rest", plan.adj_rest()),
            adj_off: ii32("adj_off", plan.adj_off()),
            pos,
            vel,
            x0: zeros("x0", ndof),
            x_tilde: zeros("x_tilde", ndof),
            values: zeros("values", nnz),
            grad: zeros("grad", ndof),
            inv_diag: zeros("inv_diag", ndof),
            sol: zeros("sol", ndof),
            r: zeros("r", ndof),
            p: zeros("p", ndof),
            ap: zeros("ap", ndof),
            zz: zeros("zz", ndof),
            u_asm: uni("u_asm", 16),
            u_step: uni("u_step", 32),
            u_axpy: uni("u_axpy", 16),
            u_xpby: uni("u_xpby", 16),
            u_n: uni("u_n", 16),
            u_cp: uni("u_cp", 16),
            // Dummy 1-collider buffer; replaced by set_colliders. ncol=0 skips it.
            colliders: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("colliders"),
                size: std::mem::size_of::<GpuCollider>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            ncol: 0,
            partials: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("partials"),
                size: (npart.max(1) as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: (npart.max(1) as u64) * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device,
            queue,
        }
    }

    /// Upload analytic colliders (run as a post-pass on the assembled system).
    pub fn set_colliders(&mut self, colliders: &[Collider]) {
        self.ncol = colliders.len() as u32;
        if colliders.is_empty() {
            return;
        }
        let encoded: Vec<GpuCollider> = colliders.iter().map(GpuCollider::from).collect();
        self.colliders = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("colliders"),
                contents: bytemuck::cast_slice(&encoded),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
    }

    fn bind(&self, pipe: &wgpu::ComputePipeline, e: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = e
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    fn run(&self, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, groups: u32) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn set_pcgp(&self, buf: &wgpu::Buffer, alpha: f32, beta: f32) {
        let p = Pcgp {
            n: (3 * self.n) as u32,
            alpha,
            beta,
            _pad: 0.0,
        };
        self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
    }

    fn dot_product(&self, a: &wgpu::Buffer, b: &wgpu::Buffer) -> f64 {
        let bg = self.bind(
            &self.dot,
            &[(10, &self.u_n), (11, a), (12, b), (13, &self.partials)],
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.dot);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.wg_dot, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.partials, 0, &self.staging, 0, (self.npart as u64) * 4);
        self.queue.submit(Some(enc.finish()));
        let slice = self.staging.slice(0..(self.npart as u64) * 4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let sum: f64 = bytemuck::cast_slice::<u8, f32>(&data)
            .iter()
            .map(|&v| v as f64)
            .sum();
        drop(data);
        self.staging.unmap();
        sum
    }

    fn copy(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, dst, 0, (3 * self.n as u64) * 4);
        self.queue.submit(Some(enc.finish()));
    }

    /// Advance one implicit step. All data stays resident; only PCG scalars are
    /// read back.
    pub fn step(&mut self) {
        let h = self.params.dt;
        let inv_h2 = 1.0 / (h * h);
        let g = self.params.gravity;
        let step = Stepp {
            n: self.n as u32,
            h,
            inv_h2,
            floor: self.params.floor_y,
            gx: g.x,
            gy: g.y,
            gz: g.z,
            has_floor: self.params.floor_y.is_finite() as u32,
        };
        self.queue
            .write_buffer(&self.u_step, 0, bytemuck::bytes_of(&step));
        self.queue.write_buffer(
            &self.u_asm,
            0,
            bytemuck::bytes_of(&Asmp {
                n: self.n as u32,
                inv_h2,
                k: self.params.stiffness,
                _pad: 0.0,
            }),
        );
        self.set_pcgp(&self.u_n, 0.0, 0.0);
        self.queue.write_buffer(
            &self.u_cp,
            0,
            bytemuck::bytes_of(&Cp {
                n: self.n as u32,
                ncol: self.ncol,
                dhat: self.params.contact_dhat,
                kappa: self.params.contact_kappa,
            }),
        );

        // predict: x0 = pos, x_tilde = inertial target
        let bg_pred = self.bind(
            &self.predict,
            &[
                (0, &self.u_step),
                (1, &self.pos),
                (2, &self.vel),
                (3, &self.x0),
                (4, &self.x_tilde),
                (5, &self.pinned),
            ],
        );
        self.run(&self.predict, &bg_pred, self.wg_v);

        for _ in 0..self.params.newton_iters.max(1) {
            self.assemble_step();
            if self.ncol > 0 {
                // Add collider barrier into values+grad (post-pass on the system).
                let bg = self.bind(
                    &self.collider,
                    &[
                        (0, &self.u_cp),
                        (1, &self.pos),
                        (2, &self.pinned),
                        (3, &self.diag_off),
                        (4, &self.colliders),
                        (5, &self.values),
                        (6, &self.grad),
                    ],
                );
                self.run(&self.collider, &bg, self.wg_v);
            }
            self.jacobi_step();
            self.pcg_solve();
            // pos -= sol
            self.set_pcgp(&self.u_axpy, -1.0, 0.0);
            let bg = self.bind(
                &self.axpy,
                &[(0, &self.u_axpy), (1, &self.sol), (2, &self.pos)],
            );
            self.run(&self.axpy, &bg, self.wg_d);
        }

        // velocity update + floor
        let bg_vu = self.bind(
            &self.velupdate,
            &[
                (11, &self.u_step),
                (12, &self.pos),
                (13, &self.x0),
                (14, &self.vel),
                (15, &self.pinned),
            ],
        );
        self.run(&self.velupdate, &bg_vu, self.wg_v);
    }

    fn assemble_step(&self) {
        // assemble reads pos (Newton iterate) as x, plus x_tilde → values, grad
        let bg = self.bind(
            &self.assemble,
            &[
                (0, &self.u_asm),
                (1, &self.pos),
                (2, &self.x_tilde),
                (3, &self.mass),
                (4, &self.pinned),
                (5, &self.diag_off),
                (6, &self.adj_ptr),
                (7, &self.adj_j),
                (8, &self.adj_rest),
                (9, &self.adj_off),
                (10, &self.values),
                (11, &self.grad),
            ],
        );
        self.run(&self.assemble, &bg, self.wg_v);
    }

    /// Jacobi inv_diag from the (final) values — run after the collider post-pass.
    fn jacobi_step(&self) {
        let bg_j = self.bind(
            &self.jacobi,
            &[
                (6, &self.u_step),
                (7, &self.values),
                (8, &self.diag_off),
                (9, &self.pinned),
                (10, &self.inv_diag),
            ],
        );
        self.run(&self.jacobi, &bg_j, self.wg_d);
    }

    /// Solve `H sol = grad` (so pos -= sol applies −H⁻¹grad). Resident PCG.
    fn pcg_solve(&self) {
        // sol = 0 ; r = grad ; zz = M⁻¹ r ; p = zz
        self.queue.write_buffer(
            &self.sol,
            0,
            bytemuck::cast_slice(&vec![0.0f32; 3 * self.n]),
        );
        self.copy(&self.grad, &self.r);
        let bg_emul = self.bind(
            &self.emul,
            &[
                (6, &self.u_n),
                (7, &self.r),
                (8, &self.inv_diag),
                (9, &self.zz),
            ],
        );
        self.run(&self.emul, &bg_emul, self.wg_d);
        self.copy(&self.zz, &self.p);

        let bnorm = self.dot_product(&self.grad, &self.grad).sqrt().max(1e-30);
        let mut rz = self.dot_product(&self.r, &self.zz);

        let bg_spmv = self.bind(
            &self.spmv,
            &[
                (14, &self.u_n),
                (15, &self.row_ptr),
                (16, &self.col_idx),
                (17, &self.values),
                (18, &self.p),
                (19, &self.ap),
            ],
        );
        let bg_axpy_sol = self.bind(
            &self.axpy,
            &[(0, &self.u_axpy), (1, &self.p), (2, &self.sol)],
        );
        let bg_axpy_r = self.bind(
            &self.axpy,
            &[(0, &self.u_axpy), (1, &self.ap), (2, &self.r)],
        );
        let bg_xpby = self.bind(
            &self.xpby,
            &[(3, &self.u_xpby), (4, &self.zz), (5, &self.p)],
        );

        for _ in 0..self.params.cg_max_iter {
            self.run(&self.spmv, &bg_spmv, self.wg_d);
            let pap = self.dot_product(&self.p, &self.ap);
            if pap <= 0.0 || pap.is_nan() {
                break;
            }
            let alpha = (rz / pap) as f32;
            self.set_pcgp(&self.u_axpy, alpha, 0.0);
            self.run(&self.axpy, &bg_axpy_sol, self.wg_d);
            self.set_pcgp(&self.u_axpy, -alpha, 0.0);
            self.run(&self.axpy, &bg_axpy_r, self.wg_d);
            self.run(&self.emul, &bg_emul, self.wg_d);
            let rz_new = self.dot_product(&self.r, &self.zz);
            let res = self.dot_product(&self.r, &self.r).sqrt();
            if res <= self.params.cg_tol as f64 * bnorm {
                break;
            }
            let beta = (rz_new / rz) as f32;
            self.set_pcgp(&self.u_xpby, 0.0, beta);
            self.run(&self.xpby, &bg_xpby, self.wg_d);
            rz = rz_new;
        }
    }

    /// Read the current positions back to the host.
    pub fn positions(&self) -> Vec<glam::Vec3> {
        let bytes = (3 * self.n as u64) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pos.read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.pos, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out = f
            .chunks_exact(3)
            .map(|c| glam::Vec3::new(c[0], c[1], c[2]))
            .collect();
        drop(data);
        staging.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::implicit::ImplicitSolver;
    use engine_core::linsolve::CgCpu;
    use engine_core::{Mesh, SimState};
    use glam::Vec3;

    #[test]
    fn gpu_resident_matches_cpu_implicit() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU-resident test, no adapter: {e}");
                return;
            }
        };
        let nn = 5;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                verts.push(Vec3::new(i as f32 * 0.1, 1.0, j as f32 * 0.1));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nn - 1 {
            for i in 0..nn - 1 {
                let v = (j * nn + i) as u32;
                let r = v + nn as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(verts, tris), 0.05);
        state.pin(0);
        state.pin(nn - 1);
        let params = ImplicitParams {
            cg_tol: 1e-7,
            ..Default::default()
        };

        let mut cpu = ImplicitSolver::new(state.clone(), params);
        let mut gpu = GpuImplicitSolver::new(&ctx, &state, params);
        for _ in 0..20 {
            cpu.step(&CgCpu::new());
            gpu.step();
        }
        let gpu_pos = gpu.positions();
        let max_err = cpu
            .state
            .positions
            .iter()
            .zip(&gpu_pos)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 2e-3,
            "GPU-resident vs CPU implicit drift: {max_err}"
        );
    }

    #[test]
    fn gpu_resident_contact_matches_cpu() {
        use engine_core::contact::Collider;
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU contact test, no adapter: {e}");
                return;
            }
        };
        let nn = 5;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                verts.push(Vec3::new(i as f32 * 0.1, 0.3, j as f32 * 0.1));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nn - 1 {
            for i in 0..nn - 1 {
                let v = (j * nn + i) as u32;
                let r = v + nn as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let state = SimState::from_mesh(&Mesh::new(verts, tris), 0.02);
        let params = ImplicitParams {
            dt: 1.0 / 240.0,
            stiffness: 500.0,
            contact_dhat: 0.02,
            contact_kappa: 1.0e3,
            cg_tol: 1e-7,
            ..Default::default()
        };
        let floor = Collider::HalfSpace {
            point: Vec3::ZERO,
            normal: Vec3::Y,
        };

        let mut cpu = ImplicitSolver::new(state.clone(), params);
        cpu.set_colliders(vec![floor]);
        let mut gpu = GpuImplicitSolver::new(&ctx, &state, params);
        gpu.set_colliders(&[floor]);
        for _ in 0..120 {
            cpu.step(&CgCpu::new());
            gpu.step();
        }
        let gpu_pos = gpu.positions();
        let max_err = cpu
            .state
            .positions
            .iter()
            .zip(&gpu_pos)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0f32, f32::max);
        assert!(max_err < 3e-3, "GPU vs CPU contact drift: {max_err}");
        assert!(
            gpu_pos.iter().all(|p| p.y > -1e-3),
            "GPU cloth penetrated floor"
        );
    }
}
