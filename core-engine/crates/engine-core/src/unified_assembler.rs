// SPDX-License-Identifier: GPL-2.0-or-later
//! Unified atomic-free **gather** assembler: inertia + springs + Baraff-Witkin
//! membrane into ONE CSR — the union of [`crate::assembler::SpringAssembler`] and
//! [`crate::membrane_assembler::MembraneAssembler`]. The sparsity is the union of
//! spring-edge blocks, membrane triangle blocks, and per-vertex diagonals; a
//! shared edge (spring == triangle edge) is ONE entry that BOTH accumulate into.
//! Worker `i` owns row `i` and gathers all of `i`'s inertia + spring + membrane
//! contributions into its own entries — no atomics. This is the CPU oracle for a
//! fully GPU-resident membrane step (the GPU stitches the same offsets); verified
//! bit-close to `ImplicitSolver::assemble` (inertia+springs+membrane, no contact).

#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

use crate::csr::CsrMatrix;
use crate::implicit::spring_block;
use crate::membrane::{self, MembraneElement};
use crate::shell_model::{self, ShellElement};
use crate::state::SimState;
use glam::Vec3;

/// Precomputed union sparsity + offsets for inertia + springs + membrane.
pub struct UnifiedAssembler {
    pub n: usize,
    pub ndof: usize,
    pub structure: CsrMatrix,
    pub pinned: Vec<bool>,
    pub mass: Vec<f32>,
    diag_off: Vec<i32>,
    // springs (per-vertex incident): neighbor, rest length, 3 off-diag row offsets.
    s_ptr: Vec<u32>,
    s_j: Vec<u32>,
    s_rest: Vec<f32>,
    s_off: Vec<i32>,
    // membrane (per-vertex incident triangles): element, local a, 9 block offsets.
    elements: Vec<MembraneElement>,
    /// Isotropic SHELL elements, occupying element indices `elements.len()..` in the
    /// SAME element-index space as the membrane elements. Sharing one index space is
    /// what lets the GPU gather stay byte-identical: both element types produce the same
    /// per-triangle `(9 grad, 81 hess)` layout, so two compute passes can fill one
    /// `egrad`/`ehess` buffer at different offsets and the gather never learns there are
    /// two kinds. The alternative — a second element stream in the shader — would have
    /// needed ~6 more storage buffers, which risks the WebGPU/Vulkan per-stage limits.
    shells: Vec<ShellElement>,
    m_ptr: Vec<u32>,
    m_elem: Vec<u32>,
    m_local: Vec<u32>,
    m_off: Vec<i32>,
}

fn find_col(s: &CsrMatrix, row: usize, col: u32) -> i32 {
    let (start, end) = (s.row_ptr[row] as usize, s.row_ptr[row + 1] as usize);
    for k in start..end {
        if s.col_idx[k] == col {
            return k as i32;
        }
    }
    -1
}

impl UnifiedAssembler {
    pub fn new(state: &SimState, elements: &[MembraneElement]) -> Self {
        Self::with_shells(state, elements, &[])
    }

    /// As [`Self::new`], plus isotropic shell elements appended to the element-index
    /// space (see the `shells` field).
    pub fn with_shells(
        state: &SimState,
        elements: &[MembraneElement],
        shells: &[ShellElement],
    ) -> Self {
        let n = state.vertex_count();
        let ndof = 3 * n;
        let pinned: Vec<bool> = (0..n).map(|i| state.inv_mass[i] <= 0.0).collect();
        let mass: Vec<f32> = (0..n)
            .map(|i| {
                if pinned[i] {
                    0.0
                } else {
                    1.0 / state.inv_mass[i]
                }
            })
            .collect();

        // Union sparsity: diagonals + spring blocks + membrane triangle blocks.
        let mut trips: Vec<crate::csr::Triplet> = Vec::new();
        for i in 0..n {
            if pinned[i] {
                for c in 0..3 {
                    trips.push(((3 * i + c) as u32, (3 * i + c) as u32, 0.0));
                }
            } else {
                for r in 0..3 {
                    for c in 0..3 {
                        trips.push(((3 * i + r) as u32, (3 * i + c) as u32, 0.0));
                    }
                }
            }
        }
        let push_block = |a: usize, b: usize, trips: &mut Vec<crate::csr::Triplet>| {
            if !pinned[a] && !pinned[b] {
                for r in 0..3 {
                    for c in 0..3 {
                        trips.push(((3 * a + r) as u32, (3 * b + c) as u32, 0.0));
                    }
                }
            }
        };
        for s in &state.stretch {
            let (a, b) = (s.a as usize, s.b as usize);
            push_block(a, b, &mut trips);
            push_block(b, a, &mut trips);
        }
        for el in elements {
            for a in 0..3 {
                for b in 0..3 {
                    push_block(el.tri[a] as usize, el.tri[b] as usize, &mut trips);
                }
            }
        }
        let structure = CsrMatrix::from_triplets(ndof, &trips);

        // Diagonal offsets.
        let mut diag_off = vec![-1i32; ndof];
        for i in 0..n {
            if pinned[i] {
                for c in 0..3 {
                    diag_off[3 * i + c] = find_col(&structure, 3 * i + c, (3 * i + c) as u32);
                }
            } else {
                for r in 0..3 {
                    diag_off[3 * i + r] = find_col(&structure, 3 * i + r, (3 * i) as u32);
                }
            }
        }
        // Never `-1`: the loop above pushes a diagonal block for every vertex before
        // any element is seen, and `from_triplets` never prunes a column. The free
        // path of `unified_gather.wgsl` casts this unguarded — see
        // `assembler.rs::SpringAssembler::new` for the full argument.
        debug_assert!(
            diag_off.iter().all(|&o| o >= 0),
            "diag_off holds -1: the structure lost a diagonal block; \
             unified_gather.wgsl indexes it unguarded and would store out of bounds"
        );

        // Spring incidence + off-diagonal offsets.
        let mut s_inc: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
        for s in &state.stretch {
            s_inc[s.a as usize].push((s.b, s.rest_len));
            s_inc[s.b as usize].push((s.a, s.rest_len));
        }
        let mut s_ptr = vec![0u32; n + 1];
        let (mut s_j, mut s_rest, mut s_off) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            for &(j, rest) in &s_inc[i] {
                s_j.push(j);
                s_rest.push(rest);
                for r in 0..3 {
                    if !pinned[i] && !pinned[j as usize] {
                        s_off.push(find_col(&structure, 3 * i + r, 3 * j));
                    } else {
                        s_off.push(-1);
                    }
                }
            }
            s_ptr[i + 1] = s_j.len() as u32;
        }

        // Membrane incidence + block offsets.
        // Triples of BOTH element kinds, membrane first then shells, in one index space.
        let triples: Vec<[u32; 3]> = elements
            .iter()
            .map(|el| el.tri)
            .chain(shells.iter().map(|el| el.v))
            .collect();
        let mut m_inc: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for (t, tri) in triples.iter().enumerate() {
            for a in 0..3 {
                m_inc[tri[a] as usize].push((t as u32, a as u32));
            }
        }
        let mut m_ptr = vec![0u32; n + 1];
        let (mut m_elem, mut m_local, mut m_off) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            for &(t, a) in &m_inc[i] {
                m_elem.push(t);
                m_local.push(a);
                let tri = &triples[t as usize];
                for b in 0..3 {
                    let vb = tri[b] as usize;
                    for r in 0..3 {
                        if !pinned[i] && !pinned[vb] {
                            m_off.push(find_col(&structure, 3 * i + r, (3 * vb) as u32));
                        } else {
                            m_off.push(-1);
                        }
                    }
                }
            }
            m_ptr[i + 1] = m_elem.len() as u32;
        }

        Self {
            n,
            ndof,
            structure,
            pinned,
            mass,
            diag_off,
            s_ptr,
            s_j,
            s_rest,
            s_off,
            elements: elements.to_vec(),
            shells: shells.to_vec(),
            m_ptr,
            m_elem,
            m_local,
            m_off,
        }
    }

    /// Assemble `(values matching structure, grad)` for inertia + springs +
    /// membrane at `x`. `rhs = -grad`. `k` = spring stiffness; `(kw,kf,ks)` =
    /// membrane warp/weft/shear.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        x: &[Vec3],
        x_tilde: &[Vec3],
        inv_h2: f32,
        k: f32,
        kw: f32,
        kf: f32,
        ks: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        self.assemble_with_shells(x, x_tilde, inv_h2, k, kw, kf, ks, 0.0, 0.0, 1.0e-6)
    }

    /// As [`Self::assemble`], with the shell elements evaluated too. `(shell_mu,
    /// shell_lambda)` of zero leaves them inert, so [`Self::assemble`] delegating here is
    /// bit-identical for a membrane-only assembler.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_with_shells(
        &self,
        x: &[Vec3],
        x_tilde: &[Vec3],
        inv_h2: f32,
        k: f32,
        kw: f32,
        kf: f32,
        ks: f32,
        shell_mu: f32,
        shell_lambda: f32,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        // Per-triangle element blocks, membrane first then shells — the same order the
        // element-index space was built in, and the same order the GPU's two compute
        // passes fill one buffer.
        let elem: Vec<([f32; 9], [f32; 81])> = self
            .elements
            .iter()
            .map(|el| {
                let p = [
                    x[el.tri[0] as usize],
                    x[el.tri[1] as usize],
                    x[el.tri[2] as usize],
                ];
                membrane::element_assembly(&p, &el.dm_inv, el.area, kw, kf, ks)
            })
            .chain(self.shells.iter().map(|el| {
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                ];
                let g = shell_model::shell_gradient_analytic(&p, el, shell_mu, shell_lambda);
                let h = shell_model::shell_hessian_analytic(&p, el, shell_mu, shell_lambda, eps);
                let mut g9 = [0.0f32; 9];
                for (a, gv) in g.iter().enumerate() {
                    g9[3 * a] = gv.x;
                    g9[3 * a + 1] = gv.y;
                    g9[3 * a + 2] = gv.z;
                }
                (g9, h)
            }))
            .collect();

        let mut values = vec![0.0f32; self.structure.nnz()];
        let mut grad = vec![0.0f32; self.ndof];
        for i in 0..self.n {
            if self.pinned[i] {
                for r in 0..3 {
                    values[self.diag_off[3 * i + r] as usize] = 1.0;
                }
                continue;
            }
            // Diagonal block accumulated in a register (inertia + spring + membrane
            // block(a,a)); off-diagonals accumulate directly into `values` (a shared
            // spring/membrane edge lands on ONE entry, both `+=`).
            let mut diag = [[0.0f32; 3]; 3];
            let mh2 = self.mass[i] * inv_h2;
            for c in 0..3 {
                diag[c][c] += mh2;
            }
            let mut gi = (x[i] - x_tilde[i]) * mh2;

            let (ss, se) = (self.s_ptr[i] as usize, self.s_ptr[i + 1] as usize);
            for e in ss..se {
                let j = self.s_j[e] as usize;
                let (hk, ga) = spring_block(x[i] - x[j], self.s_rest[e], k);
                for r in 0..3 {
                    for c in 0..3 {
                        diag[r][c] += hk[r][c];
                    }
                }
                gi += ga;
                if self.s_off[3 * e] >= 0 {
                    for r in 0..3 {
                        let base = self.s_off[3 * e + r] as usize;
                        for c in 0..3 {
                            values[base + c] += -hk[r][c];
                        }
                    }
                }
            }

            let (ms, me) = (self.m_ptr[i] as usize, self.m_ptr[i + 1] as usize);
            for kk in ms..me {
                let a = self.m_local[kk] as usize;
                let (eg, eh) = &elem[self.m_elem[kk] as usize];
                for r in 0..3 {
                    gi[r] += eg[3 * a + r];
                }
                for b in 0..3 {
                    for r in 0..3 {
                        for c in 0..3 {
                            let v = eh[(3 * a + r) * 9 + (3 * b + c)];
                            if b == a {
                                diag[r][c] += v; // vertex's own block → diagonal
                            } else {
                                let off = self.m_off[9 * kk + 3 * b + r];
                                if off >= 0 {
                                    values[off as usize + c] += v;
                                }
                            }
                        }
                    }
                }
            }

            for r in 0..3 {
                let base = self.diag_off[3 * i + r] as usize;
                for c in 0..3 {
                    values[base + c] = diag[r][c];
                }
                grad[3 * i + r] = gi[r];
            }
        }
        (values, grad)
    }

    // ---- flat accessors for the GPU port (GpuUnified uploads these) ----
    pub fn nnz(&self) -> usize {
        self.structure.nnz()
    }
    pub fn elements(&self) -> &[MembraneElement] {
        &self.elements
    }

    /// The shell elements, occupying element indices `elements().len()..`.
    pub fn shells(&self) -> &[ShellElement] {
        &self.shells
    }

    /// Total triangle elements across both kinds — the element count the GPU
    /// `egrad`/`ehess` buffers must hold.
    pub fn element_count(&self) -> usize {
        self.elements.len() + self.shells.len()
    }
    pub fn pinned_u32(&self) -> Vec<u32> {
        self.pinned.iter().map(|&p| p as u32).collect()
    }
    pub fn mass(&self) -> &[f32] {
        &self.mass
    }
    pub fn diag_off(&self) -> &[i32] {
        &self.diag_off
    }
    pub fn s_ptr(&self) -> &[u32] {
        &self.s_ptr
    }
    pub fn s_j(&self) -> &[u32] {
        &self.s_j
    }
    pub fn s_rest(&self) -> &[f32] {
        &self.s_rest
    }
    pub fn s_off(&self) -> &[i32] {
        &self.s_off
    }
    pub fn m_ptr(&self) -> &[u32] {
        &self.m_ptr
    }
    pub fn m_elem(&self) -> &[u32] {
        &self.m_elem
    }
    pub fn m_local(&self) -> &[u32] {
        &self.m_local
    }
    pub fn m_off(&self) -> &[i32] {
        &self.m_off
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Mesh;
    use crate::membrane::{build_membrane, element_assembly};

    /// Triplet reference: inertia + springs + membrane, Dirichlet-filtered — the
    /// same math `ImplicitSolver::assemble` uses (minus contact/bending).
    #[allow(clippy::too_many_arguments, clippy::manual_memcpy)]
    fn reference(
        state: &SimState,
        els: &[MembraneElement],
        x: &[Vec3],
        x_tilde: &[Vec3],
        inv_h2: f32,
        k: f32,
        kw: f32,
        kf: f32,
        ks: f32,
    ) -> (CsrMatrix, Vec<f32>) {
        let n = state.vertex_count();
        let ndof = 3 * n;
        let pinned: Vec<bool> = (0..n).map(|i| state.inv_mass[i] <= 0.0).collect();
        let mut trips = Vec::new();
        let mut grad = vec![0.0f32; ndof];
        for i in 0..n {
            if pinned[i] {
                continue;
            }
            let mh2 = (1.0 / state.inv_mass[i]) * inv_h2;
            for c in 0..3 {
                trips.push(((3 * i + c) as u32, (3 * i + c) as u32, mh2));
                grad[3 * i + c] += mh2 * (x[i][c] - x_tilde[i][c]);
            }
        }
        for s in &state.stretch {
            let (a, b) = (s.a as usize, s.b as usize);
            let (hk, ga) = spring_block(x[a] - x[b], s.rest_len, k);
            for r in 0..3 {
                for c in 0..3 {
                    let v = hk[r][c];
                    trips.push(((3 * a + r) as u32, (3 * a + c) as u32, v));
                    trips.push(((3 * b + r) as u32, (3 * b + c) as u32, v));
                    trips.push(((3 * a + r) as u32, (3 * b + c) as u32, -v));
                    trips.push(((3 * b + r) as u32, (3 * a + c) as u32, -v));
                }
                if !pinned[a] {
                    grad[3 * a + r] += ga[r];
                }
                if !pinned[b] {
                    grad[3 * b + r] -= ga[r];
                }
            }
        }
        for el in els {
            let p = [
                x[el.tri[0] as usize],
                x[el.tri[1] as usize],
                x[el.tri[2] as usize],
            ];
            let (eg, eh) = element_assembly(&p, &el.dm_inv, el.area, kw, kf, ks);
            for a in 0..3 {
                let va = el.tri[a] as usize;
                if !pinned[va] {
                    for r in 0..3 {
                        grad[3 * va + r] += eg[3 * a + r];
                    }
                }
                for b in 0..3 {
                    let vb = el.tri[b] as usize;
                    for r in 0..3 {
                        for c in 0..3 {
                            trips.push((
                                (3 * va + r) as u32,
                                (3 * vb + c) as u32,
                                eh[(3 * a + r) * 9 + (3 * b + c)],
                            ));
                        }
                    }
                }
            }
        }
        trips.retain(|&(r, c, _)| !pinned[r as usize / 3] && !pinned[c as usize / 3]);
        for i in 0..n {
            if pinned[i] {
                for c in 0..3 {
                    trips.push(((3 * i + c) as u32, (3 * i + c) as u32, 1.0));
                }
                for c in 0..3 {
                    grad[3 * i + c] = 0.0;
                }
            }
        }
        (CsrMatrix::from_triplets(ndof, &trips), grad)
    }

    #[test]
    fn unified_matches_triplet_reference() {
        let nn = 4u32;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                verts.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nn - 1 {
            for i in 0..nn - 1 {
                let v = j * nn + i;
                tris.push([v, v + 1, v + nn]);
                tris.push([v + nn, v + 1, v + nn + 1]);
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(verts.clone(), tris.clone()), 0.1);
        state.pin(0);
        state.pin((nn - 1) as usize);
        let els = build_membrane(&tris, &verts);
        let x: Vec<Vec3> = verts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                *p + Vec3::new(
                    (i as f32 * 0.3).sin() * 0.03,
                    -0.02,
                    0.04 * ((i % 3) as f32 - 1.0),
                )
            })
            .collect();
        let x_tilde: Vec<Vec3> = x.iter().map(|p| *p + Vec3::new(0.0, -0.001, 0.0)).collect();
        let (inv_h2, k, kw, kf, ks) = (1.0e4, 1000.0, 0.8, 0.5, 0.3);

        let asm = UnifiedAssembler::new(&state, &els);
        let (values, grad) = asm.assemble(&x, &x_tilde, inv_h2, k, kw, kf, ks);
        let (rmat, rgrad) = reference(&state, &els, &x, &x_tilde, inv_h2, k, kw, kf, ks);

        assert_eq!(asm.structure.row_ptr, rmat.row_ptr, "row_ptr differs");
        assert_eq!(asm.structure.col_idx, rmat.col_idx, "col_idx differs");
        let vmax = values
            .iter()
            .zip(&rmat.values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let gmax = grad
            .iter()
            .zip(&rgrad)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(vmax < 1e-2, "unified values drift {vmax}");
        assert!(gmax < 1e-3, "unified grad drift {gmax}");
    }
}

#[cfg(test)]
mod shell_stream_tests {
    use super::*;
    use crate::shell_model::build_shells;
    use crate::tet::TetModel;
    use crate::Mesh;

    fn sheet(n: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut p = Vec::new();
        for j in 0..n {
            for i in 0..n {
                p.push(Vec3::new(i as f32 * 0.3, 0.0, j as f32 * 0.3));
            }
        }
        let mut t = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                t.push([v, v + 1, r]);
                t.push([r, v + 1, r + 1]);
            }
        }
        (p, t)
    }

    /// Shells appended to the element-index space must land at the right CSR offsets.
    /// Verified two independent ways rather than against the assembler's own arithmetic:
    /// the GRADIENT against direct per-element summation (which catches any element→vertex
    /// mapping error), and the MATRIX via a matvec against direct element-wise application
    /// (which catches a wrong column offset, invisible to the gradient check).
    #[test]
    fn shell_blocks_land_at_the_right_offsets() {
        let (rest, tris) = sheet(5);
        let deformed: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let f = i as f32;
                Vec3::new(
                    p.x * 1.12 + 0.03 * (f * 0.4).sin(),
                    0.08 * (f * 0.5).sin(),
                    p.z * 0.93 + 0.02 * (f * 0.8).cos(),
                )
            })
            .collect();
        let shells = build_shells(&tris, &rest, 1.0e-12, TetModel::Arap);
        assert!(shells.len() >= 20);
        let state = SimState::from_mesh(&Mesh::new(rest.clone(), tris.clone()), 0.25);
        // Shells only: no membrane elements, so every element block is a shell block and a
        // misplacement cannot be masked by a membrane contribution at the same entry.
        let ua = UnifiedAssembler::with_shells(&state, &[], &shells);
        assert_eq!(ua.element_count(), shells.len());

        let (mu, lambda, eps) = (150.0f32, 90.0f32, 1.0e-6f32);
        let inv_h2 = 3600.0f32;
        let x_tilde: Vec<Vec3> = deformed
            .iter()
            .map(|p| *p + Vec3::new(0.0, -0.01, 0.0))
            .collect();
        // Springs off (k = 0) so the comparison isolates the shell stream.
        let (values, grad) = ua.assemble_with_shells(
            &deformed, &x_tilde, inv_h2, 0.0, 0.0, 0.0, 0.0, mu, lambda, eps,
        );

        // --- gradient, independently summed ---
        let n = rest.len();
        let mut want_g = vec![0.0f32; 3 * n];
        for i in 0..n {
            let mh2 = ua.mass()[i] * inv_h2;
            let d = deformed[i] - x_tilde[i];
            for c in 0..3 {
                want_g[3 * i + c] += mh2 * d[c];
            }
        }
        for el in &shells {
            let p = [
                deformed[el.v[0] as usize],
                deformed[el.v[1] as usize],
                deformed[el.v[2] as usize],
            ];
            let g = shell_model::shell_gradient_analytic(&p, el, mu, lambda);
            for (a, gv) in g.iter().enumerate() {
                for c in 0..3 {
                    want_g[3 * el.v[a] as usize + c] += gv[c];
                }
            }
        }
        let (mut worst, mut scale) = (0.0f32, 0.0f32);
        for i in 0..3 * n {
            worst = worst.max((grad[i] - want_g[i]).abs());
            scale = scale.max(want_g[i].abs());
        }
        assert!(scale > 0.0, "degenerate test");
        assert!(
            worst / scale < 1.0e-4,
            "gradient mismatch: worst {worst:.3e} on scale {scale:.3e}"
        );

        // --- matrix, via matvec against direct element-wise application ---
        let mut a = ua.structure.clone();
        a.values.copy_from_slice(&values);
        let mut v = vec![0.0f32; 3 * n];
        let mut seed = 0x1234_5678u32;
        for slot in v.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *slot = ((seed >> 9) as f32 / 4.194304e6) - 1.0;
        }
        let got = a.spmv_cpu(&v);
        let mut want = vec![0.0f32; 3 * n];
        for i in 0..n {
            let mh2 = ua.mass()[i] * inv_h2;
            for c in 0..3 {
                want[3 * i + c] += mh2 * v[3 * i + c];
            }
        }
        for el in &shells {
            let p = [
                deformed[el.v[0] as usize],
                deformed[el.v[1] as usize],
                deformed[el.v[2] as usize],
            ];
            let h = shell_model::shell_hessian_analytic(&p, el, mu, lambda, eps);
            for a_ in 0..3 {
                for b in 0..3 {
                    for r in 0..3 {
                        for c in 0..3 {
                            want[3 * el.v[a_] as usize + r] +=
                                h[(3 * a_ + r) * 9 + (3 * b + c)] * v[3 * el.v[b] as usize + c];
                        }
                    }
                }
            }
        }
        let (mut mworst, mut mscale) = (0.0f32, 0.0f32);
        for i in 0..3 * n {
            mworst = mworst.max((got[i] - want[i]).abs());
            mscale = mscale.max(want[i].abs());
        }
        assert!(mscale > 0.0, "degenerate matvec test");
        assert!(
            mworst / mscale < 1.0e-4,
            "matvec mismatch: worst {mworst:.3e} on scale {mscale:.3e}"
        );
    }

    /// A membrane-only assembler must be BIT-IDENTICAL through the new code path — the
    /// shell stream is purely additive.
    #[test]
    fn membrane_only_is_unchanged_by_the_shell_stream() {
        use crate::membrane::build_membrane;
        let (rest, tris) = sheet(4);
        let els = build_membrane(&tris, &rest);
        let state = SimState::from_mesh(&Mesh::new(rest.clone(), tris.clone()), 0.2);
        let a = UnifiedAssembler::new(&state, &els);
        let b = UnifiedAssembler::with_shells(&state, &els, &[]);
        let xt: Vec<Vec3> = rest
            .iter()
            .map(|p| *p + Vec3::new(0.0, -0.02, 0.0))
            .collect();
        let ra = a.assemble(&rest, &xt, 3600.0, 10.0, 50.0, 50.0, 25.0);
        let rb = b.assemble(&rest, &xt, 3600.0, 10.0, 50.0, 50.0, 25.0);
        assert_eq!(ra.0, rb.0, "values must be identical");
        assert_eq!(ra.1, rb.1, "grad must be identical");
    }
}
