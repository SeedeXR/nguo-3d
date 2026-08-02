// SPDX-License-Identifier: GPL-2.0-or-later
//! Reusable **atomic-free 4-vertex gather** topology: the CSR sparsity + value
//! offsets shared by every element type whose stencil is four vertices — bending
//! hinges ([`crate::bending_assembler`]) and tetrahedra ([`crate::tet_assembler`]).
//!
//! Sparsity is fixed by the mesh topology, so it is built once; each solve only
//! recomputes values. Assembly is per-vertex-row: worker `i` owns vertex `i`'s CSR
//! rows and gathers the contributions of `i`'s incident elements into its own
//! entries. A vertex belongs to several elements, so off-diagonal blocks accumulate
//! — but only worker `i` ever writes row `i`, so NO atomics (see docs/DEVELOPING.md §6). The
//! GPU gather kernels (`engine-gpu::shaders::bending_gather`) mirror this exactly.
//!
//! ponytail: this was extracted from `BendingAssembler` once a second 4-vertex
//! element (tets) needed the identical index bookkeeping — the fiddly `find_col`
//! offset math now lives in one place instead of two. The element-specific part is
//! injected as a closure returning the 12-gradient + 144-Hessian block.

#![allow(clippy::needless_range_loop)]

use crate::csr::CsrMatrix;

/// Per-element output the gather consumes: the 12-dof gradient and the row-major
/// 12×12 Hessian block, in the element's own vertex order.
pub type Elem4 = ([f32; 12], [f32; 144]);

/// Precomputed sparsity + value offsets for a fixed set of 4-vertex elements.
pub struct Gather4 {
    pub n: usize,
    pub ndof: usize,
    pub structure: CsrMatrix,
    pub pinned: Vec<bool>,
    /// Per-vertex incident-element adjacency (CSR-style): (element, local a ∈ 0..4).
    vt_ptr: Vec<u32>,
    vt_elem: Vec<u32>,
    vt_local: Vec<u32>,
    /// 12 offsets per incident entry: for local `b∈0..4`, `r∈0..3`, the start of the
    /// 3-run for block `(i, elem[b])` in row `3i+r`, or `-1` if pinned (block
    /// dropped by the Dirichlet filter). Index `12·e + 3·b + r`.
    vt_off: Vec<i32>,
    /// Diagonal offsets (free: start of the 3-run at col `3i`; pinned: single entry).
    diag_off: Vec<i32>,
    /// Number of elements the sparsity was built from.
    n_elems: usize,
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

impl Gather4 {
    /// Build the sparsity for `elems` (each an array of 4 vertex indices).
    pub fn new(elems: &[[u32; 4]], n: usize, pinned: &[bool]) -> Self {
        let ndof = 3 * n;
        let pinned = pinned.to_vec();

        // Diagonal block per free vertex (identity entry per pinned DOF) + every
        // free-free vertex-pair block within each element.
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
        for e in elems {
            for a in 0..4 {
                let va = e[a] as usize;
                if pinned[va] {
                    continue;
                }
                for b in 0..4 {
                    let vb = e[b] as usize;
                    if pinned[vb] {
                        continue;
                    }
                    for r in 0..3 {
                        for c in 0..3 {
                            trips.push(((3 * va + r) as u32, (3 * vb + c) as u32, 0.0));
                        }
                    }
                }
            }
        }
        let structure = CsrMatrix::from_triplets(ndof, &trips);

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

        let mut incident: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for (t, e) in elems.iter().enumerate() {
            for a in 0..4 {
                incident[e[a] as usize].push((t as u32, a as u32));
            }
        }
        let mut vt_ptr = vec![0u32; n + 1];
        let mut vt_elem = Vec::new();
        let mut vt_local = Vec::new();
        let mut vt_off = Vec::new();
        for i in 0..n {
            for &(t, a) in &incident[i] {
                vt_elem.push(t);
                vt_local.push(a);
                let e = &elems[t as usize];
                for b in 0..4 {
                    let vb = e[b] as usize;
                    for r in 0..3 {
                        if !pinned[i] && !pinned[vb] {
                            vt_off.push(find_col(&structure, 3 * i + r, (3 * vb) as u32));
                        } else {
                            vt_off.push(-1);
                        }
                    }
                }
            }
            vt_ptr[i + 1] = vt_elem.len() as u32;
        }

        Self {
            n,
            ndof,
            structure,
            pinned,
            n_elems: elems.len(),
            vt_ptr,
            vt_elem,
            vt_local,
            vt_off,
            diag_off,
        }
    }

    /// Gather element blocks into CSR values + gradient. `element(t)` supplies
    /// element `t`'s 12-gradient and 12×12 Hessian. Pinned rows get an identity
    /// diagonal. Element blocks are evaluated once each, then scattered.
    pub fn assemble(&self, element: impl Fn(usize) -> Elem4) -> (Vec<f32>, Vec<f32>) {
        let elem: Vec<Elem4> = (0..self.n_elems).map(&element).collect();

        let mut values = vec![0.0f32; self.structure.nnz()];
        let mut grad = vec![0.0f32; self.ndof];
        for i in 0..self.n {
            if self.pinned[i] {
                for r in 0..3 {
                    values[self.diag_off[3 * i + r] as usize] = 1.0;
                }
                continue;
            }
            let (s, e) = (self.vt_ptr[i] as usize, self.vt_ptr[i + 1] as usize);
            for k in s..e {
                let t = self.vt_elem[k] as usize;
                let a = self.vt_local[k] as usize;
                let (egrad, ehess) = &elem[t];
                for r in 0..3 {
                    grad[3 * i + r] += egrad[3 * a + r];
                }
                for b in 0..4 {
                    for r in 0..3 {
                        let off = self.vt_off[12 * k + 3 * b + r];
                        if off >= 0 {
                            let base = off as usize;
                            for c in 0..3 {
                                values[base + c] += ehess[(3 * a + r) * 12 + (3 * b + c)];
                            }
                        }
                    }
                }
            }
        }
        (values, grad)
    }

    // ---- flat accessors for the GPU gather port ----
    pub fn row_ptr(&self) -> &[u32] {
        &self.structure.row_ptr
    }
    pub fn col_idx(&self) -> &[u32] {
        &self.structure.col_idx
    }
    pub fn diag_off(&self) -> &[i32] {
        &self.diag_off
    }
    pub fn vt_ptr(&self) -> &[u32] {
        &self.vt_ptr
    }
    pub fn vt_elem(&self) -> &[u32] {
        &self.vt_elem
    }
    pub fn vt_local(&self) -> &[u32] {
        &self.vt_local
    }
    pub fn vt_off(&self) -> &[i32] {
        &self.vt_off
    }
    pub fn pinned_u32(&self) -> Vec<u32> {
        self.pinned.iter().map(|&p| p as u32).collect()
    }
    pub fn nnz(&self) -> usize {
        self.structure.nnz()
    }
    pub fn n_elems(&self) -> usize {
        self.n_elems
    }
}
