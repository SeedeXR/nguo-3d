// SPDX-License-Identifier: GPL-2.0-or-later
//! Atomic-free **gather** assembler for the Baraff-Witkin membrane Hessian +
//! gradient — the triangle analogue of [`crate::assembler::SpringAssembler`].
//!
//! Sparsity is fixed by the mesh topology, so it is built once; each solve only
//! recomputes the values. Assembly is per-vertex-row: worker `i` owns vertex
//! `i`'s CSR rows and gathers the contributions of `i`'s incident triangles into
//! its own entries. Unlike a spring (one edge → one off-diagonal), a membrane
//! *edge* is shared by up to two triangles, so an off-diagonal block accumulates
//! — but only worker `i` ever writes row `i`, so it accumulates sequentially with
//! NO atomics (see docs/DEVELOPING.md §6). The per-triangle element blocks come from
//! [`membrane::element_assembly`] (the shared oracle); the GPU port
//! (`engine-gpu::GpuMembrane` compute + a gather pass) mirrors this one-to-one.

#![allow(clippy::needless_range_loop)]

use crate::csr::CsrMatrix;
use crate::gather3::Gather3;
use crate::membrane::{self, MembraneElement};
use glam::Vec3;

/// Membrane (Baraff-Witkin) assembler — a thin wrapper over the shared 3-vertex
/// [`Gather3`] topology plus the membrane element oracle
/// [`membrane::element_assembly`]. The index bookkeeping lives in `Gather3` (the
/// isotropic shells reuse the identical machinery); only the element math differs.
pub struct MembraneAssembler {
    pub n: usize,
    pub ndof: usize,
    elements: Vec<MembraneElement>,
    g: Gather3,
}

impl MembraneAssembler {
    pub fn new(elements: &[MembraneElement], n: usize, pinned: &[bool]) -> Self {
        let elems: Vec<[u32; 3]> = elements.iter().map(|e| e.tri).collect();
        let g = Gather3::new(&elems, n, pinned);
        Self {
            n: g.n,
            ndof: g.ndof,
            elements: elements.to_vec(),
            g,
        }
    }

    /// Assemble the membrane CSR values + gradient at positions `x`. Only the
    /// membrane contribution; pinned rows get an identity diagonal.
    pub fn assemble(&self, x: &[Vec3], kw: f32, kf: f32, ks: f32) -> (Vec<f32>, Vec<f32>) {
        self.g.assemble(|t| {
            let el = &self.elements[t];
            let p = [
                x[el.tri[0] as usize],
                x[el.tri[1] as usize],
                x[el.tri[2] as usize],
            ];
            membrane::element_assembly(&p, &el.dm_inv, el.area, kw, kf, ks)
        })
    }

    /// The CSR sparsity the assembled values match.
    pub fn structure(&self) -> &CsrMatrix {
        &self.g.structure
    }

    // ---- flat accessors for the GPU gather port (delegated) ----
    pub fn row_ptr(&self) -> &[u32] {
        self.g.row_ptr()
    }
    pub fn col_idx(&self) -> &[u32] {
        self.g.col_idx()
    }
    pub fn diag_off(&self) -> &[i32] {
        self.g.diag_off()
    }
    pub fn vt_ptr(&self) -> &[u32] {
        self.g.vt_ptr()
    }
    pub fn vt_elem(&self) -> &[u32] {
        self.g.vt_elem()
    }
    pub fn vt_local(&self) -> &[u32] {
        self.g.vt_local()
    }
    pub fn vt_off(&self) -> &[i32] {
        self.g.vt_off()
    }
    pub fn pinned_u32(&self) -> Vec<u32> {
        self.g.pinned_u32()
    }
    pub fn elements(&self) -> &[MembraneElement] {
        &self.elements
    }
    pub fn nnz(&self) -> usize {
        self.g.nnz()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::membrane::build_membrane;

    /// Reference: direct triplet assembly of the membrane element Hessians (the
    /// same math the `ImplicitSolver` membrane path uses), Dirichlet-filtered.
    fn reference(
        els: &[MembraneElement],
        x: &[Vec3],
        n: usize,
        pinned: &[bool],
        kw: f32,
        kf: f32,
        ks: f32,
    ) -> CsrMatrix {
        let mut trips = Vec::new();
        for el in els {
            let p = [
                x[el.tri[0] as usize],
                x[el.tri[1] as usize],
                x[el.tri[2] as usize],
            ];
            let (_g, h) = membrane::element_assembly(&p, &el.dm_inv, el.area, kw, kf, ks);
            for a in 0..3 {
                let va = el.tri[a] as usize;
                for b in 0..3 {
                    let vb = el.tri[b] as usize;
                    for r in 0..3 {
                        for c in 0..3 {
                            trips.push((
                                (3 * va + r) as u32,
                                (3 * vb + c) as u32,
                                h[(3 * a + r) * 9 + (3 * b + c)],
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
            }
        }
        CsrMatrix::from_triplets(3 * n, &trips)
    }

    #[test]
    fn membrane_gather_matches_triplet_reference() {
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
        let n = verts.len();
        let els = build_membrane(&tris, &verts);
        // Pin two corners; deform.
        let mut pinned = vec![false; n];
        pinned[0] = true;
        pinned[(nn - 1) as usize] = true;
        let x: Vec<Vec3> = verts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                *p + Vec3::new(
                    (i as f32 * 0.3).sin() * 0.03,
                    (i as f32 * 0.2).cos() * 0.02,
                    0.04 * ((i % 3) as f32 - 1.0),
                )
            })
            .collect();
        let (kw, kf, ks) = (0.8f32, 0.5, 0.3);

        let asm = MembraneAssembler::new(&els, n, &pinned);
        let (values, grad) = asm.assemble(&x, kw, kf, ks);
        let reference = reference(&els, &x, n, &pinned, kw, kf, ks);

        assert_eq!(
            asm.structure().row_ptr,
            reference.row_ptr,
            "row_ptr differs"
        );
        assert_eq!(
            asm.structure().col_idx,
            reference.col_idx,
            "col_idx differs"
        );
        let mut vmax = 0.0f32;
        for (a, b) in values.iter().zip(&reference.values) {
            vmax = vmax.max((a - b).abs());
        }
        assert!(vmax < 1e-3, "membrane gather value drift {vmax}");

        // Gradient: compare to a direct sum of element gradients (free rows).
        let mut gref = vec![0.0f32; 3 * n];
        for el in &els {
            let p = [
                x[el.tri[0] as usize],
                x[el.tri[1] as usize],
                x[el.tri[2] as usize],
            ];
            let (g, _) = membrane::element_assembly(&p, &el.dm_inv, el.area, kw, kf, ks);
            for a in 0..3 {
                let va = el.tri[a] as usize;
                if !pinned[va] {
                    for r in 0..3 {
                        gref[3 * va + r] += g[3 * a + r];
                    }
                }
            }
        }
        let mut gmax = 0.0f32;
        for (a, b) in grad.iter().zip(&gref) {
            gmax = gmax.max((a - b).abs());
        }
        assert!(gmax < 1e-4, "membrane gather grad drift {gmax}");
    }
}
