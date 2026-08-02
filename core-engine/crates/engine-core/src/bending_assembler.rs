// SPDX-License-Identifier: GPL-2.0-or-later
//! Atomic-free **gather** assembler for the discrete-shell bending Hessian +
//! gradient — the 4-vertex-hinge analogue of [`crate::membrane_assembler::MembraneAssembler`].
//!
//! Sparsity is fixed by the mesh topology (which hinge vertices couple), so it is
//! built once; each solve recomputes only the values. Assembly is per-vertex-row:
//! worker `i` owns vertex `i`'s CSR rows and gathers the contributions of `i`'s
//! incident hinges into its own entries — a vertex touches several hinges, so
//! off-diagonal blocks accumulate, but only worker `i` writes row `i`, so NO
//! atomics. The per-hinge element blocks come from [`crate::bending::hinge_element`]
//! (the shared oracle); the GPU port (`engine-gpu::GpuBendingAssembler`) mirrors
//! this one-to-one, exactly as the membrane gather assembler does.

#![allow(clippy::needless_range_loop)]

use crate::bending::{self, Hinge};
use crate::csr::CsrMatrix;
use crate::gather4::Gather4;
use glam::Vec3;

/// Bending (hinge) assembler — a thin wrapper over the shared 4-vertex
/// [`Gather4`] topology plus the hinge element oracle
/// [`crate::bending::hinge_element`]. The index bookkeeping lives in `Gather4`
/// (tets reuse the identical machinery); only the element math differs.
pub struct BendingAssembler {
    pub n: usize,
    pub ndof: usize,
    hinges: Vec<Hinge>,
    g: Gather4,
}

impl BendingAssembler {
    pub fn new(hinges: &[Hinge], n: usize, pinned: &[bool]) -> Self {
        let elems: Vec<[u32; 4]> = hinges.iter().map(|h| h.v).collect();
        let g = Gather4::new(&elems, n, pinned);
        Self {
            n: g.n,
            ndof: g.ndof,
            hinges: hinges.to_vec(),
            g,
        }
    }

    /// Assemble the bending CSR values + gradient at positions `x` for stiffness
    /// `kb`. Returns `(values matching `self.structure`, grad)`; pinned rows get an
    /// identity diagonal.
    pub fn assemble(&self, x: &[Vec3], kb: f32) -> (Vec<f32>, Vec<f32>) {
        self.g.assemble(|t| {
            let h = &self.hinges[t];
            let p = [
                x[h.v[0] as usize],
                x[h.v[1] as usize],
                x[h.v[2] as usize],
                x[h.v[3] as usize],
            ];
            bending::hinge_element(&p, h.rest, h.w, kb)
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
    pub fn hinges(&self) -> &[Hinge] {
        &self.hinges
    }
    pub fn nnz(&self) -> usize {
        self.g.nnz()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bending::build_hinges;

    /// Reference: direct triplet assembly of the hinge element Hessians (the same
    /// math `ImplicitSolver`'s bending path uses), Dirichlet-filtered.
    fn reference(hinges: &[Hinge], x: &[Vec3], n: usize, pinned: &[bool], kb: f32) -> CsrMatrix {
        let mut trips = Vec::new();
        for h in hinges {
            let p = [
                x[h.v[0] as usize],
                x[h.v[1] as usize],
                x[h.v[2] as usize],
                x[h.v[3] as usize],
            ];
            let (_g, hess) = bending::hinge_element(&p, h.rest, h.w, kb);
            for a in 0..4 {
                let va = h.v[a] as usize;
                for b in 0..4 {
                    let vb = h.v[b] as usize;
                    for r in 0..3 {
                        for c in 0..3 {
                            trips.push((
                                (3 * va + r) as u32,
                                (3 * vb + c) as u32,
                                hess[(3 * a + r) * 12 + (3 * b + c)],
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
    fn bending_gather_matches_triplet_reference() {
        // A folded grid → interior hinges with nonzero angles.
        let nn = 5u32;
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                let a = 0.25 * j as f32;
                verts.push(Vec3::new(
                    i as f32 * 0.1,
                    a.cos() * j as f32 * 0.1,
                    a.sin() * j as f32 * 0.1,
                ));
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
        let hinges = build_hinges(&tris, &verts);
        assert!(hinges.len() > 5);
        let mut pinned = vec![false; n];
        pinned[0] = true;
        pinned[(nn - 1) as usize] = true;
        // Deform away from rest.
        let x: Vec<Vec3> = verts
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::new(0.0, 0.0, 0.02 * (i as f32).sin()))
            .collect();
        let kb = 0.7f32;

        let asm = BendingAssembler::new(&hinges, n, &pinned);
        let (values, grad) = asm.assemble(&x, kb);
        let reference = reference(&hinges, &x, n, &pinned, kb);

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
        assert!(vmax < 1e-4, "bending gather value drift {vmax}");

        // Gradient vs a direct sum of element gradients (free rows only).
        let mut gref = vec![0.0f32; 3 * n];
        for h in &hinges {
            let p = [
                x[h.v[0] as usize],
                x[h.v[1] as usize],
                x[h.v[2] as usize],
                x[h.v[3] as usize],
            ];
            let (g, _) = bending::hinge_element(&p, h.rest, h.w, kb);
            for a in 0..4 {
                let va = h.v[a] as usize;
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
        assert!(gmax < 1e-4, "bending gather grad drift {gmax}");
    }
}
