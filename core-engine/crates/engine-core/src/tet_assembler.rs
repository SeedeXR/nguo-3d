// SPDX-License-Identifier: GPL-2.0-or-later
//! Atomic-free gather assembler for the volumetric (tet) FEM Hessian + gradient.
//!
//! Tets are 4-vertex elements, exactly like bending hinges, so this reuses the
//! shared [`Gather4`] topology verbatim — only the element math differs
//! ([`crate::tet::tet_element_assembly`] instead of `bending::hinge_element`).
//! That is the whole point of the `Gather4` extraction: the fiddly CSR offset
//! bookkeeping is written and verified once.

use crate::csr::CsrMatrix;
use crate::gather4::Gather4;
use crate::tet::{self, TetElement};
use glam::Vec3;

/// Volumetric FEM assembler over a fixed tet topology.
pub struct TetAssembler {
    pub n: usize,
    pub ndof: usize,
    tets: Vec<TetElement>,
    g: Gather4,
}

impl TetAssembler {
    pub fn new(tets: &[TetElement], n: usize, pinned: &[bool]) -> Self {
        let elems: Vec<[u32; 4]> = tets.iter().map(|t| t.v).collect();
        let g = Gather4::new(&elems, n, pinned);
        Self {
            n: g.n,
            ndof: g.ndof,
            tets: tets.to_vec(),
            g,
        }
    }

    /// Assemble the tet CSR values + gradient at positions `x` for Lamé-style
    /// parameters `mu`/`lambda`. Pinned rows get an identity diagonal.
    pub fn assemble(&self, x: &[Vec3], mu: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
        self.g.assemble(|i| {
            let el = &self.tets[i];
            let p = [
                x[el.v[0] as usize],
                x[el.v[1] as usize],
                x[el.v[2] as usize],
                x[el.v[3] as usize],
            ];
            let (_e, grad, hess) = tet::tet_element_assembly(&p, el, mu, lambda);
            let mut g12 = [0.0f32; 12];
            for (a, gv) in grad.iter().enumerate() {
                g12[3 * a] = gv.x;
                g12[3 * a + 1] = gv.y;
                g12[3 * a + 2] = gv.z;
            }
            (g12, hess)
        })
    }

    /// Total elastic energy of all tets at `x` (useful for line search / diagnostics).
    pub fn energy(&self, x: &[Vec3], mu: f32, lambda: f32) -> f32 {
        self.tets
            .iter()
            .map(|el| {
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                    x[el.v[3] as usize],
                ];
                tet::tet_energy(&p, el, mu, lambda)
            })
            .sum()
    }

    /// The CSR sparsity the assembled values match.
    pub fn structure(&self) -> &CsrMatrix {
        &self.g.structure
    }
    pub fn tets(&self) -> &[TetElement] {
        &self.tets
    }
    pub fn nnz(&self) -> usize {
        self.g.nnz()
    }
    // ---- flat accessors for a future GPU gather port ----
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tet::{build_tets, TetModel};

    /// A 2×2×2 grid of cubes, each split into 5 tets — a real multi-element mesh
    /// where vertices are shared, so off-diagonal blocks genuinely accumulate.
    fn tet_block(nn: usize) -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
        let mut verts = Vec::new();
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    verts.push(Vec3::new(i as f32 * 0.5, j as f32 * 0.5, k as f32 * 0.5));
                }
            }
        }
        let mut tets = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
                    // 5-tet decomposition of the cube.
                    let (a, b, c, d) = (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i, j + 1, k),
                        idx(i + 1, j + 1, k),
                    );
                    let (e, f, g, h) = (
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i, j + 1, k + 1),
                        idx(i + 1, j + 1, k + 1),
                    );
                    tets.push([a, b, c, e]);
                    tets.push([b, d, c, h]);
                    tets.push([b, c, e, h]);
                    tets.push([c, e, g, h]);
                    tets.push([b, e, f, h]);
                }
            }
        }
        (verts, tets)
    }

    /// Reference: direct triplet assembly of the tet element Hessians,
    /// Dirichlet-filtered — the same independent oracle the bending and membrane
    /// assemblers are checked against.
    fn reference(
        tets: &[TetElement],
        x: &[Vec3],
        n: usize,
        pinned: &[bool],
        mu: f32,
        lambda: f32,
    ) -> CsrMatrix {
        let mut trips = Vec::new();
        for el in tets {
            let p = [
                x[el.v[0] as usize],
                x[el.v[1] as usize],
                x[el.v[2] as usize],
                x[el.v[3] as usize],
            ];
            let (_e, _g, hess) = tet::tet_element_assembly(&p, el, mu, lambda);
            for a in 0..4 {
                let va = el.v[a] as usize;
                for b in 0..4 {
                    let vb = el.v[b] as usize;
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
        for (i, &is_pinned) in pinned.iter().enumerate().take(n) {
            if is_pinned {
                for c in 0..3 {
                    trips.push(((3 * i + c) as u32, (3 * i + c) as u32, 1.0));
                }
            }
        }
        CsrMatrix::from_triplets(3 * n, &trips)
    }

    #[test]
    fn tet_gather_matches_triplet_reference() {
        let (rest, tet_idx) = tet_block(3);
        let n = rest.len();
        let els = build_tets(&tet_idx, &rest, 1e-9, TetModel::StVK);
        assert!(
            els.len() > 20,
            "expected a multi-element mesh, got {}",
            els.len()
        );
        // Pin one face; deform the rest.
        let mut pinned = vec![false; n];
        for (i, p) in rest.iter().enumerate() {
            if p.y == 0.0 {
                pinned[i] = true;
            }
        }
        let x: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| {
                *p + Vec3::new(
                    0.01 * (i as f32 * 0.7).sin(),
                    0.02 * (i as f32 * 0.3).cos(),
                    0.015 * (i as f32 * 0.5).sin(),
                )
            })
            .collect();
        let (mu, lambda) = (50.0f32, 100.0f32);

        let asm = TetAssembler::new(&els, n, &pinned);
        let (values, grad) = asm.assemble(&x, mu, lambda);
        let reference = reference(&els, &x, n, &pinned, mu, lambda);

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
        let mut vscale = 1.0f32;
        for (a, b) in values.iter().zip(&reference.values) {
            vmax = vmax.max((a - b).abs());
            vscale = vscale.max(b.abs());
        }
        assert!(
            vmax < 1e-3 * vscale,
            "tet gather value drift {vmax} (scale {vscale})"
        );

        // Gradient vs a direct sum of element gradients over free rows.
        let mut gref = vec![0.0f32; 3 * n];
        for el in &els {
            let p = [
                x[el.v[0] as usize],
                x[el.v[1] as usize],
                x[el.v[2] as usize],
                x[el.v[3] as usize],
            ];
            let (_e, g, _h) = tet::tet_element_assembly(&p, el, mu, lambda);
            for (a, gv) in g.iter().enumerate() {
                let va = el.v[a] as usize;
                if !pinned[va] {
                    for r in 0..3 {
                        gref[3 * va + r] += gv[r];
                    }
                }
            }
        }
        let mut gmax = 0.0f32;
        let mut gscale = 1.0f32;
        for (a, b) in grad.iter().zip(&gref) {
            gmax = gmax.max((a - b).abs());
            gscale = gscale.max(b.abs());
        }
        assert!(gmax < 1e-3 * gscale, "tet gather grad drift {gmax}");
    }

    /// The assembled matrix must be symmetric and PSD (every element block is
    /// PSD-projected and the gather is a congruence-style scatter), so CG can use it.
    #[test]
    fn assembled_tet_matrix_is_symmetric_psd() {
        let (rest, tet_idx) = tet_block(3);
        let n = rest.len();
        let els = build_tets(&tet_idx, &rest, 1e-9, TetModel::StVK);
        let pinned = vec![false; n];
        let x: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::splat(0.01 * (i as f32).sin()))
            .collect();
        let asm = TetAssembler::new(&els, n, &pinned);
        let (values, _g) = asm.assemble(&x, 50.0, 100.0);
        let mut m = asm.structure().clone();
        m.values = values;

        // Symmetry: A x · y == x · A y for random probes.
        let probe = |seed: u64| -> Vec<f32> {
            (0..3 * n)
                .map(|k| {
                    ((seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add((k as u64).wrapping_mul(1442695040888963407))
                        >> 40) as f32
                        / 8388608.0)
                        - 1.0
                })
                .collect()
        };
        for s in 0..4u64 {
            let (xv, yv) = (probe(s), probe(s + 100));
            let ax = m.spmv_cpu(&xv);
            let ay = m.spmv_cpu(&yv);
            let lhs: f32 = ax.iter().zip(&yv).map(|(a, b)| a * b).sum();
            let rhs: f32 = xv.iter().zip(&ay).map(|(a, b)| a * b).sum();
            assert!(
                (lhs - rhs).abs() < 1e-2 * lhs.abs().max(1.0),
                "not symmetric: {lhs} vs {rhs}"
            );
            // PSD: xᵀAx ≥ 0.
            let q: f32 = ax.iter().zip(&xv).map(|(a, b)| a * b).sum();
            assert!(q >= -1e-2, "assembled tet matrix not PSD: xᵀAx = {q}");
        }
    }

    /// At rest the tet energy, gradient, and (therefore) the assembled gradient are
    /// zero — no spurious forces on an undeformed volumetric mesh.
    #[test]
    fn rest_configuration_assembles_zero_gradient() {
        let (rest, tet_idx) = tet_block(3);
        let n = rest.len();
        let els = build_tets(&tet_idx, &rest, 1e-9, TetModel::StVK);
        let pinned = vec![false; n];
        let asm = TetAssembler::new(&els, n, &pinned);
        assert!(
            (asm.energy(&rest, 50.0, 100.0)).abs() < 1e-4,
            "rest energy nonzero"
        );
        let (_v, grad) = asm.assemble(&rest, 50.0, 100.0);
        let gmax = grad.iter().fold(0.0f32, |m, g| m.max(g.abs()));
        assert!(gmax < 1e-3, "rest gradient nonzero: {gmax}");
    }
}
