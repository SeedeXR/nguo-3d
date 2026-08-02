// SPDX-License-Identifier: GPL-2.0-or-later
//! Atomic-free gather assembler for the isotropic shell (ARAP / StVK) Hessian +
//! gradient.
//!
//! Triangles are 3-vertex elements, exactly like the Baraff-Witkin membrane, so this
//! reuses the shared [`Gather3`] topology verbatim — only the element math differs
//! ([`crate::shell_model::shell_gradient_analytic`] +
//! [`crate::shell_model::shell_hessian_analytic`] instead of
//! `membrane::element_assembly`). Same relationship `TetAssembler` has to
//! `BendingAssembler` through `Gather4`.

use crate::csr::CsrMatrix;
use crate::gather3::Gather3;
use crate::shell_model::{self, shell_gradient_analytic, shell_hessian_analytic, ShellElement};
use glam::Vec3;

/// Isotropic-shell assembler over a fixed triangle topology.
pub struct ShellAssembler {
    pub n: usize,
    pub ndof: usize,
    elements: Vec<ShellElement>,
    g: Gather3,
}

impl ShellAssembler {
    pub fn new(elements: &[ShellElement], n: usize, pinned: &[bool]) -> Self {
        let elems: Vec<[u32; 3]> = elements.iter().map(|e| e.v).collect();
        let g = Gather3::new(&elems, n, pinned);
        Self {
            n: g.n,
            ndof: g.ndof,
            elements: elements.to_vec(),
            g,
        }
    }

    /// Assemble the shell CSR values + gradient at `x` for `mu`/`lambda`. Pinned rows
    /// get an identity diagonal. `eps` is the eigensystem's degenerate-eigenvalue guard
    /// (`ImplicitParams::eiganalysis_eps`).
    pub fn assemble(&self, x: &[Vec3], mu: f32, lambda: f32, eps: f32) -> (Vec<f32>, Vec<f32>) {
        self.g.assemble(|t| {
            let el = &self.elements[t];
            let p = [
                x[el.v[0] as usize],
                x[el.v[1] as usize],
                x[el.v[2] as usize],
            ];
            let grad = shell_gradient_analytic(&p, el, mu, lambda);
            let hess = shell_hessian_analytic(&p, el, mu, lambda, eps);
            let mut g9 = [0.0f32; 9];
            for (a, gv) in grad.iter().enumerate() {
                g9[3 * a] = gv.x;
                g9[3 * a + 1] = gv.y;
                g9[3 * a + 2] = gv.z;
            }
            (g9, hess)
        })
    }

    /// Total shell elastic energy at `x`.
    pub fn energy(&self, x: &[Vec3], mu: f32, lambda: f32) -> f32 {
        self.elements
            .iter()
            .map(|el| {
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                ];
                shell_model::shell_energy(&p, el, mu, lambda)
            })
            .sum()
    }

    pub fn structure(&self) -> &CsrMatrix {
        &self.g.structure
    }
    pub fn elements(&self) -> &[ShellElement] {
        &self.elements
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
    use crate::shell_model::build_shells;
    use crate::tet::TetModel;

    /// A folded grid so triangles share vertices and off-diagonal blocks accumulate.
    fn grid(nn: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        for j in 0..nn {
            for i in 0..nn {
                let a = 0.2 * j as f32;
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
        (verts, tris)
    }

    /// Independent oracle: direct triplet assembly of the element Hessians,
    /// Dirichlet-filtered — the same check the membrane/bending/tet assemblers use.
    fn reference(
        els: &[ShellElement],
        x: &[Vec3],
        n: usize,
        pinned: &[bool],
        mu: f32,
        lambda: f32,
    ) -> CsrMatrix {
        let mut trips = Vec::new();
        for el in els {
            let p = [
                x[el.v[0] as usize],
                x[el.v[1] as usize],
                x[el.v[2] as usize],
            ];
            let hess = shell_hessian_analytic(&p, el, mu, lambda, 1.0e-6);
            for a in 0..3 {
                let va = el.v[a] as usize;
                for b in 0..3 {
                    let vb = el.v[b] as usize;
                    for r in 0..3 {
                        for c in 0..3 {
                            trips.push((
                                (3 * va + r) as u32,
                                (3 * vb + c) as u32,
                                hess[(3 * a + r) * 9 + (3 * b + c)],
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
    fn shell_gather_matches_triplet_reference() {
        let (rest, tris) = grid(5);
        let n = rest.len();
        let els = build_shells(&tris, &rest, 1e-12, TetModel::Arap);
        assert!(els.len() > 10);
        let mut pinned = vec![false; n];
        pinned[0] = true;
        pinned[4] = true;
        let x: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::new(0.0, 0.0, 0.02 * (i as f32).sin()))
            .collect();
        let (mu, lambda) = (400.0f32, 250.0f32);

        let asm = ShellAssembler::new(&els, n, &pinned);
        let (values, grad) = asm.assemble(&x, mu, lambda, 1.0e-6);
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
        assert!(vmax < 1e-3 * vscale, "shell gather value drift {vmax}");

        // Gradient vs a direct sum over free rows.
        let mut gref = vec![0.0f32; 3 * n];
        for el in &els {
            let p = [
                x[el.v[0] as usize],
                x[el.v[1] as usize],
                x[el.v[2] as usize],
            ];
            let g = shell_gradient_analytic(&p, el, mu, lambda);
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
        assert!(gmax < 1e-3 * gscale, "shell gather grad drift {gmax}");
    }

    /// The assembled matrix must be symmetric and PSD so CG can consume it.
    #[test]
    fn assembled_shell_matrix_is_symmetric_psd() {
        let (rest, tris) = grid(5);
        let n = rest.len();
        let els = build_shells(&tris, &rest, 1e-12, TetModel::Arap);
        let pinned = vec![false; n];
        let x: Vec<Vec3> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::splat(0.01 * (i as f32).sin()))
            .collect();
        let asm = ShellAssembler::new(&els, n, &pinned);
        let (values, _g) = asm.assemble(&x, 400.0, 250.0, 1.0e-6);
        let mut m = asm.structure().clone();
        m.values = values;
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
            let q: f32 = ax.iter().zip(&xv).map(|(a, b)| a * b).sum();
            assert!(q >= -1e-2, "assembled shell matrix not PSD: xᵀAx = {q}");
        }
    }

    /// Rest configuration: zero energy and zero assembled gradient.
    #[test]
    fn shell_rest_configuration_assembles_zero_gradient() {
        let (rest, tris) = grid(5);
        let n = rest.len();
        let els = build_shells(&tris, &rest, 1e-12, TetModel::Arap);
        let asm = ShellAssembler::new(&els, n, &vec![false; n]);
        assert!(asm.energy(&rest, 400.0, 250.0).abs() < 1e-4, "rest energy");
        let (_v, grad) = asm.assemble(&rest, 400.0, 250.0, 1.0e-6);
        let gmax = grad.iter().fold(0.0f32, |m, g| m.max(g.abs()));
        assert!(gmax < 1e-3, "rest gradient nonzero: {gmax}");
    }
}
