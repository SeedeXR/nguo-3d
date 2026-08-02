// SPDX-License-Identifier: GPL-2.0-or-later
//! Atomic-free **gather** assembler for the implicit mass-spring Newton system.
//!
//! The sparsity of `H` is fixed by topology (springs + pinning), so it is built
//! once; each Newton iteration only recomputes the *values*. Assembly is done
//! per-vertex-row: one worker owns vertex `i`'s CSR rows and gathers the Hessian
//! contributions of `i`'s incident springs into its own entries. No two workers
//! write the same entry, so there are **no atomics** — the design the WGSL port
//! requires (core WebGPU has no float atomics) and the one determinism requires
//! (see `docs/DEVELOPING.md` §6).
//! This CPU implementation is verified bit-close to the triplet assembler in
//! `implicit.rs`, and is mirrored one-to-one by the WGSL kernel in engine-gpu.
//!
//! Offsets: for a present block `(i,j)`, row `3i+r` stores columns `3j,3j+1,3j+2`
//! consecutively (blocks are multiples-of-3 and `from_triplets` sorts columns),
//! so a single offset per (block,row) addresses a 3-wide run.

// Numerical block assembly: fixed 3×3 index loops and 3-wide row copies read
// clearer as explicit loops than as iterator/copy_from_slice rewrites.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

use crate::csr::CsrMatrix;
use crate::implicit::spring_block;
use crate::state::SimState;
use glam::Vec3;

/// Precomputed sparsity + value offsets for a fixed spring topology.
pub struct SpringAssembler {
    pub n: usize,
    pub ndof: usize,
    /// Sparsity (values are the zero template; real values come from `assemble`).
    pub structure: CsrMatrix,
    pub pinned: Vec<bool>,
    pub mass: Vec<f32>,
    /// Per scalar-row diagonal offset. Free `i`: start of the 3-run at col `3i`
    /// in row `3i+r`. Pinned `i`: the single entry `(3i+r,3i+r)`.
    diag_off: Vec<i32>,
    // Per-vertex incident-spring adjacency (CSR-style).
    adj_ptr: Vec<u32>,
    adj_j: Vec<u32>,
    adj_rest: Vec<f32>,
    /// 3 offsets per adjacency entry: off-diagonal block `(i,j)` row starts, or
    /// `-1` when the neighbor is pinned (block dropped by the Dirichlet filter).
    adj_off: Vec<i32>,
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

impl SpringAssembler {
    pub fn new(state: &SimState) -> Self {
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

        // Structure: full 3×3 diagonal block per free vertex; full 3×3 (i,j)+(j,i)
        // per spring with both endpoints free; single identity entries per pinned.
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
        for s in &state.stretch {
            let (a, b) = (s.a as usize, s.b as usize);
            if !pinned[a] && !pinned[b] {
                for r in 0..3 {
                    for c in 0..3 {
                        trips.push(((3 * a + r) as u32, (3 * b + c) as u32, 0.0));
                        trips.push(((3 * b + r) as u32, (3 * a + c) as u32, 0.0));
                    }
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

        // Per-vertex incident-spring adjacency + off-diagonal offsets.
        let mut incident: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
        for s in &state.stretch {
            let (a, b) = (s.a, s.b);
            incident[a as usize].push((b, s.rest_len));
            incident[b as usize].push((a, s.rest_len));
        }
        let mut adj_ptr = vec![0u32; n + 1];
        let mut adj_j = Vec::new();
        let mut adj_rest = Vec::new();
        let mut adj_off = Vec::new();
        for i in 0..n {
            for &(j, rest) in &incident[i] {
                adj_j.push(j);
                adj_rest.push(rest);
                if !pinned[i] && !pinned[j as usize] {
                    for r in 0..3 {
                        adj_off.push(find_col(&structure, 3 * i + r, 3 * j));
                    }
                } else {
                    adj_off.extend_from_slice(&[-1, -1, -1]);
                }
            }
            adj_ptr[i + 1] = adj_j.len() as u32;
        }

        Self {
            n,
            ndof,
            structure,
            pinned,
            mass,
            diag_off,
            adj_ptr,
            adj_j,
            adj_rest,
            adj_off,
        }
    }

    /// Assemble the CSR values and gradient at configuration `x`. Returns
    /// `(values matching self.structure, grad)`. `rhs = -grad`.
    pub fn assemble(
        &self,
        x: &[Vec3],
        x_tilde: &[Vec3],
        inv_h2: f32,
        k: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut values = vec![0.0f32; self.structure.nnz()];
        let mut grad = vec![0.0f32; self.ndof];

        for i in 0..self.n {
            if self.pinned[i] {
                for r in 0..3 {
                    values[self.diag_off[3 * i + r] as usize] = 1.0;
                }
                continue;
            }
            let mut diag = [[0.0f32; 3]; 3];
            let mh2 = self.mass[i] * inv_h2;
            for c in 0..3 {
                diag[c][c] = mh2;
            }
            let mut gi = (x[i] - x_tilde[i]) * mh2;

            let (s, e) = (self.adj_ptr[i] as usize, self.adj_ptr[i + 1] as usize);
            for e_idx in s..e {
                let j = self.adj_j[e_idx] as usize;
                let rest = self.adj_rest[e_idx];
                let d = x[i] - x[j];
                let (hk, ga) = spring_block(d, rest, k);
                for r in 0..3 {
                    for c in 0..3 {
                        diag[r][c] += hk[r][c];
                    }
                }
                gi += ga;
                // Off-diagonal block (i,j) = -hk, if present (neighbor free).
                if self.adj_off[3 * e_idx] >= 0 {
                    for r in 0..3 {
                        let base = self.adj_off[3 * e_idx + r] as usize;
                        for c in 0..3 {
                            values[base + c] = -hk[r][c];
                        }
                    }
                }
            }
            // Write diagonal block + gradient.
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

    // ---- accessors for the GPU port (flat SoA buffers) ----
    pub fn row_ptr(&self) -> &[u32] {
        &self.structure.row_ptr
    }
    pub fn col_idx(&self) -> &[u32] {
        &self.structure.col_idx
    }
    pub fn diag_off(&self) -> &[i32] {
        &self.diag_off
    }
    pub fn adj_ptr(&self) -> &[u32] {
        &self.adj_ptr
    }
    pub fn adj_j(&self) -> &[u32] {
        &self.adj_j
    }
    pub fn adj_rest(&self) -> &[f32] {
        &self.adj_rest
    }
    pub fn adj_off(&self) -> &[i32] {
        &self.adj_off
    }
    pub fn pinned_u32(&self) -> Vec<u32> {
        self.pinned.iter().map(|&p| p as u32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Mesh;

    /// Reference: the triplet assembly (same logic as `ImplicitSolver::assemble`).
    fn reference(state: &SimState, x: &[Vec3], x_tilde: &[Vec3], inv_h2: f32, k: f32) -> CsrMatrix {
        let n = state.vertex_count();
        let ndof = 3 * n;
        let mut trips = Vec::new();
        for i in 0..n {
            if state.inv_mass[i] <= 0.0 {
                continue;
            }
            let mh2 = (1.0 / state.inv_mass[i]) * inv_h2;
            for c in 0..3 {
                trips.push(((3 * i + c) as u32, (3 * i + c) as u32, mh2));
            }
            let _ = x_tilde; // grad checked separately
        }
        for s in &state.stretch {
            let (a, b) = (s.a as usize, s.b as usize);
            let d = x[a] - x[b];
            let (hk, _) = spring_block(d, s.rest_len, k);
            for r in 0..3 {
                for c in 0..3 {
                    let v = hk[r][c];
                    trips.push(((3 * a + r) as u32, (3 * a + c) as u32, v));
                    trips.push(((3 * b + r) as u32, (3 * b + c) as u32, v));
                    trips.push(((3 * a + r) as u32, (3 * b + c) as u32, -v));
                    trips.push(((3 * b + r) as u32, (3 * a + c) as u32, -v));
                }
            }
        }
        let pinned: Vec<bool> = (0..n).map(|i| state.inv_mass[i] <= 0.0).collect();
        trips.retain(|&(r, c, _)| !pinned[r as usize / 3] && !pinned[c as usize / 3]);
        for i in 0..n {
            if pinned[i] {
                for c in 0..3 {
                    trips.push(((3 * i + c) as u32, (3 * i + c) as u32, 1.0));
                }
            }
        }
        CsrMatrix::from_triplets(ndof, &trips)
    }

    #[test]
    fn gather_assembler_matches_triplet_reference() {
        // A small cloth with two pinned corners and perturbed positions.
        let nn = 4;
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
        let mut state = SimState::from_mesh(&Mesh::new(verts, tris), 0.1);
        state.pin(0);
        state.pin(nn - 1);
        // Perturb so the Hessian is non-trivial (springs stretched/sheared).
        let x: Vec<Vec3> = state
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| *p + Vec3::new((i as f32 * 0.3).sin() * 0.02, -0.05, 0.01))
            .collect();
        let x_tilde: Vec<Vec3> = x.iter().map(|p| *p + Vec3::new(0.0, -0.001, 0.0)).collect();
        let (inv_h2, k) = (1.0e4, 1000.0);

        let asm = SpringAssembler::new(&state);
        let (values, _grad) = asm.assemble(&x, &x_tilde, inv_h2, k);
        let reference = reference(&state, &x, &x_tilde, inv_h2, k);

        // Same sparsity (col ordering), values match to f32 rounding.
        assert_eq!(asm.structure.row_ptr, reference.row_ptr, "row_ptr differs");
        assert_eq!(asm.structure.col_idx, reference.col_idx, "col_idx differs");
        let mut max_err = 0.0f32;
        for (a, b) in values.iter().zip(&reference.values) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-3, "assembler value drift {max_err}");
    }
}
