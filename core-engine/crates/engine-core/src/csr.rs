// SPDX-License-Identifier: GPL-2.0-or-later
//! Minimal CSR sparse matrix (CPU side). Used to build linear systems and as the
//! self-verification oracle for the GPU PCG. The reference solver assembles its
//! Hessian into custom `DynCSRMat`/`FixedCSRMat` (docs/KERNELS.md); this is the
//! portable analogue the WGSL SpMV consumes.

/// Compressed Sparse Row matrix, square (`n` × `n`).
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub n: usize,
    pub row_ptr: Vec<u32>, // len n+1
    pub col_idx: Vec<u32>, // len nnz
    pub values: Vec<f32>,  // len nnz
}

/// One assembly entry: `(row, col, value)`.
///
/// The indices are `u32`, not `usize`, and that is load-bearing rather than cosmetic. The
/// implicit assembly emits ~14 M of these EVERY Newton iteration on `fishingknot`; at
/// `(usize, usize, f32)` the tuple is 24 bytes (8 + 8 + 4, padded to 24) and the list alone
/// is 320 MB, which is the single largest live allocation in the solver. At `(u32, u32, f32)`
/// it is 12 bytes with no padding — exactly half. A degree of freedom index is
/// `3 * vertex + component`, so `u32` covers 1.4 billion DOFs; `from_triplets` asserts the
/// bound anyway.
pub type Triplet = (u32, u32, f32);

impl CsrMatrix {
    /// Build from a list of (row, col, value) triplets. Entries in the same
    /// (row,col) are summed. Columns within a row are sorted ascending.
    /// Built by COUNTING SORT into one flat buffer — deliberately not `Vec<Vec<_>>`.
    ///
    /// The obvious `vec![Vec::new(); n]` bucket-per-row costs `n` separate allocations that
    /// each double as they fill. At the sizes this solver actually runs (`hang`: 49,152 rows
    /// and ~12.3M triplets EVERY Newton iteration) that churned 0.9–2.6 GB of RSS per step
    /// and was OOM-killing long validation runs. Counting sort does the same job with two
    /// allocations and a flat scatter.
    pub fn from_triplets(n: usize, triplets: &[Triplet]) -> Self {
        // Pass 1: per-row counts (also the bounds check).
        let mut counts = vec![0u32; n + 1];
        for &(r, c, _) in triplets {
            let (r, c) = (r as usize, c as usize);
            assert!(r < n && c < n, "triplet ({r},{c}) out of range for n={n}");
            counts[r] += 1;
        }
        // Prefix sum → start offset of each row in the flat buffer. `counts` has n+1 slots
        // and only the first n are row counts, so the scan is over `counts[..n]` and the
        // final push closes the last row.
        let mut starts = Vec::with_capacity(n + 1);
        let mut acc = 0u32;
        for &c in &counts[..n] {
            starts.push(acc);
            acc += c;
        }
        starts.push(acc);
        // Pass 2: scatter into one buffer, using a moving cursor per row.
        let mut cursor = starts.clone();
        let mut flat = vec![(0u32, 0.0f32); triplets.len()];
        for &(r, c, v) in triplets {
            let at = cursor[r as usize] as usize;
            flat[at] = (c, v);
            cursor[r as usize] += 1;
        }
        // Pass 3: sort each row's slice in place and coalesce duplicate columns.
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0u32);
        for r in 0..n {
            let (lo, hi) = (starts[r] as usize, starts[r + 1] as usize);
            let row = &mut flat[lo..hi];
            row.sort_unstable_by_key(|&(c, _)| c);
            let mut j = 0;
            while j < row.len() {
                let c = row[j].0;
                let mut sum = 0.0;
                while j < row.len() && row[j].0 == c {
                    sum += row[j].1;
                    j += 1;
                }
                col_idx.push(c);
                values.push(sum);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        Self {
            n,
            row_ptr,
            col_idx,
            values,
        }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// y = A x on the CPU (SpMV) — the oracle the GPU SpMV must match.
    pub fn spmv_cpu(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.n);
        let mut y = vec![0.0f32; self.n];
        for (row, y_row) in y.iter_mut().enumerate() {
            let (s, e) = (self.row_ptr[row] as usize, self.row_ptr[row + 1] as usize);
            let mut sum = 0.0f32;
            for k in s..e {
                sum += self.values[k] * x[self.col_idx[k] as usize];
            }
            *y_row = sum;
        }
        y
    }

    /// Diagonal entries (0 where a row has no diagonal). For the Jacobi precond.
    pub fn diagonal(&self) -> Vec<f32> {
        let mut d = vec![0.0f32; self.n];
        for (row, d_row) in d.iter_mut().enumerate() {
            let (s, e) = (self.row_ptr[row] as usize, self.row_ptr[row + 1] as usize);
            for k in s..e {
                if self.col_idx[k] as usize == row {
                    *d_row = self.values[k];
                }
            }
        }
        d
    }

    /// 1D Laplacian with Dirichlet ends: tridiagonal SPD `[-1, 2, -1]`.
    /// A canonical, well-conditioned-enough SPD test system.
    pub fn laplacian_1d(n: usize) -> Self {
        let mut t: Vec<Triplet> = Vec::new();
        for i in 0..n as u32 {
            t.push((i, i, 2.0));
            if i > 0 {
                t.push((i, i - 1, -1.0));
            }
            if (i as usize) + 1 < n {
                t.push((i, i + 1, -1.0));
            }
        }
        Self::from_triplets(n, &t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplacian_spmv_is_correct() {
        let a = CsrMatrix::laplacian_1d(5);
        // A * ones = [1,0,0,0,1] for the Dirichlet tridiagonal.
        let y = a.spmv_cpu(&[1.0; 5]);
        assert_eq!(y, vec![1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(a.diagonal(), vec![2.0; 5]);
    }

    #[test]
    fn triplets_coalesce_and_sort() {
        let a = CsrMatrix::from_triplets(2, &[(0, 1, 1.0), (0, 1, 2.0), (0, 0, 5.0)]);
        assert_eq!(a.row_ptr, vec![0, 2, 2]);
        assert_eq!(a.col_idx, vec![0, 1]);
        assert_eq!(a.values, vec![5.0, 3.0]);
    }
}
