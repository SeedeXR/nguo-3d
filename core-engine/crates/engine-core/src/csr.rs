// SPDX-License-Identifier: GPL-2.0-or-later
//! Minimal CSR sparse matrix (CPU side). Used to build linear systems and as the
//! self-verification oracle for the GPU PCG. The reference solver assembles its
//! Hessian into custom `DynCSRMat`/`FixedCSRMat` (docs/KERNELS.md); this is the
//! portable analogue the WGSL SpMV consumes.

/// Compressed Sparse Row matrix, square (`n` × `n`).
#[derive(Clone, Debug, Default)]
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

/// **Streaming Hessian accumulator — the assembly never materialises its triplet list.**
///
/// [`CsrMatrix::from_triplets`] needs the whole list at once, and that list is what put the
/// implicit solver's ceiling at ~40 k cloth vertices. Measured on the male COL proxy
/// (10 230 v / 15 000 t) as a signed static collider, a full-body-width sheet dropped from
/// above, self-collision on, dt = 1/120, 3 Newton iterations, 12 steps, `--release`, with
/// `MGUSO_ASM_STATS=1`:
///
/// | cloth verts | pushes/assembly | old `Vec` capacity | CSR nnz | ratio |
/// |---|---|---|---|---|
/// | 5 041 | 17 304 268 | 33 554 432 (**384.0 MB**), measured | 607 923 (4.6 MB) | 28× |
/// | 10 000 | 28 499 213 | 67 108 864 (**768.0 MB**), measured | 1 182 978 (9.0 MB) | 24× |
/// | 70 312 (men-shirt) | 131 965 221 | 134 217 728 (**1 536 MB**), derived¹ | 9 993 942 (76.2 MB) | 13× |
///
/// ¹ the smallest power of two that count forces, not a run: the 70 k case was refused on the
/// old code at a projected 6.4–8.6 GB and never executed. The two smaller rows are measured.
///
/// The CSR the list produces is 13–28× smaller than the list, and nothing reads the list —
/// `implicit::Asm` gathers the `fixed_hess_in` blocks as entries stream past it. So entries are
/// folded into a running CSR in bounded chunks as they arrive:
///
/// * `push` appends to a chunk buffer of at most [`Self::CHUNK_MAX`] entries;
/// * when it fills, the chunk is counting-sorted and coalesced exactly as `from_triplets` does,
///   then **merged** into the accumulator by a linear per-row two-pointer walk;
/// * so peak assembly bytes are `20 · CHUNK_MAX` (buffer + scatter) plus `O(nnz)`, instead of
///   `20 · entries`.
///
/// The buffer **grows by doubling from [`Self::CHUNK_MIN`]** and only ceilings at `CHUNK_MAX`, so
/// a scene that stays under `CHUNK_MAX` never chunks and its arithmetic is **bit-identical** to
/// the old single-pass path. `implicit::Asm` drops Dirichlet rows before they reach `push`, so it
/// is the SURVIVING count that has to clear `CHUNK_MAX` — a 36-vertex patch on the whole male
/// proxy pushes 6 767 765 entries and keeps 33 714, and lands in a 32 768-entry buffer.
///
/// **The one numerics change, and its exact scope.** For a scene whose surviving entries exceed
/// `CHUNK_MAX`, a `(row, col)` cell that receives duplicates is summed per chunk and the chunk
/// sums added in chunk order — same values, same order, different *grouping*, so ulp-level
/// movement. It is deterministic: chunk boundaries are a function of the push count alone. It
/// does NOT touch the `fixed_hess_in` gathers, which `Asm` accumulates in push order exactly as
/// the pass they replace did; verified by position fingerprint, a 400-vertex sheet over the male
/// proxy for 8 steps is bit-identical before and after (`508029dd404be21b`).
///
/// **Do not undo that scoping.** A first cut folded the Dirichlet drop into `finish` instead, so
/// the gathers read coalesced per-cell sums — a pure re-association — and it moved
/// `engine-api`'s `cloth_dropped_on_a_real_proxy_does_not_sink_into_the_body` (female crown,
/// 36-vertex patch, 13 mm of free-fall budget, asserting `y_held > crown - 0.006`) from passing
/// to 1.894393. That scene is CHAOTIC in the summation order: the barrier stiffness is a
/// Rayleigh quotient of the gathered Hessian, and a stencil crosses or misses `d >= dhat` on its
/// last bit. Three independent, provably neutral re-associations of the OLD code were measured:
///
/// | summation order | `y_held` | verdict |
/// |---|---|---|
/// | unmodified (pdqsort, raw duplicates) | > 1.899050 | passes |
/// | `gather_elastic_blocks` iterated `.rev()` | 1.877920 | fails, below free fall |
/// | `from_triplets` row sort made STABLE | 1.896226 | fails |
///
/// So at its current threshold that assertion is a bit-exactness guard in a physics guard's
/// clothes — but the fix was to stop re-associating the gather, not to argue with the test.
///
/// The Dirichlet filter is applied by [`Self::finish`] on the **CSR**, not by a `retain` over
/// the list. That is exact rather than approximate: every entry in a cell shares the cell's
/// keep/drop verdict, so dropping-then-summing and summing-then-dropping are the same
/// floating-point expression.
pub struct TripletSink {
    n: usize,
    buf: Vec<Triplet>,
    /// Entries `buf` accepts before it is folded into `acc`. Doubles until `CHUNK_MAX`.
    flush_at: usize,
    pushed: usize,
    flushes: usize,
    /// The running sum. Empty `row_ptr` ⇒ nothing accumulated yet.
    acc: CsrMatrix,
    /// Merge target, swapped with `acc`. Kept for the life of the sink, like `buf`.
    scratch: CsrMatrix,
    chunk: CsrMatrix,
    starts: Vec<u32>,
    cursor: Vec<u32>,
    flat: Vec<(u32, f32)>,
}

impl Default for TripletSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TripletSink {
    /// Smallest chunk buffer. 16 384 entries = 196 KB of `buf` + `flat`, so a unit-test-sized
    /// scene allocates about as much as its own mesh.
    pub const CHUNK_MIN: usize = 1 << 14;
    /// Largest chunk buffer: 2 097 152 entries = **40 MB** of `buf` + `flat`. Above this the
    /// buffer folds instead of growing. The bound trades merge passes for bytes — merge cost
    /// is `(pushed / CHUNK_MAX) · nnz`, and at the measured 17× duplicate factor that is a
    /// small multiple of the single sort it replaces.
    pub const CHUNK_MAX: usize = 1 << 21;

    pub fn new() -> Self {
        Self {
            n: 0,
            buf: Vec::new(),
            flush_at: Self::CHUNK_MIN,
            pushed: 0,
            flushes: 0,
            acc: CsrMatrix::default(),
            scratch: CsrMatrix::default(),
            chunk: CsrMatrix::default(),
            starts: Vec::new(),
            cursor: Vec::new(),
            flat: Vec::new(),
        }
    }

    /// Start a fresh `n` × `n` assembly. Every heap buffer is retained, including the chunk
    /// size the last assembly grew to — that is the self-correcting size hint, so no caller
    /// has to carry one.
    pub fn begin(&mut self, n: usize) {
        self.n = n;
        self.buf.clear();
        self.buf.reserve_exact(self.flush_at);
        self.pushed = 0;
        self.flushes = 0;
        self.acc.n = n;
        self.acc.row_ptr.clear();
        self.acc.col_idx.clear();
        self.acc.values.clear();
    }

    #[inline]
    pub fn push(&mut self, t: Triplet) {
        self.pushed += 1;
        self.buf.push(t);
        if self.buf.len() >= self.flush_at {
            self.buf_full();
        }
    }

    /// Total pushes this assembly. The quantity the chunk buffer is sized from.
    pub fn pushed(&self) -> usize {
        self.pushed
    }

    /// Chunk buffer capacity, in entries — the assembly's largest single allocation.
    pub fn chunk_capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Chunk folds performed this assembly. 0 ⇒ the whole assembly fitted in one buffer and
    /// the arithmetic is bit-identical to the pre-streaming path.
    pub fn flushes(&self) -> usize {
        self.flushes
    }

    /// Non-zeros in the running sum.
    pub fn nnz(&self) -> usize {
        self.acc.values.len()
    }

    /// Grow, or fold. Cold: at most `log2(CHUNK_MAX / CHUNK_MIN)` growths and then one call
    /// per `CHUNK_MAX` entries.
    #[cold]
    fn buf_full(&mut self) {
        if self.flush_at < Self::CHUNK_MAX {
            self.flush_at = (self.flush_at * 2).min(Self::CHUNK_MAX);
            self.buf.reserve_exact(self.flush_at - self.buf.len());
        } else {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        self.build_chunk();
        if self.acc.row_ptr.is_empty() {
            std::mem::swap(&mut self.acc, &mut self.chunk);
        } else {
            merge_csr(&self.acc, &self.chunk, &mut self.scratch);
            std::mem::swap(&mut self.acc, &mut self.scratch);
        }
        self.flushes += 1;
    }

    /// Counting-sort `buf` into `chunk`, coalescing duplicates. The body of
    /// [`CsrMatrix::from_triplets`], over reused buffers.
    fn build_chunk(&mut self) {
        let n = self.n;
        let (buf, starts, cursor, flat, chunk) = (
            &mut self.buf,
            &mut self.starts,
            &mut self.cursor,
            &mut self.flat,
            &mut self.chunk,
        );
        starts.clear();
        starts.resize(n + 1, 0);
        for &(r, c, _) in buf.iter() {
            let (ru, cu) = (r as usize, c as usize);
            assert!(
                ru < n && cu < n,
                "triplet ({ru},{cu}) out of range for n={n}"
            );
            starts[ru + 1] += 1;
        }
        for k in 0..n {
            starts[k + 1] += starts[k];
        }
        cursor.clear();
        cursor.extend_from_slice(&starts[..n]);
        flat.clear();
        flat.resize(buf.len(), (0u32, 0.0f32));
        for &(r, c, v) in buf.iter() {
            let at = &mut cursor[r as usize];
            flat[*at as usize] = (c, v);
            *at += 1;
        }
        chunk.n = n;
        chunk.row_ptr.clear();
        chunk.col_idx.clear();
        chunk.values.clear();
        chunk.row_ptr.push(0);
        for r in 0..n {
            let row = &mut flat[starts[r] as usize..starts[r + 1] as usize];
            row.sort_unstable_by_key(|&(c, _)| c);
            let mut j = 0;
            while j < row.len() {
                let c = row[j].0;
                let mut sum = 0.0;
                while j < row.len() && row[j].0 == c {
                    sum += row[j].1;
                    j += 1;
                }
                chunk.col_idx.push(c);
                chunk.values.push(sum);
            }
            chunk.row_ptr.push(chunk.col_idx.len() as u32);
        }
        buf.clear();
    }

    /// The assembled matrix, with the **Dirichlet filter** applied: a pinned vertex's rows and
    /// columns are dropped (which keeps symmetry) and its diagonal set to 1, so `Δx = 0`
    /// there. `pinned` is indexed by VERTEX (`dof / 3`); an empty slice pins nothing.
    pub fn finish(&mut self, pinned: &[bool]) -> CsrMatrix {
        self.flush();
        let n = self.n;
        let a = &self.acc;
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(a.col_idx.len());
        let mut values = Vec::with_capacity(a.values.len());
        let pin = |dof: usize| pinned.get(dof / 3).copied().unwrap_or(false);
        row_ptr.push(0u32);
        for r in 0..n {
            if pin(r) {
                col_idx.push(r as u32);
                values.push(1.0);
            } else if !a.row_ptr.is_empty() {
                for k in a.row_ptr[r] as usize..a.row_ptr[r + 1] as usize {
                    let c = a.col_idx[k];
                    if !pin(c as usize) {
                        col_idx.push(c);
                        values.push(a.values[k]);
                    }
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        CsrMatrix {
            n,
            row_ptr,
            col_idx,
            values,
        }
    }
}

/// `out = a + b`, both sorted CSR of the same order. Linear per-row two-pointer walk; `out`'s
/// buffers are reused.
fn merge_csr(a: &CsrMatrix, b: &CsrMatrix, out: &mut CsrMatrix) {
    let n = a.n;
    out.n = n;
    out.row_ptr.clear();
    out.col_idx.clear();
    out.values.clear();
    out.row_ptr.reserve(n + 1);
    out.col_idx.reserve(a.col_idx.len() + b.col_idx.len());
    out.values.reserve(a.values.len() + b.values.len());
    out.row_ptr.push(0);
    for r in 0..n {
        let (mut i, ie) = (a.row_ptr[r] as usize, a.row_ptr[r + 1] as usize);
        let (mut j, je) = (b.row_ptr[r] as usize, b.row_ptr[r + 1] as usize);
        while i < ie || j < je {
            // `u32::MAX` as "exhausted": a column index is `< n`, and `n` is asserted to fit
            // in `u32` by the assembly, so it can never collide with a real column.
            let ca = if i < ie { a.col_idx[i] } else { u32::MAX };
            let cb = if j < je { b.col_idx[j] } else { u32::MAX };
            match ca.cmp(&cb) {
                std::cmp::Ordering::Less => {
                    out.col_idx.push(ca);
                    out.values.push(a.values[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.col_idx.push(cb);
                    out.values.push(b.values[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // `a` first: `a` holds the earlier chunks' partial sum, so duplicates are
                    // still summed in push order, one chunk at a time.
                    out.col_idx.push(ca);
                    out.values.push(a.values[i] + b.values[j]);
                    i += 1;
                    j += 1;
                }
            }
        }
        out.row_ptr.push(out.col_idx.len() as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sink must agree with [`CsrMatrix::from_triplets`] **exactly** when nothing chunks,
    /// and to a rounding tolerance when everything does. Both halves matter: the first is the
    /// no-regression claim for every scene under `CHUNK_MAX`, the second is the whole point.
    #[test]
    fn sink_matches_from_triplets() {
        // Deterministic pseudo-random triplets over a 3-DOF-per-vertex index space, with
        // heavy duplication so coalescing is exercised.
        let n = 3 * 64;
        let mut t: Vec<Triplet> = Vec::new();
        let mut s = 12345u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize
        };
        for _ in 0..200_000 {
            let r = next() % n;
            let c = next() % n;
            let v = (next() % 1000) as f32 * 1.0e-3;
            t.push((r as u32, c as u32, v));
        }
        let want = CsrMatrix::from_triplets(n, &t);

        let mut sink = TripletSink::new();
        sink.begin(n);
        for &e in &t {
            sink.push(e);
        }
        let got = sink.finish(&[]);
        assert_eq!(sink.pushed(), t.len());
        assert_eq!(got.row_ptr, want.row_ptr);
        assert_eq!(got.col_idx, want.col_idx);
        assert_eq!(
            got.values,
            want.values,
            "{} flushes: under CHUNK_MAX the sink must be bit-identical",
            sink.flushes()
        );
        assert!(
            sink.chunk_capacity() <= 2 * t.len(),
            "chunk buffer overshot"
        );

        // Now force chunking with a small ceiling, by pushing past CHUNK_MAX-worth. Reuse the
        // same data 12 times so the sum is large enough for grouping to matter.
        let mut sink = TripletSink::new();
        sink.begin(n);
        let mut big: Vec<Triplet> = Vec::new();
        for _ in 0..12 {
            big.extend_from_slice(&t);
        }
        for &e in &big {
            sink.push(e);
        }
        let got = sink.finish(&[]);
        let want = CsrMatrix::from_triplets(n, &big);
        assert!(
            sink.flushes() > 1,
            "fixture did not chunk ({} pushes, {} flushes)",
            big.len(),
            sink.flushes()
        );
        assert_eq!(got.col_idx, want.col_idx, "chunking changed the SPARSITY");
        for (i, (&g, &w)) in got.values.iter().zip(want.values.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1.0e-4 * w.abs().max(1.0),
                "entry {i}: chunked {g} vs single-pass {w} — that is not re-association"
            );
        }
    }

    /// The Dirichlet filter on the CSR must equal a `retain` over the triplets: same set, and
    /// bit-identical values, because every entry in a cell shares the cell's verdict.
    #[test]
    fn finish_dirichlet_equals_retaining_triplets() {
        let n = 3 * 8;
        let pinned = [false, true, false, false, true, false, false, false];
        let mut t: Vec<Triplet> = Vec::new();
        for r in 0..n as u32 {
            for c in 0..n as u32 {
                if (r * 7 + c * 3) % 5 != 0 {
                    t.push((r, c, 0.25 + (r as f32) * 0.5 - (c as f32) * 0.125));
                }
            }
        }
        let mut sink = TripletSink::new();
        sink.begin(n);
        for &e in &t {
            sink.push(e);
        }
        let got = sink.finish(&pinned);

        let mut kept: Vec<Triplet> = t
            .iter()
            .copied()
            .filter(|&(r, c, _)| !pinned[r as usize / 3] && !pinned[c as usize / 3])
            .collect();
        for (i, p) in pinned.iter().enumerate() {
            if *p {
                for c in 0..3u32 {
                    kept.push((3 * i as u32 + c, 3 * i as u32 + c, 1.0));
                }
            }
        }
        let want = CsrMatrix::from_triplets(n, &kept);
        assert_eq!(got.row_ptr, want.row_ptr);
        assert_eq!(got.col_idx, want.col_idx);
        assert_eq!(got.values, want.values);
    }

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
