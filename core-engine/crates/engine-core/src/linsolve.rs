// SPDX-License-Identifier: GPL-2.0-or-later
//! Linear-solver seam. The implicit stepper solves `H Δx = -grad` each Newton
//! iteration through a [`LinearSolver`]; the CPU CG here is the reference /
//! no-GPU path and the oracle the GPU PCG (engine-gpu) is verified against.

use crate::csr::CsrMatrix;
use std::cell::Cell;

/// Diagnostics from the most recent linear solve — fed into the invariant checker
/// (`cg_converged` / `cg_iter_under_cap`, mirroring the gauge's `advance.reresid`
/// and `advance.iter` channels).
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearStats {
    /// CG iterations performed.
    pub iters: u32,
    /// Final relative residual `‖r‖/‖b‖`.
    pub residual: f64,
}

/// Solve a symmetric-positive-(semi)definite `A x = b`. `tol` is relative to ‖b‖.
pub trait LinearSolver {
    fn solve(&self, a: &CsrMatrix, b: &[f32], tol: f32, max_iter: u32) -> Vec<f32>;
    /// Stats from the most recent [`solve`](Self::solve). Default: unavailable
    /// (zeros) — solvers that track them override this via interior mutability.
    fn last_stats(&self) -> LinearStats {
        LinearStats::default()
    }
}

/// CPU Jacobi-preconditioned conjugate gradient. Accumulates dot products in f64
/// (the reference does the same for its PCG scalars — docs/CONFIG.md), returns
/// f32. Deterministic (fixed traversal order).
pub fn solve_cg_cpu(a: &CsrMatrix, b: &[f32], tol: f32, max_iter: u32) -> Vec<f32> {
    cg_cpu(a, b, tol, max_iter, true).0
}

/// Invert a 3×3 matrix (f64); `None` if near-singular.
fn invert3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-20 {
        return None;
    }
    let id = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
        ],
    ])
}

/// Which preconditioner the CG uses.
#[derive(Clone, Copy, Debug)]
pub enum Precond {
    /// Scalar diagonal Jacobi.
    Diagonal,
    /// 3×3 per-vertex block Jacobi (default) — captures intra-vertex coupling.
    Block,
    /// One-level additive Schwarz: aggregate vertices into domains of ~`domain`
    /// vertices and solve each domain's dense block exactly. Captures within-
    /// domain inter-vertex coupling → fewer CG iterations on cloth.
    Schwarz { domain: usize },
}

/// Dense lower-triangular Cholesky `L` of a symmetric PD `m×m` matrix (row-major);
/// `None` if not PD.
fn cholesky(a: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= l[i * m + k] * l[j * m + k];
            }
            if i == j {
                if s <= 1e-14 {
                    return None;
                }
                l[i * m + i] = s.sqrt();
            } else {
                l[i * m + j] = s / l[j * m + j];
            }
        }
    }
    Some(l)
}

/// Solve `L Lᵀ x = b` given the Cholesky factor `L` (m×m row-major).
fn chol_solve(l: &[f64], m: usize, b: &[f64], out: &mut [f64]) {
    let mut y = vec![0.0f64; m];
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * m + k] * y[k];
        }
        y[i] = s / l[i * m + i];
    }
    for i in (0..m).rev() {
        let mut s = y[i];
        for k in (i + 1)..m {
            s -= l[k * m + i] * out[k];
        }
        out[i] = s / l[i * m + i];
    }
}

/// One-level additive Schwarz **topology**: disjoint vertex domains + per-domain
/// gather indices into `a.values`. Depends only on the sparsity pattern, which is
/// FIXED across the whole simulation — so this is built once and reused; only the
/// dense factorization (`factorize`) rebuilds per solve when values change. This
/// amortization is what makes Schwarz a net win vs rebuilding aggregation every
/// solve (the earlier stateless version was slower than block-Jacobi).
struct SchwarzTopology {
    domains: Vec<Vec<usize>>, // vertex indices per domain
    /// Per domain: `m*m` indices into `a.values` (row-major dense block),
    /// `usize::MAX` where the entry is structurally absent (→ 0).
    gather: Vec<Vec<usize>>,
}

impl SchwarzTopology {
    fn build(a: &CsrMatrix, domain_size: usize) -> Self {
        let nv = a.n / 3;
        // Vertex adjacency from the block-sparsity graph.
        let mut adj = vec![Vec::new(); nv];
        for row in 0..a.n {
            let i = row / 3;
            let (s, e) = (a.row_ptr[row] as usize, a.row_ptr[row + 1] as usize);
            for k in s..e {
                let j = a.col_idx[k] as usize / 3;
                if j != i {
                    adj[i].push(j);
                }
            }
        }
        for ai in adj.iter_mut() {
            ai.sort_unstable();
            ai.dedup();
        }
        // Greedy BFS aggregation into disjoint domains of ~domain_size vertices.
        let mut assigned = vec![false; nv];
        let mut domains: Vec<Vec<usize>> = Vec::new();
        for seed in 0..nv {
            if assigned[seed] {
                continue;
            }
            let mut dom = vec![seed];
            assigned[seed] = true;
            let mut fi = 0;
            while dom.len() < domain_size && fi < dom.len() {
                let v = dom[fi];
                fi += 1;
                for &nb in &adj[v] {
                    if !assigned[nb] && dom.len() < domain_size {
                        assigned[nb] = true;
                        dom.push(nb);
                    }
                }
            }
            domains.push(dom);
        }
        // Precompute the CSR index of every dense-block entry (once — sparsity
        // fixed). `find_col_idx` scans the row; done here, not per solve.
        let find_idx = |row: usize, col: usize| -> usize {
            let (s, e) = (a.row_ptr[row] as usize, a.row_ptr[row + 1] as usize);
            for k in s..e {
                if a.col_idx[k] as usize == col {
                    return k;
                }
            }
            usize::MAX
        };
        let mut gather = Vec::with_capacity(domains.len());
        for dom in &domains {
            let dofs: Vec<usize> = dom
                .iter()
                .flat_map(|&v| [3 * v, 3 * v + 1, 3 * v + 2])
                .collect();
            let m = dofs.len();
            let mut g = vec![usize::MAX; m * m];
            for (p, &gp) in dofs.iter().enumerate() {
                for (q, &gq) in dofs.iter().enumerate() {
                    g[p * m + q] = find_idx(gp, gq);
                }
            }
            gather.push(g);
        }
        SchwarzTopology { domains, gather }
    }

    /// Factor each domain's dense block from the current `a.values` (per solve).
    fn factorize(&self, a: &CsrMatrix) -> Vec<Vec<f64>> {
        let mut factors = Vec::with_capacity(self.domains.len());
        for (di, dom) in self.domains.iter().enumerate() {
            let m = 3 * dom.len();
            let g = &self.gather[di];
            let mut ad = vec![0.0f64; m * m];
            for (e, &idx) in g.iter().enumerate() {
                ad[e] = if idx == usize::MAX {
                    0.0
                } else {
                    a.values[idx] as f64
                };
            }
            // Diagonal floor (matches the reference `schwarz.cu:238-251`):
            // relative-plus-absolute shift `1e-6·(tr/d) + 1e-8`. Keeps pinned/
            // dead zero rows nonsingular and clamps the Cholesky pivot, so a
            // near-singular domain block never yields NaN.
            let tr: f64 = (0..m).map(|k| ad[k * m + k]).sum();
            let floor = 1e-6 * (tr / m as f64) + 1e-8;
            for k in 0..m {
                ad[k * m + k] += floor;
            }
            let l = cholesky(&ad, m).unwrap_or_else(|| {
                let mut d = vec![0.0f64; m * m];
                for p in 0..m {
                    let v = ad[p * m + p];
                    d[p * m + p] = if v > 1e-14 { v.sqrt() } else { 1.0 };
                }
                d
            });
            factors.push(l);
        }
        factors
    }

    /// z = M⁻¹ r using precomputed factors (disjoint domains → direct scatter).
    fn apply(&self, factors: &[Vec<f64>], r: &[f64], z: &mut [f64]) {
        for (di, dom) in self.domains.iter().enumerate() {
            let m = 3 * dom.len();
            let mut rd = vec![0.0f64; m];
            for (p, &v) in dom.iter().enumerate() {
                rd[3 * p] = r[3 * v];
                rd[3 * p + 1] = r[3 * v + 1];
                rd[3 * p + 2] = r[3 * v + 2];
            }
            let mut xd = vec![0.0f64; m];
            chol_solve(&factors[di], m, &rd, &mut xd);
            for (p, &v) in dom.iter().enumerate() {
                z[3 * v] = xd[3 * p];
                z[3 * v + 1] = xd[3 * p + 1];
                z[3 * v + 2] = xd[3 * p + 2];
            }
        }
    }
}

/// Preconditioned CG (thin wrapper: `block` → [`Precond::Block`] else
/// [`Precond::Diagonal`]). Returns `(x, iterations)`.
pub fn cg_cpu(a: &CsrMatrix, b: &[f32], tol: f32, max_iter: u32, block: bool) -> (Vec<f32>, u32) {
    let p = if block && a.n.is_multiple_of(3) {
        Precond::Block
    } else {
        Precond::Diagonal
    };
    cg_cpu_precond(a, b, tol, max_iter, p)
}

/// Preconditioned CG with an explicit [`Precond`]. All accumulation in f64.
pub fn cg_cpu_precond(
    a: &CsrMatrix,
    b: &[f32],
    tol: f32,
    max_iter: u32,
    precond_kind: Precond,
) -> (Vec<f32>, u32) {
    let n = a.n;
    assert_eq!(b.len(), n);
    let block3 = n.is_multiple_of(3);
    let kind = match precond_kind {
        // Fall back to scalar diagonal for non-3-DOF systems.
        Precond::Block | Precond::Schwarz { .. } if !block3 => Precond::Diagonal,
        k => k,
    };
    let inv_diag: Vec<f64> = a
        .diagonal()
        .iter()
        .map(|&d| if d != 0.0 { 1.0 / d as f64 } else { 1.0 })
        .collect();
    // Precompute the chosen preconditioner.
    let mut binv: Vec<[[f64; 3]; 3]> = Vec::new();
    if matches!(kind, Precond::Block) {
        let nv = n / 3;
        binv.reserve(nv);
        for i in 0..nv {
            let mut m = [[0.0f64; 3]; 3];
            for (r, mr) in m.iter_mut().enumerate() {
                let row = 3 * i + r;
                let (s, e) = (a.row_ptr[row] as usize, a.row_ptr[row + 1] as usize);
                for k in s..e {
                    let c = a.col_idx[k] as usize;
                    if c >= 3 * i && c < 3 * i + 3 {
                        mr[c - 3 * i] = a.values[k] as f64;
                    }
                }
            }
            binv.push(invert3x3(&m).unwrap_or([
                [inv_diag[3 * i], 0.0, 0.0],
                [0.0, inv_diag[3 * i + 1], 0.0],
                [0.0, 0.0, inv_diag[3 * i + 2]],
            ]));
        }
    }
    let schwarz = if let Precond::Schwarz { domain } = kind {
        let topo = SchwarzTopology::build(a, domain.max(1));
        let factors = topo.factorize(a);
        Some((topo, factors))
    } else {
        None
    };
    // z = M⁻¹ r
    let precond = |r: &[f64], z: &mut [f64]| match kind {
        Precond::Schwarz { .. } => {
            let (topo, factors) = schwarz.as_ref().unwrap();
            topo.apply(factors, r, z);
        }
        Precond::Block => {
            for (i, bi) in binv.iter().enumerate() {
                let (r0, r1, r2) = (r[3 * i], r[3 * i + 1], r[3 * i + 2]);
                z[3 * i] = bi[0][0] * r0 + bi[0][1] * r1 + bi[0][2] * r2;
                z[3 * i + 1] = bi[1][0] * r0 + bi[1][1] * r1 + bi[1][2] * r2;
                z[3 * i + 2] = bi[2][0] * r0 + bi[2][1] * r1 + bi[2][2] * r2;
            }
        }
        Precond::Diagonal => {
            for i in 0..n {
                z[i] = r[i] * inv_diag[i];
            }
        }
    };

    let (x, stats) = cg_run(a, b, tol, max_iter, precond);
    (x, stats.iters)
}

/// Core preconditioned-CG iteration (f64), shared by all preconditioner paths.
/// `precond(r, z)` computes `z = M⁻¹ r`.
fn cg_run(
    a: &CsrMatrix,
    b: &[f32],
    tol: f32,
    max_iter: u32,
    precond: impl Fn(&[f64], &mut [f64]),
) -> (Vec<f32>, LinearStats) {
    let n = a.n;
    let spmv = |x: &[f64], out: &mut [f64]| {
        for (row, o) in out.iter_mut().enumerate() {
            let (s, e) = (a.row_ptr[row] as usize, a.row_ptr[row + 1] as usize);
            let mut sum = 0.0f64;
            for k in s..e {
                sum += a.values[k] as f64 * x[a.col_idx[k] as usize];
            }
            *o = sum;
        }
    };
    let dot = |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(a, b)| a * b).sum() };

    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = dot(&bf, &bf).sqrt().max(1e-30);

    let mut x = vec![0.0f64; n];
    let mut r = bf.clone(); // r = b - A*0
    let mut z = vec![0.0f64; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0f64; n];
    let mut iters = 0u32;
    let mut residual = dot(&r, &r).sqrt() / bnorm; // reresid at x=0

    while iters < max_iter {
        spmv(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 || pap.is_nan() {
            break;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        iters += 1;
        residual = dot(&r, &r).sqrt() / bnorm;
        if residual <= tol as f64 {
            break;
        }
        precond(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    (
        x.iter().map(|&v| v as f32).collect(),
        LinearStats { iters, residual },
    )
}

/// CPU CG as a [`LinearSolver`] (block-Jacobi default). Records solve stats.
#[derive(Default)]
pub struct CgCpu {
    stats: Cell<LinearStats>,
}

impl CgCpu {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LinearSolver for CgCpu {
    fn solve(&self, a: &CsrMatrix, b: &[f32], tol: f32, max_iter: u32) -> Vec<f32> {
        let block = a.n.is_multiple_of(3);
        let kind = if block {
            Precond::Block
        } else {
            Precond::Diagonal
        };
        let (x, iters) = cg_cpu_precond(a, b, tol, max_iter, kind);
        // Recompute the final relative residual for the stats channel.
        let residual = relative_residual(a, &x, b);
        self.stats.set(LinearStats { iters, residual });
        x
    }
    fn last_stats(&self) -> LinearStats {
        self.stats.get()
    }
}

/// Relative residual `‖b − A x‖ / ‖b‖` (f64), for the cg-converged invariant.
fn relative_residual(a: &CsrMatrix, x: &[f32], b: &[f32]) -> f64 {
    let ax = a.spmv_cpu(x);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..b.len() {
        let d = b[i] as f64 - ax[i] as f64;
        num += d * d;
        den += (b[i] as f64) * (b[i] as f64);
    }
    (num.sqrt()) / den.sqrt().max(1e-30)
}

/// Stateful additive-Schwarz CG. Caches the domain aggregation + value-gather
/// indices (derived from the sparsity, which is FIXED across a simulation) and
/// only refactors each domain's dense block per solve. This amortization is what
/// makes Schwarz competitive — the stateless [`Precond::Schwarz`] rebuilds the
/// aggregation every solve and was slower than block-Jacobi. Falls back to
/// block-Jacobi CG for non-3-DOF systems.
pub struct SchwarzCpu {
    domain: usize,
    cache: std::cell::RefCell<Option<CachedSchwarz>>,
    stats: Cell<LinearStats>,
}

struct CachedSchwarz {
    n: usize,
    col_idx: Vec<u32>, // sparsity fingerprint: rebuild topology iff this changes
    topo: SchwarzTopology,
}

impl SchwarzCpu {
    /// `domain` = target vertices per Schwarz subdomain (e.g. 16).
    pub fn new(domain: usize) -> Self {
        Self {
            domain: domain.max(1),
            cache: std::cell::RefCell::new(None),
            stats: Cell::new(LinearStats::default()),
        }
    }
}

impl LinearSolver for SchwarzCpu {
    fn solve(&self, a: &CsrMatrix, b: &[f32], tol: f32, max_iter: u32) -> Vec<f32> {
        if !a.n.is_multiple_of(3) {
            let x = solve_cg_cpu(a, b, tol, max_iter);
            self.stats.set(LinearStats {
                iters: 0,
                residual: relative_residual(a, &x, b),
            });
            return x;
        }
        // (Re)build the topology only when the sparsity pattern changes.
        {
            let mut cache = self.cache.borrow_mut();
            let stale = cache
                .as_ref()
                .is_none_or(|c| c.n != a.n || c.col_idx != a.col_idx);
            if stale {
                *cache = Some(CachedSchwarz {
                    n: a.n,
                    col_idx: a.col_idx.clone(),
                    topo: SchwarzTopology::build(a, self.domain),
                });
            }
        }
        let cache = self.cache.borrow();
        let topo = &cache.as_ref().unwrap().topo;
        let factors = topo.factorize(a); // per-solve refactor (values changed)
        let (x, stats) = cg_run(a, b, tol, max_iter, |r, z| topo.apply(&factors, r, z));
        self.stats.set(stats);
        x
    }
    fn last_stats(&self) -> LinearStats {
        self.stats.get()
    }
}

// ---------------------------------------------------------------------------
// Two-level (Galerkin) additive Schwarz — the reference's `levels=2` scheme.
// ---------------------------------------------------------------------------

/// Cached two-level additive-Schwarz topology. Applies
/// `z = S₀·r + w·C₁ᵀ·S₁·(C₁·r)` (the reference's additive form,
/// `PORTING_preconditioner.md`): level-0 per-domain solve `S₀`, plus a
/// coarse-grid correction where `C₁` restricts fine vertices to their domain
/// (coarse node), `A₁ = C₁·A₀·C₁ᵀ` is the Galerkin coarse operator, and `S₁` is
/// a Schwarz smoother over `A₁` (reusing [`SchwarzTopology`]). Everything
/// topological is cached (fixed by sparsity); only the dense factorizations +
/// coarse values rebuild per solve.
struct MultilevelSchwarz {
    l0: SchwarzTopology,
    /// Fine vertex → coarse node id (= its level-0 domain).
    agg0: Vec<u32>,
    n_coarse: usize,
    /// Coarse operator sparsity (`A₁`); values filled per solve.
    coarse_structure: CsrMatrix,
    /// Fine value index → coarse value index (Galerkin aggregation, precomputed).
    coarse_gather: Vec<usize>,
    /// Smoother topology over the coarse operator.
    l1: SchwarzTopology,
    weight: f64,
}

/// Per-solve factorizations for a [`MultilevelSchwarz`] (values change each solve).
struct MlPrepared {
    f0: Vec<Vec<f64>>,
    f1: Vec<Vec<f64>>,
}

impl MultilevelSchwarz {
    fn build(a: &CsrMatrix, domain: usize, weight: f64) -> Self {
        let l0 = SchwarzTopology::build(a, domain);
        let nv = a.n / 3;
        let mut agg0 = vec![0u32; nv];
        for (d, dom) in l0.domains.iter().enumerate() {
            for &v in dom {
                agg0[v] = d as u32;
            }
        }
        let n_coarse = l0.domains.len();
        // Coarse scalar dof of fine scalar row: 3·agg0[row/3] + row%3.
        let coarse_dof = |row: usize| 3 * agg0[row / 3] as usize + row % 3;
        // Build coarse sparsity from all fine entries (u64/usize keys — no 32-bit
        // overflow, unlike the reference's CUDA sort; §7.2).
        let mut trips: Vec<crate::csr::Triplet> = Vec::with_capacity(a.nnz());
        for row in 0..a.n {
            let cr = coarse_dof(row);
            let (s, e) = (a.row_ptr[row] as usize, a.row_ptr[row + 1] as usize);
            for k in s..e {
                trips.push((cr as u32, coarse_dof(a.col_idx[k] as usize) as u32, 0.0));
            }
        }
        let coarse_structure = CsrMatrix::from_triplets(3 * n_coarse, &trips);
        // (coarse_row, coarse_col) → flat index in coarse values.
        let mut idx_map: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::with_capacity(coarse_structure.nnz());
        for cr in 0..coarse_structure.n {
            let (s, e) = (
                coarse_structure.row_ptr[cr] as usize,
                coarse_structure.row_ptr[cr + 1] as usize,
            );
            for k in s..e {
                idx_map.insert((cr as u32, coarse_structure.col_idx[k]), k);
            }
        }
        let mut coarse_gather = vec![0usize; a.nnz()];
        for row in 0..a.n {
            let cr = coarse_dof(row) as u32;
            let (s, e) = (a.row_ptr[row] as usize, a.row_ptr[row + 1] as usize);
            for (cg, &col) in coarse_gather[s..e].iter_mut().zip(&a.col_idx[s..e]) {
                let cc = coarse_dof(col as usize) as u32;
                *cg = idx_map[&(cr, cc)];
            }
        }
        let l1 = SchwarzTopology::build(&coarse_structure, domain);
        MultilevelSchwarz {
            l0,
            agg0,
            n_coarse,
            coarse_structure,
            coarse_gather,
            l1,
            weight,
        }
    }

    /// Coarse operator with the current fine values aggregated in (Galerkin).
    fn coarse_matrix(&self, a: &CsrMatrix) -> CsrMatrix {
        let mut values = vec![0.0f32; self.coarse_structure.nnz()];
        for (k, &ci) in self.coarse_gather.iter().enumerate() {
            values[ci] += a.values[k];
        }
        CsrMatrix {
            n: self.coarse_structure.n,
            row_ptr: self.coarse_structure.row_ptr.clone(),
            col_idx: self.coarse_structure.col_idx.clone(),
            values,
        }
    }

    /// Factorize both levels for this solve (values fixed within a solve).
    fn prepare(&self, a: &CsrMatrix, coarse: &CsrMatrix) -> MlPrepared {
        MlPrepared {
            f0: self.l0.factorize(a),
            f1: self.l1.factorize(coarse),
        }
    }

    /// z = S₀r + w·C₁ᵀ S₁(C₁ r).
    fn apply(&self, prep: &MlPrepared, r: &[f64], z: &mut [f64]) {
        // Level 0.
        self.l0.apply(&prep.f0, r, z);
        // Restrict: rc[g] = Σ_{i∈g} r[i] (sum, matching the reference).
        let nv = r.len() / 3;
        let mut rc = vec![0.0f64; 3 * self.n_coarse];
        for i in 0..nv {
            let g = self.agg0[i] as usize;
            rc[3 * g] += r[3 * i];
            rc[3 * g + 1] += r[3 * i + 1];
            rc[3 * g + 2] += r[3 * i + 2];
        }
        // Coarse smoother.
        let mut ec = vec![0.0f64; 3 * self.n_coarse];
        self.l1.apply(&prep.f1, &rc, &mut ec);
        // Prolong + weighted add.
        for i in 0..nv {
            let g = self.agg0[i] as usize;
            z[3 * i] += self.weight * ec[3 * g];
            z[3 * i + 1] += self.weight * ec[3 * g + 1];
            z[3 * i + 2] += self.weight * ec[3 * g + 2];
        }
    }
}

// ---------------------------------------------------------------------------
// GPU seam: flattened, POD description of the 2-level Schwarz apply so engine-gpu
// can run it on-device (factorization stays on the CPU, per solve). The CPU owns
// the algorithm; the GPU consumes plain arrays. Kept here so it can reach the
// private `MultilevelSchwarz` internals.
// ---------------------------------------------------------------------------

/// Sparsity-fixed topology for a GPU 2-level Schwarz apply — flattened arrays,
/// uploaded once. Level-0 domains double as the restriction stencil (a coarse
/// node's members are exactly its level-0 domain).
#[derive(Clone, Debug)]
pub struct SchwarzGpuPlan {
    pub n: u32,            // fine dof count
    pub nv: u32,           // fine vertices
    pub n_coarse: u32,     // coarse nodes (= number of level-0 domains)
    pub n_coarse_dom: u32, // number of level-1 (coarse) domains
    pub weight: f32,
    pub l0_vert: Vec<u32>, // per-domain fine vertex ids, concatenated
    pub l0_off: Vec<u32>,  // domain boundaries, len n_coarse+1
    pub l1_vert: Vec<u32>, // per-coarse-domain coarse-vertex ids, concatenated
    pub l1_off: Vec<u32>,  // coarse-domain boundaries, len n_coarse_dom+1
    pub agg0: Vec<u32>,    // fine vertex → coarse node
}

/// Per-solve dense Cholesky factors for both levels (row-major `m×m` lower
/// triangles, concatenated). Values change each solve; re-uploaded then.
#[derive(Clone, Debug)]
pub struct SchwarzGpuFactors {
    pub l0_fact: Vec<f32>,
    pub l0_fact_off: Vec<u32>, // start of domain d's factor in l0_fact
    pub l0_m: Vec<u32>,        // m (= 3·|domain|) per domain
    pub l1_fact: Vec<f32>,
    pub l1_fact_off: Vec<u32>,
    pub l1_m: Vec<u32>,
}

/// Builds + factorizes a 2-level Schwarz preconditioner and exposes it in a
/// GPU-friendly form: a fixed [`SchwarzGpuPlan`] (upload once), per-solve
/// [`SchwarzGpuFactors`] (re-upload), and a CPU [`apply_cpu`](Self::apply_cpu)
/// oracle to verify the GPU port against. Topology is cached; only the factors
/// rebuild per solve — same amortization as [`SchwarzMlCpu`].
pub struct SchwarzGpuBuilder {
    ml: MultilevelSchwarz,
    n: usize,
}

impl SchwarzGpuBuilder {
    pub fn new(a: &CsrMatrix, domain: usize, weight: f32) -> Self {
        Self {
            ml: MultilevelSchwarz::build(a, domain.max(1), weight as f64),
            n: a.n,
        }
    }

    /// Whether this builder's topology fits `a` (same size + sparsity count).
    pub fn matches(&self, a: &CsrMatrix) -> bool {
        self.n == a.n && self.ml.l0.gather.len() == self.ml.l0.domains.len()
    }

    /// Fixed topology (upload once).
    pub fn plan(&self) -> SchwarzGpuPlan {
        let flat = |doms: &[Vec<usize>]| -> (Vec<u32>, Vec<u32>) {
            let mut vert = Vec::new();
            let mut off = Vec::with_capacity(doms.len() + 1);
            off.push(0u32);
            for d in doms {
                vert.extend(d.iter().map(|&v| v as u32));
                off.push(vert.len() as u32);
            }
            (vert, off)
        };
        let (l0_vert, l0_off) = flat(&self.ml.l0.domains);
        let (l1_vert, l1_off) = flat(&self.ml.l1.domains);
        SchwarzGpuPlan {
            n: self.n as u32,
            nv: (self.n / 3) as u32,
            n_coarse: self.ml.n_coarse as u32,
            n_coarse_dom: self.ml.l1.domains.len() as u32,
            weight: self.ml.weight as f32,
            l0_vert,
            l0_off,
            l1_vert,
            l1_off,
            agg0: self.ml.agg0.clone(),
        }
    }

    /// Factorize both levels for the current matrix values (re-upload per solve).
    pub fn factors(&self, a: &CsrMatrix) -> SchwarzGpuFactors {
        let coarse = self.ml.coarse_matrix(a);
        let flatten = |topo: &SchwarzTopology, mat: &CsrMatrix| {
            let facs = topo.factorize(mat);
            let mut fact = Vec::new();
            let mut off = Vec::with_capacity(facs.len());
            let mut ms = Vec::with_capacity(facs.len());
            for (d, f) in facs.iter().enumerate() {
                off.push(fact.len() as u32);
                ms.push((3 * topo.domains[d].len()) as u32);
                fact.extend(f.iter().map(|&v| v as f32));
            }
            (fact, off, ms)
        };
        let (l0_fact, l0_fact_off, l0_m) = flatten(&self.ml.l0, a);
        let (l1_fact, l1_fact_off, l1_m) = flatten(&self.ml.l1, &coarse);
        SchwarzGpuFactors {
            l0_fact,
            l0_fact_off,
            l0_m,
            l1_fact,
            l1_fact_off,
            l1_m,
        }
    }

    /// CPU apply `z = S₀r + w·C₁ᵀ S₁(C₁ r)` — the oracle the GPU port must match.
    pub fn apply_cpu(&self, a: &CsrMatrix, r: &[f32]) -> Vec<f32> {
        let coarse = self.ml.coarse_matrix(a);
        let prep = self.ml.prepare(a, &coarse);
        let rf: Vec<f64> = r.iter().map(|&v| v as f64).collect();
        let mut z = vec![0.0f64; a.n];
        self.ml.apply(&prep, &rf, &mut z);
        z.iter().map(|&v| v as f32).collect()
    }
}

/// Stateful **two-level Galerkin additive Schwarz** CG. Adds a coarse-grid
/// correction to [`SchwarzCpu`] — the reference's `levels=2` scheme and the
/// measured cloth iteration-count win one-level Schwarz misses
/// (docs/GAUGE_FINDINGS.md §3). Caches the multilevel topology (sparsity-fixed);
/// refactors per solve. Block-Jacobi stays the engine default.
pub struct SchwarzMlCpu {
    domain: usize,
    weight: f64,
    cache: std::cell::RefCell<Option<MlCached>>,
    stats: Cell<LinearStats>,
}

struct MlCached {
    n: usize,
    col_idx: Vec<u32>,
    ml: MultilevelSchwarz,
}

impl SchwarzMlCpu {
    /// `domain` = target vertices per subdomain; `weight` = coarse-correction
    /// damping (the reference's `PPF_SCHWARZ_WEIGHT`, default 1.0).
    pub fn new(domain: usize, weight: f32) -> Self {
        Self {
            domain: domain.max(1),
            weight: weight as f64,
            cache: std::cell::RefCell::new(None),
            stats: Cell::new(LinearStats::default()),
        }
    }
}

impl LinearSolver for SchwarzMlCpu {
    fn solve(&self, a: &CsrMatrix, b: &[f32], tol: f32, max_iter: u32) -> Vec<f32> {
        if !a.n.is_multiple_of(3) {
            let x = solve_cg_cpu(a, b, tol, max_iter);
            self.stats.set(LinearStats {
                iters: 0,
                residual: relative_residual(a, &x, b),
            });
            return x;
        }
        {
            let mut cache = self.cache.borrow_mut();
            let stale = cache
                .as_ref()
                .is_none_or(|c| c.n != a.n || c.col_idx != a.col_idx);
            if stale {
                *cache = Some(MlCached {
                    n: a.n,
                    col_idx: a.col_idx.clone(),
                    ml: MultilevelSchwarz::build(a, self.domain, self.weight),
                });
            }
        }
        let cache = self.cache.borrow();
        let ml = &cache.as_ref().unwrap().ml;
        let coarse = ml.coarse_matrix(a);
        let prep = ml.prepare(a, &coarse);
        let (x, stats) = cg_run(a, b, tol, max_iter, |r, z| ml.apply(&prep, r, z));
        self.stats.set(stats);
        x
    }
    fn last_stats(&self) -> LinearStats {
        self.stats.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_cg_solves_laplacian() {
        let a = CsrMatrix::laplacian_1d(64);
        let x_star: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();
        let b = a.spmv_cpu(&x_star);
        let x = solve_cg_cpu(&a, &b, 1e-8, 500);
        let err = x
            .iter()
            .zip(&x_star)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "cpu cg error {err}");
    }

    /// A system with strong intra-vertex (x/y/z) coupling and weak vertex-vertex
    /// coupling: block-Jacobi (inverts the 3×3 diagonal) should converge in far
    /// fewer iterations than scalar Jacobi, to the same solution.
    #[test]
    fn block_jacobi_cuts_iterations() {
        let v = 40usize; // vertices
        let n = 3 * v;
        let mut t = Vec::new();
        // Diagonal 3×3 block D with strong off-diagonal coupling (SPD).
        let d = [[6.0, 3.0, 2.0], [3.0, 6.0, 3.0], [2.0, 3.0, 6.0]];
        for i in 0..v {
            for (r, dr) in d.iter().enumerate() {
                for (c, &val) in dr.iter().enumerate() {
                    t.push(((3 * i + r) as u32, (3 * i + c) as u32, val));
                }
            }
            // Weak coupling to the next vertex (−I), keeps it SPD/diag-dominant.
            if i + 1 < v {
                for r in 0..3 {
                    t.push(((3 * i + r) as u32, (3 * (i + 1) + r) as u32, -1.0));
                    t.push(((3 * (i + 1) + r) as u32, (3 * i + r) as u32, -1.0));
                }
            }
        }
        let a = CsrMatrix::from_triplets(n, &t);
        let x_star: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() + 0.5).collect();
        let b = a.spmv_cpu(&x_star);

        let (xs, iters_scalar) = cg_cpu(&a, &b, 1e-8, 2000, false);
        let (xb, iters_block) = cg_cpu(&a, &b, 1e-8, 2000, true);

        let err = |x: &[f32]| {
            x.iter()
                .zip(&x_star)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(err(&xs) < 1e-3 && err(&xb) < 1e-3, "both must solve");
        assert!(
            iters_block < iters_scalar,
            "block-Jacobi should need fewer iters: block={iters_block}, scalar={iters_scalar}"
        );
    }

    /// On an elliptic (2D grid) vector-Laplacian, additive Schwarz (domain solves)
    /// should converge in fewer iterations than per-vertex block-Jacobi.
    #[test]
    fn schwarz_beats_block_jacobi_on_grid() {
        let (gx, gy) = (12usize, 12usize);
        let nv = gx * gy;
        let n = 3 * nv;
        let d = [[6.0, 1.0, 0.5], [1.0, 6.0, 1.0], [0.5, 1.0, 6.0]];
        let mut t = Vec::new();
        let idx = |i: usize, j: usize| j * gx + i;
        for j in 0..gy {
            for i in 0..gx {
                let v = idx(i, j);
                for (r, dr) in d.iter().enumerate() {
                    for (c, &val) in dr.iter().enumerate() {
                        t.push(((3 * v + r) as u32, (3 * v + c) as u32, val));
                    }
                }
                // −I to the 4 grid neighbors (elliptic coupling).
                let mut nbr = Vec::new();
                if i > 0 {
                    nbr.push(idx(i - 1, j));
                }
                if i + 1 < gx {
                    nbr.push(idx(i + 1, j));
                }
                if j > 0 {
                    nbr.push(idx(i, j - 1));
                }
                if j + 1 < gy {
                    nbr.push(idx(i, j + 1));
                }
                for w in nbr {
                    for r in 0..3 {
                        t.push(((3 * v + r) as u32, (3 * w + r) as u32, -1.0));
                    }
                }
            }
        }
        let a = CsrMatrix::from_triplets(n, &t);
        let x_star: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let b = a.spmv_cpu(&x_star);

        let (_, i_block) = cg_cpu_precond(&a, &b, 1e-8, 3000, Precond::Block);
        let (xs, i_schwarz) = cg_cpu_precond(&a, &b, 1e-8, 3000, Precond::Schwarz { domain: 16 });
        let err = xs
            .iter()
            .zip(&x_star)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "Schwarz must solve, err={err}");
        assert!(
            i_schwarz < i_block,
            "Schwarz should need fewer iters: schwarz={i_schwarz}, block={i_block}"
        );
    }

    #[test]
    fn multilevel_schwarz_solves_and_cuts_iterations() {
        // Elliptic 32×32 vector-Laplacian (diag 4.01, −1 neighbours) — big enough
        // that the coarse-grid correction has room. Same base domain (8) for both
        // solvers, so the 2-level's iteration reduction is purely the coarse
        // correction. (On tiny grids 1-level already converges in a handful of
        // iters and the additive coarse term over-counts — see GAUGE_FINDINGS §3.)
        let g = 32usize;
        let nv = g * g;
        let n = 3 * nv;
        let idx = |i: usize, j: usize| j * g + i;
        let mut t = Vec::new();
        for j in 0..g {
            for i in 0..g {
                let v = idx(i, j);
                for r in 0..3 {
                    t.push(((3 * v + r) as u32, (3 * v + r) as u32, 4.01f32));
                }
                for (di, dj) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let (ni, nj) = (i as i32 + di, j as i32 + dj);
                    if ni >= 0 && ni < g as i32 && nj >= 0 && nj < g as i32 {
                        let w = idx(ni as usize, nj as usize);
                        for r in 0..3 {
                            t.push(((3 * v + r) as u32, (3 * w + r) as u32, -1.0));
                        }
                    }
                }
            }
        }
        let a = CsrMatrix::from_triplets(n, &t);
        let x_star: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();
        let b = a.spmv_cpu(&x_star);

        let one = SchwarzCpu::new(8);
        let two = SchwarzMlCpu::new(8, 0.7);
        let x1 = one.solve(&a, &b, 1e-8, 5000);
        let x2 = two.solve(&a, &b, 1e-8, 5000);
        let _ = &x1;
        let (i1, i2) = (one.last_stats().iters, two.last_stats().iters);

        let err = |x: &[f32]| {
            x.iter()
                .zip(&x_star)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            err(&x2) < 1e-3,
            "2-level Schwarz must solve, err={}",
            err(&x2)
        );
        assert!(
            i2 < i1,
            "2-level coarse correction should cut iters: two={i2}, one={i1}"
        );
        // Cache reuse: a second solve on the same sparsity is identical.
        assert_eq!(x2, two.solve(&a, &b, 1e-8, 5000));
    }

    #[test]
    fn stateful_schwarz_matches_and_caches() {
        // Same grid system: the stateful SchwarzCpu solves it correctly, and a
        // second solve on the SAME sparsity reuses the cached topology (verified
        // by matching the direct cg_cpu_precond Schwarz result).
        let (gx, gy) = (10usize, 10usize);
        let nv = gx * gy;
        let n = 3 * nv;
        let mut t = Vec::new();
        let idx = |i: usize, j: usize| j * gx + i;
        for j in 0..gy {
            for i in 0..gx {
                let v = idx(i, j);
                for r in 0..3 {
                    t.push(((3 * v + r) as u32, (3 * v + r) as u32, 5.0));
                }
                for (di, dj) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let (ni, nj) = (i as i32 + di, j as i32 + dj);
                    if ni >= 0 && ni < gx as i32 && nj >= 0 && nj < gy as i32 {
                        let w = idx(ni as usize, nj as usize);
                        for r in 0..3 {
                            t.push(((3 * v + r) as u32, (3 * w + r) as u32, -1.0));
                        }
                    }
                }
            }
        }
        let a = CsrMatrix::from_triplets(n, &t);
        let x_star: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
        let b = a.spmv_cpu(&x_star);

        let s = SchwarzCpu::new(16);
        let x1 = s.solve(&a, &b, 1e-8, 3000); // builds topology
        let x2 = s.solve(&a, &b, 1e-8, 3000); // reuses cached topology
        assert_eq!(x1, x2, "cached solve must be identical");
        let err = x1
            .iter()
            .zip(&x_star)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "stateful Schwarz must solve: err={err}");
    }
}
