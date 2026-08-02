// SPDX-License-Identifier: GPL-2.0-or-later
//! Volumetric (tetrahedral) FEM elasticity — the first increment of the tet solver
//! the gauge's hard scenes (`plastic`, `trapped`, `needle`, `stack`) need.
//!
//! Energy is the reference's **StVK on principal stretches** (singular values of the
//! deformation gradient), ported from
//! `../crates/ppf-cts-solver/src/cpp/energy/model/{stvk.hpp,detsqr.hpp}` (Apache-2.0,
//! parent reference repo):
//!
//! `Ψ(σ) = μ·Σᵢ(σᵢ−1)⁴/4 + (λ/2)·(J−1)²`, `J = σ₀σ₁σ₂ = det F`
//!
//! and the element energy is `vol · Ψ`. Because Ψ depends only on the singular
//! values, it is **rotation-invariant by construction** (no corotational fix-up
//! needed), which the tests check directly.
//!
//! **Honest note on the reference source.** `detsqr.hpp` is internally inconsistent:
//! its `energy()` shifts each stretch (`a−1`) *before* multiplying, i.e.
//! `(σ₀−1)(σ₁−1)(σ₂−1)−1`, while its `make_diff_table3()` differentiates the
//! UNSHIFTED `J = σ₀σ₁σ₂`. Those are gradients of different functions. The diff
//! table is what drives the solver (and `J−1 = 0` at rest is the physically correct
//! volume term — the shifted form would want stretches near 2 at rest), so this port
//! follows the **gradient-consistent** `(J−1)²`. Verified by FD against our own
//! energy, so the port is self-consistent either way.
//!
//! ponytail: gradient and Hessian are central finite differences of the analytic
//! energy — the same policy bending shipped with (`bending::dihedral_grad`) before
//! its analytic upgrade. That keeps this increment verifiable now; the analytic
//! stretch-space chain rule (`∂Ψ/∂σ · ∂σ/∂F · ∂F/∂x`, needing the SVD derivative)
//! is the documented perf follow-up. The Hessian is eigen-projected to PSD so the
//! Newton system stays SPD for CG.

use glam::{Mat3, Vec3};

/// Which isotropic constitutive model a tet uses. Both share the reference's volume
/// term (`detsqr`, gradient-consistent `(J−1)²`) and differ ONLY in the deviatoric
/// part, so they share all of the eigensystem machinery:
/// * `StVK`  — `μ·Σ(σᵢ−1)⁴/4`  (reference `energy/model/stvk.hpp`)
/// * `Arap`  — `½μ·Σ(σᵢ−1)²`   (reference `energy/model/arap.hpp`)
///
/// The gauge's volumetric scenes use **ARAP** (`param_summary`: "model: (arap: …)"),
/// which is why both exist here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TetModel {
    #[default]
    StVK,
    Arap,
    /// Stable Neo-Hookean (`energy/model/snhk.hpp`):
    /// `Ψ = ½μ(I₂−3) − μ(I₃−1) + ½λ(I₃−1)²` with `I₂ = Σσᵢ²`, `I₃ = J`.
    ///
    /// Its volume term is IDENTICAL to the `detsqr` term StVK/ARAP already share, so only
    /// the deviatoric half differs — and unlike theirs it does NOT separate per stretch:
    /// `−μ(I₃−1)` couples all three, contributing off-diagonal Hessian entries. Those extra
    /// coupled terms are added explicitly in [`stvk_dpsi_dsigma`] / [`stvk_d2psi_dsigma2`].
    SnHk,
}

impl TetModel {
    /// Deviatoric energy density contribution of one stretch.
    fn dev_energy(self, s: f32, mu: f32) -> f32 {
        let d = s - 1.0;
        match self {
            TetModel::StVK => mu * d.powi(4) / 4.0,
            TetModel::Arap => 0.5 * mu * d * d,
            // Separable share of ½μ(I₂−3); the coupled −μ(I₃−1) is added by the caller.
            TetModel::SnHk => 0.5 * mu * (s * s - 1.0),
        }
    }
    /// `∂/∂σ` of [`Self::dev_energy`].
    fn dev_grad(self, s: f32, mu: f32) -> f32 {
        let d = s - 1.0;
        match self {
            TetModel::StVK => mu * d.powi(3),
            TetModel::Arap => mu * d,
            TetModel::SnHk => mu * s,
        }
    }
    /// `∂²/∂σ²` of [`Self::dev_energy`] (diagonal — the deviatoric part never couples
    /// different stretches).
    fn dev_hess(self, s: f32, mu: f32) -> f32 {
        let d = s - 1.0;
        match self {
            TetModel::StVK => 3.0 * mu * d * d,
            TetModel::Arap => mu,
            TetModel::SnHk => mu,
        }
    }
}

/// One tetrahedron with its rest state baked in.
#[derive(Clone, Copy, Debug)]
pub struct TetElement {
    /// Vertex indices `[v0, v1, v2, v3]`.
    pub v: [u32; 4],
    /// Inverse rest shape matrix `Dm⁻¹` (columns `p1−p0, p2−p0, p3−p0` at rest).
    pub dm_inv: Mat3,
    /// Rest volume (always positive).
    pub volume: f32,
    /// Constitutive model for this element.
    pub model: TetModel,
}

/// Deformation gradient `F = Ds · Dm⁻¹` for the 4 deformed positions.
pub fn deformation_gradient(p: &[Vec3; 4], dm_inv: &Mat3) -> Mat3 {
    let ds = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    ds * *dm_inv
}

/// Column `c` of a row-major-ish `Mat3` (glam `Mat3` is column-major, so this is
/// just axis access by index).
fn col(m: &Mat3, c: usize) -> Vec3 {
    match c {
        0 => m.x_axis,
        1 => m.y_axis,
        _ => m.z_axis,
    }
}

/// Symmetric 3×3 eigendecomposition by cyclic Jacobi: returns `(eigenvalues,
/// eigenvectors as columns)` with `A = V·diag(λ)·Vᵀ`. Same algorithm as the GPU
/// `sym_eig` device function, kept here so `engine-core` stays GPU-free.
pub fn sym_eig3(a_in: &Mat3) -> ([f32; 3], Mat3) {
    // Row-major scratch: a[3r + c].
    let mut a = [
        a_in.x_axis.x,
        a_in.y_axis.x,
        a_in.z_axis.x,
        a_in.x_axis.y,
        a_in.y_axis.y,
        a_in.z_axis.y,
        a_in.x_axis.z,
        a_in.y_axis.z,
        a_in.z_axis.z,
    ];
    let mut v = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for _ in 0..12 {
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[3 * p + q];
            if apq.abs() < 1e-20 {
                continue;
            }
            let (app, aqq) = (a[3 * p + p], a[3 * q + q]);
            let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
            let (c, s) = (phi.cos(), phi.sin());
            for k in 0..3 {
                let (akp, akq) = (a[3 * k + p], a[3 * k + q]);
                a[3 * k + p] = c * akp - s * akq;
                a[3 * k + q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[3 * p + k], a[3 * q + k]);
                a[3 * p + k] = c * apk - s * aqk;
                a[3 * q + k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[3 * k + p], v[3 * k + q]);
                v[3 * k + p] = c * vkp - s * vkq;
                v[3 * k + q] = s * vkp + c * vkq;
            }
        }
    }
    // v is row-major with eigenvectors in COLUMNS → build Mat3 from those columns.
    let vm = Mat3::from_cols(
        Vec3::new(v[0], v[3], v[6]),
        Vec3::new(v[1], v[4], v[7]),
        Vec3::new(v[2], v[5], v[8]),
    );
    ([a[0], a[4], a[8]], vm)
}

/// Singular values of `F`, descending — the principal stretches. `σ₂` carries the
/// sign of `det F`, so an inverted element reads `J = σ₀σ₁σ₂ < 0`.
pub fn principal_stretches(f: &Mat3) -> [f32; 3] {
    let (lam, _v) = sym_eig3(&(f.transpose() * *f));
    let mut s = [
        lam[0].max(0.0).sqrt(),
        lam[1].max(0.0).sqrt(),
        lam[2].max(0.0).sqrt(),
    ];
    s.sort_by(|x, y| y.partial_cmp(x).unwrap());
    if f.determinant() < 0.0 {
        s[2] = -s[2];
    }
    s
}

/// StVK strain-energy density on principal stretches (reference `stvk.hpp` +
/// gradient-consistent `detsqr.hpp` — see the module note). `mu`/`lambda` are the
/// Lamé-style parameters the reference passes as `mu`/`lmd`.
pub fn stvk_density(sigma: &[f32; 3], model: TetModel, mu: f32, lambda: f32) -> f32 {
    let dev: f32 = sigma.iter().map(|&s| model.dev_energy(s, mu)).sum();
    let j = sigma[0] * sigma[1] * sigma[2];
    // SNHk's coupled `−μ(I₃−1)`; zero for StVK/ARAP. With it, `dev` sums to
    // `½μ(I₂−3)` and the whole expression is snhk.hpp's `energy` exactly.
    let cpl = if matches!(model, TetModel::SnHk) {
        -mu * (j - 1.0)
    } else {
        0.0
    };
    dev + cpl + 0.5 * lambda * (j - 1.0) * (j - 1.0)
}

/// Full SVD of `F`: returns `(U, sigma, V)` with `F = U·diag(sigma)·Vᵀ`, both `U`
/// and `V` **proper rotations** (`det = +1`), and `sigma[2]` carrying the sign of
/// `det F` so an inverted element is representable (Irving et al. 2004). Columns of
/// `U`/`V` are the singular vectors. Same construction as the verified GPU
/// `svd.wgsl`, kept CPU-side so `engine-core` stays GPU-free.
pub fn svd3(f: &Mat3) -> (Mat3, [f32; 3], Mat3) {
    // Eigen-decompose FᵀF; its eigenvectors are V, eigenvalues σ².
    let ata = f.transpose() * *f;
    let (lam, vmat) = sym_eig3(&ata);
    // Sort descending by eigenvalue.
    let mut ord = [0usize, 1, 2];
    ord.sort_by(|&a, &b| lam[b].partial_cmp(&lam[a]).unwrap());
    let mut v = Mat3::from_cols(
        col(&vmat, ord[0]).normalize_or_zero(),
        col(&vmat, ord[1]).normalize_or_zero(),
        col(&vmat, ord[2]).normalize_or_zero(),
    );
    // Make V a proper rotation.
    if v.determinant() < 0.0 {
        v.z_axis = -v.z_axis;
    }
    let mut sigma = [
        lam[ord[0]].max(0.0).sqrt(),
        lam[ord[1]].max(0.0).sqrt(),
        lam[ord[2]].max(0.0).sqrt(),
    ];

    // U columns: uᵢ = F·vᵢ/σᵢ, with orthonormal completion for degenerate σ.
    let eps = 1.0e-9 * sigma[0].max(1.0);
    let mut u0 = *f * v.x_axis;
    let mut u1 = *f * v.y_axis;
    u0 = if sigma[0] > eps {
        u0 / sigma[0]
    } else {
        Vec3::X
    };
    u0 = u0.normalize_or_zero();
    u1 = if sigma[1] > eps {
        u1 / sigma[1]
    } else {
        // Any unit vector orthogonal to u0.
        let t = if u0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        t - t.dot(u0) * u0
    };
    u1 = (u1 - u1.dot(u0) * u0).normalize_or_zero();
    let u2 = u0.cross(u1);
    let mut u = Mat3::from_cols(u0, u1, u2);
    // `u2 = u0×u1` already forces det U = +1. If that disagrees with F's
    // orientation, the third singular value is negative — which is precisely how an
    // inverted element is encoded while keeping both frames proper rotations.
    if (*f * v.z_axis).dot(u2) < 0.0 {
        sigma[2] = -sigma[2];
    }
    u = Mat3::from_cols(u.x_axis, u.y_axis, u.z_axis);
    (u, sigma, v)
}

/// `∂Ψ/∂σ` for [`stvk_density`] — the reference's `stvk.hpp` gradient plus the
/// gradient-consistent `detsqr.hpp` volume term (see the module note):
/// `μ(σᵢ−1)³ + λ(J−1)·∂J/∂σᵢ`, where `∂J/∂σ₀ = σ₁σ₂` etc.
pub fn stvk_dpsi_dsigma(sigma: &[f32; 3], model: TetModel, mu: f32, lambda: f32) -> [f32; 3] {
    let j = sigma[0] * sigma[1] * sigma[2];
    let dj = [
        sigma[1] * sigma[2],
        sigma[0] * sigma[2],
        sigma[0] * sigma[1],
    ];
    // SNHk's `−μ(I₃−1)` contributes `−μ·∂J/∂σᵢ`; StVK/ARAP contribute nothing here.
    let cpl = if matches!(model, TetModel::SnHk) {
        -mu
    } else {
        0.0
    };
    [
        model.dev_grad(sigma[0], mu) + lambda * (j - 1.0) * dj[0] + cpl * dj[0],
        model.dev_grad(sigma[1], mu) + lambda * (j - 1.0) * dj[1] + cpl * dj[1],
        model.dev_grad(sigma[2], mu) + lambda * (j - 1.0) * dj[2] + cpl * dj[2],
    ]
}

/// **Analytic** element energy gradient — ~2 orders of magnitude cheaper than the
/// finite-difference path in [`tet_element_assembly`] (one SVD instead of 24 energy
/// evaluations), and an independent derivation, so the two cross-check each other.
///
/// For an isotropic `Ψ(σ)` the first Piola-Kirchhoff stress is
/// `P = U·diag(∂Ψ/∂σ)·Vᵀ`. With `F = Ds·Dm⁻¹` and `E = vol·Ψ`, the chain rule gives
/// `∂E/∂Ds = vol·P·Dm⁻ᵀ`, whose three columns are the forces on `p1,p2,p3`; the
/// force on `p0` is minus their sum (translation invariance).
pub fn tet_gradient_analytic(p: &[Vec3; 4], el: &TetElement, mu: f32, lambda: f32) -> [Vec3; 4] {
    let f = deformation_gradient(p, &el.dm_inv);
    let (u, sigma, v) = svd3(&f);
    let d = stvk_dpsi_dsigma(&sigma, el.model, mu, lambda);
    // P = U diag(d) Vᵀ
    let ud = Mat3::from_cols(u.x_axis * d[0], u.y_axis * d[1], u.z_axis * d[2]);
    let pk1 = ud * v.transpose();
    // ∂E/∂Ds = vol · P · Dm⁻ᵀ
    let h = pk1 * el.dm_inv.transpose() * el.volume;
    let (g1, g2, g3) = (h.x_axis, h.y_axis, h.z_axis);
    [-(g1 + g2 + g3), g1, g2, g3]
}

/// Element energy `vol · Ψ(σ(F(x)))`.
pub fn tet_energy(p: &[Vec3; 4], el: &TetElement, mu: f32, lambda: f32) -> f32 {
    let f = deformation_gradient(p, &el.dm_inv);
    el.volume * stvk_density(&principal_stretches(&f), el.model, mu, lambda)
}

/// `∂²Ψ/∂σ²` for [`stvk_density`] — the 3×3 stretch-space Hessian (the reference's
/// `DiffTable3::d2ed2a`), ported from `stvk.hpp` (`diag(3μ(σᵢ−1)²)`) plus
/// `detsqr.hpp`'s `make_diff_table3`:
/// `∂²/∂σᵢ² = λ(σⱼσₖ)²` and `∂²/∂σᵢ∂σⱼ = λσₖ(2J−1)`.
/// Row-major 3×3.
pub fn stvk_d2psi_dsigma2(sigma: &[f32; 3], model: TetModel, mu: f32, lambda: f32) -> [f32; 9] {
    let (a, b, c) = (sigma[0], sigma[1], sigma[2]);
    let j = a * b * c;
    let t = 2.0 * j - 1.0;
    let mut h = [0.0f32; 9];
    // Deviatoric (StVK) part: diagonal only.
    h[0] = model.dev_hess(a, mu) + lambda * (b * c) * (b * c);
    h[4] = model.dev_hess(b, mu) + lambda * (a * c) * (a * c);
    h[8] = model.dev_hess(c, mu) + lambda * (a * b) * (a * b);
    // Volume-term cross derivatives.
    h[1] = lambda * c * t;
    h[3] = h[1];
    h[2] = lambda * b * t;
    h[6] = h[2];
    h[5] = lambda * a * t;
    h[7] = h[5];
    // SNHk's `−μ(I₃−1)` adds `−μ·∂²J/∂σᵢ∂σⱼ` off the diagonal (`∂²J/∂a∂b = c`, etc.);
    // its second derivative w.r.t. a single stretch is zero, so the diagonal is untouched.
    if matches!(model, TetModel::SnHk) {
        h[1] -= mu * c;
        h[3] = h[1];
        h[2] -= mu * b;
        h[6] = h[2];
        h[5] -= mu * a;
        h[7] = h[5];
    }
    h
}

/// Row-major `vec` of a `Mat3`: `v[3r + c] = M[r][c]`. Used consistently for both the
/// 9×9 `∂²Ψ/∂F²` and the `∂F/∂x` map, so the contraction is convention-independent.
fn vec9(m: &Mat3) -> [f32; 9] {
    [
        m.x_axis.x, m.y_axis.x, m.z_axis.x, // row 0
        m.x_axis.y, m.y_axis.y, m.z_axis.y, // row 1
        m.x_axis.z, m.y_axis.z, m.z_axis.z, // row 2
    ]
}

/// `∂²Ψ/∂F²` as a row-major 9×9, via the **isotropic eigensystem** — a port of the
/// reference `eigenanalysis::compute_hessian(DiffTable3, Svd3x3)`
/// (`../crates/ppf-cts-solver/src/cpp/eigenanalysis/eigenanalysis.cu`, Apache-2.0).
///
/// For an isotropic Ψ the Hessian has 9 known eigenpairs, built from the SVD:
/// * 3 **twist** modes `U·Qᵢ·Vᵀ` (antisymmetric, `1/√2`) with eigenvalues
///   `(Ψ_i + Ψ_j)/(σᵢ + σⱼ)`;
/// * 3 **flip** modes (symmetric, `1/√2`) with `(Ψ_i − Ψ_j)/(σᵢ − σⱼ)`, falling back
///   to `½(H_ii + H_jj) − ½(H_ij + H_ji)` when `σᵢ ≈ σⱼ` (the removable singularity);
/// * 3 **scaling** modes, diagonal in the eigenbasis of `∂²Ψ/∂σ²`, with that
///   matrix's eigenvalues.
///
/// Every eigenvalue is clamped at 0 exactly as the reference does, so the result is
/// **PSD by construction** — no separate projection needed for the elastic part.
/// The 9 predicted eigenpairs of `∂²Ψ/∂F²` in the SVD frame: the mode matrices
/// `Qₖ` (so the actual eigenmatrix is `U·Qₖ·Vᵀ`) and their **unclamped**
/// eigenvalues. Split out so the implementation and
/// `predicted_modes_are_true_eigenpairs_of_d2psi_df2` share one definition — the
/// test verifies these against a finite-difference Hessian.
pub fn predicted_eigenpairs(
    sigma: &[f32; 3],
    model: TetModel,
    mu: f32,
    lambda: f32,
    eps: f32,
) -> ([Mat3; 9], [f32; 9]) {
    let deda = stvk_dpsi_dsigma(sigma, model, mu, lambda);
    let d2 = stvk_d2psi_dsigma2(sigma, model, mu, lambda);
    let d2m = Mat3::from_cols(
        Vec3::new(d2[0], d2[3], d2[6]),
        Vec3::new(d2[1], d2[4], d2[7]),
        Vec3::new(d2[2], d2[5], d2[8]),
    );
    let (eigval, eigvec) = sym_eig3(&d2m);

    const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let m = |rows: [[f32; 3]; 3]| {
        Mat3::from_cols(
            Vec3::new(rows[0][0], rows[1][0], rows[2][0]),
            Vec3::new(rows[0][1], rows[1][1], rows[2][1]),
            Vec3::new(rows[0][2], rows[1][2], rows[2][2]),
        )
    };
    let mut q = [Mat3::ZERO; 9];
    // Twist (antisymmetric) modes, pairs (0,1) (0,2) (1,2).
    q[0] = m([[0.0, R, 0.0], [-R, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    q[1] = m([[0.0, 0.0, R], [0.0, 0.0, 0.0], [-R, 0.0, 0.0]]);
    q[2] = m([[0.0, 0.0, 0.0], [0.0, 0.0, R], [0.0, -R, 0.0]]);
    // Flip (symmetric) modes, same pairs.
    q[3] = m([[0.0, R, 0.0], [R, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    q[4] = m([[0.0, 0.0, R], [0.0, 0.0, 0.0], [R, 0.0, 0.0]]);
    q[5] = m([[0.0, 0.0, 0.0], [0.0, 0.0, R], [0.0, R, 0.0]]);
    // Scaling modes: diagonal in the eigenbasis of ∂²Ψ/∂σ².
    for (k, qk) in q.iter_mut().enumerate().skip(6) {
        let e = col(&eigvec, k - 6);
        *qk = Mat3::from_cols(
            Vec3::new(e.x, 0.0, 0.0),
            Vec3::new(0.0, e.y, 0.0),
            Vec3::new(0.0, 0.0, e.z),
        );
    }

    let (a, b, c) = (sigma[0], sigma[1], sigma[2]);
    let pair = |num: f32, den: f32, fallback: f32| {
        if den.abs() > eps {
            num / den
        } else {
            fallback
        }
    };
    // Removable singularity at σᵢ ≈ σⱼ (reference `eigenanalysis.cu`).
    let fb = |i: usize, j: usize| {
        0.5 * (d2[i * 3 + i] + d2[j * 3 + j]) - 0.5 * (d2[i * 3 + j] + d2[j * 3 + i])
    };
    let lam = [
        (deda[0] + deda[1]) / (a + b),
        (deda[0] + deda[2]) / (a + c),
        (deda[1] + deda[2]) / (b + c),
        pair(deda[0] - deda[1], a - b, fb(0, 1)),
        pair(deda[0] - deda[2], a - c, fb(0, 2)),
        pair(deda[1] - deda[2], b - c, fb(1, 2)),
        eigval[0],
        eigval[1],
        eigval[2],
    ];
    (q, lam)
}

/// `∂²Ψ/∂F²` as a row-major 9×9, via the **isotropic eigensystem** — a port of the
/// reference `eigenanalysis::compute_hessian(DiffTable3, Svd3x3)`
/// (`../crates/ppf-cts-solver/src/cpp/eigenanalysis/eigenanalysis.cu`, Apache-2.0).
///
/// Built from [`predicted_eigenpairs`] as `Σ max(λₖ,0)·qₖqₖᵀ`. The clamp is the
/// reference's own `fmaxf(0,·)`: this energy is **non-convex** (at realistic
/// deformations several flip/scaling eigenvalues are genuinely negative), so the
/// shipped Hessian is deliberately a PSD FILTER of the true one, which is what keeps
/// the Newton system solvable by CG. It therefore does NOT equal the true Hessian
/// wherever the energy is non-convex — see
/// `predicted_modes_are_true_eigenpairs_of_d2psi_df2` for how correctness is checked.
pub fn stvk_d2psi_df2(
    u: &Mat3,
    sigma: &[f32; 3],
    v: &Mat3,
    model: TetModel,
    mu: f32,
    lambda: f32,
    eps: f32,
) -> [f32; 81] {
    let (q, lam) = predicted_eigenpairs(sigma, model, mu, lambda, eps);
    let mut out = [0.0f32; 81];
    for (k, qk) in q.iter().enumerate() {
        let l = lam[k].max(0.0);
        if l == 0.0 {
            continue;
        }
        let qv = vec9(&(*u * *qk * v.transpose()));
        for r in 0..9 {
            for cc in 0..9 {
                out[r * 9 + cc] += l * qv[r] * qv[cc];
            }
        }
    }
    out
}

/// `∂F/∂x` as a row-major 9×12 map. `F = Ds·Dm⁻¹` is **linear** in `x`, so this is
/// constant per element and there is no second-order term from the chain rule:
/// `∂²E/∂x² = vol · Bᵀ·(∂²Ψ/∂F²)·B`.
fn dfdx(dm_inv: &Mat3) -> [f32; 108] {
    let at = |m: &Mat3, r: usize, c: usize| col(m, c)[r];
    let mut b = [0.0f32; 108]; // 9 × 12
    for i in 0..3 {
        for j in 0..3 {
            let row = 3 * i + j; // vec index of F[i][j]
            let mut sum = 0.0f32;
            for k in 0..3 {
                let d = at(dm_inv, k, j);
                b[row * 12 + 3 * (k + 1) + i] = d;
                sum += d;
            }
            b[row * 12 + i] = -sum; // vertex 0
        }
    }
    b
}

/// **Fully analytic** element Hessian: `vol · Bᵀ·(∂²Ψ/∂F²)·B`, using the isotropic
/// eigensystem ([`stvk_d2psi_df2`]) — ONE SVD, no finite differences. PSD by
/// construction (the eigenvalue clamping happens inside `stvk_d2psi_df2`).
pub fn tet_hessian_analytic(p: &[Vec3; 4], el: &TetElement, mu: f32, lambda: f32) -> [f32; 144] {
    let f = deformation_gradient(p, &el.dm_inv);
    let (u, sigma, v) = svd3(&f);
    let a9 = stvk_d2psi_df2(&u, &sigma, &v, el.model, mu, lambda, 1.0e-6);
    let b = dfdx(&el.dm_inv);
    // H = vol · Bᵀ A B  (12×12), via (A·B) then Bᵀ·(A·B).
    let mut ab = [0.0f32; 108]; // 9 × 12
    for r in 0..9 {
        for cc in 0..12 {
            let mut acc = 0.0f32;
            for k in 0..9 {
                acc += a9[r * 9 + k] * b[k * 12 + cc];
            }
            ab[r * 12 + cc] = acc;
        }
    }
    let mut h = [0.0f32; 144];
    for r in 0..12 {
        for cc in 0..12 {
            let mut acc = 0.0f32;
            for k in 0..9 {
                acc += b[k * 12 + r] * ab[k * 12 + cc];
            }
            h[r * 12 + cc] = el.volume * acc;
        }
    }
    h
}

/// Central-finite-difference energy gradient — kept as the INDEPENDENT oracle for
/// [`tet_gradient_analytic`]. Not used by the solver (24 energy evaluations vs one
/// SVD), but the two derivations must agree, and that agreement is what verifies the
/// analytic Piola-stress chain rule. Do not "optimize" this into a call to the
/// analytic path: that would turn the cross-check into a tautology.
pub fn tet_gradient_fd(p: &[Vec3; 4], el: &TetElement, mu: f32, lambda: f32) -> [Vec3; 4] {
    let scale = p.iter().map(|q| q.length()).sum::<f32>().max(1.0);
    let h = 1e-3 * scale;
    let inv = 1.0 / (2.0 * h);
    let mut g = [Vec3::ZERO; 4];
    for vi in 0..4 {
        for c in 0..3 {
            let (mut pp, mut pm) = (*p, *p);
            pp[vi][c] += h;
            pm[vi][c] -= h;
            g[vi][c] = (tet_energy(&pp, el, mu, lambda) - tet_energy(&pm, el, mu, lambda)) * inv;
        }
    }
    g
}

/// Energy, gradient, and PSD Hessian for one element — **fully analytic**.
///
/// Gradient: [`tet_gradient_analytic`] (Piola stress). Hessian:
/// [`tet_hessian_analytic`] (isotropic eigensystem). One SVD total, no finite
/// differences anywhere in the solver path. The Hessian is PSD by construction
/// (eigenvalues clamped at 0 exactly as the reference does), so no separate
/// `project_psd_12` call is needed.
pub fn tet_element_assembly(
    p: &[Vec3; 4],
    el: &TetElement,
    mu: f32,
    lambda: f32,
) -> (f32, [Vec3; 4], [f32; 144]) {
    (
        tet_energy(p, el, mu, lambda),
        tet_gradient_analytic(p, el, mu, lambda),
        tet_hessian_analytic(p, el, mu, lambda),
    )
}

/// **Plasticity** — a port of the reference `plasticity::update_tet_plasticity`
/// (`../crates/ppf-cts-solver/src/cpp/plasticity/plasticity.cu`, Apache-2.0).
///
/// Permanently relaxes the element's REST state toward the current deformation, so
/// stretch held beyond a dead zone stops being stored as elastic energy. With
/// `α = 1 − exp(−plasticity·dt)`, each principal stretch `σₖ` whose deviation
/// `|σₖ − 1|` exceeds `threshold` is pulled a fraction `α` of the way toward the
/// nearest dead-zone edge (`1 ± threshold`); the rest shape is then rewritten as
/// `Dm⁻¹ ← Ds⁻¹·F_new`, which is exactly the reference's `inv_rest = dx.inverse() *
/// F_new` (it follows from `F = Ds·Dm⁻¹`).
///
/// Returns `true` if the rest state changed. Deformation INSIDE the dead zone leaves
/// the element untouched — that is what makes this creep rather than pure damping.
///
/// Note (matching the reference): the stored rest `volume` is deliberately NOT
/// updated, only `dm_inv`. The reference likewise rewrites `inv_rest[i]` alone and
/// leaves `TetProp::volume` as the original rest volume.
pub fn apply_tet_plasticity(
    el: &mut TetElement,
    p: &[Vec3; 4],
    plasticity: f32,
    threshold: f32,
    dt: f32,
) -> bool {
    if plasticity <= 0.0 {
        return false;
    }
    let alpha = 1.0 - (-plasticity * dt).exp();
    let ds = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    let f = ds * el.dm_inv;
    let (u, sigma, v) = svd3(&f);

    let mut s_new = sigma;
    let mut changed = false;
    for k in 0..3 {
        if (sigma[k] - 1.0).abs() > threshold {
            let target = if sigma[k] < 1.0 {
                1.0 - threshold
            } else {
                1.0 + threshold
            };
            s_new[k] = sigma[k] + alpha * (target - sigma[k]);
            changed = true;
        }
    }
    if !changed {
        return false;
    }
    // F_new = U diag(S_new) Vᵀ, then Dm⁻¹ ← Ds⁻¹ F_new.
    let us = Mat3::from_cols(
        u.x_axis * s_new[0],
        u.y_axis * s_new[1],
        u.z_axis * s_new[2],
    );
    let f_new = us * v.transpose();
    let det = ds.determinant();
    if det.abs() < 1.0e-20 {
        return false; // degenerate current shape; leave rest alone
    }
    el.dm_inv = ds.inverse() * f_new;
    true
}

/// Build tet elements from an index list and rest positions. Tets whose rest volume
/// is degenerate (|vol| below `min_volume`) are skipped. Winding is normalized so
/// every stored element has a positive rest volume.
pub fn build_tets(
    tets: &[[u32; 4]],
    rest: &[Vec3],
    min_volume: f32,
    model: TetModel,
) -> Vec<TetElement> {
    let mut out = Vec::new();
    for t in tets {
        let p = [
            rest[t[0] as usize],
            rest[t[1] as usize],
            rest[t[2] as usize],
            rest[t[3] as usize],
        ];
        let dm = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
        let det = dm.determinant();
        if det.abs() <= min_volume * 6.0 {
            continue; // degenerate / sliver
        }
        // Flip winding so the rest volume is positive.
        let (v, dm) = if det < 0.0 {
            (
                [t[0], t[2], t[1], t[3]],
                Mat3::from_cols(p[2] - p[0], p[1] - p[0], p[3] - p[0]),
            )
        } else {
            (*t, dm)
        };
        out.push(TetElement {
            v,
            dm_inv: dm.inverse(),
            volume: dm.determinant().abs() / 6.0,
            model,
        });
    }
    out
}

/// The four faces of a tet `[v0,v1,v2,v3]` (positive rest volume), each wound so its
/// normal `(b−a)×(c−a)` points **outward** — away from the opposite vertex. The
/// orientation is not asserted from memory: `tet_faces_are_outward_oriented` checks
/// every face numerically against the opposite vertex.
pub fn tet_faces(v: &[u32; 4]) -> [[u32; 3]; 4] {
    [
        [v[0], v[2], v[1]], // opposite v3
        [v[0], v[1], v[3]], // opposite v2
        [v[0], v[3], v[2]], // opposite v1
        [v[1], v[2], v[3]], // opposite v0
    ]
}

/// Boundary surface of a tet mesh: the faces belonging to **exactly one** tet, kept
/// in their outward-oriented winding. Interior faces (shared by two tets) cancel and
/// are dropped. Output is deterministic (sorted by vertex key).
///
/// This is what a volumetric body needs in order to take part in surface contact —
/// self-collision and tet↔cloth coupling both act on boundary triangles, not on the
/// interior tetrahedra.
pub fn tet_surface(tets: &[TetElement]) -> Vec<[u32; 3]> {
    use std::collections::HashMap;
    let key = |f: &[u32; 3]| {
        let mut k = *f;
        k.sort_unstable();
        (k[0], k[1], k[2])
    };
    // key -> (count, first oriented occurrence)
    let mut seen: HashMap<(u32, u32, u32), (u32, [u32; 3])> = HashMap::new();
    for el in tets {
        for f in tet_faces(&el.v) {
            let e = seen.entry(key(&f)).or_insert((0, f));
            e.0 += 1;
        }
    }
    let mut out: Vec<((u32, u32, u32), [u32; 3])> = seen
        .into_iter()
        .filter(|(_k, (count, _f))| *count == 1)
        .map(|(k, (_c, f))| (k, f))
        .collect();
    out.sort_unstable_by_key(|(k, _)| *k);
    out.into_iter().map(|(_, f)| f).collect()
}

/// Volume enclosed by a closed, outward-oriented triangle surface, by the divergence
/// theorem: `V = Σ (1/6)·(a × b)·c` over faces. Used to verify that
/// [`tet_surface`] produced a correctly-oriented closed surface — it must equal the
/// sum of the tets' volumes.
pub fn enclosed_volume(surface: &[[u32; 3]], x: &[Vec3]) -> f32 {
    surface
        .iter()
        .map(|f| {
            let (a, b, c) = (x[f[0] as usize], x[f[1] as usize], x[f[2] as usize]);
            a.cross(b).dot(c) / 6.0
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single unit tet at rest.
    fn unit_tet() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        (
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
            vec![[0, 1, 2, 3]],
        )
    }

    fn pts(rest: &[Vec3], el: &TetElement) -> [Vec3; 4] {
        [
            rest[el.v[0] as usize],
            rest[el.v[1] as usize],
            rest[el.v[2] as usize],
            rest[el.v[3] as usize],
        ]
    }

    #[test]
    fn rest_state_has_zero_energy_and_gradient() {
        let (rest, tets) = unit_tet();
        let els = build_tets(&tets, &rest, 1e-9, TetModel::StVK);
        assert_eq!(els.len(), 1);
        assert!((els[0].volume - 1.0 / 6.0).abs() < 1e-6, "unit tet volume");
        let p = pts(&rest, &els[0]);
        // At rest F = I, so σ = (1,1,1), J = 1 ⇒ Ψ = 0.
        let (e, g, _h) = tet_element_assembly(&p, &els[0], 100.0, 200.0);
        assert!(e.abs() < 1e-6, "rest energy not zero: {e}");
        for (i, gi) in g.iter().enumerate() {
            assert!(gi.length() < 1e-3, "rest gradient nonzero at {i}: {gi:?}");
        }
    }

    /// Rotation invariance: energy depends only on singular values, so a rigid
    /// rotation + translation of the deformed state must not change it. This is the
    /// property that makes the stretch-space formulation corotational for free.
    #[test]
    fn energy_is_rigid_motion_invariant() {
        let (rest, tets) = unit_tet();
        let els = build_tets(&tets, &rest, 1e-9, TetModel::StVK);
        let el = els[0];
        // A genuinely deformed configuration.
        let deformed = [
            Vec3::new(0.02, -0.01, 0.0),
            Vec3::new(1.30, 0.05, -0.02),
            Vec3::new(-0.03, 0.80, 0.04),
            Vec3::new(0.01, -0.02, 1.15),
        ];
        let base = tet_energy(&deformed, &el, 100.0, 200.0);
        assert!(base > 1e-4, "test config should be deformed: {base}");
        let rot = Mat3::from_euler(glam::EulerRot::XYZ, 0.7, -1.1, 0.35);
        let shift = Vec3::new(3.0, -2.0, 5.0);
        let moved: [Vec3; 4] = std::array::from_fn(|i| rot * deformed[i] + shift);
        let after = tet_energy(&moved, &el, 100.0, 200.0);
        assert!(
            (after - base).abs() < 1e-3 * base.max(1.0),
            "energy not rigid-invariant: {base} -> {after}"
        );
    }

    /// Volume response: compression and expansion both cost energy, and the
    /// volume term grows with `lambda`.
    #[test]
    fn volume_change_costs_energy() {
        let (rest, tets) = unit_tet();
        let el = build_tets(&tets, &rest, 1e-9, TetModel::StVK)[0];
        let p0 = pts(&rest, &el);
        let scale = |s: f32| -> [Vec3; 4] { std::array::from_fn(|i| p0[i] * s) };
        let (mu, lam) = (10.0f32, 500.0f32);
        let e_rest = tet_energy(&p0, &el, mu, lam);
        let e_shrink = tet_energy(&scale(0.8), &el, mu, lam);
        let e_grow = tet_energy(&scale(1.2), &el, mu, lam);
        assert!(e_shrink > e_rest, "compression must cost energy");
        assert!(e_grow > e_rest, "expansion must cost energy");
        // Stiffer volume term ⇒ strictly more energy for the same compression.
        let e_soft = tet_energy(&scale(0.8), &el, mu, 50.0);
        assert!(
            e_shrink > e_soft,
            "higher lambda must penalize volume change more: {e_shrink} vs {e_soft}"
        );
    }

    /// The FD gradient must actually descend the energy, and the assembled 12×12
    /// Hessian must be symmetric and PSD (it is eigen-projected).
    #[test]
    fn gradient_descends_and_hessian_is_psd() {
        let (rest, tets) = unit_tet();
        let el = build_tets(&tets, &rest, 1e-9, TetModel::StVK)[0];
        let deformed = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.25, 0.03, 0.0),
            Vec3::new(0.0, 0.90, 0.02),
            Vec3::new(0.02, 0.0, 1.10),
        ];
        let (mu, lam) = (100.0f32, 200.0f32);
        let (e0, g, h) = tet_element_assembly(&deformed, &el, mu, lam);
        assert!(e0 > 0.0);
        // Step along -grad must reduce the energy.
        let gnorm: f32 = g.iter().map(|v| v.length_squared()).sum::<f32>().sqrt();
        assert!(gnorm > 1e-4, "expected a nonzero gradient");
        let step = 1e-4 / gnorm;
        let moved: [Vec3; 4] = std::array::from_fn(|i| deformed[i] - g[i] * step);
        let e1 = tet_energy(&moved, &el, mu, lam);
        assert!(e1 < e0, "gradient did not descend: {e0} -> {e1}");
        // Symmetric.
        for a in 0..12 {
            for b in 0..12 {
                assert!(
                    (h[a * 12 + b] - h[b * 12 + a]).abs() < 1e-2 * h[a * 12 + b].abs().max(1.0),
                    "Hessian not symmetric at ({a},{b})"
                );
            }
        }
        // PSD after projection: xᵀHx ≥ 0.
        for t in 0..12u64 {
            let x: [f32; 12] = std::array::from_fn(|k| {
                (((t * 7919 + k as u64 * 104729) % 200) as f32) / 100.0 - 1.0
            });
            let mut q = 0.0f32;
            for a in 0..12 {
                for b in 0..12 {
                    q += x[a] * h[a * 12 + b] * x[b];
                }
            }
            assert!(q >= -1e-2, "Hessian not PSD: xᵀHx = {q}");
        }
    }

    /// Inverted elements must be detected (negative `J`) and penalized, not
    /// silently accepted — the volume term is what recovers an inverted tet.
    #[test]
    fn inverted_element_is_penalized() {
        let (rest, tets) = unit_tet();
        let el = build_tets(&tets, &rest, 1e-9, TetModel::StVK)[0];
        let p0 = pts(&rest, &el);
        // Push v3 through the opposite face → det F < 0.
        let mut inverted = p0;
        inverted[3] = Vec3::new(0.25, 0.25, -0.6);
        let f = deformation_gradient(&inverted, &el.dm_inv);
        assert!(f.determinant() < 0.0, "test config should be inverted");
        let s = principal_stretches(&f);
        let j = s[0] * s[1] * s[2];
        assert!(j < 0.0, "J must track the inversion, got {j}");
        let e_inv = tet_energy(&inverted, &el, 100.0, 200.0);
        let e_rest = tet_energy(&p0, &el, 100.0, 200.0);
        assert!(e_inv > e_rest + 1.0, "inversion not penalized: {e_inv}");
    }

    #[test]
    fn build_tets_normalizes_winding_and_skips_slivers() {
        // Negative-volume winding gets flipped, volume stays positive.
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let flipped = build_tets(&[[0, 2, 1, 3]], &rest, 1e-9, TetModel::StVK);
        assert_eq!(flipped.len(), 1);
        assert!(flipped[0].volume > 0.0, "volume must be positive");
        // Degenerate (all coplanar) is skipped.
        let flat = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)];
        assert!(build_tets(&[[0, 1, 2, 3]], &flat, 1e-9, TetModel::StVK).is_empty());
    }

    /// The SVD must reconstruct `F` with both frames proper rotations, and `σ₂` must
    /// carry the sign of `det F` so inversion is representable.
    #[test]
    fn svd3_reconstructs_with_proper_rotations() {
        let mats = [
            Mat3::IDENTITY,
            Mat3::from_cols(
                Vec3::new(1.3, 0.05, -0.02),
                Vec3::new(-0.03, 0.8, 0.04),
                Vec3::new(0.01, -0.02, 1.15),
            ),
            // Inverted (negative determinant).
            Mat3::from_cols(
                Vec3::new(1.1, 0.0, 0.0),
                Vec3::new(0.0, 0.9, 0.0),
                Vec3::new(0.0, 0.0, -1.2),
            ),
            // Strongly sheared.
            Mat3::from_cols(
                Vec3::new(1.0, 0.4, 0.1),
                Vec3::new(0.3, 1.2, -0.2),
                Vec3::new(-0.1, 0.25, 0.95),
            ),
        ];
        for (i, f) in mats.iter().enumerate() {
            let (u, s, v) = svd3(f);
            // Proper rotations.
            assert!(
                (u.determinant() - 1.0).abs() < 1e-2,
                "det U != 1 (case {i})"
            );
            assert!(
                (v.determinant() - 1.0).abs() < 1e-2,
                "det V != 1 (case {i})"
            );
            // Reconstruction: U diag(σ) Vᵀ == F.
            let us = Mat3::from_cols(u.x_axis * s[0], u.y_axis * s[1], u.z_axis * s[2]);
            let back = us * v.transpose();
            for c in 0..3 {
                let (a, b) = (col(&back, c), col(f, c));
                assert!(
                    (a - b).length() < 2e-3,
                    "SVD reconstruction failed (case {i}, col {c}): {a:?} vs {b:?}"
                );
            }
            // σ₂ sign tracks det F.
            let j = s[0] * s[1] * s[2];
            assert!(
                j.signum() == f.determinant().signum(),
                "J sign != det F sign (case {i}): J={j} detF={}",
                f.determinant()
            );
        }
    }

    /// **Cross-check**: the analytic Piola-stress gradient must equal the
    /// finite-difference gradient. These are INDEPENDENT derivations (one from
    /// `∂Ψ/∂σ` + chain rule, one from differencing the energy), so agreement is real
    /// evidence both are right — not a tautology.
    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::StVK)[0];
        let configs: Vec<[Vec3; 4]> = vec![
            // Stretched.
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.25, 0.03, 0.0),
                Vec3::new(0.0, 0.90, 0.02),
                Vec3::new(0.02, 0.0, 1.10),
            ],
            // Compressed + sheared.
            [
                Vec3::new(0.01, -0.02, 0.0),
                Vec3::new(0.80, 0.10, 0.05),
                Vec3::new(0.15, 0.75, -0.05),
                Vec3::new(-0.05, 0.10, 0.85),
            ],
            // Rotated (rigid) — gradient should be ~0.
            {
                let rot = Mat3::from_euler(glam::EulerRot::XYZ, 0.6, -0.9, 0.4);
                [rot * rest[0], rot * rest[1], rot * rest[2], rot * rest[3]]
            },
            // Large deformation.
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.8, 0.2, 0.1),
                Vec3::new(0.1, 1.5, 0.2),
                Vec3::new(0.2, 0.1, 0.6),
            ],
        ];
        for (mu, lam) in [(100.0f32, 200.0f32), (10.0, 1000.0), (500.0, 50.0)] {
            for (ci, p) in configs.iter().enumerate() {
                let ga = tet_gradient_analytic(p, &el, mu, lam);
                // Independent oracle: FD of the energy (NOT the assembly output,
                // which now uses the analytic gradient — comparing to that would be
                // a tautology).
                let gf = tet_gradient_fd(p, &el, mu, lam);
                let scale = gf.iter().map(|v| v.length()).fold(1.0f32, f32::max);
                for k in 0..4 {
                    let err = (ga[k] - gf[k]).length();
                    assert!(
                        err < 2e-2 * scale,
                        "analytic vs FD gradient mismatch (case {ci}, mu={mu}, lam={lam}, vert {k}): \
                         analytic {:?} vs fd {:?} (err {err}, scale {scale})",
                        ga[k],
                        gf[k]
                    );
                }
                // Translation invariance: forces sum to zero.
                let sum: Vec3 = ga.iter().copied().sum();
                assert!(
                    sum.length() < 1e-2 * scale,
                    "analytic forces do not sum to zero (case {ci}): {sum:?}"
                );
            }
        }
    }

    /// The analytic gradient must be dramatically cheaper than the FD path — that is
    /// the point of deriving it. One SVD vs 24 energy evaluations.
    #[test]
    fn analytic_gradient_is_much_faster_than_fd() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::StVK)[0];
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.25, 0.03, 0.0),
            Vec3::new(0.0, 0.90, 0.02),
            Vec3::new(0.02, 0.0, 1.10),
        ];
        let iters = 2000;
        let t0 = std::time::Instant::now();
        let mut acc = Vec3::ZERO;
        for _ in 0..iters {
            acc += tet_gradient_analytic(&p, &el, 100.0, 200.0)[1];
        }
        let t_analytic = t0.elapsed();
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let (_e, g, _h) = tet_element_assembly(&p, &el, 100.0, 200.0);
            acc += g[1];
        }
        let t_fd = t1.elapsed();
        assert!(acc.is_finite());
        // The FD path also builds the 12×12 Hessian (~300 energy evals), so the
        // margin is large; require a conservative 5× to keep the test robust on any
        // machine while still failing loudly if the analytic path regresses.
        assert!(
            t_fd.as_secs_f64() > 5.0 * t_analytic.as_secs_f64(),
            "analytic gradient not meaningfully faster: analytic {t_analytic:?} vs fd {t_fd:?}"
        );
        eprintln!("analytic {t_analytic:?} vs fd(+hessian) {t_fd:?}");
    }

    /// Verify the face table's winding NUMERICALLY rather than trusting it: each
    /// face's normal must point away from the tet's opposite vertex.
    #[test]
    fn tet_faces_are_outward_oriented() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::StVK)[0];
        let faces = tet_faces(&el.v);
        let all: Vec<u32> = el.v.to_vec();
        for (fi, f) in faces.iter().enumerate() {
            // The vertex NOT on this face.
            let opp = *all
                .iter()
                .find(|v| !f.contains(v))
                .expect("opposite vertex");
            let (a, b, c) = (
                rest[f[0] as usize],
                rest[f[1] as usize],
                rest[f[2] as usize],
            );
            let n = (b - a).cross(c - a);
            let toward_opp = (rest[opp as usize] - a).dot(n);
            assert!(
                toward_opp < 0.0,
                "face {fi} {f:?} normal points TOWARD opposite vertex {opp} (dot {toward_opp})"
            );
        }
    }

    /// A single tet has all 4 faces on the boundary; two tets sharing a face expose
    /// 6, not 8 — the shared face must cancel.
    #[test]
    fn surface_drops_shared_interior_faces() {
        let rest = vec![
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let one = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::StVK);
        assert_eq!(tet_surface(&one).len(), 4, "single tet must expose 4 faces");

        // Two tets sharing face (1,2,3).
        let two = build_tets(&[[0, 1, 2, 3], [4, 1, 2, 3]], &rest, 1e-9, TetModel::StVK);
        assert_eq!(two.len(), 2, "both tets should be non-degenerate");
        let surf = tet_surface(&two);
        assert_eq!(
            surf.len(),
            6,
            "two tets sharing a face expose 6, got {}",
            surf.len()
        );
        // The shared face must not appear in any winding.
        let shared = [1u32, 2, 3];
        for f in &surf {
            let mut k = *f;
            k.sort_unstable();
            assert_ne!(k, shared, "interior face survived: {f:?}");
        }
    }

    /// The strongest orientation+closedness check: for a closed outward surface the
    /// divergence theorem gives the enclosed volume, which must equal the sum of the
    /// tets' volumes. A flipped face or a hole breaks this immediately.
    #[test]
    fn extracted_surface_encloses_the_tet_volume() {
        let nn = 3usize;
        let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
        let mut rest = Vec::new();
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    rest.push(Vec3::new(i as f32 * 0.5, j as f32 * 0.5, k as f32 * 0.5));
                }
            }
        }
        let mut tet_idx = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
                    let (a, b, _c2, d) = (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i, j + 1, k),
                        idx(i + 1, j + 1, k),
                    );
                    let c = idx(i, j + 1, k);
                    let (e, f, g, h) = (
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i, j + 1, k + 1),
                        idx(i + 1, j + 1, k + 1),
                    );
                    tet_idx.push([a, b, c, e]);
                    tet_idx.push([b, d, c, h]);
                    tet_idx.push([b, c, e, h]);
                    tet_idx.push([c, e, g, h]);
                    tet_idx.push([b, e, f, h]);
                }
            }
        }
        let els = build_tets(&tet_idx, &rest, 1e-9, TetModel::StVK);
        let tet_vol: f32 = els.iter().map(|e| e.volume).sum();
        // The 5-tet decomposition tiles the cube, so the block volume is 1.0³ here
        // (2×2×2 cubes of side 0.5).
        assert!(
            (tet_vol - 1.0).abs() < 1e-3,
            "tets should tile the block: {tet_vol}"
        );

        let surf = tet_surface(&els);
        assert!(!surf.is_empty());
        let enclosed = enclosed_volume(&surf, &rest);
        assert!(
            (enclosed - tet_vol).abs() < 1e-3 * tet_vol,
            "surface encloses {enclosed} but tets total {tet_vol} — orientation or closedness is wrong"
        );
    }

    /// End-to-end volumetric contact: a tet block dropped on a floor collider must
    /// land, stay finite, not penetrate, and keep every element un-inverted.
    #[test]
    fn tet_block_rests_on_floor_collider() {
        use crate::implicit::{ImplicitParams, ImplicitSolver};
        use crate::linsolve::CgCpu;
        use crate::state::SimState;
        use crate::Mesh;

        let nn = 3usize;
        let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
        let mut positions = Vec::new();
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    // Start above the floor at y = 0.
                    positions.push(Vec3::new(
                        i as f32 * 0.1,
                        0.25 + j as f32 * 0.1,
                        k as f32 * 0.1,
                    ));
                }
            }
        }
        let mut tet_idx = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
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
                    tet_idx.push([a, b, c, e]);
                    tet_idx.push([b, d, c, h]);
                    tet_idx.push([b, c, e, h]);
                    tet_idx.push([c, e, g, h]);
                    tet_idx.push([b, e, f, h]);
                }
            }
        }
        let tris: Vec<[u32; 3]> = vec![[idx(0, 0, 0), idx(1, 0, 0), idx(0, 1, 0)]];
        let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.5);
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 0.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                tet_mu: 5.0e3,
                tet_lambda: 1.0e4,
                contact_dhat: 0.01,
                contact_kappa: 1.0e4,
                ..Default::default()
            },
        );
        solver.set_tets(&tet_idx);
        solver.set_colliders(vec![crate::contact::Collider::HalfSpace {
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }]);
        let y_start = solver
            .state
            .positions
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        for _ in 0..300 {
            solver.step(&CgCpu::new());
        }
        let min_y = solver
            .state
            .positions
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            solver.state.positions.iter().all(|p| p.is_finite()),
            "volumetric contact produced non-finite state"
        );
        assert!(min_y < y_start, "block did not fall: {y_start} -> {min_y}");
        assert!(min_y > -0.02, "block penetrated the floor: {min_y}");
        for (t, el) in solver.tets().iter().enumerate() {
            let p = [
                solver.state.positions[el.v[0] as usize],
                solver.state.positions[el.v[1] as usize],
                solver.state.positions[el.v[2] as usize],
                solver.state.positions[el.v[3] as usize],
            ];
            let f = deformation_gradient(&p, &el.dm_inv);
            assert!(f.determinant() > 0.0, "tet {t} inverted on impact");
        }
    }

    /// FD-of-analytic-gradient Hessian, kept ONLY as the independent oracle for
    /// [`tet_hessian_analytic`]. Not used by the solver.
    fn tet_hessian_fd(p: &[Vec3; 4], el: &TetElement, mu: f32, lambda: f32) -> [f32; 144] {
        let scale = p.iter().map(|q| q.length()).sum::<f32>().max(1.0);
        let h = 1e-3 * scale;
        let inv = 1.0 / (2.0 * h);
        let mut hess = [0.0f32; 144];
        for b in 0..12 {
            let (vb, cb) = (b / 3, b % 3);
            let (mut pp, mut pm) = (*p, *p);
            pp[vb][cb] += h;
            pm[vb][cb] -= h;
            let gp = tet_gradient_analytic(&pp, el, mu, lambda);
            let gm = tet_gradient_analytic(&pm, el, mu, lambda);
            for a in 0..4 {
                for r in 0..3 {
                    hess[(3 * a + r) * 12 + b] = (gp[a][r] - gm[a][r]) * inv;
                }
            }
        }
        hess
    }

    /// **The decisive correctness test for the eigensystem.** The reference's
    /// `compute_hessian` claims 9 specific eigenpairs of `∂²Ψ/∂F²`. Verify exactly
    /// that: each predicted mode `U·Qₖ·Vᵀ` must satisfy `A_fd·qₖ = λₖ·qₖ`, where
    /// `A_fd` is the true Hessian obtained by finite-differencing the analytic Piola
    /// stress in F-space and `λₖ` is the formula's UNCLAMPED eigenvalue.
    ///
    /// This is the right check because the shipped Hessian is deliberately FILTERED:
    /// the reference clamps every eigenvalue at 0 (`fmaxf`), and at realistic
    /// deformations several are genuinely negative (this energy is non-convex in its
    /// flip modes), so comparing the filtered result against an unfiltered FD Hessian
    /// would fail for a correct implementation. Verifying the eigenpairs separates
    /// "is the transcription right" from "is the filter intentional".
    #[test]
    fn predicted_modes_are_true_eigenpairs_of_d2psi_df2() {
        let (mu, lam) = (100.0f32, 200.0f32);
        let fs = [
            Mat3::from_cols(
                Vec3::new(1.3, 0.0, 0.0),
                Vec3::new(0.0, 1.1, 0.0),
                Vec3::new(0.0, 0.0, 0.9),
            ),
            Mat3::from_cols(
                Vec3::new(1.15, 0.08, 0.02),
                Vec3::new(0.05, 0.92, -0.04),
                Vec3::new(-0.03, 0.06, 1.05),
            ),
        ];
        // FD step: a central difference of P has roundoff ~ε/h and truncation ~h²,
        // so in f32 h=1e-3 is far better conditioned than 1e-4 (which leaves ~1e-3
        // RELATIVE noise on entries of order 400 — enough to swamp a flip mode whose
        // (σᵢ−σⱼ) denominator is small). Chosen from that error balance, not tuned to
        // make the test pass.
        let h = 1e-3f32;
        let at = |m: &Mat3, r: usize, c: usize| col(m, c)[r];
        let pk1 = |f: &Mat3| -> Mat3 {
            let (u, s, v) = svd3(f);
            let d = stvk_dpsi_dsigma(&s, TetModel::StVK, mu, lam);
            Mat3::from_cols(u.x_axis * d[0], u.y_axis * d[1], u.z_axis * d[2]) * v.transpose()
        };
        for (fi, f0) in fs.iter().enumerate() {
            // True ∂²Ψ/∂F² by central differences of P.
            let mut afd = [0.0f32; 81];
            for jj in 0..9 {
                let (ri, ci) = (jj / 3, jj % 3);
                let mut cp = [f0.x_axis, f0.y_axis, f0.z_axis];
                cp[ci][ri] += h;
                let fp = Mat3::from_cols(cp[0], cp[1], cp[2]);
                let mut cm = [f0.x_axis, f0.y_axis, f0.z_axis];
                cm[ci][ri] -= h;
                let fm = Mat3::from_cols(cm[0], cm[1], cm[2]);
                let (pp, pm) = (pk1(&fp), pk1(&fm));
                for ii in 0..9 {
                    afd[ii * 9 + jj] =
                        (at(&pp, ii / 3, ii % 3) - at(&pm, ii / 3, ii % 3)) / (2.0 * h);
                }
            }
            let (u, sg, v) = svd3(f0);
            let (modes, lambdas) = predicted_eigenpairs(&sg, TetModel::StVK, mu, lam, 1.0e-6);
            for k in 0..9 {
                let qv = vec9(&(u * modes[k] * v.transpose()));
                let mut resid = 0.0f32;
                let mut mag = 1.0f32;
                for r in 0..9 {
                    let mut acc = 0.0f32;
                    for c in 0..9 {
                        acc += afd[r * 9 + c] * qv[c];
                    }
                    let want = lambdas[k] * qv[r];
                    resid = resid.max((acc - want).abs());
                    mag = mag.max(want.abs());
                }
                assert!(
                    resid < 2e-2 * mag,
                    "mode {k} (F case {fi}) is not an eigenvector: residual {resid} vs magnitude {mag}"
                );
            }
        }
    }

    /// Where the energy IS locally convex (at and near rest, all 9 eigenvalues ≥ 0),
    /// the filter is a no-op, so the analytic Hessian must equal the FD Hessian.
    #[test]
    fn analytic_hessian_matches_fd_where_energy_is_convex() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::StVK)[0];
        let (mu, lam) = (100.0f32, 200.0f32);
        // At/near rest: σ ≈ (1,1,1) ⇒ twist and flip eigenvalues are ~0 and the
        // scaling eigenvalues are ≥ 0, so no clamping occurs.
        for eps in [0.0f32, 1e-3, 3e-3] {
            let p = [
                Vec3::ZERO,
                Vec3::new(1.0 + eps, 0.0, 0.0),
                Vec3::new(0.0, 1.0 + eps, 0.0),
                Vec3::new(0.0, 0.0, 1.0 + eps),
            ];
            let ha = tet_hessian_analytic(&p, &el, mu, lam);
            let hf = tet_hessian_fd(&p, &el, mu, lam);
            let scale = hf.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            let mut worst = 0.0f32;
            for k in 0..144 {
                worst = worst.max((ha[k] - hf[k]).abs());
            }
            assert!(
                worst < 5e-2 * scale,
                "convex-regime mismatch (eps={eps}): worst {worst}, scale {scale}"
            );
        }
    }

    /// The analytic Hessian must be symmetric and PSD by construction (the reference
    /// clamps every eigenvalue at 0), and must annihilate rigid translations.
    #[test]
    fn analytic_hessian_is_symmetric_psd_and_translation_invariant() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::StVK)[0];
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.25, 0.05, 0.0),
            Vec3::new(0.0, 0.88, 0.03),
            Vec3::new(0.02, 0.0, 1.12),
        ];
        let h = tet_hessian_analytic(&p, &el, 100.0, 200.0);
        let scale = h.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for a in 0..12 {
            for b in 0..12 {
                assert!(
                    (h[a * 12 + b] - h[b * 12 + a]).abs() < 1e-4 * scale,
                    "not symmetric at ({a},{b})"
                );
            }
        }
        // PSD.
        for t in 0..12u64 {
            let x: [f32; 12] = std::array::from_fn(|k| {
                ((t.wrapping_mul(7919)
                    .wrapping_add((k as u64).wrapping_mul(104729))
                    % 200) as f32)
                    / 100.0
                    - 1.0
            });
            let mut q = 0.0f32;
            for a in 0..12 {
                for b in 0..12 {
                    q += x[a] * h[a * 12 + b] * x[b];
                }
            }
            assert!(q >= -1e-3 * scale, "analytic Hessian not PSD: xᵀHx = {q}");
        }
        // Uniform translation is a null mode: H·(1,1,1 repeated) == 0.
        for axis in 0..3 {
            let mut t = [0.0f32; 12];
            for v in 0..4 {
                t[3 * v + axis] = 1.0;
            }
            for a in 0..12 {
                let mut row = 0.0f32;
                for b in 0..12 {
                    row += h[a * 12 + b] * t[b];
                }
                assert!(
                    row.abs() < 1e-3 * scale,
                    "translation along axis {axis} is not a null mode (row {a} = {row})"
                );
            }
        }
    }

    /// The analytic Hessian must be dramatically cheaper than the FD one: one SVD
    /// versus 24 gradient evaluations (each its own SVD).
    #[test]
    fn analytic_hessian_is_much_faster_than_fd() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::StVK)[0];
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.25, 0.03, 0.0),
            Vec3::new(0.0, 0.90, 0.02),
            Vec3::new(0.02, 0.0, 1.10),
        ];
        let iters = 3000;
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..iters {
            acc += tet_hessian_analytic(&p, &el, 100.0, 200.0)[0];
        }
        let t_analytic = t0.elapsed();
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            acc += tet_hessian_fd(&p, &el, 100.0, 200.0)[0];
        }
        let t_fd = t1.elapsed();
        assert!(acc.is_finite());
        eprintln!("analytic hessian {t_analytic:?} vs fd {t_fd:?}");
        // Measured ~1.6× in an unoptimized test build: the analytic path replaces 24
        // SVDs with one, but its 9×9 eigen-assembly and the Bᵀ·A·B contractions are
        // themselves ~2.3k unoptimized flops. The margin is larger in release and
        // larger again at solver level (the FD path re-runs the SVD per perturbed
        // configuration). Assert a conservative 1.3× so this fails on a real
        // regression without being flaky about build profile.
        assert!(
            t_fd.as_secs_f64() > 1.3 * t_analytic.as_secs_f64(),
            "analytic Hessian not faster: {t_analytic:?} vs {t_fd:?}"
        );
    }

    /// ARAP's derivatives must be self-consistent: `∂Ψ/∂σ` == FD of the density, and
    /// `∂²Ψ/∂σ²` == FD of `∂Ψ/∂σ`. This is the zero-hallucination check that the
    /// transcription from the reference `arap.hpp` is right — its deviatoric part is
    /// `½μΣ(σ−1)²` with gradient `μ(σ−1)` and Hessian `μ·I`.
    #[test]
    fn arap_derivatives_are_self_consistent() {
        let (mu, lam) = (500.0f32, 300.0f32);
        for base in [[1.3f32, 1.1, 0.9], [0.85, 1.05, 1.2], [1.0, 1.0, 1.0]] {
            let h = 1e-3;
            // Gradient vs FD of density.
            let g = stvk_dpsi_dsigma(&base, TetModel::Arap, mu, lam);
            for i in 0..3 {
                let mut sp = base;
                sp[i] += h;
                let mut sm = base;
                sm[i] -= h;
                let fd = (stvk_density(&sp, TetModel::Arap, mu, lam)
                    - stvk_density(&sm, TetModel::Arap, mu, lam))
                    / (2.0 * h);
                assert!(
                    (g[i] - fd).abs() < 2e-2 * g[i].abs().max(1.0),
                    "ARAP grad[{i}] {} != fd {fd}",
                    g[i]
                );
            }
            // Hessian vs FD of gradient.
            let hh = stvk_d2psi_dsigma2(&base, TetModel::Arap, mu, lam);
            for j in 0..3 {
                let mut sp = base;
                sp[j] += h;
                let mut sm = base;
                sm[j] -= h;
                let gp = stvk_dpsi_dsigma(&sp, TetModel::Arap, mu, lam);
                let gm = stvk_dpsi_dsigma(&sm, TetModel::Arap, mu, lam);
                for i in 0..3 {
                    let fd = (gp[i] - gm[i]) / (2.0 * h);
                    assert!(
                        (hh[i * 3 + j] - fd).abs() < 3e-2 * hh[i * 3 + j].abs().max(1.0),
                        "ARAP hess[{i}][{j}] {} != fd {fd}",
                        hh[i * 3 + j]
                    );
                }
            }
        }
    }

    /// ARAP's deviatoric Hessian is exactly `μ·I` (the reference: `mu *
    /// Mat3x3f::Identity()`), so at `J = 1` — where the volume term's own Hessian is
    /// present but the deviatoric part is model-defining — the diagonal must exceed
    /// StVK's, which vanishes quadratically as `σ → 1`. This pins the difference
    /// between the two models rather than just testing each in isolation.
    #[test]
    fn arap_stays_stiff_near_rest_where_stvk_goes_slack() {
        let (mu, lam) = (500.0f32, 0.0f32); // no volume term: isolate the deviatoric part
        let near_rest = [1.001f32, 1.001, 1.001];
        let a = stvk_d2psi_dsigma2(&near_rest, TetModel::Arap, mu, lam);
        let s = stvk_d2psi_dsigma2(&near_rest, TetModel::StVK, mu, lam);
        assert!(
            (a[0] - mu).abs() < 1e-2 * mu,
            "ARAP deviatoric Hessian must be μ·I, got {}",
            a[0]
        );
        assert!(
            s[0] < 1e-3 * mu,
            "StVK's quartic deviatoric Hessian must vanish near rest, got {}",
            s[0]
        );
        assert!(a[0] > 100.0 * s[0].max(1e-9));
    }

    /// ARAP must satisfy the same physical invariants as StVK: rigid-motion
    /// invariance, zero energy/gradient at rest, and a PSD element Hessian.
    #[test]
    fn arap_element_is_rigid_invariant_and_psd() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::Arap)[0];
        let (mu, lam) = (500.0f32, 300.0f32);
        // Rest: zero energy and gradient.
        let p0 = [rest[0], rest[1], rest[2], rest[3]];
        let (e0, g0, _h0) = tet_element_assembly(&p0, &el, mu, lam);
        assert!(e0.abs() < 1e-4, "ARAP rest energy {e0}");
        assert!(g0.iter().all(|g| g.length() < 1e-3), "ARAP rest gradient");
        // Rigid motion invariance.
        let deformed = [
            Vec3::new(0.02, -0.01, 0.0),
            Vec3::new(1.30, 0.05, -0.02),
            Vec3::new(-0.03, 0.80, 0.04),
            Vec3::new(0.01, -0.02, 1.15),
        ];
        let base = tet_energy(&deformed, &el, mu, lam);
        assert!(base > 1e-3);
        let rot = Mat3::from_euler(glam::EulerRot::XYZ, 0.7, -1.1, 0.35);
        let shift = Vec3::new(3.0, -2.0, 5.0);
        let moved: [Vec3; 4] = std::array::from_fn(|i| rot * deformed[i] + shift);
        let after = tet_energy(&moved, &el, mu, lam);
        assert!(
            (after - base).abs() < 1e-3 * base.max(1.0),
            "ARAP energy not rigid-invariant: {base} -> {after}"
        );
        // Analytic gradient still matches the independent FD oracle.
        let ga = tet_gradient_analytic(&deformed, &el, mu, lam);
        let gf = tet_gradient_fd(&deformed, &el, mu, lam);
        let scale = gf.iter().map(|v| v.length()).fold(1.0f32, f32::max);
        for k in 0..4 {
            assert!(
                (ga[k] - gf[k]).length() < 2e-2 * scale,
                "ARAP analytic vs FD gradient mismatch at {k}"
            );
        }
        // Hessian PSD.
        let h = tet_hessian_analytic(&deformed, &el, mu, lam);
        let hscale = h.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for t in 0..10u64 {
            let x: [f32; 12] = std::array::from_fn(|k| {
                ((t.wrapping_mul(7919)
                    .wrapping_add((k as u64).wrapping_mul(104729))
                    % 200) as f32)
                    / 100.0
                    - 1.0
            });
            let mut q = 0.0f32;
            for a2 in 0..12 {
                for b2 in 0..12 {
                    q += x[a2] * h[a2 * 12 + b2] * x[b2];
                }
            }
            assert!(q >= -1e-3 * hscale, "ARAP Hessian not PSD: {q}");
        }
    }

    /// The dead zone is what makes this creep and not damping: deformation whose
    /// principal stretches stay within `threshold` of 1 must leave the rest state
    /// completely untouched, no matter how long it is held.
    #[test]
    fn plasticity_ignores_deformation_inside_the_dead_zone() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let mut el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::Arap)[0];
        let before = el.dm_inv;
        // 2% stretch, dead zone 10% ⇒ inside.
        let p = [
            Vec3::ZERO,
            Vec3::new(1.02, 0.0, 0.0),
            Vec3::new(0.0, 1.02, 0.0),
            Vec3::new(0.0, 0.0, 1.02),
        ];
        for _ in 0..100 {
            assert!(
                !apply_tet_plasticity(&mut el, &p, 0.5, 0.10, 1.0 / 60.0),
                "in-dead-zone deformation must not change the rest state"
            );
        }
        assert_eq!(before.to_cols_array(), el.dm_inv.to_cols_array());
        // plasticity = 0 must also be a no-op even far outside the dead zone.
        let far = [
            Vec3::ZERO,
            Vec3::new(1.8, 0.0, 0.0),
            Vec3::new(0.0, 1.8, 0.0),
            Vec3::new(0.0, 0.0, 1.8),
        ];
        assert!(!apply_tet_plasticity(&mut el, &far, 0.0, 0.10, 1.0 / 60.0));
        assert_eq!(before.to_cols_array(), el.dm_inv.to_cols_array());
    }

    /// Held beyond the dead zone, plasticity must RELAX stored elastic energy
    /// monotonically, and the residual stretch must converge to the dead-zone edge
    /// (`1 + threshold`) — not to 1 (that would be full relaxation, losing the yield
    /// surface) and not overshoot past the current deformation.
    #[test]
    fn plasticity_relaxes_energy_toward_the_dead_zone_edge() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let mut el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::Arap)[0];
        let (mu, lam) = (500.0f32, 0.0f32); // deviatoric only: isolate the stretch response
        let stretch = 1.5f32;
        let threshold = 0.10f32;
        let p = [
            Vec3::ZERO,
            Vec3::new(stretch, 0.0, 0.0),
            Vec3::new(0.0, stretch, 0.0),
            Vec3::new(0.0, 0.0, stretch),
        ];
        let mut prev = tet_energy(&p, &el, mu, lam);
        assert!(prev > 1e-3, "held stretch should store energy: {prev}");
        for step in 0..400 {
            let changed = apply_tet_plasticity(&mut el, &p, 2.0, threshold, 1.0 / 60.0);
            let e = tet_energy(&p, &el, mu, lam);
            assert!(
                e <= prev + 1e-5,
                "plasticity must not increase stored energy (step {step}): {prev} -> {e}"
            );
            if !changed {
                break;
            }
            prev = e;
        }
        // Residual stretch relative to the RELAXED rest state must sit at the
        // dead-zone edge: the yield surface is preserved, not erased.
        let f = deformation_gradient(&p, &el.dm_inv);
        let sig = principal_stretches(&f);
        for (k, &sk) in sig.iter().enumerate() {
            assert!(
                (sk - (1.0 + threshold)).abs() < 2e-2,
                "stretch {k} should converge to the dead-zone edge {}, got {sk}",
                1.0 + threshold
            );
        }
        // Energy is now only what the dead zone permits — far below the original.
        let final_e = tet_energy(&p, &el, mu, lam);
        let original = {
            let fresh = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::Arap)[0];
            tet_energy(&p, &fresh, mu, lam)
        };
        assert!(
            final_e < 0.2 * original,
            "plasticity should have relaxed most of the energy: {final_e} vs {original}"
        );
    }

    /// A larger `plasticity` rate must relax faster for the same elapsed time — the
    /// `α = 1 − exp(−plasticity·dt)` dependence, checked behaviorally.
    #[test]
    fn higher_plasticity_rate_relaxes_faster() {
        let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let (mu, lam) = (500.0f32, 0.0f32);
        let p = [
            Vec3::ZERO,
            Vec3::new(1.4, 0.0, 0.0),
            Vec3::new(0.0, 1.4, 0.0),
            Vec3::new(0.0, 0.0, 1.4),
        ];
        let run = |rate: f32| {
            let mut el = build_tets(&[[0, 1, 2, 3]], &rest, 1e-9, TetModel::Arap)[0];
            for _ in 0..20 {
                apply_tet_plasticity(&mut el, &p, rate, 0.05, 1.0 / 60.0);
            }
            tet_energy(&p, &el, mu, lam)
        };
        let slow = run(0.5);
        let fast = run(20.0);
        assert!(
            fast < slow,
            "higher plasticity must relax more in the same time: fast {fast} vs slow {slow}"
        );
    }
}

#[cfg(test)]
mod snhk_tests {
    use super::*;

    /// The SNHk derivative table, checked TWO independent ways so a transcription typo
    /// cannot hide: (a) against finite differences of `stvk_density`, and (b) against
    /// `snhk.hpp`'s closed forms written out separately here.
    ///
    /// `snhk.hpp` (3D):
    ///   Ψ       = ½μ(I₂−3) − μ(I₃−1) + ½λ(I₃−1)²
    ///   ∂Ψ/∂a   = λbc(I₃−1) + μ(a − bc)
    ///   ∂²Ψ/∂a² = λb²c² + μ
    ///   ∂²Ψ/∂a∂b = λI₃c + λc(I₃−1) − μc
    #[test]
    fn snhk_table_matches_the_reference_and_finite_differences() {
        let (mu, lam) = (120.0f32, 75.0f32);
        let cases = [
            [1.20f32, 1.05, 0.95],
            [0.90, 1.10, 1.30],
            [1.00, 1.00, 1.00], // rest
            [1.40, 0.80, 1.10],
        ];
        for s in cases {
            let (a, b, c) = (s[0], s[1], s[2]);
            let i2 = a * a + b * b + c * c;
            let i3 = a * b * c;

            // (b) energy against the reference's closed form.
            let want_e =
                0.5 * mu * (i2 - 3.0) - mu * (i3 - 1.0) + 0.5 * lam * (i3 - 1.0) * (i3 - 1.0);
            let got_e = stvk_density(&s, TetModel::SnHk, mu, lam);
            assert!(
                (got_e - want_e).abs() <= 1e-4 * want_e.abs().max(1.0),
                "energy: got {got_e} want {want_e} at {s:?}"
            );

            // (b) gradient against the reference's closed form.
            let want_g = [
                lam * b * c * (i3 - 1.0) + mu * (a - b * c),
                lam * a * c * (i3 - 1.0) + mu * (b - a * c),
                lam * a * b * (i3 - 1.0) + mu * (c - a * b),
            ];
            let got_g = stvk_dpsi_dsigma(&s, TetModel::SnHk, mu, lam);
            for k in 0..3 {
                assert!(
                    (got_g[k] - want_g[k]).abs() <= 1e-3 * want_g[k].abs().max(1.0),
                    "grad[{k}]: got {} want {} at {s:?}",
                    got_g[k],
                    want_g[k]
                );
            }

            // (a) gradient against finite differences of the energy — independent path.
            let h = 1e-3f32;
            for k in 0..3 {
                let mut sp = s;
                let mut sm = s;
                sp[k] += h;
                sm[k] -= h;
                let fd = (stvk_density(&sp, TetModel::SnHk, mu, lam)
                    - stvk_density(&sm, TetModel::SnHk, mu, lam))
                    / (2.0 * h);
                assert!(
                    (fd - got_g[k]).abs() <= 2e-2 * fd.abs().max(1.0),
                    "grad[{k}] vs FD: analytic {} fd {fd} at {s:?}",
                    got_g[k]
                );
            }

            // (b) Hessian against the reference's closed form, including the coupled
            // off-diagonals that StVK/ARAP do not have.
            let want_h = [
                lam * b * b * c * c + mu,
                lam * i3 * c + lam * c * (i3 - 1.0) - mu * c,
                lam * i3 * b + lam * b * (i3 - 1.0) - mu * b,
                lam * i3 * c + lam * c * (i3 - 1.0) - mu * c,
                lam * a * a * c * c + mu,
                lam * i3 * a + lam * a * (i3 - 1.0) - mu * a,
                lam * i3 * b + lam * b * (i3 - 1.0) - mu * b,
                lam * i3 * a + lam * a * (i3 - 1.0) - mu * a,
                lam * a * a * b * b + mu,
            ];
            let got_h = stvk_d2psi_dsigma2(&s, TetModel::SnHk, mu, lam);
            for k in 0..9 {
                assert!(
                    (got_h[k] - want_h[k]).abs() <= 1e-3 * want_h[k].abs().max(1.0),
                    "hess[{k}]: got {} want {} at {s:?}",
                    got_h[k],
                    want_h[k]
                );
            }
        }
    }

    /// SNHk must be a DISTINCT model — if it silently behaved like ARAP (the substitution
    /// the driver used to make), this test would fail.
    #[test]
    fn snhk_differs_from_arap_and_stvk() {
        let (mu, lam) = (120.0f32, 75.0f32);
        let s = [1.25f32, 1.05, 0.92];
        let snhk = stvk_dpsi_dsigma(&s, TetModel::SnHk, mu, lam);
        let arap = stvk_dpsi_dsigma(&s, TetModel::Arap, mu, lam);
        let stvk = stvk_dpsi_dsigma(&s, TetModel::StVK, mu, lam);
        let diff =
            |x: [f32; 3], y: [f32; 3]| (0..3).map(|k| (x[k] - y[k]).abs()).fold(0.0f32, f32::max);
        assert!(diff(snhk, arap) > 1.0, "SNHk must differ from ARAP");
        assert!(diff(snhk, stvk) > 1.0, "SNHk must differ from StVK");
    }
}
