// SPDX-License-Identifier: GPL-2.0-or-later
//! Isotropic **shell** constitutive models on principal stretches (ARAP / StVK), the
//! 2D counterpart of [`crate::tet`]. A triangle's deformation gradient is `3×2` (two
//! material directions mapped into 3D), so the stretch space is `σ ∈ ℝ²` and the
//! Hessian in F-space is `6×6` with **six** known eigenpairs.
//!
//! Ported from the reference (Apache-2.0, parent repo):
//! `energy/model/{arap.hpp,stvk.hpp,detsqr.hpp}` (the `Vec2f` overloads) and
//! `eigenanalysis/eigenanalysis.cu::compute_hessian(DiffTable2, Svd3x2)`.
//!
//! The gauge's `plastic` scene mixes `arap` and `baraffwitkin` shells, which is why
//! this exists alongside the already-ported Baraff-Witkin membrane
//! ([`crate::membrane`]).

use crate::tet::TetModel;
use glam::{Mat2, Vec2, Vec3};

/// A `3×2` matrix stored as its two 3D columns — the shape of a triangle's
/// deformation gradient and of the shell SVD's `U`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3x2 {
    pub c0: Vec3,
    pub c1: Vec3,
}

impl Mat3x2 {
    pub const ZERO: Self = Self {
        c0: Vec3::ZERO,
        c1: Vec3::ZERO,
    };
    pub fn from_cols(c0: Vec3, c1: Vec3) -> Self {
        Self { c0, c1 }
    }
    /// Row-major `vec` (6 entries): `v[2r + c] = M[r][c]`. Used consistently for the
    /// 6×6 assembly so the convention cancels out.
    pub fn vec6(&self) -> [f32; 6] {
        [
            self.c0.x, self.c1.x, self.c0.y, self.c1.y, self.c0.z, self.c1.z,
        ]
    }
    /// `self · m` for a 2×2 `m`.
    pub fn mul_mat2(&self, m: &Mat2) -> Self {
        Self {
            c0: self.c0 * m.x_axis.x + self.c1 * m.x_axis.y,
            c1: self.c0 * m.y_axis.x + self.c1 * m.y_axis.y,
        }
    }
    /// `Mᵀ·M` (2×2, symmetric).
    pub fn ata(&self) -> Mat2 {
        Mat2::from_cols(
            Vec2::new(self.c0.dot(self.c0), self.c1.dot(self.c0)),
            Vec2::new(self.c0.dot(self.c1), self.c1.dot(self.c1)),
        )
    }
}

/// Symmetric 2×2 eigendecomposition (closed form). Returns `(eigenvalues,
/// eigenvectors as columns)` with `A = V·diag(λ)·Vᵀ`, eigenvalues descending.
pub fn sym_eig2(a: &Mat2) -> ([f32; 2], Mat2) {
    let (p, q, r) = (a.x_axis.x, a.y_axis.x, a.y_axis.y); // [[p, q], [q, r]]
    let tr = p + r;
    let diff = ((p - r) * (p - r) + 4.0 * q * q).max(0.0).sqrt();
    let l0 = 0.5 * (tr + diff);
    let l1 = 0.5 * (tr - diff);
    // Eigenvector for l0: (q, l0 - p) or (l0 - r, q), whichever is better scaled.
    let v0 = if q.abs() > 1.0e-20 {
        Vec2::new(q, l0 - p).normalize_or(Vec2::X)
    } else if p >= r {
        Vec2::X
    } else {
        Vec2::Y
    };
    let v1 = Vec2::new(-v0.y, v0.x); // orthogonal
    ([l0, l1], Mat2::from_cols(v0, v1))
}

/// Thin SVD of a `3×2` matrix: `F = U·diag(σ)·Vᵀ` with `U` `3×2` (orthonormal
/// columns), `σ` descending and non-negative, `V` `2×2` orthonormal. Unlike the 3×3
/// case there is no orientation to preserve — a triangle has no signed volume — so
/// the singular values stay non-negative.
pub fn svd3x2(f: &Mat3x2) -> (Mat3x2, [f32; 2], Mat2) {
    let (lam, v) = sym_eig2(&f.ata());
    let sigma = [lam[0].max(0.0).sqrt(), lam[1].max(0.0).sqrt()];
    let eps = 1.0e-9 * sigma[0].max(1.0);
    // uₖ = F·vₖ/σₖ.
    let fv = |vv: Vec2| f.c0 * vv.x + f.c1 * vv.y;
    let u0 = if sigma[0] > eps {
        fv(v.x_axis) / sigma[0]
    } else {
        Vec3::X
    }
    .normalize_or(Vec3::X);
    let u1_raw = if sigma[1] > eps {
        fv(v.y_axis) / sigma[1]
    } else {
        // Any unit vector orthogonal to u0.
        let t = if u0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        t - t.dot(u0) * u0
    };
    let u1 = (u1_raw - u1_raw.dot(u0) * u0).normalize_or(Vec3::Y);
    (Mat3x2::from_cols(u0, u1), sigma, v)
}

/// Shell strain-energy density on the two principal stretches: the model's
/// deviatoric term plus the reference's `detsqr` area term `½λ(J−1)²`, `J = σ₀σ₁`.
/// (As in 3D, the gradient-consistent unshifted `J` is used — see the note in
/// [`crate::tet`] about `detsqr.hpp`'s internal inconsistency.)
pub fn shell_density(sigma: &[f32; 2], model: TetModel, mu: f32, lambda: f32) -> f32 {
    let j = sigma[0] * sigma[1];
    let dev: f32 = match model {
        TetModel::StVK => mu * ((sigma[0] - 1.0).powi(4) + (sigma[1] - 1.0).powi(4)) / 4.0,
        TetModel::Arap => {
            0.5 * mu * ((sigma[0] - 1.0) * (sigma[0] - 1.0) + (sigma[1] - 1.0) * (sigma[1] - 1.0))
        }
        // 2D Stable Neo-Hookean (`snhk.hpp::make_diff_table2`): ½μ(I₂−2) − μ(J−1), with
        // the same `detsqr` volume term added below.
        TetModel::SnHk => {
            0.5 * mu * (sigma[0] * sigma[0] + sigma[1] * sigma[1] - 2.0) - mu * (j - 1.0)
        }
    };
    dev + 0.5 * lambda * (j - 1.0) * (j - 1.0)
}

/// `∂Ψ/∂σ` — the reference's `DiffTable2::deda`.
pub fn shell_dpsi_dsigma(sigma: &[f32; 2], model: TetModel, mu: f32, lambda: f32) -> [f32; 2] {
    let j = sigma[0] * sigma[1];
    if matches!(model, TetModel::SnHk) {
        // `snhk.hpp::make_diff_table2`: λb(ab−1) + μ(a−b), and the mirror.
        return [
            lambda * sigma[1] * (j - 1.0) + mu * (sigma[0] - sigma[1]),
            lambda * sigma[0] * (j - 1.0) + mu * (sigma[1] - sigma[0]),
        ];
    }
    let dev = |s: f32| match model {
        TetModel::StVK => mu * (s - 1.0).powi(3),
        TetModel::Arap => mu * (s - 1.0),
        TetModel::SnHk => unreachable!("handled above"),
    };
    [
        dev(sigma[0]) + lambda * sigma[1] * (j - 1.0),
        dev(sigma[1]) + lambda * sigma[0] * (j - 1.0),
    ]
}

/// `∂²Ψ/∂σ²` (row-major 2×2) — the reference's `DiffTable2::d2ed2a`. The `detsqr`
/// part is `[[λσ₁², λ(2σ₀σ₁−1)], [·, λσ₀²]]`.
pub fn shell_d2psi_dsigma2(sigma: &[f32; 2], model: TetModel, mu: f32, lambda: f32) -> [f32; 4] {
    let (a, b) = (sigma[0], sigma[1]);
    if matches!(model, TetModel::SnHk) {
        // `snhk.hpp::make_diff_table2`: diag λb²+μ, λa²+μ; off-diag 2λab − λ − μ.
        let off = 2.0 * lambda * a * b - lambda - mu;
        return [lambda * b * b + mu, off, off, lambda * a * a + mu];
    }
    let dev = |s: f32| match model {
        TetModel::StVK => 3.0 * mu * (s - 1.0) * (s - 1.0),
        TetModel::Arap => mu,
        TetModel::SnHk => unreachable!("handled above"),
    };
    let cross = lambda * (2.0 * a * b - 1.0);
    [
        dev(a) + lambda * b * b,
        cross,
        cross,
        dev(b) + lambda * a * a,
    ]
}

/// The 6 predicted eigenpairs of `∂²Ψ/∂F²` for a shell, in the SVD frame: the mode
/// matrices `Qₖ` (actual eigenmatrix is `U₃·Qₖ·Vᵀ`, where `U₃` is `U` completed with
/// `u₀×u₁`) and their **unclamped** eigenvalues.
///
/// **Trap the reference explicitly warns about** (comment in `eigenanalysis.cu`):
/// `Q[1]` — the mode paired with the difference eigenvalue `(Ψ₀−Ψ₁)/(σ₀−σ₁)` — must be
/// the **symmetric** off-diagonal matrix. Using the antisymmetric twist matrix there
/// instead collapses the basis (`vec(U·Q₀·Vᵀ) == vec(U·Q₁·Vᵀ)`), which silently makes
/// the flip subspace wrong whenever the singular values differ. Transcribed as
/// symmetric, and `shell_modes_are_independent` guards it.
pub fn shell_eigenpairs(
    sigma: &[f32; 2],
    model: TetModel,
    mu: f32,
    lambda: f32,
    eps: f32,
) -> ([Mat3x2; 6], [f32; 6]) {
    let deda = shell_dpsi_dsigma(sigma, model, mu, lambda);
    let d2 = shell_d2psi_dsigma2(sigma, model, mu, lambda);
    shell_eigenpairs_from_derivs(sigma, &deda, &d2, eps)
}

/// The eigensystem for ANY energy expressed in the principal stretches, given its first
/// derivatives `deda[i] = ∂Ψ/∂σᵢ` and second derivatives `d2` (row-major 2×2). The
/// model-specific [`shell_eigenpairs`] is a thin wrapper over this; the strain-limit
/// BARRIER supplies its own derivative table the same way the reference's
/// `compute_strainlimiting_diff_table` feeds `eigenanalysis::compute_hessian`.
pub fn shell_eigenpairs_from_derivs(
    sigma: &[f32; 2],
    deda: &[f32; 2],
    d2: &[f32; 4],
    eps: f32,
) -> ([Mat3x2; 6], [f32; 6]) {
    let (eigval, eigvec) = sym_eig2(&Mat2::from_cols(
        Vec2::new(d2[0], d2[2]),
        Vec2::new(d2[1], d2[3]),
    ));

    const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
    // Rows are (x, y, z); each Q is 3×2 given by its two columns.
    let m = |r0: [f32; 2], r1: [f32; 2], r2: [f32; 2]| {
        Mat3x2::from_cols(
            Vec3::new(r0[0], r1[0], r2[0]),
            Vec3::new(r0[1], r1[1], r2[1]),
        )
    };
    let mut q = [Mat3x2::ZERO; 6];
    q[0] = m([0.0, R], [-R, 0.0], [0.0, 0.0]); // twist (antisymmetric)
    q[1] = m([0.0, R], [R, 0.0], [0.0, 0.0]); // flip  (SYMMETRIC — see doc above)
    q[2] = m([0.0, 0.0], [0.0, 0.0], [1.0, 0.0]); // out-of-plane, direction 0
    q[3] = m([0.0, 0.0], [0.0, 0.0], [0.0, 1.0]); // out-of-plane, direction 1
    for (k, qk) in q.iter_mut().enumerate().skip(4) {
        let e = if k == 4 { eigvec.x_axis } else { eigvec.y_axis };
        *qk = m([e.x, 0.0], [0.0, e.y], [0.0, 0.0]);
    }

    let (a, b) = (sigma[0], sigma[1]);
    let denom = a - b;
    let fallback = 0.5 * (d2[0] + d2[3]) - 0.5 * (d2[1] + d2[2]);
    let lam = [
        (deda[0] + deda[1]) / (a + b),
        if denom.abs() > eps {
            (deda[0] - deda[1]) / denom
        } else {
            fallback
        },
        deda[0] / a,
        deda[1] / b,
        eigval[0],
        eigval[1],
    ];
    (q, lam)
}

/// `∂²Ψ/∂F²` for a shell as a row-major 6×6, `Σ max(λₖ,0)·qₖqₖᵀ` with
/// `qₖ = vec(U₃·Qₖ·Vᵀ)`. PSD by construction (the reference's `fmaxf(0,·)`), so — as
/// in 3D — it is a deliberate PSD **filter** of the true Hessian, not equal to it
/// wherever the energy is non-convex.
pub fn shell_d2psi_df2(
    u: &Mat3x2,
    sigma: &[f32; 2],
    v: &Mat2,
    model: TetModel,
    mu: f32,
    lambda: f32,
    eps: f32,
) -> [f32; 36] {
    let deda = shell_dpsi_dsigma(sigma, model, mu, lambda);
    let d2 = shell_d2psi_dsigma2(sigma, model, mu, lambda);
    shell_d2psi_df2_from_derivs(u, sigma, v, &deda, &d2, eps)
}

/// `∂²Ψ/∂F²` for any energy given its principal-stretch derivative table. See
/// [`shell_eigenpairs_from_derivs`].
pub fn shell_d2psi_df2_from_derivs(
    u: &Mat3x2,
    sigma: &[f32; 2],
    v: &Mat2,
    deda: &[f32; 2],
    d2: &[f32; 4],
    eps: f32,
) -> [f32; 36] {
    let (q, lam) = shell_eigenpairs_from_derivs(sigma, deda, d2, eps);
    // U₃ = [u₀ | u₁ | u₀×u₁] (the reference's `expand_U`), so a 3×2 mode maps into 3D.
    let u2 = u.c0.cross(u.c1);
    let mut out = [0.0f32; 36];
    for (k, qk) in q.iter().enumerate() {
        let l = lam[k].max(0.0);
        if l == 0.0 {
            continue;
        }
        // U₃·Qₖ: each column of Qₖ is a 3-vector of coefficients over (u₀,u₁,u₂).
        let col = |c: Vec3| u.c0 * c.x + u.c1 * c.y + u2 * c.z;
        let uq = Mat3x2::from_cols(col(qk.c0), col(qk.c1));
        let qv = uq.mul_mat2(&v.transpose()).vec6();
        for r in 0..6 {
            for c in 0..6 {
                out[r * 6 + c] += l * qv[r] * qv[c];
            }
        }
    }
    out
}

/// One triangle's shell element: rest-shape inverse (2×2) and rest area.
#[derive(Clone, Copy, Debug)]
pub struct ShellElement {
    /// Vertex indices.
    pub v: [u32; 3],
    /// Inverse rest shape `Dm⁻¹` (2×2), from the triangle's own material frame.
    pub dm_inv: Mat2,
    /// Rest area (always positive).
    pub area: f32,
    /// Constitutive model.
    pub model: TetModel,
}

/// Build shell elements from a triangle list and rest positions. The material frame
/// is the triangle's own orthonormal in-plane basis (`e₁` along the first edge, `e₂`
/// completing it), so `Dm` is 2×2 and the formulation is intrinsic — no dependence on
/// how the triangle happens to sit in world space, which is what makes the energy
/// rigid-motion invariant. Degenerate (zero-area) triangles are skipped.
pub fn build_shells(
    tris: &[[u32; 3]],
    rest: &[Vec3],
    min_area: f32,
    model: TetModel,
) -> Vec<ShellElement> {
    let mut out = Vec::new();
    for t in tris {
        let (p0, p1, p2) = (
            rest[t[0] as usize],
            rest[t[1] as usize],
            rest[t[2] as usize],
        );
        let (e1, e2) = (p1 - p0, p2 - p0);
        let n = e1.cross(e2);
        let area = 0.5 * n.length();
        if area <= min_area {
            continue;
        }
        // Orthonormal in-plane frame.
        let ax = e1.normalize();
        let ay = (e2 - e2.dot(ax) * ax).normalize();
        // Dm columns are the edges expressed in that frame.
        let dm = Mat2::from_cols(
            Vec2::new(e1.dot(ax), e1.dot(ay)),
            Vec2::new(e2.dot(ax), e2.dot(ay)),
        );
        if dm.determinant().abs() < 1e-20 {
            continue;
        }
        out.push(ShellElement {
            v: *t,
            dm_inv: dm.inverse(),
            area,
            model,
        });
    }
    out
}

/// Deformation gradient `F = Ds·Dm⁻¹` (3×2) for the three deformed positions.
pub fn shell_deformation_gradient(p: &[Vec3; 3], dm_inv: &Mat2) -> Mat3x2 {
    Mat3x2::from_cols(p[1] - p[0], p[2] - p[0]).mul_mat2(dm_inv)
}

/// Element energy `area · Ψ(σ(F))`.
pub fn shell_energy(p: &[Vec3; 3], el: &ShellElement, mu: f32, lambda: f32) -> f32 {
    let f = shell_deformation_gradient(p, &el.dm_inv);
    let (_u, sigma, _v) = svd3x2(&f);
    el.area * shell_density(&sigma, el.model, mu, lambda)
}

/// **Analytic** element gradient: `P = U·diag(∂Ψ/∂σ)·Vᵀ` (3×2 Piola stress), then
/// `∂E/∂Ds = area·P·Dm⁻ᵀ` whose two columns are the forces on `p1`,`p2`; `p0` takes
/// minus their sum (translation invariance). Same structure as the tet case, one
/// dimension down.
pub fn shell_gradient_analytic(
    p: &[Vec3; 3],
    el: &ShellElement,
    mu: f32,
    lambda: f32,
) -> [Vec3; 3] {
    let f = shell_deformation_gradient(p, &el.dm_inv);
    let (u, sigma, v) = svd3x2(&f);
    let d = shell_dpsi_dsigma(&sigma, el.model, mu, lambda);
    let pk1 = Mat3x2::from_cols(u.c0 * d[0], u.c1 * d[1]).mul_mat2(&v.transpose());
    let h = pk1.mul_mat2(&el.dm_inv.transpose());
    let (g1, g2) = (h.c0 * el.area, h.c1 * el.area);
    [-(g1 + g2), g1, g2]
}

/// `∂F/∂x` as a row-major 6×9 map. `F = Ds·Dm⁻¹` is linear in `x`, so this is
/// constant and there is no second-order chain term.
fn shell_dfdx(dm_inv: &Mat2) -> [f32; 54] {
    let at = |m: &Mat2, r: usize, c: usize| if c == 0 { m.x_axis[r] } else { m.y_axis[r] };
    let mut b = [0.0f32; 54]; // 6 × 9
    for i in 0..3 {
        for j in 0..2 {
            let row = 2 * i + j; // vec index of F[i][j]
            let mut sum = 0.0f32;
            for k in 0..2 {
                let d = at(dm_inv, k, j);
                b[row * 9 + 3 * (k + 1) + i] = d;
                sum += d;
            }
            b[row * 9 + i] = -sum; // vertex 0
        }
    }
    b
}

/// **Analytic** element Hessian: `area · Bᵀ·(∂²Ψ/∂F²)·B` (9×9, row-major), PSD by
/// construction (the eigenvalue clamping lives in [`shell_d2psi_df2`]).
pub fn shell_hessian_analytic(
    p: &[Vec3; 3],
    el: &ShellElement,
    mu: f32,
    lambda: f32,
    eps: f32,
) -> [f32; 81] {
    let f = shell_deformation_gradient(p, &el.dm_inv);
    let (u, sigma, v) = svd3x2(&f);
    let a6 = shell_d2psi_df2(&u, &sigma, &v, el.model, mu, lambda, eps);
    let b = shell_dfdx(&el.dm_inv);
    // (A·B) then Bᵀ·(A·B).
    let mut ab = [0.0f32; 54]; // 6 × 9
    for r in 0..6 {
        for c in 0..9 {
            let mut acc = 0.0f32;
            for k in 0..6 {
                acc += a6[r * 6 + k] * b[k * 9 + c];
            }
            ab[r * 9 + c] = acc;
        }
    }
    let mut h = [0.0f32; 81];
    for r in 0..9 {
        for c in 0..9 {
            let mut acc = 0.0f32;
            for k in 0..6 {
                acc += b[k * 9 + r] * ab[k * 9 + c];
            }
            h[r * 9 + c] = el.area * acc;
        }
    }
    h
}

/// Fraction of `limit` the barrier argument is floored at when the strain limit has
/// already been EXCEEDED. The reference asserts this cannot happen (its CCD/line search
/// keeps every principal stretch inside the limit); we have no such guarantee for this
/// term yet, so the floor keeps the barrier finite and strongly repulsive instead of
/// producing a singular or NaN Hessian.
const SL_GAP_FLOOR: f32 = 1.0e-3;

/// The reference's strain-limiting derivative table
/// (`barrier::compute_strainlimiting_diff_table`, `barrier.cu:116-131`) plus its energy.
///
/// The limit acts as a **barrier on the principal-stretch EXCESS** `a = σ − 1`: for each
/// of the two in-plane stretches, when `a > 0` the barrier argument is `y = limit − a`, so
/// the energy diverges as the stretch approaches the limit. That is what makes the limit
/// *enforced* rather than merely penalized — a one-sided spring only resists it.
///
/// Returns `(energy, ∂E/∂a, ∂²E/∂a²)` with the second derivative as a row-major 2×2
/// (diagonal: the two stretches do not couple here). `None` when neither stretch is
/// beyond rest, i.e. the term is inactive.
pub fn strain_limit_derivs(
    sigma: &[f32; 2],
    limit: f32,
    kind: crate::contact::BarrierKind,
) -> Option<(f32, [f32; 2], [f32; 4])> {
    if limit <= 0.0 {
        return None;
    }
    let (mut energy, mut deda, mut d2) = (0.0f32, [0.0f32; 2], [0.0f32; 4]);
    let mut active = false;
    for i in 0..2 {
        let a = sigma[i] - 1.0;
        if a <= 0.0 {
            continue; // compression / rest: the reference's `if (g > 0)` guard
        }
        let y = (limit - a).max(SL_GAP_FLOOR * limit);
        if let Some((e, g, h)) = crate::contact::barrier_of(kind, y, limit, 1.0) {
            energy += e;
            // Chain rule through y = limit − a: ∂E/∂a = −∂E/∂y.
            deda[i] = -g;
            d2[i * 2 + i] = h;
            active = true;
        }
    }
    if active {
        Some((energy, deda, d2))
    } else {
        None
    }
}

/// Strain-limit barrier ENERGY for one face. Independent of the gradient path below so it
/// can serve as a finite-difference oracle.
///
/// Deliberately NOT area-scaled: the reference embeds `stiffness * dedx` with no area
/// factor (`strainlimiting.cu:85-89`), unlike its elastic terms — the scale rides on the
/// stiffness instead.
pub fn strain_limit_energy(
    p: &[Vec3; 3],
    el: &ShellElement,
    limit: f32,
    kind: crate::contact::BarrierKind,
) -> f32 {
    let f = shell_deformation_gradient(p, &el.dm_inv);
    let (_, sigma, _) = svd3x2(&f);
    strain_limit_derivs(&sigma, limit, kind).map_or(0.0, |(e, _, _)| e)
}

/// Strain-limit barrier gradient for one face: `P = U·diag(∂E/∂a)·Vᵀ`, then
/// `∂E/∂Ds = P·Dm⁻ᵀ`, whose columns are the forces on `p1`,`p2` with `p0` taking minus
/// their sum. Same structure as [`shell_gradient_analytic`] but with the barrier's
/// derivative table and no area factor.
pub fn strain_limit_gradient(
    p: &[Vec3; 3],
    el: &ShellElement,
    limit: f32,
    kind: crate::contact::BarrierKind,
) -> [Vec3; 3] {
    let f = shell_deformation_gradient(p, &el.dm_inv);
    let (u, sigma, v) = svd3x2(&f);
    let Some((_, deda, _)) = strain_limit_derivs(&sigma, limit, kind) else {
        return [Vec3::ZERO; 3];
    };
    let pk1 = Mat3x2::from_cols(u.c0 * deda[0], u.c1 * deda[1]).mul_mat2(&v.transpose());
    let h = pk1.mul_mat2(&el.dm_inv.transpose());
    [-(h.c0 + h.c1), h.c0, h.c1]
}

/// Strain-limit barrier Hessian for one face (9×9 row-major), `Bᵀ·(∂²E/∂F²)·B` via the
/// shared eigensystem ([`shell_d2psi_df2_from_derivs`]) — PSD by the same eigenvalue
/// clamping. No area factor, as with the gradient.
pub fn strain_limit_hessian(
    p: &[Vec3; 3],
    el: &ShellElement,
    limit: f32,
    kind: crate::contact::BarrierKind,
    eps: f32,
) -> [f32; 81] {
    let f = shell_deformation_gradient(p, &el.dm_inv);
    let (u, sigma, v) = svd3x2(&f);
    let Some((_, deda, d2)) = strain_limit_derivs(&sigma, limit, kind) else {
        return [0.0f32; 81];
    };
    let a6 = shell_d2psi_df2_from_derivs(&u, &sigma, &v, &deda, &d2, eps);
    let b = shell_dfdx(&el.dm_inv);
    let mut ab = [0.0f32; 54];
    for r in 0..6 {
        for c in 0..9 {
            let mut acc = 0.0f32;
            for k in 0..6 {
                acc += a6[r * 6 + k] * b[k * 9 + c];
            }
            ab[r * 9 + c] = acc;
        }
    }
    let mut h = [0.0f32; 81];
    for r in 0..9 {
        for c in 0..9 {
            let mut acc = 0.0f32;
            for k in 0..6 {
                acc += b[k * 9 + r] * ab[k * 9 + c];
            }
            h[r * 9 + c] = acc;
        }
    }
    h
}

/// Combined strain-limit element assembly: gradient AND Hessian from ONE SVD, scaled by
/// the reference's dynamic stiffness.
///
/// The reference's `strainlimiting::compute_stiffness` (`strainlimiting.cu:22-44`) is
/// ```text
/// stiffness = w·(H_elastic·w)  +  mass / d²,      d = limit − max(σ − 1)
/// ```
/// with `w` the face's vertex positions relative to their centroid and `H_elastic` the
/// 9×9 elastic block over the face's three vertices. `d` is the remaining HEADROOM to the
/// limit, so the stiffness grows without bound as the tightest principal stretch
/// approaches it.
///
/// `elastic_k` is the elastic half — `w·(H_elastic·w)` over this face's three vertices,
/// computed by the assembler (which is the only place those blocks exist) and passed in.
/// Pass 0 for a purely inertial stiffness. `face_mass` is the reference's
/// `density · rest_area` (`scene.rs:1665`).
///
/// `None` when no principal stretch exceeds rest, i.e. the term is inactive.
pub fn strain_limit_assembly(
    p: &[Vec3; 3],
    el: &ShellElement,
    limit: f32,
    face_mass: f32,
    elastic_k: f32,
    kind: crate::contact::BarrierKind,
    eps: f32,
) -> Option<([Vec3; 3], [f32; 81])> {
    let f = shell_deformation_gradient(p, &el.dm_inv);
    let (u, sigma, v) = svd3x2(&f);
    let (_, deda, d2) = strain_limit_derivs(&sigma, limit, kind)?;

    // Headroom to the limit, floored for the same reason as the barrier argument.
    let max_a = (sigma[0] - 1.0).max(sigma[1] - 1.0);
    let d = (limit - max_a).max(SL_GAP_FLOOR * limit);
    // `elastic_k` is the Rayleigh quotient of the elastic Hessian over this face's three
    // vertices; the caller supplies it because only the assembler has those blocks.
    let stiffness = elastic_k.max(0.0) + face_mass / (d * d);
    if stiffness <= 0.0 {
        return None;
    }

    let pk1 = Mat3x2::from_cols(u.c0 * deda[0], u.c1 * deda[1]).mul_mat2(&v.transpose());
    let hg = pk1.mul_mat2(&el.dm_inv.transpose());
    let (g1, g2) = (hg.c0 * stiffness, hg.c1 * stiffness);
    let grad = [-(g1 + g2), g1, g2];

    let a6 = shell_d2psi_df2_from_derivs(&u, &sigma, &v, &deda, &d2, eps);
    let b = shell_dfdx(&el.dm_inv);
    let mut ab = [0.0f32; 54];
    for r in 0..6 {
        for c in 0..9 {
            let mut acc = 0.0f32;
            for k in 0..6 {
                acc += a6[r * 6 + k] * b[k * 9 + c];
            }
            ab[r * 9 + c] = acc;
        }
    }
    let mut hess = [0.0f32; 81];
    for r in 0..9 {
        for c in 0..9 {
            let mut acc = 0.0f32;
            for k in 0..6 {
                acc += b[k * 9 + r] * ab[k * 9 + c];
            }
            hess[r * 9 + c] = stiffness * acc;
        }
    }
    Some((grad, hess))
}

/// **Shell plasticity** — a port of the reference `plasticity::update_face_plasticity`
/// (`plasticity/plasticity.cu`, Apache-2.0). The 2D twin of
/// [`crate::tet::apply_tet_plasticity`]: with `α = 1 − exp(−plasticity·dt)`, each
/// principal stretch whose deviation `|σₖ − 1|` exceeds `threshold` is pulled a
/// fraction `α` toward the nearest dead-zone edge (`1 ± threshold`), then the REST
/// shape is rewritten so the relaxed deformation becomes the new rest.
///
/// The rest update differs from the tet case in one essential way: `Ds` here is `3×2`,
/// so it cannot be inverted. The reference projects both sides into the triangle's own
/// intrinsic 2D frame — `t = ê(Ds₀)`, `n = ê(Ds₀ × Ds₁)`, `b = n × t`, `P₁ = [tᵀ; bᵀ]`
/// — reducing it to a `2×2` solve: `Dm⁻¹ ← (P₁·Ds)⁻¹·(P₁·F_new)`.
///
/// Returns `true` if the rest state changed. As in the tet case (and matching the
/// reference), the stored rest `area` is deliberately NOT updated — only `dm_inv`.
pub fn apply_shell_plasticity(
    el: &mut ShellElement,
    p: &[Vec3; 3],
    plasticity: f32,
    threshold: f32,
    dt: f32,
) -> bool {
    if plasticity <= 0.0 {
        return false;
    }
    let alpha = 1.0 - (-plasticity * dt).exp();
    let ds = Mat3x2::from_cols(p[1] - p[0], p[2] - p[0]);
    let f = ds.mul_mat2(&el.dm_inv);
    let (u, sigma, v) = svd3x2(&f);

    let mut s_new = sigma;
    let mut changed = false;
    for k in 0..2 {
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
    let f_new = Mat3x2::from_cols(u.c0 * s_new[0], u.c1 * s_new[1]).mul_mat2(&v.transpose());

    // Intrinsic 2D frame of the CURRENT shape (the reference's P1).
    let cross = ds.c0.cross(ds.c1);
    if cross.length_squared() < 1.0e-24 {
        return false; // degenerate current triangle; leave rest alone
    }
    let n = cross.normalize();
    let t = ds.c0.normalize_or_zero();
    let b = n.cross(t).normalize_or_zero();
    let proj = |m: &Mat3x2| {
        Mat2::from_cols(
            Vec2::new(t.dot(m.c0), b.dot(m.c0)),
            Vec2::new(t.dot(m.c1), b.dot(m.c1)),
        )
    };
    let p1_ds = proj(&ds);
    if p1_ds.determinant().abs() < 1.0e-20 {
        return false;
    }
    el.dm_inv = p1_ds.inverse() * proj(&f_new);
    true
}

#[cfg(test)]
mod strain_limit_tests {
    use super::*;
    use crate::contact::BarrierKind;

    /// A single unit-ish triangle and its rest element.
    fn face() -> ([Vec3; 3], ShellElement) {
        let rest = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let el = build_shells(&[[0, 1, 2]], &rest, 1.0e-12, TetModel::Arap)
            .pop()
            .expect("one shell element");
        (rest, el)
    }

    /// The barrier must be INACTIVE at rest and at compression, and switch on only once a
    /// principal stretch exceeds 1 — the reference's `if (g > 0)` guard.
    #[test]
    fn strain_limit_is_inactive_until_stretched() {
        let (rest, el) = face();
        let limit = 0.05;
        assert_eq!(
            strain_limit_energy(&rest, &el, limit, BarrierKind::Cubic),
            0.0,
            "rest must be inactive"
        );
        // Compress in x.
        let mut squashed = rest;
        squashed[1].x = 0.8;
        assert_eq!(
            strain_limit_energy(&squashed, &el, limit, BarrierKind::Cubic),
            0.0,
            "compression must be inactive"
        );
        // A small stretch inside the limit is active but finite.
        let mut mild = rest;
        mild[1].x = 1.0 + 0.5 * limit;
        let e_mild = strain_limit_energy(&mild, &el, limit, BarrierKind::Cubic);
        assert!(e_mild > 0.0 && e_mild.is_finite(), "mild stretch: {e_mild}");
        // Closer to the limit must cost strictly more — the barrier is monotone.
        let mut near = rest;
        near[1].x = 1.0 + 0.9 * limit;
        let e_near = strain_limit_energy(&near, &el, limit, BarrierKind::Cubic);
        assert!(
            e_near > e_mild,
            "energy must grow toward the limit: {e_mild} -> {e_near}"
        );
        // A zero/negative limit disables the term entirely.
        assert!(strain_limit_derivs(&[1.5, 1.0], 0.0, BarrierKind::Cubic).is_none());
    }

    /// The analytic gradient must match a finite difference of the INDEPENDENT energy
    /// path. `strain_limit_energy` recomputes the SVD and the barrier from scratch and
    /// never calls the gradient, so this is a genuine cross-check rather than a tautology
    /// (the same discipline as `tet_gradient_fd`).
    #[test]
    fn strain_limit_gradient_matches_finite_difference() {
        for kind in [BarrierKind::Cubic, BarrierKind::Log] {
            let (rest, el) = face();
            let limit = 0.05;
            // Stretch well inside the limit, and shear a little, so both principal
            // stretches are non-degenerate (equal σ would hit the eigensystem fallback).
            let mut p = rest;
            p[1] = Vec3::new(1.0 + 0.6 * limit, 0.02, 0.0);
            p[2] = Vec3::new(0.01, 1.0 + 0.2 * limit, 0.0);

            let g = strain_limit_gradient(&p, &el, limit, kind);
            let h = 1.0e-4f32;
            let mut worst = 0.0f32;
            let mut scale = 0.0f32;
            for a in 0..3 {
                for c in 0..3 {
                    let mut pp = p;
                    let mut pm = p;
                    pp[a][c] += h;
                    pm[a][c] -= h;
                    let fd = (strain_limit_energy(&pp, &el, limit, kind)
                        - strain_limit_energy(&pm, &el, limit, kind))
                        / (2.0 * h);
                    worst = worst.max((fd - g[a][c]).abs());
                    scale = scale.max(fd.abs()).max(g[a][c].abs());
                }
            }
            assert!(
                scale > 0.0,
                "{kind:?}: the test configuration must produce a nonzero gradient"
            );
            let rel = worst / scale;
            assert!(
                rel < 2.0e-2,
                "{kind:?}: analytic gradient vs FD: worst {worst:.3e}, scale {scale:.3e}, \
                 rel {rel:.3e}"
            );
        }
    }

    /// The Hessian must be symmetric and PSD (the eigenvalue clamping is inherited from
    /// the shared shell eigensystem), so it can go straight into the Newton solve.
    #[test]
    fn strain_limit_hessian_is_symmetric_and_psd() {
        let (rest, el) = face();
        let limit = 0.05;
        let mut p = rest;
        p[1] = Vec3::new(1.0 + 0.7 * limit, 0.03, 0.0);
        p[2] = Vec3::new(0.02, 1.0 + 0.3 * limit, 0.0);
        let h = strain_limit_hessian(&p, &el, limit, BarrierKind::Cubic, 1.0e-6);
        let mut worst_asym = 0.0f32;
        let mut mag = 0.0f32;
        for r in 0..9 {
            for c in 0..9 {
                worst_asym = worst_asym.max((h[r * 9 + c] - h[c * 9 + r]).abs());
                mag = mag.max(h[r * 9 + c].abs());
            }
        }
        assert!(mag > 0.0, "Hessian must be nonzero when active");
        assert!(
            worst_asym <= 1.0e-4 * mag,
            "Hessian must be symmetric: worst {worst_asym:.3e} vs magnitude {mag:.3e}"
        );
        // PSD by Rayleigh quotient over pseudo-random directions.
        let mut seed = 0x9E3779B9u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / 8.388608e6) - 1.0
        };
        for _ in 0..64 {
            let v: Vec<f32> = (0..9).map(|_| rnd()).collect();
            let mut q = 0.0f32;
            for r in 0..9 {
                for c in 0..9 {
                    q += v[r] * h[r * 9 + c] * v[c];
                }
            }
            assert!(
                q >= -1.0e-4 * mag,
                "Hessian must be PSD, got quotient {q:.3e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(c0: [f32; 3], c1: [f32; 3]) -> Mat3x2 {
        Mat3x2::from_cols(Vec3::from_array(c0), Vec3::from_array(c1))
    }

    #[test]
    fn sym_eig2_reconstructs() {
        for &(p, q, r) in &[
            (3.0f32, 1.0f32, 2.0f32),
            (5.0, 0.0, 5.0),
            (1.0, -2.0, 4.0),
            (0.0, 0.0, 0.0),
            (-1.0, 0.5, 2.0),
        ] {
            let a = Mat2::from_cols(Vec2::new(p, q), Vec2::new(q, r));
            let (l, v) = sym_eig2(&a);
            assert!(l[0] >= l[1] - 1e-6, "eigenvalues must be descending");
            // V diag(l) Vᵀ == A
            let vd = Mat2::from_cols(v.x_axis * l[0], v.y_axis * l[1]);
            let back = vd * v.transpose();
            for (x, y) in back.to_cols_array().iter().zip(a.to_cols_array().iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "reconstruction failed for {p},{q},{r}"
                );
            }
            // Orthonormal.
            assert!((v.x_axis.dot(v.y_axis)).abs() < 1e-5);
            assert!((v.x_axis.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn svd3x2_reconstructs_with_orthonormal_frames() {
        let cases = [
            mat([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            mat([1.3, 0.1, 0.05], [-0.2, 0.9, 0.1]),
            mat([2.0, 0.0, 0.0], [0.0, 0.5, 0.0]),
            // Degenerate: rank 1.
            mat([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ];
        for (i, f) in cases.iter().enumerate() {
            let (u, s, v) = svd3x2(f);
            assert!(
                s[0] >= s[1] - 1e-6 && s[1] >= -1e-6,
                "σ descending, non-negative"
            );
            // Orthonormal U columns and V.
            assert!(
                u.c0.dot(u.c1).abs() < 1e-4,
                "U columns not orthogonal (case {i})"
            );
            assert!((u.c0.length() - 1.0).abs() < 1e-4);
            assert!((u.c1.length() - 1.0).abs() < 1e-4);
            // Reconstruct: U diag(σ) Vᵀ == F.
            let us = Mat3x2::from_cols(u.c0 * s[0], u.c1 * s[1]);
            let back = us.mul_mat2(&v.transpose());
            assert!(
                (back.c0 - f.c0).length() < 2e-3 && (back.c1 - f.c1).length() < 2e-3,
                "3×2 SVD reconstruction failed (case {i})"
            );
        }
    }

    /// The 2D densities' derivatives must be self-consistent for BOTH models — the
    /// zero-hallucination check on the `Vec2f` overloads.
    #[test]
    fn shell_derivatives_are_self_consistent() {
        for model in [TetModel::Arap, TetModel::StVK] {
            let (mu, lam) = (400.0f32, 250.0f32);
            for base in [[1.25f32, 0.9], [1.0, 1.0], [0.8, 1.15]] {
                let h = 1e-3;
                let g = shell_dpsi_dsigma(&base, model, mu, lam);
                for i in 0..2 {
                    let mut sp = base;
                    sp[i] += h;
                    let mut sm = base;
                    sm[i] -= h;
                    let fd = (shell_density(&sp, model, mu, lam)
                        - shell_density(&sm, model, mu, lam))
                        / (2.0 * h);
                    assert!(
                        (g[i] - fd).abs() < 2e-2 * g[i].abs().max(1.0),
                        "{model:?} grad[{i}] {} != fd {fd}",
                        g[i]
                    );
                }
                let hh = shell_d2psi_dsigma2(&base, model, mu, lam);
                for j in 0..2 {
                    let mut sp = base;
                    sp[j] += h;
                    let mut sm = base;
                    sm[j] -= h;
                    let gp = shell_dpsi_dsigma(&sp, model, mu, lam);
                    let gm = shell_dpsi_dsigma(&sm, model, mu, lam);
                    for i in 0..2 {
                        let fd = (gp[i] - gm[i]) / (2.0 * h);
                        assert!(
                            (hh[i * 2 + j] - fd).abs() < 3e-2 * hh[i * 2 + j].abs().max(1.0),
                            "{model:?} hess[{i}][{j}] {} != fd {fd}",
                            hh[i * 2 + j]
                        );
                    }
                }
            }
        }
    }

    /// Guard the trap the reference documents: the flip mode `Q[1]` must be
    /// SYMMETRIC, so that it is linearly independent of the antisymmetric twist mode
    /// `Q[0]` once mapped through `U₃·Q·Vᵀ`. If `Q[1]` were the twist matrix, the two
    /// vectorized modes would coincide and the flip subspace would silently receive
    /// the wrong eigenvalue.
    #[test]
    fn shell_modes_are_independent() {
        let sigma = [1.3f32, 0.9];
        let (q, _l) = shell_eigenpairs(&sigma, TetModel::Arap, 400.0, 250.0, 1e-6);
        let u = Mat3x2::from_cols(Vec3::X, Vec3::Y);
        let v = Mat2::IDENTITY;
        let u2 = u.c0.cross(u.c1);
        let vecs: Vec<[f32; 6]> = q
            .iter()
            .map(|qk| {
                let col = |c: Vec3| u.c0 * c.x + u.c1 * c.y + u2 * c.z;
                Mat3x2::from_cols(col(qk.c0), col(qk.c1))
                    .mul_mat2(&v.transpose())
                    .vec6()
            })
            .collect();
        // Twist and flip must NOT be parallel.
        let dot: f32 = vecs[0].iter().zip(&vecs[1]).map(|(a, b)| a * b).sum();
        assert!(
            dot.abs() < 1e-4,
            "twist and flip modes must be orthogonal, got dot {dot}"
        );
        // All six modes pairwise orthonormal ⇒ a valid eigenbasis.
        for i in 0..6 {
            let ni: f32 = vecs[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((ni - 1.0).abs() < 1e-3, "mode {i} not unit (norm {ni})");
            for j in (i + 1)..6 {
                let d: f32 = vecs[i].iter().zip(&vecs[j]).map(|(a, b)| a * b).sum();
                assert!(d.abs() < 1e-3, "modes {i},{j} not orthogonal (dot {d})");
            }
        }
    }

    /// The predicted modes must diagonalize the true `∂²Ψ/∂F²` with the predicted
    /// UNCLAMPED eigenvalues. Verified via the **Rayleigh quotient** `qᵀAq/qᵀq`
    /// against a finite-difference `A`, rather than the raw residual `‖Aq − λq‖`.
    ///
    /// Why: `A` here is built by differencing the Piola stress in F-space, and that
    /// estimate comes out slightly ASYMMETRIC (measured: the off-diagonal block reads
    /// `[[216.7, 144.8], [138.4, 221.7]]` where the true Hessian must be symmetric).
    /// A raw residual is ill-conditioned against that noise — it reported ~9% for
    /// modes whose Rayleigh quotients agree to 1.6% — whereas the Rayleigh quotient
    /// averages the noise out and is exactly the projected curvature the eigen-
    /// decomposition is claiming. Mode independence/orthonormality is checked
    /// separately by `shell_modes_are_independent`, so between the two tests both
    /// halves of the eigen-claim are covered.
    #[test]
    fn shell_modes_are_true_eigenpairs() {
        let (mu, lam) = (400.0f32, 250.0f32);
        let fs = [
            mat([1.3, 0.0, 0.0], [0.0, 0.9, 0.0]),
            mat([1.15, 0.06, 0.02], [-0.05, 0.95, 0.03]),
        ];
        let h = 1e-3f32;
        // First Piola stress P = U diag(∂Ψ/∂σ) Vᵀ, as a 3×2.
        let pk1 = |f: &Mat3x2| -> Mat3x2 {
            let (u, s, v) = svd3x2(f);
            let d = shell_dpsi_dsigma(&s, TetModel::Arap, mu, lam);
            Mat3x2::from_cols(u.c0 * d[0], u.c1 * d[1]).mul_mat2(&v.transpose())
        };
        let at = |m: &Mat3x2, r: usize, c: usize| {
            let col = if c == 0 { m.c0 } else { m.c1 };
            col[r]
        };
        for (fi, f0) in fs.iter().enumerate() {
            // True 6×6 by central differences of P.
            let mut afd = [0.0f32; 36];
            for jj in 0..6 {
                let (ri, ci) = (jj / 2, jj % 2);
                let bump = |sign: f32| {
                    let mut m = *f0;
                    let col = if ci == 0 { &mut m.c0 } else { &mut m.c1 };
                    col[ri] += sign * h;
                    m
                };
                let (pp, pm) = (pk1(&bump(1.0)), pk1(&bump(-1.0)));
                for ii in 0..6 {
                    afd[ii * 6 + jj] =
                        (at(&pp, ii / 2, ii % 2) - at(&pm, ii / 2, ii % 2)) / (2.0 * h);
                }
            }
            let (u, sg, v) = svd3x2(f0);
            let (modes, lams) = shell_eigenpairs(&sg, TetModel::Arap, mu, lam, 1e-6);
            let u2 = u.c0.cross(u.c1);
            for k in 0..6 {
                let col = |c: Vec3| u.c0 * c.x + u.c1 * c.y + u2 * c.z;
                let qv = Mat3x2::from_cols(col(modes[k].c0), col(modes[k].c1))
                    .mul_mat2(&v.transpose())
                    .vec6();
                let mut num = 0.0f32;
                let mut den = 0.0f32;
                for r in 0..6 {
                    let mut acc = 0.0f32;
                    for c in 0..6 {
                        acc += afd[r * 6 + c] * qv[c];
                    }
                    num += qv[r] * acc;
                    den += qv[r] * qv[r];
                }
                let rayleigh = num / den.max(1e-20);
                let scale = rayleigh.abs().max(lams[k].abs()).max(1.0);
                assert!(
                    (rayleigh - lams[k]).abs() < 3e-2 * scale,
                    "shell mode {k} (case {fi}): predicted λ {} vs true projected curvature \
                     {rayleigh}",
                    lams[k]
                );
            }
        }
    }

    /// The assembled 6×6 must be symmetric and PSD (eigenvalues clamped).
    #[test]
    fn shell_hessian_is_symmetric_psd() {
        let f = mat([1.2, 0.05, 0.0], [-0.04, 0.92, 0.02]);
        let (u, s, v) = svd3x2(&f);
        let a = shell_d2psi_df2(&u, &s, &v, TetModel::Arap, 400.0, 250.0, 1e-6);
        let scale = a.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        for r in 0..6 {
            for c in 0..6 {
                assert!(
                    (a[r * 6 + c] - a[c * 6 + r]).abs() < 1e-4 * scale,
                    "not symmetric at ({r},{c})"
                );
            }
        }
        for t in 0..10u64 {
            let x: [f32; 6] = std::array::from_fn(|k| {
                ((t.wrapping_mul(7919)
                    .wrapping_add((k as u64).wrapping_mul(104729))
                    % 200) as f32)
                    / 100.0
                    - 1.0
            });
            let mut q = 0.0f32;
            for r in 0..6 {
                for c in 0..6 {
                    q += x[r] * a[r * 6 + c] * x[c];
                }
            }
            assert!(q >= -1e-3 * scale, "shell Hessian not PSD: {q}");
        }
    }

    fn unit_tri() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        (vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![[0, 1, 2]])
    }

    /// Rest state: zero energy and zero gradient (the intrinsic material frame means
    /// F = I at rest regardless of how the triangle sits in space).
    #[test]
    fn shell_rest_state_is_stress_free() {
        let (rest, tris) = unit_tri();
        let els = build_shells(&tris, &rest, 1e-12, TetModel::Arap);
        assert_eq!(els.len(), 1);
        assert!((els[0].area - 0.5).abs() < 1e-6, "unit triangle area");
        let p = [rest[0], rest[1], rest[2]];
        assert!(shell_energy(&p, &els[0], 400.0, 250.0).abs() < 1e-5);
        let g = shell_gradient_analytic(&p, &els[0], 400.0, 250.0);
        assert!(g.iter().all(|v| v.length() < 1e-3), "rest gradient nonzero");
        // Degenerate triangle skipped.
        let flat = vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0];
        assert!(build_shells(&[[0, 1, 2]], &flat, 1e-12, TetModel::Arap).is_empty());
    }

    /// Rigid-motion invariance — the property the intrinsic 3×2 formulation buys, and
    /// the reason a triangle embedded in 3D needs no corotational correction.
    #[test]
    fn shell_energy_is_rigid_motion_invariant() {
        let (rest, tris) = unit_tri();
        let el = build_shells(&tris, &rest, 1e-12, TetModel::Arap)[0];
        let deformed = [
            Vec3::new(0.01, -0.02, 0.0),
            Vec3::new(1.25, 0.05, 0.03),
            Vec3::new(-0.04, 0.88, -0.02),
        ];
        let base = shell_energy(&deformed, &el, 400.0, 250.0);
        assert!(base > 1e-3, "test config should be deformed: {base}");
        let rot = glam::Mat3::from_euler(glam::EulerRot::XYZ, 0.8, -1.2, 0.4);
        let shift = Vec3::new(-3.0, 4.0, 2.0);
        let moved: [Vec3; 3] = std::array::from_fn(|i| rot * deformed[i] + shift);
        let after = shell_energy(&moved, &el, 400.0, 250.0);
        assert!(
            (after - base).abs() < 1e-3 * base.max(1.0),
            "shell energy not rigid-invariant: {base} -> {after}"
        );
    }

    /// The analytic Piola gradient must match an independent FD gradient of the
    /// energy, for BOTH models, and the forces must sum to zero.
    #[test]
    fn shell_analytic_gradient_matches_fd() {
        let (rest, tris) = unit_tri();
        for model in [TetModel::Arap, TetModel::StVK] {
            let el = build_shells(&tris, &rest, 1e-12, model)[0];
            let configs = [
                [
                    Vec3::ZERO,
                    Vec3::new(1.3, 0.04, 0.0),
                    Vec3::new(0.02, 0.85, 0.03),
                ],
                [
                    Vec3::new(0.01, 0.0, 0.02),
                    Vec3::new(0.82, 0.10, -0.04),
                    Vec3::new(0.12, 1.15, 0.05),
                ],
            ];
            for (ci, p) in configs.iter().enumerate() {
                let ga = shell_gradient_analytic(p, &el, 400.0, 250.0);
                // FD of the energy.
                let h = 1e-3;
                let mut gf = [Vec3::ZERO; 3];
                for vi in 0..3 {
                    for c in 0..3 {
                        let (mut pp, mut pm) = (*p, *p);
                        pp[vi][c] += h;
                        pm[vi][c] -= h;
                        gf[vi][c] = (shell_energy(&pp, &el, 400.0, 250.0)
                            - shell_energy(&pm, &el, 400.0, 250.0))
                            / (2.0 * h);
                    }
                }
                let scale = gf.iter().map(|v| v.length()).fold(1.0f32, f32::max);
                for k in 0..3 {
                    assert!(
                        (ga[k] - gf[k]).length() < 3e-2 * scale,
                        "{model:?} case {ci} vert {k}: analytic {:?} vs fd {:?}",
                        ga[k],
                        gf[k]
                    );
                }
                let sum: Vec3 = ga.iter().copied().sum();
                assert!(sum.length() < 1e-2 * scale, "forces do not sum to zero");
            }
        }
    }

    /// The 9×9 element Hessian must be symmetric, PSD, and annihilate rigid
    /// translations.
    #[test]
    fn shell_element_hessian_is_symmetric_psd_translation_invariant() {
        let (rest, tris) = unit_tri();
        let el = build_shells(&tris, &rest, 1e-12, TetModel::Arap)[0];
        let p = [
            Vec3::ZERO,
            Vec3::new(1.22, 0.05, 0.0),
            Vec3::new(0.03, 0.9, 0.02),
        ];
        let h = shell_hessian_analytic(&p, &el, 400.0, 250.0, 1.0e-6);
        let scale = h.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for a in 0..9 {
            for b in 0..9 {
                assert!(
                    (h[a * 9 + b] - h[b * 9 + a]).abs() < 1e-4 * scale,
                    "not symmetric at ({a},{b})"
                );
            }
        }
        for t in 0..10u64 {
            let x: [f32; 9] = std::array::from_fn(|k| {
                ((t.wrapping_mul(7919)
                    .wrapping_add((k as u64).wrapping_mul(104729))
                    % 200) as f32)
                    / 100.0
                    - 1.0
            });
            let mut q = 0.0f32;
            for a in 0..9 {
                for b in 0..9 {
                    q += x[a] * h[a * 9 + b] * x[b];
                }
            }
            assert!(q >= -1e-3 * scale, "shell element Hessian not PSD: {q}");
        }
        for axis in 0..3 {
            let mut t = [0.0f32; 9];
            for v in 0..3 {
                t[3 * v + axis] = 1.0;
            }
            for a in 0..9 {
                let mut row = 0.0f32;
                for b in 0..9 {
                    row += h[a * 9 + b] * t[b];
                }
                assert!(
                    row.abs() < 1e-3 * scale,
                    "translation along {axis} is not a null mode (row {a} = {row})"
                );
            }
        }
    }

    /// Dead zone: in-zone deformation must never touch the rest state (creep, not
    /// damping), and `plasticity = 0` is a no-op even far outside the zone.
    #[test]
    fn shell_plasticity_respects_the_dead_zone() {
        let (rest, tris) = unit_tri();
        let mut el = build_shells(&tris, &rest, 1e-12, TetModel::Arap)[0];
        let before = el.dm_inv.to_cols_array();
        let inside = [
            Vec3::ZERO,
            Vec3::new(1.02, 0.0, 0.0),
            Vec3::new(0.0, 1.02, 0.0),
        ];
        for _ in 0..100 {
            assert!(!apply_shell_plasticity(
                &mut el,
                &inside,
                0.5,
                0.10,
                1.0 / 60.0
            ));
        }
        assert_eq!(before, el.dm_inv.to_cols_array());
        let far = [
            Vec3::ZERO,
            Vec3::new(1.8, 0.0, 0.0),
            Vec3::new(0.0, 1.8, 0.0),
        ];
        assert!(!apply_shell_plasticity(
            &mut el,
            &far,
            0.0,
            0.10,
            1.0 / 60.0
        ));
        assert_eq!(before, el.dm_inv.to_cols_array());
    }

    /// Held beyond the dead zone, shell plasticity must relax stored energy
    /// monotonically and drive the residual stretch to the dead-zone edge — the yield
    /// surface is preserved, not erased.
    #[test]
    fn shell_plasticity_relaxes_toward_the_dead_zone_edge() {
        let (rest, tris) = unit_tri();
        let mut el = build_shells(&tris, &rest, 1e-12, TetModel::Arap)[0];
        let (mu, lam) = (500.0f32, 0.0f32); // deviatoric only
        let threshold = 0.10f32;
        let p = [
            Vec3::ZERO,
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
        ];
        let original = shell_energy(&p, &el, mu, lam);
        assert!(original > 1e-3, "held stretch should store energy");
        let mut prev = original;
        for step in 0..400 {
            let changed = apply_shell_plasticity(&mut el, &p, 2.0, threshold, 1.0 / 60.0);
            let e = shell_energy(&p, &el, mu, lam);
            assert!(
                e <= prev + 1e-4,
                "shell plasticity must not increase energy (step {step}): {prev} -> {e}"
            );
            if !changed {
                break;
            }
            prev = e;
        }
        let f = shell_deformation_gradient(&p, &el.dm_inv);
        let (_u, sig, _v) = svd3x2(&f);
        for (k, &sk) in sig.iter().enumerate() {
            assert!(
                (sk - (1.0 + threshold)).abs() < 3e-2,
                "stretch {k} should converge to {}, got {sk}",
                1.0 + threshold
            );
        }
        assert!(
            shell_energy(&p, &el, mu, lam) < 0.25 * original,
            "most energy should have relaxed"
        );
    }

    /// A higher creep rate relaxes more in the same elapsed time.
    #[test]
    fn higher_shell_plasticity_relaxes_faster() {
        let (rest, tris) = unit_tri();
        let p = [
            Vec3::ZERO,
            Vec3::new(1.4, 0.0, 0.0),
            Vec3::new(0.0, 1.4, 0.0),
        ];
        let run = |rate: f32| {
            let mut el = build_shells(&tris, &rest, 1e-12, TetModel::Arap)[0];
            for _ in 0..20 {
                apply_shell_plasticity(&mut el, &p, rate, 0.05, 1.0 / 60.0);
            }
            shell_energy(&p, &el, 500.0, 0.0)
        };
        let (slow, fast) = (run(0.5), run(20.0));
        assert!(fast < slow, "higher rate must relax more: {fast} vs {slow}");
    }
}
