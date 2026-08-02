// SPDX-License-Identifier: GPL-2.0-or-later
//! Rod (strand) bending — a `½·k·(θ − θ₀)²` angle energy at each INTERIOR rod vertex.
//!
//! Ported from the reference's `embed_rod_bend_force_hessian`
//! (`ppf-cts-solver/src/cpp/energy/energy.cu:495`) and
//! `dihedral_angle::strand_compute_force_hessian`
//! (`ppf-cts-solver/src/cpp/energy/model/dihedral_angle.hpp:450`).
//!
//! Unlike cloth hinge bending (4 vertices, a dihedral between two faces), a strand bend
//! is 3 vertices: the angle `θ` at the centre vertex between its two incident rod
//! segments. `θ = atan2(|e₀×e₁|, e₀·e₁) ∈ [0, π]`, so a STRAIGHT strand has `θ = π`,
//! which is also the reference's default rest angle (`builder.rs:385-388`).
//!
//! The Hessian is the Wu & Kim analytic eigendecomposition: all six eigenpairs of the
//! 6-DOF `(e₀, e₁)` Hessian are formed in closed form and the negative ones are dropped,
//! so the result is PSD by construction. Each 6-D eigenvector `[qa; qb]` is mapped to the
//! 9 DOFs as `[qa; −qa−qb; qb]` (the centre vertex moves opposite to both edges).
//!
//! Where the eigenvalues are all positive the projection is EXACT — that is what
//! `projection_dominates_the_true_hessian_and_equals_it_when_nothing_is_clamped` verifies.

use glam::Vec3;

/// One rod bend element: the angle at interior rod vertex `i`, between neighbours `j`
/// and `k`. The reference gates these on "exactly 2 incident rod edges and 0 incident
/// faces" (`energy.cu:496`).
#[derive(Clone, Copy, Debug)]
pub struct RodHinge {
    /// First neighbour (the reference's `j`, DOF block 0).
    pub j: u32,
    /// Centre vertex (the reference's `i`, DOF block 1).
    pub i: u32,
    /// Second neighbour (the reference's `k`, DOF block 2).
    pub k: u32,
    /// Rest angle `θ₀`; π means straight. The reference defaults every vertex to π and
    /// only overrides it when an adjacent segment sets `rod-bend-rest-from-geometry`
    /// (`builder.rs:385-408`), which no gauge scene does.
    pub rest_angle: f32,
    /// `stiff_k = mean(bend of the two segments) * vertex_mass` (`energy.cu:508-511`).
    pub stiff: f32,
}

/// Below this the strand is treated as straight (or back-folded) and contributes
/// nothing. `|e₀×e₁|² = |e₀|²|e₁|² sin²θ`, so the test is scale-invariant in sin²θ.
/// Matches the reference's `kSinSqEps` (`dihedral_angle.hpp:414`). Without it an exactly
/// collinear strand normalises a zero cross product to NaN, which propagates into the
/// Hessian and breaks PCG monotonicity.
const SIN_SQ_EPS: f32 = 1.0e-12;

/// `∂θ/∂x₀` and `∂θ/∂x₂` (the reference's `gradient_theta`, `dihedral_angle.hpp:396`).
/// Returns zeros at collinearity, where the bending force vanishes anyway.
fn gradient_theta(e0: Vec3, e1: Vec3) -> [Vec3; 2] {
    let cross = e0.cross(e1);
    let cross_sq = cross.length_squared();
    let e0_sq = e0.dot(e0);
    let e1_sq = e1.dot(e1);
    if cross_sq <= SIN_SQ_EPS * e0_sq * e1_sq {
        return [Vec3::ZERO, Vec3::ZERO];
    }
    let n = cross / cross_sq.sqrt();
    [e0.cross(n) / e0_sq, -(e1.cross(n)) / e1_sq]
}

/// The angle at `x1` between `x0` and `x2`, in `[0, π]`.
pub fn strand_angle(x0: Vec3, x1: Vec3, x2: Vec3) -> f32 {
    let (e0, e1) = (x0 - x1, x2 - x1);
    e0.cross(e1).length().atan2(e0.dot(e1))
}

/// Energy `½(θ − θ₀)²` at UNIT stiffness — multiply by `stiff`.
pub fn strand_energy(x0: Vec3, x1: Vec3, x2: Vec3, rest_angle: f32) -> f32 {
    let d = strand_angle(x0, x1, x2) - rest_angle;
    0.5 * d * d
}

/// Force and PSD-projected Hessian of `½(θ − θ₀)²` at UNIT stiffness.
///
/// Returns the per-vertex gradients for `(x0, x1, x2)` and the 9×9 Hessian in the DOF
/// order `[x0; x1; x2]`. Both scale linearly with the bending stiffness, so the caller
/// multiplies by `stiff_k` (`energy.cu:520-521`).
pub fn strand_force_hessian(
    x0: Vec3,
    x1: Vec3,
    x2: Vec3,
    rest_angle: f32,
) -> ([Vec3; 3], [[f32; 9]; 9]) {
    let (e0, e1) = (x0 - x1, x2 - x1);
    let theta = e0.cross(e1).length().atan2(e0.dot(e1));
    let gt = gradient_theta(e0, e1);
    // The centre vertex takes the negative sum: θ depends only on the two edge vectors.
    let w = theta - rest_angle;
    let grad = [w * gt[0], w * (-gt[0] - gt[1]), w * gt[1]];
    let mut hess = [[0.0f32; 9]; 9];

    let (l0, l1) = (e0.length(), e1.length());
    let bcr = e0.cross(e1);
    let lb = bcr.length();
    // Straight / near-collinear: grad(θ) is already zero, so the force vanishes and the
    // bending Hessian is zero (`dihedral_angle.hpp:468`).
    if lb <= 1.0e-9 * l0 * l1 || l0 <= 1.0e-9 || l1 <= 1.0e-9 {
        return (grad, hess);
    }
    let tb = bcr / lb;
    let e0p = e0.cross(tb); // in-plane, orthogonal to e0, |e0p| = l0
    let e1p = e1.cross(tb);
    let cos_t = e0.dot(e1) / (l0 * l1);
    let sin_t = lb / (l0 * l1);

    let gamma = (l1 * l1) / (l0 * l0);
    let gm1 = gamma - 1.0;
    let gp1 = gamma + 1.0;
    let r = (4.0 * w * w * (gm1 / gp1) * (gm1 / gp1) + 1.0).sqrt();
    let rm = (2.0 * (2.0 * w * w + 1.0 - r)).max(0.0).sqrt();
    let rp = (2.0 * (2.0 * w * w + 1.0 + r)).max(0.0).sqrt();
    let lh = 1.0 / (l0 * l0) + 1.0 / (l1 * l1);
    let cc = 4.0 * w / gp1;

    // Accumulate `lam * (Jᵀq̂)(Jᵀq̂)ᵀ` for a KEPT eigenpair; dividing by `n2` folds in the
    // normalisation of `[qa; qb]`. Dropping `lam <= 0` is the PSD projection.
    let mut accum = |lam: f32, qa: Vec3, qb: Vec3| {
        if lam <= 0.0 {
            return;
        }
        let n2 = qa.dot(qa) + qb.dot(qb);
        if n2 <= 1.0e-20 {
            return;
        }
        let mut p9 = [0.0f32; 9];
        for c in 0..3 {
            p9[c] = qa[c];
            p9[3 + c] = -qa[c] - qb[c];
            p9[6 + c] = qb[c];
        }
        let s = lam / n2;
        for a in 0..9 {
            for b in 0..9 {
                hess[a][b] += s * p9[a] * p9[b];
            }
        }
    };

    // Four in-plane eigenpairs (Wu & Kim Eqns 16-24). The reference notes the b
    // component's R-term sign is FLIPPED relative to the paper (a typo in Eqns 16-19,
    // verified numerically there); reproduced as the reference has it.
    const SR: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
    const S_BIG_R: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
    for m in 0..4 {
        let big_r = if m < 2 { rm } else { rp };
        let (sr, sbr) = (SR[m], S_BIG_R[m]);
        let a = w * (gm1 * gm1 / gp1 + sr * gp1 * r + sbr * gm1 * big_r);
        let b =
            -0.5 * gamma * big_r * big_r + 2.0 * w * w - sbr * 0.5 * (gm1 + sr * gp1 * r) * big_r;
        let d = 1.0 + sr * r + sbr * big_r;
        let lam = 0.25 * lh * d;
        accum(lam, a * e0 + b * e0p, cc * e1 + d * e1p);
    }

    // Two out-of-plane eigenpairs (Wu & Kim Eqns 30-32).
    let beta = (l1 / l0 - l0 / l1) * cos_t;
    let alpha0 = 0.5 * (-beta + (beta * beta + 4.0).sqrt());
    let ginv = w / sin_t;
    let lam4 = ginv * (cos_t / (l1 * l1) - alpha0 / (l0 * l1));
    let lam5 = ginv * (cos_t / (l1 * l1) + 1.0 / (alpha0 * l0 * l1));
    accum(lam4, alpha0 * tb, tb);
    accum(lam5, tb, -alpha0 * tb);

    (grad, hess)
}

/// Interior rod vertices: exactly 2 incident rod segments and NO incident triangle
/// (`energy.cu:496`). `tris` is the scene's full face list — a vertex that belongs to any
/// face is driven by cloth/shell bending instead, never by a strand bend.
///
/// Returns `(j, i, k)` triples with `i` the centre, ordered by centre vertex.
pub fn build_rod_hinges(n_vert: usize, rods: &[[u32; 2]], tris: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let mut has_face = vec![false; n_vert];
    for t in tris {
        for &v in t {
            if (v as usize) < n_vert {
                has_face[v as usize] = true;
            }
        }
    }
    // Two slots is all a valid interior vertex can use; a third incident segment
    // disqualifies it, so count separately and read the slots only when the count is 2.
    let mut nbr = vec![[u32::MAX; 2]; n_vert];
    let mut deg = vec![0u32; n_vert];
    for r in rods {
        let (a, b) = (r[0] as usize, r[1] as usize);
        if a >= n_vert || b >= n_vert {
            continue;
        }
        for (v, other) in [(a, r[1]), (b, r[0])] {
            if deg[v] < 2 {
                nbr[v][deg[v] as usize] = other;
            }
            deg[v] += 1;
        }
    }
    (0..n_vert)
        .filter(|&v| deg[v] == 2 && !has_face[v])
        .map(|v| [nbr[v][0], v as u32, nbr[v][1]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The angle energy in f64. Central-differencing `strand_energy` in f32 is NOT
    /// accurate enough to validate anything: the energy is `O(w²)` with `w = θ − θ₀`
    /// small, so the difference of two nearly equal f32 values loses most of its digits
    /// (measured: 2% error, which would have been indistinguishable from a real bug).
    /// This f64 reference agrees with the analytic gradient to 8 digits.
    fn energy64(x: [[f64; 3]; 3], rest: f64) -> f64 {
        let e0 = [x[0][0] - x[1][0], x[0][1] - x[1][1], x[0][2] - x[1][2]];
        let e1 = [x[2][0] - x[1][0], x[2][1] - x[1][1], x[2][2] - x[1][2]];
        let cr = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ];
        let cn = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
        let dot = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
        let d = cn.atan2(dot) - rest;
        0.5 * d * d
    }

    fn fd_gradient(x: [Vec3; 3], rest: f32) -> [Vec3; 3] {
        let x64 = |x: [Vec3; 3]| {
            let f = |v: Vec3| [v.x as f64, v.y as f64, v.z as f64];
            [f(x[0]), f(x[1]), f(x[2])]
        };
        let base = x64(x);
        let h = 1.0e-6;
        let mut out = [Vec3::ZERO; 3];
        for v in 0..3 {
            for c in 0..3 {
                let mut xp = base;
                let mut xm = base;
                xp[v][c] += h;
                xm[v][c] -= h;
                out[v][c] =
                    ((energy64(xp, rest as f64) - energy64(xm, rest as f64)) / (2.0 * h)) as f32;
            }
        }
        out
    }

    /// Central-difference Hessian of the energy, via the ANALYTIC gradient (so this is a
    /// check of the Hessian only, and the gradient is checked separately against the
    /// energy).
    fn fd_hessian(x: [Vec3; 3], rest: f32) -> [[f32; 9]; 9] {
        let h = 1.0e-3;
        let mut out = [[0.0f32; 9]; 9];
        for col in 0..9 {
            let mut xp = x;
            let mut xm = x;
            xp[col / 3][col % 3] += h;
            xm[col / 3][col % 3] -= h;
            let (gp, _) = strand_force_hessian(xp[0], xp[1], xp[2], rest);
            let (gm, _) = strand_force_hessian(xm[0], xm[1], xm[2], rest);
            for row in 0..9 {
                out[row][col] = (gp[row / 3][row % 3] - gm[row / 3][row % 3]) / (2.0 * h);
            }
        }
        out
    }

    /// Symmetric 9×9 min eigenvalue by unshifted power iteration on `cI − H`.
    fn min_eigenvalue(h: &[[f32; 9]; 9]) -> f32 {
        let mut shift = 0.0f32;
        for row in h.iter() {
            // Gershgorin bound on |eigenvalue|.
            shift = shift.max(row.iter().map(|v| v.abs()).sum());
        }
        let mut v = [1.0f32; 9];
        let mut lam = 0.0;
        for _ in 0..2000 {
            let mut w = [0.0f32; 9];
            for r in 0..9 {
                w[r] = shift * v[r] - (0..9).map(|c| h[r][c] * v[c]).sum::<f32>();
            }
            let n = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n <= 1.0e-30 {
                return shift;
            }
            for r in 0..9 {
                v[r] = w[r] / n;
            }
            lam = n;
        }
        shift - lam
    }

    /// A few bent configurations, deliberately including asymmetric segment lengths
    /// (`gamma != 1`, which is what exercises the `gm1` terms) and a strongly bent case.
    fn cases() -> Vec<([Vec3; 3], f32)> {
        vec![
            // Gentle bend, equal lengths.
            (
                [
                    Vec3::new(-1.0, 0.1, 0.0),
                    Vec3::ZERO,
                    Vec3::new(1.0, 0.1, 0.0),
                ],
                std::f32::consts::PI,
            ),
            // Asymmetric lengths (gamma = 4) and out of plane.
            (
                [
                    Vec3::new(-1.0, 0.3, 0.2),
                    Vec3::ZERO,
                    Vec3::new(2.0, 0.4, -0.3),
                ],
                std::f32::consts::PI,
            ),
            // Sharply bent — nearly folded back.
            (
                [
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::ZERO,
                    Vec3::new(-0.8, 0.5, 0.1),
                ],
                std::f32::consts::PI,
            ),
            // Non-default rest angle, so `w` has the opposite sign.
            (
                [
                    Vec3::new(-1.0, 0.05, 0.0),
                    Vec3::ZERO,
                    Vec3::new(1.0, 0.05, 0.0),
                ],
                2.0,
            ),
        ]
    }

    #[test]
    fn strand_gradient_matches_finite_differences() {
        for (i, (x, rest)) in cases().into_iter().enumerate() {
            let (g, _) = strand_force_hessian(x[0], x[1], x[2], rest);
            let fd = fd_gradient(x, rest);
            for v in 0..3 {
                for c in 0..3 {
                    assert!(
                        (g[v][c] - fd[v][c]).abs() <= 2.0e-3 * fd[v][c].abs().max(1.0e-2),
                        "case {i} vertex {v} comp {c}: analytic {} vs finite difference {}",
                        g[v][c],
                        fd[v][c]
                    );
                }
            }
            // The three per-vertex gradients must sum to zero: θ is translation invariant,
            // so bending exerts no net force. This is independent of the FD check.
            let sum = g[0] + g[1] + g[2];
            assert!(
                sum.length() <= 1.0e-5 * g[0].length().max(1.0e-6),
                "case {i}: bending must exert no net force, got {sum:?}"
            );
        }
    }

    #[test]
    fn strand_hessian_is_always_psd() {
        for (i, (x, rest)) in cases().into_iter().enumerate() {
            let (_, h) = strand_force_hessian(x[0], x[1], x[2], rest);
            // Symmetry first — the projection is a sum of outer products, so any
            // asymmetry would mean an indexing bug.
            for (r, row) in h.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    assert!(
                        (v - h[c][r]).abs() <= 1.0e-6 * h[r][r].abs().max(1.0),
                        "case {i}: Hessian not symmetric at [{r}][{c}]"
                    );
                }
            }
            let lo = min_eigenvalue(&h);
            let scale: f32 = (0..9).map(|r| h[r][r]).sum::<f32>().max(1.0e-6);
            assert!(
                lo >= -1.0e-4 * scale,
                "case {i}: projected Hessian must be PSD, min eigenvalue {lo} (trace {scale})"
            );
        }
    }

    /// The eigen-projection clamps negative eigenvalues to zero, which can only RAISE the
    /// quadratic form. So `H_projected − H_true` must itself be PSD, and where nothing was
    /// clamped the two must be EQUAL.
    ///
    /// This is the test with real teeth. A PSD-only check would pass happily on a Hessian
    /// that is positive but wrong; requiring `H_proj ⪰ H_fd` pins the eigenvalues and
    /// eigenvectors to the actual second derivative of the energy. (An earlier version of
    /// this test gated on the FD Hessian being positive DEFINITE, which never fires — the
    /// 9-DOF Hessian has three translational null directions by construction.)
    #[test]
    fn projection_dominates_the_true_hessian_and_equals_it_when_nothing_is_clamped() {
        let mut equal_cases = 0usize;
        for (i, (x, rest)) in cases().into_iter().enumerate() {
            let (_, h) = strand_force_hessian(x[0], x[1], x[2], rest);
            let fd = fd_hessian(x, rest);
            let scale: f32 = (0..9).map(|r| fd[r][r].abs()).sum::<f32>().max(1.0e-6);

            // H_proj - H_fd must be PSD (up to finite-difference noise).
            let mut diff = [[0.0f32; 9]; 9];
            for r in 0..9 {
                for c in 0..9 {
                    // Symmetrise the FD Hessian: one-sided noise would otherwise show up
                    // as a spurious negative eigenvalue.
                    let sym = 0.5 * (fd[r][c] + fd[c][r]);
                    diff[r][c] = h[r][c] - sym;
                }
            }
            let lo = min_eigenvalue(&diff);
            assert!(
                lo >= -2.0e-2 * scale,
                "case {i}: projection must dominate the true Hessian, but \
                 min eig(H_proj - H_fd) = {lo} (scale {scale})"
            );

            // How much was clamped away, as a fraction of the Hessian's size.
            let dev: f32 = (0..9)
                .flat_map(|r| (0..9).map(move |c| (r, c)))
                .map(|(r, c)| (h[r][c] - 0.5 * (fd[r][c] + fd[c][r])).abs())
                .fold(0.0f32, f32::max);
            if dev <= 2.0e-2 * scale {
                equal_cases += 1;
            }
        }
        // At least one configuration must have needed NO clamping, so the equality branch
        // is actually exercised rather than every case hiding behind the inequality.
        assert!(
            equal_cases > 0,
            "no case reproduced the finite-difference Hessian exactly; the projection \
             would then never have been checked against the true second derivative"
        );
    }

    /// The eigenvalue helper the two tests above rely on must actually detect a negative
    /// eigenvalue — otherwise both would be false passes.
    #[test]
    fn min_eigenvalue_helper_is_trustworthy() {
        let mut m = [[0.0f32; 9]; 9];
        for (i, d) in [3.0, 1.0, 7.0, 2.0, 5.0, 4.0, 6.0, 8.0, 9.0]
            .iter()
            .enumerate()
        {
            m[i][i] = *d;
        }
        assert!(
            (min_eigenvalue(&m) - 1.0).abs() < 1.0e-2,
            "diagonal min should be 1.0, got {}",
            min_eigenvalue(&m)
        );
        m[4][4] = -2.5;
        assert!(
            (min_eigenvalue(&m) + 2.5).abs() < 1.0e-2,
            "must detect the negative eigenvalue -2.5, got {}",
            min_eigenvalue(&m)
        );
    }

    #[test]
    fn a_straight_strand_contributes_nothing() {
        let x = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let (g, h) = strand_force_hessian(x[0], x[1], x[2], std::f32::consts::PI);
        for gv in &g {
            assert!(
                gv.length() < 1.0e-6,
                "straight strand at rest must exert no force, got {gv:?}"
            );
        }
        assert!(
            h.iter().flatten().all(|v| v.abs() < 1.0e-6),
            "straight strand must contribute a zero Hessian"
        );
        // And it must be finite, not NaN — the collinearity guard is the whole point.
        assert!(h.iter().flatten().all(|v| v.is_finite()));
        assert!(g.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn interior_rod_vertices_are_gated_on_two_segments_and_no_face() {
        // A 4-vertex open chain 0-1-2-3: interiors are 1 and 2.
        let rods = [[0u32, 1], [1, 2], [2, 3]];
        let h = build_rod_hinges(4, &rods, &[]);
        assert_eq!(
            h.len(),
            2,
            "open chain of 3 segments has 2 interior vertices"
        );
        assert_eq!(h[0][1], 1);
        assert_eq!(h[1][1], 2);
        // Each hinge's neighbours must be the two vertices actually joined to it.
        assert_eq!([h[0][0], h[0][2]].to_vec(), vec![0, 2]);

        // A vertex belonging to a face is driven by cloth bending, never a strand bend.
        let h = build_rod_hinges(4, &rods, &[[1, 2, 3]]);
        assert!(
            h.is_empty(),
            "vertices on a face must be excluded, got {h:?}"
        );

        // A T-junction (degree 3) is not an interior rod vertex.
        let rods = [[0u32, 1], [1, 2], [1, 3]];
        assert!(
            build_rod_hinges(4, &rods, &[]).is_empty(),
            "a degree-3 junction must be excluded"
        );

        // A closed loop 0-1-2-0: every vertex is interior.
        let rods = [[0u32, 1], [1, 2], [2, 0]];
        assert_eq!(build_rod_hinges(3, &rods, &[]).len(), 3);
    }
}
