// SPDX-License-Identifier: GPL-2.0-or-later
//! One-sided **strain-limiting** spring — caps an edge's stretch to
//! `max_ratio · rest`, active ONLY when the edge is over-stretched
//! (`l > max_ratio · rest`) and silent otherwise.
//!
//! Strain limiting is what keeps woven fabric from stretching like rubber. A
//! purely elastic membrane model has no inextensibility limit, so a garment
//! under load creeps well past what a real woven textile can do; limiting the
//! per-edge (and, in the implicit tier, per-face) strain is what makes the
//! drape read as fabric rather than as a stretchy sheet. This is
//! the analytic primitive (energy + SPD gradient/Hessian) for the implicit tier,
//! reusing the same SPD-projected [`spring_block`](crate::implicit::spring_block)
//! the membrane springs use, so it drops straight into the Newton assembly.
//!
//! Opt-in: this is NOT wired into the default solver (springs stay the default
//! membrane model). Wiring it into the unified assembler as an optional pass is the
//! follow-up; landing the verified primitive first keeps the physics defaults
//! unchanged.

use crate::implicit::spring_block;
use glam::Vec3;

/// Effective one-sided rest length: the edge is unconstrained until it reaches
/// `max_ratio · rest`, then behaves like a spring with this rest length.
#[inline]
fn limit_length(rest: f32, max_ratio: f32) -> f32 {
    rest * max_ratio
}

/// Strain-limit energy on edge `(xi, xj)`: `½ k (l − r)²` for `l > r`
/// (`r = max_ratio · rest`), else `0`. `k` is the strain-limit stiffness.
pub fn strain_energy(xi: Vec3, xj: Vec3, rest: f32, max_ratio: f32, k: f32) -> f32 {
    let l = (xi - xj).length();
    let r = limit_length(rest, max_ratio);
    if l > r {
        0.5 * k * (l - r) * (l - r)
    } else {
        0.0
    }
}

/// Per-edge strain-limit force + Hessian block. Returns `(grad_i, h_ii)` where
/// `grad_i = ∂E/∂xi` and `h_ii = ∂²E/∂xi²` (the SPD-projected 3×3 block); by
/// symmetry `grad_j = −grad_i`, `h_jj = h_ii`, and `h_ij = h_ji = −h_ii`. Zero
/// when the edge is not over-stretched (`l ≤ max_ratio · rest`).
///
/// The active branch is a spring with rest length `r = max_ratio · rest`; since
/// `l > r` there the SPD clamp inside `spring_block` never triggers, so the block
/// is the *exact* energy Hessian (verified against finite differences).
pub fn strain_block(
    xi: Vec3,
    xj: Vec3,
    rest: f32,
    max_ratio: f32,
    k: f32,
) -> (Vec3, [[f32; 3]; 3]) {
    let d = xi - xj;
    let r = limit_length(rest, max_ratio);
    if d.length() <= r {
        return (Vec3::ZERO, [[0.0; 3]; 3]);
    }
    let (h, g) = spring_block(d, r, k);
    (g, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn central_grad(xi: Vec3, xj: Vec3, rest: f32, ratio: f32, k: f32) -> Vec3 {
        let eps = 1.0e-3;
        let mut g = Vec3::ZERO;
        for c in 0..3 {
            let mut p = xi;
            p[c] += eps;
            let ep = strain_energy(p, xj, rest, ratio, k);
            p[c] -= 2.0 * eps;
            let em = strain_energy(p, xj, rest, ratio, k);
            g[c] = (ep - em) / (2.0 * eps);
        }
        g
    }

    fn central_hess(xi: Vec3, xj: Vec3, rest: f32, ratio: f32, k: f32) -> [[f32; 3]; 3] {
        // FD of grad_i wrt xi.
        let eps = 1.0e-3;
        let mut h = [[0.0f32; 3]; 3];
        for c in 0..3 {
            let mut p = xi;
            p[c] += eps;
            let (gp, _) = strain_block(p, xj, rest, ratio, k);
            p[c] -= 2.0 * eps;
            let (gm, _) = strain_block(p, xj, rest, ratio, k);
            let col = (gp - gm) / (2.0 * eps);
            for r in 0..3 {
                h[r][c] = col[r];
            }
        }
        h
    }

    #[test]
    fn inactive_below_limit_is_silent() {
        // l = 1.0 < r = 1.1·1.0 → no energy, no force, no stiffness.
        let xi = Vec3::new(1.0, 0.0, 0.0);
        let xj = Vec3::ZERO;
        assert_eq!(strain_energy(xi, xj, 1.0, 1.1, 100.0), 0.0);
        let (g, h) = strain_block(xi, xj, 1.0, 1.1, 100.0);
        assert_eq!(g, Vec3::ZERO);
        assert_eq!(h, [[0.0; 3]; 3]);
    }

    #[test]
    fn active_gradient_matches_energy_fd() {
        // l = 1.3 > r = 1.1 → active. Off-axis so all 3 components are exercised.
        let xi = Vec3::new(1.0, 0.6, 0.3);
        let xj = Vec3::new(-0.1, 0.05, -0.2);
        let (rest, ratio, k) = (1.0, 1.1, 120.0);
        let (g, _) = strain_block(xi, xj, rest, ratio, k);
        let gfd = central_grad(xi, xj, rest, ratio, k);
        assert!(
            (g - gfd).length() < 1.0e-2 * g.length().max(1.0),
            "analytic grad {g:?} vs FD {gfd:?}"
        );
    }

    #[test]
    fn active_hessian_matches_grad_fd() {
        let xi = Vec3::new(1.0, 0.6, 0.3);
        let xj = Vec3::new(-0.1, 0.05, -0.2);
        let (rest, ratio, k) = (1.0, 1.1, 120.0);
        let (_, h) = strain_block(xi, xj, rest, ratio, k);
        let hfd = central_hess(xi, xj, rest, ratio, k);
        let mut max_err = 0.0f32;
        for r in 0..3 {
            for c in 0..3 {
                max_err = max_err.max((h[r][c] - hfd[r][c]).abs());
            }
        }
        // Stiffness entries are O(k)=O(120); a few e-1 absolute is well within FD noise.
        assert!(max_err < 1.0, "analytic Hessian vs FD max err {max_err}");
    }

    #[test]
    fn opposite_signs_and_symmetry() {
        // grad_j = -grad_i by construction; check the caller contract holds.
        let xi = Vec3::new(2.0, 0.0, 0.0);
        let xj = Vec3::ZERO;
        let (gi, hii) = strain_block(xi, xj, 1.0, 1.1, 50.0);
        let (gj, hjj) = strain_block(xj, xi, 1.0, 1.1, 50.0);
        assert!((gi + gj).length() < 1.0e-4, "grad_j = -grad_i");
        assert_eq!(hii, hjj, "h_jj = h_ii");
        // Over-stretched along +x → force on i points +x (restoring pull toward j).
        assert!(gi.x > 0.0);
    }
}
