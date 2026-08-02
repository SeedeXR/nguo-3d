// SPDX-License-Identifier: GPL-2.0-or-later
//! Seam stitches — a weighted barycentric point on one piece of cloth sewn to a weighted
//! barycentric point on another.
//!
//! Ported from the reference's `embed_stitch_force_hessian`
//! (`ppf-cts-solver/src/cpp/energy/energy.cu:696`).
//!
//! Structurally this is the `hook` stretch energy (see [`crate::implicit::Rod`]) generalised
//! from 2 participants to 6 weighted ones — the same `c₀`/`c₁` Hessian split — acting on
//!
//! ```text
//! t = Σ_{i<3} wᵢ·xᵢ  −  Σ_{j≥3} wⱼ·xⱼ
//! ```
//!
//! The reference subtracts a centre-of-gravity from every term; that cancels exactly because
//! each side's weights sum to 1, so it is a pure fp32 conditioning trick and is omitted here
//! (`no_cog_term_changes_the_result` pins that down).
//!
//! While `‖t‖ ≤ l_cap` the potential is `E = (‖t‖ − l0)²/(2·l0)` at unit stiffness. Past the
//! cap the reference clamps `l` but does NOT renormalise `n = t/l`, so the force grows
//! linearly in `‖t‖` instead of with the `(l/l0 − 1)` ratio; that regime is reproduced as-is
//! and is not a gradient of the same scalar.

use glam::Vec3;

/// Signed weights: `+w` on the source side, `−w` on the target. This is the reference's
/// `dtdx` (`energy.cu:745-752`), whose blocks are `±wₖ·I`.
pub fn signed_weights(w: &[f32; 6]) -> [f32; 6] {
    std::array::from_fn(|k| if k < 3 { w[k] } else { -w[k] })
}

/// The seam vector `t` for six positions and their barycentric weights.
pub fn seam_vector(x: &[Vec3; 6], w: &[f32; 6]) -> Vec3 {
    let sw = signed_weights(w);
    let mut t = Vec3::ZERO;
    for k in 0..6 {
        t += sw[k] * x[k];
    }
    t
}

/// Potential at UNIT stiffness, valid while `‖t‖ ≤ l_cap`.
pub fn energy(x: &[Vec3; 6], w: &[f32; 6], l0: f32) -> f32 {
    let e = seam_vector(x, w).length() - l0;
    0.5 * e * e / l0
}

/// Force and Hessian of one stitch at UNIT stiffness, in the DOF order
/// `[x0 … x5]` (18 DOFs). The caller scales both by the stitch's stiffness, which the
/// reference applies RAW — no mass or time-scale normalisation (`energy.cu:765-772`).
///
/// Returns per-participant gradients and the 18×18 Hessian
/// `c₀·g·gᵀ + c₁·dtdxᵀ·dtdx`, where `c₁ = max(0, r/l0)` is the PSD projection (it is the
/// term that goes negative once the seam is shorter than its rest length). Exactly the
/// reference's `hook` split, so when the seam is STRETCHED — the normal case for a seam
/// being pulled closed — this is the true Hessian.
pub fn force_hessian(
    x: &[Vec3; 6],
    w: &[f32; 6],
    l0: f32,
    l_cap: f32,
) -> ([Vec3; 6], [[f32; 18]; 18]) {
    let sw = signed_weights(w);
    let t = seam_vector(x, w);
    let tn = t.length();
    let mut hess = [[0.0f32; 18]; 18];
    if tn < 1.0e-12 || l0 <= 0.0 {
        // Coincident: no direction, and the energy is at its minimum.
        return ([Vec3::ZERO; 6], hess);
    }
    let l = tn.min(l_cap);
    let n = t / l;
    let dedt = (l / l0 - 1.0) * n;
    let grad: [Vec3; 6] = std::array::from_fn(|k| sw[k] * dedt);

    let r = (l - l0) / l;
    let c0 = (1.0 - r).max(0.0) / l0;
    let c1 = (r / l0).max(0.0);
    // g = dtdxᵀ·n, an 18-vector of 6 blocks `sw[k]·n`.
    let gv: [Vec3; 6] = std::array::from_fn(|k| sw[k] * n);
    for a in 0..6 {
        for b in 0..6 {
            // dtdxᵀ·dtdx has block (a,b) = sw[a]·sw[b]·I.
            let iden = c1 * sw[a] * sw[b];
            for rr in 0..3 {
                for cc in 0..3 {
                    let mut v = c0 * gv[a][rr] * gv[b][cc];
                    if rr == cc {
                        v += iden;
                    }
                    hess[3 * a + rr][3 * b + cc] = v;
                }
            }
        }
    }
    (grad, hess)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The energy in f64 — an f32 central difference of a squared-residual potential loses
    /// most of its digits (established while porting rod bending).
    fn energy64(x: &[[f64; 3]; 6], w: &[f32; 6], l0: f32) -> f64 {
        let mut t = [0.0f64; 3];
        for k in 0..6 {
            let s = if k < 3 { w[k] as f64 } else { -(w[k] as f64) };
            for c in 0..3 {
                t[c] += s * x[k][c];
            }
        }
        let tn = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        let e = tn - l0 as f64;
        0.5 * e * e / l0 as f64
    }

    fn to64(x: &[Vec3; 6]) -> [[f64; 3]; 6] {
        std::array::from_fn(|k| [x[k].x as f64, x[k].y as f64, x[k].z as f64])
    }

    /// A genuine barycentric-to-barycentric stitch (no degenerate side), stretched and
    /// inside its cap so the potential above is exact.
    fn stretched() -> ([Vec3; 6], [f32; 6], f32, f32) {
        let x = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.01, 0.0, 0.0),
            Vec3::new(0.0, 0.01, 0.0),
            Vec3::new(0.002, 0.001, 0.004),
            Vec3::new(0.012, 0.001, 0.004),
            Vec3::new(0.002, 0.011, 0.004),
        ];
        let w = [0.5, 0.3, 0.2, 0.25, 0.25, 0.5];
        (x, w, 0.001, 0.01)
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (x, w, l0, l_cap) = stretched();
        let tn = seam_vector(&x, &w).length();
        assert!(
            tn > l0 && tn < l_cap,
            "case must be stretched, inside the cap"
        );
        let (g, _) = force_hessian(&x, &w, l0, l_cap);
        let base = to64(&x);
        let h = 1.0e-9;
        for k in 0..6 {
            for c in 0..3 {
                let mut xp = base;
                let mut xm = base;
                xp[k][c] += h;
                xm[k][c] -= h;
                let fd = ((energy64(&xp, &w, l0) - energy64(&xm, &w, l0)) / (2.0 * h)) as f32;
                assert!(
                    (g[k][c] - fd).abs() <= 1.0e-3 * fd.abs().max(1.0e-3),
                    "participant {k} comp {c}: analytic {} vs finite difference {fd}",
                    g[k][c]
                );
            }
        }
        // A seam exerts no net force: `t` depends only on differences.
        let sum: Vec3 = (0..6).map(|k| g[k]).sum();
        assert!(
            sum.length() < 1.0e-4 * g[0].length().max(1.0e-6),
            "net force must vanish, got {sum:?}"
        );
    }

    /// The Hessian, against a central difference of the ANALYTIC gradient. A stretched seam
    /// clamps nothing (`c₁ = r/l0 > 0`), so this must be the TRUE second derivative — this
    /// is the test that catches a mis-scaled or mis-signed block, which the solver-level
    /// "does the seam close" check provably cannot: the Hessian only sets the Newton search
    /// DIRECTION, so dropping a whole term still converges to the same rest length.
    #[test]
    fn hessian_matches_finite_differences_when_stretched() {
        let (x, w, l0, l_cap) = stretched();
        let (_, hh) = force_hessian(&x, &w, l0, l_cap);
        let h = 1.0e-6;
        let mut worst = 0.0f32;
        let scale: f32 = (0..18).map(|i| hh[i][i].abs()).sum::<f32>().max(1.0e-6);
        for col in 0..18 {
            let mut xp = x;
            let mut xm = x;
            xp[col / 3][col % 3] += h;
            xm[col / 3][col % 3] -= h;
            let (gp, _) = force_hessian(&xp, &w, l0, l_cap);
            let (gm, _) = force_hessian(&xm, &w, l0, l_cap);
            for row in 0..18 {
                let fd = (gp[row / 3][row % 3] - gm[row / 3][row % 3]) / (2.0 * h);
                worst = worst.max((hh[row][col] - fd).abs());
            }
        }
        assert!(
            worst <= 2.0e-2 * scale,
            "stitch Hessian differs from the finite-difference second derivative by \
             {worst} (scale {scale})"
        );
    }

    #[test]
    fn hessian_is_symmetric_and_psd() {
        let (x, w, l0, l_cap) = stretched();
        // Include a COMPRESSED case, where `c₁` clamps and the projection kicks in.
        let mut short = x;
        short[3] = Vec3::new(0.0035, 0.0035, 0.0002);
        short[4] = Vec3::new(0.0135, 0.0035, 0.0002);
        short[5] = Vec3::new(0.0035, 0.0135, 0.0002);
        for (case, xx) in [("stretched", x), ("compressed", short)] {
            let (_, hh) = force_hessian(&xx, &w, l0, l_cap);
            for (a, row) in hh.iter().enumerate() {
                for (b, &v) in row.iter().enumerate() {
                    assert!(
                        (v - hh[b][a]).abs() <= 1.0e-5 * hh[a][a].abs().max(1.0),
                        "{case}: not symmetric at [{a}][{b}]"
                    );
                }
            }
            // PSD by probe: a sum of outer products with non-negative coefficients.
            for seed in 0..24u32 {
                let mut v = [0.0f32; 18];
                for (i, s) in v.iter_mut().enumerate() {
                    // Deterministic spread of probe directions; no RNG in tests.
                    let h = (i as u32).wrapping_mul(2654435761) ^ seed.wrapping_mul(40503);
                    *s = ((h % 1000) as f32 / 500.0) - 1.0;
                }
                let mut q = 0.0f32;
                for a in 0..18 {
                    for b in 0..18 {
                        q += v[a] * hh[a][b] * v[b];
                    }
                }
                assert!(q >= -1.0e-3, "{case}: must be PSD, got quadratic form {q}");
            }
        }
    }

    /// The reference subtracts a centre-of-gravity from all six terms before forming `t`.
    /// That must be a no-op, because each side's weights sum to 1 — verified rather than
    /// assumed, since if it were NOT a no-op omitting it would silently change the energy.
    #[test]
    fn no_cog_term_changes_the_result() {
        let (x, w, _, _) = stretched();
        let cog = (x[0] + x[1] + x[2] + x[3] + x[4] + x[5]) / 6.0;
        let sw = signed_weights(&w);
        let mut with_cog = Vec3::ZERO;
        for k in 0..6 {
            with_cog += sw[k] * (x[k] - cog);
        }
        let without = seam_vector(&x, &w);
        assert!(
            (with_cog - without).length() <= 1.0e-6 * without.length(),
            "the cog terms must cancel: {with_cog:?} vs {without:?}"
        );
    }

    /// A degenerate single-vertex source (`index = {s,s,s}`, `weight = {1,0,0}` — what all
    /// 851 of `fitting`'s stitches use) must reduce to a two-participant spring, with the
    /// target's reaction split by its barycentric weights.
    #[test]
    fn a_degenerate_source_reduces_to_a_single_vertex() {
        let p = Vec3::new(0.0, 0.0, 0.0);
        let t1 = Vec3::new(0.003, 0.001, 0.0);
        let t2 = Vec3::new(0.013, 0.001, 0.0);
        let x = [p, p, p, t1, t2, t2];
        let w = [1.0, 0.0, 0.0, 0.75, 0.25, 0.0];
        let (l0, l_cap) = (0.001f32, 0.01f32);
        let (g, _) = force_hessian(&x, &w, l0, l_cap);

        // Hand-computed equivalent.
        let target = 0.75 * t1 + 0.25 * t2;
        let t = p - target;
        let l = t.length().min(l_cap);
        let dedt = (l / l0 - 1.0) * (t / l);
        // The three source slots are the same vertex, so the caller's scatter sums them;
        // slots 1 and 2 carry weight 0 and must contribute nothing.
        assert!((g[0] - dedt).length() <= 1.0e-4 * dedt.length());
        assert!(g[1].length() < 1.0e-9 && g[2].length() < 1.0e-9);
        assert!((g[3] + 0.75 * dedt).length() <= 1.0e-4 * dedt.length());
        assert!((g[4] + 0.25 * dedt).length() <= 1.0e-4 * dedt.length());
        assert!(g[5].length() < 1.0e-9);
    }

    /// Past `l_cap` the reference clamps `l` but leaves `n = t/l` un-normalised, so the pull
    /// keeps growing LINEARLY in `‖t‖` rather than saturating. Reproduced deliberately, so
    /// pin the behaviour down rather than let a future "fix" change it silently.
    #[test]
    fn beyond_the_cap_the_pull_grows_linearly() {
        let w = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (l0, l_cap) = (0.001f32, 0.01f32);
        let mag = |sep: f32| {
            let x = [
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::new(sep, 0.0, 0.0),
                Vec3::new(sep, 0.0, 0.0),
                Vec3::new(sep, 0.0, 0.0),
            ];
            force_hessian(&x, &w, l0, l_cap).0[0].length()
        };
        let (a, b) = (mag(0.04), mag(0.08));
        assert!(
            (b / a - 2.0).abs() < 1.0e-3,
            "doubling the separation past the cap must double the pull: {a} -> {b}"
        );
    }
}
