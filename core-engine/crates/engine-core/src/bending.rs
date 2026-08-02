// SPDX-License-Identifier: GPL-2.0-or-later
//! Discrete-shell dihedral bending (Grinspun et al. 2003 "Discrete Shells").
//!
//! Each interior edge shared by two triangles is a hinge with a dihedral angle
//! θ; bending energy penalizes deviation from the rest angle:
//! `E = ½·kb·w·(θ − θ_rest)²`. Springs give zero bending resistance, so this is
//! what makes cloth drape and wrinkle rather than collapse.
//!
//! Derivative policy (zero-hallucination): the angle θ(x) is exact (triangle
//! normals + `atan2`); the gradient `∂θ/∂x` is taken by **central finite
//! differences** of θ, and the Hessian is the **Gauss-Newton** approximation
//! `kb·w·∇θ∇θᵀ`, which is PSD by construction (the reference likewise SPD-projects
//! bending Hessians via eigenanalysis — docs/KERNELS.md). Correctness is checked
//! by physical behavior in tests, not a transcribed formula. Analytic gradients
//! (Tamstorf & Grinspun 2013) are a later perf upgrade.
//!
//! ponytail: hinge weight `w = 1` (bend stiffness is the tunable knob, per
//! docs/CONFIG.md where `bend` is dimensionless); area/edge weighting for
//! mesh-independence is a refinement.

use glam::Vec3;
use std::collections::HashMap;

/// A bending hinge: `v = [opp_a, shared_0, shared_1, opp_b]`. Triangle A is
/// `(v0,v1,v2)`, triangle B is `(v1,v2,v3)`; the shared edge is `(v1,v2)`.
#[derive(Clone, Copy, Debug)]
pub struct Hinge {
    pub v: [u32; 4],
    pub rest: f32,
    pub w: f32,
}

/// Signed dihedral angle of a hinge flap. 0 when the two triangles are coplanar.
pub fn dihedral_angle(p: &[Vec3; 4]) -> f32 {
    // Triangle A = (v0,v1,v2); triangle B shares edge (v1,v2) but traverses it
    // in the opposite direction (v2→v1) as a coherently-wound mesh does, so both
    // normals point to the same side. A genuinely flat flap then reads θ = 0
    // (NOT π) — keeping rest angles off the atan2 branch cut at ±π.
    let n1 = (p[1] - p[0]).cross(p[2] - p[0]);
    let n2 = (p[1] - p[2]).cross(p[3] - p[2]);
    let e = (p[2] - p[1]).normalize_or_zero();
    let n1n = n1.normalize_or_zero();
    let n2n = n2.normalize_or_zero();
    let sin = n1n.cross(n2n).dot(e);
    let cos = n1n.dot(n2n).clamp(-1.0, 1.0);
    sin.atan2(cos)
}

/// `∂θ/∂x` for the 4 hinge vertices, by central finite differences.
pub fn dihedral_grad(p: &[Vec3; 4]) -> [Vec3; 4] {
    // Step scaled to the flap size so the FD is well-conditioned in f32.
    let scale = (p[0].length() + p[1].length() + p[2].length() + p[3].length()).max(1.0);
    let h = 1.0e-4 * scale;
    let inv = 1.0 / (2.0 * h);
    let mut g = [Vec3::ZERO; 4];
    for (vi, gv) in g.iter_mut().enumerate() {
        for c in 0..3 {
            let mut pp = *p;
            let mut pm = *p;
            pp[vi][c] += h;
            pm[vi][c] -= h;
            gv[c] = (dihedral_angle(&pp) - dihedral_angle(&pm)) * inv;
        }
    }
    g
}

/// Analytic gradient of the dihedral angle w.r.t. the four hinge vertices
/// `[opp_a, shared_0, shared_1, opp_b]`, matching [`dihedral_angle`]'s convention.
/// ~30× cheaper than the finite-difference [`dihedral_grad`] (no 24 angle evals),
/// verified equal to it. Falls back to `dihedral_grad` on a degenerate hinge.
///
/// Discrete-shell form: a tip vertex's gradient is `−(‖e‖/2A)·n̂` (rotating the
/// triangle about the shared edge; `‖e‖/2A = 1/height`). The two shared (edge)
/// vertices split the tip gradients by the tips' projection parameter along the
/// edge, so all four gradients sum to zero (translation invariance).
pub fn dihedral_grad_analytic(p: &[Vec3; 4]) -> [Vec3; 4] {
    let m1 = (p[1] - p[0]).cross(p[2] - p[0]); // tri A normal (unnormalized)
    let m2 = (p[1] - p[2]).cross(p[3] - p[2]); // tri B normal
    let e = p[2] - p[1]; // shared edge
    let (a1, a2, l2) = (m1.length(), m2.length(), e.length_squared());
    if a1 < 1.0e-12 || a2 < 1.0e-12 || l2 < 1.0e-24 {
        return dihedral_grad(p); // degenerate: fall back to FD
    }
    let l = l2.sqrt();
    let n1 = m1 / a1; // unit normals
    let n2 = m2 / a2;
    // Tip gradients: magnitude 1/height = ‖e‖/(2·area) = ‖e‖/a (a = 2·area = ‖m‖).
    let g0 = n1 * (-l / a1);
    let g3 = n2 * (-l / a2);
    // Projection parameters of the tips onto the shared edge (0 at shared_0=p1).
    let ta = (p[0] - p[1]).dot(e) / l2;
    let tb = (p[3] - p[1]).dot(e) / l2;
    // Shared vertices split the tip gradients so g0+g1+g2+g3 = 0.
    let g1 = g0 * (ta - 1.0) + g3 * (tb - 1.0);
    let g2 = g0 * (-ta) + g3 * (-tb);
    [g0, g1, g2, g3]
}

/// Per-hinge element assembly: the bending energy gradient (12 = 4 verts × 3) and
/// the Gauss-Newton Hessian block (144 = 12×12, row-major) at positions `p`.
/// `grad[3a+r] = kb·w·(θ−rest)·∂θ/∂x[a]_r`, `hess[(3a+r)·12+(3b+c)] =
/// kb·w·∂θ/∂x[a]_r·∂θ/∂x[b]_c` (PSD rank-1). The shared oracle for both the CPU
/// [`crate::bending_assembler::BendingAssembler`] and the GPU bending kernel.
pub fn hinge_element(p: &[Vec3; 4], rest: f32, w: f32, kb: f32) -> ([f32; 12], [f32; 144]) {
    let theta = dihedral_angle(p);
    let g = dihedral_grad_analytic(p);
    let kw = kb * w;
    let coef = kw * (theta - rest);
    let mut grad = [0.0f32; 12];
    let mut hess = [0.0f32; 144];
    for a in 0..4 {
        for r in 0..3 {
            grad[3 * a + r] = coef * g[a][r];
            for b in 0..4 {
                for c in 0..3 {
                    hess[(3 * a + r) * 12 + (3 * b + c)] = kw * g[a][r] * g[b][c];
                }
            }
        }
    }
    (grad, hess)
}

/// Build hinges from a triangle list; `rest` angles are read from `positions`.
pub fn build_hinges(tris: &[[u32; 3]], positions: &[Vec3]) -> Vec<Hinge> {
    // edge (sorted) -> list of opposite vertices.
    let mut map: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for t in tris {
        let edges = [(t[1], t[2], t[0]), (t[0], t[2], t[1]), (t[0], t[1], t[2])];
        for (a, b, opp) in edges {
            let key = if a < b { (a, b) } else { (b, a) };
            map.entry(key).or_default().push(opp);
        }
    }
    let mut hinges = Vec::new();
    // Deterministic order (sorted by edge key).
    let mut keys: Vec<_> = map.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let opp = &map[&key];
        if opp.len() == 2 {
            let v = [opp[0], key.0, key.1, opp[1]];
            let p = [
                positions[v[0] as usize],
                positions[v[1] as usize],
                positions[v[2] as usize],
                positions[v[3] as usize],
            ];
            hinges.push(Hinge {
                v,
                rest: dihedral_angle(&p),
                w: 1.0,
            });
        }
    }
    hinges
}

/// **Hinge (bend) plasticity** — a port of the reference
/// `plasticity::update_hinge_plasticity` (`plasticity/plasticity.cu`, Apache-2.0).
///
/// Unlike the stretch-based tet/shell plasticity, this creeps the bending **rest
/// angle**: with `α = 1 − exp(−plasticity·dt)` and `Δ = θ − θ_rest`, when `|Δ|`
/// exceeds `threshold` the rest angle advances by the reference's `_dead_zone_creep`
/// increment `α·(Δ − sign(Δ)·threshold)`. So a fold held past the dead zone gradually
/// becomes the new neutral shape (a crease), while anything inside the dead zone
/// springs back.
///
/// Returns `true` if the rest angle changed.
///
/// Note on vertex order: the reference calls `dihedral_angle::remap` first because its
/// hinge storage differs from its angle routine's expected order. Ours are already
/// consistent — [`Hinge::v`] is `[opp_a, s0, s1, opp_b]`, exactly what
/// [`dihedral_angle`] takes — so no remap is needed here.
pub fn apply_hinge_plasticity(
    hinge: &mut Hinge,
    p: &[Vec3; 4],
    plasticity: f32,
    threshold: f32,
    dt: f32,
) -> bool {
    if plasticity <= 0.0 {
        return false;
    }
    let theta = dihedral_angle(p);
    let delta = theta - hinge.rest;
    if delta.abs() <= threshold {
        return false;
    }
    let alpha = 1.0 - (-plasticity * dt).exp();
    let target = if delta > 0.0 { threshold } else { -threshold };
    hinge.rest += alpha * (delta - target);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn flap(fold_deg: f32) -> [Vec3; 4] {
        // Shared edge along X from (0,0,0) to (1,0,0). Triangle A wing at (-y).
        // Triangle B wing rotated about the X axis by `fold_deg`; with the
        // consistent-normal convention, dihedral_angle(flap(d)) == d (radians).
        let a = fold_deg.to_radians();
        [
            Vec3::new(0.5, -1.0, 0.0),        // opp A
            Vec3::new(0.0, 0.0, 0.0),         // shared 0
            Vec3::new(1.0, 0.0, 0.0),         // shared 1
            Vec3::new(0.5, a.cos(), a.sin()), // opp B, rotated
        ]
    }

    /// A skewed hinge (tips not centered over the edge) so the shared-vertex
    /// projection weights are exercised, not just the symmetric flap.
    fn skewed_flap(fold_deg: f32, sa: f32, sb: f32) -> [Vec3; 4] {
        let a = fold_deg.to_radians();
        [
            Vec3::new(sa, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.3, 0.0, 0.0), // non-unit edge
            Vec3::new(sb, a.cos(), a.sin()),
        ]
    }

    #[test]
    fn analytic_dihedral_grad_matches_fd() {
        // The analytic gradient must equal the FD gradient (verified against energy)
        // across a range of fold angles and skews — this is the correctness oracle.
        for &deg in &[10.0f32, 45.0, 90.0, 135.0, -30.0, -100.0] {
            for &(sa, sb) in &[(0.5, 0.5), (0.2, 0.9), (1.1, -0.3), (0.5, 0.5)] {
                let p = if (sa, sb) == (0.5, 0.5) {
                    flap(deg)
                } else {
                    skewed_flap(deg, sa, sb)
                };
                let ga = dihedral_grad_analytic(&p);
                let gf = dihedral_grad(&p);
                for k in 0..4 {
                    let err = (ga[k] - gf[k]).length();
                    assert!(
                        err < 2.0e-2,
                        "vertex {k} analytic {:?} vs FD {:?} (deg {deg}, skew {sa},{sb})",
                        ga[k],
                        gf[k]
                    );
                }
                // Translation invariance: the four gradients sum to ~0.
                let sum = ga[0] + ga[1] + ga[2] + ga[3];
                assert!(sum.length() < 1e-4, "grad sum not zero: {sum:?}");
            }
        }
    }

    #[test]
    fn flat_flap_is_zero_angle() {
        // fold 0° = B wing at +y, coplanar-unfolded with A wing at -y → θ = 0.
        let p = flap(0.0);
        assert!(
            dihedral_angle(&p).abs() < 1e-3,
            "flat angle {}",
            dihedral_angle(&p)
        );
    }

    #[test]
    fn folding_changes_angle_monotonically() {
        let a90 = dihedral_angle(&flap(90.0)).abs();
        let a45 = dihedral_angle(&flap(45.0)).abs();
        assert!(a90 > a45, "90° ({a90}) should bend more than 45° ({a45})");
    }

    #[test]
    fn gradient_reduces_energy_toward_rest() {
        // Moving along -∇θ·sign(θ-rest) must decrease |θ-rest|.
        let rest = 0.0f32;
        let mut p = flap(120.0);
        let theta0 = dihedral_angle(&p);
        let g = dihedral_grad(&p);
        let step = 1.0e-3 * (theta0 - rest).signum();
        for i in 0..4 {
            p[i] -= g[i] * step * 50.0;
        }
        let theta1 = dihedral_angle(&p);
        assert!(
            (theta1 - rest).abs() < (theta0 - rest).abs(),
            "energy did not decrease: {theta0} -> {theta1}"
        );
    }

    #[test]
    fn build_hinges_finds_shared_edge() {
        // Two triangles sharing edge (1,2): one interior hinge.
        let tris = vec![[0u32, 1, 2], [2, 1, 3]];
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let h = build_hinges(&tris, &pos);
        assert_eq!(h.len(), 1);
        // Flat rest.
        assert_relative_eq!(h[0].rest, 0.0, epsilon = 1e-3);
    }

    /// Dead zone: a fold within `threshold` of the rest angle must never move the rest
    /// angle, and rate 0 is a no-op even for a large fold.
    #[test]
    fn hinge_plasticity_respects_the_dead_zone() {
        let mut h = Hinge {
            v: [0, 1, 2, 3],
            rest: 0.0,
            w: 1.0,
        };
        // 5° fold, 20° dead zone ⇒ inside.
        let p = flap(5.0);
        for _ in 0..100 {
            assert!(!apply_hinge_plasticity(
                &mut h,
                &p,
                0.5,
                20.0f32.to_radians(),
                1.0 / 60.0
            ));
        }
        assert_eq!(h.rest, 0.0);
        // rate 0 with a large fold is still a no-op.
        assert!(!apply_hinge_plasticity(
            &mut h,
            &flap(120.0),
            0.0,
            0.1,
            1.0 / 60.0
        ));
        assert_eq!(h.rest, 0.0);
    }

    /// A fold held past the dead zone must creep the rest angle toward it (forming a
    /// crease), monotonically reducing bending energy, and converge so the residual
    /// `|θ − θ_rest|` equals the threshold — the yield surface is preserved.
    #[test]
    fn hinge_plasticity_creases_toward_the_dead_zone_edge() {
        let threshold = 10.0f32.to_radians();
        let mut h = Hinge {
            v: [0, 1, 2, 3],
            rest: 0.0,
            w: 1.0,
        };
        let p = flap(90.0);
        let theta = dihedral_angle(&p);
        let kb = 1.0f32;
        let energy = |hh: &Hinge| 0.5 * kb * hh.w * (theta - hh.rest) * (theta - hh.rest);
        let original = energy(&h);
        assert!(original > 1e-3);
        let mut prev = original;
        for step in 0..500 {
            let changed = apply_hinge_plasticity(&mut h, &p, 3.0, threshold, 1.0 / 60.0);
            let e = energy(&h);
            assert!(
                e <= prev + 1e-6,
                "hinge plasticity must not increase energy (step {step}): {prev} -> {e}"
            );
            if !changed {
                break;
            }
            prev = e;
        }
        // Residual deviation converges to the dead-zone edge, not to 0.
        let residual = (theta - h.rest).abs();
        assert!(
            (residual - threshold).abs() < 2e-2,
            "residual {residual} should converge to threshold {threshold}"
        );
        // The rest angle moved most of the way to the held fold: a real crease.
        assert!(
            h.rest > 0.5 * theta,
            "rest angle should have creased toward the fold: {} vs θ {theta}",
            h.rest
        );
        assert!(
            energy(&h) < 0.1 * original,
            "most bending energy should relax"
        );
    }

    /// Folding the other way must crease the other way (sign handling).
    #[test]
    fn hinge_plasticity_creases_symmetrically() {
        let threshold = 10.0f32.to_radians();
        let run = |deg: f32| {
            let mut h = Hinge {
                v: [0, 1, 2, 3],
                rest: 0.0,
                w: 1.0,
            };
            let p = flap(deg);
            for _ in 0..300 {
                apply_hinge_plasticity(&mut h, &p, 3.0, threshold, 1.0 / 60.0);
            }
            h.rest
        };
        let pos = run(90.0);
        let neg = run(-90.0);
        assert!(pos > 0.0, "positive fold should crease positive: {pos}");
        assert!(neg < 0.0, "negative fold should crease negative: {neg}");
        assert!(
            (pos + neg).abs() < 2e-2,
            "creasing should be symmetric: {pos} vs {neg}"
        );
    }
}
