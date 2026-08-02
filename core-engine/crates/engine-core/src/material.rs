// SPDX-License-Identifier: GPL-2.0-or-later
//! Material parameter conversion: engineering constants → Lamé parameters.
//!
//! Ports the reference's `convert_prop` (`ppf-cts-solver/src/builder.rs:1273-1277`), which is
//! the single place it turns a scene's `young-mod` / `poiss-rat` into the `(mu, lambda)` every
//! constitutive model actually consumes — Baraff-Witkin faces (`scene.rs:1648`), isotropic
//! shells, and tets (`scene.rs:1758`) all go through it.
//!
//! This resolves audit item A-4. Our driver previously mapped `young_mod` to
//! `(warp, weft, shear) = (y, y, 0.5·y)`, which was INFERRED — it never used `poiss_rat` at
//! all, even though every gauge scene ships it. The measured consequences on `hang`
//! (`y = 100, ν = 0.35`) were a stretch stiffness 2.7× too high (100 vs 37.04) and a shear
//! stiffness 1.7× too low (50 vs 86.42), and on `domino`/`friction` (`ν = 0.49`) a shear
//! stiffness **33× too low** (0.5·y vs 16.44·y).

/// Lamé parameters `(mu, lambda)` from Young's modulus and Poisson's ratio.
///
/// ```text
/// mu     = E / (2 (1 + ν))
/// lambda = E ν / ((1 + ν)(1 - 2ν))
/// ```
///
/// `mu` scales the STRETCH response (both principal directions, since the material is
/// isotropic) and `lambda` the SHEAR / volumetric response — that is how the reference's
/// `BaraffWitkin::stretch_*` and `shear_*` consume them (`energy.cu:317-320`).
///
/// `lambda` diverges as `ν → 0.5` (incompressibility); the reference asserts
/// `0 < ν < 0.5` (`scene.rs:1645-1646`). `ν` is clamped just inside that range here rather
/// than returning an infinity, so a malformed scene degrades instead of poisoning the solve
/// with NaN.
pub fn lame_from_young_poisson(young: f32, poisson: f32) -> (f32, f32) {
    // The reference asserts the open interval; clamp rather than divide by zero. NOTE
    // `f32::clamp` PROPAGATES NaN, so a non-finite ratio has to be replaced explicitly —
    // `degenerate_poisson_ratios_stay_finite` caught exactly that, and a NaN here would have
    // reached the Hessian.
    let nu = if poisson.is_finite() {
        poisson.clamp(1.0e-6, 0.5 - 1.0e-4)
    } else {
        0.35
    };
    let mu = young / (2.0 * (1.0 + nu));
    let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    (mu, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed against the reference formula for the three Poisson ratios the gauge
    /// actually ships, so a change to the conversion is caught with concrete numbers rather
    /// than by re-deriving the algebra.
    #[test]
    fn matches_the_reference_conversion_for_every_gauge_poisson_ratio() {
        // ν = 0.35 (most scenes), E = 100 → mu = 100/2.7 = 37.037, λ = 35/0.405 = 86.4197
        let (mu, lambda) = lame_from_young_poisson(100.0, 0.35);
        assert!((mu - 37.037_04).abs() < 1.0e-3, "mu {mu}");
        assert!((lambda - 86.419_75).abs() < 1.0e-2, "lambda {lambda}");

        // ν = 0.25 (five-twist), E = 100 → mu = 100/2.5 = 40, λ = 25/(1.25·0.5) = 40
        let (mu, lambda) = lame_from_young_poisson(100.0, 0.25);
        assert!((mu - 40.0).abs() < 1.0e-3, "mu {mu}");
        assert!((lambda - 40.0).abs() < 1.0e-3, "lambda {lambda}");

        // ν = 0.49 (domino, friction), E = 100 → mu = 100/2.98 = 33.557,
        // λ = 49/(1.49·0.02) = 1644.295. This is the case our old (y, y, 0.5y) mapping got
        // wrong by ~33x in shear.
        let (mu, lambda) = lame_from_young_poisson(100.0, 0.49);
        assert!((mu - 33.557_05).abs() < 1.0e-3, "mu {mu}");
        assert!((lambda - 1644.295).abs() < 0.1, "lambda {lambda}");
        assert!(
            lambda / 0.5 / 100.0 > 30.0,
            "near-incompressible shear must be far stiffer than the old 0.5*y mapping"
        );
    }

    /// Both parameters scale linearly in `E` at fixed `ν` — the property that made the A-4
    /// ablation (`MGUSO_ABLATE_YOUNG_SCALE`) interpretable as a pure stiffness scaling.
    #[test]
    fn both_parameters_are_linear_in_youngs_modulus() {
        let (m1, l1) = lame_from_young_poisson(100.0, 0.35);
        let (m2, l2) = lame_from_young_poisson(400.0, 0.35);
        assert!((m2 / m1 - 4.0).abs() < 1.0e-4, "mu ratio {}", m2 / m1);
        assert!((l2 / l1 - 4.0).abs() < 1.0e-4, "lambda ratio {}", l2 / l1);
    }

    /// Degenerate inputs must stay finite: the reference asserts `0 < ν < 0.5`, and a scene
    /// that violates it should degrade rather than produce NaN in the Hessian.
    #[test]
    fn degenerate_poisson_ratios_stay_finite() {
        for nu in [-1.0f32, 0.0, 0.5, 0.75, 1.0, f32::NAN] {
            let (mu, lambda) = lame_from_young_poisson(100.0, nu);
            assert!(
                mu.is_finite() && lambda.is_finite(),
                "ν = {nu} gave {mu}/{lambda}"
            );
            assert!(mu > 0.0 && lambda >= 0.0, "ν = {nu} gave {mu}/{lambda}");
        }
        // Zero stiffness stays zero (an inert element, e.g. the reference's Pdrd faces).
        let (mu, lambda) = lame_from_young_poisson(0.0, 0.35);
        assert_eq!((mu, lambda), (0.0, 0.0));
    }
}
