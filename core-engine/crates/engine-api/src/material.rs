// SPDX-License-Identifier: GPL-2.0-or-later
//! Per-object material. Defaults from the reference `object_param(Tri)`
//! (docs/CONFIG.md): baraff-witkin, young=1000, density=1.0 (areal), bend=10.

use crate::params::Model;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct Material {
    pub model: Model,
    /// Young's modulus (density-normalized, reference convention). Higher = stiffer.
    pub young_mod: f32, // Tri: 1000
    pub poisson: f32, // 0.35
    /// Areal density kg/m^2 for shells.
    pub density: f32, // 1.0
    pub bend: f32,    // 10.0
    /// Max tensile strain (0 disables). implicit-tier. When `strain_limit_weft > 0`
    /// this is the WARP (U-direction) limit for anisotropic woven fabric.
    pub strain_limit: f32, // 0.0
    /// WEFT (V-direction) max tensile strain for anisotropic strain limiting; 0 =
    /// isotropic (the warp `strain_limit` is used for every edge). Needs per-vertex
    /// UVs on the mesh to classify edges. implicit-tier.
    pub strain_limit_weft: f32, // 0.0
    /// Contact barrier activation distance `dhat` (scene units). implicit-tier.
    pub contact_gap: f32, // 1e-3
    /// Dynamic-stiffness coupling (0 = fixed κ; >0 scales κ by local inertia so
    /// contacts hold under momentum). implicit-tier.
    pub contact_dynamic: f32, // 0.0
    /// Skin thickness offset. implicit-tier.
    pub contact_offset: f32, // 0.0 for Tri
    /// Coulomb friction. implicit-tier.
    pub friction: f32, // 0.0
    /// Baraff-Witkin membrane stiffness (warp, weft, shear). All 0 = use the
    /// isotropic edge-spring model (`young_mod`) instead. Warp≠weft = anisotropic.
    pub membrane_warp: f32, // 0.0
    pub membrane_weft: f32, // 0.0
    pub membrane_shear: f32, // 0.0
}

impl Default for Material {
    fn default() -> Self {
        Self {
            model: Model::BaraffWitkin,
            young_mod: 1000.0,
            poisson: 0.35,
            density: 1.0,
            bend: 10.0,
            strain_limit: 0.0,
            strain_limit_weft: 0.0,
            contact_gap: 1.0e-3,
            contact_dynamic: 0.0,
            contact_offset: 0.0,
            friction: 0.0,
            membrane_warp: 0.0,
            membrane_weft: 0.0,
            membrane_shear: 0.0,
        }
    }
}

/// Named fabric presets → membrane (warp/weft/shear), bending, density, friction.
///
/// These are **plausible authored presets** (a calibration starting point), not
/// measured values. The intended upgrade is to derive them from standard
/// textile tests (Kawabata KES, Cusick drape, bending length); wiring that
/// measured pipeline in is future work — until then treat the numbers as
/// tunable, with the
/// *relative ordering* (denim stiff & warp-dominant, silk soft & drapey, leather
/// stiff & very high-bend, jersey stretchy) being the meaningful, tested part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FabricPreset {
    Cotton,
    Denim,
    Silk,
    Jersey,
    Wool,
    Leather,
}

impl FabricPreset {
    /// A [`Material`] configured for this fabric (membrane model; springs off).
    pub fn material(self) -> Material {
        // (warp, weft, shear, bend, density kg/m²). Membrane keeps cloth roughly
        // inextensible; bend governs drape/wrinkle. Values are tuned so drape is
        // visible at ~1 m scale — a calibration starting point (see doc above).
        let (warp, weft, shear, bend, density) = match self {
            // Woven, warp-dominant, crisp, heavy.
            FabricPreset::Denim => (600.0, 400.0, 150.0, 12.0, 0.45),
            // Woven, balanced, medium.
            FabricPreset::Cotton => (250.0, 220.0, 70.0, 3.0, 0.15),
            // Very soft, low shear resistance, fluid drape, light.
            FabricPreset::Silk => (90.0, 85.0, 12.0, 0.3, 0.06),
            // Knit: stretchy (low membrane), low bend, light.
            FabricPreset::Jersey => (50.0, 45.0, 15.0, 0.8, 0.2),
            // Thick, medium stretch, high bend (bulky), heavy.
            FabricPreset::Wool => (180.0, 170.0, 60.0, 6.0, 0.35),
            // Nearly inextensible, isotropic, very high bend, heaviest.
            FabricPreset::Leather => (1200.0, 1200.0, 400.0, 40.0, 0.9),
        };
        Material {
            young_mod: 0.0, // springs off — use the membrane model
            density,
            bend,
            membrane_warp: warp,
            membrane_weft: weft,
            membrane_shear: shear,
            ..Default::default()
        }
    }
}

impl Material {
    /// Map Young's modulus to an XPBD stretch compliance for the preview tier.
    ///
    /// ponytail: crude `alpha = 1/young` mapping — physically rough but stable
    /// and monotone (stiffer fabric → less stretch). The measured-fabric
    /// calibration workflow (Kawabata/Cusick → params) replaces
    /// this with a proper mapping. Upgrade path: fabric preset table in this crate.
    pub fn stretch_compliance(&self) -> f32 {
        if self.young_mod <= 0.0 {
            0.0
        } else {
            1.0 / self.young_mod
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_have_sensible_relative_ordering() {
        let denim = FabricPreset::Denim.material();
        let silk = FabricPreset::Silk.material();
        let leather = FabricPreset::Leather.material();
        let jersey = FabricPreset::Jersey.material();

        // All presets use the membrane model (springs off).
        for p in [
            FabricPreset::Cotton,
            FabricPreset::Denim,
            FabricPreset::Silk,
            FabricPreset::Jersey,
            FabricPreset::Wool,
            FabricPreset::Leather,
        ] {
            assert_eq!(p.material().young_mod, 0.0, "{p:?} should use membrane");
            assert!(
                p.material().membrane_warp > 0.0,
                "{p:?} needs membrane stiffness"
            );
        }
        // Stiffness: leather > denim > silk.
        assert!(leather.membrane_warp > denim.membrane_warp);
        assert!(denim.membrane_warp > silk.membrane_warp);
        // Denim is warp-dominant (anisotropic); leather is ~isotropic.
        assert!(denim.membrane_warp > denim.membrane_weft);
        assert_eq!(leather.membrane_warp, leather.membrane_weft);
        // Bending: leather stiffest, silk softest; jersey low.
        assert!(leather.bend > denim.bend && denim.bend > silk.bend);
        assert!(jersey.bend < denim.bend);
        // Silk is the lightest of these.
        assert!(silk.density < denim.density && silk.density < leather.density);
    }
}
