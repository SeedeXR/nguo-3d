// SPDX-License-Identifier: GPL-2.0-or-later
//! The skin system: `data/skin_tones.json`.
//!
//! Ten complexions from the **Monk Skin Tone Scale** (Ellis Monk / Google,
//! CC BY 4.0). `INTEGRATION.md` §4 documents two ways to apply one and this
//! module supports both:
//!
//! **A — colour factor (recommended).** One shared neutral albedo plus a
//! per-tone linear factor. Switching complexion costs four floats: no texture
//! reload, no upload, no hitch.
//!
//! ```text
//! base_color_texture = bundle.texture(SkinMap::AlbedoNeutral)?
//! base_color_factor  = tone.factor()
//! ```
//!
//! **B — texture swap (drop-in).** Each tone also ships a baked 2K albedo.
//!
//! ```text
//! base_color_texture = bundle.tone_albedo(tone)?
//! base_color_factor  = BaseColorFactor::WHITE
//! ```
//!
//! The shipped GLB/VRM use route **B** with [`BAKED_TONE`] baked in, so the
//! files look right the instant they load. Move to A when the picker exists.
//!
//! # The trap, kept shut by the type system
//! A tone carries three colours and only one of them is a factor:
//!
//! | field | space | use |
//! |---|---|---|
//! | [`Swatch::hex`] / [`Swatch::srgb`] | **sRGB** | painting a picker swatch |
//! | [`MonkTone::base_color_linear`] | linear | the published MST colour, reference only |
//! | [`MonkTone::factor`] | linear | `pbrMetallicRoughness.baseColorFactor` |
//!
//! Feeding sRGB in as a factor washes the skin out; feeding the *published
//! linear* value in is also wrong, because the factor is pre-divided by the
//! neutral albedo's mean (~0.93). So [`BaseColorFactor`] has no public
//! constructor from floats — the only ways to get one are [`MonkTone::factor`]
//! and [`BaseColorFactor::WHITE`].

use crate::AvatarError;
use serde::Deserialize;

/// The tone baked into the shipped GLB/VRM/FBX materials (`MST05 "Golden"`).
pub const BAKED_TONE: &str = "MST05";

/// Attribution required by CC BY 4.0 if the tone picker ships.
pub const MONK_ATTRIBUTION: &str = "Complexion range based on the Monk Skin Tone Scale by Ellis \
                                    Monk in partnership with Google, used under CC BY 4.0.";

/// The complete attribution notice to display alongside the application.
///
/// **This const is the single source of truth.** `assets/avatars/LICENSE.md` and
/// `LICENSING.md` quote it *by name* rather than restating the text — four
/// divergent copies of one legal notice is how the shipped build ends up
/// under-attributing, which is exactly what happened before this was fixed.
///
/// Three obligations are satisfied here, all CC BY 4.0 and all **required**:
/// the avatars themselves (Seede XR is the licensor, not merely the copyright
/// holder — saying only "© Seede XR" states ownership while omitting the grant),
/// the skin textures, and the Monk Skin Tone Scale — whose credit line names
/// Google, because the scale was created in partnership with them and
/// `assets/avatars/docs/ATTRIBUTION.md` specifies that wording.
///
/// The licence URL is part of the const rather than bolted on at the call site,
/// because CC BY 4.0 §3(a)(1)(C) asks for it *alongside* the notice; a caller
/// that renders only this string is compliant on its own.
pub const ATTRIBUTION_NOTICE: &str =
    "Avatars \u{a9} Seede XR, licensed under CC BY 4.0. Skin textures based on material from \
     texturecan.com and cgbookcase.com (CC BY 4.0). Complexion range based on the Monk Skin \
     Tone Scale by Ellis Monk in partnership with Google (CC BY 4.0). Character animation from \
     Adobe Mixamo. Licence: https://creativecommons.org/licenses/by/4.0/";

/// A **linear** RGBA multiplier for `pbrMetallicRoughness.baseColorFactor`.
///
/// No constructor from floats on purpose — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(transparent)]
pub struct BaseColorFactor([f32; 4]);

impl BaseColorFactor {
    /// The identity factor, for route B (baked per-tone albedo).
    pub const WHITE: BaseColorFactor = BaseColorFactor([1.0, 1.0, 1.0, 1.0]);

    /// The four floats, ready to write into a glTF material.
    pub fn to_array(self) -> [f32; 4] {
        self.0
    }

    /// Linear RGB without alpha.
    pub fn rgb(self) -> [f32; 3] {
        [self.0[0], self.0[1], self.0[2]]
    }
}

/// A colour for painting a picker swatch. **Never a `baseColorFactor`.**
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Swatch {
    /// `#rrggbb`, ready for CSS or an iced colour parse.
    pub hex: String,
    /// Non-linear sRGB in 0..=1, matching `hex`.
    #[serde(rename = "base_color_srgb")]
    pub srgb: [f32; 3],
}

/// One rung of the Monk Skin Tone Scale.
///
/// The scale is **not** a lightness ramp — it spans four undertone families and
/// MST03 is genuinely lighter than MST02. Present tones in scale order, never
/// sorted by luminance.
#[derive(Debug, Clone, Deserialize)]
pub struct MonkTone {
    /// `"MST01"` .. `"MST10"`.
    pub key: String,
    /// `"Porcelain"` .. `"Rich Ebony"`.
    pub name: String,
    /// `cool` / `neutral` / `warm` / `olive`.
    pub undertone: String,
    /// UI swatch only.
    #[serde(flatten)]
    pub swatch: Swatch,
    /// The published MST colour, linearised. Reference and pipeline checking;
    /// **not** a factor — the factor divides this by the albedo mean.
    pub base_color_linear: [f32; 3],
    #[serde(default)]
    pub linear_luminance: f32,
    #[serde(rename = "base_color_factor")]
    factor: BaseColorFactor,
    /// Route B texture, bundle-root relative. Resolve via
    /// [`crate::Bundle::tone_albedo`].
    pub baked_albedo: String,
    /// Advisory: core glTF has no subsurface term, so this is not baked into
    /// the exported material. Melanin absorbs scattered light, so the weight
    /// falls from 0.33 (MST01) to 0.06 (MST10) — one value for every
    /// complexion is how deep skin ends up looking like grey plastic.
    pub subsurface_weight: f32,
    pub subsurface_color_linear: [f32; 3],
}

impl MonkTone {
    /// The linear `baseColorFactor` for route A.
    pub fn factor(&self) -> BaseColorFactor {
        self.factor
    }
}

/// One of the seven shared skin maps. All 2048², 8-bit, tileable, sampled with
/// the tiling `uv0` set — so **`AddressMode::Repeat`**, not clamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SkinMap {
    /// Route A base colour. sRGB.
    AlbedoNeutral,
    /// Tangent-space normal, green-up. Correct for glTF, Blender and Unity.
    NormalOpenGl,
    /// Y-flipped normal. DirectX/Unreal only; elsewhere it inverts lighting.
    NormalDirectX,
    /// Linear, remapped into a 0.35–0.75 skin window.
    Roughness,
    AmbientOcclusion,
    Height,
    /// Advisory scatter mask; nothing in core glTF consumes it.
    Subsurface,
}

impl SkinMap {
    pub const ALL: [SkinMap; 7] = [
        SkinMap::AlbedoNeutral,
        SkinMap::NormalOpenGl,
        SkinMap::NormalDirectX,
        SkinMap::Roughness,
        SkinMap::AmbientOcclusion,
        SkinMap::Height,
        SkinMap::Subsurface,
    ];

    /// True when the texture must be uploaded in an `*Srgb` format. Only the
    /// albedo maps are colour; uploading a normal or roughness map as sRGB is
    /// the classic way to get subtly wrong lighting everywhere.
    pub fn is_srgb(self) -> bool {
        matches!(self, SkinMap::AlbedoNeutral)
    }

    /// The `shared_maps` key in `skin_tones.json`.
    pub fn as_str(self) -> &'static str {
        match self {
            SkinMap::AlbedoNeutral => "albedo_neutral",
            SkinMap::NormalOpenGl => "normal_opengl",
            SkinMap::NormalDirectX => "normal_directx",
            SkinMap::Roughness => "roughness",
            SkinMap::AmbientOcclusion => "ambient_occlusion",
            SkinMap::Height => "height",
            SkinMap::Subsurface => "subsurface",
        }
    }
}

/// The seven shared maps, bundle-root relative.
#[derive(Debug, Clone, Deserialize)]
pub struct SharedMaps {
    pub albedo_neutral: String,
    pub normal_opengl: String,
    pub normal_directx: String,
    pub roughness: String,
    pub ambient_occlusion: String,
    pub height: String,
    pub subsurface: String,
}

impl SharedMaps {
    pub fn path(&self, map: SkinMap) -> &str {
        match map {
            SkinMap::AlbedoNeutral => &self.albedo_neutral,
            SkinMap::NormalOpenGl => &self.normal_opengl,
            SkinMap::NormalDirectX => &self.normal_directx,
            SkinMap::Roughness => &self.roughness,
            SkinMap::AmbientOcclusion => &self.ambient_occlusion,
            SkinMap::Height => &self.height,
            SkinMap::Subsurface => &self.subsurface,
        }
    }
}

/// Skin is a dielectric: metallic 0, IOR 1.44.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct MaterialConstants {
    pub metallic: f32,
    pub ior: f32,
    #[serde(default)]
    pub specular: f32,
}

/// Parsed `data/skin_tones.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct SkinLibrary {
    pub standard: String,
    #[serde(default)]
    pub standard_url: String,
    #[serde(default)]
    pub standard_licence: String,
    /// 2048.
    pub resolution: u32,
    pub bit_depth: u32,
    pub tiling: bool,
    /// One `uv0` tile, in metres of skin (0.20).
    pub tile_metres: f64,
    /// `"UVMap"` — the tiling set, i.e. `TEXCOORD_0`.
    pub uv_layer: String,
    pub shared_maps: SharedMaps,
    pub material_constants: MaterialConstants,
    pub roughness_range: [f32; 2],
    tones: Vec<MonkTone>,
}

impl SkinLibrary {
    /// The credit line for the about/credits screen.
    pub fn credit(&self) -> &'static str {
        MONK_ATTRIBUTION
    }

    /// The ten tones, **in scale order**. Show them in this order.
    pub fn tones(&self) -> &[MonkTone] {
        &self.tones
    }

    /// Look a tone up by key (`"MST05"`). A typo is an error listing what
    /// exists, never a silent fallback.
    pub fn tone(&self, key: &str) -> Result<&MonkTone, AvatarError> {
        self.tones
            .iter()
            .find(|t| t.key == key)
            .ok_or_else(|| AvatarError::UnknownTone {
                id: key.to_string(),
                known: self
                    .tones
                    .iter()
                    .map(|t| t.key.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }

    /// The tone already baked into the shipped meshes ([`BAKED_TONE`]).
    pub fn baked_tone(&self) -> Result<&MonkTone, AvatarError> {
        self.tone(BAKED_TONE)
    }
}

#[cfg(test)]
mod tests {

    /// The shipped notice must satisfy all three CC BY 4.0 obligations.
    ///
    /// Regression test for a real defect: the const previously said only
    /// "Avatars (c) Seede XR" — stating ownership while omitting the *grant* —
    /// and dropped "in partnership with Google" from the Monk credit, while the
    /// call site bolted the licence URL on separately to patch half the gap.
    /// The rendered build therefore under-attributed a licence that requires
    /// attribution.
    #[test]
    fn attribution_notice_satisfies_every_cc_by_obligation() {
        let n = super::ATTRIBUTION_NOTICE;
        for needle in [
            "Seede XR",                 // the avatars' licensor
            "licensed under CC BY 4.0", // ...and the grant, not just ownership
            "texturecan.com",
            "cgbookcase.com",
            "Monk Skin Tone Scale",
            "Ellis Monk",
            "in partnership with Google", // required wording per the bundle
            "Adobe Mixamo",
            "https://creativecommons.org/licenses/by/4.0/", // CC BY 4.0 s3(a)(1)(C)
        ] {
            assert!(
                n.contains(needle),
                "attribution notice is missing {needle:?}: {n}"
            );
        }
    }

    use super::*;

    #[test]
    fn white_is_the_route_b_factor() {
        assert_eq!(BaseColorFactor::WHITE.to_array(), [1.0; 4]);
        assert_eq!(BaseColorFactor::WHITE.rgb(), [1.0; 3]);
    }

    #[test]
    fn only_albedo_is_srgb_and_map_keys_are_unique() {
        let srgb: Vec<_> = SkinMap::ALL.iter().filter(|m| m.is_srgb()).collect();
        assert_eq!(srgb, vec![&SkinMap::AlbedoNeutral]);
        let mut keys: Vec<_> = SkinMap::ALL.iter().map(|m| m.as_str()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), SkinMap::ALL.len());
    }

    #[test]
    fn a_tone_keeps_its_swatch_and_its_factor_apart() {
        // The MST05 row of skin_tones.json, trimmed to the required fields.
        // r##..##: the hex swatch contains `"#`, which would close an `r#` string.
        let json = r##"{
            "key": "MST05", "name": "Golden", "undertone": "warm",
            "hex": "#d7be93",
            "base_color_srgb": [0.843137, 0.745, 0.57549],
            "base_color_linear": [0.679542, 0.514766, 0.290685],
            "base_color_factor": [0.727758, 0.55129, 0.311309, 1.0],
            "baked_albedo": "textures/skin/tones/skin_albedo_MST05.png",
            "subsurface_weight": 0.224,
            "subsurface_color_linear": [0.779542, 0.216202, 0.087205]
        }"##;
        let t: MonkTone = serde_json::from_str(json).unwrap();
        assert_eq!(t.swatch.hex, "#d7be93");
        assert_eq!(t.factor().to_array(), [0.727758, 0.55129, 0.311309, 1.0]);
        // The factor is the published linear colour divided by the neutral
        // albedo mean, so it is brighter — never interchangeable with it.
        assert!(t.factor().rgb()[0] > t.base_color_linear[0]);
        assert!(t.swatch.srgb[0] > t.base_color_linear[0]);
    }

    #[test]
    fn an_unknown_tone_lists_what_exists() {
        let lib = SkinLibrary {
            standard: String::new(),
            standard_url: String::new(),
            standard_licence: String::new(),
            resolution: 2048,
            bit_depth: 8,
            tiling: true,
            tile_metres: 0.2,
            uv_layer: "UVMap".into(),
            shared_maps: SharedMaps {
                albedo_neutral: String::new(),
                normal_opengl: String::new(),
                normal_directx: String::new(),
                roughness: String::new(),
                ambient_occlusion: String::new(),
                height: String::new(),
                subsurface: String::new(),
            },
            material_constants: MaterialConstants {
                metallic: 0.0,
                ior: 1.44,
                specular: 0.5,
            },
            roughness_range: [0.35, 0.75],
            tones: Vec::new(),
        };
        let err = lib.tone("MST-7").unwrap_err();
        assert!(matches!(err, AvatarError::UnknownTone { .. }), "{err}");
        assert!(err.to_string().contains("MST-7"), "{err}");
    }
}
