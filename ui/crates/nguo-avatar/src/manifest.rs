// SPDX-License-Identifier: GPL-2.0-or-later
//! Serde model of `assets/avatars/data/manifest.json`.
//!
//! Field names mirror the shipped JSON exactly and unknown fields are ignored
//! (serde's default) so the bundle can grow without breaking this crate.
//! Anything the catalogue depends on is required and will fail the parse.
//!
//! # The manifest describes more than the bundle ships
//! `files` declares nine paths per avatar — the **full pipeline output**. The
//! shipped bundle is the **runtime subset**: 8 of the 18 declared paths exist.
//! `blend`, `glb.lod0` and all three `unity_fbx` entries are declared and
//! absent. That is not corruption, so nothing here resolves a path at parse
//! time; presence is a question you ask per asset via
//! [`crate::Bundle::has_asset`]. See `tests/bundle.rs`, which pins the exact
//! set so a future bundle that adds the FBX files tells us.

use serde::Deserialize;
use std::collections::BTreeMap;

/// Top level of `data/manifest.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// `"Nguo 3D Avatars"`.
    pub asset: String,
    pub version: String,
    #[serde(default)]
    pub studio: String,
    #[serde(default)]
    pub purpose: String,
    #[serde(default)]
    pub generated_with: String,
    pub conventions: Conventions,
    /// `LOD2` in every bundle shipped so far — i.e. `files.glb.runtime`.
    pub default_tier: Tier,
    pub skin: SkinBlock,
    /// Keyed `"male"` / `"female"`. A map, not a list: the bundle ships two
    /// bodies and addresses them by name.
    pub avatars: BTreeMap<String, Avatar>,
}

/// The `conventions` block.
///
/// **The three axis blocks do not agree** — Blender is Z-up/−Y-forward, glTF
/// and VRM are +Y-up/+Z-forward, Unity is +Y-up/+Z-forward at scale 1.0. Read
/// the one for the format you are consuming; [`Conventions::is_gltf_space`]
/// answers the only question this crate needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Conventions {
    /// `"metres"`. Checked on load.
    pub units: String,
    pub blender: Axes,
    pub gltf_vrm: Axes,
    pub unity_fbx: Axes,
    pub fps: f64,
    /// `UVMap` — tiling skin detail. One tile spans [`Self::UV0_TILE_METRES`]
    /// of skin and it deliberately runs outside `[0,1]`, so the sampler **must**
    /// use `AddressMode::Repeat`. Clamping smears the skin.
    #[serde(default)]
    pub uv0: String,
    /// `UVMap_Atlas` — the same unwrap packed into 0..1, for baking. Nothing
    /// samples it today.
    #[serde(default)]
    pub uv1: String,
}

impl Conventions {
    /// One `uv0` tile, in metres of skin. From `conventions.uv0` prose and
    /// `skin_tones.json.tile_metres`, which agree.
    pub const UV0_TILE_METRES: f64 = 0.20;

    /// True when the GLB/VRM files need no axis conversion to be consumed as
    /// glTF — which is also nguo's own space, so no conversion anywhere.
    pub fn is_gltf_space(&self) -> bool {
        self.gltf_vrm.up == "+Y"
            && self.gltf_vrm.forward == "+Z"
            && self.gltf_vrm.handedness.as_deref() == Some("right")
    }
}

/// One axis block. The three blocks carry different extra keys, so everything
/// past `up`/`forward` is optional.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Axes {
    pub up: String,
    pub forward: String,
    #[serde(default)]
    pub character_right: Option<String>,
    #[serde(default)]
    pub handedness: Option<String>,
    #[serde(default)]
    pub scale: Option<f64>,
}

/// The `skin` block. Colour numbers live in [`crate::SkinLibrary`]
/// (`skin_tones.json`), so there is one source of truth for them.
#[derive(Debug, Clone, Deserialize)]
pub struct SkinBlock {
    pub standard: String,
    #[serde(default)]
    pub standard_licence: String,
    pub tone_count: usize,
    /// Path to `skin_tones.json`, relative to the bundle root.
    pub detail: String,
}

/// A geometry tier. **`Lod2` is the default runtime avatar** and is carried by
/// `files.glb.runtime` — not by `glb.lod0`, which is an archival master.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
pub enum Tier {
    /// 843 443 v / 1 686 890 t (male). Archival master, **not shipped as GLB**.
    #[serde(rename = "LOD0")]
    Lod0,
    /// ~100 000 v. High-quality real-time / hero renders.
    #[serde(rename = "LOD1")]
    Lod1,
    /// ~25 000 v. The default runtime avatar.
    #[serde(rename = "LOD2")]
    Lod2,
    /// ~7 500 v. Cloth collision body — watertight and manifold.
    #[serde(rename = "COL")]
    Col,
}

impl Tier {
    pub const ALL: [Tier; 4] = [Tier::Lod0, Tier::Lod1, Tier::Lod2, Tier::Col];

    /// The manifest spelling, and the key used inside `verification`.
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Lod0 => "LOD0",
            Tier::Lod1 => "LOD1",
            Tier::Lod2 => "LOD2",
            Tier::Col => "COL",
        }
    }

    /// The GLB that carries this tier. Note [`Tier::Lod2`] maps to
    /// [`Asset::GlbRuntime`]: the plain `nguo_male.glb` *is* LOD2.
    pub fn glb(self) -> Asset {
        match self {
            Tier::Lod0 => Asset::GlbLod0,
            Tier::Lod1 => Asset::GlbLod1,
            Tier::Lod2 => Asset::GlbRuntime,
            Tier::Col => Asset::GlbCollision,
        }
    }
}

/// One file the manifest declares for an avatar. Nine per avatar; only
/// [`Asset::GlbRuntime`], [`Asset::GlbLod1`], [`Asset::GlbCollision`] and
/// [`Asset::Vrm`] are actually shipped — ask [`crate::Bundle::has_asset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Asset {
    /// Editable Blender source. Declared, not shipped.
    Blend,
    /// `avatars/glb/nguo_<sex>.glb` — LOD2, the default runtime avatar.
    GlbRuntime,
    GlbLod1,
    /// Archival master. Declared, not shipped.
    GlbLod0,
    /// `avatars/glb/nguo_<sex>_collision.glb` — the cloth collider.
    GlbCollision,
    /// `avatars/vrm/nguo_<sex>.vrm` — runtime GLB plus `VRMC_vrm` 1.0.
    Vrm,
    /// Unity Humanoid FBX. Declared, not shipped.
    FbxRuntime,
    FbxLod1,
    FbxCollision,
}

impl Asset {
    /// Every declared asset, in manifest order.
    pub const ALL: [Asset; 9] = [
        Asset::Blend,
        Asset::GlbRuntime,
        Asset::GlbLod1,
        Asset::GlbLod0,
        Asset::GlbCollision,
        Asset::Vrm,
        Asset::FbxRuntime,
        Asset::FbxLod1,
        Asset::FbxCollision,
    ];

    /// The manifest key path, e.g. `"glb.runtime"`. Used in error messages so
    /// a missing file names the manifest entry that promised it.
    pub fn as_str(self) -> &'static str {
        match self {
            Asset::Blend => "blend",
            Asset::GlbRuntime => "glb.runtime",
            Asset::GlbLod1 => "glb.lod1",
            Asset::GlbLod0 => "glb.lod0",
            Asset::GlbCollision => "glb.collision",
            Asset::Vrm => "vrm",
            Asset::FbxRuntime => "unity_fbx.runtime",
            Asset::FbxLod1 => "unity_fbx.lod1",
            Asset::FbxCollision => "unity_fbx.collision",
        }
    }

    /// True for the two formats `engine_io::load_mesh` can read.
    pub fn is_gltf(self) -> bool {
        !matches!(
            self,
            Asset::Blend | Asset::FbxRuntime | Asset::FbxLod1 | Asset::FbxCollision
        )
    }
}

/// One avatar: `manifest.avatars.<male|female>`.
#[derive(Debug, Clone, Deserialize)]
pub struct Avatar {
    /// `"Nguo Male"` / `"Nguo Female"`.
    pub name: String,
    /// Height in metres, as measured from the sculpt. The female body is
    /// **taller** (1.905 m vs 1.830 m); the pipeline never rescales.
    pub stature_m: f64,
    pub arm_span_m: f64,
    #[serde(default)]
    pub volume_m3: f64,
    #[serde(default)]
    pub surface_area_m2: f64,
    /// Girths measured off the source geometry, keyed by landmark name — see
    /// [`crate::Landmark`]. Prefer these over computing your own.
    pub measurements: BTreeMap<String, Measurement>,
    pub skeleton: SkeletonMetrics,
    pub tiers: Vec<TierInfo>,
    pub files: Files,
    #[serde(default)]
    pub animations: Vec<AnimationClip>,
    #[serde(default)]
    pub previews: Vec<String>,
    pub verification: Verification,
}

impl Avatar {
    /// The declared path for `asset`, relative to the bundle root. Says
    /// nothing about whether the file exists.
    pub fn declared(&self, asset: Asset) -> &str {
        let f = &self.files;
        match asset {
            Asset::Blend => &f.blend,
            Asset::GlbRuntime => &f.glb.runtime,
            Asset::GlbLod1 => &f.glb.lod1,
            Asset::GlbLod0 => &f.glb.lod0,
            Asset::GlbCollision => &f.glb.collision,
            Asset::Vrm => &f.vrm,
            Asset::FbxRuntime => &f.unity_fbx.runtime,
            Asset::FbxLod1 => &f.unity_fbx.lod1,
            Asset::FbxCollision => &f.unity_fbx.collision,
        }
    }

    pub fn tier(&self, tier: Tier) -> Option<&TierInfo> {
        self.tiers.iter().find(|t| t.tier == tier)
    }

    /// Clip names in bundle order. All 19 are retargeted onto both bodies, so
    /// any clip plays on either avatar; the `f_`/`m_` prefix is the Mixamo
    /// folder it came from, not the body it belongs to.
    pub fn clip_names(&self) -> impl Iterator<Item = &str> {
        self.animations.iter().map(|c| c.action.as_str())
    }

    pub fn clip(&self, action: &str) -> Option<&AnimationClip> {
        self.animations.iter().find(|c| c.action == action)
    }
}

/// One measured girth, plus where up the body it was taken.
///
/// `circumference_m` is the convex-hull perimeter of a horizontal
/// cross-section — what a taut tape reads. Upper-body girths are restricted to
/// the torso, since these are T-pose bodies and a full chest slice also cuts
/// both arms.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Measurement {
    pub circumference_m: f64,
    pub height_m: f64,
    /// `height_m / stature_m`. Absent for `thigh` and `calf`.
    #[serde(default)]
    pub height_fraction: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SkeletonMetrics {
    /// 65 on both bodies.
    pub bones: usize,
    /// `"Mixamo (mixamorig: prefix)"`.
    pub naming: String,
    pub hip_height_m: f64,
    pub inseam_m: f64,
    pub femur_m: f64,
    pub tibia_m: f64,
    pub upper_arm_m: f64,
    pub forearm_m: f64,
    /// 38 of the 65 bones carry skin weights. The other 27 (end markers,
    /// thumbs, distal fingers) animate but do not deform — deliberate, see
    /// `docs/PIPELINE_NOTES.md`.
    pub deforming_bones: usize,
    /// 4, so one `uvec4` + one `vec4` per vertex is the whole skin.
    pub max_influences_per_vertex: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TierInfo {
    pub tier: Tier,
    /// Blender object name, e.g. `Nguo_Male_Body_LOD2`. This is also the key
    /// used by [`Verification::animation_edge_growth_over_median_edge`].
    pub object: String,
    pub verts: u64,
    pub tris: u64,
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Files {
    pub blend: String,
    pub glb: GlbFiles,
    pub vrm: String,
    pub unity_fbx: FbxFiles,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlbFiles {
    /// LOD2. The default runtime avatar.
    pub runtime: String,
    pub lod1: String,
    pub lod0: String,
    pub collision: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FbxFiles {
    pub runtime: String,
    pub lod1: String,
    pub collision: String,
}

/// One baked Mixamo clip. Sampler inputs are seconds, authored at 30 fps.
/// Clips carry root motion (`mixamorig:Hips` translates) — zero that channel
/// for in-place playback.
#[derive(Debug, Clone, Deserialize)]
pub struct AnimationClip {
    /// The action name to select by; there is no index contract.
    pub action: String,
    #[serde(default)]
    pub source_clip: String,
    pub frames: u32,
    pub duration_s: f64,
}

/// The pipeline's own verification numbers. `data/test_mesh_integrity_*.json`
/// and `data/test_animation_distortion_*.json` are the full reports these are
/// summarised from — `tests/bundle.rs` cross-checks the two against each other.
#[derive(Debug, Clone, Deserialize)]
pub struct Verification {
    pub lod0_vs_source_worst_vertex_m: f64,
    /// Keyed by [`Tier::as_str`].
    pub silhouette_mismatch_fraction: BTreeMap<String, f64>,
    /// Keyed by [`TierInfo::object`].
    pub animation_edge_growth_over_median_edge: BTreeMap<String, f64>,
    pub collision_proxy_posed_deviation_mm: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod2_is_the_runtime_glb_not_lod0() {
        assert_eq!(Tier::Lod2.glb(), Asset::GlbRuntime);
        assert_eq!(Tier::Lod0.glb(), Asset::GlbLod0);
    }

    #[test]
    fn asset_keys_are_unique_and_name_the_manifest_path() {
        let mut keys: Vec<_> = Asset::ALL.iter().map(|a| a.as_str()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), Asset::ALL.len());
        assert_eq!(Asset::GlbRuntime.as_str(), "glb.runtime");
        assert!(Asset::GlbCollision.is_gltf());
        assert!(!Asset::FbxRuntime.is_gltf());
        assert!(!Asset::Blend.is_gltf());
    }
}
