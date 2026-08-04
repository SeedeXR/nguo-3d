// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-avatar` — typed access to the nguo avatar bundle.
//!
//! The bundle (`assets/avatars/`) ships **two** rigged bodies, four geometry
//! tiers each, a shared 65-bone Mixamo skeleton, ten Monk skin tones and 19
//! runway clips retargeted onto both bodies. This crate turns that directory
//! into data:
//!
//! - [`Bundle`] — open a bundle, resolve paths, ask what is present, load meshes.
//! - [`SkinLibrary`] — the ten tones, both as a linear factor and as a baked albedo.
//! - [`Measurements`] — the bundle's own measured girths, in millimetres.
//!
//! ```no_run
//! # fn main() -> Result<(), nguo_avatar::AvatarError> {
//! use nguo_avatar::{Asset, Bundle, Landmark, SkinMap, Tier};
//!
//! let bundle = Bundle::find()?;
//! let male = bundle.avatar("male")?;
//!
//! // The default runtime avatar is LOD2 — i.e. plain `nguo_male.glb`.
//! let glb = bundle.tier_path(male, bundle.manifest().default_tier)?;
//! let collider = bundle.load_mesh(male, Asset::GlbCollision)?;
//!
//! // Route A skin: neutral albedo + a linear factor.
//! let tone = bundle.skin().tone("MST05")?;
//! let albedo = bundle.texture(SkinMap::AlbedoNeutral)?;
//! let factor = tone.factor().to_array();
//!
//! let m = bundle.measurements(male);
//! let waist = m.girth(Landmark::Waist).expect("bundle measures the waist");
//! println!("{} waist {:.0} mm", male.name, waist.circumference.mm());
//! # let _ = (glb, collider, albedo, factor, Tier::Lod2); Ok(()) }
//! ```
//!
//! # Coordinate space
//! The GLB/VRM files are **metres, +Y up, +Z forward, right-handed** — glTF's
//! own convention and nguo's, so **no axis conversion is needed**. Feet sit on
//! the ground plane and the midline is at X = 0. [`Bundle::open`] checks this
//! and refuses a bundle that says otherwise. The manifest also carries the
//! Blender (Z-up) and Unity axis blocks; they do **not** agree with each other,
//! so read the one matching the format you are consuming.
//!
//! # The manifest declares more files than the bundle ships
//! `files` lists nine paths per avatar — the full pipeline output. The bundle
//! is the runtime subset: **8 of the 18 declared paths exist**. `blend`,
//! `glb.lod0` and the three `unity_fbx` entries are declared and absent.
//! Nothing is resolved at load; use [`Bundle::has_asset`] / [`Bundle::has_tier`]
//! before assuming, and [`Bundle::asset_path`] to get a typed
//! [`AvatarError::AssetAbsent`] instead of an I/O surprise later.
//!
//! # Licensing — read before shipping
//! Bodies are Seede XR's. Skin textures and the Monk Skin Tone Scale are both
//! CC BY 4.0 and **require attribution** where the avatars ship
//! ([`ATTRIBUTION_NOTICE`], [`MONK_ATTRIBUTION`]). Mixamo motion may not be
//! redistributed as standalone animation data. See
//! `assets/avatars/docs/ATTRIBUTION.md`.

#![forbid(unsafe_code)]

pub mod manifest;
pub mod measure;
mod skin;

pub use manifest::{
    AnimationClip, Asset, Avatar, Conventions, Manifest, Measurement, SkeletonMetrics, Tier,
    TierInfo, Verification,
};
pub use measure::{Girth, Landmark, Measurements};
pub use skin::{
    BaseColorFactor, MonkTone, SharedMaps, SkinLibrary, SkinMap, Swatch, ATTRIBUTION_NOTICE,
    BAKED_TONE, MONK_ATTRIBUTION,
};

use std::path::{Path, PathBuf};

/// Environment override for the bundle location.
pub const BUNDLE_ENV: &str = "NGUO_AVATAR_BUNDLE";

/// The manifest's location inside the bundle. Note it is **inside `data/`**.
pub const MANIFEST_PATH: &str = "data/manifest.json";

/// Everything that can go wrong reading a bundle.
#[derive(Debug, thiserror::Error)]
pub enum AvatarError {
    #[error("avatar bundle not found at {0} — set NGUO_AVATAR_BUNDLE to its location")]
    BundleNotFound(PathBuf),

    /// The bundle's directory tree is there but its LFS assets are not: every
    /// `.glb`/`.vrm`/`.png` is ~130 bytes of pointer text. Callers must treat
    /// this exactly like [`AvatarError::BundleNotFound`] — the bundle is not
    /// usable, and nothing but fetching the objects will change that.
    #[error(
        "avatar bundle at {root} is not hydrated — {file} is a git-LFS pointer, not the \
         asset. Run `git lfs pull` (in CI, check out with `lfs: true`)"
    )]
    LfsPointer { root: PathBuf, file: PathBuf },

    #[error("avatar bundle at {root} is missing {file}")]
    MissingFile { root: PathBuf, file: String },

    #[error("cannot read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} is not a valid nguo avatar bundle file: {source}")]
    BadJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error(
        "bundle at {root} is authored in {found} ({units}); this crate needs metres, +Y up, \
         +Z forward, right-handed — the glTF convention"
    )]
    WrongSpace {
        root: PathBuf,
        found: String,
        units: String,
    },

    /// The manifest declares this file but the bundle does not ship it. Expected
    /// for `blend`, `glb.lod0` and the Unity FBX set — the manifest describes
    /// the whole pipeline, the bundle ships the runtime subset.
    #[error(
        "{avatar} declares {asset} at {declared} but this bundle does not ship it — the manifest \
         describes the full pipeline output, the bundle ships the runtime subset"
    )]
    AssetAbsent {
        avatar: String,
        asset: &'static str,
        declared: PathBuf,
    },

    #[error("unknown avatar {id:?} — bundle has: {known}")]
    UnknownAvatar { id: String, known: String },

    #[error("unknown skin tone {id:?} — bundle has: {known}")]
    UnknownTone { id: String, known: String },

    #[error("cannot load mesh {path}: {source}")]
    MeshLoad {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },
}

/// An opened avatar bundle: manifest and skin library parsed once, plus the
/// root to resolve files against.
///
/// Opening reads two JSON files and no mesh, texture or GLB content — asset
/// paths are resolved on demand, because most of the paths the manifest
/// declares are not in the bundle. It does read the first 42 bytes of each
/// declared asset that exists, to reject a bundle whose git-LFS objects were
/// never fetched (see [`AvatarError::LfsPointer`]).
#[derive(Debug, Clone)]
pub struct Bundle {
    root: PathBuf,
    manifest: Manifest,
    skin: SkinLibrary,
}

impl Bundle {
    /// Open the bundle rooted at `root` — the directory holding `data/`,
    /// `avatars/` and `textures/`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Bundle, AvatarError> {
        let root = root.into();
        let span = tracing::info_span!("open_avatar_bundle", root = %root.display());
        let _enter = span.enter();

        if !root.join(MANIFEST_PATH).is_file() {
            return Err(AvatarError::BundleNotFound(root));
        }
        let manifest: Manifest = read_json(&root, MANIFEST_PATH)?;
        let c = &manifest.conventions;
        if c.units != "metres" || !c.is_gltf_space() {
            return Err(AvatarError::WrongSpace {
                root,
                found: format!(
                    "up {}, forward {}, {}-handed",
                    c.gltf_vrm.up,
                    c.gltf_vrm.forward,
                    c.gltf_vrm.handedness.as_deref().unwrap_or("un")
                ),
                units: c.units.clone(),
            });
        }
        let skin: SkinLibrary = read_json(&root, &manifest.skin.detail)?;
        // The two JSON inputs above are normal git; every mesh and texture is
        // git-LFS. So a checkout without LFS gets this far happily and then hands
        // out paths to ~130-byte pointer stubs — files that EXIST, so `has_asset`
        // says yes and the bundle looks present while none of its geometry is.
        // Fail here, once, rather than at each consumer's parse.
        // ponytail: LFS hydration is all-or-nothing per checkout, so the first
        // declared asset that is a pointer settles it — no need to stat textures too.
        if let Some(file) = manifest
            .avatars
            .values()
            .flat_map(|a| Asset::ALL.map(|k| root.join(a.declared(k))))
            .find(|p: &PathBuf| engine_io::is_lfs_pointer(p))
        {
            return Err(AvatarError::LfsPointer { root, file });
        }
        tracing::info!(
            version = %manifest.version,
            avatars = manifest.avatars.len(),
            tones = skin.tones().len(),
            default_tier = manifest.default_tier.as_str(),
            "bundle opened"
        );
        Ok(Bundle {
            root,
            manifest,
            skin,
        })
    }

    /// Open the bundle at `$NGUO_AVATAR_BUNDLE`, else the one vendored in this
    /// checkout at `assets/avatars/`.
    pub fn find() -> Result<Bundle, AvatarError> {
        Bundle::open(default_root())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn skin(&self) -> &SkinLibrary {
        &self.skin
    }

    /// The catalogue as `(key, avatar)`, keys being `"female"` and `"male"`.
    pub fn avatars(&self) -> impl Iterator<Item = (&str, &Avatar)> {
        self.manifest.avatars.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Look an avatar up by manifest key (`"male"` / `"female"`).
    pub fn avatar(&self, key: &str) -> Result<&Avatar, AvatarError> {
        self.manifest
            .avatars
            .get(key)
            .ok_or_else(|| AvatarError::UnknownAvatar {
                id: key.to_string(),
                known: self
                    .manifest
                    .avatars
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }

    /// Where `asset` *would* live. No I/O — the file may not exist.
    pub fn declared_path(&self, avatar: &Avatar, asset: Asset) -> PathBuf {
        self.root.join(avatar.declared(asset))
    }

    /// True when the bundle actually ships `asset` for this avatar.
    pub fn has_asset(&self, avatar: &Avatar, asset: Asset) -> bool {
        self.declared_path(avatar, asset).is_file()
    }

    /// Absolute path to `asset`, or [`AvatarError::AssetAbsent`] naming the
    /// path the manifest promised.
    pub fn asset_path(&self, avatar: &Avatar, asset: Asset) -> Result<PathBuf, AvatarError> {
        let path = self.declared_path(avatar, asset);
        if path.is_file() {
            Ok(path)
        } else {
            Err(AvatarError::AssetAbsent {
                avatar: avatar.name.clone(),
                asset: asset.as_str(),
                declared: path,
            })
        }
    }

    /// True when the GLB carrying `tier` is shipped. `Lod0` is false in every
    /// bundle so far — it is an archival master.
    pub fn has_tier(&self, avatar: &Avatar, tier: Tier) -> bool {
        self.has_asset(avatar, tier.glb())
    }

    /// Absolute path to the GLB carrying `tier`. `Tier::Lod2` resolves to the
    /// plain `nguo_<sex>.glb`.
    pub fn tier_path(&self, avatar: &Avatar, tier: Tier) -> Result<PathBuf, AvatarError> {
        self.asset_path(avatar, tier.glb())
    }

    /// Absolute path to one of the seven shared skin maps.
    pub fn texture(&self, map: SkinMap) -> Result<PathBuf, AvatarError> {
        self.texture_path(self.skin.shared_maps.path(map))
    }

    /// Absolute path to a tone's baked albedo — route B of `INTEGRATION.md` §4.
    /// Pair it with [`BaseColorFactor::WHITE`].
    pub fn tone_albedo(&self, tone: &MonkTone) -> Result<PathBuf, AvatarError> {
        self.texture_path(&tone.baked_albedo)
    }

    fn texture_path(&self, rel: &str) -> Result<PathBuf, AvatarError> {
        let path = self.root.join(rel);
        if path.is_file() {
            Ok(path)
        } else {
            Err(AvatarError::MissingFile {
                root: self.root.clone(),
                file: rel.to_string(),
            })
        }
    }

    /// The avatar's measured girths, converted to millimetres.
    pub fn measurements(&self, avatar: &Avatar) -> Measurements {
        Measurements::from_avatar(avatar)
    }

    /// Load a mesh through `engine-io`. CPU only — no GPU, no device.
    ///
    /// Only [`Asset::is_gltf`] assets can be loaded; a `.blend` or `.fbx` is
    /// rejected by `engine_io` rather than here, and both are absent anyway.
    pub fn load_mesh(
        &self,
        avatar: &Avatar,
        asset: Asset,
    ) -> Result<engine_api::Object, AvatarError> {
        let path = self.asset_path(avatar, asset)?;
        let span = tracing::info_span!("load_avatar_mesh", path = %path.display());
        let _enter = span.enter();
        engine_io::load_mesh(&path).map_err(|source| AvatarError::MeshLoad { path, source })
    }
}

fn read_json<T: serde::de::DeserializeOwned>(root: &Path, rel: &str) -> Result<T, AvatarError> {
    let path = root.join(rel);
    let bytes = std::fs::read(&path).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            AvatarError::MissingFile {
                root: root.to_path_buf(),
                file: rel.to_string(),
            }
        } else {
            AvatarError::Read {
                path: path.clone(),
                source,
            }
        }
    })?;
    serde_json::from_slice(&bytes).map_err(|source| AvatarError::BadJson { path, source })
}

/// `$NGUO_AVATAR_BUNDLE`, else `<repo>/assets/avatars` relative to this crate.
pub fn default_root() -> PathBuf {
    if let Some(p) = std::env::var_os(BUNDLE_ENV) {
        return PathBuf::from(p);
    }
    // ui/crates/nguo-avatar -> repo root.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/avatars")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_bundle_says_where_it_looked() {
        let err = Bundle::open("/definitely/not/a/bundle").unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, AvatarError::BundleNotFound(_)), "{msg}");
        assert!(msg.contains("NGUO_AVATAR_BUNDLE"), "{msg}");
    }

    #[test]
    fn an_absent_asset_names_the_manifest_entry_and_the_path() {
        let err = AvatarError::AssetAbsent {
            avatar: "Nguo Male".into(),
            asset: Asset::GlbLod0.as_str(),
            declared: PathBuf::from("/b/avatars/glb/nguo_male_lod0.glb"),
        };
        let msg = err.to_string();
        assert!(msg.contains("glb.lod0"), "{msg}");
        assert!(msg.contains("nguo_male_lod0.glb"), "{msg}");
        assert!(msg.contains("runtime subset"), "{msg}");
    }
}
