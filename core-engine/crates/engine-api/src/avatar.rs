// SPDX-License-Identifier: GPL-2.0-or-later
//! Avatar collider layer — turn a body mesh into something cloth can collide with,
//! and drape a garment on it in one call.
//!
//! # How the body actually collides
//!
//! [`ColliderSpec`](crate::ColliderSpec) is analytic-only (`Plane`, `Sphere`), so a
//! body mesh cannot become one. The body rides into the [`Scene`] as a **static
//! object** — `is_static` makes `Scene::lower` pin every one of its vertices, giving
//! infinite mass and zero integration — and cloth-versus-body contact runs through
//! the implicit tier's vertex–triangle self-collision over the concatenated mesh.
//! That is the same mechanism `auto_drape` uses; see `docs/AUTO_DRAPE.md`.
//!
//! **Therefore: build with [`Scene::build_implicit`].** The XPBD preview tier
//! ([`Scene::build`]) has no self-collision and the garment will pass straight
//! through the body.
//!
//! # Cost — the whole proxy is not steppable, and decimation would not save it
//!
//! The bundle's collision tier is **15 000 triangles / 10 230 (male) glTF vertices**
//! (7 498 unique positions; glTF splits the rest on UV/normal seams). Because the
//! body rides in as *cloth self-collision geometry*, its own triangles go through the
//! broad phase as if they could collide with each other. Measured on this machine
//! (Apple Silicon, `--release`, dt = 1/120, one 36-vertex patch dropped on the body,
//! `tests/avatar_bundle.rs::collision_proxy_step_cost`):
//!
//! | body in the scene | scene tris | self-collision | s/step |
//! |---|---|---|---|
//! | full proxy, male | 15 050 | on | **9.41** |
//! | full proxy, female | 15 050 | on | **12.53** |
//! | full proxy, either | 15 050 | off | 0.068 |
//! | region crop of the crown | 235 | on | 0.048 |
//!
//! So the body's own triangles cost **138–184×** everything else in the step, while
//! the same body with self-collision off is 0.068 s/step — the body in the linear
//! system is nearly free; only the self-collision pass is not. `drape_scene`'s
//! schedule is 12 substeps × 300 frames = 3 600 steps for a 5 s settle: **9.4 hours**
//! for one drape.
//!
//! **Decimation is not the fix.** The predecessor bundle's proxy was 5 227 triangles
//! and cost 2.27 s/step release, measured the same way in the session that wrote the
//! previous version of this layer. 2.88× the triangles cost 4.15×, an exponent of
//! ~1.35, so reaching an interactive 0.1 s/step from 9.41 needs a 94× speed-up:
//! 94^(1/1.35) ≈ 29× fewer triangles ≈ **520 triangles for a whole body**, which is
//! not a body any more. What fixes it is a **static mesh collider in the solver** — a
//! body-versus-cloth pass that never tests body-versus-body — which lives in
//! `engine-core`/`engine-gpu`, not here.
//!
//! Until then, use [`avatar_collider_near`]: a garment only ever touches part of a
//! body, and a region crop lands in the same 0.05 s/step bracket as everything else.
//!
//! # Loading
//!
//! There is no loader here. `engine_io::load_mesh` already dispatches `.glb` to the
//! glTF reader and returns exactly the [`Object`] these functions take, so loading a
//! proxy is one line at the call site. Paths come from the bundle manifest
//! (`assets/avatars/data/manifest.json`, `avatars.<sex>.files.glb.collision`) — this
//! crate never hardcodes an avatar id or a bundle layout:
//!
//! ```no_run
//! # use engine_api::{avatar, FabricPreset, Object};
//! # /// stands in for `engine_io::load_mesh(path)?`
//! # fn load(_path: &str) -> Object { unimplemented!() }
//! # fn main() -> Result<(), avatar::AvatarError> {
//! let body = load("assets/avatars/avatars/glb/nguo_female_collision.glb");
//! let panel = load("garment.glb");
//! let scene = avatar::drape_on_avatar(
//!     &body,
//!     &panel,
//!     FabricPreset::Cotton,
//!     avatar::GarmentPlacement::DropFromAbove,
//! )?;
//! let solver = scene.build_implicit();
//! # Ok(())
//! # }
//! ```
//!
//! `engine-io` depends on `engine-api`, so taking a dependency the other way would
//! be a cycle. Keeping the loader at the call site avoids it and avoids a second
//! glTF reader.
//!
//! # Units
//!
//! The avatar bundle is authored in **metres**, +Y up, +Z forward, right-handed —
//! identical to glTF and to this engine, so nothing here converts axes or scales.
//! Placement is a pure translation ([`placement_offset`]); a garment authored at a
//! different scale is the author's problem, not something guessed at here.

use crate::auto_drape::{drape_material, drape_scene, PLACEMENT_GAP};
use crate::{Aabb, FabricPreset, Object, Scene};

/// Why a mesh cannot be used as an avatar collider or a garment.
///
/// These are trust-boundary checks: both meshes come off disk, and a corrupt or
/// empty file must produce an actionable error rather than a panic deep in
/// `Scene::lower` or a scene that silently simulates nothing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AvatarError {
    #[error(
        "{role} mesh '{name}' has no vertices — expected a body proxy such as *_collision.glb"
    )]
    Empty { role: &'static str, name: String },

    #[error("{role} mesh '{name}' has {verts} vertices but no triangles — a point cloud has no surface to collide against; re-export it with faces")]
    NoTriangles {
        role: &'static str,
        name: String,
        verts: usize,
    },

    #[error("{role} mesh '{name}' has a non-finite coordinate at vertex {vertex} — the source file is corrupt")]
    NonFinite {
        role: &'static str,
        name: String,
        vertex: usize,
    },

    #[error("{role} mesh '{name}' references vertex {index} but only has {verts} — the source file is corrupt")]
    BadIndex {
        role: &'static str,
        name: String,
        index: u32,
        verts: usize,
    },
}

/// Validate a disk-loaded mesh and return its bounding box.
fn checked_bounds(role: &'static str, mesh: &Object) -> Result<Aabb, AvatarError> {
    let name = || mesh.name.clone();
    let Some(bb) = Aabb::of(&mesh.positions) else {
        return Err(AvatarError::Empty { role, name: name() });
    };
    if mesh.tris.is_empty() {
        return Err(AvatarError::NoTriangles {
            role,
            name: name(),
            verts: mesh.positions.len(),
        });
    }
    if let Some(vertex) = mesh
        .positions
        .iter()
        .position(|p| !p.iter().all(|c| c.is_finite()))
    {
        return Err(AvatarError::NonFinite {
            role,
            name: name(),
            vertex,
        });
    }
    let verts = mesh.positions.len();
    if let Some(&index) = mesh.tris.iter().flatten().find(|&&i| i as usize >= verts) {
        return Err(AvatarError::BadIndex {
            role,
            name: name(),
            index,
            verts,
        });
    }
    Ok(bb)
}

/// Lower a loaded body mesh — the bundle's `*_collision.glb` proxy, or any crop of
/// it — into a **static collider object** with the flags the implicit tier needs.
///
/// Skin, animation clips, normals and UVs are carried through untouched: the proxy
/// shares the render mesh's 65-bone skeleton and animations, so one pose drives both,
/// and glTF round-trips keep them even though the solver ignores them today.
pub fn avatar_collider(mesh: &Object) -> Result<Object, AvatarError> {
    checked_bounds("avatar", mesh)?;
    let mut collider = mesh.clone();
    // `Scene::lower` pins every vertex of a static object → infinite mass, never
    // integrated. Any authored pins are redundant.
    collider.is_static = true;
    collider.pins = Vec::new();
    // `Scene::lower` keeps UVs only if EVERY object supplies them, so a proxy
    // exported without TEXCOORD_0 would silently strip the garment's UVs and with
    // them the anisotropic warp/weft membrane. Zeros are inert on a mesh that is
    // never integrated.
    if collider.uvs.len() != collider.positions.len() {
        tracing::debug!(
            avatar = %collider.name,
            "collision proxy has no UVs; padding so the garment keeps its own"
        );
        collider.uvs = vec![[0.0, 0.0]; collider.positions.len()];
    }
    Ok(collider)
}

/// The part of a body proxy near `region` (grown by `margin` on every side), as a
/// standalone static collider: every triangle with at least one vertex inside is
/// kept whole and reindexed, so the crop has no seam holes along the boundary.
/// UVs, normals and the per-vertex skin are remapped; joints, inverse-bind matrices
/// and animation clips are carried through unchanged, so the crop still poses with
/// the full body.
///
/// This exists because of the measurement in the module docs: the bundle's 15 000-
/// triangle proxy cannot be stepped whole. A sleeve, a bodice or a waistband only
/// ever contacts a fraction of the body, and `region` is normally the garment's
/// [`Aabb`] with a margin for how far it will fall.
///
/// ponytail: an AABB is the whole selector — no capsule, no bone-region lookup.
/// Upgrade to bone-weight selection if a garment ever needs a limb rather than a box.
///
/// **Ceiling: cloth that leaves the region sees no body and will pass through it.**
/// Size `margin` to the fall distance, or pass the whole body's box to opt out.
pub fn avatar_collider_near(
    body: &Object,
    region: Aabb,
    margin: f32,
) -> Result<Object, AvatarError> {
    checked_bounds("avatar", body)?;
    let inside = |i: u32| {
        let p = body.positions[i as usize];
        (0..3).all(|k| p[k] >= region.min[k] - margin && p[k] <= region.max[k] + margin)
    };

    let mut remap = vec![u32::MAX; body.positions.len()];
    let mut crop = Object {
        name: format!("{}_region", body.name),
        positions: Vec::new(),
        tris: Vec::new(),
        material: body.material,
        preset: body.preset,
        pins: Vec::new(),
        is_static: true,
        normals: Vec::new(),
        uvs: Vec::new(),
        skin: body.skin.as_ref().map(|s| crate::rig::Skin {
            joints: s.joints.clone(),
            inverse_bind: s.inverse_bind.clone(),
            vertex_joints: Vec::new(),
            vertex_weights: Vec::new(),
        }),
        animations: body.animations.clone(),
    };
    let n = body.positions.len();
    let (has_normals, has_uvs) = (body.normals.len() == n, body.uvs.len() == n);
    for t in body.tris.iter().filter(|t| t.iter().any(|&i| inside(i))) {
        let mut out = [0u32; 3];
        for (k, &i) in t.iter().enumerate() {
            let slot = &mut remap[i as usize];
            if *slot == u32::MAX {
                *slot = crop.positions.len() as u32;
                crop.positions.push(body.positions[i as usize]);
                if has_normals {
                    crop.normals.push(body.normals[i as usize]);
                }
                if has_uvs {
                    crop.uvs.push(body.uvs[i as usize]);
                }
                if let (Some(dst), Some(src)) = (crop.skin.as_mut(), body.skin.as_ref()) {
                    // A glTF skin's per-vertex arrays are parallel to positions; a
                    // proxy exported without them keeps the joints and gets nothing.
                    if src.vertex_joints.len() == n && src.vertex_weights.len() == n {
                        dst.vertex_joints.push(src.vertex_joints[i as usize]);
                        dst.vertex_weights.push(src.vertex_weights[i as usize]);
                    }
                }
            }
            out[k] = *slot;
        }
        crop.tris.push(out);
    }
    tracing::debug!(
        avatar = %body.name,
        kept_verts = crop.positions.len(),
        kept_tris = crop.tris.len(),
        of_tris = body.tris.len(),
        "avatar collider cropped to region"
    );
    avatar_collider(&crop)
}

/// Where to put the garment relative to the avatar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GarmentPlacement {
    /// Leave the garment's authored coordinates alone — it is already modelled on
    /// this body (a fitted shirt exported in the avatar's own space).
    AsAuthored,
    /// Centre it on the avatar in XZ and drop it from [`PLACEMENT_GAP`] above the
    /// avatar's highest point — `auto_drape`'s rule, for a flat panel or a garment
    /// authored at the origin.
    DropFromAbove,
}

/// The translation that puts `garment` on `avatar`. Never a scale or a rotation:
/// the bundle and glTF share this engine's metres/+Y-up/+Z-forward convention, so
/// there is nothing to convert, and auto-scaling a garment to a stature is grading,
/// not a uniform factor.
pub fn placement_offset(avatar: Aabb, garment: Aabb, placement: GarmentPlacement) -> [f32; 3] {
    match placement {
        GarmentPlacement::AsAuthored => [0.0; 3],
        GarmentPlacement::DropFromAbove => {
            let (a, g) = (avatar.center(), garment.center());
            [
                a[0] - g[0],
                // auto_drape's placement rule: 1 mm above the target's top.
                (avatar.max[1] + PLACEMENT_GAP) - garment.min[1],
                a[2] - g[2],
            ]
        }
    }
}

/// Build a ready-to-drape [`Scene`] from a body proxy, a garment mesh and a fabric.
///
/// The avatar becomes a static collider ([`avatar_collider`]); the garment becomes
/// the dynamic cloth, translated per `placement`, wearing `fabric`'s elastic model
/// with `auto_drape`'s contact overrides. Scheduling, self-collision and the ground
/// plane at the avatar's feet all come from the same assembly `auto_drape` uses, so
/// the two paths cannot drift apart.
///
/// Any pins authored on the garment are kept — that is how you hang it from the
/// shoulders instead of dropping it.
///
/// Step it with [`Scene::build_implicit`]; see the module docs for why.
pub fn drape_on_avatar(
    avatar: &Object,
    garment: &Object,
    fabric: FabricPreset,
    placement: GarmentPlacement,
) -> Result<Scene, AvatarError> {
    let avatar_bb = checked_bounds("avatar", avatar)?;
    let garment_bb = checked_bounds("garment", garment)?;
    let offset = placement_offset(avatar_bb, garment_bb, placement);

    let mut cloth = garment.clone();
    cloth.name = format!("{}_on_{}", garment.name, avatar.name);
    cloth.is_static = false;
    cloth.preset = None; // a named preset would override the contact settings
    cloth.material = drape_material(fabric);
    for p in &mut cloth.positions {
        for k in 0..3 {
            p[k] += offset[k];
        }
    }
    if cloth.uvs.len() != cloth.positions.len() {
        tracing::warn!(
            garment = %garment.name,
            "garment has no UVs; the fabric's anisotropic warp/weft model is inert without them"
        );
    }

    tracing::info!(
        avatar = %avatar.name,
        garment = %garment.name,
        avatar_verts = avatar.positions.len(),
        garment_verts = garment.positions.len(),
        ?offset,
        "avatar drape scene built"
    );
    Ok(drape_scene(
        avatar_collider(avatar)?,
        cloth,
        avatar_bb.min[1],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_drape::cloth_plane;
    use crate::{ColliderSpec, Material};

    /// A closed axis-aligned box — a stand-in body with real surface geometry.
    pub(crate) fn box_mesh(name: &str, min: [f32; 3], max: [f32; 3]) -> Object {
        let positions = vec![
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], min[1], max[2]],
            [min[0], min[1], max[2]],
            [min[0], max[1], min[2]],
            [max[0], max[1], min[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ];
        let tris = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [1, 2, 6],
            [1, 6, 5],
        ];
        Object {
            name: name.into(),
            positions,
            tris,
            material: Material::default(),
            preset: None,
            pins: Vec::new(),
            is_static: false,
            normals: Vec::new(),
            uvs: Vec::new(),
            skin: None,
            animations: Vec::new(),
        }
    }

    /// A flat n×n panel authored at the origin, spanning [-s/2, s/2] in XZ at y=0.
    pub(crate) fn panel(name: &str, s: f32, n: usize) -> Object {
        let bb = Aabb {
            min: [-s * 0.5, 0.0, -s * 0.5],
            max: [s * 0.5, 0.0, s * 0.5],
        };
        let (positions, tris, uvs) = cloth_plane(bb, n, 0.0, 0.0);
        Object {
            name: name.into(),
            positions,
            tris,
            material: Material::default(),
            preset: None,
            pins: Vec::new(),
            is_static: false,
            normals: Vec::new(),
            uvs,
            skin: None,
            animations: Vec::new(),
        }
    }

    /// A 1.7 m "avatar" standing on y=0, off-centre in XZ so a centring bug shows.
    fn avatar() -> Object {
        box_mesh("nguo_female", [0.7, 0.0, -0.4], [1.0, 1.7, -0.1])
    }

    #[test]
    fn collider_is_static_unpinned_and_uv_parallel() {
        let mut body = avatar();
        body.pins = vec![0, 1]; // authored pins are redundant on a static object
        let c = avatar_collider(&body).unwrap();
        assert!(
            c.is_static,
            "the body must be kinematic for the implicit tier"
        );
        assert!(c.pins.is_empty());
        // Padded so `Scene::lower`'s parallel-or-empty UV rule keeps the garment's.
        assert_eq!(c.uvs.len(), c.positions.len());
        // Geometry is untouched.
        assert_eq!(c.positions, body.positions);
        assert_eq!(c.tris, body.tris);
    }

    #[test]
    fn region_crop_keeps_boundary_triangles_reindexed_and_rigged() {
        let mut body = avatar(); // x 0.7..1.0, y 0..1.7, z -0.4..-0.1
        body.uvs = vec![[0.5, 0.5]; body.positions.len()];
        body.normals = vec![[0.0, 1.0, 0.0]; body.positions.len()];
        body.skin = Some(crate::rig::Skin {
            joints: vec![crate::rig::Joint {
                name: "hips".into(),
                parent: None,
                local: [[0.0; 4]; 4],
            }],
            inverse_bind: vec![[[0.0; 4]; 4]],
            vertex_joints: (0..body.positions.len() as u32)
                .map(|i| [i, 0, 0, 0])
                .collect(),
            vertex_weights: vec![[1.0, 0.0, 0.0, 0.0]; body.positions.len()],
        });

        // Top 0.1 m: only the 4 upper corners are inside, but every side face touches
        // them, so the crop keeps whole triangles that reach below the region.
        let top = Aabb {
            min: [0.7, 1.6, -0.4],
            max: [1.0, 1.7, -0.1],
        };
        let c = avatar_collider_near(&body, top, 0.0).unwrap();
        assert!(c.is_static && c.pins.is_empty());
        assert!(
            c.tris.len() < body.tris.len() && !c.tris.is_empty(),
            "kept {} of {} triangles",
            c.tris.len(),
            body.tris.len()
        );
        // Reindexing is sound: no dangling index, no duplicated vertex.
        assert!(c
            .tris
            .iter()
            .flatten()
            .all(|&i| (i as usize) < c.positions.len()));
        assert_eq!(
            c.positions.len(),
            c.tris
                .iter()
                .flatten()
                .collect::<std::collections::HashSet<_>>()
                .len()
        );
        // Parallel attributes stay parallel, and the rig survives so the crop can
        // still be posed by the avatar's own animation clips.
        assert_eq!(c.uvs.len(), c.positions.len());
        assert_eq!(c.normals.len(), c.positions.len());
        let skin = c.skin.as_ref().expect("crop lost the skin");
        assert_eq!(skin.joints.len(), 1);
        assert_eq!(skin.vertex_joints.len(), c.positions.len());
        assert_eq!(skin.vertex_weights.len(), c.positions.len());
        // Every kept vertex kept ITS OWN weights: vertex_joints[k] == [original, ..].
        for (k, p) in c.positions.iter().enumerate() {
            let orig = body.positions.iter().position(|q| q == p).unwrap() as u32;
            assert_eq!(skin.vertex_joints[k], [orig, 0, 0, 0], "vertex {k} remap");
        }

        // The margin widens the region: the whole box comes back.
        let all = avatar_collider_near(&body, top, 2.0).unwrap();
        assert_eq!(all.tris.len(), body.tris.len());
        assert_eq!(all.positions.len(), body.positions.len());
    }

    #[test]
    fn region_crop_degenerate_inputs_error_cleanly() {
        let body = avatar();
        // A region nowhere near the body keeps nothing — an empty collider would
        // silently simulate a garment against no body at all.
        let elsewhere = Aabb {
            min: [10.0, 10.0, 10.0],
            max: [11.0, 11.0, 11.0],
        };
        assert_eq!(
            avatar_collider_near(&body, elsewhere, 0.0).unwrap_err(),
            AvatarError::Empty {
                role: "avatar",
                name: "nguo_female_region".into()
            }
        );
        // An empty body is rejected before any cropping, naming the body not the crop.
        let mut empty = body.clone();
        empty.positions.clear();
        empty.tris.clear();
        assert_eq!(
            avatar_collider_near(&empty, elsewhere, 0.0).unwrap_err(),
            AvatarError::Empty {
                role: "avatar",
                name: "nguo_female".into()
            }
        );
        // A corrupt index is caught before it can be dereferenced by the crop.
        let mut bad = body.clone();
        bad.tris[2] = [0, 1, 99];
        assert!(matches!(
            avatar_collider_near(&bad, elsewhere, 0.0).unwrap_err(),
            AvatarError::BadIndex { index: 99, .. }
        ));
    }

    #[test]
    fn collider_keeps_authored_uvs() {
        let mut body = avatar();
        body.uvs = vec![[0.25, 0.75]; body.positions.len()];
        assert_eq!(avatar_collider(&body).unwrap().uvs, body.uvs);
    }

    #[test]
    fn drop_from_above_centres_on_the_avatar_and_clears_its_head() {
        let (body, garment) = (avatar(), panel("tee", 0.5, 6));
        let (abb, gbb) = (
            Aabb::of(&body.positions).unwrap(),
            Aabb::of(&garment.positions).unwrap(),
        );
        let d = placement_offset(abb, gbb, GarmentPlacement::DropFromAbove);
        // Centred in XZ on the avatar (which sits at x=0.85, z=-0.25, not the origin).
        assert!((d[0] - 0.85).abs() < 1e-6, "{d:?}");
        assert!((d[2] + 0.25).abs() < 1e-6, "{d:?}");
        // 1 mm above the head, exactly auto_drape's rule.
        assert!((d[1] - (1.7 + PLACEMENT_GAP)).abs() < 1e-6, "{d:?}");

        let scene = drape_on_avatar(
            &body,
            &garment,
            FabricPreset::Cotton,
            GarmentPlacement::DropFromAbove,
        )
        .unwrap();
        let placed = Aabb::of(&scene.objects[1].positions).unwrap();
        assert!((placed.center()[0] - abb.center()[0]).abs() < 1e-6);
        assert!((placed.center()[2] - abb.center()[2]).abs() < 1e-6);
        assert!((placed.min[1] - (abb.max[1] + PLACEMENT_GAP)).abs() < 1e-6);
        for p in &scene.objects[1].positions {
            assert!(
                p[1] > abb.max[1],
                "garment vertex not above the head: {p:?}"
            );
        }
    }

    #[test]
    fn placement_is_a_pure_translation_never_a_scale() {
        // The bundle and this engine share metres/+Y-up, so nothing is ever rescaled.
        // Extents must survive placement bit-for-bit.
        let (body, garment) = (avatar(), panel("tee", 0.5, 6));
        let before = Aabb::of(&garment.positions).unwrap().extent();
        for placement in [
            GarmentPlacement::AsAuthored,
            GarmentPlacement::DropFromAbove,
        ] {
            let scene = drape_on_avatar(&body, &garment, FabricPreset::Silk, placement).unwrap();
            let after = Aabb::of(&scene.objects[1].positions).unwrap().extent();
            assert_eq!(before, after, "{placement:?} rescaled the garment");
        }
    }

    #[test]
    fn as_authored_leaves_the_garment_where_it_was() {
        let (body, garment) = (avatar(), panel("tee", 0.5, 4));
        assert_eq!(
            placement_offset(
                Aabb::of(&body.positions).unwrap(),
                Aabb::of(&garment.positions).unwrap(),
                GarmentPlacement::AsAuthored,
            ),
            [0.0; 3]
        );
        let scene = drape_on_avatar(
            &body,
            &garment,
            FabricPreset::Cotton,
            GarmentPlacement::AsAuthored,
        )
        .unwrap();
        assert_eq!(scene.objects[1].positions, garment.positions);
    }

    #[test]
    fn degenerate_inputs_error_cleanly() {
        let (body, garment) = (avatar(), panel("tee", 0.5, 4));

        // Empty mesh — no vertices at all.
        let mut empty = body.clone();
        empty.positions.clear();
        empty.tris.clear();
        assert_eq!(
            avatar_collider(&empty).unwrap_err(),
            AvatarError::Empty {
                role: "avatar",
                name: "nguo_female".into()
            }
        );

        // Single vertex — a point cloud has no surface.
        let mut point = body.clone();
        point.positions.truncate(1);
        point.tris.clear();
        assert_eq!(
            avatar_collider(&point).unwrap_err(),
            AvatarError::NoTriangles {
                role: "avatar",
                name: "nguo_female".into(),
                verts: 1
            }
        );

        // Corrupt coordinates.
        let mut nan = body.clone();
        nan.positions[3][1] = f32::NAN;
        assert!(matches!(
            avatar_collider(&nan).unwrap_err(),
            AvatarError::NonFinite { vertex: 3, .. }
        ));
        let mut inf = body.clone();
        inf.positions[0][2] = f32::INFINITY;
        assert!(matches!(
            avatar_collider(&inf).unwrap_err(),
            AvatarError::NonFinite { vertex: 0, .. }
        ));

        // Out-of-range triangle index would panic inside the solver's mesh build.
        let mut bad = body.clone();
        bad.tris[2] = [0, 1, 99];
        assert!(matches!(
            avatar_collider(&bad).unwrap_err(),
            AvatarError::BadIndex {
                index: 99,
                verts: 8,
                ..
            }
        ));

        // The garment is validated too, and says so — the role must not read "avatar".
        let mut empty_garment = garment.clone();
        empty_garment.positions.clear();
        let err = drape_on_avatar(
            &body,
            &empty_garment,
            FabricPreset::Cotton,
            GarmentPlacement::DropFromAbove,
        )
        .unwrap_err();
        assert_eq!(
            err,
            AvatarError::Empty {
                role: "garment",
                name: "tee".into()
            }
        );
        assert!(err.to_string().contains("garment mesh 'tee'"), "{err}");
    }

    #[test]
    fn scene_wiring_matches_auto_drape() {
        // Proves the shared assembly is actually shared: same scheduling, same
        // self-collision flag, same contact overrides, same ground plane rule.
        let (body, garment) = (avatar(), panel("tee", 0.5, 6));
        let scene = drape_on_avatar(
            &body,
            &garment,
            FabricPreset::Cotton,
            GarmentPlacement::DropFromAbove,
        )
        .unwrap();
        let reference =
            crate::auto_drape(&body, crate::DrapePreset::Normal, FabricPreset::Cotton).unwrap();

        assert_eq!(scene.config.substeps, reference.config.substeps);
        assert_eq!(scene.config.frames, reference.config.frames);
        assert!(scene.config.self_collision);
        assert_eq!(scene.config.floor_y, Some(0.0)); // the avatar's feet
        assert_eq!(scene.colliders.len(), 1);
        assert!(matches!(
            scene.colliders[0],
            ColliderSpec::Plane { point, normal } if point[1] == 0.0 && normal == [0.0, 1.0, 0.0]
        ));

        // Static body first so `Scene::lower` picks the garment's material.
        assert_eq!(scene.objects.len(), 2);
        assert!(scene.objects[0].is_static);
        assert!(!scene.objects[1].is_static);
        assert_eq!(scene.objects[1].name, "tee_on_nguo_female");
        let (m, r) = (&scene.objects[1].material, &reference.objects[1].material);
        assert_eq!(m.contact_gap, r.contact_gap);
        assert_eq!(m.contact_offset, r.contact_offset);
        assert_eq!(m.friction, r.friction);
        assert_eq!(m.membrane_warp, r.membrane_warp);
        assert_eq!(m.membrane_weft, r.membrane_weft);
        assert_eq!(m.bend, r.bend);
        assert_eq!(m.density, r.density);
        assert_eq!(scene.objects[1].preset, None);
        // Both objects carry UVs, so the anisotropic membrane survives `lower`.
        let solver = scene.build_implicit();
        assert_eq!(solver.state.uvs.len(), solver.state.vertex_count());
    }

    #[test]
    fn stepping_an_avatar_scene_is_deterministic() {
        // Ungated twin of the bundle test's determinism check, so this invariant is
        // covered on machines without the 223 MB asset pack.
        let (body, garment) = (avatar(), panel("tee", 0.3, 5));
        let run = || {
            let mut scene = drape_on_avatar(
                &body,
                &garment,
                FabricPreset::Cotton,
                GarmentPlacement::DropFromAbove,
            )
            .unwrap();
            scene.config.dt = 1.0 / 120.0;
            scene.config.iterations = 3;
            let mut solver = scene.build_implicit();
            solver.params.contact_kappa = 1.0e4;
            solver.params.contact_dhat = 1.0e-2;
            for _ in 0..8 {
                solver.step(&engine_core::linsolve::CgCpu::new());
            }
            solver
                .state
                .positions
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect::<Vec<_>>()
        };
        let (a, b) = (run(), run());
        assert!(a.iter().all(|p| p.iter().all(|c| c.is_finite())));
        // Floor: the cloth must have actually moved, or equality is vacuous.
        let start = &drape_on_avatar(
            &body,
            &garment,
            FabricPreset::Cotton,
            GarmentPlacement::DropFromAbove,
        )
        .unwrap()
        .objects[1]
            .positions
            .clone();
        let base = body.positions.len();
        assert!(
            a[base..]
                .iter()
                .zip(start.iter())
                .any(|(p, q)| (p[1] - q[1]).abs() > 1e-6),
            "no cloth vertex moved — determinism would be vacuous"
        );
        assert_eq!(a, b, "same inputs must give identical positions");
    }

    #[test]
    fn garment_pins_are_kept() {
        // Pins are how a garment hangs from the shoulders rather than dropping.
        let (body, mut garment) = (avatar(), panel("tee", 0.5, 4));
        garment.pins = vec![0, 3];
        let scene = drape_on_avatar(
            &body,
            &garment,
            FabricPreset::Cotton,
            GarmentPlacement::DropFromAbove,
        )
        .unwrap();
        assert_eq!(scene.objects[1].pins, vec![0, 3]);
        let solver = scene.build_implicit();
        let base = scene.objects[0].positions.len();
        assert!(solver.state.is_pinned(base));
        assert!(solver.state.is_pinned(base + 3));
    }
}
