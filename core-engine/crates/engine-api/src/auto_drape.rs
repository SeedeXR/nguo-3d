// SPDX-License-Identifier: GPL-2.0-or-later
//! Auto-drape — one call turns "this collider mesh" plus "this fabric" into a
//! ready-to-simulate [`Scene`]: a cloth plane sized and placed over the target's
//! bounding box, contact margins set, substeps chosen, settle length estimated.
//! No manual sim setup, no keyframing.
//!
//! # Provenance and licensing
//!
//! The *product behaviour* — one-click "drop cloth over any object" — is modelled
//! on the **ClothDrop-Reborn** Blender add-on (**GPL-3.0-or-later**), read as a
//! design reference only. Its source is not vendored or redistributed here, and
//! no code was copied; this tree stays GPL-2.0-or-later. Every number carried
//! over cites its source `file:line`. `docs/AUTO_DRAPE.md` holds the full mapping
//! table, including the ClothDrop parameters deliberately **not** carried and why.
//!
//! # Axis convention
//!
//! Blender is **Z-up** and ClothDrop drops along −Z. This engine is **Y-up**
//! (`SolverConfig::gravity` defaults to `[0,-9.8,0]`, `params.rs`). Every citation
//! below is transposed z→y, so ClothDrop's XY footprint is our XZ.

use crate::{ColliderSpec, FabricPreset, Material, Object, Scene, SolverConfig};

/// Blender cloth "Quality Steps" — solver substeps per frame. ClothDrop pins this
/// to 12 for every preset (`utils.py:307` `cs.quality = 12`). Maps 1:1 onto our
/// [`SolverConfig::substeps`], which is the same quantity (default 8).
pub const CLOTHDROP_QUALITY: u32 = 12;

/// Frames ClothDrop bakes before freezing the result (`properties.py:15`,
/// `bakeframe` default 120, min 0, max 250; `utils.py:11` clamps to >= 1).
pub const CLOTHDROP_BAKE_FRAMES: u32 = 120;

/// Blender's default scene frame rate. ClothDrop never sets `scene.render.fps`,
/// so its 120-frame bake is 120/24 = 5.0 s of simulated time. Carried as a
/// *duration* rather than a frame count because our `fps` default is 60, not 24.
pub const BLENDER_DEFAULT_FPS: f64 = 24.0;

/// Collision margin, metres. ClothDrop sets `thickness_outer = 0.001` and
/// `thickness_inner = 0.001` on the collision object (`utils.py:200-201`, and
/// again at `utils.py:256` / `operators.py:114`) and `self_distance_min = 0.001`
/// for cloth self-collision (`utils.py:309`). Metres are metres, so this one
/// transfers without reinterpretation.
pub const CONTACT_MARGIN: f32 = 1.0e-3;

/// Vertical clearance between the cloth and the highest point of the target.
/// ClothDrop drops the cloth object to `best_location.z + 0.001` — 1 mm above the
/// highest surface hit by its downward raycast grid (`operators.py:247`, and the
/// same constant at `operators.py:80`). We place at `target_max_y + this`.
pub const PLACEMENT_GAP: f32 = 1.0e-3;

/// Coulomb friction for the "High Friction" default. ClothDrop's `high_friction`
/// property defaults to `True` (`properties.py:16`, "Helps prevent sliding off
/// surface") and then sets `cloth_friction = 80.0` (`utils.py:196`, `utils.py:253`,
/// `operators.py:111`).
///
/// **Divergence:** Blender's cloth-friction slider is a dimensionless 0..80 factor,
/// not a Coulomb coefficient. ClothDrop uses the maximum of that range, so we carry
/// the *ratio* 80/80 = 1.0. mu = 1.0 is deliberately high — that is the add-on's
/// intent (stop the cloth sliding off), not a measured fabric value.
///
/// Note this field is currently inert: `Material::friction` is marked
/// implicit-tier (`material.rs`) and neither `Scene::build` nor
/// `Scene::build_implicit` reads it yet. Setting it is forward-compatible only.
pub const HIGH_FRICTION_MU: f32 = 1.0;

/// Cloth overhang past the target, as a fraction of the target's largest extent.
///
/// **This is ours, not ClothDrop's.** ClothDrop never sizes a cloth automatically:
/// the drawn-rectangle path takes the size straight from the user's mouse drag
/// (`operators.py:41-47,86`) and the apply path re-uses whatever mesh the user
/// already selected, moving only its height (`operators.py:247`). Auto-sizing is
/// the part we had to invent, so it is a named constant rather than a citation.
///
/// Scaled by the largest of all three extents (not just XZ) because the overhang
/// has to be long enough to fall *past* the target's sides — a tall, narrow object
/// needs more cloth than its footprint suggests.
pub const OVERHANG_FRAC: f32 = 0.25;

/// Axis-aligned bounding box. Y is up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// Bounds of a point set, or `None` when empty (an empty mesh has no box, and
    /// there is no sensible drape over nothing).
    pub fn of(points: &[[f32; 3]]) -> Option<Self> {
        let (&first, rest) = points.split_first()?;
        let mut bb = Aabb {
            min: first,
            max: first,
        };
        for p in rest {
            for (k, &c) in p.iter().enumerate() {
                bb.min[k] = bb.min[k].min(c);
                bb.max[k] = bb.max[k].max(c);
            }
        }
        Some(bb)
    }

    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    pub fn center(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Largest of the three extents.
    pub fn max_extent(&self) -> f32 {
        let e = self.extent();
        e[0].max(e[1]).max(e[2])
    }
}

/// ClothDrop's four presets (`presets.py:1-35`, names and blurbs `presets.py:39-44`).
///
/// These are **quality/stiffness** presets, not fabrics — an axis orthogonal to our
/// [`FabricPreset`], which is a physical material model. Only the resolution knob
/// survives translation with its meaning intact; see [`DrapePreset::subdivision`]
/// and `docs/AUTO_DRAPE.md` for the parameters that do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrapePreset {
    /// "A simple preset for speed." (`presets.py:40`)
    SimpleFast,
    /// "A sharper preset without smoothing." (`presets.py:41`)
    Sharp,
    /// "A good and efficient preset." (`presets.py:42`)
    Normal,
    /// "Does not bend much, good for paper or metals." (`presets.py:43`)
    Hard,
}

impl DrapePreset {
    /// ClothDrop's `subdivision` = `number_cuts` passed to `mesh.subdivide` on a
    /// single-quad plane (`utils.py:281`). Values: `presets.py:5,13,21,29`.
    ///
    /// This is the one ClothDrop number that is dimensionless and therefore ports
    /// exactly: N cuts on one quad yields N+1 quads and **N+2 vertices** per side.
    pub fn subdivision(self) -> u32 {
        match self {
            DrapePreset::SimpleFast => 12, // presets.py:5
            DrapePreset::Sharp => 22,      // presets.py:13
            DrapePreset::Normal => 22,     // presets.py:21
            DrapePreset::Hard => 6,        // presets.py:29
        }
    }

    /// Vertices per side of the generated cloth plane: `subdivision + 2`.
    pub fn verts_per_side(self) -> usize {
        self.subdivision() as usize + 2
    }

    /// The fabric whose behaviour best matches this preset's stated intent.
    ///
    /// A judgement call, not a ClothDrop number — ClothDrop has no fabric concept.
    /// `Hard` is "good for paper or metals" (`presets.py:43`), which is the stiffest
    /// thing we model; `Sharp` wants crisp folds. Callers pass their own fabric to
    /// [`auto_drape`]; this is only the suggested default.
    pub fn fabric(self) -> FabricPreset {
        match self {
            DrapePreset::SimpleFast => FabricPreset::Cotton,
            DrapePreset::Sharp => FabricPreset::Denim,
            DrapePreset::Normal => FabricPreset::Cotton,
            DrapePreset::Hard => FabricPreset::Leather,
        }
    }
}

/// Frames needed to settle at `fps`, preserving ClothDrop's 5.0 s of simulated
/// time (`CLOTHDROP_BAKE_FRAMES` / `BLENDER_DEFAULT_FPS`).
///
/// **This is an estimate, not a convergence test.** ClothDrop does not detect
/// settling either: it bakes a fixed frame range and freezes the mesh at the last
/// frame (`utils.py:20-26`, `utils.py:31-32`). Nothing in its 902 lines measures
/// residual velocity. We keep the same fixed-budget contract rather than invent a
/// criterion the reference does not have.
pub fn settle_frames(fps: f64) -> i32 {
    (CLOTHDROP_BAKE_FRAMES as f64 / BLENDER_DEFAULT_FPS * fps).round() as i32
}

/// A flat, triangulated, UV'd cloth plane spanning `over` in XZ (grown by
/// `margin` on every side) at height `y`, with `n` vertices per side.
///
/// Returns `(positions, tris, uvs)`. Winding and the two-triangles-per-quad split
/// match the grid idiom already used across the crate (`scene.rs`, `regression.rs`).
/// UVs are generated because our fabric presets are anisotropic — membrane
/// warp/weft and weft strain limiting only mean anything with material-space UVs
/// (`material.rs`, consumed via `Scene::lower` into `SimState::uvs`).
#[allow(clippy::type_complexity)]
pub fn cloth_plane(
    over: Aabb,
    n: usize,
    margin: f32,
    y: f32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 2]>) {
    let n = n.max(2);
    let (x0, x1) = (over.min[0] - margin, over.max[0] + margin);
    let (z0, z1) = (over.min[2] - margin, over.max[2] + margin);
    let step = (n - 1) as f32;

    let mut positions = Vec::with_capacity(n * n);
    let mut uvs = Vec::with_capacity(n * n);
    for j in 0..n {
        let tz = j as f32 / step;
        for i in 0..n {
            let tx = i as f32 / step;
            positions.push([x0 + (x1 - x0) * tx, y, z0 + (z1 - z0) * tz]);
            uvs.push([tx, tz]);
        }
    }

    let mut tris = Vec::with_capacity((n - 1) * (n - 1) * 2);
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let v = (j * n + i) as u32;
            let r = v + n as u32;
            tris.push([v, v + 1, r]);
            tris.push([r, v + 1, r + 1]);
        }
    }
    (positions, tris, uvs)
}

/// `fabric`'s elastic model with ClothDrop's contact overrides applied.
///
/// Returned as an inline [`Material`] rather than `Object::preset = Some(fabric)`
/// because `Scene::lower` lets a named preset override the whole material, which
/// would discard these overrides.
pub(crate) fn drape_material(fabric: FabricPreset) -> Material {
    let mut material = fabric.material();
    material.contact_gap = CONTACT_MARGIN;
    material.contact_offset = CONTACT_MARGIN;
    material.friction = HIGH_FRICTION_MU;
    material
}

/// Assemble the standard drape [`Scene`]: `target` forced static and placed
/// **first** (`Scene::lower` takes its material from the last *dynamic* object, so
/// the cloth must come second), ClothDrop's scheduling and self-collision, and a
/// ground plane at `floor_y` so cloth that slides off lands instead of falling
/// forever.
///
/// Shared by [`auto_drape`] and [`crate::avatar::drape_on_avatar`]; `target` is
/// taken by value because both callers already own a clone (these meshes run to
/// thousands of vertices).
pub(crate) fn drape_scene(mut target: Object, cloth: Object, floor_y: f32) -> Scene {
    target.is_static = true;
    target.pins = Vec::new(); // every vertex is already kinematic

    let config = SolverConfig {
        substeps: CLOTHDROP_QUALITY,
        self_collision: true, // utils.py:308 `collision.use_self_collision = True`
        floor_y: Some(floor_y),
        ..Default::default()
    };
    Scene {
        config: SolverConfig {
            frames: settle_frames(config.fps),
            ..config
        },
        objects: vec![target, cloth],
        colliders: vec![ColliderSpec::Plane {
            point: [0.0, floor_y, 0.0],
            normal: [0.0, 1.0, 0.0],
        }],
    }
}

/// Build a ready-to-simulate drape of `fabric` over `target`, using `preset` for
/// mesh resolution.
///
/// The returned [`Scene`] holds the target as a **static** object (infinite mass,
/// never integrated) plus the cloth, with a ground plane collider at the target's
/// base. Cloth-versus-target contact runs through the implicit tier's
/// self-collision over the concatenated mesh — `ColliderSpec` has only `Plane` and
/// `Sphere` (`scene.rs`), so an arbitrary triangle target cannot become an analytic
/// collider. Use [`Scene::build_implicit`]; the XPBD preview tier
/// ([`Scene::build`]) has no self-collision and will let the cloth pass through.
///
/// `None` when `target` has no vertices.
pub fn auto_drape(target: &Object, preset: DrapePreset, fabric: FabricPreset) -> Option<Scene> {
    let bb = Aabb::of(&target.positions)?;
    let margin = OVERHANG_FRAC * bb.max_extent();
    // 1 mm above the target's highest point — ClothDrop's placement rule
    // (operators.py:247), transposed z→y.
    let y = bb.max[1] + PLACEMENT_GAP;
    let (positions, tris, uvs) = cloth_plane(bb, preset.verts_per_side(), margin, y);

    let cloth = Object {
        name: format!("{}_cloth", target.name),
        positions,
        tris,
        material: drape_material(fabric),
        preset: None,
        pins: Vec::new(),
        is_static: false,
        normals: Vec::new(),
        uvs,
        skin: None,
        animations: Vec::new(),
    };
    Some(drape_scene(target.clone(), cloth, bb.min[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube spanning [0,1]^3, as an authored target mesh (8 corners is enough
    /// — only the bounding box is consulted).
    fn box_target(min: [f32; 3], max: [f32; 3]) -> Object {
        let mut positions = Vec::new();
        for &x in &[min[0], max[0]] {
            for &y in &[min[1], max[1]] {
                for &z in &[min[2], max[2]] {
                    positions.push([x, y, z]);
                }
            }
        }
        Object {
            name: "target".into(),
            positions,
            tris: vec![[0, 1, 2], [1, 3, 2]],
            material: Default::default(),
            preset: None,
            pins: Vec::new(),
            is_static: false,
            normals: Vec::new(),
            uvs: Vec::new(),
            skin: None,
            animations: Vec::new(),
        }
    }

    #[test]
    fn aabb_of_empty_is_none_and_of_points_is_tight() {
        assert_eq!(Aabb::of(&[]), None);
        let bb = Aabb::of(&[[-1.0, 2.0, 0.5], [3.0, -4.0, 0.5]]).unwrap();
        assert_eq!(bb.min, [-1.0, -4.0, 0.5]);
        assert_eq!(bb.max, [3.0, 2.0, 0.5]);
        assert_eq!(bb.extent(), [4.0, 6.0, 0.0]);
        assert_eq!(bb.center(), [1.0, -1.0, 0.5]);
        assert_eq!(bb.max_extent(), 6.0);
    }

    #[test]
    fn cloth_plane_covers_target_xz_with_margin_and_sits_above_it() {
        let bb = Aabb::of(&box_target([-0.5, 0.0, -0.5], [0.5, 2.0, 0.5]).positions).unwrap();
        let margin = OVERHANG_FRAC * bb.max_extent(); // 0.25 * 2.0 = 0.5
        let y = bb.max[1] + PLACEMENT_GAP;
        let (pos, tris, uvs) = cloth_plane(bb, 8, margin, y);

        // Fully covers the target's XZ extent, with margin on every side.
        let (mut x0, mut x1, mut z0, mut z1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in &pos {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            z0 = z0.min(p[2]);
            z1 = z1.max(p[2]);
            // Starts strictly above the target's max Y.
            assert!(p[1] > bb.max[1], "cloth vertex not above target: {p:?}");
            assert!(p[1].is_finite());
        }
        assert!(x0 <= bb.min[0] - margin + 1e-6, "no -X margin: {x0}");
        assert!(x1 >= bb.max[0] + margin - 1e-6, "no +X margin: {x1}");
        assert!(z0 <= bb.min[2] - margin + 1e-6, "no -Z margin: {z0}");
        assert!(z1 >= bb.max[2] + margin - 1e-6, "no +Z margin: {z1}");

        // Grid topology: n*n verts, (n-1)^2*2 tris, every index in range, UVs parallel.
        assert_eq!(pos.len(), 8 * 8);
        assert_eq!(tris.len(), 7 * 7 * 2);
        assert_eq!(uvs.len(), pos.len());
        assert!(tris.iter().flatten().all(|&v| (v as usize) < pos.len()));
        // UVs span the full unit square (needed by the anisotropic membrane model).
        assert_eq!(uvs[0], [0.0, 0.0]);
        assert_eq!(uvs[uvs.len() - 1], [1.0, 1.0]);
    }

    #[test]
    fn plane_is_flat_and_degenerate_resolution_is_clamped() {
        let bb = Aabb::of(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).unwrap();
        let (pos, tris, _) = cloth_plane(bb, 0, 0.1, 5.0);
        // n is clamped to 2 → a single quad, not an empty or 1-vertex mesh.
        assert_eq!(pos.len(), 4);
        assert_eq!(tris.len(), 2);
        assert!(pos.iter().all(|p| p[1] == 5.0), "plane must be flat");
    }

    #[test]
    fn preset_resolution_matches_clothdrop_subdivision() {
        // N cuts on one quad → N+2 vertices per side (presets.py:5,13,21,29).
        assert_eq!(DrapePreset::SimpleFast.subdivision(), 12);
        assert_eq!(DrapePreset::SimpleFast.verts_per_side(), 14);
        assert_eq!(DrapePreset::Hard.verts_per_side(), 8);
        assert_eq!(DrapePreset::Sharp.verts_per_side(), 24);
        // Sharp and Normal share a subdivision level in ClothDrop; they differ only
        // in parameters we do not carry (subsurf/wrinkles/folds/weight).
        assert_eq!(
            DrapePreset::Sharp.subdivision(),
            DrapePreset::Normal.subdivision()
        );
        // Hard is the coarsest, Sharp/Normal the finest.
        assert!(DrapePreset::Hard.verts_per_side() < DrapePreset::SimpleFast.verts_per_side());
        assert!(DrapePreset::SimpleFast.verts_per_side() < DrapePreset::Sharp.verts_per_side());
    }

    #[test]
    fn settle_frames_preserves_clothdrop_five_seconds() {
        // 120 frames at Blender's default 24 fps = 5 s; at our 60 fps that is 300.
        assert_eq!(settle_frames(24.0), 120);
        assert_eq!(settle_frames(60.0), 300);
        assert_eq!(settle_frames(SolverConfig::default().fps), 300);
    }

    #[test]
    fn auto_drape_wires_a_complete_scene() {
        let target = box_target([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5]);
        let scene = auto_drape(&target, DrapePreset::Normal, FabricPreset::Cotton).unwrap();

        // Target is static and comes first; cloth is dynamic and comes second, so
        // `Scene::lower` picks the cloth's material.
        assert_eq!(scene.objects.len(), 2);
        assert!(scene.objects[0].is_static);
        assert!(!scene.objects[1].is_static);
        assert_eq!(scene.objects[1].name, "target_cloth");

        // Cloth resolution follows the preset, and UVs are parallel to positions.
        let n = DrapePreset::Normal.verts_per_side();
        assert_eq!(scene.objects[1].positions.len(), n * n);
        assert_eq!(scene.objects[1].uvs.len(), n * n);

        // ClothDrop-derived scheduling and contact settings.
        assert_eq!(scene.config.substeps, CLOTHDROP_QUALITY);
        assert!(scene.config.self_collision);
        assert_eq!(scene.config.frames, settle_frames(scene.config.fps));
        assert_eq!(scene.config.floor_y, Some(0.0));
        assert_eq!(scene.objects[1].material.contact_gap, CONTACT_MARGIN);
        assert_eq!(scene.objects[1].material.contact_offset, CONTACT_MARGIN);
        assert_eq!(scene.objects[1].material.friction, HIGH_FRICTION_MU);

        // The fabric's elastic model survives the contact overrides.
        let cotton = FabricPreset::Cotton.material();
        assert_eq!(
            scene.objects[1].material.membrane_warp,
            cotton.membrane_warp
        );
        assert_eq!(scene.objects[1].material.bend, cotton.bend);
        // `preset` must stay None or `lower` would discard the contact overrides.
        assert_eq!(scene.objects[1].preset, None);

        // A ground plane collider sits at the target's base.
        assert_eq!(scene.colliders.len(), 1);
        match scene.colliders[0] {
            ColliderSpec::Plane { point, normal } => {
                assert_eq!(point[1], 0.0);
                assert_eq!(normal, [0.0, 1.0, 0.0]);
            }
            other => panic!("expected a ground plane, got {other:?}"),
        }
    }

    #[test]
    fn auto_drape_rejects_an_empty_target() {
        let mut empty = box_target([0.0; 3], [1.0; 3]);
        empty.positions.clear();
        assert!(auto_drape(&empty, DrapePreset::Normal, FabricPreset::Cotton).is_none());
    }

    #[test]
    fn overhang_scales_with_the_tallest_dimension() {
        // A tall, narrow target still gets enough cloth to fall past its sides: the
        // margin follows the largest extent (height, here), not just the footprint.
        let tall = box_target([-0.05, 0.0, -0.05], [0.05, 2.0, 0.05]);
        let scene = auto_drape(&tall, DrapePreset::Hard, FabricPreset::Leather).unwrap();
        let bb = Aabb::of(&scene.objects[1].positions).unwrap();
        // 0.1 footprint + 2 * (0.25 * 2.0) = 1.1 across.
        assert!((bb.extent()[0] - 1.1).abs() < 1e-5, "{:?}", bb.extent());
        assert!((bb.extent()[2] - 1.1).abs() < 1e-5, "{:?}", bb.extent());
    }
}
