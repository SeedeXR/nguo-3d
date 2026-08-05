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
use engine_core::contact::{closest_point_on_triangle_bary, surface_normal_at};
use glam::Vec3;

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
pub(crate) fn checked_bounds(role: &'static str, mesh: &Object) -> Result<Aabb, AvatarError> {
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
    /// Keep the authored coordinates, then push every vertex that starts INSIDE the
    /// body out to its surface ([`resolve_penetration`]).
    ///
    /// This is the mode for a garment modelled on somebody else's body — every vendor
    /// mesh in `samples/`, once scaled to metres, starts with 13–51 % of its vertices
    /// embedded up to 80 mm deep. [`AsAuthored`](Self::AsAuthored) does drape those
    /// without falling through (the static-collider gap is signed), but it never fits
    /// them: measured over 12 steps the cloth is frozen in barrier contact and moves
    /// < 1 mm, still embedded, at up to 72 s/step. Resolving the overlap first costs
    /// 0.2–13.7 s once (measured over the corpus, 31 k–200 k vertices) and made the same
    /// solves 6.6–50.8× cheaper.
    ///
    /// **Resize first** ([`crate::resize`]). This corrects a residual; it is not a fit, and
    /// it will decline rather than deform a garment that is the wrong size or shape — read
    /// [`FitReport::inside_after`] and do not assume it reached zero.
    FittedToBody,
}

/// The translation that puts `garment` on `avatar`. Never a scale or a rotation:
/// the bundle and glTF share this engine's metres/+Y-up/+Z-forward convention, so
/// there is nothing to convert, and auto-scaling a garment to a stature is grading,
/// not a uniform factor.
pub fn placement_offset(avatar: Aabb, garment: Aabb, placement: GarmentPlacement) -> [f32; 3] {
    match placement {
        // FittedToBody is authored coordinates plus a per-vertex correction, which is not
        // expressible as a translation — `drape_on_avatar` applies it after this offset.
        GarmentPlacement::AsAuthored | GarmentPlacement::FittedToBody => [0.0; 3],
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

/// Clearance left between a pushed-out garment vertex and the body surface. Well under
/// the contact barrier's activation distance, so a resolved garment is still *in* contact
/// — cloth resting on a body should be — just not inside it.
pub const FIT_GAP_M: f32 = 0.004;

/// Projection passes [`resolve_penetration`] will run before giving up.
///
/// One pass is not a fixed point, for two reasons. Pushing a vertex out of one facet can
/// land it inside another wherever the body is concave — armpit, crotch. And the
/// displacement is spread over the neighbourhood ([`FIT_STRAIN_RATIO`]) rather than applied
/// whole, so a deep vertex whose neighbours are outside only comes part of the way out per
/// pass; that is a pleat unfolding, and it takes several.
///
/// **The cap is a budget, not a convergence point, and an earlier version of this comment
/// claimed the opposite.** Measured over the resized sample corpus (both proxies, five
/// garments, 2026-08-04): exactly **one** of the ten pairs reaches zero — `women-skirt`/male,
/// at pass 1 — and the other nine are still non-zero at pass 24. What the extra passes buy is
/// real and large (`bike-jacket`/male 6 095 → 407, `dress-b`/female 23 264 → 213, i.e. 15–109×
/// fewer), and every sequence is still falling at 24, so the cap is where the returns stopped
/// paying for the seconds and not where the work finished.
///
/// [`FitReport::converged`] is the only thing that says whether it converged. Do not read
/// the cap as a promise, and do not read a monotone `per_pass` tail as failure — for a
/// garment that does not fit the body, a non-zero tail is the correct answer.
pub const FIT_MAX_PASSES: usize = 24;

/// Largest edge-length ratio [`resolve_penetration`] will let its projection introduce,
/// against the mesh as it arrived — the same quantity, and the same spelling, as the
/// solver's `ImplicitParams::strain_limit_ratio`.
///
/// **This is the constraint whose absence was the bug.** Projecting each vertex
/// independently to its own nearest surface point is unbounded: measured on
/// `women-skirt`/female it produced a 27.55× edge and crushed another to 1.8 % of its
/// authored length, while the solver's strain limiter would never have permitted either.
///
/// The bound is enforced on the *difference* between neighbouring vertices'
/// displacements, not on the displacements themselves, and that distinction is the whole
/// design: a region that is 74 mm inside the body can translate 74 mm outward with zero
/// differential and therefore zero distortion, while two neighbours cannot collapse onto
/// one another. An edge of authored length `L` whose ends differ by at most
/// `(ratio − 1)·L` ends between `(2 − ratio)·L` and `ratio·L`.
///
/// **Why 1.5 and not more.** Stretch and compression are the same number: the floor is
/// `2 − ratio`, so at 2.0 a collapsed edge becomes permissible and measured it happens
/// immediately — `women-skirt`/female read `edge_ratio_min` **0.0000** at 2.0 and 0.5045
/// at 1.5, with a collapsed triangle to match. Loosening it does buy penetration, and that
/// trade is real — but a degenerate triangle is not a garment, so the ceiling is set by the
/// floor. (That sensitivity sweep was taken before the relaxation's settle threshold was made
/// scale-relative; the 1.5 column is re-measured below, the 2.0 and 4.0 columns are not.)
///
/// **Measured at 1.5, over EVERY edge**, 2026-08-04, after the settle fix: the bound holds on
/// all ten corpus pairs — worst max 1.5128 (`dress-b`/female, 0.9 % over), worst min 0.4902
/// (`men-shirt`/female), zero edges under 1 % of authored length and zero triangles under
/// 1 % of authored area — re-measured 2026-08-05 over every edge and every triangle, with the
/// counts: **zero** edges under 0.01 and **zero** triangles under 0.01 on all ten pairs, and
/// 0–232 edge incidences of 1.19 M above 1.5 (worst `dress-a`/male, 0.02 % of its edges).
///
/// Read [`crate::resize::ResizeCheck::edge_ratio_max_all`] and
/// [`crate::resize::ResizeCheck::area_ratio_min_all`] for this, **not** the filtered pair: the
/// filtered pair excludes edges under [`crate::resize::EDGE_MEASURABLE_M`] and triangles under
/// its square, `men-shirt` has 108 682 of the former and 39 369 of the latter, and measured the
/// exclusion flatters both ends — `men-shirt`/male's area floor reads 0.2731 filtered against
/// **0.1252** over all triangles, and `bike-jacket`/female's ceiling 2.2379 against **4.4453**.
pub const FIT_STRAIN_RATIO: f32 = 1.5;

/// Relaxation work [`resolve_penetration`] will spend spreading one pass's displacement
/// field until it respects [`FIT_STRAIN_RATIO`], as a multiple of the garment's vertex
/// count.
///
/// A correction has to travel: the deepest overlap in the corpus is 76 mm against a
/// median 5.85 mm edge, so a spike is shared over ~26 rings. The relaxation is worklist
/// driven, so this is a *ceiling* on work and not a fixed schedule — the queue normally
/// empties first, and it only ever visits the part of the mesh the field reaches.
pub const FIT_SPREAD_WORK: usize = 256;

/// What [`resolve_penetration`] did, so a caller can log or assert on it instead of
/// guessing.
///
/// **Both halves matter.** `inside_after == 0` says the garment is out of the body;
/// [`check`](Self::check) says it is still a garment. A report with the first and not the
/// second is a regression, not a fix.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitReport {
    /// Vertices moved, summed over passes.
    pub pushed: usize,
    /// Vertices moved by each pass. A trailing `0` means it converged; a full-length
    /// vector with a non-zero tail means it ran out of passes with work still to do.
    pub per_pass: Vec<usize>,
    /// Vertices **still inside the body** when the pass loop stopped. This is the number
    /// the operation exists to drive to zero, and it is measured, not inferred from
    /// `per_pass`.
    pub inside_after: usize,
    /// How far the deepest of those is inside, in metres. Zero exactly when
    /// `inside_after == 0`.
    pub deepest_inside_m: f32,
    /// Edge-length and area distortion against the mesh as it arrived, at `factor` 1.0
    /// (see [`crate::resize::measure`]). The proof of the second half.
    pub check: crate::resize::ResizeCheck,
}

impl FitReport {
    /// Whether the garment is out of the body **and** the loop reached a fixed point.
    ///
    /// Not "the last pass moved nothing": with a wrong inside test that is vacuously true,
    /// and it was — the run that left a `men-shirt` vertex 9.1 mm inside the male armpit
    /// reported convergence and logged no warning.
    pub fn converged(&self) -> bool {
        self.inside_after == 0 && self.deepest_inside_m == 0.0
    }
}

/// A body triangle, its own normal, and the **angle-weighted pseudonormal** at each of its
/// corners. Which of the two is the surface normal depends on where the closest point
/// landed; see [`classify`].
struct BodyTri {
    p: [Vec3; 3],
    f: Vec3,
    n: [Vec3; 3],
}

impl BodyTri {
    fn of(mesh: &Object, t: [u32; 3], pn: &[Vec3]) -> Self {
        let p = t.map(|i| Vec3::from(mesh.positions[i as usize]));
        Self {
            p,
            f: (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero(),
            n: t.map(|i| pn[i as usize]),
        }
    }
}

/// Angle-weighted pseudonormals, one per vertex of `mesh` — a thin adaptor over
/// [`engine_core::contact::pseudonormals`], which is where the rule and its two measured
/// traps (bit-welding seam copies, skipping zero-area triangles) are documented. Shared
/// with the solver's static mesh collider, which takes the side of a contact from the same
/// field: the geometric pass and the barrier must not disagree about what "inside" means.
///
/// **This is what makes the inside test right at a concave crease.** Measured over all
/// 70 312 `men-shirt` vertices against the male proxy, the face normal missed 1 vertex that
/// was 9.1 mm inside the armpit and called 2 more inside that were not.
///
/// Ceiling: on a *cropped* body every vertex strictly inside the crop region has its whole
/// triangle star (`avatar_collider_near` keeps every triangle touching it), so its
/// pseudonormal is exact. Vertices dragged in only as a third corner sit on the crop's
/// fringe and get a partial angle sum — harmless here because the fringe is a full margin
/// (0.10 m) away from any garment vertex being tested.
fn pseudonormals(mesh: &Object) -> Vec<Vec3> {
    let pos: Vec<Vec3> = mesh.positions.iter().map(|p| Vec3::from(*p)).collect();
    engine_core::contact::pseudonormals(&pos, &mesh.tris)
}

/// Classify the points named by `active`, writing into `disp` the displacement that would
/// put each [`gap`](FIT_GAP_M) outside the body, or `ZERO` if it is already outside.
/// Returns how many were inside and how deep the deepest was.
///
/// **`active` is an exact restriction, not an approximation.** The body does not move, so
/// a garment vertex whose position is unchanged since it was last classified has the same
/// verdict; only vertices the previous pass actually moved — and those it left inside —
/// need looking at again. Measured on `men-shirt`/male that is 18 670 of 70 312 vertices
/// on pass 2 and a few hundred by pass 10.
fn classify(
    tri: &[BodyTri],
    pts: &[Vec3],
    gap: f32,
    active: &[u32],
    disp: &mut [Vec3],
) -> (usize, f32) {
    // O(active x triangles); spread it over the cores. Single-threaded against the
    // WHOLE body this took ~17 min for a 70k-vertex shirt. Cropping the body to the
    // garment's region (below) and threading here puts it at 0.2-6.9 s.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk = active.len().div_ceil(threads).max(1);
    let (mut inside, mut deepest) = (0usize, 0.0f32);

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, part) in active.chunks(chunk).enumerate() {
            handles.push(s.spawn(move || {
                let mut local = Vec::with_capacity(part.len());
                let (mut n_in, mut deep) = (0usize, 0.0f32);
                for &i in part {
                    let p = pts[i as usize];
                    let mut best = f32::INFINITY;
                    let mut hit: Option<(&BodyTri, Vec3, [f32; 3])> = None;
                    for t in tri {
                        // The weights come out of the same branch walk that produces `q`;
                        // re-solving for them afterwards was a second projection per vertex.
                        let (q, l) = closest_point_on_triangle_bary(p, t.p[0], t.p[1], t.p[2]);
                        let d = (p - q).length_squared();
                        if d < best {
                            best = d;
                            hit = Some((t, q, l));
                        }
                    }
                    let Some((t, q, l)) = hit else {
                        local.push(Vec3::ZERO);
                        continue;
                    };
                    // WHICH SIDE is decided by the pseudonormal, interpolated at the
                    // closest point — the face's own normal where that point is in the
                    // face's interior, since there the two agree and the interpolation is
                    // the approximation. WHICH WAY OUT is always the face's own normal.
                    //
                    // The split matters. A flat edge between two coplanar triangles has
                    // the face normal for its pseudonormal, but *interpolated vertex*
                    // pseudonormals along it carry whatever else those two corners touch:
                    // on a box's top face the corners average in the side walls, so a
                    // sheet lying on the diagonal got pushed 45° sideways and a rigid
                    // translation measured a 1.5x stretch. The face normal has no such
                    // pollution and, at a crease, is still an escape from the facet that
                    // is actually closest. A zero normal (a degenerate triangle) has no
                    // side, so it never flips one.
                    let n = surface_normal_at(l, t.f, t.n);
                    let depth = -(p - q).dot(n);
                    if n != Vec3::ZERO && depth > 0.0 {
                        local.push(q + t.f * gap - p);
                        n_in += 1;
                        deep = deep.max(depth);
                    } else {
                        local.push(Vec3::ZERO);
                    }
                }
                (ci, local, n_in, deep)
            }));
        }
        for h in handles {
            let (ci, local, n_in, deep) = h.join().expect("resolve_penetration worker panicked");
            for (k, d) in local.into_iter().enumerate() {
                disp[active[ci * chunk + k] as usize] = d;
            }
            inside += n_in;
            deepest = deepest.max(deep);
        }
    });
    (inside, deepest)
}

/// Garment vertex adjacency with the **authored** rest length of every incident edge —
/// the length the strain bound is measured against, so it never drifts with the
/// correction.
struct Adj {
    offs: Vec<u32>,
    nbr: Vec<u32>,
    rest: Vec<f32>,
}

impl Adj {
    fn of(mesh: &Object) -> Self {
        let n = mesh.positions.len();
        let mut deg = vec![0u32; n + 1];
        // Triangle edges, both directions, duplicates kept: a shared edge simply weighs
        // twice, which is the usual umbrella weighting and costs no dedup pass.
        let edges = || {
            mesh.tris.iter().flat_map(|t| {
                [
                    (t[0], t[1]),
                    (t[1], t[0]),
                    (t[1], t[2]),
                    (t[2], t[1]),
                    (t[2], t[0]),
                    (t[0], t[2]),
                ]
            })
        };
        for (u, _) in edges() {
            deg[u as usize + 1] += 1;
        }
        for i in 0..n {
            deg[i + 1] += deg[i];
        }
        let offs = deg.clone();
        let mut nbr = vec![0u32; offs[n] as usize];
        let mut rest = vec![0.0f32; offs[n] as usize];
        let mut at = offs.clone();
        for (u, v) in edges() {
            let s = at[u as usize] as usize;
            at[u as usize] += 1;
            nbr[s] = v;
            rest[s] = (Vec3::from(mesh.positions[u as usize])
                - Vec3::from(mesh.positions[v as usize]))
            .length();
        }
        Self { offs, nbr, rest }
    }

    fn range(&self, v: usize) -> std::ops::Range<usize> {
        self.offs[v] as usize..self.offs[v + 1] as usize
    }
}

/// Largest amount by which any edge's two ends' displacements differ beyond `budget ×`
/// its authored length, as a fraction of that length. Zero means the bound holds.
fn worst_violation(disp: &[Vec3], adj: &Adj, budget: f32) -> f32 {
    let mut worst = 0.0f32;
    for v in 0..disp.len() {
        for i in adj.range(v) {
            let rest = adj.rest[i];
            if rest > 0.0 {
                let over = (disp[adj.nbr[i] as usize] - disp[v]).length() - budget * rest;
                worst = worst.max(over / rest);
            }
        }
    }
    worst.max(0.0)
}

/// Spread `disp` over the mesh until no edge's two ends differ by more than
/// `(ratio − 1)` of its authored length. In place; returns the largest violation still
/// present as a fraction of the rest length, so the caller reports what happened instead
/// of assuming it converged.
///
/// Gauss-Seidel driven by a **worklist**: a vertex is revisited only when a neighbour of
/// it actually changed. The field has to travel roughly `depth / (budget × edge)` ≈ 26
/// rings on the deepest overlap in the corpus, but it only ever travels through the part
/// of the mesh that is moving, and it stops the moment nothing changes by more than a small
/// fraction of the local budget — measured, a fixed sweep schedule spent 10–40× the work for
/// the same answer.
/// Deterministic: FIFO order from a fixed initial order, single threaded, which the drape
/// tests require.
///
/// See [`FIT_STRAIN_RATIO`] for why this bounds the edge-length ratio and why the bound is
/// on the *difference* between neighbours rather than on the displacement itself.
fn spread(disp: &mut [Vec3], adj: &Adj, ratio: f32, work: usize) -> f32 {
    /// Over-relaxation. A vertex takes the mean of its incident violations, so without
    /// this it closes ~1/12 of its own gap per visit on a triangulated mesh and the
    /// long-wavelength part of the field crawls. Measured on `women-skirt`/female, the
    /// sweeps needed to reach the bound: 1.0 → over 1 024, 1.9 → 256.
    const OMEGA: f32 = 1.9;
    /// Settle threshold as a fraction of the vertex's own tightest budget.
    ///
    /// **It must be relative, and an absolute one was the bug.** The constraint is
    /// `budget × rest` per edge, so on `men-shirt`'s 108 682 authored edges under 1 mm the
    /// whole budget is sub-micron — an absolute 1 µm threshold declared such a vertex
    /// settled while it was still violating, and stopped queueing its neighbours, so the
    /// violation never propagated. Measured: the bound missed by **3.5× its budget**
    /// (`men-shirt`/male) and 4.0× (`men-shirt`/female) with an absolute 1e-6, against
    /// ≤ 0.01× with this. f32 can carry it: `disp` is a *displacement* of millimetre
    /// magnitude, whose ulp is ~1e-10, not an absolute coordinate at 1.5 m whose ulp is
    /// 1.2e-7.
    const SETTLED_FRAC: f32 = 1.0e-4;
    let budget = (ratio - 1.0).max(0.0);
    let n = disp.len();
    // Per-vertex settle floor, precomputed: the loop visits a vertex many times.
    let settled: Vec<f32> = (0..n)
        .map(|v| {
            let tight = adj
                .range(v)
                .map(|i| adj.rest[i])
                .filter(|r| *r > 0.0)
                .fold(f32::INFINITY, f32::min);
            // Never zero, or a ratio of exactly 1 (budget 0) would spin to the work cap.
            (SETTLED_FRAC * budget * tight).max(f32::MIN_POSITIVE)
        })
        .collect();
    let mut queued = vec![true; n];
    let mut q: std::collections::VecDeque<u32> = (0..n as u32).collect();
    let mut left = work.saturating_mul(n);
    while let Some(v) = q.pop_front() {
        if left == 0 {
            break;
        }
        left -= 1;
        let v = v as usize;
        queued[v] = false;
        let mut pull = Vec3::ZERO;
        for i in adj.range(v) {
            let diff = disp[adj.nbr[i] as usize] - disp[v];
            let over = diff.length() - budget * adj.rest[i];
            if over > 0.0 {
                // Halve the excess: the other end takes the rest when its turn comes.
                pull += diff.normalize_or_zero() * (over * 0.5);
            }
        }
        let step = pull * (OMEGA / adj.range(v).len().max(1) as f32);
        if step.length() <= settled[v] {
            continue;
        }
        disp[v] += step;
        for i in adj.range(v) {
            let u = adj.nbr[i];
            if !queued[u as usize] {
                queued[u as usize] = true;
                q.push_back(u);
            }
        }
    }
    worst_violation(disp, adj, budget)
}

/// Push every vertex of `garment` that starts inside `body` out to the body's surface plus
/// `gap`, and keep the result a garment. Iterates to a fixed point, at most
/// [`FIT_MAX_PASSES`] times.
///
/// **Why this exists.** The static mesh collider's gap is signed, so cloth that starts
/// inside the body is pushed the right way and does not fall through. But the barrier
/// only *arrests* it — the edge-edge terms are unsigned and brake the climb — so a vertex
/// 48 mm inside stays 48 mm inside and the garment sits locked in contact instead of
/// draping. Fitting is a geometry problem; asking the solver to dig a garment out through
/// a barrier is both wrong and, measured, up to 50× more expensive per step.
///
/// **It is the second half of drop-and-fit, not the first.** Resize the garment to the
/// body ([`crate::resize`]) before calling this: on the corpus that takes the overlap this
/// has to clear from 13–51 % of vertices down to 3–27 %, and a correction that small is one
/// this can make without deforming anything.
///
/// # Both halves are reported and both are the contract
///
/// * [`FitReport::inside_after`] must be 0 — measured with an **angle-weighted
///   pseudonormal**, which is what makes the test right in an armpit.
/// * [`FitReport::check`] must show no serious distortion — the displacement field is
///   spread over the neighbourhood until it respects [`FIT_STRAIN_RATIO`], because
///   projecting each vertex independently is unbounded and measured 27× edges.
///
/// `converged()` requires the first. A pass reporting zero penetration with a stretched
/// edge is a regression, so assert on both.
///
/// # Measured on the whole corpus, 2026-08-04 — and the trade is not free
///
/// Five drapeable samples × both shipped proxies, each first resized by [`crate::resize`],
/// `--release`, whole run at a peak RSS of **204 MB** in 74.5 s. `edge` is min…max of the
/// per-edge length ratio against the mesh this pass received, **over every edge** — see
/// [`FIT_STRAIN_RATIO`] for why the filtered pair is the wrong column to read.
///
/// | garment | body | resize | inside % → left | deepest left | edge | s |
/// |---|---|---|---|---|---|---|
/// | women-skirt | male | 1.3456 | 3.01 % → **0** | — | 0.5001…1.4963 | 0.1 |
/// | women-skirt | female | 1.1409 | 27.60 % → **3** | 0.64 mm | 0.5060…1.5008 | 0.6 |
/// | men-shirt | female | 1.1264 | 14.72 % → 210 | 43.0 mm | 0.4902…1.5022 | 2.6 |
/// | men-shirt | male | *declined* | 26.55 % → 476 | 55.2 mm | 0.4990…1.5008 | 3.5 |
/// | bike-jacket | male | *no ring* | 19.56 % → 407 | 55.9 mm | 0.5006…1.5011 | 7.5 |
/// | bike-jacket | female | *no ring* | 15.67 % → 1 072 | 47.9 mm | 0.5000…1.5013 | 4.4 |
/// | dress-b | male | 1.1348 | 11.01 % → 425 | 11.5 mm | 0.4973…1.5018 | 14.3 |
/// | dress-b | female | *declined* | 12.48 % → 213 | 6.0 mm | 0.4972…1.5128 | 16.5 |
/// | dress-a | male | 1.0475 | 13.41 % → 1 472 | 57.9 mm | 0.4975…1.5031 | 17.4 |
/// | dress-a | female | 1.1385 | 6.17 % → 773 | 46.7 mm | 0.4989…1.5010 | 6.0 |
///
/// **Read the first two rows and then the rest.** `women-skirt` is the one garment the
/// resize genuinely enlarged onto the body, and there this pass does what it is for: 0 and
/// 3 vertices left, sub-millimetre, no distortion worth the name. Everywhere else the
/// resize declined or barely moved, the garment arrives 6–27 % embedded up to 58 mm deep,
/// and **that is not a residual — it is a garment that does not fit this body.** No
/// distortion-bounded projection can fix it and this one does not pretend to.
///
/// The alternative was measured on the same corpus by removing the bound: 0 vertices inside
/// on 9 of 10 pairs after two passes, at edge ratios of **0.0000…75.77** — collapsed edges
/// and triangles, area ratios to 962× — i.e. zero penetration bought by destroying the
/// mesh. That is the regression this refuses to ship.
///
/// **Preconditions.** `body` must be closed and outward-wound — the same requirement
/// [`avatar_collider`] documents and `tests/avatar_bundle.rs` pins against the shipped
/// proxies. Both meshes must already be in metres and in the avatar's frame; this moves
/// vertices and never scales, so a garment authored in centimetres must be converted
/// before it gets here (`engine_io::load_mesh_in`).
///
/// Only the garment is touched; `body` is read.
pub fn resolve_penetration(
    body: &Object,
    garment: &mut Object,
    gap: f32,
) -> Result<FitReport, AvatarError> {
    let garment_bb = checked_bounds("garment", garment)?;
    checked_bounds("avatar", body)?;

    // The nearest body triangle to any garment vertex lies within the garment's own box,
    // so cropping loses nothing and is the difference between seconds and minutes. The
    // margin must exceed the deepest overlap we expect to correct (80 mm, measured).
    let near = avatar_collider_near(body, garment_bb, 0.10).unwrap_or_else(|_| body.clone());
    let pn = pseudonormals(&near);
    let tri: Vec<BodyTri> = near
        .tris
        .iter()
        .map(|t| BodyTri::of(&near, *t, &pn))
        .collect();
    if tri.is_empty() {
        return Ok(FitReport::default()); // garment nowhere near the body: nothing to fix
    }

    let authored: Vec<Vec3> = garment.positions.iter().map(|p| Vec3::from(*p)).collect();
    let adj = Adj::of(garment);
    // The strain bound is on the TOTAL displacement from the authored mesh, not on one
    // pass's: constraining each pass separately lets N passes compound to N times the
    // budget, and measured it did — `women-skirt`/female reached a 2.53x edge in two
    // passes of a bound that permits 1.5.
    let n = authored.len();
    let mut total = vec![Vec3::ZERO; n];
    let mut pts = authored.clone();
    let mut disp = vec![Vec3::ZERO; n];
    let mut active: Vec<u32> = (0..n as u32).collect();
    let mut report = FitReport::default();
    let mut violation = 0.0f32;
    // `FIT_MAX_PASSES` *corrections*, and one more classification than that, because what
    // the report says must be the state the garment is actually left in: measuring before
    // the last correction and calling it `inside_after` is the same class of lie as the
    // convergence flag this replaces.
    for pass in 0..=FIT_MAX_PASSES {
        disp.fill(Vec3::ZERO);
        let (inside, deepest) = classify(&tri, &pts, gap, &active, &mut disp);
        report.per_pass.push(inside);
        report.pushed += inside;
        (report.inside_after, report.deepest_inside_m) = (inside, deepest);
        if inside == 0 || pass == FIT_MAX_PASSES {
            break;
        }
        let mut cand: Vec<Vec3> = total.iter().zip(&disp).map(|(t, d)| *t + *d).collect();
        violation = spread(&mut cand, &adj, FIT_STRAIN_RATIO, FIT_SPREAD_WORK);
        // Next pass need only look at vertices that moved, plus the ones this pass found
        // inside and could not move far enough — everything else is where it was, and the
        // body has not moved either.
        active = (0..n as u32)
            .filter(|&i| {
                let i = i as usize;
                cand[i] != total[i] || disp[i] != Vec3::ZERO
            })
            .collect();
        total = cand;
        for ((p, a), t) in pts.iter_mut().zip(&authored).zip(&total) {
            *p = *a + *t;
        }
    }
    garment.positions = pts.iter().map(|p| [p.x, p.y, p.z]).collect();
    let authored: Vec<[f32; 3]> = authored.iter().map(|p| [p.x, p.y, p.z]).collect();
    report.check = crate::resize::measure(&authored, garment, 1.0);

    if !report.converged() {
        tracing::warn!(
            garment = %garment.name,
            avatar = %body.name,
            per_pass = ?report.per_pass,
            inside_after = report.inside_after,
            deepest_inside_m = report.deepest_inside_m,
            "garment still intersects the body after {FIT_MAX_PASSES} passes; the contact \
             barrier must resolve the remainder"
        );
    } else {
        tracing::info!(
            garment = %garment.name,
            avatar = %body.name,
            pushed = report.pushed,
            per_pass = ?report.per_pass,
            body_tris = tri.len(),
            edge_ratio_all = ?(report.check.edge_ratio_min_all, report.check.edge_ratio_max_all),
            area_ratio_all = ?(report.check.area_ratio_min_all, report.check.area_ratio_max_all),
            degenerate_edges = report.check.degenerate_edges,
            degenerate_tris = report.check.degenerate_tris,
            strain_violation = violation,
            "garment penetration resolved"
        );
    }
    Ok(report)
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
    // After the translation, before the scene: the projection needs the garment where it
    // will actually sit, and `drape_scene` consumes the final positions.
    if placement == GarmentPlacement::FittedToBody {
        resolve_penetration(avatar, &mut cloth, FIT_GAP_M)?;
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
pub(crate) mod tests {
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

    // ---- penetration resolution -----------------------------------------------
    // The body is `box_mesh`, which is closed and outward-wound, so it satisfies
    // `resolve_penetration`'s stated precondition. Every assertion here is about the
    // SIGN of the correction, which is the part that was wrong in the solver.

    /// A grid of cloth vertices on the plane `y`, spanning `[-half, half]` in x and z.
    fn cloth_grid(name: &str, n: usize, y: f32, half: f32) -> Object {
        let mut positions = Vec::new();
        let step = if n > 1 {
            2.0 * half / (n - 1) as f32
        } else {
            0.0
        };
        for j in 0..n {
            for i in 0..n {
                positions.push([-half + i as f32 * step, y, -half + j as f32 * step]);
            }
        }
        let mut tris = Vec::new();
        for j in 0..n.saturating_sub(1) {
            for i in 0..n.saturating_sub(1) {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, r, v + 1]);
                tris.push([v + 1, r, r + 1]);
            }
        }
        let mut o = box_mesh(name, [0.0; 3], [1.0; 3]);
        o.positions = positions;
        o.tris = tris;
        o
    }

    /// A closed box wound **outward**, and it proves it before returning.
    ///
    /// Deliberately not `box_mesh`: that fixture is wound INWARD (its bottom face
    /// `[0,2,1]` has normal `+Y`, which points into the solid), which is fine for the
    /// tests that only care about statics and reindexing, and exactly wrong here. The
    /// shipped proxies are outward-wound — `tests/avatar_bundle.rs`'s
    /// `collision_proxies_are_outward_wound` is what pins that — so a sign test must use
    /// a body with the same handedness or it asserts the opposite of the truth.
    pub(crate) fn body_box() -> Object {
        let (min, max) = ([-0.5f32, 0.0, -0.5], [0.5f32, 1.0, 0.5]);
        let p = [
            [min[0], min[1], min[2]], // 0
            [max[0], min[1], min[2]], // 1
            [max[0], min[1], max[2]], // 2
            [min[0], min[1], max[2]], // 3
            [min[0], max[1], min[2]], // 4
            [max[0], max[1], min[2]], // 5
            [max[0], max[1], max[2]], // 6
            [min[0], max[1], max[2]], // 7
        ];
        let tris = vec![
            [0, 1, 2],
            [0, 2, 3], // bottom, normal -Y
            [4, 6, 5],
            [4, 7, 6], // top,    normal +Y
            [0, 5, 1],
            [0, 4, 5], // -Z
            [3, 2, 6],
            [3, 6, 7], // +Z
            [0, 3, 7],
            [0, 7, 4], // -X
            [1, 5, 6],
            [1, 6, 2], // +X
        ];
        let centre = Vec3::new(0.0, 0.5, 0.0);
        for t in &tris {
            let (a, b, c) = (
                Vec3::from(p[t[0] as usize]),
                Vec3::from(p[t[1] as usize]),
                Vec3::from(p[t[2] as usize]),
            );
            let n = (b - a).cross(c - a);
            let outward = (a + b + c) / 3.0 - centre;
            assert!(
                n.dot(outward) > 0.0,
                "body_box triangle {t:?} is wound inward (n {n:?} vs outward {outward:?})"
            );
        }
        let mut o = box_mesh("body", min, max);
        o.positions = p.to_vec();
        o.tris = tris;
        o.is_static = true;
        o
    }

    /// A prism whose cross-section is a disc with a **30° wedge notched out**, so the
    /// vertical edge at the notch's apex is a concave crease of only 30° exterior opening
    /// — the synthetic armpit, and the smallest closed mesh on which the closest *face's*
    /// normal gets the inside test wrong.
    ///
    /// Cross-section walks the reflex apex at the origin, out along the notch's upper wall,
    /// round the unit circle, and back along the lower wall. Star-shaped from the apex, so
    /// a fan from it triangulates both caps without leaving the polygon.
    fn notched_prism() -> Object {
        use std::f32::consts::{FRAC_PI_2, PI};
        let half = 15.0f32.to_radians();
        let ring = |a: f32| [a.cos(), a.sin()];
        let xz = [
            [0.0, 0.0], // 0: apex, reflex, interior angle 330°
            ring(half), // 1: upper notch mouth
            ring(FRAC_PI_2),
            ring(PI),
            ring(3.0 * FRAC_PI_2),
            ring(-half), // 5: lower notch mouth
        ];
        let m = xz.len();
        let mut positions: Vec<[f32; 3]> = xz.iter().map(|p| [p[0], 0.0, p[1]]).collect();
        positions.extend(xz.iter().map(|p| [p[0], 1.0, p[1]]));
        let (lo, hi) = (0u32, m as u32);

        let mut tris = Vec::new();
        for i in 0..m as u32 {
            let j = (i + 1) % m as u32;
            // Side quad, wound so its normal points away from the y axis.
            tris.push([lo + i, hi + j, lo + j]);
            tris.push([lo + i, hi + i, hi + j]);
        }
        for i in 1..m as u32 - 1 {
            tris.push([lo, lo + i, lo + i + 1]); // bottom cap, fanned from the apex
            tris.push([hi, hi + i + 1, hi + i]); // top cap
        }

        // Outward winding is this fixture's whole point, so prove it rather than eyeball
        // it: a closed outward-wound mesh has positive signed volume.
        let vol: f32 = tris
            .iter()
            .map(|t| {
                let p = t.map(|i| Vec3::from(positions[i as usize]));
                p[0].dot(p[1].cross(p[2])) / 6.0
            })
            .sum();
        assert!(vol > 0.1, "notched_prism is wound inward (volume {vol})");

        let mut o = box_mesh("notch", [-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]);
        o.positions = positions;
        o.tris = tris;
        o.is_static = true;
        o
    }

    /// A point at polar angle `deg` and radius `r` about the notch apex, at mid-height.
    fn in_notch_solid(deg: f32, r: f32) -> Vec3 {
        let a = deg.to_radians();
        Vec3::new(r * a.cos(), 0.5, r * a.sin())
    }

    #[test]
    fn pseudonormals_do_not_care_how_the_surface_is_triangulated() {
        // Angle weighting is what makes a pseudonormal a property of the SURFACE rather
        // than of its triangulation. Flip the diagonal of one crease wall's quad: identical
        // geometry, identical winding, no new vertices — but the apex now has ONE incident
        // triangle on that wall instead of two, spanning the same 90° corner. Angle
        // weighting is unmoved; counting faces unweighted tilts the apex pseudonormal by
        // 2.4°, and the shipped proxies are not uniformly triangulated.
        let body = notched_prism();
        let want = pseudonormals(&body)[0];

        // The upper wall's quad in boundary order: apex_bot 0, apex_top 6, mouth_top 7,
        // mouth_bot 1. It ships split on the 0–7 diagonal; re-split it on 6–1.
        let mut flipped = body.clone();
        flipped.tris.retain(|t| {
            let s: std::collections::HashSet<u32> = t.iter().copied().collect();
            !s.is_subset(&[0u32, 6, 7, 1].into_iter().collect())
        });
        assert_eq!(flipped.tris.len(), body.tris.len() - 2, "wall not replaced");
        flipped.tris.push([0, 6, 1]);
        flipped.tris.push([6, 7, 1]);
        assert_eq!(
            flipped.tris.iter().filter(|t| t.contains(&0)).count(),
            body.tris.iter().filter(|t| t.contains(&0)).count() - 1,
            "the flip did not change the apex's triangle count, so this proves nothing"
        );

        let got = pseudonormals(&flipped)[0];
        assert!(
            got.dot(want) > 0.99999,
            "retriangulating one wall moved the apex pseudonormal {want:?} -> {got:?}"
        );
    }

    #[test]
    fn pseudonormals_survive_a_uv_seam_split() {
        // The shipped proxies are glTF, which splits vertices on UV and normal seams — the
        // male collision proxy is 10 230 glTF vertices for 7 498 unique positions. An angle
        // sum over half a vertex's incident faces is not its pseudonormal, so the split
        // copies have to weld back together.
        let body = notched_prism();
        let want = pseudonormals(&body);

        // Split vertex 0 (the notch apex, whose star spans both crease walls) into two
        // co-located copies and hand half of its triangles to each.
        let mut split = body.clone();
        let dup = split.positions.len() as u32;
        split.positions.push(body.positions[0]);
        let mut flip = false;
        for t in &mut split.tris {
            if t.contains(&0) {
                flip = !flip;
                if flip {
                    for i in t.iter_mut().filter(|i| **i == 0) {
                        *i = dup;
                    }
                }
            }
        }
        assert!(flip || split.tris.iter().flatten().any(|&i| i == dup));

        let got = pseudonormals(&split);
        for (k, w) in want.iter().enumerate() {
            assert!(
                got[k].dot(*w) > 0.9999,
                "vertex {k}: split {:?} vs welded {w:?}",
                got[k]
            );
        }
        assert!(got[dup as usize].dot(want[0]) > 0.9999, "the split copy");
    }

    #[test]
    fn a_concave_crease_needs_a_pseudonormal_and_the_closest_face_normal_gets_it_wrong() {
        let body = notched_prism();
        let pn = pseudonormals(&body);
        let tri: Vec<BodyTri> = body
            .tris
            .iter()
            .map(|t| BodyTri::of(&body, *t, &pn))
            .collect();

        // Both points are INSIDE the solid — the notch spans only ±15° about +x — and both
        // are in the apex edge's Voronoi region, so the two crease faces are exactly
        // equidistant and the closest point is the apex for both of them. One sits on the
        // wrong side of the lower wall's plane, the other on the wrong side of the upper's,
        // so a single face normal must misclassify one of them whichever way the tie falls.
        let pts = [in_notch_solid(150.0, 0.05), in_notch_solid(210.0, 0.05)];
        let mut disp = vec![Vec3::ZERO; 2];
        let (inside, deepest) = classify(&tri, &pts, FIT_GAP_M, &[0, 1], &mut disp);
        assert_eq!(inside, 2, "pseudonormal missed a vertex inside the crease");
        // 0.05 m out along a ray 150° from the +x bisector: depth = 0.05·cos(30°).
        assert!(
            (deepest - 0.05 * 30.0f32.to_radians().cos()).abs() < 1.0e-4,
            "deepest {deepest}"
        );

        // The same test signed by the closest FACE's normal — what this used to do — and
        // it gets one of the two wrong. This is the assertion that makes the fixture
        // discriminating: without it, a pseudonormal test could pass on a mesh where the
        // face normal would have done just as well.
        let face_verdicts = pts.map(|p| {
            let mut best = f32::INFINITY;
            let mut n = Vec3::ZERO;
            let mut bq = p;
            for t in &tri {
                let q = engine_core::contact::closest_point_on_triangle(p, t.p[0], t.p[1], t.p[2]);
                let d = (p - q).length_squared();
                if d < best {
                    best = d;
                    bq = q;
                    n = (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]).normalize_or_zero();
                }
            }
            (p - bq).dot(n) < 0.0
        });
        assert!(
            !face_verdicts[0] || !face_verdicts[1],
            "the closest face normal classified both crease points correctly, so this \
             fixture does not discriminate and the test proves nothing"
        );
    }

    #[test]
    fn resolve_penetration_pushes_cloth_out_and_leaves_a_gap() {
        let body = body_box();
        // A sheet 10 mm under the box's top face. Shallow and one-sided, so every vertex
        // needs the same outward push and the strain bound costs nothing — this is what a
        // *resized* garment's residual overlap looks like.
        let mut cloth = cloth_grid("sheet", 7, 0.99, 0.3);
        let before = cloth.positions.clone();

        let r = resolve_penetration(&body, &mut cloth, FIT_GAP_M).expect("resolve");

        assert_eq!(r.pushed, before.len(), "every vertex started inside");
        assert!(r.converged(), "per_pass {:?} did not end in 0", r.per_pass);
        assert_eq!(r.inside_after, 0);
        assert_eq!(r.deepest_inside_m, 0.0);
        // Nothing may remain inside, and everything must clear the surface by ~FIT_GAP_M.
        for p in &cloth.positions {
            let inside = (0..3).all(|k| {
                let (lo, hi) = ([-0.5, 0.0, -0.5][k], [0.5, 1.0, 0.5][k]);
                p[k] > lo && p[k] < hi
            });
            assert!(!inside, "vertex {p:?} is still inside the body");
            assert!(
                p[1] >= 1.0 + FIT_GAP_M - 1.0e-5,
                "vertex {p:?} did not clear the top face by the gap"
            );
        }
        let moved = cloth
            .positions
            .iter()
            .zip(&before)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(moved, before.len(), "some vertices were not corrected");
        // A uniform push is a translation, so it must cost NO distortion at all. This is
        // the other half of the contract and it is the half that used to be unmeasured.
        assert!(
            r.check.edge_spread() < 1.0e-4,
            "a uniform push distorted the sheet: {:?}",
            r.check
        );
    }

    #[test]
    fn the_strain_bound_holds_even_when_the_overlap_cannot_be_resolved() {
        // A sheet through the MIDDLE of the box: every vertex 0.5 m inside, and opposite
        // halves need pushing in opposite directions. There is no way out of this that
        // keeps the sheet a sheet, so the operation must decline honestly rather than
        // shred it — the whole point of FIT_STRAIN_RATIO.
        let body = body_box();
        let mut cloth = cloth_grid("sheet", 7, 0.5, 0.3);
        let r = resolve_penetration(&body, &mut cloth, FIT_GAP_M).expect("resolve");

        assert!(
            !r.converged() && r.inside_after > 0 && r.deepest_inside_m > 0.0,
            "an unresolvable overlap reported success: {r:?}"
        );
        // Bounded, both ways, with room for the relaxation's residual (measured 0.3 %).
        // Over **every** edge, including sub-`EDGE_MEASURABLE_M` ones: the filtered pair
        // excluded a quarter of `men-shirt`'s edges and hid a 1.796x stretch there.
        let (lo, hi) = (2.0 - FIT_STRAIN_RATIO, FIT_STRAIN_RATIO);
        assert!(
            r.check.edge_ratio_max_all <= hi * 1.02 && r.check.edge_ratio_min_all >= lo * 0.98,
            "edge ratio {:?} escaped [{lo}, {hi}]",
            (r.check.edge_ratio_min_all, r.check.edge_ratio_max_all)
        );
        // No collapsed edge and no collapsed triangle: the acceptance criterion's other
        // two halves, which the old unbounded projection failed (measured 1.8 % of an edge).
        //
        // **Both over the unfiltered pair.** Asserting an acceptance criterion on the
        // *filtered* fields was the defect: they skip edges under `EDGE_MEASURABLE_M` and
        // triangles under its square, which on `men-shirt` is a quarter of the edges and
        // 30.7 % of the triangles, so a collapse there would never have failed a test.
        assert!(
            r.check.edge_ratio_min_all > 0.01,
            "an edge collapsed: {:?}",
            r.check
        );
        assert!(
            r.check.area_ratio_min_all > 0.01,
            "a triangle collapsed: {:?}",
            r.check
        );
        // This fixture excludes nothing, so the two pairs coincide here and the assertions
        // above cannot themselves prove the `_all` fields see the excluded geometry. That is
        // pinned where the instrument lives, on a mesh built to be excluded:
        // `resize::tests::a_collapse_among_the_excluded_geometry_is_still_reported`.
        assert_eq!(
            (r.check.degenerate_edges, r.check.degenerate_tris),
            (0, 0),
            "{:?}",
            r.check
        );

        // `inside_after` is counted over an ACTIVE SUBSET of the vertices, on the argument
        // that a vertex nothing moved cannot have changed sides. Check that against an
        // independent sweep over EVERY vertex, or the report is only as good as the
        // argument. This fixture is the one that keeps the spread busiest.
        let pn = pseudonormals(&body);
        let tri: Vec<BodyTri> = body
            .tris
            .iter()
            .map(|t| BodyTri::of(&body, *t, &pn))
            .collect();
        let pts: Vec<Vec3> = cloth.positions.iter().map(|p| Vec3::from(*p)).collect();
        let all: Vec<u32> = (0..pts.len() as u32).collect();
        let mut disp = vec![Vec3::ZERO; pts.len()];
        let (inside, deepest) = classify(&tri, &pts, FIT_GAP_M, &all, &mut disp);
        assert_eq!(inside, r.inside_after, "the active set lost a vertex");
        assert!((deepest - r.deepest_inside_m).abs() < 1.0e-6, "{deepest}");
    }

    /// A partial overlap costs MANY passes and needs the whole strain budget, so this pins
    /// the two constants nothing else did — mutating [`FIT_MAX_PASSES`] to 1 or
    /// [`FIT_STRAIN_RATIO`] to 1.02 left every other test in this file green.
    ///
    /// The fixture is a sheet twice the box's width laid through it at y = 0.7: the middle
    /// 5 edges are inside and must climb 0.304 m to the top face, while the rim is already
    /// outside and must not move. The bound is on the *difference* between neighbours, so
    /// that climb can only be paid over rings — 0.304 m at a budget of 0.5 x 0.2 m per edge
    /// is at least 3 rings of slope, and each pass buys part of it. Squeeze the budget and
    /// the slope no longer fits in the rim; cap the passes at 1 and it never gets there.
    ///
    /// Measured: 25 inside at pass 0, 15 left after 24 passes at 53 mm, edge ratio
    /// 0.614…1.498, i.e. it uses essentially the whole budget and still cannot finish. That
    /// residual is a right-angle corner, not a bug — see the assertion notes.
    #[test]
    fn a_partial_overlap_needs_several_passes_and_the_whole_strain_budget() {
        let body = body_box();
        let mut cloth = cloth_grid("half-in", 21, 0.7, 2.0);
        let r = resolve_penetration(&body, &mut cloth, FIT_GAP_M).expect("resolve");

        // Measured, and stated rather than asserted away: this fixture does NOT reach zero.
        // 15 of its 441 vertices sit at the box's top-face perimeter, where the sheet has to
        // turn a right angle, and they stay ~53 mm inside. A cube is a harsher body than an
        // avatar; what matters here is that the passes make progress and the bound holds.
        assert!(
            r.inside_after < r.per_pass[0],
            "no progress at all across the passes: {:?}",
            r.per_pass
        );
        // One pass cannot do it, and that is the property FIT_MAX_PASSES exists for. The
        // count is measured; the assertion is only that it is well past 1.
        assert!(
            r.per_pass.len() > 4,
            "converged in {} passes — the fixture stopped discriminating FIT_MAX_PASSES: {:?}",
            r.per_pass.len(),
            r.per_pass
        );
        // ...and it needed most of the budget to do it, which is what makes a squeezed
        // FIT_STRAIN_RATIO fail rather than merely run slower.
        assert!(
            r.check.edge_ratio_max_all > 1.0 + 0.5 * (FIT_STRAIN_RATIO - 1.0),
            "used only {:.4} of the {FIT_STRAIN_RATIO} budget — not discriminating",
            r.check.edge_ratio_max_all
        );
        let (lo, hi) = (2.0 - FIT_STRAIN_RATIO, FIT_STRAIN_RATIO);
        assert!(
            r.check.edge_ratio_max_all <= hi * 1.02 && r.check.edge_ratio_min_all >= lo * 0.98,
            "edge ratio {:?} escaped [{lo}, {hi}]",
            (r.check.edge_ratio_min_all, r.check.edge_ratio_max_all)
        );
    }

    #[test]
    fn resolve_penetration_is_idempotent_and_leaves_outside_cloth_alone() {
        let body = body_box();
        // Well clear of the box: nothing to do, and nothing may be touched.
        let mut clear = cloth_grid("clear", 5, 3.0, 0.3);
        let untouched = clear.positions.clone();
        let r = resolve_penetration(&body, &mut clear, FIT_GAP_M).expect("resolve");
        assert_eq!(r.pushed, 0, "cloth 2 m above the body was moved");
        assert_eq!(clear.positions, untouched);
        assert!(r.converged());

        // Resolving an already-resolved garment must be a no-op — otherwise repeated
        // drapes would walk the garment away from the body.
        let mut cloth = cloth_grid("sheet", 7, 0.99, 0.3);
        resolve_penetration(&body, &mut cloth, FIT_GAP_M).expect("first");
        let after_first = cloth.positions.clone();
        let second = resolve_penetration(&body, &mut cloth, FIT_GAP_M).expect("second");
        assert_eq!(
            second.pushed, 0,
            "second pass moved {} vertices",
            second.pushed
        );
        assert_eq!(cloth.positions, after_first, "not idempotent");
    }

    #[test]
    fn a_deep_one_sided_overlap_resolves_without_distortion() {
        // The case the strain bound must NOT obstruct: the whole garment is deep inside the
        // body but every vertex needs the same displacement, so translating it out costs
        // zero differential. 0.4 m is 68 authored edges' worth of travel — an absolute
        // per-vertex clamp would never have got there, which is why the bound is on the
        // difference between neighbours instead.
        let body = body_box();
        // Narrow enough that the top face is the nearest feature for EVERY vertex (0.4 m
        // up against 0.45 m sideways), so the whole sheet wants the same displacement.
        let mut cloth = cloth_grid("sheet", 7, 0.6, 0.05);
        let r = resolve_penetration(&body, &mut cloth, FIT_GAP_M).expect("resolve");
        assert_eq!(r.inside_after, 0, "per_pass {:?}", r.per_pass);
        assert!(
            r.check.edge_spread() < 1.0e-4,
            "a rigid translation distorted the sheet: {:?}",
            r.check
        );
        let up = cloth.positions[0][1] - 0.6;
        assert!(up > 0.4, "the sheet only rose {up} m");
    }

    #[test]
    fn resolve_penetration_reports_the_residual_it_leaves() {
        // The defect class this fixes: `converged()` used to mean "the last pass moved
        // nothing", which a wrong inside test makes vacuously true. It must mean "no
        // vertex is inside", and the depth must agree with it.
        let body = notched_prism();
        let mut cloth = cloth_grid("sheet", 3, 0.5, 0.02);
        // Sit the little sheet in the crease, inside the solid, where the old face-normal
        // test read one of its corners as already outside.
        for p in &mut cloth.positions {
            let q = in_notch_solid(180.0, 0.04);
            *p = [p[0] + q.x, q.y, p[2] + q.z];
        }
        let r = resolve_penetration(&body, &mut cloth, FIT_GAP_M).expect("resolve");
        assert_eq!(
            r.converged(),
            r.inside_after == 0,
            "converged() and inside_after disagree: {r:?}"
        );
        assert_eq!(r.inside_after == 0, r.deepest_inside_m == 0.0, "{r:?}");
        assert!(r.pushed > 0, "nothing was found inside the crease at all");
    }

    #[test]
    fn fitted_to_body_placement_corrects_what_as_authored_leaves_embedded() {
        let body = body_box();
        let cloth = cloth_grid("sheet", 7, 0.5, 0.3);

        let as_authored = drape_on_avatar(
            &body,
            &cloth,
            FabricPreset::Cotton,
            GarmentPlacement::AsAuthored,
        )
        .expect("as authored");
        let fitted = drape_on_avatar(
            &body,
            &cloth,
            FabricPreset::Cotton,
            GarmentPlacement::FittedToBody,
        )
        .expect("fitted");

        // Same object layout either way; only the cloth positions differ.
        let pick = |s: &Scene| {
            s.objects
                .iter()
                .find(|o| !o.is_static)
                .expect("a dynamic cloth object")
                .positions
                .clone()
        };
        let (a, f) = (pick(&as_authored), pick(&fitted));
        assert_eq!(a.len(), f.len());
        assert_ne!(a, f, "FittedToBody made no difference to the garment");
        // AsAuthored must still be the authored coordinates — the mode's whole contract.
        assert_eq!(a, cloth.positions);
    }

    #[test]
    fn placement_offset_never_translates_a_fitted_garment() {
        // FittedToBody corrects per-vertex; if it also translated, the correction would
        // be computed against a body the garment is no longer positioned against.
        let avatar = Aabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        };
        let garment = Aabb {
            min: [-0.3, 0.4, -0.3],
            max: [0.3, 0.6, 0.3],
        };
        assert_eq!(
            placement_offset(avatar, garment, GarmentPlacement::FittedToBody),
            [0.0; 3]
        );
    }
}
