// SPDX-License-Identifier: GPL-2.0-or-later
//! The avatar the shell is showing: which body, which skin tone, the mesh the
//! 3D viewport draws and the silhouette ghosted behind the 2D canvas.
//!
//! Both views read the **same** [`nguo_viewport3d::AvatarMesh`], so
//! UX_SPEC §2's "one sprite derived from the same avatar the drape uses, so it
//! cannot drift out of agreement" is structural rather than a convention
//! someone has to remember.
//!
//! Everything here is `nguo-avatar`'s data — the catalogue, the skin library,
//! the path resolution and the glTF load all belong to that crate. This module
//! only holds the *selection* and turns the result into the two shapes the
//! shell needs.
//!
//! # Licensing
//! The bundle is **CC BY 4.0 throughout** and all three obligations are live
//! wherever this picker ships: the bodies are Seede XR's own work, the skin
//! textures derive from texturecan.com / cgbookcase.com, and the ten
//! complexions are the Monk Skin Tone Scale. There is no unverified-provenance
//! blocker any more — `assets/avatars/LICENSE.md` §"Bodies: why the release
//! blocker is closed" settles it. [`nguo_avatar::ATTRIBUTION_NOTICE`] is the
//! consolidated notice that discharges all three, and `main::avatar_picker`
//! renders it.

use std::sync::Arc;

use nguo_avatar::{Avatar, AvatarError, Bundle, MonkTone, Tier};
use nguo_viewport3d::AvatarMesh;
use vello::kurbo::{BezPath, Point};

use crate::command::Entry;

/// Command id prefixes. Both are dispatched by `main::dispatch` and both are
/// in the palette catalogue, so the picker and `⌘K` route through one
/// registry (UX_SPEC §3.2).
///
/// The body suffix is the manifest key — `male` / `female` — and the tone
/// suffix is a Monk key — `MST01`..`MST10`.
pub const BODY_COMMAND: &str = "avatar.body.";
pub const TONE_COMMAND: &str = "avatar.tone.";

/// Horizontal slices used to trace the front-view silhouette.
pub const SILHOUETTE_BANDS: usize = 64;

/// Canvas world units per metre.
///
/// **Millimetres.** UX_SPEC §1: "Geometry is millimetres everywhere internally
/// (`nguo-measure::Length`)", and the sample document agrees — its seams read
/// `380 mm` and its display unit is `Millimeter`. `nguo_canvas::view`'s doc
/// comment says "centimetres, eventually", but that crate draws a hardcoded
/// demo blob and no document geometry, so it is not evidence. If the pattern
/// ever lands in the canvas at cm, this constant is the single place to fix.
const MM_PER_M: f64 = 1000.0;

/// Logical px of stature the 2D canvas opens at.
///
/// The default layout gives the pattern pane 33 % of a 1440 px window — about
/// 475 logical px — and a T-posed body is roughly as wide as it is tall, so
/// this is bounded by the pane's *width*, not its height.
const CANVAS_FIT_PX: f64 = 420.0;

/// Canvas zoom (logical px per millimetre) that frames a body `stature_m`
/// tall. At `View`'s default zoom of 1.0 a 1.9 m body is 1 900 px of ghost in
/// a 790 px pane, i.e. an opaque wash — this is what stops that.
pub fn fit_zoom(stature_m: f64) -> f64 {
    CANVAS_FIT_PX / (stature_m.max(0.1) * MM_PER_M)
}

/// The opened bundle plus the current selection.
#[derive(Debug)]
pub struct Avatars {
    bundle: Bundle,
    /// Manifest key of the shown body. Always resolvable: only [`Self::select`]
    /// writes it, and only after the key has resolved.
    selected: String,
    /// Monk key of the shown complexion, likewise always resolvable.
    tone: String,
    base_color: [f32; 4],
    mesh: Arc<AvatarMesh>,
    silhouette: BezPath,
}

impl Avatars {
    /// Open the bundle and load its first body at the tone the shipped meshes
    /// were baked with.
    ///
    /// "First" is the first manifest key, and the manifest's `avatars` is a
    /// `BTreeMap`, so that is `female` and it is deterministic rather than
    /// whatever order a JSON object happened to be written in.
    pub fn load() -> Result<Avatars, AvatarError> {
        let span = tracing::info_span!("avatar_startup");
        let _enter = span.enter();

        let bundle = Bundle::find()?;

        // `BAKED_TONE` — the tone already in the shipped GLB materials. Starting
        // anywhere else would put the first frame and the picker's highlighted
        // swatch in disagreement.
        let tone = bundle.skin().baked_tone()?;
        let base_color = tone.factor().to_array();
        let tone = tone.key.clone();

        let (key, avatar) = bundle
            .avatars()
            .next()
            .ok_or_else(|| AvatarError::UnknownAvatar {
                id: "<first>".to_owned(),
                known: String::new(),
            })?;
        let key = key.to_owned();
        let mesh = load_mesh(&bundle, &key, avatar, bundle.manifest().default_tier)?;
        let silhouette = silhouette(&mesh, SILHOUETTE_BANDS);

        Ok(Avatars {
            selected: key,
            tone,
            base_color,
            mesh,
            silhouette,
            bundle,
        })
    }

    /// The catalogue as `(manifest key, avatar)` — two bodies, `female` then
    /// `male`.
    pub fn entries(&self) -> Vec<(&str, &Avatar)> {
        self.bundle.avatars().collect()
    }

    /// Manifest key of the shown body: what a `avatar.body.*` command carries.
    pub fn selected_key(&self) -> &str {
        &self.selected
    }

    pub fn selected(&self) -> &Avatar {
        // The invariant `select` maintains: `selected` is always a live key.
        self.bundle
            .avatar(&self.selected)
            .expect("the selected key came from the catalogue")
    }

    /// The ten tones **in Monk scale order**, which is the order to show them
    /// in — the scale is not a lightness ramp (INTEGRATION.md §4).
    pub fn tones(&self) -> &[MonkTone] {
        self.bundle.skin().tones()
    }

    /// Monk key of the shown complexion, e.g. `"MST05"`.
    pub fn tone(&self) -> &str {
        &self.tone
    }

    /// `"MST05 · Golden"`, for the read-out under the swatches.
    pub fn tone_label(&self) -> String {
        match self.bundle.skin().tone(&self.tone) {
            Ok(tone) => format!("{} · {}", tone.key, tone.name),
            Err(_) => self.tone.clone(),
        }
    }

    /// The mesh and the linear `baseColorFactor`, for the 3D viewport.
    pub fn viewport(&self) -> nguo_viewport3d::Avatar {
        nguo_viewport3d::Avatar {
            mesh: Arc::clone(&self.mesh),
            base_color: self.base_color,
        }
    }

    /// The front-view outline, for the 2D canvas. Traced from the same mesh
    /// [`Self::viewport`] hands the GPU.
    pub fn silhouette(&self) -> &BezPath {
        &self.silhouette
    }

    /// Switch body. Reloads the mesh and re-traces the silhouette together, so
    /// the two can never describe different avatars.
    pub fn select(&mut self, key: &str) -> Result<(), AvatarError> {
        if key == self.selected {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let tier = self.bundle.manifest().default_tier;
        // Borrows of `self.bundle` end with this statement, so the writes below
        // do not need a second lookup.
        let mesh = {
            let avatar = self.bundle.avatar(key)?;
            load_mesh(&self.bundle, key, avatar, tier)?
        };

        self.silhouette = silhouette(&mesh, SILHOUETTE_BANDS);
        self.mesh = mesh;
        self.selected = key.to_owned();
        tracing::info!(
            target: "nguo_app",
            avatar = key,
            ms = started.elapsed().as_secs_f64() * 1000.0,
            "avatar switched"
        );
        Ok(())
    }

    /// Switch complexion — **route A of INTEGRATION.md §4**, the recommended
    /// one: a four-float `baseColorFactor`, not a 2K texture reload. The mesh
    /// is untouched and the viewport will not re-upload it.
    pub fn set_tone(&mut self, key: &str) -> Result<(), AvatarError> {
        self.base_color = self.bundle.skin().tone(key)?.factor().to_array();
        self.tone = key.to_owned();
        tracing::info!(target: "nguo_app", tone = key, factor = ?self.base_color, "skin tone set");
        Ok(())
    }

    /// Palette rows for the two pickers.
    ///
    /// `shortcut: None` — palette-only, which is what `command::Entry`
    /// documents that to mean. These are appended to the keymap-derived
    /// catalogue rather than added to the keymap: 12 bodies-and-tones would be
    /// 12 accelerators nobody asked for.
    pub fn command_entries(&self) -> Vec<Entry> {
        self.entries()
            .iter()
            .map(|(key, avatar)| Entry {
                id: format!("{BODY_COMMAND}{key}").into(),
                label: format!("Avatar · {}", avatar.name),
                shortcut: None,
            })
            .chain(self.tones().iter().map(|t| Entry {
                id: format!("{TONE_COMMAND}{}", t.key).into(),
                // Both the key and the name, so `MST07` and `Amber` both find it.
                label: format!("Avatar · Skin tone {} {}", t.key, t.name),
                shortcut: None,
            }))
            .collect()
    }
}

/// Load one tier of one body.
///
/// The default tier is **LOD2** — the plain `nguo_<sex>.glb`, *not* `_lod0`,
/// which is an archival master the bundle does not even ship.
///
/// Measured on this machine, dev profile (`optimized + debuginfo`), three
/// repetitions of each, via `nguo_avatar::Bundle::load_mesh`:
///
/// | asset | load | triangles |
/// |---|---|---|
/// | `GlbRuntime` (LOD2) | 131.7 – 142.6 ms | 49 998 / 50 000 |
/// | `GlbCollision` (COL) | 6.1 – 9.9 ms | 15 000 |
///
/// LOD2 anyway: the cost is a one-off load, not a per-frame one, 50 k triangles
/// is nothing to draw, and the 130–140 ms is spent behind the 4-second splash.
/// COL is a decimated *collider* for the solver, and this is a display feature.
///
// ponytail: the load is synchronous, so switching body costs a ~135 ms hitch
// (about sixteen frames at 120 Hz). Move it to `Task::perform` if that ever
// reads as a stall rather than a beat.
fn load_mesh(
    bundle: &Bundle,
    key: &str,
    avatar: &Avatar,
    tier: Tier,
) -> Result<Arc<AvatarMesh>, AvatarError> {
    let obj = bundle.load_mesh(avatar, tier.glb())?;
    tracing::info!(
        target: "nguo_app",
        avatar = key,
        tier = tier.as_str(),
        vertices = obj.positions.len(),
        triangles = obj.tris.len(),
        stature_m = avatar.stature_m,
        "avatar mesh loaded"
    );
    Ok(Arc::new(AvatarMesh::new(
        &obj.positions,
        &obj.normals,
        &obj.tris,
    )))
}

/// Trace the front-view outline of `mesh` as a closed path in canvas world
/// space: millimetres, **+y downward**, centred on the origin.
///
/// The mesh is sliced into `bands` horizontal strips and each strip
/// contributes its widest left and right vertex, which for a standing body is
/// its outline. It is a per-row hull, so a T-pose reads as a cross and the gap
/// between an arm and the torso is filled in — correct for a ghost that only
/// has to say "this is where the body is", wrong if anyone ever wants a
/// cut-out.
///
// ponytail: per-scanline min/max, not a concave hull. Upgrade to a real
// outline (or an alpha shape) only if someone needs the armpit gap.
pub fn silhouette(mesh: &AvatarMesh, bands: usize) -> BezPath {
    let mut path = BezPath::new();
    let Some((lo, hi)) = mesh.bounds() else {
        return path;
    };
    let span = hi.y - lo.y;
    if bands < 2 || span <= 0.0 || !span.is_finite() {
        return path;
    }

    let mut extents = vec![None::<(f32, f32)>; bands];
    for vertex in &mesh.vertices {
        let t = (vertex.position[1] - lo.y) / span;
        let band = ((t * bands as f32) as usize).min(bands - 1);
        let x = vertex.position[0];
        extents[band] = Some(match extents[band] {
            None => (x, x),
            Some((min, max)) => (min.min(x), max.max(x)),
        });
    }

    let filled: Vec<(usize, (f32, f32))> = extents
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.map(|e| (i, e)))
        .collect();
    if filled.len() < 2 {
        return path;
    }

    // Metres, +y up, feet at y == lo.y -> millimetres, +y down, centred.
    let mid = (lo + hi) * 0.5;
    let to_canvas =
        |x: f32, y: f32| Point::new((x - mid.x) as f64 * MM_PER_M, (mid.y - y) as f64 * MM_PER_M);
    let band_centre = |i: usize| lo.y + span * (i as f32 + 0.5) / bands as f32;

    // Down the right edge from the crown, then back up the left. Highest band
    // index is the top of the body, which is the *smallest* canvas y.
    for (n, (i, (_, max))) in filled.iter().rev().enumerate() {
        let at = to_canvas(*max, band_centre(*i));
        if n == 0 {
            path.move_to(at);
        } else {
            path.line_to(at);
        }
    }
    for (i, (min, _)) in &filled {
        path.line_to(to_canvas(*min, band_centre(*i)));
    }
    path.close_path();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_viewport3d::Vertex;
    use vello::kurbo::Shape;

    /// A box `w` x `h` metres, feet at y = 0, as 8 loose vertices. No
    /// triangles: the silhouette only reads positions.
    fn box_mesh(w: f32, h: f32) -> AvatarMesh {
        let mut vertices = Vec::new();
        for x in [-w / 2.0, w / 2.0] {
            for y in [0.0, h] {
                for z in [-0.1_f32, 0.1] {
                    vertices.push(Vertex {
                        position: [x, y, z],
                        normal: [0.0, 1.0, 0.0],
                    });
                }
            }
        }
        AvatarMesh {
            vertices,
            indices: Vec::new(),
        }
    }

    #[test]
    fn the_silhouette_is_the_body_in_millimetres_centred_on_the_origin() {
        let path = silhouette(&box_mesh(0.5, 1.9), 8);
        let bbox = path.bounding_box();

        // 0.5 m wide, 1.9 m tall -> 500 x 1900 canvas units, centred.
        assert!((bbox.width() - 500.0).abs() < 1e-6, "{bbox:?}");
        assert!((bbox.x0 + 250.0).abs() < 1e-6, "{bbox:?}");
        // The band centres are half a band inside the extremes, so the traced
        // height is one band short of the full 1900 — but still centred on 0.
        assert!((bbox.y0 + bbox.y1).abs() < 1e-6, "{bbox:?}");
        assert!(bbox.height() > 1900.0 * 0.8, "{bbox:?}");
        assert!(bbox.height() <= 1900.0 + 1e-6, "{bbox:?}");
    }

    /// The zoom the canvas opens at must actually put the figure inside the
    /// default layout's 2D pane — the whole reason it is not `View::default`.
    #[test]
    fn the_fit_zoom_puts_the_whole_figure_in_the_default_pane() {
        // 33 % of a 1440 px window, less the divider (`shell::default_layout`).
        let pane = 1440.0 * 0.33;
        // The two shipped statures are 1.830 m and 1.90477 m.
        for stature in [1.2_f64, 1.83, 1.90477, 2.1] {
            let on_screen = stature * MM_PER_M * fit_zoom(stature);
            assert!(on_screen <= pane, "{stature} m fills {on_screen:.0} px");
            assert!(
                on_screen > pane * 0.5,
                "{stature} m is only {on_screen:.0} px"
            );
        }
        // A degenerate stature must not divide by zero into an infinite zoom.
        assert!(fit_zoom(0.0).is_finite());
    }

    /// A tapered body must not come out as its bounding box: the top bands
    /// have to be narrower than the bottom ones, which is the entire reason
    /// this is a per-band trace and not a rectangle.
    #[test]
    fn a_tapered_body_keeps_its_taper() {
        let mut vertices = Vec::new();
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let half = 0.4 * (1.0 - t) + 0.05 * t; // wide at the feet
            for x in [-half, half] {
                vertices.push(Vertex {
                    position: [x, t * 1.8, 0.0],
                    normal: [0.0, 1.0, 0.0],
                });
            }
        }
        let path = silhouette(
            &AvatarMesh {
                vertices,
                indices: Vec::new(),
            },
            8,
        );

        // Widest x in the top half vs the bottom half of the traced outline.
        let widest = |top: bool| {
            path.elements()
                .iter()
                .filter_map(|e| e.end_point())
                .filter(|p| (p.y < 0.0) == top)
                .fold(0.0_f64, |acc, p| acc.max(p.x.abs()))
        };
        assert!(
            widest(true) < widest(false) * 0.6,
            "top {} is not narrower than bottom {}",
            widest(true),
            widest(false),
        );
    }

    #[test]
    fn a_degenerate_mesh_produces_an_empty_path_rather_than_a_panic() {
        let empty = AvatarMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
        assert!(silhouette(&empty, SILHOUETTE_BANDS).is_empty());
        // Zero height: every vertex lands in one band.
        assert!(silhouette(&box_mesh(0.5, 0.0), 8).is_empty());
        // Fewer than two bands cannot describe an outline.
        assert!(silhouette(&box_mesh(0.5, 1.9), 1).is_empty());
        assert!(silhouette(&box_mesh(0.5, 1.9), 0).is_empty());
    }

    #[test]
    fn two_different_bodies_give_two_different_silhouettes() {
        let a = silhouette(&box_mesh(0.5, 1.83), 8).bounding_box();
        let b = silhouette(&box_mesh(0.7, 1.90477), 8).bounding_box();
        assert_ne!(a, b);
    }
}

/// Tests that need `assets/avatars/`, which is not in every clone. Each one
/// skips loudly rather than failing when the bundle is absent, and asserts a
/// floor against the shipped bundle so it cannot pass by finding nothing.
#[cfg(test)]
mod bundle_tests {
    use super::*;
    use vello::kurbo::Shape;

    /// What the shipped bundle contains. A test that walks the catalogue and
    /// finds one avatar is a broken bundle, not a passing test.
    const EXPECTED_AVATARS: usize = 2;
    const EXPECTED_TONES: usize = 10;
    /// LOD2 is 49 998 / 50 000 triangles. A floor, not the exact number — a
    /// future re-decimation may move it, a *tier* swap must not go unnoticed.
    const MIN_TRIANGLES: usize = 40_000;

    fn bundle() -> Option<Avatars> {
        match Avatars::load() {
            Ok(avatars) => Some(avatars),
            Err(error) => {
                {
                    use std::io::Write;
                    // fd 2 directly, NOT `eprintln!`: libtest captures the print
                    // macros, so a macro'd skip notice is invisible without
                    // --nocapture — i.e. not loud at all. Same fix as
                    // engine-api/tests/avatar_bundle.rs.
                    let _ = writeln!(std::io::stderr(), "SKIPPING: no avatar bundle ({error})");
                    assert!(
                        std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none(),
                        "MGUSO_REQUIRE_AVATAR_BUNDLE is set but the avatar bundle is absent"
                    );
                }
                None
            }
        }
    }

    /// The runtime tier is LOD2, carried by the plain `nguo_<sex>.glb`. LOD0 is
    /// declared by the manifest and **not shipped**, so a bundle that made it
    /// the default would be a startup toast — worth failing a test over.
    #[test]
    fn the_default_tier_is_the_shipped_runtime_glb() {
        let Some(avatars) = bundle() else { return };
        let tier = avatars.bundle.manifest().default_tier;
        assert_eq!(tier, Tier::Lod2);
        assert_eq!(tier.glb(), nguo_avatar::Asset::GlbRuntime);
        for (_, avatar) in avatars.entries() {
            assert!(avatars.bundle.has_tier(avatar, tier));
            assert!(
                !avatars.bundle.has_tier(avatar, Tier::Lod0),
                "LOD0 is archival; the bundle must not be shipping it",
            );
        }
    }

    #[test]
    fn every_body_and_every_tone_is_reachable_as_a_command() {
        let Some(avatars) = bundle() else { return };
        let entries = avatars.command_entries();
        assert_eq!(entries.len(), EXPECTED_AVATARS + EXPECTED_TONES);

        // Every generated id must survive a round trip through the same
        // prefix split `main::select_avatar` does, or the palette would list
        // commands that quietly do nothing.
        let mut avatars = avatars;
        let mut bodies = 0;
        let mut tones = 0;
        for entry in &entries {
            let id = entry.id.as_str();
            assert!(entry.shortcut.is_none(), "{id} claims an accelerator");
            match id.strip_prefix(BODY_COMMAND) {
                Some(body) => {
                    avatars.select(body).expect(id);
                    assert_eq!(avatars.selected_key(), body);
                    bodies += 1;
                }
                None => {
                    let tone = id
                        .strip_prefix(TONE_COMMAND)
                        .unwrap_or_else(|| panic!("{id} has neither picker prefix"));
                    avatars.set_tone(tone).expect(id);
                    assert_eq!(avatars.tone(), tone);
                    tones += 1;
                }
            }
        }
        assert_eq!((bodies, tones), (EXPECTED_AVATARS, EXPECTED_TONES));
    }

    /// The UX_SPEC §2 guarantee: the ghost is traced from the very mesh the
    /// viewport draws, so switching body moves both or neither.
    #[test]
    fn switching_body_moves_the_viewport_mesh_and_the_ghost_together() {
        let Some(mut avatars) = bundle() else { return };
        let keys: Vec<String> = avatars
            .entries()
            .iter()
            .map(|(k, _)| (*k).to_owned())
            .collect();
        assert_eq!(keys, vec!["female".to_owned(), "male".to_owned()]);

        let mut seen = Vec::new();
        for key in &keys {
            avatars.select(key).expect("a catalogue key must select");
            assert_eq!(avatars.selected_key(), key);

            let drawn = avatars.viewport().mesh;
            assert!(
                Arc::ptr_eq(&drawn, &avatars.mesh),
                "the viewport is being handed a different mesh from the ghost",
            );
            assert!(
                drawn.indices.len() >= 3 * MIN_TRIANGLES,
                "{key}: {} indices — is this still LOD2?",
                drawn.indices.len(),
            );

            // The traced outline is the body's real stature, to within the
            // half-band the trace loses at each end.
            let bbox = avatars.silhouette().bounding_box();
            let stature_mm = avatars.selected().stature_m * MM_PER_M;
            let band_mm = stature_mm / SILHOUETTE_BANDS as f64;
            assert!(
                (stature_mm - bbox.height()) < band_mm * 1.5,
                "{key}: silhouette {:.1} mm vs stature {stature_mm:.1} mm",
                bbox.height(),
            );
            seen.push(bbox);
        }

        // Two distinct bodies must give two distinct ghosts, or the silhouette
        // is not actually derived from the mesh. The female body is 7.5 cm
        // taller, so this is a real difference and not a rounding one.
        assert_eq!(seen.len(), EXPECTED_AVATARS);
        assert!(
            (seen[0].height() - seen[1].height()).abs() > 50.0,
            "{seen:?} — the two statures differ by 75 mm",
        );
    }

    /// Tone is a uniform, not a reload — and the value written is the
    /// `baseColorFactor`, never the swatch colour or the published linear one
    /// (INTEGRATION.md §4: three colours per tone, one of them a factor).
    #[test]
    fn switching_tone_rewrites_the_factor_and_leaves_the_mesh_alone() {
        let Some(mut avatars) = bundle() else { return };
        let before = avatars.viewport();

        let tones: Vec<(String, [f32; 3], [f32; 3])> = avatars
            .tones()
            .iter()
            .map(|t| (t.key.clone(), t.swatch.srgb, t.base_color_linear))
            .collect();
        assert_eq!(tones.len(), EXPECTED_TONES);

        let mut factors = Vec::new();
        for (key, srgb, published) in &tones {
            avatars.set_tone(key).expect("a scale key must apply");
            let after = avatars.viewport();
            assert!(
                Arc::ptr_eq(&before.mesh, &after.mesh),
                "changing skin tone re-uploaded the body",
            );
            assert_eq!(avatars.tone(), key);
            assert!(
                avatars.tone_label().contains(key),
                "{}",
                avatars.tone_label()
            );

            let rgb = [
                after.base_color[0],
                after.base_color[1],
                after.base_color[2],
            ];
            assert_ne!(&rgb, srgb, "{key}: swatch used as a factor");
            assert_ne!(
                &rgb, published,
                "{key}: the published linear tone used as a factor — the \
                 neutral-albedo-mean division was skipped",
            );
            assert_eq!(after.base_color[3], 1.0);
            factors.push(rgb);
        }
        factors.dedup();
        assert_eq!(factors.len(), EXPECTED_TONES, "two tones share a factor");
    }

    /// The startup state has to agree with the shipped meshes: they are baked
    /// at `MST05`, so the picker opens there.
    #[test]
    fn the_shell_opens_on_the_tone_the_meshes_were_baked_with() {
        let Some(avatars) = bundle() else { return };
        assert_eq!(avatars.tone(), nguo_avatar::BAKED_TONE);
        assert_eq!(avatars.selected_key(), "female");
        assert_eq!(avatars.tone_label(), "MST05 · Golden");
    }

    #[test]
    fn an_id_from_nowhere_is_an_error_that_lists_what_does_exist() {
        let Some(mut avatars) = bundle() else { return };

        let err = avatars.select("nguo_female_01").unwrap_err().to_string();
        assert!(err.contains("female"), "{err}");
        assert!(err.contains("male"), "{err}");

        let err = avatars.set_tone("MST-99").unwrap_err().to_string();
        assert!(err.contains("MST01"), "{err}");

        // Neither failure moved the selection.
        assert_eq!(avatars.selected_key(), "female");
        assert_eq!(avatars.tone(), "MST05");
    }
}
