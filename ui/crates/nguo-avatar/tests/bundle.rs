// SPDX-License-Identifier: GPL-2.0-or-later
//! Integration + regression tests against the real `assets/avatars/` bundle.
//!
//! The bundle is 227 MB and is not guaranteed to be in every checkout, so every
//! test here is gated on it being present and **skips loudly** if it is not.
//! Every test that walks something asserts a floor on how much it inspected, so
//! a bundle that silently shrinks fails rather than passing vacuously.

use nguo_avatar::{
    Asset, AvatarError, Bundle, Landmark, SkinMap, Tier, ATTRIBUTION_NOTICE, BAKED_TONE,
};
use std::collections::BTreeSet;

/// Floors the bundle is contracted to meet. A test that sees fewer than these
/// has not tested the bundle.
const EXPECTED_AVATARS: usize = 2;
const EXPECTED_TONES: usize = 10;
const EXPECTED_CLIPS: usize = 19;
const EXPECTED_BONES: usize = 65;
const EXPECTED_SHARED_MAPS: usize = 7;
/// 9 declared assets x 2 avatars.
const DECLARED_FILES: usize = 18;
/// Of those 18, the runtime subset actually on disk.
const PRESENT_FILES: usize = 8;

/// The assets this bundle ships. Everything else `files` declares is pipeline
/// output that is not redistributed. **If this list changes, the bundle
/// changed** — that is the point of pinning it.
const SHIPPED: [Asset; 4] = [
    Asset::GlbRuntime,
    Asset::GlbLod1,
    Asset::GlbCollision,
    Asset::Vrm,
];

/// `Some(bundle)`, or `None` after printing why it is skipping.
fn bundle() -> Option<Bundle> {
    match Bundle::find() {
        Ok(b) => Some(b),
        Err(AvatarError::BundleNotFound(path)) => {
            {
                use std::io::Write;
                // fd 2 directly, NOT `eprintln!`: libtest captures the print
                // macros, so a macro'd skip notice is invisible without
                // --nocapture — i.e. not loud at all. Same fix as
                // engine-api/tests/avatar_bundle.rs.
                let _ = writeln!(
                    std::io::stderr(),
                    "SKIPPING nguo-avatar bundle test: no avatar bundle at {}. \
                 Install the nguo avatar bundle to assets/avatars/, or set \
                 NGUO_AVATAR_BUNDLE.",
                    path.display()
                );
                assert!(
                    std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none(),
                    "MGUSO_REQUIRE_AVATAR_BUNDLE is set but the avatar bundle is absent"
                );
            }
            None
        }
        Err(e) => panic!("bundle present but unusable: {e}"),
    }
}

macro_rules! bundle_or_skip {
    () => {
        match bundle() {
            Some(b) => b,
            None => return,
        }
    };
}

fn read_data(b: &Bundle, rel: &str) -> serde_json::Value {
    let path = b.root().join("data").join(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn f64_at(v: &serde_json::Value, path: &[&str]) -> f64 {
    let mut cur = v;
    for key in path {
        cur = cur
            .get(key)
            .unwrap_or_else(|| panic!("missing {} in {cur}", path.join(".")));
    }
    cur.as_f64()
        .unwrap_or_else(|| panic!("{} is not a number: {cur}", path.join(".")))
}

#[test]
fn manifest_parses_and_describes_the_shipped_bundle() {
    let b = bundle_or_skip!();
    let m = b.manifest();

    assert_eq!(m.avatars.len(), EXPECTED_AVATARS, "avatar count");
    let keys: Vec<_> = b.avatars().map(|(k, _)| k).collect();
    assert_eq!(keys, vec!["female", "male"]);

    // Conventions: metres, glTF space, 30 fps, two UV sets.
    let c = &m.conventions;
    assert_eq!(c.units, "metres");
    assert!(c.is_gltf_space(), "{:?}", c.gltf_vrm);
    assert_eq!(c.fps, 30.0);
    assert!(c.uv0.contains("tiling"), "uv0 = {:?}", c.uv0);
    assert!(c.uv1.contains("Atlas"), "uv1 = {:?}", c.uv1);
    // The three axis blocks deliberately disagree — Blender is Z-up.
    assert_eq!(c.blender.up, "+Z");
    assert_eq!(c.blender.forward, "-Y");
    assert_eq!(c.unity_fbx.up, "+Y");
    assert_eq!(c.unity_fbx.scale, Some(1.0));
    assert_ne!(c.blender.up, c.gltf_vrm.up);

    // The default tier is LOD2, and LOD2 is the *runtime* GLB.
    assert_eq!(m.default_tier, Tier::Lod2);
    assert_eq!(Tier::Lod2.glb(), Asset::GlbRuntime);
    assert_eq!(m.skin.tone_count, EXPECTED_TONES);

    let mut checked = 0usize;
    for (key, a) in b.avatars() {
        assert!(!a.name.is_empty(), "{key} has no name");
        assert!(
            (1.5..2.2).contains(&a.stature_m),
            "{key} stature {} m",
            a.stature_m
        );
        assert_eq!(a.skeleton.bones, EXPECTED_BONES, "{key} bones");
        assert_eq!(a.skeleton.max_influences_per_vertex, 4, "{key} influences");
        assert_eq!(a.skeleton.deforming_bones, 38, "{key} deforming bones");
        assert_eq!(a.animations.len(), EXPECTED_CLIPS, "{key} clips");
        assert_eq!(a.tiers.len(), Tier::ALL.len(), "{key} tiers");

        for t in Tier::ALL {
            let info = a
                .tier(t)
                .unwrap_or_else(|| panic!("{key} has no {} tier", t.as_str()));
            assert!(info.verts > 0 && info.tris > 0, "{key} {info:?}");
            assert!(info.object.contains("Nguo_"), "{key} {info:?}");
        }
        // Vertex counts fall strictly with the tier, and LOD2 is the runtime one.
        let verts = |t: Tier| a.tier(t).unwrap().verts;
        assert!(verts(Tier::Lod0) > verts(Tier::Lod1));
        assert!(verts(Tier::Lod1) > verts(Tier::Lod2));
        assert!(verts(Tier::Lod2) > verts(Tier::Col));
        assert!(
            a.tier(Tier::Lod2).unwrap().role.contains("default runtime"),
            "{key} LOD2 role = {:?}",
            a.tier(Tier::Lod2).unwrap().role
        );
        assert!((24_000..26_000).contains(&verts(Tier::Lod2)), "{key} LOD2");
        assert!((7_000..8_000).contains(&verts(Tier::Col)), "{key} COL");

        // Clips are named, unique, and all 30 fps.
        let names: BTreeSet<_> = a.clip_names().collect();
        assert_eq!(names.len(), EXPECTED_CLIPS, "{key} duplicate clip names");
        for clip in &a.animations {
            assert!(clip.frames > 0, "{key} {clip:?}");
            let expected = f64::from(clip.frames) / 30.0;
            assert!(
                (clip.duration_s - expected).abs() < 0.01,
                "{key} {} is {} frames but {} s",
                clip.action,
                clip.frames,
                clip.duration_s
            );
        }
        assert!(a.clip("f_Catwalk_Walk").is_some(), "{key} lacks the walk");
        assert!(a.clip("no_such_clip").is_none());
        checked += 1;
    }
    assert_eq!(checked, EXPECTED_AVATARS);
}

/// The hazard: the manifest declares the full pipeline output, the bundle ships
/// the runtime subset. This pins **exactly** which declared paths exist, so a
/// bundle that starts shipping LOD0 or the FBX set tells us instead of silently
/// changing behaviour.
#[test]
fn declared_but_absent_files_are_pinned_and_reported_absent() {
    let b = bundle_or_skip!();
    let mut declared = 0usize;
    let mut present = 0usize;
    let mut absent = Vec::new();

    for (key, a) in b.avatars() {
        for asset in Asset::ALL {
            declared += 1;
            let expect_present = SHIPPED.contains(&asset);
            let on_disk = b.has_asset(a, asset);
            assert_eq!(
                on_disk,
                expect_present,
                "{key} {}: expected present={expect_present}, found {on_disk} at {}",
                asset.as_str(),
                b.declared_path(a, asset).display()
            );

            match b.asset_path(a, asset) {
                Ok(path) => {
                    assert!(
                        expect_present,
                        "{key} {} resolved unexpectedly",
                        asset.as_str()
                    );
                    assert!(path.is_file(), "{} is not a file", path.display());
                    assert!(
                        std::fs::metadata(&path).unwrap().len() > 1_000_000,
                        "{} is suspiciously small",
                        path.display()
                    );
                    present += 1;
                }
                Err(AvatarError::AssetAbsent { asset: name, .. }) => {
                    assert!(
                        !expect_present,
                        "{key} {name} reported absent but should ship"
                    );
                    absent.push(format!("{key}.{name}"));
                }
                Err(e) => panic!("{key} {}: wrong error {e}", asset.as_str()),
            }
        }
        // The tier question, answered per tier: LOD0 is archival, not shipped.
        assert!(!b.has_tier(a, Tier::Lod0), "{key} unexpectedly ships LOD0");
        assert!(b.has_tier(a, Tier::Lod1), "{key} lacks LOD1");
        assert!(b.has_tier(a, Tier::Lod2), "{key} lacks the runtime tier");
        assert!(b.has_tier(a, Tier::Col), "{key} lacks the collider");
        assert!(matches!(
            b.tier_path(a, Tier::Lod0),
            Err(AvatarError::AssetAbsent { .. })
        ));
        assert!(b.tier_path(a, b.manifest().default_tier).is_ok());
    }

    assert_eq!(declared, DECLARED_FILES, "declared path count");
    assert_eq!(present, PRESENT_FILES, "present path count");
    assert_eq!(absent.len(), DECLARED_FILES - PRESENT_FILES);
    assert_eq!(
        absent,
        vec![
            "female.blend",
            "female.glb.lod0",
            "female.unity_fbx.runtime",
            "female.unity_fbx.lod1",
            "female.unity_fbx.collision",
            "male.blend",
            "male.glb.lod0",
            "male.unity_fbx.runtime",
            "male.unity_fbx.lod1",
            "male.unity_fbx.collision",
        ],
        "the set of declared-but-absent files changed"
    );
}

#[test]
fn an_unknown_avatar_lists_what_exists() {
    let b = bundle_or_skip!();
    let err = b.avatar("nonbinary").unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, AvatarError::UnknownAvatar { .. }), "{msg}");
    assert!(msg.contains("male") && msg.contains("female"), "{msg}");
}

#[test]
fn all_ten_tones_resolve_to_both_a_factor_and_a_texture() {
    let b = bundle_or_skip!();
    let skin = b.skin();
    assert_eq!(skin.tones().len(), EXPECTED_TONES);
    assert_eq!(skin.resolution, 2048);
    assert_eq!(skin.tile_metres, 0.20);
    assert_eq!(skin.uv_layer, "UVMap");
    assert_eq!(skin.material_constants.metallic, 0.0);
    assert_eq!(skin.material_constants.ior, 1.44);
    assert_eq!(skin.roughness_range, [0.35, 0.75]);
    assert!(skin.credit().contains("Monk"));
    assert!(ATTRIBUTION_NOTICE.contains("CC BY 4.0"));

    let mut checked = 0usize;
    let mut last_weight = f32::INFINITY;
    for (i, tone) in skin.tones().iter().enumerate() {
        assert_eq!(tone.key, format!("MST{:02}", i + 1), "scale order");
        assert!(!tone.name.is_empty());
        assert!(
            matches!(
                tone.undertone.as_str(),
                "cool" | "neutral" | "warm" | "olive"
            ),
            "{} undertone {:?}",
            tone.key,
            tone.undertone
        );

        // Route A: a linear factor, always <= 1.
        let f = tone.factor().to_array();
        assert_eq!(f[3], 1.0, "{} alpha", tone.key);
        for ch in &f[..3] {
            assert!(*ch > 0.0 && *ch <= 1.0, "{} factor {f:?}", tone.key);
        }
        // The factor is the published linear colour divided by the neutral
        // albedo mean (~0.93), so it is brighter. Never interchangeable.
        assert!(
            f[0] > tone.base_color_linear[0],
            "{} factor is not pre-divided: {f:?} vs {:?}",
            tone.key,
            tone.base_color_linear
        );
        // Three genuinely different colours: sRGB swatch, published linear, and
        // the factor. (The factor is not bounded by the swatch — dividing by a
        // 0.93 albedo mean lifts MST01's factor above its own sRGB value.)
        assert!(
            tone.swatch.srgb[0] > tone.base_color_linear[0],
            "{}",
            tone.key
        );
        assert_ne!(tone.swatch.srgb[0], f[0], "{} swatch == factor", tone.key);
        assert!(tone.swatch.hex.starts_with('#') && tone.swatch.hex.len() == 7);

        // Route B: a baked albedo on disk.
        let albedo = b
            .tone_albedo(tone)
            .unwrap_or_else(|e| panic!("{}: {e}", tone.key));
        assert!(albedo.is_file());
        assert!(
            albedo.to_string_lossy().contains(&tone.key),
            "{} -> {}",
            tone.key,
            albedo.display()
        );

        // Melanin absorbs scattered light, so scatter weight falls down the
        // scale — but the scale is not a lightness ramp: MST03 is lighter than
        // MST02, and its weight is correspondingly higher. So the fall is only
        // monotone from MST03 on.
        assert!(
            (0.0..1.0).contains(&tone.subsurface_weight),
            "{} subsurface weight {}",
            tone.key,
            tone.subsurface_weight
        );
        if i >= 3 {
            assert!(
                tone.subsurface_weight < last_weight,
                "{} subsurface weight {} not below {last_weight}",
                tone.key,
                tone.subsurface_weight
            );
        }
        last_weight = tone.subsurface_weight;
        checked += 1;
    }
    assert_eq!(checked, EXPECTED_TONES);
    // Ends of the scale, from INTEGRATION.md §4.
    assert!(skin.tones()[0].subsurface_weight > 0.32);
    assert!(skin.tones()[EXPECTED_TONES - 1].subsurface_weight < 0.06);
    assert_eq!(skin.baked_tone().unwrap().key, BAKED_TONE);
    assert_eq!(skin.baked_tone().unwrap().name, "Golden");

    let err = skin.tone("MST-7").unwrap_err();
    assert!(matches!(err, AvatarError::UnknownTone { .. }), "{err}");
    assert!(err.to_string().contains("MST05"), "{err}");
}

#[test]
fn every_shared_skin_map_is_on_disk() {
    let b = bundle_or_skip!();
    let mut checked = 0usize;
    for map in SkinMap::ALL {
        let path = b
            .texture(map)
            .unwrap_or_else(|e| panic!("{}: {e}", map.as_str()));
        assert!(path.is_file(), "{}", path.display());
        assert!(
            std::fs::metadata(&path).unwrap().len() > 100_000,
            "{} is suspiciously small",
            path.display()
        );
        checked += 1;
    }
    assert_eq!(checked, EXPECTED_SHARED_MAPS);
    // Only the albedo is colour data; the rest must not be uploaded as sRGB.
    assert!(SkinMap::AlbedoNeutral.is_srgb());
    assert!(!SkinMap::NormalOpenGl.is_srgb());
}

#[test]
fn measurements_convert_exactly_and_land_in_human_ranges() {
    let b = bundle_or_skip!();
    let mut checked = 0usize;
    for (key, a) in b.avatars() {
        let m = b.measurements(a);
        // Conversion is exact, and stature comes from the manifest not a guess.
        assert_eq!(m.stature.mm(), a.stature_m * 1000.0, "{key} stature");
        assert_eq!(m.arm_span.mm(), a.arm_span_m * 1000.0, "{key} arm span");
        assert_eq!(m.inseam.mm(), a.skeleton.inseam_m * 1000.0, "{key} inseam");
        assert_eq!(
            m.hip_height.mm(),
            a.skeleton.hip_height_m * 1000.0,
            "{key} hip height"
        );
        assert!(
            (1500.0..2200.0).contains(&m.stature.mm()),
            "{key} stature {} mm",
            m.stature.mm()
        );
        // Runway proportions, but still human: arm span within 10 % of stature.
        assert!(
            (m.arm_span.mm() / m.stature.mm() - 1.0).abs() < 0.10,
            "{key} span/stature {}",
            m.arm_span.mm() / m.stature.mm()
        );
        assert!(m.inseam < m.hip_height, "{key} inseam above the hip joint");

        assert_eq!(m.girths().count(), Landmark::ALL.len(), "{key} girth count");
        for l in Landmark::ALL {
            let g = m
                .girth(l)
                .unwrap_or_else(|| panic!("{key} has no {}", l.key()));
            assert!(
                (300.0..1400.0).contains(&g.circumference.mm()),
                "{key} {} girth {} mm",
                l.key(),
                g.circumference.mm()
            );
            assert!(
                g.height.mm() > 0.0 && g.height < m.stature,
                "{key} {} height {} mm",
                l.key(),
                g.height.mm()
            );
            if let Some(frac) = g.height_fraction {
                let computed = g.height.mm() / m.stature.mm();
                assert!(
                    (frac - computed).abs() < 0.005,
                    "{key} {} fraction {frac} vs computed {computed}",
                    l.key()
                );
            }
            checked += 1;
        }
        // Ordering a tape measure would agree with.
        let girth = |l: Landmark| m.girth(l).unwrap().circumference;
        assert!(
            girth(Landmark::Waist) < girth(Landmark::Hip),
            "{key} waist<hip"
        );
        assert!(
            girth(Landmark::Waist) < girth(Landmark::Chest),
            "{key} waist<chest"
        );
        assert!(
            girth(Landmark::Calf) < girth(Landmark::Thigh),
            "{key} calf<thigh"
        );
        assert!(
            girth(Landmark::Neck) < girth(Landmark::Chest),
            "{key} neck<chest"
        );
        // Landmark heights descend down the body.
        let h = |l: Landmark| m.girth(l).unwrap().height;
        assert!(h(Landmark::Neck) > h(Landmark::Chest));
        assert!(h(Landmark::Chest) > h(Landmark::Waist));
        assert!(h(Landmark::Waist) > h(Landmark::Hip));
        assert!(h(Landmark::Thigh) > h(Landmark::Calf));
    }
    assert_eq!(checked, EXPECTED_AVATARS * Landmark::ALL.len());
}

/// Regression against the bundle's **own** expectations. `data/test_*.json` are
/// the reports the pipeline's Blender test suites wrote; the manifest summarises
/// them. Cross-checking the two catches a manifest that drifted from the
/// geometry, using an oracle this crate did not produce.
#[test]
fn manifest_verification_agrees_with_the_bundles_own_test_reports() {
    let b = bundle_or_skip!();
    let mut checks = 0usize;

    for (key, a) in b.avatars() {
        let integrity = read_data(&b, &format!("test_mesh_integrity_{key}.json"));
        let distortion = read_data(&b, &format!("test_animation_distortion_{key}.json"));
        assert_eq!(integrity["sex"].as_str(), Some(key));
        assert_eq!(distortion["sex"].as_str(), Some(key));

        let v = &a.verification;

        // LOD0 vertex fidelity: the manifest number is the report's number.
        assert_eq!(
            v.lod0_vs_source_worst_vertex_m,
            f64_at(&integrity, &["lod0_worst_vertex_deviation_m"]),
            "{key} lod0 deviation"
        );
        assert_eq!(
            a.tier(Tier::Lod0).unwrap().verts,
            f64_at(&integrity, &["lod0_verts"]) as u64,
            "{key} lod0 verts"
        );
        checks += 2;

        // Silhouette mismatch, per tier, and inside the report's own gate.
        for t in Tier::ALL {
            let reported = f64_at(
                &integrity,
                &["tiers", t.as_str(), "silhouette_mismatch_frac"],
            );
            let ours = *v
                .silhouette_mismatch_fraction
                .get(t.as_str())
                .unwrap_or_else(|| panic!("{key} manifest lacks {}", t.as_str()));
            assert_eq!(ours, reported, "{key} {} silhouette", t.as_str());
            let tol = f64_at(&integrity, &["tolerance_silhouette_frac", t.as_str()]);
            assert!(
                ours < tol,
                "{key} {} silhouette {ours} exceeds the bundle's own gate {tol}",
                t.as_str()
            );
            checks += 2;
        }

        // Posed collision-proxy deviation, in millimetres in both files.
        assert_eq!(
            v.collision_proxy_posed_deviation_mm,
            f64_at(&distortion, &["posed_proxy_deviation", "max_mm"]),
            "{key} proxy deviation"
        );
        assert!(
            v.collision_proxy_posed_deviation_mm < 6.0,
            "{key} proxy deviation {} mm",
            v.collision_proxy_posed_deviation_mm
        );
        checks += 2;

        // Animation edge growth, keyed by Blender object name, and every tier
        // inside the suite's 5x gate.
        let gate = f64_at(&distortion, &["gate_max_growth_ratio"]);
        let rows = distortion["tiers"].as_array().expect("tiers array");
        assert_eq!(rows.len(), Tier::ALL.len(), "{key} distortion rows");
        for row in rows {
            let object = row["tier"].as_str().expect("object name");
            let reported = f64_at(row, &["max_growth_over_median"]);
            let ours = *v
                .animation_edge_growth_over_median_edge
                .get(object)
                .unwrap_or_else(|| panic!("{key} manifest lacks {object}"));
            assert_eq!(ours, reported, "{key} {object} edge growth");
            assert!(
                ours < gate,
                "{key} {object} growth {ours} exceeds gate {gate}"
            );
            assert_eq!(
                row["worst_frame_edges_over_5x"].as_u64(),
                Some(0),
                "{key} {object}"
            );

            // The tier's vertex count is the same fact in three files.
            let tier = a
                .tiers
                .iter()
                .find(|t| t.object == object)
                .unwrap_or_else(|| panic!("{key} has no tier named {object}"));
            assert_eq!(
                tier.verts,
                f64_at(row, &["verts"]) as u64,
                "{key} {object} verts"
            );
            checks += 4;
        }
    }
    // 2 avatars x (2 + 8 + 2 + 16) checks.
    assert_eq!(checks, 56, "the oracle did fewer comparisons than expected");
}

/// The one test that actually reads geometry. The collider is the smallest GLB
/// in the bundle (~6.7 MB) and the tier the cloth solver consumes.
#[test]
fn collision_meshes_load_through_engine_io_as_finite_human_sized_bodies() {
    let b = bundle_or_skip!();
    let mut checked = 0usize;
    for (key, a) in b.avatars() {
        let obj = b
            .load_mesh(a, Asset::GlbCollision)
            .unwrap_or_else(|e| panic!("{key}: {e}"));

        let expected = a.tier(Tier::Col).unwrap();
        eprintln!(
            "{key} collider: {} verts / {} tris loaded, manifest says {} / {}",
            obj.positions.len(),
            obj.tris.len(),
            expected.verts,
            expected.tris
        );
        // Triangles are a topology invariant, so they match exactly. Vertices
        // do **not**: the manifest counts Blender mesh vertices, while glTF
        // duplicates them along every UV and normal seam (3 396 UV islands), so
        // the loaded count is higher. Never assert glTF verts == manifest verts.
        assert_eq!(obj.tris.len() as u64, expected.tris, "{key} tris");
        assert!(
            obj.positions.len() as u64 >= expected.verts,
            "{key} loaded fewer verts than the manifest's {}",
            expected.verts
        );
        assert!(
            obj.positions.len() as u64 <= expected.verts * 2,
            "{key} loaded {} verts, more than twice the manifest's {} — seam \
             duplication should not double the mesh",
            obj.positions.len(),
            expected.verts
        );

        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &obj.positions {
            for i in 0..3 {
                assert!(p[i].is_finite(), "{key} non-finite coordinate {p:?}");
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        for (i, s) in size.iter().enumerate() {
            assert!(*s > 0.01, "{key} degenerate bbox on axis {i}: {size:?}");
        }

        // Metres, +Y up: the bbox height is the stature, and the proxy tracks
        // the render mesh to a few millimetres, so allow 20 mm.
        let stature = a.stature_m as f32;
        assert!(
            (size[1] - stature).abs() < 0.02,
            "{key} bbox height {} m vs stature {stature} m",
            size[1]
        );
        // Feet on the ground plane, midline at X = 0.
        assert!(lo[1].abs() < 0.02, "{key} floor at Y = {}", lo[1]);
        assert!(
            (lo[0] + hi[0]).abs() < 0.05,
            "{key} not centred on X: {lo:?} {hi:?}"
        );
        // Every triangle indexes a real vertex.
        let n = obj.positions.len() as u32;
        assert!(
            obj.tris.iter().all(|t| t.iter().all(|&i| i < n)),
            "{key} out-of-range triangle index"
        );
        checked += 1;
    }
    assert_eq!(checked, EXPECTED_AVATARS);
}
