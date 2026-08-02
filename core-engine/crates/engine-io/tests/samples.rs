// SPDX-License-Identifier: GPL-2.0-or-later
//! Integration test over a real garment sample corpus — eight production
//! exports from industry garment-CAD and DCC tools (OBJ + FBX + textures,
//! ~352 MB). It is an internal test corpus and is **not distributed with this
//! source**; these are interoperability fixtures, exercising the file formats
//! those tools emit.
//!
//! Gated: skips cleanly and *loudly* when `samples/` is absent, by walking the
//! manifest dir's ancestors — the same approach as [`engine_io::golden::gauge_dir`],
//! which is public precisely so this idiom has one home. `NGUO_SAMPLES_DIR`
//! overrides the location.
//!
//! The corpus is not just a smoke test; it is the ground truth for what our
//! readers can actually do. Two defects were found here and are now *fixed*; the
//! assertions below pin the fixed behaviour so it cannot silently regress:
//!
//! 1. **The OBJ reader used to drop UVs, normals, panels and `mtllib`.** It now
//!    keeps all of them: [`loads_real_garment_objs`] pins the `vt` table size
//!    (larger than `v` on men-shirt and women-skirt — real UV seams) and the
//!    `usemtl`/`g` run count (the pattern panels; dress-a has 12 runs and 1
//!    material) via `engine_io::load_obj_mesh`.
//! 2. **Every real-world FBX used to fail to parse.** Root cause was in `fbxcel`
//!    0.9's *non-seekable* reader: `skip_distance_plain` discards bytes without
//!    advancing its cached position, and a zlib array attribute is the only thing
//!    that leaves an attribute partially unread (libflate stops before the adler32
//!    trailer), so the skip fired and desynchronised the parser. `fbx.rs` now uses
//!    `AnyTree::from_seekable_reader`. All five load; [`real_fbx_exports_load`]
//!    pins their counts.
//!
//! Everything asserted here was produced by running this test; nothing is
//! estimated. See `../ui/docs/SAMPLES.md` for the full table.

use engine_api::Object;
use std::path::{Path, PathBuf};

/// Locate the `samples/` corpus, or `None`. Mirrors [`engine_io::golden::gauge_dir`].
fn samples_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NGUO_SAMPLES_DIR") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join("samples"))
        .find(|p| p.is_dir())
        .map(|p| p.canonicalize().unwrap_or(p))
}

/// A sample we expect [`engine_io::load_mesh`] to read, with the counts it
/// actually produced (measured, not predicted).
struct Loadable {
    rel: &'static str,
    verts: usize,
    tris: usize,
    /// Size of the file's raw `vt` table. `vt > verts` means the file has UV
    /// seams, which only the corner-indexed `ObjMesh::uv_indices` can represent.
    vt: usize,
    /// `usemtl`/`g` runs — the garment's pattern panels.
    groups: usize,
}

/// Counts pinned from a real run. `metal`/`sphere` come out of
/// `verlet-starlight/obj.rar`, which is *not* committed — they are skipped, not
/// failed, when the archive has not been extracted.
const LOADABLE: &[Loadable] = &[
    Loadable {
        rel: "dress-a/dress_.obj",
        verts: 200_443,
        tris: 395_102,
        vt: 200_443,
        groups: 12,
    },
    Loadable {
        rel: "dress-b/dress_.obj",
        verts: 186_370,
        tris: 363_929,
        vt: 186_370,
        groups: 31,
    },
    Loadable {
        rel: "men-shirt/Men's+Shirt+OBJ/Men's Shirt.obj",
        verts: 70_312,
        tris: 128_191,
        vt: 71_745,
        groups: 45,
    },
    Loadable {
        rel: "women-skirt/Women's Pleats Skirt OBJ.obj",
        verts: 35_535,
        tris: 69_383,
        vt: 36_127,
        groups: 8,
    },
    Loadable {
        rel: "verlet-starlight/obj/metal.obj",
        verts: 1_888,
        tris: 3_752,
        vt: 2_092,
        groups: 3,
    },
    Loadable {
        rel: "verlet-starlight/obj/sphere.obj",
        verts: 482,
        tris: 960,
        vt: 559,
        groups: 1,
    },
];

/// The four OBJs that are committed outright — the floor below which this test is
/// not doing its job. (The other two live inside a `.rar`.)
const MIN_CHECKED: usize = 4;

/// Every FBX in the corpus: path, exporter FBX version, and the measured
/// `(verts, tris, normals, uvs)`. All five load. `tote-bag` reports 0 normals/UVs
/// because it is multi-geometry and `fbx.rs`'s parallel-or-empty contract drops
/// the attributes unless *every* geometry supplies them.
const FBX: &[(&str, u32, usize, usize, usize, usize)] = &[
    (
        "bike-jacket/bike+jacket.fbx",
        7400,
        31_163,
        60_454,
        31_163,
        31_163,
    ),
    (
        "dress-a/dress_.fbx",
        7500,
        200_443,
        395_102,
        200_443,
        200_443,
    ),
    (
        "dress-b/dress_.fbx",
        7500,
        186_370,
        363_929,
        186_370,
        186_370,
    ),
    (
        "men-shirt/Men's+Shirt+FBX/Men's Shirt FBX.fbx",
        7700,
        114_824,
        212_709,
        114_824,
        114_824,
    ),
    ("tote-bag/tote+bag+FBX(1).fbx", 7700, 194_171, 334_830, 0, 0),
];

/// axis-aligned bounds of a point set.
fn bbox(pts: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for p in pts {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    (mn, mx)
}

/// The invariants any mesh we hand to the solver must satisfy. Returns the bbox
/// so the caller can print it.
fn check_invariants(o: &Object, what: &str) -> ([f32; 3], [f32; 3]) {
    assert!(!o.positions.is_empty(), "{what}: no vertices");
    assert!(!o.tris.is_empty(), "{what}: no triangles");
    assert!(
        o.positions.iter().flatten().all(|c| c.is_finite()),
        "{what}: non-finite coordinate"
    );
    let nv = o.positions.len() as u32;
    assert!(
        o.tris.iter().flatten().all(|&i| i < nv),
        "{what}: triangle index out of range (nv={nv})"
    );
    assert!(
        o.tris
            .iter()
            .all(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]),
        "{what}: degenerate triangle (repeated index)"
    );
    // Parallel-or-empty is the contract every reader here documents.
    assert!(
        o.normals.is_empty() || o.normals.len() == o.positions.len(),
        "{what}: normals must be parallel to positions or absent"
    );
    assert!(
        o.uvs.is_empty() || o.uvs.len() == o.positions.len(),
        "{what}: uvs must be parallel to positions or absent"
    );
    assert!(
        o.uvs.iter().flatten().all(|c| c.is_finite()),
        "{what}: non-finite UV"
    );
    let (mn, mx) = bbox(&o.positions);
    for k in 0..3 {
        assert!(
            mx[k] - mn[k] > 0.0,
            "{what}: degenerate bounding box on axis {k} ({} .. {})",
            mn[k],
            mx[k]
        );
    }
    (mn, mx)
}

/// Load → write → re-load through both writers we have (OBJ text, binary GLB) and
/// assert the geometry is bit-identical. Both write `f32` (OBJ via the shortest
/// round-trippable `Display`, GLB as raw LE bytes), so anything short of exact
/// equality is real data loss.
fn round_trip(o: &Object, what: &str) {
    let text = engine_io::write_obj(o);
    let back =
        engine_io::parse_obj(&text, "rt").unwrap_or_else(|e| panic!("{what}: OBJ re-parse: {e:#}"));
    assert_eq!(
        back.positions, o.positions,
        "{what}: OBJ round-trip moved vertices"
    );
    assert_eq!(back.tris, o.tris, "{what}: OBJ round-trip changed topology");
    assert_eq!(back.uvs, o.uvs, "{what}: OBJ round-trip dropped UVs");
    assert_eq!(
        back.normals, o.normals,
        "{what}: OBJ round-trip dropped normals"
    );

    let glb = engine_io::write_glb(o).unwrap_or_else(|e| panic!("{what}: GLB write: {e:#}"));
    let back = engine_io::parse_gltf_slice(&glb, "rt")
        .unwrap_or_else(|e| panic!("{what}: GLB re-parse: {e:#}"));
    assert_eq!(
        back.positions, o.positions,
        "{what}: GLB round-trip moved vertices"
    );
    assert_eq!(back.tris, o.tris, "{what}: GLB round-trip changed topology");
    assert_eq!(back.uvs.len(), o.uvs.len(), "{what}: GLB dropped UVs");
    assert_eq!(
        back.normals.len(),
        o.normals.len(),
        "{what}: GLB dropped normals"
    );
}

#[test]
fn loads_real_garment_objs() {
    let Some(dir) = samples_dir() else {
        eprintln!(
            "SKIPPING samples test: no `samples/` directory found in any ancestor of {} \
             and NGUO_SAMPLES_DIR is unset. The sample corpus is ~352 MB and is not \
             committed; without it this test verifies NOTHING.",
            env!("CARGO_MANIFEST_DIR")
        );
        return;
    };
    eprintln!("samples corpus: {}", dir.display());

    let mut checked = 0usize;
    for s in LOADABLE {
        let path = dir.join(s.rel);
        if !path.is_file() {
            eprintln!("  SKIP {} — file absent (extract the .rar?)", s.rel);
            continue;
        }
        let started = std::time::Instant::now();
        let o = engine_io::load_mesh(&path).unwrap_or_else(|e| panic!("load {}: {e:#}", s.rel));
        let load = started.elapsed();
        let (mn, mx) = check_invariants(&o, s.rel);

        assert_eq!(o.positions.len(), s.verts, "{}: vertex count", s.rel);
        assert_eq!(o.tris.len(), s.tris, "{}: triangle count", s.rel);
        assert_eq!(o.uvs.len(), s.verts, "{}: UVs must survive import", s.rel);
        assert_eq!(o.normals.len(), s.verts, "{}: normals must survive", s.rel);

        // The lossless view: the raw vt table (bigger than `v` where there are UV
        // seams) and the `usemtl`/`g` runs that are the garment's pattern panels.
        let m = engine_io::load_obj_mesh(&path)
            .unwrap_or_else(|e| panic!("load_obj_mesh {}: {e:#}", s.rel));
        assert_eq!(m.uvs.len(), s.vt, "{}: raw vt table size", s.rel);
        assert_eq!(
            m.uv_indices.len(),
            s.tris,
            "{}: corner UV indices parallel to tris",
            s.rel
        );
        assert_eq!(m.groups.len(), s.groups, "{}: usemtl/g run count", s.rel);
        assert_eq!(m.groups[0].tris.start, 0, "{}: runs start at tri 0", s.rel);
        assert_eq!(
            m.groups.last().unwrap().tris.end,
            s.tris,
            "{}: runs tile the whole triangle list",
            s.rel
        );
        // Losslessness on a REAL file, not just a fixture.
        let rt = engine_io::parse_obj_mesh(&engine_io::write_obj_mesh(&m), "rt")
            .unwrap_or_else(|e| panic!("{}: ObjMesh re-parse: {e:#}", s.rel));
        assert_eq!(rt.uvs, m.uvs, "{}: vt table lost on write", s.rel);
        assert_eq!(rt.uv_indices, m.uv_indices, "{}: UV seams lost", s.rel);
        assert_eq!(rt.groups, m.groups, "{}: panel runs lost", s.rel);

        round_trip(&o, s.rel);
        checked += 1;
        eprintln!(
            "  OK   {:<44} {:>7} v {:>7} t {:>7} vt {:>3} panels  bbox [{:.2},{:.2},{:.2}]..[{:.2},{:.2},{:.2}]  load {:.1?}",
            s.rel, s.verts, s.tris, s.vt, s.groups, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], load
        );
    }

    assert!(
        checked >= MIN_CHECKED,
        "only {checked} sample(s) actually checked, expected >= {MIN_CHECKED}. \
         A sample test that loads nothing is worse than no test — five gauge tests \
         skipped silently in this repo for weeks. Fix the corpus, do not lower this."
    );
    eprintln!("  {checked} sample meshes loaded, invariant-checked and round-tripped");
}

/// Every real DCC-exported FBX in the corpus loads, with the attributes the OBJ
/// siblings do not carry. `dress-a/dress_.fbx` yields **exactly** its OBJ
/// sibling's 200 443 verts / 395 102 tris, but *with* normals and UVs.
///
/// This test used to assert the opposite (all five failed with "Node ends with
/// unexpected position"). The cause was `fbxcel` 0.9's non-seekable reader losing
/// track of its position across a skip; `fbx.rs` now parses via
/// `AnyTree::from_seekable_reader`. See the module docs.
#[test]
fn real_fbx_exports_load() {
    let Some(dir) = samples_dir() else {
        eprintln!(
            "SKIPPING FBX corpus test: no `samples/` directory (see loads_real_garment_objs)"
        );
        return;
    };
    let mut checked = 0usize;
    for &(rel, ver, verts, tris, normals, uvs) in FBX {
        let path = dir.join(rel);
        if !path.is_file() {
            eprintln!("  SKIP {rel} — file absent");
            continue;
        }
        let started = std::time::Instant::now();
        let o = engine_io::load_mesh(&path)
            .unwrap_or_else(|e| panic!("{rel} (FBX {ver}) must load: {e:#}"));
        let load = started.elapsed();
        check_invariants(&o, rel);
        assert_eq!(o.positions.len(), verts, "{rel}: vertex count");
        assert_eq!(o.tris.len(), tris, "{rel}: triangle count");
        assert_eq!(o.normals.len(), normals, "{rel}: normal count");
        assert_eq!(o.uvs.len(), uvs, "{rel}: uv count");
        checked += 1;
        eprintln!(
            "  OK   {rel:<46} (FBX {ver}) {verts:>7} v {tris:>7} t {normals:>7} n {uvs:>7} uv  load {load:.1?}"
        );
    }
    assert_eq!(
        checked,
        FBX.len(),
        "expected all {} FBX samples present and loading; only {checked} were checked",
        FBX.len()
    );
}

/// `.zprj` (the project container written by CLO3D / Marvelous Designer) and
/// `.blend` (Blender) are in the corpus and we cannot read either format.
/// `load_mesh` must say so clearly rather than mis-dispatch.
#[test]
fn unreadable_corpus_formats_are_rejected_clearly() {
    let Some(dir) = samples_dir() else {
        eprintln!("SKIPPING format-rejection test: no `samples/` directory");
        return;
    };
    let mut checked = 0usize;
    for rel in ["dress-a/dress.zprj", "dress-a/dress1.blend"] {
        let path = dir.join(rel);
        if !path.is_file() {
            eprintln!("  SKIP {rel} — file absent");
            continue;
        }
        let err = engine_io::load_mesh(&path).expect_err("must not claim to read this");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("unsupported mesh extension"),
            "{rel}: want an explicit unsupported-extension error, got: {msg}"
        );
        checked += 1;
        eprintln!("  REJECTED   {rel:<46} {msg}");
    }
    assert_eq!(
        checked, 2,
        "both unreadable formats must be present and rejected"
    );
}

/// The cross-format check that makes the FBX fix trustworthy: dress-a ships the
/// same garment as OBJ *and* FBX, and the two must agree exactly on geometry —
/// with the FBX additionally carrying the normals and UVs.
#[test]
fn dress_a_obj_and_fbx_agree() {
    let Some(dir) = samples_dir() else {
        eprintln!("SKIPPING cross-format test: no `samples/` directory");
        return;
    };
    let (obj_p, fbx_p) = (
        dir.join("dress-a/dress_.obj"),
        dir.join("dress-a/dress_.fbx"),
    );
    if !obj_p.is_file() || !fbx_p.is_file() {
        panic!("dress-a must ship both dress_.obj and dress_.fbx");
    }
    let o = engine_io::load_mesh(&obj_p).expect("dress-a OBJ");
    let f = engine_io::load_mesh(&fbx_p).expect("dress-a FBX");
    assert_eq!(o.positions.len(), f.positions.len(), "vertex count differs");
    assert_eq!(o.tris.len(), f.tris.len(), "triangle count differs");
    assert_eq!(f.normals.len(), f.positions.len(), "FBX normals");
    assert_eq!(f.uvs.len(), f.positions.len(), "FBX UVs");
    eprintln!(
        "  dress-a OBJ {} v / {} t  ==  FBX {} v / {} t (+{} normals, +{} uvs)",
        o.positions.len(),
        o.tris.len(),
        f.positions.len(),
        f.tris.len(),
        f.normals.len(),
        f.uvs.len()
    );
}
