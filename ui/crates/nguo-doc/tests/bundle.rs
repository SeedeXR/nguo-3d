// SPDX-License-Identifier: GPL-2.0-or-later
//! `.nguo` save/load (D-1): round-trip fidelity, migration through the bundle,
//! and hostile input.
//!
//! The property that matters is that save-then-load reproduces the *whole*
//! document, not a few fields — stable ids most of all, because seams,
//! constraints and undo all dereference them.

use std::path::PathBuf;

use nguo_doc::bundle::{self, Project, DOCUMENT, MANIFEST, PREVIEW};
use nguo_doc::kurbo::{Affine, Point};
use nguo_doc::measure::{Param, Unit};
use nguo_doc::model::{
    AvatarRef, Camera, Feature, Material, NotchKind, Piece, Placement, Seam, SeamSide, Symmetry,
};
use nguo_doc::pattern::{EdgeGeom, EdgeSemantic, VectorNetwork};
use nguo_doc::{Document, Edit, SCHEMA_VERSION};

// --- a project with something in it ---------------------------------------

/// A rectangle-ish piece: four vertices, one curved edge, mixed semantics, plus
/// an internal dart so the network has a T-junction.
fn piece(name: &str, w: f64, h: f64, curve: bool) -> Piece {
    let mut net = VectorNetwork::new();
    let v: Vec<_> = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        .iter()
        .map(|&(x, y)| net.add_vertex(Point::new(x, y)))
        .collect();
    let sems = [
        EdgeSemantic::Fold,
        EdgeSemantic::Seam,
        EdgeSemantic::Cut,
        EdgeSemantic::Seam,
    ];
    for i in 0..4 {
        let geom = if curve && i == 2 {
            EdgeGeom::Cubic {
                c1: Point::new(w * 0.66, h * 1.08),
                c2: Point::new(w * 0.33, h * 1.08),
            }
        } else {
            EdgeGeom::Line
        };
        net.add_edge(v[i], v[(i + 1) % 4], geom, sems[i]).unwrap();
    }
    // A dart: apex inside, two legs down to the top edge.
    let apex = net.add_vertex(Point::new(w * 0.5, h * 0.4));
    let a = net.add_vertex(Point::new(w * 0.44, 0.0));
    let b = net.add_vertex(Point::new(w * 0.56, 0.0));
    let leg_a = net
        .add_edge(apex, a, EdgeGeom::Line, EdgeSemantic::DartLeg)
        .unwrap();
    let leg_b = net
        .add_edge(apex, b, EdgeGeom::Line, EdgeSemantic::DartLeg)
        .unwrap();

    let mut p = Piece::new(name);
    p.network = net;
    p.default_allowance_mm = 12.0;
    p.features.insert(Feature::Dart {
        apex,
        legs: [leg_a, leg_b],
    });
    p
}

/// Four pieces, three seams, two fabrics, parameters, an avatar, a camera —
/// and every `Feature` variant, so nothing is round-tripped by accident.
fn dress() -> Document {
    let mut d = Document::new();

    let wool = d.next_material_id();
    d.apply_edit(Edit::AddMaterial {
        id: wool,
        material: Box::new(Material {
            name: "Wool crepe".into(),
            weight_gsm: 210.0,
            thickness_mm: 0.55,
            stretch_warp: 0.015,
            stretch_weft: 0.028,
            bend: 3.5e-5,
            friction: 0.42,
            color_srgb: [0.13, 0.18, 0.31],
        }),
    })
    .unwrap();
    let lining = d.next_material_id();
    d.apply_edit(Edit::AddMaterial {
        id: lining,
        material: Box::new(Material {
            name: "Cupro lining".into(),
            ..Material::default()
        }),
    })
    .unwrap();

    let specs = [
        ("Bodice front", 240.0, 420.0, true),
        ("Bodice back", 235.0, 430.0, false),
        ("Skirt front", 380.0, 700.0, true),
        ("Skirt back", 385.0, 700.0, true),
    ];
    let mut ids = Vec::new();
    for (i, (name, w, h, curve)) in specs.into_iter().enumerate() {
        let id = d.next_piece_id();
        let mut p = piece(name, w, h, curve);
        p.transform = Affine::translate((i as f64 * 450.0, 0.0));
        p.material = Some(if i < 2 { wool } else { lining });
        p.placement = Some(Placement {
            position_mm: [i as f64 * 30.0, 900.0 - i as f64 * 10.0, 15.0],
            rotation_quat: [0.0, 0.3826834323650898, 0.0, 0.9238795325112867],
        });
        if i == 0 {
            p.symmetry = Some(Symmetry {
                axis_from: Point::new(0.0, 0.0),
                axis_to: Point::new(0.0, 420.0),
                mirrored_piece: None,
            });
        }
        let edges: Vec<_> = p.network.edges.iter().map(|(e, _)| e).collect();
        p.features.insert(Feature::Notch {
            edge: edges[1],
            t: 0.375_912_5,
            kind: if i % 2 == 0 {
                NotchKind::V
            } else {
                NotchKind::Castle
            },
        });
        p.features.insert(Feature::GrainLine {
            from: Point::new(w * 0.5, 20.0),
            to: Point::new(w * 0.5, h - 20.0),
        });
        p.features.insert(Feature::InternalLine {
            name: "Pocket line".into(),
            edges: vec![edges[2]],
        });
        p.features.insert(Feature::Label {
            at: Point::new(30.0, 40.0),
            angle_rad: 0.261_799_387_799_149_4,
            text: format!("{name} — cut 2"),
        });
        p.features.insert(Feature::Support {
            edge: edges[0],
            landmark: if i < 2 { "underbust" } else { "waist" }.into(),
        });
        d.apply_edit(Edit::AddPiece {
            id,
            piece: Box::new(p),
        })
        .unwrap();
        ids.push(id);
    }

    // Seams that reference edges by id, including a partial span and gathering.
    for (i, &(a, b)) in [(0usize, 1usize), (2, 3), (0, 2)].iter().enumerate() {
        let ea: Vec<_> = d
            .piece(ids[a])
            .unwrap()
            .network
            .edges
            .iter()
            .map(|(e, _)| e)
            .collect();
        let eb: Vec<_> = d
            .piece(ids[b])
            .unwrap()
            .network
            .edges
            .iter()
            .map(|(e, _)| e)
            .collect();
        let id = d.next_seam_id();
        let mut seam = Seam::new(
            SeamSide::whole(ids[a], vec![(ea[1], false), (ea[2], true)]),
            SeamSide {
                piece: ids[b],
                edges: vec![(eb[3], true)],
                span: (0.125, 0.937_5),
            },
        );
        seam.allowance_mm = 9.0 + i as f64;
        seam.ease_ratio = 1.0 + i as f64 * 0.075;
        seam.note = format!("seam {i}: ease into the yoke");
        d.apply_edit(Edit::AddSeam {
            id,
            seam: Box::new(seam),
        })
        .unwrap();
    }

    d.apply_edit(Edit::SetParam {
        name: "ease".into(),
        param: Some(Param::new("2.5cm").with_description("bust ease")),
    })
    .unwrap();
    d.apply_edit(Edit::SetParam {
        name: "quarter_waist".into(),
        param: Some(Param::new("body.waist / 4 + ease").with_unit(Unit::Millimeter)),
    })
    .unwrap();

    d.apply_edit(Edit::SetAvatar {
        avatar: Some(Box::new(AvatarRef {
            name: "female".into(),
            measurements: [
                ("waist".to_string(), 745.0),
                ("hip".to_string(), 1087.3),
                ("chest".to_string(), 1116.0),
            ]
            .into_iter()
            .collect(),
            pose: "a-pose".into(),
            asset: None,
            tier: Some("col".into()),
            bundle_version: Some("1.0.0".into()),
        })),
    })
    .unwrap();
    d.apply_edit(Edit::SetMetadata {
        meta: Box::new(nguo_doc::model::Metadata {
            name: "Bias dress".into(),
            author: "Alex Mkwizu".into(),
            display_unit: Unit::Centimeter,
            notes: "Round-trip fixture: several pieces, curves, seams, ids.".into(),
        }),
    })
    .unwrap();
    d.apply_edit(Edit::SetCamera {
        camera: Camera {
            eye_mm: [120.5, 913.25, 2417.125],
            target_mm: [0.0, 880.0, 0.0],
            fov_y_rad: 0.698_131_700_797_731_8,
        },
    })
    .unwrap();
    d
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nguo-bundle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

// --- round trip -----------------------------------------------------------

#[test]
fn a_real_project_round_trips_exactly() {
    let doc = dress();
    assert!(doc.pieces().len() == 4 && doc.seams().len() == 3);

    let path = tmp("dress.nguo");
    let mut project = Project::new(doc.clone());
    project.save(&path).unwrap();
    let back = bundle::load(&path).unwrap();

    // The whole document, not a few fields.
    assert_eq!(back.document, doc);
    // And byte-identical JSON, which catches anything `PartialEq` skips.
    assert_eq!(
        back.document.to_json_pretty().unwrap(),
        doc.to_json_pretty().unwrap()
    );

    // Stable ids survive: seams still dereference the same pieces and edges.
    for (sid, seam) in doc.seams().iter() {
        let reloaded = back.document.seam(sid).unwrap();
        assert_eq!(reloaded, seam);
        for side in [&reloaded.a, &reloaded.b] {
            let net = &back.document.piece(side.piece).unwrap().network;
            for &(e, _) in &side.edges {
                assert!(net.edges.contains(e), "seam {sid:?} lost edge {e:?}");
            }
        }
    }
    // Ids are keys, not positions: the next id is still past every live one.
    assert_eq!(back.document.next_piece_id(), doc.next_piece_id());

    // Re-saving a loaded project keeps its identity and its creation time.
    let mut again = back;
    let created = again.manifest.created.clone();
    let id = again.manifest.document_id.clone();
    again.save(&tmp("dress2.nguo")).unwrap();
    assert_eq!(again.manifest.created, created);
    assert_eq!(again.manifest.document_id, id);
}

#[test]
fn the_manifest_says_what_the_file_is() {
    let path = tmp("manifest.nguo");
    let mut project = Project::new(dress());
    project.save(&path).unwrap();

    let m = &bundle::load(&path).unwrap().manifest;
    assert_eq!(m.format, "nguo-3d project");
    assert_eq!(m.schema_version, SCHEMA_VERSION);
    assert!(m.writer.starts_with("nguo-doc "));
    assert_eq!(m.name, "Bias dress");
    assert_eq!(m.units.geometry, "millimetre");
    assert_eq!(m.units.display, Unit::Centimeter);
    assert!(
        m.units.coordinates.contains("+Y up"),
        "{}",
        m.units.coordinates
    );
    assert!(m.units.coordinates.contains("millimetres"));
    assert_eq!(m.counts.pieces, 4);
    assert_eq!(m.counts.seams, 3);
    assert_eq!(m.counts.materials, 2);
    assert_eq!(m.counts.params, 2);
    assert_eq!(m.counts.vertices, 4 * 7);
    assert_eq!(m.counts.edges, 4 * 6);
    assert!(m.created.ends_with('Z') && m.modified.ends_with('Z'));
    assert!(m.document_id.starts_with("nguo-"));
    assert_eq!(m.members, [MANIFEST, DOCUMENT, PREVIEW]);

    // The avatar is referenced, never embedded.
    let a = m.avatar.as_ref().expect("the dress has an avatar");
    assert_eq!(a.key, "female");
    assert_eq!(a.tier.as_deref(), Some("col"));
    assert_eq!(a.bundle_version.as_deref(), Some("1.0.0"));
    assert!(!a.embedded);

    // Anyone can read the members with an unzip tool, and the bundle carries
    // no avatar mesh however big the project gets.
    let members = bundle::members(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(members.len(), 3);
    let manifest_text = std::str::from_utf8(&members[MANIFEST]).unwrap();
    assert!(
        manifest_text.starts_with("{\n  \"format\""),
        "{manifest_text}"
    );
    let preview = std::str::from_utf8(&members[PREVIEW]).unwrap();
    assert!(preview.starts_with("<svg"), "{preview}");
    // One outline per piece, or the thumbnail is not of this project.
    assert_eq!(preview.matches("<path").count(), 4, "{preview}");
    assert!(preview.contains("Bias dress"));
    assert!(
        members[PREVIEW].len() < 32 * 1024,
        "preview must stay small"
    );
}

#[test]
fn a_missing_preview_does_not_block_loading() {
    // Rebuild the bundle without the preview, as a stripping tool might.
    let path = tmp("nopreview.nguo");
    let mut project = Project::new(dress());
    project.save(&path).unwrap();
    let members = bundle::members(&std::fs::read(&path).unwrap()).unwrap();

    let mut w = nguo_doc::zip::Writer::new((0, 0x21));
    w.add(MANIFEST, &members[MANIFEST]).unwrap();
    w.add(DOCUMENT, &members[DOCUMENT]).unwrap();
    let loaded = bundle::from_bytes(&w.finish(), "nopreview.nguo").unwrap();
    assert_eq!(loaded.document, project.document);
}

// --- migration through the bundle -----------------------------------------

/// The frozen v1 project, packaged as a v1 `.nguo`. `doc_v1.json` is the
/// historical artifact (never regenerate it); the manifest around it is what a
/// v1 writer would have written.
fn v1_bundle() -> Vec<u8> {
    let doc = std::fs::read(golden("doc_v1.json")).expect("frozen v1 fixture");
    let manifest = serde_json::json!({
        "format": "nguo-3d project",
        "schema_version": 1,
        "writer": "nguo-doc 0.0.0",
        "document_id": "nguo-legacy-v1",
        "created": "2026-07-01T09:15:00Z",
        "modified": "2026-07-02T18:40:12Z",
        "name": "A-line skirt",
        "units": {
            "geometry": "millimetre",
            "display": "Centimeter",
            "coordinates": "pattern space: millimetres, +X right, +Y down."
        }
    });
    let mut w = nguo_doc::zip::Writer::new((0, 0x21));
    w.add(MANIFEST, manifest.to_string().as_bytes()).unwrap();
    w.add(DOCUMENT, &doc).unwrap();
    w.finish()
}

#[test]
fn a_v1_project_bundle_opens_and_is_upgraded() {
    let loaded = bundle::from_bytes(&v1_bundle(), "skirt_v1.nguo").unwrap();

    // The document came up the ladder...
    assert_eq!(loaded.document.schema_version(), SCHEMA_VERSION);
    assert_eq!(loaded.document.meta().name, "A-line skirt");
    assert_eq!(loaded.document.pieces().len(), 2);
    assert_eq!(loaded.document.seams().len(), 1);
    for (_, p) in loaded.document.pieces().iter() {
        // v1 -> v2: the project-wide allowance became a per-piece default.
        assert_eq!(p.default_allowance_mm, 12.5, "{}", p.name);
    }
    assert_eq!(loaded.document.eval("body.hip / 4").unwrap(), 245.0);
    // ...and it matches what the plain-JSON path produces, so the bundle adds
    // no second migration behaviour.
    let direct =
        Document::from_json(&std::fs::read_to_string(golden("doc_v1.json")).unwrap()).unwrap();
    assert_eq!(loaded.document, direct);

    // The manifest reports the version now in memory, and keeps the project's
    // identity and creation time from the v1 file.
    assert_eq!(loaded.manifest.schema_version, SCHEMA_VERSION);
    assert_eq!(loaded.manifest.document_id, "nguo-legacy-v1");
    assert_eq!(loaded.manifest.created, "2026-07-01T09:15:00Z");

    // Saving it back writes a v2 bundle that reloads identically.
    let path = tmp("skirt_upgraded.nguo");
    let mut loaded = loaded;
    loaded.save(&path).unwrap();
    let again = bundle::load(&path).unwrap();
    assert_eq!(again.document, loaded.document);
    assert_eq!(again.manifest.schema_version, SCHEMA_VERSION);
    // Upgrading a project must not lose when it was created or which project
    // it is; only `modified` may move.
    assert_eq!(again.manifest.document_id, "nguo-legacy-v1");
    assert_eq!(again.manifest.created, "2026-07-01T09:15:00Z");
    assert!(again.manifest.modified >= again.manifest.created);
}

#[test]
fn a_project_from_the_future_is_refused_not_downgraded() {
    let mut members = bundle::members(&v1_bundle()).unwrap();
    let mut doc: serde_json::Value = serde_json::from_slice(&members[DOCUMENT]).unwrap();
    doc["schema_version"] = serde_json::json!(SCHEMA_VERSION + 7);
    doc["something_this_build_never_heard_of"] = serde_json::json!({ "keep": "me" });
    members.insert(DOCUMENT.into(), doc.to_string().into_bytes());
    let mut manifest: serde_json::Value = serde_json::from_slice(&members[MANIFEST]).unwrap();
    manifest["schema_version"] = serde_json::json!(SCHEMA_VERSION + 7);
    members.insert(MANIFEST.into(), manifest.to_string().into_bytes());

    let mut w = nguo_doc::zip::Writer::new((0, 0x21));
    for (name, body) in &members {
        w.add(name, body).unwrap();
    }
    let err = bundle::from_bytes(&w.finish(), "future.nguo").unwrap_err();
    let msg = err.to_string() + &format!("{:?}", std::error::Error::source(&err));
    assert!(
        msg.contains("newer build") || msg.contains("FromTheFuture"),
        "unhelpful error: {msg}"
    );
}

#[test]
fn a_manifest_that_contradicts_its_document_is_refused() {
    let mut members = bundle::members(&v1_bundle()).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&members[MANIFEST]).unwrap();
    manifest["schema_version"] = serde_json::json!(2);
    members.insert(MANIFEST.into(), manifest.to_string().into_bytes());
    let mut w = nguo_doc::zip::Writer::new((0, 0x21));
    for (name, body) in &members {
        w.add(name, body).unwrap();
    }
    let err = bundle::from_bytes(&w.finish(), "mixed.nguo").unwrap_err();
    assert!(
        matches!(
            err,
            nguo_doc::BundleError::VersionMismatch {
                manifest: 2,
                document: 1,
                ..
            }
        ),
        "{err}"
    );
}

// --- hostile input --------------------------------------------------------

#[test]
fn corrupt_input_produces_an_actionable_error_never_a_panic() {
    let path = tmp("victim.nguo");
    let mut project = Project::new(dress());
    project.save(&path).unwrap();
    let good = std::fs::read(&path).unwrap();

    // Truncated at every 1/16th, plus one byte short of the end.
    for cut in (0..16).map(|i| good.len() * i / 16).chain([good.len() - 1]) {
        let err = bundle::from_bytes(&good[..cut], "cut.nguo").unwrap_err();
        assert!(!err.to_string().is_empty(), "cut at {cut}");
    }

    // A zip that is not a project.
    let mut w = nguo_doc::zip::Writer::new((0, 0x21));
    w.add("readme.txt", b"hello").unwrap();
    let err = bundle::from_bytes(&w.finish(), "plain.zip").unwrap_err();
    assert!(
        matches!(
            err,
            nguo_doc::BundleError::MissingMember {
                member: MANIFEST,
                ..
            }
        ),
        "{err}"
    );

    // A project whose document is not JSON.
    let mut w = nguo_doc::zip::Writer::new((0, 0x21));
    let members = bundle::members(&good).unwrap();
    w.add(MANIFEST, &members[MANIFEST]).unwrap();
    w.add(DOCUMENT, b"{ this is not json").unwrap();
    let err = bundle::from_bytes(&w.finish(), "bad.nguo").unwrap_err();
    assert!(
        matches!(
            err,
            nguo_doc::BundleError::BadJson {
                member: DOCUMENT,
                ..
            }
        ),
        "{err}"
    );

    // Not a zip at all.
    assert!(bundle::from_bytes(b"PK\x03\x04 nope", "junk.nguo").is_err());
    assert!(bundle::from_bytes(&[], "empty.nguo").is_err());

    // A missing file names itself.
    let err = bundle::load(&tmp("does-not-exist.nguo")).unwrap_err();
    assert!(err.to_string().contains("does-not-exist.nguo"), "{err}");
}

#[test]
fn loading_writes_nothing_and_a_member_path_cannot_escape() {
    // `../../evil.txt` is rejected by name; nothing is written either way,
    // because load decompresses into memory and never touches the filesystem.
    let escapee = std::env::temp_dir().join("nguo-bundle-escape-canary.txt");
    let _ = std::fs::remove_file(&escapee);

    let mut w = nguo_doc::zip::Writer::new((0, 0x21));
    w.add(MANIFEST, b"{}").unwrap();
    let mut raw = w.finish();
    // Patch `manifest.json` (13 bytes) to a traversal of the same length.
    let evil = b"../../../x.js";
    let mut patched = 0;
    for at in 0..raw.len() - evil.len() {
        if &raw[at..at + 13] == MANIFEST.as_bytes() {
            raw[at..at + 13].copy_from_slice(evil);
            patched += 1;
        }
    }
    assert_eq!(patched, 2);
    let err = bundle::from_bytes(&raw, "evil.nguo").unwrap_err();
    assert!(err.to_string().contains("relative"), "{err}");
    assert!(!escapee.exists());
}

// --- size and speed -------------------------------------------------------

/// A dress-sized project: 14 pieces, ~500 vertices and ~500 edges, which is
/// what a real garment's pattern actually looks like. The four-piece fixture
/// above is for fidelity; this one is for the size and time numbers, because one
/// tiny measurement hides how a format scales.
fn big_dress() -> Document {
    let mut d = dress();
    for i in 0..10 {
        let mut p = piece(&format!("Panel {i}"), 300.0, 650.0, i % 2 == 0);
        for _ in 0..3 {
            let edges: Vec<_> = p.network.edges.iter().map(|(e, _)| e).collect();
            for e in edges {
                p.network.split_edge(e, 0.5).unwrap();
            }
        }
        // Subdividing replaced the edge ids the dart referenced.
        p.features = Default::default();
        p.transform = Affine::translate((i as f64 * 500.0, 900.0));
        let id = d.next_piece_id();
        d.apply_edit(Edit::AddPiece {
            id,
            piece: Box::new(p),
        })
        .unwrap();
    }
    d
}

#[test]
fn a_dress_sized_project_is_small_and_fast() {
    let doc = big_dress();
    let path = tmp("dress-big.nguo");
    let mut project = Project::new(doc);
    let t0 = std::time::Instant::now();
    let bytes = project.save(&path).unwrap();
    let save_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    let back = bundle::load(&path).unwrap();
    let load_ms = t1.elapsed().as_secs_f64() * 1e3;
    assert_eq!(back.document, project.document);

    let m = &back.manifest;
    println!(
        "dress-sized project ({} pieces, {} vertices, {} edges, {} seams): {bytes} B on disk \
         · save {save_ms:.2} ms · load {load_ms:.2} ms · at {}",
        m.counts.pieces,
        m.counts.vertices,
        m.counts.edges,
        m.counts.seams,
        path.display(),
    );
    assert!(
        m.counts.vertices > 450 && m.counts.edges > 450,
        "{:?}",
        m.counts
    );
    assert!(bytes < 512 * 1024, "{bytes} B");
    assert!(
        save_ms < 250.0 && load_ms < 250.0,
        "{save_ms} / {load_ms} ms"
    );
}

#[test]
fn save_and_load_are_small_and_fast() {
    let doc = dress();
    let path = tmp("perf.nguo");
    let mut project = Project::new(doc);

    let t0 = std::time::Instant::now();
    let bytes = project.save(&path).unwrap();
    let save_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = std::time::Instant::now();
    let back = bundle::load(&path).unwrap();
    let load_ms = t1.elapsed().as_secs_f64() * 1e3;
    assert_eq!(back.document, project.document);

    let raw = std::fs::read(&path).unwrap();
    let members = bundle::members(&raw).unwrap();
    let plain: usize = members.values().map(Vec::len).sum();
    println!(
        "dress project: {bytes} B on disk ({plain} B uncompressed, {:.0}%) \
         · save {save_ms:.2} ms · load {load_ms:.2} ms · manifest {} B · document {} B · preview {} B",
        100.0 * bytes as f64 / plain as f64,
        members[MANIFEST].len(),
        members[DOCUMENT].len(),
        members[PREVIEW].len(),
    );

    // Ceilings, not measurements: a regression that makes a four-piece project
    // a megabyte or a tenth of a second has broken something.
    assert!(bytes < 64 * 1024, "{bytes} B is too big for four pieces");
    assert!(save_ms < 100.0, "save took {save_ms} ms");
    assert!(load_ms < 100.0, "load took {load_ms} ms");
}
