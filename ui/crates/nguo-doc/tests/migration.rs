// SPDX-License-Identifier: GPL-2.0-or-later
//! A real v1 project opens in this build (guide §4.4).
//!
//! `tests/golden/doc_v1.json` is frozen: it is what a v1 build actually wrote
//! and must never be regenerated. `tests/golden/doc_v1_migrated.json` is the
//! blessable snapshot of what this build turns it into — regenerate with
//! `NGUO_BLESS=1 cargo test -p nguo-doc --test migration`.

use std::path::PathBuf;

use nguo_doc::{migrate, Document, MigrationError, SCHEMA_VERSION};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

fn v1_source() -> String {
    std::fs::read_to_string(golden("doc_v1.json")).expect("frozen v1 fixture")
}

#[test]
fn a_v1_project_loads_into_the_current_schema() {
    let doc = Document::from_json(&v1_source()).expect("v1 must still open");

    assert_eq!(doc.schema_version(), SCHEMA_VERSION);
    assert_eq!(doc.meta().name, "A-line skirt");
    assert_eq!(doc.pieces().len(), 2);
    assert_eq!(doc.seams().len(), 1);
    assert_eq!(doc.materials().len(), 1);

    // The v1 -> v2 move: the project-wide seam allowance became a per-piece
    // default on every piece.
    for (_, piece) in doc.pieces().iter() {
        assert_eq!(piece.default_allowance_mm, 12.5, "{}", piece.name);
    }

    // Geometry, ids and expressions all survived.
    let front = doc.pieces().iter().next().unwrap().1;
    assert_eq!(front.name, "Front");
    assert_eq!(front.network.edges.len(), 4);
    assert_eq!(front.features.len(), 2);
    assert_eq!(doc.params().value("quarter_waist").unwrap(), 185.0);
    assert_eq!(doc.eval("body.hip / 4").unwrap(), 245.0);
}

#[test]
fn migrating_twice_is_the_same_as_migrating_once() {
    let once = migrate(serde_json::from_str(&v1_source()).unwrap()).unwrap();
    let twice = migrate(once.clone()).unwrap();
    assert_eq!(once, twice);
}

#[test]
fn the_migrated_document_matches_its_golden() {
    let doc = Document::from_json(&v1_source()).unwrap();
    let text = doc.to_json_pretty().unwrap() + "\n";
    let path = golden("doc_v1_migrated.json");

    if std::env::var("NGUO_BLESS").is_ok() || !path.exists() {
        std::fs::write(&path, &text).unwrap();
        eprintln!("blessed golden at {}", path.display());
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        text, want,
        "migrated v1 drifted from the golden; re-bless with NGUO_BLESS=1 if intended"
    );

    // And the golden is itself loadable, so the format is closed under save/load.
    let reloaded = Document::from_json(&want).unwrap();
    assert_eq!(reloaded, doc);
}

#[test]
fn a_future_project_is_refused_with_a_useful_error() {
    let mut v: serde_json::Value = serde_json::from_str(&v1_source()).unwrap();
    v["schema_version"] = serde_json::json!(SCHEMA_VERSION + 9);
    assert_eq!(
        migrate(v),
        Err(MigrationError::FromTheFuture {
            found: SCHEMA_VERSION + 9
        })
    );
}
