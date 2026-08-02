// SPDX-License-Identifier: GPL-2.0-or-later
//! Schema versioning and migration.
//!
//! The project format is a **zipped bundle**: a `.nguo`
//! is a zip of a JSON manifest plus binary blobs. This module owns the
//! manifest's schema and its version ladder; `nguo-io` owns the zip. The
//! handover point is [`migrate`], which takes the raw manifest value and
//! returns one that [`crate::Document`]'s `Deserialize` accepts.
//!
//! # Adding a version
//! Bump [`SCHEMA_VERSION`], add a `fn vN_to_vN1(&mut Value)`, add it to
//! [`STEPS`], and drop a fixture in `tests/golden/`. Migrations run on
//! `serde_json::Value`, never on the typed structs, so an old document can be
//! upgraded long after the Rust types for its version are gone.

use serde_json::Value;

/// The schema version this build writes.
pub const SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MigrationError {
    #[error("the manifest has no `schema_version`")]
    MissingVersion,
    #[error("the manifest is not a JSON object")]
    NotAnObject,
    #[error("this project was written by a newer build (schema {found}, this build reads {SCHEMA_VERSION})")]
    FromTheFuture { found: u32 },
    #[error("schema {0} is too old to upgrade")]
    TooOld(u32),
    #[error("migrating {from} -> {to}: {reason}")]
    Malformed { from: u32, to: u32, reason: String },
}

type Step = fn(&mut Value) -> Result<(), String>;

/// `STEPS[i]` upgrades schema `i + 1` to schema `i + 2`.
const STEPS: &[Step] = &[v1_to_v2];

/// Bring a manifest up to [`SCHEMA_VERSION`].
///
/// Idempotent for a current document, and an error rather than a silent
/// mis-parse for anything it cannot handle.
pub fn migrate(mut value: Value) -> Result<Value, MigrationError> {
    let obj = value.as_object().ok_or(MigrationError::NotAnObject)?;
    let mut version = obj
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or(MigrationError::MissingVersion)? as u32;

    if version > SCHEMA_VERSION {
        return Err(MigrationError::FromTheFuture { found: version });
    }
    if version == 0 {
        return Err(MigrationError::TooOld(version));
    }

    while version < SCHEMA_VERSION {
        let step = STEPS
            .get(version as usize - 1)
            .ok_or(MigrationError::TooOld(version))?;
        step(&mut value).map_err(|reason| MigrationError::Malformed {
            from: version,
            to: version + 1,
            reason,
        })?;
        version += 1;
        value["schema_version"] = Value::from(version);
    }
    Ok(value)
}

/// v1 -> v2: the seam allowance moved from a single project-wide setting to a
/// per-piece default, so a pattern can mix (say) 10 mm on the body and 6 mm on
/// a collar. The old value becomes every piece's default.
fn v1_to_v2(value: &mut Value) -> Result<(), String> {
    let old = value
        .get("meta")
        .and_then(|m| m.get("seam_allowance_mm"))
        .and_then(Value::as_f64)
        .unwrap_or(10.0);

    if let Some(meta) = value.get_mut("meta").and_then(Value::as_object_mut) {
        meta.remove("seam_allowance_mm");
    }

    let pieces = value
        .get_mut("pieces")
        .ok_or("v1 manifest has no `pieces`")?
        .as_object_mut()
        .ok_or("`pieces` is not an object")?;
    for (id, piece) in pieces.iter_mut() {
        let p = piece
            .as_object_mut()
            .ok_or_else(|| format!("piece {id} is not an object"))?;
        p.entry("default_allowance_mm")
            .or_insert_with(|| Value::from(old));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn v1_doc() -> Value {
        json!({
            "schema_version": 1,
            "meta": {
                "name": "Skirt",
                "author": "",
                "display_unit": "Centimeter",
                "seam_allowance_mm": 12.5
            },
            "pieces": {
                "0": {
                    "name": "Front",
                    "network": { "vertices": {}, "edges": {} },
                    "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    "material": null,
                    "features": {},
                    "symmetry": null,
                    "placement": null
                }
            },
            "seams": {},
            "materials": {},
            "params": {},
            "camera": { "eye_mm": [0.0, 0.0, 1.0], "target_mm": [0.0, 0.0, 0.0], "fov_y_rad": 0.7 }
        })
    }

    #[test]
    fn v1_upgrades_to_the_current_schema() {
        let out = migrate(v1_doc()).unwrap();
        assert_eq!(out["schema_version"], json!(SCHEMA_VERSION));
        assert_eq!(out["pieces"]["0"]["default_allowance_mm"], json!(12.5));
        assert!(out["meta"].get("seam_allowance_mm").is_none());
    }

    #[test]
    fn a_current_document_passes_through_unchanged() {
        let d = crate::Document::new();
        let value: Value = serde_json::to_value(&d).unwrap();
        assert_eq!(migrate(value.clone()).unwrap(), value);
    }

    #[test]
    fn refuses_what_it_cannot_understand() {
        assert_eq!(migrate(json!([])), Err(MigrationError::NotAnObject));
        assert_eq!(migrate(json!({})), Err(MigrationError::MissingVersion));
        assert_eq!(
            migrate(json!({ "schema_version": SCHEMA_VERSION + 1 })),
            Err(MigrationError::FromTheFuture {
                found: SCHEMA_VERSION + 1
            })
        );
        assert_eq!(
            migrate(json!({ "schema_version": 0 })),
            Err(MigrationError::TooOld(0))
        );
    }

    #[test]
    fn a_malformed_v1_reports_where_it_broke() {
        let mut bad = v1_doc();
        bad["pieces"] = json!("not an object");
        assert!(matches!(
            migrate(bad),
            Err(MigrationError::Malformed { from: 1, to: 2, .. })
        ));
    }
}
