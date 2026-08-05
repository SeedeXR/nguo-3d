// SPDX-License-Identifier: GPL-2.0-or-later
//! The `.nguo` project bundle: save and load (D-1).
//!
//! A `.nguo` is a **zip** of three members, and nothing else:
//!
//! | member          | what it is                                              |
//! |-----------------|---------------------------------------------------------|
//! | `manifest.json` | [`Manifest`] — what this file is, without reading the rest |
//! | `document.json` | the whole [`Document`], pretty-printed, migrated on load |
//! | `preview.svg`   | a 2 KB outline of the pattern for a project picker      |
//!
//! Unzip one and read it: that is the point. The manifest states the schema
//! version, the build that wrote it, the timestamps, the document id, the units
//! **and the handedness**, because every interop bug in this project traces to
//! an unstated convention.
//!
//! # What is in the file, and what is deliberately not
//! **In**, because reopening needs it: every pattern piece and its vector
//! network (vertices, curve geometry, edge semantics), features (notches, darts,
//! grain lines, internal lines, labels, `Support` = the landmark a piece hangs
//! at), seams with their per-side edge chains and spans, materials, the
//! parameter table, the avatar *reference*, per-piece 3D placement, the camera,
//! project metadata — i.e. all of [`Document`], which is by construction the
//! only authored truth (guide §4).
//!
//! **Out**, on purpose:
//! * **Derived state** — tessellations, sim meshes, spatial indices, the drape
//!   result. Rebuildable from the above, and saving it as truth is exactly what
//!   guide §4.3 forbids.
//! * **The avatar mesh.** The bundle is 227 MB, Git-LFS and CC BY 4.0 —
//!   licensed separately from this GPL code. We store the key, tier and bundle
//!   version and resolve at load; embedding it would put a quarter-gigabyte of
//!   differently-licensed asset inside every project file.
//! * **Undo history.** D-1 says no delta history in the file.
//! * **Solver settings and the resize factor + its landmark.** Not an omission
//!   we chose so much as one we inherited: neither is in [`Document`] today
//!   (they live in `nguo-engine-bridge` and `engine-api::resize`). When they
//!   move into the document they are saved automatically, and the schema bump
//!   is the migration that adds their defaults. Nothing else needs to change
//!   here.
//! * **A window layout / docking state.** Per-user, not per-project.
//!
//! # Compatibility contract
//! * **Older schema**: upgraded by [`crate::migrate`] on load. Tested against a
//!   frozen v1 project in `tests/golden/`.
//! * **Newer schema**: *refused*, with the version in the message
//!   ([`crate::MigrationError::FromTheFuture`]). It is not loaded and therefore
//!   never re-saved, so this build cannot silently delete a newer writer's data.
//! * **Unknown JSON fields** inside a *current-version* document are dropped on
//!   load, and unknown archive members are not carried over on save. That is
//!   safe only because of the rule above; if `.nguo` ever gains a second writer
//!   at the same schema version, preserve the raw `Value` instead.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use nguo_measure::Unit;
use serde::{Deserialize, Serialize};

use crate::doc::Document;
use crate::error::DocError;
use crate::migrate::SCHEMA_VERSION;
use crate::zip::{self, ZipError};

pub const MANIFEST: &str = "manifest.json";
pub const DOCUMENT: &str = "document.json";
pub const PREVIEW: &str = "preview.svg";
/// What `Document`'s millimetres mean once they reach the solver: one place
/// says it, and the manifest repeats it to the reader.
const COORDS: &str =
    "pattern space: millimetres, +X right, +Y down. 3D scene: metres, +Y up, +Z forward, \
     right-handed, feet at y=0. mm -> m happens once, at nguo_triangulate::MM_TO_M.";

#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("{path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{path} is not a .nguo project: {source}")]
    Zip {
        path: String,
        #[source]
        source: ZipError,
    },
    #[error("{path} has no {member} — it is a zip, but not a .nguo project")]
    MissingMember { path: String, member: &'static str },
    #[error("{path}: {member} is not valid JSON: {source}")]
    BadJson {
        path: String,
        member: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "{path} is inconsistent: {MANIFEST} says schema {manifest}, {DOCUMENT} says {document}"
    )]
    VersionMismatch {
        path: String,
        manifest: u32,
        document: u32,
    },
    #[error("{path}: {source}")]
    Doc {
        path: String,
        #[source]
        source: DocError,
    },
}

/// `manifest.json`: enough to understand the file without this program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Fixed marker, so a tool can tell a `.nguo` from any other zip.
    pub format: String,
    /// Document schema version — the same ladder [`crate::migrate`] walks.
    /// There is deliberately no *second* version for the archive layout: a
    /// layout change is a schema change, one number, one migration path.
    pub schema_version: u32,
    /// The build that wrote this file.
    pub writer: String,
    /// Stable identity of the project across saves.
    pub document_id: String,
    /// RFC 3339 UTC. `created` survives every save; `modified` is set by each.
    pub created: String,
    pub modified: String,
    /// The project name, so a picker need not read `document.json`.
    pub name: String,
    /// Required, never defaulted: a file that does not state its units and
    /// handedness is the bug this whole manifest exists to prevent.
    pub units: Units,
    /// Informational, recomputed on every save — hence `default`, so a partial
    /// or older manifest still loads rather than failing on a summary.
    #[serde(default)]
    pub counts: Counts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avatar: Option<AvatarLink>,
    /// Every member in the archive, so an incomplete file is obvious.
    #[serde(default)]
    pub members: Vec<String>,
    #[serde(default)]
    pub compatibility: Compatibility,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Units {
    /// Always `"millimetre"`. Patterns are authored in mm (D-18).
    pub geometry: String,
    /// What the UI shows. Never affects stored values.
    pub display: Unit,
    pub coordinates: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Counts {
    pub pieces: usize,
    pub seams: usize,
    pub materials: usize,
    pub params: usize,
    pub vertices: usize,
    pub edges: usize,
}

/// The avatar this project is drafted for — **referenced, never embedded**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvatarLink {
    /// The manifest key: `"male"` / `"female"`.
    pub key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// `bundle_version` from `assets/avatars/data/manifest.json`. `None` means
    /// the project predates the field; resolve by key and warn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_version: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pose: String,
    pub embedded: bool,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Compatibility {
    /// What a reader of this build does with a *newer* schema.
    pub newer_schema: String,
    pub older_schema: String,
    pub unknown_fields: String,
}

impl Default for Compatibility {
    fn default() -> Self {
        Compatibility {
            newer_schema: format!(
                "refused with the version in the message; this build reads up to {SCHEMA_VERSION}"
            ),
            older_schema: "migrated forward on load".into(),
            unknown_fields: "dropped on load; a newer file is refused, so nothing newer is \
                             re-saved lossily"
                .into(),
        }
    }
}

/// A loaded (or about-to-be-saved) `.nguo`: the document plus the manifest that
/// describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Project {
    pub manifest: Manifest,
    pub document: Document,
}

impl Project {
    /// A brand-new project file for `document`. Fresh id, `created` = now.
    pub fn new(document: Document) -> Project {
        let now = now_rfc3339();
        Project {
            manifest: Manifest {
                format: "nguo-3d project".into(),
                schema_version: document.schema_version(),
                writer: concat!("nguo-doc ", env!("CARGO_PKG_VERSION")).into(),
                document_id: fresh_id(),
                created: now.clone(),
                modified: now,
                name: document.meta().name.clone(),
                units: Units {
                    geometry: "millimetre".into(),
                    display: document.display_unit(),
                    coordinates: COORDS.into(),
                },
                counts: Counts::default(),
                avatar: None,
                members: Vec::new(),
                compatibility: Compatibility::default(),
            },
            document,
        }
    }

    /// Write the bundle, returning its size in bytes.
    ///
    /// Refreshes the derived parts of the manifest first — name, counts, avatar
    /// link, `modified` — so they can never disagree with the document.
    pub fn save(&mut self, path: &Path) -> Result<u64, BundleError> {
        self.manifest.schema_version = self.document.schema_version();
        self.manifest.name = self.document.meta().name.clone();
        self.manifest.units.display = self.document.display_unit();
        self.manifest.counts = counts(&self.document);
        self.manifest.avatar = self.document.avatar().map(|a| AvatarLink {
            key: a.name.clone(),
            tier: a.tier.clone(),
            bundle_version: a.bundle_version.clone(),
            pose: a.pose.clone(),
            embedded: false,
            note: "referenced by key, never embedded: resolve against \
                   assets/avatars/data/manifest.json (CC BY 4.0, Git-LFS, 227 MB). If the \
                   bundle is missing or a different version, say so — never substitute \
                   another body."
                .into(),
        });
        self.manifest.members = vec![MANIFEST.into(), DOCUMENT.into(), PREVIEW.into()];
        self.manifest.modified = now_rfc3339();

        let document = self
            .document
            .to_json_pretty()
            .map_err(|source| BundleError::Doc {
                path: shown(path),
                source,
            })?;
        let manifest = serde_json::to_string_pretty(&self.manifest).map_err(|source| {
            BundleError::BadJson {
                path: shown(path),
                member: MANIFEST,
                source,
            }
        })?;

        let mut w = zip::Writer::new(dos_datetime(unix_now()));
        // `expect`: these three names are literals and pass `check_name`.
        for (name, body) in [
            (MANIFEST, manifest.as_bytes()),
            (DOCUMENT, document.as_bytes()),
            (PREVIEW, preview_svg(&self.document).as_bytes()),
        ] {
            w.add(name, body).expect("a literal member name is valid");
        }
        let bytes = w.finish();
        std::fs::write(path, &bytes).map_err(|source| BundleError::Io {
            path: shown(path),
            source,
        })?;
        Ok(bytes.len() as u64)
    }
}

/// Open a `.nguo`. One call from a path to a live document:
/// `nguo_doc::bundle::load(path)?.document`.
///
/// Nothing is written, anywhere: members are decompressed into memory, so no
/// member name — however hostile — can name a file on disk.
pub fn load(path: &Path) -> Result<Project, BundleError> {
    let raw = std::fs::read(path).map_err(|source| BundleError::Io {
        path: shown(path),
        source,
    })?;
    from_bytes(&raw, &shown(path))
}

/// The body of [`load`], split out so the tests can hand it hostile bytes.
pub fn from_bytes(raw: &[u8], path: &str) -> Result<Project, BundleError> {
    let members = zip::read_all(raw).map_err(|source| BundleError::Zip {
        path: path.to_owned(),
        source,
    })?;
    let member = |name: &'static str| -> Result<&Vec<u8>, BundleError> {
        members.get(name).ok_or(BundleError::MissingMember {
            path: path.to_owned(),
            member: name,
        })
    };
    let parse = |name: &'static str, bytes: &[u8]| -> Result<serde_json::Value, BundleError> {
        serde_json::from_slice(bytes).map_err(|source| BundleError::BadJson {
            path: path.to_owned(),
            member: name,
            source,
        })
    };

    let manifest: Manifest =
        serde_json::from_slice(member(MANIFEST)?).map_err(|source| BundleError::BadJson {
            path: path.to_owned(),
            member: MANIFEST,
            source,
        })?;
    let value = parse(DOCUMENT, member(DOCUMENT)?)?;
    let claimed = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::from(manifest.schema_version)) as u32;
    if claimed != manifest.schema_version {
        return Err(BundleError::VersionMismatch {
            path: path.to_owned(),
            manifest: manifest.schema_version,
            document: claimed,
        });
    }
    // Migrates an older schema, refuses a newer one.
    let document = Document::from_value(value).map_err(|source| BundleError::Doc {
        path: path.to_owned(),
        source,
    })?;
    let mut manifest = manifest;
    manifest.schema_version = document.schema_version();
    Ok(Project { manifest, document })
}

fn counts(doc: &Document) -> Counts {
    let mut c = Counts {
        pieces: doc.pieces().len(),
        seams: doc.seams().len(),
        materials: doc.materials().len(),
        params: doc.params().len(),
        ..Counts::default()
    };
    for (_, p) in doc.pieces().iter() {
        c.vertices += p.network.vertices.len();
        c.edges += p.network.edges.len();
    }
    c
}

fn shown(path: &Path) -> String {
    path.display().to_string()
}

/// A 2D outline of the pattern, one `<path>` per piece, in millimetres.
///
/// SVG rather than PNG on purpose: it is text (so it diffs and inspects like
/// the rest of the bundle), it needs no encoder, and it costs a couple of KB.
/// Its absence must never block loading — [`load`] does not even look at it.
pub fn preview_svg(doc: &Document) -> String {
    use kurbo::{BezPath, Rect, Shape};

    let mut paths = Vec::new();
    let mut bbox: Option<Rect> = None;
    for (_, piece) in doc.pieces().iter() {
        let mut path = BezPath::new();
        for (id, _) in piece.network.edges.iter() {
            if let Ok(seg) = piece.network.seg(id) {
                path.extend(seg.path_elements(0.1));
            }
        }
        if path.is_empty() {
            continue;
        }
        let path = piece.transform * path;
        let b = path.bounding_box();
        bbox = Some(bbox.map_or(b, |o: Rect| o.union(b)));
        paths.push(path.to_svg());
    }

    let view = bbox
        .unwrap_or(Rect::new(0.0, 0.0, 100.0, 100.0))
        .inflate(10.0, 10.0);
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{:.1} {:.1} {:.1} {:.1}\" \
         width=\"256\">\n<title>{} — nguo-3d pattern preview (millimetres)</title>\n",
        view.x0,
        view.y0,
        view.width().max(1.0),
        view.height().max(1.0),
        escape(&doc.meta().name),
    );
    for p in paths {
        svg.push_str(&format!(
            "<path fill=\"none\" stroke=\"black\" stroke-width=\"{:.2}\" d=\"{p}\"/>\n",
            view.width().max(1.0) / 256.0
        ));
    }
    svg.push_str("</svg>\n");
    svg
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// ponytail: a project id only has to be unique on one machine's disk, so it is
/// the save time in nanoseconds, not a UUID (no `uuid` crate in this tree).
/// Swap in a v4 UUID if projects ever sync between machines.
fn fresh_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("nguo-{nanos:x}")
}

fn now_rfc3339() -> String {
    rfc3339(unix_now())
}

/// `secs` since the epoch as RFC 3339 UTC. Twelve lines instead of `chrono`,
/// which is in the lockfile but only inside `nguo-mcp`'s subtree.
pub fn rfc3339(secs: u64) -> String {
    let (y, m, d, hh, mm, ss) = civil(secs);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// MS-DOS `(time, date)` for a zip header, so `unzip -l` prints the truth.
pub fn dos_datetime(secs: u64) -> (u16, u16) {
    let (y, m, d, hh, mm, ss) = civil(secs);
    let year = y.saturating_sub(1980).clamp(0, 127) as u16;
    (
        (hh as u16) << 11 | (mm as u16) << 5 | (ss as u16 / 2),
        year << 9 | (m as u16) << 5 | d as u16,
    )
}

/// Days-since-epoch to a civil date (Howard Hinnant's `civil_from_days`).
fn civil(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + u64::from(m <= 2);
    (y, m, d, rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Every member of a `.nguo`, for tools that want more than the manifest.
pub fn members(raw: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, ZipError> {
    zip::read_all(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_epoch_and_a_known_date_format_correctly() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        // 2026-08-05T12:34:56Z — `date -u -j -f ... +%s` agrees.
        assert_eq!(rfc3339(1_785_933_296), "2026-08-05T12:34:56Z");
        // A leap day, which is where a hand-rolled calendar goes wrong.
        assert_eq!(rfc3339(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn dos_datetime_packs_the_same_instant() {
        let (time, date) = dos_datetime(1_785_933_296); // 2026-08-05 12:34:56
        assert_eq!((date >> 9) + 1980, 2026);
        assert_eq!((date >> 5) & 0xf, 8);
        assert_eq!(date & 0x1f, 5);
        assert_eq!(time >> 11, 12);
        assert_eq!((time >> 5) & 0x3f, 34);
        assert_eq!((time & 0x1f) * 2, 56);
    }

    #[test]
    fn a_preview_is_produced_even_for_an_empty_project() {
        let svg = preview_svg(&Document::new());
        assert!(svg.starts_with("<svg"), "{svg}");
        assert!(svg.ends_with("</svg>\n"));
    }
}
