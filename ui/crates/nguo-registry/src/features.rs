// SPDX-License-Identifier: GPL-2.0-or-later
//! The feature inventory: a compile-time table of everything the application
//! can do.
//!
//! # Why a `const` table and not a runtime registry
//!
//! A runtime registry (`inventory`, `linkme`, or a `register()` call in each
//! crate's init) can lie in three ways a `const` table cannot: it depends on
//! init order, it depends on the linker keeping a section it thinks is dead,
//! and it silently reports *nothing* when the registering crate was never
//! linked in. A `const` table has none of those failure modes, needs no
//! dependency, and is greppable.
//!
//! It is also the convention this tree already uses:
//! `nguo_cli::ops::SPECS` is exactly this shape (a `const` slice of
//! `&'static str`-described specs, shared by the CLI and the MCP server).
//! Two conventions for "a table of things the app can do" would be one too
//! many.
//!
//! The one thing a `const` table *can* do wrong is drift — someone adds a
//! feature and forgets the entry. Nothing prevents that in any scheme; what
//! this table does prevent, at **compile time**, is a duplicate or
//! out-of-order id (see the `const _: () = assert!(..)` below).

/// Which part of the application a feature belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Subsystem {
    /// The authored document, commands and undo.
    Document,
    /// Pattern geometry: vector networks, offsets, sewing, triangulation.
    Pattern,
    /// The 2D geometric constraint solver.
    Constraints,
    /// The 2D Vello canvas.
    Canvas,
    /// The 3D drape viewport.
    Viewport,
    /// The cloth solver and the bridge to it.
    Engine,
    /// DCC round-trips (Blender, Unity).
    Interop,
    /// Project and mesh file I/O.
    Io,
    /// Shell, widgets, measurement and the debug overlay.
    Ui,
    /// Headless surfaces: the `nguo` CLI and the MCP server.
    Tooling,
}

impl Subsystem {
    /// Lower-case stable name, for logs and the debug overlay.
    pub const fn as_str(self) -> &'static str {
        match self {
            Subsystem::Document => "document",
            Subsystem::Pattern => "pattern",
            Subsystem::Constraints => "constraints",
            Subsystem::Canvas => "canvas",
            Subsystem::Viewport => "viewport",
            Subsystem::Engine => "engine",
            Subsystem::Interop => "interop",
            Subsystem::Io => "io",
            Subsystem::Ui => "ui",
            Subsystem::Tooling => "tooling",
        }
    }
}

impl std::fmt::Display for Subsystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One thing the application can do.
#[derive(Debug)]
pub struct Feature {
    /// Stable, dotted, lower-case id. **Never change one** — it appears in
    /// logs, in `RUST_LOG` filters and in bug reports.
    pub id: &'static str,
    /// Human name, for the debug overlay and the UI.
    pub name: &'static str,
    /// Which part of the app this belongs to.
    pub subsystem: Subsystem,
    /// `false` = the code exists in the tree (or is planned) but is **not
    /// reachable from the running app** in this build. This is the field that
    /// answers "why does nothing happen when I click it".
    pub enabled: bool,
    /// The crate that owns the implementation.
    pub owner: &'static str,
    /// The `tracing` target the owning crate logs under. This is the single
    /// place that mapping lives — see [`Feature::rust_log`].
    pub target: &'static str,
    /// `true` if instances of this show up in the live process list
    /// (`crate::Registry::start`).
    pub process: bool,
    /// One line, true.
    pub summary: &'static str,
}

impl Feature {
    /// The `RUST_LOG` value that turns on debug logging for exactly this
    /// feature: the owning crate's target, plus this feature's process spans.
    ///
    /// ```
    /// let f = nguo_registry::feature_const("drape.preview");
    /// assert_eq!(
    ///     f.rust_log("debug"),
    ///     "nguo_engine_bridge=debug,nguo::process[{feature=\"drape.preview\"}]=debug"
    /// );
    /// ```
    pub fn rust_log(&self, level: &str) -> String {
        format!(
            "{}={level},{}[{{feature=\"{}\"}}]={level}",
            self.target,
            crate::PROCESS_TARGET,
            self.id
        )
    }
}

/// Every feature in the application, **sorted by `id`**.
///
/// Adding one is the whole recipe: append an entry, keep the list sorted, and
/// the compile-time assertion below does the rest. See `ui/docs/OBSERVABILITY.md`.
pub const FEATURES: &[Feature] = &[
    Feature {
        id: "canvas.render",
        name: "2D pattern canvas",
        subsystem: Subsystem::Canvas,
        enabled: true,
        owner: "nguo-canvas",
        target: "nguo_canvas",
        process: false,
        summary: "Vello scene for the pattern canvas, with pan/zoom and pointer hit-testing.",
    },
    Feature {
        id: "constraints.solve",
        name: "2D constraint solve",
        subsystem: Subsystem::Constraints,
        enabled: true,
        owner: "nguo-constraints",
        target: "nguo_constraints",
        process: true,
        summary: "Solve a parametric sketch through planegcs, with DOF diagnosis.",
    },
    Feature {
        id: "dcc.roundtrip",
        name: "DCC round-trip",
        subsystem: Subsystem::Interop,
        enabled: true,
        owner: "nguo-bridge-dcc",
        target: "nguo_bridge_dcc",
        process: true,
        summary: "Drive headless Blender or Unity and read the result back, axis-converted.",
    },
    Feature {
        id: "doc.command",
        name: "Document command",
        subsystem: Subsystem::Document,
        enabled: true,
        owner: "nguo-doc-commands",
        target: "nguo_doc_commands",
        process: false,
        summary: "Apply a transaction of primitive edits as one undo entry.",
    },
    Feature {
        id: "doc.undo",
        name: "Undo / redo",
        subsystem: Subsystem::Document,
        enabled: true,
        owner: "nguo-doc-commands",
        target: "nguo_doc_commands",
        process: false,
        summary: "Linear undo stack with gesture coalescing.",
    },
    Feature {
        id: "drape.bake",
        name: "Drape bake",
        subsystem: Subsystem::Engine,
        enabled: true,
        owner: "nguo-engine-bridge",
        target: "nguo_engine_bridge",
        process: true,
        summary: "Full-quality implicit-Euler drape; every frame is delivered (Tier::Bake).",
    },
    Feature {
        id: "drape.preview",
        name: "Drape preview",
        subsystem: Subsystem::Engine,
        enabled: true,
        owner: "nguo-engine-bridge",
        target: "nguo_engine_bridge",
        process: true,
        summary: "Interactive XPBD drape; frames may be dropped for latency (Tier::Preview).",
    },
    Feature {
        id: "gpu.handoff",
        name: "Drape GPU handoff",
        subsystem: Subsystem::Engine,
        enabled: true,
        owner: "nguo-gpu",
        target: "nguo_gpu",
        process: true,
        summary: "Async triple-buffered readback of solver positions onto the UI device.",
    },
    Feature {
        id: "io.export",
        name: "Mesh / pattern export",
        subsystem: Subsystem::Io,
        enabled: false,
        owner: "nguo-io",
        target: "nguo_io",
        process: true,
        summary: "Write OBJ/glTF/USD/DXF. Crate not on disk yet — nothing runs.",
    },
    Feature {
        id: "io.open",
        name: "Open project",
        subsystem: Subsystem::Io,
        enabled: false,
        owner: "nguo-io",
        target: "nguo_io",
        process: true,
        summary: "Read a zipped `.nguo` bundle and migrate it forward. Crate not on disk yet.",
    },
    Feature {
        id: "io.save",
        name: "Save project",
        subsystem: Subsystem::Io,
        enabled: false,
        owner: "nguo-io",
        target: "nguo_io",
        process: true,
        summary: "Write a zipped `.nguo` bundle. Crate not on disk yet.",
    },
    Feature {
        id: "mcp.serve",
        name: "MCP server",
        subsystem: Subsystem::Tooling,
        enabled: true,
        owner: "nguo-mcp",
        target: "nguo_mcp",
        process: true,
        summary: "Expose nguo_cli::ops::SPECS as MCP tools over stdio.",
    },
    Feature {
        id: "measure.expr",
        name: "Numeric expression",
        subsystem: Subsystem::Ui,
        enabled: true,
        owner: "nguo-measure",
        target: "nguo_measure",
        process: false,
        summary: "Evaluate `waist/4 + 1cm` in any numeric field, with cycle detection.",
    },
    Feature {
        id: "pattern.offset",
        name: "Seam allowance offset",
        subsystem: Subsystem::Pattern,
        enabled: true,
        owner: "nguo-pattern",
        target: "nguo_pattern",
        process: false,
        summary: "Offset a vector-network boundary by a per-edge allowance.",
    },
    Feature {
        id: "sewing.build",
        name: "Seam topology build",
        subsystem: Subsystem::Pattern,
        enabled: false,
        owner: "nguo-sewing",
        target: "nguo_sewing",
        process: true,
        summary: "Resolve seam pairings into stitch constraints. Crate not on disk yet.",
    },
    Feature {
        id: "triangulate.piece",
        name: "Pattern triangulation",
        subsystem: Subsystem::Pattern,
        enabled: false,
        owner: "nguo-triangulate",
        target: "nguo_triangulate",
        process: true,
        summary: "Mesh a pattern piece for the solver. Crate not on disk yet.",
    },
    Feature {
        id: "ui.debug_overlay",
        name: "Debug overlay",
        subsystem: Subsystem::Ui,
        // The overlay is gated on `nguo-app`'s `debug-overlay` **cargo
        // feature**, not on `debug_assertions`: a plain `cargo run -p nguo-app`
        // is a debug build with no overlay compiled in. This crate cannot see
        // another crate's features, so `nguo-app` forwards its own
        // (`debug-overlay = [.., "nguo-registry/debug-overlay"]`) and cargo's
        // feature unification makes the two flags the same flag.
        enabled: cfg!(feature = "debug-overlay"),
        owner: "nguo-app-debug",
        target: "nguo_app_debug",
        process: false,
        summary: "egui overlay with frame timings, device info and the live process list.",
    },
    Feature {
        id: "viewport.render",
        name: "3D drape viewport",
        subsystem: Subsystem::Viewport,
        enabled: true,
        owner: "nguo-viewport3d",
        target: "nguo_viewport3d",
        process: false,
        summary: "Draped garment and avatar, as an iced custom primitive on the shared device.",
    },
];

// -- compile-time integrity of the table ------------------------------------
//
// A duplicate or out-of-order id is a *compile* error, not a test failure:
// `FEATURES` is `const`, so the check costs nothing at runtime and cannot be
// skipped by running the wrong test target. Sortedness is not cosmetic — it is
// what makes the duplicate check a single linear pass, and what makes
// `feature()` a binary search.
const _: () = assert!(
    ids_strictly_ascending(FEATURES),
    "nguo_registry::FEATURES must be sorted by `id`, ascending, with no duplicates"
);

const fn ids_strictly_ascending(fs: &[Feature]) -> bool {
    let mut i = 1;
    while i < fs.len() {
        if !str_lt(fs[i - 1].id, fs[i].id) {
            return false;
        }
        i += 1;
    }
    true
}

const fn str_lt(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let n = if a.len() < b.len() { a.len() } else { b.len() };
    let mut i = 0;
    while i < n {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
        i += 1;
    }
    a.len() < b.len()
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Look a feature up by id at runtime. `None` if it is not in the inventory.
///
/// Binary search — the table is sorted by construction (see the `const`
/// assertion above).
pub fn feature(id: &str) -> Option<&'static Feature> {
    FEATURES
        .binary_search_by(|f| f.id.cmp(id))
        .ok()
        .map(|i| &FEATURES[i])
}

/// Look a feature up **at compile time**. A typo is a build error.
///
/// This is how call sites should name a feature, so that a `&'static Feature`
/// — the only thing [`crate::Registry::start`] accepts — can never refer to
/// something outside the inventory:
///
/// ```
/// const DRAPE: &nguo_registry::Feature = nguo_registry::feature_const("drape.preview");
/// assert_eq!(DRAPE.owner, "nguo-engine-bridge");
/// ```
pub const fn feature_const(id: &str) -> &'static Feature {
    let mut i = 0;
    while i < FEATURES.len() {
        if str_eq(FEATURES[i].id, id) {
            return &FEATURES[i];
        }
        i += 1;
    }
    panic!("unknown feature id — add it to nguo_registry::FEATURES")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The `const` assertion above already makes a duplicate a build error.
    /// This proves the checker itself is not vacuously true, and gives a
    /// readable failure if someone ever weakens it.
    #[test]
    fn ids_are_unique_and_sorted() {
        let ids: Vec<&str> = FEATURES.iter().map(|f| f.id).collect();
        let unique: BTreeSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "duplicate feature id");
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "FEATURES is not sorted by id");
        let out_of_order = [
            Feature {
                id: "b",
                ..copy_of(&FEATURES[0])
            },
            Feature {
                id: "a",
                ..copy_of(&FEATURES[0])
            },
        ];
        assert!(!ids_strictly_ascending(&out_of_order));
        let duplicated = [
            Feature {
                id: "a",
                ..copy_of(&FEATURES[0])
            },
            Feature {
                id: "a",
                ..copy_of(&FEATURES[0])
            },
        ];
        assert!(!ids_strictly_ascending(&duplicated));
    }

    fn copy_of(f: &Feature) -> Feature {
        Feature {
            id: f.id,
            name: f.name,
            subsystem: f.subsystem,
            enabled: f.enabled,
            owner: f.owner,
            target: f.target,
            process: f.process,
            summary: f.summary,
        }
    }

    /// The inventory is only useful if every field is filled in honestly.
    #[test]
    fn every_entry_is_complete() {
        for f in FEATURES {
            assert!(!f.id.is_empty(), "empty id");
            assert!(
                f.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c == '.' || c == '_'),
                "{}: ids are lower-case dotted ascii",
                f.id
            );
            assert!(f.id.contains('.'), "{}: id must be `subsystem.thing`", f.id);
            assert!(!f.name.is_empty(), "{}: no name", f.id);
            assert!(f.owner.starts_with("nguo-"), "{}: owner crate", f.id);
            assert_eq!(
                f.target,
                f.owner.replace('-', "_"),
                "{}: tracing target is the owning crate's module path",
                f.id
            );
            assert!(f.summary.len() > 20, "{}: summary is not a summary", f.id);
            assert!(f.summary.ends_with('.'), "{}: summary is a sentence", f.id);
        }
    }

    #[test]
    fn lookup_finds_every_id_and_nothing_else() {
        for f in FEATURES {
            assert_eq!(feature(f.id).map(|g| g.id), Some(f.id));
        }
        assert!(feature("nope.nothing").is_none());
        assert!(feature("").is_none());
    }

    #[test]
    fn rust_log_recipe_names_both_targets() {
        let f = feature_const("constraints.solve");
        assert_eq!(
            f.rust_log("trace"),
            "nguo_constraints=trace,nguo::process[{feature=\"constraints.solve\"}]=trace"
        );
    }
}
