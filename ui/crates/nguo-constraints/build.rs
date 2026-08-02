// SPDX-License-Identifier: GPL-2.0-or-later
//
// Compiles FreeCAD's planegcs (vendored, LGPL-2.1-or-later) plus our C shim
// into one static archive and links it into this crate.
//
// Everything it needs is inside the repository: the planegcs sources, the
// Eigen headers, and the compat headers that stand in for FreeCAD's
// SketcherGlobal.h/Base::Console and for the handful of Boost Graph symbols
// planegcs uses. No Homebrew, no pkg-config, no system package. `cargo test`
// works on a clean machine with only a C++ toolchain.
//
// Build-time deps: `cc` only. `cxx`/`autocxx` would generate bindings for a
// class hierarchy we deliberately do not expose (see cpp/nguo_gcs.h), and
// would add a codegen step and two more crates to gain nothing over ~30
// hand-written `extern "C"` declarations.

use std::path::{Path, PathBuf};

/// planegcs translation units, in the vendored tree.
const PLANEGCS_SOURCES: [&str; 5] = [
    "Geo.cpp",
    "qp_eq.cpp",
    "SubSystem.cpp",
    "Constraints.cpp",
    "GCS.cpp",
];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // ui/crates/nguo-constraints -> repo root -> third_party/planegcs
    let third_party = manifest
        .join("../../../third_party/planegcs")
        .canonicalize()
        .expect(
            "third_party/planegcs is missing. It is vendored into this repository; \
             see third_party/planegcs/PROVENANCE.md",
        );

    let upstream = third_party.join("upstream/planegcs");
    let compat = third_party.join("compat");
    let eigen = third_party.join("vendor/eigen");

    for dir in [&upstream, &compat, &eigen] {
        assert!(
            dir.is_dir(),
            "expected vendored directory {}",
            dir.display()
        );
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .include(&compat)
        .include(&eigen)
        .include(&upstream)
        .include(manifest.join("cpp"))
        // planegcs' own note in GCS.cpp: Eigen's SparseQR assertions crash the
        // solver, and FreeCAD sets -DEIGEN_NO_DEBUG for exactly this reason.
        .define("EIGEN_NO_DEBUG", None)
        // Upstream is not warning-clean under our flags and we do not patch
        // it; our own shim is compiled in the same unit set, so keep the noise
        // down rather than pretend otherwise.
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        .warnings(false);

    for src in PLANEGCS_SOURCES {
        build.file(upstream.join(src));
    }
    build.file(manifest.join("cpp/nguo_gcs.cpp"));

    build.compile("nguo_planegcs");

    // Rebuild triggers. Eigen is a fixed vendored snapshot; watching 337
    // headers would slow every build for a directory that only changes when
    // someone deliberately re-vendors, so the directory itself is enough.
    println!("cargo:rerun-if-changed=cpp/nguo_gcs.cpp");
    println!("cargo:rerun-if-changed=cpp/nguo_gcs.h");
    rerun_dir(&upstream);
    rerun_dir(&compat);
    println!(
        "cargo:rerun-if-changed={}",
        third_party.join("SketcherGlobal.h").display()
    );
}

fn rerun_dir(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_dir(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
