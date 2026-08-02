// SPDX-License-Identifier: GPL-2.0-or-later
//! End-to-end tests against the *real* DCCs.
//!
//! Gated: each one skips cleanly, and loudly, when its DCC is not installed —
//! the same idiom as `core-engine/crates/engine-io/tests/golden_file.rs`, which
//! skips when the reference gauge directory is absent. Run with
//! `cargo test -p nguo-bridge-dcc -- --nocapture` to see the skip reasons.

use nguo_bridge_dcc::{blender_argv, discover, run, unity_argv, BridgeError, Dcc};
use std::path::{Path, PathBuf};

/// Repo root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repo root")
}

fn skip(what: &str, why: &str) {
    println!("SKIPPED {what}: {why}");
}

/// Blender: run the add-on's headless smoke test and require it to pass.
///
/// This is the primary tested path for the Blender bridge — the N-panel is a
/// convenience on top of exactly these code paths.
#[test]
fn blender_smoke_test_passes() {
    let exe = match discover::find(Dcc::Blender) {
        Ok(e) => e,
        Err(e) => return skip("blender_smoke_test_passes", &e.to_string()),
    };
    let script = repo_root().join("bridges/blender/tests/smoke.py");
    assert!(script.is_file(), "missing {}", script.display());

    let out = run(Dcc::Blender, &exe, &blender_argv(&script, &[]))
        .unwrap_or_else(|e| panic!("blender smoke test failed:\n{e}"));
    let passed = out.result.as_ref().and_then(|v| v["passed"].as_u64());
    assert!(
        passed.is_some_and(|n| n >= 9),
        "expected >= 9 passing checks, got {passed:?}"
    );
    println!("blender {} — {passed:?} checks passed", exe.display());
}

/// A *script* error must classify as `BridgeError::Script`, never as `Dcc`.
///
/// Regression test for a real defect. `blender --python-exit-code 47` only remaps an
/// **uncaught exception**; `SystemExit` — which is what `raise SystemExit(...)` and
/// argparse's own `sys.exit(2)` produce — bypasses it entirely. So a typo'd object name
/// exited 1 and a bad flag exited 2, and `classify` read both as `BridgeError::Dcc`,
/// telling the user "Blender itself failed (bad project, no licence, crash)" when in
/// fact their argument was wrong.
///
/// Nothing exercised `headless.py`'s failure paths, which is precisely why it survived
/// review. This test covers both shapes.
#[test]
fn blender_script_errors_classify_as_script_not_dcc() {
    let exe = match discover::find(Dcc::Blender) {
        Ok(e) => e,
        Err(e) => {
            return skip(
                "blender_script_errors_classify_as_script_not_dcc",
                &e.to_string(),
            )
        }
    };
    let script = repo_root().join("bridges/blender/headless.py");
    assert!(script.is_file(), "missing {}", script.display());

    // (a) a name that does not exist in the scene, and (b) an unrecognised flag.
    for (what, args) in [
        (
            "unknown object",
            vec![
                "export-collider".to_string(),
                "--object".to_string(),
                "ThisObjectDoesNotExist".to_string(),
                "--out".to_string(),
                "/tmp/nguo-should-not-be-written.obj".to_string(),
            ],
        ),
        (
            "bad flag",
            vec![
                "export-collider".to_string(),
                "--definitely-not-a-flag".to_string(),
            ],
        ),
    ] {
        let err = run(Dcc::Blender, &exe, &blender_argv(&script, &args))
            .expect_err(&format!("{what}: expected failure, got success"));
        assert!(
            matches!(err, BridgeError::Script { .. }),
            "{what}: expected BridgeError::Script, got {err:?} \
             (SystemExit is escaping --python-exit-code again)"
        );
        println!("blender {what} -> {err}");
    }
}

/// Blender: the add-on registers and every operator we declare exists.
#[test]
fn blender_addon_registers_headlessly() {
    let exe = match discover::find(Dcc::Blender) {
        Ok(e) => e,
        Err(e) => return skip("blender_addon_registers_headlessly", &e.to_string()),
    };
    let script = repo_root().join("bridges/blender/headless.py");
    let out = run(
        Dcc::Blender,
        &exe,
        &blender_argv(&script, &["selftest".into()]),
    )
    .unwrap_or_else(|e| panic!("blender selftest failed:\n{e}"));

    let result = out.result.expect("result line");
    assert_eq!(result["registered"], true);
    let ops = result["operators"].as_array().expect("operators array");
    for want in [
        "nguo.import_glb",
        "nguo.import_sequence",
        "nguo.export_collider",
        "nguo.drape",
    ] {
        assert!(
            ops.iter().any(|o| o == want),
            "operator {want} missing from {ops:?}"
        );
    }
}

/// Blender: the extension manifest is valid per Blender's own validator.
#[test]
fn blender_extension_manifest_is_valid() {
    let exe = match discover::find(Dcc::Blender) {
        Ok(e) => e,
        Err(e) => return skip("blender_extension_manifest_is_valid", &e.to_string()),
    };
    let dir = repo_root().join("bridges/blender/nguo_bridge");
    let out = std::process::Command::new(&exe)
        .args(["--command", "extension", "validate"])
        .arg(&dir)
        .output()
        .expect("spawn blender");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && text.contains("Success"),
        "manifest rejected:\n{text}{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Unity: prove the package compiles and its axis maths agrees with ours.
///
/// Needs a project that contains the package, so it builds a throwaway one and
/// symlinks `bridges/unity` into `Packages/`.
#[test]
fn unity_selftest_passes() {
    let exe = match discover::find(Dcc::Unity) {
        Ok(e) => e,
        Err(e) => return skip("unity_selftest_passes", &e.to_string()),
    };
    let project = match scratch_unity_project() {
        Ok(p) => p,
        Err(e) => {
            return skip(
                "unity_selftest_passes",
                &format!("could not stage project: {e}"),
            )
        }
    };
    let argv = unity_argv(&project, "Seedexr.Nguo.Bridge.NguoHeadless.SelfTest", &[]);
    match run(Dcc::Unity, &exe, &argv) {
        Ok(out) => {
            assert_eq!(out.result.expect("result line")["ok"], true);
            println!("unity {} — selftest passed", exe.display());
        }
        // Not a test failure: a machine can have Unity installed without a
        // usable licence, and that is exactly the case on the dev box this was
        // written on (Unity 6000.4.8f1 exits 198).
        Err(BridgeError::NoLicence { .. }) => skip(
            "unity_selftest_passes",
            "Unity is installed but has no activated licence (exit 198)",
        ),
        Err(e) => panic!("unity selftest failed:\n{e}"),
    }
}

/// A minimal Unity project with the bridge package linked in.
fn scratch_unity_project() -> std::io::Result<PathBuf> {
    let root = std::env::temp_dir().join("nguo-bridge-dcc-unity");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("Assets"))?;
    std::fs::create_dir_all(root.join("Packages"))?;
    std::fs::create_dir_all(root.join("ProjectSettings"))?;
    std::fs::write(
        root.join("ProjectSettings/ProjectVersion.txt"),
        "m_EditorVersion: 6000.4.8f1\n",
    )?;
    std::fs::write(
        root.join("Packages/manifest.json"),
        "{\"dependencies\": {}}\n",
    )?;
    let pkg = repo_root().join("bridges/unity");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&pkg, root.join("Packages/com.seedexr.nguo-bridge"))?;
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(&pkg, root.join("Packages/com.seedexr.nguo-bridge"))?;
    Ok(root)
}
