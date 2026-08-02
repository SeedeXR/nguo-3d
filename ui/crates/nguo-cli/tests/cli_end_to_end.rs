// SPDX-License-Identifier: GPL-2.0-or-later
//! End-to-end: run the real `nguo` binary against real `core-engine/examples`
//! scenes and assert the JSON shape, the stdout/stderr split and the exit code
//! (`core-engine/docs/PROCESS_CONTRACT.md`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn repo_root() -> PathBuf {
    // ui/crates/nguo-cli -> nguo-3d
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root")
        .to_path_buf()
}

fn example(name: &str) -> PathBuf {
    repo_root().join("core-engine/examples").join(name)
}

fn nguo(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nguo"))
        .args(args)
        .output()
        .expect("running the nguo binary")
}

fn stdout_json(out: &Output) -> Value {
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "stdout was not a single JSON value ({e}):\n{}",
            String::from_utf8_lossy(&out.stdout)
        )
    })
}

/// The engine binary the drape test needs. Built by
/// `cargo build -p engine-cli` in `core-engine/`.
fn engine_available() -> bool {
    nguo_cli::ops::Env::default().engine.is_file()
}

#[test]
fn schema_lists_the_whole_surface_and_the_process_contract() {
    let out = nguo(&["schema"]);
    assert!(out.status.success(), "exit {:?}", out.status.code());
    let schema = stdout_json(&out);
    assert_eq!(schema["tool"], "nguo");
    assert!(schema["process_contract"]["exit_codes"]["2"].is_string());
    let ops = schema["operations"].as_array().expect("operations array");
    assert_eq!(ops.len(), nguo_cli::ops::SPECS.len());
    for op in ops {
        assert!(op["name"].is_string());
        assert!(op["input_schema"]["type"] == "object");
        assert!(!op["description"].as_str().unwrap_or_default().is_empty());
    }
    assert!(ops.iter().any(|o| o["name"] == "drape"));
}

#[test]
fn scene_info_on_a_real_example_scene() {
    let out = nguo(&["scene-info", example("sphere_drape.json").to_str().unwrap()]);
    assert!(out.status.success(), "exit {:?}", out.status.code());
    let info = stdout_json(&out);
    assert_eq!(info["totals"]["objects"], 1);
    assert_eq!(info["totals"]["verts"], 144);
    assert_eq!(info["totals"]["tris"], 242);
    assert_eq!(info["objects"][0]["name"], "drape");
    assert_eq!(info["colliders"][0]["type"], "sphere");
    assert_eq!(info["config"]["frames"], 90);
}

#[test]
fn scene_validate_passes_every_shipped_example() {
    for name in [
        "sphere_drape.json",
        "fabric_drape.json",
        "hanging_cloth.json",
        "large_fabric.json",
    ] {
        let out = nguo(&["scene-validate", example(name).to_str().unwrap()]);
        assert!(
            out.status.success(),
            "{name} failed validation (exit {:?}):\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout)
        );
        assert_eq!(stdout_json(&out)["ok"], true, "{name}");
    }
}

#[test]
fn an_invalid_scene_exits_2_and_still_prints_its_report() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.json");
    std::fs::write(
        &path,
        r#"{"config":{"dt":0.0,"fps":60.0,"frames":1},
            "objects":[{"name":"bad","positions":[[0,0,0],[1,0,0]],"tris":[[0,1,5]]}]}"#,
    )
    .unwrap();
    let out = nguo(&["scene-validate", path.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2), "invalid input must exit 2");
    let report = stdout_json(&out);
    assert_eq!(report["ok"], false);
    let failed: Vec<&str> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["ok"] == false)
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        failed,
        vec!["triangle-indices-in-range", "timestep-positive"]
    );
}

#[test]
fn a_missing_scene_exits_1_with_a_clean_stdout() {
    let out = nguo(&["scene-info", "/nonexistent/scene.json"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        out.stdout.is_empty(),
        "stdout must stay clean on failure, got {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ERROR"),
        "no ERROR log on stderr:\n{stderr}"
    );
}

#[test]
fn drape_runs_the_real_engine_and_writes_the_frame_sequence() {
    if !engine_available() {
        panic!(
            "mguso not found at {}. Build it with `cargo build -p engine-cli` in core-engine/, \
             or set $NGUO_MGUSO.",
            nguo_cli::ops::Env::default().engine.display()
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let out = nguo(&[
        "drape",
        example("sphere_drape.json").to_str().unwrap(),
        "--frames",
        "3",
        "--out",
        dir.path().to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "exit {:?}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let report = stdout_json(&out);
    assert_eq!(report["tier"], "preview");
    assert_eq!(report["frames_streamed"], 3);
    // The engine's own summary, verbatim.
    assert_eq!(report["engine"]["frames"], 3);
    assert_eq!(report["engine"]["verts"], 144);
    assert_eq!(report["engine"]["backend"], "xpbd");
    assert!(report["engine"]["final_time"].as_f64().unwrap() > 0.0);
    assert!(report["engine"]["bbox_min"].is_array());
    for frame in 0..3 {
        assert!(
            dir.path().join(format!("frame_{frame:04}.obj")).is_file(),
            "frame {frame} was not exported"
        );
    }
}

/// Determinism (guide §14): the same submission gives the same drape.
#[test]
fn the_same_drape_twice_gives_the_same_result() {
    if !engine_available() {
        panic!("mguso not found; see the drape test above");
    }
    let scene = example("fabric_drape.json");
    let run = || {
        let out = nguo(&["drape", scene.to_str().unwrap(), "--frames", "2"]);
        assert!(out.status.success());
        stdout_json(&out)["engine"].clone()
    };
    assert_eq!(run(), run());
}
