// SPDX-License-Identifier: GPL-2.0-or-later
//! The operation surface — **the** implementation, shared by the `nguo` CLI
//! and the `nguo-mcp` MCP server.
//!
//! There is exactly one list of operations ([`SPECS`]) and one dispatcher
//! ([`dispatch`]). The CLI derives its subcommands from it and the MCP server
//! derives its tools from it, so the two surfaces cannot drift; a test in each
//! crate fails if they do.
//!
//! Every operation here is backed by something that actually runs today: the
//! `mguso` engine binary, or local parsing of an `engine-api` scene document.
//! Nothing in this list is a stub. Document-level operations (open a project,
//! list pieces/seams, apply edits) are **absent**, not stubbed, because
//! `nguo-doc` does not exist yet — see `ui/docs/AGENTIC.md`.

use std::path::{Path, PathBuf};

use nguo_engine_bridge::{
    resolve_engine_binary, EngineBridge, ProcessTransport, Submission, Tier, Update,
};
use serde_json::{json, Value};

/// One operation, described once for both front ends.
pub struct OpSpec {
    /// Canonical snake_case name. This is the MCP tool name.
    pub name: &'static str,
    /// One line. An agent picks tools by this, so it must be true.
    pub summary: &'static str,
    /// The honest long form: what it really does, and what it does not.
    pub description: &'static str,
    /// Does not modify anything outside its own temp scratch.
    pub read_only: bool,
    /// Calling it twice with the same arguments leaves the same state.
    pub idempotent: bool,
    /// JSON Schema (draft 2020-12 subset) for the arguments object.
    pub schema: fn() -> Value,
}

impl OpSpec {
    /// The CLI subcommand name for this operation.
    pub fn cli_name(&self) -> String {
        self.name.replace('_', "-")
    }
}

pub const SPECS: &[OpSpec] = &[
    OpSpec {
        name: "schema",
        summary: "Describe every nguo operation: names, argument JSON Schemas and exit codes.",
        description: "Machine-readable discovery of this command surface. Returns the same \
                      operation list the CLI and the MCP server are both built from, plus the \
                      process contract (exit codes, stdout/stderr split). Takes no arguments.",
        read_only: true,
        idempotent: true,
        schema: || json!({"type": "object", "properties": {}, "additionalProperties": false}),
    },
    OpSpec {
        name: "gpu_info",
        summary: "Report the GPU backend and adapter the solver will use, and probe it.",
        description: "Runs `mguso gpu-info`: selects an adapter, runs a no-op SAXPY compute \
                      probe and returns the engine's own report (backend, adapter, device_type, \
                      noop_saxpy_ok). Fails if no usable adapter exists.",
        read_only: true,
        idempotent: true,
        schema: || json!({"type": "object", "properties": {}, "additionalProperties": false}),
    },
    OpSpec {
        name: "scene_info",
        summary: "Summarise an engine scene JSON file: solver config, objects, vertex/triangle \
                  counts, pins, colliders.",
        description: "Parses an `engine-api` Scene JSON document locally (no solver runs) and \
                      reports its structure. This is the query surface that exists today; it \
                      reads engine scenes, NOT nguo project documents (pattern pieces and seams \
                      are not implemented yet).",
        read_only: true,
        idempotent: true,
        schema: || {
            json!({
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "description": "Path to an engine Scene .json file."}
                },
                "required": ["scene"],
                "additionalProperties": false
            })
        },
    },
    OpSpec {
        name: "scene_validate",
        summary: "Check a scene for structural faults (non-finite positions, out-of-range \
                  triangle/pin indices, bad timestep) before spending time on a drape.",
        description: "Structural validation, run locally and always: objects present, all \
                      positions finite, every triangle and pin index within range, dt/fps/frames \
                      sane. With `solver_invariants: true` it additionally runs `mguso verify`, \
                      which simulates the scene with the implicit solver and checks the solver's \
                      own invariants (penetration-free, strain limit, CG converged, all finite) \
                      — that is slow, seconds to minutes. Returns {ok, checks:[...]}; a failing \
                      scene is reported as a failure (CLI exit 2, MCP tool error), never as a \
                      pass.",
        read_only: true,
        idempotent: true,
        schema: || {
            json!({
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "description": "Path to an engine Scene .json file."},
                    "solver_invariants": {
                        "type": "boolean",
                        "default": false,
                        "description": "Also run `mguso verify` (slow: actually simulates)."
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Solver steps for `mguso verify`. Default: derived from the scene's fps/dt/frames."
                    }
                },
                "required": ["scene"],
                "additionalProperties": false
            })
        },
    },
    OpSpec {
        name: "drape",
        summary: "Simulate a scene and return the engine's drape report; optionally write the \
                  per-frame OBJ sequence to a directory.",
        description: "Runs the solver over the engine bridge. `tier: \"preview\"` uses the fast \
                      XPBD solver, `tier: \"bake\"` uses the high-fidelity implicit backward-Euler \
                      solver (much slower). Frames are streamed off-thread while the solve runs; \
                      the returned report is the engine's own summary verbatim (frames, verts, \
                      final_time, final_kinetic_energy, bbox_min/max). With `out_dir` the OBJ \
                      sequence is kept there as frame_0000.obj... — that is also how you export. \
                      Deterministic: the same scene and tier give the same drape, so calling it \
                      twice is safe; the only side effect is overwriting frame files in `out_dir`. \
                      If the solver diverges or the engine fails, this reports the engine's real \
                      error — it never returns a fabricated success.",
        read_only: false,
        idempotent: true,
        schema: || {
            json!({
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "description": "Path to an engine Scene .json file."},
                    "tier": {
                        "type": "string",
                        "enum": ["preview", "bake"],
                        "default": "preview",
                        "description": "preview = XPBD (fast, interactive), bake = implicit (high fidelity, slow)."
                    },
                    "frames": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Frames to simulate. Default: the scene's config.frames."
                    },
                    "out_dir": {
                        "type": "string",
                        "description": "Directory for the per-frame OBJ sequence. Omit to discard the frames and keep only the summary."
                    }
                },
                "required": ["scene"],
                "additionalProperties": false
            })
        },
    },
];

pub fn spec(name: &str) -> Option<&'static OpSpec> {
    SPECS.iter().find(|s| s.name == name)
}

/// Failure modes, mapped onto the `mguso` process contract.
#[derive(Debug, thiserror::Error)]
pub enum OpError {
    #[error("unknown operation `{0}` (try `nguo schema`)")]
    Unknown(String),
    #[error("invalid arguments for `{op}`: {detail}")]
    Args { op: String, detail: String },
    #[error("{0}")]
    Failed(String),
    /// The input was understood and is bad. Exit code 2, same as
    /// `mguso verify`'s "invariants failed".
    #[error("validation failed")]
    Invalid(Value),
}

impl OpError {
    /// Process exit code (`core-engine/docs/PROCESS_CONTRACT.md`).
    pub fn exit_code(&self) -> u8 {
        match self {
            OpError::Invalid(_) => 2,
            _ => 1,
        }
    }

    fn args(op: &str, detail: impl Into<String>) -> Self {
        OpError::Args {
            op: op.to_string(),
            detail: detail.into(),
        }
    }
}

/// Where the operations find the engine.
pub struct Env {
    pub engine: PathBuf,
}

impl Default for Env {
    fn default() -> Self {
        Self {
            engine: resolve_engine_binary(),
        }
    }
}

impl Env {
    fn transport(&self) -> ProcessTransport {
        ProcessTransport::new(self.engine.clone())
    }
}

/// Validate an operation's arguments without running it. Front ends that defer
/// the work (the MCP server starts `drape` as a background job) call this
/// first, so a malformed request fails immediately instead of inside a job.
pub fn check_args(name: &str, args: &Value) -> Result<(), OpError> {
    let Some(spec) = spec(name) else {
        return Err(OpError::Unknown(name.to_string()));
    };
    reject_unknown_keys(spec, args)?;
    match name {
        "scene_info" => {
            arg_path(name, args, "scene")?;
        }
        "scene_validate" => {
            arg_path(name, args, "scene")?;
            arg_bool(name, args, "solver_invariants")?;
            arg_u32(name, args, "steps")?;
        }
        "drape" => {
            arg_path(name, args, "scene")?;
            arg_tier(name, args)?;
            arg_u32(name, args, "frames")?;
            arg_str(name, args, "out_dir")?;
        }
        _ => {}
    }
    Ok(())
}

/// Run one operation. The single entry point both front ends go through.
pub async fn dispatch(name: &str, args: &Value, env: &Env) -> Result<Value, OpError> {
    check_args(name, args)?;
    match name {
        "schema" => Ok(surface()),
        "gpu_info" => gpu_info(env),
        "scene_info" => scene_info(&arg_path(name, args, "scene")?),
        "scene_validate" => scene_validate(
            &arg_path(name, args, "scene")?,
            arg_bool(name, args, "solver_invariants")?.unwrap_or(false),
            arg_u32(name, args, "steps")?,
            env,
        ),
        "drape" => {
            drape(
                &arg_path(name, args, "scene")?,
                arg_tier(name, args)?,
                arg_u32(name, args, "frames")?,
                arg_str(name, args, "out_dir")?.map(PathBuf::from),
                env,
            )
            .await
        }
        _ => Err(OpError::Unknown(name.to_string())),
    }
}

/// The self-description an agent reads to learn this surface.
pub fn surface() -> Value {
    json!({
        "tool": "nguo",
        "version": env!("CARGO_PKG_VERSION"),
        "process_contract": {
            "stdout": "exactly one JSON value per invocation; nothing else",
            "stderr": "human or JSON logs (tracing); never mixed into stdout",
            "exit_codes": {
                "0": "success",
                "1": "failure: bad arguments, I/O error, engine failure, solver diverged",
                "2": "the input was understood and is invalid (validation or solver invariants failed)"
            }
        },
        "operations": SPECS.iter().map(|s| json!({
            "name": s.name,
            "cli": s.cli_name(),
            "summary": s.summary,
            "description": s.description,
            "read_only": s.read_only,
            "idempotent": s.idempotent,
            "input_schema": (s.schema)(),
        })).collect::<Vec<_>>(),
    })
}

fn gpu_info(env: &Env) -> Result<Value, OpError> {
    env.transport()
        .run_json(&["gpu-info"], &[0])
        .map_err(|e| OpError::Failed(e.to_string()))
}

fn load_scene(path: &Path) -> Result<Value, OpError> {
    if path.extension().and_then(|s| s.to_str()) != Some("json") {
        return Err(OpError::Failed(format!(
            "{} is not a .json scene; only engine-api Scene JSON is supported here",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| OpError::Failed(format!("reading {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| OpError::Failed(format!("parsing {}: {e}", path.display())))
}

fn scene_info(path: &Path) -> Result<Value, OpError> {
    let scene = load_scene(path)?;
    let objects = scene.get("objects").and_then(Value::as_array);
    let summarised: Vec<Value> = objects
        .map(|objs| {
            objs.iter()
                .map(|o| {
                    json!({
                        "name": o.get("name").and_then(Value::as_str).unwrap_or(""),
                        "verts": count(o, "positions"),
                        "tris": count(o, "tris"),
                        "pins": count(o, "pins"),
                        "is_static": o.get("is_static").and_then(Value::as_bool).unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let colliders: Vec<Value> = scene
        .get("colliders")
        .and_then(Value::as_array)
        .map(|cs| {
            cs.iter()
                .map(|c| json!({"type": c.get("type").and_then(Value::as_str).unwrap_or("?")}))
                .collect()
        })
        .unwrap_or_default();
    let totals = json!({
        "objects": summarised.len(),
        "verts": summarised.iter().map(|o| o["verts"].as_u64().unwrap_or(0)).sum::<u64>(),
        "tris": summarised.iter().map(|o| o["tris"].as_u64().unwrap_or(0)).sum::<u64>(),
    });
    Ok(json!({
        "scene": path.display().to_string(),
        "config": scene.get("config").cloned().unwrap_or(Value::Null),
        "objects": summarised,
        "colliders": colliders,
        "totals": totals,
    }))
}

fn count(o: &Value, key: &str) -> u64 {
    o.get(key).and_then(Value::as_array).map_or(0, |a| a.len()) as u64
}

fn scene_validate(
    path: &Path,
    solver_invariants: bool,
    steps: Option<u32>,
    env: &Env,
) -> Result<Value, OpError> {
    let scene = load_scene(path)?;
    let mut checks: Vec<Value> = Vec::new();
    let mut check = |name: &str, ok: bool, detail: String| {
        checks.push(json!({"name": name, "ok": ok, "detail": detail}));
        ok
    };

    let objects = scene
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut ok = check(
        "has-objects",
        !objects.is_empty(),
        format!("{} object(s)", objects.len()),
    );

    let mut nonfinite = 0usize;
    let mut bad_tris = 0usize;
    let mut degenerate = 0usize;
    let mut bad_pins = 0usize;
    for obj in &objects {
        let verts = obj
            .get("positions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for p in &verts {
            let finite = p.as_array().is_some_and(|c| {
                c.len() == 3 && c.iter().all(|v| v.as_f64().is_some_and(f64::is_finite))
            });
            if !finite {
                nonfinite += 1;
            }
        }
        let n = verts.len() as u64;
        let empty: Vec<Value> = Vec::new();
        for t in obj.get("tris").and_then(Value::as_array).unwrap_or(&empty) {
            let idx: Vec<u64> = t
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            if idx.len() != 3 || idx.iter().any(|&i| i >= n) {
                bad_tris += 1;
            } else if idx[0] == idx[1] || idx[1] == idx[2] || idx[0] == idx[2] {
                degenerate += 1;
            }
        }
        for p in obj.get("pins").and_then(Value::as_array).unwrap_or(&empty) {
            if p.as_u64().is_none_or(|i| i >= n) {
                bad_pins += 1;
            }
        }
    }
    ok &= check(
        "positions-finite",
        nonfinite == 0,
        format!("{nonfinite} non-finite vertex position(s)"),
    );
    ok &= check(
        "triangle-indices-in-range",
        bad_tris == 0,
        format!("{bad_tris} triangle(s) index a vertex that does not exist"),
    );
    ok &= check(
        "triangles-non-degenerate",
        degenerate == 0,
        format!("{degenerate} triangle(s) repeat a vertex"),
    );
    ok &= check(
        "pin-indices-in-range",
        bad_pins == 0,
        format!("{bad_pins} pin(s) index a vertex that does not exist"),
    );

    let config = scene.get("config").cloned().unwrap_or(json!({}));
    let dt = config.get("dt").and_then(Value::as_f64).unwrap_or(0.001);
    let fps = config.get("fps").and_then(Value::as_f64).unwrap_or(60.0);
    ok &= check(
        "timestep-positive",
        dt > 0.0 && dt.is_finite(),
        format!("dt = {dt}"),
    );
    ok &= check(
        "fps-positive",
        fps > 0.0 && fps.is_finite(),
        format!("fps = {fps}"),
    );

    let mut report = json!({
        "scene": path.display().to_string(),
        "ok": ok,
        "checks": checks,
    });

    if solver_invariants {
        let mut args: Vec<String> = vec!["verify".into(), path.display().to_string()];
        if let Some(steps) = steps {
            args.push("--steps".into());
            args.push(steps.to_string());
        }
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        // exit 2 from `mguso verify` means "invariants failed" — a result, not
        // an engine error, so accept it and read the report.
        let verify = env
            .transport()
            .run_json(&borrowed, &[0, 2])
            .map_err(|e| OpError::Failed(e.to_string()))?;
        ok &= verify
            .get("all_ok")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        report["solver_invariants"] = verify;
        report["ok"] = json!(ok);
    }

    if ok {
        Ok(report)
    } else {
        Err(OpError::Invalid(report))
    }
}

async fn drape(
    path: &Path,
    tier: Tier,
    frames: Option<u32>,
    out_dir: Option<PathBuf>,
    env: &Env,
) -> Result<Value, OpError> {
    let scene = load_scene(path)?;
    let (bridge, mut updates) = EngineBridge::new(std::sync::Arc::new(env.transport()));
    let epoch = bridge.submit(Submission {
        scene,
        // A scene file cannot carry stitches — `engine_api::Scene` has no field
        // for them — so a CLI drape of one is unsewn by construction. Sewing goes
        // through `nguo_sewing::build`, which needs a *document*, not a scene.
        stitches: Vec::new(),
        tier,
        frames,
        out_dir: out_dir.clone(),
    });

    let mut streamed = 0u32;
    while let Some(update) = updates.recv().await {
        debug_assert_eq!(update.epoch(), epoch);
        match update {
            Update::Frame { .. } => streamed += 1,
            Update::Done { summary, .. } => {
                return Ok(json!({
                    "scene": path.display().to_string(),
                    "tier": summary.tier,
                    "frames_streamed": streamed,
                    "out_dir": out_dir.map(|d| d.display().to_string()),
                    "engine": summary.report,
                }));
            }
            // The engine's real error, verbatim. Never dressed up as success.
            Update::Failed { error, .. } => return Err(OpError::Failed(error)),
        }
    }
    Err(OpError::Failed(
        "the engine bridge closed before the drape finished".into(),
    ))
}

// --- argument plumbing ----------------------------------------------------

fn reject_unknown_keys(spec: &OpSpec, args: &Value) -> Result<(), OpError> {
    let Some(obj) = args.as_object() else {
        if args.is_null() {
            return Ok(());
        }
        return Err(OpError::args(spec.name, "arguments must be a JSON object"));
    };
    let schema = (spec.schema)();
    let known = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for key in obj.keys() {
        if !known.contains_key(key) {
            return Err(OpError::args(
                spec.name,
                format!(
                    "unknown argument `{key}` (accepted: {})",
                    known.keys().cloned().collect::<Vec<_>>().join(", ")
                ),
            ));
        }
    }
    Ok(())
}

fn arg_str(op: &str, args: &Value, key: &str) -> Result<Option<String>, OpError> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(OpError::args(
            op,
            format!("`{key}` must be a string, got {other}"),
        )),
    }
}

fn arg_path(op: &str, args: &Value, key: &str) -> Result<PathBuf, OpError> {
    arg_str(op, args, key)?
        .map(PathBuf::from)
        .ok_or_else(|| OpError::args(op, format!("`{key}` is required")))
}

fn arg_bool(op: &str, args: &Value, key: &str) -> Result<Option<bool>, OpError> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(other) => Err(OpError::args(
            op,
            format!("`{key}` must be a boolean, got {other}"),
        )),
    }
}

fn arg_u32(op: &str, args: &Value, key: &str) -> Result<Option<u32>, OpError> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| {
                OpError::args(
                    op,
                    format!("`{key}` must be a non-negative integer, got {v}"),
                )
            }),
    }
}

fn arg_tier(op: &str, args: &Value) -> Result<Tier, OpError> {
    match arg_str(op, args, "tier")?.as_deref() {
        None | Some("preview") => Ok(Tier::Preview),
        Some("bake") => Ok(Tier::Bake),
        Some(other) => Err(OpError::args(
            op,
            format!("`tier` must be \"preview\" or \"bake\", got {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .join("core-engine/examples")
            .join(name)
    }

    #[test]
    fn op_names_are_unique_and_snake_case() {
        let mut seen = std::collections::HashSet::new();
        for s in SPECS {
            assert!(seen.insert(s.name), "duplicate op `{}`", s.name);
            assert!(
                s.name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "`{}` is not snake_case",
                s.name
            );
            assert!(!s.summary.is_empty() && !s.description.is_empty());
            let schema = (s.schema)();
            assert_eq!(
                schema["type"], "object",
                "`{}` schema is not an object",
                s.name
            );
        }
    }

    #[tokio::test]
    async fn scene_info_reads_a_real_example_scene() {
        let out = dispatch(
            "scene_info",
            &json!({"scene": example("sphere_drape.json").display().to_string()}),
            &Env::default(),
        )
        .await
        .expect("sphere_drape.json parses");
        assert_eq!(out["totals"]["objects"], 1);
        assert_eq!(out["totals"]["verts"], 144);
        assert_eq!(out["totals"]["tris"], 242);
        assert_eq!(out["colliders"][0]["type"], "sphere");
    }

    #[tokio::test]
    async fn scene_validate_passes_a_real_example_scene() {
        let out = dispatch(
            "scene_validate",
            &json!({"scene": example("fabric_drape.json").display().to_string()}),
            &Env::default(),
        )
        .await
        .expect("fabric_drape.json is structurally valid");
        assert_eq!(out["ok"], true);
    }

    #[tokio::test]
    async fn scene_validate_rejects_an_out_of_range_triangle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.json");
        std::fs::write(
            &path,
            json!({
                "config": {"dt": 0.001, "fps": 60.0, "frames": 1},
                "objects": [{
                    "name": "bad",
                    "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "tris": [[0, 1, 7]],
                    "pins": [9]
                }]
            })
            .to_string(),
        )
        .unwrap();
        let err = dispatch(
            "scene_validate",
            &json!({"scene": path.display().to_string()}),
            &Env::default(),
        )
        .await
        .expect_err("must not pass");
        assert_eq!(err.exit_code(), 2);
        let OpError::Invalid(report) = err else {
            panic!("expected Invalid")
        };
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
            vec!["triangle-indices-in-range", "pin-indices-in-range"]
        );
    }

    #[tokio::test]
    async fn unknown_op_and_unknown_argument_are_rejected() {
        let env = Env::default();
        assert!(matches!(
            dispatch("teleport", &json!({}), &env).await,
            Err(OpError::Unknown(_))
        ));
        assert!(matches!(
            dispatch(
                "scene_info",
                &json!({"scene": "x.json", "colour": "red"}),
                &env
            )
            .await,
            Err(OpError::Args { .. })
        ));
        assert!(matches!(
            dispatch("drape", &json!({"scene": "x.json", "tier": "fast"}), &env).await,
            Err(OpError::Args { .. })
        ));
    }
}
