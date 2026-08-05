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

use engine_api::resize::BodyGirth;
use engine_io::MeshUnits;
use nguo_engine_bridge::{
    resolve_engine_binary, EngineBridge, Flow, FrameSink, InProcessTransport, ProcessTransport,
    Stitch, Submission, Tier, Transport, Update,
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
    OpSpec {
        name: "garment_fit",
        summary: "Scale an imported garment mesh onto a body and push out the residual overlap — \
                  the drop-and-fit step that has to happen before a drape.",
        description: "Loads a garment mesh in the unit you declare, proposes a uniform resize \
                      from the body's measured girths (`engine_api::resize`), applies it, then \
                      resolves the vertices that still start inside the body \
                      (`engine_api::avatar::resolve_penetration`). Returns the whole proposal, \
                      not just the answer: the factor, the landmark that set it, the length it \
                      costs, and every landmark's girth residual — so you can see WHY and \
                      re-run with `factor` to override it. \
                      `units` is REQUIRED and has no default and no auto-detect: no mesh format \
                      records its unit, and the sample corpus contains a 297-unit bag that is a \
                      believable 0.297 m in millimetres and an absurd 2.97 m in centimetres. \
                      Guessing would silently produce a 100x garment that still \"works\" \
                      downstream. Body: either `avatar` (a bundle key, girths from its manifest) \
                      or `body` (a mesh path in metres) plus explicit `girths`. \
                      With `out` the fitted, ready-to-drape scene is written there for `drape`. \
                      Reports honestly: a garment that is simply the wrong size for the body \
                      comes back with `inside_after > 0`, never a fabricated fit.",
        read_only: false,
        idempotent: true,
        schema: || {
            json!({
                "type": "object",
                "properties": {
                    "garment": {"type": "string", "description": "Path to the garment mesh (.obj, .stl, .gltf, .glb, .usda)."},
                    "units": {
                        "type": "string",
                        "enum": ["m", "cm", "mm", "in"],
                        "description": "Unit the garment file is authored in. Required — there is deliberately no default and no auto-detect."
                    },
                    "avatar": {"type": "string", "description": "Avatar bundle key (\"male\", \"female\"). Its COL proxy is the body and its manifest supplies the girths."},
                    "body": {"type": "string", "description": "Path to a body mesh in metres, instead of `avatar`. Requires `girths`."},
                    "girths": {
                        "type": "array",
                        "description": "Body girths, overriding the manifest (required with `body`).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "landmark": {"type": "string"},
                                "circumference_m": {"type": "number", "exclusiveMinimum": 0},
                                "height_m": {"type": "number"}
                            },
                            "required": ["landmark", "circumference_m", "height_m"],
                            "additionalProperties": false
                        }
                    },
                    "ease": {"type": "number", "minimum": 0, "default": 0.03, "description": "Girth ease the factor targets. 0.03 = 3 % looser than the body."},
                    "factor": {"type": "number", "exclusiveMinimum": 0, "description": "Override the proposed factor. The proposal is still reported as `resize.proposed`."},
                    "fabric": {
                        "type": "string",
                        "enum": ["cotton", "denim", "silk", "jersey", "wool", "leather"],
                        "default": "cotton",
                        "description": "Fabric preset written into the scene."
                    },
                    "out": {"type": "string", "description": "Write the fitted scene JSON here, ready for `drape`. Omit to measure only."}
                },
                "required": ["garment", "units"],
                "additionalProperties": false
            })
        },
    },
    OpSpec {
        name: "dress_example",
        summary: "Run the whole 2D-to-worn chain on the shipped sheath-dress pattern: draft, \
                  triangulate, sew, hang on the avatar, resolve overlap, drape, save a `.nguo`.",
        description: "The end-to-end industrial example, in one process, reporting every stage. \
                      The pattern is `nguo_cli::dress::dress()` — a sleeveless sheath drafted in \
                      MILLIMETRES to the `female` avatar's own manifest girths, two panels, two \
                      shaped side seams. Stages: validate -> triangulate (`edge_mm`) -> arrange on \
                      a cylinder sized from the manifest girth at the support landmark -> sew -> \
                      `engine_api::plan_resize` (REPORTED, and applied only with `resize: true`: a \
                      drafted pattern is graded, not scaled — `nguo_doc::Feature::Support`) -> \
                      `engine_api::avatar::resolve_penetration` -> solve. \
                      Without `avatar` it stops after sewing and saving, which is the only form \
                      that runs in CI: the avatar bundle is 227 MB of Git-LFS. \
                      With `out` the project is written as a real `.nguo` (D-1: a zip of \
                      manifest.json + document.json + preview.svg), loaded straight back, and the \
                      round trip is reported as `save.round_trip_identical` — including the \
                      manifest's declared units, coordinate convention and the avatar REFERENCE \
                      (key + tier + bundle version; the 227 MB CC BY 4.0 bundle is never \
                      embedded). Reports honestly: seam gap in mm before and after, vertices still \
                      inside the body after the solve, s/step, and every count. A dress that does \
                      not close comes back with the gap it stalled at.",
        read_only: false,
        idempotent: true,
        schema: || {
            json!({
                "type": "object",
                "properties": {
                    "avatar": {"type": "string", "description": "Avatar bundle key (\"female\" is what the pattern is drafted for). Omit to run the pattern/sew/save stages only, with no body and no drape."},
                    "edge_mm": {"type": "number", "exclusiveMinimum": 0, "description": "Target triangle edge, mm. Default 60 (preview) / 30 (bake) — the single biggest input to solve cost and peak memory."},
                    "tier": {"type": "string", "enum": ["preview", "bake"], "default": "preview", "description": "preview = XPBD, bake = implicit backward Euler (far more memory)."},
                    "steps": {"type": "integer", "minimum": 1, "description": "Solver steps. Default 36 (preview) / 144 (bake)."},
                    "resize": {"type": "boolean", "default": false, "description": "Apply the proposed uniform resize. Off by default: the pattern is drafted to this body, so scaling it would be grading done wrong. The proposal is reported either way."},
                    "out": {"type": "string", "description": "Write the project here as a `.nguo` and prove the round trip. Omit to measure only."}
                },
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
        "garment_fit" => {
            arg_path(name, args, "garment")?;
            arg_units(name, args)?;
            arg_positive(name, args, "factor")?;
            arg_f32(name, args, "ease")?;
            arg_fabric(name, args)?;
            arg_str(name, args, "out")?;
            // The body pair is validated here and not only in the operation, so an
            // MCP client that defers the work still fails on the request.
            body_choice(name, args)?;
        }
        "dress_example" => {
            arg_str(name, args, "avatar")?;
            arg_positive(name, args, "edge_mm")?;
            arg_tier(name, args)?;
            arg_u32(name, args, "steps")?;
            arg_bool(name, args, "resize")?;
            arg_str(name, args, "out")?;
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
        "garment_fit" => garment_fit(args),
        "dress_example" => dress_example(args),
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

/// Drop-and-fit: import in a declared unit, resize onto the body, push out the rest.
///
/// The whole point of this operation is that **nothing is silent**. `engine_api::resize`
/// picks the supporting landmark by geometry, and its own docs say that can be wrong in a
/// way no code fixes (`men-shirt` on the female body has only its hem to go on). So the
/// return carries the proposal, every landmark that was considered, the girth residual each
/// one is left with, and the length the factor costs — and `factor` overrides it.
fn garment_fit(args: &Value) -> Result<Value, OpError> {
    const OP: &str = "garment_fit";
    let path = arg_path(OP, args, "garment")?;
    let units = arg_units(OP, args)?;
    let mut garment = engine_io::load_mesh_in(&path, units)
        .map_err(|e| OpError::Failed(format!("could not load `{}`: {e:#}", path.display())))?;
    let (body, girths, body_label) = load_body(OP, args)?;
    let ease = arg_f32(OP, args, "ease")?.unwrap_or(engine_api::resize::FIT_EASE);
    let fabric = arg_fabric(OP, args)?;

    let mut plan = engine_api::plan_resize(&body, &garment, &girths, ease)
        .map_err(|e| OpError::Failed(e.to_string()))?;
    let proposed = plan.factor;
    if let Some(f) = arg_positive(OP, args, "factor")? {
        // The anchor is independent of the factor by construction (`ResizePlan`), so an
        // override stays on the same landmark — except that an identity proposal has no
        // anchor to move to, and inventing one here would be the translation `resize`
        // deliberately does not own (D-18/D-20).
        plan.factor = f;
    }
    let overridden = plan.factor != proposed;
    let check = engine_api::apply_resize(&mut garment, &plan);
    let fit =
        engine_api::avatar::resolve_penetration(&body, &mut garment, engine_api::avatar::FIT_GAP_M)
            .map_err(|e| OpError::Failed(e.to_string()))?;

    let out = match arg_str(OP, args, "out")? {
        None => None,
        Some(dest) => {
            // `AsAuthored`: the overlap was just resolved above, and asking
            // `drape_on_avatar` for `FittedToBody` would run the same pass twice.
            let scene = engine_api::drape_on_avatar(
                &body,
                &garment,
                fabric,
                engine_api::GarmentPlacement::AsAuthored,
            )
            .map_err(|e| OpError::Failed(e.to_string()))?;
            let text = serde_json::to_string_pretty(&scene)
                .map_err(|e| OpError::Failed(format!("could not serialise the scene: {e}")))?;
            std::fs::write(&dest, text)
                .map_err(|e| OpError::Failed(format!("could not write `{dest}`: {e}")))?;
            Some(dest)
        }
    };

    Ok(json!({
        "garment": {
            "path": path.display().to_string(),
            "units": args["units"],
            "verts": garment.positions.len(),
            "tris": garment.tris.len(),
            // `load_mesh_in` scales positions but NOT the skin's `inverse_bind`
            // translations, and says so through `tracing::warn!` — which this front end
            // has no guaranteed subscriber for. The whole point of this operation is
            // that nothing is silent, so it goes in the report too.
            "skin_broken_by_unit_scale":
                garment.skin.is_some() && units != engine_io::MeshUnits::Metres,
        },
        "body": body_label,
        "resize": {
            "factor": plan.factor,
            "proposed": proposed,
            "overridden": overridden,
            "ease": plan.ease,
            "supporting": plan.supporting,
            "length_delta_m": plan.length_delta_m(),
            "anchor_from": plan.anchor_from,
            "anchor_to": plan.anchor_to,
            "edge_ratio": [check.edge_ratio_min_all, check.edge_ratio_max_all],
            "landmarks": plan.fits.iter().map(|f| json!({
                "landmark": f.landmark,
                "body_girth_m": f.body_girth_m,
                "height_m": f.height_m,
                "garment_girth_m": f.garment.map(|r| r.girth_m),
                "garment_ring_verts": f.garment.map(|r| r.verts),
                "garment_ring_aspect": f.garment.map(|r| r.aspect),
                // `None` = this landmark could not set the factor at all: no ring, a ring
                // too slanted or too sparse to trust, or an excluded landmark.
                "demands_factor": f.factor,
                // What the applied factor actually leaves at this landmark. This is the
                // residual a single uniform scale cannot remove, i.e. the case for grading.
                "slack": f.slack_at(plan.factor),
            })).collect::<Vec<_>>(),
        },
        "penetration": {
            "gap_m": engine_api::avatar::FIT_GAP_M,
            "inside_before": fit.per_pass.first(),
            "inside_after": fit.inside_after,
            "deepest_inside_m": fit.deepest_inside_m,
            "pushed": fit.pushed,
            "per_pass": fit.per_pass,
            "converged": fit.converged(),
            "edge_ratio": [fit.check.edge_ratio_min_all, fit.check.edge_ratio_max_all],
        },
        "scene": out,
    }))
}

// --- dress_example --------------------------------------------------------

/// Solver `dt`. Backward Euler is unconditionally stable, so this is chosen for
/// how much *simulated time* a step buys, not for stability: at the engine's
/// default `1e-3` a 36-step preview would advance 0.03 s and nothing would fall.
/// The same value `nguo-app` uses, so a CLI drape and the shell's drape are the
/// same solve rather than two similar ones.
const DRESS_DT: f32 = 1.0 / 120.0;
/// Clearance round the garment's own box when cropping the body proxy. It has to
/// exceed how far the garment can travel during the solve; the dress is held up
/// by its own waist and measures ~13 mm of slide (see `dress::WAIST_HALF_MM`), so
/// 150 mm is an order of magnitude of headroom. It also drops the T-pose arms,
/// which are 0.7 m out in x and can never touch a 0.2 m tube.
const DRESS_CROP_M: f32 = 0.15;

/// Total distance the seams are still open, metres, for one frame of positions.
/// The same measure as `nguo_sewing::Garment::seam_gap_m`, against solved
/// positions instead of arranged ones.
fn seam_gap_m(stitches: &[nguo_sewing::Stitch6], positions: &[[f32; 3]]) -> f64 {
    let point = |index: &[u32], weight: &[f32]| -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (&i, &w) in index.iter().zip(weight) {
            if let Some(p) = positions.get(i as usize) {
                for k in 0..3 {
                    out[k] += w * p[k];
                }
            }
        }
        out
    };
    stitches
        .iter()
        .map(|s| {
            let p = point(&s.index[0..3], &s.weight[0..3]);
            let q = point(&s.index[3..6], &s.weight[3..6]);
            ((0..3)
                .map(|k| (p[k] - q[k]) as f64)
                .map(|d| d * d)
                .sum::<f64>())
            .sqrt()
        })
        .sum()
}

/// Keeps the first and last frames and the seam-gap curve, and nothing else.
///
/// Deliberately not every frame: the closure curve is what says whether the
/// seams are still moving, and it is one `f64` per step instead of a mesh.
struct DressFrames<'a> {
    stitches: &'a [nguo_sewing::Stitch6],
    /// The garment is object 0, so its vertices are the leading prefix of every
    /// frame. A frame shorter than this is the wrong scene and is not sliced.
    cloth: usize,
    first: Option<Vec<[f32; 3]>>,
    last: Option<Vec<[f32; 3]>>,
    gaps: Vec<f64>,
}

impl FrameSink for DressFrames<'_> {
    fn push(&mut self, _index: u32, positions: Vec<[f32; 3]>) -> Flow {
        if positions.len() < self.cloth {
            tracing::error!(
                got = positions.len(),
                want = self.cloth,
                "a frame is too short for the garment; dropping it"
            );
            return Flow::Continue;
        }
        let cloth = &positions[..self.cloth];
        self.gaps.push(seam_gap_m(self.stitches, cloth));
        if self.first.is_none() {
            self.first = Some(cloth.to_vec());
        }
        self.last = Some(cloth.to_vec());
        Flow::Continue
    }

    fn is_cancelled(&self) -> bool {
        false
    }
}

/// A cloth `Object` from a garment's geometry. UVs are not optional: without
/// them `Scene::lower` throws them all away and the anisotropic warp/weft
/// membrane goes inert.
fn cloth_object(
    positions: Vec<[f32; 3]>,
    tris: &[[u32; 3]],
    uvs: &[[f32; 2]],
) -> engine_api::Object {
    engine_api::Object {
        name: "dress".into(),
        positions,
        tris: tris.to_vec(),
        material: engine_api::Material::default(),
        preset: Some(engine_api::FabricPreset::Cotton),
        pins: Vec::new(),
        is_static: false,
        normals: Vec::new(),
        uvs: uvs.to_vec(),
        skin: None,
        animations: Vec::new(),
    }
}

/// The whole chain, in one process, reporting every stage. See the [`OpSpec`].
fn dress_example(args: &Value) -> Result<Value, OpError> {
    const OP: &str = "dress_example";
    let tier = arg_tier(OP, args)?;
    let edge_mm = arg_positive(OP, args, "edge_mm")?
        .map(f64::from)
        .unwrap_or(match tier {
            Tier::Preview => 60.0,
            Tier::Bake => 30.0,
        });
    let steps = arg_u32(OP, args, "steps")?.unwrap_or(match tier {
        Tier::Preview => 36,
        Tier::Bake => 144,
    });

    // --- 1. the pattern, in millimetres -----------------------------------
    let mut doc = crate::dress::dress();
    let diagnostics = nguo_sewing::validate(&doc);
    if !nguo_sewing::is_sewable(&diagnostics) {
        return Err(OpError::Invalid(json!({
            "stage": "validate",
            "ok": false,
            "diagnostics": diagnostics.iter().map(|d| d.message()).collect::<Vec<_>>(),
        })));
    }
    let hang = nguo_sewing::hang(&doc);
    let pattern = json!({
        "name": doc.meta().name,
        "pieces": doc.pieces().len(),
        "seams": doc.seams().len(),
        "vertices": doc.pieces().iter().map(|(_, p)| p.network.vertices.len()).sum::<usize>(),
        "edges": doc.pieces().iter().map(|(_, p)| p.network.edges.len()).sum::<usize>(),
        "units": "millimetre",
        "support_landmark": hang.landmark,
        "drop_mm": hang.drop_mm,
        "support_arc_mm": doc.pieces().iter()
            .map(|(_, p)| nguo_sewing::support_arc_mm(p).unwrap_or(0.0)).sum::<f64>(),
        "seam_allowance_mm": crate::dress::SEAM_ALLOWANCE_MM,
        "draft_mm": {
            "bust_half": crate::dress::BUST_HALF_MM,
            "waist_half": crate::dress::WAIST_HALF_MM,
            "hip_half": crate::dress::HIP_HALF_MM,
            "waist_drop": crate::dress::WAIST_DROP_MM,
            "hip_drop": crate::dress::HIP_DROP_MM,
            "length": crate::dress::LENGTH_MM,
        },
        "diagnostics": diagnostics.iter().map(|d| d.message()).collect::<Vec<_>>(),
    });

    // --- 2. the body, from the manifest -----------------------------------
    let key = arg_str(OP, args, "avatar")?;
    let bundle = match &key {
        None => None,
        Some(_) => Some(
            nguo_avatar::Bundle::find()
                .map_err(|e| OpError::Failed(format!("avatar bundle: {e}")))?,
        ),
    };
    let mut arrange = nguo_sewing::ArrangeOptions::default();
    let mut body = None;
    let mut body_report = Value::Null;
    if let (Some(bundle), Some(key)) = (&bundle, &key) {
        let avatar = bundle
            .avatar(key)
            .map_err(|e| OpError::args(OP, e.to_string()))?;
        let proxy = bundle
            .load_mesh(avatar, nguo_avatar::Asset::GlbCollision)
            .map_err(|e| OpError::Failed(format!("avatar collision proxy: {e}")))?;
        let measurements = bundle.measurements(avatar);
        let landmark = hang
            .landmark
            .as_deref()
            .and_then(nguo_avatar::Landmark::from_key)
            .ok_or_else(|| {
                OpError::Failed(format!(
                    "the pattern hangs from {:?}, which is not an avatar landmark",
                    hang.landmark,
                ))
            })?;
        let at = measurements.girth(landmark).ok_or_else(|| {
            OpError::Failed(format!(
                "{key} has no {} girth in its manifest",
                landmark.key()
            ))
        })?;
        let top_y_m = (at.height.mm() / 1000.0) as f32;
        let low_y_m = top_y_m - (hang.drop_mm / 1000.0) as f32;

        // The start cylinder's radius comes from the MANIFEST girths inside the
        // band, never from a slab measured off the proxy: a horizontal slab
        // through a T-posed body catches both arms and reads the chest ~10 %
        // high. The measured maximum is reported beside it precisely so the
        // difference is visible — and where it exceeds the manifest radius, the
        // cloth genuinely does start inside the body, which is what
        // `resolve_penetration` below then measures rather than assumes.
        let in_band = |h: f32| h <= top_y_m + 1e-4 && h >= low_y_m - 1e-4;
        let manifest_r = measurements
            .girths()
            .filter(|(_, g)| in_band((g.height.mm() / 1000.0) as f32))
            .map(|(_, g)| (g.circumference.mm() / 1000.0) as f32 / std::f32::consts::TAU)
            .fold(0.0f32, f32::max);
        let mut measured = (0.0f32, 0.0f32);
        for p in proxy.positions.iter().filter(|p| in_band(p[1])) {
            let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
            if r > measured.0 {
                measured = (r, p[1]);
            }
        }

        body_report = json!({
            "avatar": key,
            "name": avatar.name,
            "tier": "COL",
            "bundle_version": bundle.manifest().version,
            "asset": avatar.declared(nguo_avatar::Asset::GlbCollision),
            "proxy_verts": proxy.positions.len(),
            "proxy_tris": proxy.tris.len(),
            "band_m": [low_y_m, top_y_m],
            "landmark": landmark.key(),
            "landmark_girth_m": at.circumference.mm() / 1000.0,
            "landmark_height_m": at.height.mm() / 1000.0,
            "manifest_radius_m": manifest_r,
            "measured_max_radius_m": measured.0,
            "measured_max_radius_at_y_m": measured.1,
            "girths_m": measurements.girths()
                .map(|(k, g)| json!({
                    "landmark": k,
                    "circumference_m": g.circumference.mm() / 1000.0,
                    "height_m": g.height.mm() / 1000.0,
                    "in_band": in_band((g.height.mm() / 1000.0) as f32),
                }))
                .collect::<Vec<_>>(),
        });
        arrange = nguo_sewing::ArrangeOptions {
            // A floor, so a proxy or a manifest that failed to describe a body
            // cannot collapse the start cylinder onto the axis.
            body_radius_m: manifest_r.max(arrange.body_radius_m),
            top_y_m,
            ..arrange
        };
        body = Some(proxy);

        // Record the body this run actually resolved, so the saved project
        // references a version and a tier rather than just a name.
        let mut avatar_ref = doc.avatar().cloned().unwrap_or_default();
        avatar_ref.tier = Some("COL".into());
        avatar_ref.bundle_version = Some(bundle.manifest().version.clone());
        avatar_ref.asset = Some(avatar.declared(nguo_avatar::Asset::GlbCollision).to_owned());
        doc.apply_edit(nguo_doc::Edit::SetAvatar {
            avatar: Some(Box::new(avatar_ref)),
        })
        .map_err(|e| OpError::Failed(format!("recording the avatar reference: {e}")))?;
    }

    // --- 3. triangulate + arrange + sew -----------------------------------
    let opts = nguo_sewing::SewOptions {
        triangulate: nguo_triangulate::TriangulateOptions::default().with_target_edge_mm(edge_mm),
        arrange,
    };
    let mut garment = nguo_sewing::build(&doc, &opts)
        .map_err(|e| OpError::Failed(format!("the dress does not sew: {e}")))?;
    let arranged_gap_m = garment.seam_gap_m();
    let sew = json!({
        "target_edge_mm": edge_mm,
        "verts": garment.vertex_count(),
        "tris": garment.triangle_count(),
        "stitches": garment.stitches.len(),
        "seams": garment.seams.iter().map(|s| json!({
            "stitches": s.stitches, "flipped": s.flipped,
        })).collect::<Vec<_>>(),
        "arrangement": {
            "radius_m": garment.arrangement.radius_m,
            "circumference_m": garment.arrangement.circumference_m,
            "slack_m": garment.arrangement.slack_m,
        },
        "bounds_m": garment.bounds().map(|b| json!({"min": b.min, "max": b.max})),
        "seam_gap_mm": arranged_gap_m * 1000.0,
        "timings_ms": {
            "validate": garment.timings.validate_ms,
            "triangulate": garment.timings.triangulate_ms,
            "arrange": garment.timings.arrange_ms,
            "sew": garment.timings.sew_ms,
            "total": garment.timings.total_ms(),
        },
    });

    // --- 4/5. resize proposal, then resolve the residual overlap ----------
    let mut resize = Value::Null;
    let mut penetration = Value::Null;
    let mut cropped = Value::Null;
    if let Some(proxy) = &body {
        let girths: Vec<BodyGirth> = body_report["girths_m"]
            .as_array()
            .map(|gs| {
                gs.iter()
                    .map(|g| BodyGirth {
                        landmark: g["landmark"].as_str().unwrap_or_default().to_owned(),
                        circumference_m: g["circumference_m"].as_f64().unwrap_or(0.0) as f32,
                        height_m: g["height_m"].as_f64().unwrap_or(0.0) as f32,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut object = cloth_object(garment.positions.clone(), &garment.tris, &garment.uvs);
        let plan = engine_api::plan_resize(proxy, &object, &girths, engine_api::resize::FIT_EASE)
            .map_err(|e| OpError::Failed(e.to_string()))?;
        let apply = arg_bool(OP, args, "resize")?.unwrap_or(false);
        // Reported always, applied only on request. A drafted pattern is graded,
        // not scaled — `nguo_doc::Feature::Support` says so in as many words —
        // and the proposal's job here is to *demonstrate* that a to-measure draft
        // needs no resize, which is only worth anything if the number is visible.
        let check = apply.then(|| engine_api::apply_resize(&mut object, &plan));
        resize = json!({
            "applied": apply,
            "factor": plan.factor,
            "ease": plan.ease,
            "supporting": plan.supporting,
            "length_delta_m": plan.length_delta_m(),
            "edge_ratio": check.map(|c| [c.edge_ratio_min_all, c.edge_ratio_max_all]),
            "landmarks": plan.fits.iter().map(|f| json!({
                "landmark": f.landmark,
                "body_girth_m": f.body_girth_m,
                "garment_girth_m": f.garment.map(|r| r.girth_m),
                "demands_factor": f.factor,
                "slack": f.slack_at(plan.factor),
            })).collect::<Vec<_>>(),
        });

        let fit = engine_api::avatar::resolve_penetration(
            proxy,
            &mut object,
            engine_api::avatar::FIT_GAP_M,
        )
        .map_err(|e| OpError::Failed(e.to_string()))?;
        penetration = json!({
            "gap_m": engine_api::avatar::FIT_GAP_M,
            "inside_before": fit.per_pass.first(),
            "inside_after": fit.inside_after,
            "deepest_inside_mm": fit.deepest_inside_m * 1000.0,
            "pushed": fit.pushed,
            "per_pass": fit.per_pass,
            "converged": fit.converged(),
            "edge_ratio": [fit.check.edge_ratio_min_all, fit.check.edge_ratio_max_all],
        });
        // Whatever the two passes changed is the garment the solver gets.
        garment.positions = object.positions;
        cropped = json!(true);
    }

    // --- 6. drape ---------------------------------------------------------
    let mut drape = Value::Null;
    if let Some(proxy) = &body {
        let box_of = engine_api::Aabb::of(&garment.positions)
            .ok_or_else(|| OpError::Failed("the sewn dress has no geometry".into()))?;
        let near = engine_api::avatar_collider_near(proxy, box_of, DRESS_CROP_M)
            .map_err(|e| OpError::Failed(e.to_string()))?;
        cropped = json!({
            "kept_verts": near.positions.len(),
            "of_verts": proxy.positions.len(),
            "kept_tris": near.tris.len(),
            "of_tris": proxy.tris.len(),
            "margin_m": DRESS_CROP_M,
        });

        // The garment is object **0**, so `Scene::lower`'s concatenation leaves
        // its vertex indices unchanged and the stitches need no offset.
        //
        // KEBAB-CASE, and it is not cosmetic: `SolverConfig` is
        // `#[serde(default, rename_all = "kebab-case")]`, so `self_collision`
        // would be an *unknown field* serde ignores, leaving the default `false`
        // — which once draped a garment 75 mm inside a body while reporting
        // success. Cloth-body contact runs through the self-contact pass; there
        // is no analytic collider in this scene, so `true` is mandatory.
        let scene = json!({
            "config": {
                "dt": DRESS_DT,
                "self-collision": true,
                "floor-y": 0.0,
                "iterations": match tier { Tier::Preview => 2, Tier::Bake => 8 },
            },
            "objects": [
                json!({
                    "name": "dress",
                    "positions": garment.positions,
                    "tris": garment.tris,
                    "uvs": garment.uvs,
                    "preset": "cotton",
                }),
                json!({
                    "name": format!("body/{}", key.as_deref().unwrap_or("?")),
                    "positions": near.positions,
                    "tris": near.tris,
                    "uvs": if near.uvs.len() == near.positions.len() {
                        near.uvs.clone()
                    } else {
                        vec![[0.0f32, 0.0]; near.positions.len()]
                    },
                    // Every vertex infinite-mass, which is also what makes the
                    // body cheap: the broadphase drops every candidate pair whose
                    // participants are all immovable (D-17).
                    "is_static": true,
                }),
            ],
        });
        let mut sink = DressFrames {
            stitches: &garment.stitches,
            cloth: garment.positions.len(),
            first: None,
            last: None,
            gaps: Vec::new(),
        };
        let started = std::time::Instant::now();
        let summary = InProcessTransport::new()
            .solve(
                &Submission {
                    scene,
                    stitches: garment
                        .stitches
                        .iter()
                        .map(|s| Stitch {
                            index: s.index,
                            weight: s.weight,
                        })
                        .collect(),
                    tier,
                    frames: Some(steps),
                    out_dir: None,
                },
                &mut sink,
            )
            .map_err(|e| OpError::Failed(format!("the drape failed: {e}")))?;
        let wall_s = started.elapsed().as_secs_f64();

        let last = sink
            .last
            .clone()
            .ok_or_else(|| OpError::Failed("the solver emitted no frame".into()))?;
        let finite = last.iter().flatten().all(|c| c.is_finite());
        // How much is still inside the body once it has settled — measured the
        // same way as before the solve, not inferred from "it looked right".
        let mut settled = cloth_object(last.clone(), &garment.tris, &garment.uvs);
        let after = engine_api::avatar::resolve_penetration(
            proxy,
            &mut settled,
            engine_api::avatar::FIT_GAP_M,
        )
        .map_err(|e| OpError::Failed(e.to_string()))?;
        let low = last.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let moved = sink.first.as_ref().map(|first| {
            first
                .iter()
                .zip(&last)
                .map(|(a, b)| ((0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>()).sqrt())
                .fold(0.0f32, f32::max)
        });

        drape = json!({
            "tier": summary.tier,
            "steps": steps,
            "frames_emitted": summary.frames_emitted,
            "engine": summary.report,
            "wall_s": wall_s,
            "all_finite": finite,
            "seam_gap_mm": {
                "arranged": arranged_gap_m * 1000.0,
                "first_frame": sink.gaps.first().map(|g| g * 1000.0),
                "final": sink.gaps.last().map(|g| g * 1000.0),
                "closed_pct": sink.gaps.first().zip(sink.gaps.last())
                    .map(|(a, b)| 100.0 * (1.0 - b / a)),
                "curve": sink.gaps.iter().enumerate()
                    .filter(|(i, _)| i % 12 == 0 || *i + 1 == sink.gaps.len())
                    .map(|(i, g)| json!([i, g * 1000.0]))
                    .collect::<Vec<_>>(),
            },
            "max_vertex_travel_mm": moved.map(|m| m * 1000.0),
            "hem_y_m": low,
            "inside_body_after_drape": after.per_pass.first(),
            "deepest_inside_after_drape_mm": after.deepest_inside_m * 1000.0,
        });
    }

    // --- 7. save the `.nguo`, and load it straight back -------------------
    let save = match arg_str(OP, args, "out")? {
        None => Value::Null,
        Some(dest) => {
            let path = std::path::PathBuf::from(&dest);
            let mut project = nguo_doc::Project::new(doc.clone());
            let bytes = project
                .save(&path)
                .map_err(|e| OpError::Failed(format!("saving {dest}: {e}")))?;
            let started = std::time::Instant::now();
            let loaded = nguo_doc::load(&path)
                .map_err(|e| OpError::Failed(format!("re-opening {dest}: {e}")))?;
            let load_ms = started.elapsed().as_secs_f64() * 1000.0;
            let members = nguo_doc::bundle::members(
                &std::fs::read(&path)
                    .map_err(|e| OpError::Failed(format!("re-reading {dest}: {e}")))?,
            )
            .map_err(|e| OpError::Failed(format!("{dest} is not a .nguo: {e}")))?;
            json!({
                "path": dest,
                "bytes": bytes,
                "load_ms": load_ms,
                // The whole point of a round trip: the same document, not a
                // similar one. `Document`'s `PartialEq` skips `revision`, which
                // is an edit counter and not content.
                "round_trip_identical": loaded.document == project.document,
                "members": members.iter().map(|(k, v)| json!([k, v.len()])).collect::<Vec<_>>(),
                "manifest": serde_json::to_value(&loaded.manifest).unwrap_or(Value::Null),
            })
        }
    };

    Ok(json!({
        "pattern": pattern,
        "body": body_report,
        "sew": sew,
        "resize": resize,
        "penetration": penetration,
        "body_cropped": cropped,
        "drape": drape,
        "save": save,
    }))
}

/// The body and the girths to fit against: a bundle avatar, or a mesh plus explicit
/// measurements. Exactly one, because a default body is a wrong body.
fn load_body(
    op: &str,
    args: &Value,
) -> Result<(engine_api::Object, Vec<BodyGirth>, String), OpError> {
    let girths = arg_girths(op, args)?;
    match body_choice(op, args)? {
        (Some(key), None) => {
            let bundle = nguo_avatar::Bundle::find()
                .map_err(|e| OpError::Failed(format!("avatar bundle: {e}")))?;
            let avatar = bundle
                .avatar(&key)
                .map_err(|e| OpError::args(op, e.to_string()))?;
            let mesh = bundle
                .load_mesh(avatar, nguo_avatar::Asset::GlbCollision)
                .map_err(|e| OpError::Failed(format!("avatar collision proxy: {e}")))?;
            let girths = girths.unwrap_or_else(|| {
                bundle
                    .measurements(avatar)
                    .girths()
                    .map(|(k, g)| BodyGirth {
                        landmark: k.to_string(),
                        circumference_m: (g.circumference.mm() / 1000.0) as f32,
                        height_m: (g.height.mm() / 1000.0) as f32,
                    })
                    .collect()
            });
            Ok((mesh, girths, key))
        }
        (None, Some(path)) => {
            let girths = girths.ok_or_else(|| {
                OpError::args(
                    op,
                    "`girths` is required with `body`: a mesh carries no measurements, and \
                     measuring them off it is what `avatar` is for",
                )
            })?;
            // Metres, as-written. A body proxy is engine-frame geometry, not an import.
            let mesh = engine_io::load_mesh(&path)
                .map_err(|e| OpError::Failed(format!("could not load `{path}`: {e:#}")))?;
            Ok((mesh, girths, path))
        }
        _ => unreachable!("body_choice returns exactly one"),
    }
}

// --- argument plumbing ----------------------------------------------------

/// Exactly one of `avatar` / `body`.
fn body_choice(op: &str, args: &Value) -> Result<(Option<String>, Option<String>), OpError> {
    match (arg_str(op, args, "avatar")?, arg_str(op, args, "body")?) {
        (Some(a), None) => Ok((Some(a), None)),
        (None, Some(b)) => Ok((None, Some(b))),
        (Some(_), Some(_)) => Err(OpError::args(
            op,
            "`avatar` and `body` are alternatives — give one, not both",
        )),
        (None, None) => Err(OpError::args(
            op,
            "one of `avatar` (a bundle key) or `body` (a mesh path in metres) is required",
        )),
    }
}

fn arg_units(op: &str, args: &Value) -> Result<MeshUnits, OpError> {
    match arg_str(op, args, "units")?.as_deref() {
        Some("m") => Ok(MeshUnits::Metres),
        Some("cm") => Ok(MeshUnits::Centimetres),
        Some("mm") => Ok(MeshUnits::Millimetres),
        Some("in") => Ok(MeshUnits::Inches),
        // No default. See the operation's description: a wrong unit is a silent 100x.
        None => Err(OpError::args(
            op,
            "`units` is required: no mesh format records its own unit, so it cannot be \
             inferred (one of \"m\", \"cm\", \"mm\", \"in\")",
        )),
        Some(other) => Err(OpError::args(
            op,
            format!("`units` must be one of \"m\", \"cm\", \"mm\", \"in\", got {other:?}"),
        )),
    }
}

fn arg_fabric(op: &str, args: &Value) -> Result<engine_api::FabricPreset, OpError> {
    match arg_str(op, args, "fabric")? {
        None => Ok(engine_api::FabricPreset::Cotton),
        Some(s) => serde_json::from_value(Value::String(s.clone()))
            .map_err(|_| OpError::args(op, format!("unknown fabric preset {s:?}"))),
    }
}

fn arg_f32(op: &str, args: &Value, key: &str) -> Result<Option<f32>, OpError> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|n| n.is_finite())
            .map(|n| Some(n as f32))
            .ok_or_else(|| OpError::args(op, format!("`{key}` must be a finite number, got {v}"))),
    }
}

fn arg_positive(op: &str, args: &Value, key: &str) -> Result<Option<f32>, OpError> {
    let v = arg_f32(op, args, key)?;
    match v {
        Some(n) if n <= 0.0 => Err(OpError::args(
            op,
            format!("`{key}` must be greater than zero, got {n}"),
        )),
        other => Ok(other),
    }
}

fn arg_girths(op: &str, args: &Value) -> Result<Option<Vec<BodyGirth>>, OpError> {
    let Some(v) = args.get("girths").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let items = v.as_array().ok_or_else(|| {
        OpError::args(
            op,
            format!(
                "`girths` must be an array of {{landmark, circumference_m, height_m}}, got {v}"
            ),
        )
    })?;
    let mut out = Vec::with_capacity(items.len());
    for (i, g) in items.iter().enumerate() {
        let at = |k: &str| -> Result<f32, OpError> {
            g.get(k)
                .and_then(Value::as_f64)
                .filter(|n| n.is_finite())
                .map(|n| n as f32)
                .ok_or_else(|| OpError::args(op, format!("girths[{i}].{k} must be a number")))
        };
        let landmark = g
            .get("landmark")
            .and_then(Value::as_str)
            .ok_or_else(|| OpError::args(op, format!("girths[{i}].landmark must be a string")))?;
        let circumference_m = at("circumference_m")?;
        if circumference_m <= 0.0 {
            return Err(OpError::args(
                op,
                format!("girths[{i}].circumference_m must be greater than zero"),
            ));
        }
        out.push(BodyGirth {
            landmark: landmark.to_string(),
            circumference_m,
            height_m: at("height_m")?,
        });
    }
    Ok(Some(out))
}

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

    // --- garment_fit ------------------------------------------------------
    //
    // Synthetic geometry only: no `samples/`, no avatar bundle. That is what lets
    // the drop-and-fit path be tested in CI at all — both are gitignored/LFS.

    /// A closed, capped cylinder on the y axis as OBJ text, in `unit` units per metre.
    ///
    /// Outward wound, and the caller checks that rather than trusting this comment —
    /// `engine_api::avatar::resolve_penetration` requires it, and the `box_mesh` fixture
    /// next door is wound INWARD, so "it looked right" is not evidence here.
    fn cylinder_obj(r: f64, y0: f64, y1: f64, sides: usize, rows: usize, per_m: f64) -> String {
        let mut s = String::from("# synthetic test cylinder\n");
        let ring = |i: usize| {
            let y = y0 + (y1 - y0) * i as f64 / rows as f64;
            (0..sides).map(move |k| {
                let t = std::f64::consts::TAU * k as f64 / sides as f64;
                [r * t.cos(), y, r * t.sin()]
            })
        };
        for i in 0..=rows {
            for p in ring(i) {
                s += &format!("v {} {} {}\n", p[0] * per_m, p[1] * per_m, p[2] * per_m);
            }
        }
        // Cap centres last, so the side indices are a clean grid.
        let n = sides * (rows + 1);
        s += &format!("v 0 {} 0\nv 0 {} 0\n", y0 * per_m, y1 * per_m);
        let (bot, top) = (n + 1, n + 2); // OBJ is 1-based
        let at = |i: usize, k: usize| i * sides + (k % sides) + 1;
        for i in 0..rows {
            for k in 0..sides {
                // (A,D,C) then (A,C,B) round the quad — the winding that puts the
                // normal on the +radial side. Verified by the volume check below.
                let (a, b) = (at(i, k), at(i, k + 1));
                let (d, c) = (at(i + 1, k), at(i + 1, k + 1));
                s += &format!("f {a} {d} {c}\nf {a} {c} {b}\n");
            }
        }
        for k in 0..sides {
            s += &format!("f {bot} {} {}\n", at(0, k), at(0, k + 1));
            s += &format!("f {top} {} {}\n", at(rows, k + 1), at(rows, k));
        }
        s
    }

    /// An open tube — a garment, no caps.
    fn tube_obj(r: f64, y0: f64, y1: f64, sides: usize, rows: usize, per_m: f64) -> String {
        let full = cylinder_obj(r, y0, y1, sides, rows, per_m);
        let n = sides * (rows + 1);
        full.lines()
            .filter(|l| {
                !l.starts_with("f ")
                    || l.split_whitespace()
                        .skip(1)
                        .all(|i| i.parse::<usize>().unwrap() <= n)
            })
            .filter(|l| !(l.starts_with("v 0 ") && l.ends_with(" 0")))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn write(dir: &Path, name: &str, text: String) -> String {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p.display().to_string()
    }

    /// Signed volume × 6. Positive exactly when a closed mesh is outward wound.
    fn six_volume(path: &str) -> f64 {
        let m = engine_io::load_mesh(path).unwrap();
        m.tris
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| m.positions[i as usize].map(f64::from));
                a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
            })
            .sum()
    }

    /// Body at r = 0.16 m (hip girth 1.0 m), garment tube at r = 0.12 m — 25 % too
    /// small to go on, which is the case drop-and-fit exists for.
    fn fixture(dir: &Path, garment_per_m: f64) -> (String, String) {
        let body = write(dir, "body.obj", cylinder_obj(0.16, 0.0, 1.6, 32, 32, 1.0));
        assert!(six_volume(&body) > 0.0, "the body fixture is inward wound");
        let garment = write(
            dir,
            "tube.obj",
            tube_obj(0.12, 0.75, 1.05, 24, 15, garment_per_m),
        );
        (body, garment)
    }

    fn hip() -> Value {
        json!([{"landmark": "hip", "circumference_m": 1.0, "height_m": 0.9}])
    }

    #[tokio::test]
    async fn garment_fit_resizes_a_too_small_tube_onto_the_body_and_shows_its_working() {
        let dir = tempfile::tempdir().unwrap();
        let (body, garment) = fixture(dir.path(), 1.0);
        let out = dir.path().join("fitted.json");
        let report = dispatch(
            "garment_fit",
            &json!({
                "garment": garment, "units": "m",
                "body": body, "girths": hip(),
                "out": out.display().to_string(),
            }),
            &Env::default(),
        )
        .await
        .expect("the tube fits");

        // A 24-gon of radius 0.12 has hull girth 2·24·0.12·sin(π/24) = 0.75186 m, so the
        // factor is 1.0 · 1.03 / 0.75186 = 1.3699 — computed, not observed.
        let f = report["resize"]["factor"].as_f64().unwrap();
        assert!((f - 1.3699).abs() < 1e-3, "factor {f}");
        assert_eq!(report["resize"]["supporting"], "hip");
        assert_eq!(report["resize"]["overridden"], false);
        // The residual the factor leaves at the landmark IS the ease, and reporting it is
        // the point: a uniform factor that cannot hit every landmark says so here.
        let slack = report["resize"]["landmarks"][0]["slack"].as_f64().unwrap();
        assert!((slack - 0.03).abs() < 1e-4, "slack {slack}");
        // A 0.30 m tube scaled 1.37× is 0.111 m longer. Never silent: a similarity that
        // fixes a girth pays for it in length.
        let dl = report["resize"]["length_delta_m"].as_f64().unwrap();
        assert!((dl - 0.111).abs() < 2e-3, "length delta {dl}");

        // r = 0.12 is inside a body of r = 0.16; the resize takes it to 0.1644, so by the
        // time `resolve_penetration` looks there is nothing left to push.
        assert_eq!(report["penetration"]["inside_before"], 0);
        assert_eq!(report["penetration"]["inside_after"], 0);
        assert_eq!(report["penetration"]["converged"], true);

        // The control, in the same test, because "0 inside" proves nothing without it:
        // decline the resize and the identical garment is 384 vertices deep in the body.
        // This is the number the operation exists to remove.
        let declined = dispatch(
            "garment_fit",
            &json!({
                "garment": garment, "units": "m",
                "body": body, "girths": hip(), "factor": 1.0,
            }),
            &Env::default(),
        )
        .await
        .expect("declining still runs");
        assert_eq!(declined["penetration"]["inside_before"], 384);
        // `resolve_penetration` does get them out — but only by stretching the mesh to
        // exactly `FIT_STRAIN_RATIO`, the bound that exists because unbounded projection
        // reached edge ratios of 75.77×. A garment that needs the whole strain budget to
        // go on is a garment that needs resizing, which is the point of the pair.
        let stretched = declined["penetration"]["edge_ratio"][1].as_f64().unwrap();
        assert!((1.49..=1.51).contains(&stretched), "{declined:#}");
        assert!(
            report["penetration"]["edge_ratio"][1].as_f64().unwrap() < 1.001,
            "the resized garment should not have been deformed at all: {report:#}",
        );

        // The scene it wrote is one the rest of the surface accepts.
        let info = dispatch(
            "scene_validate",
            &json!({"scene": out.display().to_string()}),
            &Env::default(),
        )
        .await
        .expect("the fitted scene is valid");
        assert_eq!(info["ok"], true);
    }

    /// The unit is load-bearing and cannot be inferred — the reason `units` has no
    /// default. Same bytes on disk, one declaration apart, and the answers differ by 100×.
    #[tokio::test]
    async fn the_declared_unit_changes_the_answer_and_omitting_it_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        // Authored in centimetres: the identical file, 100× the numbers.
        let (body, garment) = fixture(dir.path(), 100.0);
        let run = async |units: &str| {
            let args = json!({
                "garment": garment.clone(), "units": units,
                "body": body.clone(), "girths": hip(),
            });
            dispatch("garment_fit", &args, &Env::default()).await
        };

        let cm = run("cm").await.expect("centimetres is the truth here");
        assert!((cm["resize"]["factor"].as_f64().unwrap() - 1.3699).abs() < 1e-3);

        // Declared as metres the tube is 12 m across and 30 m tall: it reaches no
        // landmark, so there is no usable ring and the honest plan is the identity.
        let m = run("m").await.expect("still loads — it is just wrong");
        assert_eq!(m["resize"]["factor"], 1.0);
        assert_eq!(m["resize"]["supporting"], Value::Null);

        for (args, want) in [
            (
                json!({"garment": garment.clone(), "body": body.clone(), "girths": hip()}),
                "`units` is required",
            ),
            (
                json!({"garment": garment.clone(), "units": "cubits", "body": body.clone(), "girths": hip()}),
                "`units` must be one of",
            ),
            (
                json!({"garment": garment.clone(), "units": "cm"}),
                "one of `avatar`",
            ),
            (
                json!({"garment": garment.clone(), "units": "cm", "body": body.clone()}),
                "`girths` is required with `body`",
            ),
            (
                json!({"garment": garment.clone(), "units": "cm", "avatar": "female", "body": body.clone()}),
                "alternatives",
            ),
            (
                json!({"garment": garment.clone(), "units": "cm", "body": body.clone(), "girths": hip(), "factor": 0.0}),
                "greater than zero",
            ),
        ] {
            let err = dispatch("garment_fit", &args, &Env::default())
                .await
                .expect_err(&format!("must reject {args}"));
            assert!(err.to_string().contains(want), "wanted {want:?} in {err}",);
        }
    }

    /// The proposal is a proposal. Overriding it keeps the proposal visible, because a
    /// number you cannot see is a number you cannot argue with.
    #[tokio::test]
    async fn the_proposed_factor_is_overridable_and_the_proposal_survives() {
        let dir = tempfile::tempdir().unwrap();
        let (body, garment) = fixture(dir.path(), 1.0);
        let report = dispatch(
            "garment_fit",
            &json!({
                "garment": garment, "units": "m",
                "body": body, "girths": hip(),
                "factor": 1.5,
            }),
            &Env::default(),
        )
        .await
        .expect("fits");
        assert_eq!(report["resize"]["factor"], 1.5);
        assert!((report["resize"]["proposed"].as_f64().unwrap() - 1.3699).abs() < 1e-3);
        assert_eq!(report["resize"]["overridden"], true);
        // 1.5 over the 1.3699 the body asked for: 9.4 % of slack instead of 3 %.
        let slack = report["resize"]["landmarks"][0]["slack"].as_f64().unwrap();
        assert!((slack - 0.1278).abs() < 1e-3, "slack {slack}");
    }
}
