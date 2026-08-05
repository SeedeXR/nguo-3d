// SPDX-License-Identifier: GPL-2.0-or-later
//! The `nguo` command line. Every subcommand is one [`crate::ops`] operation —
//! it only turns flags into the operation's argument JSON.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Value};

#[derive(Parser)]
#[command(
    name = "nguo",
    version,
    about = "Headless, scriptable driver for the nguo-3d cloth suite",
    long_about = "Headless driver for the nguo-3d cloth suite, for agents and CI.\n\n\
                  Process contract (same as the `mguso` engine binary):\n  \
                  stdout — exactly one JSON value, nothing else\n  \
                  stderr — logs\n  \
                  exit 0 success, 1 failure, 2 the input is invalid\n\n\
                  Run `nguo schema` for the machine-readable command surface."
)]
pub struct Cli {
    /// Log format on stderr.
    #[arg(long, value_enum, global = true, default_value_t = LogFormat::Text)]
    pub log_format: LogFormat,
    /// Increase log verbosity (-v debug, -vv trace). `RUST_LOG` wins if set.
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
    /// Path to the `mguso` engine binary. Default: $NGUO_MGUSO, else the
    /// sibling core-engine target dir, else `mguso` on PATH.
    #[arg(long, global = true)]
    pub engine: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum TierArg {
    /// XPBD — fast, interactive.
    Preview,
    /// Implicit backward Euler — high fidelity, slow.
    Bake,
}

impl TierArg {
    fn as_str(self) -> &'static str {
        match self {
            TierArg::Preview => "preview",
            TierArg::Bake => "bake",
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Describe every nguo operation: names, argument JSON Schemas, exit codes.
    Schema,
    /// Report the GPU backend and adapter the solver will use, and probe it.
    GpuInfo,
    /// Summarise an engine scene: solver config, objects, vertex/triangle counts.
    SceneInfo {
        /// Path to an engine Scene .json file.
        scene: PathBuf,
    },
    /// Check a scene for structural faults before spending time on a drape.
    SceneValidate {
        /// Path to an engine Scene .json file.
        scene: PathBuf,
        /// Also run `mguso verify` — actually simulates, so this is slow.
        #[arg(long)]
        solver_invariants: bool,
        /// Solver steps for `--solver-invariants`.
        #[arg(long)]
        steps: Option<u32>,
    },
    /// Simulate a scene and print the engine's drape report.
    Drape {
        /// Path to an engine Scene .json file.
        scene: PathBuf,
        #[arg(long, value_enum, default_value_t = TierArg::Preview)]
        tier: TierArg,
        /// Frames to simulate. Default: the scene's config.frames.
        #[arg(long)]
        frames: Option<u32>,
        /// Directory for the per-frame OBJ sequence (this is also the export).
        #[arg(long = "out")]
        out_dir: Option<PathBuf>,
    },
    /// Scale an imported garment onto a body and push out the residual overlap.
    GarmentFit {
        /// Path to the garment mesh (.obj, .stl, .gltf, .glb, .usda).
        garment: PathBuf,
        /// Unit the garment file is authored in. Required: no mesh format records
        /// its own unit and guessing it is a silent 100x.
        #[arg(long, value_enum)]
        units: UnitsArg,
        /// Avatar bundle key ("male", "female"): body and girths come from it.
        #[arg(long, conflicts_with = "body")]
        avatar: Option<String>,
        /// Body mesh in metres, instead of --avatar. Needs --girths.
        #[arg(long)]
        body: Option<PathBuf>,
        /// Body girths as JSON: [{"landmark","circumference_m","height_m"}, ...].
        /// Required with --body; overrides the manifest with --avatar.
        #[arg(long)]
        girths: Option<String>,
        /// Girth ease the factor targets. 0.03 = 3 % looser than the body.
        #[arg(long)]
        ease: Option<f32>,
        /// Override the proposed factor. The proposal is still reported.
        #[arg(long)]
        factor: Option<f32>,
        /// Fabric preset written into the scene.
        #[arg(long)]
        fabric: Option<String>,
        /// Write the fitted, ready-to-drape scene JSON here.
        #[arg(long = "out")]
        out: Option<PathBuf>,
    },
}

#[derive(Copy, Clone, ValueEnum)]
pub enum UnitsArg {
    /// Metres.
    M,
    /// Centimetres — most DCC garment exports.
    Cm,
    /// Millimetres.
    Mm,
    /// Inches.
    In,
}

impl UnitsArg {
    fn as_str(self) -> &'static str {
        match self {
            UnitsArg::M => "m",
            UnitsArg::Cm => "cm",
            UnitsArg::Mm => "mm",
            UnitsArg::In => "in",
        }
    }
}

impl Command {
    /// The [`crate::ops`] operation name and argument object for this
    /// subcommand. The CLI has no logic of its own beyond this mapping.
    pub fn to_op(&self) -> (&'static str, Value) {
        match self {
            Command::Schema => ("schema", json!({})),
            Command::GpuInfo => ("gpu_info", json!({})),
            Command::SceneInfo { scene } => {
                ("scene_info", json!({"scene": scene.display().to_string()}))
            }
            Command::SceneValidate {
                scene,
                solver_invariants,
                steps,
            } => (
                "scene_validate",
                json!({
                    "scene": scene.display().to_string(),
                    "solver_invariants": solver_invariants,
                    "steps": steps,
                }),
            ),
            Command::Drape {
                scene,
                tier,
                frames,
                out_dir,
            } => (
                "drape",
                json!({
                    "scene": scene.display().to_string(),
                    "tier": tier.as_str(),
                    "frames": frames,
                    "out_dir": out_dir.as_ref().map(|d| d.display().to_string()),
                }),
            ),
            Command::GarmentFit {
                garment,
                units,
                avatar,
                body,
                girths,
                ease,
                factor,
                fabric,
                out,
            } => (
                "garment_fit",
                json!({
                    "garment": garment.display().to_string(),
                    "units": units.as_str(),
                    "avatar": avatar,
                    "body": body.as_ref().map(|p| p.display().to_string()),
                    // Passed through as JSON so the shape stays the operation's, not a
                    // second flag grammar that can drift from the schema. Unparseable
                    // text is forwarded verbatim: `arg_girths` then names the flag and
                    // echoes what it got, which beats a parse error with no context.
                    "girths": girths.as_ref().map(|g| {
                        serde_json::from_str::<Value>(g).unwrap_or_else(|_| json!(g))
                    }),
                    "ease": ease,
                    "factor": factor,
                    "fabric": fabric,
                    "out": out.as_ref().map(|p| p.display().to_string()),
                }),
            ),
        }
    }
}

/// Logs to stderr only — stdout carries the JSON result and nothing else.
pub fn init_logging(format: LogFormat, verbose: u8) {
    use tracing_subscriber::{fmt, EnvFilter};
    let default = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_env("RUST_LOG").unwrap_or_else(|_| EnvFilter::new(default));
    let builder = fmt().with_env_filter(filter).with_writer(std::io::stderr);
    match format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().init(),
    }
}
