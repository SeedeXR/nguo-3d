// SPDX-License-Identifier: GPL-2.0-or-later
//! mguso — headless engine runner and process-contract entry point.
//!
//! Process contract (see docs/PROCESS_CONTRACT.md):
//!   * exit 0 = success; 1 = failure (diverged / IO / no GPU when required);
//!     2 = `verify` ran but a solver invariant failed.
//!   * logs go to stderr; structured JSON logs via `--log-format json` or
//!     `MGUSO_LOG=json`. Level via `RUST_LOG` / `--verbose`.
//!   * machine-readable results go to stdout as JSON.

mod gauge_driver;

use anyhow::{Context, Result};

/// Allocation accounting, so peak LIVE bytes can be told apart from allocator churn.
///
/// Peak RSS on `domino` climbs to ~1.8 GB while the scene itself loads in 8 MB, and it goes
/// DOWN again between 50 and 200 steps — which rules out a leak but not a large transient.
/// RSS cannot distinguish "we hold 1.8 GB live" from "we allocated and freed 1.8 GB and the
/// system allocator kept the pages"; this can. Two relaxed atomics on the alloc path, and
/// `MGUSO_MEMSTATS=1` prints the numbers at exit.
mod memstats {
    use engine_core::alloc_stats as st;
    use std::alloc::{GlobalAlloc, Layout, System};

    pub struct Tracking;

    // SAFETY: every method forwards to `System` unchanged; only counters are added. The
    // counters live in `engine_core::alloc_stats` so the SOLVER can attribute peaks to a
    // phase of the step (`MGUSO_MEM_TRACE=1`) — a binary-local allocator cannot.
    unsafe impl GlobalAlloc for Tracking {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(l) };
            if !p.is_null() {
                st::on_alloc(l.size());
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            st::on_dealloc(l.size());
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            let q = unsafe { System.realloc(p, l, new) };
            if !q.is_null() {
                st::on_realloc(l.size(), new);
            }
            q
        }
    }

    pub fn report() {
        if std::env::var("MGUSO_MEMSTATS").is_err() {
            return;
        }
        let mb = |b: usize| b as f64 / 1048576.0;
        eprintln!(
            "memstats: peak live {:.1} MB, still live {:.1} MB, total allocated {:.1} MB \
             over {} allocations",
            mb(st::peak_bytes()),
            mb(st::live_bytes()),
            mb(st::TOTAL.load(std::sync::atomic::Ordering::Relaxed)),
            st::COUNT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
}

#[global_allocator]
static ALLOC: memstats::Tracking = memstats::Tracking;
use clap::{Parser, Subcommand, ValueEnum};
use engine_api::Scene;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mguso", version, about = "ppf-mguso-solver headless engine")]
struct Cli {
    #[arg(long, value_enum, global = true, default_value_t = LogFormat::Text)]
    log_format: LogFormat,
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, ValueEnum)]
enum LogFormat {
    Text,
    Json,
}

#[derive(Subcommand)]
enum Command {
    /// Print the selected GPU backend/adapter and run the no-op SAXPY probe.
    GpuInfo,
    /// Inspect a scene from the reference capture: report its element counts, the
    /// per-face constitutive-model split, and — importantly — the fidelity gaps this
    /// build cannot reproduce. Optionally step it to check stability. The capture is
    /// an internal corpus, not distributed with this source (docs/GAUGE_FINDINGS.md).
    GaugeScene {
        /// Path to a scene directory (e.g. `<capture>/goldens/plastic`).
        scene: PathBuf,
        /// Steps to simulate (0 = load and report only).
        #[arg(long, default_value_t = 0)]
        steps: u32,
        /// Simulate to reference output frame N and compare against
        /// `checkpoints/vert_N.bin`. Overrides `--steps`.
        #[arg(long)]
        compare_frame: Option<u32>,
    },
    /// Simulate a scene (JSON scene file, or an OBJ treated as a draping cloth).
    Run {
        /// Scene .json (engine-api::Scene) or mesh .obj
        scene: PathBuf,
        /// Frames to simulate (overrides scene config if set).
        #[arg(long)]
        frames: Option<u32>,
        /// Write per-frame OBJ sequence into this directory.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Solver tier: `xpbd` (fast preview) or `implicit` (backward-Euler).
        #[arg(long, value_enum, default_value_t = SolverKind::Xpbd)]
        solver: SolverKind,
        /// Force the CPU linear solver for `implicit` (default: GPU if available).
        #[arg(long)]
        cpu: bool,
        /// Preconditioner (implicit; CPU + GPU): `block` (default) or `schwarz-ml`.
        #[arg(long, value_enum, default_value_t = PrecondKind::default())]
        precond: PrecondKind,
    },
    /// Benchmark solver throughput (steps/sec) on a scene.
    Bench {
        scene: PathBuf,
        /// Timed steps (after a short warm-up).
        #[arg(long, default_value_t = 200)]
        steps: u32,
        #[arg(long, value_enum, default_value_t = SolverKind::Implicit)]
        solver: SolverKind,
        #[arg(long)]
        cpu: bool,
        /// Preconditioner (implicit; CPU + GPU): `block` (default) or `schwarz-ml`.
        #[arg(long, value_enum, default_value_t = PrecondKind::default())]
        precond: PrecondKind,
    },
    /// Run a scene with the implicit solver and check the solver invariants
    /// (penetration-free, strain-limit, CG-converged, zero-steady-alloc,
    /// all-finite). Exits non-zero if any invariant fails. This is the primary
    /// acceptance gate (see docs/GAUGE_FINDINGS.md).
    Verify {
        scene: PathBuf,
        /// Steps to simulate (default from scene fps/dt for `frames` frames).
        #[arg(long)]
        steps: Option<u32>,
    },
    /// Drive our solver with a golden scene directory from the reference capture
    /// and report the per-checkpoint divergence gates (max-rel / rms / centroid).
    /// The capture is an internal benchmark corpus, not distributed with this
    /// source — see docs/GAUGE_FINDINGS.md.
    /// APPROXIMATE: uses our default material + HARD pins at the initial position
    /// (the reference uses soft/kinematic FixPairs); a failing gate reflects setup
    /// approximation + our mass-spring fidelity, not a pure solver defect.
    Golden {
        /// Golden scene directory (e.g. `<capture>/goldens/hang`).
        scene: PathBuf,
        /// Only run up to this checkpoint frame (default: all).
        #[arg(long)]
        max_frame: Option<u32>,
        /// Velocity damping [0,1) approximating the reference's aerodynamic drag
        /// (param.toml `air_friction`); 0 = none. The reference damps the swing.
        #[arg(long, default_value_t = 0.0)]
        damping: f32,
        /// Enable XPBD dihedral bending (default off — its FD gradient is slow and
        /// negligible on tension-dominated cloth like hang).
        #[arg(long)]
        bend: bool,
        /// Enable the anisotropic aero drag (default off — the naive explicit
        /// coefficient over-damps vs the reference's implicit damper and worsens
        /// parity; kept for calibration experiments).
        #[arg(long)]
        aero: bool,
        #[arg(long, value_enum, default_value_t = SolverKind::Xpbd)]
        solver: SolverKind,
        #[arg(long)]
        cpu: bool,
        #[arg(long, value_enum, default_value_t = PrecondKind::default())]
        precond: PrecondKind,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum SolverKind {
    Xpbd,
    Implicit,
}

/// CPU linear-solver preconditioner for the implicit solver.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Default)]
enum PrecondKind {
    /// 3×3 per-vertex block-Jacobi (default; matches the reference default).
    #[default]
    Block,
    /// Two-level Galerkin additive Schwarz — fewer CG iterations on large cloth
    /// (measured ~9% faster than block-Jacobi on the 900-vert fabric; roadmap B).
    SchwarzMl,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    init_logging(cli.log_format, cli.verbose);
    // `verify` maps invariant failure to a non-zero exit distinct from errors.
    if let Command::Verify { scene, steps } = &cli.command {
        return match verify(scene.clone(), *steps) {
            Ok(true) => std::process::ExitCode::SUCCESS,
            Ok(false) => std::process::ExitCode::from(2), // invariants failed
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "verify failed");
                std::process::ExitCode::FAILURE
            }
        };
    }
    let out = run(cli);
    memstats::report();
    match out {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %format!("{e:#}"), "command failed");
            std::process::ExitCode::FAILURE
        }
    }
}

fn init_logging(format: LogFormat, verbose: u8) {
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

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::GpuInfo => gpu_info(),
        Command::Run {
            scene,
            frames,
            out,
            solver,
            cpu,
            precond,
        } => run_scene(scene, frames, out, solver, cpu, precond),
        Command::Bench {
            scene,
            steps,
            solver,
            cpu,
            precond,
        } => bench(scene, steps, solver, cpu, precond),
        Command::GaugeScene {
            scene,
            steps,
            compare_frame,
        } => {
            let (mut run, report) = gauge_driver::scene_run(&scene)
                .with_context(|| format!("loading gauge scene {}", scene.display()))?;
            println!("scene: {}", scene.display());
            println!(
                "  verts {}  tris {}  tets {}  rods {} (bend hinges {})",
                report.n_verts, report.n_tris, report.n_tets, report.n_rods, report.n_rod_hinges
            );
            println!(
                "  model split: baraff-witkin {}  isotropic-shell {}",
                report.n_baraff_witkin, report.n_shell
            );
            println!(
                "  solid-surface faces {}  strain-limited faces {}  pins {} ({} kinematic)  pulls {}  colliders {} ({} animated)",
                report.n_solid_surface,
                report.n_strain_limited,
                report.n_pins_applied,
                report.n_pins_kinematic,
                report.n_pulls_applied,
                report.n_colliders,
                report.n_colliders_animated
            );
            if report.unsupported.is_empty() {
                println!("  every declared feature of this scene is modeled");
            } else {
                println!("  NOT reproduced by this build:");
                for u in &report.unsupported {
                    println!("    - {u}");
                }
            }
            if let Some(frame) = compare_frame {
                use engine_core::linsolve::CgCpu;
                let p = engine_io::golden::read_param_toml(&scene)
                    .with_context(|| "param.toml is required to map a frame to a time")?;
                // Output frame N is at t = N / fps; step until the clock reaches it. dt need
                // not divide the frame interval, so the last step may overshoot slightly —
                // reported so the comparison is not silently off-time.
                let target_t = f64::from(frame) / f64::from(p.fps);
                let reference = engine_io::golden::load_checkpoint(&scene, frame)
                    .with_context(|| format!("checkpoints/vert_{frame}.bin"))?;
                let solve = CgCpu::new();
                let mut n_steps = 0u64;
                while run.solver.state().time < target_t {
                    run.step(&solve);
                    n_steps += 1;
                    if !run.solver.state.positions.iter().all(|q| q.is_finite()) {
                        anyhow::bail!("non-finite state at step {n_steps}");
                    }
                }
                let ours: Vec<[f32; 3]> = run
                    .solver
                    .state
                    .positions
                    .iter()
                    .map(|q| [q.x, q.y, q.z])
                    .collect();
                let gate = engine_io::golden::compare_frames(&ours, &reference)?;
                println!(
                    "  frame {frame}: target t {target_t:.6}, reached {:.6} in {n_steps} steps",
                    run.solver.state().time
                );
                println!(
                    "  divergence vs reference: max_rel {:.4e}  rms {:.4e}  centroid_rel {:.4e}  \
                     (diagonal {:.4})",
                    gate.max_rel, gate.rms, gate.centroid_rel, gate.diagonal
                );
                let (g0, g1, g2) = engine_io::golden::GateReport::GATES;
                println!(
                    "  gates (max_rel < {g0:.0e}, rms < {g1:.0e}, centroid_rel < {g2:.0e}): {}",
                    if gate.passes() { "PASS" } else { "FAIL" }
                );

                // A passing gate is NOT evidence on its own. The gates normalize by the
                // scene's bbox diagonal, so on a scene whose reference barely moves they are
                // satisfied by predicting nothing at all. Score against that null model —
                // "frame 0 held still" — and report the SKILL: the fraction of the
                // reference's actual motion we captured. 0 means no better than not
                // simulating; 1 means exact.
                let frame0 = engine_io::golden::load_checkpoint(&scene, 0)?;
                let null = engine_io::golden::compare_frames(&frame0, &reference)?;
                let skill = |ours: f32, none: f32| {
                    if none > 0.0 {
                        1.0 - ours / none
                    } else {
                        f32::NAN
                    }
                };
                println!(
                    "  null model (predict frame 0, no simulation): max_rel {:.4e}  rms {:.4e}  \
                     gates {}",
                    null.max_rel,
                    null.rms,
                    if null.passes() {
                        "PASS  <-- the gates cannot distinguish a solver from no solver here"
                    } else {
                        "FAIL"
                    }
                );
                println!(
                    "  SKILL vs null (1 = exact, 0 = no better than not simulating): \
                     max {:.3}  rms {:.3}",
                    skill(gate.max_rel, null.max_rel),
                    skill(gate.rms, null.rms)
                );

                // Displacement-field diagnostic. A single skill number says "wrong" but not
                // HOW: too stiff, too soft, or moving the wrong way. Comparing our
                // displacement field against the reference's separates those:
                //   ratio ~1 with low cosine  -> right magnitude, wrong direction
                //   ratio << 1                -> under-deforming (too stiff / missing force)
                //   ratio >> 1                -> over-deforming (too soft)
                let ours_d: Vec<[f32; 3]> = ours
                    .iter()
                    .zip(&frame0)
                    .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
                    .collect();
                let ref_d: Vec<[f32; 3]> = reference
                    .iter()
                    .zip(&frame0)
                    .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
                    .collect();
                let norm = |v: &[[f32; 3]]| -> f64 {
                    v.iter()
                        .map(|d| ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f64).sqrt())
                        .sum::<f64>()
                        / v.len().max(1) as f64
                };
                let (mo, mr) = (norm(&ours_d), norm(&ref_d));
                let dot: f64 = ours_d
                    .iter()
                    .zip(&ref_d)
                    .map(|(a, b)| (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) as f64)
                    .sum();
                let la: f64 = ours_d
                    .iter()
                    .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f64)
                    .sum::<f64>()
                    .sqrt();
                let lb: f64 = ref_d
                    .iter()
                    .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f64)
                    .sum::<f64>()
                    .sqrt();
                let cos = if la > 0.0 && lb > 0.0 {
                    dot / (la * lb)
                } else {
                    f64::NAN
                };
                println!(
                    "  displacement field: ours {mo:.6}  reference {mr:.6}  ratio {:.3}  \
                     cosine {cos:.3}",
                    if mr > 0.0 { mo / mr } else { f64::NAN }
                );
            } else if steps > 0 {
                use engine_core::linsolve::CgCpu;
                let solve = CgCpu::new();
                for i in 0..steps {
                    // `SceneRun::step` refreshes animated pin targets from the clock.
                    run.step(&solve);
                    if !run.solver.state.positions.iter().all(|p| p.is_finite()) {
                        anyhow::bail!("non-finite state at step {i}");
                    }
                }
                let min_y = run
                    .solver
                    .state
                    .positions
                    .iter()
                    .map(|p| p.y)
                    .fold(f32::INFINITY, f32::min);
                let m = &run.solver.metrics.steps;
                let (ni_max, cg_max) = m.iter().fold((0u32, 0u32), |(a, b), s| {
                    (a.max(s.newton_iters), b.max(s.max_cg_iters))
                });
                let nc = m.iter().map(|s| s.num_contacts).max().unwrap_or(0);
                println!("  max per step: newton {ni_max}  cg {cg_max}  contacts {nc}");
                println!(
                    "  stepped {steps} step(s) to t = {:.4}: all finite, min y = {min_y:.6}{}",
                    run.solver.state().time,
                    if run.has_animated_pins() {
                        " (pin targets animated)"
                    } else {
                        ""
                    }
                );
            }
            Ok(())
        }
        Command::Golden {
            scene,
            max_frame,
            damping,
            bend,
            aero,
            solver,
            cpu,
            precond,
        } => golden_run(scene, max_frame, damping, bend, aero, solver, cpu, precond),
        // Handled in main() for the distinct exit code.
        Command::Verify { .. } => unreachable!(),
    }
}

/// Drive our solver with a golden scene from the reference capture and report divergence gates.
#[allow(clippy::too_many_arguments)]
fn golden_run(
    scene_dir: PathBuf,
    max_frame: Option<u32>,
    damping: f32,
    enable_bend: bool,
    enable_aero: bool,
    solver: SolverKind,
    cpu: bool,
    precond: PrecondKind,
) -> Result<()> {
    let input = engine_io::load_golden_input(&scene_dir).context("golden input_bin")?;
    let init = engine_io::load_checkpoint(&scene_dir, 0).context("checkpoint 0")?;
    let pins = engine_io::read_pin_indices(&scene_dir).unwrap_or_default();
    let params = engine_io::read_param_toml(&scene_dir).context("param.toml")?;

    // Discover checkpoint frames (sorted), optionally capped.
    let mut frames: Vec<u32> = std::fs::read_dir(scene_dir.join("checkpoints"))
        .with_context(|| "reading checkpoints dir")?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.file_name().to_str().and_then(|s| {
                s.strip_prefix("vert_")?
                    .strip_suffix(".bin")?
                    .parse::<u32>()
                    .ok()
            })
        })
        .collect();
    frames.sort_unstable();
    if let Some(mf) = max_frame {
        frames.retain(|&f| f <= mf);
    }

    // Scene from the golden data: initial positions from checkpoint 0, top-row pins
    // held (hard) at their initial positions, dt/gravity/fps from param.toml.
    // Material: use the reference's own per-object stiffness when captured
    // (input_bin/param/tri-*.bin) — the dominant lever for drape parity — else our
    // default. (Our default young_mod is 10× the reference's, which skews the swing.)
    let mut material = engine_api::Material::default();
    if let Some(m) = engine_io::read_material(&scene_dir) {
        material.young_mod = m.young_mod;
        material.bend = m.bend;
        material.density = m.density;
        material.strain_limit = m.strain_limit;
        eprintln!(
            "reference material: young_mod={} bend={} density={} strain_limit={}",
            m.young_mod, m.bend, m.density, m.strain_limit
        );
    }
    let bend = material.bend;
    let obj = engine_api::Object {
        name: "golden".into(),
        positions: init.clone(),
        tris: input.tris.clone(),
        material,
        preset: None,
        pins: pins.clone(),
        is_static: false,
        normals: Vec::new(),
        uvs: Vec::new(),
        skin: None,
        animations: Vec::new(),
    };
    let mut scene = engine_api::Scene::default();
    scene.config.dt = params.dt;
    scene.config.fps = params.fps as f64;
    scene.config.gravity = params.gravity;
    scene.config.frames = params.frames as i32;
    scene.config.damping = damping;
    // Enable the anisotropic aero drag (config air_density/air_friction default to the
    // reference 1e-3/0.2). This damps broadside swing — the hang mid-swing phase gap.
    scene.config.preview_aero = enable_aero;
    // Enable XPBD dihedral bending from the reference bend stiffness (opt-in via
    // --bend; off by default because its FD gradient is slow and negligible on
    // tension-dominated cloth).
    if enable_bend && bend > 0.0 {
        scene.config.bend_compliance = Some(1.0 / bend);
    }
    scene.objects.push(obj);

    let steps_per_frame = ((params.fps as f64).recip() / params.dt as f64)
        .round()
        .max(1.0) as u32;
    let ctx = maybe_gpu(solver, cpu);
    let (mut stepper, backend) = build_stepper(&scene, solver, &ctx, precond);

    eprintln!(
        "golden run: {} — {} verts, {} tris, {} pins, backend={backend}, steps/frame={steps_per_frame}",
        scene_dir.display(),
        init.len(),
        input.tris.len(),
        pins.len()
    );
    eprintln!(
        "CAVEATS: default material (reference per-object stiffness not mapped); HARD pins at \
         initial position (reference uses soft/kinematic FixPairs — pinned row sags ~0.04); \
         gate 8e-2/2e-2/6e-3. A failing gate here is an APPROXIMATE-setup divergence, not a \
         pure solver-fidelity measurement."
    );

    let mut cur = 0u32;
    let mut all_pass = true;
    let mut rows = Vec::new();
    for &f in &frames {
        while cur < f {
            for _ in 0..steps_per_frame {
                stepper.advance();
            }
            cur += 1;
        }
        let ours: Vec<[f32; 3]> = stepper
            .sim_state()
            .positions
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
        let reference = engine_io::load_checkpoint(&scene_dir, f)?;
        let g = engine_io::compare_frames(&ours, &reference)?;
        all_pass &= g.passes();
        println!(
            "frame {f:4}: max_rel={:.4} rms={:.4} centroid_rel={:.4} diag={:.3} pass={}",
            g.max_rel,
            g.rms,
            g.centroid_rel,
            g.diagonal,
            g.passes()
        );
        rows.push(serde_json::json!({
            "frame": f, "max_rel": g.max_rel, "rms": g.rms,
            "centroid_rel": g.centroid_rel, "pass": g.passes(),
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scene": scene_dir.file_name().and_then(|s| s.to_str()),
            "backend": backend, "all_pass": all_pass, "frames": rows,
        }))?
    );
    Ok(())
}

/// Run a scene with the implicit CPU solver and check the solver invariants.
/// Prints a JSON report to stdout; returns whether all invariants passed.
fn verify(path: PathBuf, steps: Option<u32>) -> Result<bool> {
    let scene = load_scene(&path)?;
    let total = steps.unwrap_or_else(|| {
        let spf = ((scene.config.fps.recip()) / scene.config.dt as f64)
            .round()
            .max(1.0) as u32;
        (scene.config.frames.max(1) as u32)
            .saturating_mul(spf)
            .min(2000)
    });
    let mut solver = scene.build_implicit();
    let linear = engine_core::linsolve::CgCpu::new();
    for _ in 0..total {
        solver.step(&linear);
    }
    let report = engine_core::check_invariants(&solver.metrics);
    let checks: Vec<_> = report
        .checks
        .iter()
        .map(|c| serde_json::json!({ "name": c.name, "ok": c.ok, "detail": c.detail }))
        .collect();
    let out = serde_json::json!({
        "scene": path.display().to_string(),
        "steps": total,
        "verts": solver.state.vertex_count(),
        "all_ok": report.all_ok(),
        "checks": checks,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(report.all_ok())
}

fn bench(
    path: PathBuf,
    steps: u32,
    solver: SolverKind,
    cpu: bool,
    precond: PrecondKind,
) -> Result<()> {
    let scene = load_scene(&path)?;
    let ctx = maybe_gpu(solver, cpu);
    let (mut stepper, backend) = build_stepper(&scene, solver, &ctx, precond);

    // Warm up (first solve builds pipelines / caches the solver).
    let warmup = 10.min(steps);
    for _ in 0..warmup {
        stepper.advance();
    }
    let start = std::time::Instant::now();
    for _ in 0..steps {
        stepper.advance();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let sps = steps as f64 / elapsed;
    let report = serde_json::json!({
        "backend": backend,
        "verts": stepper.sim_state().vertex_count(),
        "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_sec": sps,
        "ms_per_step": 1000.0 * elapsed / steps as f64,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Anything the frame loop can drive. Lets `xpbd` and `implicit` share one loop.
trait Steppable {
    fn advance(&mut self);
    fn sim_state(&self) -> &engine_core::SimState;
}

impl Steppable for engine_core::Solver {
    fn advance(&mut self) {
        self.step();
    }
    fn sim_state(&self) -> &engine_core::SimState {
        &self.state
    }
}

struct ImplicitRun<'a> {
    solver: engine_core::ImplicitSolver,
    linear: Box<dyn engine_core::LinearSolver + 'a>,
}

impl Steppable for ImplicitRun<'_> {
    fn advance(&mut self) {
        self.solver.step(&*self.linear);
    }
    fn sim_state(&self) -> &engine_core::SimState {
        &self.solver.state
    }
}

/// Create a GPU context when the implicit solver should use the GPU linear solve.
fn maybe_gpu(solver: SolverKind, cpu: bool) -> Option<engine_gpu::GpuContext> {
    if matches!(solver, SolverKind::Implicit) && !cpu {
        match engine_gpu::GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no GPU adapter; falling back to CPU CG");
                None
            }
        }
    } else {
        None
    }
}

/// Build the right [`Steppable`] for the chosen solver, borrowing `ctx` (created
/// by [`maybe_gpu`]) for the GPU linear solve. Returns the stepper + backend name.
fn build_stepper<'a>(
    scene: &Scene,
    solver: SolverKind,
    ctx: &'a Option<engine_gpu::GpuContext>,
    precond: PrecondKind,
) -> (Box<dyn Steppable + 'a>, String) {
    match solver {
        SolverKind::Xpbd => (Box::new(scene.build()), "xpbd".to_string()),
        SolverKind::Implicit => {
            let impl_solver = scene.build_implicit();
            let (linear, backend): (Box<dyn engine_core::LinearSolver + 'a>, String) = match ctx {
                Some(c) => {
                    let (gp, tag) = match precond {
                        PrecondKind::Block => (engine_gpu::GpuPrecond::Block, "block-jacobi"),
                        PrecondKind::SchwarzMl => (
                            engine_gpu::GpuPrecond::SchwarzMl {
                                domain: 16,
                                weight: 0.7,
                            },
                            "schwarz-ml",
                        ),
                    };
                    (
                        Box::new(engine_gpu::GpuLinearSolver::with_precond(c, gp)),
                        format!("implicit/gpu-pcg-{}({})", tag, c.backend_name()),
                    )
                }
                None => match precond {
                    PrecondKind::Block => (
                        Box::new(engine_core::linsolve::CgCpu::new()),
                        "implicit/cpu-block-jacobi".to_string(),
                    ),
                    PrecondKind::SchwarzMl => (
                        Box::new(engine_core::SchwarzMlCpu::new(16, 0.7)),
                        "implicit/cpu-schwarz-ml".to_string(),
                    ),
                },
            };
            (
                Box::new(ImplicitRun {
                    solver: impl_solver,
                    linear,
                }),
                backend,
            )
        }
    }
}

fn gpu_info() -> Result<()> {
    let ctx = engine_gpu::GpuContext::new().context("GPU bring-up")?;
    let probe = ctx.saxpy(2.0, &[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0])?;
    let info = ctx.adapter.get_info();
    let report = serde_json::json!({
        "backend": ctx.backend_name(),
        "adapter": info.name,
        "device_type": format!("{:?}", info.device_type),
        "noop_saxpy_ok": probe == vec![2.0, 4.0, 6.0],
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_scene(
    path: PathBuf,
    frames: Option<u32>,
    out: Option<PathBuf>,
    solver: SolverKind,
    cpu: bool,
    precond: PrecondKind,
) -> Result<()> {
    let scene = load_scene(&path)?;
    let total_frames = frames.unwrap_or(scene.config.frames.max(0) as u32);
    let steps_per_frame = ((scene.config.fps.recip()) / scene.config.dt as f64)
        .round()
        .max(1.0) as u32;

    if let Some(dir) = &out {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
    }

    // `ctx` must outlive the GPU linear solver that borrows it.
    let ctx = maybe_gpu(solver, cpu);
    let (mut stepper, backend) = build_stepper(&scene, solver, &ctx, precond);
    tracing::info!(total_frames, steps_per_frame, %backend, "simulation start");

    for frame in 0..total_frames {
        for _ in 0..steps_per_frame {
            stepper.advance();
        }
        let state = stepper.sim_state();
        if state.positions.iter().any(|p| !p.is_finite()) {
            anyhow::bail!("solver diverged at frame {frame} (non-finite positions)");
        }
        if let Some(dir) = &out {
            write_frame(dir, frame, state)?;
        }
        tracing::debug!(frame, ke = state.kinetic_energy(), "frame done");
    }

    let state = stepper.sim_state();
    let bbox = bounds(&state.positions);
    let report = serde_json::json!({
        "backend": backend,
        "frames": total_frames,
        "verts": state.vertex_count(),
        "final_time": state.time,
        "final_kinetic_energy": state.kinetic_energy(),
        "bbox_min": bbox.0,
        "bbox_max": bbox.1,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn load_scene(path: &std::path::Path) -> Result<Scene> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("json") => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading scene {}", path.display()))?;
            serde_json::from_str(&text).context("parsing scene JSON")
        }
        Some("obj") | Some("stl") | Some("gltf") | Some("glb") | Some("usda") | Some("dae")
        | Some("fbx") => {
            // Treat a bare mesh as a cloth pinned at its two first vertices, so
            // `mguso run mesh.{obj,stl,gltf,glb,usda,dae,fbx}` does something out of the box.
            let mut obj = engine_io::load_mesh(path)?;
            obj.pins = vec![0, 1];
            let mut scene = Scene::default();
            scene.config.floor_y = Some(-2.0);
            scene.config.damping = 0.02; // let the demo drape settle
            scene.objects.push(obj);
            Ok(scene)
        }
        _ => anyhow::bail!(
            "unsupported scene extension (want .json, .obj, .stl, .gltf, .glb, .usda, .dae, or .fbx): {}",
            path.display()
        ),
    }
}

fn write_frame(dir: &std::path::Path, frame: u32, state: &engine_core::SimState) -> Result<()> {
    let obj = engine_api::Object {
        name: format!("frame_{frame:04}"),
        positions: state.positions.iter().map(|p| [p.x, p.y, p.z]).collect(),
        tris: state.tris.clone(),
        material: Default::default(),
        preset: None,
        pins: Vec::new(),
        is_static: false,
        normals: Vec::new(),
        uvs: Vec::new(),
        skin: None,
        animations: Vec::new(),
    };
    engine_io::save_obj(&obj, dir.join(format!("frame_{frame:04}.obj")))
}

fn bounds(positions: &[glam::Vec3]) -> ([f32; 3], [f32; 3]) {
    let mut lo = glam::Vec3::splat(f32::INFINITY);
    let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    (lo.into(), hi.into())
}
