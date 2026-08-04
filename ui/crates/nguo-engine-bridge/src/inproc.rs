// SPDX-License-Identifier: GPL-2.0-or-later
//! The in-process transport: call `engine-api` directly, on this thread.
//!
//! # Why this exists as well as [`ProcessTransport`]
//!
//! Not for speed, and not because the wgpu pin finally unified (it has — both
//! lockfiles resolve `wgpu 27.0.1`). For a reason the process contract cannot be
//! argued out of: **`engine_api::Scene` has no stitch field**
//! (`engine-api/src/scene.rs:71-78`). Stitches are attached to a built solver by
//! `ImplicitSolver::set_stitches`, which no JSON document reaches. So `mguso run
//! scene.json` can drape a garment but it cannot *sew* one, and sewing is the
//! whole of milestone M6. [`ProcessTransport`] therefore rejects a submission
//! that carries stitches instead of quietly draping it unsewn.
//!
//! # Why the CPU CG and not the GPU PCG
//!
//! `engine_gpu::GpuContext::new()` builds its **own** device. The app already
//! shares exactly one `wgpu::Device` between four renderers
//! (`ui/docs/DEVELOPING.md` §4.1) and a second one would double the VRAM and
//! break the one-device invariant `nguo_gpu::adopt` asserts. Joining the shared
//! device needs `GpuContext::from_device`, which exists — but the solve runs on
//! tokio's blocking pool while iced owns the queue, so the two would be
//! submitting to one queue from two threads.
//!
// ponytail: CPU conjugate gradients. The upgrade path is
// `GpuContext::from_device` plus a queue the solve is allowed to touch, and it is
// worth taking only once the linear solve is the measured bottleneck — see
// `Summary::report`'s `cg_*` fields, which is where that number comes from.

use std::time::Instant;

use engine_api::Scene;
use engine_core::implicit::Stitch as CoreStitch;

use crate::transport::{BridgeError, Flow, FrameSink, Submission, Summary, Transport};

/// Saturation cap as a multiple of the rest length.
///
/// The reference's `stitch_length_factor`, whose default is 10
/// (`engine-cli/src/gauge_driver.rs:1154-1156`). Past `l_cap` the pull grows
/// linearly instead of with the length ratio, which is exactly the regime a seam
/// that starts centimetres open lives in.
const STITCH_LENGTH_FACTOR: f32 = 10.0;

/// Stitch stiffness. The reference applies it **raw** — no mass and no
/// time-scale normalisation (`engine_core::stitch`) — and its own scenes ship
/// `1.0` when `stitch_stiffness.bin` is absent.
const STITCH_STIFFNESS: f32 = 1.0;

/// Runs drapes by building an `engine_api::Scene` in this process.
///
/// Both tiers run the **implicit** solver here, deliberately: XPBD
/// (`Scene::build`) has no `set_stitches` at all, so it cannot sew
/// (`engine-core/src/solver.rs` — `set_colliders` and `step`, and nothing else).
/// The tier still decides everything outside this file: whether the bridge drops
/// frames for a slow consumer (`bridge::ChannelSink`) and what the caller puts in
/// the scene's `self-collision`.
#[derive(Debug, Default)]
pub struct InProcessTransport {
    /// Emit a frame only every `n`th step. `1` (the default) streams every step,
    /// which is what an interactive preview wants.
    pub frame_stride: u32,
}

impl InProcessTransport {
    pub fn new() -> Self {
        Self { frame_stride: 1 }
    }
}

impl Transport for InProcessTransport {
    fn solve(&self, sub: &Submission, sink: &mut dyn FrameSink) -> Result<Summary, BridgeError> {
        let started = Instant::now();
        let scene: Scene = serde_json::from_value(sub.scene.clone())?;
        check_config_keys(&sub.scene, &scene)?;

        // Steps, not animation frames: the loop below emits one update per
        // solver step (or per `frame_stride` of them), because a drape is watched
        // as it settles rather than played back at a frame rate. `config.fps` is
        // therefore not consulted, which is a real difference from
        // `mguso run` and the reason it is spelled out here.
        let steps = sub.frames.unwrap_or(scene.config.frames.max(1) as u32);
        let stride = self.frame_stride.max(1);

        let mut solver = scene.build_implicit();
        let vertices = solver.state.vertex_count();
        let dynamic = (0..vertices)
            .filter(|&i| solver.state.inv_mass[i] > 0.0)
            .count();
        if dynamic == 0 {
            return Err(BridgeError::Unsupported(
                "every vertex in the scene is static, so there is nothing to drape — is the \
                 garment object marked `is_static`?"
                    .into(),
            ));
        }

        if !sub.stitches.is_empty() {
            let l0 = solver.params.contact_dhat.max(1.0e-4);
            let stitches: Vec<CoreStitch> = sub
                .stitches
                .iter()
                .map(|s| CoreStitch {
                    index: s.index,
                    weight: s.weight,
                    stiffness: STITCH_STIFFNESS,
                    // Uniform contact gap, so the reference's
                    // `(Σ wᵢ·ghatᵢ)/2` is just the gap: the six weights sum to 2.
                    l0,
                    l_cap: STITCH_LENGTH_FACTOR * l0,
                })
                .collect();
            // Rejects an out-of-range index or a half whose weights do not sum
            // to 1 — a caller mistake, and one worth naming rather than solving
            // a silently different energy.
            solver
                .set_stitches(stitches)
                .map_err(BridgeError::Unsupported)?;
        }

        let linear = engine_core::linsolve::CgCpu::new();
        tracing::info!(
            steps,
            stride,
            vertices,
            dynamic,
            static_verts = vertices - dynamic,
            triangles = solver.state.tris.len(),
            stitches = sub.stitches.len(),
            self_collision = scene.config.self_collision,
            dt = solver.params.dt,
            tier = ?sub.tier,
            "in-process drape starting"
        );

        let mut emitted = 0u32;
        let mut step_secs: Vec<f64> = Vec::with_capacity(steps as usize);
        for step in 0..steps {
            if sink.is_cancelled() {
                return Err(BridgeError::Cancelled);
            }
            let t = Instant::now();
            solver.step(&linear);
            step_secs.push(t.elapsed().as_secs_f64());

            if let Some(vertex) = solver.state.positions.iter().position(|p| !p.is_finite()) {
                return Err(BridgeError::Diverged {
                    frame: step,
                    vertex,
                });
            }
            // Always publish the last step, whatever the stride, or a settled
            // drape ends on a stale frame.
            if step % stride != 0 && step + 1 != steps {
                continue;
            }
            let positions = solver
                .state
                .positions
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect();
            if sink.push(emitted, positions) == Flow::Cancel {
                return Err(BridgeError::Cancelled);
            }
            emitted += 1;
        }

        let median = median(&mut step_secs);
        let wall = started.elapsed().as_secs_f64();
        tracing::info!(
            emitted,
            wall_s = wall,
            median_step_s = median,
            ke = solver.state.kinetic_energy(),
            "in-process drape done"
        );

        Ok(Summary {
            tier: sub.tier,
            frames_emitted: emitted,
            // The engine's own numbers plus the wall clock this loop measured.
            // Reported, not logged, so the shell can show `s/step` live and a
            // test can assert on it (`AGENT_BRIEF`: measure, never estimate).
            report: serde_json::json!({
                "backend": "in-process/implicit/cpu-block-jacobi",
                "steps": steps,
                "frames_emitted": emitted,
                "verts": vertices,
                "dynamic_verts": dynamic,
                "tris": solver.state.tris.len(),
                "stitches": sub.stitches.len(),
                "self_collision": scene.config.self_collision,
                "median_step_s": median,
                "total_s": wall,
                "final_time": solver.state.time,
                "final_kinetic_energy": solver.state.kinetic_energy(),
            }),
        })
    }
}

/// Reject a `config` key `SolverConfig` does not have.
///
/// `SolverConfig` is `#[serde(default, rename_all = "kebab-case")]`, so serde
/// **silently ignores** an unknown key and leaves the default in place. A caller
/// who writes `self_collision` instead of `self-collision` therefore gets a scene
/// with self-collision *off* and no error at all — which is how a garment ends up
/// draped through the body it was supposed to collide with, reported as success.
///
/// The valid key set is taken from the round trip of the parsed config, so it can
/// never drift from the struct.
fn check_config_keys(raw: &serde_json::Value, scene: &Scene) -> Result<(), BridgeError> {
    let Some(given) = raw.get("config").and_then(|c| c.as_object()) else {
        return Ok(());
    };
    let known = serde_json::to_value(scene.config)?;
    let known = known
        .as_object()
        .expect("SolverConfig serialises to an object");
    for key in given.keys() {
        if known.contains_key(key) {
            continue;
        }
        // The overwhelmingly likely mistake is snake_case for kebab-case, so say
        // so instead of just listing 30 field names.
        let kebab = key.replace('_', "-");
        let hint = if known.contains_key(&kebab) {
            format!(" — did you mean `{kebab}`?")
        } else {
            let mut names: Vec<&str> = known.keys().map(String::as_str).collect();
            names.sort_unstable();
            format!(" — known keys: {}", names.join(", "))
        };
        return Err(BridgeError::Unsupported(format!(
            "scene config has no key `{key}`, so it would be silently ignored{hint}",
        )));
    }
    Ok(())
}

/// Median rather than mean: one 300 ms hitch from a page fault should not be
/// what a per-step cost is reported as.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{Stitch, Tier};

    /// Collects every frame; never cancels.
    struct Collect(Vec<(u32, Vec<[f32; 3]>)>);

    impl FrameSink for Collect {
        fn push(&mut self, index: u32, positions: Vec<[f32; 3]>) -> Flow {
            self.0.push((index, positions));
            Flow::Continue
        }
        fn is_cancelled(&self) -> bool {
            false
        }
    }

    /// A `w x w` grid of quads in the y = `y` plane, `size` metres across,
    /// centred on the origin. Returns the scene JSON and the vertex count.
    fn sheet(cells: usize, size: f32, y: f32, pinned: bool) -> (serde_json::Value, usize) {
        let n = cells + 1;
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let (u, v) = (i as f32 / cells as f32, j as f32 / cells as f32);
                positions.push([(u - 0.5) * size, y, (v - 0.5) * size]);
                uvs.push([u, v]);
            }
        }
        let mut tris = Vec::new();
        for j in 0..cells {
            for i in 0..cells {
                let (a, b) = ((j * n + i) as u32, (j * n + i + 1) as u32);
                let (c, d) = (((j + 1) * n + i) as u32, ((j + 1) * n + i + 1) as u32);
                tris.push([a, c, b]);
                tris.push([b, c, d]);
            }
        }
        let pins: Vec<u32> = if pinned {
            vec![0, cells as u32]
        } else {
            vec![]
        };
        let scene = serde_json::json!({
            "config": { "dt": 1.0 / 120.0, "frames": 4, "self-collision": false },
            "objects": [{
                "name": "sheet",
                "positions": positions,
                "tris": tris,
                "uvs": uvs,
                "pins": pins,
                "preset": "cotton",
            }],
        });
        (scene, n * n)
    }

    fn submission(scene: serde_json::Value, stitches: Vec<Stitch>, frames: u32) -> Submission {
        Submission {
            scene,
            stitches,
            tier: Tier::Preview,
            frames: Some(frames),
            out_dir: None,
        }
    }

    /// The baseline: a free sheet falls, every frame is finite, and the report
    /// carries a real per-step cost rather than a zero.
    #[test]
    fn a_free_sheet_falls_and_every_step_is_reported() {
        let (scene, verts) = sheet(6, 0.4, 1.0, false);
        let mut sink = Collect(Vec::new());
        let summary = InProcessTransport::new()
            .solve(&submission(scene, Vec::new(), 6), &mut sink)
            .expect("a plain sheet must solve");

        assert_eq!(summary.frames_emitted, 6);
        assert_eq!(sink.0.len(), 6);
        assert_eq!(
            sink.0.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5],
            "frames must be indexed contiguously from zero",
        );
        for (index, positions) in &sink.0 {
            assert_eq!(positions.len(), verts, "frame {index} is the wrong length");
            assert!(positions.iter().flatten().all(|c| c.is_finite()));
        }
        // Gravity is -Y, so it must be lower at the end than at the start, by
        // more than fp noise.
        let first = sink.0.first().expect("a frame").1[0][1];
        let last = sink.0.last().expect("a frame").1[0][1];
        assert!(last < first - 1e-4, "{first} -> {last}: nothing fell");

        let step = summary.report["median_step_s"].as_f64().expect("measured");
        assert!(step > 0.0 && step < 5.0, "median step {step} s");
        assert_eq!(summary.report["stitches"], 0);
        assert_eq!(summary.report["dynamic_verts"], verts);
    }

    /// The point of the whole transport: a stitch closes a gap the process
    /// transport cannot even carry. Two pinned sheets a decimetre apart, sewn
    /// along their facing edges — the gap must **shrink**, and the unsewn run of
    /// the identical scene must leave it alone.
    #[test]
    fn stitches_pull_two_sheets_together_and_their_absence_does_not() {
        const CELLS: usize = 4;
        let n = (CELLS + 1) as u32;
        let (mut scene, verts) = sheet(CELLS, 0.3, 1.0, true);
        // A second sheet, offset in +X by 0.4 m: its low-x edge faces the first
        // sheet's high-x edge.
        let mut second = scene["objects"][0].clone();
        second["name"] = "sheet b".into();
        for p in second["positions"].as_array_mut().expect("positions") {
            let x = p[0].as_f64().expect("x") + 0.4;
            p[0] = serde_json::json!(x);
        }
        scene["objects"]
            .as_array_mut()
            .expect("objects")
            .push(second);

        // Sew row j of sheet A's high-x column to row j of sheet B's low-x
        // column. Single-vertex halves, which is what all 851 of the reference's
        // `fitting` stitches are.
        let stitches: Vec<Stitch> = (0..n)
            .map(|j| {
                let a = j * n + (n - 1);
                let b = verts as u32 + j * n;
                Stitch {
                    index: [a, a, a, b, b, b],
                    weight: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                }
            })
            .collect();

        let gap = |frame: &[[f32; 3]]| -> f32 {
            (0..n)
                .map(|j| {
                    let a = frame[(j * n + (n - 1)) as usize];
                    let b = frame[(verts as u32 + j * n) as usize];
                    ((0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>()).sqrt()
                })
                .sum::<f32>()
                / n as f32
        };

        let mut sewn = Collect(Vec::new());
        let summary = InProcessTransport::new()
            .solve(&submission(scene.clone(), stitches.clone(), 40), &mut sewn)
            .expect("the sewn scene must solve");
        assert_eq!(summary.report["stitches"], stitches.len());

        let mut loose = Collect(Vec::new());
        InProcessTransport::new()
            .solve(&submission(scene, Vec::new(), 40), &mut loose)
            .expect("the unsewn scene must solve");

        let start = gap(&sewn.0.first().expect("a frame").1);
        let closed = gap(&sewn.0.last().expect("a frame").1);
        let unsewn = gap(&loose.0.last().expect("a frame").1);

        // Floors, so this cannot pass by both runs doing nothing: the sheets
        // really do start 0.1 m apart, and the sewn one really does close.
        assert!(
            start > 0.05,
            "the two sheets did not start apart: {start} m"
        );
        assert!(
            closed < start * 0.7,
            "the stitches did not close the seam: {start} -> {closed} m",
        );
        assert!(
            unsewn > start * 0.9,
            "the UNSEWN run closed the seam too, so the stitches proved nothing: \
             {start} -> {unsewn} m",
        );
    }

    /// The stride must not silently drop the settled frame — that is the one the
    /// user is looking at.
    #[test]
    fn a_stride_thins_the_stream_but_always_publishes_the_last_step() {
        let (scene, _) = sheet(4, 0.3, 1.0, true);
        let mut sink = Collect(Vec::new());
        let transport = InProcessTransport { frame_stride: 4 };
        let summary = transport
            .solve(&submission(scene, Vec::new(), 10), &mut sink)
            .expect("solves");

        // Steps 0, 4, 8 by stride, plus step 9 because it is the last.
        assert_eq!(summary.frames_emitted, 4, "{:?}", sink.0.len());
        assert_eq!(summary.report["steps"], 10);
    }

    /// A scene where everything is pinned is a caller mistake with an obvious
    /// cause, so it must say so rather than emit ten identical frames.
    #[test]
    fn an_all_static_scene_is_a_named_error_not_a_silent_no_op() {
        let (mut scene, _) = sheet(3, 0.3, 1.0, false);
        scene["objects"][0]["is_static"] = serde_json::json!(true);
        let mut sink = Collect(Vec::new());
        let err = InProcessTransport::new()
            .solve(&submission(scene, Vec::new(), 3), &mut sink)
            .expect_err("an all-static scene cannot drape");
        assert!(matches!(err, BridgeError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("is_static"), "{err}");
        assert!(sink.0.is_empty(), "it emitted frames anyway");
    }

    /// The whole point of `check_config_keys`: `self_collision` is not a typo
    /// serde forgives, it is a key that does not exist, and accepting it means
    /// draping with self-collision off while believing it is on.
    #[test]
    fn a_snake_case_config_key_is_rejected_with_the_kebab_case_spelling() {
        let (mut scene, _) = sheet(3, 0.3, 1.0, true);
        scene["config"]["self_collision"] = serde_json::json!(true);
        let mut sink = Collect(Vec::new());
        let err = InProcessTransport::new()
            .solve(&submission(scene, Vec::new(), 2), &mut sink)
            .expect_err("an unknown config key must be refused");
        assert!(err.to_string().contains("self-collision"), "{err}");
        assert!(sink.0.is_empty());

        // A key that is not merely mis-cased lists what does exist.
        let (mut scene, _) = sheet(3, 0.3, 1.0, true);
        scene["config"]["gravty"] = serde_json::json!([0, -9.8, 0]);
        let err = InProcessTransport::new()
            .solve(&submission(scene, Vec::new(), 2), &mut Collect(Vec::new()))
            .expect_err("refused");
        assert!(err.to_string().contains("known keys"), "{err}");
        assert!(err.to_string().contains("gravity"), "{err}");

        // ...and the correct spelling still goes through.
        let (mut scene, _) = sheet(3, 0.3, 1.0, true);
        scene["config"]["self-collision"] = serde_json::json!(true);
        InProcessTransport::new()
            .solve(&submission(scene, Vec::new(), 2), &mut Collect(Vec::new()))
            .expect("the kebab-case key is accepted");
    }

    /// A bad stitch must be rejected by name, not accepted into a silently
    /// different energy (the weights-sum-to-1 contract in `set_stitches`).
    #[test]
    fn a_malformed_stitch_is_rejected_with_the_reason() {
        let (scene, _) = sheet(3, 0.3, 1.0, true);
        for (bad, expect) in [
            (
                Stitch {
                    index: [0, 0, 0, 1, 1, 1],
                    weight: [0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
                },
                "sum to",
            ),
            (
                Stitch {
                    index: [0, 0, 0, 9_999, 9_999, 9_999],
                    weight: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                },
                "references a vertex",
            ),
        ] {
            let mut sink = Collect(Vec::new());
            let err = InProcessTransport::new()
                .solve(&submission(scene.clone(), vec![bad], 2), &mut sink)
                .expect_err("a malformed stitch must not be attached");
            assert!(err.to_string().contains(expect), "{err}");
            assert!(sink.0.is_empty());
        }
    }
}
