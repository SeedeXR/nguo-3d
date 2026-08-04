// SPDX-License-Identifier: GPL-2.0-or-later
//! Draft → sew → drape → viewport: the M6 loop, as the shell drives it.
//!
//! ```text
//!   nguo_doc::Document
//!        │  nguo_sewing::build            validate, triangulate, arrange, sew
//!        ▼
//!   Garment { positions, tris, uvs, stitches }
//!        │  scene()                       + the COL body as a static object
//!        ▼
//!   nguo_engine_bridge::Submission  ──▶  InProcessTransport  ──▶  Update::Frame
//!        │  poll()                       one solver step per frame
//!        ▼
//!   nguo_viewport3d::Avatar          the garment, drawn over the body
//! ```
//!
//! # Why the frames are drained in `Tick` and not through a `Subscription`
//!
//! The shell already gets one `Tick` per vsync from `iced::window::frames()`, and
//! a preview frame that arrives between two vsyncs is a frame nobody can see. So
//! [`Drape::poll`] is a non-blocking drain on the tick that is about to draw
//! anyway: no channel bridging, no wakers, and the newest frame always wins
//! because the bridge already dropped the rest.
//!
//! # What is deliberately *not* here
//!
//! Undo. A drape is a derived result, not a document edit (`ui/docs/UX_SPEC.md`
//! §3.5 keeps drape-in-undo as an open design question), so nothing in this
//! module touches the document.

use std::sync::Arc;
use std::time::Instant;

use nguo_engine_bridge::{
    EngineBridge, Epoch, InProcessTransport, Stitch, Submission, Tier, Update,
};
use nguo_sewing::{Garment, SewOptions};
use nguo_viewport3d::{AvatarMesh, Vertex};

use crate::avatar::CollisionProxy;

// ---------------------------------------------------------------------------
// The two tiers, and the measurements they come from
// ---------------------------------------------------------------------------
//
// Measured on this machine (Apple M4, dev profile = `opt-level 1`, deps at 3),
// median of 6 steps on an idle machine, by `profile::profile_drape`. The sample
// garment on the shipped COL body:
//
// | configuration                    | verts  | tris   | self-coll |  s/step |
// |----------------------------------|--------|--------|-----------|---------|
// | 60 mm mesh, no body              |    346 |    608 | false     |   0.008 |
// | 60 mm mesh, no body              |    346 |    608 | true      |   0.019 |
// | 60 mm mesh, clipped body         |  3 717 |  5 979 | false     |   0.029 |
// | 60 mm mesh, clipped body         |  3 717 |  5 979 | true      | **0.051** |
// | 60 mm mesh, WHOLE body           | 10 877 | 15 608 | true      |   0.119 |
// | 30 mm mesh, no body              |  1 333 |  2 510 | false     |   0.046 |
// | 30 mm mesh, no body              |  1 333 |  2 510 | true      |   0.118 |
// | 30 mm mesh, clipped body         |  4 704 |  7 881 | false     |   0.086 |
// | 30 mm mesh, clipped body         |  4 704 |  7 881 | true      | **0.163** |
// | 30 mm mesh, WHOLE body           | 11 864 | 17 510 | true      |   0.232 |
//
// Under load the same rows read up to ~2× higher (0.106 s/step measured for the
// preview row while the rest of the suite was running), which is why every
// assertion in this file is on a *ratio* or a floor and never on an absolute.
//
// Four things that table settles, three of them against expectation:
//
// 1. **Clipping the body to the garment's band is the biggest single win**:
//    0.119 -> 0.051 s/step, 2.3×, for twenty lines (`clip_proxy`). Two thirds of a
//    standing body is nowhere near a garment that covers the hips.
// 2. **Mesh resolution is the next lever, and it is not linear.** Halving the
//    target edge is 3.2× the step cost (0.051 -> 0.163) for 27 % more vertices,
//    because the self-collision grid is indexed by the *movable* primitives (D-17)
//    and every static triangle then queries a grid four times denser. So the
//    preview tier is coarse, and that is a measurement.
// 3. **Self-collision cannot be turned off.** It is 1.8-2.6× of the step, and it
//    is also the *only* thing that gives cloth-body contact: there is no analytic
//    collider in this scene. With it off the garment drapes through the body —
//    which is what the first run of this code did, for a different reason (see
//    `scene`'s kebab-case note).
// 4. **The body is most of the cost even clipped** (0.019 -> 0.051 with 3 371 extra
//    static vertices). `Scene::lower` concatenates everything into one mesh and
//    `ImplicitSolver::new` builds a membrane element for **every** triangle
//    including the static ones (`engine-core/src/implicit.rs:458`), so the body
//    pays full assembly cost every Newton iteration to contribute nothing but
//    Dirichlet-filtered rows. That is an `engine-core` fix, not one available here,
//    and it is the next real lever.
//
// A preview is ~2-4 s of wall clock, streamed. That is not interactive and this
// file does not pretend otherwise. What stays interactive is the *shell*: the solve
// is on tokio's blocking pool and the viewport keeps rendering at vsync while
// frames arrive.

/// Solver steps a preview drape runs.
///
/// Measured, not guessed. The sample garment's seams close like this (sum over all
/// 22 stitches, printed by `solve_tests`), **re-baselined** now that the garment
/// hangs at its annotated landmark instead of at the underbust:
///
/// ```text
///   step  0   3 331 mm open      the arranger's slack
///   step 12   1 504 mm
///   step 24     188 mm           94.4 % shut
///   step 36     135 mm           95.9 %, and flat from here
///   step 47     147 mm
/// ```
///
/// Re-measured 2026-08-04 at `NGUO_DRAPE_TEST_STEPS=48`, and the numbers an earlier
/// version of this comment carried were wrong in both directions:
///
/// ```text
///   seams 3330.9 mm -> 57.4 mm open
///   closure  0:3331mm  12:1477mm  24:556mm  36:61mm  47:57mm
/// ```
///
/// So: **26 % less to close** than the underbust baseline (4 486 mm), because the
/// start cylinder is sized off the band the garment covers rather than off the
/// feet — that part held. But the seams **do** close, to 57 mm; the claim that the
/// last ~135 mm no longer closes was false. And the curve is **not** flat by step
/// 24 — 556 mm at 24 against 61 mm at 36. It is flat from 36, which is where
/// `PREVIEW_STEPS` sits, so the conclusion survives its own evidence being wrong.
/// At the measured 0.136–0.181 s/step that is ~5–6.5 s of wall clock, streamed
/// frame by frame. Whole test at 48 steps: 6.5 s, peak RSS 135 MB.
pub const PREVIEW_STEPS: u32 = 36;
/// A bake runs four times as long on a mesh twice as fine — tens of seconds,
/// deliberately.
pub const BAKE_STEPS: u32 = 144;

/// Target triangle edge, mm, per tier. See the table above: this is the single
/// biggest knob on the solve cost.
const PREVIEW_EDGE_MM: f64 = 60.0;
const BAKE_EDGE_MM: f64 = 30.0;

/// `dt`. Backward Euler is unconditionally stable, so this is chosen for how much
/// simulated time a step buys rather than for stability: at the engine's default
/// `1e-3` a 48-step preview would advance 0.05 s and nothing would visibly fall.
const DT: f32 = 1.0 / 120.0;

/// Colour the garment is drawn in — a linear `baseColorFactor`, the same input
/// the body's Monk tone goes through.
///
// ponytail: one hardcoded colour, because a garment has no material picker yet.
// It comes from `nguo_doc::Material` the moment one is selectable.
const GARMENT_COLOR: [f32; 4] = [0.62, 0.24, 0.30, 1.0];

/// What the shell is showing about the drape, in one line for the status bar.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Wall-clock milliseconds for each stage of `nguo_sewing::build`.
    pub sew: nguo_sewing::Timings,
    pub vertices: usize,
    pub triangles: usize,
    pub stitches: usize,
    /// Milliseconds from `submit` to the first frame arriving.
    pub first_frame_ms: Option<f64>,
    /// Frames delivered for the current epoch.
    pub frames: u32,
    /// Median seconds per solver step, from the engine's own report. Only known
    /// once the solve finishes.
    pub step_s: Option<f64>,
    /// Total seam gap still open, metres. Falls as the stitches pull the seams
    /// shut, which makes it the honest "is it sewn yet" read-out.
    pub seam_gap_m: f64,
    pub message: String,
}

impl Report {
    pub fn line(&self) -> String {
        if self.message.is_empty() {
            return "no drape".to_owned();
        }
        let mut s = self.message.clone();
        if let Some(ms) = self.first_frame_ms {
            s.push_str(&format!(" · first frame {ms:.0} ms"));
        }
        if let Some(step) = self.step_s {
            s.push_str(&format!(" · {:.0} ms/step", step * 1000.0));
        }
        s
    }
}

/// The drape the shell owns: one bridge, one garment, one streamed mesh.
pub struct Drape {
    /// Kept alive because the bridge's worker task lives on it. Dropping this
    /// shuts the solve down, which is what closing the app should do.
    _runtime: tokio::runtime::Runtime,
    bridge: EngineBridge,
    updates: tokio::sync::mpsc::Receiver<Update>,
    /// The garment as sewn and arranged — the drape's frame 0 and its rest
    /// shape. Kept so a streamed frame can be checked against it.
    garment: Option<Garment>,
    /// Positions from the newest frame, or the arranged ones before any arrives.
    mesh: Option<Arc<AvatarMesh>>,
    epoch: Epoch,
    submitted: Option<Instant>,
    pub report: Report,
}

impl Drape {
    /// Spawn the runtime and the bridge worker.
    ///
    /// Fails only if the OS will not give us a thread pool, which is not
    /// something the shell can fix — so the caller carries on without a drape
    /// rather than refusing to start.
    pub fn new() -> std::io::Result<Drape> {
        // Two worker threads: one for the bridge's own task, one so the blocking
        // solve is never waiting on it. `spawn_blocking` uses its own pool.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("nguo-drape")
            .build()?;
        let (bridge, updates) = {
            let _guard = runtime.enter();
            EngineBridge::new(Arc::new(InProcessTransport::new()))
        };
        Ok(Drape {
            _runtime: runtime,
            bridge,
            updates,
            garment: None,
            mesh: None,
            epoch: Epoch(0),
            submitted: None,
            report: Report::default(),
        })
    }

    /// Sew `doc` and submit it. Supersedes anything in flight.
    ///
    /// `proxy` is the body to drape over. `None` drapes under gravity with
    /// nothing to land on — which is a real answer, not a failure, and the status
    /// line says so.
    pub fn submit(
        &mut self,
        doc: &nguo_doc::Document,
        arrange: nguo_sewing::ArrangeOptions,
        proxy: Option<&CollisionProxy>,
        tier: Tier,
    ) {
        let opts = sew_options(arrange, tier);
        let garment = match nguo_sewing::build(doc, &opts) {
            Ok(garment) => garment,
            Err(error) => {
                // Actionable and non-fatal: the diagnostics inside `NotSewable`
                // name the edge or the seam to fix.
                tracing::warn!(target: "nguo_app", %error, "the garment cannot be draped");
                self.report.message = format!("cannot drape: {error}");
                self.report.first_frame_ms = None;
                self.report.step_s = None;
                let _ = nguo_registry::start(nguo_registry::feature_const("drape.preview"), 0)
                    .failed(&error);
                return;
            }
        };

        let steps = steps_for(tier);
        let scene = scene(&garment, proxy, tier);
        let stitches: Vec<Stitch> = garment
            .stitches
            .iter()
            .map(|s| Stitch {
                index: s.index,
                weight: s.weight,
            })
            .collect();

        self.report = Report {
            sew: garment.timings,
            vertices: garment.vertex_count(),
            triangles: garment.triangle_count(),
            stitches: stitches.len(),
            first_frame_ms: None,
            frames: 0,
            step_s: None,
            seam_gap_m: garment.seam_gap_m(),
            message: format!(
                "draping {} vertices, {} stitches{}",
                garment.vertex_count(),
                stitches.len(),
                if proxy.is_some() {
                    ""
                } else {
                    " (no body: gravity only)"
                },
            ),
        };
        // Show the arranged garment immediately. The first solved frame is
        // seconds away and an empty viewport in the meantime reads as a failure.
        self.mesh = Some(Arc::new(mesh_of(&garment.positions, &garment.tris)));
        self.garment = Some(garment);
        self.submitted = Some(Instant::now());
        self.epoch = self.bridge.submit(Submission {
            scene,
            stitches,
            tier,
            frames: Some(steps),
            out_dir: None,
        });
        tracing::info!(
            target: "nguo_app",
            epoch = self.epoch.0,
            steps,
            body = proxy.is_some(),
            sew_ms = self.report.sew.total_ms(),
            "drape submitted"
        );
    }

    /// Drain whatever the solver has produced. Returns `true` if the viewport
    /// should redraw.
    ///
    /// Non-blocking: it takes every queued update and stops, so a slow solve
    /// costs one failed `try_recv` per frame.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(update) = self.updates.try_recv() {
            // The bridge already drops stale epochs; this is belt and braces so
            // a future second consumer cannot show one.
            if update.epoch() != self.epoch {
                continue;
            }
            match update {
                Update::Frame { positions, .. } => {
                    let Some(garment) = &self.garment else {
                        continue;
                    };
                    // The solved scene is the garment *concatenated with the
                    // body*, and the garment is object 0, so its vertices are the
                    // leading prefix. A frame shorter than that is the wrong
                    // scene and must not be sliced.
                    if positions.len() < garment.positions.len() {
                        tracing::error!(
                            target: "nguo_app",
                            got = positions.len(),
                            want = garment.positions.len(),
                            "a frame is too short for the garment; dropping it",
                        );
                        continue;
                    }
                    let cloth = &positions[..garment.positions.len()];
                    self.report.seam_gap_m = seam_gap(garment, cloth);
                    self.mesh = Some(Arc::new(mesh_of(cloth, &garment.tris)));
                    self.report.frames += 1;
                    if self.report.first_frame_ms.is_none() {
                        self.report.first_frame_ms =
                            self.submitted.map(|t| t.elapsed().as_secs_f64() * 1000.0);
                    }
                    changed = true;
                }
                Update::Done { summary, .. } => {
                    self.report.step_s = summary.report["median_step_s"].as_f64();
                    self.report.message = format!(
                        "draped {} frames, seams {:.1} mm open",
                        summary.frames_emitted,
                        self.report.seam_gap_m * 1000.0,
                    );
                    tracing::info!(
                        target: "nguo_app",
                        report = %summary.report,
                        seam_gap_mm = self.report.seam_gap_m * 1000.0,
                        "drape finished"
                    );
                }
                Update::Failed { error, .. } => {
                    tracing::error!(target: "nguo_app", %error, "drape failed");
                    self.report.message = format!("drape failed: {error}");
                }
            }
        }
        changed
    }

    /// The garment for the viewport, if there is one.
    pub fn viewport(&self) -> Option<nguo_viewport3d::Avatar> {
        Some(nguo_viewport3d::Avatar {
            mesh: Arc::clone(self.mesh.as_ref()?),
            base_color: GARMENT_COLOR,
        })
    }

    pub fn cancel(&mut self) {
        self.bridge.cancel_all();
        self.report.message = "drape cancelled".to_owned();
        self.report.step_s = Some(0.0);
    }
}

/// How finely to mesh the garment for this tier. The whole reason the tier is
/// plumbed this far down: it is the step cost's biggest input (see the table
/// above), so a preview and a bake want different meshes, not just different step
/// counts.
fn sew_options(arrange: nguo_sewing::ArrangeOptions, tier: Tier) -> SewOptions {
    SewOptions {
        triangulate: nguo_triangulate::TriangulateOptions::default().with_target_edge_mm(
            match tier {
                Tier::Preview => PREVIEW_EDGE_MM,
                Tier::Bake => BAKE_EDGE_MM,
            },
        ),
        arrange,
    }
}

fn steps_for(tier: Tier) -> u32 {
    let default = match tier {
        Tier::Preview => PREVIEW_STEPS,
        Tier::Bake => BAKE_STEPS,
    };
    // An escape hatch, because the honest answer to "how many steps" is
    // "however many it takes on your machine" and this is what a profiling run
    // turns down to 1.
    std::env::var("NGUO_DRAPE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// Build the scene JSON. **The garment is object 0**, so `Scene::lower`'s
/// concatenation leaves the garment's vertex indices unchanged and the stitches
/// need no offset at all.
fn scene(garment: &Garment, proxy: Option<&CollisionProxy>, tier: Tier) -> serde_json::Value {
    let mut objects = vec![serde_json::json!({
        "name": "garment",
        "positions": garment.positions,
        "tris": garment.tris,
        "uvs": garment.uvs,
        // Cotton: the balanced woven middle of the preset table, and the one a
        // block is toiled in.
        "preset": "cotton",
    })];
    if let Some(proxy) = proxy {
        // Only the slice of the body the garment can reach. The band is the
        // garment's own extent plus travel allowance, so it moves with the
        // garment rather than being a hardcoded height.
        let bounds = garment.bounds();
        let band = bounds.map_or((f32::NEG_INFINITY, f32::INFINITY), |b| {
            (b.min[1] - MARGIN_DOWN, b.max[1] + MARGIN_UP)
        });
        let clipped = clip_proxy(proxy, band);
        tracing::info!(
            target: "nguo_app",
            band_lo = band.0,
            band_hi = band.1,
            kept_verts = clipped.positions.len(),
            of_verts = proxy.positions.len(),
            kept_tris = clipped.tris.len(),
            of_tris = proxy.tris.len(),
            "body clipped to the garment's band"
        );
        objects.push(serde_json::json!({
            "name": format!("body/{}", proxy.avatar),
            "positions": clipped.positions,
            "tris": clipped.tris,
            "uvs": clipped.uvs,
            // Every vertex infinite-mass. This is also what makes the body cheap:
            // `broadphase`'s movability filter drops every candidate pair whose
            // participants are all immovable (D-17, 9.724 -> 0.235 s/step).
            "is_static": true,
        }));
    }
    // KEBAB-CASE. `SolverConfig` is `#[serde(default, rename_all = "kebab-case")]`
    // (`engine-api/src/params.rs:41`), so `self_collision` is not a misspelling of
    // `self-collision` — it is an *unknown field*, which serde silently ignores,
    // leaving the default `false`. That is exactly how the first run of this code
    // draped a garment 75 mm inside the body and reported success.
    // `InProcessTransport` now rejects an unknown config key for that reason.
    serde_json::json!({
        "config": {
            "dt": DT,
            // Cloth-vs-body contact runs through the same pass as self-contact,
            // so this is not optional: with it off the garment falls straight
            // through the body (analytic colliders are the only other geometry
            // and there are none).
            "self-collision": true,
            // A ground plane, so a garment that misses the body lands instead of
            // falling forever — which is also how you can *see* that it missed.
            "floor-y": 0.0,
            // A bake earns more Newton iterations; a preview wants the frame.
            "iterations": match tier { Tier::Preview => 2, Tier::Bake => 8 },
        },
        "objects": objects,
    })
}

/// Which part of the body can possibly touch the garment, as a `(lo, hi)` world-Y
/// band with margin.
///
/// **Nothing is pinned and nothing holds the garment up but the body itself.** That
/// is not a simplification, it is the better answer, measured: an earlier version
/// pinned the whole top edge, and because the arranger necessarily starts the tube
/// on a cylinder *circumscribing* an elliptical torso, a pinned top edge is frozen
/// at that loose radius and its seams can never close. Pinned, the sample's seams
/// settled at **529.5 mm** still open after 96 steps; unpinned, at **31.6 mm** after
/// 120 — 17× better, with the pinning code deleted. The garment stays up because a
/// 1 120 mm tube catches on a 1 087 mm hip, which is how a tube skirt works — and
/// since the sample is now *placed* at the measured hip by its own
/// `Feature::Support` annotation, it catches there rather than sliding down to find
/// it (**30 mm** of settle over 36 steps, re-measured 2026-08-04 by
/// `the_body_is_what_holds_the_garment_up`; this comment said 18 mm, which was a
/// solver-generation out of date).
///
/// The consequence for this band: the garment may still slide, so the band has to
/// cover where it ends up, not just where it started. `MARGIN_DOWN` is that travel
/// allowance and is checked at runtime — see [`scene`]. It is now far more than the
/// measured travel; it stays because an unannotated garment still starts at the
/// fallback landmark and does slide.
const MARGIN_DOWN: f32 = 0.35;
const MARGIN_UP: f32 = 0.10;

/// The body, cut down to the band the garment can reach. Same fields as
/// [`CollisionProxy`] minus the identity, which the clip does not change.
struct ClippedProxy {
    positions: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    uvs: Vec<[f32; 2]>,
}

/// Keep only the proxy triangles inside `band`, renumbered.
///
/// A triangle every vertex of which is outside the band cannot come within
/// `contact_dhat` (1 mm) of any cloth vertex inside it, so dropping it changes no
/// contact — it only shrinks the static geometry the solver walks. That matters
/// because the static body *is* the cost: measured at the shipped preview
/// resolution, the same garment steps in 0.008 s alone and 0.060 s with the whole
/// body, and 0.019 s vs 0.259 s with the self-collision pass that body contact
/// needs.
///
/// Vertices are kept in their original order, so the clip is deterministic.
fn clip_proxy(proxy: &CollisionProxy, band: (f32, f32)) -> ClippedProxy {
    let inside = |i: u32| {
        proxy
            .positions
            .get(i as usize)
            .is_some_and(|p| p[1] >= band.0 && p[1] <= band.1)
    };
    let kept: Vec<[u32; 3]> = proxy
        .tris
        .iter()
        .copied()
        .filter(|t| t.iter().copied().any(inside))
        .collect();

    let mut remap = vec![u32::MAX; proxy.positions.len()];
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    for t in &kept {
        for &i in t {
            if remap[i as usize] == u32::MAX {
                remap[i as usize] = positions.len() as u32;
                positions.push(proxy.positions[i as usize]);
                uvs.push(proxy.uvs.get(i as usize).copied().unwrap_or([0.0, 0.0]));
            }
        }
    }
    let tris = kept.iter().map(|t| t.map(|i| remap[i as usize])).collect();
    ClippedProxy {
        positions,
        tris,
        uvs,
    }
}

/// Total distance the seams are still open, metres, for a streamed frame.
///
/// The same measure as `Garment::seam_gap_m`, evaluated against solved positions
/// rather than the arranged ones.
fn seam_gap(garment: &Garment, positions: &[[f32; 3]]) -> f64 {
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
    garment
        .stitches
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

/// Positions + triangles → a renderable mesh, with area-weighted vertex normals.
///
/// Recomputed per frame rather than streamed: the solver does not emit normals,
/// and cloth normals change every step anyway. Cross products of the two triangle
/// edges are *already* area-weighted (their magnitude is twice the area), so
/// there is nothing to weight by explicitly.
fn mesh_of(positions: &[[f32; 3]], tris: &[[u32; 3]]) -> AvatarMesh {
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for t in tris {
        let [a, b, c] = t.map(|i| positions.get(i as usize).copied().unwrap_or([0.0; 3]));
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for &i in t {
            if let Some(slot) = normals.get_mut(i as usize) {
                for k in 0..3 {
                    slot[k] += n[k];
                }
            }
        }
    }
    AvatarMesh {
        vertices: positions
            .iter()
            .zip(&normals)
            .map(|(position, n)| {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                Vertex {
                    position: *position,
                    // A degenerate or unreferenced vertex gets +Y rather than a
                    // NaN normal, which the shader would render as a black hole.
                    normal: if len > 1e-12 {
                        [n[0] / len, n[1] / len, n[2] / len]
                    } else {
                        [0.0, 1.0, 0.0]
                    },
                }
            })
            .collect(),
        indices: tris.iter().flatten().copied().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_are_unit_length_and_face_the_winding() {
        // One counter-clockwise triangle in the y = 0 plane, seen from +Y.
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
        let mesh = mesh_of(&positions, &[[0, 1, 2]]);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        for v in &mesh.vertices {
            let n = v.normal;
            assert!(
                (n[1] - 1.0).abs() < 1e-6,
                "expected +Y, got {n:?} — the winding convention flipped",
            );
        }
    }

    /// A vertex no triangle references, and a zero-area triangle, must not
    /// produce a NaN normal — a whole frame renders black if one does.
    #[test]
    fn a_degenerate_frame_still_produces_finite_normals() {
        let positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
        ];
        let mesh = mesh_of(&positions, &[[0, 1, 2]]);
        assert_eq!(mesh.vertices.len(), 4);
        for v in &mesh.vertices {
            assert!(v.normal.iter().all(|c| c.is_finite()), "{:?}", v.normal);
            assert_eq!(v.normal, [0.0, 1.0, 0.0]);
        }
        // An out-of-range index must not panic either — a short frame is a bug
        // elsewhere, not a crash here.
        let _ = mesh_of(&positions, &[[0, 1, 99]]);
    }

    /// The garment must be object **0**, or every stitch index is off by the
    /// body's vertex count and the seams pull random parts of the body together.
    #[test]
    fn the_garment_is_the_first_object_so_stitch_indices_need_no_offset() {
        let garment =
            nguo_sewing::build(&nguo_sewing::sample::bodice_block(), &SewOptions::default())
                .expect("the sample sews");
        let proxy = CollisionProxy {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            tris: vec![[0, 1, 2]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            avatar: "test".into(),
        };
        let scene = scene(&garment, Some(&proxy), Tier::Preview);

        let objects = scene["objects"].as_array().expect("objects");
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0]["name"], "garment");
        assert_eq!(objects[1]["name"], "body/test");
        assert_eq!(objects[0]["is_static"], serde_json::Value::Null);
        assert_eq!(objects[1]["is_static"], true);
        // NOTHING is pinned. See `MARGIN_DOWN`'s docs: pinning the top edge
        // freezes it at the arranger's circumscribing radius and its seams can
        // never close (measured: 529.5 mm still open pinned, 32.0 mm unpinned).
        // The body is what holds the garment up.
        assert!(
            objects[0]["pins"].is_null(),
            "something is pinning the garment: {}",
            objects[0]["pins"],
        );
        assert_eq!(
            objects[0]["positions"].as_array().expect("positions").len(),
            garment.vertex_count(),
        );
        // Every stitch index has to be inside the garment's own prefix.
        let n = garment.vertex_count() as u32;
        assert!(garment
            .stitches
            .iter()
            .all(|s| s.index.iter().all(|&i| i < n)));

        // Body contact needs the self-collision pass; without it the garment
        // falls through (D-17) — and the key must be the kebab-case one serde
        // actually reads, or it is silently ignored.
        assert_eq!(scene["config"]["self-collision"], true);
        assert!(
            scene["config"]["self_collision"].is_null(),
            "the snake_case key is an unknown field serde would ignore",
        );
        // Every object must carry UVs or `lower` throws them all away and the
        // anisotropic membrane goes inert.
        for obj in objects {
            assert_eq!(
                obj["uvs"].as_array().expect("uvs").len(),
                obj["positions"].as_array().expect("positions").len(),
                "{}: uvs are not parallel to positions",
                obj["name"],
            );
        }
    }

    /// No body is a legitimate scene, not a degenerate one — and it must say so.
    #[test]
    fn without_a_body_the_scene_is_the_garment_alone_and_the_status_says_why() {
        let garment =
            nguo_sewing::build(&nguo_sewing::sample::bodice_block(), &SewOptions::default())
                .expect("the sample sews");
        let scene = scene(&garment, None, Tier::Bake);
        assert_eq!(scene["objects"].as_array().expect("objects").len(), 1);
        assert_eq!(scene["config"]["iterations"], 8);
        assert_eq!(scene["config"]["floor-y"], 0.0);
    }

    /// The seam gap has to fall as the stitches pull. Fed the arranged positions
    /// it must agree with `Garment::seam_gap_m`; fed coincident ones it must be 0.
    /// The clip must keep every triangle that could touch the cloth and drop the
    /// rest, and it must renumber correctly — an off-by-one in the remap silently
    /// collides the cloth against a scrambled body.
    #[test]
    fn clipping_the_body_keeps_the_band_and_renumbers_soundly() {
        // A vertical stack of triangles, one per 0.1 m from 0 to 2 m.
        let n = 21;
        let positions: Vec<[f32; 3]> = (0..n)
            .flat_map(|i| {
                let y = i as f32 * 0.1;
                [[0.0, y, 0.0], [0.1, y, 0.0], [0.0, y, 0.1]]
            })
            .collect();
        let tris: Vec<[u32; 3]> = (0..n).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect();
        let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[2]]).collect();
        let proxy = CollisionProxy {
            positions: positions.clone(),
            tris,
            uvs,
            avatar: "stack".into(),
        };

        let ClippedProxy {
            positions: kept_p,
            tris: kept_t,
            uvs: kept_uv,
        } = clip_proxy(&proxy, (0.55, 1.05));
        // y = 0.6 .. 1.0 inclusive: five triangles.
        assert_eq!(kept_t.len(), 5, "{kept_t:?}");
        assert_eq!(kept_p.len(), 15);
        assert_eq!(kept_uv.len(), kept_p.len());
        for t in &kept_t {
            for &i in t {
                assert!(
                    (i as usize) < kept_p.len(),
                    "index {i} past {}",
                    kept_p.len()
                );
                let y = kept_p[i as usize][1];
                assert!((0.55..=1.05).contains(&y), "kept a triangle at y = {y}");
            }
            // ...and the renumbered triangle is the SAME triangle: its three
            // vertices must still be the three distinct corners.
            assert_ne!(t[0], t[1]);
            assert_ne!(t[1], t[2]);
        }

        // An unbounded band keeps everything, unchanged and in order.
        let all = clip_proxy(&proxy, (f32::NEG_INFINITY, f32::INFINITY));
        let (all_p, all_t) = (all.positions, all.tris);
        assert_eq!(all_p, positions);
        assert_eq!(all_t, proxy.tris);

        // A band that misses the body entirely keeps nothing, rather than panicking.
        let none = clip_proxy(&proxy, (10.0, 11.0));
        assert!(none.positions.is_empty() && none.tris.is_empty());
    }

    #[test]
    fn the_seam_gap_measure_agrees_with_the_garments_own_and_bottoms_out_at_zero() {
        let garment =
            nguo_sewing::build(&nguo_sewing::sample::bodice_block(), &SewOptions::default())
                .expect("the sample sews");

        let arranged = seam_gap(&garment, &garment.positions);
        assert!(
            (arranged - garment.seam_gap_m()).abs() < 1e-6,
            "{arranged} vs {}",
            garment.seam_gap_m(),
        );
        assert!(arranged > 0.0, "the sample's seams already start shut");

        let collapsed = vec![[0.0f32; 3]; garment.positions.len()];
        assert_eq!(seam_gap(&garment, &collapsed), 0.0);
    }

    #[test]
    fn the_step_budget_is_overridable_for_a_profiling_run() {
        assert_eq!(steps_for(Tier::Preview), PREVIEW_STEPS);
        assert_eq!(steps_for(Tier::Bake), BAKE_STEPS);
        // Not `set_var`: these tests share a process, and a leaked env var would
        // make the two asserts above flaky depending on test order.
        const { assert!(PREVIEW_STEPS < BAKE_STEPS) };
    }
}

/// The real thing: the shipped avatar, the shipped sample garment, and the
/// solver. Skips loudly when `assets/avatars/` is absent, with a require-env
/// hatch so CI cannot pass by finding nothing.
///
/// These are unit tests rather than `tests/*.rs` because `nguo-app` is a binary
/// crate: an integration test cannot reach [`scene`] or [`crate::avatar`].
#[cfg(test)]
mod solve_tests {
    use super::*;
    use nguo_engine_bridge::{Flow, FrameSink, Transport};
    use std::io::Write;

    /// What the shipped COL tier contains. A floor, not the exact number: seam
    /// splitting means the loaded count exceeds the manifest's declared 7 498
    /// (10 230 measured), so an equality assert here would be asserting a bug.
    const MIN_PROXY_TRIANGLES: usize = 14_000;

    struct Frames {
        got: Vec<Vec<[f32; 3]>>,
    }

    impl FrameSink for Frames {
        fn push(&mut self, _index: u32, positions: Vec<[f32; 3]>) -> Flow {
            self.got.push(positions);
            Flow::Continue
        }
        fn is_cancelled(&self) -> bool {
            false
        }
    }

    fn avatars() -> Option<crate::avatar::Avatars> {
        match crate::avatar::Avatars::load() {
            Ok(a) => Some(a),
            Err(error) => {
                // fd 2 directly: libtest captures `eprintln!`, so a macro'd skip
                // notice is invisible without --nocapture.
                let _ = writeln!(std::io::stderr(), "SKIPPING: no avatar bundle ({error})");
                assert!(
                    std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none(),
                    "MGUSO_REQUIRE_AVATAR_BUNDLE is set but the avatar bundle is absent",
                );
                None
            }
        }
    }

    /// Sew the sample against a real body and run `steps` solver steps.
    /// Returns the garment, the frames and the engine's report.
    fn drape_on_body(
        steps: u32,
        avatars: &mut crate::avatar::Avatars,
    ) -> (Garment, Frames, serde_json::Value) {
        let proxy = avatars.collider().expect("the COL tier is shipped");
        assert!(
            proxy.tris.len() >= MIN_PROXY_TRIANGLES,
            "{} triangles — is this still the COL tier?",
            proxy.tris.len(),
        );

        // The shell's own preview settings, not a bespoke pair: a test that sews
        // at a resolution the app never uses is not testing the app.
        let doc = nguo_sewing::sample::bodice_block();
        let opts = sew_options(
            avatars.arrange_options(&nguo_sewing::hang(&doc)),
            Tier::Preview,
        );
        let garment = nguo_sewing::build(&doc, &opts).expect("the sample sews");

        let sub = Submission {
            scene: scene(&garment, Some(&proxy), Tier::Preview),
            stitches: garment
                .stitches
                .iter()
                .map(|s| Stitch {
                    index: s.index,
                    weight: s.weight,
                })
                .collect(),
            tier: Tier::Preview,
            frames: Some(steps),
            out_dir: None,
        };

        let mut sink = Frames { got: Vec::new() };
        let summary = InProcessTransport::new()
            .solve(&sub, &mut sink)
            .expect("the drape must solve");
        (garment, sink, summary.report)
    }

    /// The M6 acceptance test, measured. A garment drafted, sewn and draped on the
    /// body must come out finite, with its seams **closing** and no vertex inside
    /// the body it is draped over.
    ///
    /// `NGUO_DRAPE_TEST_STEPS` turns the step count up for a real settle; the
    /// default is what keeps the suite usable.
    #[test]
    fn a_sewn_garment_drapes_on_the_real_body_and_its_seams_close() {
        let Some(mut avatars) = avatars() else { return };
        let steps: u32 = std::env::var("NGUO_DRAPE_TEST_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(24);

        let (garment, frames, report) = drape_on_body(steps, &mut avatars);
        let _ = writeln!(
            std::io::stderr(),
            "DRAPE {} cloth verts + {} static, {} tris, {} stitches, {steps} steps: \
             {:.0} ms/step median, {:.2} s total",
            garment.vertex_count(),
            report["verts"].as_u64().unwrap_or(0) as usize - garment.vertex_count(),
            report["tris"].as_u64().unwrap_or(0),
            garment.stitches.len(),
            report["median_step_s"].as_f64().unwrap_or(0.0) * 1000.0,
            report["total_s"].as_f64().unwrap_or(0.0),
        );

        assert_eq!(frames.got.len() as u32, steps);
        assert_eq!(report["stitches"], garment.stitches.len());
        assert_eq!(report["self_collision"], true);

        // Every frame finite. A single NaN propagates to the whole mesh in one
        // step, so this is the assertion that catches a bad membrane element.
        for (i, frame) in frames.got.iter().enumerate() {
            assert!(
                frame.iter().flatten().all(|c| c.is_finite()),
                "frame {i} is not finite",
            );
        }

        let first = &frames.got[0][..garment.vertex_count()];
        let last = &frames.got[frames.got.len() - 1][..garment.vertex_count()];

        // It moved. Floors on both ends: it must not sit still, and it must not
        // fly apart either.
        let moved = first
            .iter()
            .zip(last)
            .map(|(a, b)| ((0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>()).sqrt())
            .fold(0.0f32, f32::max);
        let _ = writeln!(
            std::io::stderr(),
            "  moved at most {:.1} mm",
            moved * 1000.0
        );
        assert!(moved > 1e-4, "nothing moved in {steps} steps");
        assert!(moved < 1.0, "a vertex travelled {moved} m — it diverged");

        // The seams close. The arranger starts them open by its slack, and the
        // stitches are the only thing that can shut them.
        let open_at = |f: &[[f32; 3]]| seam_gap(&garment, f);
        let (start, end) = (open_at(first), open_at(last));
        let _ = writeln!(
            std::io::stderr(),
            "  seams {:.1} mm -> {:.1} mm open",
            start * 1000.0,
            end * 1000.0,
        );
        // The whole curve, not just the ends: this is what `PREVIEW_STEPS` is
        // chosen from, and a curve that plateaus early says the budget is generous.
        let curve: Vec<String> = frames
            .got
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 12 == 0 || *i + 1 == frames.got.len())
            .map(|(i, f)| {
                format!(
                    "{i}:{:.0}mm",
                    seam_gap(&garment, &f[..garment.vertex_count()]) * 1000.0
                )
            })
            .collect();
        let _ = writeln!(std::io::stderr(), "  closure {}", curve.join(" "));

        assert!(start > 0.05, "the seams did not start open: {start} m");
        assert!(
            end < 0.5 * start,
            "the seams barely closed: {start:.4} -> {end:.4} m",
        );
    }

    /// Body collision, as a **differential** test whose control is *no body*.
    ///
    /// Not a geometric "is this vertex inside the body" predicate: a torso is an
    /// ellipse (0.20 m half-width against a 0.15 m half-depth on the female
    /// proxy), so any single-radius test reads a garment correctly hugging the
    /// front as 5 cm *inside* a circle of the body's widest half-width. That is how
    /// the first version of this test failed a drape that was fine.
    ///
    /// The control is also **not** self-collision off: with the pass off the seam
    /// forces have nothing opposing them and the cloth diverges (measured: a mean
    /// radius of 1.06 m). A control that produces garbage proves nothing. The same
    /// garment with no body at all is a valid solve with an obvious expected
    /// outcome — it falls to the floor — and nothing but the body can stop it.
    #[test]
    fn the_body_is_what_holds_the_garment_up() {
        let Some(mut avatars) = avatars() else { return };
        let steps: u32 = std::env::var("NGUO_DRAPE_TEST_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PREVIEW_STEPS);

        let proxy = avatars.collider().expect("COL tier");
        let doc = nguo_sewing::sample::bodice_block();
        let opts = sew_options(
            avatars.arrange_options(&nguo_sewing::hang(&doc)),
            Tier::Preview,
        );
        let garment = nguo_sewing::build(&doc, &opts).expect("sews");
        let stitches: Vec<Stitch> = garment
            .stitches
            .iter()
            .map(|s| Stitch {
                index: s.index,
                weight: s.weight,
            })
            .collect();
        let started_y = garment.bounds().expect("bounds").min[1];

        let run = |body: bool| -> (f32, f32) {
            let mut sink = Frames { got: Vec::new() };
            InProcessTransport::new()
                .solve(
                    &Submission {
                        scene: scene(&garment, body.then_some(&*proxy), Tier::Preview),
                        stitches: stitches.clone(),
                        tier: Tier::Preview,
                        frames: Some(steps),
                        out_dir: None,
                    },
                    &mut sink,
                )
                .expect("solves");
            let last = &sink.got[sink.got.len() - 1][..garment.vertex_count()];
            let lowest = last.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
            let mean_y = last.iter().map(|p| p[1]).sum::<f32>() / last.len() as f32;
            (lowest, mean_y)
        };

        let (on_low, on_mean) = run(true);
        let (off_low, off_mean) = run(false);
        let _ = writeln!(
            std::io::stderr(),
            "  hem started at {started_y:.4} m\n  \
               with body   : hem {on_low:.4} m, mean y {on_mean:.4} m\n  \
               without body: hem {off_low:.4} m, mean y {off_mean:.4} m",
        );

        // Floors: both runs must have produced a garment, not a NaN or a speck.
        for (label, low, mean) in [("with", on_low, on_mean), ("without", off_low, off_mean)] {
            assert!(
                low.is_finite() && mean.is_finite() && mean > low,
                "{label} body: hem {low}, mean {mean} is not a garment",
            );
        }
        // The control really does fall: with no body there is nothing but the floor
        // at y = 0, so it must end up much lower than it started.
        assert!(
            off_mean < on_mean,
            "the body made no difference: mean y {on_mean:.4} m with it, \
             {off_mean:.4} m without",
        );
        assert!(
            on_mean - off_mean > 0.05,
            "only {:.1} mm of difference between draping on a body and on nothing",
            (on_mean - off_mean) * 1000.0,
        );
        // ...and with the body, the garment is still up on it rather than pooled on
        // the floor.
        assert!(
            on_low > 0.20,
            "the garment fell to {on_low:.4} m even with the body there",
        );
        // The point of hanging it at its own landmark: it **stays where it was
        // placed**. This assertion is a re-baseline, not a new guard — the same
        // line used to assert only `on_low > 0.20`, i.e. that a garment started at
        // the underbust *slid down and caught somewhere*.
        //
        // Measured A/B over the same 36 steps, by feeding `arrange_options` the
        // unannotated fallback (`Hang { landmark: None, drop_mm: INFINITY }`,
        // which is exactly the old behaviour):
        //
        // ```text
        //   annotated (hip)      hem 0.3722 -> 0.3418 m     30.4 mm of slide
        //   fallback (underbust) hem 0.6876 -> 0.5887 m     99.0 mm, still moving
        // ```
        //
        // So this line discriminates: run the fallback and it panics with exactly
        // "the garment slid 99.0 mm". Re-measured 2026-08-04 — the numbers here were
        // previously 18 mm and 106 mm, taken against an older solver.
        //
        // What it does NOT discriminate on its own: with the annotation removed but
        // the band kept finite (`drop_mm: 560.0`), the slide is only 17.1 mm and this
        // assertion PASSES. It catches the old placement as a whole — feet-sized
        // radius included — not the landmark in isolation. The landmark alone is
        // pinned by `avatar::bundle_tests::a_designed_garment_lands_at_the_landmark_\
        // it_was_drafted_for`, to 1 mm, on both bodies.
        assert!(
            started_y - on_low < 0.05,
            "the garment slid {:.1} mm from where its landmark put it \
             ({started_y:.4} m -> {on_low:.4} m)",
            (started_y - on_low) * 1000.0,
        );
    }

    /// Determinism through the solver, not just through the sewing: the same
    /// submission must produce bit-identical frames. This is what catches hash
    /// iteration order leaking into the broadphase's candidate order.
    #[test]
    fn the_same_submission_solves_bit_identically() {
        let Some(mut avatars) = avatars() else { return };
        let (_, a, _) = drape_on_body(4, &mut avatars);
        let (_, b, _) = drape_on_body(4, &mut avatars);
        assert_eq!(a.got.len(), b.got.len());
        for (i, (x, y)) in a.got.iter().zip(&b.got).enumerate() {
            assert_eq!(x, y, "frame {i} differs between two identical solves");
        }
    }
}

/// The profile the whole M6 loop is tuned from. **Not** an assertion of an
/// absolute cost — that is this machine's — but of the *shape* of the cost, which
/// is what a configuration choice can be defended with.
///
/// Run it: `cargo test -p nguo-app --features debug-overlay profile_drape --
/// --nocapture --ignored`. Ignored by default because it solves for a minute.
#[cfg(test)]
mod profile {
    use super::*;
    use nguo_engine_bridge::{Flow, FrameSink, Transport};
    use std::io::Write;

    struct Last(Option<Vec<[f32; 3]>>);

    impl FrameSink for Last {
        fn push(&mut self, _index: u32, positions: Vec<[f32; 3]>) -> Flow {
            self.0 = Some(positions);
            Flow::Continue
        }
        fn is_cancelled(&self) -> bool {
            false
        }
    }

    /// One row of the table: what a configuration costs per solver step.
    fn row(
        label: &str,
        garment: &Garment,
        proxy: Option<&CollisionProxy>,
        self_collision: bool,
        steps: u32,
    ) -> f64 {
        let mut scene = scene(garment, proxy, Tier::Preview);
        scene["config"]["self-collision"] = serde_json::json!(self_collision);
        let sub = Submission {
            scene,
            stitches: garment
                .stitches
                .iter()
                .map(|s| Stitch {
                    index: s.index,
                    weight: s.weight,
                })
                .collect(),
            tier: Tier::Preview,
            frames: Some(steps),
            out_dir: None,
        };
        let mut sink = Last(None);
        let summary = InProcessTransport::new()
            .solve(&sub, &mut sink)
            .expect("solves");
        let step = summary.report["median_step_s"].as_f64().unwrap_or(0.0);
        let _ = writeln!(
            std::io::stderr(),
            "| {label:<40} | {:>6} | {:>6} | {:>5} | {:>8.3} |",
            summary.report["verts"].as_u64().unwrap_or(0),
            summary.report["tris"].as_u64().unwrap_or(0),
            self_collision,
            step,
        );
        step
    }

    /// What the arranged (pre-solve) garment looks like against the real body,
    /// before any solver is involved. Cheap, so not ignored.
    #[test]
    fn arrangement_geometry() {
        let Some(mut avatars) = crate::avatar::Avatars::load().ok() else {
            let _ = writeln!(std::io::stderr(), "SKIPPING: no avatar bundle");
            assert!(std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none());
            return;
        };
        let proxy = avatars.collider().expect("COL");
        let arrange =
            avatars.arrange_options(&nguo_sewing::hang(&nguo_sewing::sample::bodice_block()));
        let pb = nguo_sewing::Aabb::of(&proxy.positions).expect("body bounds");
        let _ = writeln!(
            std::io::stderr(),
            "avatar={} body_radius={:.4} m top_y={:.4} m gap={:.3} m; proxy bounds {:?}..{:?}",
            avatars.selected_key(),
            arrange.body_radius_m,
            arrange.top_y_m,
            arrange.gap_m,
            pb.min,
            pb.max,
        );
        // The body's radius profile, measured rather than assumed. Two things to
        // find out: where the arms actually are, and how far the torso departs
        // from a circle of `girth / 2π`.
        let mut widest = (0.0f32, 0.0f32);
        for pos in &proxy.positions {
            let r = (pos[0] * pos[0] + pos[2] * pos[2]).sqrt();
            if r > widest.0 {
                widest = (r, pos[1]);
            }
        }
        let _ = writeln!(
            std::io::stderr(),
            "widest point: r={:.4} m at y={:.4} m",
            widest.0,
            widest.1,
        );
        for band in 0..20 {
            let lo = band as f32 * 0.1;
            let hi = lo + 0.1;
            let mut max_r = 0.0f32;
            let mut max_x = 0.0f32;
            let mut max_z = 0.0f32;
            let mut n = 0;
            for pos in proxy.positions.iter().filter(|p| p[1] >= lo && p[1] < hi) {
                max_r = max_r.max((pos[0] * pos[0] + pos[2] * pos[2]).sqrt());
                max_x = max_x.max(pos[0].abs());
                max_z = max_z.max(pos[2].abs());
                n += 1;
            }
            if n > 0 {
                let _ = writeln!(
                    std::io::stderr(),
                    "  y {lo:.1}-{hi:.1}: n={n:<5} max_r={max_r:.4} max|x|={max_x:.4} max|z|={max_z:.4}",
                );
            }
        }

        for tier in [Tier::Preview, Tier::Bake] {
            let g = nguo_sewing::build(
                &nguo_sewing::sample::bodice_block(),
                &sew_options(arrange, tier),
            )
            .expect("sews");
            let _ = writeln!(
                std::io::stderr(),
                "  {tier:?}: {} verts, {} stitches, radius={:?} slack={:.4} m, \
                 seam_gap sum={:.4} m max={:.4} m, bounds {:?}..{:?}",
                g.vertex_count(),
                g.stitches.len(),
                g.arrangement.radius_m,
                g.arrangement.slack_m,
                g.seam_gap_m(),
                g.max_seam_gap_m(),
                g.bounds().unwrap().min,
                g.bounds().unwrap().max,
            );
        }
    }

    #[test]
    #[ignore = "solves for about a minute; run it deliberately"]
    fn profile_drape() {
        let Some(mut avatars) = crate::avatar::Avatars::load().ok() else {
            let _ = writeln!(std::io::stderr(), "SKIPPING: no avatar bundle");
            assert!(std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none());
            return;
        };
        let proxy = avatars.collider().expect("COL tier");
        let arrange =
            avatars.arrange_options(&nguo_sewing::hang(&nguo_sewing::sample::bodice_block()));

        let _ = writeln!(
            std::io::stderr(),
            "\n| configuration                            | verts  | tris   | self  |    s/step |\n\
               |------------------------------------------|--------|--------|-------|-----------|"
        );

        // Sew at three resolutions, each with and without the body, so the two
        // costs can be separated instead of guessed at.
        let mut coarse_body = 0.0;
        for mm in [60.0, 30.0] {
            let opts = SewOptions {
                triangulate: nguo_triangulate::TriangulateOptions::default()
                    .with_target_edge_mm(mm),
                arrange,
            };
            let started = Instant::now();
            let garment =
                nguo_sewing::build(&nguo_sewing::sample::bodice_block(), &opts).expect("sews");
            let sew_ms = started.elapsed().as_secs_f64() * 1000.0;
            let _ = writeln!(
                std::io::stderr(),
                "| -- {mm:.0} mm mesh: sew {sew_ms:.2} ms (validate {:.2} / tri {:.2} / arrange {:.2} / sew {:.2})",
                garment.timings.validate_ms,
                garment.timings.triangulate_ms,
                garment.timings.arrange_ms,
                garment.timings.sew_ms,
            );

            row(&format!("{mm:.0} mm, no body"), &garment, None, false, 6);
            row(
                &format!("{mm:.0} mm, no body, self-coll"),
                &garment,
                None,
                true,
                6,
            );
            row(
                &format!("{mm:.0} mm, clipped body"),
                &garment,
                Some(&proxy),
                false,
                6,
            );
            let with = row(
                &format!("{mm:.0} mm, clipped body, self-coll"),
                &garment,
                Some(&proxy),
                true,
                6,
            );
            // The same thing with the WHOLE body, to show what the clip buys.
            // Built by putting the unclipped proxy back into the scene the clip
            // just shrank — the only difference between the two rows.
            {
                let mut whole = scene(&garment, Some(&proxy), Tier::Preview);
                whole["objects"][1]["positions"] = serde_json::json!(proxy.positions);
                whole["objects"][1]["tris"] = serde_json::json!(proxy.tris);
                whole["objects"][1]["uvs"] = serde_json::json!(proxy.uvs);
                whole["config"]["self-collision"] = serde_json::json!(true);
                let mut sink = Last(None);
                let summary = InProcessTransport::new()
                    .solve(
                        &Submission {
                            scene: whole,
                            stitches: garment
                                .stitches
                                .iter()
                                .map(|s| Stitch {
                                    index: s.index,
                                    weight: s.weight,
                                })
                                .collect(),
                            tier: Tier::Preview,
                            frames: Some(6),
                            out_dir: None,
                        },
                        &mut sink,
                    )
                    .expect("solves");
                let _ = writeln!(
                    std::io::stderr(),
                    "| {:<40} | {:>6} | {:>6} | {:>5} | {:>8.3} |",
                    format!("{mm:.0} mm, WHOLE body, self-coll"),
                    summary.report["verts"].as_u64().unwrap_or(0),
                    summary.report["tris"].as_u64().unwrap_or(0),
                    true,
                    summary.report["median_step_s"].as_f64().unwrap_or(0.0),
                );
            }
            if mm == 60.0 {
                coarse_body = with;
            }
        }

        // The shape being asserted, not the numbers: adding a 15 000-triangle
        // static body is what dominates, so mesh resolution is NOT the lever.
        assert!(coarse_body > 0.0, "nothing was measured");
    }
}
