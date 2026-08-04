// SPDX-License-Identifier: GPL-2.0-or-later
//! Frame-budget profiling for the 2D canvas (guide §17).
//!
//! Budget: **8.3 ms target / 16.6 ms ceiling** for the whole frame. The canvas
//! scene build is one item in that budget, so what matters is the share it
//! takes, and this reports it in absolute milliseconds rather than a ratio.
//!
//! Numbers are printed to **fd 2**, not `println!`: libtest captures stdout and
//! a measurement nobody can read is a measurement nobody checks. Run with
//! `cargo test -p nguo-canvas --release -- --nocapture perf` to see them.
//!
//! `NGUO_PERF_STRICT=1` turns the budget into an assertion. It is off by
//! default because a shared CI box is not a frame-rate measurement, but the
//! loose ceilings below always apply, so a 10× regression fails the suite
//! wherever it runs.

use std::io::Write as _;
use std::time::{Duration, Instant};

use nguo_canvas::scene::{grid_scene, overlay_scene, Interaction, SceneCache};
use nguo_canvas::{fixture, snap::Snap, snap::SnapKind, Canvas2d, Target, View};
use nguo_doc::Document;
use nguo_ui_theme::Tokens;
use vello::kurbo::{Point, Vec2};

/// A 27" panel's worth of canvas, in logical px.
const SIZE: Vec2 = Vec2::new(1600.0, 1000.0);
/// Boundary edges per piece. A bodice front with darts and notches split in is
/// around this once it has been drafted, not the 4 of a rectangle.
const SEGMENTS: usize = 24;
/// The frame budget this session is held to.
const TARGET_MS: f64 = 8.3;
/// A regression net that holds on any machine: 10× the budget.
const SANITY_MS: f64 = 83.0;

fn note(args: std::fmt::Arguments) {
    let _ = writeln!(std::io::stderr(), "{args}");
}

macro_rules! note {
    ($($arg:tt)*) => { note(format_args!($($arg)*)) };
}

fn strict() -> bool {
    std::env::var("NGUO_PERF_STRICT").is_ok_and(|v| v == "1")
}

/// Median of `runs` timed iterations. Median, not mean: one scheduler hiccup
/// should not become the headline number.
fn bench(runs: usize, mut f: impl FnMut()) -> Duration {
    assert!(runs >= 3, "a median needs samples");
    let mut samples: Vec<Duration> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort_unstable();
    samples[runs / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn check(label: &str, took: Duration, budget_ms: f64) {
    let took_ms = ms(took);
    let share = took_ms / TARGET_MS * 100.0;
    note!("  {label:<44} {took_ms:8.3} ms  ({share:5.1}% of the 8.3 ms frame)");
    assert!(
        took_ms < SANITY_MS,
        "{label} took {took_ms:.3} ms, past the {SANITY_MS} ms sanity ceiling"
    );
    if strict() {
        assert!(
            took_ms < budget_ms,
            "{label} took {took_ms:.3} ms, over its {budget_ms} ms budget \
             (NGUO_PERF_STRICT=1)"
        );
    }
}

fn interaction(doc: &Document) -> Interaction {
    let piece = doc.pieces().ids().next().expect("a piece");
    let vertex = doc
        .piece(piece)
        .expect("a piece")
        .network
        .vertices
        .ids()
        .next()
        .expect("a vertex");
    Interaction {
        selection: vec![Target::Point { piece, vertex }],
        hover: Some(Target::Fill { piece }),
        snap: Some(Snap {
            world: Point::new(12.0, 34.0),
            kind: SnapKind::Grid,
            screen_distance: 1.0,
            source: None,
        }),
        editing: vec![piece],
        rubber_band: Some((Point::ORIGIN, Point::new(50.0, 50.0))),
    }
}

#[test]
fn scene_build_stays_inside_the_frame_budget() {
    let tokens = Tokens::light();
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    note!(
        "\nnguo-canvas scene build — {profile} build, {}×{} logical px, \
         {SEGMENTS} cubic edges per piece",
        SIZE.x,
        SIZE.y
    );

    let mut encoded = 0usize;
    for pieces in [1usize, 10, 50] {
        let doc = fixture::marker(pieces, SEGMENTS);
        assert_eq!(doc.pieces().len(), pieces);
        let it = interaction(&doc);
        let view = View::default();

        // Cold: every piece encoded from geometry. This is what a document open,
        // or any zoom step, costs.
        let cold = bench(9, || {
            let mut cache = SceneCache::new();
            let scene = cache.frame(&doc, &it, &view, SIZE, &tokens);
            encoded += scene.encoding().n_paths as usize;
            assert_eq!(cache.piece_builds(), pieces as u64);
        });
        check(
            &format!("{pieces:>2} pieces, cold (all rebuilt)"),
            cold,
            TARGET_MS,
        );

        // Steady: nothing dirty. Panning and hovering live here.
        let mut cache = SceneCache::new();
        cache.frame(&doc, &it, &view, SIZE, &tokens);
        let builds = cache.piece_builds();
        let steady = bench(31, || {
            let scene = cache.frame(&doc, &it, &view, SIZE, &tokens);
            encoded += scene.encoding().n_paths as usize;
        });
        assert_eq!(
            cache.piece_builds(),
            builds,
            "a steady frame must not rebuild a piece"
        );
        check(
            &format!("{pieces:>2} pieces, steady (cache hit)"),
            steady,
            TARGET_MS,
        );

        // Dragging one piece: the case §7.3 exists for. One piece rebuilt, the
        // rest appended from cache, interaction layer rebuilt.
        let dragged = doc.pieces().ids().next().expect("a piece");
        let drag = bench(31, || {
            cache.mark_dirty(dragged);
            let scene = cache.frame(&doc, &it, &view, SIZE, &tokens);
            encoded += scene.encoding().n_paths as usize;
        });
        check(
            &format!("{pieces:>2} pieces, drag (1 piece dirty)"),
            drag,
            TARGET_MS,
        );

        // The spatial index is derived state keyed on the document revision, so
        // a drag that emits a command per pointer move pays this every frame.
        // It is the largest single item on the canvas's frame bill — reported at
        // every complexity so the scaling is visible, not inferred.
        let index = bench(9, || {
            assert!(nguo_canvas::Index::of_document(&doc).len() >= pieces);
        });
        check(
            &format!("{pieces:>2} pieces, spatial index rebuild"),
            index,
            TARGET_MS,
        );

        if pieces == 50 {
            // The layers, separately, so a regression can be attributed.
            let grid = bench(31, || {
                let s = grid_scene(&view, SIZE, &tokens);
                encoded += s.encoding().n_paths as usize;
            });
            check("  grid + rulers alone", grid, TARGET_MS / 4.0);

            let overlay = bench(31, || {
                let s = overlay_scene(&doc, &it, &view, SIZE, &tokens);
                encoded += s.encoding().n_paths as usize;
            });
            check("  interaction layer alone", overlay, TARGET_MS / 4.0);
        }
    }

    // Not vacuous: the scenes above really did encode geometry.
    assert!(
        encoded > 1_000,
        "only {encoded} paths encoded — the benchmark measured nothing"
    );
}

#[test]
fn hit_testing_and_snapping_are_free_at_pointer_rates() {
    let doc = fixture::marker(50, SEGMENTS);
    let mut canvas = Canvas2d::new();
    // Somewhere inside the first piece, so the query has work to do.
    let screen = canvas.view.world_to_screen(Point::new(140.0, 10.0), SIZE);

    note!("\nnguo-canvas pointer path — 50 pieces, {SEGMENTS} edges each");

    // Warm the index so the query is measured on its own.
    let items = canvas.index(&doc).len();
    let hits = bench(201, || {
        canvas.hit(&doc, screen, SIZE);
    });
    check("  one hit-test query", hits, 1.0);

    let snaps = bench(201, || {
        canvas.resolve(&doc, screen, SIZE);
    });
    check("  one snap resolution", snaps, 1.0);

    note!("  index holds {items} items");
    assert!(
        canvas.hit(&doc, screen, SIZE).is_some(),
        "the profiled query must actually hit something"
    );
}
