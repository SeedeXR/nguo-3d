// SPDX-License-Identifier: GPL-2.0-or-later
//! Property tests for the view transform (guide §7.1, §18).
//!
//! HiDPI is where a coordinate round-trip silently breaks: a stray
//! `* scale_factor` in the hit-test path is invisible at 1× and a
//! factor-of-two error at Retina, so every property here sweeps the scale
//! factor as well as the zoom.

use nguo_canvas::View;
use proptest::prelude::*;
use vello::kurbo::{Point, Vec2};

/// The clamp `View` enforces. Sampling outside it would test the clamp, not the
/// transform.
const ZOOM: (f64, f64) = (0.02, 400.0);

fn view() -> impl Strategy<Value = View> {
    (
        ZOOM.0..ZOOM.1,
        -50_000.0f64..50_000.0,
        -50_000.0f64..50_000.0,
    )
        .prop_map(|(zoom, px, py)| View {
            pan: Vec2::new(px, py),
            zoom,
        })
}

fn size() -> impl Strategy<Value = Vec2> {
    (1.0f64..4096.0, 1.0f64..4096.0).prop_map(|(w, h)| Vec2::new(w, h))
}

fn world() -> impl Strategy<Value = Point> {
    (-100_000.0f64..100_000.0, -100_000.0f64..100_000.0).prop_map(|(x, y)| Point::new(x, y))
}

/// Relative tolerance: a round trip through a 400× zoom of a 100 m coordinate
/// cannot be exact in f64, and demanding it would be a test of nothing. The
/// magnitude that sets the error is the *largest* value the arithmetic passed
/// through — the pan can dwarf the coordinate itself.
fn close(a: f64, b: f64, magnitude: f64) -> bool {
    (a - b).abs() <= 1e-9 * magnitude.abs().max(1.0)
}

/// Sub-nanometre on a 4K display: far below a pixel, far above f64 noise.
const SCREEN_EPS: f64 = 1e-6;

proptest! {
    #[test]
    fn world_to_screen_inverts_screen_to_world(
        view in view(),
        size in size(),
        world in world(),
    ) {
        let back = view.screen_to_world(view.world_to_screen(world, size), size);
        let (mx, my) = (
            world.x.abs().max(view.pan.x.abs()),
            world.y.abs().max(view.pan.y.abs()),
        );
        prop_assert!(close(back.x, world.x, mx), "{back:?} vs {world:?}");
        prop_assert!(close(back.y, world.y, my), "{back:?} vs {world:?}");
    }

    /// The direction that matters for hit-testing: a pixel the user clicked must
    /// come back as the same pixel after a trip through world space.
    #[test]
    fn screen_to_world_inverts_world_to_screen(
        view in view(),
        size in size(),
        sx in 0.0f64..4096.0,
        sy in 0.0f64..4096.0,
    ) {
        let screen = Point::new(sx, sy);
        let back = view.world_to_screen(view.screen_to_world(screen, size), size);
        prop_assert!((back.x - screen.x).abs() <= SCREEN_EPS, "{back:?} vs {screen:?}");
        prop_assert!((back.y - screen.y).abs() <= SCREEN_EPS, "{back:?} vs {screen:?}");
    }

    #[test]
    fn the_physical_round_trip_holds_at_every_scale_factor(
        view in view(),
        size in size(),
        world in world(),
        scale in 1.0f64..4.0,
    ) {
        let physical = view.world_to_physical(world, size, scale);
        let back = view.physical_to_world(physical, size, scale);
        let (mx, my) = (
            world.x.abs().max(view.pan.x.abs()),
            world.y.abs().max(view.pan.y.abs()),
        );
        prop_assert!(close(back.x, world.x, mx), "{back:?} vs {world:?}");
        prop_assert!(close(back.y, world.y, my), "{back:?} vs {world:?}");
    }

    /// Vello draws with `scene_transform`; the pointer is tested with
    /// `world_to_physical`. If they ever disagree, clicks miss the shape drawn
    /// under the cursor — silently, and only on some machines.
    #[test]
    fn the_scene_transform_agrees_with_the_hit_test_transform(
        view in view(),
        size in size(),
        world in world(),
        scale in 1.0f64..4.0,
    ) {
        let drawn = view.scene_transform(size, scale) * world;
        let hit = view.world_to_physical(world, size, scale);
        // Both are pixel coordinates, so the magnitude that matters is theirs.
        prop_assert!(close(drawn.x, hit.x, drawn.x), "{drawn:?} vs {hit:?}");
        prop_assert!(close(drawn.y, hit.y, drawn.y), "{drawn:?} vs {hit:?}");
    }

    /// Zoom about the cursor: whatever is under the pointer stays under it. Any
    /// number of steps, because the error would accumulate if it existed.
    #[test]
    fn zooming_never_moves_the_point_under_the_cursor(
        mut view in view(),
        size in size(),
        cx in 0.0f64..4096.0,
        cy in 0.0f64..4096.0,
        factors in prop::collection::vec(0.2f64..5.0, 1..12),
    ) {
        let cursor = Point::new(cx, cy);
        let anchor = view.screen_to_world(cursor, size);
        for f in factors {
            view.zoom_about(f, cursor, size);
            let now = view.screen_to_world(cursor, size);
            // The anchor can be a large world coordinate; scale accordingly.
            let m = anchor.x.abs().max(anchor.y.abs()).max(1.0);
            prop_assert!(
                (now.x - anchor.x).abs() <= 1e-7 * m && (now.y - anchor.y).abs() <= 1e-7 * m,
                "drifted from {anchor:?} to {now:?} at zoom {}",
                view.zoom
            );
        }
    }

    #[test]
    fn zoom_stays_inside_its_clamp_however_hard_it_is_pushed(
        mut view in view(),
        size in size(),
        factors in prop::collection::vec(0.001f64..1000.0, 1..40),
    ) {
        for f in factors {
            view.zoom_about(f, Point::new(size.x / 2.0, size.y / 2.0), size);
            prop_assert!(
                view.zoom >= ZOOM.0 && view.zoom <= ZOOM.1,
                "zoom escaped to {}",
                view.zoom
            );
        }
    }

    /// Panning is a screen-space gesture with a world-space result, and it must
    /// be exactly reversible — otherwise a document drifts under the mouse.
    #[test]
    fn panning_is_reversible(
        view in view(),
        dx in -2000.0f64..2000.0,
        dy in -2000.0f64..2000.0,
    ) {
        let mut moved = view;
        moved.pan_by_screen(Vec2::new(dx, dy));
        moved.pan_by_screen(Vec2::new(-dx, -dy));
        let m = view.pan.x.abs().max(view.pan.y.abs()).max(1.0);
        prop_assert!((moved.pan.x - view.pan.x).abs() <= 1e-9 * m);
        prop_assert!((moved.pan.y - view.pan.y).abs() <= 1e-9 * m);
        prop_assert_eq!(moved.zoom, view.zoom, "panning must not touch the zoom");
    }
}
