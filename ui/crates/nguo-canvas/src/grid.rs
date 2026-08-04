// SPDX-License-Identifier: GPL-2.0-or-later
//! Grid and rulers, *derived from the view transform* — never a fixed bitmap
//! and never stored geometry (guide §7.2).
//!
//! Everything here is a pure function of [`View`] plus the widget size, so the
//! grid is correct at any pan, any zoom and any HiDPI scale factor without a
//! cache to invalidate.

use vello::kurbo::{Point, Rect, Vec2};

use crate::View;

/// On-screen spacing a minor grid line aims for, in logical px. Below this the
/// grid reads as a grey wash rather than a grid.
pub const TARGET_MINOR_PX: f64 = 14.0;

/// Minor lines per major line. Ten, so the major lines land on decade
/// boundaries of the 1-2-5 step and the labels stay round numbers.
pub const MINORS_PER_MAJOR: f64 = 10.0;

/// Ruler strip thickness in logical px, both axes.
pub const RULER_PX: f64 = 18.0;

/// Refuse to emit more ticks than this from one call. A step of the right size
/// yields a few hundred; anything past this is a caller bug and would otherwise
/// allocate until the frame budget is gone.
const MAX_TICKS: i64 = 4096;

/// The world-space rectangle the widget currently shows.
pub fn visible(view: &View, size: Vec2) -> Rect {
    Rect::from_points(
        view.screen_to_world(Point::ORIGIN, size),
        view.screen_to_world(Point::new(size.x, size.y), size),
    )
}

/// A 1-2-5 world step (mm) whose on-screen spacing is at least `target_px`.
///
/// This is the whole "grid derives from the transform" rule in one function:
/// zoom out and the step climbs 1 → 2 → 5 → 10 mm, so line density on screen
/// stays constant while the *world* meaning of a cell changes.
pub fn step_mm(zoom: f64, target_px: f64) -> f64 {
    debug_assert!(zoom > 0.0 && target_px > 0.0);
    let raw = target_px / zoom;
    let decade = 10f64.powf(raw.log10().floor());
    let m = raw / decade;
    decade
        * if m <= 1.0 {
            1.0
        } else if m <= 2.0 {
            2.0
        } else if m <= 5.0 {
            5.0
        } else {
            10.0
        }
}

/// Minor and major step for the current zoom, in world mm.
pub fn steps(zoom: f64) -> (f64, f64) {
    let minor = step_mm(zoom, TARGET_MINOR_PX);
    (minor, minor * MINORS_PER_MAJOR)
}

/// Multiples of `step` inside `from..=to`, in world units.
pub fn ticks(from: f64, to: f64, step: f64) -> impl Iterator<Item = f64> {
    let (first, last) = if step > 0.0 && from.is_finite() && to.is_finite() && from <= to {
        let first = (from / step).ceil() as i64;
        let last = (to / step).floor() as i64;
        (first, last.min(first.saturating_add(MAX_TICKS)))
    } else {
        (0, -1)
    };
    (first..=last).map(move |i| i as f64 * step)
}

/// Ruler tick labels for one axis: `(screen coordinate, text)`.
///
/// ponytail: this crate has no font — `nguo-gpu::overlay` owns the vendored
/// Space Mono and parley is not wired yet (guide §7.5). Returning the strings
/// keeps the numbers available to whoever can draw them; swap in
/// `Scene::draw_glyphs` when a font handle reaches the canvas.
pub fn labels(view: &View, size: Vec2, horizontal: bool) -> Vec<(f64, String)> {
    let (_, major) = steps(view.zoom);
    let world = visible(view, size);
    let (from, to) = if horizontal {
        (world.x0, world.x1)
    } else {
        (world.y0, world.y1)
    };
    ticks(from, to, major)
        .map(|v| {
            let screen = if horizontal {
                view.world_to_screen(Point::new(v, 0.0), size).x
            } else {
                view.world_to_screen(Point::new(0.0, v), size).y
            };
            (screen, format!("{v:.0}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn the_step_keeps_screen_density_roughly_constant() {
        for zoom in [0.02, 0.05, 0.3, 1.0, 2.0, 7.5, 60.0, 400.0] {
            let step = step_mm(zoom, TARGET_MINOR_PX);
            let on_screen = step * zoom;
            assert!(
                (TARGET_MINOR_PX..TARGET_MINOR_PX * 10.0).contains(&on_screen),
                "zoom {zoom}: step {step} mm is {on_screen} px on screen"
            );
        }
    }

    #[test]
    fn the_step_is_always_a_1_2_5_decade() {
        for zoom in [0.02, 0.11, 0.37, 1.0, 3.3, 88.0, 400.0] {
            let step = step_mm(zoom, TARGET_MINOR_PX);
            let decade = 10f64.powf(step.log10().floor());
            let m = step / decade;
            assert!(
                [1.0, 2.0, 5.0].iter().any(|x| (m - x).abs() < 1e-9),
                "zoom {zoom}: step {step} has mantissa {m}"
            );
        }
    }

    #[test]
    fn ticks_land_on_multiples_and_stay_inside_the_range() {
        let v: Vec<f64> = ticks(-23.0, 41.0, 10.0).collect();
        assert_eq!(v, vec![-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]);
        assert!(ticks(0.0, 10.0, 0.0).next().is_none(), "step 0 is refused");
        assert!(ticks(10.0, 0.0, 1.0).next().is_none(), "reversed is empty");
        assert_eq!(
            ticks(0.0, 1e12, 1.0).count() as i64,
            MAX_TICKS + 1,
            "an absurd range is capped, not allocated"
        );
    }

    #[test]
    fn the_grid_follows_the_view_and_not_the_widget() {
        // Pan by a whole cell and the tick pattern must shift by exactly one
        // cell in screen space — that is what "derived from the transform"
        // buys, and a bitmap grid gets it wrong.
        let size = Vec2::new(800.0, 600.0);
        let mut view = View::default();
        let (minor, _) = steps(view.zoom);
        let before: Vec<f64> = ticks(visible(&view, size).x0, visible(&view, size).x1, minor)
            .map(|v| view.world_to_screen(Point::new(v, 0.0), size).x)
            .collect();

        view.pan_by_screen(Vec2::new(-minor * view.zoom, 0.0));
        let after: Vec<f64> = ticks(visible(&view, size).x0, visible(&view, size).x1, minor)
            .map(|v| view.world_to_screen(Point::new(v, 0.0), size).x)
            .collect();

        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn labels_are_at_the_screen_position_of_their_world_value() {
        let size = Vec2::new(640.0, 480.0);
        let view = View {
            pan: Vec2::new(37.0, -12.0),
            zoom: 1.7,
        };
        let labels = labels(&view, size, true);
        assert!(!labels.is_empty(), "a visible ruler has ticks");
        for (screen, text) in &labels {
            let world: f64 = text.parse().expect("labels are numeric");
            let expected = view.world_to_screen(Point::new(world, 0.0), size).x;
            assert_abs_diff_eq!(*screen, expected, epsilon = 1e-9);
            assert!(
                *screen >= -1.0 && *screen <= size.x + 1.0,
                "label {text} at {screen} is off-widget"
            );
        }
    }
}
