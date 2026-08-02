// SPDX-License-Identifier: GPL-2.0-or-later
//! The 2D canvas view transform.
//!
//! Three spaces:
//!
//! * **world** — pattern units (centimetres, eventually). What the document
//!   stores.
//! * **screen** — logical points inside the widget, top-left origin. What iced
//!   hands us for hit-testing.
//! * **physical** — pixels in the Vello target. What the GPU sees.
//!
//! `screen = (world - pan) * zoom + size/2` and `physical = screen * scale`.
//! Nothing outside this file is allowed to know that.

use vello::kurbo::{Affine, Point, Vec2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    /// World point sitting at the centre of the widget.
    pub pan: Vec2,
    /// Logical points per world unit.
    pub zoom: f64,
}

impl Default for View {
    fn default() -> Self {
        Self {
            pan: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

const ZOOM_LIMITS: (f64, f64) = (0.02, 400.0);

impl View {
    pub fn pan_by_screen(&mut self, delta: Vec2) {
        self.pan -= delta / self.zoom;
    }

    /// Zooms about a point in the widget, so whatever is under the cursor
    /// stays under the cursor.
    pub fn zoom_about(&mut self, factor: f64, screen: Point, size: Vec2) {
        let before = self.screen_to_world(screen, size);
        self.zoom = (self.zoom * factor).clamp(ZOOM_LIMITS.0, ZOOM_LIMITS.1);
        let after = self.screen_to_world(screen, size);
        self.pan += before - after;
    }

    pub fn world_to_screen(&self, world: Point, size: Vec2) -> Point {
        Point::new(
            (world.x - self.pan.x) * self.zoom + size.x / 2.0,
            (world.y - self.pan.y) * self.zoom + size.y / 2.0,
        )
    }

    pub fn screen_to_world(&self, screen: Point, size: Vec2) -> Point {
        Point::new(
            (screen.x - size.x / 2.0) / self.zoom + self.pan.x,
            (screen.y - size.y / 2.0) / self.zoom + self.pan.y,
        )
    }

    /// World -> the physical pixel in the Vello target. This is the only place
    /// the HiDPI scale factor enters the 2D pipeline.
    pub fn world_to_physical(&self, world: Point, size: Vec2, scale_factor: f64) -> Point {
        let screen = self.world_to_screen(world, size);
        Point::new(screen.x * scale_factor, screen.y * scale_factor)
    }

    pub fn physical_to_world(&self, physical: Point, size: Vec2, scale_factor: f64) -> Point {
        self.screen_to_world(
            Point::new(physical.x / scale_factor, physical.y / scale_factor),
            size,
        )
    }

    /// The affine to hand Vello. The target is in physical pixels, so the
    /// scale factor is baked in here — and nowhere else.
    pub fn scene_transform(&self, size: Vec2, scale_factor: f64) -> Affine {
        Affine::scale(scale_factor)
            * Affine::translate((size.x / 2.0, size.y / 2.0))
            * Affine::scale(self.zoom)
            * Affine::translate((-self.pan.x, -self.pan.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const SCALES: [f64; 4] = [1.0, 1.5, 2.0, 3.0];
    const ZOOMS: [f64; 5] = [0.05, 0.5, 1.0, 7.5, 120.0];
    const SIZES: [(f64, f64); 3] = [(800.0, 600.0), (333.0, 91.0), (17.0, 4096.0)];

    fn views() -> Vec<View> {
        let mut out = Vec::new();
        for zoom in ZOOMS {
            for pan in [Vec2::ZERO, Vec2::new(-31.5, 12.25), Vec2::new(900.0, -4.0)] {
                out.push(View { pan, zoom });
            }
        }
        out
    }

    #[test]
    fn world_screen_round_trip() {
        for view in views() {
            for (w, h) in SIZES {
                let size = Vec2::new(w, h);
                for i in 0..13 {
                    let world = Point::new(i as f64 * 3.7 - 20.0, i as f64 * -1.9);
                    let back = view.screen_to_world(view.world_to_screen(world, size), size);
                    assert_abs_diff_eq!(back.x, world.x, epsilon = 1e-9);
                    assert_abs_diff_eq!(back.y, world.y, epsilon = 1e-9);
                }
            }
        }
    }

    #[test]
    fn world_physical_round_trip_at_every_scale_factor() {
        // The HiDPI bug this pins: a stray `* scale_factor` in the hit-test
        // path shows up here as a factor-of-two error at Retina.
        for view in views() {
            for (w, h) in SIZES {
                let size = Vec2::new(w, h);
                for scale in SCALES {
                    for i in 0..13 {
                        let world = Point::new(i as f64 * 2.3, 40.0 - i as f64);
                        let physical = view.world_to_physical(world, size, scale);
                        let back = view.physical_to_world(physical, size, scale);
                        assert_abs_diff_eq!(back.x, world.x, epsilon = 1e-9);
                        assert_abs_diff_eq!(back.y, world.y, epsilon = 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn the_scene_transform_agrees_with_world_to_physical() {
        // Vello draws with `scene_transform`; hit-testing uses
        // `world_to_physical`. If they ever disagree, clicks miss the shape
        // that is drawn under the cursor — silently.
        for view in views() {
            for (w, h) in SIZES {
                let size = Vec2::new(w, h);
                for scale in SCALES {
                    let affine = view.scene_transform(size, scale);
                    for i in 0..7 {
                        let world = Point::new(i as f64 * 5.5 - 9.0, 3.0);
                        let drawn = affine * world;
                        let hit = view.world_to_physical(world, size, scale);
                        assert_abs_diff_eq!(drawn.x, hit.x, epsilon = 1e-6);
                        assert_abs_diff_eq!(drawn.y, hit.y, epsilon = 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor() {
        let size = Vec2::new(800.0, 600.0);
        let cursor = Point::new(613.0, 227.0);
        let mut view = View::default();
        let anchor = view.screen_to_world(cursor, size);

        for factor in [1.1, 1.1, 1.1, 0.5, 2.7] {
            view.zoom_about(factor, cursor, size);
            let now = view.screen_to_world(cursor, size);
            assert_abs_diff_eq!(now.x, anchor.x, epsilon = 1e-9);
            assert_abs_diff_eq!(now.y, anchor.y, epsilon = 1e-9);
        }
    }

    #[test]
    fn zoom_is_clamped() {
        let size = Vec2::new(800.0, 600.0);
        let mut view = View::default();
        for _ in 0..500 {
            view.zoom_about(2.0, Point::new(1.0, 1.0), size);
        }
        assert_eq!(view.zoom, ZOOM_LIMITS.1);
        for _ in 0..5000 {
            view.zoom_about(0.5, Point::new(1.0, 1.0), size);
        }
        assert_eq!(view.zoom, ZOOM_LIMITS.0);
    }

    #[test]
    fn panning_moves_the_world_the_other_way() {
        let size = Vec2::new(800.0, 600.0);
        let mut view = View {
            zoom: 2.0,
            ..Default::default()
        };
        let before = view.world_to_screen(Point::ORIGIN, size);
        view.pan_by_screen(Vec2::new(100.0, 0.0));
        let after = view.world_to_screen(Point::ORIGIN, size);
        assert_abs_diff_eq!(after.x - before.x, 100.0, epsilon = 1e-9);
    }
}
