// SPDX-License-Identifier: GPL-2.0-or-later
//! The ground-plane grid: **what** to draw, in metres. `grid.wgsl` does the
//! drawing; everything that can be wrong about a grid is decided here, where a
//! test can see it.
//!
//! Modelled on Blender's viewport floor, from the bundled manual
//! (`data/manual/editors/3dview/display/overlays.rst`):
//!
//! - **Floor** (`show_floor`, line 42) — "Show the ground plane in perspective
//!   view." A plane, not a finite quad: it is the ray-plane intersection of
//!   every pixel, so it never has an edge you can fall off.
//! - **Axes** (`show_axis`, line 47) — "Show the X, Y and/or Z axis lines."
//!   Our floor is the XZ plane, so the two lines in it are X and Z.
//! - **Scale** (`grid_scale`, line 52) — "The distance between lines in the
//!   grid/floor", and **Subdivisions** (`grid_subdivisions`, line 57) — "The
//!   number of subdivisions between grid lines". Blender exposes both as
//!   settings; we derive the scale from the camera instead, so the density on
//!   screen stays put while the *world* meaning of a cell changes with zoom.
//!
//! Units are **metres**, because the whole tree is: the avatar bundle is
//! metres, +Y up, +Z forward, and nothing on this path converts anything
//! (`AGENT_BRIEF.md` §2b.6). Blender's own `scale_length`
//! (`data/manual/scene_layout/scene/properties.rst:67`) is 1.0 for the same
//! reason.

use crate::camera::{Camera, FAR};

/// On-screen spacing a minor line aims for, logical px. Below this a grid
/// reads as a grey wash.
pub const TARGET_MINOR_PX: f32 = 14.0;

/// Minor lines per major line — Blender's *Subdivisions*, default 10.
pub const MINORS_PER_MAJOR: f32 = 10.0;

/// How far past the pivot the floor survives, as a multiple of the camera
/// distance. Beyond it the grid is fully faded rather than ending on a hard
/// edge you can see.
const FADE_SPAN: f32 = 9.0;

/// A 1-2-5 decade step (metres) whose on-screen spacing is at least
/// `target_px`.
///
/// ponytail: `nguo_canvas::grid::step_mm` is the same rule in mm, and this is
/// deliberately a second copy rather than a dependency — reaching it would
/// pull `nguo-doc`, `nguo-pattern`, `nguo-doc-commands` and `vello` into the
/// 3D viewport for eight lines of arithmetic it would otherwise never link.
/// Hoist both into a shared units crate the next time either one changes; the
/// rule (1-2-5, decade-stepped) is pinned by a test on each side.
pub fn step_at(px_per_metre: f32, target_px: f32) -> f32 {
    // `is_finite` first, so NaN takes the fallback rather than falling through
    // a comparison that is false for every operand.
    if !px_per_metre.is_finite()
        || px_per_metre <= 0.0
        || !target_px.is_finite()
        || target_px <= 0.0
    {
        return 1.0;
    }
    let raw = (target_px / px_per_metre) as f64;
    let decade = 10f64.powf(raw.log10().floor());
    let m = raw / decade;
    let step = decade
        * if m <= 1.0 {
            1.0
        } else if m <= 2.0 {
            2.0
        } else if m <= 5.0 {
            5.0
        } else {
            10.0
        };
    let step = step as f32;
    if step.is_finite() && step > 0.0 {
        step
    } else {
        1.0
    }
}

/// Logical pixels per metre **at the pivot**, which is where the grid is being
/// looked at and therefore the distance its density should be judged from.
pub fn pixels_per_metre(camera: &Camera, height_px: f32) -> f32 {
    let world_height = 2.0 * camera.distance * (camera.fov_y * 0.5).tan();
    if !world_height.is_finite()
        || world_height <= 0.0
        || !height_px.is_finite()
        || height_px <= 0.0
    {
        return 0.0;
    }
    height_px / world_height
}

/// What one frame of floor looks like.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Floor {
    /// Minor cell, metres.
    pub minor_m: f32,
    /// Major cell, metres — [`MINORS_PER_MAJOR`] minors.
    pub major_m: f32,
    /// Distance from the eye at which the floor has faded to nothing.
    pub fade_m: f32,
}

impl Floor {
    /// Derive the floor from the camera and the widget height in logical px.
    pub fn for_camera(camera: &Camera, height_px: f32) -> Floor {
        let minor_m = step_at(pixels_per_metre(camera, height_px), TARGET_MINOR_PX);
        Floor {
            minor_m,
            major_m: minor_m * MINORS_PER_MAJOR,
            // Never past the far plane: geometry the projection has already
            // clipped cannot be faded, and a fade that ends outside the
            // frustum is a fade nobody sees.
            fade_m: (camera.distance * FADE_SPAN).clamp(1.0, FAR),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn camera(distance: f32) -> Camera {
        Camera {
            distance,
            ..Camera::default()
        }
    }

    /// The point of an adaptive grid: zoom anywhere and the lines stay roughly
    /// [`TARGET_MINOR_PX`] apart on screen. A fixed step fails this at both
    /// ends — that is what it is here to catch.
    #[test]
    fn the_step_keeps_screen_density_roughly_constant() {
        for distance in [0.4_f32, 1.0, 2.5, 4.0, 12.0, 30.0, 60.0] {
            for height_px in [240.0_f32, 648.0, 1400.0] {
                let camera = camera(distance);
                let floor = Floor::for_camera(&camera, height_px);
                let on_screen = floor.minor_m * pixels_per_metre(&camera, height_px);
                assert!(
                    (TARGET_MINOR_PX..TARGET_MINOR_PX * 10.0).contains(&on_screen),
                    "distance {distance} in {height_px}px: {} m reads {on_screen} px",
                    floor.minor_m
                );
            }
        }
    }

    #[test]
    fn the_step_is_always_a_1_2_5_decade() {
        for distance in [0.4_f32, 0.9, 3.3, 7.0, 41.0, 60.0] {
            let floor = Floor::for_camera(&camera(distance), 648.0);
            let decade = 10f32.powf(floor.minor_m.log10().floor());
            let m = floor.minor_m / decade;
            assert!(
                [1.0, 2.0, 5.0].iter().any(|x| (m - x).abs() < 1e-4),
                "distance {distance}: {} m has mantissa {m}",
                floor.minor_m
            );
            assert!((floor.major_m / floor.minor_m - MINORS_PER_MAJOR).abs() < 1e-3);
        }
    }

    /// A body is 1.8 m tall and `frame_standing` is the shipped opening view.
    /// The cell it picks has to be one a human reads as a round measurement —
    /// if this ever says 0.037 m the decade rule has been broken.
    #[test]
    fn the_default_body_view_gets_a_round_metric_cell() {
        let mut camera = Camera::default();
        camera.frame_standing(1.83);
        let floor = Floor::for_camera(&camera, 780.0);
        // Measured, not chosen: 10 cm cells with a 1 m major line, for a
        // 1.83 m body in the shell's 780 px-tall 3D pane.
        assert_eq!(floor.minor_m, 0.1, "minor cell");
        assert_eq!(floor.major_m, 1.0, "major cell");
    }

    /// Zooming out must never make the cell *smaller*: the step is monotone in
    /// camera distance, which is what stops the grid pulsing during a dolly.
    #[test]
    fn the_cell_never_shrinks_as_the_camera_pulls_back() {
        let mut previous = 0.0_f32;
        let mut distance = 0.4_f32;
        while distance <= 60.0 {
            let floor = Floor::for_camera(&camera(distance), 648.0);
            assert!(
                floor.minor_m >= previous,
                "distance {distance}: {} m after {previous} m",
                floor.minor_m
            );
            previous = floor.minor_m;
            distance *= 1.05;
        }
    }

    /// Degenerate widgets and degenerate cameras must give a drawable floor,
    /// not a NaN uniform that turns the viewport into noise.
    #[test]
    fn a_degenerate_camera_still_yields_a_finite_floor() {
        for height_px in [0.0_f32, -10.0, f32::NAN, f32::INFINITY, 1e9] {
            for distance in [0.0_f32, -1.0, 1e12] {
                let floor = Floor::for_camera(&camera(distance), height_px);
                assert!(
                    floor.minor_m.is_finite() && floor.minor_m > 0.0,
                    "{height_px}px at {distance}m: {floor:?}"
                );
                assert!(floor.major_m.is_finite() && floor.major_m > 0.0);
                assert!(floor.fade_m.is_finite() && floor.fade_m > 0.0);
            }
        }
    }

    /// The fade must die inside the frustum. A floor faded out past the far
    /// plane is a floor with a visible hard edge where the projection clips it.
    #[test]
    fn the_floor_fades_out_before_the_far_plane() {
        for distance in [0.4_f32, 4.0, 60.0] {
            let floor = Floor::for_camera(&camera(distance), 648.0);
            assert!(floor.fade_m <= FAR, "{}", floor.fade_m);
            // ...and still reaches past the thing being looked at.
            assert!(floor.fade_m > distance, "{} vs {distance}", floor.fade_m);
        }
    }

    /// The plane the shader intersects is the one the camera's own
    /// unprojection lands on — `y == 0`, the bundle's floor, feet at zero.
    #[test]
    fn the_floor_is_the_plane_the_camera_unprojects_onto() {
        let camera = Camera::default();
        let size = glam::Vec2::new(800.0, 600.0);
        let hit = camera.screen_to_world(size * 0.5, size).unwrap();
        assert!(hit.y.abs() < 1e-4, "{hit:?}");
        assert!(camera
            .world_to_screen(Vec3::new(1.0, 0.0, 1.0), size)
            .is_some());
    }
}
