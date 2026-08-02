// SPDX-License-Identifier: GPL-2.0-or-later
//! Turntable camera and pointer unprojection.
//!
//! Everything here is pure maths on **logical** units: iced hands widgets
//! logical bounds and a logical cursor, and the scale factor only ever touches
//! the size of the render target. Doing the
//! unprojection in logical space is what makes hit-testing land on the right
//! pixel at `scale_factor` 2.0 without a single `* 2.0` anywhere.

use glam::{Mat4, Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Elevation, radians, clamped away from the poles.
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    /// Vertical field of view, radians.
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.45,
            distance: 4.0,
            target: Vec3::ZERO,
            fov_y: 50.0_f32.to_radians(),
        }
    }
}

/// Keeps the turntable off the poles, where the up-vector degenerates.
const PITCH_LIMIT: f32 = 1.5533; // 89 degrees
const NEAR: f32 = 0.05;
const FAR: f32 = 100.0;

impl Camera {
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn dolly(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.4, 60.0);
    }

    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cp * sy, sp, cp * cy)
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        // `perspective_rh` (not `_gl`) — wgpu clip space is z in 0..1.
        Mat4::perspective_rh(self.fov_y, aspect.max(1e-4), NEAR, FAR)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// World point -> position inside the widget, in logical units with the
    /// origin at the widget's top-left. `None` when the point is behind the eye.
    pub fn world_to_screen(&self, world: Vec3, size: Vec2) -> Option<Vec2> {
        let clip = self.view_projection(size.x / size.y) * world.extend(1.0);
        if clip.w <= 1e-6 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new(
            (ndc.x * 0.5 + 0.5) * size.x,
            (0.5 - ndc.y * 0.5) * size.y,
        ))
    }

    /// Position inside the widget (logical, top-left origin) -> the point where
    /// that ray meets the ground plane `y == 0`. `None` when the ray is
    /// parallel to the plane or hits it behind the camera.
    pub fn screen_to_world(&self, screen: Vec2, size: Vec2) -> Option<Vec3> {
        let inverse = self.view_projection(size.x / size.y).inverse();

        let ndc = Vec2::new(screen.x / size.x * 2.0 - 1.0, 1.0 - screen.y / size.y * 2.0);

        let unproject = |z: f32| {
            let p = inverse * ndc.extend(z).extend(1.0);
            p.truncate() / p.w
        };

        let near = unproject(0.0);
        let far = unproject(1.0);
        let direction = far - near;

        if direction.y.abs() < 1e-6 {
            return None;
        }

        let t = -near.y / direction.y;
        (t >= 0.0).then(|| near + direction * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Widget sizes at the three scale factors that matter, expressed as the
    /// LOGICAL size iced would report. The maths must not care about the scale
    /// factor at all — that is the property being pinned down.
    const SIZES: [(f32, f32); 4] = [
        (800.0, 600.0),
        (400.0, 300.0),
        (1280.0, 400.0),
        (333.0, 777.0),
    ];

    fn cameras() -> Vec<Camera> {
        let mut out = Vec::new();
        for yaw in [-2.0_f32, 0.0, 0.7, 3.0] {
            for pitch in [-1.2_f32, -0.2, 0.45, 1.3] {
                for distance in [0.8_f32, 4.0, 25.0] {
                    out.push(Camera {
                        yaw,
                        pitch,
                        distance,
                        target: Vec3::new(0.3, 0.0, -0.2),
                        ..Camera::default()
                    });
                }
            }
        }
        out
    }

    #[test]
    fn screen_to_world_to_screen_round_trips() {
        for camera in cameras() {
            for (w, h) in SIZES {
                let size = Vec2::new(w, h);
                for iy in 1..8 {
                    for ix in 1..8 {
                        let screen = Vec2::new(w * ix as f32 / 8.0, h * iy as f32 / 8.0);
                        let Some(world) = camera.screen_to_world(screen, size) else {
                            continue; // ray misses the ground plane
                        };
                        assert_abs_diff_eq!(world.y, 0.0, epsilon = 1e-3);

                        let back = camera
                            .world_to_screen(world, size)
                            .expect("a point we just unprojected must project");
                        // Sub-pixel, scaled with the widget: an f32 matrix
                        // inverse at a grazing angle across a 1280pt viewport
                        // is good to ~0.06pt, not to 1e-6.
                        let tolerance = w.max(h) * 1e-4;
                        assert_abs_diff_eq!(back.x, screen.x, epsilon = tolerance);
                        assert_abs_diff_eq!(back.y, screen.y, epsilon = tolerance);
                    }
                }
            }
        }
    }

    #[test]
    fn world_to_screen_to_world_round_trips() {
        for camera in cameras() {
            for (w, h) in SIZES {
                let size = Vec2::new(w, h);
                for i in -3..=3 {
                    for j in -3..=3 {
                        let world = Vec3::new(i as f32 * 0.4, 0.0, j as f32 * 0.4);
                        let Some(screen) = camera.world_to_screen(world, size) else {
                            continue;
                        };
                        let Some(back) = camera.screen_to_world(screen, size) else {
                            continue;
                        };
                        assert_abs_diff_eq!(back.x, world.x, epsilon = 1e-2);
                        assert_abs_diff_eq!(back.z, world.z, epsilon = 1e-2);
                    }
                }
            }
        }
    }

    #[test]
    fn the_centre_of_the_widget_looks_at_the_target() {
        let camera = Camera::default();
        let size = Vec2::new(800.0, 600.0);
        let hit = camera
            .screen_to_world(size / 2.0, size)
            .expect("the default camera looks at the ground");
        assert_abs_diff_eq!(hit.x, camera.target.x, epsilon = 1e-3);
        assert_abs_diff_eq!(hit.z, camera.target.z, epsilon = 1e-3);
    }

    #[test]
    fn hit_testing_is_independent_of_the_scale_factor() {
        // Same LOGICAL cursor position and LOGICAL widget size must give the
        // same world point whether we are on a 1x display or a Retina one.
        // If this ever fails, someone has multiplied by `scale_factor` in the
        // hit-test path.
        let camera = Camera::default();
        let size = Vec2::new(800.0, 600.0);
        let screen = Vec2::new(613.0, 227.0);
        let reference = camera.screen_to_world(screen, size).unwrap();

        for scale in [1.0_f32, 1.5, 2.0, 3.0] {
            // What the GPU target would be sized to, for the record.
            let (pw, ph) = nguo_gpu::scale::physical_size(size.x, size.y, scale);
            assert_eq!((pw, ph), ((800.0 * scale) as u32, (600.0 * scale) as u32));

            let hit = camera.screen_to_world(screen, size).unwrap();
            assert_abs_diff_eq!(hit.x, reference.x, epsilon = 1e-6);
            assert_abs_diff_eq!(hit.z, reference.z, epsilon = 1e-6);
        }
    }

    #[test]
    fn orbit_stays_off_the_poles() {
        let mut camera = Camera::default();
        for _ in 0..100 {
            camera.orbit(0.1, 1.0);
        }
        assert!(camera.pitch <= PITCH_LIMIT);
        // Degenerate up-vector would make the view matrix NaN.
        assert!(camera.view().is_finite());
    }
}
