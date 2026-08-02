// SPDX-License-Identifier: GPL-2.0-or-later
//! HiDPI: iced lays out in **logical** points, GPU targets are sized in
//! **physical** pixels.
//!
//! Every conversion between the two lives here so there is exactly one place
//! to be wrong.

/// A rectangle in physical pixels, ready for `set_viewport` / `set_scissor_rect`
/// / texture allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhysicalRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PhysicalRect {
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Converts an iced (logical) widget rectangle to physical pixels.
///
/// Rounds the *edges*, not the size, so that adjacent widgets stay gap-free and
/// a widget never spills a pixel outside its layout box.
pub fn physical_rect(x: f32, y: f32, width: f32, height: f32, scale_factor: f32) -> PhysicalRect {
    let left = (x * scale_factor).round().max(0.0);
    let top = (y * scale_factor).round().max(0.0);
    let right = ((x + width) * scale_factor).round().max(left);
    let bottom = ((y + height) * scale_factor).round().max(top);

    PhysicalRect {
        x: left as u32,
        y: top as u32,
        width: (right - left) as u32,
        height: (bottom - top) as u32,
    }
}

/// Physical size of a render target for a logical widget size.
///
/// Clamped to at least 1x1: wgpu rejects zero-sized textures and iced will
/// happily lay out a zero-height widget mid-resize.
pub fn physical_size(width: f32, height: f32, scale_factor: f32) -> (u32, u32) {
    let rect = physical_rect(0.0, 0.0, width, height, scale_factor);
    (rect.width.max(1), rect.height.max(1))
}

/// Physical -> logical, for turning a pixel measurement back into iced units.
pub fn to_logical(physical: f32, scale_factor: f32) -> f32 {
    physical / scale_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALES: [f32; 6] = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0];

    #[test]
    fn round_trip_within_half_a_physical_pixel() {
        for scale in SCALES {
            for i in 0..500 {
                let logical = i as f32 * 0.37;
                let physical = logical * scale;
                let back = to_logical(physical, scale);
                assert!(
                    (back - logical).abs() < 1e-3,
                    "scale {scale}: {logical} -> {physical} -> {back}"
                );
            }
        }
    }

    #[test]
    fn adjacent_widgets_leave_no_gap_and_no_overlap() {
        // The classic HiDPI bug: two side-by-side panels whose physical rects
        // either overlap by a pixel or leave a seam.
        for scale in SCALES {
            for i in 0..200 {
                let split = 13.3 + i as f32 * 0.5;
                let left = physical_rect(0.0, 0.0, split, 100.0, scale);
                let right = physical_rect(split, 0.0, 640.0 - split, 100.0, scale);

                assert_eq!(
                    left.x + left.width,
                    right.x,
                    "scale {scale}, split {split}: seam or overlap"
                );
                assert_eq!(
                    right.x + right.width,
                    (640.0 * scale).round() as u32,
                    "scale {scale}, split {split}: right edge drifted"
                );
            }
        }
    }

    #[test]
    fn targets_are_never_zero_sized() {
        for scale in SCALES {
            let (w, h) = physical_size(0.0, 0.0, scale);
            assert_eq!((w, h), (1, 1));
        }
    }

    #[test]
    fn retina_doubles_the_target() {
        assert_eq!(physical_size(800.0, 600.0, 2.0), (1600, 1200));
        assert_eq!(physical_size(800.0, 600.0, 1.0), (800, 600));
    }
}
