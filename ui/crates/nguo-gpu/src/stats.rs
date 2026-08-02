// SPDX-License-Identifier: GPL-2.0-or-later
//! Frame timing for the debug overlay: 8.3 ms target, 16.6 ms ceiling.

use std::time::Instant;

/// Exponentially smoothed frame time. Lives in the app state, not a global.
#[derive(Debug, Clone)]
pub struct FrameStats {
    last: Option<Instant>,
    frame_ms: f32,
    pub frames: u64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            last: None,
            frame_ms: 0.0,
            frames: 0,
        }
    }
}

impl FrameStats {
    /// Call once per presented frame.
    pub fn tick(&mut self, now: Instant) {
        if let Some(last) = self.last {
            let ms = now.duration_since(last).as_secs_f32() * 1000.0;
            // 0.1 smoothing: readable numbers, still reacts within ~10 frames.
            self.frame_ms = if self.frame_ms == 0.0 {
                ms
            } else {
                self.frame_ms * 0.9 + ms * 0.1
            };
        }
        self.last = Some(now);
        self.frames += 1;
    }

    pub fn frame_ms(&self) -> f32 {
        self.frame_ms
    }

    pub fn fps(&self) -> f32 {
        if self.frame_ms > 0.0 {
            1000.0 / self.frame_ms
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn converges_on_a_steady_frame_rate() {
        let mut stats = FrameStats::default();
        let mut now = Instant::now();
        for _ in 0..200 {
            stats.tick(now);
            now += Duration::from_micros(16_667);
        }
        assert!(
            (stats.frame_ms() - 16.667).abs() < 0.1,
            "got {}",
            stats.frame_ms()
        );
        assert!((stats.fps() - 60.0).abs() < 0.5, "got {}", stats.fps());
        assert_eq!(stats.frames, 200);
    }

    #[test]
    fn first_frame_has_no_delta() {
        let mut stats = FrameStats::default();
        stats.tick(Instant::now());
        assert_eq!(stats.frame_ms(), 0.0);
        assert_eq!(stats.fps(), 0.0);
    }
}
