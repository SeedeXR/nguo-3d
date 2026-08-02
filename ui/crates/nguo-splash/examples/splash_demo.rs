// SPDX-License-Identifier: GPL-2.0-or-later
//! Standalone splash screen. `cargo run -p nguo-splash --example splash_demo`
//!
//! Shows nothing but the splash, driven by a **simulated** multi-stage load
//! that deliberately runs longer than the 4 s minimum so the "stays until
//! loading is done" branch is the one you actually see. It is not clickable —
//! it goes when `max(4 s, load)` is reached and not a frame before.

use std::time::Instant;

use iced::{Element, Subscription, Task};
use nguo_splash::Splash;

/// Simulated milestones: (elapsed seconds, label, fraction). These stand in for
/// the real ones `nguo-app` reports.
const STAGES: [(f64, &str, f32); 5] = [
    (0.5, "GPU device", 0.15),
    (1.6, "Shaders", 0.40),
    (2.9, "Theme", 0.60),
    (4.3, "Fonts", 0.80),
    (5.5, "Samples", 0.95),
];

/// Simulated total load time. Longer than `MIN_VISIBLE` on purpose.
const LOAD_SECS: f64 = 6.5;

#[derive(Debug, Clone)]
enum Message {
    Tick(Instant),
}

struct Demo {
    splash: Splash,
    started: Instant,
    next_stage: usize,
}

impl Demo {
    fn new() -> (Self, Task<Message>) {
        let now = Instant::now();

        (
            Self {
                splash: Splash::new(now),
                started: now,
                next_stage: 0,
            },
            // A splash that opens behind the window you were already using is
            // not a splash. `Level::AlwaysOnTop` alone did not do it here.
            iced::window::latest().and_then(iced::window::gain_focus),
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        let Message::Tick(now) = message;
        self.advance_fake_load(now);
        let dismissed = self.splash.tick(now);

        if dismissed {
            println!(
                "dismissed after {:.2} s",
                self.started.elapsed().as_secs_f64()
            );
            iced::exit()
        } else {
            Task::none()
        }
    }

    fn advance_fake_load(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.started).as_secs_f64();

        while let Some(&(at, label, fraction)) = STAGES.get(self.next_stage) {
            if elapsed < at {
                break;
            }

            self.splash.report(label, fraction);
            self.next_stage += 1;
        }

        if elapsed >= LOAD_SECS {
            self.splash.loaded();
        }
    }

    fn view(&self) -> Element<'_, Message> {
        self.splash.view()
    }

    fn subscription(&self) -> Subscription<Message> {
        Splash::ticks().map(Message::Tick)
    }
}

pub fn main() -> iced::Result {
    iced::application(Demo::new, Demo::update, Demo::view)
        .subscription(Demo::subscription)
        .font(nguo_splash::INTER_TTF)
        .title("nguo-3d")
        .window_size(nguo_splash::SIZE)
        .resizable(false)
        .decorations(false)
        .level(iced::window::Level::AlwaysOnTop)
        .centered()
        .run()
}
