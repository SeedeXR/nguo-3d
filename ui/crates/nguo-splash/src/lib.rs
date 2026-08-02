// SPDX-License-Identifier: GPL-2.0-or-later
//! The nguo-3d splash screen.
//!
//! Two things live here and they are deliberately kept apart:
//!
//! * [`Splash`] — the dismissal state machine. Pure. It never reads the clock,
//!   never touches a window, and is unit-tested without either. Every method
//!   that cares about time takes the [`Instant`] from the caller.
//! * [`Splash::view`] — the iced widget tree. Presentation only; it reads the
//!   state machine and returns an [`Element`].
//!
//! # Semantics (Photoshop / Lightroom)
//!
//! * The splash is visible for **at least [`MIN_VISIBLE`] (4 s)**. A splash
//!   that flashes past reads as a glitch, so a 200 ms load still gets 4 s.
//! * It stays **longer** if initialisation has not finished: the dismissal
//!   instant is `max(shown_at + MIN_VISIBLE, load_complete)`.
//! * There is **no click-to-dismiss**. Dismissal is `max(4 s, load)` and a
//!   click cannot move that instant in either direction — it may not rush the
//!   teardown of a half-initialised app, and it cannot beat the auto-dismiss
//!   that already fires on the next frame. A handler that changes no outcome
//!   is worse than none, so the splash has no click affordance at all.
//!   See `ui/docs/SPLASH.md` §2.
//!
//! # Who decides to dismiss
//!
//! Only [`Splash::tick`] can dismiss. [`Splash::report`] and [`Splash::loaded`]
//! are a *data feed* from the loader; they record what the app has achieved and
//! nothing else. Keeping the decision in one place is what makes the machine
//! testable — and it means the loader thread can report completion without a
//! window teardown happening inside its call stack.
//!
//! Consequence for the caller: after `loaded()` the splash goes away on the
//! next tick. Call `tick(Instant::now())` straight after `loaded()` if you want
//! it gone on the same frame instead of within one tick period.
//!
//! # Progress is real or it is not shown
//!
//! [`Splash::report`] is monotonic and clamped below 1.0. The bar cannot go
//! backwards and cannot read 100 % before [`Splash::loaded`] has been called.
//! There is no timer-driven fake animation anywhere in this crate.

use std::sync::LazyLock;
use std::time::{Duration, Instant};

use iced::widget::{column, container, image, row, svg, text, Space};
use iced::{Color, ContentFit, Element, Font, Length, Padding, Subscription};

/// Minimum time the splash stays on screen, however fast the app loads.
pub const MIN_VISIBLE: Duration = Duration::from_secs(4);

/// Window size the layout is designed for. 1.685:1, matching the reference
/// render (752x446).
pub const SIZE: iced::Size = iced::Size::new(900.0, 534.0);

// Assets. `include_bytes!` on purpose: a missing or renamed asset must fail the
// build, not produce a blank panel at runtime.
const ARTWORK_PNG: &[u8] = include_bytes!("../../../assets/images/splash_screen_image.png");
const BUTTERFLY_SVG: &[u8] = include_bytes!("../../../assets/brand-icon/butterfly-nguo3d-icon.svg");

/// Built **once**, not per frame.
///
/// `image::Handle::from_bytes` stamps a fresh `Id::unique()` on every call
/// (`iced_core-0.14.0/src/image.rs:136`) — the identity is the handle, not the
/// bytes. Raster decoding is asynchronous (`iced_wgpu-0.14.0/src/image/
/// cache.rs:400`, worker thread), so a handle rebuilt inside `view` is a brand
/// new cache miss on every frame: the decode is queued, the frame draws
/// nothing, the handle is thrown away, repeat. The artwork panel stays black
/// forever and nothing errors. SVG handles hash their data
/// (`iced_core-0.14.0/src/svg.rs:93`) and so do not have this problem, which is
/// exactly why the butterfly rendered and the artwork did not.
static ARTWORK: LazyLock<image::Handle> = LazyLock::new(|| image::Handle::from_bytes(ARTWORK_PNG));

/// Inter Regular, SIL OFL 1.1. Vendored — see `ui/assets/fonts/LICENSE.md`.
///
/// The shell **must** register this with the iced runtime before the splash is
/// shown, or every line falls back to the platform sans:
/// `iced::application(..).font(nguo_splash::INTER_TTF)`.
pub const INTER_TTF: &[u8] = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");

/// The face [`INTER_TTF`] registers as. All splash text uses it.
pub const INTER: Font = Font::with_name("Inter");

// ---------------------------------------------------------------------------
// Palette — see ui/docs/SPLASH.md. This screen is the ONE documented exception
// to "no hardcoded colours" (ui/docs/DEVELOPING.md §4.7): it is a fixed black-on-
// black brand plate that must not follow the user's light/dark theme, and it
// renders before `nguo-ui-theme` has necessarily loaded a theme at all.
// Values sampled from the reference render.
// ---------------------------------------------------------------------------
// #000000, sampled at (5,5).
const INK: Color = Color::BLACK;
// #ffffff — the wordmark stroke and the butterfly, the only full-contrast marks.
const PAPER: Color = Color::WHITE;
// #909090, sampled off the reference status line. Every line of type on this
// screen is this grey (maintainer's call), so nothing competes with the mark.
const MUTED: Color = Color::from_rgb(0.565, 0.565, 0.565);
// #787878 — one step quieter, for the closing line only. Same face, same size:
// the only thing separating it from the legal block is that it recedes.
const QUIET: Color = Color::from_rgb(0.471, 0.471, 0.471);

/// The wordmark, as a stroked (hollow) display face.
///
/// iced's text widget cannot stroke glyphs, so the wordmark is an SVG: usvg
/// converts `<text>` to glyph outlines and `stroke` + `fill="none"` gives the
/// true outlined face in the reference. iced's SVG rasteriser calls
/// `fontdb.load_system_fonts()` (`iced_wgpu-0.14.0/src/image/vector.rs:56`), so
/// the family list below resolves against installed fonts on every platform.
const WORDMARK_SVG: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="340" height="66" viewBox="0 0 340 66">
<text x="2" y="50" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linejoin="round"
  font-size="52" font-weight="700" letter-spacing="0.5"
  font-family="Helvetica Neue,Helvetica,Arial,Liberation Sans,DejaVu Sans,sans-serif">Nguo - 3d</text>
</svg>"##;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// The splash's dismissal state machine.
///
/// Construct with [`Splash::new`], feed it progress with [`Splash::report`] /
/// [`Splash::loaded`], drive it with [`Splash::tick`].
#[derive(Debug, Clone)]
pub struct Splash {
    shown_at: Instant,
    minimum: Duration,
    /// Set by `loaded()`. Load completion is a fact, not an instant: the
    /// dismissal time is decided by whichever tick observes it.
    load_complete: bool,
    dismissed: bool,
    fraction: f32,
    stage: String,
}

impl Splash {
    /// Creates a splash shown at `now`.
    pub fn new(now: Instant) -> Self {
        Self {
            shown_at: now,
            minimum: MIN_VISIBLE,
            load_complete: false,
            dismissed: false,
            fraction: 0.0,
            stage: String::new(),
        }
    }

    /// Overrides the minimum visible duration. Exists for the tests and the
    /// demo; product code should use the default.
    #[must_use]
    pub fn with_minimum(mut self, minimum: Duration) -> Self {
        self.minimum = minimum;
        self
    }

    /// Records a genuine initialisation milestone: what just finished, and how
    /// far through startup that leaves us.
    ///
    /// Monotonic — a lower `fraction` than the one already shown is ignored —
    /// and clamped to 0.99 so the line cannot read 100 % before
    /// [`Splash::loaded`].
    pub fn report(&mut self, stage: impl Into<String>, fraction: f32) {
        self.stage = stage.into();

        if self.load_complete {
            return;
        }

        let clamped = fraction.clamp(0.0, 0.99);
        self.fraction = self.fraction.max(clamped);
    }

    /// Records that initialisation is complete. Does not dismiss on its own;
    /// the next [`Splash::tick`] does.
    pub fn loaded(&mut self) {
        self.load_complete = true;
        self.fraction = 1.0;
    }

    /// Advances the machine. Returns `true` on the tick that dismisses.
    pub fn tick(&mut self, now: Instant) -> bool {
        if self.dismissed || !self.may_dismiss(now) {
            return false;
        }

        self.dismissed = true;
        true
    }

    /// Whether both dismissal conditions hold: minimum elapsed **and** loading
    /// complete.
    pub fn may_dismiss(&self, now: Instant) -> bool {
        self.load_complete && now.duration_since(self.shown_at) >= self.minimum
    }

    /// Whether the splash has been dismissed and should be torn down.
    pub fn is_dismissed(&self) -> bool {
        self.dismissed
    }

    /// Real progress in `0.0..=1.0`. Only ever 1.0 once [`Splash::loaded`] has
    /// been called.
    pub fn progress(&self) -> f32 {
        self.fraction
    }

    /// The raw milestone label last reported.
    pub fn stage(&self) -> &str {
        &self.stage
    }

    /// The status line as rendered.
    pub fn status(&self) -> String {
        if self.load_complete {
            return "Ready".to_owned();
        }

        let percent = (self.fraction * 100.0).round() as u32;

        if self.stage.is_empty() {
            "Loading …".to_owned()
        } else {
            format!("Loading … {}  ·  {percent}%", self.stage)
        }
    }

    /// The clock the shell should drive [`Splash::tick`] with:
    /// `Splash::ticks().map(Message::Tick)`.
    ///
    /// Built on [`iced::window::frames`]
    /// (`iced_runtime-0.14.0/src/window.rs:208`), which yields the redraw
    /// instant at the window's refresh rate. No timer backend needed, so the
    /// crate does not drag `tokio` in just for a clock — and the dismissal lag
    /// is one frame, not one timer period.
    pub fn ticks() -> Subscription<Instant> {
        iced::window::frames()
    }

    // -----------------------------------------------------------------------
    // View
    // -----------------------------------------------------------------------

    /// The splash widget tree. Display only — there is nothing to click.
    pub fn view<'a, Message: 'a>(&self) -> Element<'a, Message> {
        let panel = column![
            svg(svg::Handle::from_memory(WORDMARK_SVG))
                .width(340)
                .height(66),
            // Fill: centres the mark in whatever is left between the wordmark
            // and the type block, which is where the reference puts it.
            container(
                svg(svg::Handle::from_memory(BUTTERFLY_SVG))
                    .width(260)
                    .height(260)
                    .style(|_theme, _status| svg::Style { color: Some(PAPER) })
            )
            .center_x(Length::Fill)
            .center_y(Length::Fill),
            // Every line of type gathers here, at the bottom, small and grey.
            self.small_print(),
        ]
        // Columns and rows default to `Shrink` (`iced_widget-0.14.2/src/
        // row.rs:81`). Without this the panel sizes to its longest text line
        // instead of to half the window, and the artwork gets zero width.
        .width(Length::Fill);

        let left = container(panel)
            .width(Length::FillPortion(1))
            .height(Length::Fill)
            .padding(Padding {
                top: 34.0,
                right: 36.0,
                bottom: 32.0,
                left: 36.0,
            })
            .style(|_theme| container::Style {
                background: Some(INK.into()),
                ..container::Style::default()
            });

        // Right half, edge to edge. `Cover` crops the square artwork to the
        // panel instead of letterboxing it; the container clips the overflow.
        let artwork = container(
            image(ARTWORK.clone())
                .width(Length::Fill)
                .height(Length::Fill)
                .content_fit(ContentFit::Cover),
        )
        .width(Length::FillPortion(1))
        .height(Length::Fill)
        .clip(true);

        container(row![left, artwork].width(Length::Fill).height(Length::Fill))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme| container::Style {
                background: Some(INK.into()),
                ..container::Style::default()
            })
            .into()
    }

    /// The whole type block: live status, credits, legal, closing line. All of
    /// it small print, all of it grey, all of it at the bottom of the panel.
    fn small_print<'a, Message: 'a>(&self) -> Element<'a, Message> {
        let line = |s: String| text(s).size(10).font(INTER).color(MUTED);

        column![
            line(self.status()),
            Space::new().height(14),
            line("Artwork by Alex Mkwizu & MidJourney AI".to_owned()),
            Space::new().height(10),
            line("© 2026 Seede XR. All rights reserved.".to_owned()),
            // Corrected from the reference render, which reads "Fore more
            // details" and "About Optima Screen". See ui/docs/SPLASH.md — both
            // are errors and must not be "restored".
            line("For more details and legal notices, go to the About nguo-3d screen.".to_owned()),
            Space::new().height(12),
            // Inter like everything above it; only the grey sets it apart.
            text("Built for the community")
                .size(10)
                .font(INTER)
                .color(QUIET),
        ]
        .spacing(2)
        .into()
    }
}

// ---------------------------------------------------------------------------
// Tests — no window, no sleeps. The clock is a local `Instant` plus offsets.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test clock origin. Never advanced by the environment: every later
    /// instant is `t0 + secs(..)`, so the suite runs in microseconds.
    fn t0() -> Instant {
        Instant::now()
    }

    fn at(t0: Instant, secs: f64) -> Instant {
        t0 + Duration::from_secs_f64(secs)
    }

    #[test]
    fn fast_load_still_waits_for_the_minimum() {
        let t0 = t0();
        let mut splash = Splash::new(t0);

        splash.report("GPU device", 0.5);
        splash.loaded();

        assert!(!splash.tick(at(t0, 0.2)), "must not dismiss at 0.2 s");
        assert!(!splash.tick(at(t0, 3.999)), "must not dismiss before 4 s");
        assert!(splash.tick(at(t0, 4.0)), "must dismiss at exactly 4 s");
        assert!(splash.is_dismissed());
    }

    #[test]
    fn slow_load_holds_the_splash_past_the_minimum() {
        let t0 = t0();
        let mut splash = Splash::new(t0);

        splash.report("Shaders", 0.4);
        assert!(
            !splash.tick(at(t0, 4.0)),
            "4 s is not enough, load is running"
        );
        assert!(!splash.tick(at(t0, 8.9)));

        splash.loaded();
        assert!(splash.tick(at(t0, 9.0)), "must dismiss at load completion");
    }

    /// The behaviour a click used to claim, with no click involved: an
    /// impatient user gets exactly what a patient one gets, which is why the
    /// click handler was removed rather than kept as a no-op.
    #[test]
    fn dismissal_is_the_instant_both_conditions_hold() {
        let t0 = t0();
        let mut splash = Splash::new(t0);

        assert!(!splash.tick(at(t0, 4.0)), "loading is not done");
        splash.loaded();
        assert!(splash.tick(at(t0, 4.1)));
    }

    #[test]
    fn dismissal_happens_once() {
        let t0 = t0();
        let mut splash = Splash::new(t0);
        splash.loaded();

        assert!(splash.tick(at(t0, 4.0)));
        assert!(!splash.tick(at(t0, 4.1)), "no second dismissal edge");
    }

    #[test]
    fn progress_never_goes_backwards() {
        let mut splash = Splash::new(t0());

        splash.report("GPU device", 0.30);
        assert_eq!(splash.progress(), 0.30);

        splash.report("Shaders", 0.10);
        assert_eq!(splash.progress(), 0.30, "a lower report must not rewind");
        assert_eq!(splash.stage(), "Shaders", "but the label still updates");

        splash.report("Fonts", 0.80);
        assert_eq!(splash.progress(), 0.80);
    }

    #[test]
    fn progress_never_reaches_one_before_completion() {
        let mut splash = Splash::new(t0());

        splash.report("Samples", 1.0);
        assert!(splash.progress() < 1.0);
        splash.report("Samples", 42.0);
        assert!(splash.progress() < 1.0);
        assert!(!splash.status().contains("100%"));

        splash.loaded();
        assert_eq!(splash.progress(), 1.0);
        assert_eq!(splash.status(), "Ready");

        // A late straggler report cannot pull it back off 100 %.
        splash.report("Straggler", 0.2);
        assert_eq!(splash.progress(), 1.0);
    }

    #[test]
    fn status_line_reports_the_real_stage() {
        let mut splash = Splash::new(t0());
        assert_eq!(splash.status(), "Loading …");

        splash.report("GPU device", 0.4);
        assert_eq!(splash.status(), "Loading … GPU device  ·  40%");
    }

    #[test]
    fn minimum_is_configurable_but_still_enforced() {
        let t0 = t0();
        let mut splash = Splash::new(t0).with_minimum(Duration::from_millis(500));
        splash.loaded();

        assert!(!splash.tick(at(t0, 0.4)));
        assert!(splash.tick(at(t0, 0.5)));
    }
}
