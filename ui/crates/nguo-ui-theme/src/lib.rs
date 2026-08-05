// SPDX-License-Identifier: GPL-2.0-or-later
//! Design tokens for the nguo-3d UI.
//!
//! This crate is the **only** place a colour literal may appear. Everything
//! else — widgets, canvas, viewport, app chrome — reads a [`Tokens`] value.
//! That is what makes "restyle later without breaking functionality" true.
//!
//! Deliberately framework-free (no `iced` dependency) so the contrast tests
//! run in a second and the tokens can be reused by the Vello canvas and the
//! 3D viewport, which do not speak iced. `nguo-ui-widgets` owns the
//! `Rgba -> iced::Color` conversion.
//!
//! Design direction: dark-first professional tool chrome. Quiet neutral
//! surfaces, exactly one accent hue, precise 1px borders, generous hit
//! targets. The garment is the only saturated thing on
//! screen — which is why the canvas entity roles are the only chromatic
//! tokens outside `accent` and the status colours.

use std::path::Path;

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/// A non-linear sRGB colour with straight alpha, components in `0.0..=1.0`.
///
/// Serialises as a CSS-style hex string (`"#1c1f24"` / `"#1c1f24cc"`) so a
/// hand-edited theme file stays readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// `0xRRGGBB`, opaque.
    pub const fn hex(v: u32) -> Self {
        Self {
            r: ((v >> 16) & 0xff) as f32 / 255.0,
            g: ((v >> 8) & 0xff) as f32 / 255.0,
            b: (v & 0xff) as f32 / 255.0,
            a: 1.0,
        }
    }

    /// `0xRRGGBBAA`. Alpha is 8-bit so the value survives a JSON round-trip
    /// byte-for-byte — `0.35` would not.
    pub const fn hex_a(v: u32) -> Self {
        Self {
            r: ((v >> 24) & 0xff) as f32 / 255.0,
            g: ((v >> 16) & 0xff) as f32 / 255.0,
            b: ((v >> 8) & 0xff) as f32 / 255.0,
            a: (v & 0xff) as f32 / 255.0,
        }
    }

    /// Same colour at a different alpha (selection washes, disabled states).
    pub const fn alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// WCAG 2.x relative luminance of the *opaque* colour.
    pub fn relative_luminance(self) -> f32 {
        fn lin(c: f32) -> f32 {
            if c <= 0.040_45 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        0.2126 * lin(self.r) + 0.7152 * lin(self.g) + 0.0722 * lin(self.b)
    }

    /// Composite `self` over an opaque `bg` using straight alpha.
    pub fn over(self, bg: Rgba) -> Rgba {
        let a = self.a.clamp(0.0, 1.0);
        Rgba::new(
            self.r * a + bg.r * (1.0 - a),
            self.g * a + bg.g * (1.0 - a),
            self.b * a + bg.b * (1.0 - a),
            1.0,
        )
    }

    fn to_hex_string(self) -> String {
        let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        if self.a >= 1.0 {
            format!("#{:02x}{:02x}{:02x}", q(self.r), q(self.g), q(self.b))
        } else {
            format!(
                "#{:02x}{:02x}{:02x}{:02x}",
                q(self.r),
                q(self.g),
                q(self.b),
                q(self.a)
            )
        }
    }

    fn parse_hex(s: &str) -> Result<Self, String> {
        let h = s.strip_prefix('#').unwrap_or(s);
        if !h.is_char_boundary(0) || !h.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("bad hex colour {s:?}: non-hex digits"));
        }
        let n = |i: usize| -> Result<f32, String> {
            u8::from_str_radix(&h[i..i + 2], 16)
                .map(|v| v as f32 / 255.0)
                .map_err(|e| format!("bad hex colour {s:?}: {e}"))
        };
        match h.len() {
            6 => Ok(Rgba::new(n(0)?, n(2)?, n(4)?, 1.0)),
            8 => Ok(Rgba::new(n(0)?, n(2)?, n(4)?, n(6)?)),
            _ => Err(format!("bad hex colour {s:?}: want #rrggbb or #rrggbbaa")),
        }
    }
}

impl serde::Serialize for Rgba {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex_string())
    }
}

impl<'de> serde::Deserialize<'de> for Rgba {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = <String as serde::Deserialize>::deserialize(d)?;
        Rgba::parse_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// WCAG 2.x contrast ratio between two colours, `1.0..=21.0`.
///
/// `fg` is composited over `bg` first, so translucent text tokens are scored
/// as they actually render.
pub fn contrast_ratio(fg: Rgba, bg: Rgba) -> f32 {
    let l1 = fg.over(bg).relative_luminance();
    let l2 = bg.relative_luminance();
    let (hi, lo) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (hi + 0.05) / (lo + 0.05)
}

// ---------------------------------------------------------------------------
// Token groups
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Light,
    Dark,
}

/// Semantic colour roles. Widgets name the role, never the colour.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Palette {
    /// Window chrome behind every panel.
    pub background: Rgba,
    /// Default widget/inspector surface.
    pub surface: Rgba,
    /// Dock panel body — one step off `surface` so nested panels read.
    pub panel: Rgba,
    /// Panel headers, toolbars, status bar.
    pub header: Rgba,
    /// Tooltips, menus, toasts — floats above everything.
    pub overlay: Rgba,
    /// Hairline separators. Decorative: not held to a contrast threshold.
    pub border: Rgba,
    /// Interactive/focus outlines. Held to 3:1 (WCAG 1.4.11).
    pub border_strong: Rgba,

    pub text_primary: Rgba,
    pub text_secondary: Rgba,
    pub text_muted: Rgba,
    /// Text/icon drawn *on* `accent`.
    pub on_accent: Rgba,

    pub accent: Rgba,
    /// Selection wash; alpha applied at the use site.
    pub selection: Rgba,
    pub warning: Rgba,
    pub error: Rgba,
    pub success: Rgba,

    // --- 2D pattern canvas roles -----------------------------------------
    /// The infinite-canvas paper.
    pub canvas: Rgba,
    pub grid_line: Rgba,
    pub guide: Rgba,
    pub seam: Rgba,
    pub dart: Rgba,
    pub notch: Rgba,
    pub fold: Rgba,
    pub grain: Rgba,
    pub selected_entity: Rgba,
    pub hover_entity: Rgba,
    pub snap_indicator: Rgba,
}

/// 4px-based spacing scale. Padding, gaps, insets — nothing else.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Spacing {
    pub xs: f32,
    pub sm: f32,
    pub md: f32,
    pub lg: f32,
    pub xl: f32,
    pub xxl: f32,
}

/// Type scale in logical px. Tool UI: small and dense, but never below 11.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TypeScale {
    /// Shortcut hints, units, axis labels.
    pub micro: f32,
    /// Section headers, tab labels.
    pub small: f32,
    /// Default UI text.
    pub body: f32,
    /// Emphasised body / dialog text.
    pub body_large: f32,
    pub heading: f32,
    pub title: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Radii {
    pub none: f32,
    pub sm: f32,
    pub md: f32,
    pub lg: f32,
    /// Fully rounded (pills, badges).
    pub pill: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShadowToken {
    pub color: Rgba,
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur: f32,
}

/// Three levels only. A tool UI that needs a fourth is a tool UI that is lost.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Elevation {
    pub flat: ShadowToken,
    /// Docked panels, floating toolbars.
    pub raised: ShadowToken,
    /// Menus, tooltips, toasts, modals.
    pub overlay: ShadowToken,
}

/// Non-colour geometry every widget needs and therefore must not invent.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    /// Standard control height (text field, small button).
    pub control_height: f32,
    /// Tool buttons, primary actions — the generous hit target.
    pub control_height_lg: f32,
    pub icon: f32,
    pub icon_lg: f32,
    /// Precise 1px chrome. Physical-pixel snapping is the renderer's job.
    pub border_width: f32,
    /// Grab strip between docked panels — what the *layout* reserves and what
    /// the pointer has to hit. Wider than the line you can see, because a
    /// 6 px target is a target you miss: Blender, Figma and pane_grid all give
    /// the divider leeway around a hairline.
    pub divider: f32,
    /// The line actually painted inside that strip, centred. Marvelous
    /// Designer measures 7 px in the captured reference frames; 6 px plus the
    /// panel hairlines on either side reads the same.
    pub divider_line: f32,
    /// Height of a collapsed panel (its header only).
    pub panel_header: f32,
    pub tab_height: f32,
    pub status_bar_height: f32,
}

/// A complete theme. Serde round-trips, so `--theme path/to/tokens.json`
/// swaps the whole design language without a rebuild.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tokens {
    pub name: String,
    pub mode: Mode,
    pub color: Palette,
    pub space: Spacing,
    pub text: TypeScale,
    pub radius: Radii,
    pub elevation: Elevation,
    pub metrics: Metrics,
}

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("reading theme {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing theme {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
}

impl Tokens {
    /// Load a theme from a JSON file.
    ///
    /// ponytail: dev hot-reload (guide §22) is *loading*, not *watching*. The
    /// app can call this on a keypress today; add a `notify` watcher only when
    /// someone is actually iterating on tokens often enough to care.
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self, TokenError> {
        let path = path.as_ref();
        let name = path.display().to_string();
        let bytes = std::fs::read(path).map_err(|source| TokenError::Io {
            path: name.clone(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| TokenError::Parse { path: name, source })
    }

    pub fn to_json(&self) -> String {
        // Unwrap: `Tokens` is a plain struct tree of `String` and `f32`; the
        // only serde_json failures are non-string map keys and NaN floats,
        // neither of which this type can hold.
        serde_json::to_string_pretty(self).expect("Tokens always serialises")
    }

    pub fn dark() -> Self {
        DARK.clone()
    }

    pub fn light() -> Self {
        LIGHT.clone()
    }

    pub fn for_mode(mode: Mode) -> Self {
        match mode {
            Mode::Dark => Self::dark(),
            Mode::Light => Self::light(),
        }
    }
}

impl Default for Tokens {
    fn default() -> Self {
        Self::dark()
    }
}

// ---------------------------------------------------------------------------
// Built-in themes
// ---------------------------------------------------------------------------

const SPACE: Spacing = Spacing {
    xs: 2.0,
    sm: 4.0,
    md: 8.0,
    lg: 12.0,
    xl: 16.0,
    xxl: 24.0,
};

const TYPE: TypeScale = TypeScale {
    micro: 11.0,
    small: 12.0,
    body: 13.0,
    body_large: 14.0,
    heading: 16.0,
    title: 20.0,
};

const RADIUS: Radii = Radii {
    none: 0.0,
    sm: 2.0,
    md: 4.0,
    lg: 8.0,
    pill: 999.0,
};

const METRICS: Metrics = Metrics {
    control_height: 26.0,
    control_height_lg: 32.0,
    icon: 16.0,
    icon_lg: 20.0,
    border_width: 1.0,
    divider: 12.0,
    divider_line: 6.0,
    panel_header: 28.0,
    tab_height: 28.0,
    status_bar_height: 24.0,
};

const NO_SHADOW: ShadowToken = ShadowToken {
    color: Rgba::hex_a(0x00000000),
    offset_x: 0.0,
    offset_y: 0.0,
    blur: 0.0,
};

/// Dark is the default: this is a tool you stare at for eight hours next to a
/// saturated 3D garment.
static DARK: std::sync::LazyLock<Tokens> = std::sync::LazyLock::new(|| Tokens {
    name: "nguo dark".into(),
    mode: Mode::Dark,
    color: Palette {
        background: Rgba::hex(0x14161a),
        surface: Rgba::hex(0x1b1e23),
        panel: Rgba::hex(0x212429),
        header: Rgba::hex(0x181b1f),
        overlay: Rgba::hex(0x272b31),
        border: Rgba::hex(0x2e333a),
        border_strong: Rgba::hex(0x6b7480),

        text_primary: Rgba::hex(0xe9ecf0),
        text_secondary: Rgba::hex(0xb9c0ca),
        text_muted: Rgba::hex(0x99a1ac),
        on_accent: Rgba::hex(0x07101c),

        accent: Rgba::hex(0x59a9ff),
        selection: Rgba::hex(0x59a9ff),
        warning: Rgba::hex(0xf0b429),
        error: Rgba::hex(0xff7a7a),
        success: Rgba::hex(0x54d68a),

        canvas: Rgba::hex(0x101216),
        grid_line: Rgba::hex(0x272c33),
        guide: Rgba::hex(0x9d7bff),
        seam: Rgba::hex(0xffa14a),
        dart: Rgba::hex(0xff6f9c),
        notch: Rgba::hex(0xffd166),
        fold: Rgba::hex(0x5fd9c6),
        grain: Rgba::hex(0x8fa4ff),
        selected_entity: Rgba::hex(0xff9d2e),
        hover_entity: Rgba::hex(0xffd2a1),
        snap_indicator: Rgba::hex(0x2ce0bd),
    },
    space: SPACE,
    text: TYPE,
    radius: RADIUS,
    elevation: Elevation {
        flat: NO_SHADOW,
        raised: ShadowToken {
            color: Rgba::hex_a(0x00000059), // 35%
            offset_x: 0.0,
            offset_y: 1.0,
            blur: 3.0,
        },
        overlay: ShadowToken {
            color: Rgba::hex_a(0x0000008c), // 55%
            offset_x: 0.0,
            offset_y: 4.0,
            blur: 16.0,
        },
    },
    metrics: METRICS,
});

static LIGHT: std::sync::LazyLock<Tokens> = std::sync::LazyLock::new(|| Tokens {
    name: "nguo light".into(),
    mode: Mode::Light,
    color: Palette {
        background: Rgba::hex(0xeef0f3),
        surface: Rgba::hex(0xfdfdfe),
        panel: Rgba::hex(0xf5f6f8),
        header: Rgba::hex(0xe7eaee),
        overlay: Rgba::hex(0xffffff),
        border: Rgba::hex(0xd4d8de),
        border_strong: Rgba::hex(0x767e8a),

        text_primary: Rgba::hex(0x14171c),
        text_secondary: Rgba::hex(0x424a55),
        text_muted: Rgba::hex(0x596170),
        on_accent: Rgba::hex(0xffffff),

        accent: Rgba::hex(0x0a5fc7),
        selection: Rgba::hex(0x0a5fc7),
        warning: Rgba::hex(0x7a5100),
        error: Rgba::hex(0xb3261e),
        success: Rgba::hex(0x1a6b3c),

        canvas: Rgba::hex(0xfafbfc),
        grid_line: Rgba::hex(0xe1e5ea),
        guide: Rgba::hex(0x5b34c4),
        seam: Rgba::hex(0xa04f00),
        dart: Rgba::hex(0xb01654),
        notch: Rgba::hex(0x7a5100),
        fold: Rgba::hex(0x0c6b62),
        grain: Rgba::hex(0x3a37b8),
        selected_entity: Rgba::hex(0xb4400a),
        hover_entity: Rgba::hex(0x8a5a2b),
        snap_indicator: Rgba::hex(0x0a7d70),
    },
    space: SPACE,
    text: TYPE,
    radius: RADIUS,
    elevation: Elevation {
        flat: NO_SHADOW,
        raised: ShadowToken {
            color: Rgba::hex_a(0x0000001a), // 10%
            offset_x: 0.0,
            offset_y: 1.0,
            blur: 3.0,
        },
        overlay: ShadowToken {
            color: Rgba::hex_a(0x0000002e), // 18%
            offset_x: 0.0,
            offset_y: 4.0,
            blur: 16.0,
        },
    },
    metrics: METRICS,
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG 2.1 AA (1.4.3): 4.5:1 for body text.
    const AA_TEXT: f32 = 4.5;
    /// WCAG 2.1 AA (1.4.11): 3:1 for non-text UI components and graphics.
    /// Also the 1.4.3 threshold for large text (>= 18.66px, or 14px bold) —
    /// but every text role below is held to the stricter body figure anyway.
    const AA_NON_TEXT: f32 = 3.0;

    /// Every background a piece of text can legally land on.
    fn surfaces(p: &Palette) -> [(&'static str, Rgba); 6] {
        [
            ("background", p.background),
            ("surface", p.surface),
            ("panel", p.panel),
            ("header", p.header),
            ("overlay", p.overlay),
            ("canvas", p.canvas),
        ]
    }

    /// (role, colour, required ratio) for every foreground that renders as text.
    fn text_roles(p: &Palette) -> Vec<(&'static str, Rgba, f32)> {
        vec![
            ("text_primary", p.text_primary, AA_TEXT),
            ("text_secondary", p.text_secondary, AA_TEXT),
            ("text_muted", p.text_muted, AA_TEXT),
            // accent and the status colours are used for links, badge labels
            // and inline diagnostics, so they meet the body threshold too.
            ("accent", p.accent, AA_TEXT),
            ("warning", p.warning, AA_TEXT),
            ("error", p.error, AA_TEXT),
            ("success", p.success, AA_TEXT),
        ]
    }

    /// Non-text roles: outlines, canvas entities, indicators (WCAG 1.4.11).
    fn non_text_roles(p: &Palette) -> Vec<(&'static str, Rgba)> {
        vec![
            ("border_strong", p.border_strong),
            ("guide", p.guide),
            ("seam", p.seam),
            ("dart", p.dart),
            ("notch", p.notch),
            ("fold", p.fold),
            ("grain", p.grain),
            ("selected_entity", p.selected_entity),
            ("hover_entity", p.hover_entity),
            ("snap_indicator", p.snap_indicator),
        ]
    }

    fn check(failures: &mut Vec<String>, theme: &str, fg: &str, bg: &str, r: f32, min: f32) {
        if r < min {
            failures.push(format!(
                "{theme}: {fg} on {bg} = {r:.2}:1 (need {min:.1}:1)"
            ));
        }
    }

    #[test]
    fn every_text_on_surface_pair_meets_wcag_aa() {
        let mut failures = Vec::new();

        for tokens in [Tokens::dark(), Tokens::light()] {
            let p = &tokens.color;
            let theme = tokens.name.as_str();

            for (bg_name, bg) in surfaces(p) {
                for (fg_name, fg, min) in text_roles(p) {
                    check(
                        &mut failures,
                        theme,
                        fg_name,
                        bg_name,
                        contrast_ratio(fg, bg),
                        min,
                    );
                }
            }

            // Text drawn on the accent fill (primary buttons, active tool).
            check(
                &mut failures,
                theme,
                "on_accent",
                "accent",
                contrast_ratio(p.on_accent, p.accent),
                AA_TEXT,
            );

            // Canvas entity strokes sit on the canvas, never on chrome.
            for (fg_name, fg) in non_text_roles(p) {
                check(
                    &mut failures,
                    theme,
                    fg_name,
                    "canvas",
                    contrast_ratio(fg, p.canvas),
                    AA_NON_TEXT,
                );
            }

            // Outlines must read against every chrome surface too.
            for (bg_name, bg) in surfaces(p) {
                check(
                    &mut failures,
                    theme,
                    "border_strong",
                    bg_name,
                    contrast_ratio(p.border_strong, bg),
                    AA_NON_TEXT,
                );
            }

            // The grid is deliberately quiet, but must still be visible.
            let grid = contrast_ratio(p.grid_line, p.canvas);
            assert!(
                grid > 1.05,
                "{theme}: grid_line is invisible on canvas ({grid:.3}:1)"
            );
        }

        assert!(
            failures.is_empty(),
            "{} WCAG AA contrast failures:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }

    #[test]
    fn contrast_ratio_matches_known_values() {
        let white = Rgba::hex(0xffffff);
        let black = Rgba::hex(0x000000);
        assert!((contrast_ratio(white, black) - 21.0).abs() < 0.01);
        assert!((contrast_ratio(white, white) - 1.0).abs() < 1e-4);
        // #767676 on white is the canonical 4.54:1 AA boundary example.
        assert!((contrast_ratio(Rgba::hex(0x767676), white) - 4.54).abs() < 0.02);
    }

    #[test]
    fn translucent_foreground_is_scored_after_compositing() {
        let bg = Rgba::hex(0x000000);
        let solid = contrast_ratio(Rgba::hex(0xffffff), bg);
        let faded = contrast_ratio(Rgba::hex(0xffffff).alpha(0.5), bg);
        assert!(
            faded < solid,
            "alpha must reduce contrast: {faded} vs {solid}"
        );
    }

    #[test]
    fn tokens_round_trip_through_json() {
        for tokens in [Tokens::dark(), Tokens::light()] {
            let json = tokens.to_json();
            let back: Tokens = serde_json::from_str(&json).unwrap();
            assert_eq!(tokens, back);
        }
        // And the hex encoding is the readable one we promised.
        assert!(Tokens::dark().to_json().contains("\"#14161a\""));
    }

    #[test]
    fn hex_parsing_rejects_garbage() {
        assert!(Rgba::parse_hex("#abc").is_err());
        assert!(Rgba::parse_hex("#zzzzzz").is_err());
        assert!(Rgba::parse_hex("#ff80").is_err());
        assert_eq!(Rgba::parse_hex("#ff8000").unwrap(), Rgba::hex(0xff8000));
        assert_eq!(Rgba::parse_hex("#00000080").unwrap().a, 128.0 / 255.0);
    }

    #[test]
    fn tokens_load_from_a_path() {
        let dir = std::env::temp_dir().join("nguo-ui-theme-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.json");
        std::fs::write(&path, Tokens::light().to_json()).unwrap();
        assert_eq!(Tokens::from_json_path(&path).unwrap(), Tokens::light());

        std::fs::write(&path, "{ not json").unwrap();
        assert!(matches!(
            Tokens::from_json_path(&path),
            Err(TokenError::Parse { .. })
        ));
        assert!(matches!(
            Tokens::from_json_path(dir.join("nope.json")),
            Err(TokenError::Io { .. })
        ));
    }

    #[test]
    fn scales_are_monotonic() {
        let t = Tokens::dark();
        let s = t.space;
        assert!(s.xs < s.sm && s.sm < s.md && s.md < s.lg && s.lg < s.xl && s.xl < s.xxl);
        let f = t.text;
        assert!(f.micro < f.small && f.small < f.body && f.body < f.body_large);
        assert!(f.body_large < f.heading && f.heading < f.title);
        // 11px is the floor for legible tool UI.
        assert!(f.micro >= 11.0);
        // Hit targets stay grabbable.
        assert!(t.metrics.control_height >= 24.0);
        // The grab strip is what you have to hit; the line is what you see.
        assert!(t.metrics.divider >= 12.0, "{}", t.metrics.divider);
        assert!(t.metrics.divider_line <= t.metrics.divider);
        assert!(t.metrics.divider_line >= 4.0);
    }

    #[test]
    fn light_and_dark_share_every_non_colour_token() {
        let (d, l) = (Tokens::dark(), Tokens::light());
        assert_eq!(d.space, l.space);
        assert_eq!(d.text, l.text);
        assert_eq!(d.radius, l.radius);
        assert_eq!(d.metrics, l.metrics);
        assert_ne!(d.mode, l.mode);
    }
}
