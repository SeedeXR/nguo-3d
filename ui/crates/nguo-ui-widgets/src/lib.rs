// SPDX-License-Identifier: GPL-2.0-or-later
//! Reusable iced 0.14 widgets for nguo-3d.
//!
//! Every widget here is a **pure function of the state it is handed** and
//! reads its colours from [`nguo_ui_theme::Tokens`]. There is
//! no colour literal in this crate — the only ones are fully-transparent
//! sentinels, and even those go through `Rgba`.
//!
//! ## Layout of the crate
//! - [`icon`] — compile-time-embedded Lucide/Tabler SVGs
//! - [`dock`] — single-window docking: split panes, dividers, collapse, tabs
//! - [`tree`] — piece/layer hierarchy
//! - [`history`] — the undo history as a clickable list
//! - [`controls`] — toolbar, expression field, inspector, status bar, toasts
//!
//! ## Theming iced's built-in widgets
//! Our widgets style themselves explicitly from tokens. iced's *own* defaults
//! (scrollbars, pick lists, the text cursor) come from `iced::Theme`, so build
//! one with [`iced_theme`] and return it from your `Application::theme`.

pub mod controls;
pub mod dock;
pub mod history;
pub mod icon;
pub mod tree;

pub use controls::{
    empty_state, expr_field, inspector, property_row, section_header, status_bar, status_item,
    toast, tool_button, toolbar, toolbar_separator, ExprStatus, ToastLevel,
};
pub use dock::{dock, Axis, DividerGrab, DockEvent, DockLayout, Node, Panel, Split, Tab};
pub use history::{history, HistoryEvent, HistoryRow};
pub use icon::{icon, Icon};
pub use tree::{tree, TreeEvent, TreeNode, TreeState};

use iced::widget::{button, container};
use nguo_ui_theme::{Mode, Rgba, ShadowToken, Tokens};

/// The one place `nguo_ui_theme::Rgba` becomes an `iced::Color`.
///
/// `iced::Color::from_rgba` takes non-linear sRGB components, which is exactly
/// what `Rgba` stores (`iced_core-0.14.0/src/color.rs:79`).
pub fn color(c: Rgba) -> iced::Color {
    iced::Color::from_rgba(c.r, c.g, c.b, c.a)
}

pub fn shadow(s: ShadowToken) -> iced::Shadow {
    iced::Shadow {
        color: color(s.color),
        offset: iced::Vector::new(s.offset_x, s.offset_y),
        blur_radius: s.blur,
    }
}

/// Build the `iced::Theme` that drives iced's *built-in* widget defaults.
///
/// `Theme::custom(name, Palette { background, text, primary, success, warning,
/// danger })` — verified in `iced_core-0.14.0/src/theme.rs:91` and
/// `theme/palette.rs`. iced derives its extended palette (hover/weak/strong
/// variants) from these six.
pub fn iced_theme(tokens: &Tokens) -> iced::Theme {
    iced::Theme::custom(
        tokens.name.clone(),
        iced::theme::Palette {
            background: color(tokens.color.background),
            text: color(tokens.color.text_primary),
            primary: color(tokens.color.accent),
            success: color(tokens.color.success),
            warning: color(tokens.color.warning),
            danger: color(tokens.color.error),
        },
    )
}

/// Filled surface with a precise 1px border and a small radius. Panels,
/// popovers, inset groups.
pub fn surface(tokens: &Tokens, bg: Rgba) -> impl Fn(&iced::Theme) -> container::Style {
    let (border, width, radius) = (
        tokens.color.border,
        tokens.metrics.border_width,
        tokens.radius.sm,
    );
    move |_theme| container::Style {
        background: Some(color(bg).into()),
        border: iced::Border {
            color: color(border),
            width,
            radius: radius.into(),
        },
        ..container::Style::default()
    }
}

/// Filled strip with a square 1px border. Headers, toolbars, the status bar —
/// flush chrome that must not look like a floating card.
pub fn hairline(tokens: &Tokens, bg: Rgba) -> impl Fn(&iced::Theme) -> container::Style {
    let (border, width) = (tokens.color.border, tokens.metrics.border_width);
    move |_theme| container::Style {
        background: Some(color(bg).into()),
        border: iced::Border {
            color: color(border),
            width,
            radius: 0.0.into(),
        },
        ..container::Style::default()
    }
}

/// Floating surface: overlay fill, radius, elevation shadow.
pub fn overlay_surface(tokens: &Tokens) -> impl Fn(&iced::Theme) -> container::Style {
    let (bg, border, width, radius, elev) = (
        tokens.color.overlay,
        tokens.color.border,
        tokens.metrics.border_width,
        tokens.radius.md,
        tokens.elevation.overlay,
    );
    move |_theme| container::Style {
        background: Some(color(bg).into()),
        border: iced::Border {
            color: color(border),
            width,
            radius: radius.into(),
        },
        shadow: shadow(elev),
        ..container::Style::default()
    }
}

/// Transparent until hovered. Icon buttons in headers and tree rows.
pub fn ghost_button(tokens: &Tokens) -> impl Fn(&iced::Theme, button::Status) -> button::Style {
    let (hover, fg, radius) = (
        tokens.color.text_primary.alpha(0.08),
        tokens.color.text_secondary,
        tokens.radius.sm,
    );
    move |_theme, status| {
        let bg = match status {
            button::Status::Hovered | button::Status::Pressed => hover,
            _ => Rgba::hex_a(0x00000000),
        };
        button::Style {
            background: Some(color(bg).into()),
            text_color: color(fg),
            border: iced::Border {
                color: color(Rgba::hex_a(0x00000000)),
                width: 0.0,
                radius: radius.into(),
            },
            ..button::Style::default()
        }
    }
}

/// A ghost icon button whose **hit target is a whole control height**, not the
/// size of the glyph inside it.
///
/// Every icon-only affordance in this crate goes through here, because every one
/// of them was built by hand and every one of them was too small: the tree's
/// disclosure triangle was `padding(0)` around a 16 px glyph (16x16), the dock's
/// collapse chevron and the toast's dismiss were `padding(space.xs)` around one
/// (20x20). `Metrics::control_height` is the token that promises a grabbable
/// target — `nguo_ui_theme::tests::scales_are_monotonic` pins it at >= 24 — so the
/// size comes from there rather than from a padding value chosen per call site.
pub fn icon_button<'a, Message: Clone + 'a>(
    tokens: &Tokens,
    glyph: Icon,
    tint: Rgba,
    on_press: Message,
) -> iced::Element<'a, Message> {
    let target = iced::Length::Fixed(tokens.metrics.control_height);
    button(
        container(crate::icon::icon(glyph, tokens.metrics.icon, tint))
            .center_x(iced::Length::Fill)
            .center_y(iced::Length::Fill),
    )
    .width(target)
    .height(target)
    .padding(0)
    .style(ghost_button(tokens))
    .on_press(on_press)
    .into()
}

/// The icon for the *other* theme — what a theme-toggle button should show.
pub fn theme_toggle_icon(mode: Mode) -> Icon {
    match mode {
        Mode::Dark => Icon::Sun,
        Mode::Light => Icon::Moon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_survives_the_trip_into_iced() {
        let c = Rgba::hex_a(0x336699cc);
        let i = color(c);
        assert!((i.r - c.r).abs() < 1e-6);
        assert!((i.g - c.g).abs() < 1e-6);
        assert!((i.b - c.b).abs() < 1e-6);
        assert!((i.a - c.a).abs() < 1e-6);
    }

    #[test]
    fn the_iced_theme_carries_our_tokens() {
        for tokens in [Tokens::dark(), Tokens::light()] {
            let theme = iced_theme(&tokens);
            let p = theme.palette();
            assert_eq!(p.background, color(tokens.color.background));
            assert_eq!(p.primary, color(tokens.color.accent));
            assert_eq!(p.danger, color(tokens.color.error));
        }
    }

    #[test]
    fn shadow_tokens_map_onto_iced_shadows() {
        let s = shadow(Tokens::dark().elevation.overlay);
        assert_eq!(s.offset, iced::Vector::new(0.0, 4.0));
        assert_eq!(s.blur_radius, 16.0);
    }

    #[test]
    fn the_theme_toggle_shows_the_other_mode() {
        assert_eq!(theme_toggle_icon(Mode::Dark), Icon::Sun);
        assert_eq!(theme_toggle_icon(Mode::Light), Icon::Moon);
    }
}
