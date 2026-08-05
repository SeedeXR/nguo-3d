// SPDX-License-Identifier: GPL-2.0-or-later
//! Toolbars, inspector rows, numeric entry, status bar, toasts, empty states.
//!
//! Every function here is a pure function of the state it is handed
//! (guide §0.5). None of them owns anything the document should own.

use iced::widget::{button, column, container, row, scrollable, space, text, text_input, tooltip};
use iced::{Element, Length};
use nguo_ui_theme::{Rgba, Tokens};

use crate::dock::Axis;
use crate::icon::{icon, Icon};
use crate::{color, hairline, overlay_surface, surface};

// ---------------------------------------------------------------------------
// Toolbar + tool button
// ---------------------------------------------------------------------------

/// How much of a token a disabled control keeps. Low enough to read as "not
/// available" next to a live sibling, high enough to still be legible — the
/// button has to say what it *would* do.
pub const DISABLED_ALPHA: f32 = 0.4;

/// A tool button: icon, active state, tooltip, keyboard-shortcut hint.
///
/// `on_press: None` renders it **disabled**, and disabled has to *look* it: the
/// glyph goes to [`DISABLED_ALPHA`] and the border with it. iced reports
/// `button::Status::Disabled`, but the icon is an SVG whose tint is chosen
/// outside the style closure, so the closure alone cannot grey it — hence
/// `enabled` here rather than a fourth `match` arm. Until this existed a button
/// with no handler was pixel-identical to a live one and simply did nothing when
/// pressed.
pub fn tool_button<'a, Message: Clone + 'a>(
    tokens: &'a Tokens,
    glyph: Icon,
    label: &'a str,
    shortcut: Option<&'a str>,
    active: bool,
    on_press: Option<Message>,
) -> Element<'a, Message> {
    let c = tokens.color;
    let size = tokens.metrics.control_height_lg;
    let enabled = on_press.is_some();
    let fg = match (enabled, active) {
        (false, _) => c.text_muted.alpha(DISABLED_ALPHA),
        (true, true) => c.on_accent,
        (true, false) => c.text_secondary,
    };

    let control = button(
        container(icon(glyph, tokens.metrics.icon, fg))
            .center_x(Length::Fill)
            .center_y(Length::Fill),
    )
    .width(Length::Fixed(size))
    .height(Length::Fixed(size))
    .padding(0)
    .style(move |_theme, status| {
        let bg = match (active, status) {
            (_, button::Status::Disabled) => Rgba::hex_a(0x00000000),
            (true, _) => c.accent,
            (false, button::Status::Hovered | button::Status::Pressed) => c.overlay,
            (false, _) => Rgba::hex_a(0x00000000),
        };
        button::Style {
            background: Some(color(bg).into()),
            text_color: color(fg),
            border: iced::Border {
                color: color(match (enabled, active) {
                    (false, _) => c.border.alpha(DISABLED_ALPHA),
                    (true, true) => c.accent,
                    (true, false) => c.border,
                }),
                width: if active && enabled {
                    0.0
                } else {
                    tokens.metrics.border_width
                },
                radius: tokens.radius.md.into(),
            },
            ..button::Style::default()
        }
    })
    .on_press_maybe(on_press);

    tooltip(
        control,
        shortcut_tip(tokens, label, shortcut),
        tooltip::Position::Right,
    )
    .gap(tokens.space.sm)
    .into()
}

fn shortcut_tip<'a, Message: 'a>(
    tokens: &'a Tokens,
    label: &'a str,
    shortcut: Option<&'a str>,
) -> Element<'a, Message> {
    let c = tokens.color;
    let mut body = row![text(label)
        .size(tokens.text.small)
        .color(color(c.text_primary))]
    .spacing(tokens.space.md)
    .align_y(iced::Center);

    if let Some(keys) = shortcut {
        body = body.push(
            container(
                text(keys)
                    .size(tokens.text.micro)
                    .color(color(c.text_muted)),
            )
            .padding([0.0, tokens.space.sm])
            .style(surface(tokens, c.surface)),
        );
    }

    container(body)
        .padding([tokens.space.sm, tokens.space.md])
        .style(overlay_surface(tokens))
        .into()
}

/// A toolbar strip. `Axis::Horizontal` for the top bar, `Axis::Vertical` for
/// the left tool rail.
pub fn toolbar<'a, Message: 'a>(
    tokens: &'a Tokens,
    axis: Axis,
    items: Vec<Element<'a, Message>>,
) -> Element<'a, Message> {
    let pad = [tokens.space.sm, tokens.space.sm];
    let inner: Element<'a, Message> = match axis {
        Axis::Horizontal => row(items)
            .spacing(tokens.space.sm)
            .padding(pad)
            .align_y(iced::Center)
            .into(),
        Axis::Vertical => column(items)
            .spacing(tokens.space.sm)
            .padding(pad)
            .align_x(iced::Center)
            .into(),
    };

    container(inner)
        .width(match axis {
            Axis::Horizontal => Length::Fill,
            Axis::Vertical => Length::Shrink,
        })
        .height(match axis {
            Axis::Horizontal => Length::Shrink,
            Axis::Vertical => Length::Fill,
        })
        .style(hairline(tokens, tokens.color.header))
        .into()
}

/// A vertical hair between toolbar groups.
pub fn toolbar_separator<'a, Message: 'a>(tokens: &Tokens, axis: Axis) -> Element<'a, Message> {
    let c = tokens.color.border;
    let hair = tokens.metrics.border_width;
    let inset = tokens.space.sm;
    let (w, h) = match axis {
        Axis::Horizontal => (Length::Fixed(hair), Length::Fixed(tokens.metrics.icon_lg)),
        Axis::Vertical => (Length::Fixed(tokens.metrics.icon_lg), Length::Fixed(hair)),
    };
    container(
        container(space::horizontal())
            .width(w)
            .height(h)
            .style(move |_| container::Style {
                background: Some(color(c).into()),
                ..container::Style::default()
            }),
    )
    .padding(inset)
    .into()
}

// ---------------------------------------------------------------------------
// Numeric / expression field (guide §9.4, §20)
// ---------------------------------------------------------------------------

/// The result of evaluating what the user typed.
///
/// **This crate does not evaluate anything.** Expression and parameter-reference
/// parsing (`waist/4 + 1cm`) belongs to `nguo-measure`; the app calls it and
/// hands the outcome back here as state. That keeps `nguo-ui-widgets` free of a
/// dependency on `nguo-measure` (which is being written in parallel) and keeps
/// the widget a pure function of state.
///
/// Wiring, once `nguo-measure` lands, is one line in the app's `update`:
/// `self.width_status = ExprStatus::from_measure(measure::eval(&raw, &params));`
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ExprStatus {
    /// Nothing typed yet.
    #[default]
    Empty,
    /// Evaluated. `formatted` is the resolved value in display units, e.g.
    /// `"24.5 cm"` — shown as a quiet hint next to the raw expression.
    Ok { formatted: String },
    /// Actionable, non-fatal (guide §21): show it, keep the raw text, let the
    /// user fix it.
    Invalid { message: String },
}

impl ExprStatus {
    pub fn is_invalid(&self) -> bool {
        matches!(self, ExprStatus::Invalid { .. })
    }
}

/// A dimension field that accepts expressions and parameter references.
///
/// `raw` is exactly what the user typed — never a re-formatted value, or
/// editing `waist/4` would destroy the expression on every keystroke.
pub fn expr_field<'a, Message, F>(
    tokens: &'a Tokens,
    raw: &'a str,
    placeholder: &'a str,
    status: &'a ExprStatus,
    on_input: F,
    on_submit: Option<Message>,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(String) -> Message + 'a,
{
    let c = tokens.color;
    let invalid = status.is_invalid();

    let mut input = text_input(placeholder, raw)
        .on_input(on_input)
        .size(tokens.text.body)
        .padding([tokens.space.xs, tokens.space.md])
        .width(Length::Fill)
        .style(move |_theme, s| {
            let focused = matches!(s, text_input::Status::Focused { .. });
            let border = if invalid {
                c.error
            } else if focused {
                c.accent
            } else {
                c.border
            };
            text_input::Style {
                background: color(c.surface).into(),
                border: iced::Border {
                    color: color(border),
                    width: tokens.metrics.border_width,
                    radius: tokens.radius.md.into(),
                },
                icon: color(c.text_muted),
                placeholder: color(c.text_muted),
                value: color(c.text_primary),
                selection: color(c.selection.alpha(0.35)),
            }
        });

    if let Some(msg) = on_submit {
        input = input.on_submit(msg);
    }

    let hint: Element<'a, Message> = match status {
        ExprStatus::Empty => space::vertical().height(Length::Fixed(0.0)).into(),
        ExprStatus::Ok { formatted } => row![
            text("=").size(tokens.text.micro).color(color(c.text_muted)),
            text(formatted.as_str())
                .size(tokens.text.micro)
                .color(color(c.text_secondary))
        ]
        .spacing(tokens.space.xs)
        .into(),
        ExprStatus::Invalid { message } => row![
            icon(Icon::Warning, tokens.text.micro, c.error),
            text(message.as_str())
                .size(tokens.text.micro)
                .color(color(c.error))
        ]
        .spacing(tokens.space.xs)
        .align_y(iced::Center)
        .into(),
    };

    column![input, hint].spacing(tokens.space.xs).into()
}

// ---------------------------------------------------------------------------
// Inspector
// ---------------------------------------------------------------------------

/// A labelled control: label on the left at a fixed width, control filling the
/// rest. Consistent label width is what makes an inspector readable.
pub fn property_row<'a, Message: 'a>(
    tokens: &'a Tokens,
    label: &'a str,
    control: impl Into<Element<'a, Message>>,
) -> Element<'a, Message> {
    row![
        container(
            text(label)
                .size(tokens.text.small)
                .color(color(tokens.color.text_secondary))
        )
        .width(Length::Fixed(96.0))
        .align_y(iced::Center)
        .height(Length::Fixed(tokens.metrics.control_height)),
        container(control.into()).width(Length::Fill)
    ]
    .spacing(tokens.space.md)
    .padding([tokens.space.xs, 0.0])
    .align_y(iced::alignment::Vertical::Top)
    .into()
}

/// A section divider inside an inspector.
pub fn section_header<'a, Message: 'a>(
    tokens: &'a Tokens,
    title: &'a str,
    glyph: Option<Icon>,
) -> Element<'a, Message> {
    let c = tokens.color;
    let mut inner = row![].spacing(tokens.space.sm).align_y(iced::Center);
    if let Some(g) = glyph {
        inner = inner.push(icon(g, tokens.metrics.icon, c.text_muted));
    }
    inner = inner.push(
        text(title)
            .size(tokens.text.small)
            .color(color(c.text_muted)),
    );

    container(inner)
        .width(Length::Fill)
        .padding([tokens.space.md, tokens.space.md])
        .style(hairline(tokens, c.header))
        .into()
}

/// Scrollable container for property rows and section headers.
pub fn inspector<'a, Message: 'a>(
    tokens: &'a Tokens,
    children: Vec<Element<'a, Message>>,
) -> Element<'a, Message> {
    container(scrollable(
        column(children)
            .spacing(tokens.space.xs)
            .padding(tokens.space.md)
            .width(Length::Fill),
    ))
    .width(Length::Fill)
    .height(Length::Fill)
    .style(surface(tokens, tokens.color.panel))
    .into()
}

// ---------------------------------------------------------------------------
// Status bar + toasts
// ---------------------------------------------------------------------------

/// Bottom status bar: `left` items align left, `right` items align right.
pub fn status_bar<'a, Message: 'a>(
    tokens: &'a Tokens,
    left: Vec<Element<'a, Message>>,
    right: Vec<Element<'a, Message>>,
) -> Element<'a, Message> {
    container(
        row![
            row(left).spacing(tokens.space.lg).align_y(iced::Center),
            space::horizontal(),
            row(right).spacing(tokens.space.lg).align_y(iced::Center)
        ]
        .align_y(iced::Center)
        .padding([0.0, tokens.space.md]),
    )
    .width(Length::Fill)
    .height(Length::Fixed(tokens.metrics.status_bar_height))
    .style(hairline(tokens, tokens.color.header))
    .into()
}

/// A single status-bar readout, e.g. `zoom 100%` or `mm`.
pub fn status_item<'a, Message: 'a>(
    tokens: &'a Tokens,
    glyph: Option<Icon>,
    label: &'a str,
) -> Element<'a, Message> {
    let c = tokens.color.text_secondary;
    let mut r = row![].spacing(tokens.space.xs).align_y(iced::Center);
    if let Some(g) = glyph {
        r = r.push(icon(g, tokens.text.micro + 2.0, c));
    }
    r.push(text(label).size(tokens.text.micro).color(color(c)))
        .into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastLevel {
    Info,
    Success,
    Warning,
    Error,
}

impl ToastLevel {
    fn parts(self, tokens: &Tokens) -> (Icon, Rgba) {
        let c = tokens.color;
        match self {
            ToastLevel::Info => (Icon::Info, c.accent),
            ToastLevel::Success => (Icon::Check, c.success),
            ToastLevel::Warning => (Icon::Warning, c.warning),
            ToastLevel::Error => (Icon::Warning, c.error),
        }
    }
}

/// A non-fatal, dismissible notification (guide §21).
///
/// Errors here are *actionable and non-fatal*: `detail` says what to do, not
/// what went wrong internally. Stack these in an `iced::widget::column` and
/// float them over the app with `iced::widget::stack`.
pub fn toast<'a, Message: Clone + 'a>(
    tokens: &'a Tokens,
    level: ToastLevel,
    title: &'a str,
    detail: Option<&'a str>,
    on_dismiss: Option<Message>,
) -> Element<'a, Message> {
    let c = tokens.color;
    let (glyph, tint) = level.parts(tokens);

    let mut body = column![text(title)
        .size(tokens.text.body)
        .color(color(c.text_primary))]
    .spacing(tokens.space.xs);
    if let Some(d) = detail {
        body = body.push(
            text(d)
                .size(tokens.text.small)
                .color(color(c.text_secondary)),
        );
    }

    let mut line = row![
        icon(glyph, tokens.metrics.icon_lg, tint),
        body.width(Length::Fill)
    ]
    .spacing(tokens.space.md)
    .align_y(iced::alignment::Vertical::Top);

    if let Some(msg) = on_dismiss {
        line = line.push(crate::icon_button(tokens, Icon::Close, c.text_muted, msg));
    }

    container(line)
        .padding(tokens.space.lg)
        .max_width(420.0)
        .style(move |_| container::Style {
            background: Some(color(c.overlay).into()),
            text_color: Some(color(c.text_primary)),
            border: iced::Border {
                color: color(tint),
                width: tokens.metrics.border_width,
                radius: tokens.radius.lg.into(),
            },
            shadow: crate::shadow(tokens.elevation.overlay),
            ..container::Style::default()
        })
        .into()
}

// ---------------------------------------------------------------------------
// Empty state (guide §16)
// ---------------------------------------------------------------------------

/// What a panel shows when there is nothing in it yet — never a blank pane.
pub fn empty_state<'a, Message: Clone + 'a>(
    tokens: &'a Tokens,
    glyph: Icon,
    title: &'a str,
    body: &'a str,
    action: Option<(&'a str, Message)>,
) -> Element<'a, Message> {
    let c = tokens.color;
    let mut stack = column![
        icon(glyph, 40.0, c.text_muted),
        text(title)
            .size(tokens.text.body_large)
            .color(color(c.text_primary)),
        text(body)
            .size(tokens.text.small)
            .color(color(c.text_muted))
            .center(),
    ]
    .spacing(tokens.space.md)
    .align_x(iced::Center)
    .max_width(320.0);

    if let Some((label, msg)) = action {
        stack = stack.push(
            button(text(label).size(tokens.text.small))
                .padding([tokens.space.sm, tokens.space.lg])
                .style(move |_theme, status| {
                    let bg = match status {
                        button::Status::Hovered | button::Status::Pressed => c.accent,
                        _ => c.accent.alpha(0.85),
                    };
                    button::Style {
                        background: Some(color(bg).into()),
                        text_color: color(c.on_accent),
                        border: iced::Border {
                            color: color(c.accent),
                            width: 0.0,
                            radius: tokens.radius.md.into(),
                        },
                        ..button::Style::default()
                    }
                })
                .on_press(msg),
        );
    }

    container(stack)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .padding(tokens.space.xxl)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_status_defaults_to_empty_and_reports_invalidity() {
        assert_eq!(ExprStatus::default(), ExprStatus::Empty);
        assert!(!ExprStatus::Empty.is_invalid());
        assert!(!ExprStatus::Ok {
            formatted: "24.5 cm".into()
        }
        .is_invalid());
        assert!(ExprStatus::Invalid {
            message: "unknown parameter `waistt`".into()
        }
        .is_invalid());
    }

    #[test]
    fn every_toast_level_has_an_icon_and_a_distinct_tint() {
        let tokens = Tokens::dark();
        let levels = [
            ToastLevel::Info,
            ToastLevel::Success,
            ToastLevel::Warning,
            ToastLevel::Error,
        ];
        let tints: Vec<_> = levels
            .iter()
            .map(|l| format!("{:?}", l.parts(&tokens).1))
            .collect();
        let mut unique = tints.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(tints.len(), unique.len(), "two toast levels share a colour");
    }
}
