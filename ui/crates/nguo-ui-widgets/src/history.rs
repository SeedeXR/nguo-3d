// SPDX-License-Identifier: GPL-2.0-or-later
//! The history panel: every edit as a row, click any row to go there.
//!
//! A pure renderer, like every widget in this crate — it holds no state and
//! knows nothing about undo. The app projects its command stack into
//! [`HistoryRow`]s and gets [`HistoryEvent::Jump`] back with the row index; what
//! an index means is the app's business (`nguo-app`'s `shell::history_rows`).
//!
//! Row 0 is the document as it opened, so a click on it undoes everything —
//! which is the row Photoshop puts at the top of the same panel.

use iced::widget::{button, column, container, row, scrollable, space, text};
use iced::{Element, Length};
use nguo_ui_theme::Tokens;

use crate::color;
use crate::icon::{icon, Icon};

/// One row. `applied` is false for the **undone tail** — the states a jump
/// backwards left behind, which stay clickable but read as greyed out because
/// the next edit will truncate them (D-2: the history is linear).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryRow {
    pub label: String,
    pub applied: bool,
}

impl HistoryRow {
    pub fn new(label: impl Into<String>, applied: bool) -> Self {
        Self {
            label: label.into(),
            applied,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryEvent {
    /// Go to the state row `usize` represents.
    Jump(usize),
}

/// Where the document currently is: the last applied row.
///
/// Derived rather than passed in, so the app cannot hand the panel a cursor that
/// disagrees with the rows it handed over in the same breath. `None` only for an
/// empty list.
pub fn current(rows: &[HistoryRow]) -> Option<usize> {
    rows.iter().rposition(|r| r.applied)
}

/// Render the history.
///
/// The **current** row — the last applied one — carries the selection, so the
/// panel answers "where am I" without the app tracking a second cursor.
pub fn history<'a, Message, F>(
    tokens: &'a Tokens,
    rows: &'a [HistoryRow],
    on_event: F,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(HistoryEvent) -> Message + Clone + 'a,
{
    if rows.is_empty() {
        return crate::empty_state(
            tokens,
            Icon::Undo,
            "Nothing to undo yet",
            "Every edit lands here as one step — a whole drag included. Click a \
             step to go back to it.",
            None,
        );
    }

    let here = current(rows);
    let items = rows
        .iter()
        .enumerate()
        .map(|(i, r)| history_row(tokens, i, r, here == Some(i), &on_event))
        .collect::<Vec<_>>();

    container(scrollable(column(items).width(Length::Fill)))
        .width(Length::Fill)
        .height(Length::Fill)
        .style(crate::surface(tokens, tokens.color.panel))
        .into()
}

fn history_row<'a, Message, F>(
    tokens: &'a Tokens,
    index: usize,
    entry: &'a HistoryRow,
    current: bool,
    on_event: &F,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(HistoryEvent) -> Message + Clone + 'a,
{
    let c = tokens.color;
    // Three states, and they have to be distinguishable at a glance: where you
    // are, what is behind you, and what a new edit is about to throw away.
    let fg = match (current, entry.applied) {
        (true, _) => c.text_primary,
        (false, true) => c.text_secondary,
        (false, false) => c.text_muted,
    };

    let label = row![
        // The current row is the only one that gets a marker; without it a
        // selection tint alone is ambiguous against a hover.
        if current {
            Element::from(icon(Icon::ChevronRight, tokens.metrics.icon, fg))
        } else {
            space::horizontal()
                .width(Length::Fixed(tokens.metrics.icon))
                .into()
        },
        text(entry.label.as_str())
            .size(tokens.text.small)
            .color(color(fg)),
    ]
    .spacing(tokens.space.xs)
    .align_y(iced::Center);

    let selection = c.selection;
    button(label)
        .width(Length::Fill)
        .padding([tokens.space.xs, tokens.space.sm])
        .style(move |_theme, status| {
            let bg = match (current, status) {
                (true, _) => selection.alpha(0.28),
                (false, button::Status::Hovered | button::Status::Pressed) => {
                    c.text_primary.alpha(0.06)
                }
                _ => nguo_ui_theme::Rgba::hex_a(0x00000000),
            };
            button::Style {
                background: Some(color(bg).into()),
                text_color: color(fg),
                border: iced::Border {
                    color: color(if current { selection } else { c.border }),
                    width: if current {
                        tokens.metrics.border_width
                    } else {
                        0.0
                    },
                    radius: tokens.radius.sm.into(),
                },
                ..button::Style::default()
            }
        })
        .on_press(on_event(HistoryEvent::Jump(index)))
        .into()
}
