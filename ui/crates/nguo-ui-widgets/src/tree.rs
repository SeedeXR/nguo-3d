// SPDX-License-Identifier: GPL-2.0-or-later
//! Tree view for the piece / layer hierarchy: selection + expand/collapse.
//!
//! [`TreeState`] is view state (guide §6.5: selection is session state, not
//! authoritative geometry) and is owned by the app. The nodes themselves are
//! projected from the document each frame.

use std::collections::BTreeSet;

use iced::widget::{button, column, container, row, scrollable, space, text};
use iced::{Element, Length};
use nguo_ui_theme::Tokens;

use crate::color;
use crate::icon::{icon, Icon};

/// One row of the hierarchy. Build these from the document; do not store them.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    pub id: String,
    pub label: String,
    pub icon: Option<Icon>,
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            icon: None,
            children: Vec::new(),
        }
    }

    pub fn with_icon(mut self, glyph: Icon) -> Self {
        self.icon = Some(glyph);
        self
    }

    pub fn with_children(mut self, children: Vec<TreeNode>) -> Self {
        self.children = children;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TreeEvent {
    /// The disclosure triangle was clicked.
    Toggled(String),
    /// The row was clicked.
    ///
    /// Modifier-aware selection (shift = range, cmd/ctrl = toggle, guide §6.5)
    /// is the app's call: a click carries no modifiers through iced's `button`,
    /// and the app is the only place that knows the current keyboard state.
    /// Feed it back as [`TreeState::select`] / [`TreeState::toggle_selected`].
    Clicked(String),
}

/// Expansion + selection. Owned by the app.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TreeState {
    expanded: BTreeSet<String>,
    selected: BTreeSet<String>,
}

impl TreeState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start with a set of nodes already open.
    pub fn expanded<I: IntoIterator<Item = S>, S: Into<String>>(ids: I) -> Self {
        Self {
            expanded: ids.into_iter().map(Into::into).collect(),
            selected: BTreeSet::new(),
        }
    }

    pub fn is_expanded(&self, id: &str) -> bool {
        self.expanded.contains(id)
    }

    pub fn is_selected(&self, id: &str) -> bool {
        self.selected.contains(id)
    }

    pub fn selection(&self) -> impl Iterator<Item = &str> {
        self.selected.iter().map(String::as_str)
    }

    pub fn toggle_expanded(&mut self, id: &str) {
        if !self.expanded.remove(id) {
            self.expanded.insert(id.to_owned());
        }
    }

    /// Replace the selection with a single node.
    pub fn select(&mut self, id: &str) {
        self.selected.clear();
        self.selected.insert(id.to_owned());
    }

    /// Additive/toggle selection (cmd/ctrl-click).
    pub fn toggle_selected(&mut self, id: &str) {
        if !self.selected.remove(id) {
            self.selected.insert(id.to_owned());
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected.clear();
    }

    /// Default handling: expand/collapse toggles, a click replaces the
    /// selection. The app can intercept [`TreeEvent`] instead when it needs
    /// modifier-aware behaviour.
    pub fn update(&mut self, event: TreeEvent) {
        match event {
            TreeEvent::Toggled(id) => self.toggle_expanded(&id),
            TreeEvent::Clicked(id) => self.select(&id),
        }
    }

    /// Drop expansion/selection for nodes that no longer exist. Call after a
    /// document edit so a deleted piece does not linger in the selection.
    pub fn retain_existing(&mut self, roots: &[TreeNode]) {
        let mut live = BTreeSet::new();
        fn walk(node: &TreeNode, live: &mut BTreeSet<String>) {
            live.insert(node.id.clone());
            for c in &node.children {
                walk(c, live);
            }
        }
        for r in roots {
            walk(r, &mut live);
        }
        self.expanded.retain(|id| live.contains(id));
        self.selected.retain(|id| live.contains(id));
    }
}

/// Render the hierarchy.
pub fn tree<'a, Message, F>(
    tokens: &'a Tokens,
    roots: &'a [TreeNode],
    state: &'a TreeState,
    on_event: F,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(TreeEvent) -> Message + Clone + 'a,
{
    let mut rows: Vec<Element<'a, Message>> = Vec::new();
    for node in roots {
        push_rows(node, 0, tokens, state, &on_event, &mut rows);
    }

    scrollable(column(rows).width(Length::Fill))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn push_rows<'a, Message, F>(
    node: &'a TreeNode,
    depth: usize,
    tokens: &'a Tokens,
    state: &'a TreeState,
    on_event: &F,
    out: &mut Vec<Element<'a, Message>>,
) where
    Message: Clone + 'a,
    F: Fn(TreeEvent) -> Message + Clone + 'a,
{
    out.push(tree_row(node, depth, tokens, state, on_event));
    if state.is_expanded(&node.id) {
        for child in &node.children {
            push_rows(child, depth + 1, tokens, state, on_event, out);
        }
    }
}

fn tree_row<'a, Message, F>(
    node: &'a TreeNode,
    depth: usize,
    tokens: &'a Tokens,
    state: &'a TreeState,
    on_event: &F,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(TreeEvent) -> Message + Clone + 'a,
{
    let c = tokens.color;
    let selected = state.is_selected(&node.id);
    let fg = if selected {
        c.text_primary
    } else {
        c.text_secondary
    };
    let indent = tokens.space.lg * depth as f32;

    let disclosure: Element<'a, Message> = if node.children.is_empty() {
        space::horizontal()
            .width(Length::Fixed(tokens.metrics.icon))
            .into()
    } else {
        let glyph = if state.is_expanded(&node.id) {
            Icon::ChevronDown
        } else {
            Icon::ChevronRight
        };
        button(icon(glyph, tokens.metrics.icon, c.text_muted))
            .padding(0)
            .style(crate::ghost_button(tokens))
            .on_press(on_event(TreeEvent::Toggled(node.id.clone())))
            .into()
    };

    let mut label = row![disclosure]
        .spacing(tokens.space.xs)
        .align_y(iced::Center);
    if let Some(glyph) = node.icon {
        label = label.push(icon(glyph, tokens.metrics.icon, fg));
    }
    label = label.push(
        text(node.label.as_str())
            .size(tokens.text.small)
            .color(color(fg)),
    );

    let selection = c.selection;
    container(
        button(label)
            .width(Length::Fill)
            .padding([tokens.space.xs, tokens.space.sm])
            .style(move |_theme, status| {
                let bg = match (selected, status) {
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
                        color: color(if selected { selection } else { c.border }),
                        width: if selected {
                            tokens.metrics.border_width
                        } else {
                            0.0
                        },
                        radius: tokens.radius.sm.into(),
                    },
                    ..button::Style::default()
                }
            })
            .on_press(on_event(TreeEvent::Clicked(node.id.clone()))),
    )
    .padding(iced::Padding::new(0.0).left(indent))
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<TreeNode> {
        vec![
            TreeNode::new("front", "Front bodice").with_children(vec![
                TreeNode::new("front/dart-l", "Left dart"),
                TreeNode::new("front/dart-r", "Right dart"),
            ]),
            TreeNode::new("back", "Back bodice"),
        ]
    }

    #[test]
    fn expand_and_collapse_round_trip() {
        let mut s = TreeState::new();
        assert!(!s.is_expanded("front"));
        s.update(TreeEvent::Toggled("front".into()));
        assert!(s.is_expanded("front"));
        s.update(TreeEvent::Toggled("front".into()));
        assert!(!s.is_expanded("front"));
    }

    #[test]
    fn a_click_replaces_the_selection_and_toggle_extends_it() {
        let mut s = TreeState::new();
        s.update(TreeEvent::Clicked("front".into()));
        assert_eq!(s.selection().collect::<Vec<_>>(), ["front"]);

        s.update(TreeEvent::Clicked("back".into()));
        assert_eq!(s.selection().collect::<Vec<_>>(), ["back"]);

        s.toggle_selected("front");
        assert_eq!(s.selection().collect::<Vec<_>>(), ["back", "front"]);
        s.toggle_selected("front");
        assert_eq!(s.selection().collect::<Vec<_>>(), ["back"]);

        s.clear_selection();
        assert_eq!(s.selection().count(), 0);
    }

    #[test]
    fn stale_ids_are_dropped_when_the_document_changes() {
        let mut s = TreeState::expanded(["front", "gone"]);
        s.select("front/dart-l");
        s.toggle_selected("vanished");

        s.retain_existing(&sample());

        assert!(s.is_expanded("front"));
        assert!(!s.is_expanded("gone"));
        assert_eq!(s.selection().collect::<Vec<_>>(), ["front/dart-l"]);
    }
}
