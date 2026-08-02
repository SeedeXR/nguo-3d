// SPDX-License-Identifier: GPL-2.0-or-later
//! Single-window docking (guide §26.6 resolves to single-window docking).
//!
//! Split-pane tree with draggable dividers, collapsible panels and a tab strip.
//!
//! The layout **maths is framework-free and lives in [`DockLayout`]** — no iced
//! type appears in it — so every invariant below is unit-testable without a
//! window. The `dock()` view is a pure function of that state (guide §0.5:
//! never store authoritative state in a widget). The app owns the
//! `DockLayout` and feeds it events.
//!
//! Invariants held by construction:
//! - a split's ratios are all `> 0` and sum to `1`
//! - `sizes()` never returns a negative extent
//! - `sizes()` sums to the container extent (dividers included)
//! - collapse/expand does not touch ratios, so it round-trips exactly

use iced::widget::{button, column, container, mouse_area, row, space, text};
use iced::{mouse, Element, Length};
use nguo_ui_theme::Tokens;

use crate::icon::{icon, Icon};
use crate::{color, hairline, surface};

/// Smallest extent a panel may be dragged down to, in logical px.
pub const MIN_PANEL: f32 = 64.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Children sit side by side; dividers move horizontally.
    Horizontal,
    /// Children stack; dividers move vertically.
    Vertical,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tab {
    pub id: String,
    pub title: String,
    pub icon: Option<Icon>,
}

impl Tab {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            icon: None,
        }
    }

    pub fn with_icon(mut self, icon: Icon) -> Self {
        self.icon = Some(icon);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Panel {
    pub id: String,
    pub tabs: Vec<Tab>,
    pub active: usize,
    pub collapsed: bool,
}

impl Panel {
    pub fn new(id: impl Into<String>, tabs: Vec<Tab>) -> Self {
        Self {
            id: id.into(),
            tabs,
            active: 0,
            collapsed: false,
        }
    }

    pub fn active_tab(&self) -> Option<&Tab> {
        self.tabs.get(self.active)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Leaf(Panel),
    Split(Split),
}

impl Node {
    pub fn leaf(panel: Panel) -> Self {
        Node::Leaf(panel)
    }

    /// A collapsed node contributes only its header to the layout.
    fn is_collapsed(&self) -> bool {
        matches!(self, Node::Leaf(p) if p.collapsed)
    }
}

/// An n-way split. `ratios[i]` is child `i`'s share of the *expanded* space.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub axis: Axis,
    children: Vec<Node>,
    ratios: Vec<f32>,
}

impl Split {
    /// Equal shares.
    pub fn new(axis: Axis, children: Vec<Node>) -> Self {
        let n = children.len().max(1);
        let ratios = vec![1.0 / n as f32; children.len()];
        Self {
            axis,
            children,
            ratios,
        }
        .normalised()
    }

    /// Explicit shares, normalised to sum to 1. Non-finite or non-positive
    /// entries fall back to an equal split — a bad ratio must not be able to
    /// produce a negative panel.
    pub fn with_ratios(axis: Axis, children: Vec<Node>, ratios: Vec<f32>) -> Self {
        let ratios =
            if ratios.len() == children.len() && ratios.iter().all(|r| r.is_finite() && *r > 0.0) {
                ratios
            } else {
                vec![1.0; children.len()]
            };
        Self {
            axis,
            children,
            ratios,
        }
        .normalised()
    }

    fn normalised(mut self) -> Self {
        let sum: f32 = self.ratios.iter().sum();
        if sum > 0.0 {
            for r in &mut self.ratios {
                *r /= sum;
            }
        }
        self
    }

    pub fn children(&self) -> &[Node] {
        &self.children
    }

    pub fn ratios(&self) -> &[f32] {
        &self.ratios
    }

    /// Number of draggable dividers.
    pub fn divider_count(&self) -> usize {
        self.children.len().saturating_sub(1)
    }

    /// Extent of each child along `axis` for a container of `total` px.
    ///
    /// Collapsed children get exactly `collapsed_extent`; the rest share what
    /// is left after dividers, by ratio. The result always sums to `total`
    /// (when `total` can cover the fixed parts) and is never negative.
    pub fn sizes(&self, total: f32, collapsed_extent: f32, divider: f32) -> Vec<f32> {
        let n = self.children.len();
        if n == 0 {
            return Vec::new();
        }

        let dividers = divider * self.divider_count() as f32;
        let fixed: f32 = self
            .children
            .iter()
            .filter(|c| c.is_collapsed())
            .map(|_| collapsed_extent)
            .sum();

        let free = (total - dividers - fixed).max(0.0);
        let expanded_ratio: f32 = self
            .children
            .iter()
            .zip(&self.ratios)
            .filter(|(c, _)| !c.is_collapsed())
            .map(|(_, r)| *r)
            .sum();

        self.children
            .iter()
            .zip(&self.ratios)
            .map(|(child, r)| {
                if child.is_collapsed() {
                    collapsed_extent
                } else if expanded_ratio > 0.0 {
                    free * r / expanded_ratio
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Move divider `index` (between children `index` and `index + 1`) by
    /// `delta` px inside a container of `total` px.
    ///
    /// Both neighbours stay at or above [`MIN_PANEL`]; the pair's combined
    /// extent — and therefore the sum of all ratios — is preserved exactly.
    /// A drag against a collapsed neighbour is a no-op: you resize a collapsed
    /// panel by expanding it.
    pub fn drag_divider(
        &mut self,
        index: usize,
        delta: f32,
        total: f32,
        collapsed_extent: f32,
        divider: f32,
    ) {
        if index + 1 >= self.children.len() || !delta.is_finite() {
            return;
        }
        if self.children[index].is_collapsed() || self.children[index + 1].is_collapsed() {
            return;
        }

        let sizes = self.sizes(total, collapsed_extent, divider);
        let (a, b) = (sizes[index], sizes[index + 1]);
        let pair = a + b;
        if pair < 2.0 * MIN_PANEL {
            return; // no room to honour the minimum on both sides
        }

        let new_a = (a + delta).clamp(MIN_PANEL, pair - MIN_PANEL);
        let ratio_pair = self.ratios[index] + self.ratios[index + 1];
        self.ratios[index] = ratio_pair * new_a / pair;
        self.ratios[index + 1] = ratio_pair - self.ratios[index];
    }
}

/// Which divider the pointer grabbed, and the container it lives in.
///
/// `total` is captured by the view (which knows the laid-out size) so the app
/// never has to guess it.
#[derive(Debug, Clone, PartialEq)]
pub struct DividerGrab {
    pub path: Vec<usize>,
    pub index: usize,
    pub axis: Axis,
    pub total: f32,
}

/// Everything the dock view can emit. The app maps these into its own message
/// type and hands them straight back to [`DockLayout::update`].
#[derive(Debug, Clone, PartialEq)]
pub enum DockEvent {
    DividerGrabbed(DividerGrab),
    DividerReleased,
    CollapseToggled(String),
    TabSelected { panel: String, index: usize },
}

#[derive(Debug, Clone, PartialEq)]
struct Drag {
    grab: DividerGrab,
    last: f32,
}

/// The dock's authoritative state. Owned by the app, never by a widget.
#[derive(Debug, Clone, PartialEq)]
pub struct DockLayout {
    pub root: Node,
    /// Extent of a collapsed panel: its header strip only.
    pub collapsed_extent: f32,
    /// Divider grab-strip thickness.
    pub divider: f32,
    drag: Option<Drag>,
}

impl DockLayout {
    pub fn new(root: Node, tokens: &Tokens) -> Self {
        Self {
            root,
            collapsed_extent: tokens.metrics.panel_header,
            divider: tokens.metrics.divider,
            drag: None,
        }
    }

    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Apply a view event. Divider *motion* arrives separately via
    /// [`DockLayout::drag_to`], because only the app sees the window-level
    /// cursor once the pointer leaves the 6px grab strip.
    pub fn update(&mut self, event: DockEvent) {
        match event {
            DockEvent::DividerGrabbed(grab) => {
                self.drag = Some(Drag {
                    last: f32::NAN,
                    grab,
                });
            }
            DockEvent::DividerReleased => self.drag = None,
            DockEvent::CollapseToggled(id) => {
                if let Some(p) = self.panel_mut(&id) {
                    p.collapsed = !p.collapsed;
                }
            }
            DockEvent::TabSelected { panel, index } => {
                if let Some(p) = self.panel_mut(&panel) {
                    if index < p.tabs.len() {
                        p.active = index;
                    }
                }
            }
        }
    }

    /// Feed the current cursor position while a divider is grabbed.
    /// The first call after a grab only establishes the origin.
    pub fn drag_to(&mut self, cursor: (f32, f32)) {
        let Some(drag) = self.drag.as_mut() else {
            return;
        };
        let along = match drag.grab.axis {
            Axis::Horizontal => cursor.0,
            Axis::Vertical => cursor.1,
        };
        let last = drag.last;
        drag.last = along;
        if !last.is_finite() {
            return;
        }

        let (delta, grab) = (along - last, drag.grab.clone());
        let (collapsed_extent, divider) = (self.collapsed_extent, self.divider);
        if let Some(split) = self.split_at_mut(&grab.path) {
            split.drag_divider(grab.index, delta, grab.total, collapsed_extent, divider);
        }
    }

    pub fn split_at_mut(&mut self, path: &[usize]) -> Option<&mut Split> {
        let mut node = &mut self.root;
        for &i in path {
            let Node::Split(s) = node else { return None };
            node = s.children.get_mut(i)?;
        }
        match node {
            Node::Split(s) => Some(s),
            Node::Leaf(_) => None,
        }
    }

    pub fn panel(&self, id: &str) -> Option<&Panel> {
        fn walk<'a>(node: &'a Node, id: &str) -> Option<&'a Panel> {
            match node {
                Node::Leaf(p) if p.id == id => Some(p),
                Node::Leaf(_) => None,
                Node::Split(s) => s.children.iter().find_map(|c| walk(c, id)),
            }
        }
        walk(&self.root, id)
    }

    pub fn panel_mut(&mut self, id: &str) -> Option<&mut Panel> {
        fn walk<'a>(node: &'a mut Node, id: &str) -> Option<&'a mut Panel> {
            match node {
                Node::Leaf(p) => (p.id == id).then_some(p),
                Node::Split(s) => s.children.iter_mut().find_map(|c| walk(c, id)),
            }
        }
        walk(&mut self.root, id)
    }
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

/// Render the dock.
///
/// `content` supplies the body for a tab; the dock owns only chrome. Wrap the
/// result in nothing — it fills its container.
pub fn dock<'a, Message, F, C>(
    layout: &'a DockLayout,
    tokens: &'a Tokens,
    on_event: F,
    content: C,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(DockEvent) -> Message + Clone + 'a,
    C: Fn(&Panel, &Tab) -> Element<'a, Message> + Clone + 'a,
{
    // `responsive` hands us the laid-out size, which is how a grabbed divider
    // can report the exact container extent it is dividing.
    iced::widget::responsive(move |size| {
        node_view(
            layout,
            &layout.root,
            Vec::new(),
            size.width,
            size.height,
            tokens,
            &on_event,
            &content,
        )
    })
    .into()
}

#[allow(clippy::too_many_arguments)]
fn node_view<'a, Message, F, C>(
    layout: &'a DockLayout,
    node: &'a Node,
    path: Vec<usize>,
    width: f32,
    height: f32,
    tokens: &'a Tokens,
    on_event: &F,
    content: &C,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(DockEvent) -> Message + Clone + 'a,
    C: Fn(&Panel, &Tab) -> Element<'a, Message> + Clone + 'a,
{
    match node {
        Node::Leaf(panel) => panel_view(panel, tokens, on_event, content),
        Node::Split(split) => {
            let extent = match split.axis {
                Axis::Horizontal => width,
                Axis::Vertical => height,
            };
            let sizes = split.sizes(extent, layout.collapsed_extent, layout.divider);

            let mut items: Vec<Element<'a, Message>> = Vec::with_capacity(split.children.len() * 2);
            for (i, child) in split.children.iter().enumerate() {
                if i > 0 {
                    items.push(divider_view(
                        DividerGrab {
                            path: path.clone(),
                            index: i - 1,
                            axis: split.axis,
                            total: extent,
                        },
                        tokens,
                        on_event,
                    ));
                }

                let mut child_path = path.clone();
                child_path.push(i);
                let (w, h) = match split.axis {
                    Axis::Horizontal => (sizes[i], height),
                    Axis::Vertical => (width, sizes[i]),
                };
                let element = node_view(layout, child, child_path, w, h, tokens, on_event, content);
                items.push(
                    container(element)
                        .width(match split.axis {
                            Axis::Horizontal => Length::Fixed(w),
                            Axis::Vertical => Length::Fill,
                        })
                        .height(match split.axis {
                            Axis::Horizontal => Length::Fill,
                            Axis::Vertical => Length::Fixed(h),
                        })
                        .clip(true)
                        .into(),
                );
            }

            match split.axis {
                Axis::Horizontal => row(items).into(),
                Axis::Vertical => column(items).into(),
            }
        }
    }
}

fn divider_view<'a, Message, F>(
    grab: DividerGrab,
    tokens: &'a Tokens,
    on_event: &F,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(DockEvent) -> Message + Clone + 'a,
{
    let thickness = Length::Fixed(tokens.metrics.divider);
    let (w, h) = match grab.axis {
        Axis::Horizontal => (thickness, Length::Fill),
        Axis::Vertical => (Length::Fill, thickness),
    };
    let line = tokens.color.border;

    mouse_area(
        container(space::horizontal())
            .width(w)
            .height(h)
            .style(move |_| container::Style {
                background: Some(color(line).into()),
                ..container::Style::default()
            }),
    )
    .interaction(match grab.axis {
        Axis::Horizontal => mouse::Interaction::ResizingHorizontally,
        Axis::Vertical => mouse::Interaction::ResizingVertically,
    })
    .on_press(on_event(DockEvent::DividerGrabbed(grab)))
    .on_release(on_event(DockEvent::DividerReleased))
    .into()
}

fn panel_view<'a, Message, F, C>(
    panel: &'a Panel,
    tokens: &'a Tokens,
    on_event: &F,
    content: &C,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(DockEvent) -> Message + Clone + 'a,
    C: Fn(&Panel, &Tab) -> Element<'a, Message> + Clone + 'a,
{
    let header = tab_strip(panel, tokens, on_event);

    if panel.collapsed {
        return container(header)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(surface(tokens, tokens.color.panel))
            .into();
    }

    let body = panel
        .active_tab()
        .map(|tab| content(panel, tab))
        .unwrap_or_else(|| space::horizontal().into());

    container(column![
        header,
        container(body).width(Length::Fill).height(Length::Fill)
    ])
    .width(Length::Fill)
    .height(Length::Fill)
    .style(surface(tokens, tokens.color.panel))
    .into()
}

fn tab_strip<'a, Message, F>(
    panel: &'a Panel,
    tokens: &'a Tokens,
    on_event: &F,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
    F: Fn(DockEvent) -> Message + Clone + 'a,
{
    let mut tabs: Vec<Element<'a, Message>> = Vec::with_capacity(panel.tabs.len());
    for (i, tab) in panel.tabs.iter().enumerate() {
        let active = i == panel.active && !panel.collapsed;
        let fg = if active {
            tokens.color.text_primary
        } else {
            tokens.color.text_muted
        };

        let mut label = row![text(tab.title.as_str())
            .size(tokens.text.small)
            .color(color(fg))]
        .spacing(tokens.space.sm)
        .align_y(iced::Center);
        if let Some(glyph) = tab.icon {
            label = row![icon(glyph, tokens.metrics.icon, fg)]
                .push(
                    text(tab.title.as_str())
                        .size(tokens.text.small)
                        .color(color(fg)),
                )
                .spacing(tokens.space.sm)
                .align_y(iced::Center);
        }

        // The active tab is the one flush with the panel body below it; the
        // rest sit back in the header. No underline, no chrome — Figma/Blender
        // read the fill, not a decoration.
        let bg = if active {
            tokens.color.panel
        } else {
            tokens.color.header
        };

        tabs.push(
            button(label)
                .padding([tokens.space.xs, tokens.space.md])
                .height(Length::Fixed(tokens.metrics.tab_height))
                .style(move |_theme, status| {
                    let bg = match (active, status) {
                        (false, button::Status::Hovered | button::Status::Pressed) => {
                            tokens.color.overlay
                        }
                        _ => bg,
                    };
                    button::Style {
                        background: Some(color(bg).into()),
                        text_color: color(fg),
                        border: iced::Border {
                            color: color(tokens.color.border),
                            width: 0.0,
                            radius: tokens.radius.sm.into(),
                        },
                        ..button::Style::default()
                    }
                })
                .on_press(on_event(DockEvent::TabSelected {
                    panel: panel.id.clone(),
                    index: i,
                }))
                .into(),
        );
    }

    let chevron = if panel.collapsed {
        Icon::ChevronRight
    } else {
        Icon::ChevronDown
    };

    let toggle = button(icon(
        chevron,
        tokens.metrics.icon,
        tokens.color.text_secondary,
    ))
    .padding(tokens.space.xs)
    .style(crate::ghost_button(tokens))
    .on_press(on_event(DockEvent::CollapseToggled(panel.id.clone())));

    container(
        row![
            row(tabs).spacing(tokens.space.xs),
            space::horizontal(),
            toggle
        ]
        .align_y(iced::Center)
        .padding([0.0, tokens.space.sm]),
    )
    .width(Length::Fill)
    .height(Length::Fixed(tokens.metrics.panel_header))
    .style(hairline(tokens, tokens.color.header))
    .into()
}

// ---------------------------------------------------------------------------
// Tests — the layout maths, no window required
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const COLLAPSED: f32 = 28.0;
    const DIVIDER: f32 = 6.0;
    const EPS: f32 = 1e-3;

    fn leaf(id: &str) -> Node {
        Node::leaf(Panel::new(id, vec![Tab::new(id, id)]))
    }

    fn three() -> Split {
        Split::new(Axis::Horizontal, vec![leaf("a"), leaf("b"), leaf("c")])
    }

    fn sum(v: &[f32]) -> f32 {
        v.iter().sum()
    }

    #[test]
    fn ratios_sum_to_one_on_construction() {
        for n in 1..8 {
            let split = Split::new(
                Axis::Vertical,
                (0..n).map(|i| leaf(&i.to_string())).collect(),
            );
            assert!((sum(split.ratios()) - 1.0).abs() < EPS, "n = {n}");
        }
        // Un-normalised input is normalised.
        let s = Split::with_ratios(Axis::Horizontal, vec![leaf("a"), leaf("b")], vec![3.0, 1.0]);
        assert!((sum(s.ratios()) - 1.0).abs() < EPS);
        assert!((s.ratios()[0] - 0.75).abs() < EPS);
        // Garbage input falls back to an equal split rather than a negative panel.
        let bad = Split::with_ratios(
            Axis::Horizontal,
            vec![leaf("a"), leaf("b")],
            vec![-4.0, f32::NAN],
        );
        assert_eq!(bad.ratios(), &[0.5, 0.5]);
    }

    #[test]
    fn ratios_still_sum_to_one_after_any_drag() {
        let mut split = three();
        // A deterministic sweep of deltas, including absurd ones.
        for step in 0..200 {
            let delta = ((step * 37) % 401) as f32 - 200.0;
            let divider = step % 2;
            split.drag_divider(divider, delta, 900.0, COLLAPSED, DIVIDER);
            let s = sum(split.ratios());
            assert!(
                (s - 1.0).abs() < EPS,
                "ratios drifted to {s} after step {step}"
            );
            assert!(split.ratios().iter().all(|r| *r > 0.0));
        }
    }

    #[test]
    fn sizes_sum_to_the_container_extent() {
        let split = three();
        for total in [200.0_f32, 640.0, 1337.5, 4000.0] {
            let s = split.sizes(total, COLLAPSED, DIVIDER);
            let expected = total - DIVIDER * split.divider_count() as f32;
            assert!(
                (sum(&s) - expected).abs() < EPS,
                "total {total}: panels {} + dividers {} != {total}",
                sum(&s),
                DIVIDER * split.divider_count() as f32
            );
        }
    }

    #[test]
    fn a_divider_drag_preserves_total_size() {
        let mut split = three();
        let total = 900.0;
        let before = sum(&split.sizes(total, COLLAPSED, DIVIDER));
        for delta in [-300.0, -1.0, 0.5, 17.0, 250.0] {
            split.drag_divider(1, delta, total, COLLAPSED, DIVIDER);
            let after = sum(&split.sizes(total, COLLAPSED, DIVIDER));
            assert!(
                (after - before).abs() < EPS,
                "delta {delta}: {before} -> {after}"
            );
        }
    }

    #[test]
    fn a_drag_only_moves_the_two_panels_it_touches() {
        let mut split = three();
        let total = 900.0;
        let before = split.sizes(total, COLLAPSED, DIVIDER);
        split.drag_divider(0, 60.0, total, COLLAPSED, DIVIDER);
        let after = split.sizes(total, COLLAPSED, DIVIDER);
        assert!((after[0] - (before[0] + 60.0)).abs() < EPS);
        assert!((after[1] - (before[1] - 60.0)).abs() < EPS);
        assert!((after[2] - before[2]).abs() < EPS, "third panel moved");
    }

    #[test]
    fn sizes_never_go_negative_at_extreme_drags() {
        let mut split = three();
        for total in [150.0_f32, 400.0, 2000.0] {
            for delta in [-1e9, -5000.0, 5000.0, 1e9, f32::MAX] {
                for d in 0..split.divider_count() {
                    split.drag_divider(d, delta, total, COLLAPSED, DIVIDER);
                    let s = split.sizes(total, COLLAPSED, DIVIDER);
                    assert!(
                        s.iter().all(|x| *x >= 0.0 && x.is_finite()),
                        "total {total} delta {delta}: {s:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_drag_respects_the_minimum_panel_size() {
        let mut split = three();
        let total = 900.0;
        split.drag_divider(0, -10_000.0, total, COLLAPSED, DIVIDER);
        let s = split.sizes(total, COLLAPSED, DIVIDER);
        assert!(s[0] >= MIN_PANEL - EPS, "left panel collapsed to {}", s[0]);

        split.drag_divider(0, 10_000.0, total, COLLAPSED, DIVIDER);
        let s = split.sizes(total, COLLAPSED, DIVIDER);
        assert!(s[1] >= MIN_PANEL - EPS, "right panel collapsed to {}", s[1]);
    }

    #[test]
    fn a_drag_is_a_no_op_when_the_container_is_too_small() {
        let mut split = Split::new(Axis::Horizontal, vec![leaf("a"), leaf("b")]);
        let before = split.ratios().to_vec();
        split.drag_divider(0, 40.0, 2.0 * MIN_PANEL - 20.0, COLLAPSED, DIVIDER);
        assert_eq!(split.ratios(), before.as_slice());
    }

    #[test]
    fn collapsing_and_restoring_a_panel_round_trips() {
        let tokens = Tokens::dark();
        let mut layout = DockLayout::new(Node::Split(three()), &tokens);
        layout.collapsed_extent = COLLAPSED;
        layout.divider = DIVIDER;

        // Drag first, so we round-trip a non-default layout.
        if let Some(s) = layout.split_at_mut(&[]) {
            s.drag_divider(0, 90.0, 900.0, COLLAPSED, DIVIDER);
        }
        let before = layout
            .split_at_mut(&[])
            .unwrap()
            .sizes(900.0, COLLAPSED, DIVIDER);
        let ratios_before = layout.split_at_mut(&[]).unwrap().ratios().to_vec();

        layout.update(DockEvent::CollapseToggled("b".into()));
        assert!(layout.panel("b").unwrap().collapsed);
        let collapsed = layout
            .split_at_mut(&[])
            .unwrap()
            .sizes(900.0, COLLAPSED, DIVIDER);
        assert!((collapsed[1] - COLLAPSED).abs() < EPS);
        assert!(collapsed.iter().all(|s| *s >= 0.0));
        assert!((sum(&collapsed) - (900.0 - 2.0 * DIVIDER)).abs() < EPS);

        layout.update(DockEvent::CollapseToggled("b".into()));
        assert!(!layout.panel("b").unwrap().collapsed);
        assert_eq!(
            layout.split_at_mut(&[]).unwrap().ratios(),
            &ratios_before[..]
        );
        let after = layout
            .split_at_mut(&[])
            .unwrap()
            .sizes(900.0, COLLAPSED, DIVIDER);
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < EPS, "{before:?} != {after:?}");
        }
    }

    #[test]
    fn dragging_against_a_collapsed_neighbour_does_nothing() {
        let tokens = Tokens::dark();
        let mut layout = DockLayout::new(Node::Split(three()), &tokens);
        layout.update(DockEvent::CollapseToggled("a".into()));
        let before = layout.split_at_mut(&[]).unwrap().ratios().to_vec();
        layout
            .split_at_mut(&[])
            .unwrap()
            .drag_divider(0, 120.0, 900.0, COLLAPSED, DIVIDER);
        assert_eq!(
            layout.split_at_mut(&[]).unwrap().ratios(),
            before.as_slice()
        );
    }

    #[test]
    fn nested_splits_are_addressable_by_path() {
        let inner = Node::Split(Split::new(Axis::Vertical, vec![leaf("b"), leaf("c")]));
        let root = Node::Split(Split::new(Axis::Horizontal, vec![leaf("a"), inner]));
        let mut layout = DockLayout::new(root, &Tokens::dark());

        assert!(layout.split_at_mut(&[]).is_some());
        assert!(layout.split_at_mut(&[1]).is_some());
        assert!(layout.split_at_mut(&[0]).is_none(), "a leaf is not a split");
        assert!(layout.split_at_mut(&[9]).is_none());
        assert!(layout.panel("c").is_some());
        assert!(layout.panel("zzz").is_none());
    }

    #[test]
    fn a_full_drag_gesture_moves_the_divider_once_the_origin_is_known() {
        let mut layout = DockLayout::new(Node::Split(three()), &Tokens::dark());
        layout.collapsed_extent = COLLAPSED;
        layout.divider = DIVIDER;
        let before = layout
            .split_at_mut(&[])
            .unwrap()
            .sizes(900.0, COLLAPSED, DIVIDER);

        layout.update(DockEvent::DividerGrabbed(DividerGrab {
            path: vec![],
            index: 0,
            axis: Axis::Horizontal,
            total: 900.0,
        }));
        assert!(layout.is_dragging());

        // First move only records the origin.
        layout.drag_to((300.0, 0.0));
        let unmoved = layout
            .split_at_mut(&[])
            .unwrap()
            .sizes(900.0, COLLAPSED, DIVIDER);
        assert!((unmoved[0] - before[0]).abs() < EPS);

        layout.drag_to((350.0, 0.0));
        let moved = layout
            .split_at_mut(&[])
            .unwrap()
            .sizes(900.0, COLLAPSED, DIVIDER);
        assert!((moved[0] - (before[0] + 50.0)).abs() < EPS);

        layout.update(DockEvent::DividerReleased);
        assert!(!layout.is_dragging());
        // Motion after release is ignored.
        layout.drag_to((900.0, 0.0));
        let after = layout
            .split_at_mut(&[])
            .unwrap()
            .sizes(900.0, COLLAPSED, DIVIDER);
        assert!((after[0] - moved[0]).abs() < EPS);
    }

    #[test]
    fn tab_selection_is_clamped_to_existing_tabs() {
        let panel = Panel::new("p", vec![Tab::new("t0", "One"), Tab::new("t1", "Two")]);
        let mut layout = DockLayout::new(Node::leaf(panel), &Tokens::dark());
        layout.update(DockEvent::TabSelected {
            panel: "p".into(),
            index: 1,
        });
        assert_eq!(layout.panel("p").unwrap().active, 1);
        layout.update(DockEvent::TabSelected {
            panel: "p".into(),
            index: 99,
        });
        assert_eq!(
            layout.panel("p").unwrap().active,
            1,
            "out-of-range accepted"
        );
    }
}
