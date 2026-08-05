// SPDX-License-Identifier: GPL-2.0-or-later
//! The workspace layout: which panels exist, how wide they may get, and how
//! the document projects into the scene tree.
//!
//! Everything in this module is framework-free arithmetic or plain data, so it
//! is testable without a window. The iced side lives in `main.rs`.

use nguo_doc::{Document, PieceId};
use nguo_ui_theme::Tokens;
use nguo_ui_widgets::dock::{Axis, DockLayout, Node, Panel, Split, Tab};
use nguo_ui_widgets::icon::Icon;
use nguo_ui_widgets::tree::TreeNode;

/// Panel ids. The dock addresses panels by string, so these are the one place
/// the strings are written.
pub const SCENE: &str = "scene";
pub const CANVAS: &str = "canvas2d";
pub const VIEWPORT: &str = "viewport3d";
pub const PROPERTIES: &str = "properties";

/// Indices of the root split's children. The root is a single 4-way horizontal
/// split, which is what makes [`enforce_canvas_minimum`] a two-line rule
/// instead of a tree walk.
/// **3D sits left of 2D**, as it does in both CLO3D and Marvelous Designer
/// (instructions/ui-references/, session-2 capture). The two work areas are
/// interchangeable to [`enforce_canvas_minimum`], so the swap is these two
/// constants and the order of the children below.
const I_SCENE: usize = 0;
const I_VIEWPORT: usize = 1;
const I_CANVAS: usize = 2;
const I_PROPERTIES: usize = 3;

/// The floor a work area may not be pushed below, in logical px.
///
/// UX_SPEC §2: opening two side panels must never squeeze the work areas down
/// to a sliver. 320 px is roughly where a pattern piece stops being editable
/// rather than merely visible.
pub const CANVAS_MIN: f32 = 320.0;

/// The shipped layout. Ratios put the two work areas at two thirds of the
/// window at 1280 px, which is also where [`enforce_canvas_minimum`] would put
/// them anyway.
///
/// The chrome columns come from the references, not from taste: both CLO3D and
/// Marvelous Designer land on a **~22.5 % right dock** and a narrow left rail,
/// while their two work areas split the middle almost exactly evenly (32.3 /
/// 32.2 %). Ours were 17 / 18 — a left rail twice as wide as either reference
/// and a properties column too narrow to put a number field in.
pub fn default_layout(tokens: &Tokens) -> DockLayout {
    let root = Node::Split(Split::with_ratios(
        Axis::Horizontal,
        vec![
            Node::leaf(Panel::new(
                SCENE,
                vec![Tab::new(SCENE, "Scene").with_icon(Icon::Layers)],
            )),
            Node::leaf(Panel::new(
                VIEWPORT,
                vec![Tab::new(VIEWPORT, "3D Garment").with_icon(Icon::Garment)],
            )),
            Node::leaf(Panel::new(
                CANVAS,
                vec![Tab::new(CANVAS, "2D Pattern").with_icon(Icon::Grid)],
            )),
            Node::leaf(Panel::new(
                PROPERTIES,
                vec![
                    Tab::new("props", "Properties").with_icon(Icon::Sliders),
                    Tab::new("params", "Parameters").with_icon(Icon::Settings),
                ],
            )),
        ],
        vec![0.11, 0.33, 0.33, 0.23],
    ));

    DockLayout::new(root, tokens)
}

/// Push the side panels back until both work areas clear [`CANVAS_MIN`], or
/// until the panels hit the dock's own `MIN_PANEL` floor and there is nothing
/// left to take.
///
/// `total` is the dock's logical width. Returns `true` if anything moved.
///
/// This is a rule the dock crate cannot own: `MIN_PANEL` is symmetric — every
/// child is equally entitled to it — whereas this says *these two children are
/// the work and the other two are chrome*. That priority is an application
/// decision, so it lives in the application.
pub fn enforce_canvas_minimum(layout: &mut DockLayout, total: f32) -> bool {
    if !total.is_finite() || total <= 0.0 {
        return false;
    }
    let (collapsed, divider) = (layout.collapsed_extent, layout.divider);
    let Some(split) = layout.split_at_mut(&[]) else {
        return false;
    };
    if split.children().len() != 4 {
        return false;
    }

    let before = split.ratios().to_vec();

    // Divider 0 sits between the scene tree and the first work area: a
    // negative delta shrinks the tree. Divider 2 sits between the second work
    // area and the properties panel: a positive delta shrinks the properties
    // panel. Both drags are clamped by the dock to `MIN_PANEL`, so a panel
    // yields exactly as far as it legally can and no further.
    for (work, divider_index, sign) in [(I_VIEWPORT, 0usize, -1.0f32), (I_CANVAS, 2usize, 1.0)] {
        let deficit = CANVAS_MIN - split.sizes(total, collapsed, divider)[work];
        if deficit > 0.0 {
            split.drag_divider(divider_index, sign * deficit, total, collapsed, divider);
        }
    }

    debug_assert_eq!(split.children().len(), 4);
    let _ = (I_SCENE, I_PROPERTIES);
    split.ratios() != before.as_slice()
}

// ---------------------------------------------------------------------------
// Scene tree — garment > piece > edge (UX_SPEC §3.3)
// ---------------------------------------------------------------------------

/// Project the document into tree rows.
///
/// Three real levels, all read from `nguo-doc`: the document is the garment,
/// its pieces are the pieces, and each piece's vector-network edges are the
/// edges. Ids are `garment`, `piece/<n>`, `piece/<n>/edge/<m>` — stable for as
/// long as the arena slot is, which is what [`TreeState`] keys on.
///
/// [`TreeState`]: nguo_ui_widgets::tree::TreeState
pub fn scene_tree(doc: &Document) -> Vec<TreeNode> {
    let pieces = doc
        .pieces()
        .iter()
        .map(|(id, piece)| {
            let edges = piece
                .network
                .edges
                .iter()
                .map(|(edge_id, edge)| {
                    let mm = piece.network.arc_length(edge_id).unwrap_or(0.0);
                    TreeNode::new(
                        format!("{}/edge/{}", piece_key(id), edge_id.raw()),
                        format!("{:?} · {mm:.0} mm", edge.semantic),
                    )
                    .with_icon(Icon::Line)
                })
                .collect();

            TreeNode::new(piece_key(id), piece.name.clone())
                .with_icon(Icon::Garment)
                .with_children(edges)
        })
        .collect();

    vec![TreeNode::new("garment", doc.meta().name.clone())
        .with_icon(Icon::Layers)
        .with_children(pieces)]
}

fn piece_key(id: PieceId) -> String {
    format!("piece/{}", id.raw())
}

/// The document the shell opens with: `nguo_sewing`'s two-panel block, **sewn**.
///
/// It used to be four unsewn rectangles authored here. It is not any more, and the
/// difference is the milestone: a document with no seams cannot be draped, so the
/// opening document being sewable is what makes pressing Space do something on a
/// fresh launch.
///
/// It lives in `nguo-sewing` rather than here because three callers need the *same*
/// garment — this shell, the end-to-end integration test, and the golden drape —
/// and a golden that authors its own geometry is testing the test.
///
/// ponytail: still rectangles. Replace with a real block once `file.open` exists;
/// what a rectangle cannot exercise is curved seam correspondence, and
/// `nguo_sewing::correspond`'s own tests cover that without a document.
pub fn sample_document() -> Document {
    nguo_sewing::sample::bodice_block()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_ui_widgets::dock::{DockEvent, MIN_PANEL};

    const WIDTH: f32 = 1280.0;

    fn sizes(layout: &mut DockLayout, total: f32) -> Vec<f32> {
        let (c, d) = (layout.collapsed_extent, layout.divider);
        layout.split_at_mut(&[]).unwrap().sizes(total, c, d)
    }

    #[test]
    fn the_default_layout_already_clears_the_canvas_floor() {
        let mut layout = default_layout(&Tokens::dark());
        let s = sizes(&mut layout, WIDTH);
        assert!(s[I_CANVAS] >= CANVAS_MIN, "2D canvas {} px", s[I_CANVAS]);
        assert!(
            s[I_VIEWPORT] >= CANVAS_MIN,
            "3D viewport {} px",
            s[I_VIEWPORT]
        );
        assert!(
            !enforce_canvas_minimum(&mut layout, WIDTH),
            "the shipped layout should not need correcting"
        );
    }

    /// The pane ORDER is load-bearing, not cosmetic: `enforce_canvas_minimum`
    /// hardcodes divider 0 and divider 2 as the ones that squeeze the chrome,
    /// which is only true while the two work areas are the middle two
    /// children. Reordering the `vec!` without swapping `I_VIEWPORT` and
    /// `I_CANVAS` would silently starve the wrong panel, and every other test
    /// in this file would still pass.
    #[test]
    fn the_work_areas_are_the_middle_two_panes_with_3d_first() {
        let layout = default_layout(&Tokens::dark());
        let ids: Vec<&str> = match &layout.root {
            Node::Split(s) => s
                .children()
                .iter()
                .map(|c| match c {
                    Node::Leaf(p) => p.id.as_str(),
                    Node::Split(_) => "split",
                })
                .collect(),
            Node::Leaf(_) => vec![],
        };
        assert_eq!(ids, vec![SCENE, VIEWPORT, CANVAS, PROPERTIES]);
        assert_eq!(ids[I_SCENE], SCENE);
        assert_eq!(ids[I_VIEWPORT], VIEWPORT);
        assert_eq!(ids[I_CANVAS], CANVAS);
        assert_eq!(ids[I_PROPERTIES], PROPERTIES);
    }

    /// The chrome columns come from the reference tools; the work areas split
    /// what is left evenly. Was 17 / 33 / 32 / 18 — a left rail twice either
    /// reference's and a right dock five points short of both.
    #[test]
    fn the_shipped_layout_matches_the_reference_proportions() {
        let mut layout = default_layout(&Tokens::dark());
        let s = sizes(&mut layout, WIDTH);
        let total: f32 = s.iter().sum();
        let share = |i: usize| s[i] / total;

        assert!(
            (0.20..0.25).contains(&share(I_PROPERTIES)),
            "right dock is {:.1} %, references land on 22.5",
            share(I_PROPERTIES) * 100.0
        );
        assert!(
            share(I_SCENE) < 0.14,
            "left rail is {:.1} %",
            share(I_SCENE) * 100.0
        );
        assert!(
            (share(I_CANVAS) - share(I_VIEWPORT)).abs() < 0.01,
            "the two work areas should split the middle evenly: {s:?}"
        );
    }

    #[test]
    fn side_panels_yield_before_the_canvases_do() {
        // The failure mode being prevented: two fat side panels, both work
        // areas starved.
        let tokens = Tokens::dark();
        let mut layout = default_layout(&tokens);
        {
            let (c, d) = (layout.collapsed_extent, layout.divider);
            let split = layout.split_at_mut(&[]).unwrap();
            split.drag_divider(0, 400.0, WIDTH, c, d);
            split.drag_divider(2, -400.0, WIDTH, c, d);
        }
        let starved = sizes(&mut layout, WIDTH);
        assert!(
            starved[I_CANVAS] < CANVAS_MIN || starved[I_VIEWPORT] < CANVAS_MIN,
            "test setup failed to starve anything: {starved:?}"
        );

        assert!(enforce_canvas_minimum(&mut layout, WIDTH));
        let fixed = sizes(&mut layout, WIDTH);
        assert!(fixed[I_CANVAS] >= CANVAS_MIN - 0.01, "{fixed:?}");
        assert!(fixed[I_VIEWPORT] >= CANVAS_MIN - 0.01, "{fixed:?}");
        assert!(
            fixed[I_SCENE] < starved[I_SCENE],
            "scene tree did not yield"
        );
        assert!(
            fixed[I_PROPERTIES] < starved[I_PROPERTIES],
            "properties did not yield"
        );
        assert!(fixed.iter().all(|s| *s >= 0.0));
    }

    #[test]
    fn panels_never_yield_past_their_own_minimum() {
        // 2 * CANVAS_MIN + 2 * MIN_PANEL is more than this window has, so the
        // rule must give up gracefully rather than produce a negative panel.
        let mut layout = default_layout(&Tokens::dark());
        let total = 2.0 * CANVAS_MIN + 2.0 * MIN_PANEL - 60.0;
        enforce_canvas_minimum(&mut layout, total);
        let s = sizes(&mut layout, total);
        assert!(s.iter().all(|x| x.is_finite() && *x >= 0.0), "{s:?}");
        assert!(s[I_SCENE] >= MIN_PANEL - 0.01, "{s:?}");
        assert!(s[I_PROPERTIES] >= MIN_PANEL - 0.01, "{s:?}");
    }

    #[test]
    fn the_rule_survives_absurd_widths_and_absurd_drags() {
        let mut layout = default_layout(&Tokens::dark());
        for total in [0.0_f32, -1.0, f32::NAN, 1.0, 120.0, 3840.0, 1e9] {
            for delta in [-1e9_f32, -5000.0, 5000.0, 1e9] {
                {
                    let (c, d) = (layout.collapsed_extent, layout.divider);
                    let split = layout.split_at_mut(&[]).unwrap();
                    for i in 0..3 {
                        split.drag_divider(i, delta, total.max(1.0), c, d);
                    }
                }
                enforce_canvas_minimum(&mut layout, total);
                let s = sizes(&mut layout, total.max(1.0));
                assert!(
                    s.iter().all(|x| x.is_finite() && *x >= 0.0),
                    "total {total} delta {delta}: {s:?}"
                );
            }
        }
    }

    #[test]
    fn a_collapsed_side_panel_hands_its_space_to_the_canvases() {
        let mut layout = default_layout(&Tokens::dark());
        let open = sizes(&mut layout, WIDTH)[I_CANVAS];
        layout.update(DockEvent::CollapseToggled(SCENE.into()));
        assert!(layout.panel(SCENE).unwrap().collapsed);
        let collapsed = sizes(&mut layout, WIDTH)[I_CANVAS];
        assert!(collapsed > open, "{open} -> {collapsed}");

        // ...and round-trips exactly, which is the dock's invariant, not ours.
        layout.update(DockEvent::CollapseToggled(SCENE.into()));
        assert!((sizes(&mut layout, WIDTH)[I_CANVAS] - open).abs() < 0.01);
    }

    #[test]
    fn the_scene_tree_is_garment_then_piece_then_edge() {
        let doc = sample_document();
        let roots = scene_tree(&doc);

        assert_eq!(roots.len(), 1, "one garment root");
        assert_eq!(roots[0].label, "Bodice block");
        assert_eq!(roots[0].children.len(), 2, "two pieces");
        assert_eq!(roots[0].children[0].label, "Front bodice");
        assert_eq!(roots[0].children[0].children.len(), 4, "four edges");
        assert!(roots[0].children[0].children[0].label.contains("mm"));

        // Ids are unique across the whole tree — `TreeState` is keyed on them.
        let mut ids = Vec::new();
        fn walk(n: &TreeNode, out: &mut Vec<String>) {
            out.push(n.id.clone());
            n.children.iter().for_each(|c| walk(c, out));
        }
        walk(&roots[0], &mut ids);
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len(), "duplicate tree id: {ids:?}");
    }
}
