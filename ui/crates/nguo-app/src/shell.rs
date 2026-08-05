// SPDX-License-Identifier: GPL-2.0-or-later
//! The workspace layout: which panels exist, how wide they may get, and how
//! the document projects into the scene tree.
//!
//! Everything in this module is framework-free arithmetic or plain data, so it
//! is testable without a window. The iced side lives in `main.rs`.

use nguo_canvas::CommandStack;
use nguo_doc::{Document, PieceId};
use nguo_ui_theme::Tokens;
use nguo_ui_widgets::dock::{Axis, DockLayout, Node, Panel, Split, Tab};
use nguo_ui_widgets::history::HistoryRow;
use nguo_ui_widgets::icon::Icon;
use nguo_ui_widgets::tree::TreeNode;

/// Panel ids. The dock addresses panels by string, so these are the one place
/// the strings are written.
pub const SCENE: &str = "scene";
pub const CANVAS: &str = "canvas2d";
pub const VIEWPORT: &str = "viewport3d";
pub const PROPERTIES: &str = "properties";

/// A tab id rather than a panel id — the history shares the right dock with
/// Properties and Parameters. It is a `const` because `main.rs` matches on it
/// too, and a typo in either place is a silently blank panel.
pub const HISTORY: &str = "history";

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
                    Tab::new(HISTORY, "History").with_icon(Icon::Undo),
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

// ---------------------------------------------------------------------------
// History panel — the undo stack as rows
// ---------------------------------------------------------------------------

/// Project the command stack into history rows.
///
/// **A row index IS the `CommandStack` position it jumps to.** Row 0 is the
/// document with nothing the stack still remembers applied, row `i` is the state
/// after entry `i`. That identity is the point: `HistoryEvent::Jump(i)` becomes
/// `stack.jump_to(doc, i)` with no arithmetic in between, so there is no
/// off-by-one to get wrong between the panel and the stack.
///
/// Rows past [`CommandStack::position`] are the **undone tail** and are marked
/// unapplied, which the widget greys — a jump backwards keeps them clickable,
/// and the next edit truncates them (D-2: the history is linear). Greying is how
/// that truncation stops being silent.
///
/// Row 0 is labelled with the document name because that is the state it names.
/// Once the 256-entry cap has dropped the oldest entries it is no longer the
/// document as it was opened, only as far back as the history still goes — which
/// is the same thing Photoshop's top row means.
pub fn history_rows(stack: &CommandStack, doc: &Document) -> Vec<HistoryRow> {
    let position = stack.position();
    let mut rows = Vec::with_capacity(stack.total() + 1);
    rows.push(HistoryRow::new(doc.meta().name.clone(), true));
    rows.extend(
        stack
            .entries()
            .enumerate()
            .map(|(i, e)| HistoryRow::new(e.label(), i < position)),
    );
    rows
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

    // -----------------------------------------------------------------------
    // History panel
    // -----------------------------------------------------------------------

    /// A document with `n` recorded point-adds, plus the JSON of the document at
    /// every position 0..=n. `snapshots[p]` is the oracle for a jump to `p`.
    fn history_fixture(n: usize) -> (Document, CommandStack, Vec<String>) {
        let mut doc = sample_document();
        let mut stack = CommandStack::new();
        let piece = doc.pieces().iter().next().expect("a piece").0;
        let mut snapshots = vec![doc.to_json().unwrap()];
        for i in 1..=n {
            let id = doc.piece(piece).unwrap().network.vertices.next_id();
            stack
                .apply(
                    &mut doc,
                    nguo_doc::Edit::AddVertex {
                        piece,
                        id,
                        pos: vello::kurbo::Point::new(i as f64, 0.0),
                    },
                )
                .unwrap();
            snapshots.push(doc.to_json().unwrap());
        }
        (doc, stack, snapshots)
    }

    fn labels(rows: &[HistoryRow]) -> Vec<&str> {
        rows.iter().map(|r| r.label.as_str()).collect()
    }

    /// The panel's whole contract: **clicking row `i` lands on the document that
    /// state had.** Headless — build history, jump, compare JSON.
    ///
    /// This is also what pins `history_rows`' index/position identity. Shift the
    /// projection by one and every assertion here fails.
    #[test]
    fn clicking_a_row_lands_on_the_document_that_state_had() {
        let (mut doc, mut stack, snapshots) = history_fixture(4);
        assert_eq!(history_rows(&stack, &doc).len(), snapshots.len());

        // Back, forward, both ends, and the same row twice — not a walk, so a
        // `jump_to` that only ever stepped one way would fail here.
        for row in [2usize, 0, 4, 1, 4, 3, 3] {
            stack.jump_to(&mut doc, row).unwrap();
            assert_eq!(
                doc.to_json().unwrap(),
                snapshots[row],
                "clicking history row {row}"
            );
            // ...and the panel now marks that row as where we are.
            let rows = history_rows(&stack, &doc);
            assert_eq!(
                nguo_ui_widgets::history::current(&rows),
                Some(row),
                "row {row} should be the current one"
            );
        }
    }

    /// Row 0 is the document itself and is always behind you, never greyed —
    /// otherwise there is no row to click to undo everything.
    #[test]
    fn row_zero_is_the_document_and_is_never_greyed() {
        let (mut doc, mut stack, _) = history_fixture(3);
        for target in [3usize, 1, 0] {
            stack.jump_to(&mut doc, target).unwrap();
            let rows = history_rows(&stack, &doc);
            assert_eq!(rows[0].label, "Bodice block");
            assert!(rows[0].applied, "at position {target}");
        }
    }

    /// Jumping back greys the tail rather than hiding it, and the next edit
    /// removes those rows. That visible truncation is the whole reason the
    /// greyed state exists — a linear history (D-2) throws the tail away, and
    /// doing it silently is the failure mode.
    #[test]
    fn the_undone_tail_is_greyed_and_a_new_edit_removes_those_rows() {
        let (mut doc, mut stack, _) = history_fixture(4);
        assert!(history_rows(&stack, &doc).iter().all(|r| r.applied));

        stack.jump_to(&mut doc, 2).unwrap();
        let rows = history_rows(&stack, &doc);
        assert_eq!(rows.len(), 5, "nothing is dropped by going back");
        assert!(rows[..=2].iter().all(|r| r.applied));
        assert!(
            rows[3..].iter().all(|r| !r.applied),
            "the two undone entries must read as greyed"
        );

        // A new edit at position 2 truncates rows 3 and 4.
        let piece = doc.pieces().iter().next().unwrap().0;
        let id = doc.piece(piece).unwrap().network.vertices.next_id();
        stack
            .apply(
                &mut doc,
                nguo_doc::Edit::AddVertex {
                    piece,
                    id,
                    pos: vello::kurbo::Point::new(99.0, 0.0),
                },
            )
            .unwrap();
        let rows = history_rows(&stack, &doc);
        assert_eq!(rows.len(), 4, "the greyed rows are gone, not merely hidden");
        assert!(rows.iter().all(|r| r.applied));
    }

    /// One drag is one row. A drag emits a command per pointer move, and a panel
    /// that showed sixty "Move point" rows for one gesture would be unusable —
    /// this is the coalescing promise seen from the panel rather than the stack.
    #[test]
    fn a_whole_drag_is_a_single_row() {
        let mut doc = sample_document();
        let mut stack = CommandStack::new();
        let piece = doc.pieces().iter().next().unwrap().0;
        let vertex = doc.piece(piece).unwrap().network.vertices.next_id();
        stack
            .apply(
                &mut doc,
                nguo_doc::Edit::AddVertex {
                    piece,
                    id: vertex,
                    pos: vello::kurbo::Point::ORIGIN,
                },
            )
            .unwrap();
        let before = history_rows(&stack, &doc).len();

        let gesture = nguo_doc_commands::GestureId(1);
        for i in 1..=60 {
            stack
                .apply(
                    &mut doc,
                    nguo_doc_commands::Transaction::new("Move point")
                        .with_gesture(gesture)
                        .edit(nguo_doc::Edit::MoveVertex {
                            piece,
                            id: vertex,
                            pos: vello::kurbo::Point::new(i as f64, 0.0),
                        }),
                )
                .unwrap();
        }

        let rows = history_rows(&stack, &doc);
        assert_eq!(
            rows.len(),
            before + 1,
            "60 moves, one row: {:?}",
            labels(&rows)
        );
        assert_eq!(rows.last().unwrap().label, "Move point");

        // And clicking the row above it cancels the whole drag.
        stack.jump_to(&mut doc, rows.len() - 2).unwrap();
        assert_eq!(
            doc.piece(piece).unwrap().network.pos(vertex).unwrap(),
            vello::kurbo::Point::ORIGIN
        );
    }

    /// Rows read like actions, not like command ids. A panel of `edit.undo` /
    /// `move_vertex` is not the feature.
    #[test]
    fn every_row_reads_like_an_action() {
        let (doc, stack, _) = history_fixture(3);
        for row in history_rows(&stack, &doc) {
            assert!(
                !row.label.contains('.') && !row.label.contains('_'),
                "{:?} reads like an id",
                row.label
            );
            assert!(!row.label.is_empty());
        }
    }

    /// The History tab has to actually be in the layout under the id `main.rs`
    /// matches on, or `panel_body` falls through to its `_` arm and the panel is
    /// silently blank — a failure no other test in this file would notice.
    #[test]
    fn the_history_tab_is_in_the_right_dock_under_the_id_main_matches() {
        let layout = default_layout(&Tokens::dark());
        let panel = layout.panel(PROPERTIES).expect("the right dock");
        let ids: Vec<&str> = panel.tabs.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&HISTORY), "{ids:?}");
        assert_eq!(HISTORY, "history");
    }

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

    /// Narrow, then widen: the layout must come back.
    ///
    /// `enforce_canvas_minimum` only ever shrinks, so applying it to its own
    /// output ratchets — every narrowing squeezes the chrome further and no
    /// widening returns the space. The app avoids that by restoring the user's
    /// ratios before enforcing (`App::restore_then_enforce_canvas_minimum`),
    /// which makes the result a function of the width alone. This pins that
    /// property at the level it is actually decided: enforce against a
    /// *restored* baseline is idempotent and reversible; enforce against the
    /// previous output is neither.
    #[test]
    fn the_floor_depends_on_the_width_not_on_the_resize_history() {
        let tokens = Tokens::dark();
        let baseline = {
            let mut l = default_layout(&tokens);
            l.split_at_mut(&[]).unwrap().ratios().to_vec()
        };

        // Direct: one enforce at 900 px from the shipped ratios.
        let mut direct = default_layout(&tokens);
        enforce_canvas_minimum(&mut direct, 900.0);
        let direct = direct.split_at_mut(&[]).unwrap().ratios().to_vec();

        // Via a detour: squeeze hard, widen, come back to 900 — restoring the
        // baseline before each enforce, which is what the app does.
        let mut viaticum = default_layout(&tokens);
        for width in [640.0_f32, 2400.0, 900.0] {
            viaticum
                .split_at_mut(&[])
                .unwrap()
                .set_ratios(&baseline.clone());
            enforce_canvas_minimum(&mut viaticum, width);
        }
        let viaticum = viaticum.split_at_mut(&[]).unwrap().ratios().to_vec();

        for (a, b) in direct.iter().zip(&viaticum) {
            assert!(
                (a - b).abs() < 1e-6,
                "same width must give the same layout regardless of how it got \
                 there: direct {direct:?} vs after 640 -> 2400 -> 900 {viaticum:?}"
            );
        }

        // And without the restore it genuinely ratchets — this is the bug the
        // restore exists to prevent, asserted so nobody "simplifies" it away.
        let mut ratcheted = default_layout(&tokens);
        for width in [640.0_f32, 2400.0, 900.0] {
            enforce_canvas_minimum(&mut ratcheted, width);
        }
        let ratcheted = ratcheted.split_at_mut(&[]).unwrap().ratios().to_vec();
        assert!(
            ratcheted
                .iter()
                .zip(&direct)
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "expected enforcing on its own output to drift, but it matched the \
             direct result — if that is now genuinely idempotent, the restore \
             in App may be removable: {ratcheted:?}"
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
