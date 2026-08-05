// SPDX-License-Identifier: GPL-2.0-or-later
//! Pointer events in, geometry out — headlessly.
//!
//! This is the gap the canvas had: every tool entry point returned a
//! `Transaction` that nothing could apply, `Canvas2d::hit`/`resolve`/`applied`
//! had no production caller, and there was no `CommandStack` anywhere in the
//! app. A user could not draw a line with the mouse. These tests drive exactly
//! the loop the shell now runs — `Canvas2d::pointer` → `CommandStack::apply` →
//! `Canvas2d::applied` — through synthetic press/move/release sequences.
//!
//! Everything here is synthetic: no assets, no `samples/`, no window.

use nguo_canvas::{Canvas2d, Phase, Pointer, SnapKind, Target, Tool};
use nguo_doc::Document;
use nguo_doc_commands::CommandStack;
use nguo_pattern::EdgeSemantic;
use nguo_ui_theme::Tokens;
use vello::kurbo::{ParamCurve, Point, Vec2};

const SIZE: Vec2 = Vec2::new(900.0, 700.0);

/// The whole application-side loop, in one place, so a test cannot accidentally
/// test a different one from the shell.
struct Harness {
    canvas: Canvas2d,
    doc: Document,
    stack: CommandStack,
    tool: Tool,
    modifiers: iced::keyboard::Modifiers,
}

impl Harness {
    fn new(doc: Document, tool: Tool) -> Harness {
        Harness {
            canvas: Canvas2d::new(),
            doc,
            stack: CommandStack::new(),
            tool,
            modifiers: iced::keyboard::Modifiers::default(),
        }
    }

    /// Screen px for a world point, so the tests can say where they mean.
    fn screen(&self, world: Point) -> Point {
        self.canvas.view.world_to_screen(world, SIZE)
    }

    fn send(&mut self, phase: Phase, screen: Point) {
        let p = Pointer {
            phase,
            screen,
            size: SIZE,
            modifiers: self.modifiers,
        };
        if let Some(tx) = self.canvas.pointer(&self.doc, self.tool, p) {
            // Marked before applied: `applied` only reads the edit list, and
            // over-counting when an apply fails costs a full rebuild rather
            // than a stale piece (see `SceneCache`'s docs on that hazard).
            self.canvas.applied(&tx);
            self.stack.apply(&mut self.doc, tx).expect("a valid edit");
        }
    }

    /// One click at a world position: press, then release without moving.
    fn click(&mut self, world: Point) {
        let screen = self.screen(world);
        self.send(Phase::Pressed, screen);
        self.send(Phase::Released, screen);
    }

    fn frame(&mut self, tokens: &Tokens) {
        self.canvas.frame(&self.doc, SIZE, tokens);
    }

    fn only_piece(&self) -> &nguo_doc::Piece {
        let id = self.doc.pieces().ids().next().expect("a piece exists");
        self.doc.piece(id).expect("the piece is readable")
    }
}

#[test]
fn a_press_move_release_draws_a_segment_and_undo_removes_it() {
    let mut h = Harness::new(Document::new(), Tool::Draw(EdgeSemantic::Seam));
    let revision = h.doc.revision();

    // Two clicks 100 mm apart, off the grid pitch so nothing snaps them
    // somewhere else, with a hover between them the way a real pointer moves.
    h.click(Point::new(0.0, 0.0));
    h.send(Phase::Hovered, h.screen(Point::new(60.0, 0.0)));
    h.click(Point::new(100.0, 0.0));

    let net = &h.only_piece().network;
    assert_eq!(net.vertices.len(), 2, "two clicks, two points");
    assert_eq!(net.edges.len(), 1, "and the segment between them");
    let edge = net.edges.ids().next().unwrap();
    let seg = net.seg(edge).unwrap();
    assert_eq!(seg.start(), Point::new(0.0, 0.0));
    assert_eq!(seg.end(), Point::new(100.0, 0.0));
    assert_eq!(
        net.edges.get(edge).unwrap().semantic,
        EdgeSemantic::Seam,
        "the tool's semantic reaches the edge"
    );
    assert!(
        h.doc.revision() > revision,
        "the document must actually have moved on"
    );

    // Undo removes the segment. Two undos, because each click is its own step.
    let before_undo = h.doc.revision();
    assert!(h.stack.undo(&mut h.doc).expect("undo succeeds"));
    assert_eq!(h.only_piece().network.edges.len(), 0, "the segment is gone");
    assert_ne!(h.doc.revision(), before_undo, "undo moves the revision too");
}

#[test]
fn the_rubber_band_follows_the_pointer_between_clicks() {
    let mut h = Harness::new(Document::new(), Tool::Draw(EdgeSemantic::Seam));
    assert_eq!(
        h.canvas.interaction.rubber_band, None,
        "nothing to band yet"
    );

    h.click(Point::new(0.0, 0.0));
    h.send(Phase::Hovered, h.screen(Point::new(37.0, 42.0)));
    let (from, to) = h
        .canvas
        .interaction
        .rubber_band
        .expect("a run in progress bands to the cursor");
    assert_eq!(from, Point::new(0.0, 0.0), "banded from the last point");
    assert!(
        (to - Point::new(37.0, 42.0)).hypot() < 5.0,
        "banded to the cursor (snapped): {to:?}"
    );

    // The band is drawable, and the readout the status bar shows is its length.
    assert!((to - from).hypot() > 40.0);
    h.canvas.cancel();
    assert_eq!(h.canvas.interaction.rubber_band, None, "cancel clears it");
}

#[test]
fn clicking_back_on_the_first_point_closes_the_outline() {
    let mut h = Harness::new(Document::new(), Tool::Draw(EdgeSemantic::Seam));
    for p in [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)] {
        h.click(Point::new(p.0, p.1));
    }
    assert_eq!(h.only_piece().network.edges.len(), 3, "open so far");

    h.click(Point::new(0.0, 0.0)); // back onto the first point
    let net = &h.only_piece().network;
    assert_eq!(
        net.edges.len(),
        4,
        "the run closed instead of adding a point"
    );
    assert_eq!(net.vertices.len(), 4, "and did not duplicate the corner");
    for v in net.vertices.ids() {
        assert_eq!(net.degree(v), 2, "a closed outline: every vertex degree 2");
    }
}

/// The close gesture must use the radius the click is actually resolved at.
///
/// Snapping reaches 8 px and picking reaches 7, so a click in that 1 px band
/// landed exactly on the first vertex and then failed a close test that
/// hit-tested the RAW pointer — `pen::click` has no coincidence check, so it
/// placed a duplicate vertex and a zero-length edge, and `boundary_loop` never
/// saw a closed loop.
#[test]
fn a_click_in_the_snap_only_band_still_closes_the_outline() {
    let mut h = Harness::new(Document::new(), Tool::Draw(EdgeSemantic::Seam));
    for p in [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)] {
        h.click(Point::new(p.0, p.1));
    }
    assert_eq!(h.only_piece().network.edges.len(), 3, "open so far");

    // 7.5 logical px from the first vertex: inside the 8 px snap radius, outside
    // the 7 px pick radius. Asserted, not assumed — the whole bug lives in the
    // gap between those two numbers, so a test that silently drifted out of it
    // would pass while proving nothing.
    let first = h.screen(Point::new(0.0, 0.0));
    let off = Point::new(first.x + 7.5, first.y);
    h.send(Phase::Hovered, off);
    let snap = h.canvas.interaction.snap.expect("the snap must catch");
    assert_eq!(
        (snap.kind, snap.source),
        (
            SnapKind::Endpoint,
            Some(Target::Point {
                piece: h.doc.pieces().ids().next().unwrap(),
                vertex: h.only_piece().network.vertices.ids().next().unwrap(),
            })
        ),
        "the probe must land ON the run's FIRST vertex — an alignment snap also \
         sources a vertex and can land a metre from it: {snap:?}"
    );
    let picked = h.canvas.hit(&h.doc, off, SIZE).map(|hit| hit.target);
    assert_ne!(
        picked,
        Some(Target::Point {
            piece: h.doc.pieces().ids().next().unwrap(),
            vertex: h.only_piece().network.vertices.ids().next().unwrap(),
        }),
        "and the first vertex must be outside the 7 px PICK radius, or this is the \
         same test as the one above — it picked {picked:?}"
    );

    h.send(Phase::Pressed, off);
    h.send(Phase::Released, off);
    let net = &h.only_piece().network;
    assert_eq!(net.vertices.len(), 4, "no duplicate coincident vertex");
    assert_eq!(net.edges.len(), 4, "the outline closed");
    for v in net.vertices.ids() {
        assert_eq!(net.degree(v), 2, "a closed outline: every vertex degree 2");
    }
}

#[test]
fn dragging_a_point_with_the_select_tool_moves_it_in_one_undo_step() {
    let (doc, piece, _) = nguo_canvas::fixture::square_doc();
    let mut h = Harness::new(doc, Tool::Select);
    let corner = Point::new(100.0, 100.0);

    h.send(Phase::Pressed, h.screen(corner));
    assert_eq!(
        h.canvas.interaction.selection.len(),
        1,
        "a press selects what is under it"
    );
    assert_eq!(h.canvas.interaction.editing, vec![piece]);

    let depth = h.stack.undo_depth();
    for i in 1..=8 {
        let to = Point::new(corner.x + i as f64 * 3.0, corner.y + i as f64 * 3.0);
        h.send(Phase::Dragged, h.screen(to));
    }
    h.send(Phase::Released, h.screen(corner));

    let moved = h
        .only_piece()
        .network
        .vertices
        .iter()
        .map(|(_, v)| v.pos)
        .find(|p| p.x > 110.0)
        .expect("the corner moved");
    assert!(moved.x > 115.0 && moved.y > 115.0, "{moved:?}");
    assert_eq!(
        h.stack.undo_depth(),
        depth + 1,
        "a whole drag is one undo entry, not eight"
    );

    // And nothing moves after the release — the control that proves the drag
    // state is actually cleared rather than the test just stopping.
    let after = h.only_piece().network.vertices.len();
    h.send(Phase::Dragged, h.screen(Point::new(400.0, 400.0)));
    assert_eq!(h.stack.undo_depth(), depth + 1, "released means released");
    assert_eq!(h.only_piece().network.vertices.len(), after);
}

#[test]
fn a_second_piece_lines_up_with_the_first_and_the_modifier_suspends_it() {
    // Alignment guides: the corner of an existing piece is 100 mm away in y,
    // far outside any endpoint tolerance, but shares its x.
    let (doc, _, _) = nguo_canvas::fixture::square_doc();
    let mut h = Harness::new(doc, Tool::Draw(EdgeSemantic::Seam));

    // 2 mm off the x of the square's right edge, 200 mm below it.
    let near = Point::new(102.0, 300.0);
    h.send(Phase::Hovered, h.screen(near));
    let snap = h
        .canvas
        .interaction
        .snap
        .expect("2 mm off a vertex's x must catch the alignment guide");
    assert_eq!(snap.kind, SnapKind::AlignX, "{snap:?}");
    assert_eq!(snap.world.x, 100.0, "pulled onto the other piece's x");
    assert_eq!(snap.world.y, 300.0, "and left alone in y");
    assert!(matches!(snap.source, Some(Target::Point { .. })));

    // Hold ⌘ and it is suspended — the point lands exactly where the pointer is.
    h.modifiers.toggle(iced::keyboard::Modifiers::COMMAND);
    h.click(near);
    assert_eq!(h.canvas.interaction.snap, None, "the modifier suspends it");
    let drawn = h
        .doc
        .pieces()
        .iter()
        .find(|(_, p)| p.network.vertices.len() == 1)
        .expect("a new piece was started");
    let pos = drawn.1.network.vertices.iter().next().unwrap().1.pos;
    assert!(
        (pos - near).hypot() < 1e-9,
        "⌘ must place the point unsnapped: {pos:?}"
    );
}

/// The control the alignment test needs: with `align` off, the same probe finds
/// nothing. Without this, `a_second_piece_lines_up_with_the_first` could be
/// passing on some other snap kind entirely.
#[test]
fn with_alignment_off_the_same_probe_does_not_snap() {
    let (doc, _, _) = nguo_canvas::fixture::square_doc();
    let mut h = Harness::new(doc, Tool::Draw(EdgeSemantic::Seam));
    h.canvas.snapper.toggles = nguo_canvas::SnapToggles::NONE;
    h.send(Phase::Hovered, h.screen(Point::new(102.0, 300.0)));
    assert_eq!(h.canvas.interaction.snap, None);
}

#[test]
fn drawing_marks_only_the_piece_it_touched() {
    // `applied()` is the performance half of the wiring, and it had no call
    // site. Editing one piece of ten must rebuild one scene, not ten.
    let doc = nguo_canvas::fixture::marker(10, 8);
    let mut h = Harness::new(doc, Tool::Select);
    let tokens = Tokens::light();
    h.frame(&tokens);
    assert_eq!(
        h.canvas.piece_builds(),
        10,
        "the first frame builds all ten"
    );

    // Drag a point of one piece.
    let target = h.doc.pieces().ids().nth(3).unwrap();
    let piece = h.doc.piece(target).unwrap();
    let v = piece.network.vertices.ids().next().unwrap();
    let world = piece.transform * piece.network.pos(v).unwrap();
    h.send(Phase::Pressed, h.screen(world));
    h.send(Phase::Dragged, h.screen(world + Vec2::new(20.0, 20.0)));

    h.frame(&tokens);
    assert_eq!(
        h.canvas.piece_builds(),
        11,
        "exactly one piece rebuilt — `applied` is wired"
    );
}
