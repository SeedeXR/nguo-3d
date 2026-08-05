// SPDX-License-Identifier: GPL-2.0-or-later
//! The 2D pattern canvas: a Vello scene rendered through iced's device, with
//! pan/zoom, a grid derived from the view transform, a Bézier pen, snapping,
//! hit-testing through a spatial index, and a per-piece scene cache.
//!
//! # The rules this crate keeps
//! * **One view matrix** ([`View`]). Zoom is never baked into stored geometry;
//!   the document is millimetres, always.
//! * **No colour literals.** Every colour is a [`nguo_ui_theme`] token, asserted
//!   by `tests/no_color_literals.rs`.
//! * **No mutation.** Tools return `nguo-doc-commands` transactions. That is
//!   structural, not a convention: `Document`'s fields are private to
//!   `nguo-doc`, so this crate has no way in.
//!
//! # Layout of the pieces
//! * [`view`] — the three coordinate spaces and the one matrix.
//! * [`grid`] — grid step and tick geometry, derived from that matrix.
//! * [`hit`] — the spatial index, pixel tolerance, layer priority.
//! * [`snap`] — snap resolution: screen tolerance, world result.
//! * [`pen`] — the pen state machine and point/handle editing.
//! * [`scene`] — scene building and the id-keyed per-piece cache.
//! * [`Canvas2d`] — the one object an application holds to get all of it.

pub mod fixture;
pub mod grid;
pub mod hit;
pub mod pen;
pub mod scene;
pub mod snap;
pub mod view;

use std::sync::Arc;

use iced::keyboard::Modifiers;
use iced::widget::shader;
use iced::{mouse, Rectangle};
use nguo_doc::{Document, PieceId};
use nguo_doc_commands::{GestureId, Transaction};
use nguo_gpu::canvas::ScenePrimitive;
use nguo_gpu::scale;
use nguo_pattern::EdgeSemantic;
use nguo_ui_theme::Tokens;
use vello::kurbo::{Affine, BezPath, Circle, Point, Stroke, Vec2};
use vello::peniko::{Color, Fill};
use vello::Scene;

/// Re-exported for callers of [`Canvas2d::pointer`]: it hands back a
/// `Transaction`, and the only thing that can apply one is a `CommandStack`.
/// Making the caller add a dependency to name the type it was just given is
/// friction for nothing.
pub use nguo_doc_commands::CommandStack;

pub use hit::{HandleEnd, Hit, Index, Layer, Target};
pub use pen::{Pen, PenState};
pub use scene::{Interaction, SceneCache};
pub use snap::{Snap, SnapKind, SnapToggles, Snapper};
pub use view::View;

/// Where a pointer gesture is. Four phases and not two, because the pen needs
/// to tell "moved with the button down" (curve the segment just placed) from
/// "moved with it up" (rubber-band to the cursor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Pressed,
    /// Moved with the left button down.
    Dragged,
    /// Moved with no button down.
    Hovered,
    Released,
}

/// A pointer event, in the widget's own coordinates.
///
/// No `world` field: resolving screen to world is snapping's job, and snapping
/// needs the document, which the widget does not have. [`Canvas2d::pointer`]
/// does it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pointer {
    pub phase: Phase,
    /// Widget-relative logical px.
    pub screen: Point,
    /// The widget's logical size — the widget knows it and the app does not.
    pub size: Vec2,
    pub modifiers: Modifiers,
}

/// What the canvas asks the application to do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Pan {
        delta: Vec2,
    },
    /// `size` is the widget's logical size — the widget knows it and the
    /// application does not, and anchoring the zoom to the cursor needs it.
    Zoom {
        factor: f64,
        screen: Point,
        size: Vec2,
    },
    /// The pointer did something. Feed it to [`Canvas2d::pointer`].
    Pointer(Pointer),
}

#[derive(Debug, Default)]
pub struct State {
    /// Middle-button pan. **Not** left-drag: the left drag gesture belongs to
    /// the tools, and a canvas where the first segment you try to draw pans the
    /// view instead is a canvas you cannot draft in. Middle-drag is what
    /// Figma, Illustrator and Blender all use for the same reason.
    panning: bool,
    left_down: bool,
    last_cursor: Option<iced::Point>,
    modifiers: Modifiers,
}

/// The widget program. Rebuilt every `view()`.
#[derive(Debug, Clone, Copy)]
pub struct Canvas {
    pub view: View,
    /// Last picked world point, drawn as a crosshair. This is the visible
    /// proof that hit-testing is right on a Retina display: the crosshair
    /// lands exactly under the pointer at any scale factor or zoom.
    pub pick: Option<Point>,
}

/// The themed base colour behind the canvas: the paper the pattern is drawn on
/// (UX_SPEC §2 — a light canvas inside a dark shell).
pub fn background(tokens: &Tokens) -> Color {
    scene::ink(tokens.color.canvas)
}

/// Base colour for a caller with no theme in hand.
///
/// It is `BLACK` and not a hue, because this crate holds no colour literals
/// (`tests/no_color_literals.rs`) and a `const` cannot read a token — the
/// tokens are a `LazyLock`. Every real caller passes its theme to
/// [`background`]; the shell already does. This exists so a `const` context and
/// the headless render test have something opaque to clear with.
pub const BACKGROUND: Color = Color::BLACK;

impl shader::Program<Action> for Canvas {
    type State = State;
    type Primitive = ScenePrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Action>> {
        let size = Vec2::new(bounds.width as f64, bounds.height as f64);
        let pointer = |phase: Phase, position: iced::Point| {
            Action::Pointer(Pointer {
                phase,
                screen: Point::new(position.x as f64, position.y as f64),
                size,
                modifiers: state.modifiers,
            })
        };

        match event {
            iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(modifiers)) => {
                // Tracked here and carried on every `Pointer`, because "hold a
                // key to suspend snapping" is only knowable at the moment of
                // the pointer event.
                state.modifiers = *modifiers;
                None
            }
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let position = cursor.position_in(bounds)?;
                state.left_down = true;
                Some(shader::Action::publish(pointer(Phase::Pressed, position)).and_capture())
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                if !state.left_down {
                    return None;
                }
                state.left_down = false;
                // Widget-relative even when the release lands outside: a drag
                // that ends off-widget must still end, and the dock has the
                // same bug in the other direction (`mouse_area` early-returns).
                let position = cursor.position_in(bounds).unwrap_or_else(|| {
                    cursor
                        .position()
                        .map(|p| iced::Point::new(p.x - bounds.x, p.y - bounds.y))
                        .unwrap_or_default()
                });
                Some(shader::Action::publish(pointer(Phase::Released, position)).and_capture())
            }
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Middle)) => {
                cursor.position_in(bounds)?;
                state.panning = true;
                state.last_cursor = cursor.position();
                None
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Middle)) => {
                state.panning = false;
                state.last_cursor = None;
                None
            }
            iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if state.panning {
                    let position = cursor.position()?;
                    let last = state.last_cursor.replace(position)?;
                    return Some(
                        shader::Action::publish(Action::Pan {
                            delta: Vec2::new(
                                (position.x - last.x) as f64,
                                (position.y - last.y) as f64,
                            ),
                        })
                        .and_capture(),
                    );
                }

                let position = cursor.position_in(bounds)?;
                if state.left_down {
                    Some(shader::Action::publish(pointer(Phase::Dragged, position)).and_capture())
                } else {
                    // Published but NOT captured: hover feeds the snap
                    // indicator and the readout, and capturing every motion
                    // event would starve the dock's divider drag, which tracks
                    // the cursor at window level.
                    Some(shader::Action::publish(pointer(Phase::Hovered, position)))
                }
            }
            iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                let position = cursor.position_in(bounds)?;
                let lines = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y,
                    mouse::ScrollDelta::Pixels { y, .. } => y / 40.0,
                };
                Some(
                    shader::Action::publish(Action::Zoom {
                        factor: (1.0 + lines as f64 * 0.1).clamp(0.5, 2.0),
                        screen: Point::new(position.x as f64, position.y as f64),
                        size,
                    })
                    .and_capture(),
                )
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        // `draw` has no `Viewport`, so no scale factor. Build the scene in
        // LOGICAL units and let the primitive apply the HiDPI scale — the same
        // split the rest of the shell uses (guide §2.4).
        ScenePrimitive {
            scene: Arc::new(build_scene(
                &self.view,
                Vec2::new(bounds.width as f64, bounds.height as f64),
                self.pick,
            )),
            base_color: BACKGROUND,
        }
    }

    fn mouse_interaction(
        &self,
        state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.panning {
            mouse::Interaction::Grabbing
        } else if cursor.is_over(bounds) {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::default()
        }
    }
}

/// Everything an application needs to hold to drive the 2D authoring loop.
///
/// The document is not in here: it belongs to the app and is passed in. That is
/// deliberate — one document, many views (UX_SPEC §2 "coupled model, uncoupled
/// view").
pub struct Canvas2d {
    pub view: View,
    pub snapper: Snapper,
    pub pen: Pen,
    pub interaction: Interaction,
    /// Hit tolerance in logical px. Generous, because a 1 px target is a target
    /// nobody can hit (guide §6.2).
    pub pick_px: f64,
    /// Last resolved pointer position in world mm — after snapping, so it is
    /// the number a readout should show. `None` before the pointer has been
    /// over the widget.
    pub cursor: Option<Point>,
    /// What a left-drag is currently moving, and the gesture id that coalesces
    /// the whole drag into one undo entry.
    drag: Option<(Target, GestureId)>,
    /// Monotonic source of [`GestureId`]s. One per press.
    gestures: u64,
    index: Option<Index>,
    cache: SceneCache,
}

/// What a press means. The application maps its active toolbar button onto
/// this; the canvas knows nothing about `CommandId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tool {
    /// Pick, direct-select, and drag points and Bézier handles.
    #[default]
    Select,
    /// Draw a run of edges carrying this semantic — `Seam` for the polygon
    /// tool, `Internal` for internal lines, `DartLeg` for darts.
    Draw(EdgeSemantic),
}

impl Default for Canvas2d {
    fn default() -> Self {
        Canvas2d {
            view: View::default(),
            snapper: Snapper::default(),
            pen: Pen::default(),
            interaction: Interaction::default(),
            pick_px: 7.0,
            cursor: None,
            drag: None,
            gestures: 0,
            index: None,
            cache: SceneCache::new(),
        }
    }
}

impl Canvas2d {
    pub fn new() -> Canvas2d {
        Canvas2d::default()
    }

    /// The spatial index for `doc`, rebuilt only when the revision moved
    /// (guide §4.3: derived state, keyed on revision).
    pub fn index(&mut self, doc: &Document) -> &Index {
        let stale = self
            .index
            .as_ref()
            .is_none_or(|i| i.revision() != doc.revision());
        if stale {
            self.index = Some(Index::of_document(doc));
        }
        self.index.as_ref().expect("just built")
    }

    /// Topmost target under a pointer position in logical widget px.
    pub fn hit(&mut self, doc: &Document, screen: Point, size: Vec2) -> Option<Hit> {
        let (view, tolerance) = (self.view, self.pick_px);
        self.index(doc).hit_screen(&view, screen, size, tolerance)
    }

    /// Where a pointer position should actually land, after snapping. Falls back
    /// to the raw world point when nothing is in range, so a caller can always
    /// use the result.
    pub fn resolve(&mut self, doc: &Document, screen: Point, size: Vec2) -> Point {
        let raw = self.view.screen_to_world(screen, size);
        // The snap pitch IS the drawn minor grid. They used to be decoupled — a
        // fixed 10 mm snap under a 100 mm drawn grid at the shipped opening
        // zoom — which means you snap to lines you cannot see.
        self.snapper.grid_mm = grid::steps(self.view.zoom).0;
        let (view, snapper) = (self.view, self.snapper);
        let snap = snapper.snap(raw, view.zoom, self.index(doc));
        self.interaction.snap = snap;
        snap.map_or(raw, |s| s.world)
    }

    /// Drive one pointer event through the active tool.
    ///
    /// Returns the transaction the app must `CommandStack::apply` — and hand
    /// straight back to [`Canvas2d::applied`], or the piece cache goes stale.
    /// The canvas never mutates the document: `Document`'s fields are private
    /// to `nguo-doc`, so that is structural.
    pub fn pointer(&mut self, doc: &Document, tool: Tool, p: Pointer) -> Option<Transaction> {
        // Hold ⌘/Ctrl to place a point exactly where the pointer is, the way
        // Figma does. Everything else snaps.
        let world = if p.modifiers.command() || p.modifiers.control() {
            self.interaction.snap = None;
            self.view.screen_to_world(p.screen, p.size)
        } else {
            self.resolve(doc, p.screen, p.size)
        };
        self.cursor = Some(world);

        let tx = match tool {
            Tool::Select => self.select(doc, p, world),
            Tool::Draw(semantic) => {
                self.pen.semantic = semantic;
                self.draw(doc, p, world)
            }
        };

        // The rubber band and the angle-snap origin both hang off the pen's
        // current anchor, and both are for the *next* event, so they are
        // recomputed after the pen has moved rather than before.
        let anchor = self.pen.anchor(doc).map(|a| self.to_world(doc, a));
        self.snapper.from = anchor;
        self.interaction.rubber_band = anchor.map(|a| (a, world));
        tx
    }

    fn select(&mut self, doc: &Document, p: Pointer, world: Point) -> Option<Transaction> {
        match p.phase {
            Phase::Pressed => {
                let target = self.hit(doc, p.screen, p.size).map(|h| h.target);
                self.interaction.selection = target.into_iter().collect();
                // Direct-select: showing a piece's points is what makes them
                // draggable targets on the next press.
                self.interaction.editing = target.map(Target::piece).into_iter().collect();
                self.drag = match target {
                    Some(t @ (Target::Point { .. } | Target::Handle { .. })) => {
                        self.gestures += 1;
                        Some((t, GestureId(self.gestures)))
                    }
                    _ => None,
                };
                None
            }
            Phase::Dragged => {
                let (target, gesture) = self.drag?;
                let local = self.to_local(doc, target.piece(), world);
                let tx = match target {
                    Target::Point { piece, vertex } => {
                        pen::move_point(doc, piece, vertex, local).ok()?
                    }
                    Target::Handle { piece, edge, end } => {
                        pen::move_handle(doc, piece, edge, end, local).ok()?
                    }
                    Target::Edge { .. } | Target::Fill { .. } => return None,
                };
                Some(tx.with_gesture(gesture))
            }
            Phase::Released => {
                self.drag = None;
                None
            }
            Phase::Hovered => {
                self.interaction.hover = self.hit(doc, p.screen, p.size).map(|h| h.target);
                None
            }
        }
    }

    fn draw(&mut self, doc: &Document, p: Pointer, world: Point) -> Option<Transaction> {
        match p.phase {
            Phase::Pressed => {
                self.gestures += 1;
                // Clicking the point the run started from closes it — the
                // gesture every vector tool uses, and the only way to get a
                // closed outline (which is what `boundary_loop` needs to fill).
                if let PenState::Drawing { piece, first, .. } = self.pen.state {
                    let closing = Target::Point {
                        piece,
                        vertex: first,
                    };
                    // Either radius closes the run. Hit-testing alone left a
                    // dead band: the snap radius (8 px) is wider than the pick
                    // radius (7 px), so a click 7-8 px out snapped exactly onto
                    // `first` and then failed this test, placing a duplicate
                    // coincident vertex and a zero-length edge instead of
                    // closing — and `boundary_loop` never saw a closed loop.
                    // `world` above already resolved the snap, so this reads
                    // what the click will actually land on.
                    let snapped_to_first = self.interaction.snap.is_some_and(|s| {
                        // `Endpoint` specifically: an ALIGNMENT snap also sources
                        // a vertex, but it only shares one axis with it and can
                        // land a metre away. Only an endpoint snap puts the point
                        // ON the vertex, which is what closing means.
                        s.kind == SnapKind::Endpoint && s.source == Some(closing)
                    });
                    if snapped_to_first
                        || self.hit(doc, p.screen, p.size).map(|h| h.target) == Some(closing)
                    {
                        return self.pen.close(doc);
                    }
                }
                // A run starts a new piece, except for the semantics that only
                // mean anything *inside* one.
                let into = match (self.pen.state, self.pen.semantic) {
                    (PenState::Drawing { piece, .. }, _) => Some(piece),
                    (PenState::Idle, EdgeSemantic::Seam | EdgeSemantic::Cut) => None,
                    (PenState::Idle, _) => {
                        self.hit(doc, p.screen, p.size).map(|h| h.target.piece())
                    }
                };
                let local = into.map_or(world, |piece| self.to_local(doc, piece, world));
                self.pen.click(doc, into, local).ok()
            }
            // Dragging straight after a click curves the segment just placed.
            // Tagged, so a whole drag is one undo step and not sixty.
            Phase::Dragged => {
                let piece = match self.pen.state {
                    PenState::Drawing { piece, .. } => piece,
                    PenState::Idle => return None,
                };
                let local = self.to_local(doc, piece, world);
                self.pen
                    .drag_handle(doc, local)
                    .map(|tx| tx.with_gesture(GestureId(self.gestures)))
            }
            Phase::Released | Phase::Hovered => None,
        }
    }

    /// Ends a pen run, leaving it open. What `Esc` is bound to.
    pub fn cancel(&mut self) {
        self.pen.finish();
        self.drag = None;
        self.interaction.rubber_band = None;
        self.snapper.from = None;
    }

    /// World mm → the piece's own coordinates, which is what an `Edit` stores.
    fn to_local(&self, doc: &Document, piece: PieceId, world: Point) -> Point {
        doc.piece(piece)
            .map_or(world, |p| p.transform.inverse() * world)
    }

    /// The pen's anchor is in the piece's coordinates; the overlay draws in
    /// world. Which piece is unambiguous — the pen is only ever in one.
    fn to_world(&self, doc: &Document, local: Point) -> Point {
        match self.pen.state {
            PenState::Drawing { piece, .. } => {
                doc.piece(piece).map_or(local, |p| p.transform * local)
            }
            PenState::Idle => local,
        }
    }

    /// Call this with every transaction the app applies, so the cache knows
    /// which pieces changed. Cheap: it reads the edits it is handed.
    pub fn applied(&mut self, tx: &Transaction) {
        self.cache.mark_from_edits(tx.edits());
    }

    /// The frame, in logical units.
    pub fn frame(&mut self, doc: &Document, size: Vec2, tokens: &Tokens) -> Scene {
        self.cache
            .frame(doc, &self.interaction, &self.view, size, tokens)
    }

    /// Per-piece scene builds since startup — the number the caching claim rests
    /// on. Exposed so a profiling overlay (and the tests) can read it.
    pub fn piece_builds(&self) -> u64 {
        self.cache.piece_builds()
    }
}

/// Builds the demo scene in **logical** units; the caller scales to physical.
///
/// A filled Bézier shape and a stroked path — the two things §25.1 asks the
/// canvas to prove — plus the pick crosshair. Kept because it is the shell's
/// current canvas content and the headless render-graph test's subject;
/// [`Canvas2d::frame`] is what draws an actual document.
pub fn build_scene(view: &View, size: Vec2, pick: Option<Point>) -> Scene {
    let tokens = Tokens::dark();
    let mut scene = Scene::new();
    let to_screen = view.scene_transform(size, 1.0);

    scene.fill(
        Fill::NonZero,
        to_screen,
        scene::ink(tokens.color.accent),
        None,
        &petal(),
    );

    scene.stroke(
        &Stroke::new(3.0 / view.zoom),
        to_screen,
        scene::ink(tokens.color.notch),
        None,
        &seam_line(),
    );

    if let Some(pick) = pick {
        // Drawn in screen space at a fixed size so it stays legible at any
        // zoom, but positioned through the same transform the shapes use.
        let at = to_screen * pick;
        let mut cross = BezPath::new();
        cross.move_to((at.x - 12.0, at.y));
        cross.line_to((at.x + 12.0, at.y));
        cross.move_to((at.x, at.y - 12.0));
        cross.line_to((at.x, at.y + 12.0));

        let mark = scene::ink(tokens.color.dart);
        scene.stroke(&Stroke::new(1.5), Affine::IDENTITY, mark, None, &cross);
        scene.stroke(
            &Stroke::new(1.5),
            Affine::IDENTITY,
            mark,
            None,
            &Circle::new(at, 5.0),
        );
    }

    scene
}

/// A closed cubic-Bézier blob — stands in for a pattern piece.
fn petal() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((-90.0, 0.0));
    path.curve_to((-90.0, -70.0), (-30.0, -110.0), (0.0, -110.0));
    path.curve_to((30.0, -110.0), (90.0, -70.0), (90.0, 0.0));
    path.curve_to((90.0, 70.0), (30.0, 110.0), (0.0, 110.0));
    path.curve_to((-30.0, 110.0), (-90.0, 70.0), (-90.0, 0.0));
    path.close_path();
    path
}

/// An open path — stands in for a seam line.
fn seam_line() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((-140.0, 130.0));
    path.curve_to((-60.0, 60.0), (60.0, 200.0), (140.0, 130.0));
    path
}

/// Physical size of the canvas target, for callers that need it (tests, the
/// headless render). Kept here so the rule lives next to the transform.
pub fn target_size(bounds: Rectangle, scale_factor: f32) -> (u32, u32) {
    scale::physical_size(bounds.width, bounds.height, scale_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scene_has_content() {
        let scene = build_scene(
            &View::default(),
            Vec2::new(800.0, 600.0),
            Some(Point::new(10.0, 10.0)),
        );
        assert!(
            !scene.encoding().is_empty(),
            "the canvas scene must encode something to draw"
        );
    }

    #[test]
    fn the_demo_fill_is_still_blue_dominant() {
        // `nguo-app/tests/render_graph.rs` counts blue pixels to prove the Vello
        // pass ran. Swapping the literal for a token must not break that, and
        // this is where the coupling is cheap to check.
        let c = Tokens::dark().color.accent;
        assert!(c.b > 0.5, "accent is not blue: {c:?}");
        assert!(
            c.b - c.r > 40.0 / 255.0,
            "accent is not blue-dominant enough for the pixel test: {c:?}"
        );
    }

    #[test]
    fn the_facade_hits_snaps_and_draws_one_document() {
        // The end-to-end shape of the authoring loop, in the units it actually
        // uses: screen px in, world mm out, one frame drawn.
        let (doc, piece, edges) = fixture::square_doc();
        let size = Vec2::new(900.0, 700.0);
        let mut canvas = Canvas2d::new();
        canvas.view.pan = Vec2::new(50.0, 50.0); // centre the square

        let corner = canvas.view.world_to_screen(Point::new(0.0, 0.0), size);
        let hit = canvas
            .hit(&doc, corner, size)
            .expect("the corner is hittable");
        assert_eq!(hit.target.layer(), Layer::Point);
        assert_eq!(hit.target.piece(), piece);

        // 3 px off the same corner: snapping puts it back, in world units.
        let off = Point::new(corner.x + 3.0, corner.y);
        assert_eq!(canvas.resolve(&doc, off, size), Point::new(0.0, 0.0));
        assert_eq!(
            canvas.interaction.snap.map(|s| s.kind),
            Some(SnapKind::Endpoint)
        );

        canvas.interaction.selection = vec![Target::Edge {
            piece,
            edge: edges[0],
        }];
        let frame = canvas.frame(&doc, size, &Tokens::light());
        assert!(!frame.encoding().is_empty());
        assert_eq!(canvas.piece_builds(), 1);
    }

    #[test]
    fn the_index_is_rebuilt_only_when_the_document_moves_on() {
        let (mut doc, piece, _) = fixture::square_doc();
        let mut canvas = Canvas2d::new();
        let first = canvas.index(&doc).revision();
        assert_eq!(canvas.index(&doc).revision(), first, "no needless rebuild");

        let mut stack = nguo_doc_commands::CommandStack::new();
        let v = doc
            .piece(piece)
            .unwrap()
            .network
            .vertices
            .ids()
            .next()
            .unwrap();
        let tx = pen::move_point(&doc, piece, v, Point::new(9.0, 9.0)).unwrap();
        stack.apply(&mut doc, tx).unwrap();
        assert_ne!(canvas.index(&doc).revision(), first, "a change invalidates");
        assert!(canvas.index(&doc).hit(Point::new(9.0, 9.0), 0.5).is_some());
    }
}

#[cfg(test)]
mod routing_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use iced::widget::shader::Program as _;

    /// The widget is not at the window origin — that offset is exactly what a
    /// hit-test gets wrong when someone mixes up window and widget space.
    const BOUNDS: Rectangle = Rectangle {
        x: 641.0,
        y: 56.0,
        width: 639.0,
        height: 648.0,
    };

    fn at(x: f32, y: f32) -> mouse::Cursor {
        mouse::Cursor::Available(iced::Point::new(BOUNDS.x + x, BOUNDS.y + y))
    }

    fn send(
        canvas: &Canvas,
        state: &mut State,
        event: iced::Event,
        cursor: mouse::Cursor,
    ) -> Option<(Option<Action>, iced::event::Status)> {
        canvas.update(state, &event, BOUNDS, cursor).map(|a| {
            let (message, _, status) = a.into_inner();
            (message, status)
        })
    }

    fn press(canvas: &Canvas, state: &mut State, x: f32, y: f32) -> Pointer {
        let (message, status) = send(
            canvas,
            state,
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            at(x, y),
        )
        .expect("a click inside the widget must publish");
        assert_eq!(status, iced::event::Status::Captured);
        match message.expect("the click carries a message") {
            Action::Pointer(p) => p,
            other => panic!("a press must publish a Pointer, got {other:?}"),
        }
    }

    #[test]
    fn a_click_reports_the_widget_relative_point_under_the_cursor() {
        let canvas = Canvas {
            view: View {
                pan: Vec2::new(12.0, -3.0),
                zoom: 2.5,
            },
            pick: None,
        };
        let mut state = State::default();
        let p = press(&canvas, &mut state, 300.0, 200.0);
        assert_eq!(p.phase, Phase::Pressed);

        // Widget-relative, not window-relative.
        assert_abs_diff_eq!(p.screen.x, 300.0, epsilon = 1e-6);
        assert_abs_diff_eq!(p.screen.y, 200.0, epsilon = 1e-6);
        assert_eq!(p.size, Vec2::new(639.0, 648.0));

        // And the round trip through the view lands back on the same pixel.
        let world = canvas.view.screen_to_world(p.screen, p.size);
        let back = canvas.view.world_to_screen(world, p.size);
        assert_abs_diff_eq!(back.x, 300.0, epsilon = 1e-6);
        assert_abs_diff_eq!(back.y, 200.0, epsilon = 1e-6);
    }

    #[test]
    fn a_click_outside_the_widget_is_ignored() {
        let canvas = Canvas {
            view: View::default(),
            pick: None,
        };
        let mut state = State::default();
        assert!(send(
            &canvas,
            &mut state,
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            mouse::Cursor::Available(iced::Point::new(10.0, 10.0)),
        )
        .is_none());
        assert!(!state.left_down);
    }

    /// The regression that made drafting impossible: a left press used to set
    /// `dragging`, so the first segment anyone tried to draw panned the view
    /// instead. Pan is middle-drag now, and a left drag is the tool's.
    #[test]
    fn a_left_drag_belongs_to_the_tool_and_a_middle_drag_pans() {
        let canvas = Canvas {
            view: View::default(),
            pick: None,
        };
        let mut state = State::default();
        press(&canvas, &mut state, 100.0, 100.0);

        let (message, _) = send(
            &canvas,
            &mut state,
            iced::Event::Mouse(mouse::Event::CursorMoved {
                position: iced::Point::new(BOUNDS.x + 130.0, BOUNDS.y + 90.0),
            }),
            at(130.0, 90.0),
        )
        .expect("a left drag publishes");
        assert!(
            matches!(
                message,
                Some(Action::Pointer(Pointer {
                    phase: Phase::Dragged,
                    ..
                }))
            ),
            "a left drag must reach the tool, not the view: {message:?}"
        );

        // Middle button, same motion: now it pans.
        let mut state = State::default();
        send(
            &canvas,
            &mut state,
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Middle)),
            at(100.0, 100.0),
        );
        let (message, _) = send(
            &canvas,
            &mut state,
            iced::Event::Mouse(mouse::Event::CursorMoved {
                position: iced::Point::new(BOUNDS.x + 130.0, BOUNDS.y + 90.0),
            }),
            at(130.0, 90.0),
        )
        .expect("a middle drag publishes");
        assert_eq!(
            message,
            Some(Action::Pan {
                delta: Vec2::new(30.0, -10.0)
            })
        );
    }

    #[test]
    fn hover_publishes_without_capturing_and_a_release_always_arrives() {
        let canvas = Canvas {
            view: View::default(),
            pick: None,
        };
        let mut state = State::default();

        // Hover must not capture, or the dock's divider drag — which tracks the
        // cursor at window level — is starved whenever the pointer crosses us.
        let (message, status) = send(
            &canvas,
            &mut state,
            iced::Event::Mouse(mouse::Event::CursorMoved {
                position: iced::Point::new(BOUNDS.x + 50.0, BOUNDS.y + 50.0),
            }),
            at(50.0, 50.0),
        )
        .expect("hover publishes");
        assert_eq!(status, iced::event::Status::Ignored);
        assert!(matches!(
            message,
            Some(Action::Pointer(Pointer {
                phase: Phase::Hovered,
                ..
            }))
        ));

        // A drag released *outside* the widget still ends the drag. This is the
        // `mouse_area` early-return bug in the other direction.
        press(&canvas, &mut state, 10.0, 10.0);
        let (message, _) = send(
            &canvas,
            &mut state,
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
            mouse::Cursor::Available(iced::Point::new(5.0, 5.0)),
        )
        .expect("an off-widget release still publishes");
        assert!(matches!(
            message,
            Some(Action::Pointer(Pointer {
                phase: Phase::Released,
                ..
            }))
        ));
        assert!(!state.left_down, "the gesture ended");
    }

    #[test]
    fn modifiers_ride_along_on_the_pointer() {
        let canvas = Canvas {
            view: View::default(),
            pick: None,
        };
        let mut state = State::default();
        let mut held = iced::keyboard::Modifiers::default();
        held.toggle(iced::keyboard::Modifiers::COMMAND);
        send(
            &canvas,
            &mut state,
            iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(held)),
            at(10.0, 10.0),
        );
        assert!(
            press(&canvas, &mut state, 10.0, 10.0).modifiers.command(),
            "the suspend-snapping modifier must reach Canvas2d::pointer"
        );
    }
}
