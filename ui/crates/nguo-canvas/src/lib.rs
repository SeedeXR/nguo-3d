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
//! * [`grid`] — grid and ruler geometry, derived from that matrix.
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

use iced::widget::shader;
use iced::{mouse, Rectangle};
use nguo_doc::Document;
use nguo_doc_commands::Transaction;
use nguo_gpu::canvas::ScenePrimitive;
use nguo_gpu::scale;
use nguo_ui_theme::Tokens;
use vello::kurbo::{Affine, BezPath, Circle, Point, Stroke, Vec2};
use vello::peniko::{Color, Fill};
use vello::Scene;

pub use hit::{Hit, Index, Layer, Target};
pub use pen::Pen;
pub use scene::{Interaction, SceneCache};
pub use snap::{Snap, SnapKind, SnapToggles, Snapper};
pub use view::View;

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
    /// The user clicked; here is where, in world units.
    Picked {
        world: Point,
        screen: Point,
    },
}

#[derive(Debug, Default)]
pub struct State {
    dragging: bool,
    last_cursor: Option<iced::Point>,
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

        match event {
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let position = cursor.position_in(bounds)?;
                state.dragging = true;
                state.last_cursor = cursor.position();

                let screen = Point::new(position.x as f64, position.y as f64);
                Some(
                    shader::Action::publish(Action::Picked {
                        world: self.view.screen_to_world(screen, size),
                        screen,
                    })
                    .and_capture(),
                )
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.dragging = false;
                state.last_cursor = None;
                None
            }
            iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if !state.dragging {
                    return None;
                }
                let position = cursor.position()?;
                let last = state.last_cursor.replace(position)?;

                Some(
                    shader::Action::publish(Action::Pan {
                        delta: Vec2::new(
                            (position.x - last.x) as f64,
                            (position.y - last.y) as f64,
                        ),
                    })
                    .and_capture(),
                )
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
        if state.dragging {
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
    index: Option<Index>,
    cache: SceneCache,
}

impl Default for Canvas2d {
    fn default() -> Self {
        Canvas2d {
            view: View::default(),
            snapper: Snapper::default(),
            pen: Pen::default(),
            interaction: Interaction::default(),
            pick_px: 7.0,
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
        let (view, snapper) = (self.view, self.snapper);
        let snap = snapper.snap(raw, view.zoom, self.index(doc));
        self.interaction.snap = snap;
        snap.map_or(raw, |s| s.world)
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

    fn press_at(canvas: &Canvas, x: f32, y: f32) -> Action {
        let mut state = State::default();
        let action = canvas
            .update(
                &mut state,
                &iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
                BOUNDS,
                mouse::Cursor::Available(iced::Point::new(BOUNDS.x + x, BOUNDS.y + y)),
            )
            .expect("a click inside the widget must publish");

        let (message, _, status) = action.into_inner();
        assert_eq!(status, iced::event::Status::Captured);
        assert!(state.dragging, "a press starts the pan gesture");
        message.expect("the click carries a message")
    }

    #[test]
    fn a_click_reports_the_world_point_under_the_cursor() {
        let canvas = Canvas {
            view: View {
                pan: Vec2::new(12.0, -3.0),
                zoom: 2.5,
            },
            pick: None,
        };
        let size = Vec2::new(BOUNDS.width as f64, BOUNDS.height as f64);

        let Action::Picked { world, screen } = press_at(&canvas, 300.0, 200.0) else {
            panic!("a press must publish Picked");
        };

        // Widget-relative, not window-relative.
        assert_abs_diff_eq!(screen.x, 300.0, epsilon = 1e-6);
        assert_abs_diff_eq!(screen.y, 200.0, epsilon = 1e-6);

        let expected = canvas.view.screen_to_world(screen, size);
        assert_abs_diff_eq!(world.x, expected.x, epsilon = 1e-9);
        assert_abs_diff_eq!(world.y, expected.y, epsilon = 1e-9);

        // And the round trip lands back on the same pixel.
        let back = canvas.view.world_to_screen(world, size);
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
        assert!(canvas
            .update(
                &mut state,
                &iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
                BOUNDS,
                mouse::Cursor::Available(iced::Point::new(10.0, 10.0)),
            )
            .is_none());
        assert!(!state.dragging);
    }

    #[test]
    fn dragging_publishes_a_pan_of_the_cursor_delta() {
        let canvas = Canvas {
            view: View::default(),
            pick: None,
        };
        let mut state = State::default();
        let _ = canvas.update(
            &mut state,
            &iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            BOUNDS,
            mouse::Cursor::Available(iced::Point::new(700.0, 100.0)),
        );

        let action = canvas
            .update(
                &mut state,
                &iced::Event::Mouse(mouse::Event::CursorMoved {
                    position: iced::Point::new(730.0, 90.0),
                }),
                BOUNDS,
                mouse::Cursor::Available(iced::Point::new(730.0, 90.0)),
            )
            .expect("a drag publishes");

        let (message, _, _) = action.into_inner();
        assert_eq!(
            message,
            Some(Action::Pan {
                delta: Vec2::new(30.0, -10.0)
            })
        );
    }
}
