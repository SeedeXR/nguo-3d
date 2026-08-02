// SPDX-License-Identifier: GPL-2.0-or-later
//! The 2D pattern canvas: a Vello scene rendered
//! through iced's device, with pan/zoom and pointer hit-testing.

pub mod view;

use std::sync::Arc;

use iced::widget::shader;
use iced::{mouse, Rectangle};
use nguo_gpu::canvas::ScenePrimitive;
use nguo_gpu::scale;
use vello::kurbo::{Affine, BezPath, Circle, Point, Stroke, Vec2};
use vello::peniko::{Color, Fill};
use vello::Scene;

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

pub const BACKGROUND: Color = Color::from_rgba8(0x1a, 0x1c, 0x22, 0xff);

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

/// Builds the demo scene in **logical** units; the caller scales to physical.
///
/// A filled Bézier shape and a stroked path — the two things §25.1 asks the
/// canvas to prove — plus the pick crosshair.
pub fn build_scene(view: &View, size: Vec2, pick: Option<Point>) -> Scene {
    let mut scene = Scene::new();
    let to_screen = view.scene_transform(size, 1.0);

    scene.fill(
        Fill::NonZero,
        to_screen,
        Color::from_rgba8(0x4c, 0x9a, 0xff, 0xff),
        None,
        &petal(),
    );

    scene.stroke(
        &Stroke::new(3.0 / view.zoom),
        to_screen,
        Color::from_rgba8(0xff, 0xd0, 0x66, 0xff),
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

        scene.stroke(
            &Stroke::new(1.5),
            Affine::IDENTITY,
            Color::from_rgba8(0xff, 0x4d, 0x6d, 0xff),
            None,
            &cross,
        );
        scene.stroke(
            &Stroke::new(1.5),
            Affine::IDENTITY,
            Color::from_rgba8(0xff, 0x4d, 0x6d, 0xff),
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
