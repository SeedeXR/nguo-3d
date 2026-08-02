// SPDX-License-Identifier: GPL-2.0-or-later
//! The 3D drape viewport, as an iced custom-primitive
//! widget on the shared device.

pub mod camera;
mod renderer;

use glam::{Mat4, Vec2, Vec3};
use iced::widget::shader::{self, Primitive};
use iced::{mouse, Rectangle};
use nguo_gpu::scale::{self, PhysicalRect};

pub use camera::Camera;
pub use renderer::{ClothRenderer, DEPTH_FORMAT};

/// What the viewport asks the application to do. The widget owns no
/// authoritative state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Orbit {
        yaw: f32,
        pitch: f32,
    },
    Dolly {
        factor: f32,
    },
    /// The user clicked; here is where they clicked, on the ground plane.
    Picked {
        world: Vec3,
        screen: Vec2,
    },
}

/// Per-widget interaction state. Not authoritative — just the drag gesture.
#[derive(Debug, Default)]
pub struct State {
    dragging: bool,
    last_cursor: Option<iced::Point>,
}

/// The widget program. Cheap to rebuild every `view()`.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub camera: Camera,
    /// Animation clock, seconds.
    pub time: f32,
}

/// Radians of orbit per logical point of drag.
const ORBIT_SENSITIVITY: f32 = 0.008;

impl shader::Program<Action> for Viewport {
    type State = State;
    type Primitive = ClothPrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Action>> {
        match event {
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let position = cursor.position_in(bounds)?;
                state.dragging = true;
                state.last_cursor = cursor.position();

                let size = Vec2::new(bounds.width, bounds.height);
                let screen = Vec2::new(position.x, position.y);
                let world = self.camera.screen_to_world(screen, size)?;

                Some(shader::Action::publish(Action::Picked { world, screen }).and_capture())
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
                    shader::Action::publish(Action::Orbit {
                        yaw: -(position.x - last.x) * ORBIT_SENSITIVITY,
                        pitch: (position.y - last.y) * ORBIT_SENSITIVITY,
                    })
                    .and_capture(),
                )
            }
            iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                cursor.position_in(bounds)?;
                let lines = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y,
                    mouse::ScrollDelta::Pixels { y, .. } => y / 40.0,
                };
                Some(
                    shader::Action::publish(Action::Dolly {
                        factor: (1.0 - lines * 0.1).clamp(0.5, 2.0),
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
        _bounds: Rectangle,
    ) -> Self::Primitive {
        ClothPrimitive {
            camera: self.camera,
            time: self.time,
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
            mouse::Interaction::Grab
        } else {
            mouse::Interaction::default()
        }
    }
}

/// One frame of viewport content.
#[derive(Debug, Clone, Copy)]
pub struct ClothPrimitive {
    camera: Camera,
    time: f32,
}

#[derive(Debug)]
pub struct ClothPipeline {
    renderer: ClothRenderer,
    // ponytail: a single viewport instance, so the current frame's rectangle
    // can live on the shared pipeline. Key it per widget when we add a second
    // 3D view (split-screen / detached window, guide §2.6).
    rect: PhysicalRect,
}

impl shader::Pipeline for ClothPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        nguo_gpu::adopt(device, queue, format, "viewport3d");

        Self {
            renderer: ClothRenderer::new(device, format),
            rect: PhysicalRect::default(),
        }
    }
}

impl Primitive for ClothPrimitive {
    type Pipeline = ClothPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        nguo_gpu::note_viewport(viewport);

        pipeline.rect = scale::physical_rect(
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
            viewport.scale_factor(),
        );

        // The depth attachment has to match the colour attachment, which is
        // the whole frame.
        let frame = viewport.physical_size();
        pipeline
            .renderer
            .ensure_depth(device, frame.width, frame.height);

        // Aspect from the LOGICAL bounds: identical to the physical aspect,
        // and it does not blow up when a resize leaves a zero-width target.
        let aspect = if bounds.height > 0.0 {
            bounds.width / bounds.height
        } else {
            1.0
        };
        pipeline
            .renderer
            .upload(queue, self.camera.view_projection(aspect), self.time);
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        pipeline.renderer.draw(
            encoder,
            target,
            pipeline.rect,
            PhysicalRect {
                x: clip_bounds.x,
                y: clip_bounds.y,
                width: clip_bounds.width,
                height: clip_bounds.height,
            },
        );
    }
}

/// The view-projection a headless test should use for a given target size.
pub fn view_projection(camera: &Camera, width: u32, height: u32) -> Mat4 {
    camera.view_projection(width as f32 / height as f32)
}

#[cfg(test)]
mod routing_tests {
    use super::*;
    use iced::widget::shader::Program as _;

    /// The viewport sits at the window origin's left half, below the header.
    const BOUNDS: Rectangle = Rectangle {
        x: 0.0,
        y: 56.0,
        width: 639.0,
        height: 648.0,
    };

    #[test]
    fn a_click_reports_the_ground_point_under_the_cursor() {
        let viewport = Viewport {
            camera: Camera::default(),
            time: 0.0,
        };
        let mut state = State::default();

        let action = viewport
            .update(
                &mut state,
                &iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
                BOUNDS,
                mouse::Cursor::Available(iced::Point::new(BOUNDS.x + 320.0, BOUNDS.y + 400.0)),
            )
            .expect("a click inside the widget must publish");

        let (message, _, status) = action.into_inner();
        assert_eq!(status, iced::event::Status::Captured);
        assert!(state.dragging);

        let Some(Action::Picked { world, screen }) = message else {
            panic!("a press must publish Picked");
        };

        // Widget-relative, not window-relative.
        assert_eq!(screen, Vec2::new(320.0, 400.0));
        assert!(world.y.abs() < 1e-4, "the pick is on the ground plane");

        // Projecting it back lands on the same pixel we clicked.
        let size = Vec2::new(BOUNDS.width, BOUNDS.height);
        let back = viewport
            .camera
            .world_to_screen(world, size)
            .expect("the pick projects back");
        assert!(
            (back - screen).length() < 0.1,
            "round trip drifted: {back:?} vs {screen:?}"
        );
    }

    #[test]
    fn dragging_orbits_and_releasing_stops() {
        let viewport = Viewport {
            camera: Camera::default(),
            time: 0.0,
        };
        let mut state = State::default();

        let _ = viewport.update(
            &mut state,
            &iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            BOUNDS,
            mouse::Cursor::Available(iced::Point::new(100.0, 100.0)),
        );

        let action = viewport
            .update(
                &mut state,
                &iced::Event::Mouse(mouse::Event::CursorMoved {
                    position: iced::Point::new(150.0, 120.0),
                }),
                BOUNDS,
                mouse::Cursor::Available(iced::Point::new(150.0, 120.0)),
            )
            .expect("a drag publishes");
        let (message, _, _) = action.into_inner();
        assert_eq!(
            message,
            Some(Action::Orbit {
                yaw: -50.0 * ORBIT_SENSITIVITY,
                pitch: 20.0 * ORBIT_SENSITIVITY,
            })
        );

        let _ = viewport.update(
            &mut state,
            &iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
            BOUNDS,
            mouse::Cursor::Available(iced::Point::new(150.0, 120.0)),
        );
        assert!(!state.dragging);
        assert!(
            viewport
                .update(
                    &mut state,
                    &iced::Event::Mouse(mouse::Event::CursorMoved {
                        position: iced::Point::new(200.0, 120.0),
                    }),
                    BOUNDS,
                    mouse::Cursor::Available(iced::Point::new(200.0, 120.0)),
                )
                .is_none(),
            "moving after release must not orbit"
        );
    }
}
