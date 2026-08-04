// SPDX-License-Identifier: GPL-2.0-or-later
//! The 3D drape viewport, as an iced custom-primitive
//! widget on the shared device.

pub mod camera;
mod renderer;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use iced::widget::shader::{self, Primitive};
use iced::{mouse, Rectangle};
use nguo_gpu::scale::{self, PhysicalRect};

pub use camera::Camera;
pub use renderer::{AvatarRenderer, ClothRenderer, DEPTH_FORMAT};

/// One mesh vertex, in the bundle's own space: metres, +Y up, +Z forward,
/// right-handed. No axis conversion — that is glTF's convention and ours.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Avatar geometry, ready to upload. Immutable, and shared by `Arc` so the
/// per-frame [`Viewport`] stays cheap to clone and the renderer can tell "same
/// body" from "different body" by pointer.
#[derive(Debug, PartialEq)]
pub struct AvatarMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl AvatarMesh {
    /// Build from a loaded `engine_api::Object`-shaped mesh.
    ///
    /// Normals are optional in glTF and the bundle's are OpenGL / Y+, which is
    /// what this shader wants as-is. A mesh without them gets flat +Y normals
    /// rather than a panic — it renders dull, not missing.
    pub fn new(positions: &[[f32; 3]], normals: &[[f32; 3]], tris: &[[u32; 3]]) -> AvatarMesh {
        let aligned = normals.len() == positions.len();
        AvatarMesh {
            vertices: positions
                .iter()
                .enumerate()
                .map(|(i, position)| Vertex {
                    position: *position,
                    normal: if aligned { normals[i] } else { [0.0, 1.0, 0.0] },
                })
                .collect(),
            indices: tris.iter().flatten().copied().collect(),
        }
    }

    /// `(min, max)` corner of the axis-aligned bounds, in metres. `None` for
    /// an empty mesh.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.vertices.iter().map(|v| Vec3::from(v.position));
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }
}

/// One lit mesh the viewport should draw this frame.
///
/// Used for the body *and* for the garment: a lit indexed triangle mesh with a
/// base colour is the same thing either way, so they share the struct and the
/// pipeline rather than growing a second copy of both. The name is the body's
/// because that is what landed first.
#[derive(Debug, Clone)]
pub struct Avatar {
    pub mesh: Arc<AvatarMesh>,
    /// Linear `baseColorFactor`, from `nguo_avatar::MonkTone::factor` — **not**
    /// a Monk swatch colour, and not the published linear tone either. Those
    /// are three different colours (`INTEGRATION.md` §4).
    pub base_color: [f32; 4],
}

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

/// The widget program. Cheap to rebuild every `view()` — the avatar is an
/// `Arc` clone, not a mesh copy.
#[derive(Debug, Clone)]
pub struct Viewport {
    pub camera: Camera,
    /// Animation clock, seconds.
    pub time: f32,
    /// The body to draw. `None` — no bundle, or it failed to load — falls back
    /// to the bring-up cloth sheet, so a clone without `assets/avatars/` still
    /// has a populated viewport.
    pub avatar: Option<Avatar>,
    /// The garment, drawn over the body in the same frame. `None` until one has
    /// been sewn. A garment with no body still draws — that is the honest
    /// picture of a drape under gravity with nothing to drape over.
    pub garment: Option<Avatar>,
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
            avatar: self.avatar.clone(),
            garment: self.garment.clone(),
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
#[derive(Debug, Clone)]
pub struct ClothPrimitive {
    camera: Camera,
    time: f32,
    avatar: Option<Avatar>,
    garment: Option<Avatar>,
}

#[derive(Debug)]
pub struct ClothPipeline {
    renderer: ClothRenderer,
    avatar: AvatarRenderer,
    /// A second instance of the same pass, with its own geometry and its own
    /// uniform buffer. Two `AvatarRenderer`s rather than one renderer with two
    /// geometry slots: the pipeline, layout and shader are shared through
    /// `Arc`s inside wgpu, so the duplication is one uniform buffer.
    garment: AvatarRenderer,
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
            avatar: AvatarRenderer::new(device, format),
            garment: AvatarRenderer::new(device, format),
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

        // Aspect from the LOGICAL bounds: identical to the physical aspect,
        // and it does not blow up when a resize leaves a zero-width target.
        let aspect = if bounds.height > 0.0 {
            bounds.width / bounds.height
        } else {
            1.0
        };
        let view_projection = self.camera.view_projection(aspect);

        // The depth attachment has to match the colour attachment, which is the
        // whole frame. `pipeline.avatar` allocates the frame's ONE depth texture
        // whenever any mesh is drawn, and the garment pass shares it — that
        // sharing is what makes the body occlude the garment behind it.
        let frame = viewport.physical_size();
        let meshes = self.avatar.is_some() || self.garment.is_some();
        if meshes {
            pipeline
                .avatar
                .ensure_depth(device, frame.width, frame.height);
        }
        if let Some(avatar) = &self.avatar {
            pipeline
                .avatar
                .upload(device, queue, avatar, view_projection);
        }
        if let Some(garment) = &self.garment {
            pipeline
                .garment
                .upload(device, queue, garment, view_projection);
        }
        if !meshes {
            pipeline
                .renderer
                .ensure_depth(device, frame.width, frame.height);
            pipeline.renderer.upload(queue, view_projection, self.time);
        }
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let clip = PhysicalRect {
            x: clip_bounds.x,
            y: clip_bounds.y,
            width: clip_bounds.width,
            height: clip_bounds.height,
        };

        // Real geometry replaces the bring-up sheet rather than sharing the
        // frame with it: the sheet is a horizontal plane at y≈0, which would
        // slice through the avatar's shins.
        //
        // Body first so it clears the shared depth buffer, garment second so it
        // is depth-tested against the body.
        let Some(depth) = pipeline.avatar.depth_view() else {
            if self.avatar.is_none() && self.garment.is_none() {
                pipeline.renderer.draw(encoder, target, pipeline.rect, clip);
            }
            return;
        };
        let mut first = true;
        for (present, renderer) in [
            (self.avatar.is_some(), &pipeline.avatar),
            (self.garment.is_some(), &pipeline.garment),
        ] {
            if !present {
                continue;
            }
            renderer.draw(encoder, target, pipeline.rect, clip, depth, first);
            first = false;
        }
    }
}

/// The view-projection a headless test should use for a given target size.
pub fn view_projection(camera: &Camera, width: u32, height: u32) -> Mat4 {
    camera.view_projection(width as f32 / height as f32)
}

#[cfg(test)]
mod mesh_tests {
    use super::*;

    const POSITIONS: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.8, 0.5]];

    #[test]
    fn a_mesh_keeps_its_normals_and_flattens_its_triangles() {
        let normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let mesh = AvatarMesh::new(&POSITIONS, &normals, &[[0, 1, 2]]);

        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.vertices[2].position, [0.0, 1.8, 0.5]);
        assert_eq!(mesh.vertices[2].normal, [1.0, 0.0, 0.0]);
        assert_eq!(
            mesh.bounds(),
            Some((Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.8, 0.5)))
        );
    }

    /// A mesh whose normals do not line up with its positions must not index
    /// out of bounds, and must not silently pair the wrong normal with a
    /// vertex either — every vertex gets the flat fallback or none does.
    #[test]
    fn mismatched_normals_fall_back_flat_rather_than_panicking() {
        for normals in [vec![], vec![[0.0, 0.0, 1.0]]] {
            let mesh = AvatarMesh::new(&POSITIONS, &normals, &[[0, 1, 2]]);
            assert_eq!(mesh.vertices.len(), 3);
            assert!(
                mesh.vertices.iter().all(|v| v.normal == [0.0, 1.0, 0.0]),
                "a partial normal array must not be used at all",
            );
        }
    }

    #[test]
    fn an_empty_mesh_has_no_bounds_and_draws_nothing() {
        let mesh = AvatarMesh::new(&[], &[], &[]);
        assert_eq!(mesh.bounds(), None);
        assert!(mesh.indices.is_empty());
    }

    /// The vertex buffer layout in `renderer.rs` hardcodes two `Float32x3` at
    /// offsets 0 and 12. If `Vertex` ever grows a field, that layout is wrong
    /// and the body renders as garbage rather than failing to compile.
    #[test]
    fn the_vertex_is_still_two_float3s() {
        assert_eq!(std::mem::size_of::<Vertex>(), 24);
        assert_eq!(std::mem::align_of::<Vertex>(), 4);
    }
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
            avatar: None,
            garment: None,
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
            avatar: None,
            garment: None,
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
