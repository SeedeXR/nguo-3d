// SPDX-License-Identifier: GPL-2.0-or-later
//! The 3D pass. Plain wgpu — no iced — so it can be driven from a headless
//! test as well as from the widget.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use nguo_gpu::scale::PhysicalRect;

use crate::{Avatar, AvatarMesh, Vertex};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// 32x32 quads, two triangles each, generated in the vertex shader.
const CELLS: u32 = 32;
const VERTEX_COUNT: u32 = CELLS * CELLS * 6;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    time: f32,
    _pad: [f32; 3],
}

#[derive(Debug)]
pub struct ClothRenderer {
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth: Option<Depth>,
}

#[derive(Debug)]
struct Depth {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// Allocate (or reallocate) a frame-sized depth attachment into `slot`.
///
/// Shared by both passes: the depth buffer has to match the colour
/// attachment, which is the whole iced frame rather than either widget.
fn ensure_depth(slot: &mut Option<Depth>, device: &wgpu::Device, width: u32, height: u32) {
    if matches!(&slot, Some(d) if d.width == width && d.height == height) {
        return;
    }

    tracing::debug!(target: "nguo_viewport3d", width, height, "(re)allocating depth buffer");

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nguo viewport depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    *slot = Some(Depth {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        width,
        height,
    });
}

impl ClothRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nguo cloth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cloth.wgsl").into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nguo cloth uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nguo cloth bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nguo cloth bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nguo cloth pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nguo cloth pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // The sheet waves through itself; showing both sides is
                // the honest thing for cloth.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms,
            bind_group,
            depth: None,
        }
    }

    pub fn upload(&self, queue: &wgpu::Queue, view_projection: Mat4, time: f32) {
        queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_projection: view_projection.to_cols_array_2d(),
                time,
                _pad: [0.0; 3],
            }),
        );
    }

    /// The depth buffer must match the colour attachment, which is the whole
    /// frame, not just our widget.
    pub fn ensure_depth(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        ensure_depth(&mut self.depth, device, width, height);
    }

    /// Records the viewport pass into `encoder`. `rect` is where the widget
    /// lives in the frame, `clip` is that intersected with its clip bounds —
    /// both in physical pixels.
    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        rect: PhysicalRect,
        clip: PhysicalRect,
    ) {
        let Some(depth) = &self.depth else {
            tracing::warn!(target: "nguo_viewport3d", "no depth buffer; skipping pass");
            return;
        };
        if rect.is_empty() || clip.is_empty() {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nguo viewport3d pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The iced frame is already cleared to the theme
                    // background; we only paint our rectangle.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_viewport(
            rect.x as f32,
            rect.y as f32,
            rect.width as f32,
            rect.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(clip.x, clip.y, clip.width, clip.height);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..VERTEX_COUNT, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Avatar pass
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AvatarUniforms {
    view_projection: [[f32; 4]; 4],
    /// Linear `baseColorFactor` — see `nguo_avatar::MonkTone::factor`.
    base_color: [f32; 4],
}

/// Draws one indexed triangle mesh with two-light Lambert shading.
///
/// Geometry is uploaded only when the [`AvatarMesh`] behind the `Arc` actually
/// changes, so switching skin tone (a uniform write) never re-uploads a
/// 50 000-triangle body.
#[derive(Debug)]
pub struct AvatarRenderer {
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth: Option<Depth>,
    geometry: Option<Geometry>,
}

#[derive(Debug)]
struct Geometry {
    /// Identity of what is on the GPU, compared by pointer.
    source: Arc<AvatarMesh>,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl AvatarRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nguo avatar"),
            source: wgpu::ShaderSource::Wgsl(include_str!("avatar.wgsl").into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nguo avatar uniforms"),
            size: std::mem::size_of::<AvatarUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nguo avatar bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nguo avatar bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nguo avatar pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nguo avatar pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // No back-face culling, and this is now a *measured* decision
                // rather than a deferral. The bundle body is watertight and
                // manifold and declares `doubleSided: false`, so culling is
                // legal (INTEGRATION.md §3) — but the shell is vsync-bound:
                // 50 000 triangles drawn measured 8.33–8.74 ms against
                // 8.35–8.56 ms for the same shell with no body at all, i.e.
                // the body is already free at the 8.33 ms `Fifo` floor. Halving
                // fragment work that costs nothing buys nothing, and `abs()` in
                // the shader means a winding mistake shows up as flat shading
                // instead of a hole.
                //
                // ponytail: revisit only if a future pass is measurably
                // fragment-bound — at LOD1 (200 k tris) or with real skin
                // shading, not here.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms,
            bind_group,
            depth: None,
            geometry: None,
        }
    }

    pub fn ensure_depth(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        ensure_depth(&mut self.depth, device, width, height);
    }

    /// The depth attachment this renderer allocated, for a sibling pass to share.
    ///
    /// Two mesh passes in one frame **must** test against one depth texture, or
    /// the second one is not occluded by the first at all. So exactly one
    /// renderer per frame calls [`Self::ensure_depth`] and hands this to the
    /// others.
    pub fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.depth.as_ref().map(|d| &d.view)
    }

    /// Upload the frame's uniforms, and the geometry too if the mesh changed.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        avatar: &Avatar,
        view_projection: Mat4,
    ) {
        let stale = !matches!(&self.geometry, Some(g) if Arc::ptr_eq(&g.source, &avatar.mesh));
        if stale {
            self.geometry = Some(Geometry::upload(device, &avatar.mesh));
        }

        queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&AvatarUniforms {
                view_projection: view_projection.to_cols_array_2d(),
                base_color: avatar.base_color,
            }),
        );
    }

    /// Records the avatar pass. `rect` is the widget in the frame, `clip` that
    /// intersected with its clip bounds — both physical pixels.
    ///
    /// `depth` is the frame's shared depth attachment (see [`Self::depth_view`]).
    /// `clear_depth` must be `true` for the FIRST mesh pass of the frame and
    /// `false` for every one after it. Clearing twice is how a garment ends up
    /// drawn straight through the body it is meant to be occluded by.
    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        rect: PhysicalRect,
        clip: PhysicalRect,
        depth: &wgpu::TextureView,
        clear_depth: bool,
    ) {
        let Some(geometry) = &self.geometry else {
            tracing::warn!(target: "nguo_viewport3d", "no mesh uploaded; skipping mesh pass");
            return;
        };
        if rect.is_empty() || clip.is_empty() || geometry.index_count == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nguo avatar pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: if clear_depth {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    // Kept, not discarded, when a later pass in the same frame
                    // still has to test against it.
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_viewport(
            rect.x as f32,
            rect.y as f32,
            rect.width as f32,
            rect.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(clip.x, clip.y, clip.width, clip.height);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, geometry.vertices.slice(..));
        pass.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..geometry.index_count, 0, 0..1);
    }
}

impl Geometry {
    fn upload(device: &wgpu::Device, mesh: &Arc<AvatarMesh>) -> Geometry {
        use wgpu::util::DeviceExt as _;

        tracing::info!(
            target: "nguo_viewport3d",
            vertices = mesh.vertices.len(),
            triangles = mesh.indices.len() / 3,
            "uploading avatar geometry"
        );

        Geometry {
            source: Arc::clone(mesh),
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nguo avatar vertices"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nguo avatar indices"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: mesh.indices.len() as u32,
        }
    }
}
