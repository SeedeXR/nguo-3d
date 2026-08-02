// SPDX-License-Identifier: GPL-2.0-or-later
//! The egui debug overlay's *content*: frame timing against the 8.3 ms
//! target / 16.6 ms ceiling, plus the process registry.
//!
//! `nguo-gpu` owns the egui/iced/wgpu plumbing; this crate only decides what
//! the overlay says. Debug builds only — the app gates it behind the
//! `debug-overlay` feature.

use iced::widget::shader;
use iced::{mouse, Rectangle};
use nguo_gpu::overlay::OverlayPrimitive;

/// Everything the overlay displays. Pure data, snapshotted per frame.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub frame_ms: f32,
    pub fps: f32,
    pub frames: u64,
    pub adapter: String,
    pub surface_format: String,
    pub present_mode: String,
    pub scale_factor: f32,
    pub window_physical: (u32, u32),
    /// Last click in the 2D canvas, in canvas world units.
    pub canvas_pick: Option<(f64, f64)>,
    /// Last click in the 3D viewport, on the ground plane.
    pub viewport_pick: Option<(f32, f32, f32)>,
    /// The live process registry, copied out once per frame. Borrows nothing,
    /// so the overlay holds no lock while egui lays itself out
    /// (`ui/docs/OBSERVABILITY.md` §2.4).
    pub registry: nguo_registry::Snapshot,
}

/// The overlay widget: fills the window, draws on top, publishes nothing.
pub fn overlay<'a, Message: 'a>(snapshot: Snapshot) -> iced::Element<'a, Message> {
    shader(Overlay { snapshot })
        .width(iced::Length::Fill)
        .height(iced::Length::Fill)
        .into()
}

struct Overlay {
    snapshot: Snapshot,
}

/// The live process list (`ui/docs/OBSERVABILITY.md` §2). Reports exactly what
/// the registry holds — including "nothing", which is the honest answer while
/// no subsystem in this binary registers work.
fn processes(ui: &mut egui::Ui, snapshot: &nguo_registry::Snapshot) {
    ui.label(format!(
        "processes  {} active · {} recent · {} failed",
        snapshot.active.len(),
        snapshot.recent.len(),
        snapshot.failures.len(),
    ));

    for p in &snapshot.active {
        let progress = match p.progress {
            Some(f) => format!("  {:.0}%", f * 100.0),
            None => String::new(),
        };
        ui.label(format!(
            "  {} {:<22} {:<8} {:>6.2}s{progress}",
            p.id,
            p.feature.name,
            p.state.as_str(),
            p.elapsed.as_secs_f32(),
        ));
    }

    for c in snapshot.recent.iter().take(5) {
        ui.label(format!(
            "  {} {:<22} {:<8} {:>6.2}s",
            c.id,
            c.feature.name,
            c.outcome.as_str(),
            c.duration.as_secs_f32(),
        ));
    }

    // `title` / `detail` / `root_cause` are the same three strings the app
    // puts in a `nguo_ui_widgets::toast`; the overlay adds the whole chain.
    for f in snapshot.failures.iter().take(3) {
        ui.colored_label(
            egui::Color32::LIGHT_RED,
            format!("  {} {}: {}", f.id, f.title(), f.detail()),
        );
        ui.label(format!("      root cause: {}", f.root_cause()));
    }
}

impl<Message> shader::Program<Message> for Overlay {
    type State = ();
    type Primitive = OverlayPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        let snapshot = self.snapshot.clone();

        OverlayPrimitive::new(move |ctx| {
            egui::Window::new("nguo debug")
                // Over the 3D viewport, not the top-left: at [12, 12] it sat
                // on the toolbar and the scene tree — the two things you most
                // often have the overlay open to check against.
                .default_pos([760.0, 60.0])
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "frame  {:.2} ms  ({:.0} fps)",
                        snapshot.frame_ms, snapshot.fps
                    ));
                    ui.label(format!("frames {}", snapshot.frames));
                    ui.separator();
                    ui.label(format!("adapter {}", snapshot.adapter));
                    ui.label(format!("surface {}", snapshot.surface_format));
                    ui.label(format!("present {}", snapshot.present_mode));
                    ui.separator();
                    ui.label(format!(
                        "scale   {:.2}x   window {}x{} px",
                        snapshot.scale_factor,
                        snapshot.window_physical.0,
                        snapshot.window_physical.1
                    ));
                    ui.label(match snapshot.canvas_pick {
                        Some((x, y)) => {
                            format!("canvas pick   ({x:.2}, {y:.2})")
                        }
                        None => "canvas pick   —".to_owned(),
                    });
                    ui.label(match snapshot.viewport_pick {
                        Some((x, y, z)) => {
                            format!("viewport pick ({x:.2}, {y:.2}, {z:.2})")
                        }
                        None => "viewport pick —".to_owned(),
                    });
                    ui.separator();
                    processes(ui, &snapshot.registry);
                });
        })
    }
}
