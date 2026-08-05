// SPDX-License-Identifier: GPL-2.0-or-later
//! nguo-3d desktop shell (ui/docs/UX_SPEC.md §2).
//!
//! One window, one `wgpu::Device`, four renderers on it: the 3D viewport, the
//! Vello canvas, iced's own UI, and (with `--features debug-overlay`) egui.
//!
//! Around them, the workspace: a splash that reports real progress, a one-row
//! grouped toolbar, four dockable panels with a canvas-width floor, a scene
//! tree projected from `nguo-doc`, a `⌘K` command palette over the keymap, and
//! a status bar reading `nguo-registry`.

mod avatar;
mod command;
mod drape;
mod shell;

use std::collections::BTreeSet;
use std::time::Instant;

use iced::widget::{button, column, container, row, space, stack, text, text_input};
use iced::{Element, Length, Subscription, Task};
use nguo_canvas::Canvas;
use nguo_engine_bridge::Tier;
use nguo_gpu::stats::FrameStats;
use nguo_keymap::{CommandId, Context, Keymap, Resolution, Resolver};
use nguo_registry::Snapshot;
use nguo_splash::Splash;
use nguo_ui_theme::Tokens;
use nguo_ui_widgets as w;
use nguo_ui_widgets::dock::{Axis, DockEvent, DockLayout, Panel, Tab};
use nguo_ui_widgets::icon::Icon;
use nguo_ui_widgets::tree::{TreeEvent, TreeState};
use nguo_viewport3d::{Camera, Viewport};
use vello::kurbo::Point;

/// The toolbar, one row, grouped (UX_SPEC §3.1). Each entry is
/// `(icon, CommandId)`; `None` is a group separator. The label and the
/// shortcut in the tooltip both come from the keymap, so a rebind is reflected
/// here with no second table to update.
/// The workspace window, once the splash has gone. The splash itself opens at
/// `nguo_splash::SIZE` (900x534) — a centred panel over the desktop, never a
/// fullscreen takeover (SPLASH.md §1).
const SHELL_SIZE: iced::Size = iced::Size::new(1440.0, 900.0);

const TOOLBAR: &[Option<(Icon, &str)>] = &[
    Some((Icon::Select, "tool.select")),
    Some((Icon::DirectSelect, "tool.edit_pattern")),
    Some((Icon::Point, "tool.add_points")),
    Some((Icon::Curve, "tool.edit_curvature")),
    None,
    Some((Icon::Line, "tool.shape_rectangle")),
    Some((Icon::Pen, "tool.polygon")),
    Some((Icon::Dart, "tool.dart")),
    Some((Icon::Notch, "tool.notch")),
    Some((Icon::InternalLine, "tool.internal_line")),
    None,
    Some((Icon::Seam, "tool.sewing")),
    Some((Icon::Scissors, "tool.topstitch")),
    Some((Icon::Measure, "tool.measure")),
    None,
    Some((Icon::Snap, "constraint.toggle_display")),
    Some((Icon::Crosshair, "constraint.diagnose")),
    None,
    Some((Icon::Move3d, "tool.select_move_3d")),
    Some((Icon::Point, "tool.pin")),
    Some((Icon::Play, "sim.toggle")),
    None,
    Some((Icon::Search, "agent.palette")),
];

#[derive(Debug, Clone)]
enum Message {
    Tick(Instant),
    /// A framebuffer capture, only ever produced when `NGUO_CAPTURE` is set.
    Captured(&'static str, iced::window::Screenshot),
    /// Window-level events the widgets ignored: keys and cursor motion.
    Event(iced::Event),
    Dock(DockEvent),
    Tree(TreeEvent),
    Command(CommandId),
    PaletteQuery(String),
    PaletteAccept,
    DismissToast(u64),
    Viewport(nguo_viewport3d::Action),
    Canvas(nguo_canvas::Action),
}

/// The splash seam. `nguo-splash` is owned by another agent, so everything the
/// shell knows about it is in this enum and the four calls in `update`.
enum Phase {
    Booting(Box<Splash>),
    Running,
}

struct App {
    phase: Phase,
    /// Initialisation milestones that have genuinely completed, in order. The
    /// splash is fed from this and nothing else — no invented progress.
    stages: Vec<&'static str>,

    tokens: Tokens,
    /// The 2D canvas is light inside a dark shell (UX_SPEC §2). Held as a
    /// second token set rather than a literal: `ui-theme` is the only crate
    /// allowed a colour (ui/docs/DEVELOPING.md §4.7).
    paper: Tokens,

    doc: nguo_doc::Document,
    /// Undo/redo. The canvas returns transactions and cannot apply them —
    /// `Document`'s fields are private to `nguo-doc` — so this is the only
    /// thing in the shell that can change the drawing, and undo therefore
    /// covers every tool for free.
    stack: nguo_canvas::CommandStack,
    /// The scene tree, projected from the document. Rebuilt on a document
    /// edit, not per frame — `view` borrows it.
    scene_rows: Vec<nguo_ui_widgets::tree::TreeNode>,
    layout: DockLayout,
    tree: TreeState,

    keymap: Keymap,
    resolver: Resolver,
    focus: Context,
    active_tool: Option<CommandId>,

    palette: Option<String>,
    palette_hits: Vec<command::Entry>,
    catalogue: Vec<command::Entry>,

    registry: Snapshot,
    dismissed: BTreeSet<u64>,
    simulating: bool,

    // Status-bar readouts. `status_item` takes `&'a str`, so these are
    // recomputed once per frame in `update` rather than allocated in `view`.
    doc_label: String,
    focus_label: String,
    process_label: String,
    fps_label: String,
    /// The drape's own read-out. A `String` for the same reason the three above
    /// are: `status_item` borrows `&'a str`, so it is recomputed once per tick in
    /// `update` rather than allocated inside `view`.
    drape_label: String,
    /// Cursor position in millimetres, and the live length of the pen's rubber
    /// band while a run is in progress. This is what replaced the ruler
    /// gutters: Marvelous Designer's own transcript describes the same thing —
    /// "one tool that does both… as I am clicking and dragging I'm both drawing
    /// lines and I'm measuring them at the same time".
    cursor_label: String,

    /// The body in the viewport and behind the pattern canvas. `None` when
    /// `assets/avatars/` is absent — the assets are not in every clone, so a
    /// missing bundle is a toast and a fallback, never a failure to start.
    avatars: Option<avatar::Avatars>,

    /// The drape: bridge, garment and the streamed mesh. `None` only if the OS
    /// refused a thread pool — the shell still runs, it just cannot simulate.
    drape: Option<drape::Drape>,

    camera: Camera,
    canvas_view: nguo_canvas::View,
    /// The document renderer for the 2D pane, holding the piece scene cache.
    ///
    /// `Mutex` because `shader::Program::draw` takes `&self` while
    /// `Canvas2d::frame` needs `&mut` for its cache, and `draw` is the only
    /// place the pane's real pixel size is known. Uncontended — one lock per
    /// frame on the UI thread — and it is what keeps `piece_builds` from
    /// climbing once per frame.
    canvas2d: std::sync::Mutex<nguo_canvas::Canvas2d>,
    canvas_pick: Option<Point>,
    viewport_pick: Option<glam::Vec3>,
    stats: FrameStats,
    started: Instant,
    time: f32,
    /// Only the overlay shows it; without that feature it is still logged at
    /// startup, which is where a bug report gets it from.
    #[cfg_attr(not(feature = "debug-overlay"), allow(dead_code))]
    adapter: String,
}

impl App {
    fn new() -> Self {
        let mut stages = Vec::new();

        let tokens = Tokens::dark();
        let paper = Tokens::light();
        stages.push("theme tokens");

        let keymap = nguo_keymap::default_macos();
        let mut catalogue = command::catalogue(&keymap);
        stages.push("keymap");

        let doc = shell::sample_document();
        let scene_rows = shell::scene_tree(&doc);
        let layout = shell::default_layout(&tokens);
        stages.push("document");

        // The bundle is optional. `report_avatar_failure` turns a miss into a
        // registry failure, which the existing toast layer already renders.
        let avatars = match avatar::Avatars::load() {
            Ok(avatars) => {
                stages.push("avatar");
                Some(avatars)
            }
            Err(error) => {
                report_avatar_failure(&error);
                None
            }
        };
        // The drape worker, before the window: `EngineBridge::new` spawns onto a
        // runtime, and a runtime built after the first `view()` would have
        // nowhere to put a submission made from the toolbar.
        let drape = match drape::Drape::new() {
            Ok(drape) => {
                stages.push("drape worker");
                Some(drape)
            }
            Err(error) => {
                tracing::error!(target: "nguo_app", %error, "no drape worker; simulation disabled");
                None
            }
        };

        let mut camera = Camera::default();
        let mut canvas_view = nguo_canvas::View::default();
        if let Some(avatars) = &avatars {
            catalogue.extend(avatars.command_entries());
            catalogue.sort_by(|a, b| a.label.cmp(&b.label));
            camera.frame_standing(avatars.selected().stature_m as f32);
            canvas_view.zoom = avatar::fit_zoom(avatars.selected().stature_m);
        }

        Self {
            phase: Phase::Booting(Box::new(Splash::new(Instant::now()))),
            stages,
            doc_label: format!("{} · {} pieces", doc.meta().name, doc.pieces().len()),
            focus_label: focus_name(Context::Canvas2d).to_owned(),
            process_label: "no active process".to_owned(),
            fps_label: "— fps".to_owned(),
            drape_label: drape_label(drape.as_ref()),
            cursor_label: "—".to_owned(),
            doc,
            stack: nguo_canvas::CommandStack::new(),
            scene_rows,
            layout,
            tree: TreeState::expanded(["garment", "piece/0"]),
            keymap,
            resolver: Resolver::new(),
            focus: Context::Canvas2d,
            active_tool: None,
            palette: None,
            palette_hits: Vec::new(),
            catalogue,
            registry: Snapshot::default(),
            dismissed: BTreeSet::new(),
            simulating: false,
            avatars,
            drape,
            camera,
            canvas_view,
            canvas2d: std::sync::Mutex::new(nguo_canvas::Canvas2d::new()),
            canvas_pick: None,
            viewport_pick: None,
            stats: FrameStats::default(),
            started: Instant::now(),
            time: 0.0,
            adapter: nguo_gpu::adapter_description(),
            tokens,
            paper,
        }
    }

    /// Logical size of the window, from the compositor. `None` before the
    /// first frame, when nothing has been laid out yet anyway.
    fn logical_width(&self) -> Option<f32> {
        let (w, _, scale) = nguo_gpu::window_geometry();
        (w > 0 && scale > 0.0).then(|| w as f32 / scale)
    }

    /// What the status bar says about the solver. Read off the drape rather than
    /// off the toolbar toggle, so it cannot claim to be simulating when nothing
    /// is (`sim.toggle` is a request; this is the fact).
    fn solver_state(&self) -> &str {
        &self.drape_label
    }
}

/// Surface an avatar failure as a toast, without a bespoke notification list.
///
/// The registry already remembers failures with their whole error chain and
/// `toasts` already renders them (OBSERVABILITY.md), so a typed `AvatarError`
/// goes in one line and comes out as a dismissable card titled "3D drape
/// viewport" with the actionable root cause under it. Never fatal: every
/// caller carries on with `avatars: None`.
fn report_avatar_failure(error: &nguo_avatar::AvatarError) {
    tracing::warn!(target: "nguo_app", %error, "avatar unavailable");
    let _ = nguo_registry::start(nguo_registry::feature_const("viewport.render"), 0).failed(error);
}

/// One line about the drape, for the status bar.
fn drape_label(drape: Option<&drape::Drape>) -> String {
    match drape {
        None => "no solver".to_owned(),
        Some(drape) => drape.report.line(),
    }
}

fn focus_name(focus: Context) -> &'static str {
    match focus {
        Context::Viewport3d => "focus: 3D garment",
        Context::TextEntry => "focus: text",
        Context::Modal => "focus: modal",
        _ => "focus: 2D pattern",
    }
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

fn update(app: &mut App, message: Message) -> Task<Message> {
    match message {
        Message::Tick(now) => {
            app.stats.tick(now);
            app.time = app.started.elapsed().as_secs_f32();
            // One short lock per frame, nothing borrowed from it afterwards
            // (OBSERVABILITY.md §2.4 — 112 ns measured). The lock is released
            // before `view` is ever called.
            app.registry = nguo_registry::snapshot();
            app.process_label = process_label(&app.registry);
            app.focus_label = focus_name(app.focus).to_owned();
            app.fps_label = format!(
                "{:.0} fps · {:.2} ms",
                app.stats.fps(),
                app.stats.frame_ms()
            );
            // Drain the solver before the frame that will draw it. The redraw is
            // already scheduled — `frames()` gave us this tick — so a new frame
            // needs no extra request.
            if let Some(drape) = &mut app.drape {
                drape.poll();
            }
            app.drape_label = drape_label(app.drape.as_ref());

            let dismissed = if let Phase::Booting(splash) = &mut app.phase {
                boot_progress(splash, &app.stages);
                splash.tick(now)
            } else {
                false
            };

            // The canvas floor is re-asserted every frame because the window
            // can be resized without any dock event at all.
            if let Some(width) = app.logical_width() {
                shell::enforce_canvas_minimum(&mut app.layout, width);
            }

            let mut tasks = Vec::new();
            if dismissed {
                tracing::info!(target: "nguo_app", "splash dismissed, shell up");
                tasks.push(promote_to_shell(app));
            }
            if let Some(capture) = capture(app.stats.frames, matches!(app.phase, Phase::Running)) {
                tasks.push(capture);
            }
            return Task::batch(tasks);
        }

        Message::Captured(tag, shot) => {
            tracing::info!(
                target: "nguo_app",
                tag,
                physical_w = shot.size.width,
                physical_h = shot.size.height,
                scale_factor = shot.scale_factor,
                logical_w = shot.size.width as f32 / shot.scale_factor,
                logical_h = shot.size.height as f32 / shot.scale_factor,
                rate = %app.fps_label,
                "framebuffer captured"
            );
            if let Some(dir) = std::env::var_os("NGUO_CAPTURE") {
                let path = std::path::Path::new(&dir).join(format!(
                    "{tag}-{}x{}.rgba",
                    shot.size.width, shot.size.height
                ));
                if let Err(error) = std::fs::write(&path, &shot.rgba) {
                    tracing::error!(target: "nguo_app", %error, path = %path.display(), "capture failed");
                }
            }
        }

        Message::Event(event) => return on_event(app, event),

        Message::Dock(event) => app.layout.update(event),

        Message::Tree(event) => app.tree.update(event),

        Message::Command(id) => return dispatch(app, id),

        Message::PaletteQuery(query) => {
            app.palette_hits = command::search(&app.catalogue, &query)
                .into_iter()
                .take(40)
                .cloned()
                .collect();
            app.palette = Some(query);
        }

        Message::PaletteAccept => {
            if let Some(first) = app.palette_hits.first().map(|e| e.id.clone()) {
                close_palette(app);
                return dispatch(app, first);
            }
        }

        Message::DismissToast(id) => {
            app.dismissed.insert(id);
        }

        Message::Viewport(action) => match action {
            nguo_viewport3d::Action::Orbit { yaw, pitch } => app.camera.orbit(yaw, pitch),
            nguo_viewport3d::Action::Dolly { factor } => app.camera.dolly(factor),
            nguo_viewport3d::Action::Picked { world, screen } => {
                app.focus = Context::Viewport3d;
                app.viewport_pick = Some(world);
                let (_, _, scale) = nguo_gpu::window_geometry();
                tracing::info!(
                    target: "nguo_app",
                    widget = "viewport3d",
                    logical_x = screen.x,
                    logical_y = screen.y,
                    world_x = world.x,
                    world_y = world.y,
                    world_z = world.z,
                    scale_factor = scale,
                    "pointer picked the ground plane"
                );
            }
        },

        Message::Canvas(action) => match action {
            nguo_canvas::Action::Pan { delta } => app.canvas_view.pan_by_screen(delta),
            nguo_canvas::Action::Zoom {
                factor,
                screen,
                size,
            } => app.canvas_view.zoom_about(factor, screen, size),
            nguo_canvas::Action::Pointer(pointer) => return on_canvas_pointer(app, pointer),
        },
    }

    Task::none()
}

/// The 2D authoring loop: pointer in, transaction out, document changed.
///
/// This is the whole of it. `Canvas2d` owns the tool state and cannot mutate
/// the document; `CommandStack` is the only thing that can; `applied` tells the
/// piece cache which scenes to re-encode. Forgetting that last call is slow
/// (a full rebuild) and never stale — `SceneCache` counts revision ticks
/// against the marks it was handed, so a caller who forgets loses the caching
/// rather than rendering the wrong thing.
/// Re-derive the chrome that reads off the document: the scene tree and the
/// title-bar label. Every path that changes `App::doc` calls this, so the two
/// cannot drift apart.
fn refresh_doc_chrome(app: &mut App) {
    app.scene_rows = shell::scene_tree(&app.doc);
    app.doc_label = format!(
        "{} · {} pieces",
        app.doc.meta().name,
        app.doc.pieces().len()
    );
}

fn on_canvas_pointer(app: &mut App, pointer: nguo_canvas::Pointer) -> Task<Message> {
    app.focus = Context::Canvas2d;
    let tool = canvas_tool(app.active_tool.as_ref());

    // `get_mut`, not `lock`: `update` has `&mut App`, so there is no contention
    // to resolve and no poisoning to handle.
    let view = app.canvas_view;
    let Ok(canvas) = app.canvas2d.get_mut() else {
        // A poisoned lock means a previous `draw` panicked. Dropping input for
        // this frame beats taking the shell down.
        return Task::none();
    };
    // `App::canvas_view` is the single source of truth and `draw` copies it in —
    // but iced drains every queued message through `update` before the next
    // `draw`, so a wheel-zoom and a pointer move batched into one frame would
    // hit-test against the PRE-zoom view. Sync here too, where it is read.
    // Same reason the opening zoom (`App::new`) reaches the canvas at all.
    canvas.view = view;

    if let Some(tx) = canvas.pointer(&app.doc, tool, pointer) {
        canvas.applied(&tx);
        if let Err(error) = app.stack.apply(&mut app.doc, tx) {
            tracing::warn!(target: "nguo_app", %error, "the canvas produced an edit the document refused");
        }
        refresh_doc_chrome(app);
    }

    let canvas = app.canvas2d.get_mut().expect("checked above");
    // Only a press moves the crosshair. It is the HiDPI hit-test proof, not a
    // cursor follower — the overlay's snap diamond already tracks the pointer,
    // and two marks chasing each other is noise.
    if pointer.phase == nguo_canvas::Phase::Pressed {
        app.canvas_pick = canvas.cursor;
    }
    app.cursor_label = cursor_label(canvas);
    Task::none()
}

/// The active toolbar button, as something the canvas understands. The canvas
/// knows nothing about `CommandId`, and this is the one place the mapping lives.
///
/// ponytail: only the three drawing tools and select are wired. The rest of the
/// toolbar still logs "no handler yet" — mapping a button to a tool that does
/// not exist would be worse than leaving it visibly unimplemented.
fn canvas_tool(active: Option<&CommandId>) -> nguo_canvas::Tool {
    use nguo_doc::pattern::EdgeSemantic;
    match active.map(CommandId::as_str) {
        Some("tool.polygon") => nguo_canvas::Tool::Draw(EdgeSemantic::Seam),
        Some("tool.internal_line") => nguo_canvas::Tool::Draw(EdgeSemantic::Internal),
        Some("tool.dart") => nguo_canvas::Tool::Draw(EdgeSemantic::DartLeg),
        _ => nguo_canvas::Tool::Select,
    }
}

/// Cursor position in mm, plus the live length of the pen's rubber band.
fn cursor_label(canvas: &nguo_canvas::Canvas2d) -> String {
    let Some(at) = canvas.cursor else {
        return "—".to_owned();
    };
    match canvas.interaction.rubber_band {
        Some((from, to)) => format!(
            "{:.1}, {:.1} mm · {:.1} mm",
            at.x,
            at.y,
            (to - from).hypot()
        ),
        None => format!("{:.1}, {:.1} mm", at.x, at.y),
    }
}

/// Report only milestones that have actually completed.
///
/// `stages` is appended to by `App::new` as each one really finishes, and
/// `Splash::report` is monotonic, so replaying the whole list every tick is
/// safe and cannot show progress the app has not made.
///
/// **The shared GPU device is deliberately not a stage here.** `nguo_gpu::adopt`
/// runs inside the first shader primitive's `prepare`, and no shader widget is
/// in the tree while the splash is up — so gating dismissal on
/// `nguo_gpu::shared().is_some()` is a deadlock, which is exactly what the
/// first run of this shell did. Being *in* this function is instead the
/// evidence for the last stage: `iced::window::frames()` only fires after a
/// composited frame, so surface, renderer and font system are all up.
fn boot_progress(splash: &mut Splash, stages: &[&'static str]) {
    let total = stages.len() + 1;
    for (i, stage) in stages.iter().enumerate() {
        splash.report(*stage, (i + 1) as f32 / total as f32);
    }
    splash.report("renderer", 1.0);
    splash.loaded();
}

/// Grow the splash panel into the workspace window.
///
/// **One window, not two.** `iced::application`'s `view` is handed a window id
/// and throws it away (`iced-0.14.0/src/application.rs:137-143`), so a second
/// window would draw the *same* element tree; genuine multi-window means
/// `iced::daemon`, which the single-window-with-docking decision rules out.
/// It is also
/// unnecessary: every window knob set on the builder has a runtime counterpart
/// in `iced_runtime-0.14.0/src/window.rs`, all of them implemented in
/// `iced_winit-0.14.0/src/lib.rs` — `SetResizable:1498`,
/// `ToggleDecorations:1592`, `Resize:1446`, `GetMonitorSize:1664`. So the one
/// window simply grows.
fn promote_to_shell(app: &mut App) -> Task<Message> {
    app.phase = Phase::Running;

    // `NGUO_AUTODRAPE=1`: submit the opening document's drape as soon as the shell
    // is up, exactly as pressing Space would. It exists because there is no way to
    // press Space from a script on this machine — `osascript` has no assistive
    // access — and a capture run has to be able to photograph a *drape*, not an
    // idle window. It routes through `dispatch`, so it cannot drift from what the
    // key does.
    let autodrape = if std::env::var_os("NGUO_AUTODRAPE").is_some() {
        tracing::info!(target: "nguo_app", "NGUO_AUTODRAPE: submitting the opening drape");
        dispatch(app, CommandId::from("sim.toggle"))
    } else {
        Task::none()
    };

    Task::batch([
        autodrape,
        iced::window::latest().and_then(|id| {
            Task::batch([
                // A toggle, not a setter, in 0.14 — safe because this runs exactly
                // once: `Phase` never goes back to `Booting`.
                iced::window::toggle_decorations(id),
                iced::window::set_resizable(id, true),
                iced::window::set_level(id, iced::window::Level::Normal),
                iced::window::resize(id, SHELL_SIZE),
                // winit keeps the top-left corner where it was, so a panel centred
                // at 900x534 would hang off the bottom-right at 1440x900.
                iced::window::monitor_size(id).and_then(move |monitor| {
                    iced::window::move_to(
                        id,
                        iced::Point::new(
                            ((monitor.width - SHELL_SIZE.width) / 2.0).max(0.0),
                            ((monitor.height - SHELL_SIZE.height) / 2.0).max(0.0),
                        ),
                    )
                }),
            ])
        }),
    ])
}

/// Capture the app's own framebuffer at two fixed frames — one with the splash
/// up, one with the shell up — when `NGUO_CAPTURE` names a directory.
///
/// `screencapture` loses the race against whatever else owns the front of the
/// screen; `iced::window::screenshot` reads our surface directly and cannot
/// (SPLASH.md §5). Off by default, so it costs one integer compare per frame.
///
/// `NGUO_CAPTURE_FRAMES` is a comma-separated frame list, defaulting to
/// `120,1000` — the splash and the settled shell. The frame *list* rather than two
/// constants is now earned rather than speculative: a drape takes seconds and
/// looking at it needs several frames across that span, which is what
/// `--features debug-overlay` runs use.
fn capture(frames: u64, running: bool) -> Option<Task<Message>> {
    std::env::var_os("NGUO_CAPTURE")?;
    let wanted: Vec<u64> = match std::env::var("NGUO_CAPTURE_FRAMES") {
        Ok(list) => list
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
        Err(_) => vec![120, 1000],
    };
    if !wanted.contains(&frames) {
        return None;
    }
    // The splash holds for 4 s; the phase, not the frame number, decides which
    // is which, because the frame rate is not 60 Hz here.
    let tag: &'static str = if running {
        // Leaked deliberately and exactly once per captured frame: `Message` is
        // `Clone` and iced requires `'static`, and a handful of tags over a
        // process lifetime is not a leak that matters.
        Box::leak(format!("shell-{frames}").into_boxed_str())
    } else {
        "splash"
    };
    Some(
        iced::window::latest()
            .and_then(iced::window::screenshot)
            .map(move |shot| Message::Captured(tag, shot)),
    )
}

fn on_event(app: &mut App, event: iced::Event) -> Task<Message> {
    match event {
        // While a divider is grabbed the dock needs the window-level cursor:
        // the pointer leaves the 6 px grab strip on the first pixel of motion,
        // so the view can report grab and release but never the drag itself.
        iced::Event::Mouse(iced::mouse::Event::CursorMoved { position }) => {
            if app.layout.is_dragging() {
                app.layout.drag_to((position.x, position.y));
                if let Some(width) = app.logical_width() {
                    shell::enforce_canvas_minimum(&mut app.layout, width);
                }
            }
        }

        iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
            // `key`, never `modified_key` — KEYMAP.md §7. On macOS ⌥7 arrives
            // as `modified_key = ¶` and every ⌥ binding would break.
            ref key,
            modifiers,
            ..
        }) => {
            let Some(chord) = command::chord_from_key(key, modifiers) else {
                return Task::none();
            };
            let focus = if app.palette.is_some() {
                Context::TextEntry
            } else {
                app.focus
            };

            match app.resolver.press(&app.keymap, focus, chord) {
                Resolution::Command(id) => return dispatch(app, id),
                // A live prefix (`K`); the status bar shows the hint.
                Resolution::Pending => {}
                // The focused widget owns this chord. Do nothing at all — not
                // even a log line — and let iced deliver it.
                Resolution::Swallowed => {}
                Resolution::Unbound => {}
            }
        }

        _ => {}
    }

    Task::none()
}

/// The `CommandId` -> handler table (KEYMAP.md §9).
///
/// A `match`, not a map: the arms *are* the table, and an id from a settings
/// file written by a newer build lands in `_` — logged and ignored, never
/// fatal.
fn dispatch(app: &mut App, id: CommandId) -> Task<Message> {
    match id.as_str() {
        "agent.palette" => {
            app.palette = Some(String::new());
            app.palette_hits = app.catalogue.iter().take(40).cloned().collect();
            return iced::widget::operation::focus("palette");
        }
        "ui.cancel" => {
            if app.palette.is_some() {
                close_palette(app);
            }
            if let Ok(canvas) = app.canvas2d.get_mut() {
                // Ends a pen run, leaving it open. The points it placed stay;
                // undo is what removes them.
                canvas.cancel();
            }
            app.resolver.cancel();
        }
        // Undo covers every canvas tool, because every tool goes through the
        // stack. `CommandStack::undo` returns a `bool` and not an edit list, so
        // the piece cache is not told what changed — `SceneCache` notices the
        // revision moved further than its marks explain and rebuilds the lot,
        // which is slow for one frame and never stale.
        "edit.undo" | "edit.redo" => {
            // End any pen run FIRST. `PenState::Drawing { piece }` names a
            // piece undo may be about to remove, and a pen pointing at a gone
            // piece wedges the draw tool silently — every later click resolves
            // `doc.piece(piece)` to an error the canvas swallows, with no log
            // and no visible cause, until the user happens to press Esc.
            if let Ok(canvas) = app.canvas2d.get_mut() {
                canvas.cancel();
            }
            let moved = if id.as_str() == "edit.undo" {
                app.stack.undo(&mut app.doc)
            } else {
                app.stack.redo(&mut app.doc)
            };
            match moved {
                Ok(true) => refresh_doc_chrome(app),
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(target: "nguo_app", %error, command = id.as_str(), "history step failed")
                }
            }
        }
        "app.quit" | "window.close" => return iced::exit(),
        "view.cycle_pane_focus" => {
            app.focus = match app.focus {
                Context::Viewport3d => Context::Canvas2d,
                _ => Context::Viewport3d,
            };
        }
        // The one command that makes M6 happen: sew the document and drape it.
        // A second press cancels, because a 30-second bake with no way to stop it
        // is a hung application.
        "sim.toggle" => {
            app.simulating = !app.simulating;
            if let Some(drape) = &mut app.drape {
                if app.simulating {
                    let (arrange, proxy) = drape_inputs(&mut app.avatars, &app.doc);
                    drape.submit(&app.doc, arrange, proxy.as_deref(), Tier::Preview);
                } else {
                    drape.cancel();
                }
            }
        }
        "panel.library" => app
            .layout
            .update(DockEvent::CollapseToggled(shell::SCENE.into())),
        "agent.panel" => app
            .layout
            .update(DockEvent::CollapseToggled(shell::PROPERTIES.into())),
        "view.zoom_extents_all" => {
            app.canvas_view = nguo_canvas::View {
                // Back to the framing the document opened at, avatar and all
                // — not to a bare `zoom: 1.0`, which would leave a 1 900 mm
                // ghost filling the pane.
                zoom: app
                    .avatars
                    .as_ref()
                    .map_or(1.0, |a| avatar::fit_zoom(a.selected().stature_m)),
                ..nguo_canvas::View::default()
            };
        }
        other
            if other.starts_with(avatar::BODY_COMMAND)
                || other.starts_with(avatar::TONE_COMMAND) =>
        {
            select_avatar(app, other);
        }
        other if other.starts_with("tool.") => app.active_tool = Some(id.clone()),
        other => {
            // Not wired yet, or not ours. Either way: a log line, not a panic.
            tracing::info!(target: "nguo_app", command = other, "command has no handler yet");
        }
    }
    Task::none()
}

/// The two things a drape needs off the avatar: how to arrange the panels around
/// it, and the collision proxy to drape over.
///
/// Both degrade rather than fail. No bundle, or a proxy that will not load, means
/// `nguo_sewing`'s own defaults and a gravity-only drape — reported in the status
/// line, never pretended away.
fn drape_inputs(
    avatars: &mut Option<avatar::Avatars>,
    doc: &nguo_doc::Document,
) -> (
    nguo_sewing::ArrangeOptions,
    Option<std::sync::Arc<avatar::CollisionProxy>>,
) {
    let Some(avatars) = avatars.as_mut() else {
        return (nguo_sewing::ArrangeOptions::default(), None);
    };
    // The document decides which landmark it hangs at; the body decides where
    // that landmark is.
    let arrange = avatars.arrange_options(&nguo_sewing::hang(doc));
    let proxy = match avatars.collider() {
        Ok(proxy) => Some(proxy),
        Err(error) => {
            report_avatar_failure(&error);
            None
        }
    };
    (arrange, proxy)
}

/// `avatar.body.{male,female}` / `avatar.tone.MST01..MST10`.
///
/// Both suffixes come from the bundle itself, so a miss here means a stale
/// settings file or a bundle that changed under us — a toast, not a panic, same
/// as any other unknown command.
fn select_avatar(app: &mut App, command: &str) {
    let Some(avatars) = app.avatars.as_mut() else {
        tracing::info!(target: "nguo_app", command, "no avatar bundle; ignoring");
        return;
    };

    let result = match command.strip_prefix(avatar::BODY_COMMAND) {
        Some(key) => avatars.select(key).map(|()| {
            // Re-frame: the female sculpt is 7.5 cm taller than the male one
            // (1.90477 m vs 1.830 m — the pipeline never rescales).
            app.camera
                .frame_standing(avatars.selected().stature_m as f32);
        }),
        None => avatars.set_tone(command.trim_start_matches(avatar::TONE_COMMAND)),
    };
    if let Err(error) = result {
        report_avatar_failure(&error);
    }
}

fn close_palette(app: &mut App) {
    app.palette = None;
    app.palette_hits.clear();
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

fn view(app: &App) -> Element<'_, Message> {
    if let Phase::Booting(splash) = &app.phase {
        return splash.view();
    }

    let t = &app.tokens;
    let workspace = column![
        toolbar(app),
        w::dock::dock(&app.layout, t, Message::Dock, |panel, tab| panel_body(
            app, panel, tab
        )),
        status_bar(app),
    ];

    let mut layers: Vec<Element<'_, Message>> = vec![container(workspace)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(w::hairline(t, t.color.background))
        .into()];

    if let Some(toasts) = toasts(app) {
        layers.push(toasts);
    }
    if let Some(palette) = palette(app) {
        layers.push(palette);
    }

    #[cfg(feature = "debug-overlay")]
    layers.push(debug_overlay(app));

    stack(layers).into()
}

fn toolbar(app: &App) -> Element<'_, Message> {
    let t = &app.tokens;
    let items = TOOLBAR
        .iter()
        .map(|entry| match entry {
            None => w::toolbar_separator(t, Axis::Horizontal),
            Some((glyph, id)) => {
                let entry = app.catalogue.iter().find(|e| e.id.as_str() == *id);
                let label = entry.map_or(*id, |e| e.label.as_str());
                let shortcut = entry.and_then(|e| e.shortcut.as_deref());
                let active = app.active_tool.as_ref().is_some_and(|a| a.as_str() == *id);
                w::tool_button(
                    t,
                    *glyph,
                    label,
                    shortcut,
                    active,
                    Some(Message::Command(CommandId::from(*id))),
                )
            }
        })
        .collect();

    w::toolbar(t, Axis::Horizontal, items)
}

fn panel_body<'a>(app: &'a App, _panel: &Panel, tab: &Tab) -> Element<'a, Message> {
    let t = &app.tokens;
    match tab.id.as_str() {
        shell::SCENE => w::tree(t, &app.scene_rows, &app.tree, Message::Tree),

        shell::CANVAS => Element::from(
            iced::widget::shader(PatternCanvas {
                inner: Canvas {
                    view: app.canvas_view,
                    pick: app.canvas_pick,
                },
                paper: app.paper.color.canvas,
                doc: &app.doc,
                tokens: &app.paper,
                canvas2d: &app.canvas2d,
                // A ghosted body behind the draft (UX_SPEC §2).
                // `border_strong` and not `guide`: the canvas entity roles are
                // the chromatic tokens (UX_SPEC §2) and a violet body would
                // read as a construction line. This is neutral paper furniture.
                ghost: app.avatars.as_ref().map(|a| {
                    (
                        a.silhouette(),
                        app.paper.color.border_strong.alpha(GHOST_ALPHA),
                    )
                }),
            })
            .width(Length::Fill)
            .height(Length::Fill),
        )
        .map(Message::Canvas),

        shell::VIEWPORT => Element::from(
            iced::widget::shader(Viewport {
                camera: app.camera,
                time: app.time,
                avatar: app.avatars.as_ref().map(|a| a.viewport()),
                garment: app.drape.as_ref().and_then(drape::Drape::viewport),
            })
            .width(Length::Fill)
            .height(Length::Fill),
        )
        .map(Message::Viewport),

        "props" => properties(app),

        "params" => w::empty_state(
            t,
            Icon::Sliders,
            "No parameters yet",
            "A measurement set is a document object (UX_SPEC §3.2). Create one \
             and every dimension in the pattern becomes an input.",
            None,
        ),

        _ => space::horizontal().into(),
    }
}

fn properties(app: &App) -> Element<'_, Message> {
    let t = &app.tokens;
    let selected: Vec<&str> = app.tree.selection().collect();
    let mut rows = vec![w::section_header(t, "Selection", Some(Icon::Layers))];

    if selected.is_empty() {
        rows.push(w::property_row(
            t,
            "Nothing",
            text("Select a piece or edge")
                .size(t.text.small)
                .color(w::color(t.color.text_muted)),
        ));
    } else {
        for id in &selected {
            rows.push(w::property_row(
                t,
                "Entity",
                text(*id)
                    .size(t.text.small)
                    .color(w::color(t.color.text_secondary)),
            ));
        }
    }

    rows.extend(seam_rows(app));
    rows.extend(drape_rows(app));
    rows.extend(avatar_picker(app));

    rows.push(w::section_header(t, "Document", Some(Icon::Garment)));
    for (label, value) in [
        ("Name", app.doc.meta().name.clone()),
        ("Pieces", app.doc.pieces().len().to_string()),
        ("Seams", app.doc.seams().len().to_string()),
        ("Units", format!("{:?}", app.doc.display_unit())),
        ("Revision", app.doc.revision().to_string()),
    ] {
        rows.push(w::property_row(
            t,
            label,
            text(value)
                .size(t.text.small)
                .color(w::color(t.color.text_primary)),
        ));
    }

    w::inspector(t, rows)
}

/// What the last drape produced. Empty until one has been submitted — a panel of
/// zeroes would read as a garment with no geometry.
///
/// Every number here is measured by the code that did the work
/// (`nguo_sewing::Timings`, and the engine's own `median_step_s`), never derived
/// or estimated here.
fn drape_rows(app: &App) -> Vec<Element<'_, Message>> {
    let t = &app.tokens;
    let Some(report) = app.drape.as_ref().map(|d| &d.report) else {
        return Vec::new();
    };
    if report.message.is_empty() {
        return Vec::new();
    }

    let mut rows = vec![w::section_header(t, "Drape", Some(Icon::Play))];
    let sew = &report.sew;
    for (label, value) in [
        ("Vertices", report.vertices.to_string()),
        ("Triangles", report.triangles.to_string()),
        ("Stitches", report.stitches.to_string()),
        (
            "Seams open",
            format!("{:.1} mm", report.seam_gap_m * 1000.0),
        ),
        ("Frames", report.frames.to_string()),
        (
            "Sew",
            format!(
                "{:.1} ms (tri {:.1} · sew {:.1})",
                sew.total_ms(),
                sew.triangulate_ms,
                sew.sew_ms
            ),
        ),
        (
            "Solver",
            match report.step_s {
                Some(s) => format!("{:.0} ms/step", s * 1000.0),
                None => "running…".to_owned(),
            },
        ),
    ] {
        rows.push(w::property_row(
            t,
            label,
            text(value)
                .size(t.text.small)
                .color(w::color(t.color.text_primary)),
        ));
    }
    rows
}

/// Seam health, recomputed from the document every time this panel is drawn.
///
/// `ui/docs/UX_SPEC.md` §1: seam validation is a **maintained constraint**, not a
/// command. There is deliberately no "Check Sewing Length" menu item — the panel
/// simply always tells the truth about the document in front of you, so an unsewn
/// edge or a 2× length mismatch is visible while it is being drafted rather than
/// discovered by pressing Space and being refused a drape.
///
/// Recomputed rather than cached against `Document::revision`. That is the lazy
/// choice *and* the correct one: `nguo_sewing::validate` is a pure function of the
/// document measured at **1.1 µs/call** with 4 diagnostics
/// (`nguo_sewing::validate::tests::validation_is_cheap_enough_to_run_on_every_revision`,
/// which prints the number to fd 2), i.e. 0.013 % of the 8.3 ms frame budget. A
/// revision cache would be more code and one more thing that can go stale — the
/// exact failure mode a maintained constraint exists to prevent.
/// The seam status line. Split out from [`seam_rows`] because the panel's
/// verdict is the part worth testing and an `Element` cannot be inspected —
/// the colour beside it branches on the same two counts.
///
/// "to fix before draping" rather than "invalid": an error here is actionable and
/// never fatal, and the per-diagnostic rows below say what to do.
fn seam_status(seams: usize, errors: usize, warnings: usize) -> String {
    let verdict = if errors > 0 {
        format!("{errors} to fix before draping")
    } else if warnings > 0 {
        format!("{warnings} to check")
    } else {
        "ready to drape".to_owned()
    };
    format!("{seams} sewn · {verdict}")
}

/// How many diagnostic rows the panel shows before it starts counting the rest.
/// A draft mid-edit can earn dozens of "unsewn edge" warnings and a 40-row panel
/// is one nobody reads.
const MAX_SEAM_ROWS: usize = 6;

/// The diagnostics the panel shows, most actionable first.
///
/// The ordering is load-bearing, not cosmetic: [`seam_rows`] shows only the first
/// [`MAX_SEAM_ROWS`], so an error that sorts below a pile of "unsewn edge"
/// warnings is an error nobody sees. `Severity` orders `Error < Warning < Info`
/// and the sort is **stable**, so this lifts the actionable rows while keeping
/// `validate`'s document order within each severity — a list that reshuffles
/// frame to frame is unreadable in a panel.
///
/// A seam whose two sides match to the displayed precision is dropped: its row
/// reads "the two sides differ by +0.0 % of their length", which is four lines of
/// panel saying nothing, and on the shipped two-seam sample it filled a third of
/// the cap. `validate` still reports it — this is the *panel's* editorial choice,
/// not a change to the constraint, so anything else consuming the diagnostics
/// still sees the complete set. Ease worth naming still shows, because the
/// threshold is exactly the point at which the sentence stops reading `+0.0 %`.
fn seam_diagnostics(doc: &nguo_doc::Document) -> Vec<nguo_sewing::Diagnostic> {
    /// `{:+.1} %` of `(ratio - 1) * 100` renders as `0.0` below this.
    const SHOWS_AS_ZERO: f64 = 0.000_5;
    let mut diagnostics = nguo_sewing::validate(doc);
    diagnostics.retain(|d| match d.kind {
        nguo_sewing::Kind::Ease { ratio, .. } => (ratio - 1.0).abs() >= SHOWS_AS_ZERO,
        _ => true,
    });
    diagnostics.sort_by_key(|d| d.severity);
    diagnostics
}

fn seam_rows(app: &App) -> Vec<Element<'_, Message>> {
    let t = &app.tokens;
    if app.doc.pieces().is_empty() {
        return Vec::new();
    }

    let diagnostics = seam_diagnostics(&app.doc);
    let (errors, warnings, _) = nguo_sewing::counts(&diagnostics);

    let mut rows = vec![w::section_header(t, "Seams", Some(Icon::Seam))];
    rows.push(w::property_row(
        t,
        "Status",
        text(seam_status(app.doc.seams().len(), errors, warnings))
            .size(t.text.small)
            .color(w::color(if errors > 0 {
                t.color.error
            } else if warnings > 0 {
                t.color.warning
            } else {
                t.color.success
            })),
    ));

    // Capped, with the remainder counted rather than dropped silently.
    for d in diagnostics.iter().take(MAX_SEAM_ROWS) {
        let (label, tone) = match d.severity {
            nguo_sewing::Severity::Error => ("Error", t.color.error),
            nguo_sewing::Severity::Warning => ("Warning", t.color.warning),
            nguo_sewing::Severity::Info => ("Ease", t.color.text_secondary),
        };
        rows.push(w::property_row(
            t,
            label,
            text(d.message()).size(t.text.small).color(w::color(tone)),
        ));
    }
    if let Some(more) = diagnostics
        .len()
        .checked_sub(MAX_SEAM_ROWS)
        .filter(|n| *n > 0)
    {
        rows.push(w::property_row(
            t,
            "",
            text(format!("…and {more} more"))
                .size(t.text.small)
                .color(w::color(t.color.text_muted)),
        ));
    }
    rows
}

/// Body and skin-tone pickers (UX_SPEC §3.2). Every control emits the same
/// `CommandId` the palette does — one registry, two surfaces, no second
/// handler.
///
/// Empty when there is no bundle: the toast already says why, and a picker
/// with nothing to pick is worse than no picker.
fn avatar_picker(app: &App) -> Vec<Element<'_, Message>> {
    let t = &app.tokens;
    let Some(avatars) = app.avatars.as_ref() else {
        return Vec::new();
    };
    let selected = avatars.selected();

    let mut rows = vec![w::section_header(t, "Avatar", Some(Icon::Box3d))];

    for (key, entry) in avatars.entries() {
        let active = key == avatars.selected_key();
        rows.push(
            button(
                text(short_label(&entry.name))
                    .size(t.text.small)
                    .width(Length::Fill),
            )
            .width(Length::Fill)
            .padding([t.space.xs, t.space.sm])
            .style(move |_theme: &iced::Theme, status| picker_style(t, active, status))
            .on_press(Message::Command(CommandId::from(format!(
                "{}{key}",
                avatar::BODY_COMMAND,
            ))))
            .into(),
        );
    }

    rows.push(w::property_row(
        t,
        "Stature",
        text(format!("{:.1} cm", selected.stature_m * 100.0))
            .size(t.text.small)
            .color(w::color(t.color.text_primary)),
    ));

    rows.push(w::section_header(t, "Skin tone", Some(Icon::Palette)));
    let swatches: Vec<Element<'_, Message>> = avatars
        .tones()
        .iter()
        .map(|tone| {
            let active = tone.key == avatars.tone();
            // Not a colour literal: this is the Monk scale's own published
            // sRGB, read out of `data/skin_tones.json`. Its *linear* sibling
            // and the `baseColorFactor` never come near the UI.
            let [r, g, b] = tone.swatch.srgb;
            let fill = nguo_ui_theme::Rgba::new(r, g, b, 1.0);
            button(space::horizontal())
                .width(Length::Fixed(SWATCH))
                .height(Length::Fixed(SWATCH))
                .padding(0)
                .style(
                    move |_theme: &iced::Theme, _status| iced::widget::button::Style {
                        background: Some(w::color(fill).into()),
                        border: iced::Border {
                            color: w::color(if active {
                                t.color.accent
                            } else {
                                t.color.border
                            }),
                            width: if active { 2.0 } else { t.metrics.border_width },
                            radius: t.radius.sm.into(),
                        },
                        ..iced::widget::button::Style::default()
                    },
                )
                .on_press(Message::Command(CommandId::from(format!(
                    "{}{}",
                    avatar::TONE_COMMAND,
                    tone.key
                ))))
                .into()
        })
        .collect();

    rows.push(
        container(row(swatches).spacing(t.space.xs).wrap())
            .padding([t.space.xs, 0.0])
            .into(),
    );
    rows.push(w::property_row(
        t,
        "Tone",
        text(avatars.tone_label())
            .size(t.text.small)
            .color(w::color(t.color.text_primary)),
    ));

    // The consolidated CC BY 4.0 notice: it discharges all three obligations at
    // once — Seede XR for the bodies, texturecan/cgbookcase for the skin maps,
    // Ellis Monk / Google for the complexion scale — and credits Mixamo
    // (`assets/avatars/LICENSE.md` §"Minimum notice to ship with the
    // application"). CC BY 4.0 §3(a)(1) also asks for a link to the licence,
    // hence the second line.
    //
    // ponytail: this is the only user-reachable place there is, because there
    // is still no about/credits screen. Move it there when one exists; do not
    // remove it from here until then.
    rows.push(w::section_header(t, "Credits", Some(Icon::Layers)));
    // One string, one source of truth: `ATTRIBUTION_NOTICE` carries every credit
    // AND the licence URL that CC BY 4.0 s3(a)(1)(C) asks for, so nothing is
    // appended here. It was two lines until the const was completed.
    rows.push(
        container(
            text(nguo_avatar::ATTRIBUTION_NOTICE)
                .size(t.text.micro)
                .color(w::color(t.color.text_muted)),
        )
        .padding([t.space.xs, t.space.sm])
        .into(),
    );

    rows
}

/// Side of a Monk-tone swatch, logical px. Ten of them plus gaps fit the
/// 18 %-wide properties panel at the shell's default 1440 px window.
const SWATCH: f32 = 18.0;

/// `"Nguo Female"` -> `"Female"`. The panel is ~260 px and the studio's name is
/// already on every row.
fn short_label(name: &str) -> &str {
    name.trim_start_matches("Nguo ")
}

fn picker_style(
    t: &Tokens,
    active: bool,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::widget::button::Status;
    let bg = match (active, status) {
        (true, _) => t.color.accent,
        (false, Status::Hovered | Status::Pressed) => t.color.overlay,
        (false, _) => nguo_ui_theme::Rgba::new(0.0, 0.0, 0.0, 0.0),
    };
    iced::widget::button::Style {
        background: Some(w::color(bg).into()),
        text_color: w::color(if active {
            t.color.on_accent
        } else {
            t.color.text_secondary
        }),
        border: iced::Border {
            color: w::color(if active {
                t.color.accent
            } else {
                t.color.border
            }),
            width: t.metrics.border_width,
            radius: t.radius.sm.into(),
        },
        ..iced::widget::button::Style::default()
    }
}

fn status_bar(app: &App) -> Element<'_, Message> {
    let t = &app.tokens;
    let pending = app
        .resolver
        .pending()
        .iter()
        .map(|c| c.display(nguo_keymap::Platform::MacOs))
        .collect::<String>();

    let left = vec![
        w::status_item(t, Some(Icon::Garment), &app.doc_label),
        w::status_item(t, Some(Icon::Layers), &app.focus_label),
        w::status_item(
            t,
            Some(Icon::Crosshair),
            if pending.is_empty() {
                "ready"
            } else {
                "sequence"
            },
        ),
    ];

    let right = vec![
        w::status_item(t, Some(Icon::Measure), &app.cursor_label),
        w::status_item(t, Some(Icon::Refresh), &app.process_label),
        w::status_item(t, Some(Icon::Play), app.solver_state()),
        w::status_item(t, None, &app.fps_label),
    ];

    w::status_bar(t, left, right)
}

/// What the status bar says about the live process list. Pure so it can be
/// tested against a registry that actually has work in it — the shell binary
/// starts no processes of its own yet, and inventing one would be a lie.
fn process_label(snapshot: &Snapshot) -> String {
    match snapshot.active.first() {
        None => match snapshot.recent.first() {
            None => "no active process".to_owned(),
            Some(done) => format!("{} · {}", done.feature.name, done.outcome),
        },
        Some(p) => match p.progress {
            Some(f) => format!("{} · {} · {:.0}%", p.feature.name, p.state, f * 100.0),
            None => format!("{} · {}", p.feature.name, p.state),
        },
    }
}

/// Registry failures, as toasts (OBSERVABILITY.md; `FailureRecord::title` /
/// `detail` / `root_cause` map straight onto the widget).
fn toasts(app: &App) -> Option<Element<'_, Message>> {
    let t = &app.tokens;
    let live: Vec<_> = app
        .registry
        .failures
        .iter()
        .filter(|f| !app.dismissed.contains(&f.id.0))
        .take(4)
        .collect();
    if live.is_empty() {
        return None;
    }

    let items: Vec<Element<'_, Message>> = live
        .iter()
        .map(|f| {
            w::toast(
                t,
                w::ToastLevel::Error,
                f.title(),
                Some(f.root_cause()),
                Some(Message::DismissToast(f.id.0)),
            )
        })
        .collect();

    Some(
        container(column(items).spacing(t.space.sm))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_right(Length::Fill)
            .align_bottom(Length::Fill)
            // This layer is the whole window, status bar included, so the
            // bottom inset has to clear it or the toast sits on top of the
            // fps read-out. Only visible now that anything actually toasts.
            .padding(
                iced::Padding::from(t.space.lg).bottom(t.space.lg + t.metrics.status_bar_height),
            )
            .into(),
    )
}

fn palette(app: &App) -> Option<Element<'_, Message>> {
    let query = app.palette.as_ref()?;
    let t = &app.tokens;

    let field = text_input("Type a command…", query)
        .id("palette")
        .on_input(Message::PaletteQuery)
        .on_submit(Message::PaletteAccept)
        .size(t.text.body_large)
        .padding(t.space.md);

    let rows: Vec<Element<'_, Message>> = app
        .palette_hits
        .iter()
        .map(|e| {
            // Not `property_row`: that reserves a 96 px label gutter, which a
            // full-width command row does not want.
            container(
                row![
                    text(e.label.as_str())
                        .size(t.text.small)
                        .color(w::color(t.color.text_primary))
                        .width(Length::Fill),
                    text(e.shortcut.clone().unwrap_or_default())
                        .size(t.text.micro)
                        .color(w::color(t.color.text_muted)),
                ]
                .spacing(t.space.md)
                .align_y(iced::Center),
            )
            .padding([t.space.xs, t.space.sm])
            .width(Length::Fill)
            .into()
        })
        .collect();

    let card = container(column![field, w::inspector(t, rows)].spacing(t.space.sm))
        .width(Length::Fixed(560.0))
        .max_height(420.0)
        .padding(t.space.md)
        .style(w::overlay_surface(t));

    Some(
        container(card)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .padding(t.space.xxl)
            .into(),
    )
}

/// How much of the `border_strong` token the ghosted body gets. Enough to
/// place a piece against, faint enough that the draft stays the figure and
/// this stays the ground.
const GHOST_ALPHA: f32 = 0.28;

/// A light 2D canvas inside a dark shell (UX_SPEC §2).
///
/// `nguo_canvas::Canvas` hardcodes a dark base colour and belongs to another
/// crate, so the shell wraps it and overrides exactly one field of the
/// primitive. Everything else — pan, zoom, hit-testing, the scene — is the
/// canvas crate's, unchanged.
///
/// The avatar ghost is composited the same way: the canvas crate's scene is
/// appended *on top of* the silhouette, so the draft is never occluded and the
/// canvas crate needs to know nothing about avatars.
struct PatternCanvas<'a> {
    inner: Canvas,
    paper: nguo_ui_theme::Rgba,
    ghost: Option<(&'a vello::kurbo::BezPath, nguo_ui_theme::Rgba)>,
    /// The document to draw. `nguo_canvas::build_scene` — which is what
    /// `Canvas::draw` calls — renders a hardcoded `petal()` blob that its own
    /// docs describe as standing in for a pattern piece, so wiring the shell to
    /// it showed placeholder art in a pane labelled "2D Pattern" while the
    /// *same* document sewed and draped correctly in 3D. `Canvas2d::frame` is
    /// the real document renderer; this is what hands it the document.
    doc: &'a nguo_doc::Document,
    /// The light token set (`App::paper`), not the shell's dark one: the pattern
    /// is drawn dark-on-light (UX_SPEC §2).
    tokens: &'a Tokens,
    canvas2d: &'a std::sync::Mutex<nguo_canvas::Canvas2d>,
}

impl iced::widget::shader::Program<nguo_canvas::Action> for PatternCanvas<'_> {
    type State = nguo_canvas::State;
    type Primitive = nguo_gpu::canvas::ScenePrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &iced::Event,
        bounds: iced::Rectangle,
        cursor: iced::mouse::Cursor,
    ) -> Option<iced::widget::shader::Action<nguo_canvas::Action>> {
        self.inner.update(state, event, bounds, cursor)
    }

    fn draw(
        &self,
        state: &Self::State,
        cursor: iced::mouse::Cursor,
        bounds: iced::Rectangle,
    ) -> Self::Primitive {
        let _ = (state, cursor);
        let c = self.paper;
        let base_color = vello::peniko::Color::new([c.r, c.g, c.b, c.a]);
        let size = vello::kurbo::Vec2::new(bounds.width as f64, bounds.height as f64);

        // Logical units and scale factor 1.0 — the primitive applies HiDPI
        // (guide §2.4). Order matters: the ghost is ground, the draft is figure.
        let mut scene = vello::Scene::new();
        if let Some((outline, ink)) = self.ghost {
            scene.fill(
                vello::peniko::Fill::NonZero,
                self.inner.view.scene_transform(size, 1.0),
                vello::peniko::Color::new([ink.r, ink.g, ink.b, ink.a]),
                None,
                outline,
            );
        }

        // The document itself: grid, pieces, selection. `App::canvas_view` stays
        // the single source of truth for pan/zoom — the `Action`s in `update`
        // write it, so it is copied in here rather than mutated on both sides.
        let drafted = match self.canvas2d.lock() {
            Ok(mut canvas) => {
                canvas.view = self.inner.view;
                Some(canvas.frame(self.doc, size, self.tokens))
            }
            // A poisoned lock means a previous `draw` panicked. Losing the draft
            // for a frame is not worth taking the whole shell down for.
            Err(error) => {
                tracing::error!(target: "nguo_app", %error, "the canvas cache is poisoned");
                None
            }
        };
        if let Some(drafted) = &drafted {
            scene.append(drafted, None);
        }

        // The pick crosshair, in screen space at a fixed size so it stays
        // legible at any zoom. `build_scene` used to draw this; it is kept
        // because it is the visible proof that hit-testing is right on a Retina
        // display, which is a claim no other pixel on screen makes.
        if let Some(pick) = self.inner.pick {
            let at = self.inner.view.scene_transform(size, 1.0) * pick;
            let mut cross = vello::kurbo::BezPath::new();
            cross.move_to((at.x - 12.0, at.y));
            cross.line_to((at.x + 12.0, at.y));
            cross.move_to((at.x, at.y - 12.0));
            cross.line_to((at.x, at.y + 12.0));
            let m = self.tokens.color.dart;
            let mark = vello::peniko::Color::new([m.r, m.g, m.b, m.a]);
            let stroke = vello::kurbo::Stroke::new(1.5);
            scene.stroke(&stroke, vello::kurbo::Affine::IDENTITY, mark, None, &cross);
            scene.stroke(
                &stroke,
                vello::kurbo::Affine::IDENTITY,
                mark,
                None,
                &vello::kurbo::Circle::new(at, 5.0),
            );
        }

        nguo_gpu::canvas::ScenePrimitive {
            scene: std::sync::Arc::new(scene),
            base_color,
        }
    }

    fn mouse_interaction(
        &self,
        state: &Self::State,
        bounds: iced::Rectangle,
        cursor: iced::mouse::Cursor,
    ) -> iced::mouse::Interaction {
        self.inner.mouse_interaction(state, bounds, cursor)
    }
}

#[cfg(feature = "debug-overlay")]
fn debug_overlay(app: &App) -> Element<'_, Message> {
    let (width, height, scale) = nguo_gpu::window_geometry();
    nguo_app_debug::overlay(nguo_app_debug::Snapshot {
        frame_ms: app.stats.frame_ms(),
        fps: app.stats.fps(),
        frames: app.stats.frames,
        adapter: app.adapter.clone(),
        surface_format: nguo_gpu::shared()
            .map(|shared| format!("{:?}", shared.format))
            .unwrap_or_else(|| "—".to_owned()),
        present_mode: present_mode().to_owned(),
        scale_factor: scale,
        window_physical: (width, height),
        canvas_pick: app.canvas_pick.map(|p| (p.x, p.y)),
        viewport_pick: app.viewport_pick.map(|p| (p.x, p.y, p.z)),
        registry: app.registry.clone(),
    })
}

fn subscription(_app: &App) -> Subscription<Message> {
    // `frames()` is one message per vsync, not a spin: the cloth waves and the
    // splash needs a clock. `listen_with` adds the window-level events no
    // widget can see — cursor motion during a divider drag, and every key.
    Subscription::batch([
        iced::window::frames().map(Message::Tick),
        iced::event::listen_with(|event, _status, _window| match event {
            iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { .. })
            | iced::Event::Mouse(iced::mouse::Event::CursorMoved { .. }) => {
                Some(Message::Event(event))
            }
            _ => None,
        }),
    ])
}

/// A named fn, not a closure: `.theme(|_| ..)` fails HRTB inference in iced
/// 0.14 ("implementation of `Fn` is not general enough").
fn theme(app: &App) -> iced::Theme {
    w::iced_theme(&app.tokens)
}

#[cfg_attr(not(feature = "debug-overlay"), allow(dead_code))]
fn present_mode() -> &'static str {
    "Fifo (ICED_PRESENT_MODE=fifo)"
}

fn main() -> iced::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,nguo_app=info,nguo_gpu=info".into()),
        )
        .init();

    // Guide §2.5 wants `PresentMode::Fifo`. iced's `Settings::vsync` only gets
    // us `AutoVsync`; this env var is the exact knob
    // (`iced_wgpu-0.14.0/src/settings.rs:72-83`, consulted in
    // `window/compositor.rs:120`).
    if std::env::var_os("ICED_PRESENT_MODE").is_none() {
        std::env::set_var("ICED_PRESENT_MODE", "fifo");
    }

    tracing::info!(
        target: "nguo_app",
        adapter = %nguo_gpu::adapter_description(),
        "starting nguo-3d shell"
    );

    // Opens as the splash: a fixed-size, centred, undecorated panel over the
    // desktop (SPLASH.md §1). `promote_to_shell` grows it to `SHELL_SIZE` and
    // restores decorations, resizability and the normal stacking level the
    // moment the splash dismisses.
    iced::application(App::new, update, view)
        .title("nguo-3d")
        .subscription(subscription)
        .theme(theme)
        // Required, or every line of splash type falls back to the platform
        // sans (SPLASH.md §4).
        .font(nguo_splash::INTER_TTF)
        .window_size(nguo_splash::SIZE)
        .decorations(false)
        .resizable(false)
        .level(iced::window::Level::AlwaysOnTop)
        .centered()
        .run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_registry::{feature_const, Feature, Registry};

    const DRAPE: &Feature = feature_const("drape.preview");
    const CONSTRAINTS: &Feature = feature_const("constraints.solve");

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct Boom(&'static str, #[source] Option<Box<Boom>>);

    /// Fifteen toolbar buttons were decorative because nothing mapped them onto
    /// a canvas tool. Four of them are not any more, and this is the table.
    #[test]
    fn the_drawing_tools_reach_the_canvas() {
        use nguo_doc::pattern::EdgeSemantic;
        let tool = |id: &str| canvas_tool(Some(&CommandId::from(id)));
        assert_eq!(
            tool("tool.polygon"),
            nguo_canvas::Tool::Draw(EdgeSemantic::Seam)
        );
        assert_eq!(
            tool("tool.internal_line"),
            nguo_canvas::Tool::Draw(EdgeSemantic::Internal)
        );
        assert_eq!(
            tool("tool.dart"),
            nguo_canvas::Tool::Draw(EdgeSemantic::DartLeg)
        );
        assert_eq!(tool("tool.select"), nguo_canvas::Tool::Select);
        // An unwired button falls back to select rather than to a tool that
        // does not exist — and no tool at all is select too.
        assert_eq!(tool("tool.notch"), nguo_canvas::Tool::Select);
        assert_eq!(canvas_tool(None), nguo_canvas::Tool::Select);
    }

    /// The readout that replaced the ruler gutters. `grid::labels` could never
    /// be drawn — this crate has no font in the Vello path — so the numbers go
    /// where there is already text.
    #[test]
    fn the_cursor_readout_reports_millimetres_and_a_live_length() {
        let mut canvas = nguo_canvas::Canvas2d::new();
        assert_eq!(
            cursor_label(&canvas),
            "—",
            "nothing to report before a move"
        );

        canvas.cursor = Some(Point::new(123.45, -6.0));
        assert_eq!(cursor_label(&canvas), "123.5, -6.0 mm");

        canvas.interaction.rubber_band = Some((Point::ORIGIN, Point::new(30.0, 40.0)));
        assert_eq!(
            cursor_label(&canvas),
            "123.5, -6.0 mm · 50.0 mm",
            "a 3-4-5 rubber band measures itself while you draw it"
        );
    }

    /// The status line's three verdicts, and that an error outranks a warning.
    /// It is the one sentence that tells a drafter whether Space will work.
    #[test]
    fn the_seam_status_line_leads_with_whatever_is_most_actionable() {
        assert_eq!(seam_status(2, 0, 0), "2 sewn · ready to drape");
        assert_eq!(seam_status(2, 0, 3), "2 sewn · 3 to check");
        assert_eq!(seam_status(2, 1, 3), "2 sewn · 1 to fix before draping");
        // No seams at all is a legitimate draft, not a zero-state to hide.
        assert_eq!(seam_status(0, 0, 0), "0 sewn · ready to drape");
    }

    /// The maintained constraint itself (UX_SPEC §1), through the panel's own
    /// pipeline: the shipped sample must be clean, and a document whose seam
    /// sides differ by 2× must earn an `Error` that sorts to the **front** — the
    /// panel caps at six rows, so a mis-sorted error is an invisible one.
    ///
    /// This is the assertion that fails if `seam_rows` ever stops sorting, which
    /// no test of `nguo_sewing` alone would catch.
    #[test]
    fn seam_diagnostics_sort_errors_first_so_the_row_cap_cannot_hide_one() {
        let clean = nguo_sewing::validate(&shell::sample_document());
        let (errors, warnings, infos) = nguo_sewing::counts(&clean);
        assert_eq!(
            (errors, warnings),
            (0, 0),
            "the shipped sample must open clean: {:?}",
            clean.iter().map(|d| d.message()).collect::<Vec<_>>(),
        );
        assert!(infos > 0, "an ease report per seam is expected");
        assert_eq!(seam_status(2, errors, warnings), "2 sewn · ready to drape");
        // ...and the panel shows no rows at all for it: two "differ by +0.0 %"
        // lines are four lines of panel saying nothing.
        assert!(
            seam_diagnostics(&shell::sample_document()).is_empty(),
            "the clean sample should earn a status line and nothing else: {:?}",
            seam_diagnostics(&shell::sample_document())
                .iter()
                .map(|d| d.message())
                .collect::<Vec<_>>(),
        );

        // Drag one panel's bottom corner far below the hem: that side edge grows
        // to 3× its partner's, past `EASE_LIMIT`, so it stops being ease and
        // becomes an actionable error.
        let mut doc = shell::sample_document();
        let piece = doc.pieces().ids().next().expect("the sample has pieces");
        let (vertex, at) = doc
            .piece(piece)
            .expect("piece")
            .network
            .vertices
            .iter()
            .max_by(|a, b| a.1.pos.y.total_cmp(&b.1.pos.y))
            .map(|(id, v)| (id, v.pos))
            .expect("the panel has vertices");
        doc.apply_edit(nguo_doc::Edit::MoveVertex {
            piece,
            id: vertex,
            pos: nguo_doc::kurbo::Point::new(at.x, at.y + 2.0 * nguo_sewing::sample::PANEL_H_MM),
        })
        .expect("moving a vertex is always a valid edit");

        // The panel's own pipeline, not a local re-sort — so deleting the sort
        // from `seam_diagnostics` fails this test rather than passing quietly.
        let broken = seam_diagnostics(&doc);
        let (errors, warnings, _) = nguo_sewing::counts(&broken);
        assert!(
            errors > 0,
            "a 3x length mismatch must be an error, not silent ease: {:?}",
            broken.iter().map(|d| d.message()).collect::<Vec<_>>(),
        );
        assert_eq!(
            broken[0].severity,
            nguo_sewing::Severity::Error,
            "the error did not sort to the front: {:?}",
            broken.iter().map(|d| d.message()).collect::<Vec<_>>(),
        );
        // The property the cap rests on: the errors occupy a solid prefix, so the
        // first `MAX_SEAM_ROWS` rows can only omit an error once there are more
        // than six of them. `validate` emits per-*piece* warnings before per-*seam*
        // diagnostics, so unsorted this list starts with a warning and
        // `first_other` is 0 — which is what makes deleting the sort fail here
        // rather than pass quietly.
        let first_other = broken
            .iter()
            .position(|d| d.severity != nguo_sewing::Severity::Error)
            .unwrap_or(broken.len());
        assert!(
            warnings > 0,
            "this document must earn a warning too, or the ordering assertion \
             below cannot distinguish a sorted list from an unsorted one",
        );
        assert_eq!(
            first_other,
            errors,
            "the errors are not a prefix — one sorted below a warning: {:?}",
            broken
                .iter()
                .map(|d| (d.severity, d.message()))
                .collect::<Vec<_>>(),
        );
        assert!(
            errors <= MAX_SEAM_ROWS,
            "{errors} errors against a {MAX_SEAM_ROWS}-row cap: one is unreachable",
        );
        // ...and it says what to *do*, not merely what is wrong.
        let ease_error = broken
            .iter()
            .find(|d| d.severity == nguo_sewing::Severity::Error && d.message().contains("differ"))
            .expect("the length mismatch is reported");
        assert!(
            ease_error.message().contains("gathered") && ease_error.message().contains("shorten"),
            "not actionable: {}",
            ease_error.message(),
        );
        assert!(seam_status(2, errors, 0).contains("to fix before draping"));
    }

    #[test]
    fn the_status_bar_reads_a_snapshot_it_does_not_hold_a_lock_on() {
        let registry = Registry::new();
        assert_eq!(process_label(&registry.snapshot()), "no active process");

        let solve = registry.start(DRAPE, 7);
        solve.progress(0.42);
        let snapshot = registry.snapshot();

        // The whole point of `Snapshot`: it borrows nothing, so the registry
        // is free while the frame is being built. If it held the lock, this
        // second call from the same thread would deadlock.
        assert_eq!(process_label(&snapshot), "Drape preview · running · 42%");
        let _second = registry.snapshot();
        registry.start(CONSTRAINTS, 0).done().unwrap();
        assert!(registry.snapshot().recent.len() == 1);

        // The first snapshot is unaffected by later work — it is a copy.
        assert_eq!(snapshot.active.len(), 1);
        assert!(snapshot.recent.is_empty());
    }

    #[test]
    fn a_finished_process_still_shows_and_a_failure_becomes_a_toast() {
        let registry = Registry::new();
        registry.start(DRAPE, 1).done().unwrap();
        assert_eq!(process_label(&registry.snapshot()), "Drape preview · done");

        let err = Boom(
            "could not drape",
            Some(Box::new(Boom("solver diverged", None))),
        );
        registry.start(DRAPE, 2).failed(&err).unwrap();
        let snapshot = registry.snapshot();
        let failure = snapshot.failures.first().expect("one failure recorded");
        assert_eq!(failure.title(), "Drape preview");
        assert_eq!(failure.detail(), "could not drape");
        assert_eq!(failure.root_cause(), "solver diverged");
    }

    /// A clone without `assets/avatars/` must still start, still dispatch and
    /// still draw — the fallback path, forced regardless of whether the
    /// bundle happens to be present on this machine.
    #[test]
    fn a_missing_bundle_degrades_to_a_toast_and_the_cloth_fallback() {
        let mut app = App::new();
        app.avatars = None;

        let camera = app.camera;
        let _task = dispatch(&mut app, CommandId::from("avatar.body.male"));
        let _task = dispatch(&mut app, CommandId::from("avatar.body.female"));
        let _task = dispatch(&mut app, CommandId::from("avatar.tone.MST02"));
        assert_eq!(app.camera, camera, "a no-op command moved the camera");

        // `view` is what would panic on an unwrap; build the two shader
        // widgets the way `panel_body` does.
        assert!(app.avatars.as_ref().map(|a| a.viewport()).is_none());
        assert!(avatar_picker(&app).is_empty());

        // The failure the loader would have reported still becomes a toast,
        // not a log line nobody reads.
        let registry = Registry::new();
        registry
            .start(feature_const("viewport.render"), 0)
            .failed(&nguo_avatar::AvatarError::BundleNotFound("/nope".into()))
            .unwrap();
        let failure = registry.snapshot().failures.remove(0);
        assert_eq!(failure.title(), "3D drape viewport");
        assert!(
            failure.root_cause().contains("NGUO_AVATAR_BUNDLE"),
            "{}",
            failure.root_cause(),
        );
    }

    /// With a bundle present: the picker is on screen, its commands are in the
    /// palette, and choosing a taller body re-frames the camera onto it.
    #[test]
    fn the_pickers_are_wired_to_the_one_command_registry() {
        let mut app = App::new();
        if app.avatars.is_none() {
            {
                use std::io::Write;
                // fd 2 directly, NOT `eprintln!`: libtest captures the print
                // macros, so a macro'd skip notice is invisible without
                // --nocapture — i.e. not loud at all. Same fix as
                // engine-api/tests/avatar_bundle.rs.
                let _ = writeln!(std::io::stderr(), "SKIPPING: no avatar bundle");
                assert!(
                    std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none(),
                    "MGUSO_REQUIRE_AVATAR_BUNDLE is set but the avatar bundle is absent"
                );
            }
            return;
        }

        // 2 bodies + 10 tones, all findable by typing (UX_SPEC §3.2).
        assert_eq!(
            app.catalogue
                .iter()
                .filter(|e| e.id.as_str().starts_with("avatar."))
                .count(),
            12,
        );
        assert_eq!(command::search(&app.catalogue, "MST07").len(), 1);
        assert!(command::search(&app.catalogue, "Espresso")
            .iter()
            .any(|e| e.id.as_str() == "avatar.tone.MST09"));
        assert!(command::search(&app.catalogue, "Nguo Male")
            .iter()
            .any(|e| e.id.as_str() == "avatar.body.male"));

        // Header + 2 bodies + stature + header + swatches + tone + 2 credits.
        assert!(avatar_picker(&app).len() >= 9);

        // The shell opens on the female body, which is the *taller* of the two
        // (1.90477 m vs 1.830 m), so switching to the male one pulls the camera
        // in rather than pushing it back.
        let tall = app.camera.distance;
        let _task = dispatch(&mut app, CommandId::from("avatar.body.male"));
        let avatars = app.avatars.as_ref().expect("still loaded");
        assert_eq!(avatars.selected_key(), "male");
        assert!(
            app.camera.distance < tall,
            "the 1.830 m body did not pull the camera in from the 1.90477 m one",
        );
        assert!((app.camera.target.y - avatars.selected().stature_m as f32 / 2.0).abs() < 1e-4);

        // Every id the palette lists must dispatch — both bodies and all ten
        // tones, through the same `dispatch` a keystroke goes through, and
        // without leaving a failure behind in the registry.
        let ids: Vec<CommandId> = app
            .avatars
            .as_ref()
            .unwrap()
            .command_entries()
            .into_iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(ids.len(), 12);
        let before = nguo_registry::snapshot().failures.len();
        for id in ids {
            let _task = dispatch(&mut app, id.clone());
            let avatars = app.avatars.as_ref().unwrap();
            match id.as_str().strip_prefix(avatar::BODY_COMMAND) {
                Some(key) => assert_eq!(avatars.selected_key(), key),
                None => assert_eq!(
                    avatars.tone(),
                    id.as_str().trim_start_matches(avatar::TONE_COMMAND)
                ),
            }
        }
        assert_eq!(
            nguo_registry::snapshot().failures.len(),
            before,
            "a catalogue command toasted",
        );

        let _task = dispatch(&mut app, CommandId::from("avatar.tone.MST02"));
        assert_eq!(app.avatars.as_ref().unwrap().tone(), "MST02");
    }

    #[test]
    fn every_toolbar_entry_names_a_real_command() {
        let map = nguo_keymap::default_macos();
        let catalogue = command::catalogue(&map);
        for (_, id) in TOOLBAR.iter().flatten() {
            assert!(
                catalogue.iter().any(|e| e.id.as_str() == *id),
                "toolbar references `{id}`, which is not in the keymap",
            );
        }
        // One row, never truncating (UX_SPEC §3.4 — no `»` chevron at the
        // default window size). Measured on screen at 1440 logical px: 18
        // buttons + 5 separators occupy ~930 px. Overflow goes to the palette,
        // so this is a hard cap, not a scroll.
        assert_eq!(TOOLBAR.iter().flatten().count(), 18);
        assert!(
            TOOLBAR.len() <= 28,
            "toolbar has grown to {}",
            TOOLBAR.len()
        );
    }
}
