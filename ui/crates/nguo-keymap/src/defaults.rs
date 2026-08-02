// SPDX-License-Identifier: GPL-2.0-or-later
//! The shipped default macOS keymap, as data, and the reserved-chord list.
//!
//! Every row is `(command, context, chord)` and every chord goes through the
//! same parser a user's settings file does — there is exactly one code path, so
//! the defaults exercise it on every run of the test suite.
//!
//! Rationale for every row that carries a decision lives in
//! `ui/docs/KEYMAP.md`. Where a letter is already established practice in
//! garment CAD we keep it, because muscle memory is the largest switching cost
//! a professional tool has; where we depart from it, the reason is written
//! down.

use crate::keymap::{Binding, Context, Keymap, KeymapError, Reserved};

use Context::{Canvas2d, Global, Modal, Viewport3d};

/// `(command, context, chord)`.
pub type Row = (&'static str, Context, &'static str);

/// The default macOS keymap.
///
/// Only a macOS map is shipped. Nothing here says `Cmd`: `primary` is `⌘` on
/// macOS and `Ctrl` elsewhere, so most of this table is already portable. A
/// Windows/Linux map will still need its own rows for the handful of chords
/// that are macOS-shaped (`primary+,` for settings, `alt+7`), and will be
/// written when there is a Windows/Linux build to test it on.
pub const DEFAULT_MACOS: &[Row] = &[
    // ---- Global: application, file, edit -------------------------------
    ("app.quit", Global, "primary+q"),
    ("window.close", Global, "primary+w"),
    ("app.preferences", Global, "primary+,"),
    ("file.new", Global, "primary+n"),
    ("file.open", Global, "primary+o"),
    ("file.save", Global, "primary+s"),
    ("file.save_as", Global, "primary+shift+s"),
    ("file.export", Global, "primary+shift+e"),
    ("edit.undo", Global, "primary+z"),
    // Default first: menus print ⇧⌘Z. `primary+y` follows as an accepted
    // alias for Windows muscle memory.
    ("edit.redo", Global, "primary+shift+z"),
    ("edit.redo", Global, "primary+y"),
    ("edit.cut", Global, "primary+x"),
    ("edit.copy", Global, "primary+c"),
    ("edit.paste", Global, "primary+v"),
    ("edit.select_all", Global, "primary+a"),
    ("edit.delete", Global, "backspace"),
    ("edit.delete", Global, "delete"),
    ("edit.duplicate", Global, "primary+shift+d"),
    ("edit.find", Global, "primary+f"),
    // ---- Global: pattern properties (the high-frequency set) -----------
    ("pattern.symmetric_with_sewing", Global, "primary+d"),
    ("pattern.solidify", Global, "primary+e"),
    ("pattern.strengthen", Global, "alt+h"),
    ("pattern.deactivate", Global, "primary+j"),
    ("pattern.freeze", Global, "primary+shift+k"),
    ("pattern.hide_show", Global, "shift+q"),
    ("selection.isolate", Global, "shift+i"),
    // ---- Global: view --------------------------------------------------
    ("view.focus_selection", Global, "f"),
    ("view.zoom_extents_all", Global, "]"),
    ("view.zoom_in", Global, "primary+="),
    ("view.zoom_out", Global, "primary+-"),
    ("view.cycle_pane_focus", Global, "tab"),
    ("panel.library", Global, "y"),
    // ---- Global: simulation, agent, app --------------------------------
    ("sim.toggle", Global, "space"),
    ("agent.palette", Global, "primary+k"),
    ("agent.run", Global, "primary+return"),
    ("agent.panel", Global, "primary+shift+a"),
    ("ui.cancel", Global, "escape"),
    ("help.shortcuts", Global, "primary+/"),
    // ---- 2D canvas: tools ----------------------------------------------
    ("tool.select", Canvas2d, "q"),
    ("tool.transform_pattern", Canvas2d, "a"),
    ("tool.edit_pattern", Canvas2d, "z"),
    ("tool.add_points", Canvas2d, "x"),
    ("tool.edit_curvature", Canvas2d, "c"),
    ("tool.edit_curve_point", Canvas2d, "v"),
    ("tool.shape_rectangle", Canvas2d, "s"),
    ("tool.shape_ellipse", Canvas2d, "e"),
    ("tool.inner_ellipse", Canvas2d, "r"),
    ("tool.internal_line", Canvas2d, "g"),
    ("tool.polygon", Canvas2d, "h"),
    ("tool.trace", Canvas2d, "i"),
    ("tool.edit_texture", Canvas2d, "t"),
    ("tool.dart", Canvas2d, "d"),
    ("tool.notch", Canvas2d, "n"),
    ("tool.measure", Canvas2d, "m"),
    // Group cycling: one key per family, press again to advance. The keymap
    // stays stateless — it emits the same id every time and the tool
    // dispatcher advances the variant when the family is already active.
    ("tool.sewing", Canvas2d, "b"),
    ("tool.sewing_prev", Canvas2d, "shift+b"),
    ("tool.topstitch", Canvas2d, "j"),
    ("tool.topstitch_prev", Canvas2d, "shift+j"),
    // ---- 2D canvas: selection & clone ----------------------------------
    ("selection.invert", Canvas2d, "primary+shift+i"),
    ("pattern.layer_clone_over", Canvas2d, "primary+shift+v"),
    ("pattern.clone_as_pattern", Canvas2d, "primary+shift+c"),
    // ---- 2D canvas: parametric constraints (ours; no prior art) --------
    // `K` is a prefix, not a binding: twelve constraint types do not fit in
    // twelve spare letters, and an `⌥`-letter layer collides with macOS dead
    // keys. Sub-letters follow the sketcher convention users arrive with.
    ("constraint.coincident", Canvas2d, "k c"),
    ("constraint.horizontal", Canvas2d, "k h"),
    ("constraint.vertical", Canvas2d, "k v"),
    ("constraint.parallel", Canvas2d, "k p"),
    ("constraint.perpendicular", Canvas2d, "k n"),
    ("constraint.tangent", Canvas2d, "k t"),
    ("constraint.equal", Canvas2d, "k e"),
    ("constraint.distance", Canvas2d, "k d"),
    ("constraint.angle", Canvas2d, "k a"),
    ("constraint.symmetric", Canvas2d, "k s"),
    ("constraint.fix", Canvas2d, "k f"),
    ("constraint.remove", Canvas2d, "k x"),
    ("constraint.toggle_display", Canvas2d, "shift+k"),
    ("constraint.diagnose", Canvas2d, "alt+k"),
    // ---- 2D canvas: grading (ours; no prior art) -----------------------
    ("grading.next_size", Canvas2d, "alt+]"),
    ("grading.prev_size", Canvas2d, "alt+["),
    ("grading.toggle_size_run", Canvas2d, "shift+g"),
    ("grading.rules", Canvas2d, "alt+g"),
    // ---- 3D viewport ---------------------------------------------------
    ("tool.select_move_3d", Viewport3d, "q"),
    ("tool.pin", Viewport3d, "w"),
    ("tool.create_topology", Viewport3d, "u"),
    ("view.reset_arrangement", Viewport3d, "shift+r"),
    ("pattern.flip_horizontal", Viewport3d, "shift+h"),
    ("pattern.superimpose_under", Viewport3d, "primary+["),
    ("pattern.superimpose_over", Viewport3d, "primary+]"),
    ("view.toggle_avatar", Viewport3d, "shift+a"),
    ("view.toggle_arrangement_points", Viewport3d, "shift+f"),
    ("view.toggle_avatar_joints", Viewport3d, "shift+x"),
    ("view.toggle_strain_map", Viewport3d, "alt+7"),
    ("view.toggle_wireframe", Viewport3d, "alt+w"),
    ("view.toggle_thick_textured", Viewport3d, "alt+1"),
    // ---- Modal ----------------------------------------------------------
    ("ui.cancel", Modal, "escape"),
    ("ui.confirm", Modal, "return"),
    // Context::TextEntry deliberately has no bindings: its whole job is to
    // swallow. See Context::pierces_text_entry.
];

/// Chords that may never mean anything else.
///
/// Two kinds, both immutable: chords macOS itself owns (`⌘Q`, `⌘W`, `⌘,`) and
/// chords the application owns because losing them would be a data-loss or
/// dead-end bug (`⌘S`, `⌘Z`, `⇧⌘Z`, `⎋`). `Space` is here because it is the
/// simulate toggle — the single most-pressed key in the product.
pub const RESERVED: &[Reserved] = &[
    Reserved {
        chord: "primary+q",
        command: "app.quit",
        context: Global,
        reason: "macOS reserves ⌘Q for Quit",
    },
    Reserved {
        chord: "primary+w",
        command: "window.close",
        context: Global,
        reason: "macOS reserves ⌘W for Close Window",
    },
    Reserved {
        chord: "primary+,",
        command: "app.preferences",
        context: Global,
        reason: "macOS reserves ⌘, for Settings",
    },
    Reserved {
        chord: "primary+n",
        command: "file.new",
        context: Global,
        reason: "universal New",
    },
    Reserved {
        chord: "primary+o",
        command: "file.open",
        context: Global,
        reason: "universal Open",
    },
    Reserved {
        chord: "primary+s",
        command: "file.save",
        context: Global,
        reason: "universal Save; rebinding it is a data-loss bug",
    },
    Reserved {
        chord: "primary+z",
        command: "edit.undo",
        context: Global,
        reason: "universal Undo",
    },
    Reserved {
        chord: "primary+shift+z",
        command: "edit.redo",
        context: Global,
        reason: "the macOS Redo chord; ⌘Y stays as a removable alias",
    },
    Reserved {
        chord: "primary+f",
        command: "edit.find",
        context: Global,
        reason: "universal Find; Reset 3D Arrangement is ⇧R instead",
    },
    Reserved {
        chord: "space",
        command: "sim.toggle",
        context: Global,
        reason: "simulate; pan is a pointer gesture, not a key",
    },
    Reserved {
        chord: "escape",
        command: "ui.cancel",
        context: Global,
        reason: "the universal way out; without it a modal or half-typed \
                 sequence is a dead end",
    },
];

/// Build the default macOS keymap.
///
/// # Panics
/// Only if [`DEFAULT_MACOS`] itself is malformed, which
/// `tests/defaults.rs::default_map_is_valid` makes impossible to ship.
#[must_use]
pub fn default_macos() -> Keymap {
    try_default_macos().expect("the shipped default keymap must be valid")
}

/// Fallible form of [`default_macos`], so the test suite can assert on the
/// error instead of a panic message.
pub fn try_default_macos() -> Result<Keymap, KeymapError> {
    let bindings = DEFAULT_MACOS
        .iter()
        .map(|&(cmd, ctx, chord)| Binding::parse(cmd, ctx, chord))
        .collect::<Result<Vec<_>, _>>()?;
    Keymap::new(bindings)
}
