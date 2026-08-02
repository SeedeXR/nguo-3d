// SPDX-License-Identifier: GPL-2.0-or-later
//! The tests that make the keymap trustworthy: the shipped map is conflict
//! free, it survives a save/load cycle byte for byte, contexts resolve the way
//! the design says they do, reserved chords cannot be taken, and the menu
//! strings are the macOS glyphs.

use nguo_keymap::{
    default_macos, try_default_macos, Binding, Chord, CommandId, Context, Keymap, KeymapError,
    Mods, Platform, Resolution, Resolver, Sequence, DEFAULT_MACOS, KEYMAP_VERSION, RESERVED,
};

fn cmd(s: &str) -> CommandId {
    CommandId::from(s)
}

fn seq(s: &str) -> Sequence {
    s.parse().expect(s)
}

// ---------------------------------------------------------------- defaults

#[test]
fn default_map_is_valid() {
    // `default_macos()` panics on a bad table; this asserts on the error so a
    // regression reports *what* is wrong, not just that something is.
    if let Err(e) = try_default_macos() {
        panic!("shipped default keymap is invalid: {e}");
    }
}

#[test]
fn every_row_is_present_and_accounted_for() {
    let map = default_macos();
    assert_eq!(map.bindings().len(), DEFAULT_MACOS.len());
    // The counts in ui/docs/KEYMAP.md. Update both together.
    let n = |c: Context| map.bindings().iter().filter(|b| b.context == c).count();
    assert_eq!(n(Context::Global), 38, "global bindings");
    assert_eq!(n(Context::Canvas2d), 41, "2D canvas bindings");
    assert_eq!(n(Context::Viewport3d), 13, "3D viewport bindings");
    assert_eq!(n(Context::Modal), 2, "modal bindings");
    assert_eq!(n(Context::TextEntry), 0, "text entry binds nothing");
    assert_eq!(map.bindings().len(), 94);
}

// --------------------------------------------------------- serialisation

#[test]
fn every_default_binding_round_trips_through_serialisation() {
    let map = default_macos();
    let json = serde_json::to_string_pretty(&map).expect("serialise");
    let back: Keymap = serde_json::from_str(&json).expect("deserialise");

    assert_eq!(map.bindings().len(), back.bindings().len());
    for (a, b) in map.bindings().iter().zip(back.bindings()) {
        assert_eq!(a, b, "binding changed across the round trip");
    }
    // And the text itself is stable, so a settings file does not churn.
    let json2 = serde_json::to_string_pretty(&back).expect("re-serialise");
    assert_eq!(json, json2);
}

#[test]
fn each_chord_round_trips_through_its_string_form() {
    for &(_, _, chord) in DEFAULT_MACOS {
        let parsed = seq(chord);
        assert_eq!(
            parsed.to_string(),
            chord,
            "`{chord}` is not in canonical form; the table and the parser disagree"
        );
        assert_eq!(seq(&parsed.to_string()), parsed);
    }
}

#[test]
fn wire_format_is_the_documented_shape() {
    let map = default_macos();
    let v: serde_json::Value = serde_json::to_value(&map).unwrap();
    assert_eq!(v["version"], serde_json::json!(KEYMAP_VERSION));
    assert_eq!(
        v["bindings"][0],
        serde_json::json!({
            "command": "app.quit",
            "context": "global",
            "chord": "primary+q",
        })
    );
}

#[test]
fn a_future_version_is_refused_rather_than_misread() {
    let err = serde_json::from_str::<Keymap>(r#"{"version":99,"bindings":[]}"#).unwrap_err();
    assert!(err.to_string().contains("version 99"), "{err}");
}

#[test]
fn loading_a_conflicting_keymap_names_both_commands() {
    let mut rows: Vec<Binding> = default_macos().bindings().to_vec();
    rows.push(Binding {
        command: cmd("tool.sabotage"),
        context: Context::Canvas2d,
        chord: seq("a"),
    });
    let err = Keymap::new(rows).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("tool.transform_pattern"), "{msg}");
    assert!(msg.contains("tool.sabotage"), "{msg}");
    assert!(msg.contains("canvas2d"), "{msg}");
    assert!(matches!(err, KeymapError::Conflict { .. }));
}

/// A repeated identical row is redundant, not a conflict — but it must not
/// survive into `bindings()`, or the shortcut UI offers the user a second
/// chord for `app.quit` that is the same chord.
#[test]
fn an_identical_duplicate_row_is_collapsed() {
    let mut rows: Vec<Binding> = default_macos().bindings().to_vec();
    let n = rows.len();
    rows.push(Binding {
        command: cmd("app.quit"),
        context: Context::Global,
        chord: seq("primary+q"),
    });
    let map = Keymap::new(rows).expect("an identical row is not a conflict");
    assert_eq!(map.bindings().len(), n);
    assert_eq!(map.chords_for(&cmd("app.quit")).len(), 1);
}

#[test]
fn a_binding_that_shadows_a_sequence_prefix_is_a_conflict() {
    let mut rows: Vec<Binding> = default_macos().bindings().to_vec();
    // `k` alone would swallow every `k <letter>` constraint binding.
    rows.push(Binding {
        command: cmd("tool.kilt"),
        context: Context::Canvas2d,
        chord: seq("k"),
    });
    let err = Keymap::new(rows).unwrap_err();
    assert!(
        matches!(err, KeymapError::PrefixConflict { .. }),
        "expected a prefix conflict, got {err}"
    );
    assert!(err.to_string().contains("tool.kilt"), "{err}");
}

// ------------------------------------------------------ context resolution

#[test]
fn the_same_chord_differs_between_the_2d_canvas_and_the_3d_viewport() {
    let map = default_macos();
    let q = [Chord::ch(Mods::NONE, 'q')];
    assert_eq!(
        map.resolve(Context::Canvas2d, &q),
        Resolution::Command(cmd("tool.select"))
    );
    assert_eq!(
        map.resolve(Context::Viewport3d, &q),
        Resolution::Command(cmd("tool.select_move_3d"))
    );
}

#[test]
fn a_context_falls_through_to_global() {
    let map = default_macos();
    let space = [Chord::named(Mods::NONE, nguo_keymap::Named::Space)];
    for ctx in [Context::Canvas2d, Context::Viewport3d] {
        assert_eq!(
            map.resolve(ctx, &space),
            Resolution::Command(cmd("sim.toggle")),
            "space should simulate from {ctx}"
        );
    }
}

#[test]
fn text_entry_swallows_a_bare_a() {
    let map = default_macos();
    assert_eq!(
        map.resolve(Context::TextEntry, &[Chord::ch(Mods::NONE, 'a')]),
        Resolution::Swallowed
    );
}

/// The real bug this whole model exists to prevent: typing in a numeric or
/// expression field must never fire a tool or start a simulation.
#[test]
fn nothing_fires_while_typing_an_expression() {
    let map = default_macos();
    let mut r = Resolver::new();
    // `waist/4 + 1cm`
    for c in "waist/4 + 1cm".chars() {
        let chord = if c == ' ' {
            Chord::named(Mods::NONE, nguo_keymap::Named::Space)
        } else {
            Chord::ch(Mods::NONE, c)
        };
        assert_eq!(
            r.press(&map, Context::TextEntry, chord),
            Resolution::Swallowed,
            "`{c}` escaped the text field"
        );
    }
    // Shift and Option-only chords must not escape either: ⇧Q would hide the
    // pattern and ⌥7 would toggle the strain map.
    assert_eq!(
        map.resolve(Context::TextEntry, &[Chord::ch(Mods::SHIFT, 'q')]),
        Resolution::Swallowed
    );
    assert_eq!(
        map.resolve(Context::TextEntry, &[Chord::ch(Mods::ALT, '7')]),
        Resolution::Swallowed
    );
}

#[test]
fn documented_chords_pierce_text_entry() {
    let map = default_macos();
    // Saving must work mid-word.
    assert_eq!(
        map.resolve(Context::TextEntry, &[Chord::ch(Mods::PRIMARY, 's')]),
        Resolution::Command(cmd("file.save"))
    );
    for (chord, command) in [
        ("primary+s", "file.save"),
        ("primary+o", "file.open"),
        ("primary+n", "file.new"),
        ("primary+q", "app.quit"),
        ("primary+w", "window.close"),
        ("primary+,", "app.preferences"),
        ("primary+k", "agent.palette"),
        ("escape", "ui.cancel"),
    ] {
        assert_eq!(
            map.resolve(Context::TextEntry, seq(chord).chords()),
            Resolution::Command(cmd(command)),
            "`{chord}` should pierce text entry"
        );
    }
}

#[test]
fn text_owned_chords_stay_with_the_text_widget() {
    let map = default_macos();
    // ⌘Z in a field undoes the *text*, not the document. ⌘⇧V is
    // paste-and-match-style, not Layer Clone.
    for chord in [
        "primary+z",
        "primary+shift+z",
        "primary+y",
        "primary+a",
        "primary+c",
        "primary+v",
        "primary+x",
        "primary+shift+v",
        "primary+shift+c",
    ] {
        assert_eq!(
            map.resolve(Context::TextEntry, seq(chord).chords()),
            Resolution::Swallowed,
            "`{chord}` should be left to the text widget"
        );
    }
}

#[test]
fn a_modal_owns_the_keyboard_except_escape_confirm_and_quit() {
    let map = default_macos();
    assert_eq!(
        map.resolve(Context::Modal, seq("escape").chords()),
        Resolution::Command(cmd("ui.cancel"))
    );
    assert_eq!(
        map.resolve(Context::Modal, seq("return").chords()),
        Resolution::Command(cmd("ui.confirm"))
    );
    assert_eq!(
        map.resolve(Context::Modal, seq("primary+q").chords()),
        Resolution::Command(cmd("app.quit"))
    );
    assert_eq!(
        map.resolve(Context::Modal, seq("primary+s").chords()),
        Resolution::Swallowed
    );
    assert_eq!(
        map.resolve(Context::Modal, seq("b").chords()),
        Resolution::Swallowed
    );
}

// ------------------------------------------------------- multi-key sequences

#[test]
fn two_key_constraint_sequences_resolve() {
    let map = default_macos();
    let mut r = Resolver::new();
    assert_eq!(
        r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'k')),
        Resolution::Pending
    );
    assert_eq!(r.pending().len(), 1);
    assert_eq!(
        r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'p')),
        Resolution::Command(cmd("constraint.parallel"))
    );
    assert!(r.pending().is_empty());
}

#[test]
fn a_dead_sequence_clears_and_does_not_eat_the_next_key() {
    let map = default_macos();
    let mut r = Resolver::new();
    r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'k'));
    assert_eq!(
        r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'z')),
        Resolution::Unbound
    );
    assert!(r.pending().is_empty());
    assert_eq!(
        r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'z')),
        Resolution::Command(cmd("tool.edit_pattern"))
    );
}

#[test]
fn the_constraint_prefix_is_not_live_in_the_3d_viewport() {
    let map = default_macos();
    assert_eq!(
        map.resolve(Context::Viewport3d, &[Chord::ch(Mods::NONE, 'k')]),
        Resolution::Unbound
    );
}

/// A prefix typed in one pane means nothing in another. If it survived the
/// focus change it would silently eat the first key pressed in the new pane —
/// and the app has no hook to clear it, because focus can move without a
/// keypress (a click in the 3D view, the palette opening).
#[test]
fn a_focus_change_abandons_a_half_typed_sequence() {
    let map = default_macos();
    let mut r = Resolver::new();
    assert_eq!(
        r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'k')),
        Resolution::Pending
    );
    // `w` is Pin in the 3D viewport. With `k` still pending it resolves as
    // `k w` — unbound — and the tool never activates.
    assert_eq!(
        r.press(&map, Context::Viewport3d, Chord::ch(Mods::NONE, 'w')),
        Resolution::Command(cmd("tool.pin"))
    );
    assert!(r.pending().is_empty());
}

#[test]
fn cancel_abandons_a_half_typed_sequence() {
    let map = default_macos();
    let mut r = Resolver::new();
    r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'k'));
    r.cancel();
    assert!(r.pending().is_empty());
}

// -------------------------------------------------------------- reserved

#[test]
fn reserved_bindings_cannot_be_silently_reassigned() {
    for r in RESERVED {
        let mut map = default_macos();
        let chord = seq(r.chord);
        let err = map
            .rebind(cmd("tool.usurper"), r.context, chord.clone())
            .expect_err(&format!("`{}` was reassignable", r.chord));
        assert!(
            matches!(err, KeymapError::ReservedReassigned { .. }),
            "`{}`: {err}",
            r.chord
        );
        // ...and the map is unchanged after the refusal.
        assert_eq!(
            map.resolve(r.context, chord.chords()),
            Resolution::Command(cmd(r.command))
        );
    }
}

#[test]
fn reserved_bindings_cannot_be_removed() {
    for r in RESERVED {
        let mut map = default_macos();
        let err = map
            .unbind(r.context, &seq(r.chord))
            .expect_err(&format!("`{}` was removable", r.chord));
        assert!(
            matches!(err, KeymapError::ReservedUnbound { .. }),
            "`{}`: {err}",
            r.chord
        );
    }
}

#[test]
fn a_reserved_chord_cannot_be_hijacked_by_a_more_specific_context() {
    let mut map = default_macos();
    let err = map
        .rebind(cmd("tool.usurper"), Context::Canvas2d, seq("primary+s"))
        .unwrap_err();
    assert!(
        matches!(err, KeymapError::ReservedReassigned { .. }),
        "{err}"
    );
    assert!(err.to_string().contains("canvas2d"), "{err}");

    // …and it cannot be neutralised by a *sequence that starts with it*
    // either. `primary+s x` leaves nothing bound to ⌘S in canvas2d, so the
    // exact-match check above sees nothing — but ⌘S there now resolves to
    // `Pending` and the file is never saved. Same data loss, quieter route.
    for r in RESERVED {
        for ctx in [Context::Canvas2d, Context::Viewport3d] {
            let mut map = default_macos();
            let longer = seq(&format!("{} x", r.chord));
            let err = map
                .rebind(cmd("tool.usurper"), ctx, longer.clone())
                .expect_err(&format!("`{longer}` was accepted in {ctx}"));
            assert!(
                matches!(err, KeymapError::ReservedReassigned { .. }),
                "`{longer}` in {ctx}: {err}"
            );
            // …and the refusal left the chord doing its job.
            assert_eq!(
                map.resolve(ctx, seq(r.chord).chords()),
                Resolution::Command(cmd(r.command)),
                "`{}` stopped resolving in {ctx}",
                r.chord
            );
        }
    }
}

#[test]
fn a_non_reserved_binding_is_remappable() {
    let mut map = default_macos();
    map.rebind(cmd("tool.dart"), Context::Canvas2d, seq("shift+d"))
        .unwrap();
    assert_eq!(
        map.resolve(Context::Canvas2d, seq("shift+d").chords()),
        Resolution::Command(cmd("tool.dart"))
    );
    // The old chord is still there — rebind adds, it does not move.
    assert_eq!(
        map.resolve(Context::Canvas2d, seq("d").chords()),
        Resolution::Command(cmd("tool.dart"))
    );
    assert!(map.unbind(Context::Canvas2d, &seq("d")).unwrap());
    assert_eq!(
        map.resolve(Context::Canvas2d, seq("d").chords()),
        Resolution::Unbound
    );
}

#[test]
fn the_redo_alias_is_removable_but_the_default_is_not() {
    let mut map = default_macos();
    assert!(map.unbind(Context::Global, &seq("primary+y")).unwrap());
    assert!(map
        .unbind(Context::Global, &seq("primary+shift+z"))
        .is_err());
}

// --------------------------------------------------------------- display

#[test]
fn display_produces_the_expected_macos_glyphs() {
    for (chord, expected) in [
        // Apple HIG modifier order is ⌃⌥⇧⌘, so Cmd+Shift+Z is ⇧⌘Z, not ⌘⇧Z.
        ("primary+shift+z", "⇧⌘Z"),
        ("primary+s", "⌘S"),
        ("shift+f", "⇧F"),
        ("alt+7", "⌥7"),
        ("alt+w", "⌥W"),
        ("primary+shift+i", "⇧⌘I"),
        ("primary+[", "⌘["),
        ("]", "]"),
        ("escape", "⎋"),
        ("primary+return", "⌘↩"),
        ("backspace", "⌫"),
        ("tab", "⇥"),
        ("space", "Space"),
        ("secondary+alt+shift+primary+a", "⌃⌥⇧⌘A"),
        ("k c", "K C"),
    ] {
        assert_eq!(seq(chord).display(Platform::MacOs), expected, "{chord}");
    }
}

#[test]
fn display_has_a_non_macos_form_without_hardcoding_cmd() {
    assert_eq!(
        seq("primary+shift+z").display(Platform::Other),
        "Ctrl+Shift+Z"
    );
    assert_eq!(seq("alt+7").display(Platform::Other), "Alt+7");
    assert_eq!(seq("escape").display(Platform::Other), "Esc");
}

#[test]
fn menus_print_the_preferred_chord_not_the_alias() {
    let map = default_macos();
    assert_eq!(
        map.display_for(&cmd("edit.redo"), Platform::MacOs)
            .as_deref(),
        Some("⇧⌘Z"),
    );
    assert_eq!(map.chords_for(&cmd("edit.redo")).len(), 2);
    assert_eq!(
        map.display_for(&cmd("no.such.command"), Platform::MacOs),
        None
    );
}
