// SPDX-License-Identifier: GPL-2.0-or-later
//! The command layer: iced key events in, `CommandId`s out, and the palette
//! that lets you find one by typing (UX_SPEC §3.3).
//!
//! No iced *widgets* here — only the input translation and the matching — so
//! the whole thing is testable without a window.

use iced::keyboard::{key, Key, Modifiers};
use nguo_keymap::{Chord, CommandId, Keymap, Mods, Named, Platform};

// ---------------------------------------------------------------------------
// iced key event -> keymap chord
// ---------------------------------------------------------------------------

/// Translate one iced key press into a chord.
///
/// **Pass `KeyPressed::key`, never `modified_key`** (`ui/docs/KEYMAP.md` §7).
/// `key` is the character the key produces with no modifiers, which is the
/// only thing the keymap indexes: on macOS `⌥7` arrives as `modified_key = ¶`
/// and every `⌥` binding in the map would silently stop working.
///
/// Returns `None` for a modifier key on its own and for keys the map has no
/// vocabulary for — both mean "not a chord", not "unbound".
pub fn chord_from_key(key: &Key, modifiers: Modifiers) -> Option<Chord> {
    let mut mods = Mods::NONE;
    if modifiers.command() {
        mods = mods | Mods::PRIMARY;
    }
    if modifiers.shift() {
        mods = mods | Mods::SHIFT;
    }
    if modifiers.alt() {
        mods = mods | Mods::ALT;
    }
    // `Modifiers::command()` is already Ctrl off macOS, so only macOS has a
    // distinct fourth modifier to report.
    #[cfg(target_os = "macos")]
    if modifiers.control() {
        mods = mods | Mods::SECONDARY;
    }

    match key {
        Key::Character(s) => {
            let c = s.chars().next()?;
            Some(Chord::ch(mods, c))
        }
        Key::Named(named) => Some(Chord::new(mods, nguo_keymap::Key::Named(name(*named)?))),
        Key::Unidentified => None,
    }
}

fn name(named: key::Named) -> Option<Named> {
    use key::Named as N;
    Some(match named {
        N::Space => Named::Space,
        N::Escape => Named::Escape,
        N::Enter => Named::Return,
        N::Tab => Named::Tab,
        N::Backspace => Named::Backspace,
        N::Delete => Named::Delete,
        N::ArrowUp => Named::Up,
        N::ArrowDown => Named::Down,
        N::ArrowLeft => Named::Left,
        N::ArrowRight => Named::Right,
        N::Home => Named::Home,
        N::End => Named::End,
        N::PageUp => Named::PageUp,
        N::PageDown => Named::PageDown,
        N::F1 => Named::Function(1),
        N::F2 => Named::Function(2),
        N::F3 => Named::Function(3),
        N::F4 => Named::Function(4),
        N::F5 => Named::Function(5),
        N::F6 => Named::Function(6),
        N::F7 => Named::Function(7),
        N::F8 => Named::Function(8),
        N::F9 => Named::Function(9),
        N::F10 => Named::Function(10),
        N::F11 => Named::Function(11),
        N::F12 => Named::Function(12),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// The command catalogue
// ---------------------------------------------------------------------------

/// One palette row: a command, a human label and the chord that fires it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub id: CommandId,
    /// `"Tool · Transform pattern"` — group and title, derived from the id.
    pub label: String,
    /// Rendered accelerator, e.g. `⇧⌘Z`. `None` if the command is only
    /// reachable from the palette.
    pub shortcut: Option<String>,
}

/// Every command the keymap knows, once each, sorted by label.
///
/// Derived from the map rather than kept as a second table: a hand-written
/// label list is a thing that drifts, and the id already carries the words a
/// user would guess (UX_SPEC §3.5e — "find any command by typing a word you'd
/// plausibly guess").
pub fn catalogue(map: &Keymap) -> Vec<Entry> {
    let mut seen: Vec<CommandId> = Vec::new();
    let mut out = Vec::new();
    for binding in map.bindings() {
        if seen.contains(&binding.command) {
            continue;
        }
        seen.push(binding.command.clone());
        out.push(Entry {
            label: humanise(binding.command.as_str()),
            shortcut: map.display_for(&binding.command, Platform::MacOs),
            id: binding.command.clone(),
        });
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

/// `"tool.transform_pattern"` -> `"Tool · Transform pattern"`.
fn humanise(id: &str) -> String {
    let (group, rest) = id.split_once('.').unwrap_or(("", id));
    let title = sentence_case(rest);
    if group.is_empty() {
        title
    } else {
        format!("{} · {title}", sentence_case(group))
    }
}

fn sentence_case(s: &str) -> String {
    let spaced = s.replace(['_', '.'], " ");
    let mut chars = spaced.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Palette matching
// ---------------------------------------------------------------------------

/// How well a query matched. Lower sorts first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Rank {
    /// The query is a prefix of the command's title (`"undo"` -> `edit.undo`).
    TitlePrefix,
    /// The query is a prefix of the whole label or the dotted id.
    LabelPrefix,
    /// The query appears somewhere in the label or id.
    Substring,
    /// The query's characters appear in order but not adjacently.
    Subsequence,
}

/// Rank `entries` against `query`. An empty query returns everything, in
/// catalogue order.
///
/// The ordering rule that matters, and the one the test pins: **an exact
/// prefix always outranks a looser match**, so typing the first letters of a
/// command you already know never buries it under a fuzzy hit.
pub fn search<'a>(entries: &'a [Entry], query: &str) -> Vec<&'a Entry> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return entries.iter().collect();
    }

    let mut hits: Vec<(Rank, &Entry)> = entries
        .iter()
        .filter_map(|e| rank(e, &q).map(|r| (r, e)))
        .collect();
    // Stable, so equal ranks keep catalogue order, which is alphabetical.
    hits.sort_by_key(|(rank, _)| *rank);
    hits.into_iter().map(|(_, e)| e).collect()
}

fn rank(entry: &Entry, q: &str) -> Option<Rank> {
    let label = entry.label.to_lowercase();
    let id = entry.id.as_str().to_lowercase();
    // The part after the last `·`/`.` — the bit a user thinks of as the name.
    let title = label.rsplit('·').next().unwrap_or(&label).trim().to_owned();

    if title.starts_with(q) {
        return Some(Rank::TitlePrefix);
    }
    if label.starts_with(q) || id.starts_with(q) {
        return Some(Rank::LabelPrefix);
    }
    if label.contains(q) || id.contains(q) {
        return Some(Rank::Substring);
    }
    is_subsequence(q, &label).then_some(Rank::Subsequence)
}

fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut chars = haystack.chars();
    needle.chars().all(|c| chars.any(|h| h == c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_keymap::{default_macos, Context, Resolution, Resolver};

    fn press(focus: Context, key: Key, modifiers: Modifiers) -> Resolution {
        let map = default_macos();
        let chord = chord_from_key(&key, modifiers).expect("translatable key");
        Resolver::new().press(&map, focus, chord)
    }

    fn ch(s: &str) -> Key {
        Key::Character(s.into())
    }

    #[test]
    fn a_bare_letter_resolves_per_pane() {
        assert_eq!(
            press(Context::Canvas2d, ch("a"), Modifiers::empty()),
            Resolution::Command(CommandId::new("tool.transform_pattern")),
        );
        assert_eq!(
            press(Context::Viewport3d, ch("q"), Modifiers::empty()),
            Resolution::Command(CommandId::new("tool.select_move_3d")),
        );
        assert_eq!(
            press(Context::Canvas2d, ch("q"), Modifiers::empty()),
            Resolution::Command(CommandId::new("tool.select")),
        );
    }

    #[test]
    fn a_bare_letter_in_a_text_field_is_swallowed_but_cmd_s_pierces() {
        assert_eq!(
            press(Context::TextEntry, ch("a"), Modifiers::empty()),
            Resolution::Swallowed,
            "typing `a` in a field must not run the transform tool",
        );
        assert_eq!(
            press(Context::TextEntry, ch("s"), Modifiers::COMMAND),
            Resolution::Command(CommandId::new("file.save")),
        );
        assert_eq!(
            press(Context::TextEntry, ch("k"), Modifiers::COMMAND),
            Resolution::Command(CommandId::new("agent.palette")),
        );
    }

    #[test]
    fn the_unmodified_key_is_what_reaches_the_map() {
        // The whole reason `modified_key` is banned: on a macOS layout ⌥7
        // produces `¶`, and `view.toggle_strain_map` is bound to `alt+7`.
        assert_eq!(
            press(Context::Viewport3d, ch("7"), Modifiers::ALT),
            Resolution::Command(CommandId::new("view.toggle_strain_map")),
        );
        assert_eq!(
            press(Context::Viewport3d, ch("¶"), Modifiers::ALT),
            Resolution::Unbound,
            "the modified key must not resolve — that is the bug we avoid",
        );
    }

    #[test]
    fn named_keys_and_modifier_combinations_translate() {
        assert_eq!(
            press(
                Context::Global,
                Key::Named(key::Named::Space),
                Modifiers::empty()
            ),
            Resolution::Command(CommandId::new("sim.toggle")),
        );
        assert_eq!(
            press(
                Context::Global,
                Key::Named(key::Named::Escape),
                Modifiers::empty()
            ),
            Resolution::Command(CommandId::new("ui.cancel")),
        );
        assert_eq!(
            press(
                Context::Global,
                ch("z"),
                Modifiers::COMMAND | Modifiers::SHIFT
            ),
            Resolution::Command(CommandId::new("edit.redo")),
        );
        // A modifier held on its own is not a chord at all.
        assert!(chord_from_key(&Key::Named(key::Named::Shift), Modifiers::SHIFT).is_none());
        assert!(chord_from_key(&Key::Unidentified, Modifiers::empty()).is_none());
    }

    #[test]
    fn a_sequence_prefix_reports_pending_then_fires() {
        let map = default_macos();
        let mut r = Resolver::new();
        let k = chord_from_key(&ch("k"), Modifiers::empty()).unwrap();
        let p = chord_from_key(&ch("p"), Modifiers::empty()).unwrap();
        assert_eq!(r.press(&map, Context::Canvas2d, k), Resolution::Pending);
        assert_eq!(
            r.press(&map, Context::Canvas2d, p),
            Resolution::Command(CommandId::new("constraint.parallel")),
        );
        assert!(r.pending().is_empty());
    }

    #[test]
    fn the_catalogue_has_one_row_per_command_with_a_guessable_label() {
        let map = default_macos();
        let entries = catalogue(&map);

        let mut ids: Vec<_> = entries.iter().map(|e| e.id.clone()).collect();
        let before = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(before, ids.len(), "a command appears twice in the palette");

        let undo = entries
            .iter()
            .find(|e| e.id.as_str() == "edit.undo")
            .expect("edit.undo is in the default map");
        assert_eq!(undo.label, "Edit · Undo");
        assert_eq!(undo.shortcut.as_deref(), Some("⌘Z"));

        let dart = entries
            .iter()
            .find(|e| e.id.as_str() == "tool.dart")
            .expect("tool.dart is in the default map");
        assert_eq!(dart.label, "Tool · Dart");
    }

    #[test]
    fn an_exact_prefix_ranks_first() {
        let map = default_macos();
        let entries = catalogue(&map);

        // "undo" is a prefix of `Edit · Undo` and a subsequence of several
        // others (e.g. "Tool · Add points" has no u... pick a real check).
        let hits = search(&entries, "undo");
        assert_eq!(
            hits[0].id.as_str(),
            "edit.undo",
            "{:?}",
            &hits[..3.min(hits.len())]
        );

        // "sew" prefixes `Tool · Sewing`, and merely occurs inside
        // `Pattern · Symmetric with sewing`.
        let hits = search(&entries, "sew");
        assert_eq!(hits[0].id.as_str(), "tool.sewing");
        assert!(hits
            .iter()
            .any(|e| e.id.as_str() == "pattern.symmetric_with_sewing"));

        // The group name still finds the group.
        let hits = search(&entries, "constraint");
        assert!(hits.len() >= 12, "found {} constraint commands", hits.len());
        assert!(hits
            .iter()
            .all(|e| e.id.as_str().starts_with("constraint.")));

        // Nonsense matches nothing rather than everything.
        assert!(search(&entries, "zzzqqqxxx").is_empty());
        // An empty query is the whole list.
        assert_eq!(search(&entries, "   ").len(), entries.len());
    }

    #[test]
    fn an_unknown_command_id_is_ignored_not_fatal() {
        // KEYMAP.md §9: "an id the application does not know is logged and
        // ignored, not fatal". `dispatch` is the app's id -> handler table;
        // this is the contract it must honour for a settings file that names a
        // command from a newer build.
        let entries = catalogue(&default_macos());
        let unknown = CommandId::from("nowhere.from_the_future");
        assert!(
            !entries.iter().any(|e| e.id == unknown),
            "test premise: the id really is unknown",
        );

        // The contract is about `dispatch`, so call it. Anything but the `_`
        // arm — a panic, or a state change — fails here.
        let mut app = crate::App::new();
        let before = (
            app.active_tool.clone(),
            app.palette.clone(),
            app.focus,
            app.simulating,
            app.canvas_view,
        );
        let _task = crate::dispatch(&mut app, unknown);
        assert_eq!(
            (
                app.active_tool.clone(),
                app.palette.clone(),
                app.focus,
                app.simulating,
                app.canvas_view
            ),
            before,
            "an unknown id changed application state",
        );
    }

    /// `ui.debug_overlay.enabled` exists to answer "why does nothing happen
    /// when I press this". The overlay is gated on this crate's `debug-overlay`
    /// cargo feature (see `view`), and this crate is the only place that knows
    /// — so the inventory's answer has to agree with it in *every* build, not
    /// just in a release one.
    #[test]
    fn the_inventory_reports_the_overlay_that_was_actually_compiled_in() {
        assert_eq!(
            nguo_registry::feature("ui.debug_overlay")
                .expect("ui.debug_overlay is in the inventory")
                .enabled,
            cfg!(feature = "debug-overlay"),
        );
    }

    #[test]
    fn subsequence_matching_is_in_order() {
        assert!(is_subsequence("tp", "tool · transform pattern"));
        assert!(!is_subsequence("pt", "tp"));
        assert!(is_subsequence("", "anything"));
    }
}
