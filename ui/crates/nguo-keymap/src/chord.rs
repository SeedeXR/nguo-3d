// SPDX-License-Identifier: GPL-2.0-or-later
//! Chords (modifiers + one key) and short sequences of chords.
//!
//! # Platform-neutral modifiers
//! There is no `Cmd` in this crate. [`Mods::PRIMARY`] is the accelerator
//! modifier — `⌘` on macOS, `Ctrl` everywhere else — and [`Mods::SECONDARY`] is
//! the *other* one — `⌃` on macOS, `Super` elsewhere. A keymap file written on
//! macOS is therefore already correct on Linux; only the rendering changes
//! ([`Chord::display`]).
//!
//! # Which key is "the key"
//! [`Key::Char`] means *the character the key produces with no modifiers on the
//! user's layout*. `⌥7` is `alt+7`, never `alt+¶`; `⌘?` is `primary+shift+/`.
//! The application must feed this crate the unmodified logical key — in `iced`
//! that is `keyboard::Event::KeyPressed::key`, not `modified_key`. This crate
//! takes no input-library dependency so it cannot enforce that; the contract is
//! documented in `ui/docs/KEYMAP.md` and tested at the call site.

use std::fmt;
use std::ops::BitOr;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// A set of modifier keys.
///
/// Deliberately not `bitflags`: five constants and two operations do not earn a
/// dependency.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Mods(u8);

impl Mods {
    /// No modifiers.
    pub const NONE: Mods = Mods(0);
    /// The accelerator modifier: `⌘` on macOS, `Ctrl` elsewhere.
    pub const PRIMARY: Mods = Mods(1 << 0);
    /// `⇧`.
    pub const SHIFT: Mods = Mods(1 << 1);
    /// `⌥` / `Alt`.
    pub const ALT: Mods = Mods(1 << 2);
    /// The second modifier: `⌃` on macOS, `Super`/`Win` elsewhere.
    pub const SECONDARY: Mods = Mods(1 << 3);

    /// True if every modifier in `other` is present in `self`.
    #[must_use]
    pub const fn contains(self, other: Mods) -> bool {
        self.0 & other.0 == other.0
    }

    /// True if no modifier is held.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for Mods {
    type Output = Mods;
    fn bitor(self, rhs: Mods) -> Mods {
        Mods(self.0 | rhs.0)
    }
}

/// A key that has no printable character.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Named {
    Space,
    Escape,
    Return,
    Tab,
    Backspace,
    Delete,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    /// `F1` .. `F24`.
    Function(u8),
}

/// The key part of a [`Chord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    /// A printable key, identified by its *unmodified* character, lowercased.
    Char(char),
    Named(Named),
}

/// Modifiers plus one key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Chord {
    pub mods: Mods,
    pub key: Key,
}

impl Chord {
    #[must_use]
    pub const fn new(mods: Mods, key: Key) -> Chord {
        Chord { mods, key }
    }

    /// A chord on a printable key. `c` is lowercased.
    #[must_use]
    pub fn ch(mods: Mods, c: char) -> Chord {
        Chord::new(mods, Key::Char(c.to_ascii_lowercase()))
    }

    /// A chord on a named key.
    #[must_use]
    pub const fn named(mods: Mods, n: Named) -> Chord {
        Chord::new(mods, Key::Named(n))
    }

    /// Render for a menu or tooltip.
    ///
    /// macOS uses the Apple HIG modifier order `⌃⌥⇧⌘` and no separators, so
    /// Cmd+Shift+Z renders `⇧⌘Z` — not `⌘⇧Z`. See `ui/docs/KEYMAP.md`.
    #[must_use]
    pub fn display(&self, platform: Platform) -> String {
        let mut s = String::new();
        match platform {
            Platform::MacOs => {
                if self.mods.contains(Mods::SECONDARY) {
                    s.push('⌃');
                }
                if self.mods.contains(Mods::ALT) {
                    s.push('⌥');
                }
                if self.mods.contains(Mods::SHIFT) {
                    s.push('⇧');
                }
                if self.mods.contains(Mods::PRIMARY) {
                    s.push('⌘');
                }
                s.push_str(&mac_key(self.key));
            }
            Platform::Other => {
                if self.mods.contains(Mods::PRIMARY) {
                    s.push_str("Ctrl+");
                }
                if self.mods.contains(Mods::SECONDARY) {
                    s.push_str("Super+");
                }
                if self.mods.contains(Mods::ALT) {
                    s.push_str("Alt+");
                }
                if self.mods.contains(Mods::SHIFT) {
                    s.push_str("Shift+");
                }
                s.push_str(&other_key(self.key));
            }
        }
        s
    }
}

/// Which glyph set [`Chord::display`] renders with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Platform {
    MacOs,
    /// Windows / Linux. No map is shipped for these yet (§"door open").
    Other,
}

fn mac_key(k: Key) -> String {
    match k {
        Key::Char(c) => c.to_uppercase().to_string(),
        Key::Named(n) => match n {
            Named::Space => "Space".into(),
            Named::Escape => "⎋".into(),
            Named::Return => "↩".into(),
            Named::Tab => "⇥".into(),
            Named::Backspace => "⌫".into(),
            Named::Delete => "⌦".into(),
            Named::Up => "↑".into(),
            Named::Down => "↓".into(),
            Named::Left => "←".into(),
            Named::Right => "→".into(),
            Named::Home => "↖".into(),
            Named::End => "↘".into(),
            Named::PageUp => "⇞".into(),
            Named::PageDown => "⇟".into(),
            Named::Function(n) => format!("F{n}"),
        },
    }
}

fn other_key(k: Key) -> String {
    match k {
        Key::Char(c) => c.to_uppercase().to_string(),
        Key::Named(n) => match n {
            Named::Space => "Space".into(),
            Named::Escape => "Esc".into(),
            Named::Return => "Enter".into(),
            Named::Tab => "Tab".into(),
            Named::Backspace => "Backspace".into(),
            Named::Delete => "Delete".into(),
            Named::Up => "Up".into(),
            Named::Down => "Down".into(),
            Named::Left => "Left".into(),
            Named::Right => "Right".into(),
            Named::Home => "Home".into(),
            Named::End => "End".into(),
            Named::PageUp => "PageUp".into(),
            Named::PageDown => "PageDown".into(),
            Named::Function(n) => format!("F{n}"),
        },
    }
}

/// Why a chord string could not be parsed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChordParseError {
    #[error("empty chord")]
    Empty,
    #[error("`{0}` is not a modifier (expected primary/secondary/alt/shift, or the macOS spellings cmd/ctrl/option)")]
    UnknownModifier(String),
    #[error("`{0}` is not a key name")]
    UnknownKey(String),
    /// `"primary+"` — a modifier with nothing after it. Parsing that as a
    /// chord on `+` would bind a key the user never named, which is how a
    /// truncated settings row becomes a live binding.
    #[error("`{0}` ends in a modifier; a chord needs a key after the last `+`")]
    MissingKey(String),
    #[error("modifier `{0}` given twice")]
    DuplicateModifier(String),
}

fn parse_named(s: &str) -> Option<Named> {
    Some(match s {
        "space" => Named::Space,
        "escape" | "esc" => Named::Escape,
        "return" | "enter" => Named::Return,
        "tab" => Named::Tab,
        "backspace" => Named::Backspace,
        "delete" | "del" => Named::Delete,
        "up" => Named::Up,
        "down" => Named::Down,
        "left" => Named::Left,
        "right" => Named::Right,
        "home" => Named::Home,
        "end" => Named::End,
        "pageup" => Named::PageUp,
        "pagedown" => Named::PageDown,
        _ => {
            let n: u8 = s.strip_prefix('f')?.parse().ok()?;
            if (1..=24).contains(&n) {
                Named::Function(n)
            } else {
                return None;
            }
        }
    })
}

fn named_token(n: Named) -> String {
    match n {
        Named::Space => "space".into(),
        Named::Escape => "escape".into(),
        Named::Return => "return".into(),
        Named::Tab => "tab".into(),
        Named::Backspace => "backspace".into(),
        Named::Delete => "delete".into(),
        Named::Up => "up".into(),
        Named::Down => "down".into(),
        Named::Left => "left".into(),
        Named::Right => "right".into(),
        Named::Home => "home".into(),
        Named::End => "end".into(),
        Named::PageUp => "pageup".into(),
        Named::PageDown => "pagedown".into(),
        Named::Function(n) => format!("f{n}"),
    }
}

impl FromStr for Chord {
    type Err = ChordParseError;

    /// `"primary+shift+z"`, `"alt+7"`, `"]"`, `"space"`.
    ///
    /// `cmd`/`command` are accepted as macOS spellings of `primary`, and
    /// `ctrl`/`control` of `secondary`; `opt`/`option` of `alt`. Canonical
    /// output always uses the neutral names.
    fn from_str(s: &str) -> Result<Chord, ChordParseError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(ChordParseError::Empty);
        }
        if s == "+" {
            return Ok(Chord::ch(Mods::NONE, '+'));
        }
        // A `"++"` tail means the key itself is '+' ("primary++"). A *single*
        // trailing '+' is a truncated chord ("primary+"), not a chord on '+'.
        let plus_key = s.ends_with("++");
        let mut parts: Vec<&str> = s.split('+').collect();
        let key_tok = if plus_key {
            // "primary++" splits to ["primary", "", ""]; drop the two empties
            // and use '+' as the key.
            parts.pop();
            parts.pop();
            "+"
        } else {
            // `s` is non-empty so there is always at least one part.
            parts.pop().unwrap_or("")
        };
        if key_tok.is_empty() {
            return Err(ChordParseError::MissingKey(s.to_string()));
        }

        let mut mods = Mods::NONE;
        for p in parts {
            let m = match p.to_ascii_lowercase().as_str() {
                "primary" | "cmd" | "command" => Mods::PRIMARY,
                "shift" => Mods::SHIFT,
                "alt" | "opt" | "option" => Mods::ALT,
                "secondary" | "ctrl" | "control" => Mods::SECONDARY,
                other => return Err(ChordParseError::UnknownModifier(other.to_string())),
            };
            if mods.contains(m) {
                return Err(ChordParseError::DuplicateModifier(p.to_string()));
            }
            mods = mods | m;
        }

        let key = if key_tok.chars().count() == 1 {
            let c = key_tok.chars().next().unwrap_or(' ');
            Key::Char(c.to_ascii_lowercase())
        } else {
            let lower = key_tok.to_ascii_lowercase();
            Key::Named(
                parse_named(&lower).ok_or_else(|| ChordParseError::UnknownKey(key_tok.into()))?,
            )
        };
        Ok(Chord { mods, key })
    }
}

impl fmt::Display for Chord {
    /// The canonical serialised form: `primary+secondary+alt+shift+key`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mods.contains(Mods::PRIMARY) {
            f.write_str("primary+")?;
        }
        if self.mods.contains(Mods::SECONDARY) {
            f.write_str("secondary+")?;
        }
        if self.mods.contains(Mods::ALT) {
            f.write_str("alt+")?;
        }
        if self.mods.contains(Mods::SHIFT) {
            f.write_str("shift+")?;
        }
        match self.key {
            Key::Char(c) => write!(f, "{c}"),
            Key::Named(n) => f.write_str(&named_token(n)),
        }
    }
}

/// One or more chords pressed in order: `"j"`, or `"k c"` (press `K`, then `C`).
///
/// Sequences exist because the parametric constraint layer needs ~12 bindings
/// and there are not 12 spare top-level letters. See `ui/docs/KEYMAP.md`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Sequence(Vec<Chord>);

impl Sequence {
    /// Fails only if `chords` is empty.
    pub fn new(chords: Vec<Chord>) -> Result<Sequence, ChordParseError> {
        if chords.is_empty() {
            return Err(ChordParseError::Empty);
        }
        Ok(Sequence(chords))
    }

    /// A one-chord sequence.
    #[must_use]
    pub fn single(c: Chord) -> Sequence {
        Sequence(vec![c])
    }

    #[must_use]
    pub fn chords(&self) -> &[Chord] {
        &self.0
    }

    /// The first chord — the one that decides whether a context lets the
    /// sequence through (see [`crate::Context`]).
    #[must_use]
    pub fn first(&self) -> Chord {
        // `Sequence` cannot be constructed empty.
        self.0[0]
    }

    /// Render for a menu or tooltip: `⇧⌘Z`, or `K C` for a sequence.
    #[must_use]
    pub fn display(&self, platform: Platform) -> String {
        self.0
            .iter()
            .map(|c| c.display(platform))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl FromStr for Sequence {
    type Err = ChordParseError;
    fn from_str(s: &str) -> Result<Sequence, ChordParseError> {
        let chords = s
            .split_whitespace()
            .map(Chord::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Sequence::new(chords)
    }
}

impl TryFrom<String> for Sequence {
    type Error = ChordParseError;
    fn try_from(s: String) -> Result<Sequence, ChordParseError> {
        s.parse()
    }
}

impl From<Sequence> for String {
    fn from(s: Sequence) -> String {
        s.to_string()
    }
}

impl fmt::Display for Sequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, c) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_round_trips() {
        for s in [
            "primary+shift+z",
            "alt+7",
            "]",
            "space",
            "escape",
            "primary+,",
            "k c",
            "f12",
            "primary+secondary+alt+shift+a",
        ] {
            let seq: Sequence = s.parse().expect(s);
            assert_eq!(seq.to_string(), s, "canonical form drifted for {s}");
        }
    }

    #[test]
    fn accepts_macos_spellings_and_normalises() {
        assert_eq!(
            "cmd+Shift+Z".parse::<Chord>().unwrap(),
            "primary+shift+z".parse::<Chord>().unwrap()
        );
        assert_eq!(
            "option+w".parse::<Chord>().unwrap(),
            "alt+w".parse::<Chord>().unwrap()
        );
        assert_eq!(
            "control+k".parse::<Chord>().unwrap(),
            "secondary+k".parse::<Chord>().unwrap()
        );
    }

    #[test]
    fn plus_is_a_key() {
        let c: Chord = "primary++".parse().unwrap();
        assert_eq!(c, Chord::ch(Mods::PRIMARY, '+'));
        assert_eq!(c.to_string(), "primary++");
    }

    /// A truncated or typo'd settings row must not become a live binding on a
    /// key the user never named. `"primary+"` used to parse as a bare `+` and
    /// `"primary+shift+"` as `primary++`, silently dropping a modifier.
    #[test]
    fn a_chord_ending_in_a_modifier_is_a_parse_error() {
        for s in ["primary+", "shift+", "primary+shift+", "alt+"] {
            let err = s
                .parse::<Chord>()
                .expect_err(&format!("`{s}` parsed as a chord"));
            assert!(
                matches!(err, ChordParseError::MissingKey(_)),
                "`{s}`: {err}"
            );
        }
        // …and the same string is refused inside a sequence.
        assert!("primary+ x".parse::<Sequence>().is_err());
    }

    #[test]
    fn rejects_nonsense() {
        assert!("".parse::<Chord>().is_err());
        assert!("hyper+a".parse::<Chord>().is_err());
        assert!("shift+nosuchkey".parse::<Chord>().is_err());
        assert!("shift+shift+a".parse::<Chord>().is_err());
        assert!("f99".parse::<Chord>().is_err());
    }
}
