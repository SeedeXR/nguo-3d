// SPDX-License-Identifier: GPL-2.0-or-later
//! The nguo-3d keyboard shortcut system (ui/docs/KEYMAP.md).
//!
//! Muscle memory is the largest switching cost a professional 3D garment tool
//! has, so this crate is deliberately boring and deliberately strict.
//!
//! # Four things it does
//! 1. **Data, not code.** A [`Keymap`] is a `Vec<Binding>` that serialises to
//!    JSON and loads back. There is no `match` statement anywhere in the
//!    dispatch path. The shipped map is a table in [`defaults::DEFAULT_MACOS`]
//!    parsed by the same code that reads a user's settings file.
//! 2. **Contexts.** The same key means different things in the 2D canvas, the
//!    3D viewport and a text field. [`Context::TextEntry`] swallows almost
//!    everything, so a shortcut cannot fire while the user types in a numeric
//!    or expression field — the bug class this model exists to remove.
//! 3. **Conflict detection.** [`Keymap::new`] — and therefore every
//!    deserialisation — rejects two commands on one chord in one context, and
//!    names both. A remappable keymap without this is a bug generator.
//! 4. **Display.** [`Sequence::display`] renders `⇧⌘Z` for menus and tooltips.
//!
//! # Quick start
//! ```
//! use nguo_keymap::{default_macos, Chord, CommandId, Context, Mods, Resolution, Resolver};
//!
//! let map = default_macos();
//! let mut r = Resolver::new();
//!
//! // A bare `A` in the 2D canvas is the transform tool...
//! assert_eq!(
//!     r.press(&map, Context::Canvas2d, Chord::ch(Mods::NONE, 'a')),
//!     Resolution::Command(CommandId::new("tool.transform_pattern")),
//! );
//! // ...and nothing at all while the user is typing.
//! assert_eq!(
//!     r.press(&map, Context::TextEntry, Chord::ch(Mods::NONE, 'a')),
//!     Resolution::Swallowed,
//! );
//! // But ⌘S still saves mid-word.
//! assert_eq!(
//!     r.press(&map, Context::TextEntry, Chord::ch(Mods::PRIMARY, 's')),
//!     Resolution::Command(CommandId::new("file.save")),
//! );
//! ```
//!
//! # What is *not* here
//! No `winit`/`iced` dependency: the app translates its own key events into a
//! [`Chord`] (see [`chord`] for the "which key is the key" contract). No
//! timeouts on multi-key sequences — [`Resolver`] is driven entirely by
//! keypresses, which keeps it deterministic and testable.
//!
//! # Where the defaults come from
//! The shipped map follows the letter-tool conventions established in
//! industry-standard garment CAD where they serve the user, and platform
//! conventions (Apple's HIG on macOS) everywhere else. Every row that carries
//! a decision is justified in `ui/docs/KEYMAP.md`.

#![forbid(unsafe_code)]

pub mod chord;
pub mod defaults;
pub mod keymap;

pub use chord::{Chord, ChordParseError, Key, Mods, Named, Platform, Sequence};
pub use defaults::{default_macos, try_default_macos, DEFAULT_MACOS, RESERVED};
pub use keymap::{
    Binding, CommandId, Context, Keymap, KeymapError, Reserved, Resolution, Resolver,
    KEYMAP_VERSION,
};
