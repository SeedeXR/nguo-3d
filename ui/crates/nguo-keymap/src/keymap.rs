// SPDX-License-Identifier: GPL-2.0-or-later
//! Contexts, commands, the keymap itself, and resolution.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::chord::{Chord, ChordParseError, Key, Mods, Named, Platform, Sequence};

/// What a binding invokes.
///
/// A plain string, not an enum, and this crate does **not** depend on
/// `nguo-doc-commands`. Two reasons, both load-bearing:
///
/// 1. `nguo-doc-commands` has no command *identity* to reuse — its unit of work
///    is a runtime `Transaction` (a label plus a `Vec<Edit>` computed from the
///    document), so there is nothing here to compose with.
/// 2. Most bindable actions are not document mutations at all: tool activation,
///    `view.toggle_wireframe`, `view.zoom_extents_all`, `agent.palette`. An
///    enum living in the keymap would have to enumerate the whole application,
///    which inverts the dependency and makes the keymap unbuildable until every
///    feature crate exists.
///
/// The application owns the `CommandId -> handler` table. The cost is that a
/// typo in a user's settings file is a dangling id rather than a compile error;
/// the app should log unknown ids at load (see `ui/docs/KEYMAP.md`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CommandId(Cow<'static, str>);

impl CommandId {
    /// Const-friendly constructor for the shipped table.
    #[must_use]
    pub const fn new(s: &'static str) -> CommandId {
        CommandId(Cow::Borrowed(s))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for CommandId {
    fn from(s: &str) -> CommandId {
        CommandId(Cow::Owned(s.to_string()))
    }
}

impl From<String> for CommandId {
    fn from(s: String) -> CommandId {
        CommandId(Cow::Owned(s))
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where the keyboard focus is. The same chord means different things in each.
///
/// Ordering here is *not* specificity; specificity is fixed by
/// [`Keymap::resolve`]: the focused context is tried first, then `Global`, and
/// [`Context::TextEntry`] / [`Context::Modal`] block that fallthrough for
/// almost everything.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Context {
    /// Application-wide. Reached from any other context that lets the chord
    /// through.
    Global,
    /// The 2D pattern canvas has focus.
    Canvas2d,
    /// The 3D garment viewport has focus.
    Viewport3d,
    /// A text, numeric or expression field has focus.
    ///
    /// This is the bug class the whole context model exists for: typing `4` in
    /// a seam-allowance field must not fire a tool, and typing a space must not
    /// start the simulation. Text entry swallows everything except the chords
    /// in [`Context::pierces_text_entry`].
    TextEntry,
    /// A modal dialog or sheet is up.
    Modal,
}

impl Context {
    /// Chords that reach the application while a text field has focus.
    ///
    /// The rule: a chord pierces iff it carries [`Mods::PRIMARY`] and is not
    /// one the text widget itself owns — plus `Escape`, which must always be
    /// able to cancel the edit. Bare keys, `⇧`-only and `⌥`-only chords never
    /// pierce (`⌥` composes characters on macOS: `⌥e` is a dead key).
    #[must_use]
    pub fn pierces_text_entry(chord: Chord) -> bool {
        if chord == Chord::named(Mods::NONE, Named::Escape) {
            return true;
        }
        chord.mods.contains(Mods::PRIMARY) && !text_widget_owns(chord)
    }

    /// True if a chord unmatched in this context should be tried against
    /// [`Context::Global`].
    #[must_use]
    pub fn falls_through(self, first: Chord) -> bool {
        match self {
            Context::Global => false,
            Context::Canvas2d | Context::Viewport3d => true,
            Context::TextEntry => Context::pierces_text_entry(first),
            // A modal owns the keyboard. Quitting the app still has to work.
            Context::Modal => first == Chord::ch(Mods::PRIMARY, 'q'),
        }
    }

    /// Kebab-case name, matching the serialised form.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Context::Global => "global",
            Context::Canvas2d => "canvas2d",
            Context::Viewport3d => "viewport3d",
            Context::TextEntry => "text-entry",
            Context::Modal => "modal",
        }
    }

    /// Every context, for exhaustive checks.
    pub const ALL: [Context; 5] = [
        Context::Global,
        Context::Canvas2d,
        Context::Viewport3d,
        Context::TextEntry,
        Context::Modal,
    ];
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Chords a focused text widget handles itself, even though they carry
/// `PRIMARY`. Letting these through would break editing (`⌘Z` would undo the
/// *document* mid-word) or fire a pattern command from a text box (`⌘⇧V` is
/// Layer Clone on the canvas and paste-and-match-style in a field).
fn text_widget_owns(c: Chord) -> bool {
    let p = Mods::PRIMARY;
    let ps = Mods::PRIMARY | Mods::SHIFT;
    matches!(c.key, Key::Char(ch) if
        (c.mods == p && matches!(ch, 'a' | 'c' | 'v' | 'x' | 'z' | 'y'))
        || (c.mods == ps && matches!(ch, 'z' | 'c' | 'v')))
}

/// One row of the keymap.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub command: CommandId,
    pub context: Context,
    pub chord: Sequence,
}

impl Binding {
    /// Parse a table row. Used by the shipped default map and by tests.
    pub fn parse(
        command: &'static str,
        context: Context,
        chord: &str,
    ) -> Result<Binding, KeymapError> {
        Ok(Binding {
            command: CommandId::new(command),
            context,
            chord: chord.parse()?,
        })
    }
}

/// Everything that can be wrong with a keymap.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum KeymapError {
    /// Two commands on one chord in one context. The reason conflict detection
    /// is a feature and not a lint.
    #[error("conflict: `{chord}` in {context} is bound to both `{first}` and `{second}`")]
    Conflict {
        chord: Sequence,
        context: Context,
        first: CommandId,
        second: CommandId,
    },
    /// A complete binding that is also the prefix of a sequence. Pressing it
    /// could never fire, because the resolver would always wait for the next
    /// key.
    #[error(
        "conflict: `{prefix}` in {context} is a complete binding for `{command}` and also the \
         prefix of `{sequence}` (`{longer}`)"
    )]
    PrefixConflict {
        prefix: Sequence,
        sequence: Sequence,
        context: Context,
        command: CommandId,
        longer: CommandId,
    },
    /// A reserved chord was pointed at some other command.
    #[error("`{chord}` is reserved for `{expected}` ({reason}); it cannot be bound to `{actual}` in {context}")]
    ReservedReassigned {
        chord: Sequence,
        expected: CommandId,
        actual: CommandId,
        context: Context,
        reason: &'static str,
    },
    /// A reserved chord was removed. Silently losing `⌘S` is as bad as
    /// silently reassigning it.
    #[error("`{chord}` is reserved for `{expected}` ({reason}) and must stay bound in {context}")]
    ReservedUnbound {
        chord: Sequence,
        expected: CommandId,
        context: Context,
        reason: &'static str,
    },
    #[error("bad chord: {0}")]
    Chord(#[from] ChordParseError),
    #[error(
        "keymap format version {found} is not supported (this build reads version {expected})"
    )]
    Version { found: u32, expected: u32 },
}

/// A reserved chord: it may only ever mean one thing.
#[derive(Clone, Copy, Debug)]
pub struct Reserved {
    /// Canonical serialised chord, e.g. `"primary+s"`.
    pub chord: &'static str,
    /// The only command it may invoke.
    pub command: &'static str,
    /// The context it is reserved in.
    pub context: Context,
    /// Shown to the user when a rebind is refused.
    pub reason: &'static str,
}

/// The wire format. Separate from [`Keymap`] so that deserialisation is forced
/// through validation — you cannot construct a conflicting `Keymap` at all.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct KeymapFile {
    version: u32,
    bindings: Vec<Binding>,
}

/// A validated, context-scoped, remappable keymap.
///
/// Serialises to `{"version": 1, "bindings": [...]}` and deserialises through
/// [`Keymap::new`], so an invalid keymap is unrepresentable: a settings file
/// with two commands on one chord fails to load with an error naming both.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "KeymapFile", into = "KeymapFile")]
pub struct Keymap {
    bindings: Vec<Binding>,
    index: HashMap<(Context, Vec<Chord>), CommandId>,
    prefixes: HashSet<(Context, Vec<Chord>)>,
}

/// Bumped only on a breaking change to the file shape.
pub const KEYMAP_VERSION: u32 = 1;

impl Keymap {
    /// Validate and index a set of bindings.
    ///
    /// # Errors
    /// [`KeymapError::Conflict`], [`KeymapError::PrefixConflict`],
    /// [`KeymapError::ReservedReassigned`] or [`KeymapError::ReservedUnbound`].
    pub fn new(bindings: Vec<Binding>) -> Result<Keymap, KeymapError> {
        let mut index: HashMap<(Context, Vec<Chord>), CommandId> = HashMap::new();
        let mut rows: Vec<Binding> = Vec::with_capacity(bindings.len());
        for b in bindings {
            let key = (b.context, b.chord.chords().to_vec());
            if let Some(first) = index.get(&key) {
                // Two rows on one chord are only fine if they are the same
                // command (a duplicate row); different commands is the bug.
                if first != &b.command {
                    return Err(KeymapError::Conflict {
                        chord: b.chord,
                        context: b.context,
                        first: first.clone(),
                        second: b.command,
                    });
                }
                // An identical repeated row carries no information and is not
                // an error, but keeping it would make `chords_for` report a
                // second chord that does not exist. Keep the first.
                continue;
            }
            index.insert(key, b.command.clone());
            rows.push(b);
        }
        let bindings = rows;

        // A sequence's proper prefixes must not themselves be bindings.
        let mut prefixes: HashSet<(Context, Vec<Chord>)> = HashSet::new();
        for b in &bindings {
            let cs = b.chord.chords();
            for n in 1..cs.len() {
                let p = cs[..n].to_vec();
                if let Some(cmd) = index.get(&(b.context, p.clone())) {
                    return Err(KeymapError::PrefixConflict {
                        prefix: Sequence::new(p)?,
                        sequence: b.chord.clone(),
                        context: b.context,
                        command: cmd.clone(),
                        longer: b.command.clone(),
                    });
                }
                prefixes.insert((b.context, p));
            }
        }

        let map = Keymap {
            bindings,
            index,
            prefixes,
        };
        map.check_reserved()?;
        Ok(map)
    }

    fn check_reserved(&self) -> Result<(), KeymapError> {
        for r in crate::defaults::RESERVED {
            let seq: Sequence = r.chord.parse()?;
            let expected = CommandId::new(r.command);
            let key = seq.chords().to_vec();

            // The reserved chord must be bound to its command in its own
            // context...
            match self.index.get(&(r.context, key.clone())) {
                Some(c) if c == &expected => {}
                Some(other) => {
                    return Err(KeymapError::ReservedReassigned {
                        chord: seq,
                        expected,
                        actual: other.clone(),
                        context: r.context,
                        reason: r.reason,
                    })
                }
                None => {
                    return Err(KeymapError::ReservedUnbound {
                        chord: seq,
                        expected,
                        context: r.context,
                        reason: r.reason,
                    })
                }
            }
            // ...and must not be hijacked by a more specific context, which
            // would shadow it exactly where the user is working.
            for ctx in Context::ALL {
                if ctx == r.context {
                    continue;
                }
                if let Some(other) = self.index.get(&(ctx, key.clone())) {
                    if other != &expected {
                        return Err(KeymapError::ReservedReassigned {
                            chord: seq,
                            expected,
                            actual: other.clone(),
                            context: ctx,
                            reason: r.reason,
                        });
                    }
                }
            }
            // ...and no longer sequence may *start* with it. `primary+s x` in
            // canvas2d leaves nothing bound to `primary+s` there, so the index
            // lookup above sees nothing — but ⌘S now resolves to `Pending` and
            // the file is never saved. Same neutralisation, quieter route.
            if let Some(b) = self.bindings.iter().find(|b| {
                let cs = b.chord.chords();
                cs.len() > key.len() && cs.starts_with(&key)
            }) {
                return Err(KeymapError::ReservedReassigned {
                    chord: seq,
                    expected,
                    actual: b.command.clone(),
                    context: b.context,
                    reason: r.reason,
                });
            }
        }
        Ok(())
    }

    /// All bindings, in table order.
    #[must_use]
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// Resolve a (possibly partial) chord sequence pressed with focus in
    /// `focus`.
    ///
    /// Most specific wins: the focused context is consulted first, then
    /// `Global` if the context lets the chord through.
    #[must_use]
    pub fn resolve(&self, focus: Context, keys: &[Chord]) -> Resolution {
        let Some(&first) = keys.first() else {
            return Resolution::Unbound;
        };
        if let Some(r) = self.lookup(focus, keys) {
            return r;
        }
        if focus.falls_through(first) {
            if let Some(r) = self.lookup(Context::Global, keys) {
                return r;
            }
            return Resolution::Unbound;
        }
        match focus {
            Context::TextEntry | Context::Modal => Resolution::Swallowed,
            _ => Resolution::Unbound,
        }
    }

    fn lookup(&self, ctx: Context, keys: &[Chord]) -> Option<Resolution> {
        // ponytail: one Vec allocation per keystroke. Human typing rates make
        // this free; if a macro player ever hammers it, key the index on a
        // small trie instead.
        let k = (ctx, keys.to_vec());
        if let Some(c) = self.index.get(&k) {
            return Some(Resolution::Command(c.clone()));
        }
        if self.prefixes.contains(&k) {
            return Some(Resolution::Pending);
        }
        None
    }

    /// Every chord bound to `command`, in table order.
    #[must_use]
    pub fn chords_for(&self, command: &CommandId) -> Vec<&Binding> {
        self.bindings
            .iter()
            .filter(|b| &b.command == command)
            .collect()
    }

    /// The chord to print next to `command` in a menu, e.g. `⇧⌘Z`.
    ///
    /// The first binding in table order wins, so the shipped map lists the
    /// preferred chord first and aliases after it.
    #[must_use]
    pub fn display_for(&self, command: &CommandId, platform: Platform) -> Option<String> {
        self.bindings
            .iter()
            .find(|b| &b.command == command)
            .map(|b| b.chord.display(platform))
    }

    /// Point `chord` at `command` in `context`, replacing whatever was there.
    ///
    /// # Errors
    /// The same errors as [`Keymap::new`]; on error the map is left unchanged.
    pub fn rebind(
        &mut self,
        command: CommandId,
        context: Context,
        chord: Sequence,
    ) -> Result<(), KeymapError> {
        let mut next = self.bindings.clone();
        next.retain(|b| !(b.context == context && b.chord == chord));
        next.push(Binding {
            command,
            context,
            chord,
        });
        *self = Keymap::new(next)?;
        Ok(())
    }

    /// Remove a binding. Returns whether anything was removed.
    ///
    /// # Errors
    /// [`KeymapError::ReservedUnbound`] if the chord is reserved.
    pub fn unbind(&mut self, context: Context, chord: &Sequence) -> Result<bool, KeymapError> {
        let mut next = self.bindings.clone();
        let before = next.len();
        next.retain(|b| !(b.context == context && &b.chord == chord));
        let removed = next.len() != before;
        *self = Keymap::new(next)?;
        Ok(removed)
    }
}

impl TryFrom<KeymapFile> for Keymap {
    type Error = KeymapError;
    fn try_from(f: KeymapFile) -> Result<Keymap, KeymapError> {
        if f.version != KEYMAP_VERSION {
            return Err(KeymapError::Version {
                found: f.version,
                expected: KEYMAP_VERSION,
            });
        }
        Keymap::new(f.bindings)
    }
}

impl From<Keymap> for KeymapFile {
    fn from(k: Keymap) -> KeymapFile {
        KeymapFile {
            version: KEYMAP_VERSION,
            bindings: k.bindings,
        }
    }
}

/// What a keypress meant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// Fire this command.
    Command(CommandId),
    /// A prefix of a multi-key sequence; wait for the next key.
    Pending,
    /// The focused widget owns this chord — a text field is up and the chord
    /// does not pierce it. Do nothing, and let the widget have the event.
    Swallowed,
    /// Nothing is bound.
    Unbound,
}

/// Tracks the pending prefix of a multi-key sequence.
///
/// Stateless [`Keymap::resolve`] plus twenty lines here, rather than state
/// inside the keymap: the map stays a pure lookup table and the app cannot
/// forget to clear the prefix. There is no timeout — a dangling prefix is
/// cleared by the next key, by a change of focus, or by [`Resolver::cancel`]
/// (bind that to `Escape`).
#[derive(Clone, Debug, Default)]
pub struct Resolver {
    pending: Vec<Chord>,
    /// The context the pending prefix was typed in. A prefix is only
    /// meaningful there, so a focus change abandons it.
    focus: Option<Context>,
}

impl Resolver {
    #[must_use]
    pub fn new() -> Resolver {
        Resolver::default()
    }

    /// Feed one chord.
    pub fn press(&mut self, map: &Keymap, focus: Context, chord: Chord) -> Resolution {
        // `k` half-typed in the 2D canvas means nothing in the 3D viewport;
        // carrying it across would silently eat the first key pressed there.
        if self.focus != Some(focus) {
            self.pending.clear();
            self.focus = Some(focus);
        }
        self.pending.push(chord);
        let r = map.resolve(focus, &self.pending);
        if r != Resolution::Pending {
            self.pending.clear();
        }
        r
    }

    /// Abandon a half-typed sequence.
    pub fn cancel(&mut self) {
        self.pending.clear();
    }

    /// The keys typed so far, for a "K —" hint in the status bar.
    #[must_use]
    pub fn pending(&self) -> &[Chord] {
        &self.pending
    }
}
