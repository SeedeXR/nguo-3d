<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# KEYMAP.md — the nguo-3d keyboard shortcut system

Implemented by `ui/crates/nguo-keymap`. This document is both the design
record and the user reference: if you are a pattern maker looking for "what
key does the dart tool", the tables are for you; if you are an engineer
wiring a command up, read §5–§8 too.

**Everything below is verified against the code.** The tables were generated
from `nguo_keymap::default_macos()`, not typed by hand, and
`tests/defaults.rs` fails if the counts drift.

---

## 1. The one-paragraph version

Muscle memory is the single largest cost of adopting a new professional tool,
so where a letter is already established practice in garment CAD, the default
map keeps it. Where we depart — from that practice or from the platform — the
reason is written down, one section per departure, in §4. On top of that we
bind three things that have no established convention because they are new:
a parametric constraint layer, grading, and an agent. The whole map is data:
one JSON file you can edit, reload, and share.

Every row is a decision. **If you add a row, add its justification. A row
without one does not ship.**

---

## 2. Global bindings

Active everywhere. Reached from the 2D canvas and the 3D viewport by
fallthrough (§6); mostly blocked while a text field has focus (§7).

**These tables are not the keymap** — `crates/nguo-keymap/src/defaults.rs` is,
and it is the only place the full list lives. What is written down here is the
subset of rows that carry a *decision*: a departure from platform or drafting
convention, or a binding we invented. Rows whose only rationale was "the
obvious one" said nothing `defaults.rs` does not already say, and are not
duplicated here.

| Command | macOS | Decision |
|---|---|---|
| `app.quit` | `⌘Q` | Platform. Reserved (§5). |
| `window.close` | `⌘W` | Platform. Reserved. |
| `app.preferences` | `⌘,` | Platform. Reserved. |
| `file.new` | `⌘N` | Platform. Reserved. |
| `file.open` | `⌘O` | Platform. Reserved. |
| `file.save` | `⌘S` | Platform. Reserved. |
| `file.export` | `⇧⌘E` | New. `⌘E` is Solidify (below), so export takes the shifted form. |
| `edit.undo` | `⌘Z` | Platform and universal. Reserved. |
| `edit.redo` | `⇧⌘Z` | **See §4.1.** Reserved. |
| `edit.redo` | `⌘Y` | A second, *removable* alias for Windows habits. |
| `edit.select_all` | `⌘A` | Platform. Distinct from bare `A`. |
| `edit.delete` | `⌫`, `⌦` | Both, because both keyboards exist. |
| `edit.duplicate` | `⇧⌘D` | **Shifted** — `⌘D` goes to Symmetric Pattern; see §4.3. |
| `edit.find` | `⌘F` | **See §4.4.** Reserved. |
| `pattern.symmetric_with_sewing` | `⌘D` | The highest-frequency structural operation gets the unshifted chord. §4.3. |
| `pattern.strengthen` | `⌥H` | **See §4.5** — `⌘H` belongs to the OS. |
| `pattern.freeze` | `⇧⌘K` | **Shifted** — `⌘K` is the palette; see §4.6. |
| `view.focus_selection` | `F` | Global, not per-window: focusing the selection is useful in both panes. |
| `view.zoom_extents_all` | `]` | Global, for the same reason. |
| `view.cycle_pane_focus` | `⇥` | New. Moves focus 2D ↔ 3D. Focus is what selects the context, so this key is how you switch keymaps. |
| `sim.toggle` | `Space` | Reserved — see §4.7 for the pan-vs-simulate call. |
| `agent.palette` | `⌘K` | **New — see §4.6.** |
| `agent.run` | `⌘↩` | New. Submit the prompt in the palette / agent panel. |
| `agent.panel` | `⇧⌘A` | New. Show/hide the agent panel. |
| `ui.cancel` | `⎋` | Platform. Reserved. Cancels a tool, a modal, or a half-typed sequence. |
| `help.shortcuts` | `⌘/` | New. Opens this table in-app. |

38 global bindings in `defaults.rs`; the ones with a decision behind them are
above.

---

## 3. Viewport and canvas bindings

### 3.a 2D pattern canvas

| Command | macOS | Decision |
|---|---|---|
| `tool.select` | `Q` | The 2D counterpart of the 3D select/move tool, which is also `Q` (§3.b). Same finger, same idea, per pane. |
| `tool.dart` | `D` | New. The mnemonic is exact and the key was free. |
| `tool.notch` | `N` | New, on a key freed by §4.2. |
| `tool.measure` | `M` | New, on a key freed by §4.2. |
| `tool.sewing` | `B` | **One key for the whole sewing family — §4.2.** Press again to cycle Edit → Segment → Free. |
| `tool.sewing_prev` | `⇧B` | Cycle the sewing group backwards. |
| `tool.topstitch` | `J` | **One key for the whole topstitch family — §4.2.** Cycles Edit → Segment → Free → Seamline. |
| `tool.topstitch_prev` | `⇧J` | Cycle backwards. |
| `constraint.coincident` | `K` `C` | New — constraint layer, §4.8. |
| `constraint.horizontal` | `K` `H` | New. |
| `constraint.vertical` | `K` `V` | New. |
| `constraint.parallel` | `K` `P` | New. |
| `constraint.perpendicular` | `K` `N` | New. |
| `constraint.tangent` | `K` `T` | New. |
| `constraint.equal` | `K` `E` | New. |
| `constraint.distance` | `K` `D` | New. Dimension. |
| `constraint.angle` | `K` `A` | New. |
| `constraint.symmetric` | `K` `S` | New. |
| `constraint.fix` | `K` `F` | New. Lock a point. |
| `constraint.remove` | `K` `X` | New. Delete constraints on the selection. |
| `constraint.toggle_display` | `⇧K` | New. Show/hide constraint glyphs. |
| `constraint.diagnose` | `⌥K` | New. Degrees-of-freedom / over-constrained report. |
| `grading.next_size` | `⌥]` | New — grading, §4.9. |
| `grading.prev_size` | `⌥[` | New. |
| `grading.toggle_size_run` | `⇧G` | New. Ghost the whole size run over the base. |
| `grading.rules` | `⌥G` | New. Open the grading rule-set editor. |

41 canvas bindings in `defaults.rs`; the ones with a decision behind them are
above.

### 3.b 3D garment viewport

| Command | macOS | Decision |
|---|---|---|
| `tool.pin` | `W` | Conventional. `U` is Create Topology and is not shared with it. |
| `view.reset_arrangement` | `⇧R` | **See §4.4** — `⌘F` is Find. |

13 viewport bindings in `defaults.rs`; the ones with a decision behind them are
above.

### 3.c Modal dialogs

2 bindings, both the platform's: `ui.cancel` on `⎋` and `ui.confirm` on `↩`.
`TextEntry` binds **nothing** — its entire job is to swallow (§7).

**94 bindings total.** `tests/defaults.rs::every_row_is_present_and_accounted_for`
asserts each of these five counts.

---

## 4. The decisions, one by one

### 4.1 Redo is `⇧⌘Z`, not `⌘Y`

`⌘Y` is a Windows convention, and it reaches macOS only by port. On macOS,
Redo is `⇧⌘Z` — it is what the system text stack does, what Apple's HIG
specifies, and what every native app on the machine does. A garment tool that
disagrees will be wrong for the user in every *other* app they use that hour.

So: **`⇧⌘Z` is the default**, and it is what menus print. `⌘Y` is bound too, as
a second chord on the same command, so a Windows habit works on day one.
`⇧⌘Z` is reserved and cannot be reassigned; `⌘Y` can be removed by anyone who
wants `⌘Y` for something else.

### 4.2 One key per tool family, not three

The conventional layout spends four top-level letters on topstitch (Edit /
Segment / Free / Seamline) and three on sewing (Edit / Segment / Free) —
**seven letters on two tool families.**

Those variants are not different tools; they are the same operation at
different granularity. In practice you choose one for a task and stay in it,
so the second, third and fourth letters buy almost nothing and cost the
scarcest resource on the keyboard.

We bind **one key per family and cycle within it**: `B` advances Edit →
Segment → Free Sewing, `J` advances Edit → Segment → Free → Seamline
Topstitch, and `⇧B` / `⇧J` go backwards. The first press of `B` or `J` lands
on the conventional tool, which is the press that muscle memory actually
holds; the rest of the family is one repeat away. That frees `K` `L` `M` `N`
`;`, and `K`, `N` and `M` immediately go to the constraint layer, the notch
tool and the measure tool.

*Engineering note:* the keymap stays stateless. `B` always emits
`tool.sewing`; the tool dispatcher advances the variant when the sewing family
is **already** active. Cycling state belongs to the tool state machine, not to
a lookup table.

### 4.3 `⌘D` is Symmetric Pattern, Duplicate is `⇧⌘D`

`⌘D` = Duplicate is a strong convention in vector tools. But
symmetric-pattern-with-sewing is *the* highest-frequency structural operation
in garment work — most garments are symmetric and every one of those pieces is
made this way — and it is what a pattern maker already expects on `⌘D`. It
keeps the unshifted chord; Duplicate moves one shift key over.

### 4.4 `⌘F` is Find; Reset 3D Arrangement is `⇧R`

`⌘F` is Find in every application on every platform, and we will have search —
over pieces, parameters, seams and the agent's history. Spending it on a
viewport command is a trap. Reset 3D Arrangement takes `⇧R` instead (R for
Reset, in the 3D viewport only, where the command lives). `⌘F` is reserved.

### 4.5 Strengthen is `⌥H`, not `⌘H`

`⌘H` is **Hide Application** on macOS — the window server intercepts it before
any application sees it, so a binding there is either unreliable or requires
suppressing a system shortcut. We do neither: Strengthen takes `⌥H`, same
letter, one modifier over.

### 4.6 `⌘K` is the command palette / agent; Freeze is `⇧⌘K`

`⌘K` is the near-universal "open the command bar" chord in modern software.
A user in 2026 presses `⌘K` expecting a searchable list of everything the app
can do — which for us is also how you reach the agent. That is worth more than
any single pattern property, so Freeze takes `⇧⌘K`.

This also means the palette is the *escape valve* for the whole keymap:
anything not bound is still one `⌘K` away, which is why we can afford to keep
the bound set small.

### 4.7 `Space` simulates; pan is a gesture

The real fork. Vector-illustration tools give `Space` to temporary-pan; garment
tools give it to simulate.

We give it to **simulate**, and this is not a coin flip: pan is already served
by pointer gestures (two-finger trackpad drag, middle-mouse drag), so it does
not need a key at all, while starting and stopping the drape is the
most-pressed action in the product and deserves the biggest key on the
keyboard. `Space` is reserved.

### 4.8 The constraint layer lives behind `K` (a two-key sequence)

`nguo-constraints` (planegcs, see `ui/docs/CONSTRAINTS.md`) needs about twelve
constraint types bound. There are not twelve spare top-level letters, and an
`⌥`+letter layer is unusable on macOS because `⌥` is the dead-key/compose
modifier.

So `K` is a **prefix**: press `K`, then a letter. `K C` coincident, `K P`
parallel, `K N` perpendicular, `K D` distance, and so on. The sub-letters
follow the sketcher conventions (FreeCAD, Onshape, SolidWorks) that anyone
who has used a parametric CAD tool already has, and the status bar shows a
`K —` hint while the prefix is live.

`K` alone is therefore *not* a binding, and the loader enforces that: binding
a command to bare `K` in the 2D canvas is a `PrefixConflict` error, because
that binding could never fire. Sequences are two chords deep; nothing needs
three, and the resolver has no timeout — a prefix is cleared by the next key,
by a change of focus, or by `⎋`, which keeps it deterministic and testable.
(Focus matters because focus can move without a keypress — a click in the 3D
view, the palette opening — and a `K` left over from the 2D canvas would
otherwise eat the first key pressed in the new pane.)

### 4.9 Grading

Grading is a *navigation* problem more than a tool problem: you step through a size run and watch the pattern change. `⌥[` and
`⌥]` step previous/next size, deliberately echoing `⌘[` / `⌘]` for
superimpose so the bracket keys consistently mean "move through an ordered
set". `⇧G` ghosts the whole run over the base, `⌥G` opens the rule-set editor.

### 4.10 Modifier order in menus is `⌃⌥⇧⌘`

Apple's HIG fixes the order Control, Option, Shift, Command. So Cmd+Shift+Z
renders **`⇧⌘Z`**, not `⌘⇧Z`, which is what most third-party cheat sheets
write. We follow Apple, because these strings appear in native menus next to
system items that use Apple's order.

The order affects **display only**. The stored form is
`primary+secondary+alt+shift+key`, which is fixed and platform-neutral, so no
data depends on this choice.

---

## 5. Reserved bindings

These may never mean anything else. `rebind` and `unbind` refuse them, and a
settings file that reassigns or drops one **fails to load** with an error
naming the chord, the command it is reserved for, and why.

| Chord | Command | Why |
|---|---|---|
| `⌘Q` | `app.quit` | macOS reserves ⌘Q for Quit |
| `⌘W` | `window.close` | macOS reserves ⌘W for Close Window |
| `⌘,` | `app.preferences` | macOS reserves ⌘, for Settings |
| `⌘N` | `file.new` | universal New |
| `⌘O` | `file.open` | universal Open |
| `⌘S` | `file.save` | universal Save; rebinding it is a data-loss bug |
| `⌘Z` | `edit.undo` | universal Undo |
| `⇧⌘Z` | `edit.redo` | the macOS Redo chord (`⌘Y` stays as a removable alias) |
| `⌘F` | `edit.find` | universal Find; §4.4 |
| `Space` | `sim.toggle` | simulate; pan is a gesture, §4.7 |
| `⎋` | `ui.cancel` | the universal way out — without it a modal or half-typed sequence is a dead end |

Two further chords are owned by macOS and simply never appear in our map:
`⌘H` (Hide Application, §4.5) and `⌘M` (Minimize). Do not bind them.

Reservation is enforced **across every context**, not just `Global`: binding
`⌘S` to something in the 2D canvas would shadow Save exactly where the user
is working, so that is refused too.

It is also enforced against **sequences that start with a reserved chord**.
`⌘S X` in the 2D canvas binds nothing to `⌘S` there, so an exact-match check
sees no conflict — but `⌘S` now resolves to `Pending` and the file is never
saved. Same data loss, so the same refusal.

---

## 6. Contexts and resolution

Five contexts. The **focused** one is tried first; if it has no match, the
chord may fall through to `Global`.

```
Canvas2d   ─┐
Viewport3d ─┼─► Global        (always falls through)
TextEntry  ─┼─► Global        (only piercing chords, §7)
Modal      ─┘                 (only ⌘Q; ⎋ and ↩ are its own bindings)
```

Consequences worth knowing:

- **The same key differs per pane.** `Q` is the 2D select tool in the canvas
  and the 3D select/move tool in the viewport. Which one fires depends purely
  on which pane has focus — `⇥` switches.
- **Shadowing is legal.** A context binding beats a global one on the same
  chord. Nothing in the shipped map does this, but a user remap may.
- A chord bound in a context but unbound globally is simply dead elsewhere; it
  is not an error.

`nguo_keymap::Resolver` returns one of four answers, and the distinction
matters at the call site:

| Result | Meaning | What the app does |
|---|---|---|
| `Command(id)` | fire it | dispatch |
| `Pending` | a live sequence prefix (`K`) | show the hint, wait |
| `Swallowed` | text field or modal owns this chord | **do nothing, pass the event to the widget** |
| `Unbound` | nothing here | pass the event on |

---

## 7. Typing must never fire a shortcut

This is the bug class the context model exists for. Type `waist/4 + 1cm` into
a seam-allowance field and, without a focus model, you have just run four
tools and started a simulation.

`Context::TextEntry` therefore **swallows by default**. A chord reaches the
application from a text field only if:

1. it carries the accelerator modifier (`⌘` on macOS), **and**
2. it is not one the text widget itself owns, **or**
3. it is `⎋`.

So `⌘S` saves while you are mid-word (required behaviour), `⌘Q` quits, `⌘K`
opens the palette, and `⎋` cancels the edit — while every bare key, every
`⇧`+key and every `⌥`+key is swallowed. `⌥` is excluded on purpose: on macOS
it is the compose modifier, so `⌥e` is a dead key producing `´`, not a
shortcut.

**Chords the text widget keeps**, even though they carry `⌘`:

`⌘A` `⌘C` `⌘V` `⌘X` `⌘Z` `⌘Y` `⇧⌘Z` `⇧⌘C` `⇧⌘V`

`⌘Z` in a field must undo the *text*, not the document. `⇧⌘V` must be
paste-and-match-style, not Layer Clone. This is encoded in
`Context::pierces_text_entry`, not as an ad-hoc check at a call site, so
there is exactly one place it can be got wrong and one place it is tested.

**Which key is "the key"?** `Key::Char` means the character the key produces
*with no modifiers* on the user's layout. `⌥7` is `alt+7`, never `alt+¶`;
`⌘?` is `primary+shift+/`. The application must pass the unmodified logical
key — in iced, `keyboard::Event::KeyPressed::key`, **not** `modified_key`.
`nguo-keymap` takes no input-library dependency, so it cannot enforce this;
the contract lives here and at the call site.

---

## 8. Conflicts

A **conflict** is two *different* commands on one chord in one context. It is
detected at construction, which means at every deserialisation, which means a
bad settings file cannot load half-applied:

```
conflict: `a` in canvas2d is bound to both `tool.transform_pattern` and `tool.sabotage`
```

Also detected:

- **Prefix conflicts.** A complete binding that is also the prefix of a
  sequence could never fire. Binding bare `K` in the 2D canvas is refused,
  naming both the binding and the sequence it would swallow.
- **Reserved violations**, in either direction: reassigned (§5) or removed.
- **Version mismatch.** A file from a future format version is refused, not
  guessed at.

Not conflicts, on purpose:

- **One command, several chords.** `edit.redo` has two. Menus print the first
  in table order, which is why `⇧⌘Z` is listed before `⌘Y`.
- **Duplicate identical rows** (same command, context and chord). `Keymap::new`
  keeps the first and drops the rest, so `chords_for` never reports a chord
  twice.
- **Cross-context collisions.** That is the entire point of contexts.

`Keymap::new` itself is the checker — it is the only way to build a `Keymap`,
including through `serde`, so an invalid map is unrepresentable and
`default_macos()` panics on a bad table. `tests/defaults.rs` asserts on the
error for each class rather than re-implementing the scan:
`default_map_is_valid`, `loading_a_conflicting_keymap_names_both_commands`,
`a_binding_that_shadows_a_sequence_prefix_is_a_conflict` and
`an_identical_duplicate_row_is_collapsed`.

---

## 9. Settings format and remapping

The keymap is one JSON document. Save it, edit it, check it into a project,
mail it to a colleague.

```json
{
  "version": 1,
  "bindings": [
    { "command": "app.quit",     "context": "global",   "chord": "primary+q" },
    { "command": "tool.dart",    "context": "canvas2d", "chord": "d" },
    { "command": "constraint.parallel", "context": "canvas2d", "chord": "k p" }
  ]
}
```

**Fields**

- `command` — a dotted id. The application owns the id → handler table; an id
  it does not know is logged and ignored, not fatal.
- `context` — `global` · `canvas2d` · `viewport3d` · `text-entry` · `modal`.
- `chord` — see below. Multi-key sequences are space-separated: `"k p"`.

**Chord syntax.** `modifier+modifier+key`, all lowercase.

| Token | macOS | Windows/Linux |
|---|---|---|
| `primary` | `⌘` | `Ctrl` |
| `shift` | `⇧` | `Shift` |
| `alt` | `⌥` | `Alt` |
| `secondary` | `⌃` | `Super` |

There is **no `cmd` in the data model** — that is what keeps the file
portable. `cmd`/`command`, `ctrl`/`control` and `opt`/`option` are accepted on
input as macOS spellings of `primary`, `secondary` and `alt`, but are
normalised away on save.

The key is either a single character (`a`, `7`, `]`, `,`, `/`) — always the
unmodified one — or a name: `space` `escape` `return` `tab` `backspace`
`delete` `up` `down` `left` `right` `home` `end` `pageup` `pagedown`
`f1`…`f24`.

**Remapping in code**

```rust
let mut map = nguo_keymap::default_macos();
map.rebind("tool.dart".into(), Context::Canvas2d, "shift+d".parse()?)?;   // adds
map.unbind(Context::Canvas2d, &"d".parse()?)?;                            // removes
std::fs::write(path, serde_json::to_string_pretty(&map)?)?;
```

`rebind` **adds** a chord; it does not move one. To move a binding, rebind
then unbind. Both validate the whole map and leave it untouched on error, so
a refused edit cannot corrupt a user's settings.

---

## 10. Other platforms

Only a macOS map ships. Most of the table is already portable because
`primary` is not `⌘` — it is "the accelerator". A Windows/Linux map will still
need its own rows for the genuinely macOS-shaped chords (`⌘,` for settings,
`⌥7`), and `Chord::display(Platform::Other)` already renders `Ctrl+Shift+Z`
rather than `⇧⌘Z`.

It will be written when there is a Windows or Linux build to test it on, and
not before. A speculative map nobody has pressed a key on is not a keymap.

---

## 11. Where things are

| | |
|---|---|
| Chords, parsing, display | `ui/crates/nguo-keymap/src/chord.rs` |
| Contexts, conflicts, resolution | `ui/crates/nguo-keymap/src/keymap.rs` |
| The shipped table and reserved list | `ui/crates/nguo-keymap/src/defaults.rs` |
| Tests | `ui/crates/nguo-keymap/tests/defaults.rs` |
| How commands reach the app | `ui/crates/nguo-app/src/command.rs` |
| Interface spec | `ui/docs/UX_SPEC.md` |
