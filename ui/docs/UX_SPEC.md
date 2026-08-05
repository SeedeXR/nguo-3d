<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# UX_SPEC.md — the nguo-3d interface

This is the design specification for the nguo-3d desktop application: the
layout, the interaction model, and the decisions behind both. It is written as
a record of what we chose and why, so that an implementer can tell a decision
from an accident.

Where this document names a crate, a file or a measured number, that is
verifiable in the tree. Where it states a design position, the reasoning is
here in full — if you disagree with the reasoning, reopen the decision rather
than quietly implementing something else.

---

## 1. The thesis

Two first principles drive the whole interface. Everything in §2 and §3 is
downstream of them.

> **A dimension is an input, not an outcome.**

A pattern maker works from a specification: a 42 cm shoulder, a 4 cm hem
allowance, a quarter-waist plus one centimetre of ease. In a tool where lengths
are *reported*, you approximate that specification by dragging until the
read-out is close enough, and the specification itself is never stored — it
lives in the maker's head and in the paper spec sheet next to the keyboard.

So: **anywhere nguo-3d displays a dimension, that dimension is editable**, and
what you type is retained as the definition of the geometry, not as a one-off
nudge. Fields accept expressions (`waist/4 + 1cm`) evaluated by `nguo-measure`,
which ships with a parameter table, cycle detection and a depth limit. Geometry
is millimetres everywhere internally (`nguo-measure::Length`); the display unit
is a document setting, not a second representation.

> **A seam is a maintained constraint, not a length you check.**

If two edges are sewn together, their lengths are equal — that is what "sewn"
means. Encoding that as a constraint the solver maintains, rather than as a
property you can validate on demand, removes an entire category of user work
and an entire category of file corruption. Equal-length is one of the twelve
constraint types `nguo-constraints` already exposes (`ui/docs/CONSTRAINTS.md`).

The two principles have the same shape: **state the requirement once, let the
solver hold it true.** That is what a parametric pattern layer buys, and it is
not something that can be retrofitted onto an architecture where geometry is
the only stored truth.

**Consequence for first run.** nguo-3d is 2D-pattern-first: the pattern is the
document and the drape is a view of it. The first-run experience must therefore
lead with *type measurements, get a block* — the shortest path from an empty
document to something that is recognisably a garment and recognisably
parametric. Without that, a pattern-first tool with no asset library is a blank
page.

---

## 2. Layout

Single window, dockable panels. **Panels, not modes:** a mode selector hides
half the application from you at all times and forces a round trip whenever a
2D decision depends on a 3D observation, which in garment work is most of
them.

```
┌───────────────────────────────────────────────────────────────────────┐
│ native macOS menu bar                                                 │
├───────────────────────────────────────────────────────────────────────┤
│ toolbar — grouped tools, flyouts, ONE row, never truncating           │
├──────────┬──────────────────────────┬─────────────────┬───────────────┤
│          │                          │                 │               │
│  Scene   │   2D pattern canvas      │  3D viewport    │  Properties   │
│  tree    │   (light #C7C7C7)        │  (dark)         │  + Parameters │
│          │                          │                 │               │
│ garment  │   ← ghosted avatar       │                 │  units on     │
│  └ piece │      silhouette behind   │                 │  EVERY field  │
│    └ edge│                          │                 │               │
├──────────┴──────────────────────────┴─────────────────┴───────────────┤
│ status bar · fps · solver state · active process (nguo-registry)      │
└───────────────────────────────────────────────────────────────────────┘
```

Implemented by `nguo-app/src/shell.rs` over the dock/tree/toolbar widgets in
`nguo-ui-widgets`, with every colour from `nguo-ui-theme`.

**A light 2D canvas inside a dark shell.** Pattern drafting is ink on paper.
Dark-on-light is how markers, gradings and seam allowances have been read for a
century, and inverting it makes a printed pattern and its on-screen source look
like two different objects. The 3D viewport stays dark so that the garment is
the only saturated thing on screen — which is also why `nguo-ui-theme`'s canvas
entity roles are the only chromatic tokens outside `accent` and the status
colours.

**Two accents, two meanings.** Blue means *this tool is active*; amber means
*this display toggle is engaged*. They are distinct semantic tokens, not one
accent hue reused, because "what am I about to draw with" and "what am I
looking at" are different questions and a user answers them at different
moments.

**A canvas minimum width, enforced.** Panels must never be able to starve the
work area. `nguo-app::shell::CANVAS_MIN` is 320 logical px — roughly where a
pattern piece stops being editable rather than merely visible — and
`enforce_canvas_minimum` makes the side panels yield first when the window is
too narrow to satisfy everyone. Tested by
`side_panels_yield_before_the_canvases_do`.

That rule lives in the **application**, not in `nguo-ui-widgets`, and the
distinction is load-bearing: the dock's own `MIN_PANEL` is *symmetric* — every
child is equally entitled to it — whereas this says *these two children are the
work and the other two are chrome*. Priority between panels is a product
decision, so the dock exposes `Split::set_ratios` and the app supplies the
policy. `set_ratios` re-derives the floor from the user's own ratios on every
resize rather than correcting on top of its last correction, which is what stops
the layout drifting a little further each time the window changes size.

**Every hit target comes from a token, never from padding.** Icon-only
affordances are the ones that go wrong, because a glyph size reads as a control
size. Measured before the fix: the scene tree's disclosure triangle was 16×16
(`padding(0)` around a 16 px glyph), and the dock's collapse chevron and the
toast's dismiss were 20×20 (`padding(space.xs)` around one) — all three below
the 24 px floor. They now route through `nguo_ui_widgets::icon_button`, which
sizes itself from `Metrics::control_height` (26 px), and
`nguo_ui_theme::tests::scales_are_monotonic` pins that token at ≥ 24. Adding a
new icon button by hand reintroduces the bug; use the helper.

**The shipped column ratios come from the references, not from taste.** Both
CLO3D and Marvelous Designer land on a ~22.5 % right dock with a narrow left
rail, and split the two work areas almost exactly evenly (32.3 / 32.2 %) — see
the session-2 captures under `instructions/ui-references/`. Ours were 17 / 18:
a left rail twice as wide as either reference, and a properties column too
narrow to hold a number field. **3D sits left of 2D**, as it does in both.

**Coupled model, uncoupled view.** Each canvas has its own camera, but both
index the same document entities by the same stable ids, so selecting a 2D edge
highlights the same seam in 3D at no structural cost.

**Ghost the avatar silhouette behind the 2D pattern.** One sprite derived from
the same avatar that the drape uses, so it cannot drift out of agreement with
the 3D result. Highest value per unit of effort of anything in this document.

---

## 3. Decisions

### 3.1 The parametric core

- **Every length is an input** (§1), accepting expressions via `nguo-measure`.
- **A measurement set is a first-class document object.** A body is a named set
  of measurements, not a scale factor on a mesh — because the whole point is
  that one block generates every size and every made-to-measure fit.
- **A seam is an equal-length constraint** the solver maintains (§1).
- **Fabric is measured material.** Every fabric field carries a unit and a
  plausible range. Presets first, raw coefficients behind a disclosure. A
  dimensionless bending number between 0 and 100 is a knob, not a measurement,
  and it cannot be transferred between a swatch, a supplier and a simulation.

### 3.2 Discoverability

- **Command palette (`⌘K`).** One command registry already backs undo/redo, and
  the keymap binds `CommandId`s rather than closures, so a palette over that
  registry is close to free — and it means nothing in the application is
  unreachable just because it did not win a keyboard letter. `⌘K` is also how
  you reach the agent (`ui/docs/AGENTIC.md`).
- **A real scene tree**: garment → piece → edge, with per-class filters layered
  on top rather than replacing the hierarchy. Flat per-class lists stop being
  navigable at the point a garment has thirty panels, which is most garments.
- **One command registry serving palette, keymap, menus, undo, CLI and MCP.**
  Five front doors, one implementation. This is what makes the agent a
  first-class user rather than a bolt-on: an agent drives exactly the commands
  a human drives, so there is no second surface to keep in step and no
  capability that exists only in one of them.

### 3.3 Interaction rules

- **Keyboard scope follows selection type, not window.** A key whose meaning
  changes with which pane last received a click, distinguished only by a focus
  ring, is a key you cannot trust. `nguo-keymap` keeps `Canvas2d` and
  `Viewport3d` contexts, but a binding may only differ between them when the
  *command* is genuinely view-specific (`ui/docs/KEYMAP.md` §6).
- **Long press is never the only route to a tool.** Every tool has a keystroke
  and a palette entry as well.
- **Orbit does not share a button with the context menu.**
- **One toolbar, labelled or tooltipped**, overflowing into a menu. Never a
  truncation chevron at the default window size, and never four simultaneous
  rows of unlabelled glyphs — icon-only chrome is a memory test, and
  `ui/assets/icons/LICENSE.md` lists a deliberately small vocabulary for
  exactly that reason.
- **A schematic dress form, not a photoreal human.** Our user is accountable to
  a factory, not to a render. Photorealism costs us fidelity where it matters
  (joint landmarks, girth measurements, posture) and buys presentation value we
  are not competing on.
- **No cloud or account chrome.** nguo-3d is single-user and collects no
  telemetry of any kind. There is nothing to sign into, so there is nothing to
  render.

### 3.4 Two places we intend to go further

**a. The 2D and 3D views are one model with two presentations.**
Both views already index the same stable ids, so we can go past
cross-highlighting (§2) to **direct cross-view editing**: drag a seam in 3D and
the corresponding 2D edge updates, because the seam is one entity. Two views
that can silently disagree are the reason validation commands exist; ours
cannot disagree.

**b. Undo covers the simulation, not just the geometry.**
Our drape is deterministic, so a drape state is fully reproducible from its
inputs and belongs in the undo history like any other edit. "I simulated and it
got worse" must have a way back. This is available to us *because* determinism
is a hard requirement of the engine rather than a nice-to-have, and it is not
available to a non-deterministic solver at any price.

### 3.5 Consequences specified elsewhere

Listed so the set is visible in one place; each is fully specified in the named
document, and none of them is a new decision.

- **Honest progress everywhere.** Solves report real residual and iteration
  counts, not a spinner. `ui/docs/SPLASH.md` for the startup case.
- **Failure is a diagnostic, not a dead end.** `nguo-registry` keeps the error
  chain, `nguo-ui-widgets` has the toast, and `nguo-constraints` reports
  conflicting and redundant constraint sets rather than just refusing to solve
  (`ui/docs/CONSTRAINTS.md`, `ui/docs/OBSERVABILITY.md`).
- **Discoverability without a printed cheat sheet** — §3.2, the palette.
- **The agent as a first-class user** — `ui/docs/AGENTIC.md`.
- **Measured fabric, not folklore** — §3.1.

---

## 4. Known gaps — acquire deliberately, do not guess

Open questions that should be answered by a decision, not by whatever the first
implementation happens to do:

1. **How sewing relationships are drawn on the 2D canvas.** The *model* is
   settled (M:N segment sewing and M:N free sewing); the rendering is not. What
   distinguishes a seam from an outline at a glance, and at what zoom.
2. **Notch rendering.** Darts and internal lines are three visually distinct
   classes; notches need their own and it is not yet designed.
3. **Strain map and wireframe presentation**, and how arrangement points and
   pins are shown in the 3D viewport.
4. **The simulation-tier selector.** Two tiers exist in the bridge; how the
   user chooses, and how the high-fidelity tier gets forced at export and
   record time, is unspecified.

---

## 5. Immediate build order

1. Assemble the shell: dockable panels, the layout in §2, wired to the existing
   `nguo-ui-widgets` dock/tree/toolbar and `nguo-ui-theme` tokens.
2. Splash → shell handoff with real staged progress (`ui/docs/SPLASH.md`).
3. Command palette over the command registry (§3.2) — cheap, and it makes
   everything else discoverable while the tool set is still incomplete.
4. Editable-dimension property rows using `nguo-measure` expressions (§3.1).
5. Ghosted avatar silhouette behind the 2D canvas (§2).

Everything above §5 is decided. Anything not listed here is not yet decided,
and should be raised rather than assumed.
