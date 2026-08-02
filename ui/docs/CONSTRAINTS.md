<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# CONSTRAINTS.md — the 2D geometric constraint solver

`nguo-constraints` is safe Rust over FreeCAD's `planegcs`, vendored into this
repository. It is the foundation of the parametric pattern layer: a pattern is
drafted as geometry plus constraints, so changing a body measurement re-solves
the geometry instead of requiring it to be redrawn.

Everything below was **observed on 2026-08-01/02** — a file read, or a command
run with its output — not recalled.

## 0. Status

| | |
|---|---|
| Licence verified against the vendored sources | ✅ LGPL-2.1-or-later, SPDX line 1 of all 11 files |
| C++ builds on macOS/arm64, no manual setup | ✅ Apple clang 21.0.0, `cc` crate, vendored Eigen, no Boost |
| Safe Rust API | ✅ opaque handles, `Result`, `Drop`, every `unsafe` carries a `// SAFETY:` |
| Geometric + dimensional constraints | ✅ all ten geometric, all four dimensional |
| DOF analysis | ✅ under / well / over, plus conflicting and redundant tags |
| Incremental re-solve on parameter change | ✅ no rebuild, no re-diagnosis |
| Tests | ✅ 40 integration + 1 doctest, all green |
| Determinism | ✅ bit-identical across runs, systems and all three algorithms |
| Memory | ✅ ASan + UBSan clean, `leaks` reports 0, over 300 build/solve/drop cycles |
| Not done | curve types beyond line/circle/arc (ellipse, hyperbola, parabola, B-spline exist in planegcs but are not exposed); constraint *removal*; `nguo-pattern` wiring |

---

## 1. Licence position

### What was observed

Every file in `third_party/planegcs/upstream/planegcs/` carries, on line 1:

```
// SPDX-License-Identifier: LGPL-2.1-or-later
```

followed by the LGPL v2.1-or-later boilerplate, © Konstantinos Poulios 2011
(and Victor Titov 2014 for `Geo.cpp`). Full hashes and the commit pin are in
`third_party/planegcs/PROVENANCE.md`.

This moves the licence position from *assumed* to *checked*.

### Why LGPL-2.1-or-later inside a GPL-2.0-or-later work is fine

Two independent routes, either sufficient:

1. **The "or later" upgrade.** LGPL-2.1 §3 lets any recipient convert a copy to
   the plain GPL, "version 2 or any later version". So these sources may be
   treated as GPL-2.0-or-later, which is exactly this tree's licence. Uniform
   result, no mixed-licence question.
2. **Straight compatibility.** Even without converting, LGPL-2.1-or-later is
   GPL-compatible: the FSF lists it as such, and the combined work is
   distributed under the GPL with the LGPL parts remaining LGPL-licensed.

The reverse direction is what would have been a problem — GPL code cannot be
linked into an LGPL library. That is not what we are doing.

### The LGPL's relinking obligation, and how we satisfy it

LGPL-2.1 §6 says that if you distribute a work that uses the library, you must
let the recipient relink the work against a modified version of the library.
For a statically linked binary that normally means shipping object files or a
linkable form.

Here it is satisfied trivially and completely:

- **nguo-3d is itself GPL-2.0-or-later and ships complete corresponding
  source.** LGPL-2.1 §6(a) is met by "accompany the work with the complete
  machine-readable source code for the library". Anyone who has our source has
  planegcs' source, can modify it, and can rebuild.
- **The library sources are in the repository, unmodified, with the upstream
  commit recorded.** `third_party/planegcs/PROVENANCE.md` states the exact
  commit and a SHA-256 per file, so a recipient can diff against upstream and
  see we changed nothing.
- **The build is one command.** `cargo build -p nguo-constraints` recompiles
  planegcs from those sources. There is no prebuilt blob anywhere in the chain.

So vendoring plus source availability does more than §6 requires: the recipient
does not merely get to relink, they get to rebuild from source.

### Eigen

MPL-2.0 primarily, with BSD and LGPL-2.1 parts. MPL-2.0 §3.3 explicitly permits
distribution as part of a Larger Work under a Secondary License, which the MPL
defines to include GPL v2.0 or later. Compatible. Details in PROVENANCE.md §2.

### `deny.toml`

`LGPL-2.1-or-later`, `MPL-2.0` and `GPL-2.0-or-later` are already in the
repository-root `deny.toml` allow list (lines 46-49), so cargo-deny needs no
change. Note that cargo-deny inspects *crate* metadata and will not see
`third_party/`; the licence record for vendored C++ is PROVENANCE.md, not
cargo-deny.

---

## 2. What is vendored, and the Boost finding

Full detail: `third_party/planegcs/PROVENANCE.md`. The two things worth knowing
here:

**Eigen is vendored, not a system dependency.** 6.7 MB of headers at
`third_party/planegcs/vendor/eigen/`. A Homebrew dependency would mean either a
hard-coded `/opt/homebrew` path or a setup step, and it would make the build
depend on whichever Eigen the machine has — planegcs branches on
`EIGEN_VERSION` in three places, so that is a correctness variable. Vendoring
removes it. Eigen is header-only, so there is nothing to link.

**Boost is *not* a dependency, though planegcs still references it.** At the
pinned commit, `GCS.cpp` really does use Boost Graph — `adjacency_list`,
`add_vertex`, `add_edge`, `num_vertices`, `connected_components` — to split the
constraint system into decoupled components inside `diagnose()`. Two other
Boost headers are included but unused. Rather than vendor ~150 MB of Boost or
patch upstream, `third_party/planegcs/compat/` supplies those six symbols as a
weighted union-find. All four Boost includes are angle-bracket, so our headers
simply occupy the include path; real Boost never appears on it.

**No upstream file was modified. There are no patches.**

---

## 3. The shim API

`ui/crates/nguo-constraints/cpp/nguo_gcs.h` — about 30 `extern "C"` functions,
hand-written.

`GCS::System` has roughly 120 public methods and a class hierarchy that runs
`Curve → Circle → Arc`, plus `Ellipse`, `Hyperbola`, `Parabola`, `BSpline`.
Binding all of that is how these layers become unmaintainable. The shim is the
API contract, and it is deliberately small and boring.

`cc` was chosen over `cxx`/`autocxx`: with a hand-written C boundary there is
nothing for a binding generator to do, and it would add a codegen step plus two
build-time crates.

### The upstream API, as actually read (not from memory)

Worth recording, because it is not what one would guess:

- **There is no geometry registry.** planegcs has no `addPoint`/`addLine`/
  `addCircle`. `GCS::Point` is a pair of raw `double*`, `GCS::Line` is two
  `Point`s, `GCS::Circle` is a `Point` plus a `double* rad`. The caller owns
  the storage and planegcs reads through the pointers.
- Unknowns are declared as a flat `std::vector<double*>` via
  `declareUnknowns(VEC_pD&)`.
- `initSolution(Algorithm alg = DogLeg)` builds the subsystem partition and
  calls `diagnose()` internally when the diagnosis is stale.
- `int solve(bool isFine = true, Algorithm alg = DogLeg, bool isRedundantsolving = false)`
  returns a `SolveStatus`: `Success=0`, `Converged=1`, `Failed=2`,
  `SuccessfulSolutionInvalid=3`.
- `applySolution()` writes the solution back; `undoSolution()` restores.
- `int diagnose(Algorithm)`, `int dofsNumber() const`,
  `getConflicting/getRedundant/getPartiallyRedundant(VEC_I&) const`.
- `System` owns every `Constraint` it is given — `~System()` → `clear()` →
  `deleteAllContent(clist)`.
- Public tuning fields: `maxIter`, `convergence`, `qrAlgorithm`,
  `autoChooseAlgorithm`, `debugMode`, and the LM/DogLeg tolerances.

### Ownership

The shim owns everything: a pointer-stable `std::deque<double>` of parameters
(planegcs holds raw pointers into it for the system's whole life), the geometry
objects, and the `GCS::System`. Rust owns exactly one thing: the
`GcsSystem*`.

### Handles

Dense 0-based indices into per-kind arenas. Never recycled, geometry is never
removed, so a stale handle is impossible short of using one from a different
system — and that is range-checked, not dereferenced. Constraint ids are
planegcs *tags*, starting at 1 (tag 0 means "unlabelled" upstream, negative
means "temporary").

### Errors

- Handle-returning functions return `GCS_INVALID_ID` (`0xFFFFFFFF`).
- Constraint-returning functions return a negative `GCS_ERR_*`; success is a
  tag ≥ 1.
- Nothing throws across the boundary. Every entry point that can allocate or
  run Eigen wraps its body in a catch-all; the rest are bounds checks and
  pointer reads.

### Two decisions the shim makes on planegcs' behalf

1. **`gcs_diagnose` returns the DOF through an out-parameter**, not as its
   return value, because planegcs reports an over-determined system as a
   *negative* DOF (`unknowns − independent constraints`). A negative return
   could not be told from an error code. See §5.
2. **`gcs_solve` refuses a system with no free parameters.** planegcs returns
   `Success` for it — vacuously, since there is nothing to move — even when the
   constraints are violated. The shim returns `NotInitialised` instead, because
   a vacuous `Success` is exactly the silent-garbage answer a caller must never
   get.

---

## 4. The safe Rust API

```rust
use nguo_constraints::{System, DofState};

let mut s = System::new();

// Parameters: driving ones are the dimensions you set, free ones are unknowns.
let width = s.add_driving_param(120.0);

// Geometry as plain f64 coordinates.
let a = s.add_fixed_point(0.0, 0.0);
let b = s.add_point(90.0, 4.0);
let line = s.add_line(a, b)?;

// Constraints return a ConstraintId (a planegcs tag).
s.horizontal(line)?;
s.length(line, width)?;

// DOF feedback before solving.
let d = s.diagnose()?;
assert_eq!(d.state(), DofState::WellConstrained);

s.solve()?;
let (x, y) = s.point(b)?;          // (120.0, 0.0) to solver tolerance
# Ok::<(), nguo_constraints::Error>(())
```

- Opaque newtype handles: `ParamId`, `PointId`, `LineId`, `CircleId`, `ArcId`,
  `CurveId` (which `LineId`/`CircleId`/`ArcId` convert into), `ConstraintId`.
- Every fallible operation returns `Result<_, Error>`. No raw pointers in the
  public surface.
- `Drop` frees the C++ system, which frees every constraint, parameter and
  geometry object.
- `Send` but not `Sync`, with the reasoning written out at the `unsafe impl`.
- Every `unsafe` block carries a `// SAFETY:` naming the invariant it relies
  on, and the crate sets `#![deny(unsafe_op_in_unsafe_fn)]`.

### Constraints

Geometric: `coincident`, `horizontal`, `vertical`, `parallel`, `perpendicular`,
`tangent`, `symmetric_about_line`, `symmetric_about_point`, `equal_length`,
`collinear`, `point_on_curve`.

Dimensional: `distance`, `distance_to_line`, `length`, `angle`, `radius`.

Two notes:

- **`collinear` is synthesised.** planegcs has no single constraint for it, so
  the shim emits parallelism plus a point-on-line under one tag. It removes two
  DOF; diagnosis reports it as one constraint.
- **`length` is `distance` between the endpoints.** planegcs has no separate
  constraint and does not need one.

### "distance = parameter" — the point of the whole thing

There is no separate API for it. A dimensional constraint takes a `ParamId`. If
that parameter is *driving*, it is the binding point: give it a value from the
parameter table or a body measurement, and the geometry follows.

```rust
s.length(waist_line, waist_param)?;
s.solve()?;                              // drafted at the base measurement

s.set_param(waist_param, 78.0)?;         // a different body
s.solve()?;                              // same pattern, new size
```

---

## 5. The DOF model

A constraint UI without DOF feedback is unusable, because the user cannot tell
why the sketch will not move. `diagnose()` returns:

```rust
pub struct Diagnosis {
    pub dof: i32,
    pub conflicting: Vec<ConstraintId>,
    pub redundant: Vec<ConstraintId>,
    pub partially_redundant: Vec<ConstraintId>,
}
```

**`dof` is signed, and that matters.** planegcs computes it as
`unknowns − independent constraints`, so an over-determined sketch reports a
*negative* DOF, not zero. This was discovered by observation, not documented
upstream, and it is pinned by a test
(`contradictory_dimensions_are_reported_as_conflicting`).

`Diagnosis::state()` collapses it to what a UI needs:

| | Condition |
|---|---|
| `UnderConstrained { dof }` | `dof > 0`, nothing conflicting or redundant — the sketch can still be dragged |
| `WellConstrained` | `dof == 0`, nothing conflicting or redundant |
| `OverConstrained` | `dof < 0`, **or** any conflicting **or** any redundant constraint |

- **conflicting** — constraints that cannot all hold at once. The sketch is
  broken until one is removed. Highlight them in red.
- **redundant** — implied by the others. Harmless but they hide the sketch's
  real structure. Highlight them differently.
- **partially_redundant** — redundant only in part.

All three are `ConstraintId`s, so they map straight back to whatever the UI
drew. Constraints the shim adds for its own bookkeeping (arc rules) use tag 0
and never appear here.

`System::dof()` returns the last diagnosis. Any structural change — a new
parameter, geometry or constraint — invalidates it and it returns
`Error::NoDiagnosis` until `diagnose()` runs again. Changing a parameter's
*value* does not invalidate it, which is the whole point of the next section.

### Known limit

A sketch with **zero** free parameters cannot be diagnosed meaningfully:
planegcs reports `dof = 0` and no conflicts even when a constraint between two
fixed points is violated. `solve()` catches this (`NotInitialised`), but
`diagnose()` will call it well-constrained. In practice a pattern always has
free geometry, so this only bites on degenerate input.

---

## 6. Incremental re-solve

`set_param` changes a value without touching the structure. The next `solve()`
reuses the existing subsystem partition and DOF analysis instead of rebuilding
them — that is what makes dragging a slider affordable.

```rust
for frame in slider_values {
    s.set_param(width, frame)?;   // no rebuild
    s.solve()?;                   // no re-diagnosis
    redraw(&s);
}
```

Internally the shim keeps a `dirty` flag. Adding geometry or a constraint sets
it; `set_param` does not. `ensureInit` only re-runs `declareUnknowns` +
`initSolution` when dirty.

**Each solve restarts from the geometry as it stood at the last
initialisation**, not from the previous solution — planegcs' `solve()` calls
`resetToReference()` first. That costs a little convergence speed and buys
determinism: dragging a parameter out and back returns *bit-identical*
coordinates, with no accumulated drift. There is a test for exactly that
(`dragging_a_parameter_back_and_forth_returns_the_original_bits`).

---

## 7. Determinism

This project treats determinism as a feature, so it is tested rather than
assumed. Verified on macOS/arm64:

- Two independently built identical systems solve to bit-identical results.
- Re-solving one system eight times gives bit-identical results each time.
- All three algorithms (DogLeg, Levenberg-Marquardt, BFGS) are individually
  reproducible.
- `diagnose()` is reproducible, which was worth checking: planegcs runs part of
  it on a second thread via `std::async`. **No nondeterminism was observed.**

No wall-clock timeout is exposed, deliberately — a time-based cut would make
results depend on machine load. Use `set_max_iterations` for a bounded worst
case.

---

## 8. Robustness

- An unsatisfiable system returns `Err(NoSolution(Converged | Failed))`. On any
  failure the shim calls `undoSolution()`, so the parameters are exactly where
  they started: never NaN, never half-solved.
- A system with nothing to solve returns `Err(NoSolution(NotInitialised))`
  rather than a vacuous success.
- `set_max_iterations(1)` terminates.
- Handles from another system are rejected, not dereferenced.
- Kind mismatches (`radius` on a line, tangency between two lines) return
  `Error::BadKind`.

---

## 9. Tests

`cargo test -p nguo-constraints` — **40 integration tests + 1 doctest, all
green** (`ui/crates/nguo-constraints/tests/solver.rs`), covering: every
constraint type individually against a closed-form expectation; the rectangle
and law-of-cosines triangle sanity cases; DOF correctness for under-, well- and
over-constrained sketches; determinism; parameter-driven re-solve; unsolvable
systems; iteration caps; and 400 build/solve/drop cycles.

### Memory

The pinned toolchain is **stable**, which has no `-Zsanitizer`, so `cargo test`
cannot be sanitised. Rather than install a nightly, the memory check runs
against the layer that actually owns memory — the C++ shim — via
`cpp/asan_smoke.cpp`, which exercises every entry point, every rejection path
and 300 full build/solve/drop cycles. The build command is in that file's
header comment.

Results on macOS 26.5.2 / arm64 / Apple clang 21.0.0:

- `-fsanitize=address,undefined`: **clean**, exit 0, no reports.
- `leaks --atExit`: **0 leaks for 0 total leaked bytes**, peak footprint 3.3 MB
  after 300 cycles — flat, so nothing accumulates.
- LeakSanitizer itself is *not* supported on this platform
  (`detect_leaks is not supported on this platform`), which is why `leaks` is
  used for the leak half.

---

## 10. Integration boundary

`nguo-constraints` is standalone by design. It has one dependency, `thiserror`,
and one build dependency, `cc`. It does **not** depend on `nguo-pattern` or
`nguo-doc`.

The contract for whoever wires it up:

- **In:** plain `f64` coordinates and scalar parameters. Points, lines,
  circles, arcs.
- **Out:** solved `f64` coordinates, plus a `Diagnosis`.
- **The mapping is the caller's.** `nguo-pattern` owns the table from its
  vector-network vertex/edge ids (D-3) to `PointId`/`LineId`, and from its
  document-level constraint entities to `ConstraintId`. That table is what lets
  a DOF diagnosis be drawn back onto the canvas.
- **Parameter binding:** `nguo-measure` evaluates the parameter table and
  expressions; the resulting `f64` goes to `set_param`. This crate never
  evaluates an expression and never knows a parameter's name.
- **Suggested lifecycle:** build the system once per sketch; `diagnose()` after
  each structural edit; `set_param` + `solve()` per drag frame; rebuild only
  when geometry or constraints change.

### Not yet supported, and what it would take

- **Constraint removal.** planegcs has `removeConstraint` and `clearByTag`; the
  shim does not expose them, so an edit means rebuilding the system. Adding
  `gcs_remove_by_tag` is ~15 lines, but handle stability then needs a rule.
- **Ellipse, hyperbola, parabola, B-spline.** All exist in planegcs. B-splines
  in particular will matter for garment curves; each is a new arena kind plus
  its `addConstraint*` overloads.
- **Dragging a point directly** (as opposed to a parameter). planegcs supports
  it through negative-tag temporary constraints
  (`GCS::DefaultTemporaryConstraint`); not exposed.
