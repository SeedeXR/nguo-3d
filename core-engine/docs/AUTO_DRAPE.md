<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# AUTO_DRAPE.md — the one-click drape layer, and its ClothDrop-Reborn lineage

Implementation: `crates/engine-api/src/auto_drape.rs`.
Tests: same file (`#[cfg(test)] mod tests`, 8 unit tests) plus
`crates/engine-api/tests/regression.rs::auto_drape_preset_scene_steps_and_stays_out_of_the_target`.

## 1. What it does

```rust
let scene = engine_api::auto_drape(&target_mesh, DrapePreset::Normal, FabricPreset::Cotton)?;
let mut solver = scene.build_implicit();
```

Given a target collider mesh and a fabric, `auto_drape` returns a ready-to-simulate
`Scene`: a triangulated, UV'd cloth plane sized to the target's bounding box with
overhang, placed 1 mm above its highest point, with contact margins, substeps,
a ground plane at the target's base, and a settle-frame estimate. No manual sim
setup, no keyframing — the product behaviour ClothDrop-Reborn exposes as one button.

The target is carried into the scene as a **static** object. `ColliderSpec` has only
`Plane` and `Sphere` (`crates/engine-api/src/scene.rs`), so an arbitrary triangle mesh
cannot become an analytic collider; cloth-versus-target contact therefore runs through
the implicit tier's self-collision over the concatenated mesh. **Use `build_implicit()`**
— the XPBD preview tier (`build()`) has no self-collision and the cloth will pass
straight through the target.

## 2. Licensing position

**ClothDrop-Reborn** is a third-party Blender cloth-drop add-on licensed
**GPL-3.0-or-later**. This tree is **GPL-2.0-or-later**. Its source is not
vendored or redistributed here; the citations below are `file:line` into that
upstream project, kept so every value carried across is auditable at its
source.

GPL-2.0-or-later code can be combined into a GPLv3 work, but not the reverse: pulling
GPLv3 source into this tree would force the whole engine to GPLv3-only-or-later and
strip downstream users of the "or version 2" option. So:

- ClothDrop was read as a **behaviour/design reference only**. No code, no docstrings,
  and no data tables were copied.
- What crossed over is **factual parameter values** (a subdivision count, a frame
  budget, a 1 mm offset) and the product concept. Individual uncreative constants are
  not the protectable part of the work, and each is re-derived below with its source
  line so the provenance is auditable rather than laundered.
- Every number carried is cited in `auto_drape.rs` at its point of use, in the form
  `presets.py:5` / `utils.py:307`, relative to ClothDrop-Reborn's `src/`.
- If we ever ship an actual Blender add-on, that add-on may be GPL-3.0-or-later; this
  Rust crate stays GPL-2.0-or-later.

## 3. ClothDrop's preset table (verbatim, `presets.py:1-35`)

| Preset | `subdivision` | `subsurf` | `wrinkles` | `folds` | `weight` |
|---|---|---|---|---|---|
| Simple & Fast | 12 | 0 | 0 | 0 | 2.00 |
| Sharp | 22 | 0 | 75 | 75 | 3.10 |
| Normal | 22 | 1 | 70 | 70 | 3.00 |
| Hard | 6 | 0 | 0 | 0 | 0.60 |

**Finding worth stating plainly:** these are *quality/stiffness* presets, not fabrics —
ClothDrop has no fabric concept at all. That axis is orthogonal to our `FabricPreset`,
which is why `auto_drape` takes both. And `Sharp` vs `Normal` differ **only** in
parameters we do not carry (`subsurf` 0/1, `wrinkles` 75/70, `folds` 75/70,
`weight` 3.10/3.00) — in physics terms that this engine models, they are the same
preset. `DrapePreset::Sharp.subdivision() == DrapePreset::Normal.subdivision()` is
asserted in `preset_resolution_matches_clothdrop_subdivision`.

## 4. Mapping table — ClothDrop param → our param → source line

Axis note: Blender is **Z-up** and ClothDrop drops along −Z; this engine is **Y-up**
(`SolverConfig::gravity` = `[0,-9.8,0]`). All citations are transposed z→y, so
ClothDrop's XY footprint is our XZ.

### 4.1 Carried

| ClothDrop | Value | Source | Our param | Divergence |
|---|---|---|---|---|
| `subdivision` (`number_cuts`) | 12 / 22 / 22 / 6 | `presets.py:5,13,21,29`; applied `utils.py:281` | `DrapePreset::subdivision()` → `verts_per_side()` | None. Dimensionless. N cuts on one quad ⇒ N+1 quads and **N+2 verts** per side. |
| `cs.quality` | 12 | `utils.py:307` | `SolverConfig::substeps` (`CLOTHDROP_QUALITY`) | None — Blender's "Quality Steps" *is* substeps/frame. Our default is 8; ClothDrop pins 12 for every preset. |
| `bakeframe` | 120 | `properties.py:15`; baked `utils.py:20-26` | `settle_frames(fps)` (`CLOTHDROP_BAKE_FRAMES`) | Carried as a **duration**, not a frame count: 120 frames at Blender's default 24 fps = 5.0 s. At our 60 fps default that is 300 frames. |
| `thickness_outer` / `thickness_inner` | 0.001 | `utils.py:200-201`, `:256`, `operators.py:114` | `Material::contact_offset` (`CONTACT_MARGIN`) | None. Metres are metres. |
| `self_distance_min` | 0.001 | `utils.py:309` | `Material::contact_gap` (`CONTACT_MARGIN`) | None. Coincides with our existing `contact_gap` default of 1e-3. |
| `use_self_collision` | `True` | `utils.py:308` | `SolverConfig::self_collision` | None. Also load-bearing here: it is the only mechanism by which cloth sees the static target mesh. |
| Placement height | `best_hit_z + 0.001` | `operators.py:247` (and `:80`) | `PLACEMENT_GAP`, applied as `target_max_y + gap` | z→y. ClothDrop finds the highest point with a 10×10 downward raycast grid (`utils.py:219-242`); we take the target's AABB max Y, which is the same quantity for the drop direction and needs no raycaster. |
| `cloth_friction` (High Friction) | 80.0 | `utils.py:196`, `:253`, `operators.py:111`; default `True` at `properties.py:16` | `Material::friction` (`HIGH_FRICTION_MU` = 1.0) | **Unit change.** Blender's slider is a dimensionless 0..80 factor; ours is a Coulomb coefficient. ClothDrop uses the range maximum, so we carry the ratio 80/80 = 1.0. μ=1.0 is deliberately high — that is the add-on's stated intent ("Helps prevent sliding off surface"), not a measured fabric value. **Currently inert**: `Material::friction` is implicit-tier and neither `build()` nor `build_implicit()` reads it yet. |

### 4.2 Deliberately NOT carried

| ClothDrop | Value | Source | Why not |
|---|---|---|---|
| `weight` → `cs.mass` | 2.00 / 3.10 / 3.00 / 0.60 | `presets.py:8,17,25,33`; `utils.py:303,306` | Blender's cloth `mass` is **kg per vertex**; our `Material::density` is **areal kg/m²**. Converting exactly (`density = weight × n_verts / area`) is possible here — `auto_drape` knows both — but it yields ~392 kg/m² for Simple & Fast on a 1 m² sheet, against 0.06–0.9 kg/m² for real fabric in `material.rs`. Blender's cloth mass is a tuning knob with no physical scale, so the conversion is arithmetically exact and physically meaningless. Density comes from `FabricPreset` instead. |
| `folds` → tension/bending | 0 / 75 / 70 / 0 | `utils.py:294-300` | The formula is `fold_pwr = (folds/10)²`; `tension = max(0.5, 150/(fold_pwr+1))`; `bending = max(0.01, 1/(fold_pwr+0.1))`. So folds 0 → bending 10.0 and folds 70 → bending 0.0204: a **491× swing** in Blender's arbitrary stiffness units. Carrying that ratio onto our `Material::bend` would divide a calibrated bending stiffness by ~491 and destroy it. Bending comes from `FabricPreset`, which is at least ordered against real fabric behaviour and is tested (`silk_drapes_more_than_denim`). |
| `wrinkles` → DISPLACE modifier | 0 / 75 / 70 / 0 | `utils.py:123-146` | Not physics. It is a CLOUDS-texture displacement (`strength = 0.04 + wrinkles/8000`) applied to the **frozen mesh after the bake** — a cosmetic workaround for Blender's cloth solver being too coarse to produce real wrinkles. Our solver gets wrinkles from the bending model. Porting it would add fake noise on top of real geometry. |
| `subsurf` | 0 / 0 / 1 / 0 | `presets.py:6,14,22,30`; `utils.py:316-320` | Subdivision-surface smoothing is a render/display concern, not a simulation input. Out of scope for a `Scene`. |
| `air_damping` | 10.0 / 1.0 | `utils.py:304` | Blender's air damping is not our `air_friction` (params.rs default 0.2); no defensible conversion. |
| `collision_quality` | 5 | `utils.py:310` | Blender collision-solver iteration count with no counterpart in our contact formulation. |
| 10×10 raycast grid | — | `utils.py:221-242` | Finds the highest surface under the cloth footprint. We take the target's AABB max Y directly — same answer for the drop direction, no raycaster needed. |
| Settle **detection** | — | — | **ClothDrop has none.** Nothing in its 902 lines measures residual velocity; it bakes a fixed frame range and freezes the mesh at the last frame (`utils.py:20-26,31-32`). We keep the same fixed-budget contract rather than invent a convergence criterion the reference does not have. `settle_frames()` is documented as an *estimate*. |

### 4.3 Ours, not ClothDrop's

| Ours | Value | Why it is not a citation |
|---|---|---|
| `OVERHANG_FRAC` | 0.25 of the target's largest extent | **ClothDrop never auto-sizes a cloth.** Its drawn-rectangle path takes the size from the user's mouse drag (`operators.py:41-47,86`); its apply path reuses whatever mesh the user already selected and moves only its height (`operators.py:247`). Auto-sizing is the part we had to invent. Scaled by the largest of all three extents rather than just the XZ footprint, because the overhang must be long enough to fall *past* the target's sides — a tall, narrow object needs more cloth than its footprint suggests (`overhang_scales_with_the_tallest_dimension`). |
| `DrapePreset::fabric()` | Cotton / Denim / Cotton / Leather | A judgement call mapping each preset's blurb (`presets.py:40-43`; `Hard` is "good for paper or metals") onto our fabric table. Only a suggested default — callers pass their own fabric. |
| Generated UVs | `u = i/(n-1)`, `v = j/(n-1)` | Our fabric presets are anisotropic: membrane warp/weft and weft strain limiting are meaningless without material-space UVs (`material.rs`, consumed by `Scene::lower` into `SimState::uvs`). Blender's cloth model has no warp/weft axis, so there was nothing to carry. |
| Ground plane at `target_min_y` | — | So a cloth that slides off the target lands instead of falling forever. ClothDrop relies on whatever else is in the user's Blender scene. |

## 5. Testing

Unit (placement/sizing maths, `auto_drape.rs`):

- `aabb_of_empty_is_none_and_of_points_is_tight`
- `cloth_plane_covers_target_xz_with_margin_and_sits_above_it` — the required invariant:
  the plane fully covers the target's XZ extent **with margin on all four sides** and
  every vertex starts strictly above the target's max Y; plus grid topology and UV span.
- `plane_is_flat_and_degenerate_resolution_is_clamped` — `n=0` clamps to a single quad.
- `preset_resolution_matches_clothdrop_subdivision` — pins N+2, and pins the
  Sharp/Normal equivalence finding.
- `settle_frames_preserves_clothdrop_five_seconds` — 24 fps → 120, 60 fps → 300.
- `auto_drape_wires_a_complete_scene` — static-first ordering, contact overrides
  surviving (`preset` must stay `None` or `Scene::lower` discards them), ground plane.
- `auto_drape_rejects_an_empty_target` — `None`, not a panic or a degenerate scene.
- `overhang_scales_with_the_tallest_dimension` — a 0.1 m footprint × 2 m tall target
  still gets a 1.1 m sheet, because the margin follows the largest extent.

Integration (`regression.rs`): builds a preset scene over a closed box, steps it 300×
with the implicit solver, and asserts all-finite, no floor tunneling, no cloth vertex
inside the target, the static target's vertices not moving at all, and that it actually
draped.

**Note on measuring "it draped":** the drape is asserted as *overhang mean Y below
supported mean Y*, not as an absolute drop below the start height. The contact barrier
lifts the whole sheet by roughly `dhat` on contact, which at these scales exceeds the
droop — an absolute-drop assertion measures the barrier, not the drape. This matches
what `scene.rs` already documents: absolute droop is modest at ~0.6 m scale and the
meaningful, tested claim is the *ordering* (`silk_drapes_more_than_denim`).
