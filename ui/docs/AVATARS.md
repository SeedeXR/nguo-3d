<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# AVATARS.md — the body bundle, the loader, and what its numbers mean

The avatar is the body a garment is draped on. It enters the pipeline as **a
mesh plus a measurement set** — deliberately, so nothing downstream knows or
cares where the body came from.

Two things to read before anything else:

1. **Licensing is settled.** The bodies are **Seede XR's own work, licensed
   CC BY 4.0**; so are the skin textures and the Monk Skin Tone Scale.
   **Attribution is required for all three, including to ourselves.** §7.
   There is no provenance blocker any more — an earlier version of this file
   said there was, and that bundle was deleted.
2. **A whole body is steppable — since 2026-08-04.** An earlier version of this
   file said a 15 000-triangle proxy cost 9.4–12.5 s/step and that nothing could
   step one. It now costs **0.235 s/step** (41×, `DECISIONS.md` D-17). Two things
   in §5 still matter: the body is a **static `Object`, not a collider**, so
   `"self-collision"` must be on; and the mesh contact path was **unsigned** until
   2026-08-04, so a garment that started even 10 µm *inside* the body fell through
   it silently. That is **fixed** — the gap against a static body is signed now —
   but the recovery only arrests the garment at the skin, not outside it, so an
   overlapping placement is still not something to author. §5 is the whole story
   and it is still the most consequential thing in this document.

---

## 1. What is in the bundle

`assets/avatars/`. Counted 2026-08-03 with `find assets/avatars -type f | wc -l`
and a `stat -f %z` sum: **70 files, 227.2 MiB**. The 36 binaries — the only part
that cannot drift — are **238 053 569 bytes / 227.03 MiB**; the remaining ~190 KB
is JSON and Markdown, and the Markdown moves whenever these documents are edited,
so it is not quoted to the byte.

That splits into **68 delivered bundle files** plus **2 repo-side files** added
next to them (`LICENSE.md`, `LICENSES/CC-BY-4.0.txt`). Both counts are correct
for what they name; `.gitattributes` §3b says 68 because it is describing the
delivery.

| Path | Count | Size | What |
|---|---|---|---|
| `avatars/glb/` | 6 | 124.6 MiB | `nguo_<sex>.glb` (LOD2), `_lod1.glb`, `_collision.glb` |
| `avatars/vrm/` | 2 | 49.1 MiB | `nguo_<sex>.vrm` — the runtime GLB plus `VRMC_vrm` 1.0 |
| `textures/skin/shared/` | 7 | 29.7 MiB | albedo-neutral, normal (OpenGL + DirectX), roughness, AO, height, subsurface — all 2048², tileable |
| `textures/skin/tones/` | 10 | 18.6 MiB | `skin_albedo_MST01..MST10.png`, pre-tinted |
| `previews/` | 11 | 5.1 MiB | hero / orthos / tones / walk sheets per sex, plus the swatch strip |
| `data/` | 28 | 114.8 KiB | `manifest.json`, `skin_tones.json`, rig / UV / LOD / animation / export / self-test reports |
| `docs/` + `README.md` | 4 | 38.1 KiB | `INTEGRATION.md`, `ATTRIBUTION.md`, `PIPELINE_NOTES.md` |
| `LICENSE.md`, `LICENSES/` | 2 | 32.5 KiB | repo-side: the notice, and the canonical CC BY 4.0 text (18 657 bytes) |

**Two avatars, keyed `male` and `female`** — those strings are the manifest keys
and the API keys. There are no per-body ids like `nguo_female_01` any more.

| key | name | `stature_m` | `arm_span_m` | clips |
|---|---|---|---|---|
| `male` | Nguo Male | 1.83 | 1.90928 | 19 |
| `female` | Nguo Female | 1.90477 | 1.857649 | 19 |

Each avatar also carries 9 landmark girths measured by the pipeline (neck,
upper_chest, chest, underbust, waist, high_hip, hip, thigh, calf) — §6.

### Git LFS, and why it matters more than it looks

`assets/avatars/**/*.{glb,vrm,png}` are LFS-tracked (`.gitattributes` §3b);
the 28 JSON and 4 Markdown files stay in **normal git** so the manifest, the rig
metadata and the licence record all diff in a PR. That is 99.91 % of the bytes
in LFS and 0.09 % in git.

The rules are **path-scoped, never bare-extension**. A bare `*.png` would also
swallow the splash artwork and the icon set, and those are read by
`include_bytes!` at *compile* time — an LFS pointer file would be embedded as
text and the build would succeed with garbage. The avatar bundle is safe to LFS
precisely because it is **runtime-loaded**: a pointer file is recognised as a
pointer (`engine_io::is_lfs_pointer`) rather than fed to a parser. Since
2026-08-04 that means `Bundle::open` fails with `AvatarError::LfsPointer` —
naming `git lfs pull` — and `engine_io::load_mesh` refuses one before format
dispatch. The older `AvatarError::MeshLoad`-from-the-glTF-reader path is gone:
it produced a parse error about glTF grammar for what is really an unfetched
object, and it is what took CI red on `7ce7827`.

**A clone without `git-lfs` gets a tree whose avatars do not load.** That is the
intended failure — loud and typed. `GIT_LFS_SKIP_SMUDGE=1` plus
`NGUO_AVATAR_BUNDLE` pointing elsewhere is the small-clone escape hatch, and
with no bundle at all the app still runs (no body, no picker).

---

## 2. Conventions — and there is no axis conversion anywhere

From `data/manifest.json`:

| | |
|---|---|
| Units | **metres** |
| glTF / VRM space | **+Y up, +Z forward, right-handed** |
| Animation rate | **30 fps** |
| Skeleton | **Mixamo naming, 65 bones**, 38 deforming, **≤ 4 influences per vertex**, T-pose rest |
| `uv0` = `UVMap` | tiling skin detail, **one tile = 0.20 m of skin** → `AddressMode::Repeat` |
| `uv1` = `UVMap_Atlas` | the same unwrap packed into 0..1, for baking |

The glTF block **is** nguo world space ([`INTEROP.md`](INTEROP.md) §2 is the
authority on every conversion in this project), so **no axis conversion happens
at any point on the avatar path**. `Bundle::open` verifies units and space and
refuses a bundle that disagrees (`AvatarError::WrongSpace`,
`../crates/nguo-avatar/src/lib.rs:168`) — silently measuring a Z-up body would
produce plausible nonsense rather than an error.

**The manifest carries three axis blocks and they do not agree with each other**
— `blender` is +Z up / −Y forward / −X character-right, `unity_fbx` is +Y/+Z at
scale 1.0. Read the one matching the format you are consuming. The Blender block
is not a contradiction; it is the same bodies in Blender's space.

Both bodies share the skeleton — same bone names, same T-pose rest — so any clip
plays on either.

Feet sit on the ground plane and the midline is at X = 0. The app relies on
that: `Camera::frame_standing` frames a figure standing with feet at `y == 0`.

**Normals are OpenGL / green-up.** `skin_normal_ogl.png` is correct for glTF,
Blender and Unity as-is. `skin_normal_dx.png` exists for Unreal-style consumers
only; using it anywhere else inverts the lighting in a way that is hard to spot.

---

## 3. The four tiers, and which three actually ship

| `Tier` | male v / t | female v / t | Role | Shipped as GLB |
|---|---|---|---|---|
| `Lod0` | 843 443 / 1 686 890 | 791 487 / 1 582 974 | archival master, hero renders | **no** |
| `Lod1` | 99 998 / 200 000 | 99 999 / 199 998 | high-quality real-time | yes |
| **`Lod2`** | **24 998 / 50 000** | **24 999 / 49 998** | **default runtime avatar** | yes |
| `Col` | 7 498 / 15 000 | 7 500 / 15 000 | cloth collision body, watertight + manifold | yes |

`manifest.default_tier` is **`LOD2`**, and `Tier::Lod2` resolves to the plain
`nguo_<sex>.glb` — the unsuffixed file *is* the LOD2. `Tier::Lod0` is declared
and absent in every bundle so far.

The pipeline's own verification numbers, from `manifest.avatars.<sex>.verification`:

| | male | female |
|---|---|---|
| LOD0 worst vertex vs the source sculpt | **5.07e-7 m** | **5.69e-7 m** |
| Silhouette mismatch fraction, LOD1 / LOD2 / COL | 2.24e-4 / 7.62e-4 / 2.32e-3 | 1.66e-4 / 6.43e-4 / 2.10e-3 |
| Posed collision-proxy deviation | 5.3488 mm | 2.0117 mm |

The first row is why the CC BY 4.0 ownership claim reaches the *shipped bytes*
and not just the `.blend`: the pipeline never moves a surface vertex, so LOD0 is
the Seede XR mesh to float32 rounding at metre scale.

### The manifest over-declares, on purpose

**18 paths are declared across the two avatars; 8 exist.** Absent: both
`.blend`, both `glb.lod0`, all six `unity_fbx`, and the `textures/skin/unity/`
maps `skin_tones.json` names. The manifest describes the whole pipeline output;
the bundle ships the runtime subset.

Nothing is resolved eagerly. `Bundle::open` reads two JSON files and touches no
mesh. Ask before assuming:

```rust
bundle.has_asset(avatar, Asset::GlbLod0)      // false
bundle.has_tier(avatar, Tier::Lod0)           // false
bundle.asset_path(avatar, Asset::GlbLod0)?    // Err(AvatarError::AssetAbsent { .. })
```

`AssetAbsent` names the avatar, the manifest key and the path that was promised,
which is a better failure than an I/O surprise three layers down.

---

## 4. Loading an avatar

`nguo-avatar` (`../crates/nguo-avatar/`) is the only crate that knows the
bundle's layout. It has no UI-framework dependency.

```rust
use nguo_avatar::{Asset, Bundle, Landmark, SkinMap, Tier};

let bundle = Bundle::find()?;                 // $NGUO_AVATAR_BUNDLE, else assets/avatars
let male   = bundle.avatar("male")?;          // manifest key, not an id

// Default runtime body — LOD2, i.e. plain nguo_male.glb.
let body = bundle.load_mesh(male, Asset::GlbRuntime)?;   // engine_api::Object
let col  = bundle.load_mesh(male, Asset::GlbCollision)?; // the cloth collider

// Route A skin: one neutral albedo + a linear factor. §4.1.
let tone   = bundle.skin().tone("MST05")?;
let albedo = bundle.texture(SkinMap::AlbedoNeutral)?;
let factor = tone.factor().to_array();

let m = bundle.measurements(male);
println!("waist {:.0} mm", m.girth(Landmark::Waist).unwrap().circumference.mm());
```

Key facts about this surface:

- **`MANIFEST_PATH` is `data/manifest.json`** — inside `data/`, not at the root.
- `Bundle::open` parses `manifest.json` and `skin_tones.json` once. Everything
  after is path resolution and mesh loading.
- `AvatarError` has 9 variants, each naming the file and what to do about it.
- `load_mesh` goes through `engine_io::load_mesh`, which already dispatches
  `.glb` — the avatar path added no new format support and there is no second
  glTF reader.
- There is **no `MeshKind`**. Pick a file with `Asset`, or a geometry level with
  `Tier`; `Tier::glb()` maps one to the other.

### 4.1 Skin tone: route A, and the honest cost of it

Two routes exist. The bundle documents both; we chose **A**.

| | Route A — colour factor (**chosen**) | Route B — texture swap |
|---|---|---|
| base colour texture | `SkinMap::AlbedoNeutral`, one shared 2K map | `bundle.tone_albedo(tone)?`, one 2K map per tone |
| `baseColorFactor` | `tone.factor()` | `BaseColorFactor::WHITE` |
| cost of a tone change | **four floats** | a 2K texture reload |

The shipped GLB/VRM materials use **route B with `BAKED_TONE = "MST05"`** baked
in, so the files look right the instant any glTF viewer opens them. The app
overrides with route A at load, and the picker opens on MST05 to agree with the
bytes.

**The trap, and it is closed by the type system.** A tone carries three colours
and only one is a factor:

| field | space | use it for |
|---|---|---|
| `Swatch::hex` / `Swatch::srgb` | **sRGB** | painting a picker swatch |
| `MonkTone::base_color_linear` | linear | the published MST colour, reference only |
| `MonkTone::factor()` | linear | `pbrMetallicRoughness.baseColorFactor` |

The factor is the published linear tone **divided by the neutral albedo's mean
luminance, `0.933748`** (`data/skin_tones.json`
`neutral_albedo_mean_linear_luminance`; MST05 checks out exactly —
`0.679542 / 0.933748 = 0.727758`). Feeding sRGB in washes the skin out; feeding
the published linear value in comes out ~7 % dark. `BaseColorFactor` therefore
has **no public constructor from floats** — the intended ways to obtain one are
`MonkTone::factor()` and `BaseColorFactor::WHITE`.

> **The type is not airtight, verified 2026-08-03.** `BaseColorFactor` derives
> `Deserialize` with `#[serde(transparent)]`
> (`../crates/nguo-avatar/src/skin.rs:78-80`), so
> `serde_json::from_str::<BaseColorFactor>("[…]")` builds one from arbitrary
> floats — including an sRGB swatch. The comment above the type says "no
> constructor from floats on purpose", and serde is one. Treat the type as a
> strong convention plus a test, not a proof. Closing it means either
> `#[serde(try_from = …)]` with a range check or a private newtype for the wire
> format; tracked, not fixed.

**The cost we accepted, stated because it is a real error.** The viewport shader
binds **no albedo texture** (`../crates/nguo-viewport3d/src/avatar.wgsl:7`), so
the factor *is* the albedo. Since the factor is pre-divided by 0.933748, the
flat body reads **1 / 0.933748 = ~7 % brighter** than the published Monk tone.
The upgrade path is one change: bind `SkinMap::AlbedoNeutral` on the tiling
`uv0` set (`AddressMode::Repeat`) and the division becomes correct rather than
compensating for nothing.

Only `AlbedoNeutral` is sRGB (`SkinMap::is_srgb`). Uploading a normal or
roughness map as sRGB is the classic way to get subtly wrong lighting
everywhere.

### 4.2 Where the avatar is displayed

- **3D viewport** — `nguo-viewport3d` has an indexed mesh pass beside the cloth
  one: `Vertex` (position + normal, 24 B), `AvatarMesh`, and
  `Avatar { Arc<AvatarMesh>, base_color }`. `avatar.wgsl` is a two-light Lambert
  that gamma-encodes at the end, matching `cloth.wgsl` because iced's surface is
  gamma-space.
- **2D canvas** — a per-scanline silhouette (64 bands) ghosted behind the
  pattern, traced from the **same** `AvatarMesh` the viewport draws, so the two
  cannot drift apart.
- **Pickers** — body and tone, both on the command registry as
  `avatar.body.<male|female>` and `avatar.tone.MST01..MST10`, so both work from
  ⌘K with the properties panel collapsed.

Geometry re-uploads only when `!Arc::ptr_eq`, so **a tone change is a uniform
write**, never a re-upload of 50 k triangles. A test asserts the `Arc` identity
survives all ten tones, and that no tone's factor equals its sRGB swatch or its
published linear value.

`FRAMING_MARGIN` is **1.8** (`../crates/nguo-viewport3d/src/camera.rs:60`), up
from 1.3. A T-posed body is wider than it is tall relative to a 0.59-aspect
pane — the half-span the bundle ships is **0.522 × stature** (male) — and at
1.3 the fingertips were cut off. **1.79 is the numeric floor** at the narrowest
pane the test covers, so 1.8. The consequence to expect: the figure fills only
~56 % of the pane height, and that is the geometry, not slack.

---

## 5. Collision — the headline finding, and its resolution

`engine-api` owns this half, because the solver's own crate is where a body has
to be validated:
[`core-engine/crates/engine-api/src/avatar.rs`](../../core-engine/crates/engine-api/src/avatar.rs).

```
nguo_<sex>_collision.glb ──engine_io::load_mesh──▶ Object
                         ──avatar_collider_near──▶ cropped static Object
                         ──drape_on_avatar───────▶ Scene ──build_implicit──▶ Solver
```

**How a body actually collides.** `ColliderSpec` is analytic-only (`Plane`,
`Sphere`), so a body mesh cannot become one. The body rides into the `Scene` as
a **static object** — `is_static` makes `Scene::lower` pin every vertex, giving
infinite mass and zero integration — and cloth-versus-body contact runs through
the implicit tier's vertex–triangle **self-collision** over the concatenated
mesh. Same mechanism `auto_drape` uses.

**Therefore build with `Scene::build_implicit`.** The XPBD preview tier
(`Scene::build`) has no self-collision at all and the garment passes straight
through the body.

### 5.1 A full-body proxy **is** steppable — since 2026-08-04

**This section said the opposite until 2026-08-04, and the numbers below are
why it changed.** Measured on this machine (Apple Silicon, `--release`,
dt = 1/120, one 36-vertex cloth patch dropped on the body, medians of 3 runs) by
`engine-api/tests/avatar_bundle.rs::collision_proxy_step_cost`:

| body in the scene | scene tris | self-collision | before | after |
|---|---|---|---|---|
| full proxy, male | 15 050 | on | **9.724** | **0.235** |
| full proxy, female | 15 050 | on | **15.951** | **0.245** |
| full proxy, either | 15 050 | **off** | 0.068 | 0.068 |
| region crop of the crown | 306–333 | on | 0.065–0.087 | 0.022–0.026 |

**41×.** The 143× penalty over the self-collision-off case is now **3.5×**,
which is a normal ratio for a cloth solver with self-collision on.
`drape_scene`'s 12 substeps × 300 frames went from **9.4 hours to 14 minutes**.

**The fix was a filter, not a new collider.** A candidate contact pair whose
participants are *all* immovable is provably inert — the Dirichlet filter drops
all four of its Newton rows and all four displacements are exactly zero — so
`engine-core/src/broadphase.rs` stops emitting such pairs, keys the decision on
`inv_mass`, and indexes its spatial grid by the *movable* primitives only. No
`ColliderSpec::Mesh` was added and `engine-api/src/scene.rs` was not touched.
Full reasoning, the profile of what remains, and the two next levers:
`DECISIONS.md` D-17 (engineering memory, not published).

**Decimation was rejected on measurement and stays rejected**, now as history:
the predecessor bundle's proxy was 5 227 triangles at 2.27 s/step; 2.88× the
triangles cost 4.15×, exponent **≈1.35**, which would have put an interactive
whole body at **~520 triangles**. Nobody should ask the asset pipeline for a
coarser proxy.

> **FIXED 2026-08-04. Read the residual.** The mesh contact path used the
> **unsigned** point–triangle distance, so a cloth vertex starting even **10 µm**
> on the far side of a body triangle was pushed further out the wrong side and
> never recovered — the garment ended inside the body, with no diagnostic and no
> non-finite value (measured: mean y **−18.43 m** after 400 steps from a −10 µm
> start).
>
> A body triangle's gap is now **signed** by the triangle's own outward normal
> (`engine_core::contact::signed_point_tri_distance_grad`), and the negative
> branch uses the cubic barrier, which is finite and repulsive through zero. The
> same start now arrests at **−2.7 mm** and stays there. `Scene::build_implicit`
> turns this on for every `is_static` object; the winding is **checked**, and a
> proxy wound inward falls back to the old unsigned behaviour with a
> `tracing::warn!` rather than sucking the cloth in.
>
> **Residual:** the recovery stops at the skin, a few millimetres inside, rather
> than pushing the garment clear the way the analytic collider does. Do not author
> an overlapping placement — the default arrangement clears the body (garment
> radius 0.2481 m against a female maximum of 0.2033 m over the garment's span),
> but an authored `Placement3d::Rigid`, a reduced `gap_m` or a posed limb can
> violate it.

### 5.2 The crop — now an optimisation, not a workaround

```rust
engine_api::avatar::avatar_collider_near(&body, region, margin) -> Result<Object, AvatarError>
```

An **AABB region crop** that keeps whole triangles, remaps indices, and carries
normals, UVs, the rig and the animations through. `region` is normally the
garment's own `Aabb`; `margin` covers how far it will fall. A sleeve, a bodice
or a waistband only ever contacts a fraction of a body, and the crop lands in
the same ~0.05 s/step bracket as everything else.

**Since §5.1's fix this is a speed-up, not a requirement** — the whole body is
steppable at 0.235 s/step. Take it anyway when it is easy: `nguo-app` uses the
same idea as a y-band clip and measures **0.119 → 0.051 s/step, 2.3×, for twenty
lines**.

**Its ceiling, named in the source:** cloth that leaves the region sees no body
and passes through it. Size `margin` to the fall distance, or pass the whole
body's box to opt out. It is `ponytail:`-marked — an AABB is the whole selector,
and bone-weight selection is the upgrade if a garment ever needs a limb rather
than a box.

`avatar_collider(&Object)` is still there for the whole body: it validates a
disk-loaded mesh (a trust boundary — both body and garment come off disk, and a
corrupt or empty file must error rather than panic inside `Scene::lower`) and
lowers it to a static collider. Loading, lowering **and stepping** the full
proxy are all tested now.

### 5.3 Placement

`placement_offset(avatar, garment, placement)` returns a **translation only** —
never a scale or a rotation. There is nothing to rotate (§2), and auto-scaling a
garment to a stature is *grading*, not a uniform factor.
`GarmentPlacement::AsAuthored` leaves the garment's coordinates alone;
`DropFromAbove` centres it in XZ above the avatar's highest point, reusing
`auto_drape`'s rule.

`drape_on_avatar(...)` assembles the ready-to-drape `Scene`: body as static
collider, garment as cloth with the fabric's elastic model and `auto_drape`'s
contact overrides, ground plane at the avatar's feet. Scheduling, self-collision
and the ground plane come from the *same* assembly `auto_drape` uses, so the two
paths cannot drift. Authored pins are kept — that is how a garment hangs from
the shoulders instead of dropping.

---

## 6. Measurements

Every girth in `Measurements` is the **bundle's own published number**, read out
of `manifest.avatars.<sex>.measurements` and converted to millimetres. The
pipeline measured them in Blender against LOD0; we do not re-derive them.

**There is no measurement rig in this crate any more, on purpose.** The
predecessor bundle had no measurements, so `nguo-avatar` sliced the mesh and took
convex-hull perimeters itself — including a `MAX_TORSO_ASPECT` filter to stop a
T-pose chest slice wrapping both arms. This bundle ships the answer, measured off
LOD0 by the pipeline that owns the geometry, with the same hull-perimeter
definition and the same torso restriction. Recomputing it from the *decimated*
runtime mesh would produce a second, worse answer. **That rig is deleted; do not
reintroduce it.**

`Landmark::ALL` is the nine girths: `Neck`, `UpperChest`, `Chest` (bust on the
female body), `Underbust`, `Waist`, `HighHip`, `Hip`, `Thigh`, `Calf`. Each
`Girth` carries `circumference`, `height` above the ground plane — so a waistband
goes at the measured waist, not a guessed fraction of stature — and
`height_fraction`, which is `None` for thigh and calf.

`Measurements` also exposes `stature`, `arm_span`, and `inseam` / `hip_height`
lifted from the `skeleton` block. Those last two are **joint centres, not surface
landmarks**, which is why they are named separately rather than sitting in
`Landmark`. `femur_m`, `tibia_m`, `upper_arm_m` and `forearm_m` stay in the raw
manifest for the same reason.

`Measurements::from_avatar` is **infallible**: an absent landmark surfaces as
`girth() -> None`, never as a failed conversion.

**Units.** The bundle is **metres**; `nguo_measure::Length` is **millimetres**.
The conversion happens **exactly once** on every path — `Measurements::from_avatar`
on the way out of the crate, and the canvas ghost's `MM_PER_M` composes with
`fit_zoom` rather than converting a second time. That is the invariant to
protect: a second conversion site is a 1000× body. See
[`DEVELOPING.md`](DEVELOPING.md) §4.5, which also covers why the body → collider
path converts nothing at all.

### glTF vertex counts do not match the manifest, and that is correct

The manifest counts **unique positions**; a glTF file splits vertices on UV and
normal seams. Measured loaded-vs-declared:

| | loaded glTF verts | declared verts | loaded tris | declared tris |
|---|---|---|---|---|
| COL male | 10 230 | 7 498 | 15 000 | 15 000 |
| COL female | 10 531 | 7 500 | 15 000 | 15 000 |
| LOD2 female | 31 490 | 24 999 | 49 998 | 49 998 |

**Triangles match exactly.** And distinct positions counted by *bit pattern*
equal the manifest count exactly — that is the oracle
`engine-api/tests/avatar_bundle.rs` uses, so a bundle swap that silently changes
the mesh fails instead of passing. Never assert a raw glTF vertex count against
a manifest number.

---

## 7. Licensing and attribution

**This section is not the authority on any of it.** Per-component licences live in
`assets/avatars/LICENSE.md`; project-wide policy in
[`../../LICENSING.md`](../../LICENSING.md) §0, §4 and §5; the notice wording in a
`pub const`. What follows is only what a *code* reader has to know, which is three
things.

**1. Attribution is required and the wording is a const, not a string literal.**

> **`nguo_avatar::ATTRIBUTION_NOTICE`** — `../crates/nguo-avatar/src/skin.rs:69`

One consolidated string discharges all three CC BY 4.0 obligations — Seede XR as
licensor (we owe attribution to ourselves), the skin-texture sources, and the Monk
Skin Tone Scale — and credits Mixamo. `nguo_avatar::MONK_ATTRIBUTION`
(`skin.rs:49`) is the Monk line alone. **Render the const; never retype the
words.** Hand-copied variants are how the shipped build under-attributed earlier —
the copies drifted and the short ones dropped "licensed under CC BY 4.0" and the
licence URL. `nguo-app` renders it at `../crates/nguo-app/src/main.rs:901`, and
`../crates/nguo-avatar/tests/bundle.rs:277` asserts on it.

**There is no about/credits screen yet.** The notice renders today in a *Credits*
section of the properties panel, which is collapsible — so the credit is
technically present and practically easy to hide. Closing that gap is the tracked
follow-up and a review item on whichever PR builds the screen, because no tool in
this repo can enforce it (`cargo-deny` and `cargo-about` see Cargo metadata only).

**2. The grant is clean; two constraints bind the *manner* of redistribution.**
The CC BY 4.0 grant over Seede XR's modelling data — meshes, rigs, weights, LODs,
collision proxies, the derived tiers — has nothing carved out of it. Two of the
shipped files carry a limit on how they may be passed on, and neither is a licence
exception:

- **The baked Mixamo clips are outside the grant** (Adobe's, licensed to the
  downloading account, not ours to sub-license). The files may travel **as a
  derived character used in the application** — the intended use — and **not as a
  standalone animation pack**.
- **The `VRMC_vrm.meta` block in the two `.vrm` files is narrower than the
  grant** — `allowRedistribution: false`, `avatarPermission: onlyAuthor`, a
  VRM-Public-License `licenseUrl`. Every divergence runs restrictive, so nothing
  is over-granted, but a VRM consumer reads the block programmatically, so those
  two files carry a redistribution limitation the GLBs do not. Lifting it needs
  the upstream pipeline's `scripts/09_export_vrm.py` re-run; it cannot be done
  here. Field table and target values: `assets/avatars/LICENSE.md`.

Neither touches the loader, and neither blocks commercial use of the geometry.

**3. The former release blocker is CLOSED.** An earlier bundle (223 MB, four
avatars, unverified provenance) was **deleted** and replaced, not re-licensed.
`CC-BY-4.0` was added to `deny.toml` and both `about.toml` files, all three
agreeing at a **21-entry** allow-list (`make licenses-sync`), and the canonical
licence text is vendored at `assets/avatars/LICENSES/CC-BY-4.0.txt`
(18 657 bytes, sha256 `9ba9550a…`).

Code licensing is unchanged: `ui/**` and `core-engine/**` are GPL-2.0-or-later,
`bridges/blender/` GPL-3.0-or-later. A CC BY 4.0 asset directory does not touch
that; `../../LICENSING.md` §0 is the authority on the split.

---

## 8. Testing against the bundle

Both suites are **bundle-gated**: with no bundle they print a skip notice to
stderr and pass. Every test that *does* run asserts a **floor** — the tier
counts, the 10 tones, the 7 shared maps — so it cannot pass vacuously when the
bundle is present.

| Suite | Tests | Gate |
|---|---|---|
| `ui/crates/nguo-avatar/tests/bundle.rs` | 8 | skips on `AvatarError::BundleNotFound`; **any other error panics** |
| `core-engine/crates/engine-api/tests/avatar_bundle.rs` | 5 (1 `#[ignore]` benchmark) | skips on a missing `data/manifest.json`; **`MGUSO_REQUIRE_AVATAR_BUNDLE=1` turns the skip into a failure** |

The engine skip notice is written to fd 2 directly rather than via `eprintln!`,
because libtest's capture swallows the print macros unless the test fails.

```sh
cargo test -p nguo-avatar                              # with the bundle
NGUO_AVATAR_BUNDLE=/nope cargo test -p nguo-avatar     # exercise the skip path
make test-avatars                                      # both suites, skip = FAILURE
```

**`make test-avatars` is the gate to use.** It refuses to run without a bundle
instead of skipping, and it sets `MGUSO_REQUIRE_AVATAR_BUNDLE=1`. See
[`CI.md`](CI.md) — CI has a dedicated `avatars` job for this, because
`actions/checkout` does **not** fetch LFS by default and paying the 227 MB
download in every job would be wasteful.

**Known gap:** the UI suite has no `MGUSO_REQUIRE_AVATAR_BUNDLE` equivalent —
`make test-avatars` compensates by checking the bundle exists before invoking
cargo, but an in-crate switch would be better. `nguo-app`'s own test gate
swallows every `AvatarError`, not just a missing bundle. Both are tracked
findings, not designs.

---

## 9. Measured numbers, in one place

All measured on 2026-08-03, Apple M4 / macOS / Metal. Anything not in this
section and not cited above is not measured.

**Asset load** (`NGUO_CAPTURE`, 1440×860, release):

| | |
|---|---|
| `Bundle::open` (two JSON files) | **0.1 – 0.7 ms** |
| LOD2 body GLB (~25 MB, 50 k tris) | **131.7 – 142.6 ms** |
| COL proxy GLB | **6.1 – 9.9 ms** |

**Frame time, on screen:** **8.35 – 8.74 ms with the avatar** against
**8.35 – 8.56 ms without** — indistinguishable. The 50 k-triangle body costs
nothing measurable to draw.

> **Caveat that must travel with that number.** Present mode is `Fifo` on a
> 120 Hz display, so **8.33 ms is a hard floor**. This proves the avatar did not
> knock the app off the vsync cadence. It does **not** measure GPU headroom.

**Solver:** §5.1. That is the number that matters, and as of 2026-08-04 it is
**0.235 s/step** for a whole body rather than 9.724.

**Suites (re-run 2026-08-04, debug):** `core-engine` **399** passed / 0 failed /
3 ignored; `ui` **542** passed / **2 failed** / 3 ignored — the two being
`nguo-cli`'s end-to-end tests, which need a built `mguso` and hard-fail rather
than skip without one. `cargo fmt --all --check` and
`cargo clippy --workspace --all-targets -D warnings` **exit 0 in both**
workspaces, re-run the same day.
