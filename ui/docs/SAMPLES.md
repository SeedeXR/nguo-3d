<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# SAMPLES.md — the real garment corpus, and what we can actually do with it

Eight vendor garment sample sets live in `samples/` (352 MB, `du -sh samples`).
**`samples/` is not distributed with this repository** — the files are
third-party garment exports of unknown provenance (§7) and are development
inputs only. This document describes them so the import work and the test
corpus are reproducible for anyone who assembles an equivalent set.

Every number below came out of a command run on 2026-08-02; nothing is estimated.
The executable form of this document is
`core-engine/crates/engine-io/tests/samples.rs` — if the two disagree, the test is
right.

```
cargo test -p engine-io --test samples -- --nocapture      # 3 passed, 5.37 s (debug)
```

## 1. Load-success table

> **UPDATED 2026-08-02 — both findings below are now FIXED.** This table was
> first written when 5 of 11 mesh files failed and the other 6 silently lost
> every attribute. Both defects were repaired the same day (§2, §3). The
> executable truth is `core-engine/crates/engine-io/tests/samples.rs`; if this
> table and that test ever disagree, **the test is right**.

`engine_io::load_mesh` on every geometry file in the corpus.

| Sample | File | Size | Loads? | Verts | Tris | Normals | UVs | Panels |
|---|---|---:|---|---:|---:|---:|---:|---:|
| dress-a | `dress_.obj` | 32.0 MB | **yes** | 200 443 | 395 102 | 200 443 | 200 443 | 12 |
| dress-b | `dress_.obj` | 29.5 MB | **yes** | 186 370 | 363 929 | 186 370 | 186 370 | 31 |
| men-shirt | `Men's Shirt.obj` | 11.2 MB | **yes** | 70 312 | 128 191 | 70 312 | **71 745** ‡ | 45 |
| women-skirt | `Women's Pleats Skirt OBJ.obj` | 5.4 MB | **yes** | 35 535 | 69 383 | 35 535 | **36 127** ‡ | 8 |
| verlet-starlight | `obj/metal.obj` † | 272 KB | **yes** | 1 888 | 3 752 | 1 887 | 2 092 ‡ | 3 |
| verlet-starlight | `obj/sphere.obj` † | 61 KB | **yes** | 482 | 960 | 482 | 559 ‡ | 1 |
| bike-jacket | `bike+jacket.fbx` | 1.7 MB | **yes** | 31 163 | 60 454 | 31 163 | 31 163 | — |
| dress-a | `dress_.fbx` | 10.8 MB | **yes** | 200 443 | 395 102 | 200 443 | 200 443 | — |
| dress-b | `dress_.fbx` | 10.7 MB | **yes** | 186 370 | 363 929 | 186 370 | 186 370 | — |
| men-shirt | `Men's Shirt FBX.fbx` | 6.6 MB | **yes** | 114 824 | 212 709 | 114 824 | 114 824 | — |
| tote-bag | `tote+bag+FBX(1).fbx` | 8.8 MB | **yes** | 194 171 | 334 830 | 0 § | 0 § | — |
| dress-a/b | `dress.zprj` | 11.7 / 16.0 MB | **no** | — | — | — | — | — |
| dress-a/b | `*.blend` | 28.2 / 21.4 MB | **no** | — | — | — | — | — |
| girls-outfit | `Materials+Textures.rar` | 82.9 MB | **n/a** | — | — | — | — | — |

† extracted from `verlet-starlight/obj.rar`; not committed (see §5).
‡ `vt > v` — real UV **seams**, where one position carries several texture
coordinates. The lossless corner-indexed data lives in `ObjMesh`; the welded
`Object.uvs` array keeps a first-corner-wins value per vertex (see §2).
§ `tote-bag` is a multi-geometry FBX and hits a *pre-existing* parallel-or-empty
rule in `fbx.rs` unrelated to the zlib defect. Pinned as `0, 0` in the test so a
future fix announces itself rather than passing silently.

**All eleven mesh files now load, and attributes survive.** Was 6 of 11 with
every attribute discarded.

The strongest validation available in this corpus: **`dress-a` loads to the
identical mesh from two independent formats** — OBJ and FBX both yield 200 443
verts / 395 102 tris, and the FBX additionally carries 200 443 normals and UVs.
Two different parsers, two different files, one answer. That is pinned as
`dress_a_obj_and_fbx_agree`.

All six OBJs round-trip through both our writers (OBJ text and binary GLB) with
*bit-identical* positions and topology — `assert_eq!` on the raw
`Vec<[f32;3]>`, not a tolerance — and now also preserve `uvs`, `uv_indices` and
group runs through a `parse → write → parse` cycle.

Bounding boxes (measured; all garments are in **centimetres**, Y-up, standing on
Y≈0 apart from the ones posed on an avatar):

```
dress-a/dress_.obj        [-42.43,   4.67, -24.49] .. [ 42.32, 152.78,  13.92]
dress-b/dress_.obj        [-39.11,  34.73, -30.85] .. [ 35.67, 157.86,  28.59]
men-shirt                 [-58.02,  90.06, -15.46] .. [ 57.98, 167.23,  14.61]
women-skirt               [-24.81,  73.56, -19.59] .. [ 24.48, 116.83,  18.13]
verlet obj/metal.obj      [ -2.01,  -0.09,  -0.69] .. [  1.97,   0.29,   0.69]
verlet obj/sphere.obj     [ -1.00,  -1.00,  -1.00] .. [  1.00,   1.00,   1.00]
```

## 2. Finding 1 — the OBJ reader threw away everything but `v` and `f` — **FIXED 2026-08-02**

> **Resolved.** `obj.rs` now parses `vt`/`vn` and `usemtl`/`g`/`o` runs.
> `ObjMesh` carries the lossless corner-indexed data (`uv_indices`/
> `normal_indices` parallel to `tris`) plus `ObjGroup { name, material, tris }`
> in file order. `Object` keeps the welded position table with a
> first-corner-wins de-index, matching the contract `fbx.rs`/`gltf_io.rs`/
> `collada.rs` already use.
>
> **Why the positions are NOT split at UV seams:** the solver simulates the
> welded mesh. Splitting would change the vertex count, the lumped vertex mass
> (`area*density/n`) and the constraint graph — altering the physics to serve a
> rendering concern. **Open handoff:** `engine_api::Object` still cannot express
> a seam end-to-end; that needs either a corner-index array on `Object` or a
> render-side split at GPU upload. Reader-side work is done and the data is
> exposed.
>
> A second bug the corpus exposed and that is also fixed: naive whitespace
> splitting turned `mtllib Men's Shirt.mtl` into two filenames. `mtllib` now
> only splits when *every* token ends in `.mtl`.
>
> The analysis below is retained because its *reasoning* is what justified the
> fix — in particular that dress-a's 12 `usemtl` runs against 1 material mean
> the runs are **pattern panels**, the most valuable structure in the file for a
> pattern-first application.

`core-engine/crates/engine-io/src/obj.rs:52` is literally:

```rust
_ => {} // vt, vn, g, o, s, mtllib, usemtl — ignored for now
```

What is in the files we are ignoring (`awk` over each OBJ):

| Sample | `v` | `vt` | `vn` | `f` | `usemtl` groups | `newmtl` in `.mtl` |
|---|---:|---:|---:|---:|---:|---:|
| dress-a | 200 443 | 200 443 | 200 443 | 198 989 | 12 | 1 |
| dress-b | 186 370 | 186 370 | 186 370 | 182 334 | 31 | 3 |
| men-shirt | 70 312 | **71 745** | 70 312 | 85 913 | 45 | 5 |
| women-skirt | 35 535 | **36 127** | 35 535 | 34 693 | 8 | 2 |
| verlet metal | 1 888 | 2 092 | 1 887 | 1 876 | — | 0 ‡ |
| verlet sphere | 482 | 559 | 482 | 512 | — | 0 |

‡ `metal.obj` declares `mtllib gire.mtl`, which is **not in the archive**; the
`metal.mtl` that is there is an empty Blender stub (two comment lines). A real
`.mtl` reader must tolerate a missing/empty library, not fail the load.

Three things follow, and they are design constraints, not trivia:

1. **CLO already writes one UV per control point for the dresses** (`vt == v`), so
   for those two a straight parallel-array import is exact and cheap.
2. **men-shirt and women-skirt have UV seams** (`vt > v`, by 1 433 and 592). A
   per-vertex UV array cannot represent them; importing correctly means splitting
   positions at seams, which changes the vertex count the solver sees. Decide this
   deliberately — the FBX/glTF/COLLADA readers here already chose "first corner
   touching a vertex wins", which silently loses the seam.
3. **`usemtl` group boundaries are the pattern pieces.** dress-a has 12 `usemtl`
   runs but only 1 material: the runs are not material variety, they are the
   garment's panels. A 2D-pattern-first application (D-10) that discards them is
   throwing away the single most valuable structure in the file.

The `.mtl` files are CLO's ("MTL Exporter v1.0 by Seungwoo Oh at CLO - Virtual
Fashion Inc.") and reference the sibling `*_diffuse_1001.png` / `*_normal_1001.png`
via `map_Kd` / `map_bump`. dress-a's `.mtl` still carries a Windows absolute path
(a Windows author path, redacted — it embeds a third party's username); dress-b and the others use
relative names. Any `.mtl` reader must handle both, and must not trust the path.

`tests/samples.rs` asserts `uvs.is_empty() && normals.is_empty()` for all six. When
the reader is fixed the assertion fails and forces the table to be updated.

## 3. Finding 2 — every real-world FBX failed — **FIXED 2026-08-02, and the diagnosis below was WRONG**

> **Resolved by a one-line change in our own code**, with no fork and no
> dependency change: `AnyTree::from_reader(bytes)` →
> `AnyTree::from_seekable_reader(Cursor::new(bytes))` in `fbx.rs`.
>
> **The root cause identified in this section was incorrect, and that matters.**
> This document blamed `pull_parser/v7400/attribute.rs:104` (decompression
> skipping only forward). The fix agent vendored `fbxcel`, applied exactly that
> patch, and it changed **nothing — same errors, byte for byte.** Instrumenting
> the crate found the real defect one file over:
>
> ```rust
> // fbxcel-0.9.0/src/pull_parser/reader.rs:43
> fn skip_distance_plain(reader: &mut Reader<R>, distance: u64) -> io::Result<()> {
>     let mut limited = io::Read::take(reader.inner.by_ref(), distance);
>     io::copy(&mut limited, &mut io::sink())?;
>     Ok(())            // never calls reader.advance(distance)
> }
> ```
>
> The **non-seekable** skip path discards the bytes but never advances the
> cached position, so the counter falls permanently behind. `skip_distance_seekable`
> *does* advance. A skip only fires when an attribute is left partially unread —
> and a zlib array is the only thing that does that, because libflate stops
> before the 4-byte adler32 trailer. Hence 652 lines of `Direct` fixtures stayed
> green while 100 % of real files died.
>
> Since the working path already exists in the dependency, the fix is simply to
> reach it: a `Cursor` over the input slice is `Read + Seek`.
>
> **Upstream fix, for whoever files it** (not filed — no network in that
> environment): `skip_distance_plain` needs `reader.advance(distance as usize)`.
>
> **The lesson is the durable part:** a plausible, well-argued root cause that
> was never *tested* cost a vendored fork and a wasted patch. The failing test
> came first, which is the only reason the wrong fix was caught immediately.

All five FBX files fail identically:

```
parsing FBX binary: Tree load error: ... Data error:
Node ends with unexpected position: expected 1317564, got Some(1317557)
```

Versions and exporters (read from the 27-byte binary header + the `Creator` node):

| File | FBX ver | Creator |
|---|---:|---|
| `bike-jacket/bike+jacket.fbx` | 7400 | Blender (stable FBX IO) 4.3.2 |
| `dress-a/dress_.fbx` | 7500 | FBX SDK/FBX Plugins 2020.0.1 |
| `dress-b/dress_.fbx` | 7500 | FBX SDK/FBX Plugins 2020.0.1 |
| `men-shirt/Men's Shirt FBX.fbx` | 7700 | FBX SDK/FBX Plugins 2020.0.1 |
| `tote-bag/tote+bag+FBX(1).fbx` | 7700 | FBX SDK/FBX Plugins 2020.0.1 |

It is **not** a version problem — `fbxcel` 0.9 dispatches 32-bit vs 64-bit node
headers correctly at `src/low/v7400/node_header.rs:53`. Two experiments:

1. An independent walk of all five binary trees (Python, 40 lines, honouring the
   version-dependent header width) reports **0 node-length mismatches** in every
   file. The files are structurally sound.
2. Rewriting each file's array attributes from zlib (`encoding=1`) to Direct
   (`encoding=0`) — decompressing the payload, changing no geometry — makes them
   load, correctly and completely:

   | File (arrays inflated) | Verts | Tris | Normals | UVs | UV range |
   |---|---:|---:|---:|---:|---|
   | `bike+jacket` (7 arrays) | 31 163 | 60 454 | 31 163 | 31 163 | 0.005 .. 0.995 |
   | `dress-a/dress_` (8) | 200 443 | 395 102 | 200 443 | 200 443 | 0.015 .. 0.985 |
   | `tote+bag` (8) | 194 171 | 334 830 | 0 | 0 | — |

   `dress-a`'s FBX yields **exactly** the OBJ's 200 443 / 395 102 — same mesh, but
   with normals and UVs the OBJ path drops. tote-bag loses attributes for a
   different reason: it is multi-geometry and `fbx.rs`'s parallel-or-empty contract
   drops normals/UVs unless *every* geometry supplies them.

So: our FBX reader is good, and `fbxcel` 0.9's zlib array path desynchronises the
reader position. *Inferred mechanism* (from reading the crate, not measured):
`pull_parser/v7400/attribute.rs:104` only ever skips **forward** to the next
attribute (`if position < next_attr_start { skip_to }`), so when libflate's zlib
decoder reads ahead of the compressed block the position is never brought back —
consistent with the observed `+5` byte drift on the 7500/7700 files.

**Why we never caught this:** all 652 lines of `tests/fbx_file.rs` build fixtures
with `ArrayAttributeEncoding::Direct`. Every real DCC exporter uses zlib. The FBX
suite is green and has never once exercised the code path that all real files take.

`tests/samples.rs::real_fbx_exports_are_known_broken` now pins the failure (and
panics with "GOOD NEWS" plus the real counts if any of them starts working).

## 4. `.zprj` — CLO/Marvelous project container

`.zprj` is a **zip with a 250 584-byte prefix** (`dress-a`; 266 453 for `dress-b`).
`unzip` reads it anyway, with a warning. Layout, all observed:

```
offset 0        " ZPRJ       CLO " magic, then a binary header containing
                4 embedded PNGs (thumbnails/preview) and the literal string
                "CLO Standalone OnlineAuth"
offset 250584   standard zip local header (PK\x03\x04), 12 members:

  dress.zpac      11 007 022   " ZPAC       CLO " magic — the bulk payload
  dress.vlp        1 627 313   " VLP        CLO "
  dress.vmp          416 801   " VMP        CLO "
  dress.vrp              831   " VRP        CLO "
  dress.xml            2 322   plain XML, <Environment Version="4.0">
  clofiles.json        1 076   member -> original Windows path map
  dress.cmt              179   text: "Version: 1|^^|Type: 0|^^|<4x4 matrix>"
  dress.cmp               63   text: "Version: 1|^^|Mode: 0|^^|15 100 100000"
  dress.cpt              515   binary, no magic
  dress.osw            1 007   binary, no magic
  dress.smp              906   binary, no magic
  img_3dwindow_bg_Modelist.png  1 274
  (dress-b adds dress_0.avt, 4 513 792 — avatar)
```

Directly usable today, no reverse-engineering required:

- `dress.xml` is readable CLO scene environment: wind (position, direction,
  strength 1000, decay, frequency, turbulence 30), two lights with full 4×4
  matrices and RGBA ambient/diffuse/specular, ground-collision flag, ground height,
  background colour. This is a **wind-and-lighting reference for our solver and
  viewport**, free.
- `dress.cmt` is a camera transform (4×4, translation 1023.6 / 8004.2 → cm).
- `dress.cmp` is camera projection ("Mode: 0", `15 100 100000` — plausibly
  FOV/near/far; **unverified**).
- `clofiles.json` maps every member back to its authoring path.

**Everything else is unverified.** `.zpac` (patterns, 2D pieces, stitches, fabric
assignments — the part we actually want) is an opaque `ZPAC`-magic blob and this
document makes no claim about its internals. `.vlp`/`.vmp`/`.vrp` share the same
` XXX        CLO ` 16-byte magic form; `.cpt`/`.osw`/`.smp` have none.

## 5. Archives

| Archive | Format | Contents | Extracted? |
|---|---|---|---|
| `verlet-starlight/obj.rar` | RAR5, 66 KB | `obj/{metal,sphere}.{obj,mtl}` | **yes** |
| `verlet-starlight/textures.rar` | RAR5, 32 MB | 6 PBR maps (diffuse/normal/roughness/metalness/opacity/displacement) | no |
| `girls-outfit/Materials+Textures.rar` | RAR4, 83 MB | 20 PBR PNGs across two UV sets + 2 Substance `.spsm` — **no geometry at all** | no |

Tool availability, checked: `unar` and `unrar` are **not installed**. `7z`
(p7zip 17.05) *lists* both RAR4 and RAR5 but **cannot decompress** them —
`ERROR: Unsupported Method`, and it silently leaves 0-byte files behind.
`bsdtar` (libarchive, `/usr/bin/bsdtar`, ships with macOS) extracts RAR5 correctly
and is what was used. Use `bsdtar -xf <file>.rar`.

`girls-outfit` contains **no mesh**: it is a texture-only pack (`PBR/` +
`Unreal Engine/` variants of AO, BaseColor, Height, Metallic, Normal, Opacity,
Roughness for two UV sets). It cannot be a load-test subject at all. Not extracted
— 88 MB of textures with no geometry to apply them to.

`samples/.gitignore` (added) keeps every extracted directory out of the repo; the
archives are the source of truth.

## 6. Textures and HDRI

All garment texture sets are 2048×2048 8-bit RGBA PNG (`*_diffuse_1001.png`,
`*_normal_1001.png`) — the `1001` is the UDIM tile, and there is exactly one tile
per garment. dress-a/b additionally ship a `texture/` folder with the CLO fabric
swatches used (`Cotton_Voile_FCL1PSC006_NORMAL.jpg`,
`40s_Chambray_FCL1PSC004_NORMAL.jpg`, both 1024×1024) plus a small colour
thumbnail. `dress.png` (900×900) is the CLO render preview.

`dress-b/hdri/brown_photostudio_06_4k.exr` is 4096×2048 (dataWindow), capDate
`2022:10:08`. `engine-io` has no EXR reader and none is proposed here — it is a
**render reference**, an input to whatever IBL the viewport eventually grows.

## 7. Licence / provenance

**There is no licence file anywhere in `samples/`** (`find samples -type f` = the
37 shipped files, none named LICENSE/README/COPYING). Provenance is therefore
**unknown** for every sample. What is *observable*:

| Sample | Observable provenance | Licence |
|---|---|---|
| dress-a | CLO Standalone (OnlineAuth), authored 2024-12-06; exporter path redacted (embeds a third party's Windows username) | unknown |
| dress-b | CLO Standalone, 2026-02-27, same authoring machine | unknown |
| men-shirt | CLO `.mtl` exporter + FBX SDK 2020.0.1 | unknown |
| women-skirt | CLO `.mtl` exporter | unknown |
| tote-bag | FBX SDK 2020.0.1 | unknown |
| bike-jacket | Blender 4.3.2 FBX IO | unknown |
| girls-outfit | Substance Painter (`.spsm`), 2023-08-20, asset name `z-h_Girl's Outfit 05` | unknown |
| verlet-starlight | `metal.obj` written by RizomUV 2022.1.57, `sphere.obj` by Blender 4.2.1 LTS | unknown |

`dress-a/dress1.blend` and `dress-b/dress.blend` are Blender 3.6 files
(`BLENDER-v306` header: 64-bit, little-endian, v3.06).

The `"free1 product"` path fragment *suggests* the dresses were a free download,
and `brown_photostudio_06` matches Poly Haven's naming (Poly Haven publishes CC0),
but neither is confirmed by anything in the files. **Do not ship any of these, or
any render of them, until the maintainer confirms the source.** They are
development inputs.

## 8. What to build next, in priority order

1. **Fix FBX zlib arrays.** One dependency defect stands between us and five real
   garments *with* normals, UVs and multi-geometry structure. Options, cheapest
   first: (a) check whether a newer `fbxcel` fixes it; (b) upstream a one-line fix
   to `attribute.rs:104` so the position is always reset to `next_attr_start`, not
   only when behind; (c) vendor the patch. **And add a zlib-encoded fixture to
   `tests/fbx_file.rs` first** — the bug survived 652 lines of FBX tests because
   every fixture used `Direct`.
2. **Make the OBJ reader read the whole file.** `vt`, `vn`, `usemtl`/`g`, and
   `mtllib` are all present in all six OBJs and all six are discarded. Highest
   value per line of code in the corpus. Two decisions to make explicitly: seam
   handling for `vt > v` (men-shirt, women-skirt), and whether `usemtl`/`g` runs
   become per-panel submeshes — they should, they are the pattern pieces.
3. **Sanity-check *what* the readers hand the solver.** Every OBJ passed the
   invariant suite (finite coords, in-range indices, no degenerate triangles,
   non-degenerate bbox). Nothing here checks manifoldness, winding consistency,
   or duplicate vertices — the properties a cloth solver actually breaks on. That
   is the next tier of assertion, and the corpus is now wired up to receive it.
4. **A units/orientation contract.** Garments are cm and Y-up; `verlet-starlight`
   is unit-scale. Import must record and normalise this or drape will be nonsense.
   Nothing in `engine-io` currently looks at units at all.
5. **`.zprj` import** (lower priority). The container
   is understood (§4) and `dress.xml` / `.cmt` / `.cmp` are readable today —
   environment, lighting and camera could be imported for free. The pattern data
   in `.zpac` is opaque and is a separate, much larger project.
6. **Not proposed:** `.blend` and `.exr` readers. Blender's own exporters already
   give us FBX/OBJ, and EXR only matters once there is IBL to feed.

## 9. Which sample for which job

**UI development fixtures** (small, sub-10 ms load, safe in a hot reload loop):

- `verlet-starlight/obj/sphere.obj` — 482 v / 960 t, 2.3 ms debug. The default
  "something is on screen" mesh.
- `verlet-starlight/obj/metal.obj` — 1 888 v / 3 752 t, 8.9 ms debug. Has n-gons
  (max 5 verts/face), three `o` groups and UV seams: a better exercise of import
  paths than the sphere while staying instant.
- Both come out of `obj.rar`, so a fresh clone must run
  `bsdtar -xf samples/verlet-starlight/obj.rar -C samples/verlet-starlight`.
  The test skips them loudly if that has not happened.

**Render / visual references:**

- `dress-b` — the only complete set: mesh + 2048² diffuse/normal + fabric swatches
  + a 4K HDRI + the CLO preview render to compare against.
- `girls-outfit` — texture-only, but the richest PBR set in the corpus (AO,
  BaseColor, Height, Metallic, Normal, Opacity, Roughness, two UV sets, plus an
  Unreal-packed ORM variant). Use it to drive the material model, not the geometry.
- `dress-a/dress.png`, `dress-b/dress.png` — CLO's own 900² renders, the closest
  thing we have to a target image.

**Solver stress tests** (large, real garment topology):

- `dress-a/dress_.obj` — 200 443 v / 395 102 t, the biggest mesh that loads
  (1.0 s debug, 64 ms release). 12 pattern panels.
- `dress-b/dress_.obj` — 186 370 v / 363 929 t, 31 panels, 3 fabrics: the best
  multi-material drape test once `usemtl` is imported.
- `men-shirt` — 70 312 v / 128 191 t, 45 panels, 5 fabrics. Mid-size, most panels.
- `tote-bag` (194 171 v / 334 830 t) and `bike-jacket` (31 163 v / 60 454 t) join
  this list the moment the FBX fix lands.

**Not usable as a test subject:** `girls-outfit` (no geometry), the `.zprj` and
`.blend` files (no reader), the `.exr` (no reader).
