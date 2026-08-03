# Nguo avatars — integration guide

Rigged human avatars for the Seede XR Rust cloth design & simulation app.
Two bodies, four geometry tiers each, ten complexions, nineteen runway clips.

Everything in this document is a measured property of the shipped files, not a
plan. `data/manifest.json` is the machine-readable version of the same facts.

---

## 1. What to load

| Need | File | Tris | Size |
|---|---|---|---|
| **Runtime avatar** | `avatars/glb/nguo_{male,female}.glb` | 50 000 | ~25 MB |
| Hero / close-up render | `..._lod1.glb` | 200 000 | ~32 MB |
| Archival master | `..._lod0.glb` | 1 686 890 / 1 582 974 | 74–78 MB |
| **Cloth collision body** | `..._collision.glb` | 15 000 | ~5.9 MB |
| Humanoid variant | `avatars/vrm/nguo_*.vrm` | 50 000 | ~26 MB |
| Unity | `avatars/unity/nguo_*.fbx` + `textures/` | 50 000 | ~17 MB |
| Editing | `blend/nguo_*.blend` — all four tiers, materials, 19 actions | | 86–93 MB |

Start with `nguo_female.glb` / `nguo_male.glb`. They are self-contained:
geometry, 65-bone skeleton, all 19 animations, and three embedded 2K textures.

The **LOD0 GLB is an archival master**, not a runtime asset. Ship LOD2, keep
LOD0 for offline renders and for regenerating tiers.

---

## 2. Coordinate conventions

| | up | forward | units |
|---|---|---|---|
| glTF / GLB / VRM | **+Y** | **+Z** | metres |
| Unity FBX | +Y | +Z | metres, scale 1.0 |
| Blender source | +Z | −Y | metres |

Feet sit on the ground plane and the body midline is at X = 0, so an avatar
drops straight into a scene at the origin with no fix-up transform.

1 unit = 1 metre exactly. Garment patterns can be authored in centimetres
against the measurements in section 6 without any scale factor.

---

## 3. Loading with the `gltf` crate

```rust
let (doc, buffers, _images) = gltf::import("avatars/glb/nguo_female.glb")?;

// 19 clips, named after the source Mixamo file
for anim in doc.animations() {
    println!("{}", anim.name().unwrap_or("<unnamed>"));
}

// skin: 65 joints, 4 influences per vertex, one JOINTS_0/WEIGHTS_0 set
let skin = doc.skins().next().unwrap();
let joints: Vec<_> = skin.joints().map(|n| n.index()).collect();
let reader = skin.reader(|b| Some(&buffers[b.index()]));
let inverse_bind: Vec<[[f32; 4]; 4]> =
    reader.read_inverse_bind_matrices().unwrap().collect();

// mesh attributes
let mesh = doc.meshes().next().unwrap();
let prim = mesh.primitives().next().unwrap();
let r = prim.reader(|b| Some(&buffers[b.index()]));
let positions:  Vec<[f32; 3]> = r.read_positions().unwrap().collect();
let normals:    Vec<[f32; 3]> = r.read_normals().unwrap().collect();
let uv0:        Vec<[f32; 2]> = r.read_tex_coords(0).unwrap().into_f32().collect();
let joint_ids:  Vec<[u16; 4]> = r.read_joints(0).unwrap().into_u16().collect();
let weights:    Vec<[f32; 4]> = r.read_weights(0).unwrap().into_f32().collect();
```

Four things worth knowing before you write the shader:

- **Exactly one joint set.** Max 4 influences per vertex, so `JOINTS_1` /
  `WEIGHTS_1` never appear. One `uvec4` + one `vec4` is all you need.
- **`TEXCOORD_0` is the tiling detail UV** and deliberately runs outside
  `[0,1]` — one tile spans 0.20 m of skin. Your sampler **must** use
  `REPEAT` wrapping (`GL_REPEAT` / `AddressMode::Repeat`). Clamping produces
  smeared skin.
- **`TEXCOORD_1` is a 0..1 atlas** of the same unwrap, for baking your own
  full-body maps later (tattoos, makeup, body paint). Nothing samples it today.
- **`doubleSided` is `false`.** The body is watertight, so enable backface
  culling and halve the fragment cost.

Animation sampler inputs are in **seconds** (glTF standard), authored at 30 fps.

---

## 4. Skin complexion — two ways

Ten complexions from the **Monk Skin Tone Scale**, the 10-shade standard
developed by Dr Ellis Monk with Google specifically for global representation.
Full data in `data/skin_tones.json`.

### A. Colour factor — recommended
One shared neutral albedo plus a per-tone linear factor. Switching complexion
costs four floats: no texture reload, no GPU upload, no hitch.

```rust
// from data/skin_tones.json -> tones[i].base_color_factor
let factor: [f32; 4] = tone.base_color_factor;   // linear, always <= 1.0
material.base_color_factor = factor;             // glTF baseColorFactor
```
Pair it with `textures/skin/shared/skin_albedo_neutral.png`.

### B. Texture swap — drop-in
Each tone also ships a fully baked 2K albedo. Point `baseColorTexture` at
`textures/skin/tones/skin_albedo_MST05.png` and leave the factor white.

**The shipped GLB/VRM/FBX use path B with `MST05 "Golden"` baked in**, so the
files look right the instant they load. Move to path A when you build the picker.

| key | name | undertone | hex |
|---|---|---|---|
| MST01 | Porcelain | cool | `#f6ede5` |
| MST02 | Ivory | neutral | `#f3e7db` |
| MST03 | Cream | warm | `#f7ebce` |
| MST04 | Beige | neutral | `#eadaba` |
| MST05 | Golden | warm | `#d7be93` |
| MST06 | Olive Bronze | olive | `#a0855a` |
| MST07 | Amber Brown | warm | `#825c40` |
| MST08 | Umber | neutral | `#604134` |
| MST09 | Deep Espresso | cool | `#3a312b` |
| MST10 | Rich Ebony | neutral | `#292420` |

The scale is not a pure lightness ramp — it spans four undertone families, and
MST03 is genuinely lighter than MST02 in the published standard because it is
more saturated. Present the tones in scale order, not sorted by luminance.

### Texture set

| map | file | colour space |
|---|---|---|
| albedo (neutral) | `shared/skin_albedo_neutral.png` | sRGB |
| albedo (per tone) | `tones/skin_albedo_MST01..10.png` | sRGB |
| normal, OpenGL | `shared/skin_normal_ogl.png` | linear |
| normal, DirectX | `shared/skin_normal_dx.png` | linear |
| roughness | `shared/skin_roughness.png` | linear, 0.35–0.75 |
| ambient occlusion | `shared/skin_ao.png` | linear |
| height | `shared/skin_height.png` | linear |
| subsurface | `shared/skin_subsurface.png` | linear |

All 2048², 8-bit, tileable. Metallic is 0 — skin is a dielectric; IOR 1.44.

Inside the GLB, the exporter packs **occlusion in R and roughness in G of a
single ORM image**, which is why `occlusionTexture` and
`metallicRoughnessTexture` share a texture index. That is correct glTF.

`subsurface_weight` and `subsurface_color_linear` per tone are in
`skin_tones.json`. They are advisory: core glTF has no subsurface term, so they
are not baked into the exported material. Melanin absorbs scattered light, so
the weight falls from 0.33 on MST01 to 0.06 on MST10 — using one value for
every complexion is the classic way to make deep skin look like grey plastic.

---

## 5. Animation

19 clips, retargeted onto **both** bodies, so any clip plays on either avatar.
30 fps, 1 982 frames total per avatar.

Names are prefixed by the set they came from, because the two Mixamo folders
share filenames:

```
f_Catwalk_Sequence_02        366 f    m_Catwalk_Idle_To_Walk_Forward   41 f
f_Catwalk_Sequence_05        306 f    m_Catwalk_Idle_Twist_L           86 f
f_Catwalk_Walk               455 f    m_Catwalk_Twist_L_To_Twist_R     61 f
f_Catwalk_Walk_Forward        37 f    m_Catwalk_Walk_Forward_02        35 f
f_Catwalk_Walk_Stop           59 f    m_Catwalk_Walk_Forward_03        34 f
f_Catwalk_Walk_Stop_Twist_L   76 f    m_Catwalk_Walk_Forward_Arc_90R   73 f
f_Catwalk_Walking             36 f    m_Catwalk_Walk_Forward_Turn_90R  75 f
f_Female_Start_Walking        57 f    m_Catwalk_Walk_Stop              51 f
f_Female_Stop_Walking         47 f    m_Catwalk_Walk_Stop_Twist_L      51 f
f_Female_Walk                 36 f
```

Pick by name; there is no index contract. For a walk loop start with
`f_Catwalk_Walk` (15.2 s) or `m_Catwalk_Walk_Forward_02` (1.2 s, cyclic).

Clips carry **root motion**: `mixamorig:Hips` translates, so a walking avatar
travels. Zero the hips translation channel if your app wants in-place playback.

### Skeleton

65 bones, Mixamo names (`mixamorig:Hips`, `mixamorig:LeftUpLeg`, …).
`data/humanoid_bone_map.json` maps them to VRM 1.0 and documents which bones
are non-deforming.

**19 of the 65 bones are present but do not deform skin:**
13 Mixamo end markers (`HeadTop_End`, the `Toe_End`s, the finger `*4` tips),
the 6 thumb bones, and the 8 distal finger bones (`*3`). They still animate and
still map to Unity Humanoid and VRM — they simply carry no skin weights.

This is deliberate. Thumb and fingertip bones sit millimetres from neighbouring
digits, and letting them drive skin tore the hand open: 25 mm of edge stretch on
a 2 mm mesh, visible as a filament bridging two curled fingertips. Fingers still
articulate at the knuckle and middle joint. See `docs/PIPELINE_NOTES.md`.

---

## 6. Cloth simulation

Use `nguo_*_collision.glb` as the collision body:

- 15 000 triangles, **watertight and manifold** (0 boundary edges, 0 non-manifold
  edges) — a closed surface, so inside/outside tests are well defined.
- Volume within **0.17 %** of the full-resolution body.
- Tracks the render mesh to within **5.3 mm (male) / 2.0 mm (female)** across
  every frame of every clip while deformed.
- Same 65-bone skeleton and same animations, so it deforms in lockstep with the
  visible avatar — drive both from one pose.

Rest-pose surface deviation from LOD0 is 1.27 mm max / 0.15 mm mean.
Give your solver a collision offset of a few millimetres and the proxy is
indistinguishable from the render mesh.

### Body measurements

Measured as a taut tape would read it — the convex hull perimeter of a
horizontal cross-section — with upper-body girths restricted to the torso, since
these are T-pose bodies and a full slice at chest height also cuts both arms.

| | male | female |
|---|---|---|
| stature | 1.830 m | 1.905 m |
| neck | 53.1 cm | 36.6 cm |
| chest | 120.1 cm | 111.6 cm |
| underbust | 91.1 cm | 78.5 cm |
| waist | 85.0 cm | 74.5 cm |
| high hip | 97.3 cm | 100.6 cm |
| hip | 105.4 cm | 108.7 cm |
| thigh | 68.6 cm | 64.7 cm |
| calf | 43.1 cm | 39.0 cm |
| inseam (crotch height) | 86.0 cm | 92.6 cm |
| arm span | 1.909 m | 1.858 m |
| hip joint height | 104.9 cm | 109.2 cm |

Landmark heights are in `data/manifest.json` under
`avatars.<sex>.measurements.<name>.height_m`, so a waistband can be placed at
the measured waist rather than a guessed fraction of stature.

> **Worth a decision:** the female body is **1.905 m** and the male **1.830 m**,
> so the female avatar is 7.5 cm taller. That is how the sculpts were authored —
> the pipeline never rescales, since scaling is exactly the kind of change that
> would stop these being the artist's meshes. Both are tall even for runway. If
> you want conventional sizing, apply a uniform scale at load time (a uniform
> scale is shape-preserving) and rescale the measurements above by the same
> factor.

---

## 7. Unity

`avatars/unity/nguo_<sex>.fbx` plus `avatars/unity/textures/`.

Import settings:

- **Rig → Animation Type: Humanoid**, Avatar Definition: *Create From This Model*.
  The Mixamo bone names map automatically; no manual bone assignment.
- **Skin Weights: Standard (4 bones)** is enough — the meshes carry at most 4
  influences per vertex, so nothing is lost.
- `skin_normal.png` → Texture Type **Normal map**. It is OpenGL-style
  (green-up), which is what Unity wants.
- Turn **sRGB (Color Texture) off** for `skin_metallicSmoothness`, `skin_ao`
  and `skin_height`.

Standard shader wiring:

| slot | texture |
|---|---|
| Albedo | `skin_albedo_MST05.png` (or any tone) |
| Metallic | `skin_metallicSmoothness.png` — metallic in R, smoothness in A |
| Normal | `skin_normal.png` |
| Occlusion | `skin_ao.png` |

All 19 clips arrive as takes in the FBX. `nguo_<sex>_lod1.fbx` and
`nguo_<sex>_collision.fbx` share the skeleton, so one Animator drives all three.

---

## 8. VRM

`avatars/vrm/nguo_<sex>.vrm` is the runtime GLB plus a hand-written
`VRMC_vrm` 1.0 extension (no VRM add-on exists for Blender 5.2). It adds a
standardised humanoid bone mapping so a VRM-aware runtime can retarget motion
without knowing the skeleton uses Mixamo names. All 44 mapped bones are present,
including the 15 the spec requires.

**`meta` is set conservatively** — `avatarPermission: onlyAuthor`,
`allowRedistribution: false`, `creditNotation: required`. Those fields describe
how other people may use the avatar, which is a Seede XR business decision, not
a technical one. Set them deliberately before distributing. See
`docs/ATTRIBUTION.md`.

**Not included:** `expressions` and `lookAt`. Both are optional in VRM 1.0, and
neither can be authored honestly here — the source bodies have no shape keys, no
separate eyeballs and no mouth interior. Adding either would need face modelling
first; an empty expression set that silently does nothing is worse than its
absence.

---

## 9. Regenerating

Every artefact is generated. Edit the scripts, never the outputs.

```bash
./scripts/run_all.sh              # full rebuild, both bodies, ~25 min
./scripts/run_all.sh male         # one body
./scripts/run_all.sh --tests-only # just re-verify what is on disk
```

`avatars/`, `blend/`, `textures/`, `previews/`, `data/` and `build/` are all
build products.

## 10. Verification

Both suites are runnable, and they gate the build rather than decorating it.

```bash
blender -b --python tests/test_mesh_integrity.py -- male
blender -b --python tests/test_no_animation_distortion.py -- female
```

Latest results:

| check | male | female |
|---|---|---|
| LOD0 vertex vs original source, worst | 5.07e-7 m | 5.69e-7 m |
| LOD0 silhouette mismatch, 6 views | 2 / 388 222 px | 4 / 356 067 px |
| LOD2 silhouette mismatch | 0.076 % | 0.064 % |
| Collision silhouette mismatch | 0.23 % | 0.21 % |
| Worst edge stretch, all clips × all tiers | 3.36× median edge | 2.90× median edge |
| Edges over 5× median edge, any frame | 0 | 0 |
| Collision proxy vs posed LOD0 | 5.35 mm | 2.01 mm |

For scale: the broken weight reduction this pipeline started with produced
18.4× median-edge stretch and visible cracks across the chest.
