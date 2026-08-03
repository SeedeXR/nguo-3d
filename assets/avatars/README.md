# Nguo Avatars — asset bundle for the Seede XR cloth simulator

Rigged human avatars for a Rust cloth design & simulation app. Two bodies, three
runtime geometry tiers each, ten complexions, nineteen runway animations.

Avatars © **Seede XR**. Read `docs/ATTRIBUTION.md` before shipping — two of the
inputs are CC BY 4.0 and require credit.

---

## Load this

```
avatars/glb/nguo_female.glb            ← the runtime avatar (50k tris)
avatars/glb/nguo_male.glb
avatars/glb/nguo_female_collision.glb  ← the cloth collision body (15k tris)
avatars/glb/nguo_male_collision.glb
```

Both runtime GLBs are self-contained: geometry, 65-bone skeleton, all 19
animations, and three embedded 2K textures. Nothing else is required to get an
avatar on screen.

| file | tris | size | use |
|---|---|---|---|
| `nguo_*.glb` | 50 000 | 25 MB | **default runtime avatar** |
| `nguo_*_lod1.glb` | 200 000 | 31 MB | hero / close-up rendering |
| `nguo_*_collision.glb` | 15 000 | 6.4 MB | **cloth collision body** |
| `nguo_*.vrm` | 50 000 | 25 MB | VRM 1.0 humanoid variant |

## Contents

```
avatars/glb/        GLB per tier, 19 animations each   <- primary format
avatars/vrm/        VRM 1.0 humanoid variant
textures/skin/
  shared/           normal (GL+DX), roughness, AO, height, subsurface, neutral albedo
  tones/            10 baked complexion albedos
data/
  manifest.json     everything below, machine-readable
  skin_tones.json   the 10 complexions + colour factors
  humanoid_bone_map.json   Mixamo -> VRM bone mapping, and which bones don't deform
previews/           hero, orthographic, tone sheet, LOD sheet, walk sheet
docs/
  INTEGRATION.md    the developer guide — start here
  ATTRIBUTION.md    licensing and the required credit notice
  PIPELINE_NOTES.md how the assets were built and what not to change
```

**`docs/INTEGRATION.md` is the real documentation.** It covers loading with the
`gltf` crate, the four things to know before writing the shader, complexion
switching, animation selection, cloth collision, body measurements, and Unity.

## The five-minute version

**Coordinate system.** glTF/VRM: **+Y up, +Z forward, metres**. Feet on the
ground plane, midline at X=0 — drops in at the origin with no fix-up transform.
1 unit = 1 metre exactly, so garment patterns need no scale factor.

**Skinning.** 65 joints, **max 4 influences per vertex**, so exactly one
`JOINTS_0`/`WEIGHTS_0` set — one `uvec4` plus one `vec4`. `JOINTS_1` never
appears. Joint matrix = node world transform × `inverseBindMatrices[j]`.

**UVs.** `TEXCOORD_0` is a tiling detail UV that deliberately runs outside
`[0,1]` — one tile spans 0.20 m of skin. Your sampler **must** use `REPEAT`
wrapping or the skin smears. `TEXCOORD_1` is a 0..1 atlas for baking your own
maps later; nothing samples it today.

**Culling.** `doubleSided` is `false` — the body is watertight, so turn backface
culling on.

**Complexion.** Ten Monk Skin Tone shades. Either swap the albedo to any
`textures/skin/tones/skin_albedo_MST*.png`, or — cheaper — use
`shared/skin_albedo_neutral.png` and set `baseColorFactor` from
`data/skin_tones.json`. That second path costs four floats, no texture reload.
The shipped files have `MST05 "Golden"` baked in so they look right immediately.

**Animation.** 19 clips at 30 fps, named after the source, retargeted onto
*both* bodies so any clip plays on either avatar. Pick by name, not index.
Sampler inputs are in seconds. Clips carry **root motion** — `mixamorig:Hips`
translates, so the avatar walks; zero that channel for in-place playback.

**Cloth collision.** Use `*_collision.glb`. It is watertight and manifold, its
volume is within 0.17 % of the full-resolution body, and it tracks the render
mesh to within 5.3 mm while deformed. Same skeleton, same animations — drive
both from one pose.

**Body measurements** for pattern making are in `data/manifest.json` under
`avatars.<sex>.measurements`, including the *height* of each landmark, so a
waistband sits at the measured waist rather than a guessed fraction of stature.

| | male | female |
|---|---|---|
| stature | 1.830 m | 1.905 m |
| chest / waist / hip | 120.1 / 85.0 / 105.4 cm | 111.6 / 74.5 / 108.7 cm |
| inseam | 86.0 cm | 92.6 cm |

## Two things to be aware of

**19 of the 65 bones do not deform skin** — 13 end markers, the 6 thumb bones,
and the 8 distal finger bones. They still animate and still map to Unity Humanoid
and VRM; they simply carry no weights. Letting them drive skin tore the hand
open. Fingers still articulate at the knuckle and middle joint.

**The female is 1.905 m and the male 1.830 m**, so the female avatar is 7.5 cm
taller. That is how the sculpts were authored; the pipeline never rescales, since
rescaling is exactly the kind of change that would stop these being the original
meshes. Apply a uniform scale at load time if you want conventional sizing.

## Not in this bundle

Deliberately excluded as not needed by the app:

- `avatars/glb/*_lod0.glb` — the 1.7M-triangle archival masters (146 MB)
- `blend/` — editable Blender masters, all four tiers
- `avatars/unity/` — Unity FBX + Unity-packed textures
- `scripts/`, `tests/` — the build pipeline and its verification
- `textures/skin/unity/` — Unity metallic-smoothness packing

All of it is in the full project directory, `output/nguo-3d-avatars/`.

## Verification

These assets were checked rather than assumed. From `data/`:

| | male | female |
|---|---|---|
| LOD0 vertices vs the original sculpt | 5.07e-7 m | 5.69e-7 m |
| Silhouette match, 6 views, runtime tier | 99.92 % | 99.94 % |
| Worst edge stretch, 19 clips × 4 tiers | 3.36× median edge | 2.90× |
| Edges past 5× median edge, any frame | **0** | **0** |
| Watertight and manifold, every tier | yes | yes |

Re-runnable proofs live in `tests/` in the full project.

---

Avatars © Seede XR. Skin textures based on material from texturecan.com and
cgbookcase.com (CC BY 4.0). Complexion range based on the Monk Skin Tone Scale by
Ellis Monk (CC BY 4.0). Character animation from Adobe Mixamo — Mixamo terms do
not permit redistributing animation data standalone. Full notice in
`docs/ATTRIBUTION.md`.
