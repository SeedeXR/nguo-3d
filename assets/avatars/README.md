<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Rewritten repo-side 2026-08-03. Per LICENSING.md §0 the bundle's own docs are
     CC BY 4.0; the notice document beside this one (LICENSE.md) is not, because a
     notice is not the thing it documents. -->
# Nguo Avatars — the delivered asset bundle

Rigged human avatars for the nguo-3d cloth design & simulation app. Two bodies
(`male`, `female`), three runtime geometry tiers each, ten complexions, nineteen
runway animations. Units are **metres**, the space is **+Y up / +Z forward,
right-handed**, and there is no axis conversion anywhere on the avatar path.

**Licensing: [`LICENSE.md`](LICENSE.md).** Everything the bundle delivers is
**CC BY 4.0** and **attribution is required** — for the Seede XR modelling data, the skin textures
and the Monk Skin Tone Scale alike. Two constraints on the *manner* of
redistribution also apply (the baked Mixamo clips, and the metadata inside the
`.vrm` files); `LICENSE.md` is the authority on both. The credit wording itself
lives in `nguo_avatar::ATTRIBUTION_NOTICE` and is not restated in any document.

---

## Where to read what

This README is a signpost, not a spec. Each fact below has exactly one home:

| For | Read |
|---|---|
| **What each component is licensed under**, and the redistribution constraints | [`LICENSE.md`](LICENSE.md) |
| **The third-party inputs** — the texture sources, the Monk scale, Mixamo, Blender | [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) |
| **The file formats** — glTF layout, skinning limits, UV sets, materials, complexion routes, animation, Unity | [`docs/INTEGRATION.md`](docs/INTEGRATION.md) |
| **How this repo loads and uses the bundle** — the `nguo-avatar` API, tiers, measurements, collision cost, file sizes | [`../../ui/docs/AVATARS.md`](../../ui/docs/AVATARS.md) |
| **Why the assets are the way they are** — the measured evidence behind LOD0, the 4-influence limit, the non-deforming bones | [`docs/PIPELINE_NOTES.md`](docs/PIPELINE_NOTES.md) |
| **The machine-readable truth** — paths, tiers, vertex counts, measurements, verification | `data/manifest.json` |

## Layout

```
avatars/glb/        nguo_<sex>.glb (LOD2, the default runtime body),
                    nguo_<sex>_lod1.glb, nguo_<sex>_collision.glb
avatars/vrm/        nguo_<sex>.vrm — the runtime body plus VRMC_vrm 1.0
textures/skin/
  shared/           neutral albedo, normal (OpenGL + DirectX), roughness, AO,
                    height, subsurface — 2048², tileable
  tones/            10 baked complexion albedos, skin_albedo_MST01..MST10.png
data/
  manifest.json     every path, count, measurement and verification number
  skin_tones.json   the 10 complexions and their linear colour factors
  humanoid_bone_map.json   Mixamo -> VRM bone mapping, and which bones don't deform
previews/           hero, orthographic, tone sheet, LOD sheet, walk sheet
docs/               INTEGRATION.md, PIPELINE_NOTES.md, ATTRIBUTION.md
```

The four files to load are `nguo_<sex>.glb` (the body, 50 000 triangles, ~26 MB)
and `nguo_<sex>_collision.glb` (the cloth collider, 15 000 triangles, ~6.7 MB).
Both are self-contained: geometry, the 65-bone skeleton, all 19 animations and
three embedded 2K textures. Nothing else is needed to get an avatar on screen.

**`data/manifest.json` declares more paths than the bundle ships** — the `.blend`
masters, the LOD0 archival meshes, the Unity FBX export and the Unity texture
packing are all declared and absent, as are the pipeline `scripts/`. Resolve
nothing eagerly; `../../ui/docs/AVATARS.md` §3 has the count and the typed error.
