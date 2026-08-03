# Pipeline notes — what went wrong and what the numbers were

Written for whoever changes these scripts next. Everything here was measured on
these two bodies; the numbers are the reason the code looks the way it does.

The governing constraint was: **never move a surface vertex.** The shipped LOD0
is the artist's mesh, verified vertex-for-vertex against the original file. Every
decision below was made under that rule.

---

## 1. What the source files actually were

Not what the filenames suggested. Both `inputs/nguo-3d-avatars/*.blend` held a
single object named `mesh_node`:

| | male | female |
|---|---|---|
| vertices | 843 443 | 791 492 |
| triangles | 1 686 852 | 1 582 978 |
| UV layers | **0** | **0** |
| materials | **0** | **0** |
| armature | **none** | **none** |
| median edge | 1.961 mm | 1.955 mm |
| shells | 1 | 3 (+ 5 stray verts) |
| boundary edges | **40** | 0 |
| custom split normals | yes | no |
| origin | feet on floor | **centred (feet at Z −0.953)** |

So: raw triangle soup, no UVs, no materials, no rig. Uniform ~2 mm triangulation,
which is why decimation behaves so well later.

**Both bodies are solid, not hollow.** A ray through the chest returns exactly
two hits with 267 mm of flesh between them. This was checked *before* choosing a
skinning method — hollow shells silently break bone-heat weighting, because the
inner wall occludes the visibility rays and vertices bind to bones across the
body. Solid meant heat weighting was safe, and it produced 0 vertices whose
dominant bone is more than 80 mm past the nearest.

The male's 40 boundary edges were **one 19.6 mm disc in the sole of a foot** at
Z = 2.6 mm — a ground/boolean cut. Filled: a cloth collision body has to be
watertight. Cost, measured: 74 pixels in a straight-up-from-below render and
**3 pixels across the other five views**.

---

## 2. Custom split normals: measured, not assumed

The male source had custom split normals deviating from smooth normals by up to
**101°**. That looks alarming until you look at the distribution: mean 1.23°,
p99 7.08°, and only **356 of 5 060 556 loops (0.007 %)** past 45°.

Rendering the head, torso and hand with and without them produced **0 pixels**
differing by more than 1/255 across 810 000 pixels per view. They were baked
smooth normals with a few export artefacts, so they are cleared — which also
keeps LOD0 shading consistent with the decimated tiers, since decimation cannot
carry split normals at all.

The rule in `01_prepare.py` keys on the *fraction of genuinely hard loops*, not
the maximum, because one corrupt loop should not pin a decision.

---

## 3. UVs: TEXCOORD_0 order is a real bug

Two layers: `UVMap` (tiling detail, 20 cm per tile) and `UVMap_Atlas` (the same
unwrap packed into 0..1).

**Creation order decides the glTF slot.** The atlas layer was created first, so
the tiling UVs landed in `TEXCOORD_1` and the exporter pointed every texture at
`texCoord: 1`. That is legal glTF and renders correctly in Blender — and would
have textured the body with the atlas layout in any loader that assumes
`TEXCOORD_0`. `08_export_gltf.py` now asserts every texture slot samples
`TEXCOORD_0`.

Parameters were swept, not guessed:

| angle | margin | repack | islands | 0..1 coverage | p99/p50 stretch |
|---|---|---|---|---|---|
| 66° | 0.0015 | no | 4754 | 0.176 | 1.065 |
| 85° | 0.0015 | no | 4454 | 0.187 | 1.064 |
| **85°** | **0.0005** | **concave** | **4454** | **0.417** | **1.064** |
| 89° | 0.0005 | concave | 4403 | 0.353 | 1.075 (squash **2.35**) |

85° + concave repack more than doubles coverage at no stretch cost. 89° was
rejected on the squash figure.

The tile scale is derived from *measured* median density, because `pack_islands`
rescales the atlas — an area-based estimate landed 1.6× off (31.9 cm for a 20 cm
target).

---

## 4. Rig fitting

Topology and vertical proportions come from the reference Mixamo skeleton
imported from an actual clip, so bone names and hierarchy cannot drift from the
animation source. Everything else is measured from the mesh.

**Crotch detection.** The obvious test — "is there mesh on the midline?" — is
wrong, because on both sculpts **the thighs touch**: the measured midline half-gap
reaches 0.000 for a 70 mm run below the crotch. That test read 39.2 % of stature;
the real crotch is where *no* surface remains near the midline in the mid-depth
band, at 47.0 % (male) / 48.6 % (female), which matches human inseam proportions.

**Arms are sliced along X, legs along Z.** A T-pose arm is near-horizontal, so a
Z-slice through it degenerates.

**Knee and elbow are chain midpoints.** Verified against the template, whose
femur/tibia (0.24384 / 0.24413 H) and upper arm/forearm (0.15299 / 0.15569 H)
are equal to within 0.3 %. Measured results: femur/tibia 0.991, upper
arm/forearm 1.002.

**Fingers are measured, not templated.** A reference cross-section that resolves
exactly four Y clusters gives the index/middle/ring/pinky axes. The cluster gap
must be **adaptive**: on the male's *right* hand index and middle stay merged at
4 mm and only separate at 3 mm, while the left hand resolves fine at 4 mm. A fixed
4 mm gap failed silently on one hand in four.

Placing the template hand instead put up to 24 finger joints outside the mesh,
because these hands are ~20 % longer than the template and scaling the lateral
spread by that same factor pushed the digits out through the skin.

**The thumb took three attempts** (see §6).

---

## 5. Skin weights: the soft threshold is the whole trick

Bone heat gives good weights — but up to 7–8 influences per vertex, and
glTF/Unity want 4. How you get from 8 to 4 turned out to matter more than
anything else in the pipeline. Same weights, same stress pose, male body:

| reduction method | influences | max edge growth |
|---|---|---|
| unlimited (baseline) | 8 | 8.82 mm |
| Blender `vertex_group_limit_total(4)` | 4 | **40.53 mm** — tears the chest open |
| Blender weight smoothing ×8, then limit | 4 | 27.34 mm |
| fold discarded weight into hierarchy-nearest kept bone | 4 | 26.57 mm |
| **4 Laplacian passes, then soft threshold** | **4** | **7.57 mm** |

Tearing is a **discontinuity** problem, not a magnitude problem. Naive top-4
truncation makes neighbouring vertices discard *different* bones, so their
effective transforms diverge across an edge. The soft threshold subtracts the
5th-largest weight from every weight and clamps at zero: a bone leaving the top 4
fades to exactly zero instead of being cut off, so the field stays continuous.
It beats the unlimited baseline because the smoothing passes help too.

`soft_threshold()` is nine lines. It replaced three more elaborate approaches
that were all worse.

Smoothing runs over the **edge graph**, not spatial neighbours, so it cannot
bleed weight between the two inner thighs or from an arm to the chest.

---

## 6. The thumb, and knowing when to stop

Symptom: after everything above, the worst remaining distortion was 16.5 mm
(male) / 25.5 mm (female), and **100 % of edges past 10 mm involved a thumb bone
or an index vertex beside one**. Rendered, it was a thin filament bridging two
curled fingertips — clearly visible.

Cause: the thumb chain is laid along a straight wrist-centre → tip axis, which
passes through the palm close to the base of the index finger, so bone heat binds
index and palm vertices to `Thumb2`/`Thumb3`.

Attempts:

1. **Measure a proper thumb medial axis.** Slicing along X ignored that a thumb
   runs diagonally (Thumb2 landed 64 mm outside the mesh). A straight PCA axis
   mis-oriented itself on one of four hands and threw Thumb1 136 mm clear of the
   palm. A radial-shell centroid fell into the *gap* between thumb and thenar
   eminence, because the shell spans both.
2. **Segregate the weights** — move thumb-bone weight off non-thumb vertices.
   This made it **worse: 78 mm (male) / 84 mm (female)**, because the lobe cut is
   a hard plane and 4 smoothing passes only blur ~8 mm on a 2 mm mesh. It
   replaced a small discontinuity with a bigger one.
3. **Stop deforming with the thumb.** 9.9 mm / 10.9 mm.

Then the same logic for the distal phalanges, which were the last offenders:

| non-deforming set | worst growth, all clips (female) |
|---|---|
| end markers only | 25.5 mm |
| \+ thumb chain | 10.9 mm |
| **\+ distal phalanges (shipped)** | **4.5 mm** |
| \+ every finger bone (rigid hand) | 2.7 mm |

`distal` was chosen over a fully rigid hand: fingers still articulate at the
knuckle and middle joint, at negligible extra distortion. All 65 bones remain in
the skeleton and still animate, so Unity Humanoid and VRM still map a full hand.

The honest framing: the thumb axis is an approximation the sculpt cannot support
better without hand-authored weights, and the fix is to stop that approximation
from touching skin.

---

## 7. Distortion thresholds must scale with the mesh

The first distortion test used absolute millimetres. It was calibrated on LOD0
(1.96 mm median edge) and immediately condemned the collision proxy, whose median
edge is **19.3 mm** — where 40 mm of growth is a 2.3× stretch, i.e. ordinary joint
bending.

Growth is now normalised by each tier's own median edge length. Calibration:

| case | ratio | visible? |
|---|---|---|
| broken 4-influence truncation | 18.4× | yes, cracks across the chest |
| filament between fingertips | 6.3× | yes |
| shipped weights, LOD0 | 3.4× | no |
| shipped weights, collision proxy | 2.3× | no |

Gate: 5×. Also beware accumulating counters — the first version summed
"edges over 10 mm" across ~74 sampled frames and reported 127 145 where the real
per-frame figure was ~1 700.

---

## 8. Retargeting

Copy each bone's **rotation delta from its own rest pose**, never its absolute
world rotation:

```
delta(b) = src_pose_world_rot(b) @ src_rest_world_rot(b)^-1
want(b)  = delta(b) @ dst_rest_world_rot(b)
basis(b) = (W_parent @ P_b^-1 @ R_b)^-1 @ want(b)
```

Absolute copying forces our bones to adopt Mixamo's roll. Because both rests are
T-poses the bone head→tail directions still agree, so **every direction-based
check passes** while each bone is twisted about its own axis and the mesh shears.
The test therefore compares the rest-relative delta as a quaternion, which is
sensitive to twist. Measured error: **0.00000°**.

That test has an obvious hole — it also reads 0° if the source never animated,
since both deltas would be identity. So the clips are additionally required to
carry real motion (minimum 102.6° of source rotation) and the baked actions to
actually move the rig (minimum 1.04 m of bone travel).

Hip translation is scaled by the hip-height ratio, computed **per clip**, because
Mixamo FBX unit scale varies file to file. All 19 clips here happen to agree at
1.0057, but that is luck, not a guarantee.

---

## 9. Blender 5.2 API notes

- **Actions are layered.** `action.fcurves` no longer exists; curves live under
  `layers → strips → channelbags`. `C.action_fcurves()` handles both forms.
- `mesh.calc_normals_split()` is gone — read `mesh.corner_normals` /
  `mesh.vertex_normals` with `foreach_get("vector", …)`.
- `bmesh.ops.triangulate_faces` does not exist; it is `bmesh.ops.triangulate`.
- `vertex_group_smooth` needs weight-paint context; `mode_set` first.
- **`Image.pixels` holds raw stored values.** Writing 0.5 to an sRGB-tagged image
  reads back 0.502, not 0.735 — `colorspace_settings` only tells the *renderer*
  how to interpret the data. All sRGB↔linear maths is therefore explicit in
  `03_textures.py`. `pixels` is also bottom-up, but reading and writing both go
  through it so the convention cancels; a corner-marker round-trip asserts this.
- FBX import leaves its action behind with a fake user, so `users` never drops to
  zero. Sweep by name or the file ends up with 39 actions instead of 19.
- **FBX has no bone-length field.** On re-import Blender re-derives lengths from
  children: leaf bones get an invented one (`HeadTop_End` came back 144 mm long)
  and branch bones pick a child arbitrarily (`Spine2`, parent of `Neck` and both
  shoulders, was off by 43 mm). Verify **joint-to-joint distances**, which are
  invariant under the axis conversion.
- Importing an animated FBX assigns one of its takes, so the armature arrives
  **posed** (1.06 m of offset). Never check a re-imported rig's pose bone heads
  and conclude the bind data is wrong.
- Node translations in a glTF bone chain **cannot be summed** to get world
  positions — bone nodes carry rotations. Invert the skin's inverse bind matrices
  instead; that also validates the data the GPU really uses.

---

## 10. Things deliberately not done

- **No retopology, no remeshing, no decimation of LOD0, no remove-doubles, no
  normal recalculation, no UV re-projection, no rest-pose rebake.** Any of these
  would break the guarantee that LOD0 is the artist's mesh.
- **No rescaling**, including the female's 1.905 m stature. Flagged in
  `INTEGRATION.md` §6 as a decision for Seede XR.
- **No VRM expressions or lookAt.** Needs eyeball and mouth-interior geometry
  that does not exist. Stubs would be worse than absence.
- **The inner-surface optimisation does not apply** — these bodies are solid, so
  there is no never-visible inner wall to delete.
- **Height/subsurface maps are shipped but not wired into the exported glTF
  material.** Core glTF has no slot for either.
