<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
<!-- This notice FILE is GPL-2.0-or-later, matching every other LICENSE.md in
     the repo (e.g. ui/assets/fonts/LICENSE.md). The ASSETS it describes are
     CC BY 4.0. A notice document is not the thing it documents. -->
# Licensing — `assets/avatars/`

**Everything in this directory is licensed CC BY 4.0.** The Seede XR modelling
data (body meshes, rigs, weights, LODs, collision proxies and every derived
artifact) is licensed **CC BY 4.0 by Seede XR** — maintainer decision,
2026-08-03. The two third-party inputs, the skin textures and the Monk Skin Tone
Scale, already were. So the whole asset stack lands on **one licence and one
credit line**.

Legal text: [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) — canonical,
fetched verbatim from <https://creativecommons.org/licenses/by/4.0/legalcode.txt>
(18 657 bytes, sha256
`9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`).

> **The code is not CC BY 4.0.** nguo-3d's source is **GPL-2.0-or-later**. This
> is a deliberate split-licence arrangement — code under the GPL, assets under
> CC BY — which is the same shape Blender uses. See `../../LICENSING.md` §0 for
> which paths each licence covers.

> ⚠️ **One thing here contradicts this file: the VRM metadata.** `avatars/vrm/*.vrm`
> ships embedded terms that deny redistribution, which CC BY 4.0 grants. The
> binaries are wrong until the export pipeline is re-run. See
> [**the VRM conflict**](#vrm-metadata--open-conflict-with-the-asset-licence)
> below; it is the one open item in this directory.

This file is the repo-side licence record. It derives from the bundle's own
`docs/ATTRIBUTION.md` and `docs/INTEGRATION.md` §10, which are upstream and
authoritative on *facts of construction*; where they and this file disagree on
facts, they are right. On the **licence of the Seede XR data** this file is
newer: `docs/ATTRIBUTION.md` still says "proprietary" throughout, which the
2026-08-03 decision supersedes. That upstream file needs the same edit on the
next bundle build.

Written 2026-08-03, for the **second** bundle. The first (223 MB, four avatars)
was deleted; its bodies had unverified provenance and were an open release
blocker. **This bundle closes it** — see the bodies section.

---

## Per-component

| Component | Owner / licensor | Licence | Obligation | Notice lives in |
|---|---|---|---|---|
| **Body meshes, rigs, weights, LODs, collision proxies** — `avatars/glb/*.glb`, `avatars/vrm/*.vrm`, `previews/*.png`, and everything in `data/` describing them | **Seede XR** (owned outright) | **CC BY 4.0** | **attribution REQUIRED** — CC BY obliges crediting the licensor, so Seede XR must be named in the ship notice, not merely implied by ownership | this file; `LICENSES/CC-BY-4.0.txt`; the application's about/credits screen |
| **Skin textures** — `textures/skin/shared/*.png`, per-tone albedos `textures/skin/tones/skin_albedo_MST01..10.png` | texturecan.com, cgbookcase.com | **CC BY 4.0** | **attribution REQUIRED**, visible wherever the avatars ship | this file; `docs/ATTRIBUTION.md`; the about/credits screen |
| **Monk Skin Tone Scale** — the ten complexion values in `data/skin_tones.json` | Dr Ellis Monk in partnership with Google, <https://skintone.google> | **CC BY 4.0** | **attribution REQUIRED**, visible wherever the tone picker ships | this file; `docs/ATTRIBUTION.md`; the about/credits screen |
| **Runway animation** — 19 clips, retargeted and baked into the GLB/VRM | Adobe Mixamo | Mixamo terms — royalty-free in projects, licensed to the downloading account | **no standalone redistribution.** Note this does *not* travel under CC BY 4.0 — see the constraint below | this file; `docs/ATTRIBUTION.md` |
| **Blender 5.2.0 LTS** — the tool the bundle was built with | Blender Foundation | GNU GPL v3 | **none on these assets.** The GPL covers the application, not its output; the assets are unencumbered by it | this file; `docs/ATTRIBUTION.md` |
| **VRM metadata** — the `VRMC_vrm.meta` block inside `avatars/vrm/*.vrm` | Seede XR (our declaration about our own asset) | declares **VRM Public License 1.0**, `allowRedistribution: false` | ⚠️ **CONTRADICTS CC BY 4.0. Open item** — the export pipeline must be re-run | this file (below); `../../LICENSING.md` §5 |
| **Pipeline scripts** — upstream, not shipped in this directory | Seede XR | not covered by this file | — | `docs/ATTRIBUTION.md` |

## Why CC BY 4.0, and why not BY-SA

"Creative Commons 4.0" is six licences. `CC-BY-4.0` is the specific one chosen,
and the reasoning is recorded so it is legible if revisited:

- **`CC BY-NC 4.0`, `BY-NC-SA`, `BY-NC-ND` — excluded.** They forbid commercial
  use and are on this project's own reject list (`../../LICENSING.md` §2).
  Choosing one would put the project in violation of its own policy.
- **`CC BY-SA 4.0` — considered, not chosen.** Share-alike on derivative
  *assets* is defensible, and someone revisiting this may reasonably prefer it.
  It was not chosen because it is a stronger commitment than was asked for, and
  because it complicates downstream use — anyone retexturing or reposing an
  avatar would inherit a copyleft obligation on their asset — in a way plain BY
  does not.
- **`CC BY-ND 4.0` — excluded.** No-derivatives would forbid retargeting,
  retopology and retinting, which is exactly what this bundle is for.
- **`CC BY 4.0` — chosen.** Permits commercial use, requires attribution, and is
  already the licence on the skin textures and the Monk scale. One licence across
  the whole directory, one credit line, no per-file reasoning at the point of use.

## Bodies: why the release blocker is closed

Both source sculpts (`male_character.blend`, `female_character.blend`) were
modelled by Seede XR, who own them outright and licence them here under
CC BY 4.0. Every avatar is a derivative of those sculpts. There is no
third-party geometry in the bodies, so `../../LICENSING.md` §2 (every shipped
component must permit commercial use) is satisfied at the root — by ownership,
which is also what gives Seede XR standing to licence them at all.

The pipeline never moves a surface vertex, so the shipped LOD0 **is** the Seede XR
mesh. Verified vertex-for-vertex against the source file: worst deviation
**5.07e-7 m** (male) / **5.69e-7 m** (female) — float32 rounding at metre scale,
i.e. not a change. Source: `docs/INTEGRATION.md` §10. That verification is what
carries the ownership claim from the `.blend` to the shipped bytes.

## The three live attribution obligations

All three are CC BY 4.0, and CC BY 4.0 wants a credit **visible to the user**,
not a line in a repo file. All three ship — every avatar is Seede XR geometry
rendered with the skin textures, and the tone picker exposes the MST values — so
all three are **owed**, and the single consolidated notice below discharges them.

The Seede XR obligation is the one that is easy to miss: licensing our own work
under CC BY 4.0 means **we owe attribution to ourselves**. A downstream user must
be able to see who the licensor is, which is why the notice names Seede XR
explicitly rather than leaving it as an unstated copyright.

The skin-texture obligation is recorded as CC BY 4.0 **deliberately**, even
though texturecan's own site states CC0 for its library. That is what was
specified for this project, and crediting satisfies both terms, so the stricter
reading costs nothing and removes the question.

The shipped maps are derivative works — retinted, remapped, repacked — which
CC BY 4.0 permits.

## Mixamo: the one thing that does *not* travel under CC BY 4.0

The animation **data** originates with Adobe Mixamo and is licensed to the
account that downloaded it. Seede XR cannot sub-license it under CC BY 4.0,
because you cannot grant what you do not own. So:

- **Fine:** shipping the avatars with the motion baked in, used inside the
  application. That is the intended use of Mixamo content.
- **Not fine:** publishing `avatars/glb/*.glb`, `avatars/vrm/*.vrm` or any FBX as
  a **downloadable asset pack**, because the animation data would then travel
  standalone, and Mixamo is not licensed for redistributing clips on their own.

**This is a real tension with CC BY 4.0 and must not be smoothed over.** CC BY
4.0 on the geometry invites downstream redistribution; the Mixamo terms restrict
redistribution of the baked clips. The honest reading is that the CC BY 4.0 grant
covers the **Seede XR modelling data** — meshes, rigs, weights, LODs, collision
proxies — and not the third-party motion baked alongside it. Anyone redistributing
these files should be told that. Raise it with the maintainer before the assets
are published as a pack; the clean fix is to ship the CC BY 4.0 asset pack
**without** baked Mixamo clips.

The motion is retargeted, not copied — per-bone rotation deltas relative to each
bone's own rest pose, transferred onto the Seede XR skeleton, whose joint
positions are measured from the Seede XR meshes. The skeleton's bone *names* and
hierarchy follow the Mixamo convention, which is a naming convention and not a
licensable asset. Re-confirm against Adobe's terms current at release:
<https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html>

---

## VRM metadata — OPEN CONFLICT with the asset licence

**`avatars/vrm/nguo_female.vrm` and `nguo_male.vrm` currently state terms that
contradict CC BY 4.0.** This is the one asset in this directory whose stated
terms are wrong, and it matters more than a documentation nit: VRM consumers read
the embedded `VRMC_vrm.meta` block programmatically and will believe it over any
text file shipped beside it.

| field | shipped value | CC BY 4.0 says | conflict |
|---|---|---|---|
| `allowRedistribution` | `false` | redistribution is **granted** to anyone | **direct contradiction** |
| `avatarPermission` | `onlyAuthor` | anyone may use the material | **direct contradiction** |
| `licenseUrl` | `https://vrm.dev/licenses/1.0/` | the licence is CC BY 4.0 | **wrong licence named** |
| `creditNotation` | `required` | attribution required | consistent — CC BY already implies this |
| `commercialUsage` | `corporation` | commercial use permitted, unrestricted | narrower than the grant |
| `modification` | `allowModification` | derivatives permitted | consistent |

Two statements cannot both be true. Until this is fixed, the VRM files
misrepresent their own licence in the direction of being *more* restrictive than
granted — which is the less dangerous direction, but still wrong.

**What has to happen — an open item requiring the export pipeline to be re-run.**
The metadata is set in the pipeline's **`scripts/09_export_vrm.py`** (per
`docs/ATTRIBUTION.md`). The binaries cannot be corrected in this repo; editing a
`.vrm` by hand is not the fix and this repo does not own the export script. So:

1. Set `allowRedistribution: true` — CC BY 4.0 grants it.
2. Widen `avatarPermission` from `onlyAuthor` to match the grant
   (`everyone`, per the VRM 1.0 schema).
3. Keep `creditNotation: required` — CC BY 4.0 requires attribution anyway, so
   this field was already correct.
4. Point `licenseUrl` (or `otherLicenseUrl`, whichever the VRM 1.0 schema
   requires for a non-VRM licence) at
   <https://creativecommons.org/licenses/by/4.0/>.
5. Re-export both VRM files and replace them here.

Review the widened values against actual commercial intent as part of that
change. The `VRMC_vrm.meta` block shipped restrictive originally because
permissive defaults on intellectual property is not a call a build script should
make — that instinct was right; the CC BY 4.0 decision is what now makes the
permissive values the correct ones.

---

## Minimum notice to ship with the application

**Use this consolidated block.** It is what an about screen should paste: it
satisfies all three CC BY 4.0 attributions — Seede XR, the texture sources, and
the Monk Skin Tone Scale — and credits Mixamo, in one place.

> **Avatars.** Avatar body meshes, rigs and skin data © Seede XR, licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Skin textures based
> on material from texturecan.com and cgbookcase.com (CC BY 4.0). Complexion
> range based on the Monk Skin Tone Scale by Ellis Monk in partnership with
> Google (CC BY 4.0). Character animation from Adobe Mixamo.

For reference, the bundle's `docs/ATTRIBUTION.md` states this minimum notice
**verbatim** — it predates the CC BY 4.0 decision and so says "©" where the
consolidated block above says "licensed under CC BY 4.0":

> Avatars © Seede XR. Skin textures based on material from texturecan.com and
> cgbookcase.com (CC BY 4.0). Complexion range based on the Monk Skin Tone Scale
> by Ellis Monk (CC BY 4.0). Character animation from Adobe Mixamo.

The upstream per-component wordings, also verbatim, if the credits screen splits
the notice into lines:

> Skin textures based on material from texturecan.com and cgbookcase.com,
> licensed CC BY 4.0.

> Complexion range based on the Monk Skin Tone Scale by Ellis Monk
> (CC BY 4.0).

CC BY 4.0 §3(a)(1) also asks that a **link to the licence** and an indication of
**whether the material was modified** be retained. The link is in the block above;
the modification notice is satisfied by shipping `LICENSES/CC-BY-4.0.txt` next to
this file, which records that the skin maps are retinted/remapped derivatives.

---

## Storage note

The 36 binaries here (6 `.glb`, 2 `.vrm`, 28 `.png` — 227.1 MiB of the 227.3 MiB
total) are in **Git LFS**; see `.gitattributes` §3b. The 28 `.json`, 4 `.md` and
the licence text are 239 KB and stay in normal git so they diff in review.
Nothing in this directory is embedded at compile time — `nguo-avatar` opens it at
run time — which is what makes LFS safe here and forbidden for `ui/assets/`
(`.gitattributes` §4).
