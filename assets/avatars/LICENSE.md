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

> **Two files here carry a limit on the *manner* of redistribution, not a
> licence exception.** The grant below is CC BY 4.0 with nothing carved out of
> it. But `avatars/vrm/*.vrm` embed metadata *more restrictive* than the grant,
> and the Mixamo motion baked into the GLB/VRM may travel inside a character but
> not as an animation pack. Both are constraints on **how these particular files
> may be passed on**; see [Mixamo](#mixamo-a-limit-on-the-manner-of-redistribution)
> and [the VRM metadata](#vrm-metadata--a-redistribution-limit-on-two-files).

This file is **the authority on what each component in this directory is
licensed under.** For the *wording* of the shipped credit the authority is
`nguo_avatar::ATTRIBUTION_NOTICE`; for project-wide licence policy it is
`../../LICENSING.md`; for how the code consumes the bundle, `../../ui/docs/AVATARS.md`.

Facts of construction come from `docs/ATTRIBUTION.md` and `docs/INTEGRATION.md`
§10, which are the bundle's own documents; where they and this file disagree on a
fact, they are right. `docs/ATTRIBUTION.md` was **corrected in place** on
2026-08-03 so it no longer says "proprietary" — it is no longer byte-identical to
the delivered copy, and the header there records that.

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
| **Runway animation** — 19 clips, retargeted and baked into the GLB/VRM | Adobe Mixamo | Mixamo terms — royalty-free in projects, licensed to the downloading account | **may travel inside a derived character, not as standalone animation data.** The CC BY 4.0 grant covers the modelling data; it does not extend to this motion — see the constraint below | this file; `docs/ATTRIBUTION.md` |
| **Blender 5.2.0 LTS** — the tool the bundle was built with | Blender Foundation | GNU GPL v3 | **none on these assets.** The GPL covers the application, not its output; the assets are unencumbered by it | this file; `docs/ATTRIBUTION.md` |
| **VRM metadata** — the `VRMC_vrm.meta` block inside `avatars/vrm/*.vrm` | Seede XR (our declaration about our own asset) | declares **VRM Public License 1.0**, `allowRedistribution: false` | **narrower than the grant.** Those two files carry a redistribution limit until the export is re-run; the CC BY 4.0 grant over the modelling data is unaffected | this file (below); `../../LICENSING.md` §5 |
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

## Mixamo: a limit on the manner of redistribution

**The CC BY 4.0 grant covers the Seede XR modelling data** — meshes, rigs,
weights, LODs, collision proxies and the derived tiers. It does not cover the
Mixamo motion baked alongside them, because you cannot grant what you do not own:
the animation **data** originates with Adobe Mixamo and is licensed to the account
that downloaded it. Nothing is over-granted by that; it simply bounds what the
grant reaches.

What it means in practice is a constraint on the **manner** of redistribution:

- **Inside a derived character used in the application** — the GLB/VRM with the
  motion baked in — the clips may travel. That is the intended use of Mixamo
  content and is what this bundle ships for.
- **As standalone animation data** — republishing `avatars/glb/*.glb`,
  `avatars/vrm/*.vrm` or any FBX as an **animation pack** — they may not, because
  the clips would then travel on their own and Mixamo is not licensed for that.

So the same file is redistributable as a character and not redistributable as a
motion library, and **anyone we pass these files to has to be told that** — the
notice names Mixamo for exactly this reason. If the assets are ever published as a
CC BY 4.0 asset pack, the clean way is to export it **without** baked Mixamo
clips, which leaves the grant and the pack coextensive.

The motion is retargeted, not copied — per-bone rotation deltas relative to each
bone's own rest pose, transferred onto the Seede XR skeleton, whose joint
positions are measured from the Seede XR meshes. The skeleton's bone *names* and
hierarchy follow the Mixamo convention, which is a naming convention and not a
licensable asset. Re-confirm against Adobe's terms current at release:
<https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html>

---

## VRM metadata — a redistribution limit on two files

`avatars/vrm/nguo_female.vrm` and `nguo_male.vrm` embed a `VRMC_vrm.meta` block
that is **more restrictive than the CC BY 4.0 grant above**. The grant is not
reduced by it and nothing here is over-granted; what it means is that **those two
files, as shipped, carry a redistribution limitation** that the GLBs do not, until
the export pipeline is re-run. VRM consumers read `VRMC_vrm.meta`
programmatically, so the embedded values are what governs a VRM tool's behaviour
regardless of any text file beside it — which is why this is a real constraint and
not a documentation nit.

The field table, because it is what a downstream VRM user actually sees:

| field | shipped value | CC BY 4.0 grants | effect |
|---|---|---|---|
| `allowRedistribution` | `false` | redistribution, to anyone | **narrower than the grant** — a VRM tool will refuse to pass the file on |
| `avatarPermission` | `onlyAuthor` | use by anyone | **narrower than the grant** |
| `licenseUrl` | `https://vrm.dev/licenses/1.0/` | CC BY 4.0 | names a different licence; the CC BY 4.0 grant is the one that governs, and it is recorded in this file |
| `creditNotation` | `required` | attribution required | consistent — CC BY requires it anyway |
| `commercialUsage` | `corporation` | commercial use, unrestricted | narrower than the grant |
| `modification` | `allowModification` | derivatives permitted | consistent |

Every divergence runs in the **restrictive** direction. That is the safe direction
for a grant — a recipient of the VRM gets less than CC BY 4.0 offers, never more —
but it is still a limit we impose on our own asset by accident rather than by
intent, so it is tracked rather than accepted.

**Lifting it needs the export re-run; it cannot be done in this repo.** The
metadata is set in the pipeline's **`scripts/09_export_vrm.py`** (per
`docs/ATTRIBUTION.md`), which is not part of this repository, and hand-editing a
`.vrm` binary is not the fix. So:

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
defaulting to permissive on intellectual property is not a call a build script
should make — that instinct was right; the CC BY 4.0 decision is what now makes
the permissive values the correct ones.

---

## Minimum notice to ship with the application

**The notice text is not written out here. It lives in exactly one place:**

> **`nguo_avatar::ATTRIBUTION_NOTICE`** — `ui/crates/nguo-avatar/src/skin.rs:69`

One consolidated string discharges all three CC BY 4.0 attributions — Seede XR,
the texture sources, the Monk Skin Tone Scale — and credits Mixamo. It is a `pub
const`, so an about screen renders it rather than retyping it, and
`ui/crates/nguo-avatar/tests/bundle.rs` asserts on it.
`nguo_avatar::MONK_ATTRIBUTION` is the Monk line alone, for a credits screen that
splits per component.

**This file quotes it by name on purpose.** Four hand-copied variants of one legal
notice is how the shipped build ended up under-attributing earlier: the copies
drifted, and the shortest ones dropped "licensed under CC BY 4.0" and the licence
URL. Any document that needs the wording cites the const; nothing restates it.

CC BY 4.0 §3(a)(1) also asks that a **link to the licence** and an indication of
**whether the material was modified** be retained. The link is inside the const,
which is why it is there rather than bolted on at the call site; the modification
notice is satisfied by shipping `LICENSES/CC-BY-4.0.txt` next to this file, which
records that the skin maps are retinted/remapped derivatives.

---

## Storage note

The 36 binaries here (6 `.glb`, 2 `.vrm`, 28 `.png` — **238 053 569 bytes,
227.03 MiB** of the 227.2 MiB total) are in **Git LFS**; see `.gitattributes` §3b.
The 28 `.json`, 5 `.md` and the licence text are ~185 KB and stay in normal git so
they diff in review.
Nothing in this directory is embedded at compile time — `nguo-avatar` opens it at
run time — which is what makes LFS safe here and forbidden for `ui/assets/`
(`.gitattributes` §4).
