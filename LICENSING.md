<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Licensing policy

nguo-3d's **code** is **GPL-2.0-or-later**; its **avatar assets** are
**CC BY 4.0**. Every dependency it ships must permit **commercial use**. This
document states the policy; `deny.toml` enforces the dependency half.

---

## 0. What licence covers what — the split

Most of this document is about *other people's* work. This section is about
**ours**, because it is two licences and that has to be stated rather than
inferred. A split like this is entirely normal: Blender's code is GPL while its
bundled assets and demo files are licensed separately, for exactly the reason
below — a licence written to keep software source open is the wrong instrument
for a mesh.

| Paths | Licence | SPDX header on new files |
|---|---|---|
| **All source code** — `core-engine/**`, `ui/**`, `tools/**`, the Makefile, CI, and every `.rs` / `.wgsl` / `.toml` / `.py` / `.sh` in the tree | **GPL-2.0-or-later** | `SPDX-License-Identifier: GPL-2.0-or-later` |
| `bridges/blender/**` — the Blender extension | **GPL-3.0-or-later** | `SPDX-License-Identifier: GPL-3.0-or-later` |
| **`assets/avatars/`** — the delivered bundle: Seede XR body meshes, rigs, weights, LODs, collision proxies, previews, the JSON describing them, and the bundle's own `README.md` / `docs/*.md` | **CC BY 4.0** (licensor: Seede XR) | `SPDX-License-Identifier: CC-BY-4.0` |
| Two repo-side exceptions inside that directory: `assets/avatars/LICENSE.md` (the notice) and `assets/avatars/LICENSES/CC-BY-4.0.txt` (the licence text) | notice is **GPL-2.0-or-later** like every other `LICENSE.md`; the licence text is Creative Commons' own | a notice document is not the thing it documents |
| `ui/assets/**` — fonts, icons, brand, splash | per-component; see §5 | — |
| `third_party/**` — vendored C++ | upstream licences; see §5 | unchanged, do not add headers |

**The avatar assets are CC BY 4.0 by maintainer decision, 2026-08-03.** Seede XR
owns the modelling data outright (§4) and licenses it under CC BY 4.0 rather than
keeping it proprietary. `CC BY 4.0` specifically — not one of the other five
Creative Commons 4.0 variants:

- `CC BY-NC 4.0` / `BY-NC-SA` / `BY-NC-ND` would **forbid commercial use** and are
  on this project's own reject list (§2). Choosing one would make the project
  violate its own policy.
- `CC BY-SA 4.0` was **considered and not chosen** — share-alike on derivative
  assets is defensible but a stronger commitment than was asked for, and it
  burdens anyone retexturing or reposing an avatar.
- `CC BY 4.0` permits commercial use, requires attribution, and is already the
  licence the skin textures and the Monk Skin Tone Scale carry — so the entire
  asset stack lands on **one licence and one credit line**.

Full reasoning, per-component table, and the ship notice:
`assets/avatars/LICENSE.md`. Legal text:
`assets/avatars/LICENSES/CC-BY-4.0.txt` (canonical, fetched verbatim, sha256
`9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`) — the same
convention `core-engine/LICENSES/GPL-2.0-or-later.txt` follows for the code.

**Two consequences worth stating outright:**

1. **CC BY 4.0 obliges us to attribute ourselves.** Licensing our own work under
   CC BY means a downstream user must be able to see who the licensor is, so
   **Seede XR must be named in the shipped credit notice** — not left implicit in
   a copyright line. §5 tracks this as owed alongside the two third-party
   attributions.
2. **The grant is clean; two things limit the *manner* of redistribution.** The
   CC BY 4.0 grant over the Seede XR modelling data has nothing carved out of it.
   But the grant does not *reach* the baked Mixamo motion — we cannot sub-license
   what we do not own — so the avatar files may travel as a character and not as an
   animation pack; and the metadata embedded in the two `.vrm` files is narrower
   than the grant, so those particular files carry a redistribution limit until the
   export is re-run. Neither is a licence exception. Both are in §5.

## 1. The project licence (code)

**GPL-2.0-or-later**, deliberately — it is Blender's licence, and Blender is
the ecosystem this tool lives in. A GPL-2.0-or-later core can share code with,
be extended by, and ship alongside that ecosystem without a relicensing
argument. The `-or-later` half matters as much as the `-2.0`: it keeps the
GPLv3 upgrade path open, which is what lets us take GPL-3.0-or-later and
LGPL-3.0-or-later dependencies at all.

The Blender extension under `bridges/blender/` is **GPL-3.0-or-later**; see its
own header.

Every source file carries `SPDX-License-Identifier: GPL-2.0-or-later`.

## 2. The commercial-use rule

> **Every dependency must permit commercial use. Where a capability is only
> available under a licence that does not, we implement it ourselves.**

### Copyleft is not non-commercial

This is the single most common misreading of the rule, so it is stated once,
plainly, and should not be re-litigated:

**GPL, LGPL, MPL and AGPL all permit commercial use.** They place conditions on
the *terms of distribution* — source availability, same-licence derivatives,
no further restrictions — not on whether money changes hands. Selling GPL
software is explicitly allowed and always has been. A GPL dependency is not a
commercial-use problem and must never be flagged as one; doing so sends
somebody off to rewrite a perfectly usable library for no reason.

### What actually fails the test

Licences that restrict *who may use the software, or for what purpose*:

| Family | Examples | Why it fails |
|---|---|---|
| Non-commercial Creative Commons | `CC BY-NC`, `CC BY-NC-SA`, `CC BY-NC-ND` | forbids commercial use outright |
| Server Side Public License | `SSPL-1.0` | not OSI-approved; service-provider conditions |
| Business Source | `BUSL-1.1`, "Business Source License" | non-production/non-commercial until a change date |
| Elastic | `Elastic-2.0` | restricts managed-service and competitive use |
| Academic / research-only | "for research purposes only", "non-commercial research use", most SMPL-family model weights | use restricted by purpose |
| Field-of-use riders | JSON.org's *"The Software shall be used for Good, not Evil"* | unenforceable and non-free; a field-of-use restriction either way |

Separately rejected, for a different reason: **`GPL-2.0-only`**. It permits
commercial use fine. It is rejected because it is incompatible with a
GPL-2.0-**or-later** tree that needs to be able to upgrade to GPLv3 — a
GPLv2-only dependency pins the whole work at v2 and shuts the door on every
GPLv3 and LGPLv3 component. Note that the deprecated bare SPDX id `GPL-2.0`
normalises to `GPL-2.0-only`, which is why bare `GPL-2.0` is absent from both
allow-lists.

**`BSL-1.0` on the allow-list is the Boost Software License** — permissive,
OSI-approved, entirely fine. It is *not* the Business Source License
(`BUSL-1.1`), which is rejected above. The two are confused often enough that
it is worth the sentence.

## 3. The allow-list

`deny.toml` is the **machine-checked source of truth**. This table follows it.
If they disagree, `deny.toml` is right and this document is stale.

```
MIT                            MIT-0
Apache-2.0                     Apache-2.0 WITH LLVM-exception
BSD-2-Clause                   BSD-3-Clause
0BSD                           ISC
Zlib                           MPL-2.0
LGPL-2.1-or-later              LGPL-3.0-or-later
GPL-2.0-or-later               GPL-3.0-or-later
Unicode-3.0                    Unicode-DFS-2016
CC0-1.0                        Unlicense
BSL-1.0  (Boost)               OFL-1.1
CC-BY-4.0
```

### `CC-BY-4.0` — added 2026-08-03, and why it is on a *Cargo* allow-list

It is there because **we now ship it, and we now grant it.** Three separate
things in `assets/avatars/` are CC BY 4.0: the skin textures, the Monk Skin Tone
Scale values, and — since the 2026-08-03 decision in §0 — **all of the Seede XR
modelling data, licensed out by us**. So it is both a licence the project
receives work under and the licence it publishes its own assets under. A list
that claims to enumerate our licences has to include it. It is permissive and
GPL-compatible; its only condition is attribution, tracked as *owed* in §5.

Two honest caveats, so nobody reads more into the entry than is there:

- **Zero crates in either graph are CC-BY-4.0.** The Cargo-side widening is
  nominal today. It was still added to all three files rather than to
  `LICENSING.md` alone, because `make licenses-sync` requires the three to hold
  the same set and there is no mechanism for "documented but not enforced".
- **A CC licence arriving on an actual crate deserves review on its merits.**
  Creative Commons does not recommend CC BY for software; it says nothing about
  source availability or patents. This entry exists for assets. If a dependency
  shows up under it, treat the allow-list hit as a prompt, not a clearance.

Because no crate uses it, `cargo-deny` emits
`warning[license-not-encountered]: "CC-BY-4.0" unmatched license allowance` on
every run. That warning is **expected and must not be "fixed"** by deleting the
entry — nine other allowances (`MIT-0`, `MPL-2.0`, `OFL-1.1`, `LGPL-3.0-or-later`,
`Unicode-DFS-2016` and others) produce the same warning for the same reason. The
run is still `licenses ok`; verified 2026-08-03, `make licenses` exit 0.

**There are no scoped exceptions.** `deny.toml` has no
`[[licenses.exceptions]]` block and neither `about.toml` has a per-crate
`accepted` list. Everything in the graph is on the list above.

### The one exception there used to be, and how it was removed (2026-08-02)

Until this date the list carried a single scoped exception, for
`epaint_default_fonts` — the crate holding egui's bundled fonts. Its licence
expression was `(MIT OR Apache-2.0) AND <a font licence>`, and that third term
was the **only** licence in the tree whose GPL compatibility was not clean: it
permitted commercial use, so it passed this policy, but its rename-on-modify
clause is why the FSF does not list it as GPL-compatible. The licence is not
named here on purpose — it is no longer reachable from this tree, and the
maintainer asked that no trace of it remain in the policy files.

It is gone, and the exception was **deleted rather than narrowed**. `egui` is
now declared `default-features = false` (`ui/Cargo.toml`). `default_fonts` is
egui 0.33.3's *only* default feature — verified in the registry copy's
`Cargo.toml` (`default = ["default_fonts"]`), so nothing else was lost — and
turning it off drops `epaint_default_fonts` from the graph entirely, lockfile
included. egui then ships no font at all, so the overlay installs its own:
**Space Mono**, SIL OFL 1.1, vendored in `ui/assets/fonts/` and embedded by
`nguo-gpu/src/overlay.rs` via `include_bytes!`. OFL-1.1 was already on the
allow-list (it is Inter's licence too). Net: one fewer licence in the tree, one
fewer exception, and the overlay's live numbers stopped jittering.

Verified after the change:

```
cargo tree -p nguo-app -e normal --features debug-overlay | grep epaint_default_fonts → no output
grep -n epaint_default_fonts ui/Cargo.lock                                            → no output
```

> **Verification note, kept because the methodology error is worth preserving.**
> While the exception still existed, an audit reported that
> `epaint_default_fonts` was in `ui`'s *default* graph, which would have removed
> the blast-radius argument for scoping it. It was not. `egui` and `egui-wgpu`
> are `optional = true` behind the `debug-overlay` feature
> (`ui/crates/nguo-gpu/Cargo.toml:34-35`), and at the time:
>
> ```
> cargo tree -p nguo-app -e normal                          → 0 occurrences
> cargo tree -p nguo-app -e normal --features debug-overlay → 1 occurrence
> ```
>
> The audit had run `cargo tree` at **workspace root**, which unions features
> across every member — so `nguo-app`'s own `debug-overlay` feature leaked into
> the graph being inspected. The crate is now at 0 occurrences under both
> invocations, but the lesson stands: when checking whether an optional
> dependency reaches a shipped binary, always scope to the binary crate
> with `-p`.

### The two allow-lists, and the check that keeps them honest

- `deny.toml` — enforcement (`cargo-deny`). Blocking, runs in CI.
- `core-engine/about.toml` and `ui/about.toml` — attribution (`cargo-about` →
  `<workspace>/LICENSES/THIRD_PARTY.md`). The two are **byte-identical**; TOML
  has no include, so a check stands in for one.

All three must hold the same set. They had **silently drifted**, in both
directions, because a comment claimed they were identical and nothing verified
it: `about.toml` accepted bare `GPL-2.0` (the exact id `deny.toml` forbids) and
omitted `LGPL-2.1-or-later` (the licence planegcs is vendored under).

`make licenses-sync` now diffs the arrays and `cmp`s the two `about.toml` (and
the two `about.hbs` templates). It runs as part of `make licenses`, needs no
tools, no build and no network.

### Attribution is wired up (2026-08-02)

`cargo-about 0.9.1` is installed, `make attribution` regenerates
`core-engine/LICENSES/THIRD_PARTY.md` (172 crates) and
`ui/LICENSES/THIRD_PARTY.md` (**482** crates, measured 2026-08-03 — 459 before the
avatar work, 478 after it, then `spade` and `nguo-triangulate` landed; the count
moves whenever `ui`'s graph does, so regenerate rather than trust this number), and
`make licenses` depends on
`make attribution-check`, which regenerates into a temp dir and **fails the
build on any difference** — a stale attribution file cannot sit there quietly.
CI runs the same make target (`.github/workflows/licenses.yml`) with
`cargo-about` pinned to 0.9.1, because the check is a byte-for-byte diff.

`about.toml` pins the same five target triples as `deny.toml`'s `[graph]`, so
the generated output is identical on macOS and on the Linux runner.

Everything in §5 that `cargo-about` cannot see — planegcs, Eigen, Lucide,
Tabler, Inter, ppf-contact-solver, the ClothDrop reference — is a static table
at the bottom of `about.hbs`, so it lands in **both** generated files and
cannot be dropped without a template edit that `attribution-check` then
catches.

## 4. The ground-up rule

> Where a capability is available only under a licence that forbids commercial
> use, **we do not take the dependency. We implement it ourselves.**

Not "we take it and hope", not "we ship a non-commercial add-on". The licence
constraint decides the build-vs-buy question before the engineering merits do.

### The one live instance: D-5, the parametric body model

The SMPL family (SMPL, SMPL-X, STAR and relatives) is the research standard for
parametric human bodies, and a great deal of it — the model weights in
particular — ships under research-only or otherwise non-commercial terms. That
is exactly the failure mode in §2: not copyleft, an actual restriction on
commercial use. It is incompatible with this project, so it is not an option,
however convenient it would be.

The consequence is visible and deliberate: **the avatar milestone is blocked**
pending either an openly-licensed parametric body model or one we build
ourselves. It is decision **D-5** in the project's (unpublished) decision log,
recorded as open and needing the maintainer: *"find an openly-licensed
parametric body, build/train our own, or license one commercially."*

That block is the policy working, not a hole in it. The alternative was taking
the dependency and discovering the problem at the point of sale. Everything
downstream is designed to not care which way D-5 resolves: the avatar enters
the pipeline as a mesh plus a measurement set, so measurement binding and
cloth-body collision are written against that interface and not against
anybody's model format.

### RELEASE BLOCKER — **CLOSED, 2026-08-03**: the avatar bundle's body provenance

**The blocker is closed, by replacement rather than by argument.** It is written
up here in full because the mechanism is the lesson, not because anything is
still outstanding.

**What it was.** The first bundle (223 MB, four rigged avatars, 94 files) was
built on four source body meshes whose authors and licences could not be
established — two carried only a ZBrush exporter banner, one only a 3ds Max
banner, one nothing at all. §2 requires a licence that positively permits
commercial use; *unknown* is not that, and a derived work inherits its source.
All four avatars, their collision meshes and their preview renders inherited the
uncertainty, so shipping them commercially would have been a policy violation
rather than a grey area.

**How it was cleared.** The bundle was rebuilt, not re-licensed. That whole tree
was **deleted** and replaced with a second bundle (227 MB, two avatars, 68
files) whose bodies are **modelled by Seede XR and owned outright** — both
source sculpts (`male_character.blend`, `female_character.blend`) are ours, so
§2 is satisfied by ownership and there is no third-party grant to establish.
This was the "rebuild from a body of known origin" route; the alternative
(chase four unknown authors for a written licence) was never going to close.

That ownership is also what gives Seede XR **standing to license** the data, which
it now does: **CC BY 4.0**, per the decision in §0. The blocker asked whether we
were allowed to ship the bodies at all; the answer turned out to be that we own
them and can hand them onward.

The pipeline never moves a surface vertex, so the shipped LOD0 **is** the Seede XR
mesh — verified vertex-for-vertex against the source file, worst deviation
**5.07e-7 m** (male) / **5.69e-7 m** (female), which is float32 rounding at
metre scale. That verification is what makes the ownership claim reach the
shipped bytes instead of stopping at the `.blend`. Source:
`assets/avatars/docs/INTEGRATION.md` §10.

**What is left is smaller and is not this blocker.** The new bundle brings these
live items, all tracked in §5 and in `assets/avatars/LICENSE.md`: **three**
CC BY 4.0 attribution obligations that are **owed** (Seede XR itself, the skin
textures, the MST scale), and **two constraints on the manner of redistribution** —
the baked Mixamo clips, and the `VRMC_vrm.meta` block in the two `.vrm` files,
which is narrower than the grant until the export pipeline is re-run. None of them
blocks commercial use of the geometry, and none is an exception to the CC BY 4.0
grant; they constrain the credits screen and the distribution channel.

**D-5 is untouched and still open.** D-5 is "no openly-licensed *parametric*
body model exists for us". This was "the *static* bodies we have came from
nowhere traceable". Closing the second does not close the first — we now own two
static bodies, not a parametric model. If D-5 is ever resolved by building our
own, these CC0 sources remain the candidate starting points, and the pipeline
takes a swap by adding an entry to `AVATARS` in its `scripts/nguo_config.py`
(landmarks are measured from the geometry, so the rig fits whatever standing
biped it is pointed at):

| Source | Contents |
|---|---|
| [Blender Studio Human Base Meshes](https://studio.blender.org/characters/human-base-meshes/) | 17 meshes, male + female, clean topology |
| [thebasemesh.com](https://www.thebasemesh.com/) | 1250+ base meshes |
| [Ryan King Art base meshes](https://ryankingart.gumroad.com/l/WbtVG) | male + female sculpting bases |
| [MakeHuman](http://www.makehumancommunity.org/) | parametric human generator; its meshes/assets are CC0 — the *application* being AGPL does not affect exported meshes |

**Still excluded:** SMPL / SMPL-X / STAR, for the §2 reason already given.

None of this is enforceable by `cargo-deny` — it sees Cargo dependencies only.
The check is this document, `assets/avatars/LICENSE.md`, and review.

## 5. Component attribution

Cargo dependencies are covered wholesale by the allow-list and by
`cargo-deny` (§3). The table below covers everything that `cargo-deny` cannot
see — vendored sources, assets and references — plus the two Cargo-side items
worth naming individually.

| Component | Licence | Why it is compatible | Notice lives in |
|---|---|---|---|
| **nguo-3d** itself | GPL-2.0-or-later | — | `core-engine/LICENSE`, `core-engine/LICENSES/GPL-2.0-or-later.txt`, SPDX header on every file |
| **ppf-contact-solver** (ZOZO) — the solver is a port of it | Apache-2.0 | Apache-2.0 is one-way compatible with GPLv3; conveyed here under GPL-2.0-or-later, which the `-or-later` grant permits | `core-engine/LICENSES/README.md` |
| **planegcs** (FreeCAD) — vendored C++, 2D constraint solver (D-4) | LGPL-2.1-or-later | LGPL-2.1-**or-later** relicenses to LGPL-3.0, which is compatible with GPLv3; our `-or-later` tree can reach it. Obligation met by shipping the sources unmodified | `third_party/planegcs/PROVENANCE.md` §1 |
| **Eigen** 3.4.1 — vendored headers, planegcs' linear algebra | MPL-2.0 (with some BSD and LGPL-2.1 parts) | MPL-2.0 §3.3 explicitly permits distributing a Larger Work under a "Secondary License", which the MPL defines as including GPL v2.0 or later. BSD parts permissive; LGPL parts ride the same relicensing story as planegcs. `EIGEN_MPL2_ONLY` deliberately not set | `third_party/planegcs/PROVENANCE.md` §2, `third_party/planegcs/vendor/eigen/COPYING.*` |
| **Lucide** icons — vendored SVGs | ISC | Permissive, GPL-compatible. Some icons derive from Feather (MIT) | `ui/assets/icons/LICENSE.md` |
| **Tabler** icons — vendored SVGs, fallback where Lucide has no glyph | MIT | Permissive, GPL-compatible | `ui/assets/icons/LICENSE.md` |
| **Inter** v4.1 (Regular) — vendored TTF, embedded via `include_bytes!` | SIL OFL 1.1 | GPL-compatible free-software font licence; conditions (keep the licence, do not sell the font alone, rename on modification) met by shipping it byte-identical with `OFL.txt` | `ui/assets/fonts/LICENSE.md`, `ui/assets/fonts/OFL.txt` |
| **Space Mono** (Regular) — vendored TTF, embedded via `include_bytes!`; the debug overlay's only font | SIL OFL 1.1 | Same as Inter above. Replaced egui's bundled fonts, which is how the last non-GPL-clean licence left the tree — see §3 | `ui/assets/fonts/LICENSE.md`, `ui/assets/fonts/SpaceMono-OFL.txt` |
| **self_cell** (transitive: `iced` → `cosmic-text`) | `Apache-2.0 OR GPL-2.0-only` | Dual-licensed; we take the **Apache-2.0** branch. The GPL-2.0-only branch is not exercised and does not pin our tree | this table |
| **ClothDrop-Reborn** — design/behaviour reference. Read in a local working tree under `instructions/`, which is gitignored: it is **not** shipped and not in a clone | GPL-3.0-or-later | Read as a reference only. Pasting its code into the GPL-2.0-or-later Rust tree would force the whole work to GPLv3 — permitted by our `-or-later` grant, but a decision, not an accident. Flag it in review if it ever happens. A Blender add-on we ship may be GPL-3.0-or-later regardless | upstream `LICENSE` (verified locally: GNU GPL v3, 29 June 2007) |
| **Avatar modelling data** — the body meshes, rigs, weights, LODs, collision proxies and previews behind both avatars in `assets/avatars/`; **ours, not a third party's** | **CC BY 4.0**, licensor **Seede XR** | Owned outright, so §2 is satisfied at the root and Seede XR has standing to license it onward. Released CC BY 4.0 by maintainer decision 2026-08-03 (§0). **The former release blocker is CLOSED — see §4.** **Attribution is REQUIRED, including to ourselves** | `assets/avatars/LICENSE.md`, `assets/avatars/LICENSES/CC-BY-4.0.txt`, `assets/avatars/docs/ATTRIBUTION.md`, and the credits screen |
| **Avatar skin textures** — refined from downloaded PBR skin scans (texturecan.com *Human Face Skin Texture*; cgbookcase.com *human-skin1*) | **CC BY 4.0** | Permissive, GPL-compatible. **Attribution is REQUIRED** — see the obligation note below. Recorded as CC BY 4.0 deliberately though texturecan's site states CC0: crediting satisfies both, so the stricter reading costs nothing. Shipped maps are retinted/remapped derivatives, which CC BY 4.0 permits | `assets/avatars/LICENSE.md`, `assets/avatars/docs/ATTRIBUTION.md`, and the credits screen |
| **Monk Skin Tone (MST) Scale** — the ten skin-tone values, by Dr Ellis Monk in partnership with Google (<https://skintone.google/>), pinned in `assets/avatars/data/skin_tones.json` | CC BY 4.0 | Permissive, GPL-compatible. **Attribution is REQUIRED** — see the obligation note below this table | `assets/avatars/LICENSE.md`, `assets/avatars/docs/ATTRIBUTION.md`, and the credits screen once the tone picker ships |
| **Mixamo animation clips** — 19 Adobe Mixamo runway clips, retargeted and baked into the avatar GLBs/VRMs | Adobe Mixamo terms — free to use, including commercially; licensed to the downloading account | Baked into a derived character, which is the intended use. **Our CC BY 4.0 grant covers the modelling data and does not reach this motion** — we cannot sub-license what we do not own, and nothing is over-granted by that. It bounds the **manner of redistribution**: these files may travel as a character, **not as a standalone animation pack**, because the clips would then travel as animation data on their own. Re-confirm against Adobe's current terms before release | `assets/avatars/LICENSE.md`, `assets/avatars/docs/ATTRIBUTION.md` |
| **VRM metadata** — the `VRMC_vrm.meta` block embedded in `assets/avatars/avatars/vrm/*.vrm` | declares **VRM Public License 1.0**, `allowRedistribution: false`, `avatarPermission: onlyAuthor` | **Narrower than the CC BY 4.0 grant on the same files, so those two files carry a redistribution limitation** the GLBs do not. Every divergence runs restrictive — nothing is over-granted — but VRM consumers read the metadata programmatically, so it is what governs a VRM tool's behaviour. Lifting it means re-running the export; see the note below | `assets/avatars/LICENSE.md` (§ VRM metadata), `assets/avatars/docs/ATTRIBUTION.md` |
| **Blender** 5.2.0 LTS — the tool the avatar bundle was built with | GNU GPL v3 | **No obligation on the assets.** The GPL covers the application, not its output; a mesh exported from Blender is unencumbered by Blender's licence. Listed only because the question gets asked | `assets/avatars/docs/ATTRIBUTION.md` |
| **Brand assets** — `ui/assets/brand-icon/`, `ui/assets/images/splash_screen_image.png` | **unverified** — no attribution file present | Believed original work by Seede XR, but nothing in the tree records it. Needs a `LICENSE.md` alongside the other asset directories | *nowhere yet* |

### The three active attribution obligations — all CC BY 4.0, all OWED

**CC BY 4.0 requires a credit visible to the user**, not a line in a repo file.
Three components in `assets/avatars/` carry that condition, and all three
**ship**: every avatar is Seede XR geometry rendered with the skin textures, and
the tone picker exposes the MST values. So all three are owed **now**, not
conditionally.

| Owed to | For | Attaches to |
|---|---|---|
| **Seede XR** — us | the modelling data we licensed out under CC BY 4.0 (§0) | every avatar mesh, rig, LOD, collision proxy and preview |
| **texturecan.com, cgbookcase.com** | the skin textures | `textures/skin/**` |
| **Ellis Monk / Google** | the Monk Skin Tone Scale | the tone values in `data/skin_tones.json`, which the picker exposes |

The first is the one that is easy to miss: **licensing our own work under CC BY
4.0 means we owe attribution to ourselves.** A downstream user must be able to
see who the licensor is, so Seede XR has to be *named* in the shipped notice, not
left implicit in a copyright line.

**One consolidated notice discharges all three, and it lives in exactly one
place:** **`nguo_avatar::ATTRIBUTION_NOTICE`**
(`ui/crates/nguo-avatar/src/skin.rs:69`), with `nguo_avatar::MONK_ATTRIBUTION` for
the Monk line alone. A `pub const` an about screen renders directly, asserted on by
`ui/crates/nguo-avatar/tests/bundle.rs`.

**This document quotes it by name and does not restate the text.** Hand-copied
variants of the notice are what produced an under-attributing build earlier: the
copies drifted, and the shorter ones lost "licensed under CC BY 4.0" and the
licence URL that CC BY 4.0 §3(a)(1)(C) asks for. One string, one authority — change
it in `skin.rs` and every reader gets the change.

None of this is enforceable by any tool in this repo — `cargo-deny` sees Cargo
dependencies only, and `cargo-about` only reads crate metadata — so it is a
**review item on the PR that builds the about/credits screen**.

### The two redistribution constraints, and why neither is a licence exception

**The CC BY 4.0 grant in §0 stands with nothing carved out of it.** Seede XR's
modelling data — meshes, rigs, weights, LODs, collision proxies, the derived tiers
and the previews — is licensed CC BY 4.0, attribution required, and that is the
whole obligation on our own work. What follows constrains **how two of the shipped
files may be passed on**. Neither reduces the grant, and neither over-grants
anything.

**1. The Mixamo motion is outside the grant, so the files travel as a character
and not as a motion library.** The 19 clips are Adobe's, licensed to the account
that downloaded them; we cannot sub-license them under CC BY 4.0. Baked into a
derived character used in the application, they may travel — that is the intended
use of Mixamo content. Republished as a standalone animation pack (GLB, VRM or
FBX), they may not. If the assets are ever released as a CC BY 4.0 asset pack, the
clean way is to export it **without** baked clips, which makes the pack and the
grant coextensive.

**2. The `VRMC_vrm.meta` block in the two `.vrm` files is narrower than the
grant.** `assets/avatars/avatars/vrm/nguo_female.vrm` and `nguo_male.vrm` embed
`allowRedistribution: false`, `avatarPermission: onlyAuthor` and a `licenseUrl`
pointing at the VRM Public License 1.0. CC BY 4.0 grants redistribution and
derivatives to anyone with attribution; the embedded values withhold that. Every
divergence runs in the **restrictive** direction, so a VRM recipient gets less than
CC BY 4.0 offers and never more — but a VRM consumer reads the block
programmatically, so it is what governs that tool's behaviour, and those two files
therefore carry a redistribution limitation the GLBs do not.

**Lifting it needs the export re-run, and cannot be done in this repo.** The
metadata is written by the avatar pipeline's **`scripts/09_export_vrm.py`**, which
lives upstream and is not part of this repository; hand-editing a `.vrm` binary is
not the fix. Target values: `allowRedistribution: true`, `avatarPermission` widened
to `everyone`, `creditNotation: required` kept (CC BY requires attribution anyway,
so that field was already right), and `licenseUrl`/`otherLicenseUrl` pointed at
<https://creativecommons.org/licenses/by/4.0/> — then replace both binaries here.
Review the widened values against actual commercial intent as part of that change.
Field-by-field table: `assets/avatars/LICENSE.md`.

## 6. Adding a dependency

1. **Check the licence before you add it.** `cargo info <crate>`, or read
   `license` in its `Cargo.toml`. If it is on the §3 allow-list, continue. If
   it is on the §2 reject list, stop — §4 applies, you are writing it yourself.
   If it is neither, bring it up rather than adding it to the allow-list
   yourself.
2. **Run `make licenses`.** It runs `licenses-sync` (the two allow-lists agree)
   then `cargo-deny check licenses bans sources` over both workspaces. It is
   part of `make ci` and blocking in CI.
3. **If you widen the allow-list, widen both files** — `deny.toml` and
   `core-engine/about.toml` — and update §3 here. `licenses-sync` will fail the
   build if you do only one.
4. **Add attribution for anything `cargo-deny` cannot see.** Vendored sources,
   fonts, icons, images: a `LICENSE.md` or `PROVENANCE.md` next to the files
   recording upstream URL, version/commit, the observed licence text, and a
   hash. Add a row to §5. Copy verbatim; never re-generate or "reconstruct" an
   asset and label it as upstream.
