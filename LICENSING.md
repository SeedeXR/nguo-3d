<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Licensing policy

nguo-3d is **GPL-2.0-or-later**. Every dependency it ships must permit
**commercial use**. This document states the policy; `deny.toml` enforces it.

---

## 1. The project licence

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
```

One scoped exception, in `deny.toml` and `core-engine/about.toml` both:

| Licence | Scope | Why scoped rather than allowed |
|---|---|---|
| `Ubuntu-font-1.0` | crate `epaint_default_fonts` only | Permits commercial use and redistribution, so it passes this policy. But unlike everything above, its GPL compatibility is not clean — the rename-on-modify clause is why the FSF does not list it as GPL-compatible. It reaches a binary only under `--features debug-overlay`. If that changes, drop egui's `default_fonts` feature and supply Inter instead. |

### The two allow-lists, and the check that keeps them honest

- `deny.toml` — enforcement (`cargo-deny`). Blocking, runs in CI.
- `core-engine/about.toml` — attribution (`cargo-about` → `LICENSES/THIRD_PARTY.md`).

They must hold the same set. They had **silently drifted**, in both directions,
because a comment claimed they were identical and nothing verified it:
`about.toml` accepted bare `GPL-2.0` (the exact id `deny.toml` forbids) and
omitted `LGPL-2.1-or-later` (the licence planegcs is vendored under).

`make licenses-sync` now diffs the two arrays. It runs as part of
`make licenses`, needs no tools, no build and no network.

**Honest limit: `cargo-about` is not wired into anything.** `make licenses` runs
`cargo-deny` only; `cargo-about` is not installed on the dev machine, there is
no CI job for it, and `core-engine/LICENSES/THIRD_PARTY.md` **does not exist
yet**. The sync check guarantees the two *lists* agree. It does not generate or
freshen attribution — that is still manual, and is a release blocker rather
than a solved problem.

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
| **epaint_default_fonts** — egui's bundled fonts, `--features debug-overlay` only | `(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0` | All three permit commercial use. `Ubuntu-font-1.0` is a scoped exception — see §3 | `deny.toml` `[[licenses.exceptions]]`, `core-engine/about.toml` |
| **self_cell** (transitive: `iced` → `cosmic-text`) | `Apache-2.0 OR GPL-2.0-only` | Dual-licensed; we take the **Apache-2.0** branch. The GPL-2.0-only branch is not exercised and does not pin our tree | this table |
| **ClothDrop-Reborn** — design/behaviour reference. Read in a local working tree under `instructions/`, which is gitignored: it is **not** shipped and not in a clone | GPL-3.0-or-later | Read as a reference only. Pasting its code into the GPL-2.0-or-later Rust tree would force the whole work to GPLv3 — permitted by our `-or-later` grant, but a decision, not an accident. Flag it in review if it ever happens. A Blender add-on we ship may be GPL-3.0-or-later regardless | upstream `LICENSE` (verified locally: GNU GPL v3, 29 June 2007) |
| **Brand assets** — `ui/assets/brand-icon/`, `ui/assets/images/splash_screen_image.png` | **unverified** — no attribution file present | Believed original work by Seede XR, but nothing in the tree records it. Needs a `LICENSE.md` alongside the other asset directories | *nowhere yet* |

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
