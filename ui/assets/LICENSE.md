<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Asset licences

Every non-code asset shipped in `ui/assets/`, and where its terms come from.
Subdirectories with their own `LICENSE.md` are authoritative for their contents;
this file covers the rest and indexes the whole tree.

| Asset | Licence | Notice |
|---|---|---|
| `icons/*.svg` (61 files) | **ISC** — Lucide | [`icons/LICENSE.md`](icons/LICENSE.md) |
| `icons/tabler-*.svg` (2 files) | **MIT** — Tabler | [`icons/LICENSE.md`](icons/LICENSE.md) |
| `fonts/Inter-Regular.ttf` | **SIL OFL 1.1** | [`fonts/LICENSE.md`](fonts/LICENSE.md), [`fonts/OFL.txt`](fonts/OFL.txt) |
| `brand-icon/butterfly-nguo3d-icon.svg` | **Seede XR brand asset** — see below | this file |
| `images/splash_screen_image.png` | **Seede XR brand asset** — see below | this file |

All licences above permit commercial use, as required by
[`../../LICENSING.md`](../../LICENSING.md).

---

## Brand assets — read this before reusing them

`brand-icon/butterfly-nguo3d-icon.svg` and `images/splash_screen_image.png` are
**Seede XR brand assets**, not part of the GPL-licensed source. The application
credits the artwork on its own splash screen as *"Artwork by Alex Mkwizu &
MidJourney AI"*, © 2026 Seede XR.

That distinction is normal and deliberate: a project's code being GPL does not
place its logo and brand artwork under the same terms. Trademarks and brand
identity are conventionally reserved even in fully free software — Mozilla and
Debian both do this — because the licence governs *the software*, while the mark
identifies *who made it*.

**What this means in practice**

- You may build, run, modify and redistribute nguo-3d under GPL-2.0-or-later.
- You may **not** use the butterfly mark or the splash artwork to brand a fork
  or a derived product as though it were official. Replace them.
- Both files are embedded at build time via `include_bytes!`
  (`ui/crates/nguo-splash/src/lib.rs`), so a fork that removes them must also
  update those references — the build will fail loudly rather than silently, by
  design.

**Authorship**

The splash artwork was **created by Alex Mkwizu**, who used MidJourney as a tool
in producing it, and is held by **Seede XR**. MidJourney is credited on the
splash screen as the tool used, in the same spirit that a photograph credits
neither its camera nor its editing software as a co-author. The butterfly mark
is likewise original Seede XR work.

**One item still open for the maintainer**

The reuse terms are currently "all rights reserved", which is what the splash
screen states. Whether Seede XR wishes to grant any narrower permission — for
example, allowing the mark to appear in unmodified redistributions of nguo-3d,
which most trademark-reserving free-software projects do allow — has not been
written down. Until it is, downstream users have no stated permission at all,
and a brand asset with no stated terms is the one thing in the tree a
distributor cannot safely act on.

This does not block development or affect the GPL status of the code.
