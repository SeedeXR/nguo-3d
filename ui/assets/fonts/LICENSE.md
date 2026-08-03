<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Vendored font licences

Downloaded from upstream on **2026-08-02** and committed verbatim. Nothing here
was re-generated, subset, renamed or "reconstructed" — mislabelling a
typeface's provenance is a licensing error, not a style choice.

There is **no font crate dependency and no runtime network fetch**: both files
are embedded at compile time via `include_bytes!` — Inter by `nguo-splash`,
Space Mono by `nguo-gpu`'s debug overlay — so a missing or truncated file fails
the build. `.gitattributes` §4 pins both out of Git LFS, because an LFS pointer
would compile as a ~130-byte text file and embed cleanly as a broken font.

## Inter — SIL Open Font License 1.1

- Upstream: <https://github.com/rsms/inter>, release **v4.1**
  (`https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip`)
- File taken from the archive: `extras/ttf/Inter-Regular.ttf`
- Licence: SIL OFL 1.1 — full text in `OFL.txt`, copied from `LICENSE.txt` at
  the archive root. Copyright (c) 2016 The Inter Project Authors.
- Compatible with this project's GPL-2.0-or-later. The OFL is a
  GPL-compatible free-software licence for fonts; embedding an OFL font in a
  GPL program is fine, and the OFL's only real conditions (keep the licence,
  do not sell the font on its own, rename if you modify it) are met by
  shipping the file byte-identical with `OFL.txt` alongside.

| File | SHA-256 |
|---|---|
| `Inter-Regular.ttf` | `40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82` |

Only the Regular weight is vendored. Nothing in the UI needs Inter Bold yet;
add `extras/ttf/Inter-SemiBold.ttf` from the same release (and a row here) when
something actually does.

## Space Mono — SIL Open Font License 1.1

- Upstream: <https://github.com/googlefonts/spacemono>, `main` at commit
  `329858c2c4dbd3476f972a4ae00624b018cf4b81` (the repository has no versioned
  release; the archive fetched was
  `https://github.com/googlefonts/spacemono/archive/refs/heads/main.zip`)
- File taken from the archive: `fonts/ttf/SpaceMono-Regular.ttf`
- Licence: SIL OFL 1.1 — full text in `SpaceMono-OFL.txt`, copied verbatim from
  `OFL.txt` at the archive root. Copyright 2016 The Space Mono Project Authors.
  It is a *separate file* from Inter's `OFL.txt`: the licence body is the
  standard OFL 1.1, but the copyright line differs and the two upstream files
  are not byte-identical, so neither may stand in for the other.
- Same GPL-compatibility story as Inter above.

| File | SHA-256 |
|---|---|
| `SpaceMono-Regular.ttf` | `95837e182baeeada83368f7748db28357f0a1b75c6b84ff7065b5edf933c8e18` |
| `SpaceMono-OFL.txt` | `8e4ee42b2553e1e01504e61cb0d46d148cd8c9e5eacaa3622a7df2d4f2955b9f` |

Why this font is here: the egui debug overlay is a wall of live numbers, and in
a proportional face they jitter horizontally as digits change width. Space Mono
is monospaced (`post.isFixedPitch = 1`, verified on the vendored file) so the
columns hold still. Supplying it also let `egui`'s `default_fonts` feature — and
with it the last dependency licence that was not cleanly GPL-compatible — be
dropped outright; see `LICENSING.md` §3.

Only the Regular weight is vendored. egui renders every overlay text style from
one face, so Bold buys nothing today; take `fonts/ttf/SpaceMono-Bold.ttf` from
the same commit (and add a row here) if that changes.

Glyph coverage was checked against what the overlay actually draws, not assumed:
the vendored file's `cmap` covers 624 codepoints including `—` (U+2014), `·`
(U+00B7), `°`, `√`, `×` and the ASCII the overlay formats numbers with.
