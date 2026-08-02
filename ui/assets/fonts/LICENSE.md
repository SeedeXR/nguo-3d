<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Vendored font licences

Downloaded from upstream on **2026-08-02** and committed verbatim. Nothing here
was re-generated, subset, renamed or "reconstructed" — mislabelling a
typeface's provenance is a licensing error, not a style choice.

There is **no font crate dependency and no runtime network fetch**: the file is
embedded at compile time by `nguo-splash` via `include_bytes!`, so a missing or
truncated file fails the build.

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
