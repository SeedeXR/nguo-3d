<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Vendored icon licences

Every SVG in this directory was downloaded from its upstream repository on
**2026-08-01** and committed verbatim. No icon here was hand-authored, traced
or "reconstructed" — mislabelling provenance is a licensing error, not a style
choice.

There is **no icon-font crate dependency and no runtime network fetch**: the
files are embedded at compile time by `nguo-ui-widgets` via `include_bytes!`,
so a missing or truncated file fails the build.

## Lucide — ISC License

- Upstream: <https://github.com/lucide-icons/lucide> (`icons/<name>.svg`)
- Licence: ISC (`LICENSE` at the repository root)
- Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part
  of Feather (MIT). All other copyright (c) for Lucide are held by Lucide
  Contributors 2022.
- Compatible with this project's GPL-2.0-or-later (permissive, GPL-compatible).

Files (unprefixed filenames are Lucide):

```
arrow-up-down       box                 check               chevron-down
chevron-left        chevron-right       chevron-up          circle-dot
clipboard           copy                crosshair           download
eye                 eye-off             file-plus           flip-horizontal-2
folder-open         git-commit-vertical grid-3x3            hand
info                layers              lock                lock-open
magnet              maximize-2          minus               moon
mouse-pointer-2     mouse-pointer-click move-3d             palette
panel-bottom        panel-left          panel-right         pause
pen-tool            play                plus                redo-2
refresh-cw          rotate-3d           ruler               save
scissors            search              settings            shirt
sliders-horizontal  spline              square-dashed       sun
trash-2             triangle            triangle-alert      type
undo-2              upload              x                   zoom-in
zoom-out
```

61 files.

## Tabler Icons — MIT License

- Upstream: <https://github.com/tabler/tabler-icons> (`icons/outline/<name>.svg`)
- Licence: MIT, Copyright (c) 2020-2024 Paweł Kuna
- Used **only** where Lucide has no equivalent glyph, per the project icon
  policy. Filenames carry a `tabler-` prefix so provenance is visible on disk.

| File | Why not Lucide |
|---|---|
| `tabler-line.svg` | Lucide has no plain two-point line-segment glyph (the Line tool, guide §6.1). |
| `tabler-needle-thread.svg` | Lucide has no needle/sewing glyph (the Seam tool, guide §10). |

2 files.

## Total

63 SVG files: 61 Lucide (ISC), 2 Tabler (MIT).
