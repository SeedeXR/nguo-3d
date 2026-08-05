<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# `assets/examples/` — shipped example projects

One file today: **`dress-female-sheath.nguo`**, the end-to-end industrial
example. A `.nguo` is a zip (D-1) — `unzip -l` it and read `manifest.json`; that
is the point of the format.

```
$ unzip -l dress-female-sheath.nguo
     1321  manifest.json     what this file is, without reading the rest
    13764  document.json     the whole document, pretty-printed
     1817  preview.svg       a 2D outline of the pattern, in millimetres
```

**3 506 bytes on disk** (16 902 uncompressed), and it loads in **0.13–0.17 ms**
(five runs, release build, Apple M4) — `nguo_doc::load` straight to a live
`Document`. `nguo dress-example` reloads every file it writes and reports
`save.round_trip_identical`, so the round trip is measured rather than assumed.
The document has no drape in it, on purpose: a drape is derived state and saving
it as truth is what the guide forbids.

## What it is

A **sleeveless sheath dress**: front panel, back panel, two shaped side seams,
top edge at the bust line, hem just below the knee. Drafted **to measure** for
the `female` avatar of `assets/avatars/`, in **millimetres**.

Regenerate it, or run the whole chain on it, with the shipped CLI:

```sh
cd ui && cargo build --release -p nguo-cli
# pattern -> mesh -> sew -> save. No body, no solver: runs anywhere.
./target/release/nguo dress-example --out /tmp/dress.nguo
# the whole thing: + hang on the avatar, resolve overlap, drape.
./target/release/nguo dress-example --avatar female --edge-mm 60 --steps 36 \
    --out ../assets/examples/dress-female-sheath.nguo
```

The pattern itself is code — `ui/crates/nguo-cli/src/dress.rs` — because every
dimension is derived from a cited body measurement and a stated ease, and a
number in a data file cannot show its derivation. Read that file for the full
argument; this page is the summary.

## The draft, and where each number comes from

**Body girths: `assets/avatars/data/manifest.json`, `avatars.female`.** Read from
the file, never measured off the collision proxy: a horizontal slab through a
T-posed body catches both arms. The pipeline reports both numbers side by side
so the difference stays visible — over this dress's band the manifest gives a
body radius of **0.1776 m** and a slab of the same band gives **0.9293 m at
y = 1.4531 m**, which is a hand.

| line | body girth (manifest) | ease | finished girth | half = one panel |
|------|----------------------|------|----------------|------------------|
| bust  | 1116.03 mm @ 1471.87 mm (`chest`) | 40 mm | 1156.03 mm | **578.015 mm** |
| waist | 745.00 mm @ 1208.66 mm (`waist`)   | 20 mm | 765.00 mm  | **382.5 mm**   |
| hip   | 1087.27 mm @ 932.18 mm (`hip`)     | 40 mm | 1127.27 mm | **563.635 mm** |
| hem   | — (at `calf`, 571.43 mm)            | —     | as the hip | **563.635 mm** |

Vertical: bust line to waist **263.21 mm**, to hip **539.69 mm**, to hem
**900.44 mm** — all four straight from the manifest heights.

* **Seam allowance: 15 mm (5/8″)** on both side seams, the industrial standard.
  **Hem turn-up 40 mm. Top edge faced 25 mm.** None of the three is in the
  outline: the mesh is the **net pattern**, i.e. the stitch line, because that is
  the garment. Cutting instructions add them back.
* **Grain: the straight grain**, down the centre of each panel parallel to the
  side seams, carried as a real `Feature::GrainLine` rather than a comment.
* **Balance notches** at the waist on both side seams (`Feature::Notch`).
* **Ease and allowances are convention, stated as convention.** Nobody here has
  an edition of Aldrich or Armstrong on the desk to check a page against, and a
  fabricated citation is worse than none. What is claimed: 40/20/40 mm is what a
  close-fitting woven block is drafted with, and 15 mm is the standard seam
  allowance. Nothing here is stretch fabric.

## What size is it? None — and that is a finding

Against the UK ready-to-wear chart (UK 12 = bust 880 / waist 700 / hip 960,
+40 mm per size):

| girth | avatar | nearest UK size |
|-------|--------|-----------------|
| waist | 745.0 mm | **UK 14** |
| hip | 1087.3 mm | **UK 18** |
| bust | 1116.0 mm | **off the chart** (UK 22 is 1080) |

No single nominal size fits this body, so the dress is drafted to measure. The
bust is the reason: a bust-to-waist drop of **371 mm** against the 180–200 mm
every published chart uses is not a human shape, and `upper_chest` (1114.7 mm)
sitting 1.4 mm from `chest` (1116.0 mm) across 16 mm of height is the tell — the
manifest's chest girth almost certainly includes the T-pose arms. It is drafted
to anyway, because a **loose** bust is the safe direction (cloth must never start
inside the body) and because inventing a corrected girth would be a fabrication.

The consequence is measured rather than hidden: the dress is loose at the bust,
so **what holds it up is the waist, not the bust** — a 765 mm waist section
cannot pass a 1087 mm hip. Nothing is pinned. Measured over 36 solver steps: the
hem starts at 0.5714 m and finishes at 0.5719 m.

## Why two panels and not a bodice plus a skirt

`nguo_sewing::arrange`'s `Archetype::Tube` lays **every** piece on **one**
cylinder with **one** `top_y_m`, and sums every piece's support arc into that one
circumference. A bodice/skirt pair therefore arranges as four panels round a
circle of (bust + hip)/2π with all four starting at the bust line, overlapping —
the one initial condition a barrier method cannot recover from. Shoulder seams
fail for the same reason: on a wrapped cylinder the front and back panels meet
only along the two side seams, so a shoulder seam would fold the cloth at the
*side*. The crate's own `hang()` already says this: *"A two-tier garment needs a
placement per piece, and that is a different arrangement, not a different
string."*

So this example is the dress whose seams a tube can actually close. A
`Placement3d::Cone` variant plus a per-piece `top_y_m` unlocks the waist-seam
dress, the A-line and the peplum in one change — when there is one to measure.

## Measured, on an Apple M4, release build, one process at a time

`--tier preview`, 36 steps, female COL proxy cropped to the garment's box + 150 mm
(4 921 of 10 531 body verts — which also drops the T-pose arms). Peak RSS from
`/usr/bin/time -l`, quoted beside every timing because a number without one is
not a measurement.

| edge mm | cloth v/t | scene v | s/step | wall s | seam gap mm | closed | inside after | peak RSS |
|---------|-----------|---------|--------|--------|-------------|--------|--------------|----------|
| 120 |   446 /   776 |  5 367 | 0.131 |  4.7 | 5 506 → 161 | 97.1 % | 0 | **67.2 MiB** |
|  90 |   572 / 1 016 |  5 493 | 0.159 |  6.3 | 5 471 → 147 | 97.3 % | 0 | **64.2 MiB** |
|  60 |   769 / 1 385 |  5 692 | 0.165 |  6.0 | 6 098 → 142 | 97.7 % | 0 | **73.2 MiB** |
|  45 | 1 513 / 2 847 |  6 434 | 0.449 | 16.6 | 6 726 → 150 | 97.8 % | 0 | **105.0 MiB** |
|  30 | 2 695 / 5 155 |  7 618 | 0.660 | 23.5 | 7 388 → 165 | 97.8 % | 0 | **118.4 MiB** |

`--tier bake` (8 Newton iterations instead of 2), 30 mm, 48 steps: **1.479 s/step,
79.4 s, 115.4 MiB**, seams 5 357 → 173 mm, and flat from step 12.

Memory scaling, measured upward rather than projected (12 steps each, and RSS is
step-count independent — 30 mm reads 118.3 MiB at 12 steps and 118.4 at 36):

| cloth verts | scene verts | peak RSS |
|-------------|-------------|----------|
| 2 695 |  7 618 | 118.3 MiB |
| 6 858 | 11 781 | 318.9 MiB |
| 8 041 | 12 964 | 300.2 MiB |

2.55× the cloth vertices costs 2.70× the RSS — an exponent of 1.06, i.e. very
nearly linear. Extrapolated (and labelled as an extrapolation, not a
measurement), the 4 GB ceiling is ~**76 000 cloth vertices** away. Two things
that table shows and a projection would have missed: RSS is **not monotone** in
the requested edge, and 15 mm buys almost nothing over 20 mm (8 041 vertices
against 6 858) because the triangulator's refinement saturates — the 300 vs
319 MiB difference is run noise, not a saving.

**60 mm is the shipped resolution** (73 MiB, 6 s), 30 mm is the finest anything
here needs (118 MiB): **56× and 35× under the 4 GB cap**. So the resolution was
chosen from the seam-closure curve and the wall clock, and memory never
constrained it — which is only worth saying because it was measured rather than
assumed.

## Licensing

This file and the pattern are **GPL-2.0-or-later**, like the rest of the code.
The avatar is **referenced, never embedded** — `manifest.json` carries the key
(`female`), the tier (`COL`) and the bundle version (`1.0.0`), and that is all.
`assets/avatars/**` is 227 MB of Git-LFS under **CC BY 4.0** (licensor Seede XR),
a different licence, so a project file that carried a copy of a body would be
redistributing it. Attribution is required; see `LICENSING.md`.
