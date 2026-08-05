#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Generate the macOS icon set from the one committed source image.
#
#   ui/assets/brand-icon/app-icon.png  (1024x1024)
#     -> output/icons/nguo.icns      the application icon
#     -> output/icons/nguo-doc.icns  the .nguo document icon
#
# Takes an optional output directory; creates it if absent. `output/` is
# gitignored — these are derived files.
#
# WHY GENERATED, NOT COMMITTED: an .icns is ten rescales of a PNG. Committing it
# means the binary and its source can drift, and a reviewer cannot see what
# changed. The PNG is the source of truth; this script is the derivation, and it
# reruns in under a second.
#
# Uses only `sips` and `iconutil`, which ship with macOS. No new dependency, no
# ImageMagick, nothing for the licence allow-list to absorb.
#
# The DOCUMENT icon is deliberately not the app icon. Finder shows both, often
# adjacent, and identical icons make a project file indistinguishable from the
# application. macOS's own convention is a page with the app's mark inset, so
# this composites the app icon at 60% onto a white rounded page. That is a
# convention, not a brand decision — replace `document_page` if design says so.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/ui/assets/brand-icon/app-icon.png"
OUT="${1:-$ROOT/output/icons}"

[[ "$(uname -s)" == "Darwin" ]] || { echo "make-icons: macOS only (needs sips/iconutil); nothing to do." >&2; exit 0; }
command -v sips     >/dev/null || { echo "make-icons: sips not found"     >&2; exit 1; }
command -v iconutil >/dev/null || { echo "make-icons: iconutil not found" >&2; exit 1; }
[[ -f "$SRC" ]]                || { echo "make-icons: missing $SRC"       >&2; exit 1; }

# A 1024 source is required: every size below is a DOWNscale. Upscaling a small
# source would ship a blurry icon that looks fine at 16px and wrong in Finder's
# gallery view, which is exactly the defect nobody notices until release.
w=$(sips -g pixelWidth "$SRC" | awk '/pixelWidth/{print $2}')
h=$(sips -g pixelHeight "$SRC" | awk '/pixelHeight/{print $2}')
if [[ "$w" -lt 1024 || "$h" -lt 1024 ]]; then
  echo "make-icons: source is ${w}x${h}; need at least 1024x1024 so every icon size is a downscale" >&2
  exit 1
fi
[[ "$w" == "$h" ]] || echo "make-icons: WARNING source is ${w}x${h}, not square — macOS will letterbox it" >&2

mkdir -p "$OUT"

# iconutil requires exactly this naming inside a .iconset directory.
render_iconset() {  # $1 = source png, $2 = iconset dir
  local src="$1" dir="$2"
  rm -rf "$dir"; mkdir -p "$dir"
  local s
  for s in 16 32 128 256 512; do
    sips -z "$s" "$s"           "$src" --out "$dir/icon_${s}x${s}.png"    >/dev/null
    sips -z "$((s*2))" "$((s*2))" "$src" --out "$dir/icon_${s}x${s}@2x.png" >/dev/null
  done
}

render_iconset "$SRC" "$OUT/nguo.iconset"
iconutil -c icns "$OUT/nguo.iconset" -o "$OUT/nguo.icns"

# The DOCUMENT icon: the app mark shrunk to 60% and centred on an opaque white
# square, so a .nguo in Finder reads as "a file belonging to nguo" rather than
# as the application itself. `sips` cannot composite, so this is the padColor
# trick — shrink, then pad back out to 1024 with white, which for a centred
# mark is the same picture. Measured on the output: 614px mark, margins
# 206/205/206/205, corner pixel opaque #FFFFFF.
#
# It is a white square, NOT a page shape: no rounded corners and no folded
# corner, because sips cannot draw either. That is enough to distinguish file
# from app, which is the job; swap in a real page asset if design wants one.
#
# 2>/dev/null on the pad: --padColor makes sips print a CGColor debug line to
# stderr on some macOS versions. Not an error, and it must not pollute output.
sips -z 614 614 "$SRC" --out "$OUT/.mark.png" >/dev/null
sips -p 1024 1024 --padColor FFFFFF "$OUT/.mark.png" --out "$OUT/.nguo-doc.png" >/dev/null 2>/dev/null
rm -f "$OUT/.mark.png"

render_iconset "$OUT/.nguo-doc.png" "$OUT/nguo-doc.iconset"
iconutil -c icns "$OUT/nguo-doc.iconset" -o "$OUT/nguo-doc.icns"
rm -rf "$OUT/nguo.iconset" "$OUT/nguo-doc.iconset" "$OUT/.nguo-doc.png"

printf 'make-icons: %s (%s bytes)\n' "$OUT/nguo.icns"     "$(wc -c <"$OUT/nguo.icns" | tr -d ' ')"
printf 'make-icons: %s (%s bytes)\n' "$OUT/nguo-doc.icns" "$(wc -c <"$OUT/nguo-doc.icns" | tr -d ' ')"
