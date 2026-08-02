#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Break the UI reference videos into distinct UI states for study.
#
#   tools/extract-ui-frames.sh [threshold] [interval_s]   # defaults 0.20, 15
#
# Sampling is the UNION of two strategies, because neither alone is enough and
# we measured that:
#   * scene-change detection catches cuts and dialog pop-ups, but a screencast
#     is one continuous take — the 10-minute "3D Workspace" video yielded only
#     FIVE frames at threshold 0.20, missing almost the whole walkthrough;
#   * fixed-interval sampling never misses a stretch, but on a cut-heavy edit it
#     returns fifty near-duplicates of the same panel.
# Union of both = one frame per distinct UI state, plus a guaranteed floor of
# coverage across the timeline.
#
# Output: instructions/ui-references/frames/<video-slug>/frame_XXXX_at_<t>s.jpg
# plus a manifest.tsv (frame, timestamp) so a frame can always be traced back to
# its moment in the source — which is what makes a claim about the UI checkable.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFS="$ROOT/instructions/ui-references"
OUT="$REFS/frames"
THRESHOLD="${1:-0.20}"
INTERVAL_S="${2:-15}"

command -v ffmpeg >/dev/null 2>&1 || { echo "error: ffmpeg not on PATH" >&2; exit 127; }

shopt -s nullglob
videos=("$REFS"/*.mp4)
[ ${#videos[@]} -gt 0 ] || { echo "no .mp4 under $REFS"; exit 0; }

for v in "${videos[@]}"; do
    base="$(basename "$v" .mp4)"
    # Slug: these filenames are long YouTube-downloader strings. Keep the human
    # part, drop the noise.
    slug="$(echo "$base" \
        | sed -E 's/^YTDown\.com_YouTube_//; s/_Media_.*$//; s/[^A-Za-z0-9]+/-/g; s/^-|-$//g' \
        | tr '[:upper:]' '[:lower:]')"
    dir="$OUT/$slug"

    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "==> $slug: already extracted ($(ls "$dir"/*.jpg 2>/dev/null | wc -l | tr -d ' ') frames), skipping"
        continue
    fi
    mkdir -p "$dir"

    dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$v" | cut -d. -f1)"
    echo "==> $slug: ${dur}s, scene>$THRESHOLD OR every ${INTERVAL_S}s"

    # `isnan(prev_selected_t)` seeds the very first frame; after that,
    # `gte(t-prev_selected_t, INTERVAL)` guarantees the interval floor no matter
    # how static the recording is. OR'd with scene detection for the cuts.
    # showinfo writes pts_time for every frame that survives the select filter;
    # that is how we recover real timestamps instead of guessing from the index.
    ffmpeg -hide_banner -loglevel info -i "$v" \
        -vf "select='gt(scene,$THRESHOLD)+isnan(prev_selected_t)+gte(t-prev_selected_t,$INTERVAL_S)',showinfo" \
        -vsync vfr -q:v 3 "$dir/frame_%04d.jpg" 2>"$dir/.showinfo.log"

    # Pair each emitted frame with its pts_time, in order, into a manifest.
    : > "$dir/manifest.tsv"
    printf 'frame\ttimestamp_s\n' >> "$dir/manifest.tsv"
    paste \
        <(ls "$dir"/frame_*.jpg 2>/dev/null | xargs -n1 basename) \
        <(grep -o 'pts_time:[0-9.]*' "$dir/.showinfo.log" | cut -d: -f2) \
        >> "$dir/manifest.tsv" 2>/dev/null || true

    n="$(ls "$dir"/frame_*.jpg 2>/dev/null | wc -l | tr -d ' ')"
    echo "    $n frames -> $dir"
done

echo
echo "total: $(find "$OUT" -name 'frame_*.jpg' 2>/dev/null | wc -l | tr -d ' ') frames under $OUT"
