#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Compile for production.
#
#   scripts/build.sh                release-build both workspaces
#   scripts/build.sh --bundle       ...and wrap it in nguo.app (macOS)
#   scripts/build.sh --debug        debug build, for a compile check
#   scripts/build.sh --gated        run the PR gates first, then build
#
# This calls `make build-release` rather than repeating the cargo invocations.
# The Makefile is what CI runs, and a script that shells out to it cannot drift
# from CI; a script that copies its commands silently can, and eventually does.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE=0; PROFILE=release; GATED=0

for arg in "$@"; do
  case "$arg" in
    --bundle) BUNDLE=1 ;;
    --debug)  PROFILE=debug ;;
    --gated)  GATED=1 ;;
    -h|--help) sed -n '3,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "build: unknown option $arg (try --help)" >&2; exit 2 ;;
  esac
done

cd "$ROOT"

if [[ "$GATED" == 1 ]]; then
  echo "== build: running the PR gates first (make ci)"
  make ci
fi

if [[ "$PROFILE" == debug ]]; then
  echo "== build: debug (compile check only — not for measuring)"
  for ws in core-engine ui; do (cd "$ws" && cargo build --workspace --bins); done
else
  echo "== build: release, both workspaces"
  make build-release
fi

# Report what actually exists rather than what was hoped for. `make build-ui`
# already warns when no `nguo` binary is produced; this repeats the check for
# the app, because a bundle step that finds no binary is a confusing failure.
for b in "ui/target/$PROFILE/nguo-app" "ui/target/$PROFILE/nguo" "core-engine/target/$PROFILE/mguso"; do
  if [[ -x "$b" ]]; then
    printf '  built %-40s %s\n' "$b" "$(du -h "$b" | cut -f1 | tr -d ' ')"
  else
    printf '  MISSING %-38s (no crate produced it)\n' "$b"
  fi
done

if [[ "$BUNDLE" == 1 ]]; then
  [[ "$PROFILE" == release ]] || { echo "build: --bundle needs a release build" >&2; exit 1; }
  "$ROOT/scripts/bundle-macos.sh"
fi
