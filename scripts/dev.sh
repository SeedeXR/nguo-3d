#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Run the app in debug, with the logging and backtraces you want while working.
#
#   scripts/dev.sh              run the GUI (nguo-app)
#   scripts/dev.sh cli ...      run the CLI (nguo) with arguments
#   scripts/dev.sh --overlay    GUI with the egui debug overlay
#
# Debug, deliberately: this is the build you attach a debugger to and the one
# whose panics carry line numbers. It is SLOW for cloth — a drape that takes
# seconds in release takes minutes here. Use scripts/build.sh (release is its
# default) for anything you measure, and never quote a timing from this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# `warn` globally, `debug` for our own crates: a cloth step logs per-iteration
# at debug, and third-party crates at debug drown the signal entirely.
export RUST_LOG="${RUST_LOG:-warn,nguo_app=debug,nguo_canvas=debug,nguo_doc=debug,engine_api=debug,engine_core=info}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

# --help reads this file's own header, so it must run BEFORE the `cd` below:
# ${BASH_SOURCE[0]} is whatever path the caller used, and `./scripts/dev.sh`
# stops resolving the moment the working directory changes.
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

cd "$ROOT/ui"

# `gui` falls through to the exec at the bottom, which is also where an
# unrecognised first argument lands — so `dev.sh --fullscreen` forwards to the
# app instead of being rejected.
case "${1:-gui}" in
  cli)
    shift
    exec cargo run -p nguo-cli --bin nguo -- "$@"
    ;;
  --overlay)
    shift
    exec cargo run -p nguo-app --features nguo-app/debug-overlay -- "$@"
    ;;
  gui)
    # `|| true` is load-bearing: `${1:-gui}` also matches when there is no $1
    # at all, and `shift` on an empty argument list returns 1, which under
    # `set -e` aborts the plain `dev.sh` invocation before it ever runs cargo.
    shift || true
    ;;
esac

exec cargo run -p nguo-app -- "$@"
