#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# One table of every performance number this repo already measures.
#
#   tools/profile.sh                 measure, print the table
#   tools/profile.sh --check         also fail if a metric regressed past its tolerance
#   tools/profile.sh --only quick    one tier: quick | gpu | avatar (comma-separated)
#   tools/profile.sh --require       a SKIP is a FAILURE (CI, or "prove it ran")
#   tools/profile.sh --selftest      check the output parsers against a fixture, run nothing
#
# `--selftest` guards the parsers against a bad edit to THEM. A benchmark that
# changes its output format instead produces an empty parse, which shows up as a
# SKIP row in the table — visible, and fatal under `--require`.
#
# It runs NO measurement of its own. Every number comes from a benchmark that
# already lives in a crate and is already `#[ignore]`d there:
#
#   quick   mguso bench                              -> ms/step
#           nguo-registry  tests/overhead.rs         -> ns/op
#   gpu     nguo-gpu       tests/handoff_bench.rs    -> ms round trip / ms per frame
#   avatar  engine-api     tests/avatar_bundle.rs    -> s/step  (needs the 227 MB
#                          collision_proxy_step_cost     LFS bundle; 1.7 s measured
#                                                        2026-08-03, minutes before
#                                                        the broadphase movable mask)
#
# Baselines, hardware and the reasoning behind the tolerances: perf-baseline.tsv.
# Frame time is deliberately absent — see the bottom of that file for why (Fifo
# vsync makes 8.33 ms a hard floor, so a headless number would prove nothing).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${BASELINE:-$ROOT/tools/perf-baseline.tsv}"
MGUSO="${MGUSO:-$ROOT/core-engine/target/release/mguso}"
SCENE="${SCENE:-$ROOT/core-engine/examples/fabric_drape.json}"
AVATAR_BUNDLE="${NGUO_AVATAR_BUNDLE:-$ROOT/assets/avatars}"

## --------------------------------------------------------------------------
## Parsers. Functions, not inline awk, so `--selftest` can prove they still
## match. The first version of the blocking parser matched NOTHING and the
## table simply said SKIP — a silent no-coverage pass, which is the exact
## failure mode this repo keeps producing. The fixture makes that loud.
## --------------------------------------------------------------------------

# nguo-registry: "start + done, recording       131.7 ns/op"
parse_registry() { awk '/^start \+ done, recording/{print $(NF-1); exit}' "$1"; }

# handoff_bench blocking table: round-trip MEDIAN at 100k verts, 28 B stride.
# One rule per `###` heading so a new section can never leave `sect` set. The
# earlier `!/28 B/` reset was useless because EVERY heading in that file says
# "28 B/vertex" — and `.* 28 B` never matched "(28 B/vertex)" at all.
parse_blocking() {
  awk -F'|' '
    /^###/               {sect = ($0 ~ /Blocking path/ && $0 ~ /28 B\/vertex/); next}
    sect && $2+0==100000 {split($6,a," "); print a[1]; exit}' "$1"
}

# handoff_bench pipelined table: ms per delivered frame at 100k verts.
parse_pipelined() {
  awk -F'|' '
    /^###/               {sect = ($0 ~ /Pipelined path/); next}
    sect && $2+0==100000 {print $4+0; exit}' "$1"
}

# collision_proxy_step_cost: "male full proxy, self-collision off: 10236 verts
# / 15436 tris, 3 steps, 0.066 s/step". Both avatars print one; take the WORSE.
parse_proxy() {
  awk -v m="$1" '$0 ~ "full proxy, self-collision "m":" {print $(NF-1)}' "$2" |
    sort -g | tail -1
}

selftest() {
  local fx="$1" fail=0 want got name
  cat >"$fx" <<'FIXTURE'
start + done, recording                          131.7 ns/op
start + progress + done, recording                82.6 ns/op
### Blocking path — positions only (12 B/vertex)
| 100000 | 1.20 MB | 1.293 / 1.333 | 1.402 / 2.627 | 2.691 / 3.910 | 892 | within 8.3 ms target |
### Blocking path — positions + normals + strain (28 B/vertex)
| 50000 | 1.40 MB | 1.294 / 1.328 | 1.403 / 1.468 | 2.695 / 2.782 | 1039 | within 8.3 ms target |
| 100000 | 2.80 MB | 1.327 / 1.398 | 1.555 / 1.675 | 2.885 / 3.054 | 1941 | within 8.3 ms target |
### Pipelined path — `Handoff`, 3 slots, 28 B/vertex
| 100000 | 2.80 MB | 0.325 | 3073 | 0.296 / 0.372 | within 8.3 ms target |
male full proxy, self-collision off: 10236 verts / 15436 tris, 3 steps, 0.066 s/step
female full proxy, self-collision off: 10236 verts / 15436 tris, 3 steps, 0.061 s/step
male full proxy, self-collision on: 10236 verts / 15436 tris, 1 steps, 0.379 s/step
FIXTURE
  # Every expected value is a value that actually appeared in a real run, and
  # each one is a DIFFERENT cell of the table, so a parser that grabs the wrong
  # column cannot accidentally still pass.
  for pair in \
    "parse_registry 131.7" \
    "parse_blocking 2.885" \
    "parse_pipelined 0.325" \
    "parse_proxy_off 0.066" \
    "parse_proxy_on 0.379"; do
    name="${pair%% *}"
    want="${pair##* }"
    case "$name" in
    parse_proxy_off) got="$(parse_proxy off "$fx")" ;;
    parse_proxy_on) got="$(parse_proxy on "$fx")" ;;
    *) got="$("$name" "$fx")" ;;
    esac
    if [ "$got" = "$want" ]; then
      echo "ok   $name -> $got"
    else
      echo "FAIL $name -> '$got' (expected '$want')"
      fail=1
    fi
  done
  # The other direction: input that does not look like this must yield NOTHING,
  # so the caller reports SKIP instead of a fabricated number.
  printf 'no GPU adapter; skipping benchmark\n' >"$fx.junk"
  for name in parse_registry parse_blocking parse_pipelined; do
    if [ -z "$("$name" "$fx.junk")" ]; then
      echo "ok   $name -> empty on non-benchmark input"
    else
      echo "FAIL $name invented a value from non-benchmark input"
      fail=1
    fi
  done
  if [ "$fail" != 0 ]; then
    echo "parser selftest FAILED — benchmark output format changed; fix the parsers." >&2
    return 1
  fi
  echo "parser selftest: 8/8 — all parsers match a known-good fixture"
}

CHECK=0
SELFTEST=0
REQUIRE="${NGUO_PROFILE_REQUIRE:-0}"
TIERS="quick,gpu,avatar"
while [ $# -gt 0 ]; do
  case "$1" in
  --check) CHECK=1 ;;
  --require) REQUIRE=1 ;;
  --selftest) SELFTEST=1 ;;
  --only)
    TIERS="$2"
    shift
    ;;
  --baseline)
    BASELINE="$2"
    shift
    ;;
  -h | --help)
    sed -n '3,23p' "${BASH_SOURCE[0]}"
    exit 0
    ;;
  *)
    echo "unknown argument: $1 (try --help)" >&2
    exit 2
    ;;
  esac
  shift
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ "$SELFTEST" = 1 ]; then
  selftest "$WORK/fixture.txt"
  exit $?
fi

RESULTS="$WORK/results.tsv"
: >"$RESULTS"
SKIPS="$WORK/skips.txt"
: >"$SKIPS"

want() { [[ ",$TIERS," == *",$1,"* ]]; }
emit() { printf '%s\t%s\n' "$1" "$2" >>"$RESULTS"; }
# Skips go to fd 2 AND to a file: a skip notice that goes nowhere is how this
# session already shipped 15 tests reporting `ok` while covering nothing. Here
# it also lands in the table as SKIP and, under --require, fails the run.
skip() {
  echo "SKIP: $*" >&2
  echo "$*" >>"$SKIPS"
}
median() { sort -g | awk '{a[NR]=$1} END{if(!NR)exit; print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
# `emit` on a value, `skip` on an empty one. A function, not `A && B || C`,
# which also runs C when B fails (shellcheck SC2015).
record() {
  if [ -n "$2" ]; then emit "$1" "$2"; else skip "$3"; fi
}

# Check the parsers BEFORE spending minutes of CPU producing output they might
# not be able to read.
selftest "$WORK/fixture.txt" >/dev/null

## --------------------------------------------------------------------------
## quick — seconds. Needs the release `mguso` and a cargo build of one crate.
## --------------------------------------------------------------------------
if want quick; then
  if [ ! -x "$MGUSO" ]; then
    skip "no mguso at $MGUSO — run 'make build-engine' (or pass MGUSO=...)"
  elif ! command -v jq >/dev/null; then
    skip "jq is not installed, so 'mguso bench' JSON cannot be read (brew install jq)"
  else
    echo "== solver: mguso bench $(basename "$SCENE") x3 (median)" >&2
    # 3 runs, median. Four single runs of this spanned 20.95..28.48 ms on a
    # machine that was also compiling; one run is not a measurement.
    for _ in 1 2 3; do
      "$MGUSO" bench "$SCENE" --steps 200 2>/dev/null | jq -r '.ms_per_step'
    done >"$WORK/solver.txt"
    record solver_ms_per_step "$(median <"$WORK/solver.txt")" \
      "mguso bench produced no ms_per_step"
  fi

  echo "== registry: nguo-registry tests/overhead.rs" >&2
  (cd "$ROOT/ui" && cargo test --release -q -p nguo-registry --test overhead -- \
    --ignored --nocapture) >"$WORK/registry.txt" 2>&1 || true
  v="$(parse_registry "$WORK/registry.txt")"
  if [ -n "$v" ]; then
    emit registry_ns_per_op "$v"
  else
    skip "nguo-registry overhead printed no ns/op — build failure? last lines on stderr"
    tail -5 "$WORK/registry.txt" >&2
  fi
fi

## --------------------------------------------------------------------------
## gpu — needs a real adapter. The bench skips itself without one; so do we.
## --------------------------------------------------------------------------
if want gpu; then
  echo "== handoff: nguo-gpu tests/handoff_bench.rs (release)" >&2
  (cd "$ROOT/ui" && cargo test --release -q -p nguo-gpu --test handoff_bench -- \
    --ignored --nocapture) >"$WORK/handoff.txt" 2>&1 || true
  if grep -q 'no GPU adapter' "$WORK/handoff.txt"; then
    skip "handoff_bench found no GPU adapter — both handoff metrics unmeasured"
  else
    # 28 B/vertex is the shipped stride (DrapeVertex); 100k verts sits mid-sweep
    # and is the garment size the drape path actually targets.
    record handoff_blocking_ms_100k "$(parse_blocking "$WORK/handoff.txt")" \
      "no blocking-path row at 100k verts (the bench failed, or its table changed)"
    record handoff_pipelined_ms_100k "$(parse_pipelined "$WORK/handoff.txt")" \
      "no pipelined-path row at 100k verts (the bench failed, or its table changed)"
  fi
fi

## --------------------------------------------------------------------------
## avatar — needs the 227 MB LFS bundle. Opt-in by the bundle being present.
## --------------------------------------------------------------------------
if want avatar; then
  if [ -f "$AVATAR_BUNDLE/data/manifest.json" ]; then
    echo "== proxy: engine-api collision_proxy_step_cost (release)" >&2
    (cd "$ROOT/core-engine" && NGUO_AVATAR_BUNDLE="$AVATAR_BUNDLE" \
      MGUSO_REQUIRE_AVATAR_BUNDLE=1 cargo test --release -q -p engine-api \
      --test avatar_bundle collision_proxy_step_cost -- --ignored --nocapture) \
      >"$WORK/proxy.txt" 2>&1 || true
    for m in off on; do
      record "proxy_step_s_selfcol_$m" "$(parse_proxy "$m" "$WORK/proxy.txt")" \
        "no 'full proxy, self-collision $m' line from collision_proxy_step_cost"
    done
  else
    skip "no avatar bundle at $AVATAR_BUNDLE — 'git lfs pull' or set NGUO_AVATAR_BUNDLE"
  fi
fi

## --------------------------------------------------------------------------
## The table. Markdown on purpose: readable in a terminal AND valid for
## $GITHUB_STEP_SUMMARY, so CI and a human read the identical thing.
## --------------------------------------------------------------------------
awk -F'\t' -v check="$CHECK" -v require="$REQUIRE" -v tiers=",$TIERS," '
  FILENAME==ARGV[1] && /^[^#]/ && NF>=4 {
    order[++n]=$1; unit[$1]=$2; base[$1]=$3; tol[$1]=$4; tier[$1]=$5
  }
  FILENAME==ARGV[2] { got[$1]=$2 }
  FILENAME==ARGV[3] { skipped[++s]=$0 }
  END {
    print "| metric | unit | measured | baseline | ratio | tolerance | status |"
    print "|---|---|---:|---:|---:|---:|:--|"
    for (i=1;i<=n;i++) {
      m=order[i]
      if (index(tiers, ","tier[m]",")==0) continue
      if (!(m in got)) {
        printf "| %s | %s | — | %s | — | %.1fx | SKIP |\n", m, unit[m], base[m], tol[m]
        miss++; continue
      }
      r = got[m]/base[m]
      st = (r > tol[m]) ? "**REGRESSED**" : ((r < 1/tol[m]) ? "faster — re-baseline" : "ok")
      if (r > tol[m]) bad++
      printf "| %s | %s | %.3f | %.3f | %.2fx | %.1fx | %s |\n", m, unit[m], got[m], base[m], r, tol[m], st
    }
    for (m in got) if (!(m in base)) printf "| %s | ? | %.3f | — | — | — | NO BASELINE |\n", m, got[m]
    print ""
    for (i=1;i<=s;i++) print "- SKIPPED: " skipped[i]
    printf "\nratio = measured / baseline. The tolerance is a NOISE FLOOR, not a target:\n"
    printf "wall clock here moves up to ~2x run to run, so these guards catch STRUCTURAL\n"
    printf "regressions only (a lost pipeline, an added device sync, self-collision switched\n"
    printf "back on). They will not catch 30%%. See tools/perf-baseline.tsv.\n"
    printf "Frame time is NOT here: Fifo vsync makes 8.33 ms a floor — same file, bottom.\n"
    if (bad) printf "\n**%d metric(s) REGRESSED past tolerance.**\n", bad
    if (miss) printf "\n%d metric(s) did not run (SKIP above). No coverage is claimed for them.\n", miss
    if (check && bad) exit 1
    if (check && require && miss) { printf "\n--require: a SKIP is a failure.\n"; exit 1 }
  }
' "$BASELINE" "$RESULTS" "$SKIPS"
