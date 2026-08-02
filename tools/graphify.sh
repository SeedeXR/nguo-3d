#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Replicable knowledge-graph builds for nguo-3d.
#
#   tools/graphify.sh engine|app|product|all [--update] [extra graphify flags...]
#
# Every scope is pinned to the same flags so two runs are comparable:
#   --directed   "A calls B" != "B calls A"; an undirected code graph is useless
#   --mode deep  richer INFERRED edges (treat them as leads, not evidence)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --code-only is the DEFAULT on purpose, and it is not a limitation:
#   * it needs no LLM API key, so any contributor or CI runner can rebuild the
#     graph — without it graphify aborts with "no LLM API key found" the moment
#     the corpus contains a .md file, which ours always will;
#   * it is fully deterministic (local AST), so two runs are comparable, which
#     is the whole point of a *replicable* recipe.
# To also index docs/papers/images semantically, supply a key and pass
# --no-code-only:  GEMINI_API_KEY=... tools/graphify.sh engine --no-code-only
FLAGS=(--mode deep --directed --code-only)
for a in "$@"; do
    if [ "$a" = "--no-code-only" ]; then
        FLAGS=(--mode deep --directed)
        set -- "${@/--no-code-only/}"
    fi
done

command -v graphify >/dev/null 2>&1 || {
    echo "error: graphify not on PATH." >&2
    echo "install: uv tool install --upgrade graphifyy  (or pipx install graphifyy)" >&2
    exit 127
}

build() {
    local scope="$1" dir="$2"; shift 2
    [ -d "$dir" ] || { echo "skip $scope: $dir does not exist"; return 0; }

    echo "==> graphify [$scope] $dir"
    # graphify writes graphify-out/ relative to the CWD, so run from the scope root.
    ( cd "$dir" && graphify . "${FLAGS[@]}" "$@" )

    # `graphify .` builds graph.json but leaves GRAPH_REPORT.md from the PREVIOUS
    # run, so the report silently disagrees with the graph beside it. Always
    # re-cluster. (Skipped for --update, which regenerates the report itself.)
    case " $* " in
        *" --update "*|*" --cluster-only "*) ;;
        *) ( cd "$dir" && graphify cluster-only . ) ;;
    esac

    # Repair the scan root. graphify records it only on first install, so after a
    # directory move it keeps pointing at the OLD path and `--update` then finds
    # nothing while still reporting success — which is exactly how this repo's
    # engine graph went stale for a week. Cheap to rewrite, so rewrite it always.
    ( cd "$dir" && pwd ) > "$dir/graphify-out/.graphify_root"

    local report="$dir/graphify-out/GRAPH_REPORT.md"
    [ -f "$report" ] && grep -E '^- [0-9]+ nodes|^- Extraction:' "$report" || true
}

case "${1:-}" in
    engine)  shift; build engine  "$ROOT/core-engine" "$@" ;;
    app)     shift; build app     "$ROOT/ui" "$@" ;;
    # The product scope indexes the repo root. core-engine/ and ui/ carry their
    # own graphs; graphify skips nested graphify-out/ sidecars itself.
    product) shift; build product "$ROOT" "$@" ;;
    all)
        shift
        build engine  "$ROOT/core-engine" "$@"
        build app     "$ROOT/ui" "$@"
        build product "$ROOT" "$@"
        ;;
    *)
        sed -n '4,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
