<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

## What and why

<!-- One paragraph. What changes, and what problem it solves. Link the issue. -->

## How it was verified

<!-- Commands you actually ran and their real output. "Should work" is not
     verification. A change that has not been run is a draft. -->

```
$ make ci
```

## Gates

Everything here is checked by CI too — ticking a box you did not run wastes
a runner and your reviewer's time.

- [ ] `make fmt` — `cargo fmt --all --check`, both workspaces
- [ ] `make clippy` — `cargo clippy --workspace --all-targets -- -D warnings`, both workspaces
- [ ] `make test` — `cargo test --workspace --no-fail-fast`, both workspaces
- [ ] `make check-wgpu` — exactly one `wgpu` major per lockfile (ui/docs/DEVELOPING.md §4.1)
- [ ] `make licenses` — `cargo-deny`: allowed licences only, no git sources
- [ ] `make gpu` — adapter smoke test + the 10× determinism gate (needs a real GPU)

## Project rules

- [ ] `SPDX-License-Identifier: GPL-2.0-or-later` on every new source file
- [ ] No new dependency, or: new dep is under an allowed licence (MIT / Apache-2.0 / BSD / MPL-2.0 / LGPL / GPLv2+ / GPLv3) and `make licenses` passes
- [ ] Determinism preserved: no wall-clock, no RNG, no unordered atomics on any path that affects results
- [ ] Non-trivial logic has a test that fails if the logic breaks
- [ ] Deliberate shortcuts carry a `// ponytail:` comment naming the ceiling and the upgrade path

## Known-red

<!-- If a gate is red for a reason that is not your change, say so here so a
     red tick is not mistaken for your breakage — and link the issue.
     Last measured baseline (2026-08-01, core-engine): 356 pass / 0 fail.
     The `ui` workspace has no measured baseline yet. -->

None.
