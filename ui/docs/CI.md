<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# CI.md — what the gates are, and which of them have actually run

Written 2026-08-01. **Status rewritten 2026-08-04: CI has executed, and `main`
is RED.**

Every earlier version of this file said CI had never run. It has. The repository
has **2 commits, both pushed** — `5077aef` (2026-08-02) and `7ce7827`
(2026-08-03, the avatar bundle) — and three workflows have run on each.
Verified with `gh run list --branch main` and `gh run view`:

| Push | `CI` | `Licenses` | `Cross-platform` (DISABLED since 2026-08-05) |
|---|---|---|---|
| `5077aef` — initial | **success** | **success** | success |
| `7ce7827` — avatar bundle | **FAILURE** | **FAILURE** | success |

**What failed on `main`, and how each was fixed at the root. Both fixes are in
the working tree and NOT yet committed** — the maintainer commits.

1. **`CI → cargo test (both workspaces)`.** That job checks out with
   `lfs: false`, so `assets/avatars/**/*.glb` arrive as ~130-byte LFS **pointer
   text**. Pointer files are not *missing*, so the bundle-gated tests did not
   skip — they failed inside the glTF reader. Four `engine-api` tests panicked
   at `avatar_bundle.rs:168`
   (`cloth_dropped_on_a_real_proxy_does_not_sink_into_the_body`,
   `collision_mesh_matches_the_manifest_col_tier`,
   `full_proxy_lowers_into_a_static_collider_scene`,
   `real_proxy_scene_is_deterministic`). Every other job in that run passed,
   **including the dedicated `avatars` job with `lfs: true`** (13m34s, green) —
   so the LFS design works and the job that declined to opt into it is the one
   that broke.

   Latent behind it, and never reached because `make test` does `core-engine`
   first: **three more failures in `ui`** — `nguo-avatar`'s
   `collision_meshes_load_through_engine_io_as_finite_human_sized_bodies`,
   `declared_but_absent_files_are_pinned_and_reported_absent` and
   `every_shared_skin_map_is_on_disk`. Reproduced locally 2026-08-04 against a
   pointer fixture. Fixing only the four named `engine-api` tests would have
   turned one red job into the same red job.

   **Fixed at the one place all consumers share.** `engine_io::is_lfs_pointer`
   (`core-engine/crates/engine-io/src/lib.rs`) reads the first 42 bytes and
   matches the pointer magic. `engine_io::load_mesh` refuses a pointer before
   format dispatch, so *every* mesh format now says "run `git lfs pull`"
   instead of emitting a parse error about its own grammar. On top of that
   predicate, both bundle gates treat a pointer as an **absent** asset:
   `nguo_avatar::Bundle::open` returns the new `AvatarError::LfsPointer`, and
   `engine-api`'s `collision_tiers()` takes its loud-skip path.
   `MGUSO_REQUIRE_AVATAR_BUNDLE=1` still converts both skips into hard
   failures, so the `avatars` job keeps covering for real.

   **This was the cheap fix of the two, and it is only honest while the
   `avatars` job is watched.** The alternative — `lfs: true` on the `test` job —
   costs 227 MB per push against a **1 GiB/month free-plan allowance** (§8b),
   about four runs, and loses the "degrades gracefully with no bundle" coverage
   the design deliberately keeps. Taking the cheap fix means **no blocking job
   covers the avatar path** until `avatars` is promoted. Promote it (§12.2).

2. **`Licenses → attribution-check`** (`Makefile:189`): the generated
   `LICENSES/THIRD_PARTY.md` was stale — and would have stayed stale forever.
   Not a forgotten regeneration. See §8 for the root cause (CRLF vs
   `.gitattributes`), which is why it passed on the author's machine and failed
   on every runner.

Below, each item still says whether it was *verified locally* or *observed on a
runner*; "never executed" now means only the specific paths noted.

---

## 1. One command, two workspaces

`nguo-3d` contains **two independent cargo workspaces** — `core-engine/`
(`engine-{core,gpu,api,io,cli}`, binary `mguso`) and `ui/`
(`ui/crates/*`, binary `nguo` once it exists). Neither contains the other, so
every gate loops over both.

The loop lives in **one place**: the repo-root `Makefile`. Workflows call
`make <target>`; they never inline cargo commands. That is the whole
anti-drift mechanism — if `make ci` passes locally, CI runs the identical
commands.

```
make ci          # fmt + clippy + check-wgpu + test + licenses   (no GPU needed)
make gpu         # adapter smoke test + the 10x determinism gate (needs a GPU)
make help        # list every target
```

Useful overrides:

| Variable | Default | Why you'd change it |
|---|---|---|
| `MGUSO` | `core-engine/target/release/mguso` | Reuse a debug build: `make gpu MGUSO=core-engine/target/debug/mguso` |
| `WORKSPACES` | `core-engine ui` | Gate one workspace only: `make test WORKSPACES=core-engine` |
| `EXPECT_BACKEND` | *(empty)* | Assert the wgpu backend: `Metal` / `Vulkan` / `Dx12` |
| `DET_RUNS` | `10` | Determinism run count |
| `DET_SOLVER` | `implicit` | `xpbd` for the CPU path |

Prerequisites beyond the Rust toolchain: `jq` (adapter assertions) and
`cargo-deny` (`brew install cargo-deny`).

---

## 2. The workflows

| Workflow | Runner | Blocking? |
|---|---|---|
| `ci.yml` → `lint` | macos-15 | **Required** |
| `ci.yml` → `test` | macos-15 | **Required** |
| `ci.yml` → `avatars` | macos-15 | Advisory — **ran and PASSED** on `7ce7827` (13m34s); promote it to blocking, see §8b |
| `ci.yml` → `gpu` | macos-15 | **Required** |
| `ci.yml` → `build` | macos-15 | **Required** |
| `ci.yml` → `perf` | macos-15 | Advisory (`continue-on-error`) |
| `licenses.yml` | ubuntu-latest | **Required** |
| `cross-platform.yml` → `linux` | ubuntu-latest | **DISABLED 2026-08-05** — manual `workflow_dispatch` only |
| `cross-platform.yml` → `windows` | windows-latest | **DISABLED 2026-08-05** — manual `workflow_dispatch` only |

**Linux and Windows are DISABLED and therefore UNTESTED (2026-08-05).** Nobody on the
project has a machine to verify them on, so the `push`/`pull_request` triggers in
`cross-platform.yml` are commented out and the jobs run only via manual
`workflow_dispatch`. Do not describe either platform as supported or passing until
that workflow has been run on real hardware. `licenses.yml` also uses an
`ubuntu-latest` runner but is NOT a platform test — cargo-deny/cargo-about read
licence metadata, which is platform-independent — and it stays enabled.

Once a machine exists: uncomment the two triggers, then promote Linux to blocking at
milestone M2 and Windows at M4 by deleting the job's
`continue-on-error` line and add it to the branch-protection required-checks
list.

All three workflows set `concurrency: cancel-in-progress: true`, so a new push
to a branch cancels the older run, and `permissions: contents: read`.

---

## 3. Runner choice: macos-**15**, not macos-14

The brief specified `macos-14`. The workflows use `macos-15`. Reason:

> The macOS 14 Sonoma based runner images will begin deprecation on **July
> 6th** and will be fully unsupported by **November 2nd** for GitHub Actions
> and Azure DevOps. […] Brownout: **October 5 – October 31, 2026**. Workflows
> using the `macos-14`, `macos-14-large`, `macos-14-xlarge` image labels will
> be terminated with an error.
>
> — [actions/runner-images#13518](https://github.com/actions/runner-images/issues/13518)

Writing `runs-on: macos-14` today buys nine weeks and then hard-fails.
`macos-15` is the same Apple-Silicon (arm64) hardware class, so the Metal
behaviour this project cares about is unchanged. To move the whole pipeline,
change the four `runs-on:` lines in `ci.yml`.

---

## 4. Does the macOS runner really have Metal?

**Yes — and the citation matters, because the top search result says
otherwise and is out of date.**

- The "no Metal on GitHub-hosted macOS" claim traces to
  [actions/runner-images#7085](https://github.com/actions/runner-images/issues/7085),
  opened February 2023 and since closed. It predates the Apple-Silicon
  runners.
- The authoritative counter-evidence is **wgpu's own CI**:
  [`gfx-rs/wgpu/.github/workflows/ci.yml`](https://github.com/gfx-rs/wgpu/blob/trunk/.github/workflows/ci.yml)
  has a `gpu-test` job whose matrix includes `- name: Mac aarch64 / os: macos-14`,
  and that job runs `cargo --locked xtask test --llvm-cov` — wgpu's real
  GPU conformance suite. wgpu would not gate its own releases on a runner with
  no working Metal adapter.

Mechanically: the arm64 runners are Apple-Silicon VMs under
Virtualization.framework, which exposes a paravirtualised Metal device
(`AppleParavirtGPU`). Expect the adapter string to be something like
"Apple Paravirtual device" rather than "Apple M4".

**This is why `make gpu-smoke` asserts on `.backend == "Metal"` and on the
SAXPY probe returning the right numbers — not on the adapter name.** The
adapter string on a hosted runner is the one thing here I could not verify;
the backend and the arithmetic are what actually matter.

**No software fallback exists on Apple platforms.** There is no lavapipe for
Metal. If the `gpu` job cannot obtain an adapter, the correct fix is a
self-hosted Apple-Silicon runner — *not* weakening the assertion to make the
tick go green.

---

## 5. Test baseline

**Re-measured 2026-08-04** with `cargo test --workspace --no-fail-fast` in each
workspace, on this machine, in debug:

| Workspace | Result |
|---|---|
| `core-engine` | **399 passed, 0 failed, 3 ignored** — exit 0 |
| `ui` | **542 passed, 2 failed, 3 ignored** — exit 101 |

The two `ui` failures are `nguo-cli/tests/cli_end_to_end.rs:141,183`. They need
the `mguso` binary and **hard-fail rather than skip** when
`core-engine/target/release/mguso` is absent. On a runner that is not a problem —
the `test` job builds both workspaces — but on a clean laptop `cargo test` in
`ui/` is red until you run `cargo build -p engine-cli` in `core-engine/` or set
`$NGUO_MGUSO`. That is the "no vacuous passes" rule biting its own author, and
it is the right trade.

**Re-measured again 2026-08-04, after the LFS-pointer fix, with `mguso` built:
`ui` is 544 passed, 0 failed, 3 ignored.** Same suite, same machine — the "2
failed" above is precisely those two `nguo-cli` tests finding the binary this
time. If you see 542/2, build `mguso`; nothing is broken.

**This baseline is debug.** `cargo test -p engine-core --release` currently
fails `tet.rs:1045 analytic_gradient_is_much_faster_than_fd` (4.6× measured
against a 5× threshold, 3/3 on an idle machine) — pre-existing and tracked, but
it means "the suite is green" is only true of the debug profile.

**Do not trust this paragraph over a run.** Its previous version recorded
376/361 and was two sessions out of date, and three different `core-engine`
counts were in circulation across the docs at once. Re-measure.

`--no-fail-fast` is set so one broken target never masks the others. The `test`
job is required and is *not* softened in any way: if the suite is red, CI is
red.

---

## 6. The 10× determinism gate

The determinism gate: *"run each scenario 10×; a single failure fails the
suite."* `make determinism` implements exactly that against the **real CLI** —
no new mechanism was invented:

```
mguso run core-engine/examples/fabric_drape.json \
      --frames 5 --solver implicit --out <tmpdir>/runN
```

× 10, then SHA-256 over each run's stdout JSON summary concatenated with its
per-frame OBJ sequence. All ten hashes must be identical.

`--solver implicit` is the default because that is the GPU path
(`implicit/gpu-pcg-block-jacobi(Metal)`) — the only place unordered atomics or
nondeterministic reductions could actually bite. `DET_SOLVER=xpbd` runs the
CPU path.

**Verified locally**: 10/10 byte-identical on Apple M4 / Metal, and the
comparison logic was checked to exit 1 on differing hashes.

---

## 7. The single-wgpu-version check

wgpu version drift is the #1 integration hazard: iced, Vello, egui and the 3D
viewport must share **one** `wgpu::Device`, and wgpu majors are mutually
incompatible. Decision of record (`core-engine/docs/WGPU_PIN.md`): everything
is on **wgpu 27**.

`make check-wgpu` parses each `Cargo.lock` — no build, no network, runs in
milliseconds, so it is the *first* step in the `lint` job. If a lockfile holds
more than one wgpu major it prints `cargo tree -i wgpu@<version>` for each,
naming exactly which crates pulled which. A missing `Cargo.lock` is reported
and skipped, so it does not break while `ui/` is still being built out.

Second line of defence: `deny.toml`'s `[bans]` rejects `wgpu < 27` and
`wgpu >= 28` outright, so the check survives someone bypassing the Makefile.

**Verified locally**: both lockfiles report a single major (27.0.1).

---

## 8. Licences

`deny.toml` at the repo root is shared by both workspaces. Allowed: MIT /
Apache-2.0 / BSD / MPL-2.0 / LGPL / GPL-2.0-**or-later** / GPL-3.0-or-later
plus the usual permissive tail (ISC, Zlib, 0BSD, Unicode-3.0, CC0…).
`GPL-2.0-only` is **not** allowed — bare `GPL-2.0` normalises to it — because
it blocks the GPLv3 upgrade path our GPL-2.0-or-later tree relies on.
`[sources]` denies git dependencies and unknown registries.

This is the **enforcement** half. `core-engine/about.toml` + `about.hbs`
(cargo-about → `LICENSES/THIRD_PARTY.md`) are the **attribution** half and
already existed; the two allow-lists are deliberately kept in sync. Change one,
change the other.

**There IS a CI check for cargo-about, and it was the thing failing.**
`licenses.yml` runs `make attribution-check` (`Makefile:189`), which regenerates
each `LICENSES/THIRD_PARTY.md` and diffs it against the committed copy. On
`7ce7827` it reported **STALE** and the workflow exited 2. Regeneration itself
is still manual (`make attribution`); the check only tells you that you forgot.

**But nobody had forgotten. The root cause was line endings, and it made the
check unsatisfiable.** Reproduced 2026-08-04 by exporting `HEAD` with
`git archive` and running the same target: 208 lines of Apache-2.0 text differ
in `core-engine/LICENSES/THIRD_PARTY.md`, contributed by **`approx 0.5.1`,
whose `LICENSE` ships CRLF** (202 CRs, `file` says "with CRLF line
terminators"). `cargo-about` embeds each licence verbatim, so the generated
file carries those CRLFs. `.gitattributes` puts `*.md` on `text eol=lf`, so
**git rewrites them to LF on commit**. From that moment the committed bytes can
never equal the generator's output again, on any machine, forever.

That is also why it passed locally and failed on every runner: the author ran
`make attribution` and then `make attribution-check` against the *working tree*,
where the un-normalised CRLF was still on disk and the two matched. Git's
normalisation happens between working tree and index — a boundary nothing in
the pre-push path crossed. **A generated file drifted from its source because
the check and the runner were reading different bytes.**

**Fix:** `ABOUT_GEN` (`Makefile:39`) now pipes the generated file through
`tr -d '\r'`, so generated bytes == committed bytes on every OS, including a
Windows contributor's. Verified: after regeneration both files have 0 CRs and
`git hash-object --no-filters` equals `git hash-object --path` for each — i.e.
what is on disk is exactly what git will store.

**The clinching evidence that this was line endings and nothing else:** after
regeneration, `core-engine/LICENSES/THIRD_PARTY.md` diffs to **zero lines**
against the copy committed in `7ce7827`. The committed attribution was never
content-stale; it was only ever CRLF-stale. (`ui`'s does change, by 105 lines —
`spade`, `robust`, `allocator-api2` and the two new workspace members
`nguo-sewing` / `nguo-triangulate`. That part is ordinary, correct drift.)
And it is **not** dependabot: `glam` is 0.30.10 in both lockfiles on `main`;
0.33.2 exists only on an unmerged branch.

**Where to hook the check pre-push.** With the fix in place a plain local run
is now trustworthy, and it is cheap — `make attribution-check` measured
**25.4 s wall** for both workspaces on this machine (`/usr/bin/time -p`,
Apple M4, warm cargo registry), no network, no build. Put it in
`.git/hooks/pre-push` (not
pre-commit: too slow to pay on every commit, and attribution only matters at the
point of publishing). Until such a hook exists, run `make licenses` before any
push that touched a `Cargo.toml`, a `Cargo.lock`, an `about.toml`/`about.hbs`,
or added a workspace crate — those are the only inputs that can move it.

**Verified locally** with `cargo-deny 0.20.2`: `bans ok, licenses ok,
sources ok` for both workspaces — the `deny` half is not what is red.

---

## 8b. Avatar coverage, and why it is its own job

Added 2026-08-03, with the avatar bundle replacement.

**The problem.** `assets/avatars/` is **227 MB** (70 files, verified with `find`
+ `stat`) and its `.glb`/`.vrm`/`.png` are behind **git-LFS**. The 13 avatar
tests — 8 in `ui/crates/nguo-avatar/tests/bundle.rs`, 5 (1 `#[ignore]`) in
`core-engine/crates/engine-api/tests/avatar_bundle.rs` — are **bundle-gated**:
with no bundle they print a skip notice and *pass*. So on a runner without the
bundle, `make test` is green while covering none of the avatar path.

**Two facts drove the design, both checked rather than assumed:**

1. **`actions/checkout` does not fetch LFS objects.** It needs `lfs: true`.
   Without it the tree gets ~130-byte pointer **text** files where the GLBs
   should be. ~~Those are not *missing*, so the tests would not skip — they
   would fail inside the glTF reader as `AvatarError::MeshLoad`. That is the
   right failure, but it is a failure, not coverage.~~ **Superseded
   2026-08-04.** That was the design and it was wrong: "the right failure" in a
   *required* job is just a red build. A pointer is now recognised as such
   (`engine_io::is_lfs_pointer`) and counts as an **absent** asset, so the gates
   skip. `MGUSO_REQUIRE_AVATAR_BUNDLE=1` still makes it a hard failure, which is
   where the "loud and typed" property actually belongs — in the `avatars` job,
   the one that opted into LFS and therefore has no excuse for a pointer.
2. ~~**`assets/avatars/` is not tracked in git yet.**~~ **It was committed in
   `7ce7827` and pushed on 2026-08-03.** `git ls-files assets` → **70**,
   `git lfs ls-files` → **36** (re-verified 2026-08-04). So `lfs: true` now has
   something to fetch, the `avatars` job really ran, and **prediction (1) above
   came true the same day**: the `test` job, which does *not* set `lfs: true`,
   went red with four glTF-reader failures instead of skips — plus three more in
   `ui` that nobody saw, because `make test` runs `core-engine` first and stops.
   Fixed; see the status block at the top of this file.

**LFS bandwidth, since the `avatars` job spends it on every push.** The
`SeedeXR` org is on the **free plan** (`gh api /orgs/SeedeXR` → `plan.name:
"free"`, checked 2026-08-04), and the repo is public. GitHub's free LFS
allowance is 1 GiB storage and **1 GiB/month bandwidth**, billed to the
repository owner; a public repo gets no exemption. At 227 MB per `lfs: true`
checkout that is **~4 runs of the `avatars` job per month** before the quota is
gone — and dependabot alone opens enough PRs to reach it. Whether a data pack
has been bought could **not** be confirmed: the billing endpoints need the
`admin:org` scope this token does not have
(`/orgs/SeedeXR/settings/billing/shared-storage` → 410 "moved", the replacement
→ 403 needs `admin:org`). **The maintainer must check this directly** at
Settings → Billing → Git LFS Data. If there is no data pack, the honest options
are to run `avatars` on a schedule or on `assets/avatars/**` path changes rather
than every push, or to buy a pack. Adding `lfs: true` to a second job doubles
the burn.

**The design.**

| | |
|---|---|
| Where | `ci.yml` → `avatars`, macos-15 |
| Checkout | `lfs: true` — **the only LFS checkout in this repo's CI** |
| Gate | `make test-avatars`, which sets `MGUSO_REQUIRE_AVATAR_BUNDLE=1` and **fails** rather than skips when the bundle is absent |
| When the bundle is untracked | writes a **`⚠ AVATAR COVERAGE: NOT RUN`** block to `$GITHUB_STEP_SUMMARY` and does not run the suites — **no longer reachable on `main`; the bundle is tracked** |
| Blocking | **Not yet, and it should be now.** It was advisory only because the bundle was untracked and a required check that passes vacuously is worse than no check. That condition is gone: the job ran green on a real runner in 13m34s |

**Why not `lfs: true` on every job:** five jobs × 227 MB per push, against a
metered LFS bandwidth allowance, to make four more jobs' worth of tests skip.
One job pays it. The rest run bundle-less on purpose, which keeps "the app
degrades gracefully with no bundle" continuously exercised instead of asserted.

**Making the skip visible everywhere else.** `make test` now depends on
`make avatar-bundle`, which prints `PRESENT`/`ABSENT` plus the file count and
size before the suites run, and never fails. So a bundle-less run says so at the
top of its log instead of hiding it among ~800 test lines.

**To promote `avatars` to blocking — the precondition is met, do it:** delete
the `Bundle presence` step and the `if:` on the last step in the `avatars` job
(`make test-avatars` already fails on a missing bundle), then add `avatars` to
the branch-protection required-checks list. Until that happens the only
*blocking* job that touches the bundle is `test`, which touches it wrongly.

**Verified locally 2026-08-03:** `make avatar-bundle` on both paths (`PRESENT
… 70 files, 232796 KB`, and `ABSENT` with `NGUO_AVATAR_BUNDLE=/nope`);
`make test-avatars NGUO_AVATAR_BUNDLE=/nope` → exit 1 with the actionable
message; `make test-avatars` against the real bundle → **exit 0**, 8 + 4 bundle
tests ran. `actionlint 1.7.12` on all three workflows → **exit 0**.
**Never executed:** the job as a job — `lfs: true`, the `$GITHUB_STEP_SUMMARY`
write and the `steps.bundle.outputs` condition have no runner behind them yet.

---

## 9. Perf: compared against a baseline, still advisory

`make profile` → `tools/profile.sh` collects every benchmark that already
exists in the crates and prints one table against `tools/perf-baseline.tsv`,
which records each number, the hardware it was taken on, and the tolerance
beyond which `--check` calls it a regression.

There are still no `benches/` directories — deliberately. The measurements are
`#[ignore]`d **tests**, so `make test` never pays for them and they cannot rot:

| tier | measurement | metric |
|---|---|---|
| `quick` | `mguso bench <scene> --steps 200` | `solver_ms_per_step` |
| `quick` | `nguo-registry` `tests/overhead.rs` | `registry_ns_per_op` |
| `gpu` | `nguo-gpu` `tests/handoff_bench.rs` | `handoff_blocking_ms_100k`, `handoff_pipelined_ms_100k` |
| `avatar` | `engine-api` `tests/avatar_bundle.rs::collision_proxy_step_cost` | `proxy_step_s_selfcol_off`, `proxy_step_s_selfcol_on` |

`tools/profile.sh --selftest` checks the output parsers against a fixture and
runs no benchmark. It exists because the first version of one parser matched
nothing and the table simply said `SKIP` — silent zero coverage.

**The `perf` job stays `continue-on-error: true`, and that is a measured
decision, not caution.** On bare-metal Apple M4, *identical binaries* gave:
`mguso bench` 20.53 → 28.48 ms/step (1.39x); `nguo-registry` start+done
65.4 → 131.7 ns/op (2.01x) — and in one run the strictly-more-work variant came
out *faster*; blocking handoff at 100k verts 2.885 → 3.871 ms (1.34x); the full
collision proxy with self-collision 0.247 → 0.928 s/step (3.76x, because that
benchmark times a single step with no averaging). The "±30% on hosted runners"
figure this section used to quote is off by a factor of two on bare metal. So
the tolerances are 2x–4x and catch **structural** regressions only: pipelining
lost, a device sync added, self-collision switched back on. A `REGRESSED` row in
the job summary means "a human should look", never "this PR is broken".

CI does not run the `avatar` tier: it needs the 227 MB bundle and `checkout`
does not fetch LFS without `lfs: true`, which the `perf` job does not set. (The
bundle **is** tracked as of `7ce7827` — only the `avatars` job opts into LFS.)
The job passes `--only quick,gpu` and the two proxy rows are absent rather than
falsely green.

**Frame time is not measured here, and the reason must travel with every
frame-time number in this project.** `nguo-app` forces
`ICED_PRESENT_MODE=fifo` (`src/main.rs:1239`), so on the 120 Hz dev display
**8.33 ms is a hard floor** — the frame thread blocks on vblank. The
8.35–8.74 ms figures prove the 120 fps cadence is *kept*; they are not GPU
headroom, and a GPU 4x too slow would still read 8.33 ms until it began
dropping frames (then 16.6 ms). Where the budget *can* be enforced it is:
`handoff_bench.rs`'s `verdict()` scores every transfer against 8.3/16.6 ms.

To make it a real gate: run `tools/profile.sh --check` on a self-hosted machine
with `BASELINE=` pointing at a baseline measured **on that machine**, then drop
`continue-on-error`. Do not tighten the hosted-runner numbers instead.

---

## 10. Linux and Windows specifics

**Linux.** Installs `mesa-vulkan-drivers libvulkan1 vulkan-tools` and forces
lavapipe with
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`. That ICD
filename was checked against the Ubuntu package file list
([mesa-vulkan-drivers, noble/amd64](https://packages.ubuntu.com/noble/amd64/mesa-vulkan-drivers/filelist)),
not guessed. Note there is **no `vulkan-swiftshader` apt package** — SwiftShader
is a Google project shipped as source/prebuilt blobs; lavapipe (Mesa's LLVM
software rasteriser) is the apt-installable software Vulkan on Ubuntu. Because
`ubuntu-latest` has no GPU, this is software rendering by design: it proves the
Vulkan code path compiles, links and runs, not that it is fast. The job also
installs the winit/iced/rfd system deps (`libxkbcommon-dev`, `libwayland-dev`,
X11/xcb, `libgl1-mesa-dev`) the UI will need.

**Windows.** GNU Make is **not** preinstalled on `windows-latest` (checked
against `images/windows/Windows2022-Readme.md`; `jq 1.8.1` is present, `make`
is not), so the job runs `choco install make` first, and all steps use
`shell: bash` (Git Bash). The DX12 smoke test will select "Microsoft Basic
Render Driver" (WARP), the OS software DX12 device — still the real DX12 code
path (HLSL generation, root signatures, descriptor heaps), which is the point
of testing Windows from day one.

The Makefile is written for portability across the three: `SHA256` picks
`sha256sum` or `shasum -a 256`, and `EXE` appends `.exe` on Windows.

---

## 11. What has actually run on a runner

**Rewritten 2026-08-04.** This section used to be "verified locally vs never
executed", because nothing had been pushed. Three workflows have now run on
`main` twice each (the status block at the top of this file), so most of it is settled by observation rather than by
argument. `gh run view` is the authority; the session-1 local-verification list
is kept below it because it still records *how* each gate was proven to fail as
well as pass.

**Observed on a runner (`gh run list --branch main`, `gh run view`):**

| Job | On `5077aef` | On `7ce7827` | Note |
|---|---|---|---|
| `lint` (fmt + clippy + wgpu pin) | ✅ | ✅ 3m32s | |
| `test` (both workspaces) | ✅ | ❌ 11m30s | LFS pointer files — see the status block at the top |
| `gpu` (Metal smoke + 10× determinism) | ✅ | ✅ 2m31s | **§4's argument is now observation: Metal works on a hosted macos-15 runner** |
| `avatars` (`lfs: true`) | n/a | ✅ 13m34s | the bundle only existed from `7ce7827` |
| `build` (release binaries) | ✅ | ✅ 4m37s | |
| `perf` (measure only) | ✅ | ✅ 2m16s | |
| `licenses.yml` | ✅ | ❌ 40s | `attribution-check` — see the status block at the top |
| `cross-platform` linux + windows | ✅ | ✅ 8m25s | **that run predates 2026-08-05; the workflow is now DISABLED and does not run on push** |

So: **Metal on a hosted runner, LFS checkout, `$GITHUB_STEP_SUMMARY`,
`Swatinem/rust-cache`, `taiki-e/install-action`, lavapipe, DX12 and the whole
Makefile-driven design are no longer unverified.** Dependabot is live too — six
remote branches, and its PRs currently fail `CI` and `Licenses` for the same two
reasons `main` does. Note for the triage: the dependabot bumps in flight
(`glam 0.30.10 → 0.33.2`) are **not** why `attribution-check` fails. The stale
block is Apache-2.0 text from `approx 0.5.1`, and the cause is line endings, not
a version change — §8.

**One deprecation warning on every job, unaddressed:** `actions/checkout@v4`
targets Node 20, which GitHub is forcing onto Node 24. Not yet an error. Note
that a dependabot PR bumping `actions/checkout` from 4 to 7 is open and its CI
is red — for the `main` reasons, not for the bump.

**Still never executed:** `act` (not installed, and it runs Linux containers so
it could not do the macOS jobs anyway); a self-hosted perf runner, which is what
§9 needs before `perf` can stop being advisory.

**Kept from session 1 — locally verified, and valuable because each gate was
seen to *fail* as well as pass:**

- `actionlint 1.7.12` on all three workflows → exit 0 (it also shellchecks every
  `run:` block); `yaml.safe_load` on all three plus `dependabot.yml`.
- `make clippy WORKSPACES=core-engine` — first run **failed**
  (`needless_range_loop`, `crates/engine-gpu/src/lib.rs:397`), passed after the
  fix. Both outcomes observed, so `-D warnings` demonstrably works.
- `make gpu-smoke EXPECT_BACKEND=Metal` → `adapter=Apple M4 backend=Metal
  device_type=IntegratedGpu`; verified it **fails** with `EXPECT_BACKEND=Vulkan`.
- `make determinism` → 10/10 byte-identical, mismatch branch verified to exit
  non-zero.
- `make licenses` (cargo-deny 0.20.2) → `bans ok, licenses ok, sources ok`, both
  workspaces.
- `make fmt` caught two real in-flight problems minutes apart — a formatting
  diff and a workspace member with no `Cargo.toml`. The gate doing its job.

---

## 12. What the maintainer must do next

Steps 1–4 of the old version (git init, .gitignore, first push, first PR) are
**done**. What is left:

1. **Commit the two fixes and confirm `main` goes green.** Both failures in the
   status block at the top are fixed in the working tree, reproduced and
   re-verified locally, but **nothing is committed** — that is the maintainer's
   call. Neither was a workflow-design problem: one was a presence gate that
   could not tell a git-LFS pointer from a file, the other was CRLF versus
   `.gitattributes`.
2. **Promote `avatars` to a blocking check. This is now urgent, not tidy.**
   Fix (1) makes the `test` job *skip* the avatar suites instead of failing on
   them, so with `avatars` still advisory **no blocking job covers the avatar
   path at all**. Its precondition — a committed bundle — is met and it passed
   on a real runner. Delete the `Bundle presence` step and the `if:` on the
   job's last step, then add it to branch protection.
2b. **Check the Git LFS bandwidth bill before doing (2).** The org is on the
   free plan: 1 GiB/month, ~4 runs of `avatars`. See §8b. If there is no data
   pack, make `avatars` run on a schedule or on `assets/avatars/**` changes
   rather than on every push — a required check that runs out of quota mid-month
   blocks every PR.
3. **Set branch protection required checks** if they are not set:
   `lint`, `test`, `gpu`, `build`, `avatars`, `cargo-deny`. Leave `perf`,
   `linux` and `windows` advisory until §2's milestones.
4. **Triage the open dependabot PRs.** They are red for `main`'s reasons, not
   their own, so they cannot be assessed until step 1 lands.
5. **Bump `actions/checkout`** off Node 20 before GitHub turns the warning into
   an error (a dependabot PR already proposes v7).
6. If the `gpu` job ever cannot get a Metal adapter, add a self-hosted
   Apple-Silicon runner. Do not delete the assertion — it works today.
