<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# CI.md — what the gates are, and which of them have actually run

Written 2026-08-01, status updated 2026-08-02. **CI has still never executed.**
A git repository now exists at the tree root, but it has **0 commits and 0
tracked files**, so nothing has been pushed and no workflow has ever been
dispatched. Everything below is either *verified locally* (command run on this
Mac, output pasted into the session report) or *never executed* (needs a
runner). Each item says which.

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
| `ci.yml` → `avatars` | macos-15 | Advisory — **reports NOT RUN today**, see §8b |
| `ci.yml` → `gpu` | macos-15 | **Required** |
| `ci.yml` → `build` | macos-15 | **Required** |
| `ci.yml` → `perf` | macos-15 | Advisory (`continue-on-error`) |
| `licenses.yml` | ubuntu-latest | **Required** |
| `cross-platform.yml` → `linux` | ubuntu-latest | Advisory (`continue-on-error`) |
| `cross-platform.yml` → `windows` | windows-latest | Advisory (`continue-on-error`) |

Promote Linux to blocking at milestone M2 and Windows at M4: delete the job's
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

The session brief recorded **338 pass / 4 test targets failing** in
`core-engine` (`engine-api --test regression`; `engine-io` collada/gltf/usda).
Re-measured here at 20:25 on 2026-08-01:

```
$ make test WORKSPACES=core-engine
...
16 test targets, all ok
TOTAL passed=376 failed=0            exit 0
```

The four previously-failing targets — `regression.rs`, `collada_file.rs`,
`gltf_file.rs`, `usda_file.rs` — all ran and all passed. Another agent fixed
them mid-session. **`core-engine` is green as of that timestamp**; treat any
later red as a new regression, and re-measure rather than trusting this
paragraph, because several agents are editing concurrently.

`--no-fail-fast` is set so one broken target never masks the others. The
`test` job is required and is *not* softened in any way: if the suite is red,
CI is red. A stale internal claim of "340 tests green" was
stale for a different reason — it was never re-run.

The `ui` workspace is also green: **361 passed, 0 failed** (re-measured 2026-08-02).

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
change the other. There is no CI job for cargo-about yet — attribution
regeneration is still manual.

**Verified locally** with `cargo-deny 0.20.2`: `bans ok, licenses ok,
sources ok` for both workspaces.

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
   should be. Those are not *missing*, so the tests would not skip — they would
   fail inside the glTF reader as `AvatarError::MeshLoad`. That is the right
   failure, but it is a failure, not coverage.
2. **`assets/avatars/` is not tracked in git yet.** Verified 2026-08-03:
   `git ls-files assets | wc -l` → **0**, and `git lfs ls-files` → empty. The
   directory exists on disk and is not gitignored; it has simply never been
   added. **So `lfs: true` has nothing to fetch and there is no avatar coverage
   in CI today.**

**The design.**

| | |
|---|---|
| Where | `ci.yml` → `avatars`, macos-15 |
| Checkout | `lfs: true` — **the only LFS checkout in this repo's CI** |
| Gate | `make test-avatars`, which sets `MGUSO_REQUIRE_AVATAR_BUNDLE=1` and **fails** rather than skips when the bundle is absent |
| When the bundle is untracked | writes a **`⚠ AVATAR COVERAGE: NOT RUN`** block to `$GITHUB_STEP_SUMMARY` and does not run the suites |
| Blocking | **No.** Advisory until the bundle is committed — a required check that passes vacuously is worse than no check |

**Why not `lfs: true` on every job:** five jobs × 227 MB per push, against a
metered LFS bandwidth allowance, to make four more jobs' worth of tests skip.
One job pays it. The rest run bundle-less on purpose, which keeps "the app
degrades gracefully with no bundle" continuously exercised instead of asserted.

**Making the skip visible everywhere else.** `make test` now depends on
`make avatar-bundle`, which prints `PRESENT`/`ABSENT` plus the file count and
size before the suites run, and never fails. So a bundle-less run says so at the
top of its log instead of hiding it among ~800 test lines.

**To promote `avatars` to blocking**, once `assets/avatars/` is committed:
delete the `Bundle presence` step and the `if:` on the last step in the
`avatars` job — `make test-avatars` already fails on a missing bundle — then add
`avatars` to the branch-protection required-checks list.

**Verified locally 2026-08-03:** `make avatar-bundle` on both paths (`PRESENT
… 70 files, 232796 KB`, and `ABSENT` with `NGUO_AVATAR_BUNDLE=/nope`);
`make test-avatars NGUO_AVATAR_BUNDLE=/nope` → exit 1 with the actionable
message; `make test-avatars` against the real bundle → **exit 0**, 8 + 4 bundle
tests ran. `actionlint 1.7.12` on all three workflows → **exit 0**.
**Never executed:** the job as a job — `lfs: true`, the `$GITHUB_STEP_SUMMARY`
write and the `steps.bundle.outputs` condition have no runner behind them yet.

---

## 9. Perf: measured, not gated

The project's frame-time budgets — 8.3 ms target (120 fps), 16.6 ms hard
ceiling (60 fps), measured on canonical scenes — are meant to be enforced in
CI. **That gate is not implemented, and the
`perf` job does not pretend otherwise.**

What exists today: `mguso bench <scene> --steps 200`, which reports
`ms_per_step` / `steps_per_sec` for the solver. The `perf` job runs it, writes
the JSON to the run summary and uploads it as an artifact, and is
`continue-on-error: true`. No threshold is asserted.

Why not: §17's budgets are *UI frame times*, and there is no UI, no canonical
50-piece project, and no stored baseline to regress against. A threshold
invented now would be a number pulled from the air, and shared runners are too
noisy for a naive ms comparison anyway. There are no `benches/` directories in
either workspace (checked) — `mguso bench` is the only real measurement in the
repo.

To turn it into a gate: store a baseline (committed JSON or a
`benchmark-action` gh-pages series), compare with a tolerance wide enough for
runner noise (±30% is realistic on hosted runners), and drop
`continue-on-error`.

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

## 11. Verified locally vs never executed

**Verified — command run on this machine, output in the session report:**

- `actionlint 1.7.12` on all three workflows → exit 0 (this also shellchecks
  every `run:` block)
- `python3 -c "import yaml; yaml.safe_load(...)"` on all three workflows and
  `dependabot.yml` → all parse
- `make help`, `make check-wgpu` (both workspaces, single major 27.0.1)
- `make fmt` — ran on both workspaces. Caught two *real* in-flight problems
  minutes apart: first a formatting diff in
  `core-engine/crates/engine-api/src/auto_drape.rs`, then (after that was
  fixed) `cargo metadata` failing in `ui` because `ui/crates/nguo-app/`
  exists as a directory with no `Cargo.toml` yet. Both are other agents
  mid-edit; both are the gate doing its job, not CI defects.
- `make clippy WORKSPACES=core-engine` — first run **failed**
  (`needless_range_loop`, `crates/engine-gpu/src/lib.rs:397`), re-run after
  another agent's fix **passed**. Both outcomes observed, so the
  `-D warnings` gate demonstrably fails and passes correctly.
- `make test WORKSPACES=core-engine` → exit 0, 356 passed / 0 failed (§5)
- `make gpu-smoke EXPECT_BACKEND=Metal` → `adapter=Apple M4 backend=Metal
  device_type=IntegratedGpu`; also verified it **fails** with
  `EXPECT_BACKEND=Vulkan`
- `make determinism` → 10/10 byte-identical, and the mismatch branch verified
  to exit non-zero
- `make bench` → `ms_per_step` JSON on the canonical scene
- `make licenses` (cargo-deny 0.20.2) → `bans ok, licenses ok, sources ok`
  for **both** workspaces on the first run. A later re-run failed on `ui` with
  the same transient `nguo-app` missing-manifest error as `make fmt` — same
  cause, another agent mid-edit, not a config problem.

**Never executed — needs a runner:**

- Every job as a *job*: checkout, `Swatinem/rust-cache@v2`,
  `actions/upload-artifact@v4`, `taiki-e/install-action@v2`, the
  `$GITHUB_STEP_SUMMARY` write, concurrency cancellation. Action *references*
  are validated by actionlint; their *runtime behaviour* is not.
- Anything Metal on a hosted macOS runner. See §4: strongly evidenced, not
  observed.
- Every Linux and Windows step. No Linux or Windows machine was available.
  In particular: whether lavapipe survives the implicit/GPU determinism gate,
  whether `choco install make` + Git Bash runs this Makefile unmodified, and
  whether WARP is fast enough not to time out.
- `make clippy` / `make test` on the **`ui`** workspace. `ui/crates/*` is being
  written right now by other agents, `ui/` has no `target/` directory, and a
  cold iced+Vello+wgpu build is several GB on a disk with ~24 GiB free.
- `act` (`which act` → not found; not installed). It could not have run the
  macOS jobs anyway — `act` executes Linux containers.

---

## 12. What the maintainer must do to make any of this real

1. **`git init`** at `/nguo-3d` (the repo root, *above* `core-engine/` and
   `ui/`) — the workflows assume a single repo containing both workspaces.
   `.github/`, `Makefile` and `deny.toml` are already at that root.
2. A root `.gitignore` already exists (`target/`, `.determinism-tmp/`,
   `bench.json`, `.DS_Store`); `core-engine/.gitignore` stays as it is.
3. Commit and push to a GitHub remote, on a branch — **not** `main` directly.
4. Open a PR. That is the first real execution of any of this. Expect the
   `test` job red (§5) and, until the in-flight edits land, `lint` red too.
5. Once green, set branch protection required checks:
   `lint`, `test`, `gpu`, `build`, `cargo-deny`. Leave `perf`, `linux` and
   `windows` out until §2's milestones.
6. If the `gpu` job cannot get a Metal adapter, add a self-hosted
   Apple-Silicon runner. Do not delete the assertion.
