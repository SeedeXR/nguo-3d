<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# ppf-mguso-solver

A cross-platform **cloth / shell / rod contact solver** — pure Rust +
[wgpu](https://wgpu.rs), running on Metal (macOS), Vulkan (Linux) and DX12
(Windows) from one source tree. **No CUDA, no Python, no platform-specific
code.**

It is a port of ZOZO's [`st-tech/ppf-contact-solver`][ppf] (Apache-2.0) — a
Python + Rust + C++/CUDA, NVIDIA-only implementation — onto a portable GPU
stack. See [License](#license).

**Engine only — no UI.** Usable as a Rust library, or as a headless binary
(`mguso`) with a strict process contract: stdout is JSON, stderr is logs, the
exit code is the verdict. See [`docs/PROCESS_CONTRACT.md`](docs/PROCESS_CONTRACT.md).

## What it does

Two physics tiers. The **implicit backward-Euler tier** is the real solver; an
XPBD tier remains for fast previews and has no self-collision.

Implemented in the implicit tier: Baraff–Witkin membrane, isotropic shells
(ARAP / StVK / Stable Neo-Hookean via a 3×2 eigensystem), tet FEM, hinge
bending with a resolution-independent coefficient, cubic/log **barrier
contact** with elasticity-Rayleigh dynamic stiffness, smoothed Coulomb
friction, a per-face strain-limit barrier with its own CCD line search, pins
(static and kinematic: `move_by` / `move_to` / `spin` / `scale`) and weighted
pulls, keyframed analytic colliders, an anisotropic aerodynamic damper,
plasticity, and rod stretch. Newton terminates on `target_toi` /
`min_newton_steps`.

On the GPU (WGSL, every kernel asserted equal to its CPU implementation by
test): membrane, bending, tet and shell element kernels, a GPU-resident
unified assembly covering inertia + springs + membrane + shells, a Morton/LBVH
broad phase, and a preconditioned CG with block-Jacobi and two-level Galerkin
additive Schwarz, plus the analytic-collider IPC barrier. **Self-collision is
still CPU-only.**

**Determinism is a feature.** Same input, same backend, byte-identical output —
no wall-clock, no RNG, no float atomics on any path that affects a result. The
`make determinism` gate runs a scene 10× and requires all ten outputs to hash
identically. This is a deliberate improvement over the CUDA reference, whose
atomic-based assembly is not reproducible even against itself.

## Current state — read this honestly

- **376 tests pass, 0 fail** (`cargo test --workspace --no-fail-fast`).
- **Verified on macOS / Metal / Apple M4 only.** That is where the test suite
  and the GPU-vs-CPU oracle tests were run, and where `mguso gpu-info` brings up
  an adapter and passes its compute probe.
- **Linux (Vulkan) and Windows (DX12) are unverified.** The code is
  backend-agnostic and nothing in it is platform-specific, and the
  cross-platform CI workflow exists — but **it has never executed**, so nobody
  has run this on those backends. Treat them as expected to work, not known to
  work, and please report what you find.
- Not everything in the reference is ported. `docs/KERNELS.md` is the honest
  backlog; `docs/CONFIG.md` is the parameter-parity map.

## Quick start

```sh
cargo build --workspace
cargo test  --workspace

cargo run -p engine-cli -- gpu-info
cargo run -p engine-cli -- run examples/cloth_grid.obj --frames 60 --out /tmp/drape
cargo run -p engine-cli -- run examples/fabric_drape.json --solver implicit
cargo run -p engine-cli -- bench examples/fabric_drape.json --steps 200
```

Meshes load from OBJ, STL, glTF/GLB, COLLADA, USDA and FBX. Scenes (`.json`)
can name a fabric preset (`"preset": "denim"`) and declare sphere/plane
colliders. Runnable examples are in `examples/`.

## Layout

```
crates/
  engine-core/  solver core: sim state, step state machine, CPU physics (both tiers)
  engine-gpu/   wgpu device + compute dispatch; the WGSL kernels
  engine-api/   authoring: Scene / Object / Material / SolverConfig / auto_drape
  engine-io/    mesh & sequence I/O
  engine-cli/   `mguso`, the headless runner
docs/           DEVELOPING.md first, then PROCESS_CONTRACT / CONFIG / KERNELS / …
examples/       runnable scenes
```

## Contributing

**Read [`docs/DEVELOPING.md`](docs/DEVELOPING.md) first** — crate map, the
process contract, the test tiers, the determinism rules, the wgpu pin, and the
gates a change has to pass.

Gates are `Makefile` targets at the repository root, so local and CI cannot
drift: `make ci` (fmt, clippy, single-wgpu-major check, tests, licences) and
`make gpu` (adapter smoke test + the 10× determinism gate).

## License

**GPL-2.0-or-later.** See [`LICENSE`](LICENSE) and [`LICENSES/`](LICENSES).

This work is a port of **ZOZO's [`ppf-contact-solver`][ppf]**, which is
licensed **Apache-2.0**. Apache-2.0 is compatible with GPLv3 but not with
GPLv2 alone; combining that solver with the rest of this stack therefore
requires conveying the combined work under the GPL, which is why this
repository is GPL-2.0-**or-later** rather than GPLv2-only. The upstream
copyright and its Apache-2.0 notice are recorded in
[`LICENSES/README.md`](LICENSES/README.md) and must be preserved in any
redistribution.

Every source file carries an `SPDX-License-Identifier: GPL-2.0-or-later`
header. Dependencies must be MIT / Apache-2.0 / BSD / MPL-2.0 / GPLv2+ /
GPLv3; `cargo-deny` enforces the allow-list in `make ci`.

[ppf]: https://github.com/st-tech/ppf-contact-solver
