<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# GAUGE_FINDINGS.md — what the reference capture measured

Grounded summary of a recorded capture of `ppf-contact-solver` @ `38261ea`
running on an NVIDIA A100: 22 scenes with their inputs, per-frame checkpoints,
invariant logs and performance traces.

> **The capture is an internal benchmark corpus and is not distributed with
> this source.** It is bulk machine output, not code — ~950 MB of captured
> geometry and traces specific to one GPU. Everything it taught us that
> matters for porting is written down *here*, in `CONFIG.md` and in
> `KERNELS.md`, so this document stands on its own. Citations below name files
> *inside the capture* so the provenance of each number is auditable; they are
> not links you can follow from a clone.
>
> The code paths that read a capture (`crates/engine-io/src/golden.rs`,
> `mguso golden`, `mguso gauge-scene`) are still built and tested for their
> parsing logic; they find nothing to read in a clean clone and skip loudly.

This is the source of truth for *validation*, *performance priorities* and
*per-module port specs*. Read it before optimizing or adding regression gates.

## 1. Acceptance criteria (this changes how we test)
The reference is **not reproducible against itself**: re-running `drape` on the
same A100 from byte-identical inputs self-diverges to **5% of the bbox diagonal
by frame 60** (`perf/determinism_drape.txt`), caused by `atomicAdd` column-order
nondeterminism in `csrmat.cu:83-86`. So a tight per-vertex gate (1e-3) *fails the
reference itself*. **This validates our invariant-first testing.**

Acceptance order (the capture's `INSTRUCTIONS.md §6.3` and `harness/compare_golden.py`):
1. **The four invariants (primary gate)** — the capture's `harness/verify_invariants.py`:
   - `penetration_free`: no `"intersection detected" / "ccd failed" / "pcg failed"
     / "OVERFLOW"` markers in the log. Zero = pass.
   - `strain_limit_respected`: `max(max_sigma) ≤ (1+limit)·1.02` (2% slack).
   - `cg_converged`: `max(reresid) ≤ cg_tol·1.5`; `cg_iter_under_cap`: `max(iter)
     < cg_max_iter`.
   - `zero_steady_state_alloc`: device alloc/free counts sum to 0 over the second
     half of the run (pool warm-up allowed in first half).
   - plus `all_frames_finite` (no NaN/Inf ever) and `csr_capacity_ok`.
2. **Bounded relative error** at 6 checkpoints vs bbox-diagonal: `--max-rel 8e-2`
   (smoke alarm), `--max-rms-rel 2e-2` (meaningful), `--max-centroid-rel 6e-3`
   (tightest). Each ~1.5–3× the self-divergence floor.
3. **Divergence growth** must plateau, not climb (reported, human-judged; >100× = bug).
4. **Frame 0 bit-identical** — it's the initial condition; a diff = input-parse bug.

→ Our acceptance gate is these invariants, not tight per-vertex matching. They
are implemented in `engine_core::metrics::check_invariants` and exposed as
`mguso verify <scene>`, which exits **2** when an invariant fails (distinct
from exit 1 = the tool itself failed). The three relative gates are applied by
`mguso golden`, against a capture when one is available.

## 2. Performance priority order (measured)
Phase split (`perf/phase_breakdown.csv`, 86% of step time accounted):
| phase | % step | note |
|---|---|---|
| **linsolve** | **28.8%** | dominated by CG iteration count (mean up to 1068, max 2773) |
| **assembly** | **27.3%** | **79% is contact-Hessian** assembly |
| **linesearch** | **22.7%** | CCD + strain-limit TOI |
| intersect | 4.6% | |
| lbvh | 2.3% | |
| aabb | 0.6% | |

**The linear solve is #1, and it's iteration-bound** → a better preconditioner is
the single highest-leverage perf lever (confirms our direction). Assembly is
essentially contact-Hessian. Do not generalize one scene's mix to others.

A100 roofline ceilings (`perf/roofline/roofline.json`): HBM triad 1777 GB/s,
random gather **196.8 GB/s** (7.6× slower than sequential — assembly pattern),
3×3-block CSR SpMV 577.8 GB/s / 1.79 ms per apply @1M nodes, launch+sync tax
**~6.5–7.2 µs/dispatch** (the reference's biggest structural inefficiency:
`cudaStreamSynchronize(0)` after every launch at 107 sites — our wgpu port already
batches, see pcg.rs `dispatch_batch`).

## 3. Preconditioner (directly informs our Schwarz work)
Reference default is **block-Jacobi**; **Schwarz is opt-in** (`params.rs:264`,
`PPF_PRECOND`). So our block-Jacobi default is correct. Reference Schwarz
(the capture's `docs/PORTING_preconditioner.md`, `schwarz/schwarz.cu`):
- **kmax=16** aggregate cap (we use domain=16 — matches).
- Aggregation is **matrix-graph heavy-edge** (host `hem_partition`), NOT geometric.
  Ours is matrix-graph BFS — simpler, same class.
- Apply is the **Gram form `z = Gᵀ(G r)`** storing `G = L⁻¹` and reading ONE
  buffer both passes — the non-negotiable float32-SPD invariant. (We compute in
  f64 via Cholesky solve, equivalent; the Gram form matters for an f32 GPU port.)
- **Diagonal floor `1e-6·(tr/d) + 1e-8`** — we now match this (linsolve.rs).
- **levels=2 default: one Galerkin coarse level** `z = S₀r + Σ w·Cₗᵀ Sₗ(Cₗ r)`,
  purely additive. Galerkin coarsening `Aₗ = Cₗ Aₗ₋₁ Cₗᵀ` via sort+segmented-sum
  (use u64 keys: 32-bit overflows at nrow≳1.05M). **This has since been built**
  on both tiers (`engine-core/src/linsolve.rs::MultilevelSchwarz`,
  `engine-gpu/src/schwarz.rs`), opt-in as `--precond schwarz-ml`. It converges in
  fewer iterations but is not the default: block-Jacobi still wins wall-clock on
  the GPU, for the reason recorded in the `schwarz.rs` module header.
- All goldens ran block-Jacobi → the Schwarz path has NO golden checksum coverage;
  validate it by the invariants only.

## 4. Determinism = our advantage
A port CAN be made deterministic (canonical row order + segmented reduction
instead of float atomics), which the reference authors call "a real improvement".
**Our atomic-free per-vertex gather assembler already achieves this** — a genuine
edge over the reference for reproducible regression testing.

## 5. Per-module port backlog (specs in the capture's `docs/PORTING_*.md`)
Each is cited to `file:line` with a gotchas list. Priority by phase (§2):
- **solver_linalg** — matrix-free `M = A_dyn + B_fixed + C_diag` (block-CSR, never
  assembled), PCG with warm-start, block-Jacobi + Schwarz. Check-site order:
  converged→indefinite→non-SPD→nonfinite→cap.
- **energy_models** — BaraffWitkin (default shell, no SVD), ARAP/StVK/SNHk (SVD),
  dihedral bending (atan2, `BEND_SCALE=1.28e-5`), stitches, barriers
  (cubic/quad/log), **PSD projection via `max(0,·)` on closed-form eigenvalues**
  (no numerical eigendecomp). G-17: dropping a structural pattern pair silently
  loses a contribution — the most dangerous silent failure.
- **contact** — barrier + distance (no eps, exact-zero tests only; 4 fixed
  Gauss-Seidel iters for edge-edge; centroid subtraction for world-scale fp32),
  **ACCD** (`toi ∈ [0, line_search_max_t=1.25]`, stable `(d²−t²)/(√d²+t)` form,
  `toi_next!=toi` stall guard is the real fp32 exit), friction (lagged-λ).
- **bvh_primitives** — Karras LBVH (30-bit Morton, radix sort, 1-based child
  indices/0=leaf sentinel), build-once/refit-many, deactivation must follow every
  build+refit.
- **data_layout** — the FFI struct contract (we collapsed the two-process design,
  so this is reference-only, but `Model` enum order arap=0…pdrd=4 must match our
  `docs/CONFIG.md`).

## 6. 22 golden scenes (validation corpus)
`belt cards codim curtain domino drape fishingknot fitting five-twist friction
hang needle noodle plastic ribbon roller stack trampoline trapped twist woven
yarn` — cloth/rod/tet mixes, 512→204k verts, strain limits 0/1/5/10%. `drape`
(82k cloth over collider) is the reference/determinism scene. Inputs:
`goldens/<s>/input_bin/` (f64 vert, f32 vel, **u64** tri, AoS LE) + `param.toml`;
checkpoints `vert_{0,20,…}.bin` (f32[N][3] AoS, 6 frames); `frame_checksums.csv`
(per-frame sha + centroid/bbox/L2).
