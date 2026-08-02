<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# KERNELS.md — GPU compute porting backlog

Enumerates every GPU kernel in the reference `ppf-contact-solver`
(`crates/ppf-cts-solver/src/cpp/`) that has to exist in WGSL here, and which
ones do. Ported one at a time, each verified against the CPU implementation.
All `file:line` citations are into the **reference** repo, not this one.

Status legend:

- **☑ ported + verified** — WGSL in `crates/engine-gpu/src/shaders/`, a Rust
  module that drives it, and a `gpu_*_matches_cpu` oracle test asserting it
  equals `engine-core`. The test name is given so the claim is checkable.
- **◐ partial** — some of the family is on the GPU, the rest is not.
- **☐ not started** — CPU only. This is not a bug; the CPU tier is correct,
  and these are the throughput work that remains.

Shipped WGSL today: `assemble`, `bending`, `bvh_aabb`, `bvh_traverse`,
`collider_contact`, `gather4`, `gpu_step`, `karras`, `membrane`,
`membrane_gather`, `morton`, `pcg_ops`, `pcg_reduce`, `radix`, `saxpy`,
`scan`, `schwarz`, `shell`, `svd`, `sym_eig`, `tet`, `tree_reduce`,
`unified_gather`. (`subgroup_reduce.wgsl` is present but dead — naga does not
yet accept the `subgroups` enable-extension; see `WGPU_PIN.md`.)

## Dispatch model
- `indexed_apply<F>` (`utility/dispatcher.cu:15`): the one generic elementwise
  kernel — `if (idx < n) fn(idx)`, block size 256. Every `DISPATCH_START(n)[…]`
  body is a distinct grid-flat kernel, one thread per `i`. In WGSL this becomes
  `@compute @workgroup_size(64/128)` with a bounds guard.
- `DISPATCH_START/END` synchronize (default stream); `DISPATCH_QUEUE_*` do not
  (PCG fast loop / Schwarz apply). The WGSL port replicates the per-iteration
  dependency chain (SpMV → reduce → update → reduce → axpy), not the CUDA-graph
  capture mechanism.

## Solver step sequence (`main/main.cu::advance`)
1. Rebuild LBVH (face/edge/vertex) broad-phase + deactivate windowed AABBs — `:308-333`.
2. Per-vertex velocity `(curr-prev)/prev_dt`, max-velocity/stretch reductions — `:354-459`.
3. Inertial target `y = x1 + (x1-x0)·(dt/prev_dt) + dt²·g` (pinned snap) — `:461-483`.
4. Newton `while(true)` — `:552`:
   assemble RHS+matrix (torque/momentum/elastic/stitch/strain/contact/constraint)
   → PCG solve `:753` → max-dx rescale → CCD + strain line search (min toi) →
   apply `eval_x` → PDRD rigidify → convergence when `toi_advanced≥target_toi &&
   step≥min_newton_steps`.
5. Finalize: `check_intersection`, commit `prev=curr; curr=eval_x`, SAND integrate,
   plasticity updates — `:1052-1154`.

## Linear solver / CCD (confirmed)
- **PCG, matrix-free operator** `M = A_dyn(DynCSRMat) + B_fixed(FixedCSRMat) +
  C_diag(Mat3) + D·I`, applied by walking the CSR blocks in `solver::apply`
  (`solver/solver.cu:79`) — no external sparse lib, **but the per-Newton Hessian
  IS assembled into the custom `DynCSRMat`/`FixedCSRMat` containers**. The port
  needs: CSR assembly kernels + custom SpMV + reductions + AXPY + preconditioner
  (not merely "three WGSL kernels").
- Preconditioners in the reference: **3×3 block-Jacobi** by default
  (`solver.cu:154`), **multilevel additive Schwarz** opt-in (`schwarz::apply`,
  `solver.cu:1286`). Both are ported here; block-Jacobi is likewise the default.
- **CCD = ACCD** (additive, conservative), `contact/accd.hpp:38-70`; primitives
  point-triangle/point-edge/point-point/edge-edge `accd.hpp:122/147/170/191`.
  Strain-limit line searches use bisection (`strainlimiting.cu:207-285`).

## Kernel families (backlog)

### Vector / reduction primitives (`kernels/`) — the PCG's building blocks
- ☑ `vec_ops.cu` — fill/copy/add_scaled/combine/scalar_div (`:15-86`). No atomics. → `pcg_ops.wgsl`.
- ☑ `reduce.cu` — `inner_product`, `sum/min/max_array` (`:16-428`). Ported as a **Kahan tree reduce** (`tree_reduce.wgsl`, `reduce.rs::gpu_reduce_matches_cpu`), *not* as a warp shuffle: core WebGPU has no float atomics and naga does not yet accept `subgroups`, and the fixed-shape tree is what makes the result deterministic.
- ☑ `exclusive_scan.cu` — block scan (`:17-188`) → `scan.wgsl`, `scan.rs::gpu_exclusive_scan_matches_cpu`.
- ☑ `radix_sort.cu` — histogram + multisplit scatter (`:22-161`) → `radix.wgsl`, `radix.rs::gpu_radix_sort_matches_cpu` and `…_pairs_carries_payload`.

### PCG (`solver/solver.cu`)
- ☑ `apply` matrix-free SpMV `:79` → `pcg.rs::gpu_spmv_matches_cpu`. Our SpMV walks the assembled block-CSR built by `assemble.wgsl`; there is no atomic-scatter variant, so the `cg_apply_dot_sym_kernel` (`:439`) / `_walk_kernel` (`:523`) split does not arise.
- ☑ fused update / reduce / update-xr / beta (`:261/404/316/366/590`) → `pcg_ops.wgsl` + `pcg_reduce.wgsl`; `pcg.rs::pcg_solves_known_system`, `pcg_zero_rhs_is_zero`.
- ☑ block-Jacobi `inv_diag` build `:1230` + `precond` `:154` → `gpu_step.wgsl` diagonal extraction + `pcg.rs`.

### Elasticity / bending / inertia (`energy/energy.cu`) — per-element, atomic embed
Ported as **gather**, not atomic embed: worker `i` is the only writer of row
`i` (`unified_gather.wgsl`, `membrane_gather.wgsl`, `gather4.wgsl`), which is
what makes the GPU tier bit-reproducible.

- ☑ `embed_momentum_force_hessian` (inertia) `:608` → `gpu_step.wgsl` inertial prediction; covered by `unified_solver.rs::gpu_unified_step_matches_cpu`. Air damper / pull / torque / `fix_xz` are **not** on the GPU.
- ◐ `embed_elastic_force_hessian`: shell membrane `:656` ☑ (`membrane_asm.rs::gpu_membrane_matches_cpu`, `…_gather_matches_cpu`), tet `:667` ☑ (`tet_asm.rs::gpu_tet_elements_match_cpu`, `tet_assembler.rs::gpu_tet_assembly_matches_cpu`), hinge bending `:678` ☑ (`bending.rs::gpu_bending_matches_cpu`, `bending_asm.rs::gpu_bending_assembly_matches_cpu`), isotropic shells `shell.wgsl` ☑ (`shell_asm.rs::gpu_shell_elements_match_cpu`). Rod bend `:635` and rod stretch `:645` ☐.
- ☐ `embed_stitch_force_hessian` `:696`; `compute_torque_groups` `:551`.
- ☑ `eigenanalysis.cu` (SPD-clamped analytic Hessian projection) `:18-184` → `sym_eig.wgsl`, `sym_eig.rs::gpu_sym_eig_matches_cpu`.
- ☑ `utility.cu` SVD (`svd3x2`, `svd3x3`) `:59-166` → `svd.wgsl`, `svd.rs::gpu_svd_matches_cpu`.

### Strain limiting (`strainlimiting/strainlimiting.cu`)
- ☐ shell `embed_strainlimiting_force_hessian` (barrier on max singular value) `:46`; rod `:116`. CPU only (`engine-core/src/strain_limit.rs`).
- ☐ `line_search` shell `:236` / `rod_line_search` `:181` (bisection).
- ★ Extend to **anisotropic warp/weft** limits — new work, past parity.

### Contact barrier (`contact/contact.cu` — largest subsystem)
- ☐ `embed_contact_force_hessian` (self-contact; point/edge BVH query, two-stage dry-count/fill, cubic barrier, dynamic stiffness) `:1224`. Reference uses atomics into CSR + force. **Must be ported as gather/coloring, not atomic scatter** — atomic float accumulation is what makes the reference irreproducible (`GAUGE_FINDINGS.md` §1).
- ◐ `embed_constraint_force_hessian` `:1522`. The **analytic sphere/half-space** barrier is ported — `collider_contact.wgsl`, a post-pass over `assemble.wgsl` that adds the gradient and PSD Hessian into each free vertex's own diagonal block (no atomics); `gpu_implicit.rs::gpu_resident_contact_matches_cpu`. Collision-mesh M2C/C2M/EE is ☐.
- ☐ `line_search` (ACCD per primitive class, min toi) `:2082`.
- ☐ `check_intersection` (edge-tri, edge-edge overlap testers) `:2618` — the audit-build invariant scan.
- ☐ `barrier.cu` cubic/quad/log energy/gradient/curvature `:13-37`; `compute_stiffness<N>` `:49`.
- ☐ `contact/distance.hpp`, `aabb.hpp`, `intersect_core.hpp` — device leaf math.

### Broad phase (`lbvh/lbvh.cu`) — Morton LBVH
- ☑ centroids + morton codes `:120-155`, `:253` → `morton.wgsl`, `morton.rs::gpu_morton_matches_cpu`, `gpu_morton_then_radix_sorts`.
- ☑ Karras internal-node build `:357` → `karras.wgsl`, `karras.rs::gpu_karras_matches_cpu`.
- ☑ leaf AABBs + bottom-up merge `:623-826`, refit `:1032-1097` → `bvh_aabb.wgsl`, `bvh_aabb.rs::gpu_node_aabb_matches_cpu`.
- ☑ traversal/query → `bvh_traverse.wgsl`, `bvh_traverse.rs::gpu_lbvh_is_superset_of_brute_force` (a superset assertion: the GPU broad phase may over-report, never under-report).

### Sparse containers (`csrmat/csrmat.cu`)
- ◐ The **fixed** structure is ported: `assemble.wgsl` builds the block-CSR by
  per-row gather, so there is no atomic accumulate and no dedup pass
  (`assemble.rs::gpu_assembly_matches_cpu`, `unified_asm.rs::gpu_unified_matches_cpu`,
  `…_with_shells_matches_cpu`). The reference's `DynCSRMat` rebuild/dedup/transpose
  path `:131-289` exists to absorb *contact* fill-in and is ☐ along with GPU contact.

### Preconditioner (`schwarz/schwarz.cu`)
- ☑ Two-level Galerkin additive Schwarz (`z = S₀r + w·C₁ᵀS₁(C₁r)`) is ported on
  both tiers and selectable with `--precond schwarz-ml`: `schwarz.wgsl`,
  `schwarz.rs::gpu_schwarz_apply_matches_cpu`, `gpu_schwarz_pcg_solves_known_system`,
  `pcg.rs::gpu_schwarz_implicit_matches_block`. The CPU aggregation and dense
  Cholesky factorization stay on the host; only the apply is on-device.
  **Block-Jacobi remains the default** — matching the reference default, and
  because the GPU Schwarz apply measured *slower* than block-Jacobi on the
  900-vertex cloth despite converging in fewer iterations (the per-domain dense
  triangular solve is serial with one thread per domain). Numbers and the
  upgrade path are in the `schwarz.rs` module header.

### Plasticity (`plasticity/plasticity.cu`) — post-solve, mutate rest state
- ☐ face `:20` / tet `:87` / hinge `:146` / rod-bend `:189` plasticity creep.

### PDRD rigid + SAND grains (defer — advanced features, not core cloth)
- ☐ PDRD (`energy/model/pdrd_rigid.hpp`): fit_rigid, reconstruct, reduced 6×6 rigid PCG, rigidify. Host-side double-precision 6×6 Cholesky `:653`.
- ☐ SAND (`energy/model/sand_rigid.hpp`): condense/recover/integrate grains.

## f32 caveats to carry into every kernel
- Reference computes PCG/Newton accumulators (`err0`, `rz0`, `alpha`, `pAp`,
  `l1norm`) and the PDRD 6×6 solve in **double**; core WebGPU has no f64. Port
  these reductions with compensated (Kahan/two-sum) f32 or return the handful of
  scalars to the CPU in f64 — which is what `gpu_implicit.rs` does today. See
  the "f64 vs f32" table in [`CONFIG.md`](CONFIG.md).
