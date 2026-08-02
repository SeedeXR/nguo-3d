<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# CONFIG.md — parameter parity with the reference

Every solver parameter and its **reference default**, so a ported scene
reproduces `ppf-contact-solver` behavior. `engine-api::SolverConfig` /
`Material` mirror these.

> **Every `file:line` citation on this page points into the upstream
> [`st-tech/ppf-contact-solver`][ppf] repository, not into this one.** Paths
> like `crates/ppf-cts-core/…`, `params.rs`, `solver.cu` and `data.hpp` do not
> exist here; they are provenance for the numbers, so each default can be
> checked against its source rather than taken on trust.

## Where the reference's defaults live
Single source of truth upstream: Rust `app_param()` / `object_param()` in
`crates/ppf-cts-core/src/datamodel/params.rs`. Its Python frontend just
re-exports these dicts — no separate Python defaults. Datamodel stores every
default as **f64** (`params.rs:67`); the solver consumes them via `SimArgs`
(`ppf-cts-solver/src/args.rs`) and the FFI `ParamSet` (`data.rs:738`,
`cpp/data.hpp:421`), which are **mostly f32**.

## CRITICAL: f64 vs f32 (top port risk — core WebGPU has no f64)
| Field | Type | Reference file:line |
|---|---|---|
| All datamodel param defaults | f64 | `params.rs:67` |
| `SimArgs.fps`, `ParamSet.time`, `StepResult.time` | **f64** | `args.rs:35`, `data.rs:741/781` |
| Every other `ParamSet` numeric field (dt, cg_tol, gravity…) | f32 | `data.rs:742-765` |
| Every per-element param/prop | f32 | `data.rs:112-404` |
| PCG/Newton accumulators (`err0`,`rz0`,`alpha`,`beta`,`pAp`,`l1norm`) | **double** | `solver.cu:949-994,1091-1129` |
| PDRD reduced 6×6 solve | **double** | `pdrd_rigid.hpp:655-689` |

→ Port strategy: keep spatial math f32; do the handful of reduction/decision
scalars with compensated f32 (Kahan) or return them to CPU f64 — which is what
`engine-gpu/src/gpu_implicit.rs` does.

## App-level parameters (`app_param()`)
### Time integration
| Key | Default | Units | Meaning | file:line |
|---|---|---|---|---|
| `dt` | 1e-3 | s | Base step (clamped ≤ 0.9999/fps) | `params.rs:215` |
| `fps` | 60.0 | 1/s | Output frame rate; caps substep dt | `params.rs:233` |
| `frames` | 300 | count | Max frames | `params.rs:252` |
| `fix-xz` | 0.0 | Y units | Fix xz if y>value (×world_scaling) | `params.rs:280` |
| `world-scaling` | 1.0 | ratio | Geometry scale at ingest | `params.rs:284` |

### Newton / line search
| Key | Default | Meaning | file:line |
|---|---|---|---|
| `min-newton-steps` | 0 | Lower bound on Newton steps (rec. 32 w/ static friction) | `params.rs:221` |
| `target-toi` | 0.25 | Accumulated-TOI termination threshold | `params.rs:223` |
| `max-dx` | 1.0 | Max search-direction magnitude | `params.rs:243` |
| `line-search-max-t` | 1.25 | Extends TOI for CCD | `params.rs:227` |
_No explicit Newton max-iter or residual tol exists — termination is `target-toi`
+ `min-newton-steps`._

### Linear solver (PCG)
| Key | Default | Meaning | file:line |
|---|---|---|---|
| `cg-max-iter` | 10000 | Diverged if exceeded | `params.rs:235` |
| `cg-tol` | 1e-3 | Relative tolerance | `params.rs:237` |
| `precond` | "block-jacobi" | block-jacobi \| schwarz | `params.rs:264` |
| `schwarz-levels` | 2 | Additive Schwarz levels | `params.rs:266` |
| `csrmat-max-nnz` | 10_000_000 | Pre-allocated contact-matrix entries | `params.rs:249` |

### Contact / friction / CCD
| Key | Default | Meaning | file:line |
|---|---|---|---|
| `disable-contact` | false | Skip contact detection | `params.rs:209` |
| `barrier` | "cubic" | cubic \| quad \| log | `params.rs:260` |
| `constraint-ghat` | 1e-3 | BC-barrier gap | `args.rs:33` |
| `constraint-tol` | 0.01 | Moving-constraint CCD tol multiplier | `params.rs:231` |
| `eiganalysis-eps` | 1e-2 | Near-equal singular-value eps | `params.rs:245` |
| `friction-eps` | 1e-5 | **This is eps_v** (friction regularization) | `params.rs:247` |
| `friction-mode` | "min" | min \| max \| mean combine | `params.rs:262` |
| `ccd-reduction` | 0.01 | `eps = ccd_reduction·(√dist − offset)` | `params.rs:239`, `accd.hpp:44` |
| `ccd-max-iter` | 4096 | Max ACCD iterations | `params.rs:241` |

_Not found as standalone params: `eps_v` (= `friction-eps`), `ccd-eps`
(derived), `newton-tol/max-iter`, explicit `barrier-stiffness`._

### Aerodynamics / gravity
| Key | Default | file:line |
|---|---|---|
| `gravity` | [0, −9.8, 0] m/s² | `params.rs:274` |
| `wind` | [0,0,0] | `params.rs:276` |
| `air-density` | 1e-3 kg/m³ | `params.rs:270` |
| `air-friction` | 0.2 | `params.rs:225` |
| `isotropic-air-friction` | 0.0 | `params.rs:272` |

## Per-object material (`object_param(kind)`, defaults `params.rs:300-309`)
`(model, young_mod, density, offset, bend)`:
- **Tri** (shell): `baraff-witkin, 1000, 1.0, 0.0, 10.0`
- **Tet**: `arap, 500, 100, 0.0, 1.0`
- **Rod**: `arap, 1e4, 1.0, 1e-3, 0.0`
- **Pdrd**: `pdrd, 0, 100, 0, 0`; **Points/sand**: `arap, 0, 100, 0, 0`

| Key | Default (Tri) | Meaning | file:line |
|---|---|---|---|
| `model` | baraff-witkin | arap=0,stvk=1,baraff-witkin=2,snhk=3,pdrd=4 | `elastic_model.rs:23` |
| `young-mod` | 1000 | Lamé μ=young/(2(1+ν)), λ=young·ν/((1+ν)(1−2ν)) | `builder.rs:1274` |
| `poiss-rat` | 0.35 | Poisson ratio | `params.rs:318` |
| `density` | 1.0 | kg/m² (shell) | `params.rs:314` |
| `bend` | 10.0 | Bending stiffness | `params.rs:320` |
| `strain-limit` | 0.0 | Max tensile strain (0=off) | `params.rs:338` |
| `contact-gap` | 1e-3 | Barrier activation `dhat` (pairwise mean) | `params.rs:334` |
| `contact-offset` | 0.0 | Skin thickness (pairwise sum) | `params.rs:336` |
| `friction` | 0.0 | Coulomb μ | `params.rs:340` |
| `shrink-x/y`, `shrink` | 1.0 | Anisotropic/isotropic rest scale | `params.rs:328-332` |

## Hardcoded constants (not user-tunable)
| Constant | Value | file:line |
|---|---|---|
| `MAX_SUBSTEP_FRAME_FRACTION` | 0.9999 | `builder.rs:254` |
| `MAX_INTERSECTION_RECORDS` | 256 | `cpp/data.hpp:563` |
| `MAX_COLLISION_WINDOWS` | 8 | `main.cu:45` |
| `BVH_STACK_CAP` | 64 | `core/kernels/constants.rs:18` |
| PDRD 6×6 Cholesky eps | 1e-8 | `pdrd_rigid.hpp:672` |

[ppf]: https://github.com/st-tech/ppf-contact-solver
