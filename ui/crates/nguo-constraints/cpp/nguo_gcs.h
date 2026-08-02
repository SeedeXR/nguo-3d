/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * nguo_gcs.h — the entire C surface between Rust and FreeCAD's planegcs.
 *
 * Deliberately narrow. planegcs' `GCS::System` exposes ~120 public methods and
 * a deep C++ class hierarchy (Curve -> Circle -> Arc, Ellipse, Hyperbola,
 * Parabola, BSpline). Binding all of that is how FFI layers become
 * unmaintainable. This shim exposes only the constraint set nguo-3d's pattern
 * layer needs (../../../docs/CONSTRAINTS.md §4) plus DOF diagnosis (§5).
 *
 * Ownership model
 * ---------------
 * The shim owns everything: the parameter storage (pointer-stable, because
 * planegcs holds raw `double*` into it), the geometry objects, and the
 * `GCS::System` (which in turn owns every `GCS::Constraint` — see
 * `System::clear()` calling `deleteAllContent(clist)` in GCS.cpp).
 * Rust owns exactly one thing: the `GcsSystem*`.
 *
 * Handle model
 * ------------
 * Every id is a dense 0-based index into a per-kind arena. Handles are never
 * recycled and geometry is never removed, so a stale handle is impossible
 * short of using a handle from a different system. Out-of-range handles are
 * rejected, they do not fault.
 *
 * Error model
 * -----------
 * - Functions returning a handle return GCS_INVALID_ID on bad input.
 * - Functions returning a constraint tag return a negative GCS_ERR_* on
 *   failure; success is a tag >= 1.
 * - `gcs_solve` returns a GCS_STATUS_* value.
 * - No function may throw across the boundary. Every entry point that can
 *   allocate or run Eigen wraps its body in a catch-all; the rest are pure
 *   bounds checks and pointer reads that cannot throw.
 */

#ifndef NGUO_GCS_H
#define NGUO_GCS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GCS_INVALID_ID 0xFFFFFFFFu

/* Negative results from constraint constructors. */
#define GCS_ERR_BAD_HANDLE (-1)
#define GCS_ERR_BAD_KIND (-2)
#define GCS_ERR_EXCEPTION (-3)

/* Mirrors GCS::SolveStatus in GCS.h (verified at commit 64bc717). */
#define GCS_STATUS_SUCCESS 0
#define GCS_STATUS_CONVERGED 1
#define GCS_STATUS_FAILED 2
#define GCS_STATUS_INVALID_SOLUTION 3
/* Ours: the shim could not even run the solver. */
#define GCS_STATUS_NOT_INITIALISED 4
#define GCS_STATUS_EXCEPTION 5

/* Mirrors GCS::Algorithm in GCS.h. */
#define GCS_ALG_BFGS 0
#define GCS_ALG_LEVENBERG_MARQUARDT 1
#define GCS_ALG_DOGLEG 2

typedef struct GcsSystem GcsSystem;

GcsSystem* gcs_new(void);
void gcs_free(GcsSystem* sys);

/* ---- parameters ------------------------------------------------------- */

/* `is_free != 0` makes this an unknown the solver may move. A fixed parameter
 * is how a dimensional constraint carries a literal value or a binding to a
 * named parameter / body measurement (§9.4): set it, re-solve, geometry moves.
 */
uint32_t gcs_add_param(GcsSystem* sys, double value, int is_free);
int gcs_set_param(GcsSystem* sys, uint32_t param, double value);
int gcs_get_param(const GcsSystem* sys, uint32_t param, double* out);
uint32_t gcs_param_count(const GcsSystem* sys);

/* ---- geometry --------------------------------------------------------- */

uint32_t gcs_add_point(GcsSystem* sys, uint32_t px, uint32_t py);
int gcs_point_coords(const GcsSystem* sys, uint32_t point, double* x, double* y);

/* All three live in one curve arena; ids are not interchangeable between
 * arenas but are between curve kinds where the constraint allows it. */
uint32_t gcs_add_line(GcsSystem* sys, uint32_t p1, uint32_t p2);
uint32_t gcs_add_circle(GcsSystem* sys, uint32_t centre, uint32_t radius);
/* Adds GCS::addConstraintArcRules under tag 0, as FreeCAD does, so that
 * centre/radius/angles and the start/end points stay mutually consistent. */
uint32_t gcs_add_arc(
    GcsSystem* sys,
    uint32_t centre,
    uint32_t radius,
    uint32_t start_angle,
    uint32_t end_angle,
    uint32_t start_point,
    uint32_t end_point
);

/* ---- geometric constraints (§9.2) ------------------------------------- */

int32_t gcs_c_coincident(GcsSystem* sys, uint32_t p1, uint32_t p2);
int32_t gcs_c_horizontal(GcsSystem* sys, uint32_t line);
int32_t gcs_c_vertical(GcsSystem* sys, uint32_t line);
int32_t gcs_c_parallel(GcsSystem* sys, uint32_t l1, uint32_t l2);
int32_t gcs_c_perpendicular(GcsSystem* sys, uint32_t l1, uint32_t l2);
int32_t gcs_c_tangent(GcsSystem* sys, uint32_t c1, uint32_t c2);
int32_t gcs_c_symmetric_line(GcsSystem* sys, uint32_t p1, uint32_t p2, uint32_t line);
int32_t gcs_c_symmetric_point(GcsSystem* sys, uint32_t p1, uint32_t p2, uint32_t centre);
int32_t gcs_c_equal_length(GcsSystem* sys, uint32_t l1, uint32_t l2);
/* Two planegcs constraints (parallel + point-on-line) sharing one tag. */
int32_t gcs_c_collinear(GcsSystem* sys, uint32_t l1, uint32_t l2);
int32_t gcs_c_point_on_curve(GcsSystem* sys, uint32_t point, uint32_t curve);

/* ---- dimensional constraints (§9.2, §9.4) ----------------------------- */

int32_t gcs_c_distance_pp(GcsSystem* sys, uint32_t p1, uint32_t p2, uint32_t dist);
int32_t gcs_c_distance_pl(GcsSystem* sys, uint32_t point, uint32_t line, uint32_t dist);
int32_t gcs_c_length(GcsSystem* sys, uint32_t line, uint32_t dist);
int32_t gcs_c_angle(GcsSystem* sys, uint32_t l1, uint32_t l2, uint32_t angle);
int32_t gcs_c_radius(GcsSystem* sys, uint32_t curve, uint32_t radius);

/* ---- solving & diagnosis ---------------------------------------------- */

void gcs_set_max_iterations(GcsSystem* sys, int iters);
void gcs_set_convergence(GcsSystem* sys, double eps);

/* Declares unknowns and builds the subsystem partition if the structure
 * changed since the last call, then runs planegcs' QR-based DOF analysis.
 *
 * Writes the remaining degrees of freedom to *out_dof and returns 0, or
 * returns a negative GCS_ERR_* and leaves *out_dof untouched. The DOF is
 * signed on purpose: planegcs reports an over-determined system as
 * `parameters - independent constraints`, which is negative. That is why the
 * value cannot be the return code. */
int gcs_diagnose(GcsSystem* sys, int algorithm, int32_t* out_dof);

/* The DOF from the last diagnosis, or -1 if there is none. Ambiguous with a
 * genuine -1, so callers must track whether a diagnosis has run. */
int gcs_dof(const GcsSystem* sys);

/* Each returns the number of tags available and writes at most `cap` of them.
 * Call with cap == 0 to size the buffer. */
size_t gcs_conflicting(const GcsSystem* sys, int32_t* out, size_t cap);
size_t gcs_redundant(const GcsSystem* sys, int32_t* out, size_t cap);
size_t gcs_partially_redundant(const GcsSystem* sys, int32_t* out, size_t cap);

/* Solves and writes the result back into the parameters. Structure changes
 * since the last diagnose are picked up automatically; a bare
 * `gcs_set_param` + `gcs_solve` does NOT re-initialise, which is the
 * incremental path for dragging a parameter slider. */
int gcs_solve(GcsSystem* sys, int algorithm);

#ifdef __cplusplus
}
#endif

#endif /* NGUO_GCS_H */
