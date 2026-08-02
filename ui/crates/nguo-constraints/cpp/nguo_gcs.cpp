// SPDX-License-Identifier: GPL-2.0-or-later
//
// Implementation of the nguo <-> planegcs shim. See nguo_gcs.h for the
// contract. Keep this file boring: it is the one place where a mistake is an
// unsafety in Rust rather than a compile error.

#include "nguo_gcs.h"

#include <deque>
#include <memory>
#include <new>
#include <vector>

#include "GCS.h"

namespace
{

enum CurveKind
{
    KindLine = 0,
    KindCircle = 1,
    KindArc = 2,
};

}  // namespace

struct GcsSystem
{
    GCS::System sys;

    // Pointer-stable: planegcs stores raw `double*` into this for the whole
    // lifetime of the system. std::deque never invalidates on push_back.
    std::deque<double> params;
    std::vector<bool> paramIsFree;
    GCS::VEC_pD freeParams;

    std::deque<GCS::Point> points;

    std::vector<std::unique_ptr<GCS::Curve>> curves;
    std::vector<int> curveKinds;

    // Next constraint tag. planegcs treats tag 0 as "unlabelled" and negative
    // tags as temporary (GCS::SpecialTag), so user tags start at 1.
    int32_t nextTag = 1;

    // Structure changed since the last declareUnknowns/initSolution.
    bool dirty = true;

    bool validParam(uint32_t p) const
    {
        return p < params.size();
    }
    bool validPoint(uint32_t p) const
    {
        return p < points.size();
    }
    bool validCurve(uint32_t c) const
    {
        return c < curves.size();
    }
    bool isKind(uint32_t c, int kind) const
    {
        return validCurve(c) && curveKinds[c] == kind;
    }

    GCS::Line& line(uint32_t c)
    {
        return *static_cast<GCS::Line*>(curves[c].get());
    }
    GCS::Circle& circle(uint32_t c)
    {
        return *static_cast<GCS::Circle*>(curves[c].get());
    }
    GCS::Arc& arc(uint32_t c)
    {
        return *static_cast<GCS::Arc*>(curves[c].get());
    }

    /// Declares unknowns and rebuilds the subsystem partition, but only if the
    /// structure changed. This is what makes a parameter tweak cheap: changing
    /// a driving value leaves `dirty` false, so a re-solve skips the QR-heavy
    /// initSolution/diagnose entirely.
    bool ensureInit(int algorithm)
    {
        if (!dirty) {
            return true;
        }
        freeParams.clear();
        for (size_t i = 0; i < params.size(); ++i) {
            if (paramIsFree[i]) {
                freeParams.push_back(&params[i]);
            }
        }
        sys.declareUnknowns(freeParams);
        sys.initSolution(static_cast<GCS::Algorithm>(algorithm));
        dirty = false;
        return true;
    }
};

namespace
{

/// Runs `f`, converting any escaping C++ exception into `onError`. Used only
/// on the entry points that can realistically throw (allocation, Eigen).
template<class F>
auto guarded(F&& f, decltype(f()) onError) noexcept -> decltype(f())
{
    try {
        return f();
    }
    catch (...) {
        return onError;
    }
}

size_t copyTags(const GCS::VEC_I& tags, int32_t* out, size_t cap)
{
    const size_t n = tags.size();
    if (out != nullptr) {
        const size_t m = n < cap ? n : cap;
        for (size_t i = 0; i < m; ++i) {
            out[i] = static_cast<int32_t>(tags[i]);
        }
    }
    return n;
}

}  // namespace

extern "C" {

GcsSystem* gcs_new(void)
{
    return new (std::nothrow) GcsSystem();
}

void gcs_free(GcsSystem* sys)
{
    delete sys;
}

// ---- parameters ---------------------------------------------------------

uint32_t gcs_add_param(GcsSystem* sys, double value, int is_free)
{
    return guarded(
        [&]() -> uint32_t {
            if (sys == nullptr) {
                return GCS_INVALID_ID;
            }
            const uint32_t id = static_cast<uint32_t>(sys->params.size());
            sys->params.push_back(value);
            sys->paramIsFree.push_back(is_free != 0);
            sys->dirty = true;
            return id;
        },
        GCS_INVALID_ID
    );
}

int gcs_set_param(GcsSystem* sys, uint32_t param, double value)
{
    if (sys == nullptr || !sys->validParam(param)) {
        return GCS_ERR_BAD_HANDLE;
    }
    sys->params[param] = value;
    return 0;
}

int gcs_get_param(const GcsSystem* sys, uint32_t param, double* out)
{
    if (sys == nullptr || out == nullptr || !sys->validParam(param)) {
        return GCS_ERR_BAD_HANDLE;
    }
    *out = sys->params[param];
    return 0;
}

uint32_t gcs_param_count(const GcsSystem* sys)
{
    return sys == nullptr ? 0u : static_cast<uint32_t>(sys->params.size());
}

// ---- geometry -----------------------------------------------------------

uint32_t gcs_add_point(GcsSystem* sys, uint32_t px, uint32_t py)
{
    return guarded(
        [&]() -> uint32_t {
            if (sys == nullptr || !sys->validParam(px) || !sys->validParam(py)) {
                return GCS_INVALID_ID;
            }
            const uint32_t id = static_cast<uint32_t>(sys->points.size());
            sys->points.emplace_back(&sys->params[px], &sys->params[py]);
            return id;
        },
        GCS_INVALID_ID
    );
}

int gcs_point_coords(const GcsSystem* sys, uint32_t point, double* x, double* y)
{
    if (sys == nullptr || x == nullptr || y == nullptr || !sys->validPoint(point)) {
        return GCS_ERR_BAD_HANDLE;
    }
    *x = *sys->points[point].x;
    *y = *sys->points[point].y;
    return 0;
}

uint32_t gcs_add_line(GcsSystem* sys, uint32_t p1, uint32_t p2)
{
    return guarded(
        [&]() -> uint32_t {
            if (sys == nullptr || !sys->validPoint(p1) || !sys->validPoint(p2)) {
                return GCS_INVALID_ID;
            }
            auto l = std::make_unique<GCS::Line>();
            l->p1 = sys->points[p1];
            l->p2 = sys->points[p2];
            const uint32_t id = static_cast<uint32_t>(sys->curves.size());
            sys->curves.push_back(std::move(l));
            sys->curveKinds.push_back(KindLine);
            return id;
        },
        GCS_INVALID_ID
    );
}

uint32_t gcs_add_circle(GcsSystem* sys, uint32_t centre, uint32_t radius)
{
    return guarded(
        [&]() -> uint32_t {
            if (sys == nullptr || !sys->validPoint(centre) || !sys->validParam(radius)) {
                return GCS_INVALID_ID;
            }
            auto c = std::make_unique<GCS::Circle>();
            c->center = sys->points[centre];
            c->rad = &sys->params[radius];
            const uint32_t id = static_cast<uint32_t>(sys->curves.size());
            sys->curves.push_back(std::move(c));
            sys->curveKinds.push_back(KindCircle);
            return id;
        },
        GCS_INVALID_ID
    );
}

uint32_t gcs_add_arc(
    GcsSystem* sys,
    uint32_t centre,
    uint32_t radius,
    uint32_t start_angle,
    uint32_t end_angle,
    uint32_t start_point,
    uint32_t end_point
)
{
    return guarded(
        [&]() -> uint32_t {
            if (sys == nullptr || !sys->validPoint(centre) || !sys->validParam(radius)
                || !sys->validParam(start_angle) || !sys->validParam(end_angle)
                || !sys->validPoint(start_point) || !sys->validPoint(end_point)) {
                return GCS_INVALID_ID;
            }
            auto a = std::make_unique<GCS::Arc>();
            a->center = sys->points[centre];
            a->rad = &sys->params[radius];
            a->startAngle = &sys->params[start_angle];
            a->endAngle = &sys->params[end_angle];
            a->start = sys->points[start_point];
            a->end = sys->points[end_point];
            const uint32_t id = static_cast<uint32_t>(sys->curves.size());
            GCS::Arc& ref = *a;
            sys->curves.push_back(std::move(a));
            sys->curveKinds.push_back(KindArc);
            // Ties start/end points to centre + radius + angles. Tag 0 marks
            // it as machinery rather than a user constraint.
            sys->sys.addConstraintArcRules(ref, 0);
            sys->dirty = true;
            return id;
        },
        GCS_INVALID_ID
    );
}

// ---- geometric constraints ----------------------------------------------

// Every constraint entry point follows the same shape: validate handles, take
// the next tag, call planegcs, mark the structure dirty, return the tag.
#define GCS_CONSTRAINT_BODY(check, call) \
    return guarded( \
        [&]() -> int32_t { \
            if (sys == nullptr) { \
                return GCS_ERR_BAD_HANDLE; \
            } \
            if (!(check)) { \
                return GCS_ERR_BAD_HANDLE; \
            } \
            const int32_t tag = sys->nextTag++; \
            (void)(call); \
            sys->dirty = true; \
            return tag; \
        }, \
        GCS_ERR_EXCEPTION \
    )

int32_t gcs_c_coincident(GcsSystem* sys, uint32_t p1, uint32_t p2)
{
    GCS_CONSTRAINT_BODY(
        sys->validPoint(p1) && sys->validPoint(p2),
        sys->sys.addConstraintP2PCoincident(sys->points[p1], sys->points[p2], tag)
    );
}

int32_t gcs_c_horizontal(GcsSystem* sys, uint32_t line)
{
    GCS_CONSTRAINT_BODY(
        sys->isKind(line, KindLine),
        sys->sys.addConstraintHorizontal(sys->line(line), tag)
    );
}

int32_t gcs_c_vertical(GcsSystem* sys, uint32_t line)
{
    GCS_CONSTRAINT_BODY(
        sys->isKind(line, KindLine),
        sys->sys.addConstraintVertical(sys->line(line), tag)
    );
}

int32_t gcs_c_parallel(GcsSystem* sys, uint32_t l1, uint32_t l2)
{
    GCS_CONSTRAINT_BODY(
        sys->isKind(l1, KindLine) && sys->isKind(l2, KindLine),
        sys->sys.addConstraintParallel(sys->line(l1), sys->line(l2), tag)
    );
}

int32_t gcs_c_perpendicular(GcsSystem* sys, uint32_t l1, uint32_t l2)
{
    GCS_CONSTRAINT_BODY(
        sys->isKind(l1, KindLine) && sys->isKind(l2, KindLine),
        sys->sys.addConstraintPerpendicular(sys->line(l1), sys->line(l2), tag)
    );
}

int32_t gcs_c_symmetric_line(GcsSystem* sys, uint32_t p1, uint32_t p2, uint32_t line)
{
    GCS_CONSTRAINT_BODY(
        sys->validPoint(p1) && sys->validPoint(p2) && sys->isKind(line, KindLine),
        sys->sys.addConstraintP2PSymmetric(sys->points[p1], sys->points[p2], sys->line(line), tag)
    );
}

int32_t gcs_c_symmetric_point(GcsSystem* sys, uint32_t p1, uint32_t p2, uint32_t centre)
{
    GCS_CONSTRAINT_BODY(
        sys->validPoint(p1) && sys->validPoint(p2) && sys->validPoint(centre),
        sys->sys
            .addConstraintP2PSymmetric(sys->points[p1], sys->points[p2], sys->points[centre], tag)
    );
}

int32_t gcs_c_equal_length(GcsSystem* sys, uint32_t l1, uint32_t l2)
{
    GCS_CONSTRAINT_BODY(
        sys->isKind(l1, KindLine) && sys->isKind(l2, KindLine),
        sys->sys.addConstraintEqualLength(sys->line(l1), sys->line(l2), tag)
    );
}

int32_t gcs_c_collinear(GcsSystem* sys, uint32_t l1, uint32_t l2)
{
    // Collinearity is parallelism plus a shared point. planegcs has no single
    // constraint for it; both solver constraints carry the same tag so DOF
    // diagnosis reports them as one user-level constraint.
    GCS_CONSTRAINT_BODY(sys->isKind(l1, KindLine) && sys->isKind(l2, KindLine), [&] {
        sys->sys.addConstraintParallel(sys->line(l1), sys->line(l2), tag);
        sys->sys.addConstraintPointOnLine(sys->line(l2).p1, sys->line(l1), tag);
        return 0;
    }());
}

int32_t gcs_c_tangent(GcsSystem* sys, uint32_t c1, uint32_t c2)
{
    return guarded(
        [&]() -> int32_t {
            if (sys == nullptr || !sys->validCurve(c1) || !sys->validCurve(c2)) {
                return GCS_ERR_BAD_HANDLE;
            }
            // Normalise so a line, if present, is first.
            uint32_t a = c1;
            uint32_t b = c2;
            if (sys->curveKinds[b] == KindLine && sys->curveKinds[a] != KindLine) {
                a = c2;
                b = c1;
            }
            const int ka = sys->curveKinds[a];
            const int kb = sys->curveKinds[b];
            const int32_t tag = sys->nextTag;

            if (ka == KindLine && kb == KindCircle) {
                // `ccw` selects which side of the line the curve sits on.
                // planegcs solves either way; false is the conventional default.
                sys->sys.addConstraintTangent(sys->line(a), sys->circle(b), false, tag);
            }
            else if (ka == KindLine && kb == KindArc) {
                sys->sys.addConstraintTangent(sys->line(a), sys->arc(b), false, tag);
            }
            else if (ka == KindCircle && kb == KindCircle) {
                sys->sys.addConstraintTangent(sys->circle(a), sys->circle(b), tag);
            }
            else if (ka == KindCircle && kb == KindArc) {
                sys->sys.addConstraintTangent(sys->circle(a), sys->arc(b), tag);
            }
            else if (ka == KindArc && kb == KindCircle) {
                sys->sys.addConstraintTangent(sys->circle(b), sys->arc(a), tag);
            }
            else if (ka == KindArc && kb == KindArc) {
                sys->sys.addConstraintTangent(sys->arc(a), sys->arc(b), tag);
            }
            else {
                // line-line tangency is meaningless.
                return GCS_ERR_BAD_KIND;
            }
            sys->nextTag++;
            sys->dirty = true;
            return tag;
        },
        GCS_ERR_EXCEPTION
    );
}

int32_t gcs_c_point_on_curve(GcsSystem* sys, uint32_t point, uint32_t curve)
{
    return guarded(
        [&]() -> int32_t {
            if (sys == nullptr || !sys->validPoint(point) || !sys->validCurve(curve)) {
                return GCS_ERR_BAD_HANDLE;
            }
            const int32_t tag = sys->nextTag;
            switch (sys->curveKinds[curve]) {
                case KindLine:
                    sys->sys.addConstraintPointOnLine(sys->points[point], sys->line(curve), tag);
                    break;
                case KindCircle:
                    sys->sys.addConstraintPointOnCircle(sys->points[point], sys->circle(curve), tag);
                    break;
                case KindArc:
                    sys->sys.addConstraintPointOnArc(sys->points[point], sys->arc(curve), tag);
                    break;
                default:
                    return GCS_ERR_BAD_KIND;
            }
            sys->nextTag++;
            sys->dirty = true;
            return tag;
        },
        GCS_ERR_EXCEPTION
    );
}

// ---- dimensional constraints --------------------------------------------

int32_t gcs_c_distance_pp(GcsSystem* sys, uint32_t p1, uint32_t p2, uint32_t dist)
{
    GCS_CONSTRAINT_BODY(
        sys->validPoint(p1) && sys->validPoint(p2) && sys->validParam(dist),
        sys->sys
            .addConstraintP2PDistance(sys->points[p1], sys->points[p2], &sys->params[dist], tag)
    );
}

int32_t gcs_c_distance_pl(GcsSystem* sys, uint32_t point, uint32_t line, uint32_t dist)
{
    GCS_CONSTRAINT_BODY(
        sys->validPoint(point) && sys->isKind(line, KindLine) && sys->validParam(dist),
        sys->sys.addConstraintP2LDistance(
            sys->points[point],
            sys->line(line),
            &sys->params[dist],
            false,
            tag
        )
    );
}

int32_t gcs_c_length(GcsSystem* sys, uint32_t line, uint32_t dist)
{
    // A line's length is the distance between its endpoints; planegcs has no
    // separate constraint and does not need one.
    GCS_CONSTRAINT_BODY(
        sys->isKind(line, KindLine) && sys->validParam(dist),
        sys->sys.addConstraintP2PDistance(
            sys->line(line).p1,
            sys->line(line).p2,
            &sys->params[dist],
            tag
        )
    );
}

int32_t gcs_c_angle(GcsSystem* sys, uint32_t l1, uint32_t l2, uint32_t angle)
{
    GCS_CONSTRAINT_BODY(
        sys->isKind(l1, KindLine) && sys->isKind(l2, KindLine) && sys->validParam(angle),
        sys->sys.addConstraintL2LAngle(sys->line(l1), sys->line(l2), &sys->params[angle], tag)
    );
}

int32_t gcs_c_radius(GcsSystem* sys, uint32_t curve, uint32_t radius)
{
    return guarded(
        [&]() -> int32_t {
            if (sys == nullptr || !sys->validCurve(curve) || !sys->validParam(radius)) {
                return GCS_ERR_BAD_HANDLE;
            }
            const int32_t tag = sys->nextTag;
            switch (sys->curveKinds[curve]) {
                case KindCircle:
                    sys->sys.addConstraintCircleRadius(sys->circle(curve), &sys->params[radius], tag);
                    break;
                case KindArc:
                    sys->sys.addConstraintArcRadius(sys->arc(curve), &sys->params[radius], tag);
                    break;
                default:
                    return GCS_ERR_BAD_KIND;
            }
            sys->nextTag++;
            sys->dirty = true;
            return tag;
        },
        GCS_ERR_EXCEPTION
    );
}

// ---- solving & diagnosis -------------------------------------------------

void gcs_set_max_iterations(GcsSystem* sys, int iters)
{
    if (sys != nullptr) {
        sys->sys.maxIter = iters;
        sys->sys.maxIterRedundant = iters;
    }
}

void gcs_set_convergence(GcsSystem* sys, double eps)
{
    if (sys != nullptr) {
        sys->sys.convergence = eps;
        sys->sys.convergenceRedundant = eps;
    }
}

int gcs_diagnose(GcsSystem* sys, int algorithm, int32_t* out_dof)
{
    return guarded(
        [&]() -> int {
            if (sys == nullptr || out_dof == nullptr) {
                return GCS_ERR_BAD_HANDLE;
            }
            // initSolution() runs diagnose() internally when the diagnosis is
            // stale, so this covers both the dirty and the clean case.
            sys->ensureInit(algorithm);
            sys->sys.diagnose(static_cast<GCS::Algorithm>(algorithm));
            *out_dof = static_cast<int32_t>(sys->sys.dofsNumber());
            return 0;
        },
        GCS_ERR_EXCEPTION
    );
}

int gcs_dof(const GcsSystem* sys)
{
    if (sys == nullptr) {
        return GCS_ERR_BAD_HANDLE;
    }
    // dofsNumber() is const-correct in GCS.h but the object is not, so this
    // mirrors the header's own `const` promise.
    return sys->sys.dofsNumber();
}

size_t gcs_conflicting(const GcsSystem* sys, int32_t* out, size_t cap)
{
    return guarded(
        [&]() -> size_t {
            if (sys == nullptr) {
                return 0;
            }
            GCS::VEC_I tags;
            sys->sys.getConflicting(tags);
            return copyTags(tags, out, cap);
        },
        0
    );
}

size_t gcs_redundant(const GcsSystem* sys, int32_t* out, size_t cap)
{
    return guarded(
        [&]() -> size_t {
            if (sys == nullptr) {
                return 0;
            }
            GCS::VEC_I tags;
            sys->sys.getRedundant(tags);
            return copyTags(tags, out, cap);
        },
        0
    );
}

size_t gcs_partially_redundant(const GcsSystem* sys, int32_t* out, size_t cap)
{
    return guarded(
        [&]() -> size_t {
            if (sys == nullptr) {
                return 0;
            }
            GCS::VEC_I tags;
            sys->sys.getPartiallyRedundant(tags);
            return copyTags(tags, out, cap);
        },
        0
    );
}

int gcs_solve(GcsSystem* sys, int algorithm)
{
    return guarded(
        [&]() -> int {
            if (sys == nullptr) {
                return GCS_STATUS_NOT_INITIALISED;
            }
            sys->ensureInit(algorithm);
            if (sys->freeParams.empty()) {
                // planegcs happily reports Success for a system with nothing
                // to move, even when its constraints are violated. Refuse
                // instead: "solved" is meaningless with zero unknowns, and a
                // vacuous Success is exactly the silent-garbage answer the
                // caller must never get.
                return GCS_STATUS_NOT_INITIALISED;
            }
            const int res = sys->sys.solve(true, static_cast<GCS::Algorithm>(algorithm));
            if (res == GCS::Success) {
                sys->sys.applySolution();
            }
            else {
                // Leave the parameters where they started rather than at
                // whatever the failed iteration reached; a caller that got an
                // error must never observe half-solved geometry or NaN.
                sys->sys.undoSolution();
            }
            return res;
        },
        GCS_STATUS_EXCEPTION
    );
}

}  // extern "C"
