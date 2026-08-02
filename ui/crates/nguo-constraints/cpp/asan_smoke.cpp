// SPDX-License-Identifier: GPL-2.0-or-later
//
// Memory check for the shim, run under the address sanitiser.
//
// This exists because the pinned stable toolchain has no `-Zsanitizer`, so
// `cargo test` cannot be sanitised. All the manual memory management lives on
// this side of the boundary anyway — the Rust side owns one `NonNull` and a
// `Drop` — so exercising the shim directly covers the real risk.
//
// Build and run (from the crate root):
//
//   clang++ -std=c++20 -g -O1 -fsanitize=address,undefined \
//     -fno-omit-frame-pointer -DEIGEN_NO_DEBUG \
//     -I../../../third_party/planegcs/compat \
//     -I../../../third_party/planegcs/vendor/eigen \
//     -I../../../third_party/planegcs/upstream/planegcs -Icpp \
//     cpp/asan_smoke.cpp cpp/nguo_gcs.cpp \
//     ../../../third_party/planegcs/upstream/planegcs/{Geo,qp_eq,SubSystem,Constraints,GCS}.cpp \
//     -o /tmp/asan_smoke && /tmp/asan_smoke
//
// Exit code 0 and no sanitiser report means clean.

#include "nguo_gcs.h"

#include <cstdio>
#include <cstdlib>

namespace
{

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

/// Builds a sketch touching every geometry kind and every constraint the shim
/// exposes, solves it, and reads everything back.
void round_trip()
{
    GcsSystem* s = gcs_new();
    expect(s != nullptr, "gcs_new");

    const uint32_t ox = gcs_add_param(s, 0.0, 0);
    const uint32_t oy = gcs_add_param(s, 0.0, 0);
    const uint32_t origin = gcs_add_point(s, ox, oy);

    const uint32_t ex = gcs_add_param(s, 10.0, 0);
    const uint32_t ey = gcs_add_param(s, 0.0, 0);
    const uint32_t east = gcs_add_point(s, ex, ey);

    const uint32_t base = gcs_add_line(s, origin, east);

    const uint32_t ax = gcs_add_param(s, 1.0, 1);
    const uint32_t ay = gcs_add_param(s, 1.0, 1);
    const uint32_t a = gcs_add_point(s, ax, ay);
    const uint32_t bx = gcs_add_param(s, 5.0, 1);
    const uint32_t by = gcs_add_param(s, 2.0, 1);
    const uint32_t b = gcs_add_point(s, bx, by);
    const uint32_t arm = gcs_add_line(s, a, b);

    const uint32_t cx = gcs_add_param(s, 0.0, 1);
    const uint32_t cy = gcs_add_param(s, 8.0, 1);
    const uint32_t ccentre = gcs_add_point(s, cx, cy);
    const uint32_t rad = gcs_add_param(s, 2.0, 1);
    const uint32_t circle = gcs_add_circle(s, ccentre, rad);

    const uint32_t arad = gcs_add_param(s, 3.0, 1);
    const uint32_t a0 = gcs_add_param(s, 0.0, 1);
    const uint32_t a1 = gcs_add_param(s, 1.5, 1);
    const uint32_t asx = gcs_add_param(s, 3.0, 1);
    const uint32_t asy = gcs_add_param(s, 0.0, 1);
    const uint32_t aex = gcs_add_param(s, 0.0, 1);
    const uint32_t aey = gcs_add_param(s, 3.0, 1);
    const uint32_t acentre = gcs_add_point(s, ox, oy);
    const uint32_t astart = gcs_add_point(s, asx, asy);
    const uint32_t aend = gcs_add_point(s, aex, aey);
    const uint32_t arc = gcs_add_arc(s, acentre, arad, a0, a1, astart, aend);

    const uint32_t d = gcs_add_param(s, 4.0, 0);
    const uint32_t ang = gcs_add_param(s, 0.7, 0);

    expect(gcs_c_coincident(s, a, a) > 0, "coincident");
    expect(gcs_c_horizontal(s, base) > 0, "horizontal");
    expect(gcs_c_vertical(s, arm) > 0, "vertical");
    expect(gcs_c_parallel(s, base, arm) > 0, "parallel");
    expect(gcs_c_perpendicular(s, base, arm) > 0, "perpendicular");
    expect(gcs_c_tangent(s, base, circle) > 0, "tangent line/circle");
    expect(gcs_c_tangent(s, circle, arc) > 0, "tangent circle/arc");
    expect(gcs_c_symmetric_line(s, a, b, base) > 0, "symmetric about line");
    expect(gcs_c_symmetric_point(s, a, b, origin) > 0, "symmetric about point");
    expect(gcs_c_equal_length(s, base, arm) > 0, "equal length");
    expect(gcs_c_collinear(s, base, arm) > 0, "collinear");
    expect(gcs_c_point_on_curve(s, a, base) > 0, "point on line");
    expect(gcs_c_point_on_curve(s, a, circle) > 0, "point on circle");
    expect(gcs_c_point_on_curve(s, a, arc) > 0, "point on arc");
    expect(gcs_c_distance_pp(s, a, b, d) > 0, "distance pp");
    expect(gcs_c_distance_pl(s, a, base, d) > 0, "distance pl");
    expect(gcs_c_length(s, arm, d) > 0, "length");
    expect(gcs_c_angle(s, base, arm, ang) > 0, "angle");
    expect(gcs_c_radius(s, circle, d) > 0, "radius circle");
    expect(gcs_c_radius(s, arc, d) > 0, "radius arc");

    // Rejections must not touch memory.
    expect(gcs_c_radius(s, base, d) == GCS_ERR_BAD_KIND, "radius on line rejected");
    expect(gcs_c_tangent(s, base, arm) == GCS_ERR_BAD_KIND, "line/line tangency rejected");
    expect(gcs_c_horizontal(s, 9999) == GCS_ERR_BAD_HANDLE, "bad curve rejected");
    expect(gcs_c_coincident(s, 9999, 0) == GCS_ERR_BAD_HANDLE, "bad point rejected");
    expect(gcs_add_point(s, 9999, 0) == GCS_INVALID_ID, "bad param rejected");
    expect(gcs_set_param(s, 9999, 1.0) == GCS_ERR_BAD_HANDLE, "bad set rejected");

    int32_t dof = 0;
    expect(gcs_diagnose(s, GCS_ALG_DOGLEG, &dof) == 0, "diagnose");

    // Both the sizing call and the filling call, plus a deliberately short
    // buffer, to prove the shim honours `cap`.
    int32_t tags[4];
    (void)gcs_conflicting(s, nullptr, 0);
    (void)gcs_conflicting(s, tags, 4);
    (void)gcs_redundant(s, tags, 1);
    (void)gcs_partially_redundant(s, tags, 0);

    gcs_set_max_iterations(s, 64);
    gcs_set_convergence(s, 1e-10);
    (void)gcs_solve(s, GCS_ALG_DOGLEG);

    // Incremental path: mutate a driving parameter and re-solve repeatedly.
    for (int i = 0; i < 20; ++i) {
        expect(gcs_set_param(s, d, 4.0 + 0.1 * i) == 0, "set_param");
        (void)gcs_solve(s, GCS_ALG_DOGLEG);
    }

    double x = 0.0;
    double y = 0.0;
    expect(gcs_point_coords(s, b, &x, &y) == 0, "point_coords");
    expect(gcs_get_param(s, rad, &x) == 0, "get_param");
    expect(gcs_point_coords(s, 9999, &x, &y) == GCS_ERR_BAD_HANDLE, "bad point read rejected");

    gcs_free(s);
}

}  // namespace

int main()
{
    // Enough cycles that a leak or a dangling pointer has to show itself.
    for (int i = 0; i < 300; ++i) {
        round_trip();
    }
    // A system that is created and dropped without ever being solved, and one
    // that is solved without geometry: both are shapes real callers produce.
    gcs_free(gcs_new());
    GcsSystem* empty = gcs_new();
    (void)gcs_solve(empty, GCS_ALG_DOGLEG);
    gcs_free(empty);
    gcs_free(nullptr);

    std::puts("asan_smoke: ok");
    return 0;
}
