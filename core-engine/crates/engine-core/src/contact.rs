// SPDX-License-Identifier: GPL-2.0-or-later
//! Contact against analytic colliders (half-space + sphere) via an IPC-style
//! barrier. This is the reference's analytic-constraint path
//! (`embed_vertex_constraint_force_hessian`, sphere/floor — docs/KERNELS.md);
//! self-collision (needing the LBVH broad phase + narrow phase) layers on later.
//!
//! Barrier: `b(d) = −κ (d − d̂)² ln(d/d̂)` for `0 < d < d̂`, else 0 (Li et al.
//! 2020, "Incremental Potential Contact"). It is C² on the active interval,
//! vanishes with zero slope at `d̂`, and → +∞ as `d → 0`, which guarantees
//! non-penetration without CCD (the reference gets guarantees from barrier + CCD
//! together; ACCD is a later addition here). Derivatives are hand-derived and
//! checked by tests, not transcribed.
//!
//! ponytail: sphere Hessian drops the curvature term `b'·∂n/∂x` and SPD-projects
//! (`max(b'',0)·nnᵀ`); exact + ppf's cubic barrier are refinements.

use glam::Vec3;

/// An analytic collider. Cloth is kept on the `+normal` side / outside the sphere.
#[derive(Clone, Copy, Debug)]
pub enum Collider {
    /// Points must satisfy `(x − point)·normal ≥ 0`. `normal` must be unit.
    HalfSpace {
        point: Vec3,
        normal: Vec3,
    },
    Sphere {
        center: Vec3,
        radius: f32,
    },
}

impl Collider {
    /// Signed distance `d` (≥0 outside) and the unit gradient `∂d/∂x`.
    pub fn distance(&self, x: Vec3) -> (f32, Vec3) {
        match *self {
            Collider::HalfSpace { point, normal } => ((x - point).dot(normal), normal),
            Collider::Sphere { center, radius } => {
                let v = x - center;
                let l = v.length();
                if l < 1.0e-9 {
                    (-radius, Vec3::Y)
                } else {
                    (l - radius, v / l)
                }
            }
        }
    }
}

/// Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision
/// Detection §5.1.5). Handles the 7 Voronoi regions (face, 3 edges, 3 vertices).
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // Delegates to the weighted form: the branch conditions were duplicated verbatim in both
    // functions, so a fix to one could silently miss the other.
    closest_point_on_triangle_bary(p, a, b, c).0
}

/// Closest point on triangle `(a,b,c)` to `p`, together with its BARYCENTRIC WEIGHTS
/// `(wa, wb, wc)` — the same branch structure as [`closest_point_on_triangle`], which
/// discards them.
///
/// The weights are what make an analytic distance gradient possible: with
/// `q = wa*a + wb*b + wc*c` and `n = (p - q)/|p - q|`, the distance gradient is
/// `∂d/∂p = n` and `∂d/∂v = -w_v * n` for each triangle vertex. Vertices that are not part
/// of the closest feature simply carry weight 0. Verified against central finite
/// differences in `analytic_gradient_tests`.
pub fn closest_point_on_triangle_bary(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}

/// How two participants' friction coefficients combine at a contact.
///
/// Ports the reference's `combine_friction` (`contact/contact.cu:37-47`), selected per scene by
/// `param.toml: friction_mode`. The reference's `default:` case is `Min`, which is why that is
/// the default here.
///
/// This is NOT a detail: **14 of the 22 gauge scenes ship `"max"`** and only 8 ship `"min"`, so
/// hardwiring `min` (as this solver did until S118j) silently under-frictions the majority of
/// them. With `max`, a collider declaring friction 0 leaves the cloth's own coefficient intact;
/// with `min` the same collider makes the contact frictionless.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrictionMode {
    #[default]
    Min,
    Max,
    Mean,
}

impl FrictionMode {
    /// Combine two coefficients under this mode.
    pub fn combine(self, a: f32, b: f32) -> f32 {
        match self {
            FrictionMode::Max => a.max(b),
            FrictionMode::Mean => 0.5 * (a + b),
            FrictionMode::Min => a.min(b),
        }
    }

    /// Parse `param.toml`'s spelling; anything unrecognised falls back to the reference's
    /// `default:` branch rather than guessing.
    pub fn from_str_or_default(s: &str) -> Self {
        match s.trim().trim_matches('"') {
            "max" => FrictionMode::Max,
            "mean" => FrictionMode::Mean,
            _ => FrictionMode::Min,
        }
    }

    /// The identity element for folding a set of coefficients under this mode: combining it
    /// with anything returns that value. `Mean` has none, so folding uses the arithmetic mean
    /// of the participants directly (see `fold`).
    fn identity(self) -> f32 {
        match self {
            FrictionMode::Max => f32::NEG_INFINITY,
            FrictionMode::Mean => 0.0,
            FrictionMode::Min => f32::INFINITY,
        }
    }

    /// Combine a whole stencil's coefficients. The reference resolves a contact through TWO
    /// elements' params, which the capture does not expose, so a stencil's participants are
    /// folded instead — under the scene's own mode.
    pub fn fold(self, vals: impl IntoIterator<Item = f32>) -> f32 {
        let mut acc = self.identity();
        let mut n = 0usize;
        for v in vals {
            acc = match self {
                FrictionMode::Mean => acc + v,
                _ => self.combine(acc, v),
            };
            n += 1;
        }
        if n == 0 {
            return 0.0;
        }
        match self {
            FrictionMode::Mean => acc / n as f32,
            _ => acc,
        }
    }
}

/// Point-edge barycentric coefficients `(w0, w1)` for the projection of `p` onto edge
/// `(e0, e1)`, UNCLAMPED. Ports `distance::point_edge_distance_coeff`
/// (`contact/distance.hpp:24-35`); a degenerate edge splits the weight evenly, as there.
fn point_edge_coeff(p: Vec3, e0: Vec3, e1: Vec3) -> [f32; 2] {
    let r = e1 - e0;
    let d = r.dot(r);
    if d > 0.0 {
        let t = r.dot(p - e0) / d;
        [1.0 - t, t]
    } else {
        [0.5, 0.5]
    }
}

/// The reference's edge-edge WITNESS weights `(wa0, wa1, wb0, wb1)`, each pair summing to 1,
/// so the closest points are `wa0*ea0 + wa1*ea1` and `wb0*eb0 + wb1*eb1`.
///
/// Ports `distance::edge_edge_distance_coeff_unclassified`
/// (`contact/distance.hpp:181-238`), which the reference's contact path actually calls
/// (`contact/contact.cu:942, 2404`). The selection rule matters and is NOT the same as
/// clamping the interior solution:
///
/// 1. Solve the 2x2 interior system. Use it only if every coefficient lands in `[0, 1]`.
///    A ZERO determinant (parallel edges) falls back to weights `(1, 0, ·, ·)` — all of
///    edge A's weight on its first endpoint (`distance.hpp:86-104`).
/// 2. Otherwise enumerate the four point-edge candidates (each endpoint of A against edge B,
///    each endpoint of B against edge A), clamp each to an endpoint when its coefficient
///    leaves `[0, 1]`, and take the one with the SMALLEST distance.
///
/// Clamping the interior parameters instead — the textbook shortcut, and what this module did
/// before — picks a different witness in exactly the configurations where the interior
/// solution is invalid, which is most resting contact. Distances computed about the centroid
/// for fp32 conditioning, as the reference does.
pub fn edge_edge_witness(ea0: Vec3, ea1: Vec3, eb0: Vec3, eb1: Vec3) -> [f32; 4] {
    let cog = 0.25 * (ea0 + ea1 + eb0 + eb1);
    let (a0, a1, b0v, b1v) = (ea0 - cog, ea1 - cog, eb0 - cog, eb1 - cog);
    // Squared witness distance, evaluated about the centroid for fp32 conditioning (the
    // reference does the same).
    let dist2 = |c: &[f32; 4]| {
        let x0 = c[0] * a0 + c[1] * a1;
        let x1 = c[2] * b0v + c[3] * b1v;
        (x1 - x0).length_squared()
    };

    // --- `edge_edge_distance_coeff` (classified), `distance.hpp:75-107`.
    // The baseline candidate is edge A's FIRST endpoint against edge B; the interior solution
    // replaces it only when the 2x2 system is solvable AND actually closer.
    let pe = point_edge_coeff(ea0, eb0, eb1);
    let mut classified = [1.0, 0.0, pe[0], pe[1]];
    let r0 = ea1 - ea0;
    let r1 = eb1 - eb0;
    let a00 = r0.dot(r0);
    let a01 = -r0.dot(r1);
    let a11 = r1.dot(r1);
    let rhs = eb0 - ea0;
    let b0 = r0.dot(rhs);
    let b1 = -r1.dot(rhs);
    let det = a00 * a11 - a01 * a01;
    if det != 0.0 {
        let sp = (a11 * b0 - a01 * b1) / det;
        let tp = (a00 * b1 - a01 * b0) / det;
        let direct = [1.0 - sp, sp, 1.0 - tp, tp];
        if dist2(&direct) < dist2(&classified) {
            classified = direct;
        }
    }

    // --- `edge_edge_distance_coeff_unclassified`, `distance.hpp:181-238`.
    // Accept the classified witness only if it is a genuine convex combination. NOTE the
    // range check applies to the PARALLEL fallback too: for parallel, non-overlapping edges
    // that fallback projects outside edge B, and the enumeration below is what finds the real
    // closest pair. Returning early on the parallel branch (an earlier attempt here) gets
    // that case wrong.
    if classified.iter().all(|&x| (0.0..=1.0).contains(&x)) {
        return classified;
    }
    let snap = |c: [f32; 2]| -> [f32; 2] {
        if c[0] < 0.0 {
            [0.0, 1.0]
        } else if c[0] > 1.0 {
            [1.0, 0.0]
        } else {
            c
        }
    };
    let c1 = snap(point_edge_coeff(ea0, eb0, eb1));
    let c2 = snap(point_edge_coeff(ea1, eb0, eb1));
    let c3 = snap(point_edge_coeff(eb0, ea0, ea1));
    let c4 = snap(point_edge_coeff(eb1, ea0, ea1));
    let types = [
        [1.0, 0.0, c1[0], c1[1]],
        [0.0, 1.0, c2[0], c2[1]],
        [c3[0], c3[1], 1.0, 0.0],
        [c4[0], c4[1], 0.0, 1.0],
    ];
    let mut best = types[0];
    let mut best_d = f32::INFINITY;
    for c in types.iter() {
        let d = dist2(c);
        if d < best_d {
            best_d = d;
            best = *c;
        }
    }
    best
}

/// Unsigned distance from a point to a triangle.
pub fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (p - closest_point_on_triangle(p, a, b, c)).length()
}

/// Does segment `(e0,e1)` pierce triangle `(v0,v1,v2)`? Ports the reference's
/// intersection predicate `edge_triangle_intersect` (`intersect_core.hpp:61-81`,
/// PORTING_contact.md §6.1): the two endpoints must lie on **strictly opposite**
/// sides of the triangle plane (`s1·s2 < 0`), and the plane-crossing point must
/// fall inside the triangle. Everything uses coordinate differences (translation-
/// invariant) for fp32 conditioning. Coplanar/touching → false (matches the
/// device predicate). This is the single "surface crossing" event that both a
/// vertex-through-face and an edge-through-edge produce.
pub fn edge_triangle_intersect(e0: Vec3, e1: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> bool {
    let d1 = v1 - v0;
    let d2 = v2 - v0;
    let a0 = e0 - v0;
    let a1 = e1 - v0;
    let n = d1.cross(d2);
    let s1 = a0.dot(n);
    let s2 = a1.dot(n);
    if s1 * s2 < 0.0 {
        let t = s1 / (s1 - s2); // where the segment crosses the plane, in (0,1)
        let r = a0 * (1.0 - t) + a1 * t; // crossing point relative to v0
                                         // Inside test: r = u·d1 + v·d2 with barycentrics (1-u-v, u, v) ∈ [0,1].
        let a = d1.dot(d1);
        let b = d1.dot(d2);
        let c = d2.dot(d2);
        let det = a * c - b * b;
        if det == 0.0 {
            return false; // degenerate (zero-area) triangle contains nothing
        }
        let p = d1.dot(r);
        let q = d2.dot(r);
        let u = (c * p - b * q) / det;
        let v = (a * q - b * p) / det;
        u >= 0.0 && v >= 0.0 && u + v <= 1.0
    } else {
        false
    }
}

/// Additive CCD (conservative advancement) time-of-impact for a set of moving
/// points `x0` with per-point displacement `dx`, whose closest-feature *squared*
/// distance is `square_dist(pts)`. Returns the largest parametric fraction
/// `toi ∈ (0, max_t]` of `x(t) = x0 + t·dx` at which the primitive is guaranteed
/// not to have closed to `offset`; `max_t` means "no impact within the horizon".
///
/// Ports the reference `ccd_helper` (`accd.hpp:38-70`, PORTING_contact.md §4.2).
/// `u_max = max_{i<j}‖dx_i−dx_j‖` is the Lipschitz bound on the distance's rate
/// of change — the crucial quantity: a per-pair max over ALL unordered pairs, not
/// an additive per-vertex estimate (which under-bounds it and lets fast pairs
/// tunnel). Each step advances `toi += gap/u_max`, re-evaluating the *true*
/// distance so the bound stays both conservative and tight. `d_minus_target` uses
/// the numerically stable `(d²−target²)/(√d²+target)` form. Translation-centering
/// (`accd.hpp:14-24`) is skipped — it only affects fp32 rounding, not the result.
#[allow(clippy::too_many_arguments)]
pub fn accd_toi(
    x0: &[Vec3],
    dx: &[Vec3],
    offset: f32,
    max_t: f32,
    reduction: f32,
    max_iter: u32,
    square_dist: impl Fn(&[Vec3]) -> f32,
) -> f32 {
    // Lipschitz bound: max relative displacement over all unordered pairs.
    let mut u_max = 0.0f32;
    for i in 0..dx.len() {
        for j in (i + 1)..dx.len() {
            u_max = u_max.max((dx[i] - dx[j]).length());
        }
    }
    if u_max < 1.0e-12 {
        return max_t; // no relative motion → no impact
    }
    let inv_u_max = 1.0 / u_max;
    let d0 = square_dist(x0).max(0.0).sqrt();
    let eps = reduction * (d0 - offset);
    if eps <= 0.0 {
        return max_t; // already within offset; the barrier resolves it
    }
    let target = eps + offset;
    let eps_sqr = eps * eps;
    let mut toi = 0.0f32;
    let mut moved = x0.to_vec();
    for _ in 0..max_iter {
        for k in 0..x0.len() {
            moved[k] = x0[k] + dx[k] * toi;
        }
        let d2 = square_dist(&moved).max(0.0);
        // Stable form of (√d² − target); avoids cancellation near contact.
        let dmt = (d2 - target * target) / (d2.sqrt() + target);
        if (max_t - toi) * u_max < dmt - eps {
            return max_t; // (A) early accept: gap can't be closed within the horizon
        }
        if toi > 0.0 && dmt * dmt < eps_sqr {
            break; // (B) converged
        }
        let toi_next = toi + dmt * inv_u_max; // (C) conservative advance
        if toi_next != toi {
            toi = toi_next; // (D)
        } else {
            break;
        }
        if toi > max_t {
            return max_t; // (E) advanced past the horizon → no impact
        }
    }
    toi.max(0.0)
}

/// Closest distance between segment `p1q1` and segment `p2q2` (Ericson §5.1.9,
/// the clamped closest-point pair). This is the `_unclassified` edge-edge distance
/// the reference uses for ACCD (`distance.hpp:179-237`): the true closest distance
/// with endpoints clamped, robust to parallel/degenerate edges. Needed so a vertex
/// (or edge) sliding through a shared triangle EDGE is caught by CCD — the
/// through-edge crossing that vertex-triangle distance misses.
pub fn segment_segment_distance(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> f32 {
    let d1 = q1 - p1; // segment 1 direction
    let d2 = q2 - p2; // segment 2 direction
    let r = p1 - p2;
    let a = d1.dot(d1); // squared length of seg1
    let e = d2.dot(d2); // squared length of seg2
    let f = d2.dot(r);
    const EPS: f32 = 1.0e-12;
    let (s, t);
    if a <= EPS && e <= EPS {
        return (p1 - p2).length(); // both segments are points
    }
    if a <= EPS {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0); // seg1 is a point
    } else {
        let c = d1.dot(r);
        if e <= EPS {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0); // seg2 is a point
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b; // ≥ 0
            let s0 = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0 // parallel: any s works; pick 0
            };
            let t0 = (b * s0 + f) / e;
            // If t is out of [0,1], clamp it and recompute s for that t.
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                s = s0;
                t = t0;
            }
        }
    }
    let c1 = p1 + d1 * s;
    let c2 = p2 + d2 * t;
    (c1 - c2).length()
}

/// Segment-segment distance and its gradient w.r.t. the 4 endpoints
/// `[p1,q1,p2,q2]`, by central finite differences (same policy as the point-tri
/// barrier). For the edge-edge IPC barrier.
pub fn segment_segment_distance_grad(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (f32, [Vec3; 4]) {
    // ANALYTIC, the segment analogue of `point_tri_distance_grad`. With the witness weights
    // `w` (each pair summing to 1) the closest points are `ca = w0*p1 + w1*q1` and
    // `cb = w2*p2 + w3*q2`, and with `n = (ca − cb)/d`:
    //
    //   ∂d/∂p1 = w0·n,   ∂d/∂q1 = w1·n,   ∂d/∂p2 = −w2·n,   ∂d/∂q2 = −w3·n.
    //
    // Distance AND gradient come from ONE witness so they cannot disagree, and the witness is
    // the REFERENCE's (`edge_edge_witness`, porting
    // `edge_edge_distance_coeff_unclassified`) rather than the clamped interior solution.
    // That distinction is not cosmetic: clamping picks a different witness wherever the
    // interior solution is invalid, which is most resting contact. Replaces 24
    // `segment_segment_distance` evaluations per stencil per Newton iteration with one.
    let w = edge_edge_witness(p1, q1, p2, q2);
    let ca = w[0] * p1 + w[1] * q1;
    let cb = w[2] * p2 + w[3] * q2;
    let e = ca - cb;
    let d = e.length();
    if d < 1.0e-12 {
        return (d, [Vec3::ZERO; 4]);
    }
    let n = e / d;
    (d, [w[0] * n, w[1] * n, -w[2] * n, -w[3] * n])
}

/// Oriented (signed) point-triangle gap: `(p − closest)·n̂`, `n̂` = normalized
/// `(b−a)×(c−a)`. Positive on the `+n̂` side, negative on the `−n̂` side; the
/// magnitude is the normal component of the separation (= the true distance when
/// `p` projects inside the triangle). Unlike the unsigned distance, this does NOT
/// flip sign when `p` crosses the plane — the key to a barrier that repels a
/// crossed vertex back to its side (see `signed_barrier`).
pub fn oriented_point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let n = (b - a).cross(c - a);
    let ln = n.length();
    if ln < 1.0e-12 {
        return 0.0;
    }
    (p - closest_point_on_triangle(p, a, b, c)).dot(n / ln)
}

/// [`oriented_point_triangle`] and its gradient w.r.t. `[p,a,b,c]` (central FD).
pub fn oriented_point_tri_grad(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, [Vec3; 4]) {
    let pts = [p, a, b, c];
    let d0 = oriented_point_triangle(p, a, b, c);
    let scale = (p.length() + a.length() + b.length() + c.length()).max(1.0);
    let h = 1.0e-4 * scale;
    let inv = 1.0 / (2.0 * h);
    let mut g = [Vec3::ZERO; 4];
    for (vi, gv) in g.iter_mut().enumerate() {
        for co in 0..3 {
            let mut pp = pts;
            let mut pm = pts;
            pp[vi][co] += h;
            pm[vi][co] -= h;
            let dp = oriented_point_triangle(pp[0], pp[1], pp[2], pp[3]);
            let dm = oriented_point_triangle(pm[0], pm[1], pm[2], pm[3]);
            gv[co] = (dp - dm) * inv;
        }
    }
    (d0, g)
}

/// One-sided **cubic** contact barrier on a SIGNED gap `g` (which MAY be negative
/// = penetration): `b(g) = κ(d̂−g)³` for `g < d̂`, else `None`. Returns
/// `(energy, b', b'')`. Unlike the IPC log barrier (`barrier`, undefined for
/// `g ≤ 0`), this is finite and strongly repulsive for `g ≤ 0`, so a vertex that
/// has slipped to the wrong side is pushed BACK (penetration recovery). `b'' ≥ 0`
/// in the active range → PSD Gauss-Newton Hessian. Used with a home-side sign for
/// self-collision (the log barrier's sign-flip is the violent-slam tunneling bug).
pub fn signed_barrier(g: f32, dhat: f32, kappa: f32) -> Option<(f32, f32, f32)> {
    if g >= dhat {
        return None;
    }
    let t = dhat - g; // > 0 in the active range
    Some((kappa * t * t * t, -3.0 * kappa * t * t, 6.0 * kappa * t))
}

/// The reference's **cubic** barrier, ported verbatim from
/// `crates/ppf-cts-solver/src/cpp/barrier/cubic.hpp` (`cubic::energy`/`gradient`/
/// `curvature`, Apache-2.0, in the parent reference repo). Returns
/// `(energy, dE/dg, d²E/dg²)`, or `None` when inactive (`g − offset ≥ ghat`).
///
/// With `g ← g − offset` and `y = g − ghat`:
/// `E = −2y³/(3·ghat)`, `E' = −2y²/ghat`, `E'' = 4(1 − g/ghat) = −4y/ghat`.
///
/// Unlike the IPC log barrier ([`barrier`], which diverges as `g → 0` and is
/// undefined for `g ≤ 0`), this is *finite everywhere* and its curvature is
/// bounded — the property that lets the reference take large steps without the
/// log barrier's stiffness blow-up. `E' < 0` (repulsive) and `E'' > 0` (PSD) over
/// the whole active range. Scale by [`rayleigh_stiffness`] the way the reference
/// does (`contact.cu`: `f = stiff_k · edge_gradient`, `H = stiff_k · edge_hessian`).
pub fn cubic_barrier(g: f32, ghat: f32, offset: f32) -> Option<(f32, f32, f32)> {
    let g = g - offset;
    let y = g - ghat;
    if y >= 0.0 {
        return None;
    }
    Some((
        -2.0 * (y * y * y) / (3.0 * ghat),
        -2.0 * y * y / ghat,
        4.0 * (1.0 - g / ghat),
    ))
}

/// Smoothed Coulomb **friction**, ported from the reference `Friction`
/// (`energy/model/friction.hpp`, Apache-2.0; used in `contact/contact.cu` where its
/// gradient/Hessian are ADDED to the barrier's).
///
/// With `n` the contact normal, `f_c` the barrier force, `dx` the relative slip and
/// `P = I − nnᵀ` the tangential projector:
/// * `contact = −n·f_c` — the magnitude of the normal contact force,
/// * `λ = μ·contact / max(ε, ‖P·dx‖)`,
/// * gradient `= λ·(P·dx)`, Hessian `= λ·P`.
///
/// The `max(ε, …)` is the regularization: as slip → 0 the force would otherwise have an
/// undefined direction, so `ε` (`friction_eps`) caps `λ` and makes it differentiable. In
/// the sliding regime `‖P·dx‖ > ε` the force magnitude is `μ·contact` — the Coulomb
/// bound — and it opposes the slip direction.
///
/// **Honest note on the Hessian**: `λ·P` treats `λ` as constant, but `λ` itself depends
/// on `dx`, so this is the reference's *lagged / semi-implicit* Hessian, NOT the exact
/// Jacobian of the gradient. That choice keeps it PSD (`P` is a projector and `λ ≥ 0`),
/// which is what the Newton solve needs; the exact Jacobian is indefinite. Tests assert
/// PSD-ness and the Coulomb bound, and deliberately do NOT assert Hessian == d(grad)/d(dx).
pub fn friction(
    force_contact: Vec3,
    dx: Vec3,
    normal: Vec3,
    mu: f32,
    min_dx: f32,
) -> (Vec3, [[f32; 3]; 3]) {
    if mu <= 0.0 {
        return (Vec3::ZERO, [[0.0; 3]; 3]);
    }
    let contact = -normal.dot(force_contact);
    // Tangential projection P·dx = dx − (n·dx)n.
    let pdx = dx - normal.dot(dx) * normal;
    let eps = min_dx.max(1.0e-12);
    let denom = pdx.length().max(eps);
    let lambda = mu * contact / denom;
    if lambda <= 0.0 {
        // Separating (or zero) contact force: no friction to apply.
        return (Vec3::ZERO, [[0.0; 3]; 3]);
    }
    let grad = pdx * lambda;
    // λ·P with P = I − nnᵀ.
    let mut hess = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let p = if r == c { 1.0 } else { 0.0 } - normal[r] * normal[c];
            hess[r][c] = lambda * p;
        }
    }
    (grad, hess)
}

/// Friction for a 4-vertex self-collision stencil (vertex–triangle or edge–edge).
///
/// The reference embeds friction over a contact's stencil using its **proximity
/// weights** `w_a`: the gap vector is `ex = Σ w_a x_a`, the slip is `Σ w_a (x_a − x⁰_a)`,
/// and the friction gradient/Hessian are scattered as `w_a·f` and `w_a w_b·H`.
///
/// Those weights are recoverable from the distance gradient rather than needing a
/// separate barycentric computation: for a distance function, `∂d/∂x_a = w_a·n̂` with
/// `n̂` the unit contact normal, so `n̂ = ê(∂d/∂x₀)` and `w_a = (∂d/∂x_a)·n̂`. (For
/// vertex–triangle that recovers `(1, −b₀, −b₁, −b₂)`.) That is why this takes the
/// gradient `g` it already has rather than re-deriving the geometry.
///
/// Returns `(per-vertex gradient, 12×12 row-major Hessian)` to add to the barrier's.
pub fn stencil_friction(
    g: &[Vec3; 4],
    dx: &[Vec3; 4],
    barrier_grad: f32,
    mu: f32,
    min_dx: f32,
) -> ([Vec3; 4], [f32; 144]) {
    let zero = ([Vec3::ZERO; 4], [0.0f32; 144]);
    if mu <= 0.0 {
        return zero;
    }
    let nlen = g[0].length();
    if nlen < 1.0e-12 {
        return zero;
    }
    let n = g[0] / nlen;
    // Proximity weights and the weighted slip.
    let mut w = [0.0f32; 4];
    let mut slip = Vec3::ZERO;
    for a in 0..4 {
        w[a] = g[a].dot(n);
        slip += dx[a] * w[a];
    }
    // The barrier force along the normal for this contact (barrier_grad is dE/dd).
    let f_c = n * barrier_grad;
    let (fg, fh) = friction(f_c, slip, n, mu, min_dx);
    if fg == Vec3::ZERO {
        return zero;
    }
    let mut grad = [Vec3::ZERO; 4];
    for a in 0..4 {
        grad[a] = fg * w[a];
    }
    let mut hess = [0.0f32; 144];
    for a in 0..4 {
        for b in 0..4 {
            let wab = w[a] * w[b];
            if wab == 0.0 {
                continue;
            }
            for r in 0..3 {
                for c in 0..3 {
                    hess[(3 * a + r) * 12 + (3 * b + c)] = wab * fh[r][c];
                }
            }
        }
    }
    (grad, hess)
}

/// Which contact barrier the solver uses. `Log` (default) is the IPC log barrier
/// ([`barrier`]) — the engine's long-standing behavior; `Cubic` is the reference's
/// [`cubic_barrier`], finite and bounded-curvature through zero gap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BarrierKind {
    #[default]
    Log,
    Cubic,
}

/// Evaluate the selected barrier at distance `d`, scaled by `kappa`. Returns
/// `(energy, dE/dd, d²E/dd²)`, or `None` when inactive. The reference scales the
/// cubic barrier by the stiffness at the call site (`contact.cu`:
/// `f = stiff_k·edge_gradient`), which is exactly what `kappa` does here; the log
/// barrier already folds `kappa` in. Offset is 0 for our distance-based stencils.
pub fn barrier_of(kind: BarrierKind, d: f32, dhat: f32, kappa: f32) -> Option<(f32, f32, f32)> {
    match kind {
        BarrierKind::Log => barrier(d, dhat, kappa),
        BarrierKind::Cubic => {
            cubic_barrier(d, dhat, 0.0).map(|(e, g, h)| (kappa * e, kappa * g, kappa * h))
        }
    }
}

/// The reference's **elasticity-inclusive dynamic contact stiffness** — a port of
/// `barrier::compute_stiffness` (`crates/ppf-cts-solver/src/cpp/barrier/barrier.cu`,
/// Apache-2.0, parent reference repo). This is the piece [`dynamic_stiffness`]
/// deliberately dropped: the *elastic* Hessian blocks, which make the barrier as
/// stiff as the material it is resisting.
///
/// The local contact Hessian over the stencil's `N` participants is
/// `local[ii][jj] = elastic[ii][jj] + δ_ij·(m_ii/g²)·I`, and the stiffness is its
/// **Rayleigh quotient** along the weighted contact direction
/// `w = normalize([value[0]·e, …, value[N-1]·e])`, i.e. `κ = (local·w)·w`.
///
/// `g = ‖e‖ − offset`. Static-side participants (`mass == 0`) would collapse their
/// own diagonal term and leave the contact feeling inertia from one side only
/// (unstable at stiff collisions), so — exactly as the reference does — a zero mass
/// is replaced by the largest non-zero mass in the stencil.
///
/// `elastic` holds the `N×N` row-major 3×3 blocks (`elastic[ii*N + jj]`); pass
/// zeros for a purely inertial stiffness. Returns `≥ 0` for a PSD `elastic`.
pub fn rayleigh_stiffness(
    elastic: &[[f32; 9]],
    mass: &[f32],
    weights: &[f32],
    e: Vec3,
    offset: f32,
) -> f32 {
    let n = mass.len();
    assert_eq!(
        weights.len(),
        n,
        "rayleigh_stiffness: weights/mass length mismatch"
    );
    assert_eq!(
        elastic.len(),
        n * n,
        "rayleigh_stiffness: elastic must be N*N blocks"
    );
    let g = e.length() - offset;
    if g <= 0.0 || n == 0 {
        return 0.0;
    }
    let sqr_x = g * g;
    // Static side mimics its dynamic partner's inertia (reference comment).
    let m_max = mass.iter().copied().fold(0.0f32, f32::max);

    // w = normalize([value[ii] * e]) over the 3N-vector.
    let mut w = vec![0.0f32; 3 * n];
    for ii in 0..n {
        let v = weights[ii] * e;
        w[3 * ii] = v.x;
        w[3 * ii + 1] = v.y;
        w[3 * ii + 2] = v.z;
    }
    let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return 0.0;
    }
    for x in w.iter_mut() {
        *x /= norm;
    }

    // (local_hess · w) · w, accumulating block by block.
    let mut acc = 0.0f32;
    for ii in 0..n {
        let m_ii = if mass[ii] > 0.0 { mass[ii] } else { m_max };
        for jj in 0..n {
            let blk = &elastic[ii * n + jj];
            for r in 0..3 {
                let mut row = 0.0f32;
                for c in 0..3 {
                    let mut v = blk[3 * r + c];
                    if ii == jj && r == c {
                        v += m_ii / sqr_x;
                    }
                    row += v * w[3 * jj + c];
                }
                acc += w[3 * ii + r] * row;
            }
        }
    }
    acc
}

/// Point-triangle distance and its gradient w.r.t. the 4 points `[p,a,b,c]`, by
/// central finite differences (same zero-hallucination policy as bending).
pub fn point_tri_distance_grad(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, [Vec3; 4]) {
    // ANALYTIC. With `q` the closest point on the triangle and `w` its barycentric weights,
    // `d = |p − q|` and `n = (p − q)/d`, so
    //
    //   ∂d/∂p = n,        ∂d/∂v = −w_v · n   for each triangle vertex v.
    //
    // Exact almost everywhere: the closest FEATURE is locally constant, so within a branch
    // the weights do not vary to first order. This replaced 24 `point_triangle_distance`
    // evaluations per stencil per Newton iteration (4 vertices x 3 coordinates x 2 sides) with
    // one, and removes the finite-difference step-size compromise entirely. Agreement with
    // central differences is asserted in `analytic_gradient_tests`.
    let (q, w) = closest_point_on_triangle_bary(p, a, b, c);
    let e = p - q;
    let d = e.length();
    // Coincident: the distance is not differentiable here and the barrier does not activate
    // on a zero gap either (`cubic_barrier` needs `g − offset < ghat` with `g ≥ 0`). The
    // finite-difference version returned noise; zero is at least well defined.
    if d < 1.0e-12 {
        return (d, [Vec3::ZERO; 4]);
    }
    let n = e / d;
    (d, [n, -w[0] * n, -w[1] * n, -w[2] * n])
}

/// Per-contact **dynamic** stiffness — a pragmatic port of the reference
/// `compute_stiffness` (`barrier.cu:49-86`, PORTING_contact.md §5.1): the local
/// contact Hessian is the elastic blocks plus an inertial `m_ii/(gap-offset)²`
/// diagonal, whose Rayleigh quotient along the contact normal gives the stiffness
/// that scales the barrier — so a heavy/fast/close contact gets a proportionally
/// stronger barrier and resists high-momentum penetration (the reference's
/// robustness). Here we keep the inertial `mass/gap²` term (gap floored at
/// `0.1·dhat` to avoid double blow-up with the log barrier's own `1/d`), fold it
/// onto the base κ via a `coupling`, and drop the elastic `fixed_hess` term
/// (documented simplification). Result ≥ `kappa_base` (never weaker → resting
/// contact unchanged) and PSD (scalar multiply of the already-PSD barrier).
/// `coupling = 0` ⇒ the fixed base κ (backward-compatible default).
pub fn dynamic_stiffness(
    kappa_base: f32,
    coupling: f32,
    mass_eff: f32,
    gap: f32,
    dhat: f32,
) -> f32 {
    if coupling <= 0.0 || mass_eff <= 0.0 {
        return kappa_base;
    }
    let d = gap.max(0.1 * dhat).max(1.0e-6);
    kappa_base * (1.0 + coupling * mass_eff / (d * d))
}

/// Barrier value, first and second derivatives w.r.t. distance `d`; all scaled by
/// `kappa` (pass [`dynamic_stiffness`] for the per-contact adaptive κ). Returns
/// `None` when inactive (`d ≥ dhat`). For `d ≤ 0` (already penetrating, shouldn't
/// happen with CCD) `d` is clamped to a tiny positive so the barrier still
/// produces a strong outward push.
pub fn barrier(d: f32, dhat: f32, kappa: f32) -> Option<(f32, f32, f32)> {
    if d >= dhat {
        return None;
    }
    let d = d.max(1.0e-6);
    let t = d - dhat; // < 0
    let ld = (d / dhat).ln(); // < 0
    let e = -kappa * t * t * ld;
    let g = -kappa * (2.0 * t * ld + t * t / d);
    let h = -kappa * (2.0 * ld + 4.0 * t / d - (t * t) / (d * d));
    Some((e, g, h))
}

/// Accumulate collider-barrier gradient + PSD Hessian for one vertex position.
/// Returns `(grad3, hess3x3)` to add into the vertex's DOFs. Zero if inactive.
pub fn vertex_contact(
    x: Vec3,
    colliders: &[Collider],
    dhat: f32,
    kappa: f32,
    coupling: f32,
    mass_eff: f32,
) -> (Vec3, [[f32; 3]; 3]) {
    // Engine default: log barrier + multiplicative dynamic stiffness (unchanged).
    vertex_contact_ex(
        x,
        colliders,
        dhat,
        kappa,
        coupling,
        mass_eff,
        BarrierKind::Log,
        None,
    )
}

/// [`vertex_contact`] with the reference's options exposed. `kind` selects the
/// barrier; `elastic` (the vertex's own 3×3 elastic Hessian block, row-major)
/// switches κ to the reference's **absolute** Rayleigh stiffness instead of the
/// multiplicative [`dynamic_stiffness`].
///
/// For a vertex-vs-static-collider contact the stencil is N=1, so
/// [`rayleigh_stiffness`] reduces **exactly** (no approximation) to
/// `κ = nᵀ·elastic·n + m/d²` — the normalized direction `w` is just the contact
/// normal `n`. That makes this wiring a faithful port for colliders; multi-vertex
/// (V-T / E-E) stencils need the off-diagonal elastic blocks and are still on the
/// multiplicative path.
#[allow(clippy::too_many_arguments)]
pub fn vertex_contact_ex(
    x: Vec3,
    colliders: &[Collider],
    dhat: f32,
    kappa: f32,
    coupling: f32,
    mass_eff: f32,
    kind: BarrierKind,
    elastic: Option<&[f32; 9]>,
) -> (Vec3, [[f32; 3]; 3]) {
    vertex_contact_full(
        x, colliders, dhat, kappa, coupling, mass_eff, kind, elastic, None,
    )
}

/// [`vertex_contact_ex`] plus **friction**. `slip` is `Some((dx, mu, eps))` where `dx`
/// is this vertex's displacement since the step start; the tangential part of it drives
/// [`friction`], whose gradient/Hessian are added to the barrier's exactly as the
/// reference does in `contact.cu`.
#[allow(clippy::too_many_arguments)]
pub fn vertex_contact_full(
    x: Vec3,
    colliders: &[Collider],
    dhat: f32,
    kappa: f32,
    coupling: f32,
    mass_eff: f32,
    kind: BarrierKind,
    elastic: Option<&[f32; 9]>,
    slip: Option<(Vec3, f32, f32)>,
) -> (Vec3, [[f32; 3]; 3]) {
    let mut grad = Vec3::ZERO;
    let mut hess = [[0.0f32; 3]; 3];
    for c in colliders {
        let (d, n) = c.distance(x);
        let k = match elastic {
            // Reference semantics: κ is absolute (Rayleigh quotient), N=1 exact.
            Some(e) => rayleigh_stiffness(std::slice::from_ref(e), &[mass_eff], &[1.0], n * d, 0.0),
            None => dynamic_stiffness(kappa, coupling, mass_eff, d, dhat),
        };
        if let Some((_e, g, h)) = barrier_of(kind, d, dhat, k) {
            let f_c = n * g; // barrier force contribution for this contact
            grad += f_c;
            let hp = h.max(0.0); // SPD projection
            for r in 0..3 {
                for cc in 0..3 {
                    hess[r][cc] += hp * n[r] * n[cc];
                }
            }
            // Friction rides on THIS contact's normal force (reference: contact.cu adds
            // the Friction gradient/Hessian to the barrier's, per contact).
            if let Some((dx, mu, eps)) = slip {
                let (fg, fh) = friction(f_c, dx, n, mu, eps);
                grad += fg;
                for r in 0..3 {
                    for cc in 0..3 {
                        hess[r][cc] += fh[r][cc];
                    }
                }
            }
        }
    }
    (grad, hess)
}

#[cfg(test)]
mod friction_mode_tests {
    use super::*;

    /// All three modes, against the reference's `combine_friction` (`contact.cu:37-47`).
    #[test]
    fn combine_matches_the_reference_for_all_three_modes() {
        assert_eq!(FrictionMode::Min.combine(0.5, 0.0), 0.0);
        assert_eq!(FrictionMode::Max.combine(0.5, 0.0), 0.5);
        assert_eq!(FrictionMode::Mean.combine(0.5, 0.0), 0.25);
        // The distinction that matters in practice: a collider declaring friction 0 makes the
        // contact FRICTIONLESS under Min but leaves the cloth's own coefficient under Max.
        assert_eq!(FrictionMode::Min.combine(0.3, 0.0), 0.0);
        assert_eq!(FrictionMode::Max.combine(0.3, 0.0), 0.3);
    }

    /// Parsing follows the reference's `default:` branch for anything unrecognised, so a typo
    /// in a scene degrades to `Min` rather than to a guess.
    #[test]
    fn parsing_falls_back_to_the_reference_default() {
        assert_eq!(FrictionMode::from_str_or_default("max"), FrictionMode::Max);
        assert_eq!(
            FrictionMode::from_str_or_default("\"max\""),
            FrictionMode::Max
        );
        assert_eq!(
            FrictionMode::from_str_or_default(" mean "),
            FrictionMode::Mean
        );
        assert_eq!(FrictionMode::from_str_or_default("min"), FrictionMode::Min);
        assert_eq!(
            FrictionMode::from_str_or_default("wobble"),
            FrictionMode::Min
        );
        assert_eq!(FrictionMode::from_str_or_default(""), FrictionMode::Min);
        assert_eq!(FrictionMode::default(), FrictionMode::Min);
    }

    /// Folding a stencil's participants. `Mean` needs the arithmetic mean rather than a
    /// pairwise fold (repeated pairwise averaging would weight earlier entries more), which is
    /// the one case where `fold` is not just `combine` applied repeatedly.
    #[test]
    fn fold_handles_a_whole_stencil_including_the_mean_case() {
        let v = [0.1f32, 0.5, 0.3, 0.9];
        assert_eq!(FrictionMode::Min.fold(v), 0.1);
        assert_eq!(FrictionMode::Max.fold(v), 0.9);
        let mean = FrictionMode::Mean.fold(v);
        assert!(
            (mean - 0.45).abs() < 1.0e-6,
            "mean of {v:?} must be 0.45, got {mean} — a pairwise fold would give something else"
        );
        // Empty stencil is frictionless, not NaN or an infinity leaking from the identity.
        for m in [FrictionMode::Min, FrictionMode::Max, FrictionMode::Mean] {
            let e = m.fold(std::iter::empty());
            assert_eq!(e, 0.0, "{m:?} on an empty stencil must be 0, got {e}");
        }
        // Single participant returns itself under every mode.
        for m in [FrictionMode::Min, FrictionMode::Max, FrictionMode::Mean] {
            assert_eq!(m.fold([0.7f32]), 0.7, "{m:?} on one participant");
        }
    }
}

#[cfg(test)]
mod analytic_gradient_tests {
    use super::*;

    /// Central difference of a distance function, with the step scaled to the geometry. The
    /// distance itself is O(1) here (not a squared residual), so f32 differencing is accurate
    /// enough — unlike the energy gradients in `rod_bend`/`stitch`, which needed f64.
    fn fd(pts: [Vec3; 4], f: impl Fn([Vec3; 4]) -> f32) -> [Vec3; 4] {
        let h = 1.0e-3;
        let mut g = [Vec3::ZERO; 4];
        for vi in 0..4 {
            for co in 0..3 {
                let mut pp = pts;
                let mut pm = pts;
                pp[vi][co] += h;
                pm[vi][co] -= h;
                g[vi][co] = (f(pp) - f(pm)) / (2.0 * h);
            }
        }
        g
    }

    fn check(pts: [Vec3; 4], got: [Vec3; 4], want: [Vec3; 4], case: &str) {
        for vi in 0..4 {
            for co in 0..3 {
                assert!(
                    (got[vi][co] - want[vi][co]).abs() <= 2.0e-3,
                    "{case}: vertex {vi} comp {co}: analytic {} vs finite difference {} \
                     (pts {pts:?})",
                    got[vi][co],
                    want[vi][co]
                );
            }
        }
        // A distance depends only on RELATIVE positions, so translating everything cannot
        // change it: the four gradients must sum to zero. This is independent of the finite
        // differences and catches a mis-signed or mis-weighted term.
        let sum: Vec3 = got.iter().copied().sum();
        assert!(
            sum.length() < 1.0e-4,
            "{case}: gradients must sum to zero (translation invariance), got {sum:?}"
        );
    }

    /// Every closest-FEATURE branch of the point-triangle distance: face interior, each of the
    /// three edges, and each of the three vertices. A gradient that is right only in the
    /// interior would pass a single-case test and be wrong wherever cloth actually folds.
    #[test]
    fn point_triangle_gradient_matches_finite_differences_in_every_branch() {
        let (a, b, c) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let cases: [(&str, Vec3); 7] = [
            ("face interior", Vec3::new(0.25, 0.25, 0.4)),
            ("edge ab", Vec3::new(0.5, -0.3, 0.3)),
            ("edge ac", Vec3::new(-0.3, 0.5, 0.3)),
            ("edge bc", Vec3::new(0.8, 0.8, 0.3)),
            ("vertex a", Vec3::new(-0.4, -0.4, 0.2)),
            ("vertex b", Vec3::new(1.6, -0.4, 0.2)),
            ("vertex c", Vec3::new(-0.4, 1.6, 0.2)),
        ];
        for (name, p) in cases {
            let (d, g) = point_tri_distance_grad(p, a, b, c);
            assert!(d > 0.05, "{name}: case must be well clear of zero distance");
            let pts = [p, a, b, c];
            let want = fd(pts, |q| point_triangle_distance(q[0], q[1], q[2], q[3]));
            check(pts, g, want, name);
            // The point's own gradient is the unit normal away from the triangle.
            assert!(
                (g[0].length() - 1.0).abs() < 1.0e-4,
                "{name}: |∂d/∂p| must be 1, got {}",
                g[0].length()
            );
        }
    }

    /// Barycentric weights must be a partition of unity on the closest feature, and must be
    /// zero for vertices that feature does not touch — the property the gradient relies on.
    #[test]
    fn closest_point_weights_are_a_partition_of_unity() {
        let (a, b, c) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        for p in [
            Vec3::new(0.25, 0.25, 0.4),
            Vec3::new(0.5, -0.3, 0.3),
            Vec3::new(0.8, 0.8, 0.3),
            Vec3::new(-0.4, -0.4, 0.2),
            Vec3::new(1.6, -0.4, 0.2),
        ] {
            let (q, w) = closest_point_on_triangle_bary(p, a, b, c);
            let sum = w[0] + w[1] + w[2];
            assert!((sum - 1.0).abs() < 1.0e-5, "weights {w:?} sum to {sum}");
            assert!(w.iter().all(|&x| x >= -1.0e-6), "negative weight in {w:?}");
            // And they must actually reconstruct the closest point.
            let rebuilt = w[0] * a + w[1] * b + w[2] * c;
            assert!(
                (rebuilt - q).length() < 1.0e-5,
                "weights {w:?} rebuild {rebuilt:?}, not the closest point {q:?}"
            );
            // Consistent with the weight-free entry point.
            assert!((closest_point_on_triangle(p, a, b, c) - q).length() < 1.0e-6);
        }
    }

    /// Segment-segment: interior-interior, and the clamped endpoint cases.
    #[test]
    fn segment_segment_gradient_matches_finite_differences_in_every_branch() {
        let cases: [(&str, [Vec3; 4]); 3] = [
            (
                "interior-interior (crossing)",
                [
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.5, -1.0),
                    Vec3::new(0.0, 0.5, 1.0),
                ],
            ),
            (
                "endpoint of seg1",
                [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(2.0, 0.6, -1.0),
                    Vec3::new(2.0, 0.6, 1.0),
                ],
            ),
            (
                "endpoint-endpoint",
                [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(2.0, 0.7, 0.0),
                    Vec3::new(3.0, 0.7, 0.0),
                ],
            ),
        ];
        for (name, pts) in cases {
            let (d, g) = segment_segment_distance_grad(pts[0], pts[1], pts[2], pts[3]);
            assert!(d > 0.05, "{name}: case must be well clear of zero distance");
            let want = fd(pts, |q| segment_segment_distance(q[0], q[1], q[2], q[3]));
            check(pts, g, want, name);
        }
    }

    /// EXACTLY PARALLEL segments are a genuine non-smooth case and are checked against the
    /// REFERENCE's policy, not against a central difference.
    ///
    /// When the segments are parallel the closest pair is not unique, so the distance has a
    /// kink: a central difference returns the average of the one-sided derivatives (0.5/0.5
    /// across segment A's endpoints), which is not a derivative of anything. The reference
    /// resolves the ambiguity explicitly — `edge_edge_distance_coeff` computes the interior
    /// solution only `if (det)`, and with a zero determinant falls back to weights
    /// `(1, 0, c0, c1)`, i.e. all of segment A's weight on its FIRST endpoint
    /// (`contact/distance.hpp:86-104`). Our analytic gradient produces exactly that, so this
    /// case is a fidelity improvement over the finite-difference version it replaced, which
    /// smeared the weight 0.5/0.5.
    #[test]
    fn parallel_segments_follow_the_reference_weight_policy() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.4, 0.0),
            Vec3::new(1.0, 0.4, 0.0),
        ];
        let w = edge_edge_witness(pts[0], pts[1], pts[2], pts[3]);
        assert_eq!(
            [w[0], w[1]],
            [1.0, 0.0],
            "parallel: all of segment A's weight must sit on its first endpoint"
        );
        let (d, g) = segment_segment_distance_grad(pts[0], pts[1], pts[2], pts[3]);
        assert!(
            (d - 0.4).abs() < 1.0e-6,
            "perpendicular distance is 0.4, got {d}"
        );
        // Weight (1, 0) on segment A, matching the reference's zero-determinant fallback.
        assert!(
            (g[0].length() - 1.0).abs() < 1.0e-5 && g[1].length() < 1.0e-6,
            "expected all of A's weight on its first endpoint, got {:?} / {:?}",
            g[0],
            g[1]
        );
        // Whatever the witness, translation invariance must still hold.
        let sum: Vec3 = g.iter().copied().sum();
        assert!(
            sum.length() < 1.0e-5,
            "gradients must sum to zero, got {sum:?}"
        );
        // And the witness on B is the projection of A's endpoint, so it stays in range.
        assert!(
            (0.0..=1.0).contains(&w[2]) && (0.0..=1.0).contains(&w[3]),
            "B's weights out of range: {:?}",
            &w[2..]
        );
    }

    /// The witness-derived distance must agree with `segment_segment_distance` everywhere.
    ///
    /// This matters because the two are used in DIFFERENT places: CCD calls
    /// `segment_segment_distance` (clamped interior solution) while the barrier stencil uses
    /// `segment_segment_distance_grad` (the reference's enumerate-and-min witness). Both are
    /// valid algorithms for the same minimum, so they must return the same DISTANCE even
    /// where they pick different witnesses among ties — if they ever disagreed, CCD and the
    /// barrier would be resolving different geometry.
    #[test]
    fn witness_distance_agrees_with_the_clamped_segment_distance() {
        // Deterministic sweep over configurations: crossing, parallel, skew, endpoint-near,
        // degenerate-length. No RNG, so a failure is reproducible.
        let mut worst = 0.0f32;
        let mut checked = 0usize;
        for i in 0..12i32 {
            for j in 0..12i32 {
                let f = |k: i32, o: i32| (k as f32 * 0.37 + o as f32 * 0.11) % 2.0 - 1.0;
                let p1 = Vec3::new(f(i, 0), f(j, 1), f(i, 2));
                let q1 = p1 + Vec3::new(f(j, 3), f(i, 4), f(j, 5));
                let p2 = Vec3::new(f(i, 6), f(j, 7), f(i, 8));
                let q2 = p2 + Vec3::new(f(j, 9), f(i, 10), f(j, 11));
                let want = segment_segment_distance(p1, q1, p2, q2);
                let (got, _) = segment_segment_distance_grad(p1, q1, p2, q2);
                worst = worst.max((got - want).abs());
                checked += 1;
                assert!(
                    (got - want).abs() <= 1.0e-5 * want.max(1.0),
                    "witness distance {got} != clamped distance {want} for \
                     {p1:?} {q1:?} {p2:?} {q2:?}"
                );
            }
        }
        // Explicit degenerate and parallel cases on top of the sweep.
        let cases = [
            // Exactly parallel and overlapping.
            [
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(0.0, 0.4, 0.0),
                Vec3::new(1.0, 0.4, 0.0),
            ],
            // Parallel, NOT overlapping (the case that needs the enumeration).
            [
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(2.0, 0.7, 0.0),
                Vec3::new(3.0, 0.7, 0.0),
            ],
            // Collinear.
            [
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
            ],
            // One segment is a point.
            [Vec3::ZERO, Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0), Vec3::X],
            // Both are points.
            [Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y],
        ];
        for c in cases {
            let want = segment_segment_distance(c[0], c[1], c[2], c[3]);
            let (got, g) = segment_segment_distance_grad(c[0], c[1], c[2], c[3]);
            assert!(
                (got - want).abs() <= 1.0e-5 * want.max(1.0),
                "degenerate case {c:?}: witness {got} != clamped {want}"
            );
            assert!(
                g.iter().all(|v| v.is_finite()),
                "degenerate case {c:?} produced a non-finite gradient: {g:?}"
            );
            // Translation invariance holds even in the degenerate cases.
            let sum: Vec3 = g.iter().copied().sum();
            assert!(
                sum.length() < 1.0e-4,
                "case {c:?}: gradients must sum to zero"
            );
            checked += 1;
        }
        assert!(
            checked > 140,
            "the sweep must actually cover cases, got {checked}"
        );
        eprintln!("witness vs clamped distance: {checked} configurations, worst delta {worst:.3e}");
    }

    /// The witness weights must be a convex combination on each edge — the property the
    /// gradient's correctness rests on.
    #[test]
    fn edge_edge_witness_weights_are_convex_on_each_edge() {
        for i in 0..10i32 {
            let f = |k: i32, o: i32| (k as f32 * 0.53 + o as f32 * 0.17) % 2.0 - 1.0;
            let p1 = Vec3::new(f(i, 0), f(i, 1), f(i, 2));
            let q1 = p1 + Vec3::new(f(i, 3), f(i, 4), f(i, 5));
            let p2 = Vec3::new(f(i, 6), f(i, 7), f(i, 8));
            let q2 = p2 + Vec3::new(f(i, 9), f(i, 10), f(i, 11));
            let w = edge_edge_witness(p1, q1, p2, q2);
            assert!(
                (w[0] + w[1] - 1.0).abs() < 1.0e-5,
                "edge A weights {:?} must sum to 1",
                &w[..2]
            );
            assert!(
                (w[2] + w[3] - 1.0).abs() < 1.0e-5,
                "edge B weights {:?} must sum to 1",
                &w[2..]
            );
            assert!(
                w.iter().all(|&x| (-1.0e-6..=1.0 + 1.0e-6).contains(&x)),
                "witness weights must lie in [0,1], got {w:?}"
            );
        }
    }

    /// The gradient must be consistent with the distance it reports: stepping a small amount
    /// ALONG the gradient must increase the distance by that amount to first order. This is a
    /// property of the pair (d, g) together, so it catches the two being computed from
    /// different closest features.
    #[test]
    fn stepping_along_the_gradient_changes_the_distance_as_predicted() {
        let (a, b, c) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let p = Vec3::new(0.25, 0.25, 0.4);
        let (d0, g) = point_tri_distance_grad(p, a, b, c);
        let eps = 1.0e-3;
        let d1 = point_triangle_distance(p + eps * g[0], a, b, c);
        let predicted = d0 + eps * g[0].length_squared();
        assert!(
            (d1 - predicted).abs() < 1.0e-5,
            "predicted {predicted}, got {d1} (d0 {d0})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ported cubic barrier's analytic derivatives must equal finite
    /// differences of its own energy — the zero-hallucination check that the
    /// transcription from `cubic.hpp` is self-consistent.
    #[test]
    fn cubic_barrier_derivatives_match_fd() {
        let (ghat, offset) = (0.1f32, 0.02f32);
        // Sample across the active range, including negative gap (penetration).
        for i in 0..40 {
            let g = offset - 0.03 + (i as f32) * 0.0035;
            let Some((_e, grad, curv)) = cubic_barrier(g, ghat, offset) else {
                continue;
            };
            let h = 1e-4;
            let ep = cubic_barrier(g + h, ghat, offset).map(|t| t.0);
            let em = cubic_barrier(g - h, ghat, offset).map(|t| t.0);
            if let (Some(ep), Some(em)) = (ep, em) {
                let fd_grad = (ep - em) / (2.0 * h);
                assert!(
                    (grad - fd_grad).abs() < 1e-2 * grad.abs().max(1.0),
                    "cubic grad mismatch at g={g}: analytic {grad} vs fd {fd_grad}"
                );
            }
            // Curvature needs a LARGER step: a second difference divides by h², so
            // h=1e-4 loses ~ε/h²≈10 to f32 cancellation. The energy is exactly
            // cubic, whose 4th derivative is 0, so the central second difference is
            // truncation-exact for ANY h — bigger h is strictly more accurate here.
            let hc = 1e-2;
            if let (Some(ep), Some(em)) = (
                cubic_barrier(g + hc, ghat, offset).map(|t| t.0),
                cubic_barrier(g - hc, ghat, offset).map(|t| t.0),
            ) {
                let e0 = cubic_barrier(g, ghat, offset).unwrap().0;
                let fd_curv = (ep - 2.0 * e0 + em) / (hc * hc);
                assert!(
                    (curv - fd_curv).abs() < 2e-2 * curv.abs().max(1.0),
                    "cubic curvature mismatch at g={g}: analytic {curv} vs fd {fd_curv}"
                );
            }
        }
    }

    /// Behavioral contract: inactive past `ghat`, positive + repulsive + PSD
    /// inside, and — the property that distinguishes it from the log barrier —
    /// FINITE with bounded curvature at and below zero gap.
    #[test]
    fn cubic_barrier_is_finite_and_repulsive_including_penetration() {
        let (ghat, offset) = (0.1f32, 0.0f32);
        assert!(
            cubic_barrier(0.1, ghat, offset).is_none(),
            "inactive at ghat"
        );
        assert!(cubic_barrier(0.5, ghat, offset).is_none());
        for &g in &[0.09f32, 0.05, 0.01, 0.0, -0.05, -0.5] {
            let (e, grad, curv) = cubic_barrier(g, ghat, offset).unwrap();
            assert!(
                e > 0.0 && e.is_finite(),
                "energy not positive/finite at g={g}"
            );
            assert!(grad < 0.0 && grad.is_finite(), "not repulsive at g={g}");
            assert!(curv > 0.0 && curv.is_finite(), "not PSD/finite at g={g}");
        }
        // The log barrier diverges at g→0; the cubic one stays bounded.
        let (e_zero, _, c_zero) = cubic_barrier(1e-9, ghat, offset).unwrap();
        assert!(
            e_zero < 1.0 && c_zero <= 4.0 + 1e-5,
            "cubic blew up near zero gap"
        );
        // Deeper penetration ⇒ strictly stronger push-out.
        let g_shallow = cubic_barrier(0.01, ghat, offset).unwrap().1;
        let g_deep = cubic_barrier(-0.2, ghat, offset).unwrap().1;
        assert!(g_deep < g_shallow, "deeper penetration must push harder");
    }

    /// With a zero elastic Hessian and uniform mass, the Rayleigh quotient
    /// reduces analytically to `m/g²` (the diagonal is `(m/g²)·I` and `w` is a
    /// unit vector) — an independent closed form, so this pins the port.
    #[test]
    fn rayleigh_stiffness_reduces_to_inertial_term() {
        let e = Vec3::new(0.0, 0.03, 0.0);
        let offset = 0.0;
        let g = e.length() - offset;
        for n in [2usize, 3, 4] {
            let elastic = vec![[0.0f32; 9]; n * n];
            let mass = vec![0.7f32; n];
            // Reference weights: barycentric-style, signed, summing to zero.
            let mut weights = vec![1.0f32 / (n as f32 - 1.0); n];
            weights[0] = -1.0;
            let k = rayleigh_stiffness(&elastic, &mass, &weights, e, offset);
            let expect = 0.7 / (g * g);
            assert!(
                (k - expect).abs() < 1e-3 * expect,
                "n={n}: rayleigh {k} != analytic m/g² {expect}"
            );
        }
        // 1/g² scaling: halving the gap quadruples the stiffness.
        let elastic = vec![[0.0f32; 9]; 4];
        let mass = vec![0.7f32; 2];
        let weights = vec![-1.0f32, 1.0];
        let k_far = rayleigh_stiffness(&elastic, &mass, &weights, e, offset);
        let k_near = rayleigh_stiffness(&elastic, &mass, &weights, e * 0.5, offset);
        assert!(
            (k_near / k_far - 4.0).abs() < 1e-2,
            "expected 1/g² scaling, got ratio {}",
            k_near / k_far
        );
    }

    /// A zero mass (static collider side) must be substituted with the largest
    /// non-zero mass, so a one-sided contact feels the same inertia as a
    /// symmetric one — the reference's stated fix for unstable stiff collisions.
    #[test]
    fn rayleigh_stiffness_substitutes_static_side_mass() {
        let e = Vec3::new(0.0, 0.04, 0.0);
        let elastic = vec![[0.0f32; 9]; 4];
        let weights = vec![-1.0f32, 1.0];
        let both_dynamic = rayleigh_stiffness(&elastic, &[0.9, 0.9], &weights, e, 0.0);
        let one_static = rayleigh_stiffness(&elastic, &[0.9, 0.0], &weights, e, 0.0);
        assert!(
            (both_dynamic - one_static).abs() < 1e-4 * both_dynamic,
            "static side not substituted: {both_dynamic} vs {one_static}"
        );
        // Without the substitution the one-sided case would be ~half as stiff;
        // confirm we are NOT in that regime.
        assert!(one_static > 0.75 * both_dynamic);
    }

    /// The headline: including the ELASTIC Hessian makes the barrier stiffer than
    /// the inertia-only stiffness — a stiff material gets a proportionally stiffer
    /// contact. This is exactly what `dynamic_stiffness` drops.
    #[test]
    fn rayleigh_stiffness_grows_with_material_stiffness() {
        let e = Vec3::new(0.0, 0.03, 0.0);
        let mass = vec![0.5f32, 0.5];
        let weights = vec![-1.0f32, 1.0];
        let inertia_only = rayleigh_stiffness(&[[0.0f32; 9]; 4], &mass, &weights, e, 0.0);

        // A PSD elastic block set: k·I on each diagonal block.
        let mut prev = inertia_only;
        for k_el in [10.0f32, 100.0, 1000.0] {
            let mut elastic = vec![[0.0f32; 9]; 4];
            for ii in 0..2 {
                let blk = &mut elastic[ii * 2 + ii];
                blk[0] = k_el;
                blk[4] = k_el;
                blk[8] = k_el;
            }
            let k = rayleigh_stiffness(&elastic, &mass, &weights, e, 0.0);
            assert!(
                k > prev,
                "elastic stiffness {k_el} did not increase contact stiffness ({k} <= {prev})"
            );
            // Analytic: diagonal (k_el + m/g²)I ⇒ quotient = k_el + m/g².
            let expect = k_el + inertia_only;
            assert!(
                (k - expect).abs() < 1e-2 * expect,
                "elastic case: {k} != k_el + m/g² = {expect}"
            );
            prev = k;
        }
    }

    #[test]
    fn barrier_is_c2_at_dhat_and_blows_up_near_zero() {
        let (dhat, k) = (0.1f32, 1.0f32);
        assert!(barrier(0.1, dhat, k).is_none(), "inactive at dhat");
        assert!(barrier(0.2, dhat, k).is_none());
        let (e_edge, g_edge, _) = barrier(0.0999, dhat, k).unwrap();
        assert!(
            e_edge > 0.0 && e_edge < 1e-3,
            "≈0 just inside dhat: {e_edge}"
        );
        assert!(g_edge.abs() < 1e-2, "slope ≈0 at dhat: {g_edge}");
        let (e_near, g_near, _) = barrier(1e-4, dhat, k).unwrap();
        assert!(e_near > e_edge * 100.0, "energy grows toward contact");
        assert!(g_near < 0.0, "repulsive (b'<0 → force −b'·n outward)");
    }

    #[test]
    fn point_triangle_distance_cases() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        // Directly above the interior → perpendicular height.
        let d = point_triangle_distance(Vec3::new(0.25, 0.25, 0.7), a, b, c);
        assert!((d - 0.7).abs() < 1e-5, "face distance {d}");
        // Beyond vertex a → distance to a.
        let d = point_triangle_distance(Vec3::new(-1.0, -1.0, 0.0), a, b, c);
        assert!((d - 2.0f32.sqrt()).abs() < 1e-5, "vertex distance {d}");
        // Off an edge (x-axis edge a-b), point below in y.
        let d = point_triangle_distance(Vec3::new(0.5, -0.3, 0.0), a, b, c);
        assert!((d - 0.3).abs() < 1e-5, "edge distance {d}");
    }

    #[test]
    fn edge_triangle_intersect_cases() {
        // Triangle in the z=0 plane.
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        // Vertical edge through the interior (0.25,0.25) → pierces.
        assert!(edge_triangle_intersect(
            Vec3::new(0.25, 0.25, 1.0),
            Vec3::new(0.25, 0.25, -1.0),
            v0,
            v1,
            v2,
        ));
        // Same edge but on the SAME side of the plane (both z>0) → no pierce.
        assert!(!edge_triangle_intersect(
            Vec3::new(0.25, 0.25, 1.0),
            Vec3::new(0.25, 0.25, 0.5),
            v0,
            v1,
            v2,
        ));
        // Crosses the plane but OUTSIDE the triangle (x=2) → no pierce.
        assert!(!edge_triangle_intersect(
            Vec3::new(2.0, 0.25, 1.0),
            Vec3::new(2.0, 0.25, -1.0),
            v0,
            v1,
            v2,
        ));
        // Coplanar edge → false (matches the device predicate).
        assert!(!edge_triangle_intersect(
            Vec3::new(-1.0, 0.25, 0.0),
            Vec3::new(2.0, 0.25, 0.0),
            v0,
            v1,
            v2,
        ));
    }

    #[test]
    fn segment_segment_distance_cases() {
        // Parallel, offset by 1 in y.
        let d = segment_segment_distance(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        assert!((d - 1.0).abs() < 1e-5, "parallel offset {d}");
        // Collinear, disjoint → endpoint gap 1.
        let d = segment_segment_distance(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        );
        assert!((d - 1.0).abs() < 1e-5, "collinear gap {d}");
        // Perpendicular skew: seg2 passes 1 above seg1's midpoint.
        let d = segment_segment_distance(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, -1.0, 1.0),
            Vec3::new(0.5, 1.0, 1.0),
        );
        assert!((d - 1.0).abs() < 1e-5, "skew {d}");
        // Crossing (share a point) → 0.
        let d = segment_segment_distance(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!(d < 1e-5, "crossing {d}");
    }

    #[test]
    fn accd_toi_edge_through_edge() {
        // A horizontal edge sweeping DOWN through a static crossing edge below it —
        // the through-edge crossing vertex-triangle distance would miss.
        let a0 = Vec3::new(-1.0, 0.5, 0.0);
        let a1 = Vec3::new(1.0, 0.5, 0.0); // moving edge, 0.5 above
        let b0 = Vec3::new(0.0, 0.0, -1.0);
        let b1 = Vec3::new(0.0, 0.0, 1.0); // static edge crossing under it
        let x0 = [a0, a1, b0, b1];
        let dx = [
            Vec3::new(0.0, -2.0, 0.0),
            Vec3::new(0.0, -2.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
        ];
        let toi = accd_toi(&x0, &dx, 0.0, 1.25, 0.01, 256, |q| {
            let d = segment_segment_distance(q[0], q[1], q[2], q[3]);
            d * d
        });
        assert!(toi < 1.25, "edge-edge impact should be detected, toi={toi}");
        // At toi the edges have not crossed (positive gap).
        let m0 = a0 + dx[0] * toi;
        let m1 = a1 + dx[1] * toi;
        let gap = segment_segment_distance(m0, m1, b0, b1);
        assert!(gap > 0.0, "ACCD must not let edges cross: gap={gap}");
    }

    #[test]
    fn accd_toi_point_into_triangle() {
        // Triangle in the z=0 plane; point 1.0 above it moving straight down.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let p = Vec3::new(0.25, 0.25, 1.0);
        let x0 = [p, a, b, c];
        let dx = [
            Vec3::new(0.0, 0.0, -2.0),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ];
        let sq = |q: &[Vec3]| {
            let d = point_triangle_distance(q[0], q[1], q[2], q[3]);
            d * d
        };
        let toi = accd_toi(&x0, &dx, 0.0, 1.25, 0.01, 256, sq);
        // Impact found within the horizon, and it stops just short of contact
        // (target ≈ reduction·gap = 0.01): at toi the gap is > 0 (no crossing).
        assert!(toi < 1.25, "impact should be detected, toi={toi}");
        // Head-on at speed 2 from gap 1.0 → real contact near t≈0.5.
        assert!((0.40..0.55).contains(&toi), "toi≈0.5 expected, got {toi}");
        let moved = p + dx[0] * toi;
        let gap = point_triangle_distance(moved, a, b, c);
        assert!(gap > 0.0, "ACCD must not cross: gap={gap}");
    }

    #[test]
    fn accd_toi_no_relative_motion() {
        // All points translate together → no impact possible (u_max = 0).
        let x0 = [
            Vec3::new(0.25, 0.25, 1.0),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let v = Vec3::new(0.0, 0.0, -5.0);
        let dx = [v, v, v, v];
        let sq = |q: &[Vec3]| {
            let d = point_triangle_distance(q[0], q[1], q[2], q[3]);
            d * d
        };
        assert_eq!(accd_toi(&x0, &dx, 0.0, 1.25, 0.01, 256, sq), 1.25);
    }

    #[test]
    fn halfspace_distance_and_push() {
        let floor = Collider::HalfSpace {
            point: Vec3::ZERO,
            normal: Vec3::Y,
        };
        let (d, n) = floor.distance(Vec3::new(3.0, 0.05, -2.0));
        assert_eq!(d, 0.05);
        assert_eq!(n, Vec3::Y);
        // A vertex within dhat of the floor is pushed up (+y).
        let (grad, _) = vertex_contact(Vec3::new(0.0, 0.02, 0.0), &[floor], 0.1, 10.0, 0.0, 1.0);
        assert!(grad.y < 0.0, "grad points down → force −grad points up");
    }

    #[test]
    fn oriented_gap_signs_and_barrier_recovers() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0); // n̂ ∝ +z
                                          // Above the interior → positive; below → negative (does NOT flip to +).
        let above = oriented_point_triangle(Vec3::new(0.25, 0.25, 0.3), a, b, c);
        let below = oriented_point_triangle(Vec3::new(0.25, 0.25, -0.3), a, b, c);
        assert!((above - 0.3).abs() < 1e-5, "above {above}");
        assert!((below + 0.3).abs() < 1e-5, "below {below}");
        // Signed cubic barrier: inactive when separated beyond d̂; for a crossed
        // (negative) gap the force is repulsive toward +g (recovery).
        let dhat = 0.1;
        assert!(
            signed_barrier(0.2, dhat, 1.0).is_none(),
            "inactive when far"
        );
        let (_e, gp, hp) = signed_barrier(-0.05, dhat, 1.0).unwrap();
        assert!(gp < 0.0, "b'<0 → force −b' pushes toward +g (out)");
        assert!(hp > 0.0, "b''>0 → PSD");
        // Deeper penetration → stronger recovery.
        let (_e2, gp2, _h2) = signed_barrier(-0.15, dhat, 1.0).unwrap();
        assert!(gp2 < gp, "deeper penetration → larger restoring force");
    }

    #[test]
    fn dynamic_stiffness_scales_with_inertia() {
        let base = 100.0;
        let dhat = 0.02;
        // coupling 0 → base κ (backward-compatible).
        assert_eq!(dynamic_stiffness(base, 0.0, 1.0, 0.01, dhat), base);
        // massless/pinned → base κ.
        assert_eq!(dynamic_stiffness(base, 1.0, 0.0, 0.01, dhat), base);
        // coupling>0 → strictly stiffer, and stiffer still as the gap closes.
        let far = dynamic_stiffness(base, 1.0, 0.02, 0.015, dhat);
        let near = dynamic_stiffness(base, 1.0, 0.02, 0.005, dhat);
        assert!(far > base, "dynamic κ boosts stiffness");
        assert!(near > far, "stiffer as the contact closes (mass/gap²)");
    }

    #[test]
    fn sphere_keeps_point_outside() {
        let s = Collider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let (d, n) = s.distance(Vec3::new(1.05, 0.0, 0.0));
        assert!((d - 0.05).abs() < 1e-6);
        assert!((n - Vec3::X).length() < 1e-6);
        let (grad, hess) = vertex_contact(Vec3::new(1.02, 0.0, 0.0), &[s], 0.1, 10.0, 0.0, 1.0);
        assert!(grad.x < 0.0, "pushed outward (+x)");
        assert!(hess[0][0] >= 0.0, "PSD diagonal");
    }

    /// Friction must be purely tangential, oppose the slip, respect the Coulomb bound
    /// `‖f_t‖ ≤ μ·contact`, and vanish when `μ = 0`.
    #[test]
    fn friction_is_tangential_opposes_slip_and_respects_coulomb() {
        let n = Vec3::Y;
        // Barrier force pushing the vertex along +n ⇒ contact = -n·f_c > 0 needs f_c
        // anti-parallel to n (the barrier gradient points "into" the surface).
        let f_c = -n * 10.0;
        let mu = 0.4f32;
        let eps = 1.0e-3f32;
        let contact = -n.dot(f_c);
        assert!(contact > 0.0);

        // Sliding regime: slip well above eps.
        let slip = Vec3::new(0.05, 0.02, -0.03); // has a normal component too
        let (g, h) = friction(f_c, slip, n, mu, eps);
        // Purely tangential.
        assert!(
            g.dot(n).abs() < 1e-5,
            "friction must have no normal component"
        );
        // Opposes the tangential slip... the reference returns λ·(P·dx), which is the
        // ENERGY GRADIENT; the force is its negative, so the gradient is ALIGNED with
        // slip and the resulting force opposes it.
        let pdx = slip - n.dot(slip) * n;
        assert!(
            g.dot(pdx) > 0.0,
            "gradient should align with tangential slip"
        );
        // Coulomb bound: in the sliding regime the magnitude is exactly μ·contact.
        assert!(
            (g.length() - mu * contact).abs() < 1e-3 * mu * contact,
            "sliding friction magnitude should equal μ·contact: {} vs {}",
            g.length(),
            mu * contact
        );
        // Hessian symmetric + PSD.
        for (r, row) in h.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!((v - h[c][r]).abs() < 1e-6, "asymmetric at ({r},{c})");
            }
        }
        for t in 0..8u64 {
            let x = Vec3::new(
                ((t * 7919 % 100) as f32) / 50.0 - 1.0,
                ((t * 104729 % 100) as f32) / 50.0 - 1.0,
                ((t * 1299709 % 100) as f32) / 50.0 - 1.0,
            );
            let mut q = 0.0f32;
            for r in 0..3 {
                for c in 0..3 {
                    q += x[r] * h[r][c] * x[c];
                }
            }
            assert!(q >= -1e-5, "friction Hessian not PSD: {q}");
        }
        // μ = 0 ⇒ nothing.
        let (g0, h0) = friction(f_c, slip, n, 0.0, eps);
        assert_eq!(g0, Vec3::ZERO);
        assert_eq!(h0, [[0.0; 3]; 3]);
    }

    /// The `ε` regularization must cap the force as slip → 0 (otherwise the direction is
    /// undefined and λ diverges), and the force must grow linearly in that regime.
    #[test]
    fn friction_regularization_caps_tiny_slip() {
        let n = Vec3::Y;
        let f_c = -n * 10.0;
        let (mu, eps) = (0.5f32, 1.0e-3f32);
        let contact = -n.dot(f_c);
        let bound = mu * contact;
        let mut prev = 0.0f32;
        for k in 0..6 {
            let s = 1.0e-6 * 10f32.powi(k); // 1e-6 .. 1e-1
            let (g, _h) = friction(f_c, Vec3::new(s, 0.0, 0.0), n, mu, eps);
            let m = g.length();
            assert!(m.is_finite(), "friction blew up at slip {s}");
            assert!(
                m <= bound + 1e-4,
                "friction {m} exceeded the Coulomb bound {bound} at slip {s}"
            );
            assert!(m >= prev - 1e-6, "magnitude should be monotone in slip");
            prev = m;
            // Below eps the response is linear in slip: m ≈ μ·contact·s/eps.
            if s < eps {
                let expect = bound * s / eps;
                assert!(
                    (m - expect).abs() < 1e-3 * bound.max(1.0),
                    "sub-eps friction should be linear: {m} vs {expect}"
                );
            }
        }
    }

    /// A separating contact (no inward barrier force) must produce no friction.
    #[test]
    fn friction_vanishes_without_normal_force() {
        let n = Vec3::Y;
        // Force along +n ⇒ contact = -n·f_c < 0 (separating).
        let (g, h) = friction(n * 5.0, Vec3::new(0.1, 0.0, 0.0), n, 0.5, 1e-3);
        assert_eq!(g, Vec3::ZERO);
        assert_eq!(h, [[0.0; 3]; 3]);
    }

    /// Stencil friction must recover the proximity weights from the distance gradient,
    /// produce a translation-invariant force set (Σ w_a f = 0 when Σ w_a = 0), and be
    /// PSD. Verified on a vertex-triangle configuration where the weights are known.
    #[test]
    fn stencil_friction_uses_proximity_weights_and_is_psd() {
        // Point above the centroid of a unit triangle in the xz-plane ⇒ n = +y and the
        // barycentric weights are (1, -1/3, -1/3, -1/3).
        let p = Vec3::new(0.5, 0.2, 0.28867513);
        let (a, b, c) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.8660254),
        );
        let (_d, g) = point_tri_distance_grad(p, a, b, c);
        // Recovered weights must sum to ~0 (a rigid translation cannot change distance).
        let n = g[0].normalize();
        let wsum: f32 = g.iter().map(|gi| gi.dot(n)).sum();
        assert!(
            wsum.abs() < 1e-2,
            "proximity weights should sum to ~0, got {wsum}"
        );
        assert!(
            (g[0].dot(n) - 1.0).abs() < 1e-2,
            "point weight should be ~1"
        );

        // Tangential slip of the point only.
        let dx = [
            Vec3::new(0.02, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ];
        // Barrier gradient is negative (repulsive) for an active contact.
        let (fg, fh) = stencil_friction(&g, &dx, -50.0, 0.5, 1e-3);
        // Forces must sum to ~0 (weights sum to 0 ⇒ no net force on a translating pair).
        let net: Vec3 = fg.iter().copied().sum();
        assert!(
            net.length() < 1e-2,
            "stencil friction must not add net force: {net:?}"
        );
        // Point's friction gradient must be tangential and aligned with its slip.
        assert!(fg[0].dot(n).abs() < 1e-4, "friction must be tangential");
        assert!(fg[0].dot(dx[0]) > 0.0, "gradient should align with slip");
        // Hessian symmetric + PSD.
        for r in 0..12 {
            for c in 0..12 {
                assert!((fh[r * 12 + c] - fh[c * 12 + r]).abs() < 1e-5);
            }
        }
        for t in 0..8u64 {
            let xv: [f32; 12] = std::array::from_fn(|k| {
                ((t.wrapping_mul(7919)
                    .wrapping_add((k as u64).wrapping_mul(104729))
                    % 200) as f32)
                    / 100.0
                    - 1.0
            });
            let mut q = 0.0f32;
            for r in 0..12 {
                for c in 0..12 {
                    q += xv[r] * fh[r * 12 + c] * xv[c];
                }
            }
            assert!(q >= -1e-4, "stencil friction Hessian not PSD: {q}");
        }
        // mu = 0 ⇒ nothing.
        let (z, zh) = stencil_friction(&g, &dx, -50.0, 0.0, 1e-3);
        assert!(z.iter().all(|v| *v == Vec3::ZERO) && zh.iter().all(|x| *x == 0.0));
    }
}
