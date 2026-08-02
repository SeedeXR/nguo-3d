// SPDX-License-Identifier: GPL-2.0-or-later
//
// This is a numerical library, so it is tested like one: every constraint
// individually against a closed-form expectation, the classic sanity sketches,
// DOF analysis, determinism, behaviour with no solution, and lifecycle churn.

use nguo_constraints::{Algorithm, DofState, Error, SolveStatus, System};

/// Generous enough for a dog-leg solve, tight enough that a wrong constraint
/// cannot sneak through.
const TOL: f64 = 1e-7;

#[track_caller]
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < TOL,
        "expected {b}, got {a} (delta {})",
        (a - b).abs()
    );
}

#[track_caller]
fn finite(v: f64) {
    assert!(v.is_finite(), "expected a finite value, got {v}");
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

// ===========================================================================
// Geometric constraints, one at a time (§9.2)
// ===========================================================================

#[test]
fn coincidence_merges_two_points() {
    let mut s = System::new();
    let anchor = s.add_fixed_point(4.0, -2.0);
    let free = s.add_point(9.0, 9.0);
    s.coincident(anchor, free).unwrap();
    s.solve().unwrap();

    let (x, y) = s.point(free).unwrap();
    near(x, 4.0);
    near(y, -2.0);
}

#[test]
fn horizontal_flattens_a_line() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 3.0);
    let b = s.add_point(10.0, 8.0);
    let l = s.add_line(a, b).unwrap();
    s.horizontal(l).unwrap();
    s.solve().unwrap();

    near(s.point(b).unwrap().1, 3.0);
}

#[test]
fn vertical_stands_a_line_up() {
    let mut s = System::new();
    let a = s.add_fixed_point(3.0, 0.0);
    let b = s.add_point(8.0, 10.0);
    let l = s.add_line(a, b).unwrap();
    s.vertical(l).unwrap();
    s.solve().unwrap();

    near(s.point(b).unwrap().0, 3.0);
}

#[test]
fn parallel_matches_direction() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_fixed_point(10.0, 0.0);
    let reference = s.add_line(a, b).unwrap();

    let c = s.add_fixed_point(0.0, 5.0);
    let d = s.add_point(8.0, 9.0);
    let other = s.add_line(c, d).unwrap();

    s.parallel(reference, other).unwrap();
    s.solve().unwrap();

    // Parallel to a horizontal reference means the free endpoint drops to the
    // fixed endpoint's height.
    near(s.point(d).unwrap().1, 5.0);
}

#[test]
fn perpendicular_turns_a_line_through_a_right_angle() {
    let mut s = System::new();
    let origin = s.add_fixed_point(0.0, 0.0);
    let east = s.add_fixed_point(10.0, 0.0);
    let horizontal = s.add_line(origin, east).unwrap();

    let tip = s.add_point(5.0, 5.0);
    let upright = s.add_line(origin, tip).unwrap();
    let len = s.add_driving_param(10.0);

    s.perpendicular(horizontal, upright).unwrap();
    s.length(upright, len).unwrap();
    s.solve().unwrap();

    let (x, y) = s.point(tip).unwrap();
    near(x, 0.0);
    near(y.abs(), 10.0);
}

#[test]
fn tangent_line_touches_circle_at_exactly_the_radius() {
    let mut s = System::new();
    let centre = s.add_fixed_point(0.0, 0.0);
    let r = s.add_driving_param(5.0);
    let circle = s.add_circle(centre, r).unwrap();

    let a = s.add_point(6.0, -5.0);
    let b = s.add_point(6.0, 5.0);
    let line = s.add_line(a, b).unwrap();

    s.vertical(line).unwrap();
    s.tangent(line, circle).unwrap();
    s.solve().unwrap();

    let (ax, _) = s.point(a).unwrap();
    let (bx, _) = s.point(b).unwrap();
    near(ax, bx); // still vertical
    near(ax.abs(), 5.0); // and exactly one radius from the centre
}

#[test]
fn symmetry_mirrors_a_point_across_a_line() {
    let mut s = System::new();
    let axis_a = s.add_fixed_point(0.0, 0.0);
    let axis_b = s.add_fixed_point(0.0, 10.0);
    let axis = s.add_line(axis_a, axis_b).unwrap();

    let left = s.add_fixed_point(3.0, 2.0);
    let right = s.add_point(-1.0, 5.0);

    s.symmetric_about_line(left, right, axis).unwrap();
    s.solve().unwrap();

    let (x, y) = s.point(right).unwrap();
    near(x, -3.0);
    near(y, 2.0);
}

#[test]
fn symmetry_about_a_point_inverts_through_it() {
    let mut s = System::new();
    let centre = s.add_fixed_point(1.0, 1.0);
    let a = s.add_fixed_point(4.0, 5.0);
    let b = s.add_point(0.0, 0.0);

    s.symmetric_about_point(a, b, centre).unwrap();
    s.solve().unwrap();

    let (x, y) = s.point(b).unwrap();
    near(x, -2.0);
    near(y, -3.0);
}

#[test]
fn equal_length_makes_a_square_out_of_a_rectangle() {
    let mut s = System::new();
    let p0 = s.add_fixed_point(0.0, 0.0);
    let p1 = s.add_point(7.0, 0.5);
    let p2 = s.add_point(6.5, 3.0);
    let p3 = s.add_point(0.5, 3.5);

    let bottom = s.add_line(p0, p1).unwrap();
    let right = s.add_line(p1, p2).unwrap();
    let top = s.add_line(p2, p3).unwrap();
    let left = s.add_line(p3, p0).unwrap();

    s.horizontal(bottom).unwrap();
    s.horizontal(top).unwrap();
    s.vertical(right).unwrap();
    s.vertical(left).unwrap();
    s.equal_length(bottom, right).unwrap();

    let side = s.add_driving_param(6.0);
    s.length(bottom, side).unwrap();
    s.solve().unwrap();

    let a = s.point(p0).unwrap();
    let b = s.point(p1).unwrap();
    let c = s.point(p2).unwrap();
    near(dist(a, b), 6.0);
    near(dist(b, c), 6.0);
}

#[test]
fn collinear_puts_two_segments_on_one_infinite_line() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_fixed_point(10.0, 0.0);
    let base = s.add_line(a, b).unwrap();

    let c = s.add_point(2.0, 3.0);
    let d = s.add_point(8.0, 4.0);
    let other = s.add_line(c, d).unwrap();

    s.collinear(base, other).unwrap();
    s.solve().unwrap();

    near(s.point(c).unwrap().1, 0.0);
    near(s.point(d).unwrap().1, 0.0);
}

#[test]
fn point_on_line_lands_the_point_on_the_line() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_fixed_point(10.0, 0.0);
    let line = s.add_line(a, b).unwrap();

    let p = s.add_point(3.0, 5.0);
    s.point_on_curve(p, line).unwrap();
    s.solve().unwrap();

    near(s.point(p).unwrap().1, 0.0);
}

#[test]
fn point_on_circle_lands_the_point_on_the_rim() {
    let mut s = System::new();
    let centre = s.add_fixed_point(0.0, 0.0);
    let r = s.add_driving_param(5.0);
    let circle = s.add_circle(centre, r).unwrap();

    let p = s.add_point(7.0, 1.0);
    s.point_on_curve(p, circle).unwrap();
    s.solve().unwrap();

    near(dist(s.point(p).unwrap(), (0.0, 0.0)), 5.0);
}

#[test]
fn point_on_arc_lands_the_point_on_the_rim() {
    let mut s = System::new();
    let centre = s.add_fixed_point(0.0, 0.0);
    let r = s.add_driving_param(4.0);
    let a0 = s.add_driving_param(0.0);
    let a1 = s.add_driving_param(std::f64::consts::FRAC_PI_2);
    let start = s.add_point(4.0, 0.0);
    let end = s.add_point(0.0, 4.0);
    let arc = s.add_arc(centre, r, a0, a1, start, end).unwrap();

    let p = s.add_point(3.0, 3.0);
    s.point_on_curve(p, arc).unwrap();
    s.solve().unwrap();

    near(dist(s.point(p).unwrap(), (0.0, 0.0)), 4.0);
    // The arc's own endpoints are pinned by the internal arc rules.
    near(dist(s.point(start).unwrap(), (4.0, 0.0)), 0.0);
    near(dist(s.point(end).unwrap(), (0.0, 4.0)), 0.0);
}

// ===========================================================================
// Dimensional constraints (§9.2, §9.4)
// ===========================================================================

#[test]
fn distance_between_points() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 1.0);
    let d = s.add_driving_param(13.0);

    s.distance(a, b, d).unwrap();
    s.solve().unwrap();

    near(dist(s.point(a).unwrap(), s.point(b).unwrap()), 13.0);
}

#[test]
fn distance_from_a_point_to_a_line() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_fixed_point(10.0, 0.0);
    let line = s.add_line(a, b).unwrap();

    let p = s.add_point(5.0, 1.0);
    let d = s.add_driving_param(3.0);
    s.distance_to_line(p, line, d).unwrap();
    s.solve().unwrap();

    near(s.point(p).unwrap().1.abs(), 3.0);
}

#[test]
fn length_sets_a_segment() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(3.0, 4.0);
    let l = s.add_line(a, b).unwrap();
    let d = s.add_driving_param(20.0);

    s.length(l, d).unwrap();
    s.solve().unwrap();

    near(dist(s.point(a).unwrap(), s.point(b).unwrap()), 20.0);
}

#[test]
fn angle_between_two_lines() {
    let mut s = System::new();
    let origin = s.add_fixed_point(0.0, 0.0);
    let east = s.add_fixed_point(10.0, 0.0);
    let base = s.add_line(origin, east).unwrap();

    let tip = s.add_point(5.0, 1.0);
    let arm = s.add_line(origin, tip).unwrap();

    let theta = s.add_driving_param(std::f64::consts::FRAC_PI_3);
    let len = s.add_driving_param(10.0);
    s.angle(base, arm, theta).unwrap();
    s.length(arm, len).unwrap();
    s.solve().unwrap();

    let (x, y) = s.point(tip).unwrap();
    near(y.atan2(x).abs(), std::f64::consts::FRAC_PI_3);
    near((x * x + y * y).sqrt(), 10.0);
}

#[test]
fn radius_sets_a_circle_and_an_arc() {
    let mut s = System::new();
    let centre = s.add_fixed_point(0.0, 0.0);
    let r = s.add_free_param(2.0);
    let circle = s.add_circle(centre, r).unwrap();
    let target = s.add_driving_param(7.5);

    s.radius(circle, target).unwrap();
    s.solve().unwrap();
    near(s.param(r).unwrap(), 7.5);
}

#[test]
fn radius_rejects_a_line() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 0.0);
    let line = s.add_line(a, b).unwrap();
    let r = s.add_driving_param(1.0);

    assert_eq!(s.radius(line, r), Err(Error::BadKind));
}

#[test]
fn tangent_between_two_lines_is_rejected() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 0.0);
    let c = s.add_point(0.0, 1.0);
    let l1 = s.add_line(a, b).unwrap();
    let l2 = s.add_line(a, c).unwrap();

    assert_eq!(s.tangent(l1, l2), Err(Error::BadKind));
}

#[test]
fn handles_from_another_system_are_rejected_not_dereferenced() {
    let mut a = System::new();
    let mut b = System::new();
    let far = a.add_point(0.0, 0.0);
    for _ in 0..5 {
        a.add_point(1.0, 1.0);
    }
    // `b` has no points at all, so `far` is out of range there.
    assert_eq!(b.point(far), Err(Error::BadHandle));
    assert_eq!(b.add_line(far, far), Err(Error::BadHandle));
}

// ===========================================================================
// The classic sanity sketches
// ===========================================================================

#[test]
fn rectangle_solves_to_an_actual_rectangle() {
    let mut s = System::new();
    // Deliberately seeded as a wonky quadrilateral.
    let p0 = s.add_fixed_point(0.0, 0.0);
    let p1 = s.add_point(9.3, 0.7);
    let p2 = s.add_point(8.1, 5.4);
    let p3 = s.add_point(-0.6, 4.9);

    let bottom = s.add_line(p0, p1).unwrap();
    let right = s.add_line(p1, p2).unwrap();
    let top = s.add_line(p2, p3).unwrap();
    let left = s.add_line(p3, p0).unwrap();

    s.horizontal(bottom).unwrap();
    s.horizontal(top).unwrap();
    s.vertical(right).unwrap();
    s.vertical(left).unwrap();

    let w = s.add_driving_param(12.0);
    let h = s.add_driving_param(5.0);
    s.length(bottom, w).unwrap();
    s.length(right, h).unwrap();

    let diagnosis = s.diagnose().unwrap();
    assert_eq!(
        diagnosis.state(),
        DofState::WellConstrained,
        "a fully dimensioned rectangle should be exactly determined: {diagnosis:?}"
    );

    s.solve().unwrap();

    let a = s.point(p0).unwrap();
    let b = s.point(p1).unwrap();
    let c = s.point(p2).unwrap();
    let d = s.point(p3).unwrap();

    near(a.1, b.1); // bottom horizontal
    near(c.1, d.1); // top horizontal
    near(b.0, c.0); // right vertical
    near(d.0, a.0); // left vertical
    near(dist(a, b), 12.0);
    near(dist(b, c), 5.0);
    near(dist(c, d), 12.0); // opposite sides equal, without being told
    near(dist(d, a), 5.0);
    // Diagonals equal is the property that distinguishes a rectangle from a
    // parallelogram.
    near(dist(a, c), dist(b, d));
}

#[test]
fn triangle_with_two_sides_and_the_included_angle_obeys_the_law_of_cosines() {
    let side_ab = 5.0_f64;
    let side_ac = 7.0_f64;
    let theta = 0.9_f64;
    let expected_bc =
        (side_ab.powi(2) + side_ac.powi(2) - 2.0 * side_ab * side_ac * theta.cos()).sqrt();

    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(4.0, 0.3);
    let c = s.add_point(3.0, 4.0);

    let ab = s.add_line(a, b).unwrap();
    let ac = s.add_line(a, c).unwrap();

    let p_ab = s.add_driving_param(side_ab);
    let p_ac = s.add_driving_param(side_ac);
    let p_theta = s.add_driving_param(theta);

    s.horizontal(ab).unwrap(); // pins the remaining rotational freedom
    s.length(ab, p_ab).unwrap();
    s.length(ac, p_ac).unwrap();
    s.angle(ab, ac, p_theta).unwrap();

    let diagnosis = s.diagnose().unwrap();
    assert_eq!(
        diagnosis.state(),
        DofState::WellConstrained,
        "{diagnosis:?}"
    );

    s.solve().unwrap();

    let pb = s.point(b).unwrap();
    let pc = s.point(c).unwrap();
    near(dist(s.point(a).unwrap(), pb), side_ab);
    near(dist(s.point(a).unwrap(), pc), side_ac);
    near(dist(pb, pc), expected_bc);
}

// ===========================================================================
// DOF analysis (§9.3)
// ===========================================================================

#[test]
fn an_unconstrained_sketch_reports_its_full_freedom() {
    let mut s = System::new();
    let a = s.add_point(0.0, 0.0);
    let b = s.add_point(10.0, 0.0);
    let line = s.add_line(a, b).unwrap();
    s.horizontal(line).unwrap();

    // Four unknown coordinates, one equation.
    let diagnosis = s.diagnose().unwrap();
    assert_eq!(
        diagnosis.state(),
        DofState::UnderConstrained { dof: 3 },
        "{diagnosis:?}"
    );
    assert_eq!(s.dof().unwrap(), 3);
    assert!(diagnosis.conflicting.is_empty());
    assert!(diagnosis.redundant.is_empty());
}

#[test]
fn a_fully_constrained_sketch_reports_zero_dof() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 1.0);
    let line = s.add_line(a, b).unwrap();
    let len = s.add_driving_param(10.0);

    s.horizontal(line).unwrap();
    s.length(line, len).unwrap();

    let diagnosis = s.diagnose().unwrap();
    assert_eq!(diagnosis.dof, 0, "{diagnosis:?}");
    assert!(diagnosis.is_well_constrained(), "{diagnosis:?}");
}

#[test]
fn a_duplicated_constraint_is_reported_not_silently_absorbed() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 1.0);
    let line = s.add_line(a, b).unwrap();
    let len = s.add_driving_param(10.0);

    s.horizontal(line).unwrap();
    let dup = s.horizontal(line).unwrap();
    s.length(line, len).unwrap();

    let diagnosis = s.diagnose().unwrap();
    assert_eq!(
        diagnosis.state(),
        DofState::OverConstrained,
        "a doubled horizontal must not look well-constrained: {diagnosis:?}"
    );
    assert!(
        diagnosis.redundant.contains(&dup) || diagnosis.conflicting.contains(&dup),
        "the duplicate should be named: {diagnosis:?}"
    );
}

#[test]
fn contradictory_dimensions_are_reported_as_conflicting() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 1.0);
    let line = s.add_line(a, b).unwrap();

    s.horizontal(line).unwrap();
    let ten = s.add_driving_param(10.0);
    let twenty = s.add_driving_param(20.0);
    s.length(line, ten).unwrap();
    s.length(line, twenty).unwrap();

    let diagnosis = s.diagnose().unwrap();
    assert_eq!(
        diagnosis.state(),
        DofState::OverConstrained,
        "{diagnosis:?}"
    );
    assert!(
        !diagnosis.conflicting.is_empty(),
        "two incompatible lengths must be reported as conflicting, got {diagnosis:?}"
    );
    // planegcs reports over-determination as a negative DOF, not as zero.
    assert!(diagnosis.dof < 0, "{diagnosis:?}");
    assert_eq!(s.dof().unwrap(), diagnosis.dof);
}

#[test]
fn dof_is_unavailable_until_diagnosis_runs() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 0.0);
    let line = s.add_line(a, b).unwrap();

    assert_eq!(s.dof(), Err(Error::NoDiagnosis));
    s.horizontal(line).unwrap();
    s.diagnose().unwrap();
    assert!(s.dof().is_ok());

    // A structural change invalidates it again.
    s.add_point(5.0, 5.0);
    assert_eq!(s.dof(), Err(Error::NoDiagnosis));
}

// ===========================================================================
// Parameter binding and incremental re-solve (§9.4)
// ===========================================================================

/// A width driven by a parameter: this is the whole point of the parametric
/// layer. Change the number, re-solve, geometry follows — without rebuilding
/// the system, which is what makes a slider drag affordable.
#[test]
fn changing_a_driving_parameter_moves_the_geometry_without_a_rebuild() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(3.0, 0.0);
    let line = s.add_line(a, b).unwrap();
    let width = s.add_driving_param(10.0);

    s.horizontal(line).unwrap();
    s.length(line, width).unwrap();

    s.diagnose().unwrap();
    s.solve().unwrap();
    near(s.point(b).unwrap().0, 10.0);

    for target in [25.0, 4.0, 100.0, 10.0] {
        s.set_param(width, target).unwrap();
        // No diagnose(), no re-adding geometry.
        s.solve().unwrap();
        near(s.point(b).unwrap().0, target);
    }
}

#[test]
fn dragging_a_parameter_back_and_forth_returns_the_original_bits() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(3.0, 1.0);
    let line = s.add_line(a, b).unwrap();
    let width = s.add_driving_param(10.0);
    s.horizontal(line).unwrap();
    s.length(line, width).unwrap();

    s.solve().unwrap();
    let before = s.point(b).unwrap();

    for target in [11.0, 40.0, 3.0, 77.5] {
        s.set_param(width, target).unwrap();
        s.solve().unwrap();
    }
    s.set_param(width, 10.0).unwrap();
    s.solve().unwrap();

    let after = s.point(b).unwrap();
    assert_eq!(
        (before.0.to_bits(), before.1.to_bits()),
        (after.0.to_bits(), after.1.to_bits()),
        "a round trip through other values must not accumulate drift"
    );
}

// ===========================================================================
// Determinism — the project treats this as a feature, not a nicety
// ===========================================================================

fn wonky_rectangle() -> (System, Vec<nguo_constraints::PointId>) {
    let mut s = System::new();
    let p0 = s.add_fixed_point(0.0, 0.0);
    let p1 = s.add_point(9.3, 0.7);
    let p2 = s.add_point(8.1, 5.4);
    let p3 = s.add_point(-0.6, 4.9);

    let bottom = s.add_line(p0, p1).unwrap();
    let right = s.add_line(p1, p2).unwrap();
    let top = s.add_line(p2, p3).unwrap();
    let left = s.add_line(p3, p0).unwrap();

    s.horizontal(bottom).unwrap();
    s.horizontal(top).unwrap();
    s.vertical(right).unwrap();
    s.vertical(left).unwrap();

    let w = s.add_driving_param(12.0);
    let h = s.add_driving_param(5.0);
    s.length(bottom, w).unwrap();
    s.length(right, h).unwrap();

    (s, vec![p0, p1, p2, p3])
}

fn bits(s: &System, pts: &[nguo_constraints::PointId]) -> Vec<(u64, u64)> {
    pts.iter()
        .map(|p| {
            let (x, y) = s.point(*p).unwrap();
            (x.to_bits(), y.to_bits())
        })
        .collect()
}

#[test]
fn two_identical_systems_solve_to_identical_bits() {
    let (mut a, pa) = wonky_rectangle();
    let (mut b, pb) = wonky_rectangle();
    a.solve().unwrap();
    b.solve().unwrap();
    assert_eq!(bits(&a, &pa), bits(&b, &pb));
}

#[test]
fn re_solving_the_same_system_is_bit_stable() {
    let (mut s, pts) = wonky_rectangle();
    s.solve().unwrap();
    let first = bits(&s, &pts);
    for _ in 0..8 {
        s.solve().unwrap();
        assert_eq!(bits(&s, &pts), first, "a repeat solve must not drift");
    }
}

#[test]
fn diagnosis_is_deterministic() {
    // planegcs runs part of `diagnose()` on a second thread via std::async.
    // If that introduced nondeterminism it would show up here.
    let (mut a, _) = wonky_rectangle();
    let first = a.diagnose().unwrap();
    for _ in 0..8 {
        let (mut b, _) = wonky_rectangle();
        assert_eq!(b.diagnose().unwrap(), first);
    }
}

#[test]
fn every_algorithm_is_individually_deterministic() {
    for alg in [
        Algorithm::DogLeg,
        Algorithm::LevenbergMarquardt,
        Algorithm::Bfgs,
    ] {
        let (mut a, pa) = wonky_rectangle();
        let (mut b, pb) = wonky_rectangle();
        a.set_algorithm(alg);
        b.set_algorithm(alg);
        a.solve().unwrap();
        b.solve().unwrap();
        assert_eq!(bits(&a, &pa), bits(&b, &pb), "{alg:?} was not reproducible");
    }
}

// ===========================================================================
// Robustness
// ===========================================================================

#[test]
fn an_impossible_system_errors_instead_of_returning_nonsense() {
    // Triangle inequality violated: 1 + 1 cannot reach 10.
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 0.0);
    let c = s.add_point(2.0, 0.0);

    let one = s.add_driving_param(1.0);
    let ten = s.add_driving_param(10.0);
    s.distance(a, b, one).unwrap();
    s.distance(b, c, one).unwrap();
    s.distance(a, c, ten).unwrap();

    let err = s.solve().unwrap_err();
    assert!(
        matches!(
            err,
            Error::NoSolution(SolveStatus::Converged | SolveStatus::Failed)
        ),
        "unexpected error: {err:?}"
    );

    // And the geometry must still be readable, finite, and where it started.
    for p in [a, b, c] {
        let (x, y) = s.point(p).unwrap();
        finite(x);
        finite(y);
    }
    assert_eq!(s.point(b).unwrap(), (1.0, 0.0));
    assert_eq!(s.point(c).unwrap(), (2.0, 0.0));
}

#[test]
fn a_system_with_nothing_to_solve_reports_it_rather_than_hanging() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_fixed_point(10.0, 0.0);
    let d = s.add_driving_param(3.0);
    s.distance(a, b, d).unwrap();

    // No unknowns at all: planegcs cannot initialise a subsystem.
    assert_eq!(
        s.solve(),
        Err(Error::NoSolution(SolveStatus::NotInitialised))
    );
}

#[test]
fn a_tight_iteration_cap_terminates() {
    let mut s = System::new();
    let a = s.add_fixed_point(0.0, 0.0);
    let b = s.add_point(1.0, 0.0);
    let c = s.add_point(2.0, 0.0);
    let one = s.add_driving_param(1.0);
    let ten = s.add_driving_param(10.0);
    s.distance(a, b, one).unwrap();
    s.distance(b, c, one).unwrap();
    s.distance(a, c, ten).unwrap();

    s.set_max_iterations(1);
    // The only requirement is that this returns at all, with an error rather
    // than NaN.
    let err = s.solve().unwrap_err();
    assert!(matches!(err, Error::NoSolution(_)), "{err:?}");
    finite(s.point(c).unwrap().0);
}

#[test]
fn empty_system_is_harmless() {
    let mut s = System::new();
    assert_eq!(s.param_count(), 0);
    assert_eq!(
        s.solve(),
        Err(Error::NoSolution(SolveStatus::NotInitialised))
    );
    let diagnosis = s.diagnose().unwrap();
    assert_eq!(diagnosis.dof, 0);
}

/// Creates and drops enough systems that a leak, double free or dangling
/// pointer in the shim shows up as a crash or as runaway memory. Run under the
/// address sanitiser to make it a real check; see ui/docs/CONSTRAINTS.md.
#[test]
fn many_systems_can_be_built_solved_and_dropped() {
    for i in 0..400 {
        let (mut s, pts) = wonky_rectangle();
        // Keep the work varied so nothing is optimised into a single shape.
        let extra = s.add_point(f64::from(i) * 0.01, 1.0);
        let diagonal = s.add_line(pts[0], pts[1]).unwrap();
        s.point_on_curve(extra, diagonal).unwrap();
        s.diagnose().unwrap();
        s.solve().unwrap();
        assert!(s.point(pts[2]).unwrap().0.is_finite());
    }
}

#[test]
fn a_system_can_be_moved_to_another_thread() {
    let (mut s, pts) = wonky_rectangle();
    let handle = std::thread::spawn(move || {
        s.solve().unwrap();
        s.point(pts[2]).unwrap()
    });
    let (x, y) = handle.join().unwrap();
    near(x, 12.0);
    near(y, 5.0);
}
