// SPDX-License-Identifier: GPL-2.0-or-later
//! Volumetric parity groundwork: build our tet elements from a REAL reference
//! (gauge) scene and validate them against the reference's own reported statistics.
//! Skips gracefully when the gauge isn't present.

// One implementation, in engine-io. This copy used `exists()` for the env-var override,
// which is true for a FILE too. See `engine_io::golden::gauge_dir`.
use engine_io::golden::gauge_dir;

/// The reference's `plastic` scene: 12,670 tets, mean rest volume 5.1503e-6 and max
/// 2.6277e-5 per its `param_summary.txt`. Building our `TetElement`s from that rest
/// state must reproduce those statistics — which validates the index decode, the
/// winding normalization, and the rest-volume computation together, against data we
/// did not produce.
#[test]
fn builds_valid_tet_elements_from_reference_scene() {
    let Some(gauge) = gauge_dir() else {
        eprintln!("no gauge dir; skipping");
        return;
    };
    let scene = gauge.join("goldens/plastic");
    if !scene.exists() {
        eprintln!("plastic scene absent; skipping");
        return;
    }
    let input = engine_io::golden::load_golden_input(&scene).expect("load plastic");
    assert_eq!(input.tets.len(), 12_670);

    let rest: Vec<glam::Vec3> = input
        .positions
        .iter()
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();
    let els = engine_core::tet::build_tets(
        &input.tets,
        &rest,
        1.0e-12,
        engine_core::tet::TetModel::Arap,
    );
    // The reference mesh is well-formed: essentially nothing should be rejected.
    assert!(
        els.len() * 100 >= input.tets.len() * 99,
        "too many reference tets rejected as degenerate: {} of {}",
        els.len(),
        input.tets.len()
    );
    assert!(
        els.iter().all(|e| e.volume > 0.0),
        "winding normalization must give positive rest volumes"
    );

    // Compare against the reference's REPORTED volume statistics.
    let total: f32 = els.iter().map(|e| e.volume).sum();
    let mean = total / els.len() as f32;
    let max = els.iter().map(|e| e.volume).fold(0.0f32, f32::max);
    assert!(
        (mean - 5.1503e-6).abs() < 0.02 * 5.1503e-6,
        "mean rest volume {mean} != param_summary's 5.1503e-6"
    );
    assert!(
        (max - 2.6277e-5).abs() < 0.05 * 2.6277e-5,
        "max rest volume {max} != param_summary's 2.6277e-5"
    );

    // The reference rest state must be stress-free under our energy.
    let pinned = vec![false; rest.len()];
    let asm = engine_core::tet_assembler::TetAssembler::new(&els, rest.len(), &pinned);
    let e_rest = asm.energy(&rest, 100.0, 200.0);
    assert!(
        e_rest.abs() < 1e-4,
        "reference rest state should be stress-free, got energy {e_rest}"
    );

    // And the assembled gradient at rest must vanish (no phantom forces).
    let (_v, g) = asm.assemble(&rest, 100.0, 200.0);
    let gmax = g.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(gmax < 1e-3, "phantom force at reference rest state: {gmax}");
}
