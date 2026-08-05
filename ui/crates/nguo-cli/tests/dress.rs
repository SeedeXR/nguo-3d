// SPDX-License-Identifier: GPL-2.0-or-later
//! The industrial example, end to end: `dress_example` on the shipped sheath
//! dress.
//!
//! # Which of these can run in CI, and why
//!
//! The first test is **synthetic in the only sense that matters here**: it uses
//! no avatar bundle and no `samples/`, both of which CI cannot see — the bundle
//! is 227 MB of Git-LFS that the `test` job checks out as pointer text, and
//! `samples/` is gitignored. It still runs the real pattern through the real
//! triangulator, the real arranger, the real sewing and the real `.nguo` writer,
//! so everything except the body and the solver is covered unconditionally.
//!
//! The second needs the body. It uses the **loud-skip** pattern: it writes to
//! fd 2 directly (libtest swallows `eprintln!` without `--nocapture`) and then
//! asserts that `MGUSO_REQUIRE_AVATAR_BUNDLE` is *not* set, so `make
//! test-avatars` turns the skip into a failure and CI cannot pass by finding
//! nothing.

use std::io::Write;

use serde_json::{json, Value};

fn run(args: Value) -> Value {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime")
        .block_on(nguo_cli::ops::dispatch(
            "dress_example",
            &args,
            &nguo_cli::ops::Env::default(),
        ))
        .expect("the dress example runs")
}

/// `Some(key)` when the bundle is really there; `None` after a loud skip.
fn avatar_bundle() -> Option<&'static str> {
    match nguo_avatar::Bundle::find() {
        Ok(_) => Some("female"),
        Err(error) => {
            let _ = writeln!(std::io::stderr(), "SKIPPING: no avatar bundle ({error})");
            assert!(
                std::env::var_os("MGUSO_REQUIRE_AVATAR_BUNDLE").is_none(),
                "MGUSO_REQUIRE_AVATAR_BUNDLE is set but the avatar bundle is absent",
            );
            None
        }
    }
}

/// The pattern, the mesh, the seams and the file — no body, so this is the form
/// that runs everywhere.
#[test]
fn the_dress_drafts_sews_and_saves_a_self_describing_nguo_with_no_body() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("dress.nguo");
    let report = run(json!({"edge_mm": 60.0, "out": out.display().to_string()}));

    // --- the pattern, in millimetres --------------------------------------
    let p = &report["pattern"];
    assert_eq!(p["units"], "millimetre");
    assert_eq!(p["pieces"], 2);
    assert_eq!(p["seams"], 2);
    assert_eq!(p["vertices"], 16);
    assert_eq!(p["edges"], 16);
    assert_eq!(p["support_landmark"], "chest");
    assert_eq!(p["seam_allowance_mm"], 15.0);
    // The support arc is the finished bust girth: 1116.03 mm of body + 40 mm of
    // ease, summed over the two panels. If the manifest is re-measured this is
    // the assertion that notices.
    let arc = p["support_arc_mm"].as_f64().expect("arc");
    assert!(
        (arc - 1156.03).abs() < 1e-6,
        "{arc} mm of finished bust girth",
    );

    // --- triangulate + arrange + sew --------------------------------------
    let s = &report["sew"];
    assert_eq!(s["target_edge_mm"], 60.0);
    let (verts, tris, stitches) = (
        s["verts"].as_u64().expect("verts"),
        s["tris"].as_u64().expect("tris"),
        s["stitches"].as_u64().expect("stitches"),
    );
    // Floors and ceilings, not exact counts: the triangulator is free to choose
    // its own interior points, and pinning that here would make every refinement
    // a test failure. What must hold is that the panels really were meshed at
    // roughly the requested edge and that both seams really resolved onto it.
    assert!(
        (400..3_000).contains(&verts) && tris > verts,
        "{verts} verts / {tris} tris is not a meshed pair of panels",
    );
    assert!(
        stitches >= 40,
        "only {stitches} stitches for two side seams"
    );
    assert_eq!(s["seams"].as_array().expect("seams").len(), 2);
    for seam in s["seams"].as_array().expect("seams") {
        assert_eq!(seam["flipped"], false, "a seam is sewn inside out");
        assert!(seam["stitches"].as_u64().expect("stitches") > 4);
    }
    // The seams start open — that is the arranger's slack plus the waist
    // shaping, and it is the number the solver exists to close. A dress that
    // started shut would make the drape assertions vacuous.
    let gap = s["seam_gap_mm"].as_f64().expect("gap");
    assert!(gap > 1_000.0, "the seams start only {gap:.0} mm open");

    // Without a body there is nothing to fit to, nothing to push out of and
    // nothing to drape on — and all three say so rather than reporting zeros.
    for stage in ["body", "resize", "penetration", "drape"] {
        assert_eq!(report[stage], Value::Null, "{stage} ran without a body");
    }

    // --- the `.nguo` -------------------------------------------------------
    let save = &report["save"];
    assert_eq!(save["round_trip_identical"], true);
    let bytes = save["bytes"].as_u64().expect("bytes");
    assert!((1_000..100_000).contains(&bytes), "{bytes} byte .nguo");
    assert_eq!(std::fs::metadata(&out).expect("written").len(), bytes);
    // Exactly the three members D-1 specifies, and it really is a zip.
    let members: Vec<&str> = save["members"]
        .as_array()
        .expect("members")
        .iter()
        .map(|m| m[0].as_str().unwrap_or_default())
        .collect();
    assert_eq!(
        members,
        vec!["document.json", "manifest.json", "preview.svg"]
    );

    // Self-describing: the manifest has to answer "what is this, in what units,
    // in what handedness, against which body" without the rest of the file.
    let m = &save["manifest"];
    assert_eq!(m["format"], "nguo-3d project");
    assert_eq!(m["schema_version"], nguo_doc::SCHEMA_VERSION);
    assert_eq!(m["units"]["geometry"], "millimetre");
    let coords = m["units"]["coordinates"].as_str().expect("coordinates");
    for claim in [
        "millimetres",
        "metres",
        "+Y up",
        "+Z forward",
        "right-handed",
    ] {
        assert!(
            coords.contains(claim),
            "the manifest never states {claim:?}"
        );
    }
    assert_eq!(m["counts"]["pieces"], 2);
    assert_eq!(m["counts"]["seams"], 2);
    assert_eq!(m["counts"]["materials"], 1);
    assert_eq!(m["counts"]["params"], 4);
    // The avatar is REFERENCED, never embedded. The bundle is 227 MB, Git-LFS
    // and CC BY 4.0 — a different licence from this GPL code — so a project file
    // that carried a copy would be redistributing it.
    assert_eq!(m["avatar"]["key"], "female");
    assert_eq!(m["avatar"]["embedded"], false);
    assert!(bytes < 100_000, "{bytes} bytes — is the body in there?");
}

/// A resolution the sewing must survive but that nothing else asserts on:
/// coarse enough for a test, fine enough that both panels get interior points.
const TEST_EDGE_MM: f64 = 120.0;
/// Steps. Measured, not picked: at [`TEST_EDGE_MM`] the seam gap goes
/// 5 506 → 1 980 → 162 mm at steps 0 / 12 / 24 and is flat from there
/// (`--steps 36` finishes at 161 mm), so 24 is where the curve has arrived.
const TEST_STEPS: u32 = 24;

/// The whole point: the dress is **worn**. Real avatar, real solver.
///
/// Three things are asserted and each one has a way of failing that the others
/// do not catch:
///
/// * the seams **close** — the arranger starts them open by the waist shaping
///   and only the stitches can shut them;
/// * **nothing is left inside the body**, measured after the solve by the same
///   `resolve_penetration` that measured it before, not inferred from the drape
///   looking right;
/// * it **stays where it was put** — the hem does not slide, because a 765 mm
///   waist section cannot pass a 1 087 mm hip. Nothing is pinned.
#[test]
fn the_dress_is_worn_by_the_real_avatar_and_its_seams_close() {
    let Some(avatar) = avatar_bundle() else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("worn.nguo");
    let report = run(json!({
        "avatar": avatar,
        "edge_mm": TEST_EDGE_MM,
        "steps": TEST_STEPS,
        "out": out.display().to_string(),
    }));

    // The body came from the manifest, not from a ruler held against the proxy.
    let body = &report["body"];
    assert_eq!(body["tier"], "COL");
    assert_eq!(body["landmark"], "chest");
    let manifest_r = body["manifest_radius_m"].as_f64().expect("radius");
    assert!(
        (manifest_r - 1.11603 / std::f64::consts::TAU).abs() < 1e-6,
        "{manifest_r} m is not the manifest chest girth / 2pi",
    );
    // ...and the reason it must not come from the proxy is *in the report*: a
    // slab through this band catches a T-posed arm 0.93 m off the axis, 5.2x the
    // radius the girths give. This assertion is the trap, kept visible.
    let measured_r = body["measured_max_radius_m"].as_f64().expect("measured");
    assert!(
        measured_r > 3.0 * manifest_r,
        "the T-pose arms have stopped showing up in a slab of this band \
         ({measured_r} m vs {manifest_r} m) — re-read why the girths come from \
         the manifest before relaxing this",
    );

    // A pattern drafted to this body needs no resize, and the proposal proves it
    // rather than the comment claiming it.
    assert_eq!(report["resize"]["applied"], false);
    assert_eq!(report["resize"]["factor"], 1.0);

    // The arranger necessarily starts the cloth on a cylinder of the *chest*
    // radius, so a few vertices sit inside the wider-than-circular hip. They are
    // pushed out before the solver ever sees them, and that has to converge.
    let pen = &report["penetration"];
    assert!(
        pen["inside_before"].as_u64().expect("before") > 0,
        "nothing started inside the body — resolve_penetration is doing nothing \
         here and this test would not notice if it broke: {pen}",
    );
    assert_eq!(pen["inside_after"], 0);
    assert_eq!(pen["converged"], true);

    let drape = &report["drape"];
    assert_eq!(drape["all_finite"], true);
    assert_eq!(drape["frames_emitted"], TEST_STEPS);
    assert_eq!(drape["engine"]["self_collision"], true);

    // 1. The seams close.
    let gap = &drape["seam_gap_mm"];
    let (start, end) = (
        gap["first_frame"].as_f64().expect("start"),
        gap["final"].as_f64().expect("end"),
    );
    let _ = writeln!(
        std::io::stderr(),
        "DRESS {} cloth verts, {} scene verts, {} stitches, {TEST_STEPS} steps: \
         {:.0} ms/step, seams {start:.0} -> {end:.0} mm ({:.1} % shut)",
        report["sew"]["verts"],
        drape["engine"]["verts"],
        report["sew"]["stitches"],
        drape["engine"]["median_step_s"].as_f64().unwrap_or(0.0) * 1000.0,
        gap["closed_pct"].as_f64().unwrap_or(0.0),
    );
    assert!(start > 1_000.0, "the seams did not start open: {start} mm");
    assert!(
        end < 0.1 * start,
        "the seams barely closed: {start:.0} -> {end:.0} mm",
    );

    // 2. Nothing is inside the body afterwards.
    assert_eq!(
        drape["inside_body_after_drape"], 0,
        "the dress ended up inside the body: {drape}",
    );

    // 3. It is worn where it was drafted to be worn. The hem starts at the
    // `calf` landmark (0.57143 m) and nothing pins it there.
    let hem = drape["hem_y_m"].as_f64().expect("hem");
    assert!(
        (hem - 0.57143).abs() < 0.02,
        "the dress moved to {hem:.4} m — it should still be at the calf line",
    );

    // The saved file records the body it was drafted against, by reference.
    let m = &report["save"]["manifest"];
    assert_eq!(report["save"]["round_trip_identical"], true);
    assert_eq!(m["avatar"]["key"], avatar);
    assert_eq!(m["avatar"]["tier"], "COL");
    assert_eq!(m["avatar"]["embedded"], false);
    assert!(
        m["avatar"]["bundle_version"].is_string(),
        "the .nguo does not say which bundle version it was drafted against",
    );
}
