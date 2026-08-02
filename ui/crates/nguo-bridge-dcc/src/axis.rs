// SPDX-License-Identifier: GPL-2.0-or-later
//! Coordinate-system conversion between nguo-3d and the two DCCs.
//!
//! # The three conventions (all observed, none recalled)
//!
//! * **nguo-3d** — right-handed, **+Y up**. `engine-api`'s default gravity is
//!   `[0.0, -9.8, 0.0]` (`core-engine/crates/engine-api/src/params.rs:99`), and
//!   `engine-io`'s glTF writer stores positions, normals and UVs **verbatim**
//!   (`gltf_io.rs:326` `write_glb`, `gltf_io.rs:161` "UVs (TEXCOORD_0),
//!   untransformed"). nguo space is therefore *identical to glTF space*:
//!   +Y up, +Z forward, −X right, UV origin top-left (v grows downwards).
//! * **Blender** — right-handed, **+Z up**, −Y forward.
//! * **Unity** — **left-handed**, +Y up, **+Z forward**, +X right.
//!
//! # nguo -> Blender
//!
//! Measured, not assumed. Blender 5.2.0 LTS, `bpy.ops.wm.obj_export` at its
//! defaults (`forward_axis='NEGATIVE_Z'`, `up_axis='Y'`) exported the three
//! Blender basis vertices as:
//!
//! ```text
//!   blender (1,0,0) -> file (1, 0,  0)
//!   blender (0,1,0) -> file (0, 0, -1)
//!   blender (0,0,1) -> file (0, 1,  0)
//! ```
//!
//! so `nguo = (b.x, b.z, -b.y)` and `blender = (n.x, -n.z, n.y)`. That is a
//! −90° rotation about X: `det = +1`, **handedness preserved, winding
//! unchanged**.
//!
//! # nguo -> Unity
//!
//! glTF declares +Z as forward and −X as right; Unity declares +Z forward and
//! +X right. The conversion that keeps *up* and *forward* meaning the same
//! thing is therefore **negate X**: `unity = (-n.x, n.y, n.z)`. This is also
//! what Unity's own glTFast >= 5.0 does (it changed from negate-Z to negate-X
//! precisely to conform to the spec's "+Z is forward"), so a GLB imported via
//! glTFast and an OBJ sequence imported via our own reader land in the same
//! place. `det = -1`: **handedness flips, so triangle winding MUST flip** or
//! the garment renders inside-out.
//!
//! # Normals
//!
//! Every matrix here is orthogonal, so `M^-T == M` and the *same* call converts
//! positions and normals. For the Unity mirror there is a second subtlety: a
//! `det < 0` map negates the geometric normal of a triangle, and flipping the
//! winding negates it back. So "flip winding **and** transform the shading
//! normal by M" is the only combination in which the shading normal still
//! agrees with the geometric one. `nguo_to_unity_mesh` does both.
//!
//! # UVs
//!
//! glTF (and therefore nguo) puts the UV origin at the **top-left**, v growing
//! down. Blender and Unity both put it at the **bottom-left**. Both directions
//! are `v -> 1 - v`.

/// Row-major 3x3 orthogonal basis change.
pub type Basis = [[f32; 3]; 3];

/// nguo (Y-up RH) -> Blender (Z-up RH). Rx(-90 deg), `det = +1`.
pub const NGUO_TO_BLENDER: Basis = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
/// Blender (Z-up RH) -> nguo (Y-up RH). Rx(+90 deg), `det = +1`.
pub const BLENDER_TO_NGUO: Basis = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]];
/// nguo (RH) -> Unity (LH). Negate X, `det = -1` — winding must flip.
pub const NGUO_TO_UNITY: Basis = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// Unity (LH) -> nguo (RH). Negate X is an involution, so this is the same map.
pub const UNITY_TO_NGUO: Basis = NGUO_TO_UNITY;

/// `m * v`. Valid for positions and, because every `Basis` here is orthogonal
/// (`M^-T == M`), for normals too.
#[must_use]
pub fn apply(m: &Basis, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Determinant. `+1` = rotation, `-1` = mirror.
#[must_use]
pub fn det(m: &Basis) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// A `det < 0` basis change mirrors the mesh; triangle winding must be
/// reversed to keep front faces facing front.
#[must_use]
pub fn flips_winding(m: &Basis) -> bool {
    det(m) < 0.0
}

/// Reverse a triangle's winding iff `m` mirrors.
#[must_use]
pub fn wind(m: &Basis, t: [u32; 3]) -> [u32; 3] {
    if flips_winding(m) {
        [t[0], t[2], t[1]]
    } else {
        t
    }
}

/// glTF/nguo UV origin (top-left) <-> Blender/Unity UV origin (bottom-left).
/// Its own inverse.
#[must_use]
pub fn flip_uv_v(uv: [f32; 2]) -> [f32; 2] {
    [uv[0], 1.0 - uv[1]]
}

/// Convert a whole mesh in place-ish: positions, normals, UVs and winding, all
/// consistently. This is the entry point callers should use — getting three of
/// the four right is the classic silent interop bug.
pub fn convert_mesh(
    m: &Basis,
    positions: &mut [[f32; 3]],
    normals: &mut [[f32; 3]],
    uvs: &mut [[f32; 2]],
    tris: &mut [[u32; 3]],
) {
    for p in positions.iter_mut() {
        *p = apply(m, *p);
    }
    for n in normals.iter_mut() {
        *n = apply(m, *n);
    }
    for uv in uvs.iter_mut() {
        *uv = flip_uv_v(*uv);
    }
    if flips_winding(m) {
        for t in tris.iter_mut() {
            *t = [t[0], t[2], t[1]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Geometric normal of a triangle, from the winding.
    fn geo_normal(p: &[[f32; 3]], t: [u32; 3]) -> [f32; 3] {
        let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// The exact triple measured from Blender 5.2.0's OBJ exporter at defaults.
    #[test]
    fn blender_basis_matches_measured_obj_export() {
        assert_eq!(apply(&BLENDER_TO_NGUO, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
        assert_eq!(apply(&BLENDER_TO_NGUO, [0.0, 1.0, 0.0]), [0.0, 0.0, -1.0]);
        assert_eq!(apply(&BLENDER_TO_NGUO, [0.0, 0.0, 1.0]), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn semantic_axes_survive_the_trip() {
        // up
        assert_eq!(apply(&NGUO_TO_BLENDER, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(apply(&NGUO_TO_UNITY, [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]);
        // forward: glTF/nguo +Z forward -> Blender -Y forward, Unity +Z forward
        assert_eq!(apply(&NGUO_TO_BLENDER, [0.0, 0.0, 1.0]), [0.0, -1.0, 0.0]);
        assert_eq!(apply(&NGUO_TO_UNITY, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]);
        // right: glTF/nguo -X right -> Unity +X right
        assert_eq!(apply(&NGUO_TO_UNITY, [-1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn handedness_is_what_we_claim() {
        assert_eq!(det(&NGUO_TO_BLENDER), 1.0);
        assert_eq!(det(&BLENDER_TO_NGUO), 1.0);
        assert_eq!(det(&NGUO_TO_UNITY), -1.0);
        assert!(!flips_winding(&NGUO_TO_BLENDER));
        assert!(flips_winding(&NGUO_TO_UNITY));
    }

    #[test]
    fn round_trip_positions_and_normals_are_identity() {
        let samples = [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-0.5, 12.25, -7.125],
            [1e-3, -1e3, 0.25],
        ];
        for v in samples {
            assert_eq!(apply(&BLENDER_TO_NGUO, apply(&NGUO_TO_BLENDER, v)), v);
            assert_eq!(apply(&NGUO_TO_BLENDER, apply(&BLENDER_TO_NGUO, v)), v);
            assert_eq!(apply(&UNITY_TO_NGUO, apply(&NGUO_TO_UNITY, v)), v);
            assert_eq!(apply(&NGUO_TO_UNITY, apply(&UNITY_TO_NGUO, v)), v);
        }
    }

    #[test]
    fn round_trip_uvs_are_identity() {
        for uv in [[0.0, 0.0], [1.0, 1.0], [0.25, 0.75], [0.5, 0.5]] {
            assert_eq!(flip_uv_v(flip_uv_v(uv)), uv);
        }
        // and the flip is a real flip, not a no-op
        assert_eq!(flip_uv_v([0.25, 0.75]), [0.25, 0.25]);
    }

    #[test]
    fn round_trip_winding_is_identity() {
        let t = [0u32, 1, 2];
        assert_eq!(wind(&BLENDER_TO_NGUO, wind(&NGUO_TO_BLENDER, t)), t);
        assert_eq!(wind(&UNITY_TO_NGUO, wind(&NGUO_TO_UNITY, t)), t);
        // Blender preserves handedness, so it must NOT touch winding.
        assert_eq!(wind(&NGUO_TO_BLENDER, t), t);
        // Unity mirrors, so it must.
        assert_eq!(wind(&NGUO_TO_UNITY, t), [0, 2, 1]);
    }

    /// The bug this whole module exists to prevent: after conversion the
    /// triangle's *geometric* normal must still point the same way as its
    /// *shading* normal. Without the winding flip on the Unity path it points
    /// the opposite way and the garment renders inside-out.
    #[test]
    fn converted_mesh_faces_the_same_way_it_did_before() {
        // CCW triangle in the XZ plane, normal pointing at +Y.
        let base_pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
        let base_nrm = [[0.0, 1.0, 0.0]; 3];
        let base_tri = [0u32, 1, 2];
        assert!(
            dot(geo_normal(&base_pos, base_tri), base_nrm[0]) > 0.0,
            "fixture is wound the wrong way"
        );

        for m in [&NGUO_TO_BLENDER, &NGUO_TO_UNITY] {
            let mut p = base_pos;
            let mut n = base_nrm;
            let mut uv = [[0.0f32, 0.0f32]; 3];
            let mut t = [base_tri];
            convert_mesh(m, &mut p, &mut n, &mut uv, &mut t);
            assert!(
                dot(geo_normal(&p, t[0]), n[0]) > 0.0,
                "converted mesh is inside-out for det={}",
                det(m)
            );
        }
    }

    /// Skipping the winding flip really does break it — proof the assertion
    /// above has teeth.
    #[test]
    fn unity_without_winding_flip_is_inside_out() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
        let tri = [0u32, 1, 2];
        let mirrored: Vec<[f32; 3]> = pos.iter().map(|p| apply(&NGUO_TO_UNITY, *p)).collect();
        let n = apply(&NGUO_TO_UNITY, [0.0, 1.0, 0.0]);
        assert!(dot(geo_normal(&mirrored, tri), n) < 0.0);
    }

    #[test]
    fn full_mesh_round_trip_nguo_dcc_nguo() {
        let pos0 = [[0.1, 2.0, -3.0], [1.0, -0.5, 4.0], [-2.0, 0.0, 0.5]];
        let nrm0 = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let uv0 = [[0.0, 0.0], [1.0, 0.25], [0.5, 1.0]];
        let tri0 = [[0u32, 1, 2]];

        for (out, back) in [
            (&NGUO_TO_BLENDER, &BLENDER_TO_NGUO),
            (&NGUO_TO_UNITY, &UNITY_TO_NGUO),
        ] {
            let (mut p, mut n, mut uv, mut t) = (pos0, nrm0, uv0, tri0);
            convert_mesh(out, &mut p, &mut n, &mut uv, &mut t);
            convert_mesh(back, &mut p, &mut n, &mut uv, &mut t);
            assert_eq!(p, pos0);
            assert_eq!(n, nrm0);
            assert_eq!(uv, uv0);
            assert_eq!(t, tri0);
        }
    }
}
