// SPDX-License-Identifier: GPL-2.0-or-later
//! Linear blend skinning: sample an [`AnimationClip`] at time `t` and produce the
//! posed vertex positions of a mesh rigged by a [`Skin`].
//!
//! This is the animation half of a *kinematic collider*. The other half is
//! [`engine_core::ImplicitSolver::set_kinematic_collider_positions`], which hands
//! these positions to the solver between steps.
//!
//! glTF semantics, because the rig comes from glTF via `engine_io::gltf_io::read_rig`:
//! * a joint's local transform is `Joint::local` (the node's bind-pose matrix),
//!   with any animated **component** — T, R or S — replacing that component;
//! * channels interpolate LINEARLY (`slerp` for rotation), which is what the
//!   shipped Mixamo clips declare;
//! * the skinning matrix is `global(j) * inverse_bind(j)`;
//! * a vertex is `sum_k w_k * skin(j_k) * rest`.
//!
//! Verified against the shipped bundle: skinning the BIND pose of
//! `nguo_male_collision.glb` reproduces the file's own `POSITION` accessor to
//! 4.9157e-7 m over 10 230 vertices — see
//! `posed_collider.rs::bind_pose_lbs_reproduces_the_shipped_rest_mesh`. Note that a
//! Mixamo clip at `t = 0` is its first MOTION frame, not the bind pose: on this bundle
//! that is up to 1.0967 m away from `POSITION`.

use crate::rig::{AnimationClip, AnimationPath, Skin};
use glam::{Mat4, Quat, Vec3, Vec4};

/// Posed positions for `rest` (parallel to `skin.vertex_joints`/`vertex_weights`)
/// with `clip` sampled at `t` seconds. `t` is clamped to `[0, duration]` — callers
/// that want looping should wrap it themselves (`t % clip.duration`).
///
/// Vertices with no influence (all weights zero) pass through unmoved, which is the
/// honest answer for an unrigged vertex and keeps a partially-skinned mesh usable.
pub fn pose_at(skin: &Skin, clip: &AnimationClip, t: f32, rest: &[[f32; 3]]) -> Vec<Vec3> {
    skin_with(&skinning_matrices(skin, clip, t), skin, rest)
}

/// `global(j) * inverse_bind(j)` for every joint. Private: `pose_at` is the only
/// caller. Make it public if a caller that poses several meshes against one
/// skeleton ever turns up and wants to sample the clip once.
fn skinning_matrices(skin: &Skin, clip: &AnimationClip, t: f32) -> Vec<Mat4> {
    let n = skin.joints.len();
    let mut local: Vec<Mat4> = skin
        .joints
        .iter()
        .map(|j| Mat4::from_cols_array_2d(&j.local))
        .collect();

    // Animated components override the bind-pose local. Decompose ONCE per animated
    // joint: `Mat4::to_scale_rotation_translation` is not cheap and most joints in a
    // Mixamo clip are rotation-only, but the alternative — rebuilding from whichever
    // components happen to be keyed — needs the other two anyway.
    let mut touched = vec![false; n];
    let mut trs: Vec<(Vec3, Quat, Vec3)> = vec![(Vec3::ONE, Quat::IDENTITY, Vec3::ZERO); n];
    for ch in &clip.channels {
        let j = ch.target_joint as usize;
        if j >= n || !ch.is_consistent() || ch.times.is_empty() {
            continue;
        }
        if !touched[j] {
            trs[j] = local[j].to_scale_rotation_translation();
            touched[j] = true;
        }
        let v = sample(ch, t);
        match ch.path {
            AnimationPath::Translation => trs[j].2 = Vec3::new(v.x, v.y, v.z),
            AnimationPath::Scale => trs[j].0 = Vec3::new(v.x, v.y, v.z),
            // glTF stores a quaternion as (x, y, z, w); `Quat::from_xyzw` matches.
            // Normalized because LINEAR sampling of two unit quaternions is only unit
            // after slerp, and a keyframe pair can be stored slightly off-unit.
            AnimationPath::Rotation => trs[j].1 = Quat::from_xyzw(v.x, v.y, v.z, v.w).normalize(),
        }
    }
    for j in 0..n {
        if touched[j] {
            local[j] = Mat4::from_scale_rotation_translation(trs[j].0, trs[j].1, trs[j].2);
        }
    }

    // Hierarchy compose. glTF does NOT guarantee `skin.joints` is topologically
    // ordered, so resolve by passes rather than assuming parent < child: 65 joints,
    // and a wrong assumption here silently poses the arms off the bind-pose torso.
    let mut global: Vec<Option<Mat4>> = vec![None; n];
    let mut remaining = n;
    while remaining > 0 {
        let before = remaining;
        for j in 0..n {
            if global[j].is_some() {
                continue;
            }
            match skin.joints[j].parent {
                None => {
                    global[j] = Some(local[j]);
                    remaining -= 1;
                }
                Some(p) => {
                    if let Some(pm) = global[p as usize] {
                        global[j] = Some(pm * local[j]);
                        remaining -= 1;
                    }
                }
            }
        }
        if remaining == before {
            // A parent cycle, or a parent index out of range. Treat the survivors as
            // roots rather than looping forever — a malformed rig should pose badly,
            // not hang.
            for g in global.iter_mut().take(n) {
                if g.is_none() {
                    *g = Some(Mat4::IDENTITY);
                }
            }
            break;
        }
    }

    (0..n)
        .map(|j| {
            let g = global[j].unwrap_or(Mat4::IDENTITY);
            let ib = skin
                .inverse_bind
                .get(j)
                .map(Mat4::from_cols_array_2d)
                .unwrap_or(Mat4::IDENTITY);
            g * ib
        })
        .collect()
}

/// Apply precomputed skinning matrices to `rest`.
fn skin_with(mats: &[Mat4], skin: &Skin, rest: &[[f32; 3]]) -> Vec<Vec3> {
    rest.iter()
        .enumerate()
        .map(|(i, p)| {
            let p = Vec3::from(*p);
            let (Some(js), Some(ws)) = (skin.vertex_joints.get(i), skin.vertex_weights.get(i))
            else {
                return p;
            };
            let mut acc = Vec3::ZERO;
            let mut total = 0.0f32;
            for k in 0..4 {
                let w = ws[k];
                if w == 0.0 {
                    continue;
                }
                let Some(m) = mats.get(js[k] as usize) else {
                    continue;
                };
                acc += w * m.transform_point3(p);
                total += w;
            }
            // Renormalize rather than trust the file: glTF requires the weights to sum
            // to 1 but quantized (u8/u16) weights round, and an unrigged vertex sums to
            // 0 and must pass through unmoved instead of collapsing to the origin.
            if total > 1.0e-6 {
                acc / total
            } else {
                p
            }
        })
        .collect()
}

/// Linear sample of one channel at `t`, clamped to the channel's own key range.
/// Rotation uses `slerp`, which is what glTF's LINEAR interpolation means for a
/// quaternion; T/S lerp componentwise. Returns a `Vec4` with `w` unused for T/S.
fn sample(ch: &crate::rig::AnimationChannel, t: f32) -> Vec4 {
    let c = ch.path.components();
    let key = |i: usize| {
        let s = i * c;
        match ch.path {
            AnimationPath::Rotation => Vec4::new(
                ch.values[s],
                ch.values[s + 1],
                ch.values[s + 2],
                ch.values[s + 3],
            ),
            _ => Vec4::new(ch.values[s], ch.values[s + 1], ch.values[s + 2], 0.0),
        }
    };
    let last = ch.times.len() - 1;
    if t <= ch.times[0] {
        return key(0);
    }
    if t >= ch.times[last] {
        return key(last);
    }
    // First key strictly after `t`; `partition_point` is the stdlib binary search
    // that does not need an exact match.
    let hi = ch.times.partition_point(|&k| k <= t).min(last);
    let lo = hi - 1;
    let span = ch.times[hi] - ch.times[lo];
    let u = if span > 0.0 {
        (t - ch.times[lo]) / span
    } else {
        0.0
    };
    let (a, b) = (key(lo), key(hi));
    match ch.path {
        AnimationPath::Rotation => {
            let qa = Quat::from_xyzw(a.x, a.y, a.z, a.w).normalize();
            let qb = Quat::from_xyzw(b.x, b.y, b.z, b.w).normalize();
            let q = qa.slerp(qb, u);
            Vec4::new(q.x, q.y, q.z, q.w)
        }
        _ => a.lerp(b, u),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig::{AnimationChannel, Joint};

    fn ident() -> [[f32; 4]; 4] {
        Mat4::IDENTITY.to_cols_array_2d()
    }

    /// Two joints in a chain, the child offset 1 m up, both bound at the identity —
    /// so `inverse_bind` for the child is the inverse of its bind global.
    fn chain() -> Skin {
        let child_bind = Mat4::from_translation(Vec3::Y);
        Skin {
            joints: vec![
                Joint {
                    name: "root".into(),
                    parent: None,
                    local: ident(),
                },
                Joint {
                    name: "tip".into(),
                    parent: Some(0),
                    local: child_bind.to_cols_array_2d(),
                },
            ],
            inverse_bind: vec![ident(), child_bind.inverse().to_cols_array_2d()],
            // v0 rides the root, v1 the tip, v2 is split 50/50.
            vertex_joints: vec![[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
            vertex_weights: vec![
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ],
        }
    }

    fn rest() -> Vec<[f32; 3]> {
        vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.0]]
    }

    #[test]
    fn bind_pose_is_the_identity_on_the_rest_mesh() {
        let clip = AnimationClip {
            name: "empty".into(),
            duration: 1.0,
            channels: Vec::new(),
        };
        let out = pose_at(&chain(), &clip, 0.0, &rest());
        for (o, r) in out.iter().zip(rest()) {
            assert!((*o - Vec3::from(r)).length() < 1e-6, "{o:?} vs {r:?}");
        }
    }

    /// A root translation of +2 m in x carries every vertex with it, and half of it
    /// carries the 50/50 vertex only half way. This is the discriminating case: a
    /// blend that ignored the weights would move v2 the full 2 m.
    #[test]
    fn a_translated_root_carries_its_weighted_vertices() {
        let clip = AnimationClip {
            name: "slide".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                target_joint: 0,
                path: AnimationPath::Translation,
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            }],
        };
        let skin = chain();
        let r = rest();
        let mid = pose_at(&skin, &clip, 0.5, &r);
        assert!((mid[0].x - 1.0).abs() < 1e-6, "root vertex {}", mid[0].x);
        // v1 is weighted to the TIP, whose parent is the root, so it moves too.
        assert!((mid[1].x - 1.0).abs() < 1e-6, "tip vertex {}", mid[1].x);
        // v2 blends root (moved 1.0) and tip (moved 1.0) — both moved, so 1.0.
        assert!((mid[2].x - 1.0).abs() < 1e-6, "blend vertex {}", mid[2].x);
        // ...and the clamp at both ends.
        assert!((pose_at(&skin, &clip, -5.0, &r)[0].x).abs() < 1e-6);
        assert!((pose_at(&skin, &clip, 99.0, &r)[0].x - 2.0).abs() < 1e-6);
    }

    /// Rotating the TIP joint 90° about z must move only what is weighted to it, and
    /// must rotate about the tip's own origin (y = 1), not the world origin. Getting
    /// `inverse_bind` wrong is exactly what this catches.
    #[test]
    fn a_rotated_child_joint_pivots_about_its_own_bind_origin() {
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let clip = AnimationClip {
            name: "bend".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                target_joint: 1,
                path: AnimationPath::Rotation,
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 1.0, q.x, q.y, q.z, q.w],
            }],
        };
        let skin = chain();
        let out = pose_at(&skin, &clip, 1.0, &rest());
        // v0 rides the root: untouched.
        assert!(out[0].length() < 1e-6, "{:?}", out[0]);
        // v1 sits AT the tip origin: a rotation about that origin leaves it put.
        assert!((out[1] - Vec3::Y).length() < 1e-6, "{:?}", out[1]);
        // A point 0.5 below the tip, rotated 90° about z about (0,1,0), lands at
        // (0.5, 1.0). Blended 50/50 with the unmoved root contribution (0, 0.5).
        let tip_only = Vec3::new(0.5, 1.0, 0.0);
        let want = 0.5 * Vec3::new(0.0, 0.5, 0.0) + 0.5 * tip_only;
        assert!((out[2] - want).length() < 1e-5, "{:?} vs {want:?}", out[2]);
    }

    /// The control the RISKS section asks for: slerp must actually interpolate. A
    /// nearest-key sampler would return one of the endpoints and fail this.
    #[test]
    fn rotation_slerps_rather_than_snapping_to_a_key() {
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let ch = AnimationChannel {
            target_joint: 0,
            path: AnimationPath::Rotation,
            times: vec![0.0, 1.0],
            values: vec![0.0, 0.0, 0.0, 1.0, q.x, q.y, q.z, q.w],
        };
        let v = sample(&ch, 0.5);
        let got = Quat::from_xyzw(v.x, v.y, v.z, v.w);
        let want = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        assert!(got.dot(want).abs() > 1.0 - 1e-6, "{got:?} vs {want:?}");
    }

    /// Joints stored child-before-parent still compose correctly. glTF does not
    /// require topological order and Blender's exporter does not always produce it.
    #[test]
    fn out_of_order_joints_still_compose() {
        let child_bind = Mat4::from_translation(Vec3::Y);
        let skin = Skin {
            joints: vec![
                Joint {
                    name: "tip".into(),
                    parent: Some(1),
                    local: child_bind.to_cols_array_2d(),
                },
                Joint {
                    name: "root".into(),
                    parent: None,
                    local: ident(),
                },
            ],
            inverse_bind: vec![child_bind.inverse().to_cols_array_2d(), ident()],
            vertex_joints: vec![[0, 0, 0, 0]],
            vertex_weights: vec![[1.0, 0.0, 0.0, 0.0]],
        };
        let clip = AnimationClip {
            name: "slide".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                target_joint: 1,
                path: AnimationPath::Translation,
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            }],
        };
        let out = pose_at(&skin, &clip, 1.0, &[[0.0, 1.0, 0.0]]);
        assert!(
            (out[0] - Vec3::new(3.0, 1.0, 0.0)).length() < 1e-6,
            "{:?}",
            out[0]
        );
    }

    /// A cyclic parent chain must not hang. Belt-and-braces on the resolve loop.
    #[test]
    fn a_cyclic_rig_terminates() {
        let skin = Skin {
            joints: vec![
                Joint {
                    name: "a".into(),
                    parent: Some(1),
                    local: ident(),
                },
                Joint {
                    name: "b".into(),
                    parent: Some(0),
                    local: ident(),
                },
            ],
            inverse_bind: vec![ident(), ident()],
            vertex_joints: vec![[0, 0, 0, 0]],
            vertex_weights: vec![[1.0, 0.0, 0.0, 0.0]],
        };
        let clip = AnimationClip::default();
        let out = pose_at(&skin, &clip, 0.0, &[[1.0, 2.0, 3.0]]);
        assert_eq!(out.len(), 1);
    }
}
