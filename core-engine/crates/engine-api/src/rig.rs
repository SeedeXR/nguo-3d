// SPDX-License-Identifier: GPL-2.0-or-later
//! Rig / skinning / animation data model — carried through glTF import/export so
//! simulated cloth can stay attached to a rigged, animated character (production
//! pipelines: game engines, DCC, animation). Backend-agnostic plain data; the
//! solver does not consume it yet (pass-through), but it round-trips.
//!
//! Matrices are stored **column-major** `[[f32; 4]; 4]` (glam / glTF convention:
//! `m[col][row]`), matching `glam::Mat4::to_cols_array_2d`.

use serde::{Deserialize, Serialize};

/// One joint (bone) in a skin's skeleton.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    pub name: String,
    /// Parent joint index within [`Skin::joints`], or `None` for a root.
    pub parent: Option<u32>,
    /// Local (bind-pose) transform relative to the parent, column-major.
    pub local: [[f32; 4]; 4],
}

/// A skin: joint hierarchy + inverse-bind matrices + per-vertex influences.
/// `inverse_bind` is parallel to `joints`; `vertex_joints`/`vertex_weights` are
/// parallel to the mesh's `positions` (up to 4 influences per vertex).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Skin {
    pub joints: Vec<Joint>,
    pub inverse_bind: Vec<[[f32; 4]; 4]>,
    pub vertex_joints: Vec<[u32; 4]>,
    pub vertex_weights: Vec<[f32; 4]>,
}

/// Which node property an animation channel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnimationPath {
    Translation,
    /// Quaternion `[x, y, z, w]`.
    Rotation,
    Scale,
}

impl AnimationPath {
    /// Number of f32 components per keyframe (3 for T/S, 4 for a quaternion).
    pub fn components(self) -> usize {
        match self {
            AnimationPath::Rotation => 4,
            _ => 3,
        }
    }
}

/// One animation channel: keyframed samples driving a joint's TRS.
/// `values` is flattened — `channels().components()` floats per key, so
/// `times.len() * path.components() == values.len()`. Linear interpolation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationChannel {
    /// Target joint index within the associated [`Skin::joints`].
    pub target_joint: u32,
    pub path: AnimationPath,
    pub times: Vec<f32>,
    pub values: Vec<f32>,
}

impl AnimationChannel {
    /// Whether `times`/`values` lengths are consistent with `path`.
    pub fn is_consistent(&self) -> bool {
        self.times.len() * self.path.components() == self.values.len()
    }
}

/// A named animation clip = a set of channels + total duration (seconds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_skin() -> Skin {
        Skin {
            joints: vec![
                Joint {
                    name: "root".into(),
                    parent: None,
                    local: identity(),
                },
                Joint {
                    name: "bone1".into(),
                    parent: Some(0),
                    local: identity(),
                },
            ],
            inverse_bind: vec![identity(), identity()],
            vertex_joints: vec![[0, 1, 0, 0], [1, 0, 0, 0]],
            vertex_weights: vec![[0.7, 0.3, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        }
    }

    fn identity() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn skin_serde_round_trips() {
        let s = sample_skin();
        let json = serde_json::to_string(&s).unwrap();
        let back: Skin = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn animation_serde_round_trips_and_is_consistent() {
        let clip = AnimationClip {
            name: "spin".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                target_joint: 1,
                path: AnimationPath::Rotation,
                times: vec![0.0, 1.0],
                // 2 keys × 4 (quaternion) = 8 values.
                values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.707, 0.707],
            }],
        };
        assert!(clip.channels[0].is_consistent());
        let json = serde_json::to_string(&clip).unwrap();
        let back: AnimationClip = serde_json::from_str(&json).unwrap();
        assert_eq!(clip, back);
    }

    #[test]
    fn path_component_counts() {
        assert_eq!(AnimationPath::Translation.components(), 3);
        assert_eq!(AnimationPath::Scale.components(), 3);
        assert_eq!(AnimationPath::Rotation.components(), 4);
    }
}
