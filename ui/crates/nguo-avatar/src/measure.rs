// SPDX-License-Identifier: GPL-2.0-or-later
//! Body measurements, in the app's canonical millimetres.
//!
//! # There is no measurement rig here, on purpose
//! The bundle ships **measured** girths: `manifest.avatars.<sex>.measurements
//! .<name>.circumference_m`, taken off the source geometry as the convex-hull
//! perimeter of a horizontal cross-section — what a taut tape reads — with
//! upper-body girths restricted to the torso so a T-pose chest slice does not
//! also wrap both arms. Recomputing that here from the decimated runtime mesh
//! would produce a second, worse answer to a question already answered by the
//! pipeline that owns the geometry. So this module is a unit conversion and a
//! set of names, and nothing else.
//!
//! # One conversion point
//! The manifest is metres, [`Length`] is millimetres. [`Measurements`] is the
//! only place metres become millimetres in this crate.

use crate::manifest::Avatar;
use nguo_measure::Length;
use std::collections::BTreeMap;

/// Metres to the canonical millimetre. **The only conversion in this crate.**
fn mm(metres: f64) -> Length {
    Length::from_mm(metres * 1000.0)
}

/// The nine girths the bundle measures, as types rather than strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Landmark {
    Neck,
    /// Just below the armpit; within 2 mm of [`Landmark::Chest`] on both bodies.
    UpperChest,
    /// Chest on the male body, bust on the female.
    Chest,
    Underbust,
    /// The narrowest point between ribs and pelvis.
    Waist,
    HighHip,
    Hip,
    Thigh,
    Calf,
}

impl Landmark {
    /// Every measured landmark, top of the body down.
    pub const ALL: [Landmark; 9] = [
        Landmark::Neck,
        Landmark::UpperChest,
        Landmark::Chest,
        Landmark::Underbust,
        Landmark::Waist,
        Landmark::HighHip,
        Landmark::Hip,
        Landmark::Thigh,
        Landmark::Calf,
    ];

    /// The `measurements` key in the manifest.
    pub fn key(self) -> &'static str {
        match self {
            Landmark::Neck => "neck",
            Landmark::UpperChest => "upper_chest",
            Landmark::Chest => "chest",
            Landmark::Underbust => "underbust",
            Landmark::Waist => "waist",
            Landmark::HighHip => "high_hip",
            Landmark::Hip => "hip",
            Landmark::Thigh => "thigh",
            Landmark::Calf => "calf",
        }
    }
}

/// A girth and the height it was taken at, so a waistband can be placed at the
/// measured waist rather than a guessed fraction of stature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Girth {
    pub circumference: Length,
    /// Height above the ground plane. Feet sit on it, so this is also the
    /// height in the scene.
    pub height: Length,
    /// `height / stature`. Absent for thigh and calf.
    pub height_fraction: Option<f64>,
}

/// One avatar's measurements in millimetres.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurements {
    /// Bounding height of the body. 1.830 m male, 1.905 m female — the female
    /// sculpt really is 7.5 cm taller.
    pub stature: Length,
    pub arm_span: Length,
    /// Crotch height, from the skeleton block.
    pub inseam: Length,
    /// Hip joint centre height, from the skeleton block.
    pub hip_height: Length,
    girths: BTreeMap<String, Girth>,
}

impl Measurements {
    /// Convert an avatar's manifest entry. Infallible: an absent landmark is
    /// reported by [`Measurements::girth`] returning `None`, not by failing the
    /// whole conversion.
    pub fn from_avatar(avatar: &Avatar) -> Measurements {
        Measurements {
            stature: mm(avatar.stature_m),
            arm_span: mm(avatar.arm_span_m),
            inseam: mm(avatar.skeleton.inseam_m),
            hip_height: mm(avatar.skeleton.hip_height_m),
            girths: avatar
                .measurements
                .iter()
                .map(|(name, m)| {
                    (
                        name.clone(),
                        Girth {
                            circumference: mm(m.circumference_m),
                            height: mm(m.height_m),
                            height_fraction: m.height_fraction,
                        },
                    )
                })
                .collect(),
        }
    }

    pub fn girth(&self, landmark: Landmark) -> Option<Girth> {
        self.girths.get(landmark.key()).copied()
    }

    /// Every girth the manifest carried, keyed by its manifest name — including
    /// any this crate has no [`Landmark`] for yet.
    pub fn girths(&self) -> impl Iterator<Item = (&str, &Girth)> {
        self.girths.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metres_become_millimetres_exactly() {
        assert_eq!(mm(1.83).mm(), 1830.0);
        assert_eq!(mm(0.85004).mm(), 850.04);
        assert_eq!(mm(0.0), Length::ZERO);
    }

    #[test]
    fn landmark_keys_are_unique() {
        let mut keys: Vec<_> = Landmark::ALL.iter().map(|l| l.key()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), Landmark::ALL.len());
    }
}
