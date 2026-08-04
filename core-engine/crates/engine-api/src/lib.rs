// SPDX-License-Identifier: GPL-2.0-or-later
//! engine-api — authoring surface for ppf-mguso-solver (replaces the Python
//! frontend). Build a [`Scene`] of [`Object`]s with [`Material`]s and
//! [`SolverConfig`], then `scene.build()` to get a runnable `engine_core::Solver`.

pub mod auto_drape;
pub mod avatar;
pub mod material;
pub mod params;
pub mod resize;
pub mod rig;
pub mod scene;

pub use auto_drape::{auto_drape, Aabb, DrapePreset};
pub use avatar::{
    avatar_collider, avatar_collider_near, drape_on_avatar, AvatarError, GarmentPlacement,
};
pub use material::{FabricPreset, Material};
pub use params::{Barrier, Model, SolverConfig};
pub use resize::{apply_resize, plan_resize, BodyGirth, ResizeCheck, ResizePlan};
pub use rig::{AnimationChannel, AnimationClip, AnimationPath, Joint, Skin};
pub use scene::{ColliderSpec, Object, Scene};
