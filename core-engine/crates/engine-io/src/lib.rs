// SPDX-License-Identifier: GPL-2.0-or-later
//! engine-io — mesh & sequence I/O. OBJ + STL + glTF/GLB + USD (`.usda` text, and
//! `.usdc`/`.usdz` binary/package via the `usdcat` tool) + COLLADA (`.dae`) + FBX
//! binary (`.fbx`). glTF is the web/game-engine/DCC interchange standard and the
//! base for rigging/animation support; USD is the VFX/animation format; COLLADA is
//! the legacy DCC XML format; FBX is the Autodesk DCC format. [`golden`] reads
//! the reference capture used for correctness checking against the upstream
//! CUDA solver — an internal corpus not distributed with this source, see
//! `docs/GAUGE_FINDINGS.md`.

pub mod collada;
pub mod fbx;
pub mod gltf_io;
pub mod golden;
pub mod obj;
pub mod stl;
pub mod usd;

pub use collada::{load_collada, parse_collada};
pub use fbx::{load_fbx, parse_fbx};
pub use gltf_io::{load_gltf, parse_gltf_slice, save_glb, write_glb};
pub use golden::{
    compare_frames, load_checkpoint, load_golden_input, read_material, read_param_toml,
    read_pin_indices, GateReport, GoldenInput, GoldenMaterial, GoldenParams,
};
pub use obj::{
    load_obj, load_obj_mesh, parse_obj, parse_obj_mesh, save_obj, write_obj, write_obj_mesh,
    ObjGroup, ObjMesh,
};
pub use stl::{load_stl, save_stl, write_stl_binary};
pub use usd::{load_usd, load_usd_binary, load_usda, parse_usda};

use std::path::Path;

/// Load a triangle mesh by file extension (`.obj`, `.stl`, `.gltf`, `.glb`,
/// `.usda`).
pub fn load_mesh(path: impl AsRef<Path>) -> anyhow::Result<engine_api::Object> {
    let path = path.as_ref();
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("obj") => load_obj(path),
        Some("stl") => load_stl(path),
        Some("gltf") | Some("glb") => load_gltf(path),
        Some("usda") => load_usda(path),
        Some("usdc") | Some("usdz") => load_usd_binary(path),
        Some("usd") => load_usd(path),
        Some("dae") => load_collada(path),
        Some("fbx") => load_fbx(path),
        other => {
            anyhow::bail!(
                "unsupported mesh extension {other:?} (want .obj, .stl, .gltf, .glb, .usda, .usdc, .usdz, .usd, .dae, or .fbx)"
            )
        }
    }
}
