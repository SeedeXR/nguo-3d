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

/// First line of a git-LFS pointer file, verbatim
/// (<https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md>).
pub const LFS_POINTER_MAGIC: &[u8] = b"version https://git-lfs.github.com/spec/v1";

/// True when `path` holds a git-LFS **pointer stub** rather than the real file.
///
/// A clone made without `git lfs pull`, or a CI checkout without `lfs: true`,
/// leaves ~130 bytes of pointer text where every LFS-tracked asset should be.
/// The file *exists*, so `Path::is_file` says yes and every presence gate built
/// on it is wrong: the bundle looks present while none of its geometry is, and
/// the format reader then fails with a parse error naming the format instead of
/// the actual problem. Callers that gate on presence must treat a pointer as an
/// **absent** file; callers that load must report it as a pointer.
///
/// Cheap: one open and a 42-byte read, no allocation.
pub fn is_lfs_pointer(path: impl AsRef<Path>) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = [0u8; LFS_POINTER_MAGIC.len()];
    f.read_exact(&mut head).is_ok() && head == LFS_POINTER_MAGIC
}

/// The unit a mesh file's coordinates are written in.
///
/// This engine works in **metres** everywhere, and no mesh format reliably records its
/// own unit — OBJ has no unit field at all, and glTF only *specifies* metres without
/// anything enforcing it. So the unit is the caller's to declare, and
/// [`load_mesh_in`] converts once at the boundary.
///
/// It cannot be inferred from size, which is why there is no `Auto`. Measured on the
/// sample corpus: four garments are centimetres, one is metres, and `tote-bag` is
/// millimetres — its 297 units are a believable 0.297 m bag and an absurd 2.97 m one,
/// and nothing about the numbers distinguishes those. Guessing would silently produce a
/// 100× garment, which is [`crate`]'s worst failure mode because everything downstream
/// still "works".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshUnits {
    /// Already metres — the engine's convention, and what the avatar bundle ships.
    #[default]
    Metres,
    /// Centimetres. Most DCC garment exports, including four of the `samples/` meshes.
    Centimetres,
    /// Millimetres. Also what authored 2D patterns use (see `nguo_triangulate::MM_TO_M`).
    Millimetres,
    /// Inches, for imperial CAD sources.
    Inches,
}

impl MeshUnits {
    /// Factor that converts a coordinate in this unit to metres.
    pub fn to_metres(self) -> f32 {
        match self {
            MeshUnits::Metres => 1.0,
            MeshUnits::Centimetres => 1.0e-2,
            MeshUnits::Millimetres => 1.0e-3,
            MeshUnits::Inches => 0.0254,
        }
    }
}

/// Load a triangle mesh and convert it from `units` to metres.
///
/// The conversion happens exactly here and scales `positions` only: normals are
/// direction vectors and a uniform positive scale leaves them unchanged, and UVs are
/// dimensionless. A skin's `inverse_bind` matrices carry translations in the source
/// unit, so a rigged mesh loaded in anything but [`MeshUnits::Metres`] would need those
/// converted too — that is not done here, and [`MeshUnits::Metres`] (the default, and
/// what every rigged asset in this project uses) is unaffected.
pub fn load_mesh_in(
    path: impl AsRef<Path>,
    units: MeshUnits,
) -> anyhow::Result<engine_api::Object> {
    let mut o = load_mesh(path)?;
    let s = units.to_metres();
    if s != 1.0 {
        for p in &mut o.positions {
            for c in p.iter_mut() {
                *c *= s;
            }
        }
        if o.skin.is_some() {
            tracing::warn!(
                object = %o.name,
                ?units,
                "rigged mesh scaled to metres: `inverse_bind` translations were NOT \
                 converted and the skin will be wrong"
            );
        }
        tracing::info!(object = %o.name, ?units, scale = s, "mesh converted to metres");
    }
    Ok(o)
}

/// Load a triangle mesh by file extension (`.obj`, `.stl`, `.gltf`, `.glb`,
/// `.usda`).
///
/// Coordinates are taken **as written**, i.e. assumed to be metres already. For a file
/// in any other unit use [`load_mesh_in`] — a centimetre mesh loaded here is a 100×
/// object and nothing downstream will complain.
pub fn load_mesh(path: impl AsRef<Path>) -> anyhow::Result<engine_api::Object> {
    let path = path.as_ref();
    // Before dispatch, so every format gets the actionable message instead of
    // its own "this is not a valid X" for 130 bytes of pointer text.
    anyhow::ensure!(
        !is_lfs_pointer(path),
        "{} is a git-LFS pointer, not the file itself — run `git lfs pull` \
         (in CI, check out with `lfs: true`)",
        path.display()
    );
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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::io::Write;

    /// One triangle with coordinates of 100, so a cm→m conversion is unmistakable.
    fn tri_obj(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("nguo_units_{tag}.obj"));
        let mut f = std::fs::File::create(&p).expect("temp obj");
        writeln!(f, "v 0 0 0\nv 100 0 0\nv 0 100 0\nf 1 2 3").expect("write obj");
        p
    }

    #[test]
    fn factors_are_exact_and_metres_is_the_default() {
        assert_eq!(MeshUnits::default(), MeshUnits::Metres);
        assert_eq!(MeshUnits::Metres.to_metres(), 1.0);
        assert_eq!(MeshUnits::Centimetres.to_metres(), 1.0e-2);
        assert_eq!(MeshUnits::Millimetres.to_metres(), 1.0e-3);
        assert_eq!(MeshUnits::Inches.to_metres(), 0.0254);
        // Agrees with the pattern path's `nguo_triangulate::MM_TO_M`, which is f64 1e-3.
        // NOT an exact comparison: the nearest f32 to 1e-3 is 0.001000000047497451, so
        // `1.0e-3f32 as f64 == 1.0e-3f64` is false. A garment 1000x off is the failure
        // being guarded against, and half an ULP of f32 is nowhere near it.
        assert!((MeshUnits::Millimetres.to_metres() as f64 - 1.0e-3).abs() < 1.0e-9);
    }

    #[test]
    fn load_mesh_in_scales_positions_and_load_mesh_does_not() {
        let p = tri_obj("scale");

        // `load_mesh` is unchanged: coordinates as written.
        let raw = load_mesh(&p).expect("load raw");
        assert_eq!(raw.positions[1], [100.0, 0.0, 0.0]);

        // Metres is a genuine no-op, not a 1.0 multiply that could drift.
        let m = load_mesh_in(&p, MeshUnits::Metres).expect("load m");
        assert_eq!(m.positions, raw.positions);

        for (u, want) in [
            (MeshUnits::Centimetres, 1.0f32),
            (MeshUnits::Millimetres, 0.1),
            (MeshUnits::Inches, 2.54),
        ] {
            let o = load_mesh_in(&p, u).expect("load scaled");
            assert!(
                (o.positions[1][0] - want).abs() < 1e-6,
                "{u:?}: x {} want {want}",
                o.positions[1][0]
            );
            // Topology must survive the conversion untouched.
            assert_eq!(o.tris, raw.tris, "{u:?} changed the triangles");
            assert_eq!(o.positions.len(), raw.positions.len());
        }
        std::fs::remove_file(&p).ok();
    }

    /// The failure this type exists to prevent: a centimetre garment loaded as metres is
    /// a 100x object, and every downstream stage still reports success.
    #[test]
    fn a_centimetre_mesh_read_as_metres_is_a_hundred_times_too_big() {
        let p = tri_obj("wrong");
        let wrong = load_mesh(&p).expect("as metres");
        let right = load_mesh_in(&p, MeshUnits::Centimetres).expect("as cm");
        let span = |o: &engine_api::Object| {
            o.positions
                .iter()
                .map(|q| q[0])
                .fold(f32::NEG_INFINITY, f32::max)
        };
        assert!((span(&wrong) / span(&right) - 100.0).abs() < 1e-3);
        std::fs::remove_file(&p).ok();
    }
}

#[cfg(test)]
mod lfs_tests {
    use super::*;

    /// The exact bytes `actions/checkout` leaves behind without `lfs: true`.
    const POINTER: &str = "version https://git-lfs.github.com/spec/v1\n\
                           oid sha256:98f985ffb4f05e36ffe919ba388d990ef54bcc57c8126df48f31b8cd448e4850\n\
                           size 6723496\n";

    #[test]
    fn a_pointer_is_detected_and_named_instead_of_parsed_as_a_mesh() {
        let dir = std::env::temp_dir().join("engine_io_lfs_pointer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let glb = dir.join("body.glb");
        std::fs::write(&glb, POINTER).unwrap();
        assert!(is_lfs_pointer(&glb));

        // The whole point: the error names the cause, not the glTF grammar.
        let err = load_mesh(&glb).unwrap_err().to_string();
        assert!(
            err.contains("git-LFS pointer") && err.contains("git lfs pull"),
            "{err}"
        );

        // A real (if invalid) GLB is NOT a pointer — the guard must not swallow
        // genuine parse errors.
        let real = dir.join("real.glb");
        std::fs::write(&real, b"glTF\x02\x00\x00\x00").unwrap();
        assert!(!is_lfs_pointer(&real));
        assert!(!load_mesh(&real)
            .unwrap_err()
            .to_string()
            .contains("git-LFS"));

        // Shorter than the magic, and absent, are both "not a pointer".
        let short = dir.join("short.glb");
        std::fs::write(&short, b"version ").unwrap();
        assert!(!is_lfs_pointer(&short));
        assert!(!is_lfs_pointer(dir.join("nope.glb")));
        std::fs::remove_dir_all(&dir).ok();
    }
}
