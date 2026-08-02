// SPDX-License-Identifier: GPL-2.0-or-later
//! Minimal USD ASCII (`.usda`) mesh reader — the VFX/animation interchange text
//! format (Pixar OpenUSD).
//!
//! ponytail: hand-rolled reader for the mesh core (`points`, `faceVertexIndices`,
//! `faceVertexCounts`, plus `normals` / `primvars:st` when present), rather than a
//! dependency. Rationale — the license-clean pure-Rust crates surveyed
//! (`openusd-rs` MIT/Apache-2.0, `openusd` MIT) read USDC *binary* well but their
//! USDA-text path is WIP, and USDC can't be fixture-generated here without Pixar's
//! toolchain, so it can't be self-verified. This text reader is one file, fully
//! deterministic, and testable against a hand-written `.usda`.
//!
//! USDC binary + USDZ packages + composition are handled by [`load_usd_binary`],
//! which flattens the file to USD-ASCII with the OpenUSD `usdcat` tool (subprocess,
//! not linked → no linking-license obligation) and feeds it back to `parse_usda`.
//! `usdcat --flatten` resolves the crate binary format, USDZ zip containers, and
//! references/sublayers/variants — reading production USD the way DCCs do.
//!
//! No-external-tool alternatives (deferred): an embedded pure-Rust USDC reader
//! (`openusd-rs`, MIT/Apache-2.0) or Pixar's OpenUSD C++ library via FFI. Those
//! remove the runtime `usdcat` dependency but add a heavy build/link dependency;
//! the subprocess path is chosen for verifiability + license cleanliness.
//!
//! Scope/limits (v1): reads the FIRST `def Mesh` prim (concatenation of multiple
//! meshes needs a per-prim vertex-base walk — the upgrade). Polygons are
//! fan-triangulated. No transforms/composition/references.

use anyhow::{bail, Context, Result};
use engine_api::Object;
use std::path::Path;

/// Load a `.usda` (USD ASCII) file into an [`Object`].
pub fn load_usda(path: impl AsRef<Path>) -> Result<Object> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading USDA {}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".into());
    parse_usda(&text, name)
}

/// Load a USD binary/package (`.usdc` crate binary, `.usdz` zip package, or a
/// binary `.usd`) by flattening it to USD-ASCII with the OpenUSD `usdcat` tool and
/// parsing the result with [`parse_usda`]. `usdcat --flatten` resolves the crate
/// binary format, USDZ containers, and full composition (references / sublayers /
/// variants), so this reads production USD the same way DCCs do.
///
/// Requires `usdcat` on `PATH` (an OpenUSD / `usd-core` install). It runs as a
/// SUBPROCESS — not linked — so this adds no OpenUSD linking-license obligation.
/// No-external-tool alternatives (deferred): an embedded pure-Rust USDC reader
/// (`openusd-rs`, MIT/Apache) or the OpenUSD C++ library via FFI.
pub fn load_usd_binary(path: impl AsRef<Path>) -> Result<Object> {
    let path = path.as_ref();
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".into());
    let output = std::process::Command::new("usdcat")
        .arg("--flatten")
        .arg(path)
        .output()
        .with_context(|| {
            format!(
                "running `usdcat` to read {} — install OpenUSD / `usd-core` (provides \
                 `usdcat`) to read .usdc/.usdz",
                path.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "usdcat failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let text = String::from_utf8(output.stdout).context("usdcat produced non-UTF-8 output")?;
    parse_usda(&text, name)
}

/// Load any USD file, auto-routing by content: the crate binary magic `PXR-USDC`
/// (or a read error) goes through [`load_usd_binary`] (usdcat), otherwise the file
/// is treated as USD-ASCII text ([`load_usda`]). Handy for the ambiguous `.usd`
/// extension, which may be either ASCII or binary.
pub fn load_usd(path: impl AsRef<Path>) -> Result<Object> {
    use std::io::Read;
    let path = path.as_ref();
    let mut magic = [0u8; 8];
    let is_binary = std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|()| &magic == b"PXR-USDC")
        .unwrap_or(false);
    if is_binary {
        load_usd_binary(path)
    } else {
        load_usda(path)
    }
}

/// Parse USD-ASCII text into an [`Object`] (first `def Mesh` prim).
pub fn parse_usda(text: &str, name: impl Into<String>) -> Result<Object> {
    // Isolate the first Mesh prim's body so its arrays don't cross into a
    // sibling prim (`def Mesh` .. up to the next top-level `def `).
    let window = first_mesh_window(text).context("no `def Mesh` prim found in USDA")?;

    let flat_pts = number_array(window, "point3f[] points")
        .or_else(|| number_array(window, "points"))
        .context("mesh has no `points` attribute")?;
    if flat_pts.len() % 3 != 0 {
        bail!("`points` length {} is not a multiple of 3", flat_pts.len());
    }
    let positions: Vec<[f32; 3]> = flat_pts
        .chunks_exact(3)
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect();
    if positions.is_empty() {
        bail!("USDA mesh has no points");
    }

    let indices = number_array(window, "faceVertexIndices").context("no `faceVertexIndices`")?;
    let counts = number_array(window, "faceVertexCounts").context("no `faceVertexCounts`")?;
    let idx: Vec<u32> = indices.iter().map(|&v| v as u32).collect();

    // Fan-triangulate each face; validate indices are in range.
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut cursor = 0usize;
    let nverts = positions.len() as u32;
    for (fi, &cf) in counts.iter().enumerate() {
        let c = cf as usize;
        if c < 3 {
            bail!("face {fi} has {c} verts (need ≥3)");
        }
        if cursor + c > idx.len() {
            bail!("faceVertexIndices too short for faceVertexCounts");
        }
        let face = &idx[cursor..cursor + c];
        for k in 1..c - 1 {
            let t = [face[0], face[k], face[k + 1]];
            if t[0] >= nverts || t[1] >= nverts || t[2] >= nverts {
                bail!("face {fi} references vertex out of range (nverts={nverts})");
            }
            tris.push(t);
        }
        cursor += c;
    }
    if tris.is_empty() {
        bail!("USDA mesh produced no triangles");
    }

    // Optional per-vertex normals / UVs — kept only if parallel to points.
    let normals = number_array(window, "normal3f[] normals")
        .or_else(|| number_array(window, "normals"))
        .filter(|n| n.len() == positions.len() * 3)
        .map(|n| {
            n.chunks_exact(3)
                .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let uvs = number_array(window, "primvars:st")
        .filter(|u| u.len() == positions.len() * 2)
        .map(|u| {
            u.chunks_exact(2)
                .map(|c| [c[0] as f32, c[1] as f32])
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(Object {
        name: name.into(),
        positions,
        tris,
        material: Default::default(),
        preset: None,
        pins: Vec::new(),
        is_static: false,
        normals,
        uvs,
        skin: None,
        animations: Vec::new(),
    })
}

/// Substring covering the first `def Mesh` prim: from that keyword to the next
/// top-level `def ` (or end). Isolates the prim's arrays from siblings.
fn first_mesh_window(text: &str) -> Option<&str> {
    let start = text.find("def Mesh")?;
    let rest = &text[start + "def Mesh".len()..];
    let end = rest.find("\ndef ").map(|e| start + "def Mesh".len() + e);
    Some(match end {
        Some(e) => &text[start..e],
        None => &text[start..],
    })
}

/// Extract the numeric array assigned to `key`: the numbers inside the first
/// `[ ... ]` that follows `key`. Tuple parens/commas are treated as separators,
/// so `[(0, 0, 0), (1, 0, 0)]` flattens to `[0,0,0,1,0,0]`. Returns None if the
/// key or its bracketed value is absent. (USDA mesh arrays are flat or arrays of
/// tuples — never nested `[]` — so the first `]` closes the value.)
fn number_array(window: &str, key: &str) -> Option<Vec<f64>> {
    let kpos = window.find(key)?;
    let after = &window[kpos + key.len()..];
    let open = after.find('[')?;
    let inner = &after[open + 1..];
    let close = inner.find(']')?;
    let body = &inner[..close];
    let cleaned: String = body
        .chars()
        .map(|ch| {
            if matches!(ch, '(' | ')' | ',') {
                ' '
            } else {
                ch
            }
        })
        .collect();
    let nums: Vec<f64> = cleaned
        .split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect();
    if nums.is_empty() {
        None
    } else {
        Some(nums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUAD: &str = r#"#usda 1.0
(
    defaultPrim = "cloth"
    upAxis = "Y"
)

def Mesh "cloth"
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    normal3f[] normals = ( interpolation = "vertex" ) [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)]
}
"#;

    #[test]
    fn parses_quad_mesh_and_fan_triangulates() {
        let obj = parse_usda(QUAD, "cloth").unwrap();
        assert_eq!(obj.positions.len(), 4);
        assert_eq!(obj.positions[1], [1.0, 0.0, 0.0]);
        // Quad [0,1,2,3] fans to (0,1,2) and (0,2,3).
        assert_eq!(obj.tris, vec![[0, 1, 2], [0, 2, 3]]);
        assert_eq!(obj.normals.len(), 4, "normals parsed past ( ) metadata");
        assert_eq!(obj.uvs.len(), 4);
        assert_eq!(obj.uvs[2], [1.0, 1.0]);
    }

    #[test]
    fn two_triangles_direct() {
        let src = r#"def Mesh "m" {
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 2, 1, 3]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
}"#;
        let obj = parse_usda(src, "m").unwrap();
        assert_eq!(obj.tris, vec![[0, 1, 2], [2, 1, 3]]);
        assert!(obj.normals.is_empty(), "no normals → empty");
        assert!(obj.uvs.is_empty());
    }

    #[test]
    fn rejects_out_of_range_index() {
        let src = r#"def Mesh "m" {
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 9]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
}"#;
        assert!(parse_usda(src, "m").is_err());
    }

    /// Whether the OpenUSD `usdcat` tool is available (USDC/USDZ tests need it).
    fn have_usdcat() -> bool {
        std::process::Command::new("usdcat")
            .arg("--help")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn loads_usdc_binary_round_trip() {
        // Write a .usda, convert to the .usdc CRATE BINARY with usdcat, then load the
        // binary through our usdcat path and confirm the mesh matches the text parse.
        if !have_usdcat() {
            eprintln!("usdcat not found; skipping USDC binary test");
            return;
        }
        let dir = std::env::temp_dir();
        let usda = dir.join("mguso_usdc_fixture.usda");
        let usdc = dir.join("mguso_usdc_fixture.usdc");
        std::fs::write(
            &usda,
            "#usda 1.0\ndef Mesh \"Quad\" {\n int[] faceVertexCounts = [3, 3]\n \
             int[] faceVertexIndices = [0, 1, 2, 2, 1, 3]\n \
             point3f[] points = [(0,0,0),(1,0,0),(0,1,0),(1,1,0)]\n}\n",
        )
        .unwrap();
        // .usdc extension makes usdcat emit the crate binary format.
        let ok = std::process::Command::new("usdcat")
            .arg("-o")
            .arg(&usdc)
            .arg(&usda)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "usdcat failed to produce .usdc");
        // Sanity: the output really is the crate binary (PXR-USDC magic).
        let magic = std::fs::read(&usdc).unwrap();
        assert_eq!(&magic[..8], b"PXR-USDC", "not a crate binary");

        let obj = load_usd_binary(&usdc).expect("load usdc via usdcat");
        let via_load = crate::load_mesh(&usdc).expect("dispatcher routes .usdc");
        let _ = std::fs::remove_file(&usda);
        let _ = std::fs::remove_file(&usdc);
        assert_eq!(obj.positions.len(), 4);
        assert_eq!(obj.positions[3], [1.0, 1.0, 0.0]);
        assert_eq!(obj.tris, vec![[0, 1, 2], [2, 1, 3]]);
        // The extension dispatcher (`mguso run scene.usdc`) yields the same mesh.
        assert_eq!(via_load.tris, obj.tris);
    }

    #[test]
    fn loads_usdz_package_round_trip() {
        // Package a .usda into a .usdz (zip) with usdzip, then load it.
        if !have_usdcat() {
            eprintln!("usdcat/usdzip not found; skipping USDZ test");
            return;
        }
        let dir = std::env::temp_dir();
        let usda = dir.join("mguso_usdz_fixture.usda");
        let usdz = dir.join("mguso_usdz_fixture.usdz");
        let _ = std::fs::remove_file(&usdz); // usdzip refuses to overwrite
        std::fs::write(
            &usda,
            "#usda 1.0\ndef Mesh \"Quad\" {\n int[] faceVertexCounts = [3]\n \
             int[] faceVertexIndices = [0, 1, 2]\n \
             point3f[] points = [(0,0,0),(2,0,0),(0,2,0)]\n}\n",
        )
        .unwrap();
        let ok = std::process::Command::new("usdzip")
            .arg(&usdz)
            .arg(&usda)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("usdzip failed; skipping USDZ assertion");
            let _ = std::fs::remove_file(&usda);
            return;
        }
        let obj = crate::load_mesh(&usdz).expect("load usdz via dispatcher");
        let _ = std::fs::remove_file(&usda);
        let _ = std::fs::remove_file(&usdz);
        assert_eq!(obj.positions.len(), 3);
        assert_eq!(obj.positions[1], [2.0, 0.0, 0.0]);
        assert_eq!(obj.tris, vec![[0, 1, 2]]);
    }
}
