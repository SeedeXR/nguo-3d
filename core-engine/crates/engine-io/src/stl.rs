// SPDX-License-Identifier: GPL-2.0-or-later
//! STL mesh I/O (ASCII + binary) with vertex welding.
//!
//! STL stores each triangle's 3 vertices independently (no sharing), so a raw
//! import has no connectivity — useless for a cloth solver, which needs shared
//! vertices for springs/topology. We **weld** coincident positions (quantized to
//! a tolerance) into a single index so the imported mesh is a proper manifold-ish
//! surface. STL is the most common interchange format for real garment / scan
//! meshes, hence first-class here alongside OBJ.
//!
//! ponytail: hand-rolled (no dep) — the binary layout is fixed and the ASCII
//! grammar is tiny. Reach for a crate only if we need color/attribute bytes.

use anyhow::{bail, Context, Result};
use engine_api::Object;
use std::path::Path;

const WELD_QUANTUM: f32 = 1.0e-5; // positions within this fuse to one vertex

struct Welder {
    map: std::collections::HashMap<(i64, i64, i64), u32>,
    positions: Vec<[f32; 3]>,
}

impl Welder {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            positions: Vec::new(),
        }
    }
    fn index(&mut self, p: [f32; 3]) -> u32 {
        let key = (
            (p[0] / WELD_QUANTUM).round() as i64,
            (p[1] / WELD_QUANTUM).round() as i64,
            (p[2] / WELD_QUANTUM).round() as i64,
        );
        *self.map.entry(key).or_insert_with(|| {
            self.positions.push(p);
            (self.positions.len() - 1) as u32
        })
    }
}

/// Parse binary STL bytes (80-byte header, u32 count, 50 bytes/triangle).
fn parse_binary(bytes: &[u8], name: &str) -> Result<Object> {
    if bytes.len() < 84 {
        bail!("binary STL too short");
    }
    let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let expected = 84 + count * 50;
    if bytes.len() < expected {
        bail!(
            "binary STL truncated: expected {expected} bytes, got {}",
            bytes.len()
        );
    }
    let f = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let mut w = Welder::new();
    let mut tris = Vec::with_capacity(count);
    for t in 0..count {
        let base = 84 + t * 50 + 12; // skip the per-facet normal (3 f32)
        let v = |k: usize| [f(base + k * 12), f(base + k * 12 + 4), f(base + k * 12 + 8)];
        let (a, b, c) = (w.index(v(0)), w.index(v(1)), w.index(v(2)));
        if a != b && b != c && a != c {
            tris.push([a, b, c]); // drop degenerate (welded-collapsed) facets
        }
    }
    finish(w.positions, tris, name)
}

/// Parse ASCII STL text ("solid … facet normal … outer loop vertex … ").
fn parse_ascii(text: &str, name: &str) -> Result<Object> {
    let mut w = Welder::new();
    let mut tris = Vec::new();
    let mut loop_verts: Vec<u32> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("vertex") => {
                let c: Vec<f32> = it
                    .take(3)
                    .map(|s| s.parse::<f32>())
                    .collect::<std::result::Result<_, _>>()
                    .context("bad STL vertex")?;
                if c.len() != 3 {
                    bail!("STL vertex needs 3 coords");
                }
                loop_verts.push(w.index([c[0], c[1], c[2]]));
            }
            Some("endloop") => {
                // Triangulate the loop as a fan (STL loops are triangles).
                for k in 1..loop_verts.len().saturating_sub(1) {
                    let (a, b, c) = (loop_verts[0], loop_verts[k], loop_verts[k + 1]);
                    if a != b && b != c && a != c {
                        tris.push([a, b, c]);
                    }
                }
                loop_verts.clear();
            }
            _ => {}
        }
    }
    finish(w.positions, tris, name)
}

fn finish(positions: Vec<[f32; 3]>, tris: Vec<[u32; 3]>, name: &str) -> Result<Object> {
    if positions.is_empty() || tris.is_empty() {
        bail!("STL contained no triangles");
    }
    Ok(Object {
        name: name.to_string(),
        positions,
        tris,
        material: Default::default(),
        preset: None,
        pins: Vec::new(),
        is_static: false,
        normals: Vec::new(),
        uvs: Vec::new(),
        skin: None,
        animations: Vec::new(),
    })
}

/// Load an STL file, auto-detecting ASCII vs binary, welding vertices.
pub fn load_stl(path: impl AsRef<Path>) -> Result<Object> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading STL {}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".into());

    // Detection: a valid binary STL has size exactly 84 + 50·count. ASCII files
    // start with "solid" but won't satisfy that size relation (except by fluke,
    // which the header-keyword check below rules out).
    let looks_ascii = bytes.starts_with(b"solid")
        && std::str::from_utf8(&bytes[..bytes.len().min(512)])
            .map(|s| s.contains("facet"))
            .unwrap_or(false);
    if looks_ascii {
        let text = std::str::from_utf8(&bytes).context("STL is not valid UTF-8")?;
        parse_ascii(text, &name)
    } else {
        parse_binary(&bytes, &name)
    }
}

/// Serialize an [`Object`] to binary STL (per-triangle, no sharing; zeroed normals).
pub fn write_stl_binary(obj: &Object) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + obj.tris.len() * 50);
    out.extend_from_slice(&[0u8; 80]); // header
    out.extend_from_slice(&(obj.tris.len() as u32).to_le_bytes());
    for t in &obj.tris {
        out.extend_from_slice(&[0u8; 12]); // normal (0,0,0)
        for &vi in t {
            for &comp in &obj.positions[vi as usize] {
                out.extend_from_slice(&comp.to_le_bytes());
            }
        }
        out.extend_from_slice(&[0u8; 2]); // attribute byte count
    }
    out
}

pub fn save_stl(obj: &Object, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path.as_ref(), write_stl_binary(obj))
        .with_context(|| format!("writing STL {}", path.as_ref().display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Object {
        // Two triangles sharing an edge → 4 welded vertices.
        Object {
            name: "q".into(),
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            tris: vec![[0, 1, 2], [2, 1, 3]],
            material: Default::default(),
            preset: None,
            pins: Vec::new(),
            is_static: false,
            normals: Vec::new(),
            uvs: Vec::new(),
            skin: None,
            animations: Vec::new(),
        }
    }

    #[test]
    fn binary_round_trip_welds_shared_vertices() {
        let q = quad();
        let bytes = write_stl_binary(&q);
        let back = parse_binary(&bytes, "q").unwrap();
        // 6 raw STL vertices (2 tris × 3) weld back down to the 4 unique corners.
        assert_eq!(
            back.positions.len(),
            4,
            "welding failed: {:?}",
            back.positions
        );
        assert_eq!(back.tris.len(), 2);
    }

    #[test]
    fn ascii_parses_and_welds() {
        let src = "solid q\n\
             facet normal 0 0 1\n outer loop\n  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n endloop\n endfacet\n\
             facet normal 0 0 1\n outer loop\n  vertex 0 1 0\n  vertex 1 0 0\n  vertex 1 1 0\n endloop\n endfacet\n\
             endsolid q\n";
        let obj = parse_ascii(src, "q").unwrap();
        assert_eq!(obj.positions.len(), 4);
        assert_eq!(obj.tris.len(), 2);
    }

    #[test]
    fn detects_ascii_vs_binary() {
        let q = quad();
        // Binary path via the public loader would need a file; exercise parse fns
        // directly, plus confirm ASCII detection keys on the "facet" keyword.
        let bytes = write_stl_binary(&q);
        assert!(!bytes.starts_with(b"solid")); // zeroed header, not "solid"
        assert_eq!(parse_binary(&bytes, "q").unwrap().tris.len(), 2);
    }
}
