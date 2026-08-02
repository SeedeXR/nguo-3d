// SPDX-License-Identifier: GPL-2.0-or-later
//! Wavefront OBJ read/write for triangle meshes.
//!
//! ponytail: hand-rolled parser instead of `tobj` — OBJ's grammar is a few lines,
//! `tobj` would still not give us the panel runs in the shape we want, and we need
//! our own writer regardless. USD/glTF/COLLADA/FBX are separate modules.
//!
//! # What is preserved, and where
//!
//! [`parse_obj`] returns the solver's [`Object`]: positions are the file's `v`
//! table **welded** (one vertex per `v`, never split), because the cloth solver
//! must simulate the welded mesh — splitting at UV seams would change the vertex
//! count, the mass lumping and the constraint graph for a purely cosmetic reason.
//!
//! That makes `Object::uvs`/`normals` (which are *parallel to positions*) unable
//! to represent a **UV seam**, where one position carries several texture
//! coordinates. Real files have them: of the sample corpus, `men-shirt` has
//! 71 745 `vt` for 70 312 `v` and `women-skirt` 36 127 for 35 535. So `Object`
//! gets the "first face corner touching this vertex wins" de-index — the same
//! lossy-but-parallel contract the FBX/glTF/COLLADA readers already use — and the
//! **lossless** data is returned by [`parse_obj_mesh`] as [`ObjMesh`]: the raw
//! `vt`/`vn` tables plus per-triangle-corner index arrays.
//!
//! HANDOFF: representing seams end-to-end needs a corner-index array on
//! `engine_api::Object` (or a render-side split at upload). `engine-api` is not
//! ours to change; until it grows one, a renderer that wants exact UVs must call
//! [`parse_obj_mesh`] and use `uv_indices`.
//!
//! `usemtl`/`g`/`o` runs are kept as named triangle ranges ([`ObjGroup`]). They
//! are **not** material variety — `dress-a` has 12 `usemtl` runs and its `.mtl`
//! declares 1 material — they are the garment's pattern panels, which is the most
//! valuable structure in the file for a 2D-pattern-first garment application.

use anyhow::{bail, Context, Result};
use engine_api::Object;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::Path;

/// One `g`/`o`/`usemtl` run: the triangles emitted between one such directive and
/// the next. Ranges index [`ObjMesh::object`]`.tris` and are contiguous and
/// non-overlapping, in file order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjGroup {
    /// The `g`/`o` name in effect (empty if the file only used `usemtl`).
    pub name: String,
    /// The `usemtl` name in effect, if any.
    pub material: Option<String>,
    /// Triangle range in `object.tris`.
    pub tris: Range<usize>,
}

/// An OBJ with everything the file actually contained.
#[derive(Clone, Debug)]
pub struct ObjMesh {
    /// Welded mesh for the solver: positions parallel to the file's `v` table,
    /// with first-corner-wins `uvs`/`normals` (empty if the file has none, or if
    /// any face omitted the corresponding index).
    pub object: Object,
    /// The raw `vt` table, in file order.
    pub uvs: Vec<[f32; 2]>,
    /// Per-triangle corner indices into [`Self::uvs`]; parallel to `object.tris`,
    /// or empty. This is the only lossless representation of a UV seam.
    pub uv_indices: Vec<[u32; 3]>,
    /// The raw `vn` table, in file order.
    pub normals: Vec<[f32; 3]>,
    /// Per-triangle corner indices into [`Self::normals`]; parallel or empty.
    pub normal_indices: Vec<[u32; 3]>,
    /// `usemtl`/`g`/`o` runs, in file order. Empty if the file has no such
    /// directive.
    pub groups: Vec<ObjGroup>,
    /// `mtllib` names, verbatim. **Not resolved and not trusted** — the corpus has
    /// one with a Windows absolute path and one naming a file that does not exist.
    pub mtllib: Vec<String>,
}

/// Parse an OBJ into the solver's [`Object`] (see the module docs for what this
/// drops). Faces with >3 vertices are fan-triangulated.
pub fn parse_obj(text: &str, name: impl Into<String>) -> Result<Object> {
    Ok(parse_obj_mesh(text, name)?.object)
}

/// Parse an OBJ, keeping `vt`/`vn` corner indices, `usemtl`/`g` runs and `mtllib`.
pub fn parse_obj_mesh(text: &str, name: impl Into<String>) -> Result<ObjMesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uv_indices: Vec<[u32; 3]> = Vec::new();
    let mut normal_indices: Vec<[u32; 3]> = Vec::new();
    let mut mtllib: Vec<String> = Vec::new();
    let mut groups: Vec<ObjGroup> = Vec::new();
    // Every face corner must carry the index for the array to stay meaningful;
    // one bare `f 1 2 3` in a file with `vt` invalidates the whole UV index array.
    let mut all_have_uv = true;
    let mut all_have_normal = true;
    let mut group_name = String::new();
    let mut material: Option<String> = None;
    let mut run_start = 0usize;
    let mut saw_group_directive = false;

    for (lineno, line) in text.lines().enumerate() {
        // OBJ has no escaping: `#` starts a comment *anywhere* on the line, not
        // only at column 0, and assimp and `tobj` both read it that way. Real
        // exports put them after `f` and `usemtl`, and a trailing comment used
        // to abort the whole file.
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else { continue };
        match key {
            "v" => positions.push(floats3(&mut it, lineno, "vertex")?),
            "vn" => normals.push(floats3(&mut it, lineno, "normal")?),
            "vt" => {
                // `vt u [v] [w]` — v defaults to 0 (1-D textures are legal OBJ).
                let u: f32 = it
                    .next()
                    .with_context(|| format!("vt needs >=1 coord on line {}", lineno + 1))?
                    .parse()
                    .with_context(|| format!("bad texcoord on line {}", lineno + 1))?;
                let v: f32 = match it.next() {
                    Some(s) => s
                        .parse()
                        .with_context(|| format!("bad texcoord on line {}", lineno + 1))?,
                    None => 0.0,
                };
                uvs.push([u, v]);
            }
            "f" => {
                let corners: Vec<Corner> = it
                    .map(|tok| parse_corner(tok, positions.len(), uvs.len(), normals.len()))
                    .collect::<Result<_>>()
                    .with_context(|| format!("bad face on line {}", lineno + 1))?;
                if corners.len() < 3 {
                    bail!("face needs >=3 verts on line {}", lineno + 1);
                }
                all_have_uv &= corners.iter().all(|c| c.uv.is_some());
                all_have_normal &= corners.iter().all(|c| c.normal.is_some());
                // Fan triangulation; the attribute indices fan identically.
                for k in 1..corners.len() - 1 {
                    let (a, b, c) = (&corners[0], &corners[k], &corners[k + 1]);
                    tris.push([a.pos, b.pos, c.pos]);
                    uv_indices.push([a.uv.unwrap_or(0), b.uv.unwrap_or(0), c.uv.unwrap_or(0)]);
                    normal_indices.push([
                        a.normal.unwrap_or(0),
                        b.normal.unwrap_or(0),
                        c.normal.unwrap_or(0),
                    ]);
                }
            }
            "g" | "o" | "usemtl" => {
                saw_group_directive = true;
                if tris.len() > run_start {
                    groups.push(ObjGroup {
                        name: group_name.clone(),
                        material: material.clone(),
                        tris: run_start..tris.len(),
                    });
                }
                run_start = tris.len();
                let rest = it.collect::<Vec<_>>().join(" ");
                if key == "usemtl" {
                    material = Some(rest);
                } else {
                    group_name = rest;
                }
            }
            "mtllib" => {
                // `mtllib` takes a whitespace-separated list, but CLO writes
                // `mtllib Men's Shirt.mtl` — a single name with a space in it. The
                // list form is only unambiguous when every token looks like a
                // library, so fall back to "the rest of the line is one name".
                //
                // ponytail: the format itself is ambiguous here, so this is a
                // heuristic with a known ceiling — `mtllib my.mtl.backup b.mtl`
                // comes back as one name. Resolving it needs the filesystem
                // (does `my.mtl.backup` exist next to the OBJ?), which this
                // function does not have and should not grow.
                let toks: Vec<&str> = it.collect();
                if toks.len() > 1
                    && toks
                        .iter()
                        .all(|t| t.to_ascii_lowercase().ends_with(".mtl"))
                {
                    mtllib.extend(toks.iter().map(|s| (*s).to_owned()));
                } else if !toks.is_empty() {
                    mtllib.push(toks.join(" "));
                }
            }
            _ => {} // s, vp, curv, … — no meaning for a triangle mesh
        }
    }

    if positions.is_empty() {
        bail!("OBJ contained no vertices");
    }
    if saw_group_directive && tris.len() > run_start {
        groups.push(ObjGroup {
            name: group_name,
            material,
            tris: run_start..tris.len(),
        });
    }
    if !all_have_uv || uvs.is_empty() {
        uvs.clear();
        uv_indices.clear();
    }
    if !all_have_normal || normals.is_empty() {
        normals.clear();
        normal_indices.clear();
    }

    // Parallel-to-positions view for `Object`: first corner touching a vertex wins.
    let obj_uvs = deindex(&tris, &uv_indices, &uvs, positions.len());
    let obj_normals = deindex(&tris, &normal_indices, &normals, positions.len());

    Ok(ObjMesh {
        object: Object {
            name: name.into(),
            positions,
            tris,
            material: Default::default(),
            preset: None,
            pins: Vec::new(),
            is_static: false,
            normals: obj_normals,
            uvs: obj_uvs,
            skin: None,
            animations: Vec::new(),
        },
        uvs,
        uv_indices,
        normals,
        normal_indices,
        groups,
        mtllib,
    })
}

/// One parsed `f` corner: `v`, `v/vt`, `v//vn` or `v/vt/vn` (all 0-based here).
struct Corner {
    pos: u32,
    uv: Option<u32>,
    normal: Option<u32>,
}

fn floats3<'a>(
    it: &mut impl Iterator<Item = &'a str>,
    lineno: usize,
    what: &str,
) -> Result<[f32; 3]> {
    let coords: Vec<f32> = it
        .take(3)
        .map(|s| s.parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("bad {what} on line {}", lineno + 1))?;
    if coords.len() != 3 {
        bail!("{what} needs 3 coords on line {}", lineno + 1);
    }
    Ok([coords[0], coords[1], coords[2]])
}

fn parse_corner(tok: &str, nv: usize, nvt: usize, nvn: usize) -> Result<Corner> {
    let mut parts = tok.split('/');
    let pos = resolve(parts.next().unwrap_or(tok), nv, "vertex")?
        .with_context(|| format!("face corner {tok:?} has no vertex index"))?;
    let uv = match parts.next() {
        Some(s) => resolve(s, nvt, "texcoord")?,
        None => None,
    };
    let normal = match parts.next() {
        Some(s) => resolve(s, nvn, "normal")?,
        None => None,
    };
    Ok(Corner { pos, uv, normal })
}

/// OBJ indices are 1-based; negative indices count back from the end of the table
/// *as it stands at this point in the file*. An empty field (`v//vn`) is `None`.
fn resolve(field: &str, count: usize, what: &str) -> Result<Option<u32>> {
    if field.is_empty() {
        return Ok(None);
    }
    let raw: i64 = field
        .parse()
        .with_context(|| format!("bad {what} index {field:?}"))?;
    let resolved = if raw < 0 { count as i64 + raw } else { raw - 1 };
    if resolved < 0 || resolved as usize >= count {
        bail!("{what} index {raw} out of range ({what}s so far: {count})");
    }
    Ok(Some(resolved as u32))
}

/// Corner-indexed attribute → one value per position (first corner wins). Empty
/// in, empty out; unreferenced vertices get the zero value.
fn deindex<const N: usize>(
    tris: &[[u32; 3]],
    indices: &[[u32; 3]],
    table: &[[f32; N]],
    nverts: usize,
) -> Vec<[f32; N]> {
    if indices.is_empty() || table.is_empty() {
        return Vec::new();
    }
    let mut out = vec![None; nverts];
    for (t, a) in tris.iter().zip(indices) {
        for k in 0..3 {
            let slot = &mut out[t[k] as usize];
            if slot.is_none() {
                *slot = Some(table[a[k] as usize]);
            }
        }
    }
    out.into_iter().map(|v| v.unwrap_or([0.0; N])).collect()
}

/// Serialize an [`Object`] to OBJ text (1-based indices). Writes `vt`/`vn` when
/// the object carries them; since they are parallel to positions, every corner
/// uses the same index for all three.
pub fn write_obj(obj: &Object) -> String {
    let mut s = String::with_capacity(obj.positions.len() * 24);
    let _ = writeln!(s, "# ppf-mguso-solver OBJ export: {}", obj.name);
    for p in &obj.positions {
        let _ = writeln!(s, "v {} {} {}", p[0], p[1], p[2]);
    }
    for t in &obj.uvs {
        let _ = writeln!(s, "vt {} {}", t[0], t[1]);
    }
    for n in &obj.normals {
        let _ = writeln!(s, "vn {} {} {}", n[0], n[1], n[2]);
    }
    let (uv, nrm) = (!obj.uvs.is_empty(), !obj.normals.is_empty());
    for t in &obj.tris {
        let _ = write!(s, "f");
        for &i in t {
            let _ = write!(
                s,
                " {}",
                corner(i + 1, uv.then_some(i + 1), nrm.then_some(i + 1))
            );
        }
        let _ = writeln!(s);
    }
    s
}

/// Serialize an [`ObjMesh`] — positions, the full `vt`/`vn` tables with their
/// per-corner indices, the `usemtl`/`g` runs and `mtllib`. Lossless for anything
/// [`parse_obj_mesh`] produced: parse → write → parse is a fixed point.
pub fn write_obj_mesh(m: &ObjMesh) -> String {
    let o = &m.object;
    let mut s = String::with_capacity(o.positions.len() * 24);
    let _ = writeln!(s, "# ppf-mguso-solver OBJ export: {}", o.name);
    // One line per library. Joining them would re-enter the reader's
    // "is this a list or one name with spaces?" heuristic and come back as a
    // single name, so `parse → write → parse` would not be a fixed point.
    for lib in &m.mtllib {
        let _ = writeln!(s, "mtllib {lib}");
    }
    for p in &o.positions {
        let _ = writeln!(s, "v {} {} {}", p[0], p[1], p[2]);
    }
    for t in &m.uvs {
        let _ = writeln!(s, "vt {} {}", t[0], t[1]);
    }
    for n in &m.normals {
        let _ = writeln!(s, "vn {} {} {}", n[0], n[1], n[2]);
    }
    let mut next_group = 0usize;
    for (i, t) in o.tris.iter().enumerate() {
        // A run start re-emits both directives, so runs survive verbatim even when
        // two adjacent ones share a name or a material.
        while next_group < m.groups.len() && m.groups[next_group].tris.start == i {
            let g = &m.groups[next_group];
            let _ = writeln!(s, "g {}", g.name);
            if let Some(mat) = &g.material {
                let _ = writeln!(s, "usemtl {mat}");
            }
            next_group += 1;
        }
        let _ = write!(s, "f");
        for k in 0..3 {
            let uv = m.uv_indices.get(i).map(|a| a[k] + 1);
            let nrm = m.normal_indices.get(i).map(|a| a[k] + 1);
            let _ = write!(s, " {}", corner(t[k] + 1, uv, nrm));
        }
        let _ = writeln!(s);
    }
    s
}

/// `v`, `v/vt`, `v//vn` or `v/vt/vn`.
fn corner(v: u32, uv: Option<u32>, nrm: Option<u32>) -> String {
    match (uv, nrm) {
        (None, None) => format!("{v}"),
        (Some(t), None) => format!("{v}/{t}"),
        (None, Some(n)) => format!("{v}//{n}"),
        (Some(t), Some(n)) => format!("{v}/{t}/{n}"),
    }
}

pub fn load_obj(path: impl AsRef<Path>) -> Result<Object> {
    Ok(load_obj_mesh(path)?.object)
}

/// [`load_obj`] keeping `vt`/`vn` corner indices, `usemtl`/`g` runs and `mtllib`.
pub fn load_obj_mesh(path: impl AsRef<Path>) -> Result<ObjMesh> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading OBJ {}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".into());
    parse_obj_mesh(&text, name)
}

pub fn save_obj(obj: &Object, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path.as_ref(), write_obj(obj))
        .with_context(|| format!("writing OBJ {}", path.as_ref().display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_quad() {
        let src = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 3 2 4\n";
        let obj = parse_obj(src, "q").unwrap();
        assert_eq!(obj.positions.len(), 4);
        assert_eq!(obj.tris.len(), 2);
        let out = write_obj(&obj);
        let reparsed = parse_obj(&out, "q").unwrap();
        assert_eq!(reparsed.positions, obj.positions);
        assert_eq!(reparsed.tris, obj.tris);
    }

    #[test]
    fn triangulates_quad_face() {
        let obj = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", "q").unwrap();
        assert_eq!(obj.tris.len(), 2);
    }

    #[test]
    fn handles_slashed_and_negative_indices() {
        let obj = parse_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\n\
             f -3/-3/-3 -2/-2/-2 -1/-1/-1\n",
            "q",
        )
        .unwrap();
        assert_eq!(obj.tris, vec![[0, 1, 2]]);
        assert_eq!(obj.uvs, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn rejects_out_of_range_index() {
        assert!(parse_obj("v 0 0 0\nf 1 2 3\n", "q").is_err());
        // A vt index past the end of the vt table is just as broken.
        assert!(parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/2 3/1\n", "q").is_err());
    }

    #[test]
    fn reads_uvs_and_normals() {
        let src = "v 0 0 0\nv 1 0 0\nv 0 1 0\n\
                   vt 0 0\nvt 1 0\nvt 0 1\n\
                   vn 1 0 0\nvn 0 1 0\nvn 0 0 1\n\
                   f 1/1/1 2/2/2 3/3/3\n";
        let obj = parse_obj(src, "t").unwrap();
        assert_eq!(obj.uvs, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(obj.normals[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn face_without_uv_index_drops_the_uv_array() {
        // Parallel-or-empty: one bare corner invalidates the whole array.
        let src = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3\n";
        let m = parse_obj_mesh(src, "t").unwrap();
        assert!(m.uv_indices.is_empty());
        assert!(m.object.uvs.is_empty());
    }

    #[test]
    fn vt_defaults_v_to_zero() {
        let m = parse_obj_mesh("v 0 0 0\nvt 0.25\n", "t").unwrap();
        assert_eq!(m.uvs, vec![[0.25, 0.0]]);
    }

    #[test]
    fn mtllib_is_kept_verbatim() {
        let m = parse_obj_mesh("mtllib C:/Users/AP/dress.mtl\nv 0 0 0\n", "t").unwrap();
        assert_eq!(m.mtllib, vec!["C:/Users/AP/dress.mtl".to_string()]);
        // A name with a space (CLO writes these) stays one name…
        let m = parse_obj_mesh("mtllib Men's Shirt.mtl\nv 0 0 0\n", "t").unwrap();
        assert_eq!(m.mtllib, vec!["Men's Shirt.mtl".to_string()]);
        // …but a genuine list stays a list.
        let m = parse_obj_mesh("mtllib a.mtl b.mtl\nv 0 0 0\n", "t").unwrap();
        assert_eq!(m.mtllib, vec!["a.mtl".to_string(), "b.mtl".to_string()]);
    }
}
