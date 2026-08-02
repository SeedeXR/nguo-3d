// SPDX-License-Identifier: GPL-2.0-or-later
//! OBJ reader: the two structures real garment exports carry that a naive
//! `v`+`f` parser throws away — **UV seams** (`vt > v`) and **`usemtl`/`g` runs**
//! (the pattern panels). Both fixtures below are miniatures of a case measured in
//! `samples/`: men-shirt has 71 745 `vt` for 70 312 `v`, and dress-a has 12
//! `usemtl` runs but only 1 material in its `.mtl`.

use engine_io::{parse_obj, parse_obj_mesh, write_obj, write_obj_mesh};

/// Two triangles sharing the edge v2–v3, split across a UV seam: the shared
/// vertices carry a different `vt` in each triangle. `vt` (6) > `v` (4).
const SEAM: &str = "\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vt 0 0
vt 1 0
vt 0 1
vt 1 1
vt 0.5 0
vt 0.5 1
f 1/1 2/5 4/6
f 1/1 4/6 3/3
";

#[test]
fn uv_seam_survives_as_corner_indices() {
    let m = parse_obj_mesh(SEAM, "seam").unwrap();
    // The seam exists: more texcoords than positions.
    assert_eq!(m.object.positions.len(), 4);
    assert_eq!(m.uvs.len(), 6, "the whole vt table is kept, not just 4");
    assert_eq!(m.uv_indices.len(), m.object.tris.len());

    // Vertex 3 (0-based) carries vt 5 in BOTH triangles here, but vertex 1 carries
    // vt 4 while the parallel array could only hold one value. The lossless proof
    // is that the corner indices point at the file's own table.
    assert_eq!(m.uv_indices[0], [0, 4, 5], "f 1/1 2/5 4/6 corner indices");
    assert_eq!(m.uv_indices[1], [0, 5, 2], "f 1/1 4/6 3/3 corner indices");
    assert_eq!(m.uvs[4], [0.5, 0.0]);
    assert_eq!(m.uvs[5], [0.5, 1.0]);

    // The parallel `Object::uvs` is the documented lossy view: first corner wins,
    // one entry per position. It must still be exactly parallel.
    assert_eq!(m.object.uvs.len(), 4);
    assert_eq!(m.object.uvs[0], [0.0, 0.0]);
    assert_eq!(m.object.uvs[1], [0.5, 0.0], "v1's only corner");
    assert_eq!(m.object.uvs[3], [0.5, 1.0]);
}

#[test]
fn uv_seam_round_trips_losslessly() {
    let m = parse_obj_mesh(SEAM, "seam").unwrap();
    let back = parse_obj_mesh(&write_obj_mesh(&m), "seam").unwrap();
    assert_eq!(back.object.positions, m.object.positions);
    assert_eq!(back.object.tris, m.object.tris);
    assert_eq!(back.uvs, m.uvs, "vt table survives the write");
    assert_eq!(back.uv_indices, m.uv_indices, "the seam survives the write");
    // And it is a fixed point: writing the re-parsed mesh gives the same bytes.
    assert_eq!(write_obj_mesh(&back), write_obj_mesh(&m));
}

/// Four triangles in three `usemtl` runs, only two of which name distinct
/// materials — the dress-a case in miniature (runs are panels, not materials).
const PANELS: &str = "\
mtllib dress_.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
g front
usemtl fabric
f 1 2 3
g back
usemtl fabric
f 2 4 3
f 1 3 4
g trim
usemtl binding
f 1 2 4
";

#[test]
fn usemtl_runs_are_preserved_as_named_face_ranges() {
    let m = parse_obj_mesh(PANELS, "panels").unwrap();
    assert_eq!(m.object.tris.len(), 4);
    assert_eq!(m.mtllib, vec!["dress_.mtl".to_string()]);
    assert_eq!(m.groups.len(), 3, "three runs, two distinct materials");

    let g: Vec<(&str, Option<&str>, usize, usize)> = m
        .groups
        .iter()
        .map(|g| {
            (
                g.name.as_str(),
                g.material.as_deref(),
                g.tris.start,
                g.tris.end,
            )
        })
        .collect();
    assert_eq!(
        g,
        vec![
            ("front", Some("fabric"), 0, 1),
            ("back", Some("fabric"), 1, 3),
            ("trim", Some("binding"), 3, 4),
        ]
    );
    // The ranges tile the triangle list exactly — that is what makes them usable
    // as pattern pieces.
    assert_eq!(m.groups[0].tris.start, 0);
    assert_eq!(m.groups.last().unwrap().tris.end, m.object.tris.len());
}

#[test]
fn panel_runs_round_trip_losslessly() {
    let m = parse_obj_mesh(PANELS, "panels").unwrap();
    let back = parse_obj_mesh(&write_obj_mesh(&m), "panels").unwrap();
    assert_eq!(back.groups, m.groups, "runs survive the write");
    assert_eq!(back.mtllib, m.mtllib);
    assert_eq!(back.object.tris, m.object.tris);
    assert_eq!(write_obj_mesh(&back), write_obj_mesh(&m));
}

#[test]
fn adjacent_runs_with_the_same_name_stay_separate() {
    // 12 `usemtl` runs / 1 material means repeats must NOT be merged.
    let src = "v 0 0 0\nv 1 0 0\nv 0 1 0\n\
               usemtl m\nf 1 2 3\nusemtl m\nf 1 2 3\nusemtl m\nf 1 2 3\n";
    let m = parse_obj_mesh(src, "rep").unwrap();
    assert_eq!(m.groups.len(), 3, "three runs of the same material");
    let back = parse_obj_mesh(&write_obj_mesh(&m), "rep").unwrap();
    assert_eq!(back.groups, m.groups, "and they survive a write");
}

/// A file with two `mtllib` entries, one of them a CLO-style name with a space
/// in it. The corpus has no such sample — every file there has exactly one —
/// so the writer used to join them onto one line and the round trip came back
/// as a single bogus name.
const TWO_LIBS: &str = "\
mtllib Men's Shirt.mtl
mtllib trim.mtl
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
";

#[test]
fn every_mtllib_entry_gets_its_own_line() {
    let m = parse_obj_mesh(TWO_LIBS, "libs").unwrap();
    assert_eq!(
        m.mtllib,
        vec!["Men's Shirt.mtl".to_string(), "trim.mtl".to_string()]
    );

    let out = write_obj_mesh(&m);
    assert!(
        out.contains("mtllib Men's Shirt.mtl\nmtllib trim.mtl\n"),
        "libraries were not written one per line:\n{out}"
    );

    let back = parse_obj_mesh(&out, "libs").unwrap();
    assert_eq!(
        back.mtllib, m.mtllib,
        "parse → write → parse is not a fixed point"
    );
    assert_eq!(write_obj_mesh(&back), out);
}

/// `#` starts a comment anywhere on the line, not only at column 0. A trailing
/// comment on an `f` line used to abort the whole file, and one on `usemtl`
/// became part of the material name.
#[test]
fn a_comment_is_stripped_from_anywhere_on_any_line_type() {
    let src = "\
mtllib dress_.mtl  # written by CLO
v 0 0 0 # corner
v 1 0 0
v 0 1 0
vt 0 0 # uv
vt 1 0
vt 0 1
usemtl fabric # the only material
f 1/1 2/2 3/3   # the only face
";
    let m = parse_obj_mesh(src, "commented").unwrap();
    assert_eq!(m.mtllib, vec!["dress_.mtl".to_string()]);
    assert_eq!(m.object.positions.len(), 3);
    assert_eq!(m.object.tris, vec![[0, 1, 2]]);
    assert_eq!(m.uvs.len(), 3);
    assert_eq!(m.groups.len(), 1);
    assert_eq!(m.groups[0].material.as_deref(), Some("fabric"));
}

#[test]
fn files_without_grouping_have_no_runs() {
    let m = parse_obj_mesh("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", "bare").unwrap();
    assert!(m.groups.is_empty());
    assert!(m.mtllib.is_empty());
}

#[test]
fn object_writer_carries_uvs_and_normals() {
    // `write_obj(&Object)` is the solver-side writer: parallel arrays only, but it
    // must no longer silently drop them.
    let src = "v 0 0 0\nv 1 0 0\nv 0 1 0\n\
               vt 0 0\nvt 1 0\nvt 0 1\n\
               vn 1 0 0\nvn 0 1 0\nvn 0 0 1\n\
               f 1/1/1 2/2/2 3/3/3\n";
    let o = parse_obj(src, "t").unwrap();
    let back = parse_obj(&write_obj(&o), "t").unwrap();
    assert_eq!(back.positions, o.positions);
    assert_eq!(back.tris, o.tris);
    assert_eq!(back.uvs, o.uvs, "write_obj kept the UVs");
    assert_eq!(back.normals, o.normals, "write_obj kept the normals");
}
