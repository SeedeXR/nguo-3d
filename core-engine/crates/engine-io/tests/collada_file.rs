// SPDX-License-Identifier: GPL-2.0-or-later
//! File-path COLLADA read test: `load_collada` / `load_mesh` on a hand-written
//! `.dae` fixture (a 2-quad cloth grid via `<polylist>` with interleaved
//! VERTEX/NORMAL offsets).

/// Reads `tests/fixtures/cloth_grid.dae`: 6 points, two quads in a `<polylist>`
/// (each with a 2-wide index tuple) → four fan triangles.
#[test]
fn collada_file_reads_grid_mesh() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cloth_grid.dae");

    // Via the extension dispatcher (the path `mguso run scene.dae` takes).
    let obj = engine_io::load_mesh(&path).expect("load dae");

    assert_eq!(obj.positions.len(), 6, "6 grid points");
    assert_eq!(obj.positions[2], [2.0, 0.0, 0.0]);
    assert_eq!(obj.positions[5], [2.0, 1.0, 0.0]);
    // Two quads [0,1,4,3] and [1,2,5,4], each fanned into 2 triangles; VERTEX
    // indices pulled from offset 0 of each 2-wide tuple.
    assert_eq!(obj.tris, vec![[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]);
    // Face-varying NORMAL (offset 1) de-indexed to per-control-point: the single
    // normal (0,0,1) is shared by every corner → parallel per-vertex normals.
    assert_eq!(obj.normals.len(), 6, "normals parallel to positions");
    assert!(obj.normals.iter().all(|n| *n == [0.0, 0.0, 1.0]));
    assert!(obj.uvs.is_empty(), "no TEXCOORD input → empty");
    for t in &obj.tris {
        for &v in t {
            assert!((v as usize) < obj.positions.len());
        }
    }
}
