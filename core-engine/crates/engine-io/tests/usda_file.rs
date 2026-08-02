// SPDX-License-Identifier: GPL-2.0-or-later
//! File-path USDA read test: `load_usda` / `load_mesh` on a hand-written `.usda`
//! fixture (a 2-quad cloth grid). Exercises the extension dispatcher too.

/// Reads `tests/fixtures/cloth_grid.usda`: 6 points, two quads → four triangles
/// (fan), with per-vertex normals + `primvars:st`.
#[test]
fn usda_file_reads_grid_mesh() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cloth_grid.usda");

    // Via the extension dispatcher (the path `mguso run scene.usda` takes).
    let obj = engine_io::load_mesh(&path).expect("load usda");

    assert_eq!(obj.positions.len(), 6, "6 grid points");
    assert_eq!(obj.positions[2], [2.0, 0.0, 0.0]);
    // Two quads, each fanned into 2 triangles.
    assert_eq!(obj.tris.len(), 4, "2 quads → 4 triangles");
    assert_eq!(obj.tris[0], [0, 1, 4]);
    assert_eq!(obj.tris[1], [0, 4, 3]);
    assert_eq!(obj.tris[2], [1, 2, 5]);
    assert_eq!(obj.tris[3], [1, 5, 4]);
    assert_eq!(obj.normals.len(), 6, "per-vertex normals kept");
    assert_eq!(obj.uvs.len(), 6, "primvars:st kept");
    assert_eq!(obj.uvs[1], [0.5, 0.0]);
    // Every triangle references a valid vertex.
    for t in &obj.tris {
        for &v in t {
            assert!((v as usize) < obj.positions.len());
        }
    }
}
