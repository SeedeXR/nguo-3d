// SPDX-License-Identifier: GPL-2.0-or-later
//! FBX binary read test: build a minimal binary FBX in-memory with `fbxcel`'s
//! writer (Objects → Geometry → Vertices + PolygonVertexIndex), then parse it
//! with our reader. Verifies the vertex + polygon (negative-terminator) decoding.

use fbxcel::low::v7400::ArrayAttributeEncoding;
use fbxcel::low::v7400::ArrayAttributeEncoding as Enc;
use fbxcel::low::FbxVersion;
use fbxcel::writer::v7400::binary::{FbxFooter, Writer};
use std::io::Cursor;

/// Write a binary FBX with one Geometry: `verts` (flat f64 xyz) and `poly_idx`
/// (i32 PolygonVertexIndex, negative = polygon terminator).
fn build_fbx(verts: &[f64], poly_idx: &[i32]) -> Vec<u8> {
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap(); // container, no attributes
    w.new_node("Geometry").unwrap();
    {
        let mut a = w.new_node("Vertices").unwrap();
        a.append_arr_f64_from_iter(ArrayAttributeEncoding::Direct, verts.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap(); // Vertices
    {
        let mut a = w.new_node("PolygonVertexIndex").unwrap();
        a.append_arr_i32_from_iter(ArrayAttributeEncoding::Direct, poly_idx.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap(); // PolygonVertexIndex
    w.close_node().unwrap(); // Geometry
    w.close_node().unwrap(); // Objects
    let out = w.finalize_and_flush(&FbxFooter::default()).unwrap();
    out.into_inner()
}

/// Like `build_fbx` but also emits a `LayerElementNormal` with the given
/// `mapping` (`"ByVertice"`/`"ByPolygonVertex"`), Direct reference, and `normals`
/// (flat xyz; one per control point for ByVertice, one per polygon-corner for
/// ByPolygonVertex).
fn build_fbx_normals(verts: &[f64], poly_idx: &[i32], mapping: &str, normals: &[f64]) -> Vec<u8> {
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap();
    w.new_node("Geometry").unwrap();
    {
        let mut a = w.new_node("Vertices").unwrap();
        a.append_arr_f64_from_iter(ArrayAttributeEncoding::Direct, verts.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("PolygonVertexIndex").unwrap();
        a.append_arr_i32_from_iter(ArrayAttributeEncoding::Direct, poly_idx.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    // LayerElementNormal { MappingInformationType, ReferenceInformationType, Normals }
    w.new_node("LayerElementNormal").unwrap();
    {
        let mut a = w.new_node("MappingInformationType").unwrap();
        a.append_string_direct(mapping).unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("ReferenceInformationType").unwrap();
        a.append_string_direct("Direct").unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("Normals").unwrap();
        a.append_arr_f64_from_iter(ArrayAttributeEncoding::Direct, normals.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap(); // Normals
    w.close_node().unwrap(); // LayerElementNormal
    w.close_node().unwrap(); // Geometry
    w.close_node().unwrap(); // Objects
    w.finalize_and_flush(&FbxFooter::default())
        .unwrap()
        .into_inner()
}

#[test]
fn fbx_reads_normals_by_control_point() {
    // 4 verts, one normal each (ByVertice/Direct) → per-vertex normals recovered.
    let verts = [
        0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        1.0, 1.0, 0.0, //
        0.0, 1.0, 0.0,
    ];
    let poly = [0i32, 1, 2, !3];
    let normals = [
        0.0, 0.0, 1.0, //
        0.0, 0.0, 1.0, //
        0.1, 0.0, 0.99, //
        0.0, 0.2, 0.98,
    ];
    let bytes = build_fbx_normals(&verts, &poly, "ByVertice", &normals);
    let obj = engine_io::parse_fbx(&bytes, "quad").expect("parse fbx");
    assert_eq!(obj.positions.len(), 4);
    assert_eq!(
        obj.normals.len(),
        4,
        "per-vertex normals parallel to positions"
    );
    assert!((obj.normals[2][0] - 0.1).abs() < 1e-5, "normal survives");
    assert!((obj.normals[3][1] - 0.2).abs() < 1e-5);
}

#[test]
fn fbx_reads_normals_by_polygon_vertex() {
    // ByPolygonVertex: one normal per polygon corner (parallel to poly), de-indexed
    // to per-control-point (first corner wins). Two triangles, 4 verts.
    let verts = [
        0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        1.0, 1.0, 0.0,
    ];
    let poly = [0i32, 1, !2, 2, 1, !3]; // 6 corners referencing verts 0,1,2,2,1,3
                                        // 6 corner normals; corner 0→v0, 1→v1, 2→v2, 3→v2, 4→v1, 5→v3.
    let normals = [
        1.0, 0.0, 0.0, // v0
        0.0, 1.0, 0.0, // v1
        0.0, 0.0, 1.0, // v2 (first)
        0.5, 0.5, 0.0, // v2 (second, ignored — first wins)
        0.9, 0.1, 0.0, // v1 (ignored)
        0.0, 0.5, 0.5, // v3
    ];
    let bytes = build_fbx_normals(&verts, &poly, "ByPolygonVertex", &normals);
    let obj = engine_io::parse_fbx(&bytes, "tris").expect("parse fbx");
    assert_eq!(obj.normals.len(), 4);
    assert_eq!(obj.normals[0], [1.0, 0.0, 0.0]);
    assert_eq!(obj.normals[2], [0.0, 0.0, 1.0], "first corner for v2 wins");
    assert_eq!(obj.normals[3], [0.0, 0.5, 0.5]);
}

/// Column-major (glam) flat translation matrix, 16 f64.
fn translation(tx: f64, ty: f64, tz: f64) -> [f64; 16] {
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        tx, ty, tz, 1.0,
    ]
}

type W = Writer<Cursor<Vec<u8>>>;

/// Write the shared skin objects (Geometry 100 quad + bones 200/201 + Skin 300 +
/// Clusters 400/401) inside an already-open `Objects` node. Does NOT open/close
/// `Objects`. Returns the skin OO edges `(src, dst)`.
fn write_skin_objects(w: &mut W) -> Vec<(i64, i64)> {
    // Geometry (id 100) — a unit quad.
    {
        let mut a = w.new_node("Geometry").unwrap();
        a.append_i64(100).unwrap();
        a.append_string_direct("Geometry::mesh").unwrap();
        a.append_string_direct("Mesh").unwrap();
    }
    {
        let verts = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0f64,
        ];
        let mut a = w.new_node("Vertices").unwrap();
        a.append_arr_f64_from_iter(Enc::Direct, verts.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let poly = [0i32, 1, 2, !3];
        let mut a = w.new_node("PolygonVertexIndex").unwrap();
        a.append_arr_i32_from_iter(Enc::Direct, poly.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    w.close_node().unwrap(); // Geometry
    for (id, name) in [(200i64, "Model::Root"), (201, "Model::Child")] {
        {
            let mut a = w.new_node("Model").unwrap();
            a.append_i64(id).unwrap();
            a.append_string_direct(name).unwrap();
            a.append_string_direct("LimbNode").unwrap();
        }
        w.close_node().unwrap();
    }
    {
        let mut a = w.new_node("Deformer").unwrap();
        a.append_i64(300).unwrap();
        a.append_string_direct("Deformer::skin").unwrap();
        a.append_string_direct("Skin").unwrap();
    }
    w.close_node().unwrap();
    let clusters = [
        (
            400i64,
            vec![0i32, 1],
            vec![1.0f64, 0.5],
            translation(0.0, 0.0, 0.0),
        ),
        (
            401,
            vec![1, 2, 3],
            vec![0.5, 1.0, 1.0],
            translation(2.0, 0.0, 0.0),
        ),
    ];
    for (id, idxs, wts, link) in clusters {
        {
            let mut a = w.new_node("Deformer").unwrap();
            a.append_i64(id).unwrap();
            a.append_string_direct("SubDeformer::cluster").unwrap();
            a.append_string_direct("Cluster").unwrap();
        }
        {
            let mut a = w.new_node("Indexes").unwrap();
            a.append_arr_i32_from_iter(Enc::Direct, idxs.iter().copied())
                .unwrap();
        }
        w.close_node().unwrap();
        {
            let mut a = w.new_node("Weights").unwrap();
            a.append_arr_f64_from_iter(Enc::Direct, wts.iter().copied())
                .unwrap();
        }
        w.close_node().unwrap();
        {
            let mut a = w.new_node("TransformLink").unwrap();
            a.append_arr_f64_from_iter(Enc::Direct, link.iter().copied())
                .unwrap();
        }
        w.close_node().unwrap();
        w.close_node().unwrap(); // Deformer (Cluster)
    }
    vec![
        (300, 100), // Skin → Geometry
        (400, 300), // Cluster0 → Skin
        (401, 300), // Cluster1 → Skin
        (200, 400), // bone Root → Cluster0
        (201, 401), // bone Child → Cluster1
        (201, 200), // Child → Root (hierarchy)
    ]
}

/// Write an `AnimationCurve` object (KeyTime i64 + KeyValueFloat f32) inside an
/// open `Objects` node.
fn write_anim_curve(w: &mut W, id: i64, times: &[i64], vals: &[f32]) {
    {
        let mut a = w.new_node("AnimationCurve").unwrap();
        a.append_i64(id).unwrap();
        a.append_string_direct("AnimCurve::").unwrap();
        a.append_string_direct("").unwrap();
    }
    {
        let mut a = w.new_node("KeyTime").unwrap();
        a.append_arr_i64_from_iter(Enc::Direct, times.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("KeyValueFloat").unwrap();
        a.append_arr_f32_from_iter(Enc::Direct, vals.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    w.close_node().unwrap(); // AnimationCurve
}

/// Write a single-attribute object node `[id, name, subtype]` (no children) inside
/// an open `Objects` node.
fn write_obj_node(w: &mut W, node: &str, id: i64, name: &str, subtype: &str) {
    {
        let mut a = w.new_node(node).unwrap();
        a.append_i64(id).unwrap();
        a.append_string_direct(name).unwrap();
        a.append_string_direct(subtype).unwrap();
    }
    w.close_node().unwrap();
}

/// Write the `Connections` node from `(kind, src, dst, prop)` edges (`prop` is the
/// OP property string, empty for OO edges).
fn write_connections(w: &mut W, edges: &[(&str, i64, i64, &str)]) {
    w.new_node("Connections").unwrap();
    for &(kind, src, dst, prop) in edges {
        {
            let mut a = w.new_node("C").unwrap();
            a.append_string_direct(kind).unwrap();
            a.append_i64(src).unwrap();
            a.append_i64(dst).unwrap();
            if kind == "OP" {
                a.append_string_direct(prop).unwrap();
            }
        }
        w.close_node().unwrap();
    }
    w.close_node().unwrap(); // Connections
}

/// Write one quad Geometry + a Skin + a single Cluster binding all 4 of its verts
/// (weight 1) to whatever bone the caller wires the cluster to. Used to build a
/// multi-geometry shared-skeleton FBX. Returns the Skin id.
fn write_geom_with_cluster(w: &mut W, geom_id: i64, skin_id: i64, cluster_id: i64, x_off: f64) {
    {
        let mut a = w.new_node("Geometry").unwrap();
        a.append_i64(geom_id).unwrap();
        a.append_string_direct("Geometry::mesh").unwrap();
        a.append_string_direct("Mesh").unwrap();
    }
    {
        let verts = [
            x_off,
            0.0,
            0.0, //
            x_off + 1.0,
            0.0,
            0.0, //
            x_off + 1.0,
            1.0,
            0.0, //
            x_off,
            1.0,
            0.0,
        ];
        let mut a = w.new_node("Vertices").unwrap();
        a.append_arr_f64_from_iter(Enc::Direct, verts.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("PolygonVertexIndex").unwrap();
        a.append_arr_i32_from_iter(Enc::Direct, [0i32, 1, 2, !3].iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    w.close_node().unwrap(); // Geometry
    {
        let mut a = w.new_node("Deformer").unwrap();
        a.append_i64(skin_id).unwrap();
        a.append_string_direct("Deformer::skin").unwrap();
        a.append_string_direct("Skin").unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("Deformer").unwrap();
        a.append_i64(cluster_id).unwrap();
        a.append_string_direct("SubDeformer::cluster").unwrap();
        a.append_string_direct("Cluster").unwrap();
    }
    {
        let mut a = w.new_node("Indexes").unwrap();
        a.append_arr_i32_from_iter(Enc::Direct, [0i32, 1, 2, 3].iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("Weights").unwrap();
        a.append_arr_f64_from_iter(Enc::Direct, [1.0f64, 1.0, 1.0, 1.0].iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("TransformLink").unwrap();
        a.append_arr_f64_from_iter(Enc::Direct, translation(0.0, 0.0, 0.0).iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    w.close_node().unwrap(); // Deformer (Cluster)
}

/// Build a MULTI-geometry FBX: two quads (8 verts total), each skinned to the SAME
/// bone (Root, id 200) via its own Skin+Cluster — the shared-skeleton case.
fn build_fbx_two_geom_shared_skeleton() -> Vec<u8> {
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap();
    write_obj_node(&mut w, "Model", 200, "Model::Root", "LimbNode");
    write_geom_with_cluster(&mut w, 100, 300, 400, 0.0);
    write_geom_with_cluster(&mut w, 101, 301, 401, 2.0);
    w.close_node().unwrap(); // Objects
    write_connections(
        &mut w,
        &[
            ("OO", 300, 100, ""), // Skin A → Geom A
            ("OO", 400, 300, ""), // Cluster A → Skin A
            ("OO", 200, 400, ""), // Root → Cluster A
            ("OO", 301, 101, ""), // Skin B → Geom B
            ("OO", 401, 301, ""), // Cluster B → Skin B
            ("OO", 200, 401, ""), // Root → Cluster B (SAME bone → shared skeleton)
        ],
    );
    w.finalize_and_flush(&FbxFooter::default())
        .unwrap()
        .into_inner()
}

#[test]
fn fbx_reads_multi_geometry_shared_skeleton_skin() {
    let bytes = build_fbx_two_geom_shared_skeleton();
    let obj = engine_io::parse_fbx(&bytes, "multi").expect("parse multi-geom fbx");
    assert_eq!(obj.positions.len(), 8, "two quads concatenated");
    let skin = obj.skin.expect("merged skin present");
    assert_eq!(skin.joints.len(), 1, "shared skeleton: one joint (Root)");
    assert_eq!(skin.joints[0].name, "Root");
    // Weights parallel to all 8 positions; every vertex (both geometries, at their
    // base offsets) bound to joint 0.
    assert_eq!(skin.vertex_joints.len(), 8);
    assert_eq!(skin.vertex_weights.len(), 8);
    for v in 0..8 {
        assert_eq!(skin.vertex_joints[v][0], 0, "vertex {v} → Root");
        assert!(
            (skin.vertex_weights[v][0] - 1.0).abs() < 1e-5,
            "vertex {v} weight"
        );
    }
    // Geometry B's verts (4..8) are offset +2 in x — confirms concatenation order.
    assert!((obj.positions[4][0] - 2.0).abs() < 1e-5);
}

/// Build a minimal RIGGED FBX: one Geometry (4-vert quad) + two LimbNode bones
/// (root, child) + a Skin deformer with two Clusters, wired by the Connections
/// OO graph. Cluster0→bone0 (verts 0,1), Cluster1→bone1 (verts 1,2,3).
fn build_fbx_skin() -> Vec<u8> {
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap();
    let skin_edges = write_skin_objects(&mut w);
    w.close_node().unwrap(); // Objects
    let edges: Vec<(&str, i64, i64, &str)> =
        skin_edges.iter().map(|&(s, d)| ("OO", s, d, "")).collect();
    write_connections(&mut w, &edges);
    w.finalize_and_flush(&FbxFooter::default())
        .unwrap()
        .into_inner()
}

/// FBX time unit (`FBXSDK_TC_SECOND`): ticks per second.
const FBX_TICK: i64 = 46_186_158_000;

/// Build a rigged + ANIMATED FBX: the skin fixture above plus an AnimationStack
/// ("Take001") → Layer → two curve nodes on bone Root (id 200): a Translation
/// node (X 0→5, Y/Z 0) and a Rotation node (Z 0→90°, X/Y absent → 0). Two keys at
/// t=0 and t=1s.
fn build_fbx_animated() -> Vec<u8> {
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap();
    let skin_edges = write_skin_objects(&mut w);
    // Animation objects.
    write_obj_node(&mut w, "AnimationStack", 500, "AnimStack::Take001", "");
    write_obj_node(&mut w, "AnimationLayer", 501, "AnimLayer::Base", "");
    write_obj_node(&mut w, "AnimationCurveNode", 600, "AnimCurveNode::T", "");
    write_obj_node(&mut w, "AnimationCurveNode", 610, "AnimCurveNode::R", "");
    // Translation sub-curves X/Y/Z (700/701/702).
    write_anim_curve(&mut w, 700, &[0, FBX_TICK], &[0.0, 5.0]);
    write_anim_curve(&mut w, 701, &[0, FBX_TICK], &[0.0, 0.0]);
    write_anim_curve(&mut w, 702, &[0, FBX_TICK], &[0.0, 0.0]);
    // Rotation sub-curve Z only (712) — 0→90°.
    write_anim_curve(&mut w, 712, &[0, FBX_TICK], &[0.0, 90.0]);
    w.close_node().unwrap(); // Objects

    let mut edges: Vec<(&str, i64, i64, &str)> =
        skin_edges.iter().map(|&(s, d)| ("OO", s, d, "")).collect();
    edges.extend_from_slice(&[
        ("OO", 501, 500, ""),                // layer → stack
        ("OO", 600, 501, ""),                // T node → layer
        ("OO", 610, 501, ""),                // R node → layer
        ("OP", 600, 200, "Lcl Translation"), // T node → bone Root
        ("OP", 700, 600, "d|X"),             // curve → T node
        ("OP", 701, 600, "d|Y"),
        ("OP", 702, 600, "d|Z"),
        ("OP", 610, 200, "Lcl Rotation"), // R node → bone Root
        ("OP", 712, 610, "d|Z"),          // curve → R node
    ]);
    write_connections(&mut w, &edges);
    w.finalize_and_flush(&FbxFooter::default())
        .unwrap()
        .into_inner()
}

/// A Translation track on bone Root whose X and Y sub-curves key at DIFFERENT
/// times: X at {0, 1s}→{0,10}, Y at {0, 0.5s}→{0,4}. The channel should sample at
/// the union {0, 0.5, 1} with each axis linearly interpolated/clamped.
fn build_fbx_animated_unaligned() -> Vec<u8> {
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap();
    let skin_edges = write_skin_objects(&mut w);
    write_obj_node(&mut w, "AnimationStack", 500, "AnimStack::Take001", "");
    write_obj_node(&mut w, "AnimationLayer", 501, "AnimLayer::Base", "");
    write_obj_node(&mut w, "AnimationCurveNode", 600, "AnimCurveNode::T", "");
    write_anim_curve(&mut w, 700, &[0, FBX_TICK], &[0.0, 10.0]); // X keys at 0, 1s
    write_anim_curve(&mut w, 701, &[0, FBX_TICK / 2], &[0.0, 4.0]); // Y keys at 0, 0.5s
    w.close_node().unwrap(); // Objects

    let mut edges: Vec<(&str, i64, i64, &str)> =
        skin_edges.iter().map(|&(s, d)| ("OO", s, d, "")).collect();
    edges.extend_from_slice(&[
        ("OO", 501, 500, ""),
        ("OO", 600, 501, ""),
        ("OP", 600, 200, "Lcl Translation"),
        ("OP", 700, 600, "d|X"),
        ("OP", 701, 600, "d|Y"),
    ]);
    write_connections(&mut w, &edges);
    w.finalize_and_flush(&FbxFooter::default())
        .unwrap()
        .into_inner()
}

#[test]
fn fbx_reads_unaligned_animation() {
    let bytes = build_fbx_animated_unaligned();
    let obj = engine_io::parse_fbx(&bytes, "anim").expect("parse fbx");
    let clip = &obj.animations[0];
    let trans = clip
        .channels
        .iter()
        .find(|c| c.path == engine_api::rig::AnimationPath::Translation)
        .expect("translation channel");
    assert!(trans.is_consistent());
    // Union of key times {0, 0.5, 1}.
    assert_eq!(trans.times, vec![0.0, 0.5, 1.0]);
    // key@0 = (0,0,0); key@0.5 = (X lerp=5, Y exact=4, Z=0);
    // key@1 = (X=10, Y clamped=4, Z=0).
    assert_eq!(trans.values[0..3], [0.0, 0.0, 0.0]);
    assert!(
        (trans.values[3] - 5.0).abs() < 1e-4,
        "X lerped to 5 at t=0.5"
    );
    assert!((trans.values[4] - 4.0).abs() < 1e-4, "Y exact 4 at t=0.5");
    assert!((trans.values[6] - 10.0).abs() < 1e-4, "X=10 at t=1");
    assert!(
        (trans.values[7] - 4.0).abs() < 1e-4,
        "Y clamped to 4 at t=1"
    );
}

#[test]
fn fbx_reads_skin_weights_and_joints() {
    let bytes = build_fbx_skin();
    let obj = engine_io::parse_fbx(&bytes, "rigged").expect("parse rigged fbx");
    assert_eq!(obj.positions.len(), 4);
    let skin = obj.skin.expect("skin present");
    // Two joints, root first (Cluster0 discovered first).
    assert_eq!(skin.joints.len(), 2);
    assert_eq!(skin.joints[0].name, "Root");
    assert_eq!(skin.joints[1].name, "Child");
    assert_eq!(skin.joints[0].parent, None);
    assert_eq!(skin.joints[1].parent, Some(0));
    // Child bind is translation(+2) in x → inverse_bind is translation(-2).
    assert_eq!(skin.inverse_bind.len(), 2);
    assert!((skin.inverse_bind[1][3][0] - (-2.0)).abs() < 1e-5);
    // Child local (parent⁻¹·world) = translation(+2).
    assert!((skin.joints[1].local[3][0] - 2.0).abs() < 1e-5);
    // Per-vertex weights: v0→j0 only; v1→j0(0.5)+j1(0.5); v2,v3→j1.
    assert_eq!(skin.vertex_joints.len(), 4);
    assert_eq!(skin.vertex_joints[0][0], 0);
    assert!((skin.vertex_weights[0][0] - 1.0).abs() < 1e-5);
    // v1 has two influences (sorted by weight desc; equal 0.5 → either order).
    let v1: Vec<(u32, f32)> = skin.vertex_joints[1]
        .iter()
        .zip(&skin.vertex_weights[1])
        .filter(|(_, &w)| w > 0.0)
        .map(|(&j, &w)| (j, w))
        .collect();
    assert_eq!(v1.len(), 2, "v1 shared by both joints");
    assert!(v1.iter().any(|&(j, w)| j == 0 && (w - 0.5).abs() < 1e-5));
    assert!(v1.iter().any(|&(j, w)| j == 1 && (w - 0.5).abs() < 1e-5));
    // v2 → joint 1 fully.
    assert_eq!(skin.vertex_joints[2][0], 1);
    assert!((skin.vertex_weights[2][0] - 1.0).abs() < 1e-5);
}

#[test]
fn fbx_reads_animation_curves() {
    let bytes = build_fbx_animated();
    let obj = engine_io::parse_fbx(&bytes, "anim").expect("parse animated fbx");
    assert!(obj.skin.is_some(), "skin still read");
    assert_eq!(obj.animations.len(), 1, "one AnimationStack → one clip");
    let clip = &obj.animations[0];
    assert_eq!(clip.name, "Take001");
    assert!((clip.duration - 1.0).abs() < 1e-4, "duration 1s");
    assert_eq!(clip.channels.len(), 2, "translation + rotation channels");

    let trans = clip
        .channels
        .iter()
        .find(|c| c.path == engine_api::rig::AnimationPath::Translation)
        .expect("translation channel");
    assert_eq!(trans.target_joint, 0, "targets bone Root = joint 0");
    assert!(trans.is_consistent());
    assert_eq!(trans.times, vec![0.0, 1.0]);
    // Key 0 = (0,0,0); key 1 = (5,0,0).
    assert_eq!(trans.values, vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0]);

    let rot = clip
        .channels
        .iter()
        .find(|c| c.path == engine_api::rig::AnimationPath::Rotation)
        .expect("rotation channel");
    assert!(rot.is_consistent());
    assert_eq!(rot.times, vec![0.0, 1.0]);
    // Key 0 = identity quat (0,0,0,1); key 1 = 90° about Z = (0,0,√½,√½).
    assert!((rot.values[0]).abs() < 1e-5);
    assert!((rot.values[1]).abs() < 1e-5);
    assert!((rot.values[2]).abs() < 1e-5);
    assert!((rot.values[3] - 1.0).abs() < 1e-5);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    assert!((rot.values[4]).abs() < 1e-5, "qx≈0");
    assert!((rot.values[5]).abs() < 1e-5, "qy≈0");
    assert!((rot.values[6] - s).abs() < 1e-5, "qz≈√½");
    assert!((rot.values[7] - s).abs() < 1e-5, "qw≈√½");
}

#[test]
fn fbx_reads_quad_geometry() {
    // A unit quad: 4 vertices, 1 polygon [0,1,2,3] (last index negated as the
    // FBX polygon terminator: !3 = -4).
    let verts = [
        0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        1.0, 1.0, 0.0, //
        0.0, 1.0, 0.0,
    ];
    let poly = [0i32, 1, 2, !3]; // !3 == -4
    let bytes = build_fbx(&verts, &poly);

    let obj = engine_io::parse_fbx(&bytes, "quad").expect("parse fbx");
    assert_eq!(obj.positions.len(), 4);
    assert_eq!(obj.positions[2], [1.0, 1.0, 0.0]);
    // Quad fan-triangulated → (0,1,2), (0,2,3).
    assert_eq!(obj.tris, vec![[0, 1, 2], [0, 2, 3]]);
}

#[test]
fn fbx_reads_two_triangles() {
    // 4 verts, two triangles: [0,1,2] and [2,1,3] (each terminated by a negative).
    let verts = [
        0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        1.0, 1.0, 0.0,
    ];
    let poly = [0i32, 1, !2, 2, 1, !3];
    let bytes = build_fbx(&verts, &poly);

    // Exercise the extension dispatcher too by writing to a temp file.
    let path = std::env::temp_dir().join("mguso_fbx_two_tri.fbx");
    std::fs::write(&path, &bytes).unwrap();
    let obj = engine_io::load_mesh(&path).expect("load fbx via dispatcher");
    let _ = std::fs::remove_file(&path);

    assert_eq!(obj.positions.len(), 4);
    assert_eq!(obj.tris, vec![[0, 1, 2], [2, 1, 3]]);
}

// ---------------------------------------------------------------------------
// zlib-encoded array attributes (`encoding = 1`).
//
// EVERY other fixture in this file uses `ArrayAttributeEncoding::Direct`, but
// every real DCC exporter (Blender FBX IO, Autodesk FBX SDK) writes zlib. The
// whole suite was green while 100% of real-world files failed to load.
// ---------------------------------------------------------------------------

/// Same shape as `build_fbx`, with the array encoding under the caller's control.
fn build_fbx_enc(verts: &[f64], poly_idx: &[i32], enc: ArrayAttributeEncoding) -> Vec<u8> {
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap();
    w.new_node("Geometry").unwrap();
    {
        let mut a = w.new_node("Vertices").unwrap();
        a.append_arr_f64_from_iter(enc, verts.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("PolygonVertexIndex").unwrap();
        a.append_arr_i32_from_iter(enc, poly_idx.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    w.close_node().unwrap(); // Geometry
    w.close_node().unwrap(); // Objects
    w.finalize_and_flush(&FbxFooter::default())
        .unwrap()
        .into_inner()
}

/// An `n`×`n` grid of quads: flat f64 positions + PolygonVertexIndex with the
/// last corner of each quad negated. Big enough that zlib actually compresses.
fn grid(n: usize) -> (Vec<f64>, Vec<i32>) {
    let mut verts = Vec::new();
    for j in 0..n {
        for i in 0..n {
            verts.extend_from_slice(&[i as f64, j as f64, 0.0]);
        }
    }
    let mut poly = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let v = (j * n + i) as i32;
            let r = v + n as i32;
            poly.extend_from_slice(&[v, v + 1, r + 1, !r]);
        }
    }
    (verts, poly)
}

#[test]
fn fbx_reads_zlib_encoded_arrays() {
    let (verts, poly) = grid(20);
    let expect_verts = verts.len() / 3;
    let expect_tris = poly.len() / 4 * 2;

    // Control: the identical mesh with Direct arrays must parse (it always has).
    let direct = engine_io::parse_fbx(
        &build_fbx_enc(&verts, &poly, ArrayAttributeEncoding::Direct),
        "grid",
    )
    .expect("Direct-encoded fixture parses");
    assert_eq!(direct.positions.len(), expect_verts);
    assert_eq!(direct.tris.len(), expect_tris);

    // The real-world case: zlib-compressed arrays.
    let obj = engine_io::parse_fbx(
        &build_fbx_enc(&verts, &poly, ArrayAttributeEncoding::Zlib),
        "grid",
    )
    .expect("zlib-encoded fixture parses");
    assert_eq!(obj.positions, direct.positions, "zlib == direct positions");
    assert_eq!(obj.tris, direct.tris, "zlib == direct topology");
}

#[test]
fn fbx_reads_zlib_arrays_with_following_siblings() {
    // A zlib array must not disturb the reader position for the nodes that come
    // after it: LayerElementNormal follows PolygonVertexIndex here.
    let (verts, poly) = grid(12);
    let mut w = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
    w.new_node("Objects").unwrap();
    w.new_node("Geometry").unwrap();
    {
        let mut a = w.new_node("Vertices").unwrap();
        a.append_arr_f64_from_iter(Enc::Zlib, verts.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("PolygonVertexIndex").unwrap();
        a.append_arr_i32_from_iter(Enc::Zlib, poly.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap();
    w.new_node("LayerElementNormal").unwrap();
    {
        let mut a = w.new_node("MappingInformationType").unwrap();
        a.append_string_direct("ByVertice").unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("ReferenceInformationType").unwrap();
        a.append_string_direct("Direct").unwrap();
    }
    w.close_node().unwrap();
    {
        let normals: Vec<f64> = (0..verts.len() / 3).flat_map(|_| [0.0, 0.0, 1.0]).collect();
        let mut a = w.new_node("Normals").unwrap();
        a.append_arr_f64_from_iter(Enc::Zlib, normals.iter().copied())
            .unwrap();
    }
    w.close_node().unwrap(); // Normals
    w.close_node().unwrap(); // LayerElementNormal
    w.close_node().unwrap(); // Geometry
    w.close_node().unwrap(); // Objects
    let bytes = w
        .finalize_and_flush(&FbxFooter::default())
        .unwrap()
        .into_inner();

    let obj = engine_io::parse_fbx(&bytes, "grid").expect("zlib + siblings parses");
    assert_eq!(obj.positions.len(), verts.len() / 3);
    assert_eq!(
        obj.normals.len(),
        verts.len() / 3,
        "normals after zlib array"
    );
    assert_eq!(obj.normals[0], [0.0, 0.0, 1.0]);
}
