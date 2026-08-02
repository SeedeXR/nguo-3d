// SPDX-License-Identifier: GPL-2.0-or-later
//! glTF 2.0 / GLB mesh I/O — the web/game-engine/DCC interchange standard.
//!
//! Import uses the `gltf` crate (MIT/Apache-2.0), which parses both text `.gltf`
//! (+ external/embedded buffers) and binary `.glb`, and — importantly for the
//! roadmap — natively exposes **skins and animations**. Imports positions,
//! triangle indices, per-vertex normals/UVs (when every primitive supplies
//! them, node world transforms applied), plus the **rig**: the first skin's
//! joints (hierarchy + inverse-bind matrices), per-vertex JOINTS_0/WEIGHTS_0,
//! and animation clips whose channels target skin joints (TRS keyframes) →
//! `engine_api::{Skin, AnimationClip}`.
//!
//! Export writes a self-contained binary **GLB** (single mesh/primitive) — a
//! hand-assembled container for full control over the byte layout — including
//! NORMAL / TEXCOORD_0 when present, and the full **rig** when the Object carries
//! one: JOINTS_0/WEIGHTS_0 vertex attrs, the joint node hierarchy + skin, the
//! inverse-bind-matrix accessor, and animation sampler/channel JSON. A rigged
//! mesh round-trips load→save→load (`rigged_glb_write_round_trip`).

use anyhow::{bail, Context, Result};
use engine_api::rig::{AnimationChannel, AnimationClip, AnimationPath, Joint, Skin};
use engine_api::Object;
use glam::Mat4;
use std::path::Path;

/// Load a `.gltf` or `.glb` file into an [`Object`]. All mesh primitives across
/// all scene nodes are concatenated into one vertex space (node world transforms
/// applied), mirroring the OBJ loader's single-object model.
pub fn load_gltf(path: impl AsRef<Path>) -> Result<Object> {
    let path = path.as_ref();
    let (doc, buffers, _images) =
        gltf::import(path).with_context(|| format!("importing glTF {}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".into());
    from_document(&doc, &buffers, name)
}

/// Parse an in-memory `.gltf`/`.glb` byte slice (auto-detected by the crate).
pub fn parse_gltf_slice(bytes: &[u8], name: impl Into<String>) -> Result<Object> {
    let (doc, buffers, _images) = gltf::import_slice(bytes).context("parsing glTF/GLB bytes")?;
    from_document(&doc, &buffers, name.into())
}

/// Accumulator kept parallel to `positions`; normals/uvs are dropped entirely if
/// any imported primitive lacks them (so the arrays stay either empty or aligned).
struct MeshAccum {
    positions: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    /// Per-vertex skin influences (JOINTS_0 / WEIGHTS_0), parallel to positions.
    vjoints: Vec<[u32; 4]>,
    vweights: Vec<[f32; 4]>,
    has_normals: bool,
    has_uvs: bool,
    has_skin_verts: bool,
}

fn from_document(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    name: String,
) -> Result<Object> {
    let mut acc = MeshAccum {
        positions: Vec::new(),
        tris: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        vjoints: Vec::new(),
        vweights: Vec::new(),
        has_normals: true,
        has_uvs: true,
        has_skin_verts: true,
    };
    // Walk each scene's node hierarchy, accumulating world transforms.
    for scene in doc.scenes() {
        for node in scene.nodes() {
            walk_node(&node, Mat4::IDENTITY, buffers, &mut acc)?;
        }
    }
    if acc.positions.is_empty() || acc.tris.is_empty() {
        bail!("glTF contained no triangle mesh data");
    }
    // Keep attributes only if every primitive supplied them (arrays parallel).
    let normals = if acc.has_normals && acc.normals.len() == acc.positions.len() {
        acc.normals
    } else {
        Vec::new()
    };
    let uvs = if acc.has_uvs && acc.uvs.len() == acc.positions.len() {
        acc.uvs
    } else {
        Vec::new()
    };
    // Per-vertex skin influences, kept only if parallel to positions.
    let (vjoints, vweights) = if acc.has_skin_verts
        && acc.vjoints.len() == acc.positions.len()
        && acc.vweights.len() == acc.positions.len()
    {
        (acc.vjoints, acc.vweights)
    } else {
        (Vec::new(), Vec::new())
    };
    // Skin + animations (first skin only for now; JOINTS_0 already index the
    // skin's joint list, which we preserve in order).
    let (skin, animations) = read_rig(doc, buffers, vjoints, vweights);
    Ok(Object {
        name,
        positions: acc.positions,
        tris: acc.tris,
        material: Default::default(),
        preset: None,
        pins: Vec::new(),
        is_static: false,
        normals,
        uvs,
        skin,
        animations,
    })
}

fn walk_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    acc: &mut MeshAccum,
) -> Result<()> {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue; // v1: triangle meshes only
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let Some(pos_iter) = reader.read_positions() else {
                continue;
            };
            let base = acc.positions.len() as u32;
            for p in pos_iter {
                let w = world.transform_point3(glam::Vec3::from(p));
                acc.positions.push([w.x, w.y, w.z]);
            }
            let count = acc.positions.len() as u32 - base;
            // Normals (rotated by the node's linear part).
            match reader.read_normals() {
                Some(nrm) => {
                    for nn in nrm {
                        let w = world
                            .transform_vector3(glam::Vec3::from(nn))
                            .normalize_or_zero();
                        acc.normals.push([w.x, w.y, w.z]);
                    }
                }
                None => acc.has_normals = false,
            }
            // UVs (TEXCOORD_0), untransformed.
            match reader.read_tex_coords(0).map(|t| t.into_f32()) {
                Some(uv) => {
                    for c in uv {
                        acc.uvs.push(c);
                    }
                }
                None => acc.has_uvs = false,
            }
            // Skin influences (JOINTS_0 → indices into the skin's joint list,
            // WEIGHTS_0 → weights). Parallel to positions.
            match (
                reader.read_joints(0).map(|j| j.into_u16()),
                reader.read_weights(0).map(|w| w.into_f32()),
            ) {
                (Some(js), Some(ws)) => {
                    for j in js {
                        acc.vjoints
                            .push([j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32]);
                    }
                    for w in ws {
                        acc.vweights.push(w);
                    }
                }
                _ => acc.has_skin_verts = false,
            }
            // Indexed or non-indexed (implicit 0,1,2,…) triangles.
            match reader.read_indices() {
                Some(idx) => {
                    let flat: Vec<u32> = idx.into_u32().collect();
                    for t in flat.chunks_exact(3) {
                        acc.tris.push([base + t[0], base + t[1], base + t[2]]);
                    }
                }
                None => {
                    for t in (0..count).step_by(3) {
                        if t + 2 < count {
                            acc.tris.push([base + t, base + t + 1, base + t + 2]);
                        }
                    }
                }
            }
        }
    }
    for child in node.children() {
        walk_node(&child, world, buffers, acc)?;
    }
    Ok(())
}

/// Read the first skin (joints + inverse-bind + the given per-vertex influences)
/// and all animation clips whose channels target skin joints. Returns
/// `(None, [])` when the document has no skin.
fn read_rig(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    vjoints: Vec<[u32; 4]>,
    vweights: Vec<[f32; 4]>,
) -> (Option<Skin>, Vec<AnimationClip>) {
    let Some(gskin) = doc.skins().next() else {
        return (None, Vec::new());
    };
    // glTF node index → our joint index (skin.joints() order is preserved).
    let joint_nodes: Vec<usize> = gskin.joints().map(|n| n.index()).collect();
    let node_to_joint: std::collections::HashMap<usize, u32> = joint_nodes
        .iter()
        .enumerate()
        .map(|(ji, &ni)| (ni, ji as u32))
        .collect();

    // Parent of each joint = the joint whose glTF node lists it as a child.
    let mut parents = vec![None; gskin.joints().count()];
    for parent in gskin.joints() {
        if let Some(&pj) = node_to_joint.get(&parent.index()) {
            for child in parent.children() {
                if let Some(&cj) = node_to_joint.get(&child.index()) {
                    parents[cj as usize] = Some(pj);
                }
            }
        }
    }
    let joints: Vec<Joint> = gskin
        .joints()
        .enumerate()
        .map(|(ji, node)| Joint {
            name: node.name().unwrap_or("joint").to_string(),
            parent: parents[ji],
            local: node.transform().matrix(),
        })
        .collect();

    let reader = gskin.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
    let inverse_bind: Vec<[[f32; 4]; 4]> = match reader.read_inverse_bind_matrices() {
        Some(it) => it.collect(),
        None => vec![Mat4::IDENTITY.to_cols_array_2d(); joints.len()],
    };

    let skin = Skin {
        joints,
        inverse_bind,
        vertex_joints: vjoints,
        vertex_weights: vweights,
    };

    // Animations: channels targeting joint nodes become joint TRS tracks.
    let mut clips = Vec::new();
    for anim in doc.animations() {
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for ch in anim.channels() {
            let Some(&target_joint) = node_to_joint.get(&ch.target().node().index()) else {
                continue; // channel targets a non-joint node
            };
            let path = match ch.target().property() {
                gltf::animation::Property::Translation => AnimationPath::Translation,
                gltf::animation::Property::Rotation => AnimationPath::Rotation,
                gltf::animation::Property::Scale => AnimationPath::Scale,
                gltf::animation::Property::MorphTargetWeights => continue,
            };
            let creader = ch.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let times: Vec<f32> = match creader.read_inputs() {
                Some(it) => it.collect(),
                None => continue,
            };
            if let Some(&t) = times.last() {
                duration = duration.max(t);
            }
            let mut values: Vec<f32> = Vec::new();
            match creader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
                    for v in it {
                        values.extend_from_slice(&v);
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
                    for v in it {
                        values.extend_from_slice(&v);
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(rot)) => {
                    for v in rot.into_f32() {
                        values.extend_from_slice(&v);
                    }
                }
                _ => continue,
            }
            channels.push(AnimationChannel {
                target_joint,
                path,
                times,
                values,
            });
        }
        if !channels.is_empty() {
            clips.push(AnimationClip {
                name: anim.name().unwrap_or("clip").to_string(),
                duration,
                channels,
            });
        }
    }
    (Some(skin), clips)
}

/// Serialize an [`Object`] to a self-contained binary GLB (glTF 2.0).
/// Single mesh, single triangle primitive: POSITION (f32 VEC3) + u32 indices.
pub fn write_glb(obj: &Object) -> Result<Vec<u8>> {
    let n = obj.positions.len();
    if n == 0 || obj.tris.is_empty() {
        bail!("cannot export empty mesh to GLB");
    }
    // Binary buffer: positions, then indices (both 4-byte aligned naturally).
    let mut bin: Vec<u8> = Vec::with_capacity(n * 12 + obj.tris.len() * 12);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in &obj.positions {
        for c in 0..3 {
            min[c] = min[c].min(p[c]);
            max[c] = max[c].max(p[c]);
            bin.extend_from_slice(&p[c].to_le_bytes());
        }
    }
    let pos_bytes = bin.len();
    for t in &obj.tris {
        for &i in t {
            bin.extend_from_slice(&i.to_le_bytes());
        }
    }
    let idx_bytes = bin.len() - pos_bytes;
    let idx_count = obj.tris.len() * 3;

    // Build bufferViews/accessors/attributes, adding NORMAL / TEXCOORD_0 when the
    // Object carries them (parallel to positions). Component types: 5126=FLOAT,
    // 5125=UNSIGNED_INT. Targets: 34962=ARRAY_BUFFER, 34963=ELEMENT_ARRAY_BUFFER.
    let mut buffer_views = vec![
        serde_json::json!({ "buffer": 0, "byteOffset": 0, "byteLength": pos_bytes, "target": 34962 }),
        serde_json::json!({ "buffer": 0, "byteOffset": pos_bytes, "byteLength": idx_bytes, "target": 34963 }),
    ];
    let mut accessors = vec![
        serde_json::json!({ "bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": min, "max": max }),
        serde_json::json!({ "bufferView": 1, "componentType": 5125, "count": idx_count, "type": "SCALAR" }),
    ];
    let mut attributes = serde_json::Map::new();
    attributes.insert("POSITION".into(), serde_json::json!(0));

    if obj.normals.len() == n {
        let off = bin.len();
        for nn in &obj.normals {
            for &c in nn {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        buffer_views.push(serde_json::json!({ "buffer": 0, "byteOffset": off, "byteLength": n * 12, "target": 34962 }));
        let acc_idx = accessors.len();
        accessors.push(serde_json::json!({ "bufferView": buffer_views.len() - 1, "componentType": 5126, "count": n, "type": "VEC3" }));
        attributes.insert("NORMAL".into(), serde_json::json!(acc_idx));
    }
    if obj.uvs.len() == n {
        let off = bin.len();
        for uv in &obj.uvs {
            for &c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        buffer_views.push(serde_json::json!({ "buffer": 0, "byteOffset": off, "byteLength": n * 8, "target": 34962 }));
        let acc_idx = accessors.len();
        accessors.push(serde_json::json!({ "bufferView": buffer_views.len() - 1, "componentType": 5126, "count": n, "type": "VEC2" }));
        attributes.insert("TEXCOORD_0".into(), serde_json::json!(acc_idx));
    }

    // Node 0 = the mesh node; joint nodes (when skinned) are nodes 1..=J so
    // `node index = 1 + joint index` (the read side maps them back the same way).
    let mut mesh_node = serde_json::json!({ "mesh": 0, "name": obj.name });
    let mut nodes: Vec<serde_json::Value> = Vec::new();
    let mut scene_nodes: Vec<u32> = vec![0];
    let mut skins: Vec<serde_json::Value> = Vec::new();
    let mut animations: Vec<serde_json::Value> = Vec::new();

    if let Some(skin) = &obj.skin {
        // JOINTS_0 (u16 VEC4) + WEIGHTS_0 (f32 VEC4) when parallel to positions.
        if skin.vertex_joints.len() == n && skin.vertex_weights.len() == n {
            let off = bin.len();
            for j in &skin.vertex_joints {
                for &v in j {
                    bin.extend_from_slice(&(v as u16).to_le_bytes());
                }
            }
            buffer_views.push(serde_json::json!({ "buffer": 0, "byteOffset": off, "byteLength": n * 8, "target": 34962 }));
            let a = accessors.len();
            accessors.push(serde_json::json!({ "bufferView": buffer_views.len() - 1, "componentType": 5123, "count": n, "type": "VEC4" }));
            attributes.insert("JOINTS_0".into(), serde_json::json!(a));

            let off = bin.len();
            for w in &skin.vertex_weights {
                for &c in w {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
            buffer_views.push(serde_json::json!({ "buffer": 0, "byteOffset": off, "byteLength": n * 16, "target": 34962 }));
            let a = accessors.len();
            accessors.push(serde_json::json!({ "bufferView": buffer_views.len() - 1, "componentType": 5126, "count": n, "type": "VEC4" }));
            attributes.insert("WEIGHTS_0".into(), serde_json::json!(a));
        }
        // Inverse-bind matrices (MAT4, column-major flatten).
        let off = bin.len();
        for m in &skin.inverse_bind {
            for col in m {
                for &c in col {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        buffer_views.push(serde_json::json!({ "buffer": 0, "byteOffset": off, "byteLength": skin.joints.len() * 64 }));
        let ibm_acc = accessors.len();
        accessors.push(serde_json::json!({ "bufferView": buffer_views.len() - 1, "componentType": 5126, "count": skin.joints.len(), "type": "MAT4" }));

        // Joint nodes (node index = 1 + joint index); children from `parent`.
        for (j, joint) in skin.joints.iter().enumerate() {
            let children: Vec<u32> = skin
                .joints
                .iter()
                .enumerate()
                .filter(|(_, jj)| jj.parent == Some(j as u32))
                .map(|(k, _)| 1 + k as u32)
                .collect();
            let matrix: Vec<f32> = joint.local.iter().flatten().copied().collect();
            let mut nd = serde_json::json!({ "name": joint.name, "matrix": matrix });
            if !children.is_empty() {
                nd["children"] = serde_json::json!(children);
            }
            nodes.push(nd);
            if joint.parent.is_none() {
                scene_nodes.push(1 + j as u32); // roots into the scene
            }
        }
        let joint_node_indices: Vec<u32> = (0..skin.joints.len() as u32).map(|j| 1 + j).collect();
        skins.push(
            serde_json::json!({ "joints": joint_node_indices, "inverseBindMatrices": ibm_acc }),
        );
        mesh_node["skin"] = serde_json::json!(0);

        // Animation clips → samplers + channels targeting joint nodes.
        for clip in &obj.animations {
            let mut samplers: Vec<serde_json::Value> = Vec::new();
            let mut channels: Vec<serde_json::Value> = Vec::new();
            for ch in &clip.channels {
                // input: keyframe times (SCALAR f32 with min/max).
                let toff = bin.len();
                for &t in &ch.times {
                    bin.extend_from_slice(&t.to_le_bytes());
                }
                let tmin = ch.times.iter().copied().fold(f32::INFINITY, f32::min);
                let tmax = ch.times.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                buffer_views.push(serde_json::json!({ "buffer": 0, "byteOffset": toff, "byteLength": ch.times.len() * 4 }));
                let in_acc = accessors.len();
                accessors.push(serde_json::json!({ "bufferView": buffer_views.len() - 1, "componentType": 5126, "count": ch.times.len(), "type": "SCALAR", "min": [tmin], "max": [tmax] }));
                // output: values (VEC3 for T/S, VEC4 for rotation).
                let voff = bin.len();
                for &v in &ch.values {
                    bin.extend_from_slice(&v.to_le_bytes());
                }
                let comps = ch.path.components();
                buffer_views.push(serde_json::json!({ "buffer": 0, "byteOffset": voff, "byteLength": ch.values.len() * 4 }));
                let out_acc = accessors.len();
                let vtype = if comps == 4 { "VEC4" } else { "VEC3" };
                accessors.push(serde_json::json!({ "bufferView": buffer_views.len() - 1, "componentType": 5126, "count": ch.values.len() / comps, "type": vtype }));
                let samp_idx = samplers.len();
                samplers.push(serde_json::json!({ "input": in_acc, "output": out_acc, "interpolation": "LINEAR" }));
                let path_str = match ch.path {
                    engine_api::AnimationPath::Translation => "translation",
                    engine_api::AnimationPath::Rotation => "rotation",
                    engine_api::AnimationPath::Scale => "scale",
                };
                channels.push(serde_json::json!({ "sampler": samp_idx, "target": { "node": 1 + ch.target_joint, "path": path_str } }));
            }
            animations.push(serde_json::json!({ "name": clip.name, "samplers": samplers, "channels": channels }));
        }
    }
    // Prepend the mesh node (index 0), then the joint nodes.
    nodes.insert(0, mesh_node);

    let mut doc = serde_json::json!({
        "asset": { "version": "2.0", "generator": "ppf-mguso-solver" },
        "buffers": [ { "byteLength": bin.len() } ],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [ { "primitives": [ { "attributes": attributes, "indices": 1, "mode": 4 } ] } ],
        "nodes": nodes,
        "scenes": [ { "nodes": scene_nodes } ],
        "scene": 0
    });
    if !skins.is_empty() {
        doc["skins"] = serde_json::json!(skins);
    }
    if !animations.is_empty() {
        doc["animations"] = serde_json::json!(animations);
    }
    let mut json = serde_json::to_vec(&doc).context("serializing glTF JSON")?;
    // GLB chunks are 4-byte aligned: JSON padded with spaces, BIN with zeros.
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes()); // magic 'glTF' (LE)
    out.extend_from_slice(&2u32.to_le_bytes()); // version 2
    out.extend_from_slice(&(total as u32).to_le_bytes());
    // JSON chunk.
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // 'JSON'
    out.extend_from_slice(&json);
    // BIN chunk.
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes()); // 'BIN\0'
    out.extend_from_slice(&bin);
    Ok(out)
}

/// Save an [`Object`] as a binary `.glb`.
pub fn save_glb(obj: &Object, path: impl AsRef<Path>) -> Result<()> {
    let bytes = write_glb(obj)?;
    std::fs::write(path.as_ref(), bytes)
        .with_context(|| format!("writing GLB {}", path.as_ref().display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Object {
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
    fn glb_round_trip_preserves_positions_and_indices() {
        let q = quad();
        let bytes = write_glb(&q).unwrap();
        // Valid GLB magic.
        assert_eq!(&bytes[0..4], b"glTF");
        let back = parse_gltf_slice(&bytes, "q").unwrap();
        assert_eq!(back.positions.len(), 4);
        assert_eq!(back.tris.len(), 2);
        // Positions survive exactly (identity node transform).
        for (a, b) in q.positions.iter().zip(&back.positions) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-6, "pos drift {a:?} vs {b:?}");
            }
        }
        // Triangle connectivity survives.
        assert_eq!(back.tris, q.tris);
    }

    #[test]
    fn glb_round_trip_preserves_normals_and_uvs() {
        let mut q = quad();
        q.normals = vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        q.uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let bytes = write_glb(&q).unwrap();
        let back = parse_gltf_slice(&bytes, "q").unwrap();
        assert_eq!(back.normals.len(), 4, "normals must survive");
        assert_eq!(back.uvs.len(), 4, "uvs must survive");
        for (a, b) in q.normals.iter().zip(&back.normals) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-6, "normal drift");
            }
        }
        for (a, b) in q.uvs.iter().zip(&back.uvs) {
            for c in 0..2 {
                assert!((a[c] - b[c]).abs() < 1e-6, "uv drift {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn glb_is_four_byte_aligned() {
        let bytes = write_glb(&quad()).unwrap();
        assert_eq!(bytes.len() % 4, 0, "GLB must be 4-byte aligned");
        // Total length field matches actual length.
        let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(total as usize, bytes.len());
    }

    #[test]
    fn rejects_empty_mesh() {
        let mut e = quad();
        e.tris.clear();
        assert!(write_glb(&e).is_err());
    }
}
