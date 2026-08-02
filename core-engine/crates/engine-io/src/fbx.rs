// SPDX-License-Identifier: GPL-2.0-or-later
//! Autodesk FBX (`.fbx`) mesh reader — the DCC/game-engine interchange format.
//!
//! Uses `fbxcel` (MIT/Apache-2.0), a pure-Rust low-level parser for the FBX
//! **binary** format (v7400), via its `tree` DOM. We walk `Objects` → `Geometry`
//! nodes and read `Vertices` (flat f64 xyz) + `PolygonVertexIndex` (i32, where a
//! NEGATIVE index `i` is the polygon terminator with real index `!i` = `-i-1`).
//! Polygons are fan-triangulated; all geometries are concatenated (vertex-offset)
//! into one [`Object`], mirroring the OBJ/glTF/COLLADA loaders.
//!
//! Also reads per-vertex normals + UVs from `LayerElementNormal`/`LayerElementUV`
//! for the common `MappingInformationType` ∈ {ByControlPoint, ByPolygonVertex}
//! with `ReferenceInformationType` ∈ {Direct, IndexToDirect} — de-indexed to one
//! value per control point (parallel-or-empty, like the OBJ/glTF loaders).
//!
//! Rig: for a single-geometry file we also read the SKIN (`Deformer`/`Cluster` +
//! the `Connections` OO graph → `engine_api::rig::Skin`) and ANIMATION
//! (`AnimationStack`/`Layer`/`CurveNode`/`Curve` via OO+OP edges → `AnimationClip`,
//! `Lcl Translation`/`Rotation`/`Scaling` retargeted onto skin joints). FBX ASCII
//! (`.fbx` text, a separate grammar `fbxcel` does not parse) is deferred.

use anyhow::{bail, Context, Result};
use engine_api::rig::{AnimationChannel, AnimationClip, AnimationPath, Joint, Skin};
use engine_api::Object;
use fbxcel::low::v7400::AttributeValue;
use fbxcel::tree::any::AnyTree;
use fbxcel::tree::v7400::NodeHandle;
use glam::Mat4;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;

/// Load a binary `.fbx` file into an [`Object`].
pub fn load_fbx(path: impl AsRef<Path>) -> Result<Object> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading FBX {}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".into());
    parse_fbx(&bytes, name)
}

/// Parse binary FBX bytes into an [`Object`].
pub fn parse_fbx(bytes: &[u8], name: impl Into<String>) -> Result<Object> {
    // MUST be the *seekable* entry point. `fbxcel` 0.9's plain (non-seekable)
    // skip path — `pull_parser::reader::ReaderFnTable::skip_distance_plain` —
    // discards bytes from the inner stream but never calls `reader.advance()`,
    // so its cached position silently falls behind the real one. A skip only
    // happens when an attribute was left partially unread, which is exactly what
    // a zlib array does (libflate stops before the 4-byte adler32 trailer), so
    // every real DCC-exported FBX desynchronised the parser and died with "Node
    // ends with unexpected position". `skip_distance_seekable` does advance the
    // counter. `Cursor` over the input slice is Read + Seek and costs nothing.
    let tree =
        match AnyTree::from_seekable_reader(Cursor::new(bytes)).context("parsing FBX binary")? {
            AnyTree::V7400(_ver, tree, _footer) => tree,
            other => bail!("unsupported FBX version {:?}", other.fbx_version()),
        };
    let root = tree.root();
    let objects = root
        .first_child_by_name("Objects")
        .context("FBX has no `Objects` node")?;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    // Keep normals/UVs only if EVERY geometry supplies them (arrays stay parallel).
    let mut has_normals = true;
    let mut has_uvs = true;
    // (geometry id, vertex base offset, vertex count) per mesh geometry — for the
    // multi-geometry skin merge (each geometry's cluster indices are local to its
    // control points; the base offset places them in the concatenated arrays).
    let mut geoms: Vec<(i64, u32, u32)> = Vec::new();
    for geom in objects.children_by_name("Geometry") {
        let base = positions.len() as u32;
        let verts = read_f64_array(geom, "Vertices");
        let idx = read_i32_array(geom, "PolygonVertexIndex");
        let (Some(verts), Some(idx)) = (verts, idx) else {
            continue; // not a mesh geometry (e.g. NURBS) or empty
        };
        if verts.len() % 3 != 0 {
            bail!(
                "Geometry Vertices length {} not a multiple of 3",
                verts.len()
            );
        }
        let nverts = (verts.len() / 3) as u32;
        if let Some(id) = node_id(geom) {
            geoms.push((id, base, nverts));
        }
        for c in verts.chunks_exact(3) {
            positions.push([c[0] as f32, c[1] as f32, c[2] as f32]);
        }
        // Decode polygons: a negative index terminates the current polygon; its
        // real vertex index is `!raw` (bitwise NOT). Fan-triangulate each polygon.
        let mut poly: Vec<u32> = Vec::new();
        for &raw in &idx {
            let (v, end) = if raw < 0 {
                ((!raw) as u32, true)
            } else {
                (raw as u32, false)
            };
            if v >= nverts {
                bail!("FBX PolygonVertexIndex {v} out of range (nverts={nverts})");
            }
            poly.push(base + v);
            if end {
                for k in 1..poly.len().saturating_sub(1) {
                    tris.push([poly[0], poly[k], poly[k + 1]]);
                }
                poly.clear();
            }
        }
        // Per-vertex normals / UVs from LayerElement* (parallel-or-empty).
        match layer_element(geom, "LayerElementNormal", "Normals", 3, nverts, &idx) {
            Some(nrm) => normals.extend(nrm.into_iter().map(|v| [v[0], v[1], v[2]])),
            None => has_normals = false,
        }
        match layer_element(geom, "LayerElementUV", "UV", 2, nverts, &idx) {
            Some(uv) => uvs.extend(uv.into_iter().map(|v| [v[0], v[1]])),
            None => has_uvs = false,
        }
    }
    if positions.is_empty() || tris.is_empty() {
        bail!("FBX contained no triangle mesh geometry");
    }
    let normals = if has_normals && normals.len() == positions.len() {
        normals
    } else {
        Vec::new()
    };
    let uvs = if has_uvs && uvs.len() == positions.len() {
        uvs
    } else {
        Vec::new()
    };
    // Skin + animation, merged across all mesh geometries (shared-skeleton case).
    let (skin, animations) = build_fbx_skin_merged(root, &geoms, positions.len());
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
        skin,
        animations,
    })
}

/// Object id (attribute 0, an `i64`) of an FBX Objects node.
fn node_id(node: NodeHandle) -> Option<i64> {
    match node.attributes().first()? {
        AttributeValue::I64(id) => Some(*id),
        _ => None,
    }
}

/// The type string (attribute 2, e.g. "Skin"/"Cluster"/"LimbNode"/"Mesh").
fn node_type(node: NodeHandle<'_>) -> Option<&str> {
    match node.attributes().get(2)? {
        AttributeValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// The name part of attribute 1 (`"Model::Bone"` → `"Bone"`).
fn node_name(node: NodeHandle) -> String {
    match node.attributes().get(1) {
        Some(AttributeValue::String(s)) => s.rsplit("::").next().unwrap_or(s).to_string(),
        _ => String::new(),
    }
}

/// Read + merge the skin across all mesh `geoms` (`(id, base, nverts)`) into one
/// [`Skin`] parallel to the concatenated `total` positions, plus the animation clips.
/// Handles the SINGLE-geometry case (base 0) and the common MULTI-geometry
/// SHARED-SKELETON case (several mesh parts rigged to the same bones): each geometry's
/// per-control-point weights are placed at its base offset, and its joint indices are
/// consistent because the skeleton is shared. When geometries have DIFFERENT skeletons
/// (joint names differ), falls back to the first skinned geometry's skeleton (a
/// documented limitation — a cross-skeleton merge would need parent/index remapping).
fn build_fbx_skin_merged(
    root: NodeHandle<'_>,
    geoms: &[(i64, u32, u32)],
    total: usize,
) -> (Option<Skin>, Vec<AnimationClip>) {
    // Read each geometry's skin (base offset, skin, bone-id→joint-index map).
    let per_geom: Vec<(u32, Skin, HashMap<i64, u32>)> = geoms
        .iter()
        .filter_map(|&(id, base, nv)| {
            read_fbx_skin(root, id, nv as usize).map(|(s, ji)| (base, s, ji))
        })
        .collect();
    let Some((_, first_skin, first_ji)) = per_geom.first() else {
        return (None, Vec::new());
    };
    // Shared skeleton iff every geometry's joint list matches the first (same names,
    // same order → same indices).
    let shared = per_geom.iter().all(|(_, s, _)| {
        s.joints.len() == first_skin.joints.len()
            && s.joints
                .iter()
                .zip(&first_skin.joints)
                .all(|(a, b)| a.name == b.name)
    });
    let mut vertex_joints = vec![[0u32; 4]; total];
    let mut vertex_weights = vec![[0.0f32; 4]; total];
    // Which geometries' weights to scatter: all (shared) or just the first (fallback).
    let contributing: &[(u32, Skin, HashMap<i64, u32>)] = if shared {
        &per_geom
    } else {
        std::slice::from_ref(&per_geom[0])
    };
    for (base, s, _) in contributing {
        for (v, (vj, vw)) in s.vertex_joints.iter().zip(&s.vertex_weights).enumerate() {
            let idx = *base as usize + v;
            if idx < total {
                vertex_joints[idx] = *vj;
                vertex_weights[idx] = *vw;
            }
        }
    }
    let skin = Skin {
        joints: first_skin.joints.clone(),
        inverse_bind: first_skin.inverse_bind.clone(),
        vertex_joints,
        vertex_weights,
    };
    let animations = read_fbx_animations(root, first_ji);
    (Some(skin), animations)
}

/// Read the FBX skin (Deformer/Cluster + the Connections OO graph) for geometry
/// `geom_id` into an [`engine_api::rig::Skin`]. Maps: Deformer(Skin)→Geometry,
/// Deformer(Cluster)→Skin, Model(LimbNode bone)→Cluster, Model→Model (hierarchy).
/// Each Cluster gives per-control-point (Indexes,Weights); its bone gives a joint
/// with `inverse_bind = inverse(TransformLink)` and a local bind derived from the
/// parent's TransformLink. Per-vertex influences are reduced to the top 4. Returns
/// `None` if no skin is wired to this geometry, else the skin plus the bone-Model
/// id → joint-index map (used to retarget animation curves onto skin joints).
fn read_fbx_skin(
    root: NodeHandle,
    geom_id: i64,
    nverts: usize,
) -> Option<(Skin, HashMap<i64, u32>)> {
    let objects = root.first_child_by_name("Objects")?;
    let connections = root.first_child_by_name("Connections")?;

    // OO connections as directed edges (src → dst); src is the "child".
    let mut oo: Vec<(i64, i64)> = Vec::new();
    for c in connections.children_by_name("C") {
        let a = c.attributes();
        if let (
            Some(AttributeValue::String(kind)),
            Some(AttributeValue::I64(src)),
            Some(AttributeValue::I64(dst)),
        ) = (a.first(), a.get(1), a.get(2))
        {
            if kind == "OO" {
                oo.push((*src, *dst));
            }
        }
    }

    // Index Deformer + Model nodes by id.
    let mut deformers: HashMap<i64, NodeHandle> = HashMap::new();
    let mut models: HashMap<i64, NodeHandle> = HashMap::new();
    for d in objects.children_by_name("Deformer") {
        if let Some(id) = node_id(d) {
            deformers.insert(id, d);
        }
    }
    for m in objects.children_by_name("Model") {
        if let Some(id) = node_id(m) {
            models.insert(id, m);
        }
    }

    // The Skin deformer wired to this geometry (src=skin, dst=geom).
    let skin_id = oo.iter().find_map(|&(src, dst)| {
        if dst == geom_id && deformers.get(&src).copied().and_then(node_type) == Some("Skin") {
            Some(src)
        } else {
            None
        }
    })?;

    // Clusters of that skin (src=cluster, dst=skin).
    let cluster_ids: Vec<i64> = oo
        .iter()
        .filter(|&&(_, dst)| dst == skin_id)
        .filter_map(|&(src, _)| {
            (deformers.get(&src).copied().and_then(node_type) == Some("Cluster")).then_some(src)
        })
        .collect();
    if cluster_ids.is_empty() {
        return None;
    }

    // For each cluster, the bone Model that drives it (src=bone, dst=cluster).
    // Assign each distinct bone a joint index (order of first appearance).
    let mut joint_index: HashMap<i64, u32> = HashMap::new();
    let mut bone_ids: Vec<i64> = Vec::new();
    let mut cluster_bone: Vec<(i64, i64)> = Vec::new(); // (cluster_id, bone_id)
    for &cid in &cluster_ids {
        let bone = oo.iter().find_map(|&(src, dst)| {
            (dst == cid && models.get(&src).copied().and_then(node_type) == Some("LimbNode"))
                .then_some(src)
        });
        if let Some(bid) = bone {
            joint_index.entry(bid).or_insert_with(|| {
                bone_ids.push(bid);
                (bone_ids.len() - 1) as u32
            });
            cluster_bone.push((cid, bid));
        }
    }
    if bone_ids.is_empty() {
        return None;
    }

    // Per-bone TransformLink (world bind) → inverse_bind; joints + hierarchy.
    let mut inverse_bind: Vec<[[f32; 4]; 4]> = Vec::new();
    let mut link_world: Vec<Mat4> = Vec::new(); // TransformLink per joint
    for &bid in &bone_ids {
        // TransformLink lives on the bone's cluster; find it.
        let cid = cluster_bone
            .iter()
            .find_map(|&(c, b)| (b == bid).then_some(c));
        let link = cid
            .and_then(|c| deformers.get(&c).copied())
            .and_then(|cl| read_f64_array(cl, "TransformLink"))
            .filter(|v| v.len() == 16)
            .map(|v| {
                let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                Mat4::from_cols_slice(&f)
            })
            .unwrap_or(Mat4::IDENTITY);
        link_world.push(link);
        inverse_bind.push(link.inverse().to_cols_array_2d());
    }

    // Joints: name + parent (from Model→Model OO among the bones) + local bind
    // (parent_world⁻¹ · world; = world for a root).
    let mut joints: Vec<Joint> = Vec::new();
    for (j, &bid) in bone_ids.iter().enumerate() {
        let name = models.get(&bid).map(|&m| node_name(m)).unwrap_or_default();
        // Parent bone: this bone (src) connected to another LimbNode (dst).
        let parent = oo.iter().find_map(|&(src, dst)| {
            (src == bid && dst != bid)
                .then(|| joint_index.get(&dst).copied())
                .flatten()
        });
        let local = match parent {
            Some(p) => (link_world[p as usize].inverse() * link_world[j]).to_cols_array_2d(),
            None => link_world[j].to_cols_array_2d(),
        };
        joints.push(Joint {
            name,
            parent,
            local,
        });
    }

    // Per-vertex top-4 influences from each cluster's (Indexes, Weights).
    let mut infl: Vec<Vec<(u32, f32)>> = vec![Vec::new(); nverts];
    for &(cid, bid) in &cluster_bone {
        let j = joint_index[&bid];
        let cl = deformers[&cid];
        let (idx, wts) = match (read_i32_array(cl, "Indexes"), read_f64_array(cl, "Weights")) {
            (Some(i), Some(w)) if i.len() == w.len() => (i, w),
            _ => continue,
        };
        for (&ci, &w) in idx.iter().zip(&wts) {
            if ci >= 0 && (ci as usize) < nverts && w != 0.0 {
                infl[ci as usize].push((j, w as f32));
            }
        }
    }
    let mut vertex_joints = vec![[0u32; 4]; nverts];
    let mut vertex_weights = vec![[0.0f32; 4]; nverts];
    for (v, list) in infl.iter_mut().enumerate() {
        // Keep the 4 largest weights.
        list.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (s, &(j, w)) in list.iter().take(4).enumerate() {
            vertex_joints[v][s] = j;
            vertex_weights[v][s] = w;
        }
    }

    Some((
        Skin {
            joints,
            inverse_bind,
            vertex_joints,
            vertex_weights,
        },
        joint_index,
    ))
}

/// FBX time unit: `KTime` counts 1/46186158000 s per tick (the FBX SDK constant
/// `FBXSDK_TC_SECOND`; every exporter writes `KeyTime` in these units). Verified
/// only against fixtures we author with this same constant — the value itself is
/// the documented FBX SDK constant, not measured here.
const FBX_TIME_UNIT: f64 = 46_186_158_000.0;

/// Read FBX animation (`AnimationStack`→`AnimationLayer`→`AnimationCurveNode`→
/// `AnimationCurve`) into [`AnimationClip`]s, retargeting each curve node's Model
/// onto the skin joint via `joint_index`. Handles the `Lcl Translation`/`Lcl
/// Scaling` (3-float, direct) and `Lcl Rotation` (Euler XYZ degrees → quaternion)
/// properties. A curve node's X/Y/Z sub-curves may key at independent times: the
/// channel samples at the UNION of their key times, each axis linearly interpolated
/// (`sample_curve`); a missing sub-curve defaults to 0 (T/R) or 1 (S). Curve nodes
/// whose target Model is not a skin joint are skipped. Non-XYZ Euler orders
/// (`RotationOrder`) are a documented follow-up.
fn read_fbx_animations(
    root: NodeHandle<'_>,
    joint_index: &HashMap<i64, u32>,
) -> Vec<AnimationClip> {
    let (Some(objects), Some(connections)) = (
        root.first_child_by_name("Objects"),
        root.first_child_by_name("Connections"),
    ) else {
        return Vec::new();
    };

    // Connections: OO edges (src→dst) and OP edges (src→dst, property string).
    let mut oo: Vec<(i64, i64)> = Vec::new();
    let mut op: Vec<(i64, i64, String)> = Vec::new();
    for c in connections.children_by_name("C") {
        let a = c.attributes();
        match (a.first(), a.get(1), a.get(2), a.get(3)) {
            (
                Some(AttributeValue::String(kind)),
                Some(AttributeValue::I64(src)),
                Some(AttributeValue::I64(dst)),
                prop,
            ) if kind == "OP" => {
                let p = match prop {
                    Some(AttributeValue::String(s)) => s.clone(),
                    _ => String::new(),
                };
                op.push((*src, *dst, p));
            }
            (
                Some(AttributeValue::String(kind)),
                Some(AttributeValue::I64(src)),
                Some(AttributeValue::I64(dst)),
                _,
            ) if kind == "OO" => oo.push((*src, *dst)),
            _ => {}
        }
    }

    // Index animation objects by id.
    let mut curves: HashMap<i64, NodeHandle> = HashMap::new();
    for cv in objects.children_by_name("AnimationCurve") {
        if let Some(id) = node_id(cv) {
            curves.insert(id, cv);
        }
    }

    let mut clips = Vec::new();
    for stack in objects.children_by_name("AnimationStack") {
        let Some(stack_id) = node_id(stack) else {
            continue;
        };
        let name = node_name(stack);
        // Layers of this stack, then curve nodes of those layers.
        let layer_ids: Vec<i64> = oo
            .iter()
            .filter(|&&(_, dst)| dst == stack_id)
            .map(|&(src, _)| src)
            .collect();
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for &lid in &layer_ids {
            let node_ids: Vec<i64> = oo
                .iter()
                .filter(|&&(_, dst)| dst == lid)
                .map(|&(src, _)| src)
                .collect();
            for &nid in &node_ids {
                // Curve node → Model (OP) with the driven property.
                let Some((model_id, path)) = op.iter().find_map(|(src, dst, p)| {
                    (*src == nid)
                        .then(|| anim_path(p).map(|path| (*dst, path)))
                        .flatten()
                }) else {
                    continue;
                };
                let Some(&target_joint) = joint_index.get(&model_id) else {
                    continue; // curve targets a non-skin Model
                };
                // The X/Y/Z sub-curves feeding this node (OP prop "d|X"/"d|Y"/"d|Z").
                let mut axes: [Option<(Vec<i64>, Vec<f32>)>; 3] = [None, None, None];
                for (src, dst, p) in &op {
                    if *dst != nid {
                        continue;
                    }
                    let axis = match p.as_str() {
                        "d|X" => 0,
                        "d|Y" => 1,
                        "d|Z" => 2,
                        _ => continue,
                    };
                    if let Some(cv) = curves.get(src) {
                        let t = read_i64_array(*cv, "KeyTime").unwrap_or_default();
                        let v = read_f32_array(*cv, "KeyValueFloat").unwrap_or_default();
                        if t.len() == v.len() && !t.is_empty() {
                            axes[axis] = Some((t, v));
                        }
                    }
                }
                // Union of all present axes' key times (each axis may key at its own
                // times; e.g. only X moves at t=1 while Z rotates at t=0.5). Each
                // axis is then linearly sampled at every union time.
                let mut ticks: Vec<i64> = axes
                    .iter()
                    .flatten()
                    .flat_map(|(t, _)| t.iter().copied())
                    .collect();
                ticks.sort_unstable();
                ticks.dedup();
                if ticks.is_empty() {
                    continue;
                }
                let times: Vec<f32> = ticks
                    .iter()
                    .map(|&t| (t as f64 / FBX_TIME_UNIT) as f32)
                    .collect();
                // Sample each axis at a tick (default 0 for T/R, 1 for S when absent).
                let default = if path == AnimationPath::Scale {
                    1.0
                } else {
                    0.0
                };
                let sample = |axis: usize, q: i64| -> f32 {
                    match &axes[axis] {
                        Some((t, v)) => sample_curve(t, v, q),
                        None => default,
                    }
                };
                let mut values = Vec::with_capacity(ticks.len() * path.components());
                for &q in &ticks {
                    let (x, y, z) = (sample(0, q), sample(1, q), sample(2, q));
                    match path {
                        AnimationPath::Rotation => {
                            // FBX Lcl Rotation = Euler degrees, default order XYZ.
                            let quat = glam::Quat::from_euler(
                                glam::EulerRot::XYZ,
                                x.to_radians(),
                                y.to_radians(),
                                z.to_radians(),
                            );
                            values.extend_from_slice(&[quat.x, quat.y, quat.z, quat.w]);
                        }
                        _ => values.extend_from_slice(&[x, y, z]),
                    }
                }
                if let Some(&last) = times.last() {
                    duration = duration.max(last);
                }
                channels.push(AnimationChannel {
                    target_joint,
                    path,
                    times,
                    values,
                });
            }
        }
        if !channels.is_empty() {
            clips.push(AnimationClip {
                name,
                duration,
                channels,
            });
        }
    }
    clips
}

/// Linearly sample a piecewise-linear FBX curve (ascending `ticks` + parallel
/// `vals`) at query tick `q`: clamped to the endpoints outside the range, lerped
/// within the bracketing segment. Empty/mismatched curves sample to 0.
fn sample_curve(ticks: &[i64], vals: &[f32], q: i64) -> f32 {
    let n = ticks.len().min(vals.len());
    if n == 0 {
        return 0.0;
    }
    if q <= ticks[0] {
        return vals[0];
    }
    if q >= ticks[n - 1] {
        return vals[n - 1];
    }
    // Bracket [i, i+1] with ticks[i] <= q < ticks[i+1].
    let i = match ticks[..n].binary_search(&q) {
        Ok(hit) => return vals[hit],
        Err(ins) => ins - 1,
    };
    let (t0, t1) = (ticks[i] as f64, ticks[i + 1] as f64);
    let f = if t1 > t0 {
        (q as f64 - t0) / (t1 - t0)
    } else {
        0.0
    };
    (vals[i] as f64 + (vals[i + 1] - vals[i]) as f64 * f) as f32
}

/// Map an FBX Model property name to an [`AnimationPath`].
fn anim_path(prop: &str) -> Option<AnimationPath> {
    match prop {
        "Lcl Translation" => Some(AnimationPath::Translation),
        "Lcl Rotation" => Some(AnimationPath::Rotation),
        "Lcl Scaling" => Some(AnimationPath::Scale),
        _ => None,
    }
}

/// Read a `LayerElement*` node (e.g. `LayerElementNormal`/`Normals`, comp=3) and
/// map it to ONE value per control point (`nverts` × `comp`), returning `None`
/// when absent or in an unsupported layout. Handles `MappingInformationType`
/// ∈ {ByVertex/ByControlPoint, ByPolygonVertex} with `ReferenceInformationType`
/// ∈ {Direct, IndexToDirect}. For `ByPolygonVertex` (a value per polygon-corner,
/// parallel to `poly_idx`) the corner values are de-indexed to the control point
/// they reference — first corner wins (per-corner seams are collapsed; documented).
fn layer_element(
    geom: NodeHandle,
    layer: &str,
    data_name: &str,
    comp: usize,
    nverts: u32,
    poly_idx: &[i32],
) -> Option<Vec<Vec<f32>>> {
    let node = geom.first_child_by_name(layer)?;
    let mapping = read_string(node, "MappingInformationType")?;
    let reference = read_string(node, "ReferenceInformationType")?;
    let data: Vec<f32> = read_f64_array(node, data_name)?
        .iter()
        .map(|&v| v as f32)
        .collect();
    if !data.len().is_multiple_of(comp) {
        return None;
    }
    // Resolve reference: a function from "slot" → the comp-vector at that slot.
    let index: Option<Vec<i32>> = if reference.contains("IndexToDirect") {
        // Index array is named e.g. NormalsIndex / UVIndex.
        Some(read_i32_array(node, &format!("{data_name}Index"))?)
    } else if reference == "Direct" {
        None
    } else {
        return None; // unknown reference type
    };
    let n_direct = data.len() / comp;
    let fetch = |slot: usize| -> Option<Vec<f32>> {
        let di = match &index {
            Some(ix) => *ix.get(slot)? as usize,
            None => slot,
        };
        if di >= n_direct {
            return None;
        }
        Some(data[di * comp..di * comp + comp].to_vec())
    };

    let mut out = vec![Vec::<f32>::new(); nverts as usize];
    if mapping == "ByVertice" || mapping == "ByVertex" || mapping == "ByControlPoint" {
        // One slot per control point (parallel to Vertices).
        for (v, slot) in out.iter_mut().enumerate() {
            *slot = fetch(v)?;
        }
    } else if mapping == "ByPolygonVertex" {
        // One slot per polygon corner (parallel to PolygonVertexIndex); de-index
        // to the referenced control point, first corner wins.
        for (corner, &raw) in poly_idx.iter().enumerate() {
            let v = (if raw < 0 { !raw } else { raw }) as usize;
            if v < out.len() && out[v].is_empty() {
                out[v] = fetch(corner)?;
            }
        }
    } else {
        return None; // ByPolygon / AllSame not supported here
    }
    // Every control point must have received a value.
    if out.iter().any(|v| v.len() != comp) {
        return None;
    }
    Some(out)
}

/// First child named `name`, first attribute as a UTF-8 string.
fn read_string(node: NodeHandle, name: &str) -> Option<String> {
    match node.first_child_by_name(name)?.attributes().first()? {
        AttributeValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// First child named `name`, first attribute as an f64 array (clone).
fn read_f64_array(node: NodeHandle, name: &str) -> Option<Vec<f64>> {
    match node.first_child_by_name(name)?.attributes().first()? {
        AttributeValue::ArrF64(v) => Some(v.clone()),
        _ => None,
    }
}

/// First child named `name`, first attribute as an i32 array (clone).
fn read_i32_array(node: NodeHandle, name: &str) -> Option<Vec<i32>> {
    match node.first_child_by_name(name)?.attributes().first()? {
        AttributeValue::ArrI32(v) => Some(v.clone()),
        _ => None,
    }
}

/// First child named `name`, first attribute as an i64 array (clone).
fn read_i64_array(node: NodeHandle, name: &str) -> Option<Vec<i64>> {
    match node.first_child_by_name(name)?.attributes().first()? {
        AttributeValue::ArrI64(v) => Some(v.clone()),
        _ => None,
    }
}

/// First child named `name`, first attribute as an f32 array (clone).
fn read_f32_array(node: NodeHandle, name: &str) -> Option<Vec<f32>> {
    match node.first_child_by_name(name)?.attributes().first()? {
        AttributeValue::ArrF32(v) => Some(v.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::sample_curve;

    #[test]
    fn sample_curve_clamps_and_lerps() {
        let ticks = [0i64, 10, 30];
        let vals = [0.0f32, 10.0, 30.0];
        // Clamp below/above the range.
        assert_eq!(sample_curve(&ticks, &vals, -5), 0.0);
        assert_eq!(sample_curve(&ticks, &vals, 40), 30.0);
        // Exact keys.
        assert_eq!(sample_curve(&ticks, &vals, 10), 10.0);
        // Midpoint of [0,10] → 5; midpoint of [10,30] (t=20) → 20.
        assert!((sample_curve(&ticks, &vals, 5) - 5.0).abs() < 1e-6);
        assert!((sample_curve(&ticks, &vals, 20) - 20.0).abs() < 1e-6);
        // Non-uniform value slope: [10,30] spans 20 ticks / 20 value → t=25 → 25.
        assert!((sample_curve(&ticks, &vals, 25) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn sample_curve_empty_is_zero() {
        assert_eq!(sample_curve(&[], &[], 5), 0.0);
    }
}
