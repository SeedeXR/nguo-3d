// SPDX-License-Identifier: GPL-2.0-or-later
//! COLLADA (`.dae`) mesh reader — the DCC/interchange XML format.
//!
//! Reads the first `<mesh>`: resolves its `<vertices>` POSITION `<source>` to a
//! `<float_array>` (positions), then reads `<triangles>` or `<polylist>`
//! primitives, pulling the VERTEX-semantic index out of each interleaved index
//! tuple in `<p>` (honouring per-input `offset`), and fan-triangulates polygons.
//!
//! Uses `roxmltree` (MIT/Apache-2.0) for read-only DOM navigation. Tags are
//! matched by *local* name (`tag_name().name()`) because COLLADA puts everything
//! in a default namespace, so namespaced matching would miss.
//!
//! Face-varying NORMAL/TEXCOORD inputs (per-corner, own `offset` in the `<p>`
//! tuple) are de-indexed to one value per control point (first corner touching a
//! vertex wins), parallel to `positions` — the same parallel-or-empty contract as
//! the FBX/glTF loaders (kept only if every control point is covered). UVs are the
//! raw source ST (no V flip).
//!
//! Rig: a `<library_controllers><skin>` bound to the mesh is read into
//! `engine_api::rig::Skin` — JOINT names + INV_BIND_MATRIX (row-major → glam),
//! `<vertex_weights>` top-4 per control point, and parent/local from the
//! `<library_visual_scenes>` JOINT-node hierarchy.
//!
//! Animation: `<library_animations>` matrix tracks (a `<sampler>` with a 4×4
//! TRANSFORM OUTPUT per key) are decomposed to Translation/Rotation/Scale channels
//! on the target joint → `engine_api::rig::AnimationClip`.
//!
//! A `<vertices>`-level (shared, per-control-point) NORMAL input is also read and
//! seeds the per-vertex normals before any face-varying pass.
//!
//! Scope/limits (v1): first `<mesh>` only; no transforms/`<up_axis>` remap;
//! `<polygons>`/`<trifans>`/`<tristrips>` deferred; per-corner seams (a control
//! point with differing normals collapses to the first) are a follow-up.
//! TRS-decomposed joint nodes are a later upgrade. (Matrix, vector, AND
//! single-component `Node/location.X` animation OUTPUT are all handled.)

use anyhow::{bail, Context, Result};
use engine_api::rig::{AnimationChannel, AnimationClip, AnimationPath, Joint, Skin};
use engine_api::Object;
use glam::Mat4;
use roxmltree::{Document, Node};
use std::path::Path;

/// Load a `.dae` (COLLADA) file into an [`Object`].
pub fn load_collada(path: impl AsRef<Path>) -> Result<Object> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading COLLADA {}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".into());
    parse_collada(&text, name)
}

/// Parse COLLADA XML text into an [`Object`] (first `<mesh>`).
pub fn parse_collada(text: &str, name: impl Into<String>) -> Result<Object> {
    let doc = Document::parse(text).context("parsing COLLADA XML")?;
    let mesh = descendant(doc.root(), "mesh").context("no <mesh> in COLLADA document")?;
    // The enclosing <geometry id> — used to match a <controller><skin source="#id">.
    let geom_id = mesh.parent().and_then(|g| g.attribute("id"));

    // Positions: <vertices> → its POSITION <input source="#id"> → <source id> →
    // <float_array>, chunked by the accessor stride (default 3).
    let vertices = child_by_tag(mesh, "vertices").context("mesh has no <vertices>")?;
    let pos_input =
        input_with_semantic(vertices, "POSITION").context("<vertices> has no POSITION input")?;
    let pos_source_id = pos_input
        .attribute("source")
        .map(strip_hash)
        .context("POSITION input has no source")?;
    let source = element_by_id(&doc, pos_source_id)
        .with_context(|| format!("POSITION source #{pos_source_id} not found"))?;
    let farray = descendant(source, "float_array").context("position source has no float_array")?;
    let floats: Vec<f64> = parse_numbers(farray.text().unwrap_or(""));
    let stride = accessor_stride(source).unwrap_or(3).max(1);
    if floats.len() < stride {
        bail!("position float_array too short");
    }
    let positions: Vec<[f32; 3]> = floats
        .chunks(stride)
        .filter(|c| c.len() >= 3)
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect();
    if positions.is_empty() {
        bail!("COLLADA mesh has no positions");
    }
    let nverts = positions.len() as u32;

    // Primitives: <triangles> (flat, every 3 tuples = a tri) or <polylist>
    // (<vcount> per-face sizes, fan-triangulated). Face-varying NORMAL/TEXCOORD
    // are de-indexed to one value per control point (first corner wins), parallel
    // to `positions` — matching the FBX/glTF loaders' parallel-or-empty contract.
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut norm_by_v: Vec<Option<[f32; 3]>> = vec![None; nverts as usize];
    let mut uv_by_v: Vec<Option<[f32; 2]>> = vec![None; nverts as usize];
    // Vertices-level (shared) NORMAL: some exporters put a per-control-point normal
    // input on `<vertices>` (indexed by the vertex index, not per-corner). Seed
    // `norm_by_v` from it; any face-varying NORMAL below only fills gaps.
    if let Some((floats, stride)) =
        input_with_semantic(vertices, "NORMAL").and_then(|nin| resolve_input_source(&doc, nin, 3))
    {
        for (v, slot) in norm_by_v.iter_mut().enumerate() {
            let base = v * stride;
            if base + 3 <= floats.len() {
                *slot = Some([
                    floats[base] as f32,
                    floats[base + 1] as f32,
                    floats[base + 2] as f32,
                ]);
            }
        }
    }
    for prim in mesh.children().filter(|n| n.is_element()) {
        let kind = prim.tag_name().name();
        if kind != "triangles" && kind != "polylist" {
            continue;
        }
        // Index tuple layout: stride = max(input offset)+1; VERTEX picks its slot.
        let inputs: Vec<Node> = prim
            .children()
            .filter(|n| n.tag_name().name() == "input")
            .collect();
        let vertex_offset = inputs
            .iter()
            .find(|n| n.attribute("semantic") == Some("VERTEX"))
            .and_then(|n| n.attribute("offset"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let index_stride = inputs
            .iter()
            .filter_map(|n| n.attribute("offset"))
            .filter_map(|s| s.parse::<usize>().ok())
            .max()
            .unwrap_or(0)
            + 1;
        let p_node = match child_by_tag(prim, "p") {
            Some(p) => p,
            None => continue,
        };
        let p: Vec<u32> = parse_numbers(p_node.text().unwrap_or(""))
            .into_iter()
            .map(|v| v as u32)
            .collect();
        if index_stride == 0 || p.len() < index_stride {
            continue;
        }
        // Per-corner VERTEX indices, in order.
        let ntuples = p.len() / index_stride;
        let vidx: Vec<u32> = (0..ntuples)
            .map(|i| p[i * index_stride + vertex_offset])
            .collect();

        // De-index face-varying NORMAL (3-comp) / TEXCOORD (first 2 comps) to the
        // control point (first corner to touch a vertex wins).
        deindex_corner_attr(
            &inputs,
            "NORMAL",
            &doc,
            &p,
            &vidx,
            index_stride,
            ntuples,
            3,
            &mut norm_by_v,
            |c| [c[0] as f32, c[1] as f32, c[2] as f32],
        );
        deindex_corner_attr(
            &inputs,
            "TEXCOORD",
            &doc,
            &p,
            &vidx,
            index_stride,
            ntuples,
            2,
            &mut uv_by_v,
            |c| [c[0] as f32, c[1] as f32],
        );

        if kind == "triangles" {
            for t in vidx.chunks_exact(3) {
                push_tri(&mut tris, [t[0], t[1], t[2]], nverts, "triangles")?;
            }
        } else {
            // polylist: <vcount> gives face sizes; fan each face.
            let vcounts: Vec<usize> = child_by_tag(prim, "vcount")
                .and_then(|n| n.text())
                .map(parse_numbers)
                .unwrap_or_default()
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let mut cursor = 0usize;
            for (fi, &c) in vcounts.iter().enumerate() {
                if c < 3 {
                    bail!("polylist face {fi} has {c} verts (need ≥3)");
                }
                if cursor + c > vidx.len() {
                    bail!("polylist <p> too short for <vcount>");
                }
                let face = &vidx[cursor..cursor + c];
                for k in 1..c - 1 {
                    push_tri(
                        &mut tris,
                        [face[0], face[k], face[k + 1]],
                        nverts,
                        "polylist",
                    )?;
                }
                cursor += c;
            }
        }
    }
    if tris.is_empty() {
        bail!("COLLADA mesh produced no triangles (need <triangles> or <polylist>)");
    }

    // Parallel-or-empty: keep normals/UVs only if EVERY control point got one.
    let normals: Vec<[f32; 3]> = if norm_by_v.iter().all(Option::is_some) {
        norm_by_v.into_iter().flatten().collect()
    } else {
        Vec::new()
    };
    let uvs: Vec<[f32; 2]> = if uv_by_v.iter().all(Option::is_some) {
        uv_by_v.into_iter().flatten().collect()
    } else {
        Vec::new()
    };

    let skin = read_collada_skin(&doc, geom_id, positions.len());
    // Animation channels target skin joints, so read them only when a skin exists.
    let animations = match &skin {
        Some(s) => {
            let names: Vec<String> = s.joints.iter().map(|j| j.name.clone()).collect();
            read_collada_animations(&doc, &names)
        }
        None => Vec::new(),
    };

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

/// Read `<library_animations>` into [`AnimationClip`]s. Each `<animation>`'s
/// `<channel target="Node/component">` samples a `<sampler>` whose INPUT is TIME
/// (seconds) and OUTPUT is either a 4×4 TRANSFORM matrix per key (row-major),
/// decomposed via `Mat4::to_scale_rotation_translation` into Translation / Rotation
/// (quaternion) / Scale channels, OR a 3-float `<translate>`/`<scale>` vector per
/// key (used directly, classified by the targeted component name). Retargeted onto
/// the joint matched by node `sid`/`id`; all `<animation>`s merge into one clip.
/// Also handles axis-angle `<rotate>` samplers — both the 4-float form (axis + angle in
/// DEGREES per key) and the single-component `Node/rotateZ.ANGLE` form, whose static axis is
/// resolved from the node's own `<rotate sid="rotateZ">` element — and single-component
/// `Node/location.X` channels.
fn read_collada_animations(doc: &Document, joint_names: &[String]) -> Vec<AnimationClip> {
    use std::collections::{BTreeMap, HashMap};
    let name_index: HashMap<&str, u32> = joint_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i as u32))
        .collect();

    let mut channels: Vec<AnimationChannel> = Vec::new();
    let mut duration = 0.0f32;
    let mut clip_name = String::new();
    // Single-component channels (`Node/location.X`) grouped by (joint, base transform
    // name) → per-component (times, values). BTreeMap keeps the output deterministic.
    #[allow(clippy::type_complexity)]
    let mut components: BTreeMap<(u32, &str), [Option<(Vec<f32>, Vec<f32>)>; 3]> = BTreeMap::new();

    for anim in doc
        .descendants()
        .filter(|n| n.tag_name().name() == "animation")
    {
        for channel in anim.children().filter(|n| n.tag_name().name() == "channel") {
            let Some(target) = channel.attribute("target") else {
                continue;
            };
            // target = "NodeToken/property"; resolve the node token → joint index.
            let token = target.split('/').next().unwrap_or(target);
            let Some(ji) = resolve_target_joint(doc, token, &name_index) else {
                continue;
            };
            // Sampler → INPUT (TIME) + OUTPUT (4×4 matrices).
            let Some(sampler) = channel
                .attribute("source")
                .map(strip_hash)
                .and_then(|id| element_by_id(doc, id))
            else {
                continue;
            };
            let times: Vec<f32> = input_source(doc, sampler, "INPUT")
                .and_then(|s| descendant(s, "float_array"))
                .map(|fa| parse_numbers(fa.text().unwrap_or("")))
                .unwrap_or_default()
                .into_iter()
                .map(|t| t as f32)
                .collect();
            let out_el = input_source(doc, sampler, "OUTPUT");
            let out: Vec<f64> = out_el
                .and_then(|s| descendant(s, "float_array"))
                .map(|fa| parse_numbers(fa.text().unwrap_or("")))
                .unwrap_or_default();
            let out_stride = out_el.and_then(accessor_stride).unwrap_or(16);
            let n = times.len();
            if n == 0 {
                continue;
            }
            let last = *times.last().unwrap();
            // The transform component the channel targets ("Root/translate" → "translate").
            let prop = target.split_once('/').map(|(_, p)| p).unwrap_or("");
            if out_stride == 16 && out.len() >= n * 16 {
                // Full 4×4 TRANSFORM per key → decompose into T/R/S channels.
                let mut t_vals = Vec::with_capacity(n * 3);
                let mut r_vals = Vec::with_capacity(n * 4);
                let mut s_vals = Vec::with_capacity(n * 3);
                for k in 0..n {
                    let m = mat4_rowmajor(&out[k * 16..k * 16 + 16]);
                    let (scale, rot, trans) = m.to_scale_rotation_translation();
                    t_vals.extend_from_slice(&[trans.x, trans.y, trans.z]);
                    r_vals.extend_from_slice(&[rot.x, rot.y, rot.z, rot.w]);
                    s_vals.extend_from_slice(&[scale.x, scale.y, scale.z]);
                }
                duration = duration.max(last);
                if clip_name.is_empty() {
                    clip_name = anim.attribute("id").unwrap_or("anim").to_string();
                }
                channels.push(AnimationChannel {
                    target_joint: ji,
                    path: AnimationPath::Translation,
                    times: times.clone(),
                    values: t_vals,
                });
                channels.push(AnimationChannel {
                    target_joint: ji,
                    path: AnimationPath::Rotation,
                    times: times.clone(),
                    values: r_vals,
                });
                channels.push(AnimationChannel {
                    target_joint: ji,
                    path: AnimationPath::Scale,
                    times,
                    values: s_vals,
                });
            } else if out_stride == 3 && out.len() >= n * 3 {
                // Vector OUTPUT: a `<translate>`/`<scale>` transform channel (3 floats
                // per key, used directly). Classify by the targeted component name.
                let base = prop.split(['.', '(']).next().unwrap_or(prop);
                let path = match base {
                    "translate" | "location" => Some(AnimationPath::Translation),
                    "scale" => Some(AnimationPath::Scale),
                    _ => None,
                };
                if let Some(path) = path {
                    let vals: Vec<f32> = out[..n * 3].iter().map(|&x| x as f32).collect();
                    duration = duration.max(last);
                    if clip_name.is_empty() {
                        clip_name = anim.attribute("id").unwrap_or("anim").to_string();
                    }
                    channels.push(AnimationChannel {
                        target_joint: ji,
                        path,
                        times,
                        values: vals,
                    });
                }
            } else if out_stride == 4 && out.len() >= n * 4 {
                // Axis-angle OUTPUT: a `<rotate>` transform channel. COLLADA 1.4/1.5 spec:
                // `<rotate>` holds four floats, the rotation AXIS followed by the angle in
                // DEGREES, right-handed. Convert each key to the quaternion the rig format
                // wants (`AnimationPath::Rotation` is `[x, y, z, w]`).
                let base = prop.split(['.', '(']).next().unwrap_or(prop);
                if base.starts_with("rotate") || base == "rotation" {
                    let mut vals = Vec::with_capacity(n * 4);
                    for k in 0..n {
                        let q = axis_angle_quat(
                            out[k * 4] as f32,
                            out[k * 4 + 1] as f32,
                            out[k * 4 + 2] as f32,
                            out[k * 4 + 3] as f32,
                        );
                        vals.extend_from_slice(&[q.x, q.y, q.z, q.w]);
                    }
                    duration = duration.max(last);
                    if clip_name.is_empty() {
                        clip_name = anim.attribute("id").unwrap_or("anim").to_string();
                    }
                    channels.push(AnimationChannel {
                        target_joint: ji,
                        path: AnimationPath::Rotation,
                        times,
                        values: vals,
                    });
                }
            } else if out.len() >= n && is_rotate_angle_channel(prop) {
                // Single-component ANGLE channel (`Node/rotateZ.ANGLE`): one float per key,
                // and the AXIS is static — it lives on the node's own
                // `<rotate sid="rotateZ">` element. Without resolving that axis the channel
                // is unusable, which is why this case was a follow-up rather than folded in
                // with the translate/scale component channels.
                let base = prop.split(['.', '(']).next().unwrap_or(prop);
                if let Some(axis) = resolve_rotate_axis(doc, token, base) {
                    let mut vals = Vec::with_capacity(n * 4);
                    for &angle in out.iter().take(n) {
                        let q = axis_angle_quat(axis[0], axis[1], axis[2], angle as f32);
                        vals.extend_from_slice(&[q.x, q.y, q.z, q.w]);
                    }
                    duration = duration.max(last);
                    if clip_name.is_empty() {
                        clip_name = anim.attribute("id").unwrap_or("anim").to_string();
                    }
                    channels.push(AnimationChannel {
                        target_joint: ji,
                        path: AnimationPath::Rotation,
                        times,
                        values: vals,
                    });
                }
            } else if out.len() >= n {
                // Single-component channel (`Node/location.X`, `Node/scale(1)`): one
                // float per key targeting one axis. Collect for post-loop merge — the
                // X/Y/Z sub-channels combine into one T/S channel at the union of times.
                if let Some((base, comp)) = collada_component(prop) {
                    if matches!(base, "translate" | "location" | "scale") {
                        let vals: Vec<f32> = out[..n].iter().map(|&x| x as f32).collect();
                        components.entry((ji, base)).or_default()[comp] = Some((times, vals));
                        duration = duration.max(last);
                        if clip_name.is_empty() {
                            clip_name = anim.attribute("id").unwrap_or("anim").to_string();
                        }
                    }
                }
            }
        }
    }
    // Assemble grouped single-component channels: union each group's key times and
    // sample every axis (missing axis → 0) → one Translation/Scale channel.
    for ((ji, base), comps) in components {
        let path = match base {
            "translate" | "location" => AnimationPath::Translation,
            "scale" => AnimationPath::Scale,
            _ => continue,
        };
        let mut ts: Vec<f32> = comps
            .iter()
            .flatten()
            .flat_map(|(t, _)| t.iter().copied())
            .collect();
        ts.sort_by(|a, b| a.total_cmp(b));
        ts.dedup();
        if ts.is_empty() {
            continue;
        }
        let mut values = Vec::with_capacity(ts.len() * 3);
        for &q in &ts {
            for c in &comps {
                values.push(match c {
                    Some((t, v)) => collada_sample(t, v, q),
                    None => 0.0,
                });
            }
        }
        channels.push(AnimationChannel {
            target_joint: ji,
            path,
            times: ts,
            values,
        });
    }
    if channels.is_empty() {
        return Vec::new();
    }
    vec![AnimationClip {
        name: clip_name,
        duration,
        channels,
    }]
}

/// Resolve an animation channel target node token to a joint index: the token
/// directly (sid==joint name), else the token as a `<node>` id/sid whose `sid`
/// matches a joint name.
/// Parse a single-component channel property into (base transform, axis index):
/// `"location.X"` → `("location", 0)`, `"scale(1)"` → `("scale", 1)`. Handles
/// `.X`/`.Y`/`.Z` (and `.S`/`.T` for 2D) and `(0)`/`(1)`/`(2)` COLLADA suffixes.
/// Quaternion from a COLLADA `<rotate>` axis and an angle in DEGREES.
///
/// The COLLADA 1.4/1.5 spec fixes this: `<rotate>` is `axis.x axis.y axis.z angle`, angle in
/// degrees, right-handed. A zero-length axis is not a rotation — return identity rather than
/// normalising a zero vector into NaN.
fn axis_angle_quat(ax: f32, ay: f32, az: f32, angle_deg: f32) -> glam::Quat {
    let axis = glam::Vec3::new(ax, ay, az);
    if axis.length_squared() <= 1.0e-20 {
        return glam::Quat::IDENTITY;
    }
    glam::Quat::from_axis_angle(axis.normalize(), angle_deg.to_radians())
}

/// Is `prop` the ANGLE component of a `<rotate>` channel (`rotateZ.ANGLE`, `rotate(3)`)?
fn is_rotate_angle_channel(prop: &str) -> bool {
    let base = prop.split(['.', '(']).next().unwrap_or(prop);
    if !(base.starts_with("rotate") || base == "rotation") {
        return false;
    }
    // `.ANGLE`, or the 4th element by index — the angle is component 3 of the axis-angle
    // quadruple, NOT component 2. (`collada_component` maps "ANGLE" to 2 because it was
    // written for the .X/.Y/.Z of translate and scale; that mapping must not be reused here.)
    prop.ends_with(".ANGLE") || prop.ends_with("(3)")
}

/// The static rotation AXIS for a node's `<rotate sid="...">` element.
///
/// A single-component ANGLE channel animates only the angle; the axis is whatever the node
/// declares. Returns `None` when the node or the named `<rotate>` cannot be found, so the
/// channel is skipped rather than silently given a wrong axis.
fn resolve_rotate_axis(doc: &Document, node_token: &str, sid: &str) -> Option<[f32; 3]> {
    let node = doc.descendants().find(|n| {
        n.tag_name().name() == "node"
            && (n.attribute("id") == Some(node_token)
                || n.attribute("sid") == Some(node_token)
                || n.attribute("name") == Some(node_token))
    })?;
    let rot = node.children().find(|c| {
        c.tag_name().name() == "rotate" && (c.attribute("sid") == Some(sid) || sid == "rotate")
    })?;
    let text = rot.text()?;
    let v: Vec<f32> = text
        .split_whitespace()
        .filter_map(|t| t.parse::<f32>().ok())
        .collect();
    (v.len() >= 3).then(|| [v[0], v[1], v[2]])
}

fn collada_component(prop: &str) -> Option<(&str, usize)> {
    if let Some((base, comp)) = prop.rsplit_once('.') {
        let idx = match comp {
            "X" | "S" | "U" | "R" => 0,
            "Y" | "T" | "V" | "G" => 1,
            "Z" | "P" | "B" | "ANGLE" => 2,
            _ => return None,
        };
        return Some((base, idx));
    }
    if let Some((base, rest)) = prop.split_once('(') {
        let idx = rest.trim_end_matches(')').parse::<usize>().ok()?;
        if idx < 3 {
            return Some((base, idx));
        }
    }
    None
}

/// Linear-sample a piecewise-linear channel (`times` ascending, parallel `vals`) at
/// query time `q`; endpoint-clamped. Empty → 0.
fn collada_sample(times: &[f32], vals: &[f32], q: f32) -> f32 {
    let n = times.len().min(vals.len());
    if n == 0 {
        return 0.0;
    }
    if q <= times[0] {
        return vals[0];
    }
    if q >= times[n - 1] {
        return vals[n - 1];
    }
    let mut i = 0;
    while i + 1 < n && times[i + 1] < q {
        i += 1;
    }
    let (t0, t1) = (times[i], times[i + 1]);
    let f = if t1 > t0 { (q - t0) / (t1 - t0) } else { 0.0 };
    vals[i] + (vals[i + 1] - vals[i]) * f
}

fn resolve_target_joint(
    doc: &Document,
    token: &str,
    name_index: &std::collections::HashMap<&str, u32>,
) -> Option<u32> {
    if let Some(&i) = name_index.get(token) {
        return Some(i);
    }
    let node = doc.descendants().find(|n| {
        n.tag_name().name() == "node"
            && (n.attribute("id") == Some(token) || n.attribute("sid") == Some(token))
    })?;
    let sid = node.attribute("sid")?;
    name_index.get(sid).copied()
}

/// Read a `<library_controllers><controller><skin>` bound to geometry `geom_id`
/// into an [`engine_api::rig::Skin`]. Joints come from the `<joints>` block (JOINT
/// `Name_array` names + `INV_BIND_MATRIX` 4×4 float arrays, COLLADA row-major →
/// transposed to glam column-major); per-vertex influences from `<vertex_weights>`
/// (`<vcount>`/`<v>` (joint,weight) index pairs, reduced to the top 4). Parent +
/// local bind come from the `<library_visual_scenes>` JOINT node hierarchy (by
/// `sid`; a node's `<matrix>` is its local transform, row-major → transposed).
/// Returns `None` if no skin binds this geometry or its vertex count disagrees.
fn read_collada_skin(doc: &Document, geom_id: Option<&str>, nverts: usize) -> Option<Skin> {
    // The <skin> whose source is this geometry (else the first skin in the doc).
    let skin_el = doc
        .descendants()
        .filter(|n| n.tag_name().name() == "skin")
        .find(|s| match (geom_id, s.attribute("source")) {
            (Some(g), Some(src)) => strip_hash(src) == g,
            _ => false,
        })
        .or_else(|| descendant(doc.root(), "skin"))?;

    // Joints: JOINT names + INV_BIND_MATRIX (16 floats each, row-major).
    let joints_el = child_by_tag(skin_el, "joints")?;
    let names = input_source(doc, joints_el, "JOINT").and_then(read_name_array)?;
    let njoints = names.len();
    if njoints == 0 {
        return None;
    }
    let ibm: Vec<f64> = input_source(doc, joints_el, "INV_BIND_MATRIX")
        .and_then(|s| descendant(s, "float_array"))
        .map(|fa| parse_numbers(fa.text().unwrap_or("")))
        .unwrap_or_default();
    let inverse_bind: Vec<[[f32; 4]; 4]> = (0..njoints)
        .map(|j| {
            if ibm.len() >= (j + 1) * 16 {
                mat4_rowmajor(&ibm[j * 16..j * 16 + 16]).to_cols_array_2d()
            } else {
                Mat4::IDENTITY.to_cols_array_2d()
            }
        })
        .collect();

    // Per-vertex influences: <vertex_weights> <vcount> + <v> (joint,weight pairs).
    let vw = child_by_tag(skin_el, "vertex_weights")?;
    let joint_off = input_offset(vw, "JOINT")?;
    let weight_off = input_offset(vw, "WEIGHT")?;
    let stride = vw
        .children()
        .filter(|n| n.tag_name().name() == "input")
        .filter_map(|n| n.attribute("offset"))
        .filter_map(|s| s.parse::<usize>().ok())
        .max()
        .unwrap_or(0)
        + 1;
    let weights: Vec<f64> = input_source(doc, vw, "WEIGHT")
        .and_then(|s| descendant(s, "float_array"))
        .map(|fa| parse_numbers(fa.text().unwrap_or("")))?;
    let vcount: Vec<usize> = child_by_tag(vw, "vcount")
        .and_then(|n| n.text())
        .map(parse_numbers)
        .unwrap_or_default()
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let vidx: Vec<i64> = child_by_tag(vw, "v")
        .and_then(|n| n.text())
        .map(parse_numbers)
        .unwrap_or_default()
        .into_iter()
        .map(|v| v as i64)
        .collect();
    // Vertex-weight count must match the mesh control points to stay parallel.
    if vcount.len() != nverts {
        return None;
    }

    let mut vertex_joints = vec![[0u32; 4]; nverts];
    let mut vertex_weights = vec![[0.0f32; 4]; nverts];
    let mut cursor = 0usize;
    for (vtx, &c) in vcount.iter().enumerate() {
        let mut infl: Vec<(u32, f32)> = Vec::with_capacity(c);
        for k in 0..c {
            let base = cursor + k * stride;
            if base + stride > vidx.len() {
                break;
            }
            let ji = vidx[base + joint_off];
            let wi = vidx[base + weight_off];
            if ji >= 0 && (ji as usize) < njoints && wi >= 0 && (wi as usize) < weights.len() {
                let w = weights[wi as usize] as f32;
                if w != 0.0 {
                    infl.push((ji as u32, w));
                }
            }
        }
        cursor += c * stride;
        infl.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (s, &(j, w)) in infl.iter().take(4).enumerate() {
            vertex_joints[vtx][s] = j;
            vertex_weights[vtx][s] = w;
        }
    }

    let joints = build_collada_joints(doc, &names);
    Some(Skin {
        joints,
        inverse_bind,
        vertex_joints,
        vertex_weights,
    })
}

/// Build joints (name + parent + local bind) by matching each JOINT `sid`/`id`
/// to a `<node type="JOINT">` in `<library_visual_scenes>`: `parent` is the nearest
/// enclosing joint node that is itself in `names`; `local` is that node's `<matrix>`
/// (row-major → transposed), or identity when absent. Falls back to a flat root
/// list when no visual scene is present.
fn build_collada_joints(doc: &Document, names: &[String]) -> Vec<Joint> {
    // sid/id → joint index, for parent lookup.
    let index: std::collections::HashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i as u32))
        .collect();
    names
        .iter()
        .map(|nm| {
            let node = doc.descendants().find(|n| {
                n.tag_name().name() == "node"
                    && (n.attribute("sid") == Some(nm) || n.attribute("id") == Some(nm))
            });
            let local = node
                .and_then(|n| child_by_tag(n, "matrix"))
                .and_then(|m| m.text())
                .map(parse_numbers)
                .filter(|f| f.len() >= 16)
                .map(|f| mat4_rowmajor(&f).to_cols_array_2d())
                .unwrap_or_else(|| Mat4::IDENTITY.to_cols_array_2d());
            // Parent: nearest ancestor <node> whose sid/id is also a joint.
            let mut parent = None;
            if let Some(n) = node {
                let mut a = n.parent();
                while let Some(p) = a {
                    if p.tag_name().name() == "node" {
                        if let Some(&pi) = p
                            .attribute("sid")
                            .or_else(|| p.attribute("id"))
                            .and_then(|k| index.get(k))
                        {
                            parent = Some(pi);
                            break;
                        }
                    }
                    a = p.parent();
                }
            }
            Joint {
                name: nm.clone(),
                parent,
                local,
            }
        })
        .collect()
}

/// Resolve a `<input semantic=…>`'s `source="#id"` to the referenced element.
fn input_source<'a>(doc: &'a Document<'a>, parent: Node, semantic: &str) -> Option<Node<'a, 'a>> {
    let src = input_with_semantic(parent, semantic)?
        .attribute("source")
        .map(strip_hash)?;
    element_by_id(doc, src)
}

/// The `offset` of a `<vertex_weights>` `<input semantic=…>`.
fn input_offset(parent: Node, semantic: &str) -> Option<usize> {
    input_with_semantic(parent, semantic)?
        .attribute("offset")
        .and_then(|s| s.parse::<usize>().ok())
}

/// Read a `<source>`'s `<Name_array>`/`<IDREF_array>` as whitespace-split strings.
fn read_name_array(source: Node) -> Option<Vec<String>> {
    let arr = descendant(source, "Name_array").or_else(|| descendant(source, "IDREF_array"))?;
    Some(
        arr.text()
            .unwrap_or("")
            .split_whitespace()
            .map(String::from)
            .collect(),
    )
}

/// A COLLADA row-major 4×4 (16 floats) as a glam column-major [`Mat4`] (transpose).
fn mat4_rowmajor(f: &[f64]) -> Mat4 {
    let a: [f32; 16] = std::array::from_fn(|i| f[i] as f32);
    Mat4::from_cols_array(&a).transpose()
}

/// De-index a face-varying primitive input (`NORMAL`/`TEXCOORD`) to one value per
/// control point (first corner touching a vertex wins). `comps` is how many source
/// components to keep (3 for normals, 2 for UV); the source accessor stride may be
/// larger (e.g. texcoord STPQ). No-op if the semantic is absent or unresolved.
#[allow(clippy::too_many_arguments)]
fn deindex_corner_attr<T: Copy>(
    inputs: &[Node],
    semantic: &str,
    doc: &Document,
    p: &[u32],
    vidx: &[u32],
    index_stride: usize,
    ntuples: usize,
    comps: usize,
    out: &mut [Option<T>],
    make: impl Fn(&[f64]) -> T,
) {
    let Some(input) = inputs
        .iter()
        .find(|n| n.attribute("semantic") == Some(semantic))
    else {
        return;
    };
    let offset = input
        .attribute("offset")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let Some((floats, src_stride)) = resolve_input_source(doc, *input, comps) else {
        return;
    };
    for i in 0..ntuples {
        let v = vidx[i] as usize;
        if v >= out.len() || out[v].is_some() {
            continue;
        }
        let ai = p[i * index_stride + offset] as usize;
        let base = ai * src_stride;
        if base + comps <= floats.len() {
            out[v] = Some(make(&floats[base..base + comps]));
        }
    }
}

/// Resolve a primitive `<input source="#id">` to its `<source>`'s `<float_array>`
/// floats + accessor stride (defaulting to `min_stride` when unspecified).
fn resolve_input_source(
    doc: &Document,
    input: Node,
    min_stride: usize,
) -> Option<(Vec<f64>, usize)> {
    let id = input.attribute("source").map(strip_hash)?;
    let src = element_by_id(doc, id)?;
    let farray = descendant(src, "float_array")?;
    let floats = parse_numbers(farray.text().unwrap_or(""));
    let stride = accessor_stride(src).unwrap_or(min_stride).max(min_stride);
    Some((floats, stride))
}

fn push_tri(tris: &mut Vec<[u32; 3]>, t: [u32; 3], nverts: u32, ctx: &str) -> Result<()> {
    if t[0] >= nverts || t[1] >= nverts || t[2] >= nverts {
        bail!("{ctx} references vertex out of range (nverts={nverts})");
    }
    tris.push(t);
    Ok(())
}

fn strip_hash(s: &str) -> &str {
    s.strip_prefix('#').unwrap_or(s)
}

/// First descendant element (any depth) with the given local tag name.
fn descendant<'a>(root: Node<'a, 'a>, tag: &str) -> Option<Node<'a, 'a>> {
    root.descendants()
        .find(|n| n.is_element() && n.tag_name().name() == tag)
}

/// First direct child element with the given local tag name.
fn child_by_tag<'a>(node: Node<'a, 'a>, tag: &str) -> Option<Node<'a, 'a>> {
    node.children()
        .find(|n| n.is_element() && n.tag_name().name() == tag)
}

/// Direct-child `<input>` with the given `semantic`.
fn input_with_semantic<'a>(node: Node<'a, 'a>, semantic: &str) -> Option<Node<'a, 'a>> {
    node.children()
        .find(|n| n.tag_name().name() == "input" && n.attribute("semantic") == Some(semantic))
}

/// Element anywhere in the document whose `id` attribute equals `id`.
fn element_by_id<'a>(doc: &'a Document<'a>, id: &str) -> Option<Node<'a, 'a>> {
    doc.descendants()
        .find(|n| n.is_element() && n.attribute("id") == Some(id))
}

/// The `stride` of a `<source>`'s `<technique_common><accessor>` (component count
/// per element, e.g. 3 for XYZ).
fn accessor_stride(source: Node) -> Option<usize> {
    descendant(source, "accessor")?
        .attribute("stride")
        .and_then(|s| s.parse::<usize>().ok())
}

/// Whitespace-separated numbers → `Vec<f64>` (skips unparseable tokens).
fn parse_numbers(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A quad as two <triangles>, POSITION at offset 0, NORMAL at offset 1 (so
    // index_stride=2 — exercises the interleaved-tuple stride handling).
    const TRIS: &str = r##"<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <library_geometries>
  <geometry id="Cloth-mesh" name="Cloth">
   <mesh>
    <source id="Cloth-positions">
     <float_array id="Cloth-positions-array" count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
     <technique_common>
      <accessor source="#Cloth-positions-array" count="4" stride="3">
       <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor>
     </technique_common>
    </source>
    <source id="Cloth-normals">
     <float_array id="Cloth-normals-array" count="3">0 0 1</float_array>
     <technique_common>
      <accessor source="#Cloth-normals-array" count="1" stride="3"/>
     </technique_common>
    </source>
    <vertices id="Cloth-vertices">
     <input semantic="POSITION" source="#Cloth-positions"/>
    </vertices>
    <triangles count="2">
     <input semantic="VERTEX" source="#Cloth-vertices" offset="0"/>
     <input semantic="NORMAL" source="#Cloth-normals" offset="1"/>
     <p>0 0 1 0 2 0  0 0 2 0 3 0</p>
    </triangles>
   </mesh>
  </geometry>
 </library_geometries>
</COLLADA>"##;

    #[test]
    fn parses_triangles_with_interleaved_offsets() {
        let obj = parse_collada(TRIS, "cloth").unwrap();
        assert_eq!(obj.positions.len(), 4);
        assert_eq!(obj.positions[2], [1.0, 1.0, 0.0]);
        // VERTEX indices pulled from offset-0 slot of each 2-wide tuple.
        assert_eq!(obj.tris, vec![[0, 1, 2], [0, 2, 3]]);
        // NORMAL (offset 1) de-indexed: every corner references normal 0 = (0,0,1),
        // so all 4 control points get it → parallel per-vertex normals.
        assert_eq!(obj.normals.len(), 4, "normals parallel to positions");
        assert!(obj.normals.iter().all(|n| *n == [0.0, 0.0, 1.0]));
        assert!(obj.uvs.is_empty(), "no TEXCOORD input → empty");
    }

    #[test]
    fn parses_facevarying_normals_and_uvs() {
        // Two triangles over 4 verts, with per-corner NORMAL (offset 1) + TEXCOORD
        // (offset 2). Each control point's first corner picks its normal/uv.
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <source id="n"><float_array count="12">1 0 0  0 1 0  0 0 1  0 0 -1</float_array>
   <technique_common><accessor source="#na" count="4" stride="3"/></technique_common></source>
  <source id="t"><float_array count="8">0 0  1 0  1 1  0 1</float_array>
   <technique_common><accessor source="#ta" count="4" stride="2"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="2">
   <input semantic="VERTEX" source="#v" offset="0"/>
   <input semantic="NORMAL" source="#n" offset="1"/>
   <input semantic="TEXCOORD" source="#t" offset="2"/>
   <p>0 0 0  1 1 1  2 2 2   0 0 0  2 2 2  3 3 3</p>
  </triangles>
 </mesh></geometry></library_geometries></COLLADA>"##;
        let obj = parse_collada(src, "g").unwrap();
        assert_eq!(obj.tris, vec![[0, 1, 2], [0, 2, 3]]);
        assert_eq!(obj.normals.len(), 4);
        assert_eq!(obj.normals[1], [0.0, 1.0, 0.0], "v1 → normal 1");
        assert_eq!(obj.normals[3], [0.0, 0.0, -1.0], "v3 → normal 3");
        assert_eq!(obj.uvs.len(), 4);
        assert_eq!(obj.uvs[2], [1.0, 1.0], "v2 → uv 2");
        assert_eq!(obj.uvs[3], [0.0, 1.0], "v3 → uv 3");
    }

    #[test]
    fn reads_vertices_level_shared_normals() {
        // NORMAL input on <vertices> (per-control-point, indexed by vertex) — some
        // exporters use this instead of a face-varying primitive input.
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="9">0 0 0  1 0 0  0 1 0</float_array>
   <technique_common><accessor source="#pa" count="3" stride="3"/></technique_common></source>
  <source id="n"><float_array count="9">1 0 0  0 1 0  0 0 1</float_array>
   <technique_common><accessor source="#na" count="3" stride="3"/></technique_common></source>
  <vertices id="v">
   <input semantic="POSITION" source="#p"/>
   <input semantic="NORMAL" source="#n"/>
  </vertices>
  <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/>
   <p>0 1 2</p></triangles>
 </mesh></geometry></library_geometries></COLLADA>"##;
        let obj = parse_collada(src, "g").unwrap();
        assert_eq!(obj.normals.len(), 3, "per-vertex normals from <vertices>");
        assert_eq!(obj.normals[0], [1.0, 0.0, 0.0]);
        assert_eq!(obj.normals[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn parses_polylist_and_fans() {
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="12">0 0 0 1 0 0 1 1 0 0 1 0</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <polylist count="1"><input semantic="VERTEX" source="#v" offset="0"/>
   <vcount>4</vcount><p>0 1 2 3</p></polylist>
 </mesh></geometry></library_geometries></COLLADA>"##;
        let obj = parse_collada(src, "g").unwrap();
        // One quad → fan (0,1,2),(0,2,3).
        assert_eq!(obj.tris, vec![[0, 1, 2], [0, 2, 3]]);
        assert!(obj.normals.is_empty(), "COLLADA normals deferred in v1");
    }

    #[test]
    fn reads_skin_controller() {
        // Quad mesh + a 2-joint skin (Root, Child) via <library_controllers>, with a
        // visual-scene joint hierarchy (Child under Root).
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="2"><input semantic="VERTEX" source="#v" offset="0"/>
   <p>0 1 2  0 2 3</p></triangles>
 </mesh></geometry></library_geometries>
 <library_controllers><controller id="c"><skin source="#g">
  <source id="jn"><Name_array count="2">Root Child</Name_array>
   <technique_common><accessor source="#jna" count="2" stride="1"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="ibm"><float_array count="32">1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1   1 0 0 -2  0 1 0 0  0 0 1 0  0 0 0 1</float_array>
   <technique_common><accessor source="#ibma" count="2" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
  <source id="wt"><float_array count="2">1.0 0.5</float_array>
   <technique_common><accessor source="#wta" count="2" stride="1"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints>
   <input semantic="JOINT" source="#jn"/>
   <input semantic="INV_BIND_MATRIX" source="#ibm"/>
  </joints>
  <vertex_weights count="4">
   <input semantic="JOINT" source="#jn" offset="0"/>
   <input semantic="WEIGHT" source="#wt" offset="1"/>
   <vcount>1 2 1 1</vcount>
   <v>0 0   0 1 1 1   1 0   1 0</v>
  </vertex_weights>
 </skin></controller></library_controllers>
 <library_visual_scenes><visual_scene id="scene">
  <node id="Arm">
   <node id="Root" sid="Root" type="JOINT">
    <matrix>1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1</matrix>
    <node id="Child" sid="Child" type="JOINT">
     <matrix>1 0 0 2  0 1 0 0  0 0 1 0  0 0 0 1</matrix>
    </node>
   </node>
  </node>
 </visual_scene></library_visual_scenes>
</COLLADA>"##;
        let obj = parse_collada(src, "g").unwrap();
        let skin = obj.skin.expect("skin present");
        assert_eq!(skin.joints.len(), 2);
        assert_eq!(skin.joints[0].name, "Root");
        assert_eq!(skin.joints[1].name, "Child");
        assert_eq!(skin.joints[0].parent, None);
        assert_eq!(skin.joints[1].parent, Some(0), "Child under Root");
        // Child local = translation(+2) (row-major <matrix> transposed to glam col3).
        assert!((skin.joints[1].local[3][0] - 2.0).abs() < 1e-5);
        // Child inverse-bind = translation(-2).
        assert!((skin.inverse_bind[1][3][0] - (-2.0)).abs() < 1e-5);
        // Weights: v0→j0(1.0); v1→j0(0.5)+j1(0.5); v2→j1(1.0).
        assert_eq!(skin.vertex_joints[0][0], 0);
        assert!((skin.vertex_weights[0][0] - 1.0).abs() < 1e-5);
        let v1: Vec<(u32, f32)> = skin.vertex_joints[1]
            .iter()
            .zip(&skin.vertex_weights[1])
            .filter(|(_, &w)| w > 0.0)
            .map(|(&j, &w)| (j, w))
            .collect();
        assert_eq!(v1.len(), 2, "v1 shared");
        assert!(v1.iter().any(|&(j, w)| j == 0 && (w - 0.5).abs() < 1e-5));
        assert!(v1.iter().any(|&(j, w)| j == 1 && (w - 0.5).abs() < 1e-5));
        assert_eq!(skin.vertex_joints[2][0], 1);
        assert!((skin.vertex_weights[2][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn reads_matrix_animation() {
        // Skin (Root/Child) + a library_animations matrix track on Root: key0 =
        // identity, key1 = rotZ(90°) with translation (3,0,0). Decomposed per key
        // into Translation / Rotation / Scale channels.
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="2"><input semantic="VERTEX" source="#v" offset="0"/>
   <p>0 1 2  0 2 3</p></triangles>
 </mesh></geometry></library_geometries>
 <library_controllers><controller id="c"><skin source="#g">
  <source id="jn"><Name_array count="2">Root Child</Name_array>
   <technique_common><accessor source="#jna" count="2" stride="1"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="wt"><float_array count="1">1.0</float_array>
   <technique_common><accessor source="#wta" count="1" stride="1"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#jn"/></joints>
  <vertex_weights count="4">
   <input semantic="JOINT" source="#jn" offset="0"/>
   <input semantic="WEIGHT" source="#wt" offset="1"/>
   <vcount>1 1 1 1</vcount><v>0 0  0 0  0 0  0 0</v>
  </vertex_weights>
 </skin></controller></library_controllers>
 <library_animations><animation id="Root-anim">
  <source id="in"><float_array count="2">0 1</float_array>
   <technique_common><accessor source="#ina" count="2" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
  <source id="out"><float_array count="32">1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1   0 -1 0 3  1 0 0 0  0 0 1 0  0 0 0 1</float_array>
   <technique_common><accessor source="#outa" count="2" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
  <sampler id="samp">
   <input semantic="INPUT" source="#in"/>
   <input semantic="OUTPUT" source="#out"/>
  </sampler>
  <channel source="#samp" target="Root/transform"/>
 </animation></library_animations>
 <library_visual_scenes><visual_scene id="scene">
  <node id="Root" sid="Root" type="JOINT"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
   <node id="Child" sid="Child" type="JOINT"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>
  </node>
 </visual_scene></library_visual_scenes>
</COLLADA>"##;
        let obj = parse_collada(src, "g").unwrap();
        assert!(obj.skin.is_some());
        assert_eq!(obj.animations.len(), 1, "one clip");
        let clip = &obj.animations[0];
        assert_eq!(clip.name, "Root-anim");
        assert!((clip.duration - 1.0).abs() < 1e-5);
        // T / R / S channels, all on joint 0 (Root).
        let trans = clip
            .channels
            .iter()
            .find(|c| c.path == AnimationPath::Translation)
            .unwrap();
        assert_eq!(trans.target_joint, 0);
        assert!(trans.is_consistent());
        assert_eq!(trans.times, vec![0.0, 1.0]);
        // key0 t=(0,0,0), key1 t=(3,0,0).
        assert!((trans.values[0]).abs() < 1e-5);
        assert!((trans.values[3] - 3.0).abs() < 1e-5);
        let rot = clip
            .channels
            .iter()
            .find(|c| c.path == AnimationPath::Rotation)
            .unwrap();
        // key0 identity quat; key1 = 90° about Z = (0,0,±√½,±√½) (sign-agnostic).
        assert!((rot.values[3] - 1.0).abs() < 1e-5, "key0 w≈1");
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!((rot.values[4]).abs() < 1e-4, "key1 qx≈0");
        assert!((rot.values[5]).abs() < 1e-4, "key1 qy≈0");
        assert!((rot.values[6].abs() - s).abs() < 1e-4, "key1 |qz|≈√½");
        assert!((rot.values[7].abs() - s).abs() < 1e-4, "key1 |qw|≈√½");
        // Scale ≈ 1 both keys.
        let scale = clip
            .channels
            .iter()
            .find(|c| c.path == AnimationPath::Scale)
            .unwrap();
        assert!((scale.values[0] - 1.0).abs() < 1e-4);
        assert!((scale.values[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn reads_translate_vector_animation() {
        // Skin + a non-matrix <translate> sampler (stride 3) targeting Root/translate:
        // (0,0,0) → (5,0,0). Maps directly to one Translation channel.
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="2"><input semantic="VERTEX" source="#v" offset="0"/>
   <p>0 1 2  0 2 3</p></triangles>
 </mesh></geometry></library_geometries>
 <library_controllers><controller id="c"><skin source="#g">
  <source id="jn"><Name_array count="1">Root</Name_array>
   <technique_common><accessor source="#jna" count="1" stride="1"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="wt"><float_array count="1">1.0</float_array>
   <technique_common><accessor source="#wta" count="1" stride="1"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#jn"/></joints>
  <vertex_weights count="4">
   <input semantic="JOINT" source="#jn" offset="0"/>
   <input semantic="WEIGHT" source="#wt" offset="1"/>
   <vcount>1 1 1 1</vcount><v>0 0  0 0  0 0  0 0</v>
  </vertex_weights>
 </skin></controller></library_controllers>
 <library_animations><animation id="Root-loc">
  <source id="in"><float_array count="2">0 1</float_array>
   <technique_common><accessor source="#ina" count="2" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
  <source id="out"><float_array count="6">0 0 0  5 0 0</float_array>
   <technique_common><accessor source="#outa" count="2" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
  <sampler id="samp"><input semantic="INPUT" source="#in"/><input semantic="OUTPUT" source="#out"/></sampler>
  <channel source="#samp" target="Root/translate"/>
 </animation></library_animations>
 <library_visual_scenes><visual_scene id="scene">
  <node id="Root" sid="Root" type="JOINT"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>
 </visual_scene></library_visual_scenes>
</COLLADA>"##;
        let obj = parse_collada(src, "g").unwrap();
        let clip = &obj.animations[0];
        assert_eq!(clip.channels.len(), 1, "one translate channel");
        let ch = &clip.channels[0];
        assert_eq!(ch.path, AnimationPath::Translation);
        assert_eq!(ch.target_joint, 0);
        assert!(ch.is_consistent());
        assert_eq!(ch.times, vec![0.0, 1.0]);
        assert_eq!(ch.values, vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn reads_single_component_animation_channels() {
        // Three single-component channels (Root/location.X/.Y/.Z) with X and Y keyed
        // at DIFFERENT times → combined into one Translation channel at the union of
        // times, each axis linearly sampled (Z absent → 0).
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="2"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2  0 2 3</p></triangles>
 </mesh></geometry></library_geometries>
 <library_controllers><controller id="c"><skin source="#g">
  <source id="jn"><Name_array count="1">Root</Name_array>
   <technique_common><accessor source="#jna" count="1" stride="1"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="wt"><float_array count="1">1.0</float_array>
   <technique_common><accessor source="#wta" count="1" stride="1"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#jn"/></joints>
  <vertex_weights count="4"><input semantic="JOINT" source="#jn" offset="0"/><input semantic="WEIGHT" source="#wt" offset="1"/>
   <vcount>1 1 1 1</vcount><v>0 0  0 0  0 0  0 0</v></vertex_weights>
 </skin></controller></library_controllers>
 <library_animations>
  <animation id="ax"><source id="axi"><float_array count="2">0 1</float_array>
    <technique_common><accessor source="#axia" count="2" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
   <source id="axo"><float_array count="2">0 4</float_array>
    <technique_common><accessor source="#axoa" count="2" stride="1"><param name="X" type="float"/></accessor></technique_common></source>
   <sampler id="axs"><input semantic="INPUT" source="#axi"/><input semantic="OUTPUT" source="#axo"/></sampler>
   <channel source="#axs" target="Root/location.X"/></animation>
  <animation id="ay"><source id="ayi"><float_array count="2">0 0.5</float_array>
    <technique_common><accessor source="#ayia" count="2" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
   <source id="ayo"><float_array count="2">0 2</float_array>
    <technique_common><accessor source="#ayoa" count="2" stride="1"><param name="Y" type="float"/></accessor></technique_common></source>
   <sampler id="ays"><input semantic="INPUT" source="#ayi"/><input semantic="OUTPUT" source="#ayo"/></sampler>
   <channel source="#ays" target="Root/location.Y"/></animation>
 </library_animations>
 <library_visual_scenes><visual_scene id="s"><node id="Root" sid="Root" type="JOINT"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node></visual_scene></library_visual_scenes>
</COLLADA>"##;
        let obj = parse_collada(src, "g").unwrap();
        let clip = &obj.animations[0];
        assert_eq!(clip.channels.len(), 1, "one merged Translation channel");
        let ch = &clip.channels[0];
        assert_eq!(ch.path, AnimationPath::Translation);
        assert_eq!(ch.target_joint, 0);
        assert!(ch.is_consistent());
        // Union of {0,1} (X) and {0,0.5} (Y) = {0, 0.5, 1}.
        assert_eq!(ch.times, vec![0.0, 0.5, 1.0]);
        // X: lerp 0→4 over [0,1] → 0, 2, 4. Y: 0→2 over [0,0.5] → 0, 2, 2(clamped). Z: 0.
        assert_eq!(ch.values, vec![0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 4.0, 2.0, 0.0]);
    }

    /// Skin boilerplate shared by the rotate tests: one joint `Root`, one quad bound to it.
    /// `anim` and `node` are spliced in so each test only shows what it is actually testing.
    fn collada_with(anim: &str, node: &str) -> String {
        format!(
            r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="2"><input semantic="VERTEX" source="#v" offset="0"/>
   <p>0 1 2  0 2 3</p></triangles>
 </mesh></geometry></library_geometries>
 <library_controllers><controller id="c"><skin source="#g">
  <source id="jn"><Name_array count="1">Root</Name_array>
   <technique_common><accessor source="#jna" count="1" stride="1"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="wt"><float_array count="1">1.0</float_array>
   <technique_common><accessor source="#wta" count="1" stride="1"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#jn"/></joints>
  <vertex_weights count="4">
   <input semantic="JOINT" source="#jn" offset="0"/>
   <input semantic="WEIGHT" source="#wt" offset="1"/>
   <vcount>1 1 1 1</vcount><v>0 0  0 0  0 0  0 0</v>
  </vertex_weights>
 </skin></controller></library_controllers>
 <library_animations>{anim}</library_animations>
 <library_visual_scenes><visual_scene id="scene">{node}</visual_scene></library_visual_scenes>
</COLLADA>"##
        )
    }

    /// Apply a quaternion channel's key `k` to a vector, so the assertion is about GEOMETRY
    /// rather than about quaternion components: a 90 deg turn about +Z maps (1,0,0) -> (0,1,0).
    fn rotate_key(ch: &engine_api::rig::AnimationChannel, k: usize, v: glam::Vec3) -> glam::Vec3 {
        let q = glam::Quat::from_xyzw(
            ch.values[k * 4],
            ch.values[k * 4 + 1],
            ch.values[k * 4 + 2],
            ch.values[k * 4 + 3],
        );
        q * v
    }

    /// A 4-float `<rotate>` sampler: axis + angle in DEGREES per key. Checked by the geometry
    /// it produces, and specifically that the angle is read as degrees (90 -> quarter turn),
    /// not radians.
    #[test]
    fn reads_axis_angle_rotate_sampler() {
        let anim = r##"<animation id="Root-rot">
  <source id="in"><float_array count="3">0 1 2</float_array>
   <technique_common><accessor source="#ina" count="3" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
  <source id="out"><float_array count="12">0 0 1 0   0 0 1 90   1 0 0 180</float_array>
   <technique_common><accessor source="#outa" count="3" stride="4"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/><param name="ANGLE" type="float"/></accessor></technique_common></source>
  <sampler id="samp"><input semantic="INPUT" source="#in"/><input semantic="OUTPUT" source="#out"/></sampler>
  <channel source="#samp" target="Root/rotateZ"/>
 </animation>"##;
        let node = r##"<node id="Root" sid="Root" type="JOINT"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>"##;
        let obj = parse_collada(&collada_with(anim, node), "g").unwrap();
        let clip = &obj.animations[0];
        assert_eq!(clip.channels.len(), 1, "one rotate channel");
        let ch = &clip.channels[0];
        assert_eq!(ch.path, AnimationPath::Rotation);
        assert_eq!(ch.target_joint, 0);
        assert!(ch.is_consistent(), "4 floats per key");
        assert_eq!(ch.times, vec![0.0, 1.0, 2.0]);

        let x = glam::Vec3::X;
        let near = |a: glam::Vec3, b: glam::Vec3| (a - b).length() < 1.0e-5;
        // angle 0 about +Z: identity.
        assert!(near(rotate_key(ch, 0, x), x), "key 0 must be identity");
        // 90 DEGREES about +Z: (1,0,0) -> (0,1,0). If degrees were mistaken for radians this
        // would be a ~5157 deg rotation and land nowhere near +Y.
        assert!(
            near(rotate_key(ch, 1, x), glam::Vec3::Y),
            "key 1 must be a quarter turn about +Z, got {:?}",
            rotate_key(ch, 1, x)
        );
        // 180 degrees about +X: (0,1,0) -> (0,-1,0).
        assert!(
            near(rotate_key(ch, 2, glam::Vec3::Y), -glam::Vec3::Y),
            "key 2 must be a half turn about +X, got {:?}",
            rotate_key(ch, 2, glam::Vec3::Y)
        );
    }

    /// A single-component `Root/rotateY.ANGLE` channel: only the ANGLE is keyed, and the axis
    /// must come from the node's own `<rotate sid="rotateY">`. This is the case that needs the
    /// node lookup — using the translate/scale component path would silently treat ANGLE as a
    /// Z translation.
    #[test]
    fn reads_single_component_rotate_angle_channel_using_the_node_axis() {
        let anim = r##"<animation id="Root-rotY">
  <source id="in"><float_array count="2">0 1</float_array>
   <technique_common><accessor source="#ina" count="2" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
  <source id="out"><float_array count="2">0 90</float_array>
   <technique_common><accessor source="#outa" count="2" stride="1"><param name="ANGLE" type="float"/></accessor></technique_common></source>
  <sampler id="samp"><input semantic="INPUT" source="#in"/><input semantic="OUTPUT" source="#out"/></sampler>
  <channel source="#samp" target="Root/rotateY.ANGLE"/>
 </animation>"##;
        // The node declares the axis: +Y.
        let node = r##"<node id="Root" sid="Root" type="JOINT">
   <rotate sid="rotateY">0 1 0 0</rotate>
   <matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>"##;
        let obj = parse_collada(&collada_with(anim, node), "g").unwrap();
        let clip = &obj.animations[0];
        assert_eq!(clip.channels.len(), 1, "one rotate channel");
        let ch = &clip.channels[0];
        assert_eq!(
            ch.path,
            AnimationPath::Rotation,
            "an ANGLE channel is a ROTATION, not a translation component"
        );
        assert!(ch.is_consistent());
        assert_eq!(ch.times, vec![0.0, 1.0]);
        let near = |a: glam::Vec3, b: glam::Vec3| (a - b).length() < 1.0e-5;
        assert!(near(rotate_key(ch, 0, glam::Vec3::X), glam::Vec3::X));
        // 90 deg about +Y takes (1,0,0) to (0,0,-1).
        assert!(
            near(rotate_key(ch, 1, glam::Vec3::X), -glam::Vec3::Z),
            "must rotate about the NODE's +Y axis, got {:?}",
            rotate_key(ch, 1, glam::Vec3::X)
        );
    }

    /// Without a resolvable axis the channel must be DROPPED, not guessed. A wrong axis
    /// produces a rig that looks plausible and animates incorrectly, which is worse.
    #[test]
    fn a_rotate_angle_channel_with_no_node_axis_is_dropped() {
        let anim = r##"<animation id="Root-rotY">
  <source id="in"><float_array count="2">0 1</float_array>
   <technique_common><accessor source="#ina" count="2" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
  <source id="out"><float_array count="2">0 90</float_array>
   <technique_common><accessor source="#outa" count="2" stride="1"><param name="ANGLE" type="float"/></accessor></technique_common></source>
  <sampler id="samp"><input semantic="INPUT" source="#in"/><input semantic="OUTPUT" source="#out"/></sampler>
  <channel source="#samp" target="Root/rotateY.ANGLE"/>
 </animation>"##;
        // No <rotate sid="rotateY"> on the node.
        let node = r##"<node id="Root" sid="Root" type="JOINT"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>"##;
        let obj = parse_collada(&collada_with(anim, node), "g").unwrap();
        assert!(
            obj.animations.is_empty() || obj.animations[0].channels.is_empty(),
            "an ANGLE channel with no axis must be dropped, got {:?}",
            obj.animations
        );
    }

    #[test]
    fn no_controller_means_no_skin() {
        let obj = parse_collada(TRIS, "cloth").unwrap();
        assert!(obj.skin.is_none(), "mesh without a controller has no skin");
    }

    #[test]
    fn rejects_out_of_range() {
        let src = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array count="9">0 0 0 1 0 0 0 1 0</float_array>
   <technique_common><accessor source="#pa" count="3" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/>
   <p>0 1 9</p></triangles>
 </mesh></geometry></library_geometries></COLLADA>"##;
        assert!(parse_collada(src, "g").is_err());
    }
}
