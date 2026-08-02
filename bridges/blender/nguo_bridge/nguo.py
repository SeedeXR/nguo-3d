# SPDX-License-Identifier: GPL-3.0-or-later
"""Everything the nguo-3d Blender bridge does that is not UI.

Kept free of operator/panel code so the headless entry point and the smoke test
can call it directly.

Axis convention (see ui/docs/INTEROP.md; measured, not recalled):

    nguo is glTF space  — right-handed, +Y up, +Z forward, -X right
    Blender is          — right-handed, +Z up, -Y forward

    blender = (n.x, -n.z,  n.y)      # Rx(-90 deg), det +1
    nguo    = (b.x,  b.z, -b.y)      # Rx(+90 deg), det +1

det is +1 both ways: handedness is preserved, so triangle winding is NOT
touched on the Blender path (unlike Unity, which mirrors).

UVs: glTF/nguo put the origin top-left (v grows down); Blender puts it
bottom-left. Both directions are ``v -> 1 - v``.
"""

import json
import os
import shutil
import subprocess

import bpy

# ---------------------------------------------------------------------------
# axis
# ---------------------------------------------------------------------------


def n2b(v):
    """nguo -> Blender."""
    return (v[0], -v[2], v[1])


def b2n(v):
    """Blender -> nguo."""
    return (v[0], v[2], -v[1])


def flip_v(uv):
    """glTF/nguo UV origin (top-left) <-> Blender UV origin (bottom-left)."""
    return (uv[0], 1.0 - uv[1])


# ---------------------------------------------------------------------------
# Blender mesh -> nguo
# ---------------------------------------------------------------------------

#: The six fabrics `engine_api::FabricPreset` knows about, serialised kebab-case.
FABRIC_PRESETS = ("cotton", "denim", "silk", "jersey", "wool", "leather")


def mesh_to_nguo(obj, depsgraph=None, apply_modifiers=True):
    """A Blender mesh object as an ``engine_api::Object`` dict.

    Coordinates are baked to world space first, because nguo has no scene
    graph — an object's transform would otherwise be silently dropped.
    """
    src = obj
    if apply_modifiers and depsgraph is not None:
        src = obj.evaluated_get(depsgraph)
    mesh = src.to_mesh()
    try:
        mesh.calc_loop_triangles()
        mw = obj.matrix_world
        positions = [b2n(mw @ v.co) for v in mesh.vertices]
        tris = [tuple(t.vertices) for t in mesh.loop_triangles]

        # nguo UVs are per-VERTEX; Blender's are per-LOOP. Collapse by taking
        # the first loop that touches each vertex. ponytail: that silently
        # welds UV seams. Fine for a collision proxy and for a garment whose
        # UVs came from nguo in the first place (no seams introduced); split
        # the vertices instead if a seamful DCC mesh ever has to round-trip.
        uvs = []
        layer = mesh.uv_layers.active
        if layer is not None:
            uvs = [(0.0, 0.0)] * len(mesh.vertices)
            seen = [False] * len(mesh.vertices)
            for li, loop in enumerate(mesh.loops):
                vi = loop.vertex_index
                if not seen[vi]:
                    seen[vi] = True
                    uvs[vi] = flip_v(tuple(layer.uv[li].vector))

        out = {
            "name": obj.name,
            "positions": [list(p) for p in positions],
            "tris": [list(t) for t in tris],
        }
        if uvs:
            out["uvs"] = [list(u) for u in uvs]
        return out
    finally:
        src.to_mesh_clear()


#: Vertex group that marks kinematic (pinned) vertices for the solver.
PIN_GROUP = "nguo_pin"


def pinned_indices(obj, group=PIN_GROUP):
    """Vertex indices in the `nguo_pin` vertex group -> `Object.pins`.

    A vertex group is the obvious Blender-native way to author this and costs
    nothing; without it a free cloth just falls out of frame.
    """
    vg = obj.vertex_groups.get(group)
    if vg is None:
        return []
    return [v.index for v in obj.data.vertices if any(g.group == vg.index for g in v.groups)]


def nguo_to_mesh(name, positions, tris, uvs=None):
    """Build a Blender object from nguo-space arrays. Returns the object."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([n2b(p) for p in positions], [], [tuple(t) for t in tris])
    mesh.update()
    if uvs:
        layer = mesh.uv_layers.new(name="UVMap")
        for li, loop in enumerate(mesh.loops):
            layer.uv[li].vector = flip_v(uvs[loop.vertex_index])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# nguo OBJ frame sequence -> Blender
# ---------------------------------------------------------------------------


def read_obj(path):
    """``(positions, tris)`` in nguo space from an OBJ.

    ``mguso run --out DIR`` writes ``frame_%04d.obj`` containing only ``v`` and
    ``f`` lines (core-engine/crates/engine-io/src/obj.rs:92 ``write_obj``), so
    this stays deliberately small. ``f a/b/c`` is tolerated for hand-authored
    files.
    """
    positions, tris = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                positions.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):  # fan-triangulate
                    tris.append((idx[0], idx[k], idx[k + 1]))
    return positions, tris


def frame_files(directory):
    """Sorted ``frame_*.obj`` paths written by ``mguso run --out``."""
    names = sorted(
        n for n in os.listdir(directory) if n.startswith("frame_") and n.endswith(".obj")
    )
    return [os.path.join(directory, n) for n in names]


def import_sequence(directory, target=None, start_frame=1, name="nguo_drape", n_verts=None):
    """Import an nguo OBJ frame sequence as a shape-key animation.

    Frame 0 becomes the Basis (or is checked against ``target``); every later
    frame becomes one relative shape key, keyframed 0 -> 1 -> 0 across its
    neighbours so exactly one key is fully on at each integer timeline frame.

    Shape keys rather than a mesh-sequence cache because a cache needs Alembic
    or USD and ``engine-io`` writes neither today (it writes OBJ, STL and GLB
    only). Shape keys also keep the object's existing UVs and materials, which
    the positions-only OBJ sequence does not carry.

    ``mguso`` lowers every scene object into ONE vertex space and writes them
    all to each frame, so a drape against a collider produces cloth+collider in
    one mesh. ``n_verts`` keeps only the leading `n` vertices — the cloth, which
    ``build_scene`` always emits first.

    Returns ``(object, n_frames)``.
    """
    paths = frame_files(directory)
    if not paths:
        raise RuntimeError("no frame_*.obj files in %s" % directory)

    positions, tris = read_obj(paths[0])
    if n_verts is not None and n_verts < len(positions):
        positions = positions[:n_verts]
        tris = [t for t in tris if max(t) < n_verts]
    if target is None:
        obj = nguo_to_mesh(name, positions, tris)
    else:
        obj = target
        if len(obj.data.vertices) != len(positions):
            raise RuntimeError(
                "target %r has %d vertices, the sequence has %d"
                % (obj.name, len(obj.data.vertices), len(positions))
            )

    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    basis = obj.data.shape_keys.key_blocks[0]
    basis.data.foreach_set("co", [c for p in positions for c in n2b(p)])

    for i, path in enumerate(paths[1:], start=1):
        pos, _ = read_obj(path)
        if n_verts is not None and n_verts < len(pos):
            pos = pos[:n_verts]
        if len(pos) != len(positions):
            raise RuntimeError(
                "frame %d has %d vertices, frame 0 has %d — topology must be "
                "constant across an nguo sequence" % (i, len(pos), len(positions))
            )
        key = obj.shape_key_add(name="frame_%04d" % i, from_mix=False)
        key.data.foreach_set("co", [c for p in pos for c in n2b(p)])
        # 0 -> 1 -> 0 over three timeline frames. O(3n) keys, not O(n^2).
        f = start_frame + i
        for at, value in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
            key.value = value
            key.keyframe_insert("value", frame=at)
        key.value = 0.0

    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = start_frame + len(paths) - 1
    return obj, len(paths)


def import_glb(path):
    """Import a drape result with UVs and materials.

    ``engine-io``'s only material-carrying writer is ``save_glb``
    (gltf_io.rs:543) — it has no FBX/USD/COLLADA *writer* at all — so GLB is
    the format for anything richer than positions. Blender's bundled glTF
    importer is the reference implementation of the axis conversion, so we let
    it do the conversion rather than duplicating it.

    Returns the list of objects it created.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def export_collider(objs, path, depsgraph=None):
    """Export objects as an OBJ for nguo to use as a collider / avatar target.

    Blender's OBJ exporter at its defaults (``forward_axis='NEGATIVE_Z'``,
    ``up_axis='Y'``) already writes exactly nguo space — verified on Blender
    5.2.0 by exporting the three basis vertices — so the axes are left alone
    on purpose. ``export_triangulated_mesh`` is not optional: nguo's
    ``Object.tris`` is triangles-only.
    """
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.wm.obj_export(
        filepath=path,
        export_selected_objects=True,
        export_triangulated_mesh=True,
        apply_modifiers=True,
        export_materials=False,
        export_uv=True,
        export_normals=True,
    )
    return path


# ---------------------------------------------------------------------------
# driving the engine
# ---------------------------------------------------------------------------


def build_scene(cloth, colliders, preset="cotton", frames=120, fps=30.0, floor_y=None,
                damping=0.02, self_collision=True, depsgraph=None):
    """An ``engine_api::Scene`` JSON dict.

    Field names follow the Rust types exactly: ``Scene``/``Object`` are plain
    snake_case, ``SolverConfig`` is ``rename_all = "kebab-case"``
    (engine-api/src/params.rs:36), hence ``air-friction`` style keys.
    """
    if preset not in FABRIC_PRESETS:
        raise ValueError("unknown fabric preset %r (want one of %r)" % (preset, FABRIC_PRESETS))
    objects = []
    for o in cloth:
        d = mesh_to_nguo(o, depsgraph)
        d["preset"] = preset
        pins = pinned_indices(o)
        if pins:
            d["pins"] = pins
        objects.append(d)
    for o in colliders:
        d = mesh_to_nguo(o, depsgraph)
        d["is_static"] = True
        objects.append(d)
    config = {
        "frames": int(frames),
        "fps": float(fps),
        "damping": float(damping),
        "self-collision": bool(self_collision),
    }
    if floor_y is not None:
        config["floor-y"] = float(floor_y)
    return {"config": config, "objects": objects}


def find_mguso():
    """Locate the engine binary.

    ``NGUO_MGUSO`` wins, then ``nguo``/``mguso`` on PATH, then the debug build
    beside this checkout. ``nguo`` is the CLI another agent is building;
    ``mguso`` is what exists today (core-engine/crates/engine-cli).
    """
    env = os.environ.get("NGUO_MGUSO")
    if env and os.path.isfile(env):
        return env
    for name in ("nguo", "mguso"):
        found = shutil.which(name)
        if found:
            return found
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("../../../core-engine/target/release/mguso",
                "../../../core-engine/target/debug/mguso"):
        cand = os.path.normpath(os.path.join(here, rel))
        if os.path.isfile(cand):
            return cand
    return None


def run_mguso(scene_path, out_dir, frames=None, solver="xpbd", exe=None):
    """``mguso run <scene.json> --out <dir>``. Returns the CLI's JSON report.

    Subcommand and flags read from core-engine/crates/engine-cli/src/main.rs
    (``Command::Run``): ``--frames``, ``--out``, ``--solver {xpbd,implicit}``.
    The report goes to stdout, logs to stderr; exit 0 = success.
    """
    exe = exe or find_mguso()
    if exe is None:
        raise RuntimeError(
            "engine binary not found; set NGUO_MGUSO or put `nguo`/`mguso` on PATH"
        )
    os.makedirs(out_dir, exist_ok=True)
    argv = [exe, "run", scene_path, "--out", out_dir, "--solver", solver]
    if frames is not None:
        argv += ["--frames", str(int(frames))]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "%s exited %d\n%s" % (" ".join(argv), proc.returncode, proc.stderr[-4000:])
        )
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {"stdout": proc.stdout}


def drape(cloth, colliders, work_dir, **kw):
    """Build a scene, solve it, import the result. Returns ``(object, n_frames)``."""
    os.makedirs(work_dir, exist_ok=True)
    scene_path = os.path.join(work_dir, "scene.json")
    out_dir = os.path.join(work_dir, "frames")
    solver = kw.pop("solver", "xpbd")
    exe = kw.pop("exe", None)
    scene = build_scene(cloth, colliders, **kw)
    with open(scene_path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh)
    run_mguso(scene_path, out_dir, frames=kw.get("frames"), solver=solver, exe=exe)
    # The engine concatenates every object into one vertex space; keep only the
    # cloth, which build_scene emitted first.
    n_cloth = sum(len(o["positions"]) for o in scene["objects"][: len(cloth)])
    return import_sequence(out_dir, n_verts=n_cloth)
