# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless smoke test for the nguo-3d Blender bridge.

    blender --background --factory-startup --python-exit-code 47 \
            --python bridges/blender/tests/smoke.py

Exits 0 when everything passes; an assertion failure becomes exit 47 via
`--python-exit-code`, which is what CI and `nguo_bridge_dcc` check.

It does NOT need the engine binary: the engine round-trip is covered by the
Rust integration test. What it covers is the part only Blender can answer —
that the add-on registers, that its operators and panel exist, and that the
axis conversion and the shape-key sequence import actually behave.
"""

import os
import sys
import tempfile

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import nguo_bridge  # noqa: E402
from nguo_bridge import nguo  # noqa: E402

PASSED = []


def check(name, fn):
    fn()
    PASSED.append(name)
    print("  ok  %s" % name)


# ---------------------------------------------------------------------------


def t_registers_and_unregisters():
    nguo_bridge.register()
    assert hasattr(bpy.types.Scene, "nguo"), "Scene.nguo property missing"
    for cls in nguo_bridge.operators.classes:
        module, name = cls.bl_idname.split(".", 1)
        assert module == "nguo", cls.bl_idname
        assert hasattr(bpy.ops.nguo, name), "operator %s not registered" % cls.bl_idname
    assert bpy.types.NGUO_PT_panel.bl_space_type == "VIEW_3D"
    assert bpy.types.NGUO_PT_panel.bl_region_type == "UI"
    nguo_bridge.unregister()
    assert not hasattr(bpy.types.Scene, "nguo")
    # register/unregister must be idempotent, or reloading the add-on breaks.
    nguo_bridge.register()


def t_scene_properties_default_sanely():
    props = bpy.context.scene.nguo
    assert props.fabric in nguo.FABRIC_PRESETS
    props.fabric = "denim"
    assert bpy.context.scene.nguo.fabric == "denim"
    props.fabric = "cotton"


def t_axis_round_trip():
    for v in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-0.5, 12.25, -7.125)]:
        assert nguo.b2n(nguo.n2b(v)) == v, v
        assert nguo.n2b(nguo.b2n(v)) == v, v
    # up stays up: nguo +Y -> blender +Z
    assert nguo.n2b((0.0, 1.0, 0.0)) == (0.0, 0.0, 1.0)
    # forward stays forward: nguo/glTF +Z -> blender -Y
    assert nguo.n2b((0.0, 0.0, 1.0)) == (0.0, -1.0, 0.0)
    for uv in [(0.0, 0.0), (0.25, 0.75), (1.0, 1.0)]:
        assert nguo.flip_v(nguo.flip_v(uv)) == uv


def t_axis_matches_blenders_own_obj_exporter():
    """The Python conversion must agree with `bpy.ops.wm.obj_export` at its
    defaults, or a mesh exported by the operator and a mesh exported by our
    code land in different places."""
    obj = nguo.nguo_to_mesh(
        "axisprobe", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], [(0, 1, 2)]
    )
    path = os.path.join(tempfile.mkdtemp(), "axis.obj")
    nguo.export_collider([obj], path)
    positions, _ = nguo.read_obj(path)
    for want, got in zip([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], positions):
        assert max(abs(a - b) for a, b in zip(want, got)) < 1e-5, (want, got)
    bpy.data.objects.remove(obj, do_unlink=True)


def t_mesh_round_trips_through_nguo_space():
    src = nguo.nguo_to_mesh(
        "rt",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 2.0, -1.0)],
        [(0, 1, 2), (0, 2, 3)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )
    d = nguo.mesh_to_nguo(src)
    assert len(d["positions"]) == 4 and len(d["tris"]) == 2
    assert max(abs(a - b) for a, b in zip(d["positions"][3], (0.0, 2.0, -1.0))) < 1e-5
    # UVs survive the top-left <-> bottom-left origin flip
    assert max(abs(a - b) for a, b in zip(d["uvs"][2], (1.0, 1.0))) < 1e-5
    # Blender preserves handedness, so winding must be untouched.
    assert tuple(d["tris"][0]) == (0, 1, 2)
    bpy.data.objects.remove(src, do_unlink=True)


def t_scene_json_shape_matches_engine_api():
    src = nguo.nguo_to_mesh("cloth", [(0, 0, 0), (1, 0, 0), (0, 0, 1)], [(0, 1, 2)])
    coll = nguo.nguo_to_mesh("body", [(0, -1, 0), (1, -1, 0), (0, -1, 1)], [(0, 1, 2)])
    scene = nguo.build_scene([src], [coll], preset="denim", frames=7, fps=24.0, floor_y=-2.0)
    assert set(scene) == {"config", "objects"}
    # SolverConfig is rename_all = "kebab-case"
    assert scene["config"]["floor-y"] == -2.0
    assert scene["config"]["self-collision"] is True
    assert scene["config"]["frames"] == 7
    # Object is plain snake_case
    assert scene["objects"][0]["preset"] == "denim"
    assert scene["objects"][1]["is_static"] is True
    assert "preset" not in scene["objects"][1]
    try:
        nguo.build_scene([src], [], preset="pleather")
        raise AssertionError("unknown preset was accepted")
    except ValueError:
        pass
    for o in (src, coll):
        bpy.data.objects.remove(o, do_unlink=True)


def t_pins_come_from_a_vertex_group():
    obj = nguo.nguo_to_mesh("pinned", [(0, 0, 0), (1, 0, 0), (0, 0, 1)], [(0, 1, 2)])
    assert nguo.pinned_indices(obj) == [], "no group means no pins"
    obj.vertex_groups.new(name=nguo.PIN_GROUP).add([0, 2], 1.0, "REPLACE")
    assert nguo.pinned_indices(obj) == [0, 2]
    scene = nguo.build_scene([obj], [])
    assert scene["objects"][0]["pins"] == [0, 2]
    bpy.data.objects.remove(obj, do_unlink=True)


def t_sequence_keeps_only_the_cloth_when_told_a_vertex_count():
    """mguso welds every scene object into one vertex space, so a drape against
    a collider arrives as cloth+collider unless we slice."""
    d = tempfile.mkdtemp()
    for f in range(2):
        with open(os.path.join(d, "frame_%04d.obj" % f), "w", encoding="utf-8") as fh:
            fh.write("v 0 0 0\nv 1 0 0\nv 0 0 1\nv 5 5 5\nv 6 5 5\nv 5 5 6\n"
                     "f 1 2 3\nf 4 5 6\n")
    obj, _ = nguo.import_sequence(d, n_verts=3)
    assert len(obj.data.vertices) == 3, len(obj.data.vertices)
    assert len(obj.data.polygons) == 1, len(obj.data.polygons)
    bpy.data.objects.remove(obj, do_unlink=True)


def t_sequence_imports_as_shape_keys():
    d = tempfile.mkdtemp()
    n_frames = 4
    for f in range(n_frames):
        with open(os.path.join(d, "frame_%04d.obj" % f), "w", encoding="utf-8") as fh:
            fh.write("v 0 %f 0\nv 1 %f 0\nv 0 %f 1\nf 1 2 3\n" % (-f, -f, -f))
    obj, n = nguo.import_sequence(d)
    assert n == n_frames, n
    keys = obj.data.shape_keys.key_blocks
    assert len(keys) == n_frames, list(keys.keys())
    assert keys[0].name == "Basis"
    # Basis is frame 0 in Blender space: nguo (0,0,0) -> blender (0,0,0)
    assert tuple(keys[0].data[0].co) == (0.0, 0.0, 0.0)
    # frame 3 dropped 3 units along nguo -Y == blender -Z
    assert abs(keys[3].data[0].co[2] + 3.0) < 1e-5, tuple(keys[3].data[0].co)
    assert obj.data.shape_keys.animation_data is not None, "shape keys are not animated"
    assert bpy.context.scene.frame_end == n_frames
    bpy.data.objects.remove(obj, do_unlink=True)


def t_sequence_rejects_changing_topology():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "frame_0000.obj"), "w", encoding="utf-8") as fh:
        fh.write("v 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\n")
    with open(os.path.join(d, "frame_0001.obj"), "w", encoding="utf-8") as fh:
        fh.write("v 0 0 0\nv 1 0 0\nf 1 2 2\n")
    try:
        nguo.import_sequence(d)
        raise AssertionError("changing topology was accepted")
    except RuntimeError as exc:
        assert "topology" in str(exc), exc


def t_glb_importer_is_available_under_factory_startup():
    """--factory-startup must still leave the bundled glTF add-on enabled, or
    the GLB path silently dies in CI."""
    assert hasattr(bpy.ops.import_scene, "gltf")
    assert bpy.ops.import_scene.gltf.poll()


def main():
    print("nguo-3d Blender bridge smoke test — Blender %s" % bpy.app.version_string)
    for name, fn in sorted(globals().items()):
        if name.startswith("t_"):
            check(name[2:], fn)
    print("PASSED %d/%d" % (len(PASSED), len(PASSED)))
    print("NGUO-BRIDGE-RESULT: {\"passed\": %d}" % len(PASSED))


if __name__ == "__main__":
    main()
