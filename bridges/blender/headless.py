# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless entry point for the nguo-3d Blender bridge.

    blender --background --factory-startup --python-exit-code 47 \
            --python bridges/blender/headless.py -- <subcommand> [options]

That argv is exactly what `nguo_bridge_dcc::blender_argv` builds, and this is
the *primary tested path* — CI and agents drive Blender through here, the
N-panel is a convenience on top.

Every subcommand prints one machine-readable line

    NGUO-BRIDGE-RESULT: {...}

which `nguo_bridge_dcc::parse_result` lifts back out of Blender's log noise.
Any failure raises, and `--python-exit-code 47` turns that into exit 47, which
the Rust side maps to `BridgeError::Script`.
"""

import argparse
import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nguo_bridge  # noqa: E402
from nguo_bridge import nguo  # noqa: E402

RESULT_PREFIX = "NGUO-BRIDGE-RESULT:"


def emit(**payload):
    print("%s %s" % (RESULT_PREFIX, json.dumps(payload, sort_keys=True)))
    sys.stdout.flush()


def script_args():
    """Everything after the bare `--`; Blender keeps its own args before it."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def open_blend(path):
    if path:
        bpy.ops.wm.open_mainfile(filepath=path)


def pick(names):
    """Objects by name, or every mesh in the scene when no names are given."""
    if names:
        missing = [n for n in names if n not in bpy.data.objects]
        if missing:
            # RuntimeError, NOT SystemExit: `blender --python-exit-code 47` only
            # remaps an *uncaught exception*. SystemExit bypasses it and exits 1,
            # which nguo_bridge_dcc::classify reads as BridgeError::Dcc ("Blender
            # itself failed") instead of BridgeError::Script. A typo'd object name
            # would have been reported to the user as a Blender crash.
            raise RuntimeError("no such object(s): %r" % missing)
        return [bpy.data.objects[n] for n in names]
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def save(path):
    if path:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))


# ---------------------------------------------------------------------------


def cmd_selftest(_args):
    """Register the add-on and assert its operators and panel exist.

    The smoke test CI actually runs — see tests/smoke.py for the assertions.
    """
    nguo_bridge.register()
    ops = [c.bl_idname for c in nguo_bridge.operators.classes]
    present = [o for o in ops if hasattr(bpy.ops.nguo, o.split(".", 1)[1])]
    nguo_bridge.unregister()
    emit(registered=True, operators=present, engine=nguo.find_mguso())


def cmd_import_glb(args):
    open_blend(args.blend)
    created = nguo.import_glb(args.path)
    save(args.save)
    emit(objects=[o.name for o in created])


def cmd_import_sequence(args):
    open_blend(args.blend)
    target = bpy.data.objects[args.target] if args.target else None
    obj, n = nguo.import_sequence(args.dir, target=target)
    save(args.save)
    emit(object=obj.name, frames=n, vertices=len(obj.data.vertices))


def cmd_export_collider(args):
    open_blend(args.blend)
    objs = pick(args.object)
    nguo.export_collider(objs, os.path.abspath(args.out))
    emit(path=os.path.abspath(args.out), objects=[o.name for o in objs])


def cmd_scene(args):
    """Write the engine scene JSON without solving — useful on its own and the
    part of `drape` that does not need the engine binary."""
    open_blend(args.blend)
    cloth = pick(args.cloth)
    colliders = pick(args.collider) if args.collider else []
    scene = nguo.build_scene(
        cloth,
        colliders,
        preset=args.fabric,
        frames=args.frames,
        fps=args.fps,
        floor_y=args.floor_y,
        depsgraph=bpy.context.evaluated_depsgraph_get(),
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(scene, fh)
    emit(
        path=os.path.abspath(args.out),
        objects=len(scene["objects"]),
        vertices=sum(len(o["positions"]) for o in scene["objects"]),
    )


def cmd_drape(args):
    open_blend(args.blend)
    cloth = pick(args.cloth)
    colliders = pick(args.collider) if args.collider else []
    obj, n = nguo.drape(
        cloth,
        colliders,
        args.work,
        preset=args.fabric,
        frames=args.frames,
        fps=args.fps,
        floor_y=args.floor_y,
        solver=args.solver,
        depsgraph=bpy.context.evaluated_depsgraph_get(),
    )
    save(args.save)
    emit(object=obj.name, frames=n, vertices=len(obj.data.vertices))


def main():
    p = argparse.ArgumentParser(prog="nguo blender bridge")
    p.add_argument("--blend", help="open this .blend first")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)

    q = sub.add_parser("import-glb")
    q.add_argument("--path", required=True)
    q.add_argument("--save")
    q.set_defaults(fn=cmd_import_glb)

    q = sub.add_parser("import-sequence")
    q.add_argument("--dir", required=True)
    q.add_argument("--target", help="apply the shape keys to this existing object")
    q.add_argument("--save")
    q.set_defaults(fn=cmd_import_sequence)

    q = sub.add_parser("export-collider")
    q.add_argument("--object", action="append", help="repeatable; default = every mesh")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_export_collider)

    for name, fn in (("scene", cmd_scene), ("drape", cmd_drape)):
        q = sub.add_parser(name)
        q.add_argument("--cloth", action="append", help="repeatable; default = every mesh")
        q.add_argument("--collider", action="append", default=None)
        q.add_argument("--fabric", default="cotton", choices=list(nguo.FABRIC_PRESETS))
        q.add_argument("--frames", type=int, default=120)
        q.add_argument("--fps", type=float, default=30.0)
        q.add_argument("--floor-y", dest="floor_y", type=float, default=None)
        if name == "scene":
            q.add_argument("--out", required=True)
        else:
            q.add_argument("--work", required=True)
            q.add_argument("--solver", default="xpbd", choices=["xpbd", "implicit"])
            q.add_argument("--save")
        q.set_defaults(fn=fn)

    # argparse calls sys.exit(2) on a bad flag, and SystemExit is NOT covered by
    # `blender --python-exit-code 47`, so a usage error would surface to the caller
    # as exit 2 -> BridgeError::Dcc, i.e. "Blender failed" rather than "you passed
    # a bad argument". Re-raise as a normal exception so the 47 mapping applies and
    # the caller gets BridgeError::Script with argparse's message intact.
    try:
        args = p.parse_args(script_args())
    except SystemExit as e:
        if e.code:  # 0 is --help / --version, which really should exit cleanly
            raise RuntimeError("bad arguments: %s" % " ".join(script_args())) from e
        raise
    args.fn(args)


if __name__ == "__main__":
    main()
