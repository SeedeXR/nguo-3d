# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators. Structure follows ClothDrop Reborn's `operators.py`
(GPL-3.0-or-later): one class per verb, a flat `classes` list at the bottom
that `__init__.py` registers.

Every operator is a thin shell over `nguo.py` so the headless entry point can
do the same work without an operator context.
"""

import os

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import nguo


class NGUO_OT_import_glb(bpy.types.Operator, ImportHelper):
    bl_idname = "nguo.import_glb"
    bl_label = "Import Drape (.glb)"
    bl_description = "Import an nguo-3d drape result with its UVs and materials"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".glb"
    filter_glob: bpy.props.StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})

    def execute(self, context):
        created = nguo.import_glb(self.filepath)
        self.report({"INFO"}, "imported %d object(s)" % len(created))
        return {"FINISHED"}


class NGUO_OT_import_sequence(bpy.types.Operator, ImportHelper):
    bl_idname = "nguo.import_sequence"
    bl_label = "Import Drape Sequence"
    bl_description = (
        "Import an nguo-3d frame_*.obj sequence as a shape-key animation. With an "
        "object selected, the sequence is applied to it (keeping its UVs and "
        "materials); otherwise a new object is created"
    )
    bl_options = {"REGISTER", "UNDO"}

    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filter_glob: bpy.props.StringProperty(default="*.obj", options={"HIDDEN"})
    use_active: bpy.props.BoolProperty(
        name="Apply To Active Object",
        description="Add the shape keys to the active object instead of creating a new one",
        default=False,
    )

    def execute(self, context):
        target = context.object if self.use_active else None
        try:
            obj, n = nguo.import_sequence(self.directory, target=target)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "imported %d frames onto %r" % (n, obj.name))
        return {"FINISHED"}


class NGUO_OT_export_collider(bpy.types.Operator, ExportHelper):
    bl_idname = "nguo.export_collider"
    bl_label = "Export Collider (.obj)"
    bl_description = "Export the selected objects as an nguo-3d collider / avatar target"
    bl_options = {"REGISTER"}

    filename_ext = ".obj"
    filter_glob: bpy.props.StringProperty(default="*.obj", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" for o in context.selected_objects)

    def execute(self, context):
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        nguo.export_collider(objs, self.filepath)
        self.report({"INFO"}, "exported %d mesh(es)" % len(objs))
        return {"FINISHED"}


class NGUO_OT_drape(bpy.types.Operator):
    bl_idname = "nguo.drape"
    bl_label = "Drape"
    bl_description = (
        "Solve the active object as cloth against the other selected meshes as "
        "static colliders, then import the result as a shape-key animation"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def execute(self, context):
        props = context.scene.nguo
        cloth = [context.object]
        colliders = [
            o for o in context.selected_objects if o.type == "MESH" and o is not context.object
        ]
        work = bpy.path.abspath(props.work_dir)
        if not work:
            work = os.path.join(bpy.app.tempdir, "nguo_drape")
        try:
            obj, n = nguo.drape(
                cloth,
                colliders,
                work,
                preset=props.fabric,
                frames=props.frames,
                fps=props.fps,
                damping=props.damping,
                self_collision=props.self_collision,
                floor_y=props.floor_y if props.use_floor else None,
                solver=props.solver,
                depsgraph=context.evaluated_depsgraph_get(),
            )
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "draped %d frames onto %r" % (n, obj.name))
        return {"FINISHED"}


classes = [
    NGUO_OT_import_glb,
    NGUO_OT_import_sequence,
    NGUO_OT_export_collider,
    NGUO_OT_drape,
]
