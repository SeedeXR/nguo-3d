# SPDX-License-Identifier: GPL-3.0-or-later
"""N-panel in the 3D viewport. Registration pattern (VIEW_3D / UI /
`bl_category`, plus a flat `classes` list) follows ClothDrop Reborn's
`panel.py`, GPL-3.0-or-later.
"""

import bpy

from . import nguo


class NGUO_PT_panel(bpy.types.Panel):
    bl_idname = "NGUO_PT_panel"
    bl_label = "nguo-3d"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "nguo-3d"

    def draw(self, context):
        layout = self.layout
        props = context.scene.nguo

        col = layout.column(align=True)
        col.prop(props, "fabric")
        col.prop(props, "solver")

        col = layout.column(align=True)
        col.prop(props, "frames")
        col.prop(props, "fps")
        col.prop(props, "damping")

        col = layout.column(align=True)
        col.prop(props, "self_collision", toggle=True)
        row = col.row(align=True)
        row.prop(props, "use_floor", toggle=True)
        sub = row.row(align=True)
        sub.enabled = props.use_floor
        sub.prop(props, "floor_y", text="")

        layout.prop(props, "work_dir")

        layout.separator()
        layout.operator("nguo.drape", icon="MOD_CLOTH")

        layout.separator(type="LINE")
        col = layout.column(align=True)
        col.operator("nguo.import_glb", icon="IMPORT")
        col.operator("nguo.import_sequence", icon="SEQUENCE")
        col.operator("nguo.export_collider", icon="EXPORT")

        if nguo.find_mguso() is None:
            box = layout.box()
            box.alert = True
            box.label(text="engine binary not found", icon="ERROR")
            box.label(text="set NGUO_MGUSO or add nguo/mguso to PATH")


classes = [NGUO_PT_panel]
