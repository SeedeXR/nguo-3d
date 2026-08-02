# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene-level settings for the nguo-3d bridge.

Structure follows ClothDrop Reborn's `properties.py` (GPL-3.0-or-later) — a
single PropertyGroup hung off `bpy.types.Scene`, with the preset list living
next to it. The fabric names are not ours to invent: they are exactly
`engine_api::FabricPreset`, serialised kebab-case
(core-engine/crates/engine-api/src/material.rs:70).
"""

import bpy

from . import nguo

#: (identifier, label, description) for the EnumProperty.
FABRIC_ITEMS = [
    ("cotton", "Cotton", "Woven, balanced, medium weight"),
    ("denim", "Denim", "Woven, warp-dominant, crisp and heavy"),
    ("silk", "Silk", "Very soft, low shear resistance, fluid drape"),
    ("jersey", "Jersey", "Knit: stretchy, low bend, light"),
    ("wool", "Wool", "Thick, bulky, high bend"),
    ("leather", "Leather", "Nearly inextensible, very high bend, heaviest"),
]

assert {i[0] for i in FABRIC_ITEMS} == set(nguo.FABRIC_PRESETS), "preset list drifted"


class NGUO_Properties(bpy.types.PropertyGroup):
    fabric: bpy.props.EnumProperty(
        name="Fabric",
        description="Fabric preset handed to the nguo-3d solver",
        items=FABRIC_ITEMS,
        default="cotton",
    )
    frames: bpy.props.IntProperty(
        name="Frames", description="Frames to simulate", min=1, max=2000, default=120
    )
    fps: bpy.props.FloatProperty(name="FPS", min=1.0, max=240.0, default=30.0)
    solver: bpy.props.EnumProperty(
        name="Solver",
        description="xpbd = fast preview, implicit = backward-Euler bake",
        items=[
            ("xpbd", "Preview (XPBD)", "Fast preview tier"),
            ("implicit", "Bake (implicit)", "Backward-Euler, higher fidelity, slower"),
        ],
        default="xpbd",
    )
    self_collision: bpy.props.BoolProperty(
        name="Self Collision", description="Vertex-triangle self collision", default=True
    )
    use_floor: bpy.props.BoolProperty(name="Ground Plane", default=False)
    floor_y: bpy.props.FloatProperty(
        name="Ground Height", description="Ground plane height in nguo (Y-up) space", default=0.0
    )
    damping: bpy.props.FloatProperty(
        name="Damping", description="Per-step velocity damping; lets a drape settle",
        min=0.0, max=0.99, default=0.02,
    )
    work_dir: bpy.props.StringProperty(
        name="Work Dir",
        description="Where the scene JSON and the frame sequence are written",
        subtype="DIR_PATH",
        default="//nguo_drape",
    )


classes = [NGUO_Properties]
