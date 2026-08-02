# SPDX-License-Identifier: GPL-3.0-or-later
#
# nguo-3d Blender bridge.
# Creator: Seede XR. First dev: Alex Mkwizu <a.mkwizu@seedexr.com>.
#
# LICENCE NOTE: this directory (bridges/blender/**) is GPL-3.0-or-later, NOT
# GPL-2.0-or-later like the rest of nguo-3d. Two reasons: Blender add-ons must
# be GPL anyway, and the add-on's structure — the extension manifest, the flat
# `classes` list registered from `__init__.py`, the N-panel registration, the
# preset table — follows ClothDrop Reborn
# (GPL-3.0-or-later), read as a design reference in a local working tree and
# not redistributed here.
#
# The register/unregister shape below is derived from that add-on's
# `src/__init__.py`.

import bpy

from . import operators, panel, properties

classes = []
for module in (properties, operators, panel):
    classes.extend(module.classes)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.nguo = bpy.props.PointerProperty(type=properties.NGUO_Properties)


def unregister():
    if hasattr(bpy.types.Scene, "nguo"):
        del bpy.types.Scene.nguo
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
