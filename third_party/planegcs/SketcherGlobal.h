// SPDX-License-Identifier: GPL-2.0-or-later
//
// NOT UPSTREAM. Written for nguo-3d.
//
// planegcs sources include "../../SketcherGlobal.h", which in FreeCAD defines
// the Sketcher module's DLL import/export attribute. We link planegcs
// statically into one Rust rlib, so the attribute is empty.
#pragma once
#define SketcherExport
