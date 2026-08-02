// SPDX-License-Identifier: GPL-2.0-or-later
// NOT UPSTREAM. Constraints.cpp includes this but uses no symbol from it.
//
// It does, however, call assert() while including <cassert> only under
// `#if DEBUG_DERIVS` — upstream gets the declaration transitively through real
// Boost. We reproduce that one transitive include rather than patch
// Constraints.cpp, so the vendored source stays byte-identical to upstream.
#pragma once
#include <cassert>
