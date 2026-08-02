<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# third_party/planegcs — provenance

Everything under this directory that is *not* marked "written for nguo-3d" is
third-party code copied verbatim. Nothing upstream has been modified. This file
records exactly what was taken, from where, and under what licence, so that the
next update is a re-copy rather than an investigation.

Recorded 2026-08-01. Every hash and licence line below was read off the files
in this tree, not recalled.

---

## 1. planegcs (the solver)

| | |
|---|---|
| Upstream | <https://github.com/FreeCAD/FreeCAD> |
| Path | `src/Mod/Sketcher/App/planegcs/` |
| Commit | `64bc717c82c7188e9f102e5c121a2b5283320d17` |
| Commit date | 2026-03-17 ("Sketcher: Remove unused arguments") |
| Why this commit | the most recent commit touching `planegcs/` as of 2026-08-01; pinning to it rather than to `main` means re-vendoring only when the solver itself changes |
| Vendored to | `upstream/planegcs/` |
| Licence | **LGPL-2.1-or-later** |

### Licence, as observed

Every one of the eleven files carries, on **line 1**:

```
// SPDX-License-Identifier: LGPL-2.1-or-later
```

followed by a copyright block ("This library is free software; you can
redistribute it and/or modify it under the terms of the GNU Library General
Public License … version 2 of the License, or (at your option) any later
version"), copyright Konstantinos Poulios (2011) and, for `Geo.cpp`, Victor
Titov (2014).

FreeCAD's repository-root `LICENSE` covers the project as a whole; the
per-file SPDX tags above are the operative statement for these files and are
what was checked. **This matches the decision of record to adopt planegcs
(`ui/docs/CONSTRAINTS.md` §1) and the maintainer's stated position. No
contradiction was found.**

### Files copied (verbatim, SHA-256)

```
023f2f7753aff95ded246f41521527a07a1b82ba8815da327ad692e864156d7f  Constraints.cpp
e088865959562632c27ce5c29d67933f4f543dc716679e696d8473eb60a198d6  Constraints.h
4ce413be489276e228160fa52f3aa533dd6b31136d41c36f7f0a9f34f9e8fab2  GCS.cpp
20d89779bbc17af5e3b3e4f070cb6efaacac5e3c08fea393cbffcc0522060b16  GCS.h
169aad09829b81288d3a81d81cd00fdb5ce9723a4ee34450fb0ac8667b8e9e2b  Geo.cpp
20c2de0ed0b5b6e2aa6e7f7f302ec11ce798bcb653b051d44cf5de7b60590af5  Geo.h
28c94e84ff50187ec9b6060ac72e05fb6671a51fae2be40f479d2b4dd151e393  SubSystem.cpp
99e1380a0a08d9fd7ee0fd69e26d3d9dcb46fdf033f4c397c9a05e8df8c53dc4  SubSystem.h
33aee54b020e565faf96629a6d4a57a75a3699acb2336aa96d3663cd70c7e53e  Util.h
dc71350766c1a4edf7024d593cb5d55b0736a4f43d319d25a9c961e8342ca11c  qp_eq.cpp
34e373d9b3b0f5938c849ea7489a8405617692a5062198e8ae0f0c6ac5e4d468  qp_eq.h
```

That is the whole `planegcs/` directory — 11 files, ~13 300 lines. Nothing else
from FreeCAD is vendored: no `Sketcher` module, no `Base`, no `App`, no build
system.

**Zero patches.** There is no `patches/` directory because none was needed. If
one ever is, it goes there as a `.patch` file and the copies above stay
byte-identical, so `git diff` against a fresh upstream copy stays meaningful.

---

## 2. Eigen (the linear algebra planegcs needs)

| | |
|---|---|
| Upstream | <https://gitlab.com/libeigen/eigen> |
| Release | **3.4.1** (tagged 2025-09-29) |
| Tarball | `https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz` |
| Tarball SHA-256 | `b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2` |
| Vendored to | `vendor/eigen/` — the `Eigen/` header tree plus the `COPYING.*` files, 344 files, 6.7 MB |
| Excluded | `unsupported/`, `doc/`, `test/`, `bench/`, `demos/`, CMake files — none of it is reachable from planegcs |
| Licence | primarily **MPL-2.0**, with some BSD and LGPL-2.1 parts; see `vendor/eigen/COPYING.README` |

### Vendored rather than taken from the system — and why

Requirement: a clean macOS machine with only a C++ toolchain must be able to
run `cargo test` with no manual setup.

- **System / Homebrew Eigen** would mean either a hard-coded `/opt/homebrew`
  path (forbidden, and wrong on Intel Macs and on Linux) or a `pkg-config`
  probe that fails until someone runs `brew install eigen`. That is manual
  setup, and it makes the build depend on whichever Eigen version the machine
  happens to have — planegcs branches on `EIGEN_VERSION` in three places, so
  that is a real correctness variable, not just a convenience one.
- **Vendoring** costs 6.7 MB of headers, once, and removes the variable
  entirely. Eigen is header-only, so there is nothing to build or link.

3.4.1 rather than 5.0.x: 3.4.x is the branch FreeCAD builds planegcs against,
and planegcs' `EIGEN_VERSION` feature gates (`EIGEN_SPARSEQR_COMPATIBLE`,
`EIGEN_STOCK_FULLPIVLU_COMPUTE`) were written for it. Both gates evaluate
correctly at 3.4.1 with no manual `-D` flags.

### Licence compatibility

MPL-2.0 is compatible with GPL-2.0-or-later: MPL-2.0 §3.3 explicitly permits
distributing a Larger Work under a "Secondary License", which the MPL defines
as including GPL v2.0 or later. The BSD parts are permissive. The LGPL-2.1
parts sit inside the same relicensing story as planegcs itself (see
`ui/docs/CONSTRAINTS.md`). `EIGEN_MPL2_ONLY` is *not* set, because the LGPL
parts are acceptable here and setting it would be a compile error waiting to
happen on an Eigen bump.

---

## 3. Written for nguo-3d, not upstream

These stand in for FreeCAD infrastructure and for Boost, so that planegcs
compiles unmodified with no external dependency. All carry
`SPDX-License-Identifier: GPL-2.0-or-later` and a "NOT UPSTREAM" banner.

| File | Replaces | Why |
|---|---|---|
| `SketcherGlobal.h` | FreeCAD's DLL export macro header | `#define SketcherExport` to nothing; we link statically. Must sit here, because the sources include it as `"../../SketcherGlobal.h"`. |
| `compat/FCConfig.h` | FreeCAD's generated build config | Empty; planegcs uses nothing from it. |
| `compat/Base/Console.h` | FreeCAD's `Base::Console()` logger | ~50 lines providing `log()` and `warning()`. Silent unless `NGUO_GCS_VERBOSE=1`. Pulling in FreeCAD's Base library for a `printf` would be absurd. |
| `compat/boost/graph/connected_components.hpp` | Boost Graph | A weighted union-find providing the exact six symbols `GCS.cpp` uses. See below. |
| `compat/boost_graph_adjacency_list.hpp` | FreeCAD's Boost Graph wrapper | Forwards to the above. |
| `compat/boost/graph/graph_concepts.hpp` | Boost Graph concepts | `Constraints.cpp` includes it but uses nothing from it — except that it relies on it transitively supplying `<cassert>`, which this stub reproduces. |
| `compat/boost/math/constants/constants.hpp` | Boost Math | `Geo.h` includes it and uses nothing from it (verified: no `math::` or `constants::` reference anywhere in the tree). Empty stub. |

### The Boost question — the material build-story finding

planegcs at this commit **does still reference Boost**, in three headers:

- `Geo.h` → `boost/math/constants/constants.hpp` (included, never used)
- `Constraints.cpp` → `boost/graph/graph_concepts.hpp` (included, never used)
- `GCS.cpp` → `boost/graph/connected_components.hpp` and FreeCAD's
  `boost_graph_adjacency_list.hpp`, and it **does** use them, in `diagnose()`,
  to split the constraint system into decoupled components.

The used surface is six symbols:

```
boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>
boost::add_vertex(g)   boost::add_edge(u, v, g)
boost::num_vertices(g) boost::connected_components(g, &out[0])
```

Boost Graph is header-only but not vendorable in isolation — extracting it
drags in mpl, type_traits, iterator and more, and vendoring all of Boost is
~150 MB. All four includes are angle-bracket, so supplying our own headers on
the include path satisfies them without touching a line of upstream. Real Boost
is never on the include path.

**Net effect: the build needs Eigen only. Boost is not a dependency of this
tree.** The union-find replacement reproduces Boost's labelling contract
(labels dense, `0..n-1`, assigned in ascending order of first appearance),
which `GCS.cpp` depends on.

The risk this buys is bounded and named: if a future planegcs starts using more
of Boost Graph, the shim headers stop compiling — loudly, at build time, not
silently at runtime.

---

## 4. Re-vendoring procedure

1. Pick the new upstream commit; note its hash and date.
2. Replace the eleven files in `upstream/planegcs/` wholesale.
3. `cargo test -p nguo-constraints`. A new `#include` or a new Boost symbol
   shows up here as a compile error.
4. Update section 1 above: commit, date, hashes.
5. If a patch became unavoidable, add `patches/*.patch`, apply it in
   `build.rs`, and say so here.
