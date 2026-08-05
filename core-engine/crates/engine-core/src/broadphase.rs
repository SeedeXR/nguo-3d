// SPDX-License-Identifier: GPL-2.0-or-later
//! Broad phase: uniform spatial hash → candidate (vertex, triangle) pairs whose
//! separation *might* be below `dhat`, for the self-collision narrow phase.
//!
//! ponytail: a uniform grid hash is the simplest correct broad phase and ports
//! cleanly to the GPU later. The reference uses an LBVH
//! (Morton + radix sort + Karras) — that is the *performance* variant of this
//! same "emit AABB-overlapping candidates" interface; swap it in during the GPU
//! perf pass. Correctness here is checked against brute force (must be a
//! superset of every truly-close pair).

use glam::Vec3;
use std::collections::HashMap;

type Cell = (i32, i32, i32);

/// Grid map: cell key → the `(start, len)` range of that cell's primitives in
/// [`CellGrid::items`]. Every key here is three small cell indices, so `std`'s
/// DoS-resistant SipHash is pure overhead — this is not attacker-controlled input. Measured with `sample` on the male proxy at 15 050 static triangles,
/// `HashMap::get` alone accounted for **1844 of 10 130** samples of a step (18%).
///
/// [`FxHasher`] is the multiply-xor hash rustc itself uses. It is written out rather than
/// taken as a dependency: `rustc-hash` is in the lock file only transitively, so making it
/// direct would touch `about.toml`, `deny.toml` and the licence allow-list for twelve lines
/// of arithmetic.
type Grid = HashMap<Cell, (u32, u32), std::hash::BuildHasherDefault<FxHasher>>;

/// Cell → primitive indices, with the indices in ONE flat buffer and the map holding a
/// `(start, len)` range into it — not a `Vec` per cell.
///
/// This is the same counting-sort-into-a-flat-buffer shape `csr::from_triplets` already
/// uses, and for the same reason: a `Vec` per cell is one allocation per occupied cell,
/// each doubling as it fills, and this grid is rebuilt on every Newton iteration AND every
/// CCD widening pass.
///
/// The flat buffer is still the right shape — one `Vec` per occupied cell would be an
/// allocation per cell on a grid rebuilt every Newton iteration. But the REASON this comment
/// used to give for it has been measured and is **false**, so it is recorded here rather than
/// left to be rediscovered:
///
/// > "Peak LIVE bytes did not predict the footprint; the allocation COUNT did."
///
/// Allocation count does **not** predict the footprint. Measured on the same scene the claim
/// was made about — 5 041 cloth vertices over the male COL proxy, 12 steps, `memprobe` under
/// `/usr/bin/time -l`, the two runs differing only in whether
/// [`crate::contact::rayleigh_stiffness`] puts its 48-byte `w` buffer on the stack:
///
/// | | allocations | peak live | max RSS |
/// |---|---|---|---|
/// | before | 2 350 318 | 834.2 MB | 1 116 700 672 B |
/// | after | 8 693 | 834.2 MB | 1 119 076 352 B |
///
/// **270x fewer allocations moved max RSS by +0.2 % and the wall clock by +1.0 %**, with a
/// bit-identical position fingerprint (`d3e5bf44d601fba3`) proving nothing else changed. At
/// 10 000 cloth vertices it is 732x fewer for −0.14 % RSS. The old 8.4x RSS-over-peak-live
/// retention figure is also gone: it now reads 1.28x here and 1.05x at 10 000 vertices, so
/// the libmalloc-retention story those numbers told no longer describes this tree.
///
/// What peak bytes are actually made of is in `implicit::assemble`'s `asm:stencils-done`
/// checkpoint. Reach for churned BYTES, not count, when this comes up again.
///
/// Bucket CONTENTS are unchanged: within a cell, indices land in insertion order exactly as
/// `Vec::push` left them, so every consumer sees the same slice and the pair lists are
/// bit-identical. The map's iteration order decides only WHERE in `items` a bucket sits.
struct CellGrid {
    at: Grid,
    items: Vec<u32>,
}

impl CellGrid {
    /// `ins` is the full `(cell, primitive)` insertion list. Staging it costs one 8-byte-key
    /// allocation that grows by doubling; the alternative is running the caller's cell loops
    /// twice, which is three copies of the same loop for no memory saved.
    fn build(ins: &[(Cell, u32)]) -> Self {
        let mut at = Grid::default();
        for (c, _) in ins {
            at.entry(*c).or_insert((0, 0)).1 += 1;
        }
        let mut acc = 0u32;
        for v in at.values_mut() {
            v.0 = acc;
            acc += v.1;
            v.1 = 0; // reused as the scatter cursor, and ends equal to the length again
        }
        let mut items = vec![0u32; ins.len()];
        for &(c, i) in ins {
            let e = at.get_mut(&c).expect("counted in pass 1");
            items[(e.0 + e.1) as usize] = i;
            e.1 += 1;
        }
        Self { at, items }
    }

    #[inline]
    fn get(&self, c: &Cell) -> &[u32] {
        match self.at.get(c) {
            Some(&(s, l)) => &self.items[s as usize..(s + l) as usize],
            None => &[],
        }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn cells(&self) -> usize {
        self.at.len()
    }

    fn biggest(&self) -> usize {
        self.at.values().map(|v| v.1 as usize).max().unwrap_or(0)
    }
}

/// The rustc/Firefox multiply-xor hash. `write_i32` is the path a `(i32, i32, i32)` key
/// actually takes; the byte-wise `write` is the correctness fallback for any other key,
/// so this is a valid `Hasher` regardless of what is hashed through it.
#[derive(Default)]
struct FxHasher(u64);

impl FxHasher {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(Self::K);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.add(i as u32 as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// The STATIC MESH COLLIDER (`memory/DECISIONS.md` D-17), in one predicate.
///
/// Every `movable: &[bool]` parameter below is the per-vertex "this vertex can move"
/// mask (`inv_mass > 0`). A candidate pair whose participants are **all** immovable is
/// **inert**, in both consumers of this module:
///
/// * In the Newton assembly, all of the stencil's rows and columns are dropped by the
///   Dirichlet filter for pinned DOFs (`implicit.rs`, "Dirichlet filter for pinned
///   DOFs"), so its barrier and friction blocks produce **exactly zero** displacement.
/// * In the CCD line search, every one of the stencil's displacements is exactly zero
///   (pinned DOFs get `Δx = 0`), so `accd_toi` cannot return anything but the full step.
///
/// So dropping such a pair is not an approximation — the result is unchanged. And it is
/// the ENTIRE cost of putting a body mesh in the scene: the avatar's collision proxy
/// arrives as a static object, and without this filter its own 15 000 triangles are
/// tested against each other as if they could collide. Measured, one 36-vertex cloth
/// patch on the male proxy, dt = 1/120: **9.724 → 0.235 s/step**, against 0.068 for the
/// same scene with self-collision switched off entirely.
///
/// This is deliberately a MOVABILITY filter and not an object-id filter. Movability is
/// the property that actually makes a pair inert, it is already in `SimState::inv_mass`
/// so nothing has to be plumbed, and it is strictly more correct: two *distinct* static
/// objects must not be tested against each other either, which an object-id filter
/// would still do.
///
/// An EMPTY mask means "everything is movable", i.e. no filtering. That keeps direct
/// callers and the unit tests below honest about the un-filtered behaviour.
#[inline]
fn any_movable(movable: &[bool], verts: &[u32]) -> bool {
    movable.is_empty() || verts.iter().any(|&v| movable[v as usize])
}

/// Candidate vertex–triangle pairs within (roughly) `dhat`. Triangles that share
/// any vertex with `excluded[v]` (the vertex's 1-ring, incl. itself) are dropped
/// so flat neighbors don't register as contacts. Deterministic (sorted output).
/// Grid cell size for the broad phase.
///
/// The cell must be at least the SEARCH radius *and* at least the typical primitive
/// extent. Using the search radius alone — which every function here used to do — is
/// catastrophic whenever primitives are much larger than the radius: a primitive's
/// search-expanded AABB then spans `extent/search` cells per axis, so insertion cost grows
/// CUBICALLY in that ratio. Measured on the gauge's `domino` (mean edge 0.153, `dhat`
/// 1e-3, so ~150 cells per axis ⇒ millions of cell insertions per edge): one step of a
/// 512-vertex scene took ~29 s, with `edge_edge_pairs` accounting for 3352 of 3353
/// profiler samples.
///
/// Enlarging the cell is always SAFE. Each primitive is inserted into every cell its
/// expanded AABB covers, so two primitives within `search` have overlapping expanded
/// AABBs; a non-empty overlap region intersects at least one cell, and that cell holds
/// both — for ANY cell size. A larger cell only admits extra candidates, which the exact
/// narrow-phase test then rejects; it can never drop a true pair.
/// `MGUSO_BP_STATS=1` prints per-call broadphase shape (cell size, occupied cells, biggest
/// bucket, raw vs deduped pair count). Read ONCE — this is called from the hot path, and
/// `std::env::var` takes a lock.
///
/// This is the diagnostic that located B-1: on `domino` the CCD call builds a 2.0-unit cell
/// (because its radius is `dhat + 2*max_disp` and `max_dx` is 1.0 world unit), which puts all
/// 1344 edges in ONE bucket and emits **17.3 M raw pairs** against the barrier call's 14.6 K.
fn bp_stats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MGUSO_BP_STATS").is_ok())
}

fn grid_cell(search: f32, typical_extent: f32) -> f32 {
    search.max(typical_extent).max(1e-6)
}

/// Mean edge length, the characteristic extent for edge primitives.
fn mean_edge_len(positions: &[Vec3], edges: &[[u32; 2]]) -> f32 {
    if edges.is_empty() {
        return 0.0;
    }
    let sum: f32 = edges
        .iter()
        .map(|e| positions[e[0] as usize].distance(positions[e[1] as usize]))
        .sum();
    sum / edges.len() as f32
}

/// Mean triangle AABB extent (largest axis), the characteristic extent for triangles.
fn mean_tri_extent(positions: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    if tris.is_empty() {
        return 0.0;
    }
    let sum: f32 = tris
        .iter()
        .map(|t| {
            let (a, b, c) = (
                positions[t[0] as usize],
                positions[t[1] as usize],
                positions[t[2] as usize],
            );
            let d = a.max(b).max(c) - a.min(b).min(c);
            d.x.max(d.y).max(d.z)
        })
        .sum();
    sum / tris.len() as f32
}

/// `movable`: per-vertex `inv_mass > 0` mask; empty ⇒ unfiltered. See [`any_movable`].
///
/// Sorted and deduplicated, which costs the whole pair list in memory — see
/// [`for_each_vertex_triangle_pair`] for the streaming form the CCD line search uses and
/// why.
pub fn vertex_triangle_pairs(
    positions: &[Vec3],
    tris: &[[u32; 3]],
    dhat: f32,
    excluded: &[Vec<u32>],
    movable: &[bool],
) -> Vec<(u32, u32)> {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for_each_vertex_triangle_pair(positions, tris, dhat, excluded, movable, |v, t| {
        pairs.push((v, t))
    });
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Every candidate vertex–triangle pair, handed to `visit` as it is found, possibly more
/// than once. The primitive behind [`vertex_triangle_pairs`]; see
/// [`for_each_edge_edge_pair`] for the measurement that motivates streaming.
pub fn for_each_vertex_triangle_pair(
    positions: &[Vec3],
    tris: &[[u32; 3]],
    dhat: f32,
    excluded: &[Vec<u32>],
    movable: &[bool],
    mut visit: impl FnMut(u32, u32),
) {
    if positions.is_empty() || tris.is_empty() {
        return;
    }
    let cell = grid_cell(dhat, mean_tri_extent(positions, tris));
    let inv = 1.0 / cell;
    let key = |p: Vec3| -> (i32, i32, i32) {
        (
            (p.x * inv).floor() as i32,
            (p.y * inv).floor() as i32,
            (p.z * inv).floor() as i32,
        )
    };

    // Hash vertices into cells. One cell per vertex, so the insertion list is exactly
    // `positions.len()` long.
    let mut ins: Vec<(Cell, u32)> = Vec::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        ins.push((key(*p), i as u32));
    }
    let grid = CellGrid::build(&ins);
    drop(ins);

    for (ti, t) in tris.iter().enumerate() {
        let (pa, pb, pc) = (
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        let lo = pa.min(pb).min(pc) - Vec3::splat(dhat);
        let hi = pa.max(pb).max(pc) + Vec3::splat(dhat);
        // Hoisted out of the cell loops: if the triangle can move, every candidate
        // vertex qualifies and the per-candidate mask test is skipped entirely. When it
        // cannot, only a movable vertex can make the pair non-inert.
        let tri_movable = any_movable(movable, t);
        let (kl, kh) = (key(lo), key(hi));
        for cx in kl.0..=kh.0 {
            for cy in kl.1..=kh.1 {
                for cz in kl.2..=kh.2 {
                    for &v in grid.get(&(cx, cy, cz)) {
                        // Cheapest rejection first — this is the D-17 hot path.
                        if !tri_movable && !movable[v as usize] {
                            continue;
                        }
                        if excluded[v as usize].iter().any(|&e| t.contains(&e)) {
                            continue;
                        }
                        visit(v, ti as u32);
                    }
                }
            }
        }
    }
}

/// Unique undirected mesh edges (each triangle contributes its 3 edges, dedup'd).
pub fn unique_edges(tris: &[[u32; 3]]) -> Vec<[u32; 2]> {
    let mut set: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for t in tris {
        for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
            let (i, j) = (t[a], t[b]);
            set.insert((i.min(j), i.max(j)));
        }
    }
    let mut e: Vec<[u32; 2]> = set.into_iter().map(|(a, b)| [a, b]).collect();
    e.sort_unstable(); // deterministic order
    e
}

/// Candidate edge–edge pairs within `search` distance (spatial hash on edge
/// AABBs). Returns index pairs `(ei, ej)` with `ei < ej`, skipping edges that
/// share a vertex (adjacent → can't cross, and the distance degenerates).
///
/// `movable`: per-vertex `inv_mass > 0` mask; empty ⇒ unfiltered. See [`any_movable`].
///
/// **Sorted and deduplicated, and that costs the whole pair list in memory.** Use it from
/// the barrier assembly, which needs each stencil EXACTLY once (a duplicate would
/// double-count the barrier's Hessian block). Use [`for_each_edge_edge_pair`] from the CCD
/// line search, which only takes a `min` and is therefore indifferent to duplicates — see
/// the measurement on that function.
pub fn edge_edge_pairs(
    positions: &[Vec3],
    edges: &[[u32; 2]],
    search: f32,
    movable: &[bool],
) -> Vec<(u32, u32)> {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for_each_edge_edge_pair(positions, edges, search, movable, |a, b| pairs.push((a, b)));
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Every candidate edge–edge pair, handed to `visit` as it is found — **`(ei, ej)` with
/// `ei < ej`, and possibly more than once**.
///
/// This is the primitive; [`edge_edge_pairs`] is this function plus a sort and a dedup. The
/// difference is the entire footprint of the CCD line search. Measured, 20 164 cloth
/// vertices over the male COL proxy with `MGUSO_BP_STATS=1`, the three widening passes of
/// ONE Newton iteration:
///
/// | `allowed` pass | cell | cells | biggest bucket | raw pairs | deduped |
/// |---|---|---|---|---|---|
/// | 1 | 0.01340 | 10 716 | 33 | 6 533 443 | 3 233 750 |
/// | 2 | 0.02126 | 2 862 | 65 | 13 397 243 | 9 907 192 |
/// | 3 | 0.05506 | 382 | **294** | **63 252 345** | 56 469 056 |
///
/// The widening quadruples `allowed`, `grid_cell` is floored to `search`, so the cell
/// outgrows the mesh spacing and the pair count goes quadratic. At 8 B per pair the third
/// pass alone is 483 MB raw plus 431 MB of deduped result held while the caller iterates,
/// per Newton iteration — and that is what put peak LIVE bytes at **4 153.2 MB** and
/// `/usr/bin/time -l` maximum RSS at 4 794 105 856 B for a 12-step drape, over the 4 GB
/// ceiling. Streaming makes the CCD's broadphase footprint O(cells + insertions) instead of
/// O(pairs).
///
/// Duplicates are not a defect for the CCD consumer: it computes `t = min(t, toi(pair))`,
/// which is idempotent. The measured duplicate rate is ~10% on the widest pass
/// (63 252 345 → 56 469 056), i.e. 10% more narrow-phase calls in exchange for ~900 MB.
pub fn for_each_edge_edge_pair(
    positions: &[Vec3],
    edges: &[[u32; 2]],
    search: f32,
    movable: &[bool],
    mut visit: impl FnMut(u32, u32),
) {
    if edges.len() < 2 {
        return;
    }
    // Per-edge movability, so the bucket loop tests a bool instead of two vertices.
    let em: Vec<bool> = if movable.is_empty() {
        Vec::new()
    } else {
        edges.iter().map(|e| any_movable(movable, e)).collect()
    };
    let e_movable = |i: u32| em.is_empty() || em[i as usize];
    let cell = grid_cell(search, mean_edge_len(positions, edges));
    let inv = 1.0 / cell;
    let key = |v: f32| -> i32 { (v * inv).floor() as i32 };

    // The grid is INDEXED BY THE MOVABLE EDGES ONLY, and then every edge queries it.
    //
    // A pair is non-inert iff at least one of its edges can move (see `any_movable`).
    // That edge is in the grid; the other one queries. So no non-inert pair is missed,
    // and with an empty mask every edge is both indexed and querying — exactly the
    // original all-pairs relation. What changes is that **the grid is sized by the cloth
    // rather than by the scene**. Measured on the male proxy with `MGUSO_BP_STATS=1`, the
    // old grid inserted 25 409 edges over ~16 000 cells — that many `Vec` allocations per
    // call, rebuilt on every Newton iteration and every CCD widening pass, for body
    // geometry that never moves. `sample` put `HashMap::get` at 18% of a step and
    // `edge_edge_pairs` at another 12%.
    //
    // Expansion by `search` belongs on ONE side only — the querying side. It used to be
    // applied to both, which catches pairs up to `2*search` apart and cubes the cells each
    // edge occupies. One-sided is still a superset, and that is provable: if `dist(A,B) <=
    // search` then the closest point `p` on A satisfies `p ∈ aabb(A)` and, being within
    // `search` of B on every axis, also `p ∈ aabb(B)+search`. Both cell ranges cover every
    // cell their box touches, so `cell(p)` is in both — A's unpadded range and B's padded
    // one. This is also the shape `vertex_triangle_pairs` already has (vertices indexed
    // unpadded, the triangle's AABB padded on query), so the two agree now.
    //
    // Measured, K=142 sheet over the male proxy, the widest CCD call of one step:
    // 2 078 712 cell-insertions -> 114 454, biggest bucket 177 -> 37, 127 125 006 raw pairs
    // -> 6 973 780 (18.2x), and `/usr/bin/time -l` maximum RSS for the run
    // 2 507 751 424 -> 928 743 424 B (2.70x), 48.46 -> 41.60 s. The raw-pair `Vec` at
    // 8 B/pair plus its doubling realloc WAS the excess.
    //
    // The 2.70x RATIO is what that A/B established and it still stands. The ABSOLUTE
    // 928 743 424 B does NOT: re-measured 2026-08-05, the same K=142 12-step run reads
    // **4 539 121 664 B** max RSS and 12 124.4 MB peak live. Do not project a memory budget
    // from the old figure — one agent did, and launched a run that breached the 4 GB cap.
    // `edge_edge_pairs_is_a_superset_of_brute_force` pins the contract and
    // `edge_edge_candidate_count_stays_bounded` pins the count.
    //
    // The cell range of edge `e`, padded by `pad`, as inclusive index bounds.
    let cells = |e: &[u32; 2], pad: f32| {
        let (p, q) = (positions[e[0] as usize], positions[e[1] as usize]);
        let lo = p.min(q) - Vec3::splat(pad);
        let hi = p.max(q) + Vec3::splat(pad);
        (
            [key(lo.x), key(lo.y), key(lo.z)],
            [key(hi.x), key(hi.y), key(hi.z)],
        )
    };

    let mut ins: Vec<(Cell, u32)> = Vec::new();
    for (ei, e) in edges.iter().enumerate() {
        if !e_movable(ei as u32) {
            continue;
        }
        let (lo, hi) = cells(e, 0.0);
        for cx in lo[0]..=hi[0] {
            for cy in lo[1]..=hi[1] {
                for cz in lo[2]..=hi[2] {
                    ins.push(((cx, cy, cz), ei as u32));
                }
            }
        }
    }
    if ins.is_empty() {
        return; // nothing can move ⇒ every pair is inert
    }
    let grid = CellGrid::build(&ins);
    drop(ins);

    // Pair edges that co-occur in any cell; skip shared-vertex edges.
    let mut raw = 0usize;
    for (j, eb) in edges.iter().enumerate() {
        let ej = j as u32;
        let (lo, hi) = cells(eb, search);
        for cx in lo[0]..=hi[0] {
            for cy in lo[1]..=hi[1] {
                for cz in lo[2]..=hi[2] {
                    for &ei in grid.get(&(cx, cy, cz)) {
                        // Emit each unordered pair ONCE rather than leaning on the dedup
                        // to halve it: when both edges are movable both are indexed, so
                        // both directions fire and `ei < ej` picks one. When only `ei` is
                        // movable this is the only direction there is, so it must not be
                        // gated on the ordering.
                        if ei == ej || (ei > ej && e_movable(ej)) {
                            continue;
                        }
                        let ea = edges[ei as usize];
                        if ea[0] == eb[0] || ea[0] == eb[1] || ea[1] == eb[0] || ea[1] == eb[1] {
                            continue; // shared vertex
                        }
                        raw += 1;
                        visit(ei.min(ej), ei.max(ej));
                    }
                }
            }
        }
    }
    if bp_stats() {
        // `indexed` is the MOVABLE edge count — the grid is sized by those, while all
        // `edges.len()` of them query it. Reporting both is the point: a scene whose
        // indexed count is a rounding error against its total is one where the static
        // mesh collider is doing its job.
        eprintln!(
            "bp ee: {} edges ({} cell-insertions from movable edges), \
             cell {cell:.5}, {} cells, biggest bucket {}, {raw} raw pairs",
            edges.len(),
            grid.items.len(),
            grid.cells(),
            grid.biggest(),
        );
    }
}

/// Candidate edge–triangle pairs within `search` (spatial hash on edge AABBs,
/// queried per triangle). Returns `(edge_index, triangle_index)`, skipping pairs
/// that share any vertex (adjacent → can't pierce). For the post-step
/// intersection check ([`crate::contact::edge_triangle_intersect`]).
pub fn edge_triangle_pairs(
    positions: &[Vec3],
    edges: &[[u32; 2]],
    tris: &[[u32; 3]],
    search: f32,
) -> Vec<(u32, u32)> {
    if edges.is_empty() || tris.is_empty() {
        return Vec::new();
    }
    let cell = grid_cell(
        search,
        mean_edge_len(positions, edges).max(mean_tri_extent(positions, tris)),
    );
    let inv = 1.0 / cell;
    let key = |v: f32| -> i32 { (v * inv).floor() as i32 };

    // Hash each edge into every cell its (search-expanded) AABB covers.
    let mut ins: Vec<(Cell, u32)> = Vec::with_capacity(edges.len());
    for (ei, e) in edges.iter().enumerate() {
        let (p, q) = (positions[e[0] as usize], positions[e[1] as usize]);
        let lo = p.min(q) - Vec3::splat(search);
        let hi = p.max(q) + Vec3::splat(search);
        for cx in key(lo.x)..=key(hi.x) {
            for cy in key(lo.y)..=key(hi.y) {
                for cz in key(lo.z)..=key(hi.z) {
                    ins.push(((cx, cy, cz), ei as u32));
                }
            }
        }
    }
    let grid = CellGrid::build(&ins);
    drop(ins);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for (ti, t) in tris.iter().enumerate() {
        let (pa, pb, pc) = (
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        let lo = pa.min(pb).min(pc) - Vec3::splat(search);
        let hi = pa.max(pb).max(pc) + Vec3::splat(search);
        for cx in key(lo.x)..=key(hi.x) {
            for cy in key(lo.y)..=key(hi.y) {
                for cz in key(lo.z)..=key(hi.z) {
                    for &ei in grid.get(&(cx, cy, cz)) {
                        let e = edges[ei as usize];
                        if t.contains(&e[0]) || t.contains(&e[1]) {
                            continue; // shared vertex → adjacent, skip
                        }
                        pairs.push((ei, ti as u32));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Stream candidate edge–triangle pairs to `hit`, stopping at the first `true`.
///
/// Same candidate set as [`edge_triangle_pairs`], but it never materialises the pair list.
/// That list is the single largest allocation in the whole solver: on `fishingknot`
/// (56,464 edges x 36,864 triangles) the post-step self-intersection check was allocating
/// **~718 MB** of transient pairs — measured, it took peak live from 338.8 MB to 1057.2 MB in
/// one call — purely to answer a yes/no question that short-circuits on the first hit.
///
/// Returns `true` iff `hit` returned `true` for some candidate. Duplicates are possible
/// (an edge can share several cells with a triangle), which is harmless for a predicate:
/// `edge_triangle_pairs` sorts and dedups only because it returns a list.
///
/// `movable`: per-vertex `inv_mass > 0` mask; empty ⇒ unfiltered.
///
/// Unlike the force and CCD queries, skipping all-immovable pairs here is a deliberate
/// SEMANTIC choice, not a provable no-op: it means an intersection baked into static
/// input geometry (a body proxy whose own mesh self-intersects) no longer reports the
/// invariant violated. That is the wanted reading — the invariant asks whether the
/// SOLVER created a penetration, and no solver step can undo a pierce between two
/// vertices it is forbidden to move. It is also required for cost: a 15 000-triangle
/// static proxy would otherwise run 22 000 edges against 15 000 triangles every step.
pub fn any_edge_triangle_pair(
    positions: &[Vec3],
    edges: &[[u32; 2]],
    tris: &[[u32; 3]],
    search: f32,
    movable: &[bool],
    mut hit: impl FnMut(u32, u32) -> bool,
) -> bool {
    if edges.is_empty() || tris.is_empty() {
        return false;
    }
    let cell = grid_cell(
        search,
        mean_edge_len(positions, edges).max(mean_tri_extent(positions, tris)),
    );
    let inv = 1.0 / cell;
    let key = |v: f32| -> i32 { (v * inv).floor() as i32 };

    // ponytail: both grids here still pad the INDEXING side by `search`, which
    // `edge_edge_pairs` no longer does — see the proof there that one-sided is still a
    // superset. Not changed with it because this function materialises nothing, so it is a
    // TIME cost and not a memory one, and it is the pair the
    // `streaming_offers_the_same_candidate_set_as_the_materialised_list` test pins against
    // `edge_triangle_pairs`, so the two must move together. Worth doing if the post-step
    // invariant check shows up in a profile again (it was 23% of a step once); measure first.
    let edge_cells = |e: &[u32; 2]| {
        let (p, q) = (positions[e[0] as usize], positions[e[1] as usize]);
        let lo = p.min(q) - Vec3::splat(search);
        let hi = p.max(q) + Vec3::splat(search);
        (
            [key(lo.x), key(lo.y), key(lo.z)],
            [key(hi.x), key(hi.y), key(hi.z)],
        )
    };
    let tri_cells = |t: &[u32; 3]| {
        let (pa, pb, pc) = (
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        let lo = pa.min(pb).min(pc) - Vec3::splat(search);
        let hi = pa.max(pb).max(pc) + Vec3::splat(search);
        (
            [key(lo.x), key(lo.y), key(lo.z)],
            [key(hi.x), key(hi.y), key(hi.z)],
        )
    };

    // TWO passes, each indexing only the MOVABLE primitives of one kind and scanning all
    // of the other. A pair is non-inert iff its edge or its triangle can move, so pass A
    // (movable edges indexed, every triangle scanned) and pass B (movable triangles
    // indexed, every edge scanned) together cover every non-inert pair. A pair with both
    // sides movable is offered twice, which is harmless — this is a short-circuiting
    // predicate and the doc above already permits duplicates.
    //
    // The point is that BOTH grids are sized by the movable geometry. This function was
    // the single largest item in the step profile (`sample`, male proxy: 2313 of 10 130
    // samples, 23%), essentially all of it spent hashing 25 409 never-moving body edges
    // into ~16 000 freshly allocated buckets once per step to answer a diagnostic.
    let mut ins: Vec<(Cell, u32)> = Vec::new();
    for (ei, e) in edges.iter().enumerate() {
        if !any_movable(movable, e) {
            continue;
        }
        let (lo, hi) = edge_cells(e);
        for cx in lo[0]..=hi[0] {
            for cy in lo[1]..=hi[1] {
                for cz in lo[2]..=hi[2] {
                    ins.push(((cx, cy, cz), ei as u32));
                }
            }
        }
    }
    let grid = CellGrid::build(&ins);
    drop(ins);
    if !grid.is_empty() {
        for (ti, t) in tris.iter().enumerate() {
            let (lo, hi) = tri_cells(t);
            for cx in lo[0]..=hi[0] {
                for cy in lo[1]..=hi[1] {
                    for cz in lo[2]..=hi[2] {
                        for &ei in grid.get(&(cx, cy, cz)) {
                            let e = edges[ei as usize];
                            if t.contains(&e[0]) || t.contains(&e[1]) {
                                continue; // shared vertex → adjacent, skip
                            }
                            if hit(ei, ti as u32) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Pass B: movable triangles indexed, every edge scanned. Skipped entirely when the
    // mask is empty — then pass A already indexed every edge and saw every triangle, so
    // this would only re-offer the same pairs.
    if movable.is_empty() {
        return false;
    }
    let mut ins: Vec<(Cell, u32)> = Vec::new();
    for (ti, t) in tris.iter().enumerate() {
        if !any_movable(movable, t) {
            continue;
        }
        let (lo, hi) = tri_cells(t);
        for cx in lo[0]..=hi[0] {
            for cy in lo[1]..=hi[1] {
                for cz in lo[2]..=hi[2] {
                    ins.push(((cx, cy, cz), ti as u32));
                }
            }
        }
    }
    let tgrid = CellGrid::build(&ins);
    drop(ins);
    if tgrid.is_empty() {
        return false;
    }
    for (ei, e) in edges.iter().enumerate() {
        // Already covered by pass A, which offered this edge against every triangle.
        if any_movable(movable, e) {
            continue;
        }
        let (lo, hi) = edge_cells(e);
        for cx in lo[0]..=hi[0] {
            for cy in lo[1]..=hi[1] {
                for cz in lo[2]..=hi[2] {
                    for &ti in tgrid.get(&(cx, cy, cz)) {
                        let t = tris[ti as usize];
                        if t.contains(&e[0]) || t.contains(&e[1]) {
                            continue; // shared vertex → adjacent, skip
                        }
                        if hit(ei as u32, ti) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// The 1-ring exclusion sets (each vertex + its edge-neighbors) for a mesh.
pub fn one_ring_exclusion(vertex_count: usize, tris: &[[u32; 3]]) -> Vec<Vec<u32>> {
    let mut ex = vec![Vec::new(); vertex_count];
    for (i, e) in ex.iter_mut().enumerate() {
        e.push(i as u32);
    }
    for t in tris {
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    ex[t[a] as usize].push(t[b]);
                }
            }
        }
    }
    for e in ex.iter_mut() {
        e.sort_unstable();
        e.dedup();
    }
    ex
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::point_triangle_distance;

    /// **The footprint guard for the CCD broad phase.** Streaming candidates must cost
    /// memory in the number of PRIMITIVES; collecting them costs memory in the number of
    /// PAIRS, and the two diverge quadratically.
    ///
    /// This is the regression that put a 12-step drape of 20 164 cloth vertices at peak live
    /// 4 153.2 MB and `/usr/bin/time -l` maximum RSS 4 794 105 856 B, over the 4 GB ceiling.
    /// `ccd_max_step` widens its search radius by 4x per pass until it covers the step, and
    /// `grid_cell` is floored to that radius, so the cell outgrows the mesh spacing and the
    /// candidate count explodes: measured with `MGUSO_BP_STATS=1`, the three passes of one
    /// Newton iteration emitted 6 533 443 / 13 397 243 / **63 252 345** raw pairs.
    ///
    /// The fixture reproduces that regime deliberately — `search` is 8x the point spacing,
    /// which is what a widened CCD pass looks like. It asserts the RATIO, not an absolute
    /// byte count, so it is machine independent.
    #[test]
    fn streaming_candidates_do_not_allocate_per_pair() {
        // A dense 12^3 lattice of points wired into edges: many primitives in few cells once
        // `search` is large, which is exactly the widened-CCD regime.
        let n = 12usize;
        let mut positions = Vec::new();
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.01, j as f32 * 0.01, k as f32 * 0.01));
                }
            }
        }
        let mut edges: Vec<[u32; 2]> = Vec::new();
        for a in 0..positions.len() as u32 {
            // A short chain per point, so no two edges of a pair share a vertex by accident.
            if (a as usize) + n * n < positions.len() {
                edges.push([a, a + (n * n) as u32]);
            }
        }
        let search = 0.08; // 8x the spacing — a widened CCD pass

        let mut streamed_count = 0usize;
        crate::alloc_stats::reset_thread_peak();
        let base = crate::alloc_stats::thread_peak_bytes();
        for_each_edge_edge_pair(&positions, &edges, search, &[], |_, _| streamed_count += 1);
        let streamed = crate::alloc_stats::thread_peak_bytes() - base;

        crate::alloc_stats::reset_thread_peak();
        let base = crate::alloc_stats::thread_peak_bytes();
        let collected_pairs = edge_edge_pairs(&positions, &edges, search, &[]);
        let collected = crate::alloc_stats::thread_peak_bytes() - base;

        assert!(
            streamed_count > 200_000,
            "fixture is too small to discriminate: only {streamed_count} candidates"
        );
        assert!(
            collected >= streamed_count * std::mem::size_of::<(u32, u32)>(),
            "the tracking allocator is not feeding alloc_stats — the guard is vacuous \
             (collected {collected} B for {streamed_count} pairs)"
        );
        // Streaming holds the grid only: `items` (4 B) plus the staged insertion list (16 B)
        // per cell-insertion, and insertions are bounded by primitives x cells-per-AABB, not
        // by pairs. Measured on the fixture as it stands: 40 024 B streamed against
        // 16 784 472 B collected over 1 324 656 candidates, a 419x ratio.
        //
        // WHAT THIS DOES NOT GUARD, stated because it was mistaken for a guard that does:
        // it pins the PRIMITIVE, not that `implicit.rs`'s CCD line search calls it.
        // Reverting those two call sites to `edge_edge_pairs`/`vertex_triangle_pairs` costs
        // 1.5 GB of max RSS at 10 000 cloth vertices and both this test and
        // `implicit::tests::one_step_footprint_is_bounded` still pass.
        //
        // There is no cheap test that would catch it, and that is a property of the failure
        // and not an omission: the collected list is O(pairs), which is O(vertices) on any
        // scene small enough for a unit test. It only dominates once the CCD's widening
        // passes go quadratic — measured 6.5M -> 13.4M -> 63.3M raw pairs across ONE Newton
        // iteration's three passes at 20 164 cloth vertices, because `allowed` quadruples per
        // pass while `grid_cell` is floored to `search`, so the cell outgrows the mesh
        // spacing. Reproducing that costs GBs and minutes. It is tracked in memory/todo.md as
        // a scaling check for the perf harness, not as a unit test.
        assert!(
            streamed * 8 < collected,
            "streaming the {streamed_count} candidates peaked at {streamed} B against \
             {collected} B for collecting them ({:.1}x) — the candidate list is being \
             materialised again",
            collected as f64 / streamed.max(1) as f64
        );
        drop(collected_pairs);
    }

    #[test]
    fn hash_is_superset_of_brute_force() {
        // Two separate horizontal layers, close together (within dhat in places).
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        let n = 5;
        for (layer, y) in [(0usize, 0.0f32), (1, 0.03)] {
            let base = positions.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.1, y, j as f32 * 0.1));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = base + (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([r, v + 1, r + 1]);
                }
            }
            let _ = layer;
        }
        let dhat = 0.05;
        let excluded = one_ring_exclusion(positions.len(), &tris);
        let pairs = vertex_triangle_pairs(&positions, &tris, dhat, &excluded, &[]);
        let set: std::collections::HashSet<_> = pairs.iter().copied().collect();

        // Brute force: every (v,tri) with true distance < dhat and not excluded
        // MUST appear in the broad-phase output.
        for (vi, p) in positions.iter().enumerate() {
            for (ti, t) in tris.iter().enumerate() {
                if excluded[vi].iter().any(|&e| t.contains(&e)) {
                    continue;
                }
                let d = point_triangle_distance(
                    *p,
                    positions[t[0] as usize],
                    positions[t[1] as usize],
                    positions[t[2] as usize],
                );
                if d < dhat {
                    assert!(
                        set.contains(&(vi as u32, ti as u32)),
                        "broad phase missed close pair (v={vi}, t={ti}, d={d})"
                    );
                }
            }
        }
        // Cross-layer pairs exist (layers are 0.03 apart, dhat 0.05).
        assert!(!pairs.is_empty(), "expected candidate pairs between layers");
    }

    /// Two grid layers plus a `search` several times the mean edge length — i.e. the shape
    /// of the CCD call, whose radius is `dhat + 2*max_disp` and which is the one that
    /// actually blows up.
    fn two_layer_grid(n: usize, spacing: f32, gap: f32) -> (Vec<Vec3>, Vec<[u32; 2]>) {
        let mut pos = Vec::new();
        let mut tris = Vec::new();
        for y in [0.0f32, gap] {
            let base = pos.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    pos.push(Vec3::new(i as f32 * spacing, y, j as f32 * spacing));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = base + (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([v + 1, r + 1, r]);
                }
            }
        }
        (pos.clone(), unique_edges(&tris))
    }

    /// THE CONTRACT of `edge_edge_pairs`: a superset of every truly-close pair. It was
    /// untested, which is how an AABB pad applied to BOTH the indexing and the querying
    /// side survived — that catches pairs up to `2*search` apart and cubes the cells each
    /// edge occupies, for candidates the narrow phase then throws away.
    #[test]
    fn edge_edge_pairs_is_a_superset_of_brute_force() {
        for &(n, spacing, gap, search) in &[
            (12usize, 0.02f32, 0.03f32, 0.10f32), // search 5x the edge length (CCD shape)
            (12, 0.02, 0.03, 0.02),               // search == the edge length (barrier shape)
            (18, 0.05, 0.01, 0.02),               // search < the edge length
        ] {
            let (pos, edges) = two_layer_grid(n, spacing, gap);
            let pairs = edge_edge_pairs(&pos, &edges, search, &[]);
            let set: std::collections::HashSet<_> = pairs.iter().copied().collect();
            let mut true_close = 0usize;
            for (i, a) in edges.iter().enumerate() {
                for (j, b) in edges.iter().enumerate().skip(i + 1) {
                    if a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1] {
                        continue; // shared vertex — deliberately excluded
                    }
                    let d = crate::contact::segment_segment_distance(
                        pos[a[0] as usize],
                        pos[a[1] as usize],
                        pos[b[0] as usize],
                        pos[b[1] as usize],
                    );
                    if d <= search {
                        true_close += 1;
                        assert!(
                            set.contains(&(i as u32, j as u32)),
                            "n={n} search={search}: missed close pair ({i},{j}) d={d}"
                        );
                    }
                }
            }
            assert!(
                true_close > 100,
                "n={n} search={search}: only {true_close} true close pairs — not a real case"
            );
        }
    }

    /// The candidate COUNT is the memory ceiling — each pair is 8 B and the `Vec` doubles —
    /// so pin it as a number rather than trusting a comment. This fixture is deliberately
    /// the shape that made the two-sided indexing pad expensive: `search` ≈ the mean edge
    /// length ≈ the cell, over a mesh many cells across, so padding the indexing side too
    /// takes each edge from ~2 cells to 3x3x3 = 27. A fixture only a few cells wide shows
    /// nothing (measured: 292 569 -> 276 741, i.e. 5%).
    #[test]
    fn edge_edge_candidate_count_stays_bounded() {
        let (pos, edges) = two_layer_grid(30, 0.02, 0.01);
        assert_eq!(edges.len(), 5162, "fixture changed");
        // Two-sided indexing pad: 563 685. One-sided: this — 2.1x on a fixture with no
        // static geometry. The real win is larger because a body's edges still QUERY with
        // the pad while only the cloth is indexed, so shrinking the buckets compounds:
        // K=142 sheet over the male proxy, one CCD call, 127 125 006 -> 13 922 922.
        assert_eq!(edge_edge_pairs(&pos, &edges, 0.02, &[]).len(), 268075);
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    /// A cloth-like grid plus a second sheet stabbed through it, so the candidate set is
    /// non-trivial and the two code paths have something to disagree about.
    fn scene() -> (Vec<Vec3>, Vec<[u32; 2]>, Vec<[u32; 3]>) {
        let (nx, nz) = (7usize, 7usize);
        let mut pos = Vec::new();
        for j in 0..nz {
            for i in 0..nx {
                pos.push(Vec3::new(i as f32 * 0.1, 0.0, j as f32 * 0.1));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nz - 1 {
            for i in 0..nx - 1 {
                let v = (j * nx + i) as u32;
                let r = v + nx as u32;
                tris.push([v, v + 1, r]);
                tris.push([v + 1, r + 1, r]);
            }
        }
        // A vertical edge pair that pierces the sheet near its middle.
        let base = pos.len() as u32;
        pos.push(Vec3::new(0.32, -0.2, 0.31));
        pos.push(Vec3::new(0.32, 0.2, 0.31));
        let mut edges = unique_edges(&tris);
        edges.push([base, base + 1]);
        (pos, edges, tris)
    }

    /// `any_edge_triangle_pair` must offer exactly the candidates `edge_triangle_pairs`
    /// returns — that equivalence is the whole basis for replacing the materialised list with
    /// a stream in the self-intersection predicate.
    #[test]
    fn streaming_offers_the_same_candidate_set_as_the_materialised_list() {
        let (pos, edges, tris) = scene();
        for &search in &[0.05f32, 0.15, 0.4] {
            let want = edge_triangle_pairs(&pos, &edges, &tris, search);
            let mut got = Vec::new();
            let exhausted = any_edge_triangle_pair(&pos, &edges, &tris, search, &[], |ei, ti| {
                got.push((ei, ti));
                false // never stop: enumerate everything
            });
            assert!(
                !exhausted,
                "a never-matching predicate must exhaust the candidates"
            );
            got.sort_unstable();
            got.dedup();
            assert_eq!(
                got, want,
                "search {search}: streaming candidate set differs from the materialised one"
            );
            assert!(
                !want.is_empty(),
                "search {search}: the case must be non-trivial"
            );
        }
    }

    /// It must short-circuit: a predicate that matches immediately sees far fewer candidates
    /// than the full set. This is the property that makes the streaming version cheap as well
    /// as small.
    #[test]
    fn streaming_short_circuits_on_the_first_match() {
        let (pos, edges, tris) = scene();
        let total = edge_triangle_pairs(&pos, &edges, &tris, 0.4).len();
        let mut seen = 0usize;
        let found = any_edge_triangle_pair(&pos, &edges, &tris, 0.4, &[], |_, _| {
            seen += 1;
            true // match the very first candidate
        });
        assert!(found, "must report a match");
        assert_eq!(seen, 1, "must stop after the first match, saw {seen}");
        assert!(
            total > 10,
            "the full set should be much larger than 1, got {total}"
        );
    }

    /// And it must agree with the materialised path on the actual PREDICATE, not just the
    /// candidate set: a genuinely piercing edge is found, and the same scene with the spike
    /// pulled clear of the sheet is not.
    #[test]
    fn streaming_agrees_on_whether_an_edge_pierces() {
        let (mut pos, edges, tris) = scene();
        let pierce = |pos: &Vec<Vec3>| {
            any_edge_triangle_pair(pos, &edges, &tris, 0.15, &[], |ei, ti| {
                let e = edges[ei as usize];
                let t = tris[ti as usize];
                crate::contact::edge_triangle_intersect(
                    pos[e[0] as usize],
                    pos[e[1] as usize],
                    pos[t[0] as usize],
                    pos[t[1] as usize],
                    pos[t[2] as usize],
                )
            })
        };
        assert!(pierce(&pos), "the spike through the sheet must be detected");
        // Lift both spike ends above the sheet: no crossing any more.
        let n = pos.len();
        pos[n - 2].y = 0.2;
        pos[n - 1].y = 0.6;
        assert!(
            !pierce(&pos),
            "a spike clear of the sheet must not be detected"
        );
    }
}

/// The static mesh collider filter (D-17). Two properties, both asserted against the
/// UNFILTERED result of the very same query, so neither can pass by doing nothing:
///
/// 1. An all-movable mask changes nothing (the filter is inert when nothing is static).
/// 2. A partly-static mask removes **exactly** the pairs whose every participant is
///    immovable, and keeps every cloth-vs-body pair.
#[cfg(test)]
mod static_collider_tests {
    use super::*;

    /// Two stacked sheets 0.02 apart: the LOWER one is the "body" (static), the upper the
    /// "cloth". Close enough that the broad phase offers plenty of pairs of all three
    /// kinds — body-body, cloth-cloth and cloth-body.
    #[allow(clippy::type_complexity)]
    fn two_sheets() -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<[u32; 2]>, Vec<bool>) {
        let n = 6usize;
        let (mut pos, mut tris, mut movable) = (Vec::new(), Vec::new(), Vec::new());
        for (y, is_cloth) in [(0.0f32, false), (0.02, true)] {
            let base = pos.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    pos.push(Vec3::new(i as f32 * 0.05, y, j as f32 * 0.05));
                    movable.push(is_cloth);
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = base + (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([v + 1, r + 1, r]);
                }
            }
        }
        // One PINNED vertex inside the cloth sheet, so the scene contains MIXED primitives
        // (a triangle with two movable corners and one immovable one). Without this the
        // scene cannot tell `any_movable` from an `all`-quantified version of itself, and a
        // filter that required EVERY participant to be movable — which would silently stop
        // a pinned waistband or a hung curtain from colliding — passes every assertion.
        // Verified by mutation: flipping `any` to `all` fails only because of this vertex.
        let mid = n * n + (n / 2) * n + n / 2;
        movable[mid] = false;

        let edges = unique_edges(&tris);
        (pos, tris, edges, movable)
    }

    /// Is every vertex of this primitive immovable?
    fn inert(movable: &[bool], vs: &[u32]) -> bool {
        vs.iter().all(|&v| !movable[v as usize])
    }

    #[test]
    fn an_all_movable_mask_changes_nothing() {
        let (pos, tris, edges, _) = two_sheets();
        let all = vec![true; pos.len()];
        let ex = one_ring_exclusion(pos.len(), &tris);
        for &search in &[0.01f32, 0.05, 0.2] {
            let vt_off = vertex_triangle_pairs(&pos, &tris, search, &ex, &[]);
            let vt_on = vertex_triangle_pairs(&pos, &tris, search, &ex, &all);
            assert!(!vt_off.is_empty(), "search {search}: no vertex-triangle candidates — the assertion below would be vacuous");
            assert_eq!(
                vt_off, vt_on,
                "search {search}: an all-movable mask must be a no-op for vertex-triangle"
            );

            let ee_off = edge_edge_pairs(&pos, &edges, search, &[]);
            let ee_on = edge_edge_pairs(&pos, &edges, search, &all);
            assert!(
                !ee_off.is_empty(),
                "search {search}: no edge-edge candidates"
            );
            assert_eq!(
                ee_off, ee_on,
                "search {search}: an all-movable mask must be a no-op for edge-edge"
            );
        }
    }

    #[test]
    fn a_static_body_drops_exactly_the_inert_pairs() {
        let (pos, tris, edges, movable) = two_sheets();
        let ex = one_ring_exclusion(pos.len(), &tris);
        // Accumulated over every `search`, because the edge-edge half below counts TRULY
        // CLOSE pairs and the sheets are 0.02 apart — at search 0.01 there are none of
        // either kind, so a per-search non-vacuity check would be a false alarm.
        let (mut ee_dropped, mut ee_cross) = (0usize, 0usize);
        for &search in &[0.01f32, 0.05, 0.2] {
            // ---- vertex-triangle ----
            let off = vertex_triangle_pairs(&pos, &tris, search, &ex, &[]);
            let on = vertex_triangle_pairs(&pos, &tris, search, &ex, &movable);
            let want: Vec<(u32, u32)> = off
                .iter()
                .copied()
                .filter(|&(v, ti)| !(inert(&movable, &[v]) && inert(&movable, &tris[ti as usize])))
                .collect();
            let dropped = off.len() - want.len();
            let cross = want
                .iter()
                .filter(|&&(v, ti)| movable[v as usize] != inert(&movable, &tris[ti as usize]))
                .count();
            assert!(
                dropped > 0 && cross > 0,
                "search {search}: vt must have both inert pairs to drop ({dropped}) and \
                 cloth-vs-body pairs to keep ({cross}) or this test proves nothing"
            );
            assert_eq!(
                on, want,
                "search {search}: vt filtered set must be exactly the non-inert pairs"
            );

            // ---- edge-edge ----
            // NOT compared against a filtered copy of the unfiltered run. `edge_edge_pairs`
            // pads the AABB on the QUERY side only, so "i indexed, j queries" is a different
            // predicate from "j indexed, i queries" — and which of the two is available
            // depends on which edge is movable. Checked against GROUND TRUTH instead, which
            // is the real contract and a stronger statement than agreeing with a sibling
            // call of the same function.
            let on = edge_edge_pairs(&pos, &edges, search, &movable);
            let seen: std::collections::HashSet<_> = on.iter().copied().collect();
            assert!(
                !on.iter().any(|&(i, j)| {
                    inert(&movable, &edges[i as usize]) && inert(&movable, &edges[j as usize])
                }),
                "search {search}: ee kept an inert pair — the filter is not filtering"
            );
            for (i, a) in edges.iter().enumerate() {
                for (j, b) in edges.iter().enumerate().skip(i + 1) {
                    if a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1] {
                        continue; // shared vertex — deliberately excluded
                    }
                    let d = crate::contact::segment_segment_distance(
                        pos[a[0] as usize],
                        pos[a[1] as usize],
                        pos[b[0] as usize],
                        pos[b[1] as usize],
                    );
                    if d > search {
                        continue;
                    }
                    let (ia, ib) = (inert(&movable, a), inert(&movable, b));
                    if ia && ib {
                        ee_dropped += 1;
                        continue; // inert: must NOT be offered, asserted above
                    }
                    ee_cross += usize::from(ia != ib);
                    assert!(
                        seen.contains(&(i as u32, j as u32)),
                        "search {search}: ee dropped non-inert close pair ({i},{j}) d={d}"
                    );
                }
            }
        }
        assert!(
            ee_dropped > 0 && ee_cross > 0,
            "ee must have both inert close pairs to drop ({ee_dropped}) and cloth-vs-body \
             close pairs to keep ({ee_cross}) or this test proves nothing"
        );
    }

    /// The streaming predicate must still see a cloth edge piercing a STATIC body triangle
    /// — the case the whole feature exists for. And it must ignore a pierce that is
    /// entirely inside the static geometry, since no step can undo one.
    #[test]
    fn the_intersection_predicate_sees_cloth_through_body_but_not_body_through_body() {
        let (mut pos, tris, mut edges, mut movable) = two_sheets();
        let any = |pos: &Vec<Vec3>, edges: &Vec<[u32; 2]>, movable: &[bool]| {
            any_edge_triangle_pair(pos, edges, &tris, 0.05, movable, |ei, ti| {
                let e = edges[ei as usize];
                let t = tris[ti as usize];
                crate::contact::edge_triangle_intersect(
                    pos[e[0] as usize],
                    pos[e[1] as usize],
                    pos[t[0] as usize],
                    pos[t[1] as usize],
                    pos[t[2] as usize],
                )
            })
        };
        assert!(
            !any(&pos, &edges, &movable),
            "the undisturbed two-sheet scene must not intersect"
        );

        // A MOVABLE spike driven down through the static lower sheet: must be reported.
        // It stops at y = 0.015, BELOW the movable upper sheet at y = 0.02, so the only
        // triangles it pierces belong to the static body — otherwise the static case
        // below would still (correctly) fire on the cloth sheet and prove nothing.
        let base = pos.len() as u32;
        pos.push(Vec3::new(0.12, 0.015, 0.13));
        pos.push(Vec3::new(0.12, -0.03, 0.13));
        movable.extend([true, true]);
        edges.push([base, base + 1]);
        assert!(
            any(&pos, &edges, &movable),
            "a movable edge piercing the static body MUST be reported"
        );

        // The same spike declared STATIC: now every participant is immovable, so it is
        // baked-in input geometry and deliberately not an invariant violation.
        movable[base as usize] = false;
        movable[base as usize + 1] = false;
        assert!(
            !any(&pos, &edges, &movable),
            "a pierce entirely between immovable primitives must NOT be reported"
        );
        // ...and it is still reported with no mask at all, proving the mask is what
        // silenced it rather than the geometry having drifted apart.
        assert!(
            any(&pos, &edges, &[]),
            "unfiltered, the same pierce must still be found"
        );

        // The MIRROR case, which is what the predicate's second pass exists for: a STATIC
        // edge piercing a MOVABLE triangle. Indexing only movable edges would miss this
        // entirely, so without this assertion half the function is uncovered. The spike
        // runs from above the movable upper sheet (y = 0.02) down to between the sheets, so
        // the only triangles it crosses are the movable ones.
        let base = pos.len() as u32;
        pos.push(Vec3::new(0.07, 0.05, 0.08));
        pos.push(Vec3::new(0.07, 0.008, 0.08));
        movable.extend([false, false]);
        edges.push([base, base + 1]);
        assert!(
            any(&pos, &edges, &movable),
            "a STATIC edge piercing a MOVABLE triangle must be reported (second pass)"
        );
    }
}

#[cfg(test)]
mod cell_size_tests {
    use super::*;

    /// Elements MUCH larger than the search radius — the regime that used to make the grid
    /// degenerate. Every candidate list must still be a superset of brute force.
    ///
    /// This is the regression guard for the cell-size fix. Before it, `cell = search` meant
    /// a primitive's expanded AABB spanned `extent/search` cells per axis (~150 on the
    /// gauge's `domino`), so insertion cost grew cubically and one 512-vertex step took
    /// ~29 s. The risk in fixing it was the opposite error — a cell so large that pairs get
    /// MISSED — so this pins correctness, not speed.
    #[test]
    fn large_elements_with_tiny_search_still_find_every_pair() {
        // Stations along x; at each, a PAIR of unit-length edges separated by 0.4*search in
        // y. Elements are 1000x the search radius — under the old `cell = search` rule each
        // edge would have spanned ~1000 cells per axis.
        let search = 1.0e-3f32;
        let mut positions = Vec::new();
        for i in 0..6 {
            let x = i as f32 * 0.5;
            positions.push(Vec3::new(x, 0.0, 0.0)); // 4i+0  edge A start
            positions.push(Vec3::new(x, 0.0, 1.0)); // 4i+1  edge A end
            positions.push(Vec3::new(x, 0.4e-3, 0.0)); // 4i+2  edge B start (within search)
            positions.push(Vec3::new(x, 0.4e-3, 1.0)); // 4i+3  edge B end
        }
        // Two disjoint edges per station, so near pairs exist with NO shared vertex.
        let mut edges: Vec<[u32; 2]> = Vec::new();
        for i in 0..6u32 {
            edges.push([4 * i, 4 * i + 1]);
            edges.push([4 * i + 2, 4 * i + 3]);
        }
        let tris: Vec<[u32; 3]> = (0..6u32).map(|i| [4 * i, 4 * i + 1, 4 * i + 2]).collect();

        // --- edge_edge: superset of brute force ---
        let got: std::collections::HashSet<(u32, u32)> =
            edge_edge_pairs(&positions, &edges, search, &[])
                .into_iter()
                .collect();
        let mut brute = 0usize;
        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                let (a, b) = (edges[i], edges[j]);
                if a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1] {
                    continue;
                }
                // Conservative: AABBs within `search` must appear as candidates.
                let box_of = |e: [u32; 2]| {
                    let (p, q) = (positions[e[0] as usize], positions[e[1] as usize]);
                    (p.min(q), p.max(q))
                };
                let (alo, ahi) = box_of(a);
                let (blo, bhi) = box_of(b);
                let near = (0..3).all(|k| alo[k] - search <= bhi[k] && blo[k] - search <= ahi[k]);
                if near {
                    brute += 1;
                    assert!(
                        got.contains(&(i as u32, j as u32)),
                        "edge_edge dropped a true candidate ({i},{j})"
                    );
                }
            }
        }
        assert!(brute > 0, "test must exercise at least one near pair");

        // --- vertex_triangle: superset of brute force ---
        let excluded: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
        let vt: std::collections::HashSet<(u32, u32)> =
            vertex_triangle_pairs(&positions, &tris, search, &excluded, &[])
                .into_iter()
                .collect();
        let mut vt_brute = 0usize;
        for (ti, t) in tris.iter().enumerate() {
            let (pa, pb, pc) = (
                positions[t[0] as usize],
                positions[t[1] as usize],
                positions[t[2] as usize],
            );
            let (lo, hi) = (pa.min(pb).min(pc), pa.max(pb).max(pc));
            for (vi, p) in positions.iter().enumerate() {
                if (0..3).all(|k| p[k] >= lo[k] - search && p[k] <= hi[k] + search) {
                    vt_brute += 1;
                    assert!(
                        vt.contains(&(vi as u32, ti as u32)),
                        "vertex_triangle dropped a true candidate ({vi},{ti})"
                    );
                }
            }
        }
        assert!(
            vt_brute > 0,
            "test must exercise at least one near vertex/triangle"
        );

        // --- edge_triangle: superset of brute force ---
        let et: std::collections::HashSet<(u32, u32)> =
            edge_triangle_pairs(&positions, &edges, &tris, search)
                .into_iter()
                .collect();
        assert!(!et.is_empty(), "edge_triangle must find candidates here");
    }

    /// The cell rule itself: never below the search radius, never below the primitive
    /// extent. The second half is what stops the cubic blowup.
    #[test]
    fn grid_cell_is_bounded_below_by_both_inputs() {
        assert_eq!(
            grid_cell(1.0e-3, 0.15),
            0.15,
            "must not stay at the tiny search radius"
        );
        assert_eq!(
            grid_cell(0.5, 0.15),
            0.5,
            "must not drop below the search radius"
        );
        assert!(
            grid_cell(0.0, 0.0) > 0.0,
            "must never be zero (division by cell)"
        );
    }
}
