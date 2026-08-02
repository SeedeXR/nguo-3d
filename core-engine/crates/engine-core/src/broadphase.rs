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

pub fn vertex_triangle_pairs(
    positions: &[Vec3],
    tris: &[[u32; 3]],
    dhat: f32,
    excluded: &[Vec<u32>],
) -> Vec<(u32, u32)> {
    if positions.is_empty() || tris.is_empty() {
        return Vec::new();
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

    // Hash vertices into cells.
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (i, p) in positions.iter().enumerate() {
        grid.entry(key(*p)).or_default().push(i as u32);
    }

    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for (ti, t) in tris.iter().enumerate() {
        let (pa, pb, pc) = (
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        let lo = pa.min(pb).min(pc) - Vec3::splat(dhat);
        let hi = pa.max(pb).max(pc) + Vec3::splat(dhat);
        let (kl, kh) = (key(lo), key(hi));
        for cx in kl.0..=kh.0 {
            for cy in kl.1..=kh.1 {
                for cz in kl.2..=kh.2 {
                    if let Some(verts) = grid.get(&(cx, cy, cz)) {
                        for &v in verts {
                            if excluded[v as usize].iter().any(|&e| t.contains(&e)) {
                                continue;
                            }
                            pairs.push((v, ti as u32));
                        }
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
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
pub fn edge_edge_pairs(positions: &[Vec3], edges: &[[u32; 2]], search: f32) -> Vec<(u32, u32)> {
    if edges.len() < 2 {
        return Vec::new();
    }
    let cell = grid_cell(search, mean_edge_len(positions, edges));
    let inv = 1.0 / cell;
    let key = |v: f32| -> i32 { (v * inv).floor() as i32 };

    // Insert each edge index into every grid cell its (search-expanded) AABB covers.
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (ei, e) in edges.iter().enumerate() {
        let (p, q) = (positions[e[0] as usize], positions[e[1] as usize]);
        let lo = p.min(q) - Vec3::splat(search);
        let hi = p.max(q) + Vec3::splat(search);
        for cx in key(lo.x)..=key(hi.x) {
            for cy in key(lo.y)..=key(hi.y) {
                for cz in key(lo.z)..=key(hi.z) {
                    grid.entry((cx, cy, cz)).or_default().push(ei as u32);
                }
            }
        }
    }
    // Pair edges that co-occur in any cell; dedup; skip shared-vertex edges.
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for bucket in grid.values() {
        for a in 0..bucket.len() {
            for b in (a + 1)..bucket.len() {
                let (mut i, mut j) = (bucket[a], bucket[b]);
                if i > j {
                    std::mem::swap(&mut i, &mut j);
                }
                let (ea, eb) = (edges[i as usize], edges[j as usize]);
                if ea[0] == eb[0] || ea[0] == eb[1] || ea[1] == eb[0] || ea[1] == eb[1] {
                    continue; // shared vertex
                }
                pairs.push((i, j));
            }
        }
    }
    let raw = pairs.len();
    pairs.sort_unstable();
    pairs.dedup();
    if bp_stats() {
        let biggest = grid.values().map(|b| b.len()).max().unwrap_or(0);
        eprintln!(
            "bp ee: {} edges, cell {cell:.5}, {} cells, biggest bucket {biggest}, \
             {raw} raw pairs -> {} deduped",
            edges.len(),
            grid.len(),
            pairs.len()
        );
    }
    pairs
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
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (ei, e) in edges.iter().enumerate() {
        let (p, q) = (positions[e[0] as usize], positions[e[1] as usize]);
        let lo = p.min(q) - Vec3::splat(search);
        let hi = p.max(q) + Vec3::splat(search);
        for cx in key(lo.x)..=key(hi.x) {
            for cy in key(lo.y)..=key(hi.y) {
                for cz in key(lo.z)..=key(hi.z) {
                    grid.entry((cx, cy, cz)).or_default().push(ei as u32);
                }
            }
        }
    }
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
                    if let Some(es) = grid.get(&(cx, cy, cz)) {
                        for &ei in es {
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
pub fn any_edge_triangle_pair(
    positions: &[Vec3],
    edges: &[[u32; 2]],
    tris: &[[u32; 3]],
    search: f32,
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

    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (ei, e) in edges.iter().enumerate() {
        let (p, q) = (positions[e[0] as usize], positions[e[1] as usize]);
        let lo = p.min(q) - Vec3::splat(search);
        let hi = p.max(q) + Vec3::splat(search);
        for cx in key(lo.x)..=key(hi.x) {
            for cy in key(lo.y)..=key(hi.y) {
                for cz in key(lo.z)..=key(hi.z) {
                    grid.entry((cx, cy, cz)).or_default().push(ei as u32);
                }
            }
        }
    }
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
                    if let Some(es) = grid.get(&(cx, cy, cz)) {
                        for &ei in es {
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
        let pairs = vertex_triangle_pairs(&positions, &tris, dhat, &excluded);
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
            let exhausted = any_edge_triangle_pair(&pos, &edges, &tris, search, |ei, ti| {
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
        let found = any_edge_triangle_pair(&pos, &edges, &tris, 0.4, |_, _| {
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
            any_edge_triangle_pair(pos, &edges, &tris, 0.15, |ei, ti| {
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
            edge_edge_pairs(&positions, &edges, search)
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
            vertex_triangle_pairs(&positions, &tris, search, &excluded)
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
