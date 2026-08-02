// SPDX-License-Identifier: GPL-2.0-or-later
//! Minimal SoA geometry the solver operates on. This is the *solver-side* mesh
//! (dense vertex indices, derived edges), distinct from the authoring `Mesh` in
//! `engine-api` — mirroring the reference's two-model split (frontend scene vs
//! solver scene).

use glam::Vec3;
use std::collections::BTreeSet;

/// A triangle shell mesh with dense vertex indices `[0, positions.len())`.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    /// Triangle vertex indices (winding preserved).
    pub tris: Vec<[u32; 3]>,
    /// Unique undirected edges, derived from `tris` (springs / stretch elements).
    pub edges: Vec<[u32; 2]>,
}

impl Mesh {
    pub fn new(positions: Vec<Vec3>, tris: Vec<[u32; 3]>) -> Self {
        let edges = derive_unique_edges(&tris);
        Self {
            positions,
            tris,
            edges,
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
}

/// Collect the unique undirected edges of a triangle set. Deterministic order
/// (sorted) so downstream constraint solving is reproducible run-to-run.
pub fn derive_unique_edges(tris: &[[u32; 3]]) -> Vec<[u32; 2]> {
    let mut set: BTreeSet<(u32, u32)> = BTreeSet::new();
    for t in tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            set.insert(if a < b { (a, b) } else { (b, a) });
        }
    }
    set.into_iter().map(|(a, b)| [a, b]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_unique_and_deterministic() {
        // Two triangles sharing edge (1,2): quad split.
        let tris = vec![[0u32, 1, 2], [2, 1, 3]];
        let e = derive_unique_edges(&tris);
        // 5 unique edges: 0-1,0-2,1-2,1-3,2-3.
        assert_eq!(e.len(), 5);
        // Deterministic sorted order.
        assert_eq!(e, derive_unique_edges(&tris));
        // Shared edge (1,2) appears exactly once, normalized low<high.
        assert_eq!(e.iter().filter(|x| **x == [1, 2]).count(), 1);
    }
}
