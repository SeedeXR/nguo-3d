// SPDX-License-Identifier: GPL-2.0-or-later
//! Pattern geometry for nguo-3d: stable ids, the vector network, and the
//! seam-allowance offset.
//!
//! UI-framework-free by construction: the only dependencies are
//! `kurbo`, `serde` and `thiserror`. Everything here is testable headless.

#![forbid(unsafe_code)]

pub mod id;
pub mod network;
pub mod offset;

pub use id::{Arena, Id};
pub use network::{
    Edge, EdgeGeom, EdgeId, EdgeSemantic, NetworkError, Split, VectorNetwork, Vertex, VertexId,
    GEOM_ACCURACY,
};
pub use offset::{is_simple, offset_closed, offset_uniform, Join, OffsetError, OffsetOptions};

/// Re-exported so downstream crates agree on one `kurbo`.
pub use kurbo;

use kurbo::BezPath;

/// Seam allowance for one boundary loop of a network, one width per edge.
///
/// The bridge between [`VectorNetwork`] and [`offset_closed`]: walk the
/// boundary containing `start`, ask `width` for each edge, offset.
pub fn seam_allowance(
    net: &VectorNetwork,
    start: EdgeId,
    mut width: impl FnMut(EdgeId, EdgeSemantic) -> f64,
    opts: &OffsetOptions,
) -> Result<BezPath, PatternError> {
    let walk = net.boundary_loop(start)?;
    let segs = net.loop_segments(&walk)?;
    let widths: Vec<f64> = walk
        .iter()
        .map(|&(id, _)| {
            let sem = net
                .edges
                .get(id)
                .map(|e| e.semantic)
                .unwrap_or(EdgeSemantic::Cut);
            width(id, sem)
        })
        .collect();
    Ok(offset_closed(&segs, &widths, opts)?)
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PatternError {
    #[error(transparent)]
    Network(#[from] NetworkError),
    #[error(transparent)]
    Offset(#[from] OffsetError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::{Point, Shape};

    #[test]
    fn seam_allowance_only_grows_the_seam_edges() {
        let mut n = VectorNetwork::new();
        let v: Vec<VertexId> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        let e: Vec<EdgeId> = (0..4)
            .map(|i| n.add_line(v[i], v[(i + 1) % 4]).unwrap())
            .collect();
        // The left edge is a fold: no allowance there.
        n.edges.get_mut(e[3]).unwrap().semantic = EdgeSemantic::Fold;

        let out = seam_allowance(
            &n,
            e[0],
            |_, sem| if sem.takes_allowance() { 10.0 } else { 0.0 },
            &OffsetOptions::default(),
        )
        .unwrap();
        let bb = out.bounding_box();
        assert!(
            (bb.x0 - 0.0).abs() < 1e-6,
            "the fold edge must not move: {bb:?}"
        );
        assert!((bb.x1 - 110.0).abs() < 1e-6);
        assert!((bb.y0 + 10.0).abs() < 1e-6);
        assert!((bb.y1 - 110.0).abs() < 1e-6);
    }
}
