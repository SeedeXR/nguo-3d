// SPDX-License-Identifier: GPL-2.0-or-later
//! The vector network — chosen over closed paths, deliberately.
//!
//! Vertices and edges meet freely. That is not a stylistic preference: a
//! garment piece has T-junctions where an internal seam line lands on the
//! boundary, darts branching off an edge, and edges shared between pieces.
//! Closed paths model none of those without duplicating geometry.

use kurbo::{
    CubicBez, Line, ParamCurve, ParamCurveArclen, ParamCurveNearest, PathSeg, Point, Rect, Shape,
};
use serde::{Deserialize, Serialize};

use crate::id::{Arena, Id};

/// Accuracy for arc length and nearest-point queries, in mm. A hundredth of a
/// millimetre is two orders below anything a cutter can hold.
pub const GEOM_ACCURACY: f64 = 1e-2;

/// A point in the piece's local space, in millimetres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    pub pos: Point,
}

impl Vertex {
    pub fn new(x: f64, y: f64) -> Vertex {
        Vertex {
            pos: Point::new(x, y),
        }
    }
}

pub type VertexId = Id<Vertex>;

/// What an edge *means*, which is what drives everything downstream: whether it
/// gets a seam allowance, whether it can be sewn, how it is drawn, whether the
/// triangulator treats it as a boundary (guide §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeSemantic {
    /// Sewn to another edge. Gets a seam allowance.
    Seam,
    /// Mirrored, not sewn and not cut.
    Fold,
    /// Cut edge with no seam (a selvedge or finished edge).
    Cut,
    /// The grain direction marker.
    Grain,
    /// A drawn line inside the piece (pocket placement, style line).
    Internal,
    /// One side of a dart.
    DartLeg,
}

impl EdgeSemantic {
    /// Does this edge form part of the piece outline?
    pub fn is_boundary(self) -> bool {
        matches!(
            self,
            EdgeSemantic::Seam | EdgeSemantic::Fold | EdgeSemantic::Cut
        )
    }

    /// Does this edge get a seam allowance by default?
    pub fn takes_allowance(self) -> bool {
        matches!(self, EdgeSemantic::Seam)
    }
}

/// The shape of an edge between its two vertices. Endpoints live in the
/// vertices, never duplicated here — that is what keeps a shared vertex shared.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EdgeGeom {
    Line,
    /// Cubic Bézier with the two interior control points.
    Cubic {
        c1: Point,
        c2: Point,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub start: VertexId,
    pub end: VertexId,
    pub geom: EdgeGeom,
    pub semantic: EdgeSemantic,
}

pub type EdgeId = Id<Edge>;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum NetworkError {
    #[error("no such vertex {0:?}")]
    NoSuchVertex(VertexId),
    #[error("no such edge {0:?}")]
    NoSuchEdge(EdgeId),
    #[error("vertex {0:?} still has {1} incident edge(s)")]
    VertexInUse(VertexId, usize),
    #[error("parameter {0} is outside the open interval (0, 1)")]
    ParamOutOfRange(f64),
    #[error("edges {0:?} and {1:?} do not share exactly one degree-2 vertex")]
    NotJoinable(EdgeId, EdgeId),
    #[error("only straight edges can be joined; {0:?} is curved")]
    JoinNeedsLines(EdgeId),
    #[error("edge {0:?} is not a boundary edge")]
    NotABoundary(EdgeId),
    #[error("the boundary walk from {0:?} hit vertex {1:?} with degree {2}, not 2")]
    OpenBoundary(EdgeId, VertexId, usize),
}

/// Result of [`VectorNetwork::split_edge`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Split {
    /// The vertex created at the split parameter.
    pub vertex: VertexId,
    /// The original edge, now covering `[0, t]`.
    pub first: EdgeId,
    /// The new edge covering `[t, 1]`.
    pub second: EdgeId,
}

/// A vertex + edge graph. No implicit ordering, no closed-path assumption.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorNetwork {
    pub vertices: Arena<Vertex>,
    pub edges: Arena<Edge>,
}

impl VectorNetwork {
    pub fn new() -> VectorNetwork {
        VectorNetwork::default()
    }

    // --- construction -----------------------------------------------------

    pub fn add_vertex(&mut self, pos: Point) -> VertexId {
        self.vertices.insert(Vertex { pos })
    }

    pub fn add_edge(
        &mut self,
        start: VertexId,
        end: VertexId,
        geom: EdgeGeom,
        semantic: EdgeSemantic,
    ) -> Result<EdgeId, NetworkError> {
        self.check_vertex(start)?;
        self.check_vertex(end)?;
        Ok(self.edges.insert(Edge {
            start,
            end,
            geom,
            semantic,
        }))
    }

    /// Convenience: a straight seam edge.
    pub fn add_line(&mut self, start: VertexId, end: VertexId) -> Result<EdgeId, NetworkError> {
        self.add_edge(start, end, EdgeGeom::Line, EdgeSemantic::Seam)
    }

    pub fn remove_edge(&mut self, id: EdgeId) -> Result<Edge, NetworkError> {
        self.edges.remove(id).ok_or(NetworkError::NoSuchEdge(id))
    }

    /// Removing a vertex that still carries edges would dangle references, so
    /// it is an error rather than a silent cascade.
    pub fn remove_vertex(&mut self, id: VertexId) -> Result<Vertex, NetworkError> {
        let degree = self.degree(id);
        if degree > 0 {
            return Err(NetworkError::VertexInUse(id, degree));
        }
        self.vertices
            .remove(id)
            .ok_or(NetworkError::NoSuchVertex(id))
    }

    fn check_vertex(&self, id: VertexId) -> Result<(), NetworkError> {
        self.vertices
            .contains(id)
            .then_some(())
            .ok_or(NetworkError::NoSuchVertex(id))
    }

    // --- incidence --------------------------------------------------------

    /// Edges touching `v`.
    ///
    /// ponytail: linear scan over the edge arena. An incidence index is derived
    /// state (guide §4.3) and belongs in a rebuildable cache next to the
    /// spatial index, not in the document; add one there if a piece ever grows
    /// past a few thousand edges.
    pub fn edges_at(&self, v: VertexId) -> impl Iterator<Item = (EdgeId, &Edge)> + '_ {
        self.edges
            .iter()
            .filter(move |(_, e)| e.start == v || e.end == v)
    }

    pub fn degree(&self, v: VertexId) -> usize {
        self.edges
            .values()
            .map(|e| usize::from(e.start == v) + usize::from(e.end == v))
            .sum()
    }

    // --- geometry ---------------------------------------------------------

    /// The edge as a kurbo segment, in piece-local millimetres.
    pub fn seg(&self, id: EdgeId) -> Result<PathSeg, NetworkError> {
        let e = self.edges.get(id).ok_or(NetworkError::NoSuchEdge(id))?;
        let p0 = self.pos(e.start)?;
        let p3 = self.pos(e.end)?;
        Ok(match e.geom {
            EdgeGeom::Line => PathSeg::Line(Line::new(p0, p3)),
            EdgeGeom::Cubic { c1, c2 } => PathSeg::Cubic(CubicBez::new(p0, c1, c2, p3)),
        })
    }

    pub fn pos(&self, v: VertexId) -> Result<Point, NetworkError> {
        self.vertices
            .get(v)
            .map(|x| x.pos)
            .ok_or(NetworkError::NoSuchVertex(v))
    }

    pub fn arc_length(&self, id: EdgeId) -> Result<f64, NetworkError> {
        Ok(self.seg(id)?.arclen(GEOM_ACCURACY))
    }

    /// Closest point on `id` to `p`: the curve parameter and the distance.
    /// This is the hit-test and snap primitive (guide §6.2, §6.3).
    /// The distance is recomputed at the returned parameter rather than taken
    /// from kurbo's `distance_sq`, so `eval(t)` and the distance always agree —
    /// callers snap to `eval(t)`, and a distance that did not match it would
    /// make the snap indicator lie.
    pub fn nearest_point(&self, id: EdgeId, p: Point) -> Result<(f64, f64), NetworkError> {
        let seg = self.seg(id)?;
        let t = seg.nearest(p, GEOM_ACCURACY).t;
        Ok((t, (seg.eval(t) - p).hypot()))
    }

    /// Closest point across the whole network. `None` for an empty network.
    pub fn nearest_on_network(&self, p: Point) -> Option<(EdgeId, f64, f64)> {
        self.edges
            .ids()
            .filter_map(|id| {
                let (t, d) = self.nearest_point(id, p).ok()?;
                Some((id, t, d))
            })
            .reduce(|a, b| if b.2 < a.2 { b } else { a })
    }

    pub fn bounds(&self) -> Option<Rect> {
        self.edges
            .ids()
            .filter_map(|id| self.seg(id).ok())
            .map(|s| s.bounding_box())
            .reduce(|a, b| a.union(b))
    }

    // --- editing ----------------------------------------------------------

    /// Split `id` at curve parameter `t`, keeping the original edge as the
    /// first half so that anything referencing it (a seam, a feature) still
    /// points at a live edge. Ids are stable: `first == id`.
    pub fn split_edge(&mut self, id: EdgeId, t: f64) -> Result<Split, NetworkError> {
        if !(t > 0.0 && t < 1.0) {
            return Err(NetworkError::ParamOutOfRange(t));
        }
        let seg = self.seg(id)?;
        let edge = *self.edges.get(id).ok_or(NetworkError::NoSuchEdge(id))?;
        let mid = seg.eval(t);
        let (a, b) = (seg.subsegment(0.0..t), seg.subsegment(t..1.0));

        let v = self.add_vertex(mid);
        let second = self.edges.insert(Edge {
            start: v,
            end: edge.end,
            geom: EdgeGeom::of_seg(b),
            semantic: edge.semantic,
        });
        let first = self.edges.get_mut(id).expect("checked above");
        first.end = v;
        first.geom = EdgeGeom::of_seg(a);
        Ok(Split {
            vertex: v,
            first: id,
            second,
        })
    }

    /// The inverse of [`VectorNetwork::split_edge`] for straight edges: drop the
    /// shared degree-2 vertex and keep one edge.
    ///
    /// ponytail: lines only. Two arbitrary cubics do not concatenate into one
    /// cubic, so a general join needs curve fitting (`kurbo::fit_to_bezpath`)
    /// and an error tolerance the caller has to accept. Add that when a user
    /// actually asks to merge two curved edges.
    pub fn join_edges(&mut self, a: EdgeId, b: EdgeId) -> Result<EdgeId, NetworkError> {
        let ea = *self.edges.get(a).ok_or(NetworkError::NoSuchEdge(a))?;
        let eb = *self.edges.get(b).ok_or(NetworkError::NoSuchEdge(b))?;
        if !matches!(ea.geom, EdgeGeom::Line) {
            return Err(NetworkError::JoinNeedsLines(a));
        }
        if !matches!(eb.geom, EdgeGeom::Line) {
            return Err(NetworkError::JoinNeedsLines(b));
        }
        let shared = [ea.start, ea.end]
            .into_iter()
            .find(|v| eb.start == *v || eb.end == *v)
            .ok_or(NetworkError::NotJoinable(a, b))?;
        if self.degree(shared) != 2 {
            return Err(NetworkError::NotJoinable(a, b));
        }
        let outer_a = if ea.start == shared { ea.end } else { ea.start };
        let outer_b = if eb.start == shared { eb.end } else { eb.start };
        self.edges.remove(b);
        let e = self.edges.get_mut(a).expect("checked above");
        e.start = outer_a;
        e.end = outer_b;
        self.vertices.remove(shared);
        Ok(a)
    }

    /// Merge `drop` into `keep`: every edge that referenced `drop` now
    /// references `keep`. This is the network join that makes T-junctions and
    /// shared seam edges possible.
    pub fn weld_vertices(&mut self, keep: VertexId, drop: VertexId) -> Result<(), NetworkError> {
        self.check_vertex(keep)?;
        self.check_vertex(drop)?;
        if keep == drop {
            return Ok(());
        }
        for (_, e) in self.edges.iter_mut() {
            if e.start == drop {
                e.start = keep;
            }
            if e.end == drop {
                e.end = keep;
            }
        }
        // Any edge that collapsed to a self-loop is degenerate; drop it.
        let loops: Vec<EdgeId> = self
            .edges
            .iter()
            .filter(|(_, e)| e.start == e.end)
            .map(|(id, _)| id)
            .collect();
        for id in loops {
            self.edges.remove(id);
        }
        self.vertices.remove(drop);
        Ok(())
    }

    pub fn set_vertex_pos(&mut self, v: VertexId, pos: Point) -> Result<Point, NetworkError> {
        let vert = self
            .vertices
            .get_mut(v)
            .ok_or(NetworkError::NoSuchVertex(v))?;
        Ok(std::mem::replace(&mut vert.pos, pos))
    }

    // --- outline ----------------------------------------------------------

    /// Walk the boundary cycle containing `start`, in edge order.
    ///
    /// Returns `(edge, reversed)` pairs: `reversed` means the edge's `end` comes
    /// first in the walk. Only boundary-semantic edges are followed, so
    /// internal lines and dart legs hanging off the outline do not derail it —
    /// but a boundary vertex whose boundary-degree is not 2 is a genuine
    /// modelling error and is reported as one.
    pub fn boundary_loop(&self, start: EdgeId) -> Result<Vec<(EdgeId, bool)>, NetworkError> {
        let e0 = *self
            .edges
            .get(start)
            .ok_or(NetworkError::NoSuchEdge(start))?;
        if !e0.semantic.is_boundary() {
            return Err(NetworkError::NotABoundary(start));
        }
        let mut out = vec![(start, false)];
        let mut at = e0.end;
        let origin = e0.start;
        let mut prev = start;
        while at != origin {
            let mut next = None;
            for (id, e) in self.edges.iter() {
                if id == prev || !e.semantic.is_boundary() {
                    continue;
                }
                if e.start == at {
                    next = Some((id, false, e.end));
                    break;
                }
                if e.end == at {
                    next = Some((id, true, e.start));
                    break;
                }
            }
            let boundary_degree = self
                .edges_at(at)
                .filter(|(_, e)| e.semantic.is_boundary())
                .count();
            if boundary_degree != 2 {
                return Err(NetworkError::OpenBoundary(start, at, boundary_degree));
            }
            let (id, rev, to) = next.ok_or(NetworkError::OpenBoundary(start, at, 0))?;
            out.push((id, rev));
            prev = id;
            at = to;
            if out.len() > self.edges.len() {
                return Err(NetworkError::OpenBoundary(start, at, boundary_degree));
            }
        }
        Ok(out)
    }

    /// The boundary walk as oriented kurbo segments, ready for offsetting.
    pub fn loop_segments(&self, walk: &[(EdgeId, bool)]) -> Result<Vec<PathSeg>, NetworkError> {
        walk.iter()
            .map(|&(id, rev)| {
                let s = self.seg(id)?;
                Ok(if rev { s.reverse() } else { s })
            })
            .collect()
    }
}

impl EdgeGeom {
    /// The interior shape of a kurbo segment, dropping its endpoints (which
    /// live in the vertices). Quads are raised to cubics: the network stores
    /// one curve kind, not three.
    pub fn of_seg(seg: PathSeg) -> EdgeGeom {
        match seg {
            PathSeg::Line(_) => EdgeGeom::Line,
            PathSeg::Quad(q) => {
                let c = q.raise();
                EdgeGeom::Cubic { c1: c.p1, c2: c.p2 }
            }
            PathSeg::Cubic(c) => EdgeGeom::Cubic { c1: c.p1, c2: c.p2 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> (VectorNetwork, Vec<EdgeId>) {
        let mut n = VectorNetwork::new();
        let v: Vec<VertexId> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        let e = (0..4)
            .map(|i| n.add_line(v[i], v[(i + 1) % 4]).unwrap())
            .collect();
        (n, e)
    }

    #[test]
    fn arc_length_of_a_line() {
        let (n, e) = square();
        assert!((n.arc_length(e[0]).unwrap() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn split_keeps_the_original_id_and_halves_the_length() {
        let (mut n, e) = square();
        let s = n.split_edge(e[0], 0.25).unwrap();
        assert_eq!(s.first, e[0], "the original id must survive a split");
        assert!((n.arc_length(s.first).unwrap() - 25.0).abs() < 1e-9);
        assert!((n.arc_length(s.second).unwrap() - 75.0).abs() < 1e-9);
        assert_eq!(n.pos(s.vertex).unwrap(), Point::new(25.0, 0.0));
        assert_eq!(n.degree(s.vertex), 2);
    }

    #[test]
    fn split_a_curve_preserves_the_shape() {
        let mut n = VectorNetwork::new();
        let a = n.add_vertex(Point::new(0.0, 0.0));
        let b = n.add_vertex(Point::new(100.0, 0.0));
        let e = n
            .add_edge(
                a,
                b,
                EdgeGeom::Cubic {
                    c1: Point::new(0.0, 60.0),
                    c2: Point::new(100.0, 60.0),
                },
                EdgeSemantic::Seam,
            )
            .unwrap();
        let before = n.arc_length(e).unwrap();
        let s = n.split_edge(e, 0.4).unwrap();
        let after = n.arc_length(s.first).unwrap() + n.arc_length(s.second).unwrap();
        assert!((before - after).abs() < 1e-6, "{before} vs {after}");
    }

    #[test]
    fn split_rejects_endpoints() {
        let (mut n, e) = square();
        assert!(matches!(
            n.split_edge(e[0], 0.0),
            Err(NetworkError::ParamOutOfRange(_))
        ));
        assert!(matches!(
            n.split_edge(e[0], 1.0),
            Err(NetworkError::ParamOutOfRange(_))
        ));
    }

    #[test]
    fn join_undoes_split_for_lines() {
        let (mut n, e) = square();
        let before = n.arc_length(e[0]).unwrap();
        let s = n.split_edge(e[0], 0.3).unwrap();
        let joined = n.join_edges(s.first, s.second).unwrap();
        assert_eq!(joined, e[0]);
        assert!((n.arc_length(joined).unwrap() - before).abs() < 1e-9);
        assert_eq!(n.vertices.len(), 4);
        assert_eq!(n.edges.len(), 4);
    }

    #[test]
    fn join_refuses_curves() {
        let mut n = VectorNetwork::new();
        let a = n.add_vertex(Point::new(0.0, 0.0));
        let b = n.add_vertex(Point::new(10.0, 0.0));
        let e = n
            .add_edge(
                a,
                b,
                EdgeGeom::Cubic {
                    c1: Point::new(3.0, 5.0),
                    c2: Point::new(7.0, 5.0),
                },
                EdgeSemantic::Seam,
            )
            .unwrap();
        let s = n.split_edge(e, 0.5).unwrap();
        assert!(matches!(
            n.join_edges(s.first, s.second),
            Err(NetworkError::JoinNeedsLines(_))
        ));
    }

    #[test]
    fn t_junctions_are_first_class() {
        // An internal line landing on the middle of a boundary edge: the thing
        // closed paths cannot express.
        let (mut n, e) = square();
        let s = n.split_edge(e[0], 0.5).unwrap();
        let apex = n.add_vertex(Point::new(50.0, 40.0));
        n.add_edge(s.vertex, apex, EdgeGeom::Line, EdgeSemantic::Internal)
            .unwrap();
        assert_eq!(n.degree(s.vertex), 3);
        // The boundary walk ignores the internal spur.
        let walk = n.boundary_loop(e[1]).unwrap();
        assert_eq!(walk.len(), 5);
    }

    #[test]
    fn weld_merges_vertices() {
        let (mut n, _) = square();
        let extra = n.add_vertex(Point::new(0.5, 0.0));
        let apex = n.add_vertex(Point::new(50.0, 50.0));
        n.add_edge(extra, apex, EdgeGeom::Line, EdgeSemantic::Internal)
            .unwrap();
        let corner = n.vertices.ids().next().unwrap();
        n.weld_vertices(corner, extra).unwrap();
        assert!(!n.vertices.contains(extra));
        assert_eq!(n.degree(corner), 3);
    }

    #[test]
    fn remove_vertex_refuses_while_edges_reference_it() {
        let (mut n, _) = square();
        let v = n.vertices.ids().next().unwrap();
        assert!(matches!(
            n.remove_vertex(v),
            Err(NetworkError::VertexInUse(_, 2))
        ));
    }

    #[test]
    fn nearest_point_finds_the_foot_of_the_perpendicular() {
        let (n, e) = square();
        let (t, d) = n.nearest_point(e[0], Point::new(30.0, 12.0)).unwrap();
        assert!((t - 0.3).abs() < 1e-6, "t = {t}");
        assert!((d - 12.0).abs() < 1e-6, "d = {d}");
    }

    #[test]
    fn nearest_on_network_picks_the_right_edge() {
        let (n, e) = square();
        let (id, _, d) = n.nearest_on_network(Point::new(98.0, 50.0)).unwrap();
        assert_eq!(id, e[1]);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn boundary_loop_walks_the_outline_once() {
        let (n, e) = square();
        let walk = n.boundary_loop(e[0]).unwrap();
        assert_eq!(
            walk,
            vec![(e[0], false), (e[1], false), (e[2], false), (e[3], false)]
        );
    }

    #[test]
    fn boundary_loop_reports_an_open_outline() {
        let (mut n, e) = square();
        n.remove_edge(e[2]).unwrap();
        assert!(matches!(
            n.boundary_loop(e[0]),
            Err(NetworkError::OpenBoundary(..))
        ));
    }

    #[test]
    fn serde_round_trip() {
        let (n, _) = square();
        let json = serde_json::to_string(&n).unwrap();
        let back: VectorNetwork = serde_json::from_str(&json).unwrap();
        assert_eq!(n, back);
    }
}
