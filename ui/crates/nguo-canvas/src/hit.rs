// SPDX-License-Identifier: GPL-2.0-or-later
//! Hit-testing through a spatial index, with a **pixel** tolerance and layer
//! priority (guide §6.2).
//!
//! The index is built in **world** space from a [`Document`] and keyed on the
//! document revision, so it is derived state (guide §4.3) and never saved.
//! Queries take a screen-space tolerance in logical px and divide by the zoom
//! exactly once, which is what makes a handle grabbable at 0.02× and at 400×
//! with the same code.

use std::collections::BTreeMap;

use nguo_doc::{Document, Piece, PieceId};
use nguo_pattern::{EdgeGeom, EdgeId, VertexId};
use vello::kurbo::{BezPath, ParamCurve, ParamCurveNearest, PathSeg, Point, Rect, Shape, Vec2};

use crate::View;

/// Hit layers in priority order: a click that is within tolerance of both a
/// point and an edge takes the point (guide §6.2). `Ord` *is* the priority, so
/// there is no second place where the order is written down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    Fill,
    Edge,
    Handle,
    Point,
}

/// Which Bézier control point of an edge a handle belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandleEnd {
    /// The control point next to the edge's start vertex.
    Start,
    /// The control point next to the edge's end vertex.
    End,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Target {
    Fill {
        piece: PieceId,
    },
    Edge {
        piece: PieceId,
        edge: EdgeId,
    },
    Handle {
        piece: PieceId,
        edge: EdgeId,
        end: HandleEnd,
    },
    Point {
        piece: PieceId,
        vertex: VertexId,
    },
}

impl Target {
    pub fn layer(self) -> Layer {
        match self {
            Target::Fill { .. } => Layer::Fill,
            Target::Edge { .. } => Layer::Edge,
            Target::Handle { .. } => Layer::Handle,
            Target::Point { .. } => Layer::Point,
        }
    }

    pub fn piece(self) -> PieceId {
        match self {
            Target::Fill { piece }
            | Target::Edge { piece, .. }
            | Target::Handle { piece, .. }
            | Target::Point { piece, .. } => piece,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub target: Target,
    /// Where on the target, in world mm.
    pub at: Point,
    /// World-space distance from the query point.
    pub distance: f64,
    /// Curve parameter for a [`Target::Edge`] hit; `0.0` for everything else.
    pub t: f64,
}

enum Geom {
    Dot(Point),
    Seg(PathSeg),
    /// A closed boundary; hit by containment, not by distance.
    Area(BezPath),
}

struct Item {
    target: Target,
    geom: Geom,
    bbox: Rect,
}

/// A uniform-grid spatial index over the document's world-space geometry.
///
/// ponytail: a uniform grid, not a BVH. Pattern pieces are laid out on a marker
/// with similar-sized entities and no deep nesting, which is exactly the case a
/// grid handles as well as a tree for a tenth of the code. The ceiling is a
/// document with wildly mixed entity scales; swap in a BVH then.
///
/// ponytail: the index is rebuilt **whole** whenever the revision moves, which
/// during a drag is every frame. Measured (`tests/perf.rs`, debug profile,
/// 50 pieces × 24 cubic edges = 4 850 items): **0.81 ms**, 10% of an 8.3 ms
/// frame — the largest single item on the canvas's bill, and comfortably inside
/// it. A query is 1 µs. The upgrade, if a bigger document ever needs it, is to
/// key the items per piece and rebuild only the dirty ones — the same id-keyed
/// dirty flags `scene::SceneCache` already carries, which is why that is a small
/// change rather than a redesign.
pub struct Index {
    items: Vec<Item>,
    /// Cell size in world mm.
    cell: f64,
    buckets: BTreeMap<(i64, i64), Vec<u32>>,
    /// Items too big to bucket (a piece fill covers a hundred cells). Checked on
    /// every query instead — cheaper both to build and to query than smearing
    /// them across the grid, measured in `tests/perf.rs`.
    oversized: Vec<u32>,
    revision: u64,
    bounds: Option<Rect>,
}

/// Fallback cell size (mm) for a document with no measurable extent.
const MIN_CELL: f64 = 1.0;

/// Above this many cells an item goes in [`Index::oversized`] instead.
const MAX_CELLS: usize = 16;

impl Index {
    /// Builds the index for `doc` as it stands now.
    pub fn of_document(doc: &Document) -> Index {
        // One item per vertex, one per edge, two per curved edge, one fill per
        // piece: reserve it in one allocation instead of a dozen reallocs.
        let expected: usize = doc
            .pieces()
            .values()
            .map(|p| p.network.vertices.len() + p.network.edges.len() * 3 + 1)
            .sum();
        let mut items = Vec::with_capacity(expected);
        for (id, piece) in doc.pieces().iter() {
            collect(id, piece, &mut items);
        }

        let bounds = items.iter().map(|i| i.bbox).reduce(|a, b| a.union(b));
        // One cell per item on average: enough to cut the candidate set without
        // paying for a bucket per entity.
        let cell = match bounds {
            Some(b) if !items.is_empty() => ((b.width().max(1.0) * b.height().max(1.0))
                / items.len() as f64)
                .sqrt()
                .max(MIN_CELL),
            _ => MIN_CELL,
        };

        let mut buckets: BTreeMap<(i64, i64), Vec<u32>> = BTreeMap::new();
        let mut oversized = Vec::new();
        for (i, item) in items.iter().enumerate() {
            let span = cell_range(item.bbox, cell);
            if span.map_or(0, |(x0, x1, y0, y1)| {
                ((x1 - x0 + 1) * (y1 - y0 + 1)) as usize
            }) > MAX_CELLS
            {
                oversized.push(i as u32);
                continue;
            }
            for key in cells(item.bbox, cell) {
                buckets.entry(key).or_default().push(i as u32);
            }
        }

        Index {
            items,
            cell,
            buckets,
            oversized,
            revision: doc.revision(),
            bounds,
        }
    }

    /// The document revision this index was built from (guide §4.3).
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn bounds(&self) -> Option<Rect> {
        self.bounds
    }

    /// Indices of every item that could be within `radius` of `world`: the grid
    /// cells it touches, plus the oversized items whose bbox contains the query.
    fn candidates(&self, world: Point, radius: f64) -> Vec<u32> {
        let query = Rect::new(
            world.x - radius,
            world.y - radius,
            world.x + radius,
            world.y + radius,
        );
        let mut out: Vec<u32> = cells(query, self.cell)
            .filter_map(|key| self.buckets.get(&key))
            .flatten()
            .copied()
            .chain(
                self.oversized
                    .iter()
                    .copied()
                    .filter(|&i| self.items[i as usize].bbox.overlaps(query)),
            )
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The topmost target within `tolerance_px` of `screen`.
    ///
    /// The pixel tolerance becomes a world tolerance exactly once, here.
    pub fn hit_screen(
        &self,
        view: &View,
        screen: Point,
        size: Vec2,
        tolerance_px: f64,
    ) -> Option<Hit> {
        self.hit(view.screen_to_world(screen, size), tolerance_px / view.zoom)
    }

    /// The topmost target within `tolerance` **world** units of `world`.
    ///
    /// One code path with [`Index::hits`] on purpose: two copies of a
    /// priority rule are two copies that drift.
    pub fn hit(&self, world: Point, tolerance: f64) -> Option<Hit> {
        self.hits(world, tolerance).into_iter().next()
    }

    /// Every target within `tolerance`, best first: layer priority, then
    /// proximity (guide §6.2). Feeds [`Index::hit`], a
    /// cycle-through-overlapping-entities click, and the tests that need to see
    /// the losers.
    pub fn hits(&self, world: Point, tolerance: f64) -> Vec<Hit> {
        let mut out: Vec<Hit> = self
            .candidates(world, tolerance)
            .into_iter()
            .filter_map(|i| {
                let item = &self.items[i as usize];
                let (at, t, distance) = match &item.geom {
                    Geom::Dot(p) => (*p, 0.0, (*p - world).hypot()),
                    Geom::Seg(s) => {
                        let n = s.nearest(world, nguo_pattern::GEOM_ACCURACY);
                        let at = s.eval(n.t);
                        (at, n.t, (at - world).hypot())
                    }
                    // A fill is hit by containment, so its distance is zero —
                    // it must not lose a tie to a far edge of its own piece.
                    Geom::Area(path) => {
                        if !path.contains(world) {
                            return None;
                        }
                        (world, 0.0, 0.0)
                    }
                };
                (distance <= tolerance).then_some(Hit {
                    target: item.target,
                    at,
                    distance,
                    t,
                })
            })
            .collect();
        out.sort_by(|a, b| {
            b.target
                .layer()
                .cmp(&a.target.layer())
                .then(a.distance.total_cmp(&b.distance))
        });
        out
    }

    // --- snap feeds --------------------------------------------------------

    /// Vertices near `world`, in world mm. The snapper's endpoint candidates.
    pub fn vertices_near(&self, world: Point, radius: f64) -> Vec<(Target, Point)> {
        self.candidates(world, radius)
            .into_iter()
            .filter_map(|i| {
                let item = &self.items[i as usize];
                match (&item.geom, item.target) {
                    (Geom::Dot(p), t @ Target::Point { .. }) => Some((t, *p)),
                    _ => None,
                }
            })
            .collect()
    }

    /// Edge segments near `world`, in world mm. Midpoint and intersection
    /// candidates come from these.
    pub fn segments_near(&self, world: Point, radius: f64) -> Vec<(Target, PathSeg)> {
        self.candidates(world, radius)
            .into_iter()
            .filter_map(|i| {
                let item = &self.items[i as usize];
                match (&item.geom, item.target) {
                    (Geom::Seg(s), t @ Target::Edge { .. }) => Some((t, *s)),
                    _ => None,
                }
            })
            .collect()
    }
}

/// Inclusive cell bounds `(x0, x1, y0, y1)` covering `r`, or `None` for a
/// non-finite rect — which would otherwise loop to the end of `i64`.
fn cell_range(r: Rect, cell: f64) -> Option<(i64, i64, i64, i64)> {
    (cell > 0.0 && r.x0.is_finite() && r.y0.is_finite() && r.x1.is_finite() && r.y1.is_finite())
        .then(|| {
            (
                (r.x0 / cell).floor() as i64,
                (r.x1 / cell).floor() as i64,
                (r.y0 / cell).floor() as i64,
                (r.y1 / cell).floor() as i64,
            )
        })
}

/// Cell keys covering `r`.
fn cells(r: Rect, cell: f64) -> impl Iterator<Item = (i64, i64)> {
    let (x0, x1, y0, y1) = cell_range(r, cell).unwrap_or((0, -1, 0, -1));
    (x0..=x1).flat_map(move |x| (y0..=y1).map(move |y| (x, y)))
}

/// Pads a degenerate bbox (an axis-aligned line, a dot) so it still lands in a
/// cell.
fn padded(bbox: Rect) -> Rect {
    bbox.inflate(1e-6, 1e-6)
}

fn collect(piece_id: PieceId, piece: &Piece, out: &mut Vec<Item>) {
    let to_world = piece.transform;

    for (vertex, v) in piece.network.vertices.iter() {
        let p = to_world * v.pos;
        out.push(Item {
            target: Target::Point {
                piece: piece_id,
                vertex,
            },
            geom: Geom::Dot(p),
            bbox: padded(Rect::from_center_size(p, (0.0, 0.0))),
        });
    }

    for (edge, e) in piece.network.edges.iter() {
        let Ok(seg) = piece.network.seg(edge) else {
            continue;
        };
        // `t` survives the transform: an affine of a Bézier is the Bézier of the
        // transformed control points, so an edge hit's parameter is equally
        // valid in piece-local space.
        let seg = to_world * seg;
        out.push(Item {
            target: Target::Edge {
                piece: piece_id,
                edge,
            },
            geom: Geom::Seg(seg),
            bbox: padded(seg.bounding_box()),
        });

        if let EdgeGeom::Cubic { c1, c2 } = e.geom {
            for (end, c) in [(HandleEnd::Start, c1), (HandleEnd::End, c2)] {
                let p = to_world * c;
                out.push(Item {
                    target: Target::Handle {
                        piece: piece_id,
                        edge,
                        end,
                    },
                    geom: Geom::Dot(p),
                    bbox: padded(Rect::from_center_size(p, (0.0, 0.0))),
                });
            }
        }
    }

    if let Some(path) = boundary_path(piece) {
        let bbox = path.bounding_box();
        out.push(Item {
            target: Target::Fill { piece: piece_id },
            geom: Geom::Area(path),
            bbox: padded(bbox),
        });
    }
}

/// The piece outline as a closed world-space path, or `None` if the boundary
/// does not close (a piece still being drawn).
///
/// Reuses `nguo_pattern`'s boundary walk rather than guessing at an outline —
/// which is also why a concave piece gets a correct fill hit-test instead of a
/// bounding box.
pub fn boundary_path(piece: &Piece) -> Option<BezPath> {
    let start = piece
        .network
        .edges
        .iter()
        .find(|(_, e)| e.semantic.is_boundary())
        .map(|(id, _)| id)?;
    let walk = piece.network.boundary_loop(start).ok()?;
    let segs = piece.network.loop_segments(&walk).ok()?;
    let first = segs.first()?;

    let mut path = BezPath::new();
    path.move_to(first.start());
    for s in &segs {
        path.push(s.as_path_el());
    }
    path.close_path();
    Some(piece.transform * path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{square_doc, square_piece};
    use nguo_doc::Edit;
    use vello::kurbo::Affine;

    #[test]
    fn a_point_beats_an_edge_it_sits_on() {
        let (doc, piece, _) = square_doc();
        let index = Index::of_document(&doc);

        // The corner (0,0) is a vertex AND the end of two edges.
        let hit = index
            .hit(Point::new(0.4, 0.4), 2.0)
            .expect("something is hit");
        assert_eq!(hit.target.layer(), Layer::Point, "got {:?}", hit.target);
        assert_eq!(hit.target.piece(), piece);

        // Every layer is genuinely a candidate there, so the priority is doing
        // work rather than being the only answer.
        let layers: Vec<Layer> = index
            .hits(Point::new(0.4, 0.4), 2.0)
            .iter()
            .map(|h| h.target.layer())
            .collect();
        assert!(layers.contains(&Layer::Point));
        assert!(layers.contains(&Layer::Edge));
    }

    #[test]
    fn mid_edge_hits_the_edge_and_reports_the_parameter() {
        let (doc, _, edges) = square_doc();
        let index = Index::of_document(&doc);
        let hit = index.hit(Point::new(50.0, 0.6), 2.0).expect("edge hit");
        match hit.target {
            Target::Edge { edge, .. } => assert_eq!(edge, edges[0]),
            other => panic!("expected an edge, got {other:?}"),
        }
        assert!((hit.t - 0.5).abs() < 1e-6, "t = {}", hit.t);
        assert!((hit.at.y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn inside_the_piece_is_a_fill_hit_and_outside_is_nothing() {
        let (doc, piece, _) = square_doc();
        let index = Index::of_document(&doc);
        assert_eq!(
            index.hit(Point::new(50.0, 50.0), 0.5).map(|h| h.target),
            Some(Target::Fill { piece })
        );
        assert!(index.hit(Point::new(-40.0, -40.0), 0.5).is_none());
    }

    #[test]
    fn the_pixel_tolerance_scales_with_zoom() {
        let (doc, _, _) = square_doc();
        let index = Index::of_document(&doc);
        let size = Vec2::new(800.0, 600.0);

        // A pointer 6 mm (world) off the bottom edge. A fixed *pixel* tolerance
        // covers more world distance the further out you are zoomed, so the same
        // pointer catches the edge when zoomed out and misses it when zoomed in.
        // Getting this backwards is the classic tolerance bug.
        let world = Point::new(50.0, -6.0);
        for (zoom, expect_hit) in [(0.5, true), (1.0, true), (4.0, false)] {
            let view = View {
                pan: Vec2::ZERO,
                zoom,
            };
            let screen = view.world_to_screen(world, size);
            let got = index.hit_screen(&view, screen, size, 8.0).is_some();
            assert_eq!(got, expect_hit, "8 px at zoom {zoom} is {} mm", 8.0 / zoom);
        }
    }

    #[test]
    fn a_handle_is_grabbable_at_any_zoom() {
        // The regression this pins: a handle indexed in world space with a
        // world tolerance becomes unclickable when you zoom out.
        let (mut doc, piece, edges) = square_doc();
        let mut stack = nguo_doc_commands::CommandStack::new();
        stack
            .apply(
                &mut doc,
                Edit::SetEdgeGeom {
                    piece,
                    id: edges[0],
                    geom: EdgeGeom::Cubic {
                        c1: Point::new(20.0, -30.0),
                        c2: Point::new(80.0, -30.0),
                    },
                },
            )
            .unwrap();
        let index = Index::of_document(&doc);
        let size = Vec2::new(800.0, 600.0);

        let handle = Target::Handle {
            piece,
            edge: edges[0],
            end: HandleEnd::Start,
        };
        for zoom in [0.05, 1.0, 50.0] {
            let view = View {
                pan: Vec2::ZERO,
                zoom,
            };
            // Two logical px off the handle, wherever that lands in the world.
            let at = Point::new(20.0, -30.0);
            let screen = view.world_to_screen(at, size) + Vec2::new(2.0, 0.0);
            let world = view.screen_to_world(screen, size);
            let caught = index.hits(world, 6.0 / view.zoom);
            assert!(
                caught.iter().any(|h| h.target == handle),
                "the handle was not within 6 px at zoom {zoom}: {caught:?}"
            );

            // At 0.05× the entire 100 mm square is inside a 6 px (120 mm)
            // capture radius, so a *vertex* is legitimately in range too and
            // layer priority prefers it. Wherever the handle is the nearest
            // thing, it must be what a click gets.
            if zoom >= 1.0 {
                assert_eq!(
                    index.hit_screen(&view, screen, size, 6.0).map(|h| h.target),
                    Some(handle),
                    "zoom {zoom}"
                );
            }
        }
    }

    #[test]
    fn the_index_narrows_the_candidate_set() {
        // A grid that returns everything is a grid that is not working. 40
        // pieces spread out on a marker; a query near one of them must not
        // consider the others' geometry.
        let mut doc = nguo_doc::Document::new();
        let mut stack = nguo_doc_commands::CommandStack::new();
        for i in 0..40 {
            let id = doc.next_piece_id();
            let mut piece = square_piece();
            piece.transform = Affine::translate((i as f64 * 300.0, 0.0));
            stack
                .apply(
                    &mut doc,
                    Edit::AddPiece {
                        id,
                        piece: Box::new(piece),
                    },
                )
                .unwrap();
        }
        let index = Index::of_document(&doc);
        assert_eq!(index.len(), 40 * 9, "4 vertices + 4 edges + 1 fill each");
        let considered = index.candidates(Point::new(0.0, 0.0), 5.0).len();
        assert!(
            considered < index.len() / 4,
            "the grid considered {considered} of {} items",
            index.len()
        );
    }

    #[test]
    fn a_piece_transform_moves_its_geometry_into_world_space() {
        let mut doc = nguo_doc::Document::new();
        let mut stack = nguo_doc_commands::CommandStack::new();
        let id = doc.next_piece_id();
        let mut piece = square_piece();
        piece.transform = Affine::translate((1000.0, 0.0));
        stack
            .apply(
                &mut doc,
                Edit::AddPiece {
                    id,
                    piece: Box::new(piece),
                },
            )
            .unwrap();
        let index = Index::of_document(&doc);
        assert!(index.hit(Point::new(0.0, 0.0), 1.0).is_none());
        assert!(index.hit(Point::new(1000.0, 0.0), 1.0).is_some());
    }
}
