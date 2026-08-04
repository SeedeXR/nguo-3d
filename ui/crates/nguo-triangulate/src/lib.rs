// SPDX-License-Identifier: GPL-2.0-or-later
//! Pattern piece -> simulation mesh (`frontend-guide.md` §11.1, `DATAFLOW.md` §5).
//!
//! One entry point, [`triangulate`]. It takes a [`VectorNetwork`] in
//! **millimetres** and returns a [`SimMesh`] in **metres**, together with the
//! pattern-space ↔ mesh-vertex map that everything downstream needs: 3D
//! selection, seam correspondence and UVs.
//!
//! # The pipeline
//! 1. **Flatten.** Every non-`Grain` edge becomes a polyline. Curves go through
//!    [`kurbo::flatten`] at [`TriangulateOptions::curve_tolerance_mm`]; every
//!    chain is then subdivided so no link exceeds
//!    [`TriangulateOptions::target_edge_mm`]. The curve parameter `t` of every
//!    generated point is retained — that is the map.
//! 2. **Sanitise.** Snap-round to a [`TriangulateOptions::weld_mm`] grid and
//!    dedup. CDT implementations fail on near-duplicate input, not on hard
//!    geometry, so this happens *before* anything is inserted.
//! 3. **CDT.** `spade`'s constrained Delaunay triangulation, one constraint
//!    edge per polyline link. Each constraint carries the semantic of its
//!    pattern edge.
//! 4. **Refine.** `spade`'s Ruppert/Chew refinement to an angle bound and an
//!    area bound. Constraint edges are never split (see [`SimMesh::chains`]).
//! 5. **Classify.** Faces are labelled by how many outline crossings separate
//!    them from infinity; odd means inside the piece, which handles holes for
//!    free. Dart wedges are then cut out.
//! 6. **Emit.** Fabric faces only, counter-clockwise, mm -> m.
//!
//! # Why `spade`
//! Rung 5 of the ladder: a maintained CDT *and* a Ruppert-style refiner in one
//! crate, `MIT OR Apache-2.0`, three new transitive crates (`spade`, `robust`,
//! `allocator-api2`), all `MIT OR Apache-2.0` and all already on the allow-list
//! in `deny.toml`. Writing a robust CDT plus refinement by hand is a month of
//! work and a permanent liability. Of the alternatives `frontend-guide.md` §11
//! names: the `cdt` crate (0.1.0) is CDT only, with no refinement pass; CGAL
//! means an FFI build dependency and GPLv3.
//!
//! Two of `spade`'s knobs are set against their defaults for reasons that were
//! *measured*, not assumed — see the comments on `keep_constraint_edges` and
//! `exclude_outer_faces` in [`triangulate`]. Both cost real mesh quality if you
//! flip them back.
//!
//! # Units
//! **The mm -> m conversion lives here and nowhere else** (`DATAFLOW.md` §2.1).
//! [`SimMesh::positions`] is metres; [`SimMesh::pattern_mm`] is the same points
//! in millimetres, and [`SimMesh::check`] pins the ratio to [`MM_TO_M`] exactly,
//! so a second conversion site cannot hide.

#![forbid(unsafe_code)]

use std::collections::{HashMap, VecDeque};

use kurbo::{ParamCurve, ParamCurveNearest, PathEl, Point, Rect};
use nguo_pattern::{EdgeId, EdgeSemantic, NetworkError, VectorNetwork, VertexId};
use spade::{
    handles::FixedVertexHandle, AngleLimit, ConstrainedDelaunayTriangulation, Point2,
    RefinementParameters, Triangulation,
};

/// Pattern millimetres to solver metres. **The only unit conversion on this
/// path** (`DATAFLOW.md` §2.1). A second one is a 1000× garment.
pub const MM_TO_M: f64 = 1e-3;

/// Ceiling on sanitised input points, i.e. on boundary subdivision. Reached only
/// by a unit mistake — a pattern drafted in µm read as mm, or a target edge
/// orders of magnitude below the piece — so it is a typed error rather than a
/// silent 40-minute triangulation. A whole graded nest of realistic pieces at
/// 1 mm resolution is comfortably under it.
pub const MAX_INPUT_POINTS: usize = 2_000_000;

/// Quality and resolution knobs. Defaults are tuned for a garment panel at
/// interactive solver resolution; `tests/mesh.rs` measures what each costs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangulateOptions {
    /// Chord tolerance for flattening curved edges, in mm. 0.5 mm of chord
    /// error is invisible on a draped garment; the 0.01 mm cutting tolerance
    /// (`nguo_pattern::GEOM_ACCURACY`) would cost several times the boundary
    /// points for no simulation benefit.
    pub curve_tolerance_mm: f64,
    /// Target triangle edge length, in mm. Drives both boundary subdivision and
    /// the refiner's area bound, so resolution is *one* knob.
    pub target_edge_mm: f64,
    /// Snap-round grid, in mm. Two input points closer than this become one
    /// vertex. 1 µm: far below any pattern's meaningful precision, far above
    /// f64 noise.
    pub weld_mm: f64,
    /// Minimum angle the refiner aims for, in degrees. `0.0` disables
    /// angle-driven refinement. Above 30° the algorithm may not terminate and
    /// falls back on [`TriangulateOptions::max_steiner`].
    pub min_angle_deg: f64,
    /// Faces smaller than this are left alone by the refiner. Guards against
    /// over-refinement around an acute input corner.
    pub min_area_mm2: Option<f64>,
    /// Hard cap on refinement-inserted points, which is what guarantees
    /// termination. Exceeding it is reported in
    /// [`SimMesh::refinement_complete`], not as an error.
    pub max_steiner: usize,
    /// Run the refinement pass at all. `false` gives the raw CDT of the
    /// flattened, subdivided input — the cheap path.
    pub refine: bool,
}

impl Default for TriangulateOptions {
    fn default() -> Self {
        TriangulateOptions {
            curve_tolerance_mm: 0.5,
            target_edge_mm: 10.0,
            weld_mm: 1e-3,
            min_angle_deg: 30.0,
            min_area_mm2: None,
            max_steiner: 50_000,
            refine: true,
        }
    }
}

impl TriangulateOptions {
    /// Same quality, a different resolution.
    pub fn with_target_edge_mm(mut self, mm: f64) -> Self {
        self.target_edge_mm = mm;
        self
    }

    /// The raw CDT of the flattened, subdivided input — no quality pass. The
    /// cheap path, for previews.
    pub fn without_refinement(mut self) -> Self {
        self.refine = false;
        self
    }

    fn validate(&self) -> Result<(), TriangulateError> {
        for (name, value, upper) in [
            ("curve_tolerance_mm", self.curve_tolerance_mm, f64::MAX),
            ("target_edge_mm", self.target_edge_mm, f64::MAX),
            ("weld_mm", self.weld_mm, self.target_edge_mm),
        ] {
            if !(value.is_finite() && value > 0.0 && value <= upper) {
                return Err(TriangulateError::BadOption { name, value });
            }
        }
        if !(self.min_angle_deg.is_finite() && (0.0..60.0).contains(&self.min_angle_deg)) {
            return Err(TriangulateError::BadOption {
                name: "min_angle_deg",
                value: self.min_angle_deg,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TriangulateError {
    #[error(transparent)]
    Network(#[from] NetworkError),
    /// Actionable: the piece outline is not closed, so there is no inside.
    #[error(
        "the piece has no fabric region: its outline (Seam/Fold/Cut edges) does not enclose an \
         area — check for a gap in the outline or an unwelded corner"
    )]
    NoRegion,
    #[error(
        "only {distinct} distinct point(s) survived snap-rounding to {weld_mm} mm; the piece is \
         degenerate or weld_mm is too coarse"
    )]
    Degenerate { distinct: usize, weld_mm: f64 },
    #[error("notch parameter {1} on edge {0:?} is outside [0, 1]")]
    NotchOutOfRange(EdgeId, f64),
    #[error(
        "edge {0:?} crosses another pattern edge away from a shared vertex; split both edges at \
         the crossing (VectorNetwork::split_edge) and weld them first"
    )]
    CrossingConstraint(EdgeId),
    #[error("the pattern contains a non-finite coordinate")]
    NonFinite,
    #[error("spade could not insert a vertex: {0}")]
    Insertion(String),
    #[error("option {name} is out of range: {value}")]
    BadOption { name: &'static str, value: f64 },
    #[error(
        "the piece needs more than {limit} boundary points at a {target_edge_mm} mm target edge; \
         raise target_edge_mm, or check the piece is really {span_mm} mm across (the document unit \
         is millimetres)"
    )]
    TooDense {
        limit: usize,
        target_edge_mm: f64,
        span_mm: f64,
    },
    /// An internal invariant broke. Never expected; reported rather than
    /// panicked so a bad mesh cannot reach the solver unnoticed.
    #[error("internal error: the triangulation violated its own invariants: {0}")]
    Invalid(String),
}

/// Where a mesh vertex came from in pattern space. Together with
/// [`SimMesh::chains`] this is the pattern-space ↔ mesh-vertex map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VertexSource {
    /// Coincides with a network vertex — a piece corner, a dart apex.
    Vertex(VertexId),
    /// Inserted along a pattern edge: curve flattening, resolution
    /// subdivision, or a notch.
    Edge(EdgeId),
    /// A Steiner point added by refinement. Interior to the piece.
    Interior,
}

/// The mesh vertices lying on one pattern edge, in order along the edge.
///
/// This is what seam correspondence (M6) walks: a seam side is a chain of
/// pattern edges, and pulling one seam onto another means pairing these runs.
/// `t = 0.0` and `t = 1.0` are always present, so consecutive edges in a seam
/// side share their end vertices.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EdgeChain {
    pub edge: EdgeId,
    pub semantic: EdgeSemantic,
    /// `(mesh vertex index, curve parameter)`, ascending in `t`.
    pub points: Vec<(u32, f64)>,
}

/// A pattern piece as the solver wants it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimMesh {
    /// **Metres**, in the piece's own plane: pattern `+x` -> `+X`, pattern `+y`
    /// -> `+Y` (up), `z = 0`. Arrangement (M6) places and rotates it; no axis
    /// conversion happens here or there — nguo world space *is* glTF space
    /// (`DATAFLOW.md` §2.2).
    pub positions: Vec<[f32; 3]>,
    /// Counter-clockwise in pattern space, so the face normal is `+Z`.
    pub indices: Vec<[u32; 3]>,
    /// **Millimetres**: the same vertices in pattern space. The map. Also the
    /// UV source — see [`SimMesh::uvs`].
    pub pattern_mm: Vec<[f64; 2]>,
    pub sources: Vec<VertexSource>,
    /// One entry per pattern edge that contributed geometry, ascending by edge
    /// id. `Grain` edges are markers and never appear.
    pub chains: Vec<EdgeChain>,
    /// `false` if refinement hit [`TriangulateOptions::max_steiner`]; the mesh
    /// is still valid, just unevenly refined in places.
    pub refinement_complete: bool,
    /// Zero-area faces dropped on emit. Non-zero means the input had exactly
    /// collinear points; the mesh is still watertight because the dropped area
    /// is zero.
    pub degenerate_dropped: usize,
}

impl SimMesh {
    /// UVs, from the pattern coordinates normalised into `0..=1` by the piece's
    /// pattern-space bounding box.
    ///
    /// There is no unwrapping step and there never will be: a garment is
    /// developable from its pattern by construction, so the flat coordinates
    /// *are* the texture coordinates (`DATAFLOW.md` §5.1).
    pub fn uvs(&self) -> Vec<[f32; 2]> {
        let b = self.pattern_bounds();
        let w = b.width().max(f64::MIN_POSITIVE);
        let h = b.height().max(f64::MIN_POSITIVE);
        self.pattern_mm
            .iter()
            .map(|&[x, y]| [((x - b.x0) / w) as f32, ((y - b.y0) / h) as f32])
            .collect()
    }

    /// Bounding box of the piece in pattern millimetres.
    pub fn pattern_bounds(&self) -> Rect {
        self.pattern_mm.iter().fold(
            Rect::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN),
            |r, &[x, y]| Rect::new(r.x0.min(x), r.y0.min(y), r.x1.max(x), r.y1.max(y)),
        )
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// The chain for one pattern edge, if that edge contributed geometry.
    pub fn chain(&self, edge: EdgeId) -> Option<&EdgeChain> {
        self.chains
            .binary_search_by_key(&edge, |c| c.edge)
            .ok()
            .map(|i| &self.chains[i])
    }

    /// Mesh vertices on the piece outline (`Seam`/`Fold`/`Cut` edges), sorted
    /// and deduplicated. What sewing constraints attach to.
    pub fn outline_vertices(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .chains
            .iter()
            .filter(|c| c.semantic.is_boundary())
            .flat_map(|c| c.points.iter().map(|&(i, _)| i))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Signed area of triangle `i` in pattern mm². Positive means
    /// counter-clockwise. The mesh invariant is that every one of these is
    /// strictly positive.
    pub fn signed_area_mm2(&self, i: usize) -> f64 {
        let [a, b, c] = self.indices[i].map(|k| self.pattern_mm[k as usize]);
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    }

    /// Total fabric area in mm².
    pub fn area_mm2(&self) -> f64 {
        (0..self.indices.len())
            .map(|i| self.signed_area_mm2(i))
            .sum()
    }

    /// Smallest interior angle of each triangle, in degrees, in triangle order.
    ///
    /// Report a percentile of this, not just the minimum: Ruppert refinement
    /// provably cannot open up an angle between two *input* edges, and a dart
    /// apex is a 7° wedge by design. One 3° sliver at the apex says nothing
    /// about the rest of the mesh.
    pub fn min_angles_deg(&self) -> Vec<f64> {
        self.indices
            .iter()
            .map(|tri| {
                let p = tri.map(|k| self.pattern_mm[k as usize]);
                (0..3)
                    .map(|k| {
                        let (o, a, b) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
                        let u = [a[0] - o[0], a[1] - o[1]];
                        let v = [b[0] - o[0], b[1] - o[1]];
                        let cross = u[0] * v[1] - u[1] * v[0];
                        let dot = u[0] * v[0] + u[1] * v[1];
                        cross.abs().atan2(dot).to_degrees()
                    })
                    .fold(180.0f64, f64::min)
            })
            .collect()
    }

    /// Smallest interior angle anywhere in the mesh, in degrees.
    pub fn min_angle_deg(&self) -> f64 {
        self.min_angles_deg().into_iter().fold(180.0, f64::min)
    }

    /// The `q`-quantile of [`SimMesh::min_angles_deg`], e.g. `0.01` for "1% of
    /// triangles are worse than this". The mesh-quality number to assert on.
    pub fn angle_quantile_deg(&self, q: f64) -> f64 {
        let mut a = self.min_angles_deg();
        if a.is_empty() {
            return 0.0;
        }
        a.sort_by(f64::total_cmp);
        a[((q.clamp(0.0, 1.0) * (a.len() - 1) as f64).round() as usize).min(a.len() - 1)]
    }

    /// Largest triangle area in mm² — what the refiner's area bound controls.
    pub fn max_area_mm2(&self) -> f64 {
        (0..self.indices.len())
            .map(|i| self.signed_area_mm2(i))
            .fold(0.0, f64::max)
    }

    /// Every invariant this crate promises, as one check. O(triangles), cheap
    /// enough for tests and for the `debug_assert` at the end of
    /// [`triangulate`].
    ///
    /// Returns the offending description rather than panicking, so a caller can
    /// surface it.
    pub fn check(&self) -> Result<(), String> {
        let n = self.positions.len();
        if n != self.pattern_mm.len() || n != self.sources.len() {
            return Err(format!(
                "vertex arrays disagree: {n} positions, {} pattern_mm, {} sources",
                self.pattern_mm.len(),
                self.sources.len()
            ));
        }
        if self.indices.is_empty() {
            return Err("no triangles".into());
        }
        for (i, (p, m)) in self.positions.iter().zip(&self.pattern_mm).enumerate() {
            if !p.iter().all(|c| c.is_finite()) || !m.iter().all(|c| c.is_finite()) {
                return Err(format!("vertex {i} is not finite"));
            }
            if p[2] != 0.0 {
                return Err(format!("vertex {i} left the pattern plane: z = {}", p[2]));
            }
            // The single-conversion-site guard: positions must be pattern_mm
            // scaled by MM_TO_M and by nothing else.
            for (axis, (&got, &mm)) in p.iter().zip(m.iter()).enumerate() {
                let want = (mm * MM_TO_M) as f32;
                if got != want {
                    return Err(format!(
                        "vertex {i} axis {axis}: {got} m but pattern is {mm} mm (expected \
                         {want}); a second unit conversion has crept in"
                    ));
                }
            }
        }
        for (i, tri) in self.indices.iter().enumerate() {
            if tri.iter().any(|&k| k as usize >= n) {
                return Err(format!("triangle {i} indexes a missing vertex: {tri:?}"));
            }
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                return Err(format!("triangle {i} repeats a vertex: {tri:?}"));
            }
            let a = self.signed_area_mm2(i);
            if a <= 0.0 {
                return Err(format!(
                    "triangle {i} is flipped or degenerate: area {a} mm²"
                ));
            }
        }
        for c in &self.chains {
            if c.points.len() < 2 {
                return Err(format!(
                    "chain for {:?} has {} point(s)",
                    c.edge,
                    c.points.len()
                ));
            }
            if c.points.windows(2).any(|w| w[0].1 >= w[1].1) {
                return Err(format!("chain for {:?} is not ascending in t", c.edge));
            }
            if c.points.iter().any(|&(i, _)| i as usize >= n) {
                return Err(format!("chain for {:?} indexes a missing vertex", c.edge));
            }
        }
        if self.chains.windows(2).any(|w| w[0].edge >= w[1].edge) {
            return Err("chains are not sorted by edge id".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

/// What crossing an edge means to region classification. `Ord` is the priority
/// used when two pattern edges land on the same CDT edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
enum EdgeKind {
    /// Not a pattern edge: an ordinary Delaunay edge.
    #[default]
    Free,
    /// An `Internal` pattern edge. Present in the mesh, but it does not divide
    /// the fabric — you can walk around its ends.
    Internal,
    /// A dart leg. Divides the fabric: the wedge behind it is sewn out.
    Dart,
    /// The piece outline. Crossing it flips inside/outside.
    Outline,
}

impl EdgeKind {
    fn of(semantic: EdgeSemantic) -> Option<EdgeKind> {
        match semantic {
            EdgeSemantic::Seam | EdgeSemantic::Fold | EdgeSemantic::Cut => Some(EdgeKind::Outline),
            EdgeSemantic::DartLeg => Some(EdgeKind::Dart),
            EdgeSemantic::Internal => Some(EdgeKind::Internal),
            // A grain marker is a direction, not geometry. It may even lie
            // outside the piece, so constraining it would drag stray vertices
            // into the mesh.
            EdgeSemantic::Grain => None,
        }
    }
}

type Cdt = ConstrainedDelaunayTriangulation<Point2<f64>, (), EdgeKind, ()>;

/// A pattern edge flattened and snapped: `(vertex index, t)` ascending in `t`.
struct Chain {
    edge: EdgeId,
    semantic: EdgeSemantic,
    kind: EdgeKind,
    points: Vec<(usize, f64)>,
}

/// Flatten + subdivide one edge into `(t, point)` samples, `t` ascending, with
/// `0.0` and `1.0` included.
fn samples(
    net: &VectorNetwork,
    edge: EdgeId,
    notches: &[f64],
    opts: &TriangulateOptions,
) -> Result<Vec<(f64, Point)>, TriangulateError> {
    let seg = net.seg(edge)?;
    if !seg.is_finite() {
        return Err(TriangulateError::NonFinite);
    }

    let mut ts: Vec<f64> = vec![0.0, 1.0];
    if !matches!(seg, kurbo::PathSeg::Line(_)) {
        // kurbo's adaptive flattener, then recover each point's parameter. The
        // points are exactly on the curve, so `nearest` is well conditioned.
        // ponytail: reusing kurbo rather than writing a subdivider. The
        // `nearest` calls are microseconds and there are a few hundred per piece.
        let path = [PathEl::MoveTo(seg.start()), seg.as_path_el()];
        kurbo::flatten(path, opts.curve_tolerance_mm, |el| {
            if let PathEl::LineTo(p) = el {
                ts.push(seg.nearest(p, 1e-9).t);
            }
        });
    }
    for &t in notches {
        if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
            return Err(TriangulateError::NotchOutOfRange(edge, t));
        }
        ts.push(t);
    }
    ts.retain(|t| t.is_finite());
    for t in &mut ts {
        *t = t.clamp(0.0, 1.0);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    ts.dedup();

    // Resolution: no link longer than target_edge_mm. Splitting in t rather
    // than arc length is fine — these are quality knobs, not geometry — and it
    // keeps t monotone, which seam correspondence relies on.
    let mut out: Vec<(f64, Point)> = Vec::with_capacity(ts.len() * 2);
    for w in ts.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let (p0, p1) = (seg.eval(t0), seg.eval(t1));
        out.push((t0, p0));
        // Clamp before the cast, not after: `f64 as usize` saturates, so a
        // pattern accidentally drafted in µm-as-mm (or a 1 mm target edge on a
        // 10 km edge) would otherwise ask for usize::MAX subdivisions and hang.
        // The total is checked against `MAX_INPUT_POINTS` by the caller.
        let extra = ((p1 - p0).hypot() / opts.target_edge_mm)
            .ceil()
            .min(MAX_INPUT_POINTS as f64) as usize;
        for k in 1..extra.max(1) {
            let t = t0 + (t1 - t0) * (k as f64 / extra as f64);
            out.push((t, seg.eval(t)));
        }
    }
    out.push((1.0, seg.eval(1.0)));
    if out.iter().any(|(_, p)| !p.is_finite()) {
        return Err(TriangulateError::NonFinite);
    }
    Ok(out)
}

/// Snap-rounding grid key. `round`, not a truncating cast: two points either
/// side of a grid line must not straddle it.
fn key(p: Point, weld: f64) -> (i64, i64) {
    ((p.x / weld).round() as i64, (p.y / weld).round() as i64)
}

/// Insert the sanitised points and their constraint edges. Separate from
/// [`triangulate`] because the refinement fallback has to rebuild it.
fn build_cdt(positions: &[Point], chains: &[Chain]) -> Result<Cdt, TriangulateError> {
    let mut cdt = Cdt::new();
    let mut handles: Vec<FixedVertexHandle> = Vec::with_capacity(positions.len());
    for p in positions {
        handles.push(
            cdt.insert(Point2::new(p.x, p.y))
                .map_err(|e| TriangulateError::Insertion(e.to_string()))?,
        );
    }
    for c in chains {
        for w in c.points.windows(2) {
            let (a, b) = (handles[w[0].0], handles[w[1].0]);
            let added = cdt.try_add_constraint(a, b);
            if added.is_empty() && !cdt.exists_constraint(a, b) {
                // `try_add_constraint` refuses rather than panicking when a new
                // constraint would cross an existing one. That is a pattern
                // error the user can fix, so say so.
                return Err(TriangulateError::CrossingConstraint(c.edge));
            }
            for e in added {
                let slot = cdt.undirected_edge_data_mut(e.as_undirected()).data_mut();
                *slot = (*slot).max(c.kind);
            }
        }
    }
    Ok(cdt)
}

/// Triangulate one pattern piece.
///
/// `net` is a single piece's network in millimetres, in its own local space
/// (`Piece::transform` is a canvas placement and is deliberately not applied —
/// `DATAFLOW.md` §4.1). `notches` are `(edge, t)` pairs from `Feature::Notch`;
/// each becomes a mesh vertex exactly on that edge so seam correspondence can
/// pin to it.
///
/// # Errors
/// Every failure is a typed, actionable [`TriangulateError`] — nothing here
/// panics on bad geometry, because a user drafting a piece will hand it an open
/// outline several times a minute.
pub fn triangulate(
    net: &VectorNetwork,
    notches: &[(EdgeId, f64)],
    opts: &TriangulateOptions,
) -> Result<SimMesh, TriangulateError> {
    opts.validate()?;

    let mut by_edge: HashMap<EdgeId, Vec<f64>> = HashMap::new();
    for &(e, t) in notches {
        by_edge.entry(e).or_default().push(t);
    }

    // Cost is set by boundary subdivision, and the arc lengths are cheap to sum,
    // so refuse an impossible job before allocating for it rather than after.
    let boundary_mm: f64 = net
        .edges
        .iter()
        .filter(|(_, e)| EdgeKind::of(e.semantic).is_some())
        .map(|(id, _)| net.arc_length(id).unwrap_or(0.0))
        .sum();
    if boundary_mm / opts.target_edge_mm > MAX_INPUT_POINTS as f64 {
        let b = net.bounds().unwrap_or_default();
        return Err(TriangulateError::TooDense {
            limit: MAX_INPUT_POINTS,
            target_edge_mm: opts.target_edge_mm,
            span_mm: b.width().max(b.height()),
        });
    }

    // --- 1 & 2: flatten, snap-round, dedup -------------------------------
    let mut positions: Vec<Point> = Vec::new();
    let mut sources: Vec<VertexSource> = Vec::new();
    let mut index: HashMap<(i64, i64), usize> = HashMap::new();
    let mut chains: Vec<Chain> = Vec::new();

    // The arena iterates in id order, so this loop and everything keyed off it
    // is deterministic.
    for (edge, e) in net.edges.iter() {
        let Some(kind) = EdgeKind::of(e.semantic) else {
            continue;
        };
        let notches = by_edge.get(&edge).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut points: Vec<(usize, f64)> = Vec::new();
        for (t, p) in samples(net, edge, notches, opts)? {
            let k = key(p, opts.weld_mm);
            let snapped = Point::new(k.0 as f64 * opts.weld_mm, k.1 as f64 * opts.weld_mm);
            let source = if t == 0.0 {
                VertexSource::Vertex(e.start)
            } else if t == 1.0 {
                VertexSource::Vertex(e.end)
            } else {
                VertexSource::Edge(edge)
            };
            let slot = *index.entry(k).or_insert_with(|| {
                positions.push(snapped);
                sources.push(source);
                positions.len() - 1
            });
            // A corner shared by two edges must read as a network vertex, not
            // as whichever edge reached it first.
            if matches!(source, VertexSource::Vertex(_)) {
                sources[slot] = source;
            }
            // Snapping can collapse two samples into one; a zero-length
            // constraint is not a thing, so drop the repeat.
            if points.last().map(|&(i, _)| i) != Some(slot) {
                points.push((slot, t));
            }
        }
        if points.len() >= 2 {
            chains.push(Chain {
                edge,
                semantic: e.semantic,
                kind,
                points,
            });
        }
    }

    if positions.len() < 3 {
        return Err(TriangulateError::Degenerate {
            distinct: positions.len(),
            weld_mm: opts.weld_mm,
        });
    }
    if positions.len() > MAX_INPUT_POINTS {
        let b = net.bounds().unwrap_or_default();
        return Err(TriangulateError::TooDense {
            limit: MAX_INPUT_POINTS,
            target_edge_mm: opts.target_edge_mm,
            span_mm: b.width().max(b.height()),
        });
    }

    // --- 3: constrained Delaunay ----------------------------------------
    let mut cdt = build_cdt(&positions, &chains)?;

    // --- 4: Ruppert-style refinement ------------------------------------
    // Equilateral triangle of side `target_edge_mm`. One resolution knob.
    let max_area = 3.0f64.sqrt() / 4.0 * opts.target_edge_mm * opts.target_edge_mm;
    let mut params = RefinementParameters::<f64>::new()
        .with_angle_limit(AngleLimit::from_deg(opts.min_angle_deg))
        .with_max_allowed_area(max_area)
        .with_max_additional_vertices(opts.max_steiner)
        // Constraint edges are never split. Two reasons, in order of weight:
        //  1. Every boundary vertex keeps the exact `(edge, t)` it was created
        //     with, which is what makes `SimMesh::chains` — the whole point of
        //     this milestone — trustworthy. A split point sits on the *chord*,
        //     so its curve parameter would be a reconstruction, not a fact.
        //  2. Splitting also destroys this crate's `EdgeKind` tag: the two
        //     halves of a split edge get `UE::default()`, and region
        //     classification then loses the outline.
        // The price is that refinement cannot fix the boundary itself, so
        // boundary resolution is set by subdivision in `samples` instead.
        // MEASURED: with the boundary pre-subdivided to `target_edge_mm` this
        // costs nothing — a 10 mm-subdivided concave panel refines 44 -> 169
        // vertices with this flag against 44 -> 244 without. Without the
        // pre-subdivision it is fatal (5 -> 5), which is why the two belong
        // together.
        .keep_constraint_edges();
    // NOT `exclude_outer_faces(true)`, deliberately. spade's own inner/outer
    // winding test is wrong for dart topology: on a panel with a dart it
    // classifies *every* face as outer (MEASURED: 64 of 64), so refinement
    // becomes a no-op and the mesh stays at its input resolution. Refining the
    // dead space between the outline and the convex hull instead costs ~20%
    // extra vertices, all of which classification below discards. Correct and
    // slightly wasteful beats fast and unrefined.
    if let Some(a) = opts.min_area_mm2 {
        params = params.with_min_required_area(a);
    }
    let mut refinement_complete = true;
    if opts.refine {
        // `spade::refine` has a reachable `panic!("Failed to locate position.
        // This is a bug in spade.")` (spade-2.15.1
        // delaunay_core/triangulation_ext.rs:562) that fires on extreme
        // geometry — a sliver whose circumcenter lands absurdly far from the
        // triangulation. Found by the property test below, not by review. A
        // panic here would kill the app while a user drags a point, so catch it
        // and fall back to the unrefined CDT: rebuilt from scratch, because a
        // triangulation abandoned mid-refinement cannot be trusted. Refinement
        // is a quality pass, so losing it degrades the mesh instead of the
        // session.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cdt.refine(params))) {
            Ok(r) => refinement_complete = r.refinement_complete,
            Err(_) => {
                tracing::warn!(
                    "spade panicked during refinement; falling back to the unrefined \
                     triangulation for this piece"
                );
                cdt = build_cdt(&positions, &chains)?;
                refinement_complete = false;
            }
        }
        if !refinement_complete {
            tracing::warn!(
                max_steiner = opts.max_steiner,
                "triangulation refinement did not complete; the mesh is valid but unevenly refined"
            );
        }
    }

    // --- 5: classify ----------------------------------------------------
    let faces = collect_faces(&cdt);
    let inside: Vec<bool> = outline_depth(&faces).iter().map(|d| d % 2 == 1).collect();
    let keep = largest_dart_component(&faces, &inside);

    // --- 6: emit --------------------------------------------------------
    let n_input = positions.len();
    let mut remap: HashMap<usize, u32> = HashMap::new();
    let mut mesh = SimMesh {
        positions: Vec::new(),
        indices: Vec::new(),
        pattern_mm: Vec::new(),
        sources: Vec::new(),
        chains: Vec::new(),
        refinement_complete,
        degenerate_dropped: 0,
    };
    for (f, &k) in faces.iter().zip(&keep) {
        if !k {
            continue;
        }
        // `Face::area` is spade's, which is **absolute**, so this drops only
        // exactly-degenerate faces. That is deliberate: spade guarantees inner
        // faces are counter-clockwise, so a *negative* signed area would be an
        // upstream bug, and it should surface as the loud `Invalid` error from
        // `check` below rather than be quietly patched into a hole here.
        if f.area <= 0.0 {
            mesh.degenerate_dropped += 1;
            continue;
        }
        let tri = f.verts.map(|v| {
            *remap.entry(v).or_insert_with(|| {
                let p = cdt.vertex(FixedVertexHandle::from_index(v)).position();
                mesh.pattern_mm.push([p.x, p.y]);
                mesh.positions
                    .push([(p.x * MM_TO_M) as f32, (p.y * MM_TO_M) as f32, 0.0]);
                mesh.sources.push(if v < n_input {
                    sources[v]
                } else {
                    VertexSource::Interior
                });
                (mesh.pattern_mm.len() - 1) as u32
            })
        });
        mesh.indices.push(tri);
    }
    if mesh.indices.is_empty() {
        return Err(TriangulateError::NoRegion);
    }
    for c in chains {
        // A chain point can be absent if its edge strayed outside the fabric
        // region (a stray Internal line). Report what survived, not a lie.
        let points: Vec<(u32, f64)> = c
            .points
            .iter()
            .filter_map(|&(v, t)| Some((*remap.get(&v)?, t)))
            .collect();
        if points.len() >= 2 {
            mesh.chains.push(EdgeChain {
                edge: c.edge,
                semantic: c.semantic,
                points,
            });
        }
    }
    // Always, not `debug_assert`. MEASURED (release, bodice fixture, 1332
    // triangles): 0.003 ms against 0.398 ms of triangulation — 0.8%. What it
    // buys is that a flipped triangle can never reach the solver, where it shows
    // up as an inexplicable drape rather than as an error anyone can trace.
    // `profile_triangulation` re-measures the ratio and fails above 25%.
    mesh.check().map_err(TriangulateError::Invalid)?;
    Ok(mesh)
}

/// One inner CDT face, flattened out of the DCEL so classification is plain
/// array work instead of handle chasing.
struct Face {
    verts: [usize; 3],
    /// Neighbour slot across each of the three edges, and what that edge is.
    across: [(Option<usize>, EdgeKind); 3],
    area: f64,
}

fn collect_faces(cdt: &Cdt) -> Vec<Face> {
    let mut slot = vec![usize::MAX; cdt.num_all_faces()];
    for (i, f) in cdt.inner_faces().enumerate() {
        slot[f.fix().index()] = i;
    }
    cdt.inner_faces()
        .map(|f| Face {
            verts: f.vertices().map(|v| v.fix().index()),
            across: f.adjacent_edges().map(|e| {
                let n = e.rev().face().as_inner().map(|n| slot[n.fix().index()]);
                (n, *e.as_undirected().data().data())
            }),
            area: f.area(),
        })
        .collect()
}

/// How many outline crossings separate each face from infinity, by 0-1 BFS from
/// the convex hull. Odd = inside the piece.
///
/// This is the winding-parity test done on the dual graph: for properly nested
/// closed loops the minimum crossing count has the same parity as the winding
/// number, so holes fall out for free and an open `Internal` line — which you
/// can walk around — costs nothing.
fn outline_depth(faces: &[Face]) -> Vec<u32> {
    let mut depth = vec![u32::MAX; faces.len()];
    let mut q: VecDeque<usize> = VecDeque::new();
    // Seed: every face with an edge on the convex hull, i.e. an edge whose
    // other side is the outer face. A hull edge that is itself part of the
    // outline already costs one crossing.
    for (i, f) in faces.iter().enumerate() {
        let d = f
            .across
            .iter()
            .filter(|(n, _)| n.is_none())
            .map(|&(_, k)| u32::from(k == EdgeKind::Outline))
            .min();
        if let Some(d) = d {
            if depth[i] > d {
                depth[i] = d;
                // The deque must stay sorted by depth for 0-1 BFS to be correct,
                // which it is: every seed is 0 or 1, and 0s go to the front.
                if d == 0 {
                    q.push_front(i);
                } else {
                    q.push_back(i);
                }
            }
        }
    }
    while let Some(i) = q.pop_front() {
        let d = depth[i];
        for &(n, kind) in &faces[i].across {
            let Some(j) = n else { continue };
            let w = u32::from(kind == EdgeKind::Outline);
            if depth[j] > d + w {
                depth[j] = d + w;
                if w == 0 {
                    q.push_front(j);
                } else {
                    q.push_back(j);
                }
            }
        }
    }
    depth
}

/// Split the inside faces into components across dart legs and keep the largest
/// by area — the fabric. The others are dart wedges: the triangle of cloth a
/// dart folds out, which must not be simulated.
///
/// ponytail: largest-by-area, not a full dart-topology analysis. A dart wedge is
/// a sliver next to a panel by construction, and a closed (fisheye) dart is a
/// lens inside one. If a piece ever legitimately splits into two comparable
/// fabric regions across a dart leg, this picks one — upgrade it then to
/// "discard only components bounded entirely by dart legs and outline".
fn largest_dart_component(faces: &[Face], inside: &[bool]) -> Vec<bool> {
    let mut comp = vec![usize::MAX; faces.len()];
    let mut areas: Vec<f64> = Vec::new();
    for start in 0..faces.len() {
        if !inside[start] || comp[start] != usize::MAX {
            continue;
        }
        let id = areas.len();
        let mut area = 0.0;
        let mut stack = vec![start];
        comp[start] = id;
        while let Some(i) = stack.pop() {
            area += faces[i].area;
            for &(n, kind) in &faces[i].across {
                let Some(j) = n else { continue };
                if inside[j] && comp[j] == usize::MAX && kind != EdgeKind::Dart {
                    comp[j] = id;
                    stack.push(j);
                }
            }
        }
        areas.push(area);
    }
    match areas
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
    {
        None => vec![false; faces.len()],
        Some(best) => comp.iter().map(|&c| c == best).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_pattern::EdgeGeom;
    use std::collections::HashSet;

    /// A `size` mm square of `Cut` edges. Corners in CCW order.
    fn square(size: f64) -> (VectorNetwork, Vec<VertexId>, Vec<EdgeId>) {
        let mut n = VectorNetwork::new();
        let v: Vec<VertexId> = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        let e: Vec<EdgeId> = (0..4)
            .map(|i| {
                n.add_edge(v[i], v[(i + 1) % 4], EdgeGeom::Line, EdgeSemantic::Cut)
                    .unwrap()
            })
            .collect();
        (n, v, e)
    }

    fn mesh_edges(m: &SimMesh) -> HashSet<(u32, u32)> {
        let mut edges = HashSet::new();
        for t in &m.indices {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                edges.insert((a.min(b), a.max(b)));
            }
        }
        edges
    }

    #[test]
    fn a_square_triangulates_to_a_valid_mesh() {
        let (n, _, _) = square(100.0);
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        assert!(
            (m.area_mm2() - 10_000.0).abs() < 1e-6,
            "area {} mm²",
            m.area_mm2()
        );
        assert!(m.refinement_complete);
        assert_eq!(m.degenerate_dropped, 0);
    }

    #[test]
    fn the_square_has_the_euler_characteristic_of_a_disc() {
        let (n, _, _) = square(100.0);
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        // V - E + F = 1 for a triangulated disc.
        let (v, e, f) = (
            m.vertex_count() as i64,
            mesh_edges(&m).len() as i64,
            m.triangle_count() as i64,
        );
        assert_eq!(v - e + f, 1, "V={v} E={e} F={f}");
    }

    #[test]
    fn millimetres_become_metres_exactly_once() {
        let (n, _, _) = square(1000.0);
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        let b = m.pattern_bounds();
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (0.0, 0.0, 1000.0, 1000.0));
        let max_x = m.positions.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let max_y = m.positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        assert_eq!((max_x, max_y), (1.0, 1.0), "1000 mm must be 1.0 m");
    }

    #[test]
    fn pattern_coordinates_are_the_uvs() {
        let (n, _, _) = square(250.0);
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        let uvs = m.uvs();
        assert_eq!(uvs.len(), m.vertex_count());
        for (uv, mm) in uvs.iter().zip(&m.pattern_mm) {
            assert!((uv[0] as f64 - mm[0] / 250.0).abs() < 1e-6);
            assert!((uv[1] as f64 - mm[1] / 250.0).abs() < 1e-6);
        }
    }

    #[test]
    fn an_internal_line_survives_as_mesh_edges() {
        let (mut n, _, e) = square(100.0);
        // Split the left and right edges at mid-height, then run a style line
        // between the two new vertices: a T-junction at both ends.
        let right = n.split_edge(e[1], 0.5).unwrap();
        let left = n.split_edge(e[3], 0.5).unwrap();
        let line = n
            .add_edge(
                left.vertex,
                right.vertex,
                EdgeGeom::Line,
                EdgeSemantic::Internal,
            )
            .unwrap();

        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();

        let chain = m.chain(line).expect("the internal line must be mapped");
        assert_eq!(chain.semantic, EdgeSemantic::Internal);
        assert_eq!(chain.points.len(), 11, "100 mm at a 10 mm target edge");

        // Every consecutive pair in the chain must be an actual mesh edge.
        let edges = mesh_edges(&m);
        for w in chain.points.windows(2) {
            let (x, y) = (w[0].0, w[1].0);
            assert!(
                edges.contains(&(x.min(y), x.max(y))),
                "internal line segment {x}-{y} is not a mesh edge"
            );
        }
        // The line does not cut the piece in two: area is still the full square.
        assert!((m.area_mm2() - 10_000.0).abs() < 1e-6, "{}", m.area_mm2());
    }

    #[test]
    fn a_notch_becomes_a_mesh_vertex_on_its_edge() {
        let (n, _, e) = square(100.0);
        let m = triangulate(&n, &[(e[0], 0.37)], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        let chain = m.chain(e[0]).unwrap();
        let (idx, t) = *chain
            .points
            .iter()
            .find(|&&(_, t)| (t - 0.37).abs() < 1e-12)
            .expect("the notch parameter must appear in the chain");
        assert_eq!(t, 0.37);
        assert_eq!(m.pattern_mm[idx as usize], [37.0, 0.0]);
        assert_eq!(m.sources[idx as usize], VertexSource::Edge(e[0]));
    }

    #[test]
    fn a_notch_outside_the_unit_interval_is_an_error() {
        let (n, _, e) = square(100.0);
        assert_eq!(
            triangulate(&n, &[(e[0], 1.5)], &TriangulateOptions::default()),
            Err(TriangulateError::NotchOutOfRange(e[0], 1.5))
        );
    }

    #[test]
    fn an_open_outline_is_reported_not_panicked() {
        let (mut n, _, e) = square(100.0);
        n.remove_edge(e[2]).unwrap();
        assert_eq!(
            triangulate(&n, &[], &TriangulateOptions::default()),
            Err(TriangulateError::NoRegion)
        );
    }

    #[test]
    fn near_duplicate_input_is_welded_not_fatal() {
        let (mut n, _, _) = square(100.0);
        // A second corner a ten-thousandth of a micron from the first, joined by
        // an internal spur: raw CDT input that is a near-degenerate triangle.
        let ghost = n.add_vertex(Point::new(1e-7, 1e-7));
        let apex = n.add_vertex(Point::new(50.0, 50.0));
        n.add_edge(ghost, apex, EdgeGeom::Line, EdgeSemantic::Internal)
            .unwrap();
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        assert!((m.area_mm2() - 10_000.0).abs() < 1e-3, "{}", m.area_mm2());
    }

    #[test]
    fn a_grain_marker_contributes_nothing() {
        let (mut n, _, _) = square(100.0);
        // Deliberately outside the piece: a grain edge must not drag the mesh
        // out with it.
        let a = n.add_vertex(Point::new(-500.0, -500.0));
        let b = n.add_vertex(Point::new(-400.0, -500.0));
        let g = n
            .add_edge(a, b, EdgeGeom::Line, EdgeSemantic::Grain)
            .unwrap();
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        assert!(m.chain(g).is_none());
        assert!((m.area_mm2() - 10_000.0).abs() < 1e-6, "{}", m.area_mm2());
        assert!(m.pattern_bounds().x0 >= 0.0);
    }

    #[test]
    fn a_curved_edge_is_flattened_within_tolerance() {
        let mut n = VectorNetwork::new();
        let v: Vec<VertexId> = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        // Bulge the bottom edge out into a cubic.
        let curved = n
            .add_edge(
                v[0],
                v[1],
                EdgeGeom::Cubic {
                    c1: Point::new(30.0, -40.0),
                    c2: Point::new(70.0, -40.0),
                },
                EdgeSemantic::Cut,
            )
            .unwrap();
        for i in 1..4 {
            n.add_edge(v[i], v[(i + 1) % 4], EdgeGeom::Line, EdgeSemantic::Cut)
                .unwrap();
        }
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        let chain = m.chain(curved).unwrap();
        // Chord error of every link against the true curve.
        let seg = n.seg(curved).unwrap();
        for w in chain.points.windows(2) {
            let a = m.pattern_mm[w[0].0 as usize];
            let b = m.pattern_mm[w[1].0 as usize];
            let mid = Point::new((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0);
            let d = (seg.eval(seg.nearest(mid, 1e-9).t) - mid).hypot();
            assert!(d < 0.6, "chord error {d} mm exceeds the 0.5 mm tolerance");
        }
        // The bulge dips below y = 0, so the mesh must reach there.
        assert!(m.pattern_bounds().y0 < -25.0, "{:?}", m.pattern_bounds());
    }

    #[test]
    fn refinement_bounds_the_triangle_area() {
        // A 200 mm square subdivided at 50 mm: the raw CDT is a handful of huge
        // triangles, which is exactly what the area bound exists to fix.
        let (n, _, _) = square(200.0);
        let opts = TriangulateOptions::default().with_target_edge_mm(50.0);
        let cap = 3.0f64.sqrt() / 4.0 * 50.0 * 50.0;

        let raw = triangulate(&n, &[], &opts.without_refinement()).unwrap();
        let refined = triangulate(&n, &[], &opts).unwrap();

        assert!(
            raw.max_area_mm2() > cap * 4.0,
            "the raw CDT was already fine, so this proves nothing: {} mm²",
            raw.max_area_mm2()
        );
        // The area bound is a refinement *hint*, not a guarantee: with
        // `keep_constraint_edges` a face against the boundary cannot always be
        // split. MEASURED worst case on this fixture is 1437 mm² against a 1083
        // mm² bound, i.e. 1.33x, so 1.5x is the honest ceiling to assert.
        assert!(
            refined.max_area_mm2() <= cap * 1.5,
            "refined mesh still has a {} mm² face against a {cap} mm² bound",
            refined.max_area_mm2()
        );
        assert!(
            refined.triangle_count() > raw.triangle_count() * 2,
            "{} -> {} triangles",
            raw.triangle_count(),
            refined.triangle_count()
        );
        // Refinement adds vertices; it must not move the cloth.
        assert!((refined.area_mm2() - raw.area_mm2()).abs() < 1e-6);
    }

    #[test]
    fn refinement_holds_the_angle_bound_away_from_acute_input_corners() {
        let (n, _, _) = square(200.0);
        let opts = TriangulateOptions::default().with_target_edge_mm(20.0);
        let raw = triangulate(&n, &[], &opts.without_refinement()).unwrap();
        let refined = triangulate(&n, &[], &opts).unwrap();
        // A square has no acute input angle, so here the bound really is global.
        assert!(
            refined.min_angle_deg() > 25.0,
            "min angle {} (raw was {})",
            refined.min_angle_deg(),
            raw.min_angle_deg()
        );
    }

    #[test]
    fn a_panel_thinner_than_the_target_edge_is_still_valid_just_skinny() {
        // 400 x 12 mm at a 200 mm target edge. The boundary cannot be
        // subdivided below the panel's own width, so `keep_constraint_edges`
        // leaves slivers. Documented limitation: the mesh must still be
        // *correct*, and the caller's fix is a smaller target_edge_mm.
        let mut n = VectorNetwork::new();
        let v: Vec<VertexId> = [(0.0, 0.0), (400.0, 0.0), (400.0, 12.0), (0.0, 12.0)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        for i in 0..4 {
            n.add_edge(v[i], v[(i + 1) % 4], EdgeGeom::Line, EdgeSemantic::Cut)
                .unwrap();
        }
        let coarse = triangulate(
            &n,
            &[],
            &TriangulateOptions::default().with_target_edge_mm(200.0),
        )
        .unwrap();
        coarse.check().unwrap();
        assert!((coarse.area_mm2() - 4_800.0).abs() < 1e-6);
        // The fix, and proof that it is a fix.
        let fine = triangulate(
            &n,
            &[],
            &TriangulateOptions::default().with_target_edge_mm(6.0),
        )
        .unwrap();
        fine.check().unwrap();
        assert!(
            fine.min_angle_deg() > coarse.min_angle_deg() + 10.0,
            "{} -> {}",
            coarse.min_angle_deg(),
            fine.min_angle_deg()
        );
        assert!((fine.area_mm2() - 4_800.0).abs() < 1e-6);
    }

    #[test]
    fn a_hole_is_excluded() {
        let (mut n, _, _) = square(200.0);
        // A 40 mm square hole; parity does the rest.
        let h: Vec<VertexId> = [(80.0, 80.0), (120.0, 80.0), (120.0, 120.0), (80.0, 120.0)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        for i in 0..4 {
            n.add_edge(h[i], h[(i + 1) % 4], EdgeGeom::Line, EdgeSemantic::Cut)
                .unwrap();
        }
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        assert!(
            (m.area_mm2() - (40_000.0 - 1_600.0)).abs() < 1e-6,
            "area {} mm² should be the square minus the hole",
            m.area_mm2()
        );
    }

    #[test]
    fn a_dart_removes_its_wedge() {
        // 200 mm square with a 40 mm-wide dart rising 120 mm from the bottom.
        let mut n = VectorNetwork::new();
        let c: Vec<VertexId> = [
            (0.0, 0.0),
            (80.0, 0.0),
            (120.0, 0.0),
            (200.0, 0.0),
            (200.0, 200.0),
            (0.0, 200.0),
        ]
        .iter()
        .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
        .collect();
        for i in 0..6 {
            n.add_edge(c[i], c[(i + 1) % 6], EdgeGeom::Line, EdgeSemantic::Cut)
                .unwrap();
        }
        let apex = n.add_vertex(Point::new(100.0, 120.0));
        let leg_a = n
            .add_edge(c[1], apex, EdgeGeom::Line, EdgeSemantic::DartLeg)
            .unwrap();
        let leg_b = n
            .add_edge(apex, c[2], EdgeGeom::Line, EdgeSemantic::DartLeg)
            .unwrap();

        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();

        // The wedge is (40 x 120)/2 = 2400 mm² of cloth the dart folds out.
        let want = 40_000.0 - 2_400.0;
        assert!(
            (m.area_mm2() - want).abs() < 1e-6,
            "area {} mm², expected {want} — the dart wedge was not removed",
            m.area_mm2()
        );
        // Both legs are mapped, so M6 can sew them to each other, and the apex
        // is a single shared vertex.
        let ca = m.chain(leg_a).unwrap();
        let cb = m.chain(leg_b).unwrap();
        assert_eq!(ca.semantic, EdgeSemantic::DartLeg);
        assert_eq!(
            ca.points.last().unwrap().0,
            cb.points[0].0,
            "the apex is not shared"
        );
        assert_eq!(
            m.sources[ca.points.last().unwrap().0 as usize],
            VertexSource::Vertex(apex)
        );
        // No triangle may sit inside the wedge.
        for i in 0..m.triangle_count() {
            let p = m.indices[i].map(|k| m.pattern_mm[k as usize]);
            let cx = (p[0][0] + p[1][0] + p[2][0]) / 3.0;
            let cy = (p[0][1] + p[1][1] + p[2][1]) / 3.0;
            let in_wedge = cy < 120.0 && (cx - 100.0).abs() < 20.0 * (1.0 - cy / 120.0);
            assert!(
                !in_wedge,
                "triangle {i} lies in the dart wedge at ({cx}, {cy})"
            );
        }
    }

    #[test]
    fn a_closed_dart_removes_its_lens() {
        // A fisheye dart: two legs meeting at both ends, wholly inside the piece.
        let (mut n, _, _) = square(200.0);
        let lo = n.add_vertex(Point::new(100.0, 60.0));
        let hi = n.add_vertex(Point::new(100.0, 140.0));
        let l = n.add_vertex(Point::new(85.0, 100.0));
        let r = n.add_vertex(Point::new(115.0, 100.0));
        for (a, b) in [(lo, l), (l, hi), (hi, r), (r, lo)] {
            n.add_edge(a, b, EdgeGeom::Line, EdgeSemantic::DartLeg)
                .unwrap();
        }
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        m.check().unwrap();
        // The lens is a 30 x 80 rhombus: 1200 mm².
        let want = 40_000.0 - 1_200.0;
        assert!(
            (m.area_mm2() - want).abs() < 1e-6,
            "area {} mm², expected {want}",
            m.area_mm2()
        );
    }

    #[test]
    fn crossing_pattern_edges_are_refused_with_a_fixable_error() {
        let (mut n, _, _) = square(100.0);
        // Two internal lines crossing at (50, 50) with no vertex there.
        let a = n.add_vertex(Point::new(20.0, 20.0));
        let b = n.add_vertex(Point::new(80.0, 80.0));
        let c = n.add_vertex(Point::new(20.0, 80.0));
        let d = n.add_vertex(Point::new(80.0, 20.0));
        n.add_edge(a, b, EdgeGeom::Line, EdgeSemantic::Internal)
            .unwrap();
        n.add_edge(c, d, EdgeGeom::Line, EdgeSemantic::Internal)
            .unwrap();
        assert!(matches!(
            triangulate(&n, &[], &TriangulateOptions::default()),
            Err(TriangulateError::CrossingConstraint(_))
        ));
    }

    #[test]
    fn bad_options_are_rejected_before_any_work() {
        let (n, _, _) = square(100.0);
        for bad in [
            TriangulateOptions {
                target_edge_mm: 0.0,
                ..Default::default()
            },
            TriangulateOptions {
                weld_mm: 1e6,
                ..Default::default()
            },
            TriangulateOptions {
                curve_tolerance_mm: f64::NAN,
                ..Default::default()
            },
            TriangulateOptions {
                min_angle_deg: 90.0,
                ..Default::default()
            },
        ] {
            assert!(
                matches!(
                    triangulate(&n, &[], &bad),
                    Err(TriangulateError::BadOption { .. })
                ),
                "{bad:?} was accepted"
            );
        }
    }

    #[test]
    fn resolution_is_one_knob() {
        let (n, _, _) = square(200.0);
        let coarse = triangulate(
            &n,
            &[],
            &TriangulateOptions::default().with_target_edge_mm(50.0),
        )
        .unwrap();
        let fine = triangulate(
            &n,
            &[],
            &TriangulateOptions::default().with_target_edge_mm(10.0),
        )
        .unwrap();
        assert!(
            fine.triangle_count() > coarse.triangle_count() * 10,
            "{} vs {}",
            coarse.triangle_count(),
            fine.triangle_count()
        );
        // Both cover the same cloth.
        assert!((fine.area_mm2() - coarse.area_mm2()).abs() < 1e-6);
    }

    #[test]
    fn a_collinear_network_is_a_typed_error() {
        // Three points on a line with edges between them: distinct points, but
        // no area, so there is no triangulation and no fabric.
        let mut n = VectorNetwork::new();
        let v: Vec<VertexId> = [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]
            .iter()
            .map(|&(x, y)| n.add_vertex(Point::new(x, y)))
            .collect();
        for i in 0..2 {
            n.add_edge(v[i], v[i + 1], EdgeGeom::Line, EdgeSemantic::Cut)
                .unwrap();
        }
        assert_eq!(
            triangulate(&n, &[], &TriangulateOptions::default()),
            Err(TriangulateError::NoRegion)
        );
    }

    #[test]
    fn an_empty_network_is_a_typed_error() {
        let n = VectorNetwork::new();
        assert_eq!(
            triangulate(&n, &[], &TriangulateOptions::default()),
            Err(TriangulateError::Degenerate {
                distinct: 0,
                weld_mm: 1e-3
            })
        );
    }

    #[test]
    fn outline_vertices_are_the_boundary_ring() {
        let (n, _, _) = square(100.0);
        let m = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        let ring = m.outline_vertices();
        // 4 sides x 10 links = 40 distinct vertices on the ring.
        assert_eq!(ring.len(), 40, "{ring:?}");
        for &i in &ring {
            let [x, y] = m.pattern_mm[i as usize];
            assert!(
                x == 0.0 || y == 0.0 || x == 100.0 || y == 100.0,
                "vertex {i} at ({x}, {y}) is not on the outline"
            );
        }
    }

    #[test]
    fn an_absurd_resolution_is_refused_before_any_work() {
        // 100 mm square at a 10 nm target edge: 4 x 10^7 boundary points. The
        // realistic cause is a document in the wrong unit, which is what the
        // error says.
        let (n, _, _) = square(100.0);
        let err = triangulate(
            &n,
            &[],
            &TriangulateOptions {
                weld_mm: 1e-9,
                ..TriangulateOptions::default().with_target_edge_mm(1e-5)
            },
        )
        .unwrap_err();
        assert!(matches!(err, TriangulateError::TooDense { .. }), "{err:?}");
        assert!(err.to_string().contains("millimetres"), "{err}");
    }

    /// `check` is the crate's own safety net and every mesh goes through it, so
    /// it had better not be vacuous. Break a valid mesh in each of the ways that
    /// matter and confirm it notices.
    #[test]
    fn the_mesh_validator_has_teeth() {
        let (n, _, _) = square(100.0);
        let good = triangulate(&n, &[], &TriangulateOptions::default()).unwrap();
        assert_eq!(good.check(), Ok(()));

        let mut flipped = good.clone();
        flipped.indices[0].swap(0, 1);
        assert!(
            flipped
                .check()
                .unwrap_err()
                .contains("flipped or degenerate"),
            "a reversed triangle went unnoticed"
        );

        let mut collapsed = good.clone();
        collapsed.indices[0][1] = collapsed.indices[0][0];
        assert!(collapsed.check().unwrap_err().contains("repeats a vertex"));

        // The unit guard: a second mm -> m conversion anywhere would look
        // exactly like this.
        let mut twice = good.clone();
        for p in &mut twice.positions {
            p[0] *= 1e-3;
            p[1] *= 1e-3;
        }
        assert!(
            twice
                .check()
                .unwrap_err()
                .contains("second unit conversion"),
            "a 1000x garment went unnoticed"
        );

        let mut lifted = good.clone();
        lifted.positions[0][2] = 0.1;
        assert!(lifted.check().unwrap_err().contains("pattern plane"));

        let mut unsorted = good.clone();
        unsorted.chains.swap(0, 1);
        assert!(unsorted.check().unwrap_err().contains("not sorted"));

        let mut backwards = good.clone();
        let pts = &mut backwards.chains[0].points;
        pts.reverse();
        assert!(backwards.check().unwrap_err().contains("ascending in t"));
    }

    #[test]
    fn the_same_input_triangulates_identically() {
        let (n, _, e) = square(180.0);
        let a = triangulate(&n, &[(e[0], 0.25)], &TriangulateOptions::default()).unwrap();
        let b = triangulate(&n, &[(e[0], 0.25)], &TriangulateOptions::default()).unwrap();
        assert_eq!(a, b);
    }
}
