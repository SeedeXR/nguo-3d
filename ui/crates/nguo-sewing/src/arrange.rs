// SPDX-License-Identifier: GPL-2.0-or-later
//! 3D arrangement: where each flat piece goes before the solver runs
//! (`frontend-guide.md` §11.2).
//!
//! # It does not have to be good, but it does have to be isometric
//!
//! The solver untangles from overlapping starts, which is why it was chosen, so
//! the placement only has to be roughly right. There is one thing it must not
//! get wrong: **`SimState::from_mesh` takes every rest length from the
//! positions it is handed**, so whatever we submit *is* the garment's rest
//! shape. A placement that stretches a panel bakes that stretch in as the
//! fabric's natural state and the garment never springs back.
//!
//! Wrapping a flat panel onto a cylinder is *isometric* — a cylinder is
//! developable, so arc length equals the pattern width exactly — which is why
//! [`Archetype::Tube`] wraps rather than scales. The only thing the wrap
//! changes is the bending rest angle, so a tube "remembers" being round; that
//! is a real artefact, and a small one next to remembering being stretched.
//!
//! # Why the radius is a `max`
//!
//! Panels are laid on a cylinder of radius `max(body + gap, ΣW/2π)`:
//!
//! * `ΣW/2π` is the radius at which the panels exactly close the circle. Below
//!   it they would overlap.
//! * `body + gap` keeps them *outside* the body. Starting inside a barrier
//!   method's collision geometry is the one initial condition it cannot fix.
//!
//! Whatever arc is left over is shared out between the panels, so the seams
//! start slightly open and the stitches pull them shut. A garment narrower than
//! the body therefore starts as a loose tube around it and is drawn in by its
//! own seams — never stretched over it by the arranger.

use nguo_doc::{Document, Feature, PieceId, Placement};

/// Clearance between the body surface and the fabric, metres. 2 cm: bigger than
/// any contact `dhat` in the solver (1 mm) by a wide margin, small enough that
/// the seams do not have far to travel.
pub const DEFAULT_GAP_M: f32 = 0.02;

/// An axis-aligned box in world metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn of(points: &[[f32; 3]]) -> Option<Aabb> {
        let mut it = points.iter();
        let first = *it.next()?;
        Some(it.fold(
            Aabb {
                min: first,
                max: first,
            },
            |mut b, p| {
                for (k, c) in p.iter().enumerate() {
                    b.min[k] = b.min[k].min(*c);
                    b.max[k] = b.max[k].max(*c);
                }
                b
            },
        ))
    }

    pub fn center(&self) -> [f32; 3] {
        std::array::from_fn(|k| 0.5 * (self.min[k] + self.max[k]))
    }

    pub fn extent(&self) -> [f32; 3] {
        std::array::from_fn(|k| self.max[k] - self.min[k])
    }
}

/// Garment archetype — the preset that decides the *shape* of the arrangement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archetype {
    /// Panels wrapped around a vertical cylinder in seam order: a bodice, a
    /// skirt, a sleeve, a tube. The only preset with a body in it.
    Tube,
    /// Panels stood up side by side in a plane in front of the body. The
    /// fallback for a garment with no seam graph to walk, and what you want
    /// while checking a draft rather than a drape.
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrangeOptions {
    pub archetype: Archetype,
    /// Cylinder axis, world XZ metres.
    pub center_xz: [f32; 2],
    /// The body's radius where the garment sits: a measured girth / 2π, **not**
    /// half a bounding box (a T-posed body's box is its arm span).
    pub body_radius_m: f32,
    pub gap_m: f32,
    /// World Y of the garment's top edge.
    pub top_y_m: f32,
}

impl Default for ArrangeOptions {
    fn default() -> Self {
        ArrangeOptions {
            archetype: Archetype::Tube,
            center_xz: [0.0, 0.0],
            body_radius_m: 0.15,
            gap_m: DEFAULT_GAP_M,
            top_y_m: 1.3,
        }
    }
}

/// Where one piece's flat plane goes in 3D.
///
/// Every variant maps pattern millimetres to world metres and every one maps
/// pattern `+y` to world `−Y`: pattern space is `+y` **downward** (that is what
/// the canvas draws and what the document's own sample geometry assumes), so a
/// piece's `y = 0` edge is its top edge and belongs at the top in 3D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement3d {
    /// Wrapped on a vertical cylinder. `origin_mm` is the piece's pattern-space
    /// top-left corner, i.e. `(bounds.x0, bounds.y0)`.
    Wrap {
        center_xz: [f32; 2],
        radius_m: f32,
        /// Angle of the piece's left edge. `0` faces `+Z`, increasing toward `+X`.
        theta0: f32,
        top_y_m: f32,
        origin_mm: [f64; 2],
    },
    /// Rigid: the flat plane rotated by a quaternion and translated. This is
    /// what an authored [`nguo_doc::Placement`] becomes — the manual override.
    Rigid {
        /// `[x, y, z, w]`.
        quat: [f32; 4],
        offset_m: [f32; 3],
        origin_mm: [f64; 2],
    },
}

impl Placement3d {
    /// Place one pattern point. `p_mm` is piece-local pattern millimetres, the
    /// same space as [`nguo_triangulate::SimMesh::pattern_mm`].
    pub fn apply(&self, p_mm: [f64; 2]) -> [f32; 3] {
        match *self {
            Placement3d::Wrap {
                center_xz,
                radius_m,
                theta0,
                top_y_m,
                origin_mm,
            } => {
                let along = (p_mm[0] - origin_mm[0]) * 1e-3;
                let down = (p_mm[1] - origin_mm[1]) * 1e-3;
                let theta = theta0 as f64 + along / radius_m.max(1e-6) as f64;
                [
                    center_xz[0] + radius_m * theta.sin() as f32,
                    top_y_m - down as f32,
                    center_xz[1] + radius_m * theta.cos() as f32,
                ]
            }
            Placement3d::Rigid {
                quat,
                offset_m,
                origin_mm,
            } => {
                let local = [
                    ((p_mm[0] - origin_mm[0]) * 1e-3) as f32,
                    (-(p_mm[1] - origin_mm[1]) * 1e-3) as f32,
                    0.0,
                ];
                let r = rotate(quat, local);
                std::array::from_fn(|k| r[k] + offset_m[k])
            }
        }
    }

    /// Both variants map pattern `+y` to world `−Y`, which mirrors the plane, so
    /// a counter-clockwise pattern triangle comes out clockwise. The assembler
    /// re-winds; this is the flag that says it must.
    pub fn flips_winding(&self) -> bool {
        true
    }
}

/// `q · v · q*` for a unit quaternion `[x, y, z, w]`. Ten lines rather than a
/// `glam` dependency in a crate that needs no other vector maths.
fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    // t = 2 · (q_vec × v)
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

/// One piece as the arranger sees it: how wide and tall it is in pattern space,
/// and whether the author has already placed it by hand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PieceExtent {
    pub piece: PieceId,
    /// `(x0, y0, x1, y1)` of the piece's pattern-space bounding box, mm.
    pub bounds_mm: [f64; 4],
    /// An authored placement, which always wins over the preset.
    pub manual: Option<Placement>,
    /// Arc length of the piece's `Feature::Support` edge, pattern mm — the girth
    /// this piece actually contributes *where it hangs*. `None` means the piece
    /// is unannotated and its bounding box is the best available answer.
    pub support_arc_mm: Option<f64>,
}

impl PieceExtent {
    pub fn width_mm(&self) -> f64 {
        self.bounds_mm[2] - self.bounds_mm[0]
    }

    /// The arc this piece occupies round the body: its support edge if it has
    /// one, its bounding-box width otherwise.
    ///
    /// The distinction only bites on a **flared** panel, and there it decides the
    /// whole placement: an A-line skirt's bounding box is its *hem*, so sizing
    /// the cylinder by it hangs the waistband out at hem radius and every side
    /// seam starts open by (hem width − waist width). Sizing by the support edge
    /// closes the waistband onto the body, which is where the garment is worn.
    ///
    // ponytail: one radius for the whole panel, so a flared panel's hem now
    // starts overlapping its neighbours instead of gaping — which is what flared
    // cloth does, and the solver's problem. The real answer is a *conical*
    // `Placement3d` variant; build it when a flared sample exists to measure it
    // on. Unmeasured hazard, stated rather than hidden: overlapping cloth at
    // step 0 is what stalls ACCD.
    pub fn arc_mm(&self) -> f64 {
        self.support_arc_mm.unwrap_or_else(|| self.width_mm())
    }

    pub fn origin_mm(&self) -> [f64; 2] {
        [self.bounds_mm[0], self.bounds_mm[1]]
    }
}

/// Where the garment wants to hang, read straight off the document.
///
/// The caller needs this **before** sewing, because it is what turns a body into
/// [`ArrangeOptions`] — and it is deliberately cheap: no triangulation, no mesh,
/// just the pieces' own networks.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Hang {
    /// The avatar manifest key from the first `Feature::Support` in the document,
    /// e.g. `"hip"`. `None` if nothing is annotated.
    pub landmark: Option<String>,
    /// Pattern millimetres the garment reaches **below** its top edge: the
    /// tallest piece. With the landmark height this is the band of body the
    /// garment can touch, which is the only band worth measuring a radius over.
    ///
    /// Off the network's own bounding box, so a curved hem reads its true extent
    /// and not its flattening — which makes this a hair larger than the
    /// triangulated mesh, i.e. a hair *looser* band. That is the safe direction.
    pub drop_mm: f64,
}

/// Read [`Hang`] off `doc`.
///
/// `landmark` is the first annotation in piece-id order.
///
// ponytail: one landmark per garment, because [`Archetype::Tube`] has one
// `top_y_m`. A two-tier garment (bodice at the underbust, skirt at the waist)
// needs a placement per piece, and that is a different arrangement, not a
// different string.
pub fn hang(doc: &Document) -> Hang {
    let mut out = Hang::default();
    for (_, piece) in doc.pieces().iter() {
        if let Some(b) = piece.network.bounds() {
            out.drop_mm = out.drop_mm.max(b.height());
        }
        if out.landmark.is_none() {
            out.landmark = piece.features.values().find_map(|f| match f {
                Feature::Support { landmark, .. } => Some(landmark.clone()),
                _ => None,
            });
        }
    }
    out
}

/// Arc length of a piece's support edge, pattern mm — [`PieceExtent`]'s
/// `support_arc_mm`.
///
/// The *network's* arc length, not the triangulated chain's: same edge, and the
/// exact curve rather than its flattening.
pub fn support_arc_mm(piece: &nguo_doc::Piece) -> Option<f64> {
    piece.features.values().find_map(|f| match f {
        Feature::Support { edge, .. } => piece.network.arc_length(*edge).ok(),
        _ => None,
    })
}

/// What the arranger decided, for reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct Arrangement {
    pub placements: Vec<Placement3d>,
    /// The cylinder the tube was laid on, metres. `None` for [`Archetype::Flat`].
    pub radius_m: Option<f32>,
    /// Total arc wrapped, metres — the garment's circumference *where it hangs*
    /// ([`PieceExtent::arc_mm`]), which is the sum of the panel widths only when
    /// no panel is flared.
    pub circumference_m: f32,
    /// Arc left over and shared between the seams, metres. Each seam therefore
    /// starts about `slack_m / seams` open.
    pub slack_m: f32,
}

/// Place every piece. `pieces` is in the order the wrap should follow — use
/// [`seam_order`] to get one that walks the seam graph.
pub fn arrange(pieces: &[PieceExtent], opts: &ArrangeOptions) -> Arrangement {
    match opts.archetype {
        Archetype::Flat => {
            // Bounding-box widths, not support arcs: a flat layout stands the
            // whole panel up side by side and must not let two of them overlap.
            let total_m = (pieces.iter().map(|p| p.width_mm()).sum::<f64>() * 1e-3) as f32;
            let mut x = 0.0f32;
            let placements = pieces
                .iter()
                .map(|p| {
                    let placement = manual(p).unwrap_or(Placement3d::Rigid {
                        quat: [0.0, 0.0, 0.0, 1.0],
                        offset_m: [
                            opts.center_xz[0] - 0.5 * total_m + x,
                            opts.top_y_m,
                            opts.center_xz[1] + opts.body_radius_m + opts.gap_m,
                        ],
                        origin_mm: p.origin_mm(),
                    });
                    x += (p.width_mm() * 1e-3) as f32;
                    placement
                })
                .collect();
            Arrangement {
                placements,
                radius_m: None,
                circumference_m: total_m,
                slack_m: 0.0,
            }
        }
        Archetype::Tube => {
            // The girth where the garment *hangs*, which is the bounding-box
            // width only for a panel with parallel sides. See `arc_mm`.
            let total_m = (pieces.iter().map(|p| p.arc_mm()).sum::<f64>() * 1e-3) as f32;
            let closing = total_m / std::f32::consts::TAU;
            let radius = (opts.body_radius_m + opts.gap_m).max(closing).max(1e-3);
            let slack = (std::f32::consts::TAU * radius - total_m).max(0.0);
            let per_seam = if pieces.is_empty() {
                0.0
            } else {
                slack / pieces.len() as f32
            };

            // Centre the first panel on the front (+Z) so a two-panel garment
            // reads as front/back rather than left/right.
            let first = pieces.first().map_or(0.0, |p| (p.arc_mm() * 1e-3) as f32);
            let mut theta = -0.5 * (first + per_seam) / radius;
            let placements = pieces
                .iter()
                .map(|p| {
                    let placement = manual(p).unwrap_or(Placement3d::Wrap {
                        center_xz: opts.center_xz,
                        radius_m: radius,
                        theta0: theta + 0.5 * per_seam / radius,
                        top_y_m: opts.top_y_m,
                        origin_mm: p.origin_mm(),
                    });
                    theta += ((p.arc_mm() * 1e-3) as f32 + per_seam) / radius;
                    placement
                })
                .collect();

            tracing::debug!(
                pieces = pieces.len(),
                circumference_m = total_m,
                radius_m = radius,
                slack_m = slack,
                "garment arranged as a tube"
            );
            Arrangement {
                placements,
                radius_m: Some(radius),
                circumference_m: total_m,
                slack_m: slack,
            }
        }
    }
}

fn manual(p: &PieceExtent) -> Option<Placement3d> {
    let m = p.manual?;
    Some(Placement3d::Rigid {
        quat: m.rotation_quat.map(|c| c as f32),
        offset_m: m.position_mm.map(|c| (c * 1e-3) as f32),
        origin_mm: p.origin_mm(),
    })
}

/// Piece ids ordered so that pieces sewn to each other are adjacent — the order
/// to wrap them in.
///
/// A greedy chain walk over the seam graph from the lowest id, restarting at the
/// next unvisited piece when the chain runs out. For the shapes that matter
/// (front/back, a four-panel skirt, a sleeve) this is exactly the order round
/// the body, and for anything it cannot chain it degrades to document order,
/// which is the order a person drafted them in.
///
// ponytail: greedy, not a Hamiltonian path. The seam graph of a garment panel
// ring is a path or a cycle; a piece with three seamed neighbours (a gusset)
// picks the lowest-id branch first. Upgrade to a real traversal when a garment
// with branching seams exists to test it on.
pub fn seam_order(doc: &Document) -> Vec<PieceId> {
    let all: Vec<PieceId> = doc.pieces().ids().collect();
    let neighbours = |p: PieceId| -> Vec<PieceId> {
        let mut n: Vec<PieceId> = doc
            .seams()
            .values()
            .filter(|s| s.references(p))
            .map(|s| if s.a.piece == p { s.b.piece } else { s.a.piece })
            .filter(|&q| q != p)
            .collect();
        n.sort_by_key(|q| q.raw());
        n.dedup();
        n
    };

    let mut out: Vec<PieceId> = Vec::with_capacity(all.len());
    for start in &all {
        if out.contains(start) {
            continue;
        }
        let mut current = *start;
        loop {
            out.push(current);
            let Some(next) = neighbours(current)
                .into_iter()
                .find(|q| !out.contains(q) && doc.pieces().contains(*q))
            else {
                break;
            };
            current = next;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_doc::{Edit, Piece, Seam, SeamSide};
    use nguo_pattern::{EdgeGeom, EdgeSemantic};

    fn extent(id: u64, w: f64, h: f64) -> PieceExtent {
        PieceExtent {
            piece: PieceId::from_raw(id),
            bounds_mm: [0.0, 0.0, w, h],
            manual: None,
            support_arc_mm: None,
        }
    }

    fn opts() -> ArrangeOptions {
        ArrangeOptions {
            archetype: Archetype::Tube,
            center_xz: [0.0, 0.0],
            body_radius_m: 0.15,
            gap_m: 0.02,
            top_y_m: 1.2,
        }
    }

    /// The invariant the whole design rests on: wrapping preserves arc length,
    /// so the panel is not stretched by being placed.
    #[test]
    fn wrapping_a_panel_preserves_its_width_as_arc_length() {
        let a = arrange(&[extent(0, 500.0, 600.0)], &opts());
        let p = a.placements[0];
        let steps = 200;
        let mut walked = 0.0f64;
        let mut prev = p.apply([0.0, 0.0]);
        for i in 1..=steps {
            let x = 500.0 * i as f64 / steps as f64;
            let q = p.apply([x, 0.0]);
            walked += ((0..3)
                .map(|k| (q[k] - prev[k]) as f64)
                .map(|d| d * d)
                .sum::<f64>())
            .sqrt();
            prev = q;
        }
        assert!(
            (walked - 0.5).abs() < 1e-4,
            "500 mm of pattern became {walked:.6} m of arc"
        );
    }

    /// Pattern y is downward, so the piece's y = 0 edge is its top edge and
    /// must end up at `top_y_m`, with the hem below it.
    #[test]
    fn pattern_y_is_down_so_the_hem_ends_up_below_the_top_edge() {
        let a = arrange(&[extent(0, 500.0, 600.0)], &opts());
        let top = a.placements[0].apply([250.0, 0.0]);
        let hem = a.placements[0].apply([250.0, 600.0]);
        assert!((top[1] - 1.2).abs() < 1e-6, "top edge at top_y_m: {top:?}");
        assert!((hem[1] - 0.6).abs() < 1e-6, "600 mm below it: {hem:?}");
    }

    /// Two panels of equal width put their shared edges at the same place, up
    /// to the slack that is deliberately left for the seam to close.
    #[test]
    fn adjacent_panels_meet_at_their_shared_edge_within_the_slack() {
        let a = arrange(&[extent(0, 500.0, 600.0), extent(1, 500.0, 600.0)], &opts());
        let r = a.radius_m.unwrap();
        // 1.0 m of panel needs r = 0.159; the body needs 0.17, so the body wins
        // and there is slack.
        assert!((r - 0.17).abs() < 1e-6, "radius {r}");
        assert!(a.slack_m > 0.0);

        let a_right = a.placements[0].apply([500.0, 0.0]);
        let b_left = a.placements[1].apply([0.0, 0.0]);
        let gap = ((0..3)
            .map(|k| (a_right[k] - b_left[k]) as f64)
            .map(|d| d * d)
            .sum::<f64>())
        .sqrt();
        assert!(
            gap <= (a.slack_m / 2.0) as f64 + 1e-6,
            "the seam starts {gap:.4} m open, slack/2 is {:.4}",
            a.slack_m / 2.0
        );
        assert!(gap > 0.0, "with slack the seam is not already closed");
    }

    /// Every placed vertex must be outside the body cylinder, because a barrier
    /// method cannot recover from starting inside the collision geometry.
    #[test]
    fn no_panel_starts_inside_the_body_cylinder() {
        let o = opts();
        let a = arrange(&[extent(0, 400.0, 600.0), extent(1, 400.0, 600.0)], &o);
        for (i, p) in a.placements.iter().enumerate() {
            for x in [0.0, 100.0, 200.0, 300.0, 400.0] {
                for y in [0.0, 300.0, 600.0] {
                    let w = p.apply([x, y]);
                    let radial =
                        ((w[0] - o.center_xz[0]).powi(2) + (w[2] - o.center_xz[1]).powi(2)).sqrt();
                    assert!(
                        radial >= o.body_radius_m + o.gap_m - 1e-5,
                        "piece {i} at ({x},{y}) is {radial} from the axis, inside the body+gap"
                    );
                }
            }
        }
    }

    /// A garment wider than the body opens the circle out instead of overlapping.
    #[test]
    fn a_garment_wider_than_the_body_sets_its_own_radius() {
        let a = arrange(
            &[extent(0, 1000.0, 600.0), extent(1, 1000.0, 600.0)],
            &opts(),
        );
        let r = a.radius_m.unwrap();
        assert!(
            (r - 2.0 / std::f32::consts::TAU).abs() < 1e-6,
            "2 m of panel should close at r = {}, got {r}",
            2.0 / std::f32::consts::TAU
        );
        assert!(a.slack_m.abs() < 1e-6, "a closed tube has no slack");
    }

    #[test]
    fn an_authored_placement_overrides_the_preset() {
        let mut e = extent(0, 500.0, 600.0);
        e.manual = Some(Placement {
            position_mm: [1000.0, 2000.0, 3000.0],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
        });
        let a = arrange(&[e], &opts());
        let at_origin = a.placements[0].apply([0.0, 0.0]);
        assert_eq!(at_origin, [1.0, 2.0, 3.0], "identity rotation, mm -> m");
    }

    #[test]
    fn a_quarter_turn_about_y_sends_pattern_x_to_minus_z() {
        // 90 deg about +Y: (1,0,0) -> (0,0,-1).
        let s = (std::f32::consts::FRAC_PI_4).sin();
        let mut e = extent(0, 1000.0, 100.0);
        e.manual = Some(Placement {
            position_mm: [0.0; 3],
            rotation_quat: [0.0, s as f64, 0.0, s as f64],
        });
        let a = arrange(&[e], &opts());
        let p = a.placements[0].apply([1000.0, 0.0]);
        assert!(p[0].abs() < 1e-6 && (p[2] + 1.0).abs() < 1e-6, "{p:?}");
    }

    /// The seam graph, not the arena order, decides who is next around the body.
    #[test]
    fn seam_order_walks_the_seam_graph() {
        let mut doc = Document::new();
        let mut ids = Vec::new();
        let mut edges = Vec::new();
        for i in 0..3 {
            let mut piece = Piece::new(format!("p{i}"));
            let v: Vec<_> = [(0.0, 0.0), (100.0, 0.0), (100.0, 200.0), (0.0, 200.0)]
                .iter()
                .map(|&(x, y)| piece.network.add_vertex(nguo_doc::kurbo::Point::new(x, y)))
                .collect();
            let e: Vec<_> = (0..4)
                .map(|k| {
                    piece
                        .network
                        .add_edge(v[k], v[(k + 1) % 4], EdgeGeom::Line, EdgeSemantic::Seam)
                        .unwrap()
                })
                .collect();
            let id = doc.next_piece_id();
            doc.apply_edit(Edit::AddPiece {
                id,
                piece: Box::new(piece),
            })
            .unwrap();
            ids.push(id);
            edges.push(e);
        }
        // Sew 0-2 and 2-1, so the ring order is 0, 2, 1 — not arena order.
        for (x, y) in [(0usize, 2usize), (2, 1)] {
            let id = doc.next_seam_id();
            doc.apply_edit(Edit::AddSeam {
                id,
                seam: Box::new(Seam::new(
                    SeamSide::whole(ids[x], vec![(edges[x][1], false)]),
                    SeamSide::whole(ids[y], vec![(edges[y][3], true)]),
                )),
            })
            .unwrap();
        }
        assert_eq!(seam_order(&doc), vec![ids[0], ids[2], ids[1]]);
    }

    /// The C2 defect, as a number. Two A-line panels 300 mm at the waist and
    /// 700 mm at the hem: the bounding box says 1 400 mm of girth and hangs the
    /// waistband at r = 0.223 m, four times further from a 0.15 m body than the
    /// gap allows. The support edge says 600 mm, which closes on the body.
    #[test]
    fn a_flared_panel_is_sized_by_its_support_edge_and_not_its_hem() {
        let flared = |id| PieceExtent {
            support_arc_mm: Some(300.0),
            ..extent(id, 700.0, 600.0)
        };

        let hem = arrange(&[extent(0, 700.0, 600.0), extent(1, 700.0, 600.0)], &opts());
        assert!(
            (hem.radius_m.unwrap() - 1.4 / std::f32::consts::TAU).abs() < 1e-6,
            "unannotated: the hem sets the radius, {:?}",
            hem.radius_m,
        );

        let waist = arrange(&[flared(0), flared(1)], &opts());
        // 0.6 m closes at r = 0.0955, below the body, so the body wins: the
        // waistband lands at body + gap, which is the whole point.
        assert!(
            (waist.radius_m.unwrap() - 0.17).abs() < 1e-6,
            "annotated: {:?}",
            waist.radius_m,
        );
        assert!((waist.circumference_m - 0.6).abs() < 1e-6);
        // The support edges tile the circle: the two seams share the slack and
        // nothing else, exactly as for a rectangular panel.
        let a_right = waist.placements[0].apply([300.0, 0.0]);
        let b_left = waist.placements[1].apply([0.0, 0.0]);
        let gap = ((0..3)
            .map(|k| (a_right[k] - b_left[k]) as f64)
            .map(|d| d * d)
            .sum::<f64>())
        .sqrt();
        assert!(
            gap <= (waist.slack_m / 2.0) as f64 + 1e-6,
            "{gap} m open against {} m of slack",
            waist.slack_m,
        );
    }

    /// The annotation the app reads before it has a mesh: which landmark, and how
    /// far down the garment reaches.
    #[test]
    fn hang_reads_the_support_annotation_and_the_garment_drop() {
        let mut doc = Document::new();
        assert_eq!(
            hang(&doc),
            Hang::default(),
            "an empty document hangs nowhere"
        );

        let mut piece = Piece::new("skirt");
        let v: Vec<_> = [(0.0, 0.0), (400.0, 0.0), (400.0, 700.0), (0.0, 700.0)]
            .iter()
            .map(|&(x, y)| piece.network.add_vertex(nguo_doc::kurbo::Point::new(x, y)))
            .collect();
        let e: Vec<_> = (0..4)
            .map(|k| {
                piece
                    .network
                    .add_edge(v[k], v[(k + 1) % 4], EdgeGeom::Line, EdgeSemantic::Cut)
                    .unwrap()
            })
            .collect();
        piece.features.insert(nguo_doc::Feature::Support {
            edge: e[0],
            landmark: "waist".into(),
        });
        assert_eq!(super::support_arc_mm(&piece), Some(400.0));

        let id = doc.next_piece_id();
        doc.apply_edit(Edit::AddPiece {
            id,
            piece: Box::new(piece),
        })
        .unwrap();

        let h = hang(&doc);
        assert_eq!(h.landmark.as_deref(), Some("waist"));
        assert!((h.drop_mm - 700.0).abs() < 1e-9, "{h:?}");
    }

    #[test]
    fn seam_order_covers_every_piece_even_with_no_seams() {
        let mut doc = Document::new();
        let mut ids = Vec::new();
        for i in 0..3 {
            let id = doc.next_piece_id();
            doc.apply_edit(Edit::AddPiece {
                id,
                piece: Box::new(Piece::new(format!("p{i}"))),
            })
            .unwrap();
            ids.push(id);
        }
        assert_eq!(seam_order(&doc), ids);
    }
}
