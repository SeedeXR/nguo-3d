// SPDX-License-Identifier: GPL-2.0-or-later
//! Scene building and the per-piece cache (guide §7.3).
//!
//! # Layering
//! grid/rulers → cached per-piece scenes (fill, edges, features) → interaction
//! overlay (points, handles, snap indicator, rubber band).
//!
//! ponytail: the guide's layer list is global (all fills, then all edges). This
//! caches per *piece* instead, which the guide explicitly allows, because that is
//! what makes a drag cost one piece rebuild instead of fifty. The visible
//! difference is only where two pieces overlap on the marker. Split into two
//! cached scenes per piece if overlapping pieces ever need strict global order.
//!
//! # Why a cached piece scene is in world space
//! Each piece is encoded once with its own `transform`, then appended under the
//! view matrix. Pan is therefore free — no rebuild. Stroke widths are the
//! exception: they are stored in world units so that a `1.5 px` line stays
//! `1.5 px`, which means the cache is keyed on the zoom and a zoom step costs one
//! full rebuild. Measured in `tests/perf.rs`; the upgrade path is screen-space
//! strokes in Vello.

use std::collections::{BTreeMap, BTreeSet};

use nguo_doc::{Document, Edit, Feature, Piece, PieceId};
use nguo_pattern::{EdgeGeom, EdgeSemantic};
use nguo_ui_theme::{Rgba, Tokens};
use vello::kurbo::{Affine, BezPath, Circle, Line, ParamCurve, Point, Rect, Stroke, Vec2};
use vello::peniko::{Color, Fill};
use vello::Scene;

use crate::hit::{boundary_path, HandleEnd, Target};
use crate::snap::Snap;
use crate::{grid, View};

/// Theme colour → Vello colour. `Rgba` is non-linear sRGB with straight alpha,
/// which is exactly what `peniko::Color::new` wants.
pub fn ink(c: Rgba) -> Color {
    Color::new([c.r, c.g, c.b, c.a])
}

// Line weights and handle sizes, in logical px. Not tokens: `nguo-ui-theme`'s
// `Metrics` covers widget chrome, and inventing canvas entries there is a change
// to a crate this one does not own.
const EDGE_PX: f64 = 1.6;
const OUTLINE_PX: f64 = 2.2;
const HAIRLINE_PX: f64 = 1.0;
const POINT_R_PX: f64 = 3.5;
const HANDLE_R_PX: f64 = 2.75;
const NOTCH_PX: f64 = 7.0;
/// Piece fill opacity. The canvas is paper: a piece is a tint, not a slab.
const FILL_ALPHA: f32 = 0.10;

/// What the interaction layer is showing this frame. Rebuilt every frame on
/// purpose — it is the only thing that has to be.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Interaction {
    pub selection: Vec<Target>,
    pub hover: Option<Target>,
    pub snap: Option<Snap>,
    /// Pieces whose points and handles are shown (direct-select).
    pub editing: Vec<PieceId>,
    /// Pen rubber band, in world mm.
    pub rubber_band: Option<(Point, Point)>,
}

/// The grid and rulers, in **screen** space. Cheap and view-dependent, so it is
/// rebuilt every frame rather than cached and invalidated.
pub fn grid_scene(view: &View, size: Vec2, tokens: &Tokens) -> Scene {
    let mut scene = Scene::new();
    if size.x <= 0.0 || size.y <= 0.0 {
        return scene;
    }
    let (minor, major) = grid::steps(view.zoom);
    let world = grid::visible(view, size);
    let line = ink(tokens.color.grid_line);
    let faint = ink(tokens.color.grid_line.alpha(0.45));

    let mut minors = BezPath::new();
    let mut majors = BezPath::new();
    for x in grid::ticks(world.x0, world.x1, minor) {
        let sx = view.world_to_screen(Point::new(x, 0.0), size).x;
        let path = if (x / major).fract().abs() < 1e-9 {
            &mut majors
        } else {
            &mut minors
        };
        path.move_to((sx, 0.0));
        path.line_to((sx, size.y));
    }
    for y in grid::ticks(world.y0, world.y1, minor) {
        let sy = view.world_to_screen(Point::new(0.0, y), size).y;
        let path = if (y / major).fract().abs() < 1e-9 {
            &mut majors
        } else {
            &mut minors
        };
        path.move_to((0.0, sy));
        path.line_to((size.x, sy));
    }

    let stroke = Stroke::new(HAIRLINE_PX);
    scene.stroke(&stroke, Affine::IDENTITY, faint, None, &minors);
    scene.stroke(&stroke, Affine::IDENTITY, line, None, &majors);

    // Rulers: a strip per axis with a tick at every major line. The numbers come
    // from `grid::labels`; drawing them needs a font this crate does not have.
    let strip = ink(tokens.color.header.alpha(0.9));
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        strip,
        None,
        &Rect::new(0.0, 0.0, size.x, grid::RULER_PX),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        strip,
        None,
        &Rect::new(0.0, 0.0, grid::RULER_PX, size.y),
    );
    let mut ticks = BezPath::new();
    for x in grid::ticks(world.x0, world.x1, major) {
        let sx = view.world_to_screen(Point::new(x, 0.0), size).x;
        ticks.move_to((sx, 0.0));
        ticks.line_to((sx, grid::RULER_PX));
    }
    for y in grid::ticks(world.y0, world.y1, major) {
        let sy = view.world_to_screen(Point::new(0.0, y), size).y;
        ticks.move_to((0.0, sy));
        ticks.line_to((grid::RULER_PX, sy));
    }
    scene.stroke(
        &Stroke::new(HAIRLINE_PX),
        Affine::IDENTITY,
        ink(tokens.color.text_muted),
        None,
        &ticks,
    );
    scene
}

/// One piece, encoded in **world** space. `px` is world units per logical
/// pixel (`1.0 / zoom`), which is how a constant-width line survives being
/// drawn through the view matrix.
pub fn piece_scene(piece: &Piece, tokens: &Tokens, px: f64) -> Scene {
    let mut scene = Scene::new();
    let to_world = piece.transform;

    if let Some(path) = boundary_path(piece) {
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY, // `boundary_path` is already world space.
            ink(tokens.color.surface.alpha(FILL_ALPHA)),
            None,
            &path,
        );
    }

    // One path per semantic, so the whole piece is a handful of Vello draws
    // rather than one per edge.
    let mut by_semantic: BTreeMap<EdgeSemantic, BezPath> = BTreeMap::new();
    for (id, edge) in piece.network.edges.iter() {
        let Ok(seg) = piece.network.seg(id) else {
            continue;
        };
        let path = by_semantic.entry(edge.semantic).or_default();
        path.move_to(seg.start());
        path.push(seg.as_path_el());
    }
    for (semantic, path) in &by_semantic {
        let width = if semantic.is_boundary() {
            OUTLINE_PX
        } else {
            EDGE_PX
        };
        let stroke = match semantic {
            // Fold and internal lines are dashed, the way a paper pattern marks
            // a line you do not cut.
            EdgeSemantic::Fold | EdgeSemantic::Internal => {
                Stroke::new(width * px).with_dashes(0.0, [6.0 * px, 4.0 * px])
            }
            _ => Stroke::new(width * px),
        };
        scene.stroke(
            &stroke,
            to_world,
            ink(semantic_color(*semantic, tokens)),
            None,
            path,
        );
    }

    for (_, feature) in piece.features.iter() {
        draw_feature(&mut scene, piece, feature, tokens, px);
    }
    scene
}

fn semantic_color(semantic: EdgeSemantic, tokens: &Tokens) -> Rgba {
    match semantic {
        EdgeSemantic::Seam => tokens.color.seam,
        EdgeSemantic::Fold => tokens.color.fold,
        EdgeSemantic::Cut => tokens.color.text_primary,
        EdgeSemantic::Grain => tokens.color.grain,
        EdgeSemantic::Internal => tokens.color.guide,
        EdgeSemantic::DartLeg => tokens.color.dart,
    }
}

fn draw_feature(scene: &mut Scene, piece: &Piece, feature: &Feature, tokens: &Tokens, px: f64) {
    let to_world = piece.transform;
    match feature {
        Feature::GrainLine { from, to } => {
            if (*to - *from).hypot() < 1e-9 {
                return; // A zero-length grain line has no direction to draw.
            }
            let mut path = BezPath::new();
            path.move_to(*from);
            path.line_to(*to);
            // Arrowheads both ends, sized in px so they stay legible.
            let d = (*to - *from).normalize() * 8.0 * px;
            let n = Vec2::new(-d.y, d.x) * 0.45;
            for (tip, back) in [(*to, *to - d), (*from, *from + d)] {
                path.move_to(back + n);
                path.line_to(tip);
                path.line_to(back - n);
            }
            scene.stroke(
                &Stroke::new(EDGE_PX * px),
                to_world,
                ink(tokens.color.grain),
                None,
                &path,
            );
        }
        Feature::Notch { edge, t, .. } => {
            let Ok(seg) = piece.network.seg(*edge) else {
                return;
            };
            let t = t.clamp(0.0, 1.0);
            let at = seg.eval(t);
            // Across the edge, from the tangent, so a notch on a curve still
            // points the right way. A degenerate edge has no tangent.
            let chord = seg.eval((t + 0.01).min(1.0)) - seg.eval((t - 0.01).max(0.0));
            if chord.hypot() < 1e-12 {
                return;
            }
            let tangent = chord.normalize();
            let n = Vec2::new(-tangent.y, tangent.x) * NOTCH_PX * px;
            let mut path = BezPath::new();
            path.move_to(at - n * 0.3);
            path.line_to(at + n);
            scene.stroke(
                &Stroke::new(EDGE_PX * px),
                to_world,
                ink(tokens.color.notch),
                None,
                &path,
            );
        }
        Feature::Dart { apex, .. } => {
            if let Ok(p) = piece.network.pos(*apex) {
                scene.stroke(
                    &Stroke::new(HAIRLINE_PX * px),
                    to_world,
                    ink(tokens.color.dart),
                    None,
                    &Circle::new(p, POINT_R_PX * px),
                );
            }
        }
        // Internal lines are already `Internal` edges, a label needs a font
        // (guide §7.5), and a support edge is an existing outline edge that is
        // already stroked. Nothing to draw here that is not drawn above.
        Feature::InternalLine { .. } | Feature::Label { .. } | Feature::Support { .. } => {}
    }
}

/// The interaction layer, in **screen** space so handles are exactly the size
/// they claim to be at any zoom.
pub fn overlay_scene(
    doc: &Document,
    it: &Interaction,
    view: &View,
    size: Vec2,
    tokens: &Tokens,
) -> Scene {
    let mut scene = Scene::new();
    let to_screen = |p: Point| view.world_to_screen(p, size);

    if let Some((from, to)) = it.rubber_band {
        let mut path = BezPath::new();
        path.move_to(to_screen(from));
        path.line_to(to_screen(to));
        scene.stroke(
            &Stroke::new(HAIRLINE_PX).with_dashes(0.0, [4.0, 3.0]),
            Affine::IDENTITY,
            ink(tokens.color.accent),
            None,
            &path,
        );
    }

    for &piece_id in &it.editing {
        let Ok(piece) = doc.piece(piece_id) else {
            continue;
        };
        let to_world = piece.transform;
        // Handles first: the stem must sit under the point that owns it.
        for (id, edge) in piece.network.edges.iter() {
            let EdgeGeom::Cubic { c1, c2 } = edge.geom else {
                continue;
            };
            let Ok(seg) = piece.network.seg(id) else {
                continue;
            };
            for (anchor, control) in [(seg.start(), c1), (seg.end(), c2)] {
                let a = to_screen(to_world * anchor);
                let c = to_screen(to_world * control);
                scene.stroke(
                    &Stroke::new(HAIRLINE_PX),
                    Affine::IDENTITY,
                    ink(tokens.color.border_strong),
                    None,
                    &Line::new(a, c),
                );
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    ink(tokens.color.accent),
                    None,
                    &Circle::new(c, HANDLE_R_PX),
                );
            }
        }
        for (_, v) in piece.network.vertices.iter() {
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                ink(tokens.color.text_primary),
                None,
                &Circle::new(to_screen(to_world * v.pos), POINT_R_PX),
            );
        }
    }

    for (target, color) in it
        .selection
        .iter()
        .map(|t| (t, tokens.color.selected_entity))
        .chain(it.hover.iter().map(|t| (t, tokens.color.hover_entity)))
    {
        highlight(&mut scene, doc, *target, view, size, color);
    }

    if let Some(snap) = it.snap {
        let at = to_screen(snap.world);
        // A diamond, so a snap indicator never reads as a point you can drag.
        let r = POINT_R_PX + 2.5;
        let mut path = BezPath::new();
        path.move_to((at.x, at.y - r));
        path.line_to((at.x + r, at.y));
        path.line_to((at.x, at.y + r));
        path.line_to((at.x - r, at.y));
        path.close_path();
        scene.stroke(
            &Stroke::new(HAIRLINE_PX * 1.5),
            Affine::IDENTITY,
            ink(tokens.color.snap_indicator),
            None,
            &path,
        );
    }
    scene
}

fn highlight(
    scene: &mut Scene,
    doc: &Document,
    target: Target,
    view: &View,
    size: Vec2,
    color: Rgba,
) {
    let Ok(piece) = doc.piece(target.piece()) else {
        return;
    };
    let to_world = piece.transform;
    let stroke = Stroke::new(OUTLINE_PX);
    match target {
        Target::Point { vertex, .. } => {
            if let Ok(p) = piece.network.pos(vertex) {
                let at = view.world_to_screen(to_world * p, size);
                scene.stroke(
                    &stroke,
                    Affine::IDENTITY,
                    ink(color),
                    None,
                    &Circle::new(at, POINT_R_PX + 2.0),
                );
            }
        }
        Target::Handle { edge, end, .. } => {
            if let Some(EdgeGeom::Cubic { c1, c2 }) = piece.network.edges.get(edge).map(|e| e.geom)
            {
                let c = match end {
                    HandleEnd::Start => c1,
                    HandleEnd::End => c2,
                };
                let at = view.world_to_screen(to_world * c, size);
                scene.stroke(
                    &stroke,
                    Affine::IDENTITY,
                    ink(color),
                    None,
                    &Circle::new(at, HANDLE_R_PX + 2.0),
                );
            }
        }
        Target::Edge { edge, .. } => {
            if let Ok(seg) = piece.network.seg(edge) {
                let mut path = BezPath::new();
                path.move_to(seg.start());
                path.push(seg.as_path_el());
                // The path is transformed into screen space and stroked with
                // identity, so the highlight is a constant width at any zoom.
                let full = view.scene_transform(size, 1.0) * to_world;
                scene.stroke(&stroke, Affine::IDENTITY, ink(color), None, &(full * path));
            }
        }
        Target::Fill { .. } => {
            if let Some(path) = boundary_path(piece) {
                let screen = view.scene_transform(size, 1.0) * path;
                scene.stroke(&stroke, Affine::IDENTITY, ink(color), None, &screen);
            }
        }
    }
}

/// Per-piece scene cache with id-keyed dirty flags (guide §7.3).
///
/// The contract: tell it which pieces changed — [`SceneCache::mark_from_edits`]
/// does that straight from a transaction — and it rebuilds only those. If the
/// document revision moves by more than the marks account for, it rebuilds
/// everything, so a caller who forgets is slow rather than wrong.
///
/// "Account for" is exact, not a heuristic: [`Document::apply_edit`] bumps the
/// revision once per edit, so a transaction's `edits().len()` is precisely its
/// contribution, and [`SceneCache::mark_from_edits`] adds it up. Anything the
/// cache was not told about — an undo (`CommandStack::undo` returns a `bool`, so
/// there is nothing to hand to `mark_from_edits`), a rolled-back transaction, a
/// direct `apply_edit` — leaves the counts unequal and triggers the full rebuild.
///
/// The earlier gate was `revision changed && dirty.is_empty()`, i.e. the fallback
/// only fired when the caller had marked *nothing*. Mark one transaction, forget
/// another in the same frame, and the forgotten piece rendered stale forever —
/// which is exactly the shape undo/redo has.
pub struct SceneCache {
    pieces: BTreeMap<PieceId, Scene>,
    dirty: BTreeSet<PieceId>,
    zoom: f64,
    revision: u64,
    /// Revision ticks the caller has accounted for since the last [`Self::frame`].
    accounted: u64,
    piece_builds: u64,
}

impl Default for SceneCache {
    fn default() -> Self {
        SceneCache {
            pieces: BTreeMap::new(),
            dirty: BTreeSet::new(),
            zoom: f64::NAN,
            revision: u64::MAX,
            accounted: 0,
            piece_builds: 0,
        }
    }
}

impl SceneCache {
    pub fn new() -> SceneCache {
        SceneCache::default()
    }

    /// How many per-piece scenes have been encoded since this cache was made.
    /// The dirty-flag tests assert on this; do not remove it.
    pub fn piece_builds(&self) -> u64 {
        self.piece_builds
    }

    pub fn cached(&self) -> usize {
        self.pieces.len()
    }

    pub fn mark_dirty(&mut self, piece: PieceId) {
        self.dirty.insert(piece);
    }

    pub fn mark_all_dirty(&mut self) {
        self.pieces.clear();
    }

    /// The pieces a transaction touched. Every geometry `Edit` names its piece,
    /// so this is exact rather than a guess.
    ///
    /// Also records that `edits.len()` revision ticks are accounted for — see the
    /// type docs. Pass the transaction's WHOLE edit list, including the variants
    /// no piece scene depends on; the count is what keeps the staleness fallback
    /// honest, and dropping edits from it only costs a full rebuild.
    pub fn mark_from_edits(&mut self, edits: &[Edit]) {
        self.accounted += edits.len() as u64;
        for edit in edits {
            match edit {
                Edit::AddPiece { id, .. }
                | Edit::RemovePiece { id }
                | Edit::SetPieceName { id, .. }
                | Edit::SetPieceTransform { id, .. }
                | Edit::SetPieceMaterial { id, .. }
                | Edit::SetPieceAllowance { id, .. }
                | Edit::SetPieceSymmetry { id, .. }
                | Edit::SetPiecePlacement { id, .. } => {
                    self.dirty.insert(*id);
                }
                Edit::AddVertex { piece, .. }
                | Edit::RemoveVertex { piece, .. }
                | Edit::MoveVertex { piece, .. }
                | Edit::AddEdge { piece, .. }
                | Edit::RemoveEdge { piece, .. }
                | Edit::SetEdgeGeom { piece, .. }
                | Edit::SetEdgeSemantic { piece, .. }
                | Edit::SetEdgeEnds { piece, .. }
                | Edit::AddFeature { piece, .. }
                | Edit::RemoveFeature { piece, .. }
                | Edit::SetFeature { piece, .. } => {
                    self.dirty.insert(*piece);
                }
                // Seams, materials, params, avatar, camera, metadata: nothing
                // in a piece's own scene depends on them today. A material
                // colour would — mark the pieces using it here when it does.
                _ => {}
            }
        }
    }

    /// The whole frame, in logical units. The caller applies the HiDPI scale
    /// (guide §2.4), exactly as [`crate::build_scene`] does.
    pub fn frame(
        &mut self,
        doc: &Document,
        it: &Interaction,
        view: &View,
        size: Vec2,
        tokens: &Tokens,
    ) -> Scene {
        self.sync(doc, view.zoom);

        let mut scene = grid_scene(view, size, tokens);
        let to_view = view.scene_transform(size, 1.0);
        let px = 1.0 / view.zoom;

        for (id, piece) in doc.pieces().iter() {
            if self.dirty.contains(&id) || !self.pieces.contains_key(&id) {
                self.piece_builds += 1;
                self.pieces.insert(id, piece_scene(piece, tokens, px));
            }
            if let Some(cached) = self.pieces.get(&id) {
                scene.append(cached, Some(to_view));
            }
        }
        self.dirty.clear();

        scene.append(&overlay_scene(doc, it, view, size, tokens), None);
        scene
    }

    /// Drops what the view or the document has invalidated wholesale.
    fn sync(&mut self, doc: &Document, zoom: f64) {
        if self.zoom != zoom {
            // Stroke widths are baked in world units, so they are only correct
            // for the zoom they were built at.
            self.pieces.clear();
            self.zoom = zoom;
        }
        if doc.revision() != self.revision.wrapping_add(self.accounted) {
            // The document moved further than the marks explain: something changed
            // it without telling us which piece. Rebuild the lot.
            self.pieces.clear();
        }
        self.revision = doc.revision();
        self.accounted = 0;
        self.pieces.retain(|id, _| doc.pieces().contains(*id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{curved_piece, marker, square_doc};
    use nguo_doc_commands::CommandStack;

    const SIZE: Vec2 = Vec2::new(1200.0, 800.0);

    fn tokens() -> Tokens {
        Tokens::light()
    }

    #[test]
    fn a_piece_scene_encodes_something() {
        let (doc, piece, _) = square_doc();
        let scene = piece_scene(doc.piece(piece).unwrap(), &tokens(), 1.0);
        assert!(!scene.encoding().is_empty());
    }

    #[test]
    fn the_grid_encodes_and_survives_a_zero_sized_widget() {
        let t = tokens();
        assert!(!grid_scene(&View::default(), SIZE, &t).encoding().is_empty());
        assert!(grid_scene(&View::default(), Vec2::ZERO, &t)
            .encoding()
            .is_empty());
    }

    #[test]
    fn editing_one_piece_does_not_rebuild_the_others() {
        let mut doc = marker(10, 8);
        let mut stack = CommandStack::new();
        let mut cache = SceneCache::new();
        let it = Interaction::default();
        let view = View::default();
        let t = tokens();

        cache.frame(&doc, &it, &view, SIZE, &t);
        assert_eq!(cache.piece_builds(), 10, "the first frame builds all ten");

        // A second frame with nothing changed builds nothing at all.
        cache.frame(&doc, &it, &view, SIZE, &t);
        assert_eq!(cache.piece_builds(), 10, "an idle frame rebuilds nothing");

        // Move a vertex of piece 3 and mark from the transaction itself.
        let target = doc.pieces().ids().nth(3).unwrap();
        let v = doc
            .piece(target)
            .unwrap()
            .network
            .vertices
            .ids()
            .next()
            .unwrap();
        let tx = crate::pen::move_point(&doc, target, v, Point::new(1.0, 2.0)).unwrap();
        cache.mark_from_edits(tx.edits());
        stack.apply(&mut doc, tx).unwrap();

        cache.frame(&doc, &it, &view, SIZE, &t);
        assert_eq!(
            cache.piece_builds(),
            11,
            "exactly one piece rebuilt, not ten"
        );
    }

    #[test]
    fn a_drag_of_the_interaction_layer_alone_rebuilds_no_piece() {
        let doc = marker(10, 8);
        let mut cache = SceneCache::new();
        let view = View::default();
        let t = tokens();
        cache.frame(&doc, &Interaction::default(), &view, SIZE, &t);
        let after_first = cache.piece_builds();

        // 60 frames of hover/snap/selection churn: the whole point of §7.3.
        for i in 0..60 {
            let it = Interaction {
                hover: Some(Target::Fill {
                    piece: doc.pieces().ids().next().unwrap(),
                }),
                snap: Some(Snap {
                    world: Point::new(i as f64, 0.0),
                    kind: crate::snap::SnapKind::Grid,
                    screen_distance: 1.0,
                    source: None,
                }),
                ..Interaction::default()
            };
            cache.frame(&doc, &it, &view, SIZE, &t);
        }
        assert_eq!(
            cache.piece_builds(),
            after_first,
            "the interaction layer must not touch the piece cache"
        );
    }

    #[test]
    fn panning_does_not_invalidate_the_cache_but_zooming_does() {
        let doc = marker(4, 8);
        let mut cache = SceneCache::new();
        let mut view = View::default();
        let t = tokens();
        cache.frame(&doc, &Interaction::default(), &view, SIZE, &t);
        assert_eq!(cache.piece_builds(), 4);

        view.pan_by_screen(Vec2::new(120.0, -75.0));
        cache.frame(&doc, &Interaction::default(), &view, SIZE, &t);
        assert_eq!(cache.piece_builds(), 4, "pan is free — world-space cache");

        view.zoom_about(1.25, Point::new(10.0, 10.0), SIZE);
        cache.frame(&doc, &Interaction::default(), &view, SIZE, &t);
        assert_eq!(
            cache.piece_builds(),
            8,
            "zoom rebuilds: stroke widths are world units"
        );
    }

    #[test]
    fn a_document_change_with_no_marks_rebuilds_rather_than_going_stale() {
        let mut doc = marker(3, 8);
        let mut stack = CommandStack::new();
        let mut cache = SceneCache::new();
        let t = tokens();
        cache.frame(&doc, &Interaction::default(), &View::default(), SIZE, &t);
        assert_eq!(cache.piece_builds(), 3);

        // Deliberately do NOT call `mark_from_edits`.
        let id = doc.next_piece_id();
        stack
            .apply(
                &mut doc,
                Edit::AddPiece {
                    id,
                    piece: Box::new(curved_piece("Late", 8, 100.0)),
                },
            )
            .unwrap();
        cache.frame(&doc, &Interaction::default(), &View::default(), SIZE, &t);
        assert_eq!(
            cache.piece_builds(),
            3 + 4,
            "an unmarked change is slow, never stale"
        );
    }

    /// The half of the fallback the `dirty.is_empty()` gate never reached: one
    /// transaction marked, another in the same frame forgotten. The forgotten piece
    /// used to render its stale cached scene forever — `piece_builds` went 3 → 4
    /// (only the marked piece) where a fully unmarked frame correctly gives 6.
    ///
    /// Undo is exactly this shape: `CommandStack::undo` returns a `bool`, so there
    /// is nothing to hand to `mark_from_edits`, and any marked edit in the same
    /// frame used to suppress the rebuild.
    #[test]
    fn a_frame_that_mixes_marked_and_unmarked_changes_still_rebuilds() {
        let mut doc = marker(3, 8);
        let mut stack = CommandStack::new();
        let mut cache = SceneCache::new();
        let (it, view, t) = (Interaction::default(), View::default(), tokens());
        cache.frame(&doc, &it, &view, SIZE, &t);
        assert_eq!(cache.piece_builds(), 3);

        let ids: Vec<_> = doc.pieces().ids().collect();
        let moved = |doc: &Document, p| {
            let v = doc.piece(p).unwrap().network.vertices.ids().next().unwrap();
            crate::pen::move_point(doc, p, v, Point::new(7.0, 9.0)).unwrap()
        };

        // Piece 0: marked, the way a well-behaved caller does it.
        let marked = moved(&doc, ids[0]);
        cache.mark_from_edits(marked.edits());
        stack.apply(&mut doc, marked).unwrap();
        // Piece 1: applied in the same frame and NOT marked.
        let forgotten = moved(&doc, ids[1]);
        stack.apply(&mut doc, forgotten).unwrap();

        cache.frame(&doc, &it, &view, SIZE, &t);
        assert_eq!(
            cache.piece_builds(),
            3 + 3,
            "one forgotten transaction must still cost a full rebuild, not a stale piece"
        );
    }

    #[test]
    fn a_removed_piece_leaves_the_cache() {
        let mut doc = marker(3, 8);
        let mut stack = CommandStack::new();
        let mut cache = SceneCache::new();
        cache.frame(
            &doc,
            &Interaction::default(),
            &View::default(),
            SIZE,
            &tokens(),
        );
        assert_eq!(cache.cached(), 3);

        let gone = doc.pieces().ids().next().unwrap();
        cache.mark_dirty(gone);
        stack
            .apply(&mut doc, Edit::RemovePiece { id: gone })
            .unwrap();
        cache.frame(
            &doc,
            &Interaction::default(),
            &View::default(),
            SIZE,
            &tokens(),
        );
        assert_eq!(cache.cached(), 2, "the cache does not leak dead pieces");
    }

    #[test]
    fn mark_from_edits_names_the_piece_for_every_geometry_edit() {
        let (doc, piece, edges) = square_doc();
        let v = doc
            .piece(piece)
            .unwrap()
            .network
            .vertices
            .ids()
            .next()
            .unwrap();
        let cases = [
            Edit::MoveVertex {
                piece,
                id: v,
                pos: Point::ORIGIN,
            },
            Edit::SetEdgeGeom {
                piece,
                id: edges[0],
                geom: EdgeGeom::Line,
            },
            Edit::SetPieceTransform {
                id: piece,
                transform: Affine::IDENTITY,
            },
            Edit::RemoveEdge {
                piece,
                id: edges[0],
            },
        ];
        for edit in cases {
            let mut cache = SceneCache::new();
            cache.mark_from_edits(std::slice::from_ref(&edit));
            assert!(
                cache.dirty.contains(&piece),
                "{edit:?} did not mark its piece"
            );
        }

        // And a non-geometry edit does not mark anything.
        let mut cache = SceneCache::new();
        cache.mark_from_edits(&[Edit::SetParam {
            name: "ease".into(),
            param: None,
        }]);
        assert!(cache.dirty.is_empty());
    }

    #[test]
    fn the_overlay_draws_what_it_is_given_and_nothing_when_empty() {
        let (doc, piece, edges) = square_doc();
        let t = tokens();
        let view = View::default();
        assert!(
            overlay_scene(&doc, &Interaction::default(), &view, SIZE, &t)
                .encoding()
                .is_empty(),
            "an empty interaction state draws nothing"
        );
        let it = Interaction {
            selection: vec![Target::Edge {
                piece,
                edge: edges[0],
            }],
            hover: Some(Target::Point {
                piece,
                vertex: doc
                    .piece(piece)
                    .unwrap()
                    .network
                    .vertices
                    .ids()
                    .next()
                    .unwrap(),
            }),
            snap: Some(Snap {
                world: Point::new(5.0, 5.0),
                kind: crate::snap::SnapKind::Grid,
                screen_distance: 0.5,
                source: None,
            }),
            editing: vec![piece],
            rubber_band: Some((Point::ORIGIN, Point::new(10.0, 10.0))),
        };
        assert!(!overlay_scene(&doc, &it, &view, SIZE, &t)
            .encoding()
            .is_empty());
    }
}
