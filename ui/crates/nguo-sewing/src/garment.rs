// SPDX-License-Identifier: GPL-2.0-or-later
//! Draft → triangulate → sew → arrange, in one call.
//!
//! [`build`] turns a [`Document`] into a [`Garment`]: one concatenated cloth
//! mesh in world metres, plus the stitch list that sews it together. That pair
//! is exactly what the solver takes — `engine_api::Scene` for the mesh,
//! `engine_core::implicit::Stitch` for the seams — but nothing in this crate
//! knows either type. The engine seam is `nguo-engine-bridge`'s job; keeping it
//! out of here is what lets every test in this crate run in microseconds with no
//! solver in the build.
//!
//! # Why one mesh and not one object per piece
//!
//! `Scene::lower` concatenates all objects into one vertex space anyway, and a
//! stitch indexes that space. Concatenating here means the indices in
//! [`Garment::stitches`] are the garment's own and need one offset (the body's
//! vertex count) rather than a per-piece table the caller has to keep straight.

use nguo_doc::Document;
use nguo_triangulate::{triangulate, SimMesh, TriangulateOptions};

use crate::arrange::{arrange, seam_order, Aabb, ArrangeOptions, PieceExtent, Placement3d};
use crate::correspond::{
    correspond, pairing_distance_m, side_samples, SeamError, SewnSeam, Stitch6,
};
use crate::validate::{is_sewable, validate, Diagnostic, Kind, Severity};

#[derive(Debug, Clone, PartialEq)]
pub struct SewOptions {
    pub triangulate: TriangulateOptions,
    pub arrange: ArrangeOptions,
}

impl Default for SewOptions {
    fn default() -> Self {
        SewOptions {
            // 30 mm, not the 10 mm `nguo-triangulate` defaults to. That is a
            // *measured* choice, not a guess: a 500 x 560 mm panel at 10 mm is
            // ~5 600 triangles, and two of them put ~11 000 cloth triangles into
            // an implicit solve that also carries the body. See
            // `tests/end_to_end.rs::mesh_resolution_cost` for the numbers.
            triangulate: TriangulateOptions::default().with_target_edge_mm(30.0),
            arrange: ArrangeOptions::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SewError {
    #[error("the garment cannot be sewn as authored: {}", first_error(.0))]
    NotSewable(Vec<Diagnostic>),
    #[error("piece '{piece}' could not be triangulated: {source}")]
    Triangulate {
        piece: String,
        #[source]
        source: nguo_triangulate::TriangulateError,
    },
    #[error("seam {seam} could not be resolved onto the mesh: {source}")]
    Seam {
        seam: u64,
        #[source]
        source: SeamError,
    },
    #[error("the document has no pieces to sew")]
    Empty,
}

fn first_error(d: &[Diagnostic]) -> String {
    d.iter()
        .find(|d| d.severity == Severity::Error)
        .map(|d| d.message())
        .unwrap_or_else(|| "no error reported".to_owned())
}

/// One piece, triangulated and placed.
#[derive(Debug, Clone, PartialEq)]
pub struct PieceMesh {
    pub piece: nguo_doc::PieceId,
    pub name: String,
    pub mesh: SimMesh,
    pub placement: Placement3d,
    /// Where this piece's vertices start in [`Garment::positions`].
    pub vertex_offset: u32,
}

/// Wall-clock cost of each stage, milliseconds. Reported rather than logged so
/// the shell can show it and a test can assert on it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Timings {
    pub validate_ms: f64,
    pub triangulate_ms: f64,
    pub arrange_ms: f64,
    pub sew_ms: f64,
}

impl Timings {
    pub fn total_ms(&self) -> f64 {
        self.validate_ms + self.triangulate_ms + self.arrange_ms + self.sew_ms
    }
}

/// A sewn, arranged garment, ready to submit.
#[derive(Debug, Clone, PartialEq)]
pub struct Garment {
    /// World metres, `+Y` up / `+Z` forward — glTF space, no conversion
    /// anywhere on this path.
    pub positions: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
    /// Material-space UVs from the flat pattern. A garment is developable from
    /// its pattern by construction, so the flat coordinates *are* the UVs — and
    /// the solver needs them: the anisotropic warp/weft membrane is inert
    /// without them.
    pub uvs: Vec<[f32; 2]>,
    pub stitches: Vec<Stitch6>,
    pub pieces: Vec<PieceMesh>,
    pub seams: Vec<SewnSeam>,
    /// Everything [`validate`] said, plus anything the geometry added (a flipped
    /// seam). Never contains an [`Severity::Error`] — that is a [`SewError`].
    pub diagnostics: Vec<Diagnostic>,
    pub timings: Timings,
    /// What the arranger decided.
    pub arrangement: crate::arrange::Arrangement,
}

impl Garment {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    pub fn bounds(&self) -> Option<Aabb> {
        Aabb::of(&self.positions)
    }

    /// Total distance the seams still have to close, metres. `0` means every
    /// stitch is already coincident; it is the honest measure of "is it sewn".
    pub fn seam_gap_m(&self) -> f64 {
        self.stitches
            .iter()
            .map(|s| {
                let p = self.stitch_point(&s.index[0..3], &s.weight[0..3]);
                let q = self.stitch_point(&s.index[3..6], &s.weight[3..6]);
                ((0..3)
                    .map(|k| (p[k] - q[k]) as f64)
                    .map(|d| d * d)
                    .sum::<f64>())
                .sqrt()
            })
            .sum()
    }

    /// The largest single seam gap, metres — the one a user would see.
    pub fn max_seam_gap_m(&self) -> f64 {
        self.stitches
            .iter()
            .map(|s| {
                let p = self.stitch_point(&s.index[0..3], &s.weight[0..3]);
                let q = self.stitch_point(&s.index[3..6], &s.weight[3..6]);
                ((0..3)
                    .map(|k| (p[k] - q[k]) as f64)
                    .map(|d| d * d)
                    .sum::<f64>())
                .sqrt()
            })
            .fold(0.0, f64::max)
    }

    fn stitch_point(&self, index: &[u32], weight: &[f32]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (&i, &w) in index.iter().zip(weight) {
            if w == 0.0 {
                continue;
            }
            if let Some(p) = self.positions.get(i as usize) {
                for k in 0..3 {
                    out[k] += w * p[k];
                }
            }
        }
        out
    }

    /// Everything a mesh handed to a solver must satisfy. Cheap; call it before
    /// submitting rather than debugging a NaN four seconds into a solve.
    pub fn check(&self) -> Result<(), String> {
        if self.positions.is_empty() || self.tris.is_empty() {
            return Err("the garment has no geometry".into());
        }
        if self.uvs.len() != self.positions.len() {
            return Err(format!(
                "{} uvs for {} vertices",
                self.uvs.len(),
                self.positions.len()
            ));
        }
        if let Some(i) = self
            .positions
            .iter()
            .position(|p| !p.iter().all(|c| c.is_finite()))
        {
            return Err(format!("vertex {i} is not finite: {:?}", self.positions[i]));
        }
        let n = self.positions.len() as u32;
        if let Some(t) = self.tris.iter().find(|t| t.iter().any(|&i| i >= n)) {
            return Err(format!("triangle {t:?} indexes past {n} vertices"));
        }
        if let Some(s) = self
            .stitches
            .iter()
            .find(|s| s.index.iter().any(|&i| i >= n))
        {
            return Err(format!("stitch {:?} indexes past {n} vertices", s.index));
        }
        for s in &self.stitches {
            for half in [&s.weight[0..3], &s.weight[3..6]] {
                let sum: f32 = half.iter().sum();
                if (sum - 1.0).abs() > 1e-4 {
                    return Err(format!("stitch half weights sum to {sum}, not 1"));
                }
            }
        }
        Ok(())
    }
}

/// Draft → triangulate → sew → arrange.
///
/// Fails only on things the user can act on: an unsewable document (which
/// carries the diagnostics), a piece that is not a closed region, a seam that
/// does not resolve onto its own mesh.
pub fn build(doc: &Document, opts: &SewOptions) -> Result<Garment, SewError> {
    let mut timings = Timings::default();

    let t = std::time::Instant::now();
    let mut diagnostics = validate(doc);
    timings.validate_ms = ms(t);
    if !is_sewable(&diagnostics) {
        return Err(SewError::NotSewable(diagnostics));
    }

    let order = seam_order(doc);
    if order.is_empty() {
        return Err(SewError::Empty);
    }

    // 1. Triangulate every piece, in wrap order.
    let t = std::time::Instant::now();
    let mut meshes = Vec::with_capacity(order.len());
    for &pid in &order {
        let Ok(piece) = doc.piece(pid) else { continue };
        // Notches are registration marks the seam has to be able to pin to, so
        // they become mesh vertices exactly on their edge.
        let notches: Vec<_> = piece
            .features
            .values()
            .filter_map(|f| match f {
                nguo_doc::Feature::Notch { edge, t, .. } => Some((*edge, *t)),
                _ => None,
            })
            .collect();
        let mesh = triangulate(&piece.network, &notches, &opts.triangulate).map_err(|source| {
            SewError::Triangulate {
                piece: piece.name.clone(),
                source,
            }
        })?;
        meshes.push((pid, piece, mesh));
    }
    timings.triangulate_ms = ms(t);

    // 2. Arrange.
    let t = std::time::Instant::now();
    let extents: Vec<PieceExtent> = meshes
        .iter()
        .map(|(pid, piece, mesh)| {
            let b = mesh.pattern_bounds();
            PieceExtent {
                piece: *pid,
                bounds_mm: [b.x0, b.y0, b.x1, b.y1],
                manual: piece.placement,
                support_arc_mm: crate::arrange::support_arc_mm(piece),
            }
        })
        .collect();
    let arrangement = arrange(&extents, &opts.arrange);
    timings.arrange_ms = ms(t);

    // 3. Concatenate, placed.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut pieces: Vec<PieceMesh> = Vec::new();
    for ((pid, piece, mesh), placement) in meshes.into_iter().zip(arrangement.placements.iter()) {
        let base = positions.len() as u32;
        positions.extend(mesh.pattern_mm.iter().map(|&p| placement.apply(p)));
        uvs.extend(mesh.uvs());
        // The placements mirror the pattern plane (pattern +y is world -Y), so a
        // counter-clockwise pattern triangle arrives clockwise. Re-wind, or every
        // garment normal points inward.
        tris.extend(mesh.indices.iter().map(|t| {
            if placement.flips_winding() {
                [t[0] + base, t[2] + base, t[1] + base]
            } else {
                [t[0] + base, t[1] + base, t[2] + base]
            }
        }));
        pieces.push(PieceMesh {
            piece: pid,
            name: piece.name.clone(),
            mesh,
            placement: *placement,
            vertex_offset: base,
        });
    }

    // 4. Sew.
    let t = std::time::Instant::now();
    let mut stitches = Vec::new();
    let mut seams = Vec::new();
    for (sid, seam) in doc.seams().iter() {
        let mut resolved = Vec::with_capacity(2);
        for side in [&seam.a, &seam.b] {
            let Some(pm) = pieces.iter().find(|p| p.piece == side.piece) else {
                // `validate` already reported it as a dangling reference, and it
                // was not an error, so the piece simply is not in the garment.
                break;
            };
            let piece = doc.piece(side.piece).map_err(|_| SewError::Empty)?;
            let mut samples =
                side_samples(&piece.network, &pm.mesh, side).map_err(|source| SewError::Seam {
                    seam: sid.raw(),
                    source,
                })?;
            for v in &mut samples.vertices {
                v.0 += pm.vertex_offset;
            }
            resolved.push(samples);
        }
        let [a, b] = match <[_; 2]>::try_from(resolved) {
            Ok(pair) => pair,
            Err(_) => continue,
        };

        // Direction: pair straight or reversed, whichever actually brings the
        // two sides together. A twisted seam is the classic sewing mistake and
        // this is where the geometry can see it — `validate` can only see a
        // discontinuous chain.
        let straight = pairing_distance_m(&a, &b, &positions, false);
        let flipped = pairing_distance_m(&a, &b, &positions, true);
        let flip = flipped < 0.5 * straight;
        if flip {
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                kind: Kind::Flipped { seam: sid },
            });
        }

        let c = correspond(&a, &b, flip);
        seams.push(SewnSeam {
            seam: sid,
            stitches: c.stitches.len(),
            ease: c.ease(),
            flipped: flip,
            open_m: if flip { flipped } else { straight },
        });
        stitches.extend(c.stitches);
    }
    timings.sew_ms = ms(t);

    let garment = Garment {
        positions,
        tris,
        uvs,
        stitches,
        pieces,
        seams,
        diagnostics,
        timings,
        arrangement,
    };
    if let Err(why) = garment.check() {
        // An internal invariant, not a user mistake — but reported, never
        // panicked, because a bad mesh must not reach the solver unnoticed.
        return Err(SewError::Triangulate {
            piece: "<garment>".into(),
            source: nguo_triangulate::TriangulateError::Invalid(why),
        });
    }
    tracing::info!(
        pieces = garment.pieces.len(),
        vertices = garment.vertex_count(),
        triangles = garment.triangle_count(),
        seams = garment.seams.len(),
        stitches = garment.stitches.len(),
        max_seam_gap_mm = garment.max_seam_gap_m() * 1000.0,
        ms = garment.timings.total_ms(),
        "garment sewn and arranged"
    );
    Ok(garment)
}

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
