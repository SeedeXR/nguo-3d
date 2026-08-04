// SPDX-License-Identifier: GPL-2.0-or-later
//! Seam validation as a **maintained constraint**, not a command
//! (`ui/docs/UX_SPEC.md` §1).
//!
//! There is deliberately no "Check Sewing Length" menu item. [`validate`] is a
//! pure function of the document, cheap enough to run on every revision, and
//! the shell shows what it returns continuously. A length mismatch is therefore
//! *surfaced* — as ease while it is plausible, as an actionable error once it is
//! not — rather than discovered when somebody remembers to ask.
//!
//! Everything here is document geometry only: no triangulation, no arrangement,
//! no solver. That is what makes it affordable per keystroke.

use nguo_doc::{Document, PieceId, SeamId, SeamSide};
use nguo_pattern::{EdgeId, EdgeSemantic};

use crate::correspond::{check_chain, side_length_mm, SeamError, EASE_LIMIT};

/// How far the measured ease may drift from [`nguo_doc::Seam::ease_ratio`]
/// before the drift itself is worth reporting. 1 % of the seam length.
pub const DECLARED_EASE_TOLERANCE: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// The garment cannot be sewn as authored. Actionable, and never fatal to
    /// the app — the drape simply refuses until it is fixed.
    Error,
    /// Sewable, but probably not what was meant.
    Warning,
    /// A fact worth showing, e.g. how much ease a seam is absorbing.
    Info,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// An edge marked [`EdgeSemantic::Seam`] that no seam sews.
    Unsewn { piece: PieceId, edge: EdgeId },
    /// A seam side naming a piece that is not in the document.
    DanglingPiece { seam: SeamId, piece: PieceId },
    /// A seam side naming an edge that is not in that piece's network.
    DanglingEdge {
        seam: SeamId,
        piece: PieceId,
        edge: EdgeId,
    },
    /// A seam side whose `(edge, reversed)` chain is discontinuous — the
    /// direction mismatch that is visible without any geometry.
    BadDirection { seam: SeamId, piece: PieceId },
    /// The seam pairs the two sides only when one is reversed: as authored it is
    /// twisted. Only [`crate::build`] can see this — it needs the arrangement,
    /// so [`validate`] never emits it.
    Flipped { seam: SeamId },
    /// The seam absorbs a length difference as ease.
    Ease { seam: SeamId, ratio: f64 },
    /// The length difference is past what a plain seam absorbs.
    EaseTooLarge { seam: SeamId, ratio: f64 },
    /// The measured ease is not the ease the seam says it was built for.
    EaseNotAsDeclared {
        seam: SeamId,
        declared: f64,
        measured: f64,
    },
    /// The side could not be measured at all.
    Unmeasurable { seam: SeamId, error: SeamError },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub kind: Kind,
}

impl Diagnostic {
    /// The user-facing sentence. Every one of them names what to do, because a
    /// diagnostic that only names the problem is a dead end (`AGENT_BRIEF`:
    /// user-facing errors actionable and non-fatal).
    pub fn message(&self) -> String {
        match &self.kind {
            Kind::Unsewn { edge, .. } => format!(
                "edge {} is marked as a seam edge but is not sewn to anything — sew it, or set it \
                 to Cut or Fold if it is a finished edge",
                edge.raw()
            ),
            Kind::DanglingPiece { piece, .. } => format!(
                "the seam references piece {}, which is not in the document — delete the seam or \
                 undo the piece deletion",
                piece.raw()
            ),
            Kind::DanglingEdge { piece, edge, .. } => format!(
                "the seam references edge {} of piece {}, which no longer exists — re-sew that \
                 side",
                edge.raw(),
                piece.raw()
            ),
            Kind::BadDirection { piece, .. } => format!(
                "the seam's side on piece {} runs in two directions at once: its edges do not join \
                 end to start — flip the mis-ordered edge",
                piece.raw()
            ),
            Kind::Flipped { .. } => "the seam's two sides run in opposite directions, so it was \
                 sewn reversed to close it — check which way round the pieces go if the drape \
                 looks twisted"
                .to_owned(),
            Kind::Ease { ratio, .. } => format!(
                "the two sides differ by {:+.1} % of their length; the seam absorbs it as ease",
                (ratio - 1.0) * 100.0
            ),
            Kind::EaseTooLarge { ratio, .. } => format!(
                "the two sides differ by {:+.1} % of their length, past the {:.0} % a plain seam \
                 absorbs — shorten the longer side, or split the seam so the extra length is \
                 gathered where you want it",
                (ratio - 1.0) * 100.0,
                (EASE_LIMIT - 1.0) * 100.0
            ),
            Kind::EaseNotAsDeclared {
                declared, measured, ..
            } => format!(
                "the seam is set up for {:+.1} % ease but the pieces now differ by {:+.1} % — \
                 update the seam's ease or the piece geometry",
                (declared - 1.0) * 100.0,
                (measured - 1.0) * 100.0
            ),
            Kind::Unmeasurable { error, .. } => format!("the seam cannot be measured: {error}"),
        }
    }

    /// The seam this is about, if any.
    pub fn seam(&self) -> Option<SeamId> {
        match self.kind {
            Kind::Unsewn { .. } => None,
            Kind::DanglingPiece { seam, .. }
            | Kind::DanglingEdge { seam, .. }
            | Kind::BadDirection { seam, .. }
            | Kind::Flipped { seam }
            | Kind::Ease { seam, .. }
            | Kind::EaseTooLarge { seam, .. }
            | Kind::EaseNotAsDeclared { seam, .. }
            | Kind::Unmeasurable { seam, .. } => Some(seam),
        }
    }
}

/// Every diagnostic the document currently earns, in document order (pieces
/// then seams, both by id) so the list is stable frame to frame — a list that
/// reshuffles is unreadable in a panel.
pub fn validate(doc: &Document) -> Vec<Diagnostic> {
    let mut out = Vec::new();

    // 1. Unsewn seam edges. `Seam` is the semantic that *means* "sewn", so an
    //    unsewn one is either an unfinished draft or a mislabelled edge.
    for (pid, piece) in doc.pieces().iter() {
        for (eid, edge) in piece.network.edges.iter() {
            if edge.semantic != EdgeSemantic::Seam {
                continue;
            }
            if !doc
                .seams()
                .values()
                .any(|seam| seam.references_edge(pid, eid))
            {
                out.push(Diagnostic {
                    severity: Severity::Warning,
                    kind: Kind::Unsewn {
                        piece: pid,
                        edge: eid,
                    },
                });
            }
        }
    }

    // 2. Per seam: references resolve, each side has one direction, and the two
    //    lengths agree (or the difference is named).
    for (sid, seam) in doc.seams().iter() {
        let mut lengths = [None, None];
        for (slot, side) in [&seam.a, &seam.b].into_iter().enumerate() {
            match measure_side(doc, sid, side, &mut out) {
                Some(len) => lengths[slot] = Some(len),
                None => continue,
            }
        }
        let (Some(a), Some(b)) = (lengths[0], lengths[1]) else {
            continue;
        };
        let measured = b / a;
        let severity = if measured.max(1.0 / measured) > EASE_LIMIT {
            Severity::Error
        } else {
            Severity::Info
        };
        out.push(Diagnostic {
            severity,
            kind: if severity == Severity::Error {
                Kind::EaseTooLarge {
                    seam: sid,
                    ratio: measured,
                }
            } else {
                Kind::Ease {
                    seam: sid,
                    ratio: measured,
                }
            },
        });
        if (measured - seam.ease_ratio).abs() > DECLARED_EASE_TOLERANCE {
            out.push(Diagnostic {
                severity: Severity::Warning,
                kind: Kind::EaseNotAsDeclared {
                    seam: sid,
                    declared: seam.ease_ratio,
                    measured,
                },
            });
        }
    }

    out
}

/// Length of one side, pushing a diagnostic and returning `None` if it cannot
/// be measured.
fn measure_side(
    doc: &Document,
    seam: SeamId,
    side: &SeamSide,
    out: &mut Vec<Diagnostic>,
) -> Option<f64> {
    let Ok(piece) = doc.piece(side.piece) else {
        out.push(Diagnostic {
            severity: Severity::Error,
            kind: Kind::DanglingPiece {
                seam,
                piece: side.piece,
            },
        });
        return None;
    };
    if let Some(&(edge, _)) = side
        .edges
        .iter()
        .find(|&&(e, _)| !piece.network.edges.contains(e))
    {
        out.push(Diagnostic {
            severity: Severity::Error,
            kind: Kind::DanglingEdge {
                seam,
                piece: side.piece,
                edge,
            },
        });
        return None;
    }
    if check_chain(&piece.network, side).is_err() {
        out.push(Diagnostic {
            severity: Severity::Error,
            kind: Kind::BadDirection {
                seam,
                piece: side.piece,
            },
        });
        return None;
    }
    match side_length_mm(&piece.network, side) {
        Ok(len) => Some(len),
        Err(error) => {
            out.push(Diagnostic {
                severity: Severity::Error,
                kind: Kind::Unmeasurable { seam, error },
            });
            None
        }
    }
}

/// Is this document sewable at all? `true` when no diagnostic is an
/// [`Severity::Error`].
pub fn is_sewable(diagnostics: &[Diagnostic]) -> bool {
    !diagnostics.iter().any(|d| d.severity == Severity::Error)
}

/// `(errors, warnings, infos)`, for a status line.
pub fn counts(diagnostics: &[Diagnostic]) -> (usize, usize, usize) {
    diagnostics
        .iter()
        .fold((0, 0, 0), |(e, w, i), d| match d.severity {
            Severity::Error => (e + 1, w, i),
            Severity::Warning => (e, w + 1, i),
            Severity::Info => (e, w, i + 1),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nguo_doc::{Edit, Piece, Seam};
    use nguo_pattern::EdgeGeom;

    /// Two rectangles, `b` optionally taller, with the seam left to the test.
    fn doc_with(h_b: f64) -> (Document, Vec<PieceId>, Vec<Vec<EdgeId>>) {
        let mut doc = Document::new();
        let mut pieces = Vec::new();
        let mut edges = Vec::new();
        for (i, h) in [200.0, h_b].into_iter().enumerate() {
            let mut piece = Piece::new(format!("p{i}"));
            let v: Vec<_> = [(0.0, 0.0), (100.0, 0.0), (100.0, h), (0.0, h)]
                .iter()
                .map(|&(x, y)| piece.network.add_vertex(nguo_doc::kurbo::Point::new(x, y)))
                .collect();
            let e = (0..4)
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
            pieces.push(id);
            edges.push(e);
        }
        (doc, pieces, edges)
    }

    fn add_seam(doc: &mut Document, seam: Seam) -> SeamId {
        let id = doc.next_seam_id();
        doc.apply_edit(Edit::AddSeam {
            id,
            seam: Box::new(seam),
        })
        .unwrap();
        id
    }

    #[test]
    fn every_unsewn_seam_edge_is_flagged_and_sewing_one_clears_it() {
        let (mut doc, p, e) = doc_with(200.0);
        let before = validate(&doc);
        assert_eq!(
            before
                .iter()
                .filter(|d| matches!(d.kind, Kind::Unsewn { .. }))
                .count(),
            8,
            "two rectangles, four Seam edges each"
        );
        assert!(is_sewable(&before), "unsewn is a warning, not an error");

        add_seam(
            &mut doc,
            Seam::new(
                SeamSide::whole(p[0], vec![(e[0][1], false)]),
                SeamSide::whole(p[1], vec![(e[1][3], true)]),
            ),
        );
        let after = validate(&doc);
        assert_eq!(
            after
                .iter()
                .filter(|d| matches!(d.kind, Kind::Unsewn { .. }))
                .count(),
            6,
            "the two sewn edges are no longer unsewn"
        );
        assert!(after.iter().any(|d| matches!(d.kind, Kind::Ease { .. })));
        assert!(is_sewable(&after));
    }

    #[test]
    fn equal_sides_report_zero_ease_and_no_warning() {
        let (mut doc, p, e) = doc_with(200.0);
        let s = add_seam(
            &mut doc,
            Seam::new(
                SeamSide::whole(p[0], vec![(e[0][1], false)]),
                SeamSide::whole(p[1], vec![(e[1][3], true)]),
            ),
        );
        let d = validate(&doc);
        let ease = d
            .iter()
            .find(|d| d.seam() == Some(s) && matches!(d.kind, Kind::Ease { .. }))
            .expect("an ease report");
        assert_eq!(ease.severity, Severity::Info);
        assert!(matches!(ease.kind, Kind::Ease { ratio, .. } if (ratio - 1.0).abs() < 1e-9));
        assert!(
            !d.iter()
                .any(|d| matches!(d.kind, Kind::EaseNotAsDeclared { .. })),
            "a flat seam matches its default ease_ratio of 1.0"
        );
    }

    /// The distinction the brief asks for: a small mismatch is ease, a large one
    /// is an error. Both are the same code path, so the boundary is tested from
    /// both sides.
    #[test]
    fn a_length_mismatch_is_ease_below_the_limit_and_an_error_above_it() {
        // 220/200 = 1.10, inside EASE_LIMIT.
        let (mut doc, p, e) = doc_with(220.0);
        add_seam(
            &mut doc,
            Seam::new(
                SeamSide::whole(p[0], vec![(e[0][1], false)]),
                SeamSide::whole(p[1], vec![(e[1][3], true)]),
            ),
        );
        let d = validate(&doc);
        assert!(
            matches!(
                d.iter().find_map(|d| match d.kind {
                    Kind::Ease { ratio, .. } => Some(ratio),
                    _ => None,
                }),
                Some(r) if (r - 1.1).abs() < 1e-9
            ),
            "{:?}",
            d.iter().map(|d| d.message()).collect::<Vec<_>>()
        );
        assert!(is_sewable(&d));

        // 400/200 = 2.0, well past it.
        let (mut doc, p, e) = doc_with(400.0);
        add_seam(
            &mut doc,
            Seam::new(
                SeamSide::whole(p[0], vec![(e[0][1], false)]),
                SeamSide::whole(p[1], vec![(e[1][3], true)]),
            ),
        );
        let d = validate(&doc);
        let err = d
            .iter()
            .find(|d| matches!(d.kind, Kind::EaseTooLarge { .. }))
            .expect("2x is an error");
        assert_eq!(err.severity, Severity::Error);
        assert!(!is_sewable(&d));
        assert!(
            err.message().contains("gathered"),
            "the error must say what to do: {}",
            err.message()
        );
    }

    /// The ratio is direction-blind: b shorter than a by the same factor is
    /// just as wrong as b longer.
    #[test]
    fn the_ease_limit_applies_in_both_directions() {
        let (mut doc, p, e) = doc_with(100.0); // 100/200 = 0.5
        add_seam(
            &mut doc,
            Seam::new(
                SeamSide::whole(p[0], vec![(e[0][1], false)]),
                SeamSide::whole(p[1], vec![(e[1][3], true)]),
            ),
        );
        assert!(!is_sewable(&validate(&doc)));
    }

    #[test]
    fn a_side_whose_edges_do_not_join_is_a_direction_error() {
        let (mut doc, p, e) = doc_with(200.0);
        // bottom forward + right reversed: both end at (100, 0).
        add_seam(
            &mut doc,
            Seam::new(
                SeamSide::whole(p[0], vec![(e[0][0], false), (e[0][1], true)]),
                SeamSide::whole(p[1], vec![(e[1][3], true)]),
            ),
        );
        let d = validate(&doc);
        let bad = d
            .iter()
            .find(|d| matches!(d.kind, Kind::BadDirection { .. }))
            .expect("a discontinuous side is a direction error");
        assert_eq!(bad.severity, Severity::Error);
        assert!(!is_sewable(&d));
    }

    /// A dangling seam reference cannot be *edited* into a document —
    /// `nguo-doc` refuses both `AddSeam` with an unknown piece
    /// (`edit.rs:507` `check_seam`) and `RemovePiece` while a seam still names it
    /// (`StillReferenced`). It arrives one way only: through
    /// [`Document::from_json`], which migrates the schema but does **not**
    /// re-check referential integrity. That file boundary is the trust boundary,
    /// so that is where this is reproduced — deleting the piece through the edit
    /// API instead would assert on a state the model forbids and pass by
    /// panicking on the `unwrap`.
    #[test]
    fn a_seam_referencing_a_deleted_piece_is_an_error_not_a_panic() {
        let (mut doc, p, e) = doc_with(200.0);
        add_seam(
            &mut doc,
            Seam::new(
                SeamSide::whole(p[0], vec![(e[0][1], false)]),
                SeamSide::whole(p[1], vec![(e[1][3], true)]),
            ),
        );
        assert!(
            doc.apply_edit(Edit::RemovePiece { id: p[1] }).is_err(),
            "the document model let a referenced piece be deleted",
        );

        let mut value: serde_json::Value =
            serde_json::from_str(&doc.to_json().expect("serialises")).expect("valid json");
        let removed = value["pieces"]
            .as_object_mut()
            .expect("pieces is a map")
            .remove(&p[1].raw().to_string());
        assert!(
            removed.is_some(),
            "the second piece was not in the manifest"
        );
        let doc = nguo_doc::Document::from_value(value).expect("loads without integrity checks");
        assert_eq!(doc.pieces().len(), 1);

        let d = validate(&doc);
        assert!(
            d.iter()
                .any(|d| matches!(d.kind, Kind::DanglingPiece { .. })),
            "{:?}",
            d.iter().map(|d| d.message()).collect::<Vec<_>>()
        );
        assert!(!is_sewable(&d));
    }

    #[test]
    fn a_declared_ease_that_no_longer_matches_the_geometry_is_warned_about() {
        let (mut doc, p, e) = doc_with(200.0);
        let mut seam = Seam::new(
            SeamSide::whole(p[0], vec![(e[0][1], false)]),
            SeamSide::whole(p[1], vec![(e[1][3], true)]),
        );
        seam.ease_ratio = 1.05;
        add_seam(&mut doc, seam);
        let d = validate(&doc);
        assert!(d
            .iter()
            .any(|d| matches!(d.kind, Kind::EaseNotAsDeclared { .. })));
    }

    /// It is a maintained constraint, so it has to be cheap. Measured, not
    /// asserted: this prints the real number and fails only if it is far enough
    /// off that it could not run per keystroke.
    #[test]
    fn validation_is_cheap_enough_to_run_on_every_revision() {
        let (mut doc, p, e) = doc_with(200.0);
        for (a, b) in e[0].iter().zip(&e[1]) {
            add_seam(
                &mut doc,
                Seam::new(
                    SeamSide::whole(p[0], vec![(*a, false)]),
                    SeamSide::whole(p[1], vec![(*b, true)]),
                ),
            );
        }
        let runs = 200;
        let started = std::time::Instant::now();
        let mut total = 0;
        for _ in 0..runs {
            total += validate(&doc).len();
        }
        let per_call_us = started.elapsed().as_secs_f64() * 1e6 / runs as f64;
        // fd 2, so it is visible even under libtest's capture.
        let _ = writeln!(
            std::io::stderr(),
            "validate(): {per_call_us:.1} us/call over {runs} calls, {} diagnostics",
            total / runs
        );
        assert!(total > 0, "the test document must produce diagnostics");
        assert!(
            per_call_us < 1000.0,
            "{per_call_us:.1} us/call is too slow for a per-keystroke constraint"
        );
    }

    use std::io::Write as _;
}
