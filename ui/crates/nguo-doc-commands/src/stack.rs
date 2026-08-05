// SPDX-License-Identifier: GPL-2.0-or-later
//! Transactions and the undo/redo stack.

use nguo_doc::{DocError, Document, Edit};

/// Identifies one continuous user gesture (a drag, a scrub, a nudge-repeat).
///
/// Commands sharing a gesture id coalesce into a single undo entry. Allocate
/// one per gesture and never reuse it — reuse would silently merge two
/// unrelated actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GestureId(pub u64);

/// A group of edits that becomes exactly one undo entry.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transaction {
    label: String,
    gesture: Option<GestureId>,
    edits: Vec<Edit>,
}

impl Transaction {
    pub fn new(label: impl Into<String>) -> Transaction {
        Transaction {
            label: label.into(),
            gesture: None,
            edits: Vec::new(),
        }
    }

    /// Builder-style append.
    pub fn edit(mut self, e: Edit) -> Transaction {
        self.edits.push(e);
        self
    }

    pub fn push(&mut self, e: Edit) {
        self.edits.push(e);
    }

    pub fn extend(&mut self, edits: impl IntoIterator<Item = Edit>) {
        self.edits.extend(edits);
    }

    /// Mark this as part of a continuous gesture so it merges with the
    /// preceding entry from the same gesture.
    pub fn with_gesture(mut self, g: GestureId) -> Transaction {
        self.gesture = Some(g);
        self
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn gesture(&self) -> Option<GestureId> {
        self.gesture
    }

    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn len(&self) -> usize {
        self.edits.len()
    }
}

/// A single edit is a single-entry transaction, labelled from the edit.
impl From<Edit> for Transaction {
    fn from(e: Edit) -> Transaction {
        Transaction {
            label: e.label().to_string(),
            gesture: None,
            edits: vec![e],
        }
    }
}

/// One entry on the history stack.
///
/// `redo` applied in order performs the command; `undo` applied in order
/// reverses it. Both are exact — they came out of
/// [`Document::apply_edit`], not from a second implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    label: String,
    gesture: Option<GestureId>,
    redo: Vec<Edit>,
    undo: Vec<Edit>,
}

impl Entry {
    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn gesture(&self) -> Option<GestureId> {
        self.gesture
    }

    /// The edits this entry replays, in order. Useful for an autosave journal
    /// (guide §5.4) and for inspecting history in the debug overlay.
    pub fn edits(&self) -> &[Edit] {
        &self.redo
    }
}

/// Linear undo + redo (guide §5.3).
#[derive(Debug, Clone)]
pub struct CommandStack {
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    limit: usize,
}

impl Default for CommandStack {
    fn default() -> Self {
        CommandStack::new()
    }
}

impl CommandStack {
    /// **256 entries.** A drag is one entry, so this is a long session's worth.
    ///
    /// The cap is an entry count and not a byte budget because the per-entry
    /// cost is now bounded by what the entry touches in the document, which the
    /// document already bounds. Measured on this tree with a counting global
    /// allocator (`tests/memory.rs`), live heap the stack retains:
    ///
    /// | entry | bytes |
    /// |---|---|
    /// | add one point | **185** |
    /// | 256 of them (the full cap) | 47 360 |
    /// | one 20 000-move drag | 635 |
    /// | delete a 200-vertex piece | 37 361 |
    ///
    /// So a full history of ordinary edits is 46 KiB and a full history of
    /// piece deletions is single-digit MB. The one shape that had **no** bound
    /// was a coalesced drag — see the collapse in [`CommandStack::apply`].
    ///
    /// ponytail: the residual ceiling is 256 × the largest piece in the
    /// document, because a delete's inverse carries the piece back. Add a byte
    /// budget if a real project ever makes that hurt; it needs a size function
    /// per `Edit` variant, which is not worth writing on a 47 KiB measurement.
    pub const DEFAULT_LIMIT: usize = 256;

    pub fn new() -> CommandStack {
        CommandStack::with_limit(CommandStack::DEFAULT_LIMIT)
    }

    pub fn with_limit(limit: usize) -> CommandStack {
        CommandStack {
            undo: Vec::new(),
            redo: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Apply a transaction and record it.
    ///
    /// Atomic: if any edit fails, the ones already applied are rolled back and
    /// the document is exactly as it was. Nothing is recorded.
    pub fn apply(
        &mut self,
        doc: &mut Document,
        tx: impl Into<Transaction>,
    ) -> Result<(), DocError> {
        let tx = tx.into();
        if tx.is_empty() {
            return Ok(());
        }
        let mut inverses: Vec<Edit> = Vec::with_capacity(tx.edits.len());
        for edit in &tx.edits {
            match doc.apply_edit(edit.clone()) {
                Ok(inv) => inverses.push(inv),
                Err(e) => {
                    rollback(doc, &mut inverses);
                    return Err(e);
                }
            }
        }

        let coalesced = tx.gesture.is_some()
            && self
                .undo
                .last()
                .is_some_and(|last| last.gesture == tx.gesture);
        if coalesced {
            let last = self.undo.last_mut().expect("checked above");
            // Collapse the scrub rather than append it. A drag emits one
            // `MoveVertex` per pointer move, and the intermediate positions are
            // unobservable the moment the next one lands — so an entry that
            // keeps them all grows without any bound. Measured: 20 000 moves of
            // one vertex (about five minutes of dragging) held **5 767 363 B**
            // in a single entry appending, 635 B collapsing.
            //
            // Replacing in place and keeping the ORIGINAL inverse is exact:
            // `undo` restores the value from before the gesture started, which
            // is what one undo of a whole drag has to do.
            //
            // ponytail: O(distinct targets) scan per edit, not O(moves) —
            // replacing in place is what keeps the vector short. A drag over a
            // whole selection is still linear in the selection, which is the
            // right size for it to be.
            let mut kept: Vec<Edit> = Vec::new();
            for (edit, inverse) in tx.edits.into_iter().zip(inverses) {
                match last.redo.iter_mut().rev().find(|p| supersedes(p, &edit)) {
                    Some(slot) => *slot = edit,
                    None => {
                        last.redo.push(edit);
                        kept.push(inverse);
                    }
                }
            }
            // Undo runs newest-first.
            kept.reverse();
            last.undo.splice(0..0, kept);
        } else {
            // Undo runs newest-first.
            inverses.reverse();
            self.undo.push(Entry {
                label: tx.label,
                gesture: tx.gesture,
                redo: tx.edits,
                undo: inverses,
            });
            while self.undo.len() > self.limit {
                self.undo.remove(0);
            }
        }
        self.redo.clear();
        Ok(())
    }

    /// Undo the newest entry. `Ok(false)` when there is nothing to undo.
    pub fn undo(&mut self, doc: &mut Document) -> Result<bool, DocError> {
        let Some(entry) = self.undo.pop() else {
            return Ok(false);
        };
        match run(doc, &entry.undo) {
            Ok(()) => {
                self.redo.push(entry);
                Ok(true)
            }
            Err(e) => {
                self.undo.push(entry);
                Err(e)
            }
        }
    }

    /// Redo the newest undone entry. `Ok(false)` when there is nothing to redo.
    pub fn redo(&mut self, doc: &mut Document) -> Result<bool, DocError> {
        let Some(entry) = self.redo.pop() else {
            return Ok(false);
        };
        match run(doc, &entry.redo) {
            Ok(()) => {
                self.undo.push(entry);
                Ok(true)
            }
            Err(e) => {
                self.redo.push(entry);
                Err(e)
            }
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// For the Edit menu: "Undo Add dart".
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(Entry::label)
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(Entry::label)
    }

    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_depth(&self) -> usize {
        self.redo.len()
    }

    /// How many entries are currently applied to the document.
    ///
    /// Also the index into [`CommandStack::entries`] where the **undone tail**
    /// starts: everything from here on has been rolled back and is what a
    /// history panel greys out.
    pub fn position(&self) -> usize {
        self.undo.len()
    }

    /// Every entry the stack is holding, **oldest first** — the applied ones,
    /// then the undone tail in the order redo would replay it.
    ///
    /// The tail is `redo` read backwards because `redo` is a stack: its last
    /// element is the next one [`CommandStack::redo`] pops, i.e. the *oldest*
    /// undone entry.
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &Entry> + Clone {
        self.undo.iter().chain(self.redo.iter().rev())
    }

    /// Applied plus undone. `position() == total()` means nothing is greyed.
    pub fn total(&self) -> usize {
        self.undo.len() + self.redo.len()
    }

    /// Move the document to the state it had after `position` entries — the
    /// Photoshop history click, in either direction.
    ///
    /// `0` is the state before the oldest entry the stack still holds;
    /// [`CommandStack::total`] is everything redone. Positions past `total` are
    /// clamped. `Ok(false)` when already there.
    ///
    /// Just repeated [`CommandStack::undo`]/[`CommandStack::redo`], which is the
    /// point: a linear history (D-2) makes jumping a walk, and reusing the two
    /// steps means there is no second implementation to drift. On error it stops
    /// where it got to — every step is individually atomic, so the document is
    /// consistent and [`CommandStack::position`] says where.
    pub fn jump_to(&mut self, doc: &mut Document, position: usize) -> Result<bool, DocError> {
        let target = position.min(self.total());
        let from = self.undo.len();
        while self.undo.len() > target {
            if !self.undo(doc)? {
                break;
            }
        }
        while self.undo.len() < target {
            if !self.redo(doc)? {
                break;
            }
        }
        Ok(self.undo.len() != from)
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// Does `next` make `prev` unobservable?
///
/// True only for a last-write-wins setter on the same target: once `next` has
/// run, nothing can tell whether `prev` ever did. That is what makes dropping
/// `prev` from a coalesced entry exact rather than lossy.
///
/// These two are precisely what a shipped gesture emits — `nguo-canvas`
/// `pen::move_point` produces `MoveVertex`, and `pen::move_handle` /
/// `PenState::drag_handle` produce `SetEdgeGeom`.
///
/// ponytail: every other `Set*` variant is equally collapsible; add the arm when
/// something actually coalesces one. An unlisted pair only falls through to the
/// append path, so a miss costs memory, never correctness.
fn supersedes(prev: &Edit, next: &Edit) -> bool {
    match (prev, next) {
        (
            Edit::MoveVertex {
                piece: p0, id: i0, ..
            },
            Edit::MoveVertex {
                piece: p1, id: i1, ..
            },
        ) => p0 == p1 && i0 == i1,
        (
            Edit::SetEdgeGeom {
                piece: p0, id: i0, ..
            },
            Edit::SetEdgeGeom {
                piece: p1, id: i1, ..
            },
        ) => p0 == p1 && i0 == i1,
        _ => false,
    }
}

/// Apply a run of edits, discarding the inverses.
fn run(doc: &mut Document, edits: &[Edit]) -> Result<(), DocError> {
    let mut inverses = Vec::with_capacity(edits.len());
    for e in edits {
        match doc.apply_edit(e.clone()) {
            Ok(inv) => inverses.push(inv),
            Err(err) => {
                rollback(doc, &mut inverses);
                return Err(err);
            }
        }
    }
    Ok(())
}

/// Put the document back after a partially applied run.
///
/// These inverses came straight out of `apply_edit` against this document, so
/// they cannot fail; if one ever did, the document would be inconsistent and we
/// want to know in a debug build rather than limp on.
fn rollback(doc: &mut Document, inverses: &mut Vec<Edit>) {
    while let Some(inv) = inverses.pop() {
        let r = doc.apply_edit(inv);
        debug_assert!(r.is_ok(), "rollback failed: {:?}", r.err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::Point;
    use nguo_doc::{Piece, PieceId};
    use nguo_pattern::VertexId;

    fn seeded() -> (Document, CommandStack, PieceId) {
        let mut doc = Document::new();
        let mut stack = CommandStack::new();
        let id = doc.next_piece_id();
        stack
            .apply(
                &mut doc,
                Edit::AddPiece {
                    id,
                    piece: Box::new(Piece::new("Front")),
                },
            )
            .unwrap();
        (doc, stack, id)
    }

    fn add_point(doc: &Document, piece: PieceId, x: f64, y: f64) -> (Edit, VertexId) {
        let id = doc.piece(piece).unwrap().network.vertices.next_id();
        (
            Edit::AddVertex {
                piece,
                id,
                pos: Point::new(x, y),
            },
            id,
        )
    }

    #[test]
    fn undo_and_redo_walk_the_history() {
        let (mut doc, mut stack, pid) = seeded();
        let empty = doc.to_json().unwrap();

        let (e, _) = add_point(&doc, pid, 1.0, 2.0);
        stack.apply(&mut doc, e).unwrap();
        let (e, _) = add_point(&doc, pid, 3.0, 4.0);
        stack.apply(&mut doc, e).unwrap();
        let full = doc.to_json().unwrap();

        assert_eq!(stack.undo_depth(), 3);
        assert!(stack.undo(&mut doc).unwrap());
        assert!(stack.undo(&mut doc).unwrap());
        assert_eq!(doc.to_json().unwrap(), empty);

        assert!(stack.redo(&mut doc).unwrap());
        assert!(stack.redo(&mut doc).unwrap());
        assert_eq!(doc.to_json().unwrap(), full);
        assert!(!stack.redo(&mut doc).unwrap());
    }

    #[test]
    fn a_transaction_is_one_undo_entry() {
        let (mut doc, mut stack, pid) = seeded();
        let before = doc.to_json().unwrap();

        let (e1, v1) = add_point(&doc, pid, 0.0, 0.0);
        let v2 = nguo_pattern::Id::from_raw(v1.raw() + 1);
        let tx = Transaction::new("Draw line")
            .edit(e1)
            .edit(Edit::AddVertex {
                piece: pid,
                id: v2,
                pos: Point::new(10.0, 0.0),
            })
            .edit(Edit::AddEdge {
                piece: pid,
                id: doc.piece(pid).unwrap().network.edges.next_id(),
                edge: nguo_pattern::Edge {
                    start: v1,
                    end: v2,
                    geom: nguo_pattern::EdgeGeom::Line,
                    semantic: nguo_pattern::EdgeSemantic::Seam,
                },
            });

        let depth = stack.undo_depth();
        stack.apply(&mut doc, tx).unwrap();
        assert_eq!(stack.undo_depth(), depth + 1, "three edits, one entry");
        assert_eq!(doc.piece(pid).unwrap().network.edges.len(), 1);

        assert_eq!(stack.undo_label(), Some("Draw line"));
        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.to_json().unwrap(), before, "one undo must take it all");
    }

    #[test]
    fn a_drag_coalesces_into_one_entry() {
        let (mut doc, mut stack, pid) = seeded();
        let (e, vid) = add_point(&doc, pid, 0.0, 0.0);
        stack.apply(&mut doc, e).unwrap();
        let anchor = doc.to_json().unwrap();
        let depth = stack.undo_depth();

        let g = GestureId(7);
        for i in 1..=25 {
            let tx = Transaction::new("Move point")
                .with_gesture(g)
                .edit(Edit::MoveVertex {
                    piece: pid,
                    id: vid,
                    pos: Point::new(i as f64, i as f64 * 2.0),
                });
            stack.apply(&mut doc, tx).unwrap();
        }
        assert_eq!(
            stack.undo_depth(),
            depth + 1,
            "25 moves must land in a single entry"
        );
        assert_eq!(
            doc.piece(pid).unwrap().network.pos(vid).unwrap(),
            Point::new(25.0, 50.0)
        );

        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.to_json().unwrap(), anchor, "one undo cancels the drag");
        stack.redo(&mut doc).unwrap();
        assert_eq!(
            doc.piece(pid).unwrap().network.pos(vid).unwrap(),
            Point::new(25.0, 50.0)
        );
    }

    #[test]
    fn separate_gestures_do_not_merge() {
        let (mut doc, mut stack, pid) = seeded();
        let (e, vid) = add_point(&doc, pid, 0.0, 0.0);
        stack.apply(&mut doc, e).unwrap();
        let depth = stack.undo_depth();
        for (i, g) in [GestureId(1), GestureId(2)].into_iter().enumerate() {
            stack
                .apply(
                    &mut doc,
                    Transaction::new("Move point")
                        .with_gesture(g)
                        .edit(Edit::MoveVertex {
                            piece: pid,
                            id: vid,
                            pos: Point::new(i as f64, 0.0),
                        }),
                )
                .unwrap();
        }
        assert_eq!(stack.undo_depth(), depth + 2);
    }

    #[test]
    fn a_new_command_clears_redo() {
        let (mut doc, mut stack, pid) = seeded();
        let (e, _) = add_point(&doc, pid, 1.0, 1.0);
        stack.apply(&mut doc, e).unwrap();
        stack.undo(&mut doc).unwrap();
        assert!(stack.can_redo());
        let (e, _) = add_point(&doc, pid, 2.0, 2.0);
        stack.apply(&mut doc, e).unwrap();
        assert!(!stack.can_redo());
    }

    #[test]
    fn a_failing_transaction_is_all_or_nothing() {
        let (mut doc, mut stack, pid) = seeded();
        let before = doc.to_json().unwrap();
        let depth = stack.undo_depth();

        let (good, _) = add_point(&doc, pid, 1.0, 1.0);
        let tx = Transaction::new("Half-broken")
            .edit(good)
            .edit(Edit::MoveVertex {
                piece: pid,
                id: nguo_pattern::Id::from_raw(9999),
                pos: Point::ORIGIN,
            });

        assert!(stack.apply(&mut doc, tx).is_err());
        assert_eq!(
            doc.to_json().unwrap(),
            before,
            "partial edits must roll back"
        );
        assert_eq!(
            stack.undo_depth(),
            depth,
            "a failed command is not recorded"
        );
    }

    #[test]
    fn the_limit_drops_the_oldest_entries() {
        let (mut doc, _, pid) = seeded();
        let mut stack = CommandStack::with_limit(4);
        for i in 0..10 {
            let (e, _) = add_point(&doc, pid, i as f64, 0.0);
            stack.apply(&mut doc, e).unwrap();
        }
        assert_eq!(stack.undo_depth(), 4);
        // What is left still undoes cleanly.
        while stack.undo(&mut doc).unwrap() {}
        assert_eq!(doc.piece(pid).unwrap().network.vertices.len(), 6);
    }

    /// The history panel's whole feature: click any row, land on that state.
    /// Headless — build history, jump, compare documents.
    #[test]
    fn jumping_lands_on_the_document_that_state_had() {
        // `snapshots[p]` is the document at position `p`. Built here rather than
        // via `seeded()` so position 0 — the empty document — has an oracle too.
        let mut doc = Document::new();
        let mut stack = CommandStack::new();
        let mut snapshots = vec![doc.to_json().unwrap()];

        let pid = doc.next_piece_id();
        stack
            .apply(
                &mut doc,
                Edit::AddPiece {
                    id: pid,
                    piece: Box::new(Piece::new("Front")),
                },
            )
            .unwrap();
        snapshots.push(doc.to_json().unwrap());
        for i in 1..=5 {
            let (e, _) = add_point(&doc, pid, i as f64, 0.0);
            stack.apply(&mut doc, e).unwrap();
            snapshots.push(doc.to_json().unwrap());
        }
        assert_eq!(stack.position(), 6);
        assert_eq!(stack.total(), 6);

        // Back, forward, to both ends, and twice to the same row — an order
        // that is deliberately not a walk, because a walk would also pass if
        // `jump_to` only ever stepped one way.
        for target in [3usize, 1, 6, 0, 6, 4, 4] {
            stack.jump_to(&mut doc, target).unwrap();
            assert_eq!(stack.position(), target, "jump to {target}");
            assert_eq!(
                doc.to_json().unwrap(),
                snapshots[target],
                "document at position {target}"
            );
        }
    }

    /// Jumping back does not throw the tail away — that is what makes the
    /// greyed rows in the panel real state and not a drawing.
    #[test]
    fn jumping_back_keeps_the_tail_until_a_new_edit_truncates_it() {
        let (mut doc, mut stack, pid) = seeded();
        for i in 1..=4 {
            let (e, _) = add_point(&doc, pid, i as f64, 0.0);
            stack.apply(&mut doc, e).unwrap();
        }
        let total = stack.total();

        stack.jump_to(&mut doc, 2).unwrap();
        assert_eq!(stack.position(), 2);
        assert_eq!(stack.total(), total, "the tail is still there, just undone");
        assert_eq!(stack.entries().count(), total);

        // A new edit at position 2 truncates everything after it (D-2, linear).
        let (e, _) = add_point(&doc, pid, 99.0, 0.0);
        stack.apply(&mut doc, e).unwrap();
        assert_eq!(stack.position(), 3);
        assert_eq!(stack.total(), 3, "the redo tail is gone, not merely hidden");
        assert!(!stack.can_redo());
    }

    /// `entries()` is what the panel iterates, so its order is load-bearing:
    /// oldest first, applied then undone. `redo` is a stack, so getting this
    /// wrong shows the tail upside down and every click jumps to the wrong row.
    #[test]
    fn entries_reads_oldest_first_across_the_undone_tail() {
        let (mut doc, mut stack, pid) = seeded();
        for name in ["A", "B", "C"] {
            let (e, _) = add_point(&doc, pid, 0.0, 0.0);
            stack
                .apply(&mut doc, Transaction::new(name).edit(e))
                .unwrap();
        }
        let all: Vec<&str> = stack.entries().map(Entry::label).collect();
        assert_eq!(all, ["Add piece", "A", "B", "C"]);

        stack.jump_to(&mut doc, 2).unwrap();
        let all: Vec<&str> = stack.entries().map(Entry::label).collect();
        assert_eq!(all, ["Add piece", "A", "B", "C"], "order must not flip");
        assert_eq!(stack.position(), 2, "B and C are the greyed ones");
    }

    /// Jumping is `undo`/`redo` in a loop, so an out-of-range click must clamp
    /// rather than panic or spin. A panel row index can outlive the history it
    /// came from by one frame.
    #[test]
    fn a_jump_past_either_end_clamps() {
        let (mut doc, mut stack, pid) = seeded();
        let (e, _) = add_point(&doc, pid, 1.0, 0.0);
        stack.apply(&mut doc, e).unwrap();
        let full = doc.to_json().unwrap();

        assert!(
            !stack.jump_to(&mut doc, usize::MAX).unwrap(),
            "already there"
        );
        assert_eq!(stack.position(), 2);
        assert_eq!(doc.to_json().unwrap(), full);

        assert!(stack.jump_to(&mut doc, 0).unwrap());
        assert!(
            !stack.jump_to(&mut doc, 0).unwrap(),
            "already at the bottom"
        );
        assert_eq!(stack.position(), 0);
    }

    /// A coalesced drag must not accumulate the positions it passed through.
    /// The entry count alone never bounded this: 20 000 moves measured
    /// 5 767 363 B in ONE entry before the collapse (`tests/memory.rs`).
    #[test]
    fn a_coalesced_drag_collapses_to_one_edit_per_target() {
        let (mut doc, mut stack, pid) = seeded();
        let (e, vid) = add_point(&doc, pid, 0.0, 0.0);
        stack.apply(&mut doc, e).unwrap();
        let anchor = doc.to_json().unwrap();

        let g = GestureId(3);
        for i in 1..=500 {
            stack
                .apply(
                    &mut doc,
                    Transaction::new("Move point")
                        .with_gesture(g)
                        .edit(Edit::MoveVertex {
                            piece: pid,
                            id: vid,
                            pos: Point::new(i as f64, 0.0),
                        }),
                )
                .unwrap();
        }
        let entry = stack.entries().last().expect("the drag");
        assert_eq!(
            entry.edits().len(),
            1,
            "500 moves of one vertex must keep one edit, not 500"
        );

        // ...and it is still the LAST position, undoing to the FIRST.
        assert_eq!(
            doc.piece(pid).unwrap().network.pos(vid).unwrap(),
            Point::new(500.0, 0.0)
        );
        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.to_json().unwrap(), anchor);
        stack.redo(&mut doc).unwrap();
        assert_eq!(
            doc.piece(pid).unwrap().network.pos(vid).unwrap(),
            Point::new(500.0, 0.0)
        );
    }

    /// Two targets in one gesture — a multi-select drag — collapse per target
    /// and keep both. Replacing only the *newest* edit would leave this one
    /// growing at one edit per move while looking fixed for a single target.
    #[test]
    fn a_gesture_over_two_targets_collapses_to_two_edits() {
        let (mut doc, mut stack, pid) = seeded();
        let (e, v1) = add_point(&doc, pid, 0.0, 0.0);
        stack.apply(&mut doc, e).unwrap();
        let (e, v2) = add_point(&doc, pid, 1.0, 0.0);
        stack.apply(&mut doc, e).unwrap();
        let anchor = doc.to_json().unwrap();

        let g = GestureId(4);
        for i in 1..=100 {
            let mut tx = Transaction::new("Move points").with_gesture(g);
            for v in [v1, v2] {
                tx.push(Edit::MoveVertex {
                    piece: pid,
                    id: v,
                    pos: Point::new(i as f64, 0.0),
                });
            }
            stack.apply(&mut doc, tx).unwrap();
        }
        let entry = stack.entries().last().expect("the drag");
        assert_eq!(entry.edits().len(), 2, "one edit per target, not 200");

        stack.undo(&mut doc).unwrap();
        assert_eq!(doc.to_json().unwrap(), anchor, "both points must go back");
    }

    /// The panel shows these strings, so they have to read like actions. A
    /// command id leaking through ("edit.undo", "move_vertex") is the failure
    /// this catches — that is a panel of opaque ids, not a history.
    #[test]
    fn every_label_is_prose_and_not_an_id() {
        for edit_label in [
            Edit::AddPiece {
                id: PieceId::from_raw(0),
                piece: Box::new(Piece::new("x")),
            }
            .label(),
            Edit::MoveVertex {
                piece: PieceId::from_raw(0),
                id: nguo_pattern::Id::from_raw(0),
                pos: Point::ORIGIN,
            }
            .label(),
        ] {
            assert!(
                !edit_label.contains('.') && !edit_label.contains('_'),
                "{edit_label:?} reads like an id"
            );
            let mut chars = edit_label.chars();
            assert!(
                chars.next().is_some_and(char::is_uppercase),
                "{edit_label:?}"
            );
        }
    }

    #[test]
    fn undo_on_an_empty_stack_is_not_an_error() {
        let mut doc = Document::new();
        let mut stack = CommandStack::new();
        assert!(!stack.undo(&mut doc).unwrap());
        assert!(!stack.redo(&mut doc).unwrap());
    }
}
