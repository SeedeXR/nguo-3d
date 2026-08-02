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
    /// 256 entries. A drag is one entry, so this is a long session's worth.
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
        // Undo runs newest-first.
        inverses.reverse();

        let coalesced = tx.gesture.is_some()
            && self
                .undo
                .last()
                .is_some_and(|last| last.gesture == tx.gesture);
        if coalesced {
            let last = self.undo.last_mut().expect("checked above");
            last.redo.extend(tx.edits);
            last.undo.splice(0..0, inverses);
        } else {
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

    pub fn history(&self) -> &[Entry] {
        &self.undo
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
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

    #[test]
    fn undo_on_an_empty_stack_is_not_an_error() {
        let mut doc = Document::new();
        let mut stack = CommandStack::new();
        assert!(!stack.undo(&mut doc).unwrap());
        assert!(!stack.redo(&mut doc).unwrap());
    }
}
