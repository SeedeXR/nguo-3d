// SPDX-License-Identifier: GPL-2.0-or-later
//! Rebuildable state that must never become document truth (guide §4.3).

use crate::doc::Document;

/// A cache of something computed *from* a [`Document`], tagged with the
/// revision it was computed at.
///
/// This type has no `Serialize` and no `Deserialize` impl, and that is the
/// point: tessellated Vello scenes, sim meshes, spatial indices, thumbnails and
/// render buffers are derived. Because they can only be held in a `Derived<T>`,
/// and a `Derived<T>` cannot be serialized, they cannot be smuggled into a
/// saved project as truth. Guide §4.3 states the rule; this makes the compiler
/// enforce it.
#[derive(Debug, Clone)]
pub struct Derived<T> {
    value: T,
    revision: u64,
}

impl<T> Derived<T> {
    pub fn new(doc: &Document, value: T) -> Derived<T> {
        Derived {
            value,
            revision: doc.revision(),
        }
    }

    /// The document revision this was built from.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn is_stale(&self, doc: &Document) -> bool {
        self.revision != doc.revision()
    }

    /// The cached value, or `None` if the document has moved on.
    pub fn get(&self, doc: &Document) -> Option<&T> {
        (!self.is_stale(doc)).then_some(&self.value)
    }

    /// The cached value, rebuilt first if the document has moved on.
    pub fn get_or_rebuild(&mut self, doc: &Document, build: impl FnOnce(&Document) -> T) -> &T {
        if self.is_stale(doc) {
            self.value = build(doc);
            self.revision = doc.revision();
        }
        &self.value
    }

    /// The cached value regardless of staleness. For "draw the old frame while
    /// the new one is computing" — say so at the call site.
    pub fn peek(&self) -> &T {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edit, Piece};

    #[test]
    fn a_cache_goes_stale_when_the_document_changes() {
        let mut doc = Document::new();
        let mut cache = Derived::new(&doc, 0usize);
        assert_eq!(cache.get(&doc), Some(&0));

        let id = doc.next_piece_id();
        doc.apply_edit(Edit::AddPiece {
            id,
            piece: Box::new(Piece::new("Front")),
        })
        .unwrap();

        assert!(cache.is_stale(&doc));
        assert_eq!(cache.get(&doc), None);
        assert_eq!(*cache.get_or_rebuild(&doc, |d| d.pieces().len()), 1);
        assert_eq!(cache.get(&doc), Some(&1));
    }
}
