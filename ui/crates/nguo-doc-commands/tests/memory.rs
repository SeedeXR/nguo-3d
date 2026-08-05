// SPDX-License-Identifier: GPL-2.0-or-later
//! What the history costs, measured — not estimated.
//!
//! A counting global allocator makes "how many bytes is the stack holding"
//! answerable exactly: build history, read live heap, [`CommandStack::clear`],
//! read again. The difference is what the stack was retaining and nothing else,
//! because `clear` does not touch the document.
//!
//! # Why this is one test function
//! The counter is process-wide, and libtest runs `#[test]`s on parallel threads.
//! A second test in this binary would allocate inside the first one's
//! measurement window and both numbers would be junk. One function, in order.
//!
//! The numbers it prints are quoted in [`CommandStack::DEFAULT_LIMIT`]'s docs.
//! Re-run with `--nocapture` before changing them there.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use kurbo::Point;
use nguo_doc::{Document, Edit, Piece, PieceId};
use nguo_doc_commands::{CommandStack, GestureId, Transaction};

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
            FREED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Bytes currently on the heap.
fn live() -> usize {
    ALLOCATED.load(Ordering::Relaxed) - FREED.load(Ordering::Relaxed)
}

/// Heap the stack is holding: exactly what `clear()` hands back.
fn held_by(stack: &mut CommandStack) -> usize {
    let before = live();
    stack.clear();
    before - live()
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn seeded() -> (Document, PieceId) {
    let mut doc = Document::new();
    let id = doc.next_piece_id();
    doc.apply_edit(Edit::AddPiece {
        id,
        piece: Box::new(Piece::new("Front")),
    })
    .unwrap();
    (doc, id)
}

fn add_point(doc: &Document, piece: PieceId, x: f64, y: f64) -> Edit {
    Edit::AddVertex {
        piece,
        id: doc.piece(piece).unwrap().network.vertices.next_id(),
        pos: Point::new(x, y),
    }
}

/// A piece with `n` vertices chained by `n - 1` line edges. Stands in for a
/// traced outline; what matters is that the arenas are full, because deleting it
/// puts the whole thing into one undo entry.
fn chained_piece(n: usize) -> Piece {
    let mut piece = Piece::new("Front");
    let mut prev = piece.network.add_vertex(Point::ORIGIN);
    for i in 1..n {
        let v = piece.network.add_vertex(Point::new(i as f64, 0.0));
        piece
            .network
            .add_edge(
                prev,
                v,
                nguo_pattern::EdgeGeom::Line,
                nguo_pattern::EdgeSemantic::Seam,
            )
            .unwrap();
        prev = v;
    }
    piece
}

// ---------------------------------------------------------------------------
// The cap
// ---------------------------------------------------------------------------

/// The history is capped at [`CommandStack::DEFAULT_LIMIT`] = **256 entries**,
/// and that is only a real memory bound because no single entry can grow
/// without limit.
///
/// The second half is the one that was broken. Coalescing means one entry
/// absorbs a whole drag, so before the collapse in `CommandStack::apply` a
/// 20 000-move drag held **5 767 363 B in one entry** and had no ceiling at all
/// — the entry count bounded nothing. It now holds three figures.
#[test]
fn the_history_is_bounded_in_entries_and_in_bytes_per_entry() {
    assert_eq!(CommandStack::DEFAULT_LIMIT, 256);
    println!("size_of::<Edit>() = {} B", std::mem::size_of::<Edit>());

    // --- 1. an ordinary edit, and a full history of them ------------------
    let (mut doc, pid) = seeded();
    let mut stack = CommandStack::new();
    for i in 0..4 * CommandStack::DEFAULT_LIMIT {
        let e = add_point(&doc, pid, i as f64, 0.0);
        stack.apply(&mut doc, e).unwrap();
    }
    assert_eq!(
        stack.total(),
        CommandStack::DEFAULT_LIMIT,
        "1024 edits must be capped at the entry limit"
    );
    let ordinary = held_by(&mut stack);
    let per_entry = ordinary / CommandStack::DEFAULT_LIMIT;
    println!("add one point:              {per_entry} B/entry");
    println!("a full 256-entry history:   {ordinary} B");
    assert!(
        ordinary <= 64 * 1024,
        "a full history of point edits is {ordinary} B, was 47 360"
    );

    // --- 2. one long drag: the case an entry count cannot bound -----------
    let (mut doc, pid) = seeded();
    let mut stack = CommandStack::new();
    let e = add_point(&doc, pid, 0.0, 0.0);
    stack.apply(&mut doc, e).unwrap();
    let vid = doc
        .piece(pid)
        .unwrap()
        .network
        .vertices
        .iter()
        .next()
        .unwrap()
        .0;
    for i in 1..=20_000u32 {
        stack
            .apply(
                &mut doc,
                Transaction::new("Move point")
                    .with_gesture(GestureId(1))
                    .edit(Edit::MoveVertex {
                        piece: pid,
                        id: vid,
                        pos: Point::new(i as f64, 0.0),
                    }),
            )
            .unwrap();
    }
    assert_eq!(stack.total(), 2, "20 000 moves are one drag, one entry");
    let drag = held_by(&mut stack);
    println!("a 20 000-move drag:         {drag} B  (5 767 363 B before the collapse)");
    assert!(
        drag <= 4 * 1024,
        "the drag holds {drag} B — the coalescing collapse is gone and the \
         history is unbounded again"
    );

    // --- 3. the expensive entry: a delete carries the piece back ----------
    let mut doc = Document::new();
    let pid = doc.next_piece_id();
    doc.apply_edit(Edit::AddPiece {
        id: pid,
        piece: Box::new(chained_piece(200)),
    })
    .unwrap();
    let mut stack = CommandStack::new();
    stack
        .apply(&mut doc, Edit::RemovePiece { id: pid })
        .unwrap();
    let delete = held_by(&mut stack);
    println!("delete a 200-vertex piece:  {delete} B");
    // This is the residual ceiling: 256 x the largest piece. 37 KB x 256 is
    // under 10 MB, which is why the cap is still an entry count and not a byte
    // budget (see `DEFAULT_LIMIT`). If this ever measures megabytes, revisit.
    assert!(
        delete <= 256 * 1024,
        "one piece-delete entry holds {delete} B; 256 of those would be \
         {} MB and the entry count would stop being a memory bound",
        delete * 256 / 1_000_000
    );
}
