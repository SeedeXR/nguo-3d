// SPDX-License-Identifier: GPL-2.0-or-later
//! Commands, transactions and undo/redo over [`nguo_doc::Document`].
//!
//! # The rule
//! Nothing outside a command may mutate the document, and that is enforced by
//! the type system rather than by convention: `Document`'s fields are
//! `pub(crate)` to `nguo-doc`, and its only `&mut self` method takes an
//! [`Edit`] — a closed enum of primitive mutations. This crate composes those
//! primitives; it has no other way in either.
//!
//! # Undo
//! [`Document::apply_edit`] returns the exact inverse of what it just did, so
//! undo is not a parallel implementation that can drift. A [`Transaction`] is a
//! list of edits that becomes **one** undo entry — "add dart" is create + split
//! + connect, and the user presses undo once.
//!
//! # Coalescing
//! A drag emits a command per pointer move. Tag them all with the same
//! [`GestureId`] and the stack merges them into the entry it is already
//! holding, so the whole drag undoes in one step (§5.2).
//!
//! # History model
//! A linear stack (§5.3/§26.2 fork, resolved). Commands stay operation-based
//! and ids stay stable, so a node-graph history remains possible later without
//! rewriting the document.

#![forbid(unsafe_code)]

mod ops;
mod stack;

pub use ops::{add_dart, split_edge};
pub use stack::{CommandStack, Entry, GestureId, Transaction};

pub use nguo_doc::{DocError, Document, Edit};
