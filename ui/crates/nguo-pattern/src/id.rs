// SPDX-License-Identifier: GPL-2.0-or-later
//! Stable entity ids and the arena that hands them out.
//!
//! # Why not a generational index
//! The obvious options are a generational index or a UUID. We use a
//! **monotone `u64` handle**, the UUID side of that fork with a cheaper
//! representation.
//! The reason is undo: a generational arena bumps a slot's generation on
//! removal, so `add` followed by `undo` leaves a document that *behaves*
//! identically but does not *serialize* identically. Handles are never reused,
//! so removing an entity and undoing the removal restores the byte-exact
//! document — which is exactly the invariant `nguo-doc-commands` property-tests.
//!
//! The allocation counter is derived from the contents (`max key + 1`) and is
//! never serialized, so it cannot drift and cannot make two loads of the same
//! file disagree.

use std::collections::{btree_map, BTreeMap};
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

/// A stable handle to a `T` inside an [`Arena`].
///
/// `PhantomData<fn() -> T>` keeps `Id<T>` `Copy`/`Send`/`Sync` regardless of
/// `T`, which is why every trait impl below is hand-written rather than derived
/// (a derive would add a spurious `T: Copy` bound).
#[derive(Serialize, Deserialize)]
#[serde(transparent, bound = "")]
pub struct Id<T> {
    raw: u64,
    #[serde(skip)]
    _marker: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    pub const fn from_raw(raw: u64) -> Id<T> {
        Id {
            raw,
            _marker: PhantomData,
        }
    }

    pub const fn raw(self) -> u64 {
        self.raw
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Id<T> {}
impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl<T> Eq for Id<T> {}
impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}
impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}
impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.raw)
    }
}

/// An id-keyed store with deterministic iteration order.
///
/// `BTreeMap`, not `Vec` + free list: iteration order is part of the serialized
/// document, and pattern-scale arenas (hundreds of entities) do not care about
/// the log factor.
///
/// `next` is the allocation watermark. It is **not** content: it is skipped by
/// `Serialize`, rebuilt as `max key + 1` by `Deserialize`, and ignored by
/// `PartialEq`. That is what lets "add an entity, then undo" compare and
/// serialize equal to the document before the add, while still guaranteeing an
/// id is never handed out twice within a session.
#[derive(Clone)]
pub struct Arena<T> {
    items: BTreeMap<u64, T>,
    next: u64,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena {
            items: BTreeMap::new(),
            next: 0,
        }
    }
}

impl<T: Serialize> Serialize for Arena<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.items.serialize(s)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Arena<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let items = BTreeMap::<u64, T>::deserialize(d)?;
        let next = items.keys().next_back().map_or(0, |k| k + 1);
        Ok(Arena { items, next })
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.items.iter().map(|(k, v)| (Id::<T>::from_raw(*k), v)))
            .finish()
    }
}

impl<T: PartialEq> PartialEq for Arena<T> {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl<T> Arena<T> {
    pub fn new() -> Arena<T> {
        Arena::default()
    }

    /// The id the next [`Arena::insert`] will use. Pure, so a command can plan
    /// an edit before applying it.
    pub fn next_id(&self) -> Id<T> {
        Id::from_raw(self.next)
    }

    pub fn insert(&mut self, value: T) -> Id<T> {
        let id = Id::from_raw(self.next);
        self.next += 1;
        self.items.insert(id.raw, value);
        id
    }

    /// Put `value` back at an id that was handed out earlier. This is how undo
    /// restores a removed entity without changing its identity.
    ///
    /// Returns the previous occupant, if any.
    pub fn insert_at(&mut self, id: Id<T>, value: T) -> Option<T> {
        self.next = self.next.max(id.raw + 1);
        self.items.insert(id.raw, value)
    }

    pub fn remove(&mut self, id: Id<T>) -> Option<T> {
        self.items.remove(&id.raw)
    }

    pub fn get(&self, id: Id<T>) -> Option<&T> {
        self.items.get(&id.raw)
    }

    pub fn get_mut(&mut self, id: Id<T>) -> Option<&mut T> {
        self.items.get_mut(&id.raw)
    }

    pub fn contains(&self, id: Id<T>) -> bool {
        self.items.contains_key(&id.raw)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Id<T>, &T)> + '_ {
        self.items.iter().map(|(k, v)| (Id::from_raw(*k), v))
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (Id<T>, &mut T)> + '_ {
        self.items.iter_mut().map(|(k, v)| (Id::from_raw(*k), v))
    }

    pub fn ids(&self) -> impl DoubleEndedIterator<Item = Id<T>> + '_ {
        self.items.keys().map(|k| Id::from_raw(*k))
    }

    pub fn values(&self) -> btree_map::Values<'_, u64, T> {
        self.items.values()
    }
}

impl<T> FromIterator<T> for Arena<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut a = Arena::new();
        for v in iter {
            a.insert(v);
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Thing(u32);

    #[test]
    fn ids_are_never_reused() {
        let mut a: Arena<Thing> = Arena::new();
        let x = a.insert(Thing(1));
        let y = a.insert(Thing(2));
        a.remove(y);
        let z = a.insert(Thing(3));
        assert_ne!(y, z, "a removed id must not be handed out again");
        assert_ne!(x, z);
    }

    #[test]
    fn next_id_matches_what_insert_uses() {
        let mut a: Arena<Thing> = Arena::new();
        let planned = a.next_id();
        assert_eq!(a.insert(Thing(7)), planned);
        let planned = a.next_id();
        assert_eq!(a.insert(Thing(8)), planned);
    }

    #[test]
    fn remove_then_insert_at_is_the_identity() {
        let mut a: Arena<Thing> = Arena::new();
        a.insert(Thing(1));
        let mid = a.insert(Thing(2));
        a.insert(Thing(3));
        let before = a.clone();
        let v = a.remove(mid).unwrap();
        a.insert_at(mid, v);
        assert_eq!(a, before);
    }

    #[test]
    fn iteration_is_ordered_by_id() {
        let mut a: Arena<Thing> = Arena::new();
        for i in 0..8 {
            a.insert(Thing(i));
        }
        a.remove(Id::from_raw(3));
        let seen: Vec<u64> = a.ids().map(|i| i.raw()).collect();
        assert_eq!(seen, vec![0, 1, 2, 4, 5, 6, 7]);
    }

    #[test]
    fn serde_round_trip_keeps_ids() {
        let mut a: Arena<u32> = Arena::new();
        a.insert(10);
        let keep = a.insert(20);
        a.insert(30);
        a.remove(Id::from_raw(0));
        let json = serde_json::to_string(&a).unwrap();
        let back: Arena<u32> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get(keep), Some(&20));
        assert_eq!(back.next_id(), a.next_id());
    }
}
