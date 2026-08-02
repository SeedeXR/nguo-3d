// SPDX-License-Identifier: GPL-2.0-or-later
//! Units, length parse/format, and the expression evaluator behind every
//! numeric field in the app.
//!
//! # Canonical unit
//! **Millimetres, always.** [`Length`] stores mm; [`Unit`] only exists to
//! convert at the UI edge. Nothing downstream of a numeric field ever sees
//! centimetres or inches.
//!
//! # Expressions
//! A numeric field accepts `waist/4 + 1cm`. Bare literals are read in the
//! field's *default unit*; a suffixed literal (`1cm`) converts to mm
//! regardless. Identifiers resolve through a [`Resolve`] (usually a
//! [`ParamTable`]), which detects reference cycles rather than blowing the
//! stack.
//!
//! This crate is UI-framework-free (guide §1) and has no floating dependency
//! beyond `serde` + `thiserror`.

#![forbid(unsafe_code)]

mod expr;
mod params;
mod unit;

pub use expr::{eval, EvalError, NoParams, Resolve};
pub use params::{Param, ParamTable};
pub use unit::{Length, ParseLengthError, Unit};
