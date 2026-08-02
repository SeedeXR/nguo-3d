// SPDX-License-Identifier: GPL-2.0-or-later
//! The parameter table: named expressions that reference each other.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::expr::{eval, EvalError, Resolve};
use crate::unit::Unit;

/// One row of the parameter table.
///
/// `expr` is the *source text* the user typed, kept verbatim so the table is
/// re-editable and round-trips through the project file. `unit` is the unit
/// bare literals in that expression are read in; declare dimensionless
/// parameters (ease ratios, grade factors) as [`Unit::Millimeter`] so the
/// implicit conversion is the identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub expr: String,
    #[serde(default)]
    pub unit: Unit,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

impl Param {
    pub fn new(expr: impl Into<String>) -> Param {
        Param {
            expr: expr.into(),
            unit: Unit::Millimeter,
            description: String::new(),
        }
    }

    pub fn with_unit(mut self, unit: Unit) -> Param {
        self.unit = unit;
        self
    }

    pub fn with_description(mut self, d: impl Into<String>) -> Param {
        self.description = d.into();
        self
    }
}

/// A name -> expression table with cycle-safe resolution.
///
/// `BTreeMap`, not `HashMap`: iteration order is part of the serialized
/// document and determinism is a feature.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParamTable {
    entries: BTreeMap<String, Param>,
}

impl ParamTable {
    pub fn new() -> ParamTable {
        ParamTable::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, param: Param) -> Option<Param> {
        self.entries.insert(name.into(), param)
    }

    /// Shorthand for `insert(name, Param::new(expr))`.
    pub fn set(&mut self, name: impl Into<String>, expr: impl Into<String>) -> Option<Param> {
        self.insert(name, Param::new(expr))
    }

    pub fn remove(&mut self, name: &str) -> Option<Param> {
        self.entries.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&Param> {
        self.entries.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Param)> {
        self.entries.iter()
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// The value of one parameter, in millimetres.
    ///
    /// Returns [`EvalError::Circular`] with the offending chain rather than
    /// recursing forever.
    pub fn value(&self, name: &str) -> Result<f64, EvalError> {
        let mut r = TableResolver {
            table: self,
            extra: &|_| None,
            stack: Vec::new(),
            on_stack: BTreeSet::new(),
        };
        r.value(name)
    }

    /// Evaluate an arbitrary expression against this table, in millimetres.
    pub fn eval(&self, src: &str, default_unit: Unit) -> Result<f64, EvalError> {
        self.eval_with(src, default_unit, &|_| None)
    }

    /// Like [`ParamTable::eval`], but `extra` supplies values (in millimetres)
    /// for names the table does not define.
    ///
    /// This is how a document layers avatar measurements underneath the
    /// parameter table without copying them into it — one source of truth, and
    /// still cycle-safe, because the cycle check lives in the walk rather than
    /// in the table.
    pub fn eval_with(
        &self,
        src: &str,
        default_unit: Unit,
        extra: &dyn Fn(&str) -> Option<f64>,
    ) -> Result<f64, EvalError> {
        let mut r = TableResolver {
            table: self,
            extra,
            stack: Vec::new(),
            on_stack: BTreeSet::new(),
        };
        eval(src, default_unit, &mut r)
    }

    /// Resolve every parameter. Deterministic order; failures reported per row
    /// so the UI can flag exactly the broken cells (guide §9.3 diagnostics).
    pub fn resolve_all(&self) -> BTreeMap<String, Result<f64, EvalError>> {
        self.entries
            .keys()
            .map(|k| (k.clone(), self.value(k)))
            .collect()
    }
}

/// Depth-first resolver carrying the visiting stack, which is what turns
/// `a = b + 1; b = a` into a typed error instead of a stack overflow.
struct TableResolver<'a> {
    table: &'a ParamTable,
    extra: &'a dyn Fn(&str) -> Option<f64>,
    stack: Vec<String>,
    on_stack: BTreeSet<String>,
}

impl Resolve for TableResolver<'_> {
    fn value(&mut self, name: &str) -> Result<f64, EvalError> {
        if self.on_stack.contains(name) {
            let mut chain: Vec<String> = self
                .stack
                .iter()
                .skip_while(|n| n.as_str() != name)
                .cloned()
                .collect();
            chain.push(name.to_string());
            return Err(EvalError::Circular(chain));
        }
        // Copy the shared reference out so the recursive `eval` below may hold
        // `&mut self` while we still read the (immutable) table.
        let table = self.table;
        let Some(param) = table.entries.get(name) else {
            // Not a parameter: fall through to whatever the caller layered
            // underneath (avatar measurements, in a document).
            return (self.extra)(name).ok_or_else(|| EvalError::UnknownParameter(name.to_string()));
        };
        let (src, unit) = (param.expr.clone(), param.unit);

        self.stack.push(name.to_string());
        self.on_stack.insert(name.to_string());
        let out = eval(&src, unit, self);
        self.on_stack.remove(name);
        self.stack.pop();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(rows: &[(&str, &str)]) -> ParamTable {
        let mut t = ParamTable::new();
        for (k, v) in rows {
            t.set(*k, *v);
        }
        t
    }

    #[test]
    fn resolves_chains() {
        let t = table(&[
            ("waist", "80cm"),
            ("quarter", "waist/4"),
            ("front", "quarter + 1cm"),
        ]);
        assert_eq!(t.value("waist").unwrap(), 800.0);
        assert_eq!(t.value("quarter").unwrap(), 200.0);
        assert_eq!(t.value("front").unwrap(), 210.0);
    }

    #[test]
    fn direct_cycle_is_a_typed_error() {
        let t = table(&[("a", "a + 1")]);
        assert_eq!(
            t.value("a"),
            Err(EvalError::Circular(vec!["a".into(), "a".into()]))
        );
    }

    #[test]
    fn indirect_cycle_is_a_typed_error() {
        let t = table(&[("a", "b + 1"), ("b", "a")]);
        let err = t.value("a").unwrap_err();
        assert_eq!(
            err,
            EvalError::Circular(vec!["a".into(), "b".into(), "a".into()])
        );
        // And from the other end.
        assert!(matches!(t.value("b"), Err(EvalError::Circular(_))));
    }

    #[test]
    fn a_diamond_is_not_a_cycle() {
        let t = table(&[("a", "10"), ("b", "a * 2"), ("c", "a * 3"), ("d", "b + c")]);
        assert_eq!(t.value("d").unwrap(), 50.0);
    }

    #[test]
    fn resolve_all_reports_per_row() {
        let t = table(&[("ok", "1cm"), ("bad", "nope"), ("loop", "loop")]);
        let r = t.resolve_all();
        assert_eq!(r["ok"], Ok(10.0));
        assert!(matches!(r["bad"], Err(EvalError::UnknownParameter(_))));
        assert!(matches!(r["loop"], Err(EvalError::Circular(_))));
    }

    #[test]
    fn unit_of_a_row_scales_its_bare_literals() {
        let mut t = ParamTable::new();
        t.insert("hem", Param::new("3").with_unit(Unit::Centimeter));
        assert_eq!(t.value("hem").unwrap(), 30.0);
    }

    #[test]
    fn eval_against_the_table() {
        let t = table(&[("waist", "80cm")]);
        assert_eq!(t.eval("waist/4 + 1cm", Unit::Millimeter).unwrap(), 210.0);
    }
}
