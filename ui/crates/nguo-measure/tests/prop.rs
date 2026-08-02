// SPDX-License-Identifier: GPL-2.0-or-later
//! Property tests for units and the expression evaluator (guide §18).

use nguo_measure::{eval, EvalError, NoParams, Param, ParamTable, Unit};
use proptest::prelude::*;

fn any_unit() -> impl Strategy<Value = Unit> {
    prop_oneof![
        Just(Unit::Millimeter),
        Just(Unit::Centimeter),
        Just(Unit::Inch)
    ]
}

proptest! {
    /// Format at full precision, parse it back, get the same length.
    #[test]
    fn format_parse_round_trip(mm in -1.0e6f64..1.0e6, unit in any_unit()) {
        let l = nguo_measure::Length::from_mm(mm);
        let text = l.format(unit, 12);
        let back = nguo_measure::Length::parse(&text, unit).unwrap();
        prop_assert!(
            (back.mm() - mm).abs() <= 1e-6 + mm.abs() * 1e-12,
            "{text} -> {} != {mm}", back.mm()
        );
    }

    /// The formatted text is also a valid *expression* and evaluates the same.
    #[test]
    fn formatted_text_evaluates(mm in -1.0e6f64..1.0e6, unit in any_unit()) {
        let l = nguo_measure::Length::from_mm(mm);
        let text = l.format(unit, 12);
        let v = eval(&text, Unit::Millimeter, &mut NoParams).unwrap();
        prop_assert!((v - mm).abs() <= 1e-6 + mm.abs() * 1e-12);
    }

    /// Unit conversion is a bijection at the resolution we care about.
    #[test]
    fn unit_round_trip(v in -1.0e5f64..1.0e5, unit in any_unit()) {
        let l = nguo_measure::Length::from_unit(v, unit);
        prop_assert!((l.to_unit(unit) - v).abs() <= v.abs() * 1e-15 + 1e-12);
    }
}

/// A small arithmetic expression that we can also evaluate directly in `f64`,
/// so "the parser is exact" is checkable rather than asserted.
#[derive(Debug, Clone)]
enum Ast {
    Num(f64),
    Bin(char, Box<Ast>, Box<Ast>),
}

impl Ast {
    fn render(&self) -> String {
        match self {
            Ast::Num(n) => format!("{n:?}"),
            Ast::Bin(op, a, b) => format!("({} {} {})", a.render(), op, b.render()),
        }
    }

    fn value(&self) -> f64 {
        match self {
            Ast::Num(n) => *n,
            Ast::Bin(op, a, b) => {
                let (x, y) = (a.value(), b.value());
                match op {
                    '+' => x + y,
                    '-' => x - y,
                    '*' => x * y,
                    _ => x / y,
                }
            }
        }
    }
}

fn ast() -> impl Strategy<Value = Ast> {
    let leaf = (-1.0e4f64..1.0e4).prop_map(Ast::Num);
    leaf.prop_recursive(6, 48, 2, |inner| {
        (
            prop_oneof![Just('+'), Just('-'), Just('*'), Just('/')],
            inner.clone(),
            inner,
        )
            .prop_map(|(op, a, b)| Ast::Bin(op, Box::new(a), Box::new(b)))
    })
}

proptest! {
    /// Evaluation is bit-exact for anything f64 can represent. Millimetres is
    /// the default unit, so the dimensionless promotion is the identity and any
    /// difference would be the parser's fault.
    #[test]
    fn evaluation_is_exact(a in ast()) {
        let src = a.render();
        let expected = a.value();
        match eval(&src, Unit::Millimeter, &mut NoParams) {
            Ok(got) => {
                prop_assert!(expected.is_finite(), "{src}: got {got} for non-finite expected");
                prop_assert_eq!(got, expected, "{}", src);
            }
            Err(EvalError::DivisionByZero) | Err(EvalError::NotFinite) => {
                // Only acceptable when the direct computation also degenerates.
                prop_assert!(!expected.is_finite() || expected == 0.0, "{src}", );
            }
            Err(e) => prop_assert!(false, "{src} failed to parse: {e}"),
        }
    }
}

fn small_table() -> impl Strategy<Value = Vec<(String, String)>> {
    let names = ["a", "b", "c", "d"];
    let atom = prop_oneof![
        (0usize..4).prop_map(move |i| names[i].to_string()),
        (0.0f64..100.0).prop_map(|n| format!("{n:?}")),
        Just("1cm".to_string()),
    ];
    let expr = (
        atom.clone(),
        prop_oneof![Just('+'), Just('*'), Just('/')],
        atom,
    )
        .prop_map(|(l, op, r)| format!("{l} {op} {r}"));
    proptest::collection::vec(expr, 1..5).prop_map(move |exprs| {
        exprs
            .into_iter()
            .enumerate()
            .map(|(i, e)| (names[i % 4].to_string(), e))
            .collect()
    })
}

proptest! {
    /// No parameter table, however tangled, may hang or overflow the stack.
    #[test]
    fn param_resolution_always_terminates(rows in small_table()) {
        let mut t = ParamTable::new();
        for (k, v) in rows {
            t.insert(k, Param::new(v));
        }
        for (_, r) in t.resolve_all() {
            if let Ok(v) = r {
                prop_assert!(v.is_finite());
            }
        }
    }
}
