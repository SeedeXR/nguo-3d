// SPDX-License-Identifier: GPL-2.0-or-later
//! The numeric-field expression evaluator.
//!
//! Grammar (recursive descent, left-associative):
//!
//! ```text
//! expr   := term (('+' | '-') term)*
//! term   := unary (('*' | '/') unary)*
//! unary  := ('-' | '+')* atom
//! atom   := NUMBER UNIT? | IDENT | '(' expr ')'
//! ```
//!
//! # Dimensions
//! Values carry a tiny degree tag: `0` = plain number, `1` = a length in mm.
//! A suffixed literal (`1cm`) and every parameter reference are degree 1; a
//! bare literal is degree 0. `*` adds degrees, `/` subtracts, `+`/`-` promote a
//! degree-0 operand into the default unit. The final result must be degree 0 or
//! 1; degree 0 is promoted by the default unit. That is what makes
//! `waist/4 + 1cm` mean "a quarter of the waist plus ten millimetres" rather
//! than "a quarter of the waist divided by four millimetres".
//!
//! Dimensionless parameters (ease ratios, grade factors) should be declared in
//! [`Unit::Millimeter`] so the final promotion multiplies by 1.0.

use crate::unit::Unit;

/// Anything an expression can go wrong with. No panics, no stack overflows —
/// user input is a trust boundary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    #[error("empty expression")]
    Empty,
    #[error("unexpected character `{ch}` at byte {at}")]
    UnexpectedChar { ch: char, at: usize },
    #[error("`{text}` at byte {at} is not a number")]
    BadNumber { text: String, at: usize },
    #[error("unexpected end of expression")]
    UnexpectedEnd,
    #[error("unexpected `{found}` at byte {at}")]
    UnexpectedToken { found: String, at: usize },
    #[error("trailing input at byte {at}")]
    TrailingInput { at: usize },
    #[error("unknown parameter `{0}`")]
    UnknownParameter(String),
    #[error("division by zero")]
    DivisionByZero,
    #[error("result is not a length (degree {degree})")]
    DimensionMismatch { degree: i32 },
    #[error("circular parameter reference: {}", .0.join(" -> "))]
    Circular(Vec<String>),
    #[error("expression nested deeper than {0} levels")]
    DepthLimit(u32),
    #[error("expression evaluated to a non-finite number")]
    NotFinite,
}

/// Resolves an identifier to its value **in millimetres**.
pub trait Resolve {
    fn value(&mut self, name: &str) -> Result<f64, EvalError>;
}

/// A resolver for expressions that may not reference anything.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoParams;

impl Resolve for NoParams {
    fn value(&mut self, name: &str) -> Result<f64, EvalError> {
        Err(EvalError::UnknownParameter(name.to_string()))
    }
}

impl<F: FnMut(&str) -> Option<f64>> Resolve for F {
    fn value(&mut self, name: &str) -> Result<f64, EvalError> {
        self(name).ok_or_else(|| EvalError::UnknownParameter(name.to_string()))
    }
}

/// Evaluate `src` and return **millimetres**.
///
/// `default_unit` is the unit of the field being edited: it scales bare
/// literals and the final result when the expression is dimensionless.
pub fn eval(src: &str, default_unit: Unit, resolve: &mut dyn Resolve) -> Result<f64, EvalError> {
    let tokens = lex(src)?;
    if tokens.is_empty() {
        return Err(EvalError::Empty);
    }
    let mut p = Parser {
        tokens: &tokens,
        pos: 0,
        default_unit,
        resolve,
        depth: 0,
    };
    let v = p.expr()?;
    if p.pos != p.tokens.len() {
        return Err(EvalError::TrailingInput {
            at: p.tokens[p.pos].at,
        });
    }
    let mm = match v.degree {
        1 => v.value,
        0 => v.value * default_unit.mm_per(),
        d => return Err(EvalError::DimensionMismatch { degree: d }),
    };
    if !mm.is_finite() {
        return Err(EvalError::NotFinite);
    }
    Ok(mm)
}

// --- lexer ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

#[derive(Debug, Clone)]
struct Spanned {
    tok: Tok,
    at: usize,
}

impl Spanned {
    fn describe(&self) -> String {
        match &self.tok {
            Tok::Num(n) => n.to_string(),
            Tok::Ident(s) => s.clone(),
            Tok::Plus => "+".into(),
            Tok::Minus => "-".into(),
            Tok::Star => "*".into(),
            Tok::Slash => "/".into(),
            Tok::LParen => "(".into(),
            Tok::RParen => ")".into(),
        }
    }
}

fn lex(src: &str) -> Result<Vec<Spanned>, EvalError> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        let at = i;
        match c {
            c if c.is_ascii_whitespace() => i += 1,
            '+' => {
                out.push(Spanned { tok: Tok::Plus, at });
                i += 1;
            }
            '-' => {
                out.push(Spanned {
                    tok: Tok::Minus,
                    at,
                });
                i += 1;
            }
            '*' => {
                out.push(Spanned { tok: Tok::Star, at });
                i += 1;
            }
            '/' => {
                out.push(Spanned {
                    tok: Tok::Slash,
                    at,
                });
                i += 1;
            }
            '(' => {
                out.push(Spanned {
                    tok: Tok::LParen,
                    at,
                });
                i += 1;
            }
            ')' => {
                out.push(Spanned {
                    tok: Tok::RParen,
                    at,
                });
                i += 1;
            }
            '"' => {
                // Inch suffix. Lexed as an identifier so the parser's unit
                // lookup handles it uniformly.
                out.push(Spanned {
                    tok: Tok::Ident("\"".into()),
                    at,
                });
                i += 1;
            }
            '0'..='9' | '.' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                    i += 1;
                }
                // Exponent, but only when it really is one (`1e-3`), so that
                // `2e` still lexes as `2` followed by the identifier `e`.
                if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                    let mut j = i + 1;
                    if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                        j += 1;
                    }
                    if j < b.len() && b[j].is_ascii_digit() {
                        while j < b.len() && b[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                }
                let text = &src[start..i];
                let n: f64 = text.parse().map_err(|_| EvalError::BadNumber {
                    text: text.to_string(),
                    at: start,
                })?;
                out.push(Spanned {
                    tok: Tok::Num(n),
                    at: start,
                });
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() {
                    let ch = b[i] as char;
                    if ch.is_alphanumeric() || ch == '_' || ch == '.' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                out.push(Spanned {
                    tok: Tok::Ident(src[start..i].to_string()),
                    at: start,
                });
            }
            _ => return Err(EvalError::UnexpectedChar { ch: c, at }),
        }
    }
    Ok(out)
}

// --- parser / evaluator ---------------------------------------------------

/// A value plus its dimension degree (0 = number, 1 = length in mm).
#[derive(Debug, Clone, Copy)]
struct Val {
    value: f64,
    degree: i32,
}

/// Guard against a pathological `((((((…` from a text field.
const MAX_DEPTH: u32 = 64;

struct Parser<'a> {
    tokens: &'a [Spanned],
    pos: usize,
    default_unit: Unit,
    resolve: &'a mut dyn Resolve,
    depth: u32,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos).map(|s| &s.tok)
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    /// Bring two operands to a common degree for `+`/`-` by reading the
    /// dimensionless one in the field's default unit.
    fn unify(&self, a: Val, b: Val) -> (f64, f64, i32) {
        match (a.degree, b.degree) {
            (x, y) if x == y => (a.value, b.value, x),
            (0, y) => (a.value * self.default_unit.mm_per(), b.value, y),
            (x, 0) => (a.value, b.value * self.default_unit.mm_per(), x),
            (x, _) => (a.value, b.value, x),
        }
    }

    fn expr(&mut self) -> Result<Val, EvalError> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => 1.0,
                Some(Tok::Minus) => -1.0,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.term()?;
            let (a, b, degree) = self.unify(lhs, rhs);
            lhs = Val {
                value: a + op * b,
                degree,
            };
        }
    }

    fn term(&mut self) -> Result<Val, EvalError> {
        let mut lhs = self.unary()?;
        loop {
            let mul = match self.peek() {
                Some(Tok::Star) => true,
                Some(Tok::Slash) => false,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.unary()?;
            lhs = if mul {
                Val {
                    value: lhs.value * rhs.value,
                    degree: lhs.degree + rhs.degree,
                }
            } else {
                if rhs.value == 0.0 {
                    return Err(EvalError::DivisionByZero);
                }
                Val {
                    value: lhs.value / rhs.value,
                    degree: lhs.degree - rhs.degree,
                }
            };
        }
    }

    fn unary(&mut self) -> Result<Val, EvalError> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.bump();
                let v = self.unary()?;
                Ok(Val {
                    value: -v.value,
                    degree: v.degree,
                })
            }
            Some(Tok::Plus) => {
                self.bump();
                self.unary()
            }
            _ => self.atom(),
        }
    }

    fn atom(&mut self) -> Result<Val, EvalError> {
        let Some(next) = self.tokens.get(self.pos) else {
            return Err(EvalError::UnexpectedEnd);
        };
        match next.tok.clone() {
            Tok::Num(n) => {
                self.bump();
                // An identifier glued onto a number is a unit suffix.
                if let Some(Tok::Ident(word)) = self.peek() {
                    if let Some(u) = Unit::from_suffix(word) {
                        self.bump();
                        return Ok(Val {
                            value: n * u.mm_per(),
                            degree: 1,
                        });
                    }
                }
                Ok(Val {
                    value: n,
                    degree: 0,
                })
            }
            Tok::Ident(name) => {
                self.bump();
                if Unit::from_suffix(&name).is_some() {
                    // A bare `cm` with nothing in front of it.
                    return Err(EvalError::UnexpectedToken {
                        found: name,
                        at: next.at,
                    });
                }
                let v = self.resolve.value(&name)?;
                Ok(Val {
                    value: v,
                    degree: 1,
                })
            }
            Tok::LParen => {
                self.bump();
                self.depth += 1;
                if self.depth > MAX_DEPTH {
                    return Err(EvalError::DepthLimit(MAX_DEPTH));
                }
                let v = self.expr()?;
                self.depth -= 1;
                match self.peek() {
                    Some(Tok::RParen) => {
                        self.bump();
                        Ok(v)
                    }
                    Some(_) => Err(EvalError::UnexpectedToken {
                        found: self.tokens[self.pos].describe(),
                        at: self.tokens[self.pos].at,
                    }),
                    None => Err(EvalError::UnexpectedEnd),
                }
            }
            _ => Err(EvalError::UnexpectedToken {
                found: next.describe(),
                at: next.at,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(src: &str) -> Result<f64, EvalError> {
        eval(src, Unit::Millimeter, &mut NoParams)
    }

    #[test]
    fn arithmetic_is_exact() {
        assert_eq!(e("1 + 2 * 3").unwrap(), 7.0);
        assert_eq!(e("(1 + 2) * 3").unwrap(), 9.0);
        assert_eq!(e("-4").unwrap(), -4.0);
        assert_eq!(e("10 / 4").unwrap(), 2.5);
        assert_eq!(e("2 - 3 - 4").unwrap(), -5.0);
    }

    #[test]
    fn unit_suffixes_convert_to_mm() {
        assert_eq!(e("1cm").unwrap(), 10.0);
        assert_eq!(e("1in").unwrap(), 25.4);
        assert_eq!(e("1\"").unwrap(), 25.4);
        assert_eq!(e("1cm + 5mm").unwrap(), 15.0);
    }

    #[test]
    fn bare_numbers_use_the_field_unit() {
        assert_eq!(eval("7", Unit::Centimeter, &mut NoParams).unwrap(), 70.0);
        // ... but not when they are only a divisor.
        let mut r = |n: &str| (n == "waist").then_some(800.0);
        assert_eq!(eval("waist/4", Unit::Centimeter, &mut r).unwrap(), 200.0);
    }

    #[test]
    fn the_readme_expression() {
        let mut r = |n: &str| (n == "waist").then_some(800.0);
        assert_eq!(
            eval("waist/4 + 1cm", Unit::Millimeter, &mut r).unwrap(),
            210.0
        );
    }

    #[test]
    fn errors_are_typed_not_panics() {
        assert!(matches!(e(""), Err(EvalError::Empty)));
        assert!(matches!(e("1 +"), Err(EvalError::UnexpectedEnd)));
        assert!(matches!(e("1 2"), Err(EvalError::TrailingInput { .. })));
        assert!(matches!(e("1/0"), Err(EvalError::DivisionByZero)));
        assert!(matches!(e("#"), Err(EvalError::UnexpectedChar { .. })));
        assert!(matches!(e("cm"), Err(EvalError::UnexpectedToken { .. })));
        assert!(matches!(e("waist"), Err(EvalError::UnknownParameter(_))));
        assert!(matches!(e("(1"), Err(EvalError::UnexpectedEnd)));
    }

    #[test]
    fn deep_nesting_is_rejected_not_overflowed() {
        let src = format!("{}1{}", "(".repeat(500), ")".repeat(500));
        assert!(matches!(e(&src), Err(EvalError::DepthLimit(_))));
    }

    #[test]
    fn number_then_identifier_that_is_not_a_unit() {
        let mut r = |n: &str| (n == "e").then_some(3.0);
        // `2e` -> 2 followed by `e`; two atoms in a row is trailing input.
        assert!(matches!(
            eval("2e", Unit::Millimeter, &mut r),
            Err(EvalError::TrailingInput { .. })
        ));
        assert_eq!(eval("2e-1", Unit::Millimeter, &mut r).unwrap(), 0.2);
    }

    #[test]
    fn dotted_identifiers_resolve() {
        let mut r = |n: &str| (n == "body.bust").then_some(940.0);
        assert_eq!(
            eval("body.bust / 2", Unit::Millimeter, &mut r).unwrap(),
            470.0
        );
    }
}
