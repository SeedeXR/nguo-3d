// SPDX-License-Identifier: GPL-2.0-or-later
//! Display units and the canonical [`Length`] (always millimetres).

use serde::{Deserialize, Serialize};

/// A unit a human may type or read. Internal maths never uses these — see
/// [`Length`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub enum Unit {
    /// The canonical internal unit.
    #[default]
    Millimeter,
    Centimeter,
    Inch,
}

impl Unit {
    /// Every unit we support, in menu order.
    pub const ALL: [Unit; 3] = [Unit::Millimeter, Unit::Centimeter, Unit::Inch];

    /// Millimetres in one of this unit. The single source of conversion truth.
    pub const fn mm_per(self) -> f64 {
        match self {
            Unit::Millimeter => 1.0,
            Unit::Centimeter => 10.0,
            Unit::Inch => 25.4,
        }
    }

    /// The canonical suffix used when formatting.
    pub const fn suffix(self) -> &'static str {
        match self {
            Unit::Millimeter => "mm",
            Unit::Centimeter => "cm",
            Unit::Inch => "in",
        }
    }

    /// Sensible number of decimals for this unit at garment scale.
    pub const fn default_decimals(self) -> usize {
        match self {
            Unit::Millimeter => 1,
            Unit::Centimeter => 2,
            Unit::Inch => 3,
        }
    }

    /// Parse a suffix a user might type. Case-insensitive; accepts `"` for inch.
    pub fn from_suffix(s: &str) -> Option<Unit> {
        // ponytail: no locale-specific suffixes (e.g. `pouce`) — i18n of unit
        // input lands with fluent-rs (guide §20), not here.
        match s.trim().to_ascii_lowercase().as_str() {
            "mm" | "millimeter" | "millimetre" | "millimeters" | "millimetres" => {
                Some(Unit::Millimeter)
            }
            "cm" | "centimeter" | "centimetre" | "centimeters" | "centimetres" => {
                Some(Unit::Centimeter)
            }
            "in" | "\"" | "inch" | "inches" => Some(Unit::Inch),
            _ => None,
        }
    }
}

impl core::fmt::Display for Unit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.suffix())
    }
}

/// A length in the canonical unit: **millimetres**.
///
/// Deliberately a newtype so a raw `f64` in some other unit cannot be assigned
/// to a geometry field by accident.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Length(f64);

impl Length {
    pub const ZERO: Length = Length(0.0);

    pub const fn from_mm(mm: f64) -> Length {
        Length(mm)
    }

    pub fn from_unit(value: f64, unit: Unit) -> Length {
        Length(value * unit.mm_per())
    }

    /// The canonical value, in millimetres.
    pub const fn mm(self) -> f64 {
        self.0
    }

    /// Convert for display.
    pub fn to_unit(self, unit: Unit) -> f64 {
        self.0 / unit.mm_per()
    }

    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Format for display, e.g. `12.50cm`. Round-trips through [`parse_length`]
    /// to within the printed precision.
    pub fn format(self, unit: Unit, decimals: usize) -> String {
        format!("{:.*}{}", decimals, self.to_unit(unit), unit.suffix())
    }
}

impl core::ops::Add for Length {
    type Output = Length;
    fn add(self, rhs: Length) -> Length {
        Length(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Length {
    type Output = Length;
    fn sub(self, rhs: Length) -> Length {
        Length(self.0 - rhs.0)
    }
}

impl core::ops::Mul<f64> for Length {
    type Output = Length;
    fn mul(self, rhs: f64) -> Length {
        Length(self.0 * rhs)
    }
}

/// A plain `<number><unit?>` failed to parse. Full expressions go through
/// [`crate::eval`] instead.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseLengthError {
    #[error("empty input")]
    Empty,
    #[error("`{0}` is not a number")]
    NotANumber(String),
    #[error("unknown unit `{0}`")]
    UnknownUnit(String),
}

/// Parse `"12.5cm"`, `"3 in"`, `"7"` (the last one in `default_unit`).
///
/// This is the fast path for a field that only ever holds a literal. Fields
/// that accept formulas call [`crate::eval`], which subsumes this.
pub fn parse_length(s: &str, default_unit: Unit) -> Result<Length, ParseLengthError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ParseLengthError::Empty);
    }
    // Split at the first byte that cannot continue a float literal.
    let split = s
        .char_indices()
        .position(|(_, c)| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(s.len());
    let (num, suffix) = s.split_at(split);
    let num = num.trim();
    let suffix = suffix.trim();
    let value: f64 = num
        .parse()
        .map_err(|_| ParseLengthError::NotANumber(num.to_string()))?;
    let unit = if suffix.is_empty() {
        default_unit
    } else {
        Unit::from_suffix(suffix)
            .ok_or_else(|| ParseLengthError::UnknownUnit(suffix.to_string()))?
    };
    Ok(Length::from_unit(value, unit))
}

impl Length {
    /// Convenience wrapper over [`parse_length`].
    pub fn parse(s: &str, default_unit: Unit) -> Result<Length, ParseLengthError> {
        parse_length(s, default_unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_are_exact_where_they_can_be() {
        assert_eq!(Length::from_unit(1.0, Unit::Centimeter).mm(), 10.0);
        assert_eq!(Length::from_unit(1.0, Unit::Inch).mm(), 25.4);
        assert_eq!(Length::from_mm(25.4).to_unit(Unit::Inch), 1.0);
    }

    #[test]
    fn parses_suffixes() {
        assert_eq!(
            parse_length("12.5cm", Unit::Millimeter).unwrap().mm(),
            125.0
        );
        assert_eq!(parse_length("2 in", Unit::Millimeter).unwrap().mm(), 50.8);
        assert_eq!(
            parse_length("3\"", Unit::Millimeter).unwrap().mm(),
            3.0 * 25.4
        );
        assert_eq!(parse_length("7", Unit::Centimeter).unwrap().mm(), 70.0);
        assert_eq!(parse_length("-1.5e1mm", Unit::Inch).unwrap().mm(), -15.0);
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(
            parse_length("", Unit::Millimeter),
            Err(ParseLengthError::Empty)
        );
        assert!(matches!(
            parse_length("abc", Unit::Millimeter),
            Err(ParseLengthError::NotANumber(_))
        ));
        assert!(matches!(
            parse_length("5furlongs", Unit::Millimeter),
            Err(ParseLengthError::UnknownUnit(_))
        ));
    }

    #[test]
    fn formats() {
        assert_eq!(
            Length::from_mm(125.0).format(Unit::Centimeter, 2),
            "12.50cm"
        );
        assert_eq!(Length::from_mm(25.4).format(Unit::Inch, 3), "1.000in");
    }
}
