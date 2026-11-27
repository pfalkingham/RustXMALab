//! Parsing of single cells.

/// Parses a number as C's `printf` family writes it, or as XMALab's exports write it.
///
/// Accepts everything Rust's `f64::from_str` accepts, plus the spellings C libraries
/// and XMALab use for special values, case-insensitively and with an optional sign:
/// `nan`, `-nan`, `NaN`, `nan(...)`, `-nan(ind)`, `inf`, `-inf`, `infinity`.
/// Surrounding spaces and tabs are ignored (some XMALab files separate cells with `", "`).
/// A leading `-` on a NaN is kept as the sign bit.
///
/// Returns `None` for anything else, including the empty string.
pub fn parse_f64(s: &str) -> Option<f64> {
    let t = s.trim_matches([' ', '\t']);
    if t.is_empty() {
        return None;
    }
    let (negative, body) = match t.as_bytes()[0] {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    let lower = body.to_ascii_lowercase();
    if lower == "nan" || (lower.starts_with("nan(") && lower.ends_with(')')) {
        return Some(if negative { -f64::NAN } else { f64::NAN });
    }
    if lower == "inf" || lower == "infinity" {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    // Reject a second sign, which `from_str` would otherwise see after we stripped one.
    if body.starts_with(['+', '-']) {
        return None;
    }
    t.parse::<f64>().ok()
}

/// Parses a signed decimal integer. Surrounding spaces and tabs are ignored.
pub fn parse_i64(s: &str) -> Option<i64> {
    s.trim_matches([' ', '\t']).parse::<i64>().ok()
}

/// Parses a non-negative decimal integer. Surrounding spaces and tabs are ignored.
pub fn parse_usize(s: &str) -> Option<usize> {
    let t = s.trim_matches([' ', '\t']);
    if t.starts_with('+') {
        return None;
    }
    t.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_as_printf_writes_them() {
        assert_eq!(parse_f64("0.29999999999999999"), Some(0.3));
        assert_eq!(parse_f64("1e+100"), Some(1e100));
        assert_eq!(parse_f64("-2.5e-07"), Some(-2.5e-7));
        assert_eq!(parse_f64("4.9406564584124654e-324"), Some(5e-324));
        assert_eq!(parse_f64(" 381.82788"), Some(381.82788));
        assert_eq!(parse_f64("-0").map(f64::to_bits), Some((-0.0f64).to_bits()));
        assert_eq!(parse_f64("inf"), Some(f64::INFINITY));
        assert_eq!(parse_f64("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(parse_f64("-Infinity"), Some(f64::NEG_INFINITY));
        let n = parse_f64("nan").unwrap();
        assert!(n.is_nan() && n.is_sign_positive());
        let n = parse_f64("-nan").unwrap();
        assert!(n.is_nan() && n.is_sign_negative());
        assert!(parse_f64("NaN").unwrap().is_nan());
        assert!(parse_f64("-nan(ind)").unwrap().is_nan());
        assert_eq!(parse_f64(""), None);
        assert_eq!(parse_f64("--1"), None);
        assert_eq!(parse_f64("abc"), None);
        assert_eq!(parse_f64("1,2"), None);
    }

    #[test]
    fn integers() {
        assert_eq!(parse_i64("-2"), Some(-2));
        assert_eq!(parse_i64(" 40"), Some(40));
        assert_eq!(parse_i64("1.0"), None);
        assert_eq!(parse_usize("7"), Some(7));
        assert_eq!(parse_usize("-1"), None);
    }
}
