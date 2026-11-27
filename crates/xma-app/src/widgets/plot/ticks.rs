//! Axis ticks and tick labels, as QCustomPlot 2.1 makes them for XMALab's plot.
//!
//! XMALab never changes QCustomPlot's ticker, so every axis uses the default
//! `QCPAxisTicker`: about five ticks, steps with a mantissa of 1, 2, 2.5, 5 or
//! 10, sub-ticks chosen from the step's mantissa, and labels formatted with
//! `QLocale::toString(value, 'g', 6)`. Tick placement does not depend on the
//! axis length in pixels, only on the visible range.

/// Ticks for one axis over one range.
#[derive(Clone, Debug, PartialEq)]
pub struct Ticks {
    /// The step between major ticks.
    pub step: f64,
    /// Major ticks inside the range, ascending.
    pub major: Vec<f64>,
    /// Sub-ticks inside the range, ascending.
    pub minor: Vec<f64>,
    /// One label per major tick.
    pub labels: Vec<String>,
}

/// QCustomPlot's default tick count (`QCPAxisTicker::mTickCount`).
const TICK_COUNT: f64 = 5.0;

/// Mantissa and magnitude of `input` (`142.6` gives `1.426` and `100`).
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisTicker::getMantissa.
fn mantissa(input: f64) -> (f64, f64) {
    let mag = 10f64.powf(input.log10().floor());
    (input / mag, mag)
}

/// The candidate closest to `target`; ties go to the lower candidate.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisTicker::pickClosest.
fn pick_closest(target: f64, candidates: &[f64]) -> f64 {
    let i = candidates.partition_point(|c| *c < target);
    if i == candidates.len() {
        candidates[i - 1]
    } else if i == 0 {
        candidates[0]
    } else if target - candidates[i - 1] < candidates[i] - target {
        candidates[i - 1]
    } else {
        candidates[i]
    }
}

/// The major tick step for a visible range of size `span`.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisTicker::getTickStep
/// and QCPAxisTicker::cleanMantissa (strategy `tssReadability`).
pub fn tick_step(span: f64) -> f64 {
    let exact = span / (TICK_COUNT + 1e-10);
    let (m, mag) = mantissa(exact);
    pick_closest(m, &[1.0, 2.0, 2.5, 5.0, 10.0]) * mag
}

/// The number of sub-ticks between two major ticks for a tick step.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisTicker::getSubTickCount.
pub fn sub_tick_count(step: f64) -> usize {
    let epsilon = 0.01;
    let (m, _) = mantissa(step);
    let mut int_part = m.trunc() as i64;
    let frac = m - m.trunc();
    if frac < epsilon || 1.0 - frac < epsilon {
        if 1.0 - frac < epsilon {
            int_part += 1;
        }
        match int_part {
            1 => 4,
            2 => 3,
            3 => 2,
            4 => 3,
            5 => 4,
            6 => 2,
            7 => 6,
            8 => 3,
            9 => 2,
            _ => 1,
        }
    } else if (frac - 0.5).abs() < epsilon {
        match int_part {
            1 => 2,
            2 => 4,
            3 => 4,
            4 => 2,
            5 => 4,
            6 => 4,
            7 => 2,
            8 => 4,
            9 => 4,
            _ => 1,
        }
    } else {
        1
    }
}

/// Drops ticks outside `[lo, hi]`, keeping one outlier on each side if asked.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisTicker::trimTicks.
fn trim(lo: f64, hi: f64, ticks: &mut Vec<f64>, keep_one_outlier: bool) {
    let low = ticks.iter().position(|t| *t >= lo);
    let high = ticks.iter().rposition(|t| *t <= hi);
    match (low, high) {
        (Some(low), Some(high)) => {
            let keep = usize::from(keep_one_outlier);
            let front = low.saturating_sub(keep);
            let back = ticks.len().saturating_sub(1 + keep + high);
            let end = ticks.len() - back;
            if front >= end {
                ticks.clear();
            } else {
                ticks.truncate(end);
                ticks.drain(..front);
            }
        }
        _ => ticks.clear(),
    }
}

/// Major ticks, sub-ticks and labels for the visible range `(lo, hi)`.
///
/// Ported from external/QCustomPlot/qcustomplot.cpp, QCPAxisTicker::generate
/// with createTickVector and createSubTickVector (tick origin 0).
pub fn ticks(range: (f64, f64)) -> Ticks {
    let (lo, hi) = (range.0.min(range.1), range.0.max(range.1));
    let span = hi - lo;
    let empty = Ticks {
        step: 1.0,
        major: Vec::new(),
        minor: Vec::new(),
        labels: Vec::new(),
    };
    if span.is_nan() || span <= 0.0 || !span.is_finite() {
        return empty;
    }
    let step = tick_step(span);
    if step.is_nan() || step <= 0.0 || !step.is_finite() {
        return empty;
    }
    let first = (lo / step).floor() as i64;
    let last = (hi / step).ceil() as i64;
    // QCustomPlot has no cap; a cap only guards against absurd ranges.
    let count = (last - first + 1).clamp(0, 10_000);
    let mut major: Vec<f64> = (0..count).map(|i| (first + i) as f64 * step).collect();
    trim(lo, hi, &mut major, true);
    let mut minor = Vec::new();
    let subs = sub_tick_count(step);
    if subs > 0 && major.len() >= 2 {
        for w in major.windows(2) {
            let sub_step = (w[1] - w[0]) / (subs + 1) as f64;
            for k in 1..=subs {
                minor.push(w[0] + k as f64 * sub_step);
            }
        }
    }
    trim(lo, hi, &mut minor, false);
    trim(lo, hi, &mut major, false);
    let labels = major.iter().map(|v| format_g(*v, 6)).collect();
    Ticks {
        step,
        major,
        minor,
        labels,
    }
}

/// `QString::number(value)`, Qt's `'g'` with precision 6, which is also what
/// QCustomPlot's tick labels give (`QLocale::toString(value, 'g', 6)` with group
/// separators off: the same text, checked with Qt 6.12). Exact ties round away
/// from zero and `-0` is `0`. Calls the engine's
/// [`xma_engine::view::plots::qt_number`], so there is one implementation.
///
/// Only precision 6 exists in XMALab's labels; `precision` is kept for the
/// callers and must be 6.
pub fn format_g(value: f64, precision: usize) -> String {
    debug_assert_eq!(precision, 6, "only Qt's default precision 6 is implemented");
    xma_engine::view::plots::qt_number(value)
}

/// Turns `1.5e+06` into `1.5·10⁶`, as QCustomPlot does with "beautiful powers"
/// (its default number format `gbd`). Other labels are returned unchanged.
pub fn beautify_exponent(label: &str) -> String {
    let Some((mant, exp)) = label.split_once('e') else {
        return label.to_owned();
    };
    let exp: i32 = match exp.parse() {
        Ok(e) => e,
        Err(_) => return label.to_owned(),
    };
    const SUP: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let mut out = String::new();
    if mant != "1" {
        out.push_str(mant);
        out.push('·');
    }
    out.push_str("10");
    if exp < 0 {
        out.push('⁻');
    }
    for d in exp.unsigned_abs().to_string().bytes() {
        out.push(SUP[(d - b'0') as usize]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_match_reference_screenshot() {
        // docs/ui-reference/pig/PlotWindow.png: X 1..145 in steps of 25,
        // left 636.6..670.8 in steps of 5, right ~290..640 in steps of 50.
        assert_eq!(tick_step(144.0), 25.0);
        assert_eq!(tick_step(34.2), 5.0);
        assert_eq!(tick_step(350.0), 50.0);
    }

    #[test]
    fn format_g_matches_c() {
        assert_eq!(format_g(670.0, 6), "670");
        assert_eq!(format_g(0.30000000000000004, 6), "0.3");
        assert_eq!(format_g(1234567.0, 6), "1.23457e+06");
        assert_eq!(format_g(0.0001, 6), "0.0001");
        assert_eq!(format_g(0.00001, 6), "1e-05");
        assert_eq!(format_g(-2.5, 6), "-2.5");
        assert_eq!(format_g(100000.0, 6), "100000");
        assert_eq!(format_g(999999.5, 6), "1e+06");
        assert_eq!(beautify_exponent("1.5e+06"), "1.5·10⁶");
        assert_eq!(beautify_exponent("1e-05"), "10⁻⁵");
    }

    /// Review 12: labels round as Qt does (`QString::number` and QCustomPlot's
    /// `QLocale::toString(v, 'g', 6)`, which give the same text; checked with Qt 6.12):
    /// exact ties away from zero, and `-0` written `0`.
    #[test]
    fn review12_format_g_rounds_as_qt() {
        assert_eq!(format_g(1234565.0, 6), "1.23457e+06");
        assert_eq!(format_g(-1234565.0, 6), "-1.23457e+06");
        assert_eq!(format_g(123456.5, 6), "123457");
        assert_eq!(format_g(-0.0, 6), "0");
        // A tick label on an exact tie: the range 1234550..1234580 has step 5.
        let t = ticks((1234550.0, 1234580.0));
        let i = t.major.iter().position(|v| *v == 1234565.0).unwrap();
        assert_eq!(t.labels[i], "1.23457e+06");
    }
}
