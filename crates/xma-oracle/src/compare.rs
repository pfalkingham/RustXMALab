//! Comparing computed values with reference values.

use std::fmt;

/// A tolerance for [`Comparison::check`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// Passes if `|got − want| ≤ tol`.
    Abs(f64),
    /// Passes if `|got − want| ≤ tol · |want|`. A zero reference passes only an exact zero.
    Rel(f64),
    /// Passes if the value is within the absolute tolerance OR within the relative one.
    Either {
        /// Absolute tolerance.
        abs: f64,
        /// Relative tolerance.
        rel: f64,
    },
}

impl fmt::Display for Tolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tolerance::Abs(a) => write!(f, "abs {a:e}"),
            Tolerance::Rel(r) => write!(f, "rel {r:e}"),
            Tolerance::Either { abs, rel } => write!(f, "abs {abs:e} or rel {rel:e}"),
        }
    }
}

/// The differences between one computed value and its reference.
///
/// Two NaNs are equal (both differences 0). A NaN against a number, or an infinity
/// against anything else, gives infinite differences. The relative difference is
/// `abs / |want|`; when `want` is 0 it is 0 if `got` is also 0 and infinite otherwise.
pub fn diff(got: f64, want: f64) -> (f64, f64) {
    if got.is_nan() && want.is_nan() {
        return (0.0, 0.0);
    }
    if got.is_nan() || want.is_nan() {
        return (f64::INFINITY, f64::INFINITY);
    }
    if got == want {
        return (0.0, 0.0);
    }
    let abs = (got - want).abs();
    let abs = if abs.is_nan() { f64::INFINITY } else { abs };
    let rel = if want == 0.0 {
        f64::INFINITY
    } else {
        abs / want.abs()
    };
    (abs, rel)
}

fn ratio(d: f64, tol: f64) -> f64 {
    if d == 0.0 {
        0.0
    } else if tol > 0.0 {
        d / tol
    } else {
        f64::INFINITY
    }
}

/// How far a value is outside a tolerance: at most 1 means it passes.
fn score(abs: f64, rel: f64, tol: Tolerance) -> f64 {
    match tol {
        Tolerance::Abs(a) => ratio(abs, a),
        Tolerance::Rel(r) => ratio(rel, r),
        Tolerance::Either { abs: a, rel: r } => ratio(abs, a).min(ratio(rel, r)),
    }
}

/// One compared value, as reported by the queries of [`Comparison`].
#[derive(Debug, Clone, PartialEq)]
pub struct Worst {
    /// The label given when the value was added.
    pub label: String,
    /// The computed value.
    pub got: f64,
    /// The reference value.
    pub want: f64,
    /// Absolute difference (see [`diff`]).
    pub abs: f64,
    /// Relative difference (see [`diff`]).
    pub rel: f64,
}

impl fmt::Display for Worst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: got {:.16e}, want {:.16e} (abs diff {:.3e}, rel diff {:.3e})",
            self.label, self.got, self.want, self.abs, self.rel
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    got: f64,
    want: f64,
    label: u32,
    /// Index within a slice added by `add_slice`, or `u32::MAX` for a single value.
    sub: u32,
}

/// An accumulator of `(label, got, want)` triples.
///
/// Add values with [`Comparison::add`] or [`Comparison::add_slice`], then query the
/// largest differences, or check against a [`Tolerance`]. The checks fail when no
/// value was added, since comparing nothing is almost always a mistake in a test.
///
/// `Display` gives a one-line summary for a parity report, for example
/// `points3d x: 1450 values, max abs 1.137e-13 (marker 3 frame 17), max rel 4.012e-15 (marker 0 frame 2)`.
/// Failure messages show the worst value's label and both values with 17 significant digits.
#[derive(Debug, Clone, Default)]
pub struct Comparison {
    name: String,
    labels: Vec<String>,
    entries: Vec<Entry>,
}

impl Comparison {
    /// An empty comparison. `name` starts the summary and failure messages.
    pub fn new(name: impl Into<String>) -> Comparison {
        Comparison {
            name: name.into(),
            labels: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// The name given to [`Comparison::new`].
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adds one value.
    pub fn add(&mut self, label: impl fmt::Display, got: f64, want: f64) {
        self.labels.push(label.to_string());
        self.entries.push(Entry {
            got,
            want,
            label: (self.labels.len() - 1) as u32,
            sub: u32::MAX,
        });
    }

    /// Adds two equally long slices element by element; element `i` is labelled
    /// `label[i]`. Panics if the lengths differ.
    #[track_caller]
    pub fn add_slice(&mut self, label: impl fmt::Display, got: &[f64], want: &[f64]) {
        let label = label.to_string();
        assert_eq!(
            got.len(),
            want.len(),
            "{}: {label}: got {} values, want {}",
            self.name,
            got.len(),
            want.len()
        );
        self.labels.push(label);
        let id = (self.labels.len() - 1) as u32;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            self.entries.push(Entry {
                got: g,
                want: w,
                label: id,
                sub: i as u32,
            });
        }
    }

    /// Number of values added.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// True if no value was added.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn label_of(&self, e: &Entry) -> String {
        let l = &self.labels[e.label as usize];
        if e.sub == u32::MAX {
            l.clone()
        } else {
            format!("{l}[{}]", e.sub)
        }
    }

    fn report(&self, e: &Entry) -> Worst {
        let (abs, rel) = diff(e.got, e.want);
        Worst {
            label: self.label_of(e),
            got: e.got,
            want: e.want,
            abs,
            rel,
        }
    }

    /// The first entry with the largest key (so ties report the earliest value).
    fn max_by(&self, key: impl Fn(&Entry) -> f64) -> Option<&Entry> {
        let mut best: Option<(&Entry, f64)> = None;
        for e in &self.entries {
            let k = key(e);
            if best.is_none_or(|(_, b)| k > b) {
                best = Some((e, k));
            }
        }
        best.map(|(e, _)| e)
    }

    /// Largest absolute difference (0 if empty; infinite if a NaN met a number).
    pub fn max_abs(&self) -> f64 {
        self.worst_abs().map_or(0.0, |w| w.abs)
    }

    /// Largest relative difference (0 if empty).
    pub fn max_rel(&self) -> f64 {
        self.worst_rel().map_or(0.0, |w| w.rel)
    }

    /// The value with the largest absolute difference.
    pub fn worst_abs(&self) -> Option<Worst> {
        self.max_by(|e| diff(e.got, e.want).0)
            .map(|e| self.report(e))
    }

    /// The value with the largest relative difference.
    pub fn worst_rel(&self) -> Option<Worst> {
        self.max_by(|e| diff(e.got, e.want).1)
            .map(|e| self.report(e))
    }

    /// Number of values where exactly one of `got` and `want` is NaN.
    pub fn nan_mismatches(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.got.is_nan() != e.want.is_nan())
            .count()
    }

    /// Number of values outside the tolerance.
    pub fn failures(&self, tol: Tolerance) -> usize {
        self.entries
            .iter()
            .filter(|e| {
                let (a, r) = diff(e.got, e.want);
                score(a, r, tol) > 1.0
            })
            .count()
    }

    /// The value furthest outside the tolerance (relative to the tolerance), whether or
    /// not it fails.
    pub fn worst(&self, tol: Tolerance) -> Option<Worst> {
        self.max_by(|e| {
            let (a, r) = diff(e.got, e.want);
            score(a, r, tol)
        })
        .map(|e| self.report(e))
    }

    /// `Ok` if every value is within the tolerance and at least one value was added;
    /// otherwise a message naming the worst value.
    pub fn check(&self, tol: Tolerance) -> Result<(), String> {
        if self.entries.is_empty() {
            return Err(format!("{}: no values were compared", self.name));
        }
        let n = self.failures(tol);
        if n == 0 {
            return Ok(());
        }
        let worst = self.worst(tol).expect("not empty");
        Err(format!(
            "{}: {n} of {} values outside {tol}; worst {worst}",
            self.name,
            self.count()
        ))
    }

    /// Panics with [`Comparison::check`]'s message unless every value is within `tol`.
    #[track_caller]
    pub fn assert(&self, tol: Tolerance) {
        if let Err(m) = self.check(tol) {
            panic!("{m}");
        }
    }

    /// Asserts `|got − want| ≤ tol` for every value.
    #[track_caller]
    pub fn assert_abs(&self, tol: f64) {
        self.assert(Tolerance::Abs(tol));
    }

    /// Asserts `|got − want| ≤ tol · |want|` for every value.
    #[track_caller]
    pub fn assert_rel(&self, tol: f64) {
        self.assert(Tolerance::Rel(tol));
    }

    /// Asserts that every value is within `abs` absolutely or within `rel` relatively.
    #[track_caller]
    pub fn assert_either(&self, abs: f64, rel: f64) {
        self.assert(Tolerance::Either { abs, rel });
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (Some(wa), Some(wr)) = (self.worst_abs(), self.worst_rel()) else {
            return write!(f, "{}: no values compared", self.name);
        };
        write!(
            f,
            "{}: {} values, max abs {:.3e} ({}), max rel {:.3e} ({})",
            self.name,
            self.count(),
            wa.abs,
            wa.label,
            wr.rel,
            wr.label
        )?;
        let nan = self.nan_mismatches();
        if nan > 0 {
            write!(f, ", {nan} NaN mismatches")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffs() {
        assert_eq!(diff(1.0, 1.0), (0.0, 0.0));
        assert_eq!(diff(f64::NAN, -f64::NAN), (0.0, 0.0));
        assert_eq!(diff(f64::NAN, 1.0), (f64::INFINITY, f64::INFINITY));
        assert_eq!(diff(1.0, f64::NAN).0, f64::INFINITY);
        assert_eq!(diff(f64::INFINITY, f64::INFINITY), (0.0, 0.0));
        assert_eq!(diff(f64::INFINITY, 1.0).0, f64::INFINITY);
        assert_eq!(diff(f64::INFINITY, f64::NEG_INFINITY).0, f64::INFINITY);
        assert_eq!(diff(1e-20, 0.0), (1e-20, f64::INFINITY));
        assert_eq!(diff(0.0, -0.0), (0.0, 0.0));
        assert_eq!(diff(3.0, 2.0), (1.0, 0.5));
    }

    #[test]
    fn queries_and_checks() {
        let mut c = Comparison::new("pts");
        c.add("a", 1.0, 1.0);
        c.add("b", 100.0 + 1e-9, 100.0);
        c.add("c", 1e-12, 0.0);
        c.add_slice("v", &[1.0, 2.0 + 1e-15], &[1.0, 2.0]);
        assert_eq!(c.count(), 5);
        let wa = c.worst_abs().unwrap();
        assert_eq!(wa.label, "b");
        assert!((c.max_abs() - 1e-9).abs() < 1e-13);
        let wr = c.worst_rel().unwrap();
        assert_eq!(wr.label, "c");
        assert_eq!(c.max_rel(), f64::INFINITY);
        assert!(c.check(Tolerance::Abs(1e-8)).is_ok());
        assert_eq!(c.failures(Tolerance::Abs(1e-10)), 1);
        // "c" fails any relative tolerance (reference 0) but passes the absolute part.
        assert_eq!(c.failures(Tolerance::Rel(1e-6)), 1);
        c.assert_either(1e-11, 1e-10);
        assert_eq!(
            c.failures(Tolerance::Either {
                abs: 1e-13,
                rel: 1e-12
            }),
            2
        );
        assert_eq!(
            c.failures(Tolerance::Either {
                abs: 1e-11,
                rel: 1e-12
            }),
            1
        );
        assert_eq!(c.worst(Tolerance::Abs(1e-10)).unwrap().label, "b");
        assert_eq!(c.worst(Tolerance::Rel(1.0)).unwrap().label, "c");
        assert_eq!(c.nan_mismatches(), 0);
        let s = c.to_string();
        assert!(
            s.starts_with("pts: 5 values, max abs 1.000e-9 (b), max rel inf (c)"),
            "{s}"
        );
    }

    #[test]
    fn slice_labels_and_nan() {
        let mut c = Comparison::new("x");
        c.add_slice("row", &[1.0, f64::NAN, f64::NAN], &[1.0, f64::NAN, 5.0]);
        assert_eq!(c.nan_mismatches(), 1);
        let w = c.worst_abs().unwrap();
        assert_eq!(w.label, "row[2]");
        assert!(w.got.is_nan());
        let msg = c.check(Tolerance::Abs(1.0)).unwrap_err();
        assert!(msg.contains("1 of 3 values outside abs 1e0"), "{msg}");
        assert!(
            msg.contains("row[2]: got NaN, want 5.0000000000000000e0"),
            "{msg}"
        );
        assert!(c.to_string().ends_with(", 1 NaN mismatches"));
    }

    #[test]
    fn failure_shows_17_digits() {
        let mut c = Comparison::new("t");
        c.add("p", 0.1 + 0.2, 0.3);
        let msg = c.check(Tolerance::Abs(0.0)).unwrap_err();
        assert!(
            msg.contains("got 3.0000000000000004e-1, want 2.9999999999999999e-1"),
            "{msg}"
        );
        c.assert_rel(1e-15);
    }

    #[test]
    fn empty_comparison_fails_checks() {
        let c = Comparison::new("none");
        assert_eq!(c.max_abs(), 0.0);
        assert!(c.check(Tolerance::Abs(1.0)).is_err());
        assert_eq!(c.to_string(), "none: no values compared");
    }

    #[test]
    #[should_panic(expected = "worst q: got")]
    fn assert_abs_panics() {
        let mut c = Comparison::new("t");
        c.add("q", 1.0, 2.0);
        c.assert_abs(0.5);
    }
}
