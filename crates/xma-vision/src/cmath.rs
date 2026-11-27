//! The C maths library's `pow`, as XMALab calls it.
//!
//! XMALab's standard deviations add `pow(x, 2)` (core/RigidBody.cpp:438, :1267, :1313,
//! core/Trial.cpp:1392, :1479, :2488, core/Marker.cpp:1683). The reference build does not
//! turn that call into `x * x`: it calls the C library's `pow`, which on glibc differs from
//! the correctly rounded `x * x` in the last bit for about 1 argument in 1,100 (measured on
//! a million random numbers in [0, 1)), so the standard deviations differ in their last bits
//! too. [`pow`] calls the platform's `pow` for every exponent, so on Linux it gives glibc's
//! values; on other platforms the maths library's own (parity ledger A57).

/// The C library's `pow(x, y)`, never replaced by a multiplication: the exponent is hidden
/// from the compiler, which would otherwise fold `pow(x, 2.0)` into `x * x`.
#[must_use]
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    x.powf(std::hint::black_box(y))
}

/// The C library's `sqrt`, as the generated expressions of [`crate::multicam::funcs`] and
/// [`crate::pose_refine`] call it.
#[must_use]
#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// The C library's `sin`, as the generated expressions call it.
#[must_use]
#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}

/// The C library's `cos`, as the generated expressions call it.
#[must_use]
#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}

/// The C library's `fabs`, as the generated expressions call it.
#[must_use]
#[inline]
pub(crate) fn fabs(x: f64) -> f64 {
    x.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_the_library() {
        assert_eq!(pow(3.0, 2.0), 9.0);
        assert_eq!(pow(2.0, 0.5), 2f64.sqrt());
        // An argument where glibc's pow and the rounded product differ (found by search on
        // the reference platform); elsewhere the two may agree.
        let x = 0.011_930_246_636_300_074_f64;
        let p = pow(x, 2.0);
        assert!((p - x * x).abs() <= f64::EPSILON * x * x);
    }
}
