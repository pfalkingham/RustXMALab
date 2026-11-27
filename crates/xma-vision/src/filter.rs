//! Low-pass filtering and gap interpolation of time series.
//!
//! Work package: filtering and interpolation. Three origins:
//!
//! - The Butterworth coefficients ([`ButterworthLowPass::new`]) are ported from
//!   `processing/ButterworthLowPassFilter.cpp`, which takes them from Exstrom's `liir.c`.
//! - The zero-phase filtering ([`ButterworthLowPass::filter`]) and the cubic spline
//!   ([`cubic_spline`]) are original code of this project, written on 16 November 2026 from
//!   the mathematical specification `docs/filter-spline-spec.md`, not ported (parity ledger,
//!   item A208). They give the same numbers as XMALab 3, bit for bit.
//! - The segment handling is ported from `core/Marker.cpp` (`filterMarker`, `filterData`)
//!   and `core/RigidBody.cpp` (`filterTransformations`, `filterData`).
//!
//! # How XMALab filters a series with gaps
//!
//! Both callers do the same thing, and the pieces here are meant to be used the same way:
//!
//! 1. [`can_filter`]: refuse unless `cutoff > 0`, `recordingSpeed > 0` and
//!    `0 < cutoff / (recordingSpeed / 2) < 1`, always on the entered cutoff. On refusal the
//!    C++ filters nothing. Otherwise the filter is designed with
//!    [`ButterworthLowPass::zero_phase`], whose [`CutoffCorrection`] corrects the cutoff for
//!    the two passes (ledger A7; not in the XMALab 3 mode).
//! 2. [`runs`]: split frames `0..nbImages` into runs of consecutive valid frames (marker
//!    `status3D > 0`; body `poseComputed`).
//! 3. [`filter_run`] per run and per component: runs of [`MIN_FILTERED_RUN`] frames or fewer
//!    are copied unchanged (parity ledger, item A8); longer runs go through
//!    [`ButterworthLowPass::filter`] (order [`FILTER_ORDER`]). [`filter_runs`] does steps 2
//!    and 3 for one component into a caller-initialised output, so frames outside every run
//!    keep whatever the caller put there (the C++ markers use -1000, bodies keep the old
//!    filtered values).
//!
//! # Floating point
//!
//! Everything is `f64`, with the order of operations fixed by the C++ (coefficients) or by
//! the specification's section 4 (filtering and spline), never fusing a multiply and an add
//! except in the one place the specification fuses (the steady-state right-hand side, see
//! [`ButterworthLowPass::filter`]). `sin`, `cos` and `pow` are the platform's libm, as in the
//! C++ (`f64::powf`, not `powi`, because the C++ calls `pow(double, int)`, which is
//! `pow(double, double)`).

use std::f64::consts::PI;
use std::ops::Range;

use nalgebra::DMatrix;

use crate::linalg;

/// Filter order XMALab uses everywhere. Orders 1 to 4 reproduce the C++ on every series
/// tested; orders 5 and above differ in the last bits, see
/// [`ButterworthLowPass::filter`].
pub const FILTER_ORDER: usize = 4;

/// Runs of this many frames or fewer are copied through unfiltered (parity ledger, item A8).
///
/// Equal to `3 * FILTER_ORDER`, the length at or below which `filtfilt` throws.
pub const MIN_FILTERED_RUN: usize = 12;

/// The highest filter order there is a filter for (parity ledger, item A209).
///
/// The numerator integers ([`ButterworthLowPass::c`]) are binomial coefficients computed in
/// `i32` as liir and XMALab compute them (`ccof[i] = (n - i + 1) * ccof[i - 1] / i`, in C's
/// 32-bit `int` in XMALab). The intermediate product is `i * C(n, i)`; its largest value
/// for order 29 is `14 * C(29, 14) = 1 085 822 640`, below `i32::MAX`, and for order 30 it is
/// `15 * C(30, 15) = 2 326 762 800`, above it, so from order 30 the C++ overflows (undefined
/// behaviour) and so would the port. Up to 29 the arithmetic is exact and the coefficients
/// are XMALab's. The bound also keeps the work of the design and of the steady state (an
/// `order x order` matrix and its inverse) small and fixed.
///
/// Outside `1..=MAX_FILTER_ORDER` there is no filter: [`ButterworthLowPass::new`] and
/// [`ButterworthLowPass::zero_phase`] return the filter of order 0, which filters nothing
/// ([`ButterworthLowPass::filter`] gives `None`, [`filter_run`] copies), and
/// [`design_cutoff`] returns the cutoff unchanged. A filter built through the public fields
/// whose order (the longer of `d` and `c`, less one) is outside the range is treated the
/// same way.
pub const MAX_FILTER_ORDER: usize = 29;

/// `true` when there is a filter of this order ([`MAX_FILTER_ORDER`]).
fn order_in_range(order: usize) -> bool {
    (1..=MAX_FILTER_ORDER).contains(&order)
}

/// A Butterworth low-pass filter applied forwards and backwards (zero phase).
#[derive(Clone, Debug, PartialEq)]
pub struct ButterworthLowPass {
    /// Denominator coefficients (C++ `dcof`), `order + 1` of them; `d[0]` is 1.
    pub d: Vec<f64>,
    /// Numerator coefficients before scaling (C++ `ccof`), `order + 1` binomial coefficients.
    pub c: Vec<i32>,
    /// Scaling factor (C++ `sf`).
    pub scale: f64,
}

/// The refusal test both callers apply before filtering: `true` when filtering goes ahead.
///
/// Ported from `core/Marker.cpp:1126-1128`, `Marker::filterMarker`, and the identical test in
/// `core/RigidBody.cpp:1914-1916`, `RigidBody::filterTransformations`. The ratio is computed
/// as the C++ does, `cutoff / (recording * 0.5)`, twice. NaN in either argument refuses.
#[must_use]
pub fn can_filter(cutoff_hz: f64, recording_hz: f64) -> bool {
    cutoff_hz > 0.0
        && recording_hz > 0.0
        && 0.0 < (cutoff_hz / (recording_hz * 0.5))
        && (cutoff_hz / (recording_hz * 0.5)) < 1.0
}

/// How the entered cutoff is turned into the cutoff of the single-pass design (parity
/// ledger, item A7).
///
/// [`ButterworthLowPass::filter`] applies the filter twice, forwards and backwards, so the
/// amplitude response as applied is the square of the single-pass one. Designed at the
/// entered cutoff, the data are attenuated to 0.5 there, not 1/sqrt(2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CutoffCorrection {
    /// The default (ledger A7 fixed): the single pass is designed at [`design_cutoff`], so
    /// that the filter as applied (both passes) has amplitude 1/sqrt(2) at the entered
    /// cutoff.
    #[default]
    BothPasses,
    /// XMALab 3: the single pass is designed at the entered cutoff, so the filter as applied
    /// has amplitude 0.5 there (ledger A7 as the C++ has it).
    SinglePass,
}

/// The cutoff handed to the single-pass design ([`ButterworthLowPass::new`]) for an entered
/// cutoff `cutoff_hz`.
///
/// Implements parity ledger item A7 (not ported; the C++ makes no correction). For
/// [`CutoffCorrection::SinglePass`] it is `cutoff_hz`, bit for bit. For
/// [`CutoffCorrection::BothPasses`], the cutoff at which a bilinear-transform Butterworth of
/// order `order`, applied twice, has amplitude 1/sqrt(2) at `cutoff_hz`:
///
/// `(recording / PI) * atan(tan(PI * cutoff / recording) / k)`, `k = (sqrt(2) - 1)^(1 / (2 order))`
///
/// (the single pass has `|H|^2 = 1 / (1 + (tan(w/2) / tan(wc/2))^(2 order))`; setting its
/// square to 1/2 at the entered cutoff gives `tan(wc/2) = tan(w/2) / k`). The result is
/// above the entered cutoff (k = 0.8957 for order 4: 25 Hz at 250 Hz becomes 27.69 Hz), and
/// never below it.
///
/// Whenever [`can_filter`] accepts `cutoff_hz`, it accepts the result too and the result is
/// finite: where rounding would put the design cutoff at or beyond the Nyquist frequency
/// (or at 0), the nearest value [`can_filter`] accepts is returned instead. When
/// [`can_filter`] refuses `cutoff_hz` (the callers then filter nothing), `cutoff_hz` is
/// returned unchanged.
///
/// For an order outside `1..=`[`MAX_FILTER_ORDER`] (no filter) `cutoff_hz` is returned
/// unchanged (parity ledger, item A209).
#[must_use]
pub fn design_cutoff(
    order: usize,
    cutoff_hz: f64,
    recording_hz: f64,
    correction: CutoffCorrection,
) -> f64 {
    if !order_in_range(order)
        || correction == CutoffCorrection::SinglePass
        || !can_filter(cutoff_hz, recording_hz)
    {
        return cutoff_hz;
    }
    // `order` is at most MAX_FILTER_ORDER here.
    let n = f64::from(u32::try_from(order).unwrap_or(u32::MAX));
    let k = (2.0f64.sqrt() - 1.0).powf(1.0 / (2.0 * n));
    let x = PI * (cutoff_hz / recording_hz);
    let d = if x < 1e-100 {
        // tan and atan are the identity to far below rounding here, and the formula would
        // lose precision to subnormals (or underflow to 0).
        cutoff_hz / k
    } else {
        (recording_hz / PI) * (x.tan() / k).atan()
    };
    // The exact value is above the entered cutoff (k < 1); close to the Nyquist frequency,
    // where `tan` is ill-conditioned, rounding could put it below.
    nearest_accepted(d.max(cutoff_hz), cutoff_hz, recording_hz)
}

/// `d` when [`can_filter`] accepts it and it is finite; otherwise the nearest value it
/// accepts, stepping one ulp at a time towards the accepted range, and `fallback` (which it
/// accepts) if that takes more than a few hundred steps.
fn nearest_accepted(d: f64, fallback: f64, recording_hz: f64) -> f64 {
    if !d.is_finite() {
        return fallback;
    }
    let mut v = d;
    for _ in 0..256 {
        if can_filter(v, recording_hz) {
            return v;
        }
        let ratio = v / (recording_hz * 0.5);
        v = if ratio >= 1.0 {
            v.next_down()
        } else {
            v.next_up()
        };
    }
    fallback
}

impl ButterworthLowPass {
    /// Designs the zero-phase filter for an entered cutoff: [`ButterworthLowPass::new`] at
    /// [`design_cutoff`] (parity ledger, item A7). This is what every caller uses; callers
    /// test [`can_filter`] on the entered cutoff first.
    ///
    /// For an order outside `1..=`[`MAX_FILTER_ORDER`], the filter of order 0, which filters
    /// nothing (see [`ButterworthLowPass::new`]).
    #[must_use]
    pub fn zero_phase(
        order: usize,
        cutoff_hz: f64,
        recording_hz: f64,
        correction: CutoffCorrection,
    ) -> Self {
        Self::new(
            order,
            design_cutoff(order, cutoff_hz, recording_hz, correction),
            recording_hz,
        )
    }

    /// Designs the filter for one pass at `cutoff_hz`.
    ///
    /// Ported from `ButterworthLowPassFilter::ButterworthLowPassFilter(order, cutOffFrequency,
    /// recordingFrequency)` and the `liir.c` functions it calls (`dcof_bwlp`, `binomial_mult`,
    /// `ccof_bwlp`, `sf_bwlp`). The cutoff is normalised as `cutoff / (recording * 0.5)` and
    /// not corrected for the two passes (parity ledger, item A7): callers do not call this
    /// directly but [`ButterworthLowPass::zero_phase`], which applies the correction (or not,
    /// in the XMALab 3 mode). No range check is made here (the C++ constructor makes none
    /// either); callers test [`can_filter`] first. Tier A.
    ///
    /// For an order outside `1..=`[`MAX_FILTER_ORDER`] there is no filter (the C++ reads
    /// outside its arrays for order 0 and overflows `int` from order 30): the result is the
    /// filter of order 0, `d = [1]`, `c = [1]`, `scale = 1`, which filters nothing
    /// ([`ButterworthLowPass::order`] is 0, [`ButterworthLowPass::filter`] gives `None`,
    /// [`filter_run`] copies). Parity ledger, item A209.
    #[must_use]
    pub fn new(order: usize, cutoff_hz: f64, recording_hz: f64) -> Self {
        if !order_in_range(order) {
            return Self {
                d: vec![1.0],
                c: vec![1],
                scale: 1.0,
            };
        }
        let fcf = cutoff_hz / (recording_hz * 0.5);
        let d = dcof_bwlp(order, fcf);
        let c = ccof_bwlp(order);
        let scale = sf_bwlp(order, fcf);
        Self { d, c, scale }
    }

    /// Filter order (`d.len() - 1`; 0 when `d` is empty).
    #[must_use]
    pub fn order(&self) -> usize {
        self.d.len().saturating_sub(1)
    }

    /// Filters one contiguous run of samples forwards and backwards.
    ///
    /// Returns `None` when the input has `3 * order` samples or fewer (12 for order 4): the
    /// C++ `filtfilt` throws `std::domain_error` there, which XMALab never catches; its
    /// callers never get there because they copy such runs instead ([`filter_run`]).
    ///
    /// Written from docs/filter-spline-spec.md, sections 2 and 5; not a port (parity ledger,
    /// item A208). The numerator is `c[k] * scale`, the denominator `d`. The signal is
    /// extended at each end by `3 * order` samples reflected through the end sample (odd
    /// extension), filtered forwards from the step steady state scaled by the first extended
    /// sample, reversed, filtered again from the steady state scaled by the first sample of
    /// the reversed series, reversed back, and the extension cut off. The method is the one
    /// documented for MATLAB's and SciPy's `filtfilt` (odd padding, `lfilter_zi` initial
    /// state); see also Gustafsson, "Determining the initial states in forward-backward
    /// filtering", IEEE Trans. Signal Processing 44(4), 1996. Tier A.
    ///
    /// The steady-state right-hand side `b[k] - a[k] b[0]` is rounded once, as
    /// `fma(a[k], -b[0], b[k])`, because that is what XMALab computed on the reference machine
    /// (its matrix library fused the operation there); [`filter_unfused_zi`] rounds twice.
    ///
    /// Orders: order 4 (the only one XMALab uses), 2 and 3 match the C++ bit for bit; order 1
    /// matched on every series tested. Orders 5 and above differ from the C++ in the last
    /// bits. A filter with no order, or of an order above [`MAX_FILTER_ORDER`] (the longer of
    /// `d` and `c`, less one), gives `None` (parity ledger, item A209).
    /// Oracle: `trial_<t>/points3d_filtered.csv`, `trial_<t>/rb_pose_filtered.csv`, and the
    /// direct fixtures in `tests/data/filter/`.
    #[must_use]
    pub fn filter(&self, input: &[f64]) -> Option<Vec<f64>> {
        zero_phase(&design_coefficients(self), &self.d, input, StepRhs::Fused)
    }
}

// ---------------------------------------------------------------------------------------
// Coefficients (liir.c)
// ---------------------------------------------------------------------------------------

/// Ported from `processing/ButterworthLowPassFilter.cpp`, `binomial_mult`.
fn binomial_mult(n: usize, p: &[f64]) -> Vec<f64> {
    let mut a = vec![0.0f64; 2 * n];
    for i in 0..n {
        for j in (1..=i).rev() {
            a[2 * j] += p[2 * i] * a[2 * (j - 1)] - p[2 * i + 1] * a[2 * (j - 1) + 1];
            a[2 * j + 1] += p[2 * i] * a[2 * (j - 1) + 1] + p[2 * i + 1] * a[2 * (j - 1)];
        }
        a[0] += p[2 * i];
        a[1] += p[2 * i + 1];
    }
    a
}

/// Ported from `processing/ButterworthLowPassFilter.cpp`, `dcof_bwlp`. Returns the first
/// `n + 1` entries, the ones the filter reads.
fn dcof_bwlp(n: usize, fcf: f64) -> Vec<f64> {
    let mut rcof = vec![0.0f64; 2 * n];
    let theta = PI * fcf;
    let st = theta.sin();
    let ct = theta.cos();
    let n_i = i32::try_from(n).expect("filter order fits in i32");
    for k in 0..n {
        let k_i = i32::try_from(k).expect("filter order fits in i32");
        let parg = PI * f64::from(2 * k_i + 1) / f64::from(2 * n_i);
        let sparg = parg.sin();
        let cparg = parg.cos();
        let a = 1.0 + st * sparg;
        rcof[2 * k] = -ct / a;
        rcof[2 * k + 1] = -st * cparg / a;
    }
    let mut dcof = binomial_mult(n, &rcof);
    dcof[1] = dcof[0];
    dcof[0] = 1.0;
    for k in 3..=n {
        dcof[k] = dcof[2 * k - 2];
    }
    dcof.truncate(n + 1);
    dcof
}

/// Ported from `processing/ButterworthLowPassFilter.cpp`, `ccof_bwlp` (integer arithmetic).
fn ccof_bwlp(n: usize) -> Vec<i32> {
    let n_i = i32::try_from(n).expect("filter order fits in i32");
    let mut ccof = vec![0i32; n + 1];
    ccof[0] = 1;
    ccof[1] = n_i;
    let m = n / 2;
    for i in 2..=m {
        let i_i = i32::try_from(i).expect("filter order fits in i32");
        ccof[i] = (n_i - i_i + 1) * ccof[i - 1] / i_i;
        ccof[n - i] = ccof[i];
    }
    ccof[n - 1] = n_i;
    ccof[n] = 1;
    ccof
}

/// Ported from `processing/ButterworthLowPassFilter.cpp`, `sf_bwlp`.
fn sf_bwlp(n: usize, fcf: f64) -> f64 {
    let n_i = i32::try_from(n).expect("filter order fits in i32");
    let omega = PI * fcf;
    let mut fomega = omega.sin();
    let parg0 = PI / f64::from(2 * n_i);
    let mut sf = 1.0f64;
    for k in 0..(n_i / 2) {
        sf *= 1.0 + fomega * (f64::from(2 * k + 1) * parg0).sin();
    }
    fomega = (omega / 2.0).sin();
    if n % 2 == 1 {
        sf *= fomega + (omega / 2.0).cos();
    }
    fomega.powf(f64::from(n_i)) / sf
}

// ---------------------------------------------------------------------------------------
// Zero-phase filtering
// ---------------------------------------------------------------------------------------

/// How the right-hand side `b_k - a_k b_0` of the steady-state system is rounded
/// (`docs/filter-spline-spec.md`, section 2.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepRhs {
    /// `fma(a_k, -b_0, b_k)`, one rounding: what the reference machine computed; used by
    /// [`ButterworthLowPass::filter`].
    Fused,
    /// `(a_k * -b_0) + b_k`, two roundings. For the tests only.
    Unfused,
}

/// The two coefficient vectors brought to a common length `N = max(len b, len a)` by
/// appending `+0` entries (specification, section 2.2).
///
/// Written from docs/filter-spline-spec.md, section 2.2; not a port.
struct Coefficients {
    b: Vec<f64>,
    a: Vec<f64>,
}

impl Coefficients {
    fn padded(b: &[f64], a: &[f64]) -> Self {
        let n = b.len().max(a.len());
        let pad = |v: &[f64]| {
            let mut out = v.to_vec();
            out.resize(n, 0.0);
            out
        };
        Self {
            b: pad(b),
            a: pad(a),
        }
    }

    /// The filter order `m = N - 1`, or `None` when it is outside `1..=MAX_FILTER_ORDER`
    /// (no filter; parity ledger, item A209).
    fn order(&self) -> Option<usize> {
        self.a.len().checked_sub(1).filter(|&m| order_in_range(m))
    }
}

/// The state in which the filter rests after an infinitely long unit step: the solution
/// `zeta` of `(I - C) zeta = b[1..] - a[1..] b[0]`, where `C` is the companion-type matrix
/// with first column `-a[1..]` and ones on the superdiagonal. Unscaled (a unit step);
/// computed from the padded coefficients without normalising by `a[0]`. Length `m`.
///
/// The system is solved by forming the inverse with [`linalg::invert`] (OpenCV's `DECOMP_LU`
/// procedure, Appendix B of the specification; its matrix is used whether or not it reports
/// a singular matrix, in which case it is all zeros) and multiplying, each row summed left
/// to right with separately rounded products; a `-0` result becomes `+0`.
///
/// Written from docs/filter-spline-spec.md, section 2.4 and Appendix B; not a port. The
/// steady-state initial condition is that of Gustafsson (1996) and of SciPy's `lfilter_zi`.
fn step_steady_state(coef: &Coefficients, m: usize, rhs: StepRhs) -> Vec<f64> {
    let (b, a) = (&coef.b, &coef.a);
    let matrix = DMatrix::from_fn(m, m, |row, col| {
        if col == 0 {
            if row == 0 { 1.0 + a[1] } else { a[row + 1] }
        } else if col == row {
            1.0
        } else if col == row + 1 {
            -1.0
        } else {
            0.0
        }
    });
    let rho: Vec<f64> = (1..=m)
        .map(|k| match rhs {
            StepRhs::Fused => a[k].mul_add(-b[0], b[k]),
            StepRhs::Unfused => a[k] * -b[0] + b[k],
        })
        .collect();
    let inverse = linalg::invert(&matrix).x;
    (0..m)
        .map(|row| {
            let mut t = inverse[(row, 0)] * rho[0];
            for (col, &r) in rho.iter().enumerate().skip(1) {
                t += inverse[(row, col)] * r;
            }
            // `* 1 + 0`: the value unchanged, except that -0 becomes +0.
            t * 1.0 + 0.0
        })
        .collect()
}

/// One pass of the recursive filter over `v`, started from the state `state` (length `m`),
/// by the difference equation
///
/// `w[i] = beta[0] v[i] + sum_{k=1}^{min(i,m)} (beta[k] v[i-k] - alpha[k] w[i-k]) + [i < m] state[i]`,
///
/// which is the transposed direct form II started from `state`. `beta` and `alpha` are the
/// coefficients divided by `a[0]` when `a[0]` is not exactly 1. The sum is accumulated from
/// the highest lag down, starting from the state entry (or `+0`), and `beta[0] v[i]` is
/// added last.
///
/// Written from docs/filter-spline-spec.md, section 2.5; not a port. The difference equation
/// and the transposed direct form II are in any signal-processing text (Oppenheim and
/// Schafer, *Discrete-Time Signal Processing*).
fn recursive_pass(coef: &Coefficients, m: usize, v: &[f64], state: &[f64]) -> Vec<f64> {
    let a0 = coef.a[0];
    let (beta, alpha): (Vec<f64>, Vec<f64>) = if a0 == 1.0 {
        (coef.b.clone(), coef.a.clone())
    } else {
        (
            coef.b.iter().map(|&x| x / a0).collect(),
            coef.a.iter().map(|&x| x / a0).collect(),
        )
    };
    let mut w: Vec<f64> = Vec::with_capacity(v.len());
    for (i, &vi) in v.iter().enumerate() {
        let mut s = if i < m { state[i] } else { 0.0 };
        for k in (1..=i.min(m)).rev() {
            s += beta[k] * v[i - k] - alpha[k] * w[i - k];
        }
        w.push(beta[0] * vi + s);
    }
    w
}

/// Applies the filter `(b, a)` forwards and then backwards (zero phase), with an odd
/// extension of `3m` samples at each end and the step steady state, scaled by the first
/// sample each pass sees, as initial state. `None` when the signal has `3m` samples or
/// fewer, or when the padded coefficients have no order (`m < 1`).
///
/// Written from docs/filter-spline-spec.md, sections 2.2, 2.3 and 2.6; not a port. The method
/// is that documented for MATLAB's `filtfilt` and SciPy's `filtfilt` with `padtype='odd'`.
fn zero_phase(b: &[f64], a: &[f64], x: &[f64], rhs: StepRhs) -> Option<Vec<f64>> {
    let coef = Coefficients::padded(b, a);
    let m = coef.order()?;
    let edge = m.checked_mul(3)?;
    let len = x.len();
    if len <= edge {
        return None;
    }

    // Odd (point-reflected) extension about the end samples, which are not repeated.
    let first = x[0];
    let last = x[len - 1];
    let mut u = Vec::with_capacity(len + 2 * edge);
    u.extend((0..edge).map(|j| 2.0 * first - x[edge - j]));
    u.extend_from_slice(x);
    u.extend((0..edge).map(|j| 2.0 * last - x[len - 2 - j]));

    let zeta = step_steady_state(&coef, m, rhs);
    let scaled = |by: f64| zeta.iter().map(|&z| z * by).collect::<Vec<f64>>();

    let forward = recursive_pass(&coef, m, &u, &scaled(u[0]));
    let reversed: Vec<f64> = forward.into_iter().rev().collect();
    let backward = recursive_pass(&coef, m, &reversed, &scaled(reversed[0]));

    // Drop `edge` samples at each end and restore the order.
    Some(backward[edge..edge + len].iter().rev().copied().collect())
}

/// The numerator `c[k] * scale` and denominator of a designed filter (specification,
/// section 2.7).
fn design_coefficients(f: &ButterworthLowPass) -> Vec<f64> {
    f.c.iter().map(|&c| f64::from(c) * f.scale).collect()
}

#[doc(hidden)]
/// [`ButterworthLowPass::filter`] with the initial conditions computed without fused
/// multiply-adds, as OpenCV does on a CPU without FMA. For tests only.
#[must_use]
pub fn filter_unfused_zi(f: &ButterworthLowPass, input: &[f64]) -> Option<Vec<f64>> {
    zero_phase(&design_coefficients(f), &f.d, input, StepRhs::Unfused)
}

#[doc(hidden)]
/// The step steady-state initial condition (unscaled), fused (`fma = true`, as
/// [`ButterworthLowPass::filter`] uses) or unfused, for the tests. Empty when there is no
/// filter: no order (`d` and `c` of length 1 or less) or an order above
/// [`MAX_FILTER_ORDER`].
#[must_use]
pub fn initial_conditions_for(f: &ButterworthLowPass, fma: bool) -> Vec<f64> {
    let coef = Coefficients::padded(&design_coefficients(f), &f.d);
    let rhs = if fma {
        StepRhs::Fused
    } else {
        StepRhs::Unfused
    };
    coef.order()
        .map_or_else(Vec::new, |m| step_steady_state(&coef, m, rhs))
}

// ---------------------------------------------------------------------------------------
// Runs with gaps
// ---------------------------------------------------------------------------------------

/// Splits frames into runs of consecutive valid frames, in order.
///
/// Ported from the loops of `core/Marker.cpp:1130-1154`, `Marker::filterMarker` (valid:
/// `status3D > 0`) and `core/RigidBody.cpp:1918-1942`, `RigidBody::filterTransformations`
/// (valid: `poseComputed`). `valid` has one entry per frame of the trial (`nbImages`).
#[must_use]
pub fn runs(valid: &[bool]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &v) in valid.iter().enumerate() {
        match (v, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(s..valid.len());
    }
    out
}

/// Filters one run of one component as `filterData` does: a run of [`MIN_FILTERED_RUN`]
/// samples or fewer is returned unchanged (parity ledger, item A8), a longer one goes through
/// [`ButterworthLowPass::filter`].
///
/// Ported from `core/Marker.cpp:1515-1569`, `Marker::filterData`, and
/// `core/RigidBody.cpp:1615-1697`, `RigidBody::filterData` (per component).
///
/// A longer run that [`ButterworthLowPass::filter`] still refuses (a filter of order above
/// 4 and a run of 13 to `3 * order` samples, or no filter: order 0 or above
/// [`MAX_FILTER_ORDER`]) is also returned unchanged (parity ledger, item A209). XMALab only uses
/// order 4, for which this cannot happen.
#[must_use]
pub fn filter_run(filter: &ButterworthLowPass, values: &[f64]) -> Vec<f64> {
    if values.len() <= MIN_FILTERED_RUN {
        values.to_vec()
    } else {
        filter.filter(values).unwrap_or_else(|| values.to_vec())
    }
}

/// Filters one component over the given runs, writing into `out` only the frames that
/// belong to a run. Frames outside every run keep what `out` held.
///
/// Ported from `Marker::filterMarker` / `Marker::filterData` and
/// `RigidBody::filterTransformations` / `RigidBody::filterData` (pose mode), one component
/// at a time; get `runs` from [`runs`] once and call this for x, y, z (or the six pose
/// components). The caller applies [`can_filter`] first.
///
/// Misuse does not panic (parity ledger, item A209): a run that is reversed or does not lie
/// inside both `values` and `out` is skipped, leaving `out` as it was for those frames; the
/// runs that do lie inside both are filtered even when `values` and `out` differ in length.
pub fn filter_runs(
    filter: &ButterworthLowPass,
    values: &[f64],
    runs: &[Range<usize>],
    out: &mut [f64],
) {
    let limit = values.len().min(out.len());
    for run in runs {
        if run.start > run.end || run.end > limit {
            continue;
        }
        let filtered = filter_run(filter, &values[run.clone()]);
        out[run.clone()].copy_from_slice(&filtered);
    }
}

// ---------------------------------------------------------------------------------------
// Cubic spline (not-a-knot, Hermite form)
// ---------------------------------------------------------------------------------------

/// Solves a tridiagonal system by Gaussian elimination down the diagonal without row
/// exchanges, then back-substitution (the Thomas algorithm). Row `i` reads
/// `sub[i] x[i-1] + diag[i] x[i] + sup[i] x[i+1] = rhs[i]` (`sub[0]` and the last `sup` are
/// not read). The first pivot is not tested; a later pivot that is exactly zero (either
/// sign) makes the solve fail (`None`). A NaN pivot does not fail. All four slices have the
/// same length, at least 1.
///
/// Written from docs/filter-spline-spec.md, section 3.4; not a port. The algorithm is in any
/// numerical-analysis text (Burden and Faires, *Numerical Analysis*; Press et al.,
/// *Numerical Recipes*).
fn thomas_solve(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    // ratio[i] = sup[i-1] / pivot[i-1]; ratio[0] is unused.
    let mut ratio = vec![0.0f64; n];
    let mut reduced = vec![0.0f64; n];
    let mut pivot = diag[0];
    reduced[0] = rhs[0] / pivot;
    for i in 1..n {
        ratio[i] = sup[i - 1] / pivot;
        pivot = diag[i] - sub[i] * ratio[i];
        if pivot == 0.0 {
            return None;
        }
        reduced[i] = (rhs[i] - reduced[i - 1] * sub[i]) / pivot;
    }
    let mut x = reduced;
    for i in (0..n - 1).rev() {
        x[i] -= x[i + 1] * ratio[i + 1];
    }
    Some(x)
}

/// The slopes at the knots of the cubic spline through `(knots, values)` with not-a-knot
/// ends: continuity of the second derivative at the interior knots, and of the third
/// derivative at the second and the second-to-last knots. `None` when the tridiagonal solve
/// fails. Needs at least 3 knots and `values.len() == knots.len()`.
///
/// Written from docs/filter-spline-spec.md, section 3.3; not a port. The slope formulation
/// and the not-a-knot condition are those of de Boor, *A Practical Guide to Splines*, and of
/// Wolberg, "Cubic spline interpolation: a review" (Columbia University, CUCS-389-88, 1988).
fn not_a_knot_slopes(knots: &[f64], values: &[f64]) -> Option<Vec<f64>> {
    let n = knots.len();
    let width: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();
    let slope: Vec<f64> = (0..n - 1)
        .map(|i| (values[i + 1] - values[i]) / width[i])
        .collect();

    let mut sub = vec![0.0f64; n];
    let mut diag = vec![0.0f64; n];
    let mut sup = vec![0.0f64; n];
    let mut rhs = vec![0.0f64; n];

    // Left end: h1 s0 + (h0 + h1) s1 = ((3 h0 h1 + 2 h1^2) d0 + h0^2 d1) / (h0 + h1),
    // multiplied through by (h0 + h1).
    let (h0, h1) = (width[0], width[1]);
    let sum = h0 + h1;
    diag[0] = h1 * sum;
    sup[0] = sum * sum;
    rhs[0] = slope[0] * (3.0 * h0 * h1 + 2.0 * h1 * h1) + slope[1] * h0 * h0;

    // Interior: h_i s_{i-1} + 2 (h_{i-1} + h_i) s_i + h_{i-1} s_{i+1}
    //         = 3 (h_i d_{i-1} + h_{i-1} d_i).
    for i in 1..n - 1 {
        sub[i] = width[i];
        diag[i] = 2.0 * (width[i - 1] + width[i]);
        sup[i] = width[i - 1];
        rhs[i] = 3.0 * (slope[i - 1] * width[i] + slope[i] * width[i - 1]);
    }

    // Right end: (P + Q)^2 s_{n-2} + P (P + Q) s_{n-1} = Q^2 d_{n-3} + (3 P Q + 2 P^2) d_{n-2}.
    let (p, q) = (width[n - 3], width[n - 2]);
    let sum = p + q;
    sub[n - 1] = sum * sum;
    diag[n - 1] = p * sum;
    rhs[n - 1] = slope[n - 3] * q * q + slope[n - 2] * (3.0 * p * q + 2.0 * p * p);

    thomas_solve(&sub, &diag, &sup, &rhs)
}

/// Interpolates `(x, y)` by a cubic spline with not-a-knot ends and evaluates it at `at`.
///
/// `x` are the knots (increasing), `y` the values, `at` the evaluation points (in XMALab
/// increasing and strictly inside the knots). Returns one value per entry of `at`, or `None`:
/// when `x` and `y` differ in length (parity ledger, item A209), when there are fewer than 3
/// knots or no evaluation points, when the slope system's solve fails, when the first point
/// is below the first knot or the last point above the last knot, or when a point would open
/// an interval at or beyond the last knot.
///
/// On each knot interval the spline is the cubic Hermite polynomial with the knot values and
/// the slopes of [`not_a_knot_slopes`], in power form about the interval's left knot,
/// evaluated by Horner's rule. The points are taken in order; a new interval is looked for
/// only when a point is beyond the right end of the current one, and the search only moves
/// forwards. Points that do not leave the current interval to the right are evaluated on its
/// cubic, by extrapolation if they lie left of it. The behaviour this gives on unusual input
/// (three knots, a point on the last knot, a huge or NaN first point, repeated knots) is
/// listed in parity ledger item A208.
///
/// Written from docs/filter-spline-spec.md, sections 3.1, 3.2 and 3.5; not a port. Cubic
/// Hermite interpolation is in any numerical-analysis text (Burden and Faires); the
/// not-a-knot spline is in de Boor, *A Practical Guide to Splines*, and Wolberg (1988).
#[must_use]
pub fn cubic_spline(x: &[f64], y: &[f64], at: &[f64]) -> Option<Vec<f64>> {
    let n = x.len();
    if y.len() != n || n < 3 || at.is_empty() {
        return None;
    }
    let s = not_a_knot_slopes(x, y)?;
    if at[0] < x[0] || at[at.len() - 1] > x[n - 1] {
        return None;
    }

    let mut j = 0usize;
    let mut left = 0.0f64;
    let mut right = at[0] - 1.0;
    let mut c = [0.0f64; 4];
    let mut out = Vec::with_capacity(at.len());
    for &t in at {
        if t > right {
            while j < n && t > x[j] {
                j += 1;
            }
            if j == n {
                return None;
            }
            if t < x[j] {
                if j == 0 {
                    return None;
                }
                j -= 1;
            }
            if j == n - 1 {
                return None;
            }
            left = x[j];
            right = x[j + 1];
            let inv_width = 1.0 / (x[j + 1] - x[j]);
            let chord = (y[j + 1] - y[j]) * inv_width;
            c = [
                y[j],
                s[j],
                inv_width * (3.0 * chord - 2.0 * s[j] - s[j + 1]),
                inv_width * inv_width * (-2.0 * chord + s[j] + s[j + 1]),
            ];
        }
        let theta = t - left;
        out.push(((c[3] * theta + c[2]) * theta + c[1]) * theta + c[0]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccof_is_binomial() {
        assert_eq!(ccof_bwlp(4), vec![1, 4, 6, 4, 1]);
        assert_eq!(ccof_bwlp(1), vec![1, 1]);
        assert_eq!(ccof_bwlp(5), vec![1, 5, 10, 10, 5, 1]);
    }
}
