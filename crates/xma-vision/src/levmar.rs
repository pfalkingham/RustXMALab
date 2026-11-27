//! Levenberg-Marquardt least squares with an analytic Jacobian.
//!
//! Ported from levmar 2.6 (Manolis Lourakis, GPL-2.0-or-later, see `oracle/THIRD_PARTY.md`),
//! `dlevmar_der`, exactly as the reference build runs it. The C++ uses it for rigid-body pose
//! refinement (`processing/RigidBodyPoseOptimization.cpp`) and for multi-camera bundle
//! adjustment (`processing/MultiCameraCalibration.cpp`).
//!
//! # What "as the reference build runs it" means
//!
//! The reference build links Fedora's levmar 2.6, compiled with `HAVE_LAPACK`, double
//! precision and without `LINSOLVERS_RETAIN_MEMORY`. Its parts are ported as follows.
//!
//! - `lm_core.c`, `LEVMAR_DER` instantiated for `double` (`lm.c`): [`dlevmar_der`], line by
//!   line, with every stopping rule, the damping update, the `info[]` contents and the return
//!   value ([`Info::return_value`]).
//! - `misc_core.c`, `LEVMAR_L2NRMXMY`: [`l2nrmxmy`], with its unrolled summation order.
//! - The `J^T J` product. Problems with `n*m < 1024` (the 6-parameter pose problem with up to
//!   170 residuals) use the loop in `lm_core.c`, ported exactly. Larger problems call
//!   `LEVMAR_TRANS_MAT_MAT_MULT`, which with `HAVE_LAPACK` is BLAS `dgemm`; the port
//!   reproduces OpenBLAS's `dgemm` for six parameters ([`trans_mat_mat_mult`]), but not
//!   necessarily for other parameter counts.
//! - `Axb_core.c`, `Ax_eq_b_BK`, which calls LAPACK `dsytrf("L")` and `dsytrs("L")`
//!   (Bunch-Kaufman `L D L^T`): [`ax_eq_b_bk`]. `dsytrf` uses the unblocked `dsytf2` when the
//!   order is at most its block size (64), which covers the pose problem; the port always
//!   uses `dsytf2` ([`dsytf2_lower`]). Problems with more than 64 parameters (bundle
//!   adjustment) take LAPACK's blocked path (`dlasyf`) in the reference build and may round
//!   differently.
//!
//! The reference build's LAPACK is FlexiBLAS 3.5 with its default backend, OpenBLAS 0.3.29
//! (OpenMP), which picks the "Cooperlake" kernels on the reference machine. `dsytf2` and
//! `dsytrs` are netlib Fortran inside OpenBLAS (the older `dsytf2` that folds `1/D21` into
//! the 2x2 update, compiled without fused multiply-add), but the BLAS routines they call are
//! OpenBLAS kernels that use fused multiply-add and their own summation order. The port
//! reproduces those kernels as they behave for the sizes used here ([`dsyr_lower`],
//! [`dger_column`], [`dgemv_t_column`]); each was checked bit for bit against the library for
//! vectors of length 1 to 40. With netlib BLAS, or OpenBLAS on a CPU without FMA, the
//! reference results would differ in the last bits.
//!
//! The port keeps no global state and uses no `unsafe`, so unlike the C library under the
//! C++'s mutex it may be called from several threads at once.

use std::fmt;

/// Default scale factor for the initial damping `mu` (`LM_INIT_MU` in `levmar.h`).
pub const LM_INIT_MU: f64 = 1e-3;
/// Default stopping threshold (`LM_STOP_THRESH` in `levmar.h`).
pub const LM_STOP_THRESH: f64 = 1e-17;
/// Number of entries in levmar's `info[]` array (`LM_INFO_SZ`).
pub const LM_INFO_SZ: usize = 10;

/// `EPSILON` in `lm.c`, used by the "almost singular" test.
const EPSILON: f64 = 1e-12;
/// `ONE_THIRD` in `lm.c` (sic: not exactly 1/3).
const ONE_THIRD: f64 = 0.3333333334;
/// `__BLOCKSZ__SQ` in `misc.h`: below this many Jacobian entries `lm_core.c` uses its own
/// `J^T J` loop.
const BLOCKSZ_SQ: usize = 32 * 32;

/// Minimisation options, levmar's `opts[0..4]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// `opts[0]`: scale factor for the initial damping, `mu = tau * max(diag(J^T J))`.
    pub mu: f64,
    /// `opts[1]`: stop when `||J^T e||_inf <= eps1`.
    pub eps1: f64,
    /// `opts[2]`: stop when `||Dp||_2 <= eps2 * ||p||_2`.
    pub eps2: f64,
    /// `opts[3]`: stop when `||e||_2^2 <= eps3`.
    pub eps3: f64,
}

impl Options {
    /// levmar's defaults, used when `opts` is `NULL`: `{LM_INIT_MU, LM_STOP_THRESH,
    /// LM_STOP_THRESH, LM_STOP_THRESH}`.
    pub const DEFAULT: Options = Options {
        mu: LM_INIT_MU,
        eps1: LM_STOP_THRESH,
        eps2: LM_STOP_THRESH,
        eps3: LM_STOP_THRESH,
    };

    /// The options of the C++ rigid-body pose refinement
    /// (`RigidBodyPoseOptimization::optimizeRigidBodySetup`): `{0.01, 1e-30, 1e-50, 1e-50}`.
    pub const XMALAB_POSE: Options = Options {
        mu: 0.01,
        eps1: 1e-30,
        eps2: 1e-50,
        eps3: 1e-50,
    };

    /// The options as levmar's `opts[0..4]` array.
    pub fn to_array(&self) -> [f64; 4] {
        [self.mu, self.eps1, self.eps2, self.eps3]
    }
}

impl Default for Options {
    fn default() -> Self {
        Options::DEFAULT
    }
}

/// Why the minimisation stopped, levmar's `info[6]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StopReason {
    /// 1: small gradient, `||J^T e||_inf <= eps1`.
    SmallGradient = 1,
    /// 2: small step, `||Dp||_2 <= eps2 * ||p||_2`.
    SmallStep = 2,
    /// 3: `itmax` iterations reached.
    MaxIterations = 3,
    /// 4: singular matrix (the step is enormous). The return value is an error.
    SingularMatrix = 4,
    /// 5: no further error reduction is possible (the damping counter overflowed).
    NoFurtherReduction = 5,
    /// 6: small error, `||e||_2^2 <= eps3`.
    SmallError = 6,
    /// 7: the function returned NaN or infinity. The return value is an error.
    InvalidValues = 7,
}

impl StopReason {
    /// The numeric code levmar stores in `info[6]`.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// The reason for a numeric code from levmar's `info[6]`.
    pub fn from_code(code: i32) -> Option<StopReason> {
        Some(match code {
            1 => StopReason::SmallGradient,
            2 => StopReason::SmallStep,
            3 => StopReason::MaxIterations,
            4 => StopReason::SingularMatrix,
            5 => StopReason::NoFurtherReduction,
            6 => StopReason::SmallError,
            7 => StopReason::InvalidValues,
            _ => return None,
        })
    }
}

/// Information about a minimisation, levmar's `info[0..10]`.
///
/// levmar's comments call `info[0]`, `info[1]` and `info[3]` 2-norms, but the code stores the
/// squared norms; so does this port.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Info {
    /// `info[0]`: `||e||_2^2` at the initial `p`.
    pub initial_error: f64,
    /// `info[1]`: `||e||_2^2` at the final `p`.
    pub error: f64,
    /// `info[2]`: `||J^T e||_inf` at the last Jacobian (0 if none was computed).
    pub gradient_inf: f64,
    /// `info[3]`: `||Dp||_2^2` of the last step tried (0 after a small-gradient stop,
    /// `f64::MAX` if no step was ever computed).
    pub step: f64,
    /// `info[4]`: `mu / max(diag(J^T J))`. When the run stops before the first Jacobian the C
    /// reads uninitialised memory here; the port reads zeros and gives NaN.
    pub mu_ratio: f64,
    /// `info[5]`: number of iterations.
    pub iterations: usize,
    /// `info[6]`: why the minimisation stopped.
    pub stop: StopReason,
    /// `info[7]`: number of function evaluations.
    pub function_evals: usize,
    /// `info[8]`: number of Jacobian evaluations.
    pub jacobian_evals: usize,
    /// `info[9]`: number of linear systems solved (attempts to reduce the error).
    pub linear_solves: usize,
}

impl Info {
    /// The values as levmar's `info[]` array.
    pub fn to_array(&self) -> [f64; LM_INFO_SZ] {
        [
            self.initial_error,
            self.error,
            self.gradient_inf,
            self.step,
            self.mu_ratio,
            self.iterations as f64,
            f64::from(self.stop.code()),
            self.function_evals as f64,
            self.jacobian_evals as f64,
            self.linear_solves as f64,
        ]
    }

    /// The value `dlevmar_der` returns: the number of iterations, or `LM_ERROR` (-1) when the
    /// run stopped with a singular matrix (4) or invalid function values (7). In those cases
    /// `p` and the rest of the information are still set, as in the C.
    pub fn return_value(&self) -> i32 {
        match self.stop {
            StopReason::SingularMatrix | StopReason::InvalidValues => -1,
            _ => i32::try_from(self.iterations).unwrap_or(i32::MAX),
        }
    }
}

/// Errors for which `dlevmar_der` returns `LM_ERROR` before doing any work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LevmarError {
    /// Fewer measurements than unknowns (`n < m`).
    TooFewMeasurements {
        /// Number of parameters.
        m: usize,
        /// Number of measurements.
        n: usize,
    },
    /// The measurement vector is shorter than `n`. (The C cannot check this.)
    MeasurementsTooShort {
        /// Number of measurements.
        n: usize,
        /// Length of the given vector.
        len: usize,
    },
}

impl fmt::Display for LevmarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LevmarError::TooFewMeasurements { m, n } => write!(
                f,
                "dlevmar_der(): cannot solve a problem with fewer measurements [{n}] than unknowns [{m}]"
            ),
            LevmarError::MeasurementsTooShort { n, len } => write!(
                f,
                "dlevmar_der(): measurement vector has {len} values, {n} needed"
            ),
        }
    }
}

impl std::error::Error for LevmarError {}

/// Minimises `||x - func(p)||^2` over `p` by Levenberg-Marquardt with the analytic Jacobian
/// `jacf`.
///
/// - `func(p, hx)` writes the `n` model values for the parameters `p` (length `m`) into `hx`.
/// - `jacf(p, jac)` writes the `n x m` Jacobian of `func`, row-major, into `jac`.
/// - `p` holds the initial parameters and receives the result; its length is `m`.
/// - `x` is the measurement vector; only its first `n` values are read. `None` means zeros,
///   as levmar's `NULL`.
/// - `opts` are `opts[0..4]`; `None` means levmar's defaults ([`Options::DEFAULT`]).
///
/// Returns levmar's `info[]` ([`Info`]); [`Info::return_value`] gives the C return value.
/// The C's `work` and `covar` arguments are not ported: work memory is always allocated (as
/// with `work == NULL`) and no covariance is computed (as with `covar == NULL`), which is how
/// XMALab calls it. The C's `adata` is whatever the closures capture.
///
/// Ported from levmar 2.6 `lm_core.c`, `LEVMAR_DER` (`dlevmar_der`, double precision via
/// `lm.c`, built with `HAVE_LAPACK`).
pub fn dlevmar_der<F, J>(
    mut func: F,
    mut jacf: J,
    p: &mut [f64],
    x: Option<&[f64]>,
    n: usize,
    itmax: usize,
    opts: Option<&Options>,
) -> Result<Info, LevmarError>
where
    F: FnMut(&[f64], &mut [f64]),
    J: FnMut(&[f64], &mut [f64]),
{
    let m = p.len();
    if n < m {
        return Err(LevmarError::TooFewMeasurements { m, n });
    }
    let x = match x {
        Some(x) if x.len() < n => {
            return Err(LevmarError::MeasurementsTooShort { n, len: x.len() });
        }
        Some(x) => Some(&x[..n]),
        None => None,
    };

    let (tau, eps1, eps2, eps2_sq, eps3) = match opts {
        Some(o) => (o.mu, o.eps1, o.eps2, o.eps2 * o.eps2, o.eps3),
        None => (
            LM_INIT_MU,
            LM_STOP_THRESH,
            LM_STOP_THRESH,
            LM_STOP_THRESH * LM_STOP_THRESH,
            LM_STOP_THRESH,
        ),
    };

    let nm = n * m;
    // Work arrays. Zeroed, where the C's malloc gives uninitialised memory; this only shows
    // in info[4] when no Jacobian is ever computed.
    let mut e = vec![0.0; n];
    let mut hx = vec![0.0; n];
    let mut jac_te = vec![0.0; m];
    let mut jac = vec![0.0; nm];
    let mut jac_tjac = vec![0.0; m * m];
    let mut dp = vec![0.0; m];
    let mut diag_jac_tjac = vec![0.0; m];
    let mut pdp = vec![0.0; m];
    let mut solver = BkWork::new(m);

    let mut mu = 0.0;
    let mut jac_te_inf = 0.0;
    let mut p_l2;
    let mut dp_l2 = f64::MAX;
    let mut nu: i32 = 2;
    let mut stop: i32 = 0;
    let mut njev = 0usize;
    let mut nlss = 0usize;

    // Compute e = x - f(p) and its squared L2 norm.
    func(p, &mut hx);
    let mut nfev = 1usize;
    e.copy_from_slice(&hx);
    let mut p_el2 = l2nrmxmy(x, &mut e);
    let init_p_el2 = p_el2;
    if !p_el2.is_finite() {
        stop = 7;
    }

    let mut k = 0usize;
    while k < itmax && stop == 0 {
        // p and e have been updated at a previous iteration.
        if p_el2 <= eps3 {
            // Error is small.
            stop = 6;
            break;
        }

        // Compute the Jacobian J at p, J^T J, J^T e, ||J^T e||_inf and ||p||^2.
        jacf(p, &mut jac);
        njev += 1;

        if nm < BLOCKSZ_SQ {
            // Small problem: lower triangle accumulated row by row of J, looping downwards.
            jac_tjac.fill(0.0);
            jac_te.fill(0.0);
            for l in (0..n).rev() {
                let jaclm = &jac[l * m..(l + 1) * m];
                for i in (0..m).rev() {
                    let alpha = jaclm[i];
                    let row = &mut jac_tjac[i * m..(i + 1) * m];
                    for j in (0..=i).rev() {
                        row[j] += jaclm[j] * alpha;
                    }
                    jac_te[i] += alpha * e[l];
                }
            }
            // Copy to the upper part.
            for i in (0..m).rev() {
                for j in i + 1..m {
                    jac_tjac[i * m + j] = jac_tjac[j * m + i];
                }
            }
        } else {
            // Large problem: J^T J by BLAS dgemm in the C, then J^T e row by row.
            trans_mat_mat_mult(&jac, &mut jac_tjac, n, m);
            jac_te.fill(0.0);
            for i in 0..n {
                let jacrow = &jac[i * m..(i + 1) * m];
                let tmp = e[i];
                for l in 0..m {
                    jac_te[l] += jacrow[l] * tmp;
                }
            }
        }

        // Compute ||J^T e||_inf and ||p||^2.
        p_l2 = 0.0;
        jac_te_inf = 0.0;
        for i in 0..m {
            let tmp = jac_te[i].abs();
            if jac_te_inf < tmp {
                jac_te_inf = tmp;
            }
            // Save diagonal entries so that augmentation can be cancelled later.
            diag_jac_tjac[i] = jac_tjac[i * m + i];
            p_l2 += p[i] * p[i];
        }

        // Check for convergence.
        if jac_te_inf <= eps1 {
            dp_l2 = 0.0; // No increment for p in this case.
            stop = 1;
            break;
        }

        // Compute the initial damping factor.
        if k == 0 {
            let mut tmp = -f64::MAX;
            for &d in &diag_jac_tjac {
                if d > tmp {
                    tmp = d; // Largest diagonal element.
                }
            }
            mu = tau * tmp;
        }

        // Determine the increment using adaptive damping.
        loop {
            // Augment the normal equations.
            for i in 0..m {
                jac_tjac[i * m + i] += mu;
            }

            // Solve the augmented equations.
            let issolved = ax_eq_b_bk(&jac_tjac, &jac_te, &mut dp, m, &mut solver);
            nlss += 1;

            if issolved {
                // Compute p's new estimate and ||Dp||^2.
                dp_l2 = 0.0;
                for i in 0..m {
                    let tmp = dp[i];
                    pdp[i] = p[i] + tmp;
                    dp_l2 += tmp * tmp;
                }

                if dp_l2 <= eps2_sq * p_l2 {
                    // Relative change in p is small, stop.
                    stop = 2;
                    break;
                }

                if dp_l2 >= (p_l2 + eps2) / (EPSILON * EPSILON) {
                    // Almost singular.
                    stop = 4;
                    break;
                }

                // Evaluate the function at p + Dp and compute ||e(p + Dp)||^2.
                func(&pdp, &mut hx);
                nfev += 1;
                let pdp_el2 = l2nrmxmy(x, &mut hx);
                if !pdp_el2.is_finite() {
                    // Sum of squares is not finite, most probably due to a user error.
                    stop = 7;
                    break;
                }

                let mut dl = 0.0;
                for i in 0..m {
                    dl += dp[i] * (mu * dp[i] + jac_te[i]);
                }

                let df = p_el2 - pdp_el2;

                if dl > 0.0 && df > 0.0 {
                    // Reduction in error, the increment is accepted.
                    let mut tmp = 2.0 * df / dl - 1.0;
                    tmp = 1.0 - tmp * tmp * tmp;
                    mu *= if tmp >= ONE_THIRD { tmp } else { ONE_THIRD };
                    nu = 2;

                    p.copy_from_slice(&pdp);
                    e.copy_from_slice(&hx);
                    p_el2 = pdp_el2;
                    break;
                }
            }

            // Either the linear system could not be solved or the error did not reduce; in
            // either case the increment is rejected.
            mu *= f64::from(nu);
            // The C shifts a signed int, which wraps to a negative value after 2^30.
            let nu2 = nu.wrapping_shl(1);
            if nu2 <= nu {
                // nu has wrapped around (overflowed).
                stop = 5;
                break;
            }
            nu = nu2;

            // Restore the diagonal J^T J entries.
            for i in 0..m {
                jac_tjac[i * m + i] = diag_jac_tjac[i];
            }
        }

        k += 1;
    }

    if k >= itmax {
        stop = 3;
    }

    // Restore the diagonal J^T J entries.
    for i in 0..m {
        jac_tjac[i * m + i] = diag_jac_tjac[i];
    }

    let mut tmp = -f64::MAX;
    for i in 0..m {
        if tmp < jac_tjac[i * m + i] {
            tmp = jac_tjac[i * m + i];
        }
    }

    Ok(Info {
        initial_error: init_p_el2,
        error: p_el2,
        gradient_inf: jac_te_inf,
        step: dp_l2,
        mu_ratio: mu / tmp,
        iterations: k,
        stop: StopReason::from_code(stop).unwrap_or(StopReason::MaxIterations),
        function_evals: nfev,
        jacobian_evals: njev,
        linear_solves: nlss,
    })
}

/// Replaces `y` by `e = x - y` (or `-y` when `x` is `None`) and returns `||e||_2^2`, summed in
/// the order of levmar's unrolled loop: blocks of eight from the top down into four partial
/// sums, then the remainder upwards, then `((sum0 + sum1) + sum2) + sum3`.
///
/// Ported from levmar 2.6 `misc_core.c`, `LEVMAR_L2NRMXMY`. The C also allows `e` distinct
/// from `y`; callers here copy first.
fn l2nrmxmy(x: Option<&[f64]>, y: &mut [f64]) -> f64 {
    let n = y.len();
    let mut sum = [0.0f64; 4];
    let diff = |y: &mut [f64], i: usize| -> f64 {
        let e = match x {
            Some(x) => x[i] - y[i],
            None => -y[i],
        };
        y[i] = e;
        e * e
    };

    let blockn = (n >> 3) << 3;
    // for(i=blockn-1; i>0; i-=blocksize)
    let mut i = blockn as isize - 1;
    while i > 0 {
        let top = i as usize;
        for off in 0..8 {
            let sq = diff(y, top - off);
            sum[off % 4] += sq;
        }
        i -= 8;
    }

    // The remainder, entered at `case n - blockn` of the C's fall-through switch.
    const ORDER: [usize; 7] = [0, 1, 2, 3, 0, 1, 2];
    let rest = n - blockn;
    for (t, &s) in ORDER[7 - rest..].iter().enumerate() {
        let sq = diff(y, blockn + t);
        sum[s] += sq;
    }

    sum[0] + sum[1] + sum[2] + sum[3]
}

/// `b = a^T a` for the `n x m` row-major matrix `a`; `b` is `m x m`.
///
/// Ported from levmar 2.6 `misc_core.c`, `LEVMAR_TRANS_MAT_MAT_MULT`, `HAVE_LAPACK` branch,
/// which calls `dgemm("N", "T", m, m, n, 1, a, m, a, m, 0, b, m)`. Computed as the reference
/// build's OpenBLAS 0.3.29 `dgemm` does it for six parameters: each entry is one fused
/// multiply-add chain over the rows of `a` in ascending order, starting from zero. This was
/// checked bit for bit against the library for `m = 6` and 171 to 6000 rows. For other `m`
/// OpenBLAS may use other kernels (and reference BLAS sums without fusing), so larger
/// problems such as bundle adjustment can differ in the last bits. Only problems with
/// `n*m >= 1024` come here.
///
/// The result is bit for bit that of the dense definition, `b[j*m + i]` = the chain
/// `acc = fma(a[l][i], a[l][j], acc)` over every row `l` in ascending order from `+0`
/// ([`trans_mat_mat_mult_dense_entry`]), but the work is done sparsely, which matters for the
/// bundle adjustment, whose rows hold 16 or 21 non-zeros out of hundreds of columns:
///
/// - Rows are taken in the outer loop; a row whose entries are all finite contributes only
///   to the pairs of its non-zero columns. A skipped product is an exact zero (both factors
///   finite, one of them zero), which leaves any accumulator unchanged except that it may
///   turn `-0` into `+0`; a later product of two non-zero factors gives the same result
///   whichever zero the accumulator held. So the sparse chain equals the dense one, except
///   that it may end as `-0` where the dense one ends as `+0` (it never ends as `+0` where
///   the dense one ends as `-0`). An entry that ends as `-0` is recomputed by the dense
///   chain. A row holding a NaN or an infinity is taken whole (`0 * inf` is NaN).
/// - Only the entries `i <= j` are accumulated; `b[i*m + j]` is mirrored from `b[j*m + i]`,
///   since `fma(x, y, c) == fma(y, x, c)` whenever no operand is NaN. When an entry ends as
///   NaN its mirror is recomputed by the dense chain with its own operand order, since the
///   NaN that `fma` propagates can depend on that order.
fn trans_mat_mat_mult(a: &[f64], b: &mut [f64], n: usize, m: usize) {
    b.fill(0.0);
    let mut cols: Vec<usize> = Vec::with_capacity(m);
    for l in 0..n {
        let row = &a[l * m..(l + 1) * m];
        cols.clear();
        if row.iter().all(|v| v.is_finite()) {
            cols.extend((0..m).filter(|&i| row[i] != 0.0));
        } else {
            cols.extend(0..m);
        }
        for (q, &j) in cols.iter().enumerate() {
            let rj = row[j];
            let bj = &mut b[j * m..(j + 1) * m];
            for &i in &cols[..=q] {
                bj[i] = row[i].mul_add(rj, bj[i]);
            }
        }
    }
    for j in 0..m {
        for i in 0..=j {
            let v = b[j * m + i];
            if v == 0.0 && v.is_sign_negative() {
                let d = trans_mat_mat_mult_dense_entry(a, n, m, i, j);
                b[j * m + i] = d;
                b[i * m + j] = d;
            } else if v.is_nan() {
                b[i * m + j] = trans_mat_mat_mult_dense_entry(a, n, m, j, i);
            } else {
                b[i * m + j] = v;
            }
        }
    }
}

/// One entry of [`trans_mat_mat_mult`] by its dense definition: `b[j*m + i]`, the chain
/// `acc = fma(a[l][i], a[l][j], acc)` over every row `l` in ascending order from `+0`.
fn trans_mat_mat_mult_dense_entry(a: &[f64], n: usize, m: usize, i: usize, j: usize) -> f64 {
    let mut acc = 0.0f64;
    for l in 0..n {
        let row = &a[l * m..(l + 1) * m];
        acc = row[i].mul_add(row[j], acc);
    }
    acc
}

/// Scratch memory for [`ax_eq_b_bk`]: the factorised matrix and the pivots.
struct BkWork {
    a: Vec<f64>,
    ipiv: Vec<isize>,
}

impl BkWork {
    fn new(m: usize) -> Self {
        BkWork {
            a: vec![0.0; m * m],
            ipiv: vec![0; m],
        }
    }
}

/// Solves `A x = b` for the symmetric `m x m` matrix `A` by Bunch-Kaufman `L D L^T`
/// factorisation of its lower triangle (in LAPACK's column-major sense; `A` is symmetric so
/// the storage order does not matter). Returns `false` if the block diagonal `D` is exactly
/// singular (`dsytrf` info > 0), leaving `x` holding `b`.
///
/// Ported from levmar 2.6 `Axb_core.c`, `AX_EQ_B_BK` (`dAx_eq_b_BK`), which calls LAPACK
/// `dsytrf("L")` and `dsytrs("L")`; see [`dsytf2_lower`] and [`dsytrs_lower`].
fn ax_eq_b_bk(a_in: &[f64], b: &[f64], x: &mut [f64], m: usize, work: &mut BkWork) -> bool {
    let a = &mut work.a;
    a.copy_from_slice(a_in);
    x.copy_from_slice(b);

    // dsytrf with order <= 64 is dsytf2 on the whole matrix.
    let info = dsytf2_lower(m, a, &mut work.ipiv);
    if info != 0 {
        return false;
    }
    dsytrs_lower(m, a, &work.ipiv, x);
    true
}

/// Index of the first element of largest absolute value among `len` elements of `a`
/// starting at `start` with stride `inc` (0-based offset from the first element).
///
/// Ported from BLAS `IDAMAX` (as OpenBLAS returns it for finite values: the first maximum).
fn idamax(a: &[f64], start: usize, len: usize, inc: usize) -> usize {
    let mut best = 0;
    let mut dmax = a[start].abs();
    for t in 1..len {
        let v = a[start + t * inc].abs();
        if v > dmax {
            best = t;
            dmax = v;
        }
    }
    best
}

/// Bunch-Kaufman factorisation `A = L D L^T` of the lower triangle of the column-major
/// `n x n` matrix `a`, in place. Pivots are written to `ipiv` with LAPACK's 1-based
/// convention (positive: 1x1 block, row and column `ipiv[k]` interchanged; negative: 2x2
/// block). Returns LAPACK's `info`: 0, or `k > 0` if `D(k, k)` is exactly zero.
///
/// Ported from LAPACK `DSYTF2`, `UPLO = 'L'`, in the version inside the reference build's
/// OpenBLAS 0.3.29 (checked against its disassembly): `D11 = 1/A(K,K)` with no small-pivot
/// branch, and the 2x2 update with `D21 = T/D21` folded into `WK` and `WKP1`.
fn dsytf2_lower(n: usize, a: &mut [f64], ipiv: &mut [isize]) -> usize {
    let idx = |i: usize, j: usize| i + j * n;
    // ALPHA = ( ONE+SQRT( SEVTEN ) ) / EIGHT
    let alpha = (1.0 + 17.0f64.sqrt()) / 8.0;
    let mut info = 0;

    let mut k = 0;
    while k < n {
        let mut kstep = 1;
        let kp;

        // Determine rows and columns to be interchanged and whether a 1x1 or 2x2 pivot block
        // will be used.
        let absakk = a[idx(k, k)].abs();

        // IMAX is the row index of the largest off-diagonal element in column K, and COLMAX
        // is its absolute value.
        let (imax, colmax) = if k + 1 < n {
            let imax = k + 1 + idamax(a, idx(k + 1, k), n - k - 1, 1);
            (imax, a[idx(imax, k)].abs())
        } else {
            (k, 0.0)
        };

        if absakk.max(colmax) == 0.0 || absakk.is_nan() {
            // Column K is zero or contains a NaN: set INFO and continue.
            if info == 0 {
                info = k + 1;
            }
            kp = k;
        } else {
            if absakk >= alpha * colmax {
                // No interchange, use 1x1 pivot block.
                kp = k;
            } else {
                // JMAX is the column index of the largest off-diagonal element in row IMAX,
                // and ROWMAX is its absolute value.
                let jmax = k + idamax(a, idx(imax, k), imax - k, n);
                let mut rowmax = a[idx(imax, jmax)].abs();
                if imax + 1 < n {
                    let jmax = imax + 1 + idamax(a, idx(imax + 1, imax), n - imax - 1, 1);
                    rowmax = rowmax.max(a[idx(jmax, imax)].abs());
                }

                if absakk >= alpha * colmax * (colmax / rowmax) {
                    // No interchange, use 1x1 pivot block.
                    kp = k;
                } else if a[idx(imax, imax)].abs() >= alpha * rowmax {
                    // Interchange rows and columns K and IMAX, use 1x1 pivot block.
                    kp = imax;
                } else {
                    // Interchange rows and columns K+1 and IMAX, use 2x2 pivot block.
                    kp = imax;
                    kstep = 2;
                }
            }

            let kk = k + kstep - 1;
            if kp != kk {
                // Interchange rows and columns KK and KP in the trailing submatrix A(k:n,k:n).
                for i in kp + 1..n {
                    a.swap(idx(i, kk), idx(i, kp));
                }
                for t in kk + 1..kp {
                    a.swap(idx(t, kk), idx(kp, t));
                }
                a.swap(idx(kk, kk), idx(kp, kp));
                if kstep == 2 {
                    a.swap(idx(k + 1, k), idx(kp, k));
                }
            }

            // Update the trailing submatrix.
            if kstep == 1 {
                // 1x1 pivot block D(k): A := A - L(k)*D(k)*L(k)**T.
                if k + 1 < n {
                    let d11 = 1.0 / a[idx(k, k)];
                    dsyr_lower(n, a, k + 1, k, -d11);
                    // DSCAL( N-K, D11, A( K+1, K ), 1 )
                    for i in k + 1..n {
                        a[idx(i, k)] *= d11;
                    }
                }
            } else if k + 2 < n {
                // 2x2 pivot block D(k): A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**T.
                let mut d21 = a[idx(k + 1, k)];
                let d11 = a[idx(k + 1, k + 1)] / d21;
                let d22 = a[idx(k, k)] / d21;
                let t = 1.0 / (d11 * d22 - 1.0);
                d21 = t / d21;

                for j in k + 2..n {
                    let wk = d21 * (d11 * a[idx(j, k)] - a[idx(j, k + 1)]);
                    let wkp1 = d21 * (d22 * a[idx(j, k + 1)] - a[idx(j, k)]);
                    for i in j..n {
                        a[idx(i, j)] = a[idx(i, j)] - a[idx(i, k)] * wk - a[idx(i, k + 1)] * wkp1;
                    }
                    a[idx(j, k)] = wk;
                    a[idx(j, k + 1)] = wkp1;
                }
            }
        }

        // Store details of the interchanges in IPIV.
        let kp1 = kp as isize + 1;
        if kstep == 1 {
            ipiv[k] = kp1;
        } else {
            ipiv[k] = -kp1;
            ipiv[k + 1] = -kp1;
        }

        k += kstep;
    }
    info
}

/// `A(s:n, s:n) += alpha * x x^T` on the lower triangle, where `x = A(s:n, c)` (column-major,
/// order `n`), as `dsytf2` calls `DSYR( 'L', N-K, -D11, A( K+1, K ), 1, A( K+1, K+1 ), LDA )`.
///
/// Ported from BLAS `DSYR` (lower) as OpenBLAS 0.3.29 computes it for orders below 100: column
/// by column, `temp = alpha * x(j)` (skipped when `x(j)` is zero), then
/// `A(i, j) = fma(temp, x(i), A(i, j))` (its `daxpy` kernel fuses the multiply-add).
fn dsyr_lower(n: usize, a: &mut [f64], s: usize, c: usize, alpha: f64) {
    let idx = |i: usize, j: usize| i + j * n;
    for j in s..n {
        let xj = a[idx(j, c)];
        if xj != 0.0 {
            let temp = alpha * xj;
            for i in j..n {
                a[idx(i, j)] = temp.mul_add(a[idx(i, c)], a[idx(i, j)]);
            }
        }
    }
}

/// `b(i) -= b(k) * A(i, c)` for `i` in `s..n`: `DGER( N-S, 1, -ONE, A( S, C ), 1, B( K ),
/// LDB, B( S ), LDB )` with one right-hand side.
///
/// Ported from BLAS `DGER` as OpenBLAS 0.3.29 computes it: `y0 = alpha * y`, then
/// `b(i) = fma(y0, x(i), b(i))`.
fn dger_column(n: usize, a: &[f64], c: usize, s: usize, b: &mut [f64], k: usize) {
    let y0 = -b[k];
    for i in s..n {
        b[i] = y0.mul_add(a[i + c * n], b[i]);
    }
}

/// `b(k) -= sum_{i in s..n} b(i) * A(i, c)`: `DGEMV( 'Transpose', N-S, 1, -ONE, B( S ), LDB,
/// A( S, C ), 1, ONE, B( K ), LDB )` with one right-hand side.
///
/// Ported from BLAS `DGEMV` ('T') as OpenBLAS 0.3.29 computes it for one column
/// (`kernel/x86_64/dgemv_t_4.c`): the rows in whole blocks of four are multiplied lane by
/// lane, the lanes summed over the blocks, reduced as `(l0 + l2) + (l1 + l3)` and subtracted
/// from `b(k)`; then the last one to three rows are added with fused multiply-adds against
/// `-A(i, c)`. Checked bit for bit against the library for 1 to 40 rows.
fn dgemv_t_column(n: usize, a: &[f64], c: usize, s: usize, b: &mut [f64], k: usize) {
    let len = n - s;
    let col = &a[s + c * n..(c + 1) * n];
    let rows = &b[s..n];
    let m1 = len & !3;
    let mut y = b[k];

    if m1 > 0 {
        let mut lanes = [0.0f64; 4];
        for blk in 0..m1 / 4 {
            for (j, lane) in lanes.iter_mut().enumerate() {
                let r = blk * 4 + j;
                let prod = rows[r] * col[r];
                *lane = if blk == 0 { prod } else { *lane + prod };
            }
        }
        let t = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
        y -= t;
    }

    let r = &rows[m1..];
    let xt: Vec<f64> = col[m1..].iter().map(|&v| -v).collect();
    match len - m1 {
        3 => {
            let e = r[2].mul_add(xt[2], r[0].mul_add(xt[0], r[1] * xt[1]));
            y += e;
        }
        2 => {
            let e = r[0].mul_add(xt[0], r[1] * xt[1]);
            y += e;
        }
        1 => {
            y = r[0].mul_add(xt[0], y);
        }
        _ => {}
    }
    b[k] = y;
}

/// Solves `A x = b` with the factorisation from [`dsytf2_lower`], one right-hand side, in
/// place in `b`.
///
/// Ported from LAPACK `DSYTRS`, `UPLO = 'L'`, `NRHS = 1`. The BLAS calls are
/// [`dger_column`], `DSCAL` (a multiplication by `ONE / A(K,K)`), `DSWAP` and
/// [`dgemv_t_column`].
fn dsytrs_lower(n: usize, a: &[f64], ipiv: &[isize], b: &mut [f64]) {
    let idx = |i: usize, j: usize| i + j * n;

    // First solve L*D*X = B, overwriting B with X. K increases from 1 to N.
    let mut k = 0;
    while k < n {
        if ipiv[k] > 0 {
            // 1x1 diagonal block: interchange rows K and IPIV(K).
            let kp = (ipiv[k] - 1) as usize;
            if kp != k {
                b.swap(k, kp);
            }
            // Multiply by inv(L(K)), where L(K) is the transformation stored in column K of A.
            if k + 1 < n {
                dger_column(n, a, k, k + 1, b, k);
            }
            // Multiply by the inverse of the diagonal block.
            b[k] *= 1.0 / a[idx(k, k)];
            k += 1;
        } else {
            // 2x2 diagonal block: interchange rows K+1 and -IPIV(K).
            let kp = (-ipiv[k] - 1) as usize;
            if kp != k + 1 {
                b.swap(k + 1, kp);
            }
            // Multiply by inv(L(K)).
            if k + 2 < n {
                dger_column(n, a, k, k + 2, b, k);
                dger_column(n, a, k + 1, k + 2, b, k + 1);
            }
            // Multiply by the inverse of the diagonal block.
            let akm1k = a[idx(k + 1, k)];
            let akm1 = a[idx(k, k)] / akm1k;
            let ak = a[idx(k + 1, k + 1)] / akm1k;
            let denom = akm1 * ak - 1.0;
            let bkm1 = b[k] / akm1k;
            let bk = b[k + 1] / akm1k;
            b[k] = (ak * bkm1 - bk) / denom;
            b[k + 1] = (akm1 * bk - bkm1) / denom;
            k += 2;
        }
    }

    // Next solve L**T *X = B, overwriting B with X. K decreases from N to 1.
    let mut k = n as isize - 1;
    while k >= 0 {
        let ku = k as usize;
        if ipiv[ku] > 0 {
            // 1x1 diagonal block: multiply by inv(L**T(K)).
            if ku + 1 < n {
                dgemv_t_column(n, a, ku, ku + 1, b, ku);
            }
            // Interchange rows K and IPIV(K).
            let kp = (ipiv[ku] - 1) as usize;
            if kp != ku {
                b.swap(ku, kp);
            }
            k -= 1;
        } else {
            // 2x2 diagonal block: multiply by inv(L**T(K-1)).
            if ku + 1 < n {
                dgemv_t_column(n, a, ku, ku + 1, b, ku);
                dgemv_t_column(n, a, ku - 1, ku + 1, b, ku - 1);
            }
            // Interchange rows K and -IPIV(K).
            let kp = (-ipiv[ku] - 1) as usize;
            if kp != ku {
                b.swap(ku, kp);
            }
            k -= 2;
        }
    }
}

/// Test access to the linear solver, so the integration tests can check it against
/// `dsytrf`/`dsytrs` directly. Not part of the supported interface.
#[doc(hidden)]
pub mod internals {
    /// Factorises and solves as levmar's `dAx_eq_b_BK`. Returns `(solved, x, factor, ipiv)`
    /// where `factor` and `ipiv` are what `dsytrf` leaves.
    pub fn bunch_kaufman(a: &[f64], b: &[f64]) -> (bool, Vec<f64>, Vec<f64>, Vec<isize>) {
        let m = b.len();
        let mut work = super::BkWork::new(m);
        let mut x = vec![0.0; m];
        let solved = super::ax_eq_b_bk(a, b, &mut x, m, &mut work);
        (solved, x, work.a, work.ipiv)
    }

    /// The largest `m` [`trans_mat_mat_mult`] accepts: its result of `m * m` values is then
    /// at most 8 MiB, whatever the arguments (the tests go to `m = 340`).
    pub const TRANS_MAT_MAT_MULT_MAX_M: usize = 1024;

    /// `J^T J` as `dlevmar_der` computes it for problems with `n*m >= 1024` (levmar's
    /// `LEVMAR_TRANS_MAT_MAT_MULT`): `a` is `n x m` row-major, the result `m x m`. Empty
    /// when `a` holds fewer than `n*m` values, `m` exceeds its length, or `m` exceeds
    /// [`TRANS_MAT_MAT_MULT_MAX_M`] (so that no argument value allocates without bound).
    pub fn trans_mat_mat_mult(a: &[f64], n: usize, m: usize) -> Vec<f64> {
        if m > TRANS_MAT_MAT_MULT_MAX_M
            || m > a.len()
            || n.checked_mul(m).is_none_or(|nm| nm > a.len())
        {
            return Vec::new();
        }
        let mut b = vec![0.0; m * m];
        super::trans_mat_mat_mult(&a[..n * m], &mut b, n, m);
        b
    }
}
