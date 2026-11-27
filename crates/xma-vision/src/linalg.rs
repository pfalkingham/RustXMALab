//! Dense linear algebra that behaves as OpenCV's does: singular value decomposition,
//! null vector, least-squares solve, inverse, determinant and index sorting.
//!
//! Work package: linear algebra. Ported from OpenCV 4.13 (`modules/core/src/lapack.cpp`,
//! `matrix_decomp.cpp`, `matrix_operations.cpp`, `include/opencv2/core/operations.hpp`).
//! Every other package that needs a decomposition uses this module, so results come from
//! one implementation and do not depend on the platform or on a system LAPACK.
//!
//! # The functions
//!
//! | Rust | OpenCV | XMALab call sites |
//! |---|---|---|
//! | [`svd`] | `cv::SVD::compute(a, w, u, vt, flags)` | `core/RigidBody.cpp:1038,1406,1857`, `processing/RigidBodyPoseFrom2D.cpp:199`, inside `cv::Rodrigues` |
//! | [`solve_z`] | `cv::SVD::solveZ` | `core/Marker.cpp:279,364`, `processing/Calibration.cpp:176`, `processing/CubeCalibration.cpp:832` |
//! | [`solve_svd`] | `cv::solve(.., DECOMP_SVD)` | `core/Marker.cpp:429,513,607` |
//! | [`solve_qr`] | `cv::solve(.., DECOMP_QR)` | `processing/MarkerDetection.cpp:539` |
//! | [`invert_svd`] | `cv::invert(.., DECOMP_SVD)` (pseudo-inverse) | `processing/LocalUndistortion.cpp:327` |
//! | [`invert`] | `cv::invert(.., DECOMP_LU)`, `Mat::inv()` | `core/Marker.cpp:574,1765,1942`, `core/Camera.cpp:585`, `core/CalibrationImage.cpp:400`, `core/RigidBody.cpp:1201`, `processing/ButterworthLowPassFilter.cpp:335`, `processing/RigidBodyPoseOptimization.cpp:189,190,239,240`, `processing/RigidBodyPoseFrom2D.cpp:334,341`, `processing/CubeCalibration.cpp:235`, `processing/MultiCameraCalibration.cpp:481,524,525` |
//! | [`determinant`] | `cv::determinant` | `core/RigidBody.cpp:1045,1046,1413,1414,1864,1865`, `processing/RigidBodyPoseFrom2D.cpp:206,207` |
//! | [`sort_idx`], [`sort_idx_matrix`] | `cv::sortIdx` | `processing/LocalUndistortion.cpp:301,547` (columns), `:605-679` (rows) |
//!
//! Every matrix in those call sites is `CV_64F`, so everything here is `f64`. `cv::eigen` is
//! not used by XMALab and is not provided.
//!
//! Inputs are any nalgebra matrix (`Matrix3`, `DMatrix`, a view, ...); outputs are
//! `DMatrix`/`DVector`. Shapes follow OpenCV exactly (for example [`solve_z`] of an `m x n`
//! matrix returns `n` values; the thin [`svd`] of a `3 x 5` matrix has `u` of `3 x 3` and
//! `vt` of `3 x 5`).
//!
//! # Which backend the reference numbers come from
//!
//! OpenCV hands some decompositions to LAPACK when it is built with LAPACK
//! (`modules/core/src/hal_internal.cpp`), but only above fixed sizes:
//!
//! | Routine | LAPACK used when | LAPACK routine |
//! |---|---|---|
//! | SVD (all of [`svd`], [`solve_z`], [`solve_svd`], [`invert_svd`]) | `max(rows, cols) >= 25` | `dgesdd` |
//! | QR ([`solve_qr`]) | `rows >= 30` | `dgels` |
//! | LU ([`invert`] for `n >= 4`, [`determinant`] for `n >= 4`) | `n >= 100` | `dgesv` / `dgetrf` |
//!
//! Both references are built with LAPACK: the reference C++ build links Fedora's OpenCV 4.13.0
//! (`opencv_version -v` in the `xmalabDev` container: `Lapack: YES (/usr/lib64/libflexiblas.so)`,
//! and `libopencv_core.so.4.13.0` links `libflexiblas.so.3`), and the primitive oracle was
//! made with pip `cv2` 4.14.0 (`Lapack: YES (/usr/lib64/libopenblasp.so)`, IPP on).
//!
//! This module always runs OpenCV's own code (`JacobiSVDImpl_`, `QRImpl`, `LUImpl`), at every
//! size. Consequences:
//!
//! - Below the thresholds (every triangulation, every rigid-body fit, the LWM pseudo-inverse
//!   at its default size, every `inv()` in XMALab, DLTs with at most 12 points) the
//!   reference uses the same code and the results are bit-identical. This was checked
//!   against the real OpenCV 4.13 on several thousand inputs, including NaN, infinities,
//!   denormals and signed zeros; a subset is the committed fixture `tests/data/linalg/`,
//!   which `tests/linalg.rs` compares bit for bit.
//! - At or above them the reference uses LAPACK through FlexiBLAS/OpenBLAS, whose results
//!   cannot be reproduced bit for bit:
//!   - **SVD, full-rank input with `rows >= cols`**: singular values, solved `x` and
//!     pseudo-inverses agree to rounding (about 1e-15 relative on well-conditioned input),
//!     but singular vectors may differ in sign, and in basis for repeated singular values.
//!     XMALab's DLT (`processing/Calibration.cpp:176`, `processing/CubeCalibration.cpp:832`)
//!     has `rows >= cols` (`2N x 12`); with 13 or more points (`2N >= 26`) its null vector
//!     may come back negated, nothing worse.
//!   - **SVD, `rows < cols`**: the null space has dimension `cols - rows` or more, and
//!     [`solve_z`] returns OpenCV's random-completed vector while `dgesdd` returns some
//!     other vector of that space: an arbitrary, unrelated null vector, not just a sign flip
//!     (measured at 4 x 25: difference about 0.93). The extra vectors of a
//!     [`SvdMode::Full`] decomposition differ the same way. No XMALab call site has this
//!     shape at that size.
//!   - **QR ([`solve_qr`]) with 30 or more rows**: OpenCV uses `dgels`
//!     (`hal_internal.cpp:330-445`). On full-rank, well-scaled input `x` agrees to rounding.
//!     But `dgels` has no `100 * DBL_EPSILON` test: it fails only for an exactly zero
//!     diagonal element of R. So on rank-deficient, tiny or zero-column input the two
//!     disagree completely (measured at 30 x 15; see [`solve_qr`]). This concerns
//!     `processing/MarkerDetection.cpp:539` (`N x 15`, `N` usually 49 or more), which reads
//!     the result without checking it. Recorded in the parity ledger (A37); the behaviour
//!     here is OpenCV's own `QRImpl`, and the choice is made when detection is ported.
//!
//! ## Fit sizes
//!
//! The thresholds are crossed by settings, not only by data. The LWM pseudo-inverse
//! (`processing/LocalUndistortion.cpp:327`) is `6 x LocalUndistortionNeighbours`, 12 by
//! default (`core/Settings.cpp:73`); a settings file that sets 25 or more moves the
//! reference onto LAPACK `dgesdd` for it. Triangulation is `2 * cameras x 4` (or `x 3`),
//! so it crosses only with 13 or more cameras; the DLT crosses with 13 or more points.
//!
//! # Floating point
//!
//! The port follows OpenCV operation by operation and never fuses a multiply and an add.
//! OpenCV's `lapack.cpp` is compiled for baseline x86-64 (SSE2), which has no FMA, and its
//! SIMD paths (`VBLAS<double>::givens`, the 2x2 inverse) compute the same IEEE results as the
//! scalar code they replace, so the Jacobi SVD here is bit-identical to OpenCV's.

// The ports keep OpenCV's index loops so they can be checked line by line against the C++.
#![allow(clippy::needless_range_loop)]

use nalgebra::{DMatrix, DVector, Dim, Matrix, RawStorage};

/// `DBL_EPSILON`.
const DBL_EPSILON: f64 = f64::EPSILON;
/// `DBL_MIN`, the smallest positive normal double.
const DBL_MIN: f64 = f64::MIN_POSITIVE;

// ---------------------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------------------

/// How many singular vectors [`svd`] returns: OpenCV's `flags` argument of `SVD::compute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvdMode {
    /// `flags = 0`: `u` is `m x min(m, n)`, `vt` is `min(m, n) x n`.
    #[default]
    Thin,
    /// `SVD::FULL_UV`: `u` is `m x m` and `vt` is `n x n`. The extra vectors (and those of
    /// zero singular values) are made by OpenCV's seeded random-vector Gram-Schmidt, which
    /// is ported exactly.
    Full,
}

/// Result of [`svd`]: `a = u * diag(w) * vt` (using the first `min(m, n)` columns of `u`
/// and rows of `vt`).
#[derive(Debug, Clone, PartialEq)]
pub struct Svd {
    /// Singular values, `min(m, n)` of them, in descending order.
    pub w: DVector<f64>,
    /// Left singular vectors as columns.
    pub u: DMatrix<f64>,
    /// Right singular vectors as rows (V transposed).
    pub vt: DMatrix<f64>,
}

/// Result of a solve or an inverse that can fail, as OpenCV returns it: the matrix plus the
/// `bool` return value. When `ok` is false, `x` is all zeros (OpenCV sets `dst = Scalar(0)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Solved {
    /// The solution or inverse; zeros when `ok` is false.
    pub x: DMatrix<f64>,
    /// OpenCV's return value.
    pub ok: bool,
}

/// Direction for [`sort_idx_matrix`]: OpenCV's `SORT_EVERY_ROW` / `SORT_EVERY_COLUMN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    /// `SORT_EVERY_ROW`: each row is sorted on its own.
    EveryRow,
    /// `SORT_EVERY_COLUMN`: each column is sorted on its own.
    EveryColumn,
}

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

/// Copies any nalgebra matrix into a row-major vector, returning `(rows, cols, data)`.
fn to_row_major<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
    a: &Matrix<f64, R, C, S>,
) -> (usize, usize, Vec<f64>) {
    let (m, n) = a.shape();
    let mut v = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            v.push(a[(i, j)]);
        }
    }
    (m, n, v)
}

/// Builds a `DMatrix` from row-major data.
fn from_row_major(rows: usize, cols: usize, data: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, cols, data)
}

/// `cv::RNG` (multiply-with-carry), as used by `JacobiSVDImpl_` for zero singular values.
///
/// Ported from `modules/core/include/opencv2/core/operations.hpp`, `RNG::next`.
struct CvRng {
    state: u64,
}

impl CvRng {
    /// `CV_RNG_COEFF`.
    const COEFF: u64 = 4_164_903_690;

    fn new(seed: u64) -> Self {
        CvRng { state: seed }
    }

    /// `state = (uint64)(unsigned)state*CV_RNG_COEFF + (unsigned)(state >> 32)`.
    fn next(&mut self) -> u32 {
        self.state = (self.state & 0xffff_ffff)
            .wrapping_mul(Self::COEFF)
            .wrapping_add(self.state >> 32);
        self.state as u32
    }
}

/// OpenCV's own `hypot` (not `libm`'s).
///
/// Ported from `modules/core/src/lapack.cpp`, `hypot`.
fn cv_hypot(a: f64, b: f64) -> f64 {
    let mut a = a.abs();
    let mut b = b.abs();
    if a > b {
        b /= a;
        return a * (1.0 + b * b).sqrt();
    }
    if b > 0.0 {
        a /= b;
        return b * (1.0 + a * a).sqrt();
    }
    0.0
}

/// One-sided Jacobi SVD on the rows of `at` (`n` rows of length `m`, row-major, `m >= n`
/// for the callers). On return `w` holds the `n` singular values in descending order,
/// rows `0..n1` of `at` hold the left singular vectors (rows `n..n1` must be allocated, and
/// are produced by the random Gram-Schmidt), and `vt` (if given, `n x n`) the right ones.
///
/// Ported from `modules/core/src/lapack.cpp`, `JacobiSVDImpl_<double>`, called as
/// `hal::SVD64f` does: `minval = DBL_MIN`, `eps = DBL_EPSILON * 10`.
fn jacobi_svd(
    at: &mut [f64],
    w_out: &mut [f64],
    vt: Option<&mut [f64]>,
    m: usize,
    n: usize,
    n1: usize,
) {
    let minval = DBL_MIN;
    let eps = DBL_EPSILON * 10.0;
    let max_iter = m.max(30);
    let mut w = vec![0.0f64; n];
    let mut vt = vt;

    for i in 0..n {
        let mut sd = 0.0f64;
        for k in 0..m {
            let t = at[i * m + k];
            sd += t * t;
        }
        w[i] = sd;

        if let Some(v) = vt.as_deref_mut() {
            for k in 0..n {
                v[i * n + k] = 0.0;
            }
            v[i * n + i] = 1.0;
        }
    }

    for _iter in 0..max_iter {
        let mut changed = false;

        for i in 0..n.saturating_sub(1) {
            for j in (i + 1)..n {
                let mut a = w[i];
                let mut p = 0.0f64;
                let mut b = w[j];

                for k in 0..m {
                    p += at[i * m + k] * at[j * m + k];
                }

                if p.abs() <= eps * (a * b).sqrt() {
                    continue;
                }

                p *= 2.0;
                let beta = a - b;
                let gamma = cv_hypot(p, beta);
                let (c, s);
                if beta < 0.0 {
                    let delta = (gamma - beta) * 0.5;
                    s = (delta / gamma).sqrt();
                    c = p / (gamma * s * 2.0);
                } else {
                    c = ((gamma + beta) / (gamma * 2.0)).sqrt();
                    s = p / (gamma * c * 2.0);
                }

                a = 0.0;
                b = 0.0;
                for k in 0..m {
                    let ai = at[i * m + k];
                    let aj = at[j * m + k];
                    let t0 = c * ai + s * aj;
                    let t1 = -s * ai + c * aj;
                    at[i * m + k] = t0;
                    at[j * m + k] = t1;

                    a += t0 * t0;
                    b += t1 * t1;
                }
                w[i] = a;
                w[j] = b;

                changed = true;

                if let Some(v) = vt.as_deref_mut() {
                    // VBLAS<double>::givens computes Vi*c + Vj*s and Vj*c - Vi*s, which are
                    // the same IEEE results as the scalar loop below.
                    for k in 0..n {
                        let vi = v[i * n + k];
                        let vj = v[j * n + k];
                        let t0 = c * vi + s * vj;
                        let t1 = -s * vi + c * vj;
                        v[i * n + k] = t0;
                        v[j * n + k] = t1;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    for i in 0..n {
        let mut sd = 0.0f64;
        for k in 0..m {
            let t = at[i * m + k];
            sd += t * t;
        }
        w[i] = sd.sqrt();
    }

    for i in 0..n.saturating_sub(1) {
        let mut j = i;
        for k in (i + 1)..n {
            if w[j] < w[k] {
                j = k;
            }
        }
        if i != j {
            w.swap(i, j);
            if let Some(v) = vt.as_deref_mut() {
                for k in 0..m {
                    at.swap(i * m + k, j * m + k);
                }
                for k in 0..n {
                    v.swap(i * n + k, j * n + k);
                }
            }
        }
    }

    w_out[..n].copy_from_slice(&w);

    if vt.is_none() {
        return;
    }

    let mut rng = CvRng::new(0x1234_5678);
    for i in 0..n1 {
        let mut sd = if i < n { w[i] } else { 0.0 };

        let mut ii = 0;
        while ii < 100 && sd <= minval {
            // A zero singular value: make the left singular vector from a random vector,
            // orthogonalised against the previous ones and normalised.
            let val0 = 1.0 / m as f64;
            for k in 0..m {
                let val = if (rng.next() & 256) != 0 { val0 } else { -val0 };
                at[i * m + k] = val;
            }
            for _iter in 0..2 {
                for j in 0..i {
                    sd = 0.0;
                    for k in 0..m {
                        sd += at[i * m + k] * at[j * m + k];
                    }
                    let mut asum = 0.0f64;
                    for k in 0..m {
                        let t = at[i * m + k] - sd * at[j * m + k];
                        at[i * m + k] = t;
                        asum += t.abs();
                    }
                    asum = if asum > eps * 100.0 { 1.0 / asum } else { 0.0 };
                    for k in 0..m {
                        at[i * m + k] *= asum;
                    }
                }
            }
            sd = 0.0;
            for k in 0..m {
                let t = at[i * m + k];
                sd += t * t;
            }
            sd = sd.sqrt();
            ii += 1;
        }

        let s = if sd > minval { 1.0 / sd } else { 0.0 };
        for k in 0..m {
            at[i * m + k] *= s;
        }
    }
}

/// `y[0:m, 0:n] += diag(a[0:m]) * x[0:m, 0:n]`, with strides as in OpenCV (a stride of 0
/// reuses the same row).
///
/// Ported from `modules/core/src/lapack.cpp`, `MatrAXPY`.
#[allow(clippy::too_many_arguments)]
fn matr_axpy(
    m: usize,
    n: usize,
    x: &[f64],
    dx: usize,
    a: &[f64],
    a_off: usize,
    inca: usize,
    y: &mut [f64],
    dy: usize,
) {
    for i in 0..m {
        let s = a[a_off + i * inca];
        let xr = i * dx;
        let yr = i * dy;
        for j in 0..n {
            y[yr + j] += s * x[xr + j];
        }
    }
}

/// Back substitution with an SVD: `x = V * diag(1/w) * U^T * b`, skipping singular values
/// `<= sum(w) * 2 * DBL_EPSILON`. `u` is read as `u(row j, col i) = u[j*udelta1 + i*udelta0]`
/// and `v` as `v(j, i) = v[j*vdelta1 + i*vdelta0]`. `b` is `m x nb` row-major or `None`
/// (identity, giving the pseudo-inverse). Returns `x`, `n x nb` row-major.
///
/// Ported from `modules/core/src/lapack.cpp`, `SVBkSbImpl_<double>` (with
/// `eps = DBL_EPSILON * 2`, as `SVBkSb` passes).
#[allow(clippy::too_many_arguments)]
fn sv_bk_sb(
    m: usize,
    n: usize,
    w: &[f64],
    u: &[f64],
    udelta0: usize,
    udelta1: usize,
    v: &[f64],
    vdelta0: usize,
    vdelta1: usize,
    b: Option<&[f64]>,
    nb: usize,
) -> Vec<f64> {
    let eps = DBL_EPSILON * 2.0;
    let nm = m.min(n);
    let nb = if b.is_none() { m } else { nb };
    let ldb = nb;
    let ldx = nb;
    let mut x = vec![0.0f64; n * nb];
    let mut buffer = vec![0.0f64; nb];

    let mut threshold = 0.0f64;
    for &wi in w.iter().take(nm) {
        threshold += wi;
    }
    threshold *= eps;

    for i in 0..nm {
        let uo = i * udelta0;
        let vo = i * vdelta0;
        let mut wi = w[i];
        if wi.abs() <= threshold {
            continue;
        }
        wi = 1.0 / wi;

        if nb == 1 {
            let mut s = 0.0f64;
            if let Some(b) = b {
                for j in 0..m {
                    s += u[uo + j * udelta1] * b[j * ldb];
                }
            } else {
                s = u[uo];
            }
            s *= wi;

            for j in 0..n {
                x[j * ldx] += s * v[vo + j * vdelta1];
            }
        } else {
            if let Some(b) = b {
                buffer.iter_mut().for_each(|e| *e = 0.0);
                matr_axpy(m, nb, b, ldb, u, uo, udelta1, &mut buffer, 0);
                for e in buffer.iter_mut() {
                    *e *= wi;
                }
            } else {
                for j in 0..nb {
                    buffer[j] = u[uo + j * udelta1] * wi;
                }
            }
            matr_axpy(n, nb, &buffer, 0, v, vo, vdelta1, &mut x, ldx);
        }
    }
    x
}

/// Shared by [`svd`] and [`invert_svd`]: `_SVDcompute`, returning `(w, u, vt)` as row-major
/// vectors with their shapes: `u` is `rows x ucols`, `vt` is `vrows x cols`.
struct RawSvd {
    w: Vec<f64>,
    u: Vec<f64>,
    ucols: usize,
    vt: Vec<f64>,
    vrows: usize,
}

/// Ported from `modules/core/src/lapack.cpp`, `_SVDcompute` (with `u` and `vt` requested).
fn svd_compute(rows: usize, cols: usize, src: &[f64], full_uv: bool) -> RawSvd {
    let (mut m, mut n) = (rows, cols);
    let at = m < n;
    if at {
        std::mem::swap(&mut m, &mut n);
    }
    let urows = if full_uv { m } else { n };

    // temp_a (n x m) and temp_u (urows x m) share one buffer; rows past n start at zero.
    let mut temp_u = vec![0.0f64; urows * m];
    if !at {
        // transpose(src, temp_a): temp_a(i, k) = src(k, i)
        for i in 0..n {
            for k in 0..m {
                temp_u[i * m + k] = src[k * cols + i];
            }
        }
    } else {
        temp_u[..n * m].copy_from_slice(&src[..n * m]);
    }
    let mut temp_w = vec![0.0f64; n];
    let mut temp_v = vec![0.0f64; n * n];

    jacobi_svd(&mut temp_u, &mut temp_w, Some(&mut temp_v), m, n, urows);

    if !at {
        // u = transpose(temp_u): rows x urows; vt = temp_v: n x n.
        let mut u = vec![0.0f64; m * urows];
        for r in 0..m {
            for c in 0..urows {
                u[r * urows + c] = temp_u[c * m + r];
            }
        }
        RawSvd {
            w: temp_w,
            u,
            ucols: urows,
            vt: temp_v,
            vrows: n,
        }
    } else {
        // u = transpose(temp_v): n x n; vt = temp_u: urows x m.
        let mut u = vec![0.0f64; n * n];
        for r in 0..n {
            for c in 0..n {
                u[r * n + c] = temp_v[c * n + r];
            }
        }
        RawSvd {
            w: temp_w,
            u,
            ucols: n,
            vt: temp_u,
            vrows: urows,
        }
    }
}

// ---------------------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------------------

/// Singular value decomposition `a = u * diag(w) * vt`, with OpenCV's one-sided Jacobi
/// method and its ordering and sign conventions.
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::SVD::compute` / `_SVDcompute` and
/// `JacobiSVDImpl_`. Serves `core/RigidBody.cpp:1038,1406,1857` and
/// `processing/RigidBodyPoseFrom2D.cpp:199` (`svd.compute(K, W, U, VT)` on 3x3, flags 0) and
/// the 3x3 orthonormalisation inside `cv::Rodrigues`.
///
/// - Singular values are in descending order (ties keep the first index found).
/// - For `m < n` OpenCV decomposes `a^T`, so the Jacobi rotations act on the rows of `a`.
/// - Signs are whatever the Jacobi rotations produce from the identity; they are not
///   normalised.
/// - Zero singular values (`<= DBL_MIN`) and, in [`SvdMode::Full`], the extra vectors get
///   OpenCV's seeded random vectors (`cv::RNG(0x12345678)`), orthogonalised: deterministic.
///
/// Size limit for parity: when `max(rows, cols) >= 25`, OpenCV built with LAPACK uses
/// `dgesdd` instead (see the module docs); then only `w` and `u * diag(w) * vt` are
/// comparable, not the signs of the vectors.
///
/// Panics if `a` is empty.
pub fn svd<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
    a: &Matrix<f64, R, C, S>,
    mode: SvdMode,
) -> Svd {
    let (rows, cols, data) = to_row_major(a);
    assert!(rows > 0 && cols > 0, "svd of an empty matrix");
    let r = svd_compute(rows, cols, &data, mode == SvdMode::Full);
    Svd {
        w: DVector::from_vec(r.w),
        u: from_row_major(rows, r.ucols, &r.u),
        vt: from_row_major(r.vrows, cols, &r.vt),
    }
}

/// The unit vector `z` minimising `|a z|`: the last row of `vt` from [`svd`], with
/// [`SvdMode::Thin`] when `rows >= cols` and [`SvdMode::Full`] otherwise. Returns `cols`
/// values. Its sign is the Jacobi SVD's and is not normalised.
///
/// Ported from `modules/core/include/opencv2/core/operations.hpp`, `cv::SVD::solveZ`.
/// Serves `core/Marker.cpp:279,364` (triangulation, `2 * cameras x 4`),
/// `processing/Calibration.cpp:176` and `processing/CubeCalibration.cpp:832` (DLT,
/// `2 * points x 12`).
///
/// Size limit for parity: for `max(rows, cols) >= 25` the reference OpenCV uses LAPACK
/// `dgesdd`. With `rows >= cols` (a DLT with 13 or more points) its `z` may be the negative
/// of this one. With `rows < cols` the null space has more than one dimension and `dgesdd`
/// returns an arbitrary vector of it, unrelated to this one (see the module docs).
pub fn solve_z<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(a: &Matrix<f64, R, C, S>) -> DVector<f64> {
    let (rows, cols, data) = to_row_major(a);
    assert!(rows > 0 && cols > 0, "solve_z of an empty matrix");
    let r = svd_compute(rows, cols, &data, rows < cols);
    let last = r.vrows - 1;
    DVector::from_row_slice(&r.vt[last * cols..(last + 1) * cols])
}

// ---------------------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------------------

/// Least-squares solution of `a x = b` through the Jacobi SVD, with singular values below
/// `sum(w) * 2 * DBL_EPSILON` dropped (minimum-norm solution for rank-deficient `a`).
/// `a` is `m x n` with `m >= n`; `b` is `m x nb`; returns `n x nb`. OpenCV's return value
/// is always `true` on this path, so only `x` is returned.
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::solve` with `DECOMP_SVD` (the `JacobiSVD`
/// and `SVBkSb` branch). Serves `core/Marker.cpp:429,513` (`2 * cameras x 3`) and `:607`
/// (`3 x 3`).
///
/// Size limit for parity: for `m >= 25` the reference uses LAPACK `dgesdd`; `x` then agrees
/// to rounding.
///
/// Panics if `m < n` (OpenCV: "can not solve under-determined linear systems") or if the
/// row counts of `a` and `b` differ.
pub fn solve_svd<
    R: Dim,
    C: Dim,
    S: RawStorage<f64, R, C>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<f64, R2, C2>,
>(
    a: &Matrix<f64, R, C, S>,
    b: &Matrix<f64, R2, C2, S2>,
) -> DMatrix<f64> {
    let (m, n, src) = to_row_major(a);
    let (bm, nb, bdata) = to_row_major(b);
    assert_eq!(m, bm, "solve_svd: a has {m} rows, b has {bm}");
    assert!(m >= n, "solve_svd: under-determined system ({m} x {n})");
    assert!(n > 0, "solve_svd of an empty matrix");

    // a = transpose(src): n x m.
    let mut at = vec![0.0f64; n * m];
    for i in 0..n {
        for k in 0..m {
            at[i * m + k] = src[k * n + i];
        }
    }
    let mut w = vec![0.0f64; n];
    let mut v = vec![0.0f64; n * n];
    jacobi_svd(&mut at, &mut w, Some(&mut v), m, n, n);

    // SVBkSb(m_, n, w, u = a (uT), v (vT), b): u(j, i) = at[i*m + j], v(j, i) = v[i*n + j].
    let x = sv_bk_sb(m, n, &w, &at, m, 1, &v, n, 1, Some(&bdata), nb);
    from_row_major(n, nb, &x)
}

/// Least-squares solution of `a x = b` by Householder QR (`m >= n`), as OpenCV's own
/// `QRImpl`. Returns `n x nb`. `ok` is false, and `x` zero, when a diagonal element of R is
/// below `100 * DBL_EPSILON` in absolute value. An exactly zero column of `a` divides 0 by 0
/// and gives `ok = true` with NaN, as `QRImpl` does.
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::solve` with `DECOMP_QR`, and
/// `modules/core/src/matrix_decomp.cpp`, `QRImpl<double>`. Serves
/// `processing/MarkerDetection.cpp:539` (quadric fit, `N x 15`, `N` usually 49 or more),
/// which uses the result without checking `ok`.
///
/// Size limit for parity: up to 29 rows this is bit-identical to OpenCV, including every
/// failure case. With 30 or more rows (nearly always at that call site) the reference
/// uses LAPACK `dgels` (`hal_internal.cpp:330-445`), which fails only for an exactly zero
/// diagonal element of R:
///
/// - full-rank, well-scaled input: `x` agrees to rounding, not bit for bit;
/// - rank-deficient input (measured at 30 x 15, rank 1 to 14): OpenCV returns `ok = true`
///   with entries up to about 8e16, this function `ok = false` and zeros;
/// - tiny input (uniform noise times 1e-15): OpenCV `ok = true` with entries about 3e14,
///   this function `ok = false` and zeros;
/// - one exactly zero column: OpenCV `ok = false` and zeros, this function `ok = true` and NaN;
/// - the all-zero matrix: OpenCV `ok = true` and zeros, this function `ok = true` and NaN.
///
/// The behaviour is left as OpenCV's own code; parity ledger A37 decides it when detection
/// is ported.
///
/// Panics if `m < n` or if the row counts of `a` and `b` differ.
pub fn solve_qr<
    R: Dim,
    C: Dim,
    S: RawStorage<f64, R, C>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<f64, R2, C2>,
>(
    a: &Matrix<f64, R, C, S>,
    b: &Matrix<f64, R2, C2, S2>,
) -> Solved {
    let (m, n, mut am) = to_row_major(a);
    let (bm, k, mut bmat) = to_row_major(b);
    assert_eq!(m, bm, "solve_qr: a has {m} rows, b has {bm}");
    assert!(m >= n, "solve_qr: under-determined system ({m} x {n})");

    let ok = qr_impl(&mut am, m, n, k, &mut bmat, DBL_EPSILON * 100.0);
    let x = if ok {
        from_row_major(n, k, &bmat[..n * k])
    } else {
        DMatrix::zeros(n, k)
    };
    Solved { x, ok }
}

/// Ported from `modules/core/src/matrix_decomp.cpp`, `QRImpl<double>` (with `b` given and
/// `hFactors == NULL`). `a` is `m x n`, `b` is `m x k`, both row-major and overwritten.
fn qr_impl(a: &mut [f64], m: usize, n: usize, k: usize, b: &mut [f64], eps: f64) -> bool {
    let astep = n;
    let bstep = k;
    let mut vl = vec![0.0f64; m];
    let mut h_factors = vec![0.0f64; n];

    for l in 0..n {
        let vl_size = m - l;
        let mut vl_norm = 0.0f64;
        for i in 0..vl_size {
            vl[i] = a[(l + i) * astep + l];
            vl_norm += vl[i] * vl[i];
        }
        let tmp_v = vl[0];
        let sign = if vl[0] >= 0.0 { 1.0 } else { -1.0 };
        vl[0] += sign * vl_norm.sqrt();
        vl_norm = (vl_norm + vl[0] * vl[0] - tmp_v * tmp_v).sqrt();
        for e in vl.iter_mut().take(vl_size) {
            *e /= vl_norm;
        }
        for j in l..n {
            let mut v_la = 0.0f64;
            for i in l..m {
                v_la += vl[i - l] * a[i * astep + j];
            }
            for i in l..m {
                a[i * astep + j] -= 2.0 * vl[i - l] * v_la;
            }
        }

        h_factors[l] = vl[0] * vl[0];
        for i in 1..vl_size {
            a[(l + i) * astep + l] = vl[i] / vl[0];
        }
    }

    for l in 0..n {
        vl[0] = 1.0;
        for j in 1..(m - l) {
            vl[j] = a[(j + l) * astep + l];
        }
        for j in 0..k {
            let mut v_lb = 0.0f64;
            for i in l..m {
                v_lb += vl[i - l] * b[i * bstep + j];
            }
            for i in l..m {
                b[i * bstep + j] -= 2.0 * vl[i - l] * v_lb * h_factors[l];
            }
        }
    }

    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            for p in 0..k {
                b[i * bstep + p] -= b[j * bstep + p] * a[i * astep + j];
            }
        }
        if a[i * astep + i].abs() < eps {
            return false;
        }
        for p in 0..k {
            b[i * bstep + p] /= a[i * astep + i];
        }
    }
    true
}

// ---------------------------------------------------------------------------------------
// Inverse
// ---------------------------------------------------------------------------------------

/// Pseudo-inverse by the Jacobi SVD: `n x m` for an `m x n` input, with singular values
/// below `sum(w) * 2 * DBL_EPSILON` dropped. Also returns OpenCV's return value.
///
/// The return value reproduces an OpenCV quirk: it is `w[cols - 1] / w[0]` (or 0 when
/// `w[0] < DBL_EPSILON`), and for `rows < cols` the index `cols - 1` is past the
/// `rows` singular values, so OpenCV reads the next buffer in memory, which is `vt`
/// (row-major): the value is `vt[cols - 1 - rows] / w[0]` (`vt` from the thin [`svd`]).
/// XMALab ignores the value.
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::invert` with `DECOMP_SVD` and
/// `SVD::backSubst`. Serves `processing/LocalUndistortion.cpp:327` (`6 x nbNeighbours`).
///
/// Size limit for parity: for `max(rows, cols) >= 25` the reference uses LAPACK `dgesdd`;
/// the pseudo-inverse then agrees to rounding, and the return value for `rows < cols`
/// (which reads `vt`) may not be comparable: it depends on `dgesdd`'s singular vectors
/// (it agreed in 4 of 4 measured 4 x 25 cases). At XMALab's default size (6 x 12) the
/// Jacobi code runs and everything is bit-identical.
pub fn invert_svd<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
    a: &Matrix<f64, R, C, S>,
) -> (DMatrix<f64>, f64) {
    let (m, n, src) = to_row_major(a);
    assert!(m > 0 && n > 0, "invert_svd of an empty matrix");
    let nm = m.min(n);
    let r = svd_compute(m, n, &src, false);
    // backSubst(w, u, vt, Mat()): u is m x nm (not transposed), vt is nm x n (transposed).
    // u(j, i) = u[j*nm + i] -> udelta0 = 1, udelta1 = nm; v(j, i) = vt[i*n + j].
    let x = sv_bk_sb(m, n, &r.w, &r.u, 1, nm, &r.vt, n, 1, None, m);

    // w and vt are contiguous in OpenCV's buffer: element k of w past nm is vt[k - nm].
    let w_at = |k: usize| if k < nm { r.w[k] } else { r.vt[k - nm] };
    let retval = if r.w[0] >= DBL_EPSILON {
        w_at(n - 1) / r.w[0]
    } else {
        0.0
    };
    (from_row_major(n, m, &x), retval)
}

/// `det2` macro of `lapack.cpp`.
fn det2(s: &[f64], st: usize) -> f64 {
    s[0] * s[st + 1] - s[1] * s[st]
}

/// `det3` macro of `lapack.cpp`.
fn det3(s: &[f64]) -> f64 {
    let m = |y: usize, x: usize| s[y * 3 + x];
    m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
        - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
}

/// Inverse of a square matrix by OpenCV's default method, `DECOMP_LU`: what `Mat::inv()`
/// and `cv::invert(src, dst)` do.
///
/// - `n <= 3`: closed forms (cofactors divided by the determinant). The matrix counts as
///   singular only when the determinant is exactly 0.
/// - `n >= 4`: Gaussian elimination with partial pivoting (`LUImpl`) applied to the
///   identity. The matrix counts as singular when a pivot is below `100 * DBL_EPSILON` in
///   absolute value. This test is absolute, not relative to the matrix's scale.
///
/// For a singular matrix `ok` is false and `x` is all zeros, as in OpenCV (which XMALab
/// never checks).
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::invert` (`DECOMP_LU` branch), and
/// `modules/core/src/matrix_decomp.cpp`, `LUImpl<double>`. Serves every `.inv()` and
/// `cv::invert(P, P)` in XMALab: `core/Marker.cpp:574` (4x4), `:1765,1942`,
/// `core/Camera.cpp:585` (3x3), `core/CalibrationImage.cpp:400`, `core/RigidBody.cpp:1201`,
/// `processing/ButterworthLowPassFilter.cpp:335`, `processing/RigidBodyPoseOptimization.cpp:189,239`,
/// `processing/RigidBodyPoseFrom2D.cpp:334,341`, `processing/CubeCalibration.cpp:235`,
/// `processing/MultiCameraCalibration.cpp:481,524,525`.
///
/// Size limit for parity: for `n >= 100` the reference uses LAPACK `dgesv`.
///
/// Panics if `a` is not square or is empty.
pub fn invert<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(a: &Matrix<f64, R, C, S>) -> Solved {
    let (m, n, s) = to_row_major(a);
    assert_eq!(m, n, "invert: matrix is {m} x {n}, not square");
    assert!(n > 0, "invert of an empty matrix");
    let mut d = vec![0.0f64; n * n];
    let ok;

    if n == 1 {
        let det = s[0];
        ok = det != 0.0;
        if ok {
            d[0] = 1.0 / det;
        }
    } else if n == 2 {
        let mut det = det2(&s, 2);
        ok = det != 0.0;
        if ok {
            det = 1.0 / det;
            // CV_SIMD128_64F path: products by det, off-diagonal negated as 0 - x.
            d[0] = s[3] * det;
            d[1] = 0.0 - s[1] * det;
            d[2] = 0.0 - s[2] * det;
            d[3] = s[0] * det;
        }
    } else if n == 3 {
        let mut det = det3(&s);
        ok = det != 0.0;
        if ok {
            det = 1.0 / det;
            let sd = |y: usize, x: usize| s[y * 3 + x];
            d[0] = (sd(1, 1) * sd(2, 2) - sd(1, 2) * sd(2, 1)) * det;
            d[1] = (sd(0, 2) * sd(2, 1) - sd(0, 1) * sd(2, 2)) * det;
            d[2] = (sd(0, 1) * sd(1, 2) - sd(0, 2) * sd(1, 1)) * det;

            d[3] = (sd(1, 2) * sd(2, 0) - sd(1, 0) * sd(2, 2)) * det;
            d[4] = (sd(0, 0) * sd(2, 2) - sd(0, 2) * sd(2, 0)) * det;
            d[5] = (sd(0, 2) * sd(1, 0) - sd(0, 0) * sd(1, 2)) * det;

            d[6] = (sd(1, 0) * sd(2, 1) - sd(1, 1) * sd(2, 0)) * det;
            d[7] = (sd(0, 1) * sd(2, 0) - sd(0, 0) * sd(2, 1)) * det;
            d[8] = (sd(0, 0) * sd(1, 1) - sd(0, 1) * sd(1, 0)) * det;
        }
    } else {
        let mut a1 = s;
        for i in 0..n {
            d[i * n + i] = 1.0;
        }
        ok = lu_impl(&mut a1, n, Some((&mut d[..], n)), DBL_EPSILON * 100.0) != 0;
    }

    if !ok {
        d.iter_mut().for_each(|e| *e = 0.0);
    }
    Solved {
        x: from_row_major(n, n, &d),
        ok,
    }
}

/// `cv::solve(a, b, x, DECOMP_LU)` for a square `a` (`n x n`) and a `b` of `n x nb`, on the
/// general path: `n >= 4` or `nb > 1`. This is also what `a.inv() * b` evaluates to for two
/// `Mat`s (`MatOp_Invert::matmul` turns it into `MatOp_Solve`), as
/// `processing/MultiCameraCalibration.cpp:524-525` uses it. When a pivot is below
/// `100 * DBL_EPSILON` (`ok` false) `x` holds what the elimination left in `b`, as in
/// OpenCV (whose callers there ignore the result).
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::solve` (`DECOMP_LU`, not normal), and
/// `modules/core/src/matrix_decomp.cpp`, `LUImpl<double>`. Size limit for parity: for
/// `n >= 100` the reference uses LAPACK.
///
/// Panics if `a` is not square, is empty, if `b` has not `n` rows, or for the small-system
/// shortcut OpenCV takes instead (`n <= 3` with one column), which is not ported.
pub fn solve_lu<R: Dim, C: Dim, S: RawStorage<f64, R, C>, R2: Dim, C2: Dim, S2>(
    a: &Matrix<f64, R, C, S>,
    b: &Matrix<f64, R2, C2, S2>,
) -> Solved
where
    S2: RawStorage<f64, R2, C2>,
{
    let (m, n, mut s) = to_row_major(a);
    let (bm, nb, mut d) = to_row_major(b);
    assert_eq!(m, n, "solve_lu: matrix is {m} x {n}, not square");
    assert!(n > 0, "solve_lu of an empty matrix");
    assert_eq!(bm, n, "solve_lu: right-hand side has {bm} rows, {n} needed");
    assert!(
        n > 3 || nb > 1,
        "solve_lu: the n <= 3, one-column shortcut of cv::solve is not ported"
    );
    let ok = lu_impl(&mut s, n, Some((&mut d[..], nb)), DBL_EPSILON * 100.0) != 0;
    Solved {
        x: from_row_major(n, nb, &d),
        ok,
    }
}

/// Gaussian elimination with partial pivoting on the `m x m` row-major `a`, applied to the
/// `m x nb` row-major `b` if given (which ends up as `a^-1 b`). Returns the permutation sign
/// (`1` or `-1`), or `0` when a pivot is below `eps` in absolute value.
///
/// Ported from `modules/core/src/matrix_decomp.cpp`, `LUImpl<double>`.
fn lu_impl(a: &mut [f64], m: usize, mut b: Option<(&mut [f64], usize)>, eps: f64) -> i32 {
    let astep = m;
    let mut p = 1;

    for i in 0..m {
        let mut k = i;
        for j in (i + 1)..m {
            if a[j * astep + i].abs() > a[k * astep + i].abs() {
                k = j;
            }
        }

        if a[k * astep + i].abs() < eps {
            return 0;
        }

        if k != i {
            for j in i..m {
                a.swap(i * astep + j, k * astep + j);
            }
            if let Some((b, n)) = b.as_mut() {
                for j in 0..*n {
                    b.swap(i * *n + j, k * *n + j);
                }
            }
            p = -p;
        }

        let d = -1.0 / a[i * astep + i];

        for j in (i + 1)..m {
            let alpha = a[j * astep + i] * d;

            for k in (i + 1)..m {
                a[j * astep + k] += alpha * a[i * astep + k];
            }

            if let Some((b, n)) = b.as_mut() {
                for k in 0..*n {
                    b[j * *n + k] += alpha * b[i * *n + k];
                }
            }
        }
    }

    if let Some((b, n)) = b.as_mut() {
        for i in (0..m).rev() {
            for j in 0..*n {
                let mut s = b[i * *n + j];
                for k in (i + 1)..m {
                    s -= a[i * astep + k] * b[k * *n + j];
                }
                b[i * *n + j] = s / a[i * astep + i];
            }
        }
    }

    p
}

// ---------------------------------------------------------------------------------------
// Symmetric eigen decomposition (Jacobi) and the solves built on it
// ---------------------------------------------------------------------------------------

/// Eigenvalues (descending) and eigenvectors (row `i` belongs to value `i`) of a symmetric
/// row-major `n x n` matrix by OpenCV's own Jacobi rotations; only the upper triangle is
/// read.
///
/// Ported from `modules/core/src/lapack.cpp`, `JacobiImpl_<double>` (what `cv::solve` and
/// `cv::invert` with `DECOMP_EIG` run, and what `cv::eigen` runs in a build without Eigen).
/// The reference OpenCV is built with Eigen, so its `cv::eigen` itself is Eigen's
/// `SelfAdjointEigenSolver`, which is not ported; there the results agree to rounding and
/// eigenvector signs may differ.
///
/// Panics if `a.len() != n * n`.
#[must_use]
pub fn eigen_jacobi(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n, "eigen_jacobi: matrix is not {n} x {n}");
    let eps = f64::EPSILON;
    let mut a = a.to_vec();
    let astep = n;
    let mut w = vec![0.0f64; n];
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_iters = n * n * 30;
    let mut ind_r = vec![0usize; n];
    let mut ind_c = vec![0usize; n];
    for k in 0..n {
        w[k] = a[(astep + 1) * k];
        if k + 1 < n {
            let mut m = k + 1;
            let mut mv = a[astep * k + m].abs();
            for i in k + 2..n {
                let val = a[astep * k + i].abs();
                if mv < val {
                    mv = val;
                    m = i;
                }
            }
            ind_r[k] = m;
        }
        if k > 0 {
            let mut m = 0;
            let mut mv = a[k].abs();
            for i in 1..k {
                let val = a[astep * i + k].abs();
                if mv < val {
                    mv = val;
                    m = i;
                }
            }
            ind_c[k] = m;
        }
    }
    if n > 1 {
        for _ in 0..max_iters {
            let mut k = 0usize;
            let mut mv = a[ind_r[0]].abs();
            for i in 1..n - 1 {
                let val = a[astep * i + ind_r[i]].abs();
                if mv < val {
                    mv = val;
                    k = i;
                }
            }
            let mut l = ind_r[k];
            for i in 1..n {
                let val = a[astep * ind_c[i] + i].abs();
                if mv < val {
                    mv = val;
                    k = ind_c[i];
                    l = i;
                }
            }
            let p = a[astep * k + l];
            if p.abs() <= eps {
                break;
            }
            let y = (w[l] - w[k]) * 0.5;
            let mut t = y.abs() + p.hypot(y);
            let mut s = p.hypot(t);
            let c = t / s;
            s = p / s;
            t = (p / t) * p;
            if y < 0.0 {
                s = -s;
                t = -t;
            }
            a[astep * k + l] = 0.0;
            w[k] -= t;
            w[l] += t;
            let rot = |x: &mut [f64], i0: usize, i1: usize| {
                let a0 = x[i0];
                let b0 = x[i1];
                x[i0] = a0 * c - b0 * s;
                x[i1] = a0 * s + b0 * c;
            };
            for i in 0..k {
                rot(&mut a, astep * i + k, astep * i + l);
            }
            for i in k + 1..l {
                rot(&mut a, astep * k + i, astep * i + l);
            }
            for i in l + 1..n {
                rot(&mut a, astep * k + i, astep * l + i);
            }
            for i in 0..n {
                rot(&mut v, n * k + i, n * l + i);
            }
            for j in 0..2 {
                let idx = if j == 0 { k } else { l };
                if idx + 1 < n {
                    let mut m = idx + 1;
                    let mut mv = a[astep * idx + m].abs();
                    for i in idx + 2..n {
                        let val = a[astep * idx + i].abs();
                        if mv < val {
                            mv = val;
                            m = i;
                        }
                    }
                    ind_r[idx] = m;
                }
                if idx > 0 {
                    let mut m = 0;
                    let mut mv = a[idx].abs();
                    for i in 1..idx {
                        let val = a[astep * i + idx].abs();
                        if mv < val {
                            mv = val;
                            m = i;
                        }
                    }
                    ind_c[idx] = m;
                }
            }
        }
    }
    for k in 0..n.saturating_sub(1) {
        let mut m = k;
        for i in k + 1..n {
            if w[m] < w[i] {
                m = i;
            }
        }
        if k != m {
            w.swap(m, k);
            for i in 0..n {
                v.swap(n * m + i, n * k + i);
            }
        }
    }
    (w, v)
}

/// `cv::solve(a, b, x, DECOMP_EIG)` for a square symmetric row-major `n x n` `a` and an
/// `n x nb` `b`: Jacobi eigen decomposition, then `SVBkSb` with the eigenvectors as both
/// `u` and `v`. Returns `n x nb`, row-major.
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::solve` (the `DECOMP_EIG` branch).
#[must_use]
pub fn solve_eig(a: &[f64], n: usize, b: &[f64], nb: usize) -> Vec<f64> {
    let (w, v) = eigen_jacobi(a, n);
    sv_bk_sb(n, n, &w, &v, n, 1, &v, n, 1, Some(b), nb)
}

/// The diagonal of `cv::invert(a, ai, DECOMP_EIG)` for a symmetric row-major `n x n` `a`:
/// the eigen decomposition (here OpenCV's own Jacobi, see [`eigen_jacobi`]), then
/// `SVD::backSubst(w, u = v^T, vt = v, Mat())`.
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::invert` (the `DECOMP_EIG` branch).
#[must_use]
pub fn invert_eig(a: &[f64], n: usize) -> Vec<f64> {
    let (w, v) = eigen_jacobi(a, n);
    // u = transpose(vt): u(j, i) = v[i*n + j] -> udelta0 = n... in backSubst u is not
    // transposed (u(j, i) at u[j*n + i]), here u = vt^T so u[j*n + i] = v[i*n + j].
    sv_bk_sb(n, n, &w, &v, n, 1, &v, n, 1, None, n)
}

// ---------------------------------------------------------------------------------------
// Determinant
// ---------------------------------------------------------------------------------------

/// Determinant of a square matrix: closed forms for `n <= 3`, otherwise the product of the
/// LU pivots times the permutation sign. As in OpenCV, an LU pivot below `100 * DBL_EPSILON`
/// (absolute) makes the result exactly 0.
///
/// Ported from `modules/core/src/lapack.cpp`, `cv::determinant`. Serves
/// `core/RigidBody.cpp:1045,1046,1413,1414,1864,1865` and
/// `processing/RigidBodyPoseFrom2D.cpp:206,207` (3x3, the sign test of the Kabsch fit).
///
/// Size limit for parity: for `n >= 100` the reference uses LAPACK `dgetrf`.
///
/// Panics if `a` is not square or is empty.
pub fn determinant<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(a: &Matrix<f64, R, C, S>) -> f64 {
    let (m, n, mut s) = to_row_major(a);
    assert_eq!(m, n, "determinant: matrix is {m} x {n}, not square");
    assert!(n > 0, "determinant of an empty matrix");
    match n {
        1 => s[0],
        2 => det2(&s, 2),
        3 => det3(&s),
        _ => {
            let mut result = lu_impl(&mut s, n, None, DBL_EPSILON * 100.0) as f64;
            if result != 0.0 {
                for i in 0..n {
                    result *= s[i * n + i];
                }
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------------------
// sortIdx
// ---------------------------------------------------------------------------------------

/// Indices that sort `values` ascending (or descending), with OpenCV's order for equal
/// values.
///
/// OpenCV sorts indices with `std::sort` and the comparator `arr[a] < arr[b]` (IPP has no
/// `f64` index sort, so it is not used). `std::sort` is not stable; this is a port of
/// libstdc++'s introsort (median-of-three quicksort down to 16 elements, heapsort past
/// `2 * floor(log2(n))` levels, then insertion sort), so ties come out in the same order as
/// in the GCC-built reference. For `n <= 16` that is plain insertion sort, which keeps
/// equal values in index order. Descending order is the ascending result reversed, so
/// there ties come out in reverse.
///
/// NaN breaks the comparator's ordering in C++ (undefined behaviour); here it cannot read
/// out of bounds, but the order is then unspecified.
///
/// Ported from `modules/core/src/matrix_operations.cpp`, `sortIdx_<double>`, and
/// libstdc++ `bits/stl_algo.h` / `bits/stl_heap.h` (`std::__sort`). Serves
/// `processing/LocalUndistortion.cpp:301,547` (a column of distances, `SORT_EVERY_COLUMN`) and
/// `:605-679` (a row of six angles, `SORT_EVERY_ROW`); there `values` is that column or row.
pub fn sort_idx(values: &[f64], descending: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    libstdcxx_sort(&mut idx, &|a: usize, b: usize| values[a] < values[b]);
    if descending {
        idx.reverse();
    }
    idx
}

/// `cv::sortIdx` on a matrix: every row or every column sorted on its own, as [`sort_idx`].
/// Returns a matrix of the same shape holding, in each row (column), the indices into that
/// row (column).
///
/// Ported from `modules/core/src/matrix_operations.cpp`, `cv::sortIdx` / `sortIdx_<double>`.
pub fn sort_idx_matrix<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
    a: &Matrix<f64, R, C, S>,
    by: SortBy,
    descending: bool,
) -> DMatrix<usize> {
    let (m, n) = a.shape();
    let mut out = DMatrix::<usize>::zeros(m, n);
    match by {
        SortBy::EveryRow => {
            for i in 0..m {
                let row: Vec<f64> = (0..n).map(|j| a[(i, j)]).collect();
                for (j, k) in sort_idx(&row, descending).into_iter().enumerate() {
                    out[(i, j)] = k;
                }
            }
        }
        SortBy::EveryColumn => {
            for j in 0..n {
                let col: Vec<f64> = (0..m).map(|i| a[(i, j)]).collect();
                for (i, k) in sort_idx(&col, descending).into_iter().enumerate() {
                    out[(i, j)] = k;
                }
            }
        }
    }
    out
}

/// libstdc++ `_S_threshold`.
const SORT_THRESHOLD: usize = 16;

/// libstdc++ `std::sort(first, last, comp)` (`std::__sort`).
fn libstdcxx_sort<F: Fn(usize, usize) -> bool>(v: &mut [usize], comp: &F) {
    let len = v.len();
    if len == 0 {
        return;
    }
    // std::__lg(n) * 2
    let depth = 2 * (usize::BITS - 1 - len.leading_zeros()) as usize;
    introsort_loop(v, 0, len, depth, comp);
    final_insertion_sort(v, 0, len, comp);
}

/// libstdc++ `std::__introsort_loop`.
fn introsort_loop<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    first: usize,
    mut last: usize,
    mut depth_limit: usize,
    comp: &F,
) {
    while last - first > SORT_THRESHOLD {
        if depth_limit == 0 {
            // std::__partial_sort(first, last, last): heap select (make_heap) + sort_heap.
            make_heap(v, first, last, comp);
            sort_heap(v, first, last, comp);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(v, first, last, comp);
        introsort_loop(v, cut, last, depth_limit, comp);
        last = cut;
    }
}

/// libstdc++ `std::__unguarded_partition_pivot`.
fn unguarded_partition_pivot<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    first: usize,
    last: usize,
    comp: &F,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, comp);
    unguarded_partition(v, first + 1, last, first, comp)
}

/// libstdc++ `std::__move_median_to_first`.
fn move_median_to_first<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    comp: &F,
) {
    let cmp = |x: usize, y: usize, v: &[usize]| comp(v[x], v[y]);
    if cmp(a, b, v) {
        if cmp(b, c, v) {
            v.swap(result, b);
        } else if cmp(a, c, v) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if cmp(a, c, v) {
        v.swap(result, a);
    } else if cmp(b, c, v) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

/// libstdc++ `std::__unguarded_partition`. The bounds checks only matter for a comparator
/// that is not a strict weak ordering (NaN); otherwise they never trigger.
fn unguarded_partition<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    comp: &F,
) -> usize {
    let lo = pivot;
    loop {
        while first < last && comp(v[first], v[pivot]) {
            first += 1;
        }
        last -= 1;
        while last > lo && comp(v[pivot], v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// libstdc++ `std::__final_insertion_sort`.
fn final_insertion_sort<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    first: usize,
    last: usize,
    comp: &F,
) {
    if last - first > SORT_THRESHOLD {
        insertion_sort(v, first, first + SORT_THRESHOLD, comp);
        for i in (first + SORT_THRESHOLD)..last {
            unguarded_linear_insert(v, first, i, comp);
        }
    } else {
        insertion_sort(v, first, last, comp);
    }
}

/// libstdc++ `std::__insertion_sort`.
fn insertion_sort<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    first: usize,
    last: usize,
    comp: &F,
) {
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        if comp(v[i], v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            unguarded_linear_insert(v, first, i, comp);
        }
    }
}

/// libstdc++ `std::__unguarded_linear_insert`. `lo` bounds the scan (see
/// [`unguarded_partition`]).
fn unguarded_linear_insert<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    lo: usize,
    mut last: usize,
    comp: &F,
) {
    let val = v[last];
    while last > lo && comp(val, v[last - 1]) {
        v[last] = v[last - 1];
        last -= 1;
    }
    v[last] = val;
}

/// libstdc++ `std::__make_heap`.
fn make_heap<F: Fn(usize, usize) -> bool>(v: &mut [usize], first: usize, last: usize, comp: &F) {
    let len = last - first;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        adjust_heap(v, first, parent, len, value, comp);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// libstdc++ `std::__sort_heap`.
fn sort_heap<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    first: usize,
    mut last: usize,
    comp: &F,
) {
    while last - first > 1 {
        last -= 1;
        // std::__pop_heap(first, last, last)
        let value = v[last];
        v[last] = v[first];
        adjust_heap(v, first, 0, last - first, value, comp);
    }
}

/// libstdc++ `std::__adjust_heap` followed by `std::__push_heap`.
fn adjust_heap<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    first: usize,
    mut hole: usize,
    len: usize,
    value: usize,
    comp: &F,
) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if comp(v[first + second], v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    // std::__push_heap
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && comp(v[first + parent], value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[first + hole] = value;
}

/// Test hook, not part of the API: the heapsort fallback of [`sort_idx`] on its own
/// (libstdc++ `std::__partial_sort(first, last, last)`: `make_heap` then `sort_heap`), so
/// `tests/linalg.rs` can compare it with libstdc++ directly. Ordinary inputs rarely reach
/// that fallback.
#[doc(hidden)]
pub fn heap_sort_idx_for_tests(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    let comp = |a: usize, b: usize| values[a] < values[b];
    let n = idx.len();
    make_heap(&mut idx, 0, n, &comp);
    sort_heap(&mut idx, 0, n, &comp);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_opencv_first_values() {
        // RNG(0x12345678).next(): state = 0x12345678 * 4164903690 + 0.
        let mut r = CvRng::new(0x1234_5678);
        let expect = (0x1234_5678u64 * 4_164_903_690u64) as u32;
        assert_eq!(r.next(), expect);
    }

    #[test]
    fn heap_fallback_sorts() {
        // Force the heapsort path with a zero depth limit.
        let values: Vec<f64> = (0..40).map(|i| ((i * 37) % 23) as f64).collect();
        let mut v: Vec<usize> = (0..values.len()).collect();
        let comp = |a: usize, b: usize| values[a] < values[b];
        introsort_loop(&mut v, 0, 40, 0, &comp);
        final_insertion_sort(&mut v, 0, 40, &comp);
        for w in v.windows(2) {
            assert!(values[w[0]] <= values[w[1]]);
        }
        let mut seen = v.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn nan_does_not_panic() {
        let values = [
            3.0,
            f64::NAN,
            1.0,
            2.0,
            f64::NAN,
            0.0,
            5.0,
            4.0,
            1.0,
            1.0,
            7.0,
            2.0,
            9.0,
            8.0,
            6.0,
            0.5,
            2.5,
            3.5,
            f64::NAN,
            1.5,
        ];
        let idx = sort_idx(&values, false);
        let mut seen = idx.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..values.len()).collect::<Vec<_>>());
    }
}
