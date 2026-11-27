//! 3D reconstruction of a point from two or more cameras.
//!
//! Work package: triangulation. Ported from `core/Marker.cpp`, `Marker::reconstruct3DPoint`
//! and the five functions it dispatches to.
//!
//! # What is ported and what is left to the caller
//!
//! [`triangulate`] is the numerical part of the five `reconstruct3DPoint*` functions: from
//! the views it is given (the cameras with `status2D > 0`, in camera order, each with its
//! projection matrix and undistorted point) it returns the 3D point the C++ writes into
//! `points3D[frame]`, or `None` where the C++ writes nothing. Choosing the cameras, setting
//! `status3D` (the minimum participating `status2D`, else `UNDEFINED`) and `reprojectPoint`
//! are the caller's.
//!
//! # Floating point
//!
//! Every expression keeps the C++ operand order and is never fused. The OpenCV operations
//! the C++ uses besides the decompositions in [`crate::linalg`] are written out here
//! following OpenCV 4.13:
//!
//! - `Mat /= double` and `Mat / double` are `convertTo(.., 1./s)`
//!   (`core/include/opencv2/core/operations.hpp:312`, `core/src/matrix_expressions.cpp:893`
//!   and `MatOp_AddEx::assign`): each element becomes `v * (1/s) + 0`, and the matrix is
//!   copied unchanged when `|1/s - 1| < DBL_EPSILON` (`core/src/convert.dispatch.cpp:268`).
//!   See [`scale_like_convert_to`].
//! - `P_inv * v` and `P_inv * v - origin` are `cv::gemm` with `len == 4` and a one-column
//!   result, the unrolled small-matrix branch of `gemmImpl` (`core/src/matmul.simd.hpp`):
//!   `t = a0*b0 + a1*b1 + a2*b2 + a3*b3` left to right, then `t*alpha + c*beta`. See
//!   [`gemm_4x4_by_4x1`].
//! - `cv::norm(dir)` of four doubles is `sqrt` of `s += v*v` in order from `s = 0`
//!   (`core/src/norm.simd.hpp`, `normL2_` and `NormL2_SIMD<double, double>`, whose vector
//!   loop needs `4 * lanes` elements: eight on the SSE2 baseline, sixteen in the AVX2
//!   dispatch the reference machine runs; four values take the scalar tail either way).
//!
//! # Bit-identity holds for 2 to 12 cameras
//!
//! Methods 0 to 3 solve a `2n x 4` (methods 0, 1) or `2n x 3` (methods 2, 3) system by SVD
//! for `n` cameras. The reference OpenCV (built with LAPACK) hands an SVD to LAPACK `dgesdd`
//! when `max(rows, cols) >= 25`, that is `2n >= 25`, 13 cameras or more; [`crate::linalg`]
//! always runs OpenCV's own Jacobi SVD (parity ledger, item A37). So these methods are
//! bit-identical to the C++ for 2 to 12 cameras only. Measured against the real OpenCV with
//! 13 to 30 cameras:
//!
//! - on ordinary data, methods 0, 2 and 3 agree to about 1e-13 relative, and method 1 to
//!   about 6e-10 relative (its reweighting passes amplify the difference);
//! - on degenerate input the two give unrelated answers: identical cameras; a point at
//!   infinity; a `P(2,3)` of 0 (methods 2 and 3: below 13 cameras both give NaN, with 13 or
//!   more the reference may give (0, 0, 0), and the port may too, not necessarily for the
//!   same input); a NaN image coordinate.
//!
//! Method 4 solves a 3x3 system whatever the camera count, and is bit-identical at every
//! count tested (up to 30). `tests/triangulation.rs` checks all of this against a fixture
//! from the reference build (`tests/data/triangulation/`).

use nalgebra::{DMatrix, Matrix4};

use crate::linalg::{invert, solve_svd, solve_z};
use crate::{Matrix3x4, Point2, Point3};

/// The triangulation variants, numbered as the `TriangulationMethod` setting numbers them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TriangulationMethod {
    /// 0: `reconstruct3DPointZisserman`, linear homogeneous.
    Zisserman = 0,
    /// 1: `reconstruct3DPointZissermanIncremental`, iteratively reweighted. XMALab's default.
    #[default]
    ZissermanIncremental = 1,
    /// 2: `reconstruct3DPointZissermanMatlab`.
    ZissermanMatlab = 2,
    /// 3: `reconstruct3DPointZissermanIncrementalMatlab`.
    ZissermanIncrementalMatlab = 3,
    /// 4: `reconstruct3DPointRayIntersection`.
    RayIntersection = 4,
}

impl TriangulationMethod {
    /// The method for a setting value. Unknown values select method 0, as the C++ `switch`
    /// does with its `default` label.
    ///
    /// Ported from `core/Marker.cpp`, `Marker::reconstruct3DPoint` (the `switch`, `:216-234`).
    #[must_use]
    pub const fn from_setting(value: i32) -> Self {
        match value {
            1 => Self::ZissermanIncremental,
            2 => Self::ZissermanMatlab,
            3 => Self::ZissermanIncrementalMatlab,
            4 => Self::RayIntersection,
            _ => Self::Zisserman,
        }
    }
}

/// One camera's view of the point.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    /// The camera's projection matrix for the trial's reference calibration image.
    pub projection: Matrix3x4,
    /// The point in undistorted pixels (`Camera::undistortPoint(p, true)`).
    pub point: Point2,
}

/// Reconstructs a 3D point from the cameras that see it.
///
/// Ported from `core/Marker.cpp`, the function named on each [`TriangulationMethod`] variant
/// (dispatch in `Marker::reconstruct3DPoint`, `:209-237`). `observations` are the cameras
/// with `status2D > 0`, in camera order: the order of the rows of each linear system.
///
/// Returns `None` where the C++ writes nothing into `points3D[frame]`:
/// - with fewer than two observations (every method);
/// - for [`TriangulationMethod::Zisserman`] and [`TriangulationMethod::ZissermanIncremental`],
///   when the fourth homogeneous component of the solution is exactly zero.
///
/// The other three methods always return a point, which may hold NaN or infinities on
/// degenerate input, exactly as the C++ writes them. Tier A.
///
/// Bit-identical to the C++ for 2 to 12 observations. With 13 or more, methods 0 to 3 differ
/// from the reference build, whose OpenCV then hands their SVD (`2n >= 25` rows) to LAPACK:
/// by about 1e-13 relative on ordinary data (method 1 by about 6e-10), and arbitrarily on
/// degenerate input (identical cameras, a point at infinity, `P(2,3)` of 0, a NaN
/// coordinate). Method 4 stays bit-identical at every count tested (up to 30). See the
/// module docs and parity ledger item A37.
/// Oracle: `trial_<t>/points3d_method<k>.csv`, from `trial_<t>/points2d_undistorted.csv` and
/// `trial_<t>/projection_matrices.csv`.
#[must_use]
pub fn triangulate(method: TriangulationMethod, observations: &[Observation]) -> Option<Point3> {
    if observations.len() < 2 {
        return None;
    }
    match method {
        TriangulationMethod::Zisserman => zisserman(observations),
        TriangulationMethod::ZissermanIncremental => zisserman_incremental(observations),
        TriangulationMethod::ZissermanMatlab => Some(zisserman_matlab(observations)),
        TriangulationMethod::ZissermanIncrementalMatlab => {
            Some(zisserman_incremental_matlab(observations))
        }
        TriangulationMethod::RayIntersection => Some(ray_intersection(observations)),
    }
}

/// `DBL_EPSILON`.
const DBL_EPSILON: f64 = f64::EPSILON;

/// The weight change below which the incremental methods stop (`double eps = 0.0000001`).
const INCREMENTAL_EPS: f64 = 0.000_000_1;

/// The number of passes of the incremental methods (`for (int j = 0; j < 10; j++)`).
const INCREMENTAL_PASSES: usize = 10;

/// `f / w` for a homogeneous solution, or `None` when `w == 0`.
///
/// Ported from `core/Marker.cpp`, the `w != 0.0` tail of `Marker::reconstruct3DPointZisserman`
/// (`:280-289`) and `Marker::reconstruct3DPointZissermanIncremental` (`:368-376`).
fn dehomogenise(f: &[f64]) -> Option<Point3> {
    let w = f[3];
    if w != 0.0 {
        Some(Point3::new(f[0] / w, f[1] / w, f[2] / w))
    } else {
        None
    }
}

/// Method 0: rows `x*P(2,k) - P(0,k)` and `y*P(2,k) - P(1,k)` per view, null vector by
/// `cv::SVD::solveZ`, divided by its fourth component.
///
/// Ported from `core/Marker.cpp`, `Marker::reconstruct3DPointZisserman` (`:239-291`).
fn zisserman(observations: &[Observation]) -> Option<Point3> {
    let mut a = DMatrix::<f64>::zeros(2 * observations.len(), 4);
    for (count, obs) in observations.iter().enumerate() {
        let (x, y) = (obs.point.x, obs.point.y);
        let p = &obs.projection;
        for k in 0..4 {
            a[(count * 2, k)] = x * p[(2, k)] - p[(0, k)];
            a[(count * 2 + 1, k)] = y * p[(2, k)] - p[(1, k)];
        }
    }
    let f = solve_z(&a);
    dehomogenise(f.as_slice())
}

/// Method 1: method 0 with each view's two rows divided by a weight. The weights start at
/// 1; from the second pass on each is `P(2,:) . f` with `f` the previous (unit-norm)
/// solution. At most 10 passes; a pass in which every weight changed by less than 1e-7
/// stops before solving, so the result is the solution of the previous pass.
///
/// Ported from `core/Marker.cpp`, `Marker::reconstruct3DPointZissermanIncremental`
/// (`:294-378`).
fn zisserman_incremental(observations: &[Observation]) -> Option<Point3> {
    let n = observations.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 4);
    let mut w = vec![1.0f64; n];
    // `f` is only read from the second pass on, after the first solve has set it.
    let mut f: Vec<f64> = Vec::new();

    for j in 0..INCREMENTAL_PASSES {
        let mut stop = true;
        for (count, obs) in observations.iter().enumerate() {
            let (x, y) = (obs.point.x, obs.point.y);
            let p = &obs.projection;
            if j != 0 {
                let w_old = w[count];
                w[count] =
                    f[0] * p[(2, 0)] + f[1] * p[(2, 1)] + f[2] * p[(2, 2)] + f[3] * p[(2, 3)];
                // `stop = (fabs(w - w_old) < eps) ? stop : false;` (NaN clears `stop`).
                stop = stop && (w[count] - w_old).abs() < INCREMENTAL_EPS;
            } else {
                stop = false;
            }
            for k in 0..4 {
                a[(count * 2, k)] = (x * p[(2, k)] - p[(0, k)]) / w[count];
                a[(count * 2 + 1, k)] = (y * p[(2, k)] - p[(1, k)]) / w[count];
            }
        }
        if stop {
            break;
        }
        f = solve_z(&a).as_slice().to_vec();
    }
    dehomogenise(&f)
}

/// `projMatrs /= projMatrs.at<double>(2, 3)`: OpenCV's `Mat /= double`, which is
/// `convertTo(m, -1, 1./s)`. Every element becomes `v * (1/s) + 0`, unless
/// `|1/s - 1| < DBL_EPSILON`, where OpenCV copies the matrix unchanged. The `+ 0` turns a
/// negative zero into a positive one, as OpenCV's `src*a + b` does.
///
/// Ported from `modules/core/include/opencv2/core/operations.hpp:312` (`operator /=`),
/// `modules/core/src/convert.dispatch.cpp`, `Mat::convertTo` (the `noScale` test), and
/// `modules/core/src/convert_scale.simd.hpp`, `cvt_64f`. The same holds for `Mat / double`
/// assigned to a matrix (`MatOp_AddEx::assign` calls `convertTo` with `alpha = 1./s`).
fn scale_like_convert_to(values: &mut [f64], s: f64) {
    let alpha = 1.0 / s;
    let beta = 0.0;
    let no_scale = (alpha - 1.0).abs() < DBL_EPSILON && f64::abs(beta) < DBL_EPSILON;
    if no_scale {
        return;
    }
    for v in values {
        *v = *v * alpha + beta;
    }
}

/// A view's projection matrix divided by its `P(2,3)` as the "Matlab" methods do it.
///
/// Ported from `core/Marker.cpp:408` and `:479` (`projMatrs /= projMatrs.at<double>(2, 3)`).
fn normalised_projection(p: &Matrix3x4) -> Matrix3x4 {
    let mut q = *p;
    // Read before scaling: OpenCV evaluates `1./b` once, before `convertTo` runs.
    let s = q[(2, 3)];
    scale_like_convert_to(q.as_mut_slice(), s);
    q
}

/// Method 2: P divided by `P(2,3)`, then the inhomogeneous `2n x 3` system
/// `A f = B` with rows `x*P(2,k) - P(0,k)`, `y*P(2,k) - P(1,k)` and right-hand sides
/// `P(0,3) - x`, `P(1,3) - y`, solved by `cv::solve(.., DECOMP_SVD)`.
///
/// Ported from `core/Marker.cpp`, `Marker::reconstruct3DPointZissermanMatlab` (`:381-438`).
fn zisserman_matlab(observations: &[Observation]) -> Point3 {
    let n = observations.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 3);
    let mut b = DMatrix::<f64>::zeros(2 * n, 1);
    for (count, obs) in observations.iter().enumerate() {
        let (x, y) = (obs.point.x, obs.point.y);
        let p = normalised_projection(&obs.projection);
        for k in 0..3 {
            a[(count * 2, k)] = x * p[(2, k)] - p[(0, k)];
            a[(count * 2 + 1, k)] = y * p[(2, k)] - p[(1, k)];
        }
        b[(count * 2, 0)] = p[(0, 3)] - x;
        b[(count * 2 + 1, 0)] = p[(1, 3)] - y;
    }
    let f = solve_svd(&a, &b);
    Point3::new(f[(0, 0)], f[(1, 0)], f[(2, 0)])
}

/// Method 3: method 2 with each view's rows and right-hand sides divided by a weight. The
/// weights start at 1; from the second pass on each is `P(2,0..2) . f + P(2,3)` (with P
/// normalised). At most 10 passes, with the stopping rule of method 1.
///
/// Ported from `core/Marker.cpp`, `Marker::reconstruct3DPointZissermanIncrementalMatlab`
/// (`:440-524`).
fn zisserman_incremental_matlab(observations: &[Observation]) -> Point3 {
    let n = observations.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 3);
    let mut b = DMatrix::<f64>::zeros(2 * n, 1);
    let mut w = vec![1.0f64; n];
    // Set by the first pass's solve; only read from the second pass on.
    let mut f = [0.0f64; 3];

    for j in 0..INCREMENTAL_PASSES {
        let mut stop = true;
        for (count, obs) in observations.iter().enumerate() {
            let (x, y) = (obs.point.x, obs.point.y);
            let p = normalised_projection(&obs.projection);
            if j != 0 {
                let w_old = w[count];
                w[count] = f[0] * p[(2, 0)] + f[1] * p[(2, 1)] + f[2] * p[(2, 2)] + p[(2, 3)];
                stop = stop && (w[count] - w_old).abs() < INCREMENTAL_EPS;
            } else {
                stop = false;
            }
            for k in 0..3 {
                a[(count * 2, k)] = (x * p[(2, k)] - p[(0, k)]) / w[count];
                a[(count * 2 + 1, k)] = (y * p[(2, k)] - p[(1, k)]) / w[count];
            }
            b[(count * 2, 0)] = (p[(0, 3)] - x) / w[count];
            b[(count * 2 + 1, 0)] = (p[(1, 3)] - y) / w[count];
        }
        if stop {
            break;
        }
        let s = solve_svd(&a, &b);
        f = [s[(0, 0)], s[(1, 0)], s[(2, 0)]];
    }
    Point3::new(f[0], f[1], f[2])
}

/// `alpha * a * b + beta * c` for a 4x4 `a` and a 4x1 `b`, as OpenCV's `gemmImpl` computes
/// it in its unrolled `len == 4` branch for a one-column result: each row's dot product is
/// summed left to right, then `t*alpha + c*beta`. With no `c` OpenCV reads zeros.
///
/// Ported from `modules/core/src/matmul.simd.hpp`, `gemmImpl` (`CV_64F`, `case 4`, the
/// `d_size.width <= 16 && a != d` branch).
fn gemm_4x4_by_4x1(
    a: &Matrix4<f64>,
    b: &[f64; 4],
    alpha: f64,
    c: Option<&[f64; 4]>,
    beta: f64,
) -> [f64; 4] {
    let zero = [0.0f64; 4];
    let c = c.unwrap_or(&zero);
    let mut d = [0.0f64; 4];
    for r in 0..4 {
        let t = a[(r, 0)] * b[0] + a[(r, 1)] * b[1] + a[(r, 2)] * b[2] + a[(r, 3)] * b[3];
        d[r] = t * alpha + c[r] * beta;
    }
    d
}

/// `cv::norm(v)` (`NORM_L2`) of four doubles: `sqrt` of `s += v*v` from `s = 0`, in order.
///
/// Ported from `modules/core/src/norm.dispatch.cpp`, `cv::norm`, and
/// `modules/core/src/norm.simd.hpp`, `normL2_` / `NormL2_SIMD<double, double>` (the vector
/// loop needs `4 * lanes` elements: eight with the SSE2 baseline, sixteen with the AVX2
/// dispatch the reference machine runs; so four values take the scalar tail).
fn norm_l2(v: &[f64; 4]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Method 4: for each view, the inverse of `[P; 0 0 0 1]` gives the camera centre
/// (`P_inv * (0,0,0,1)`) and a second point on the ray (`P_inv * (x,y,1,1)`); the direction
/// is their difference divided by its norm (all four components, the fourth being rounding
/// noise of the inverse). The view adds `I - d d^T` (from the first three components of d)
/// to a 3x3 `A` and `(I - d d^T) * origin` to `B`; `A f = B` is solved by
/// `cv::solve(.., DECOMP_SVD)`.
///
/// The C++ indexes the column matrices `origin`, `dir` and `B` as `(0, k)`
/// (`core/Marker.cpp:581-602`); they are continuous, so that is element `k`, as here.
///
/// Ported from `core/Marker.cpp`, `Marker::reconstruct3DPointRayIntersection` (`:526-616`).
fn ray_intersection(observations: &[Observation]) -> Point3 {
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    let tmp2 = [0.0, 0.0, 0.0, 1.0];

    for obs in observations {
        let (x, y) = (obs.point.x, obs.point.y);
        let p = &obs.projection;

        // cv::vconcat(P, (0,0,0,1), P_inv); cv::invert(P_inv, P_inv) (DECOMP_LU; a singular
        // matrix gives zeros, which the C++ does not check).
        let mut m = Matrix4::<f64>::zeros();
        for r in 0..3 {
            for c in 0..4 {
                m[(r, c)] = p[(r, c)];
            }
        }
        m[(3, 3)] = 1.0;
        let inv = invert(&m).x;
        let p_inv = Matrix4::<f64>::from_fn(|r, c| inv[(r, c)]);

        let tmp3 = [x, y, 1.0, 1.0];
        // origin = P_inv * tmp2: gemm(P_inv, tmp2, 1, Mat(), 0).
        let origin = gemm_4x4_by_4x1(&p_inv, &tmp2, 1.0, None, 0.0);
        // dir = P_inv * tmp3 - origin: gemm(P_inv, tmp3, 1, origin, -1).
        let mut dir = gemm_4x4_by_4x1(&p_inv, &tmp3, 1.0, Some(&origin), -1.0);
        // dir = dir / cv::norm(dir): MatOp_AddEx with alpha = 1./norm. When alpha is exactly
        // 1 OpenCV takes `cv::add(dir, Scalar(0))`: `v + 0`; otherwise `convertTo`, with its
        // own no-scale test.
        let nrm = norm_l2(&dir);
        if 1.0 / nrm != 1.0 {
            scale_like_convert_to(&mut dir, nrm);
        } else {
            for v in &mut dir {
                *v += 0.0;
            }
        }

        let t11 = 1.0 - dir[0] * dir[0];
        let t22 = 1.0 - dir[1] * dir[1];
        let t33 = 1.0 - dir[2] * dir[2];
        let t12 = -dir[0] * dir[1];
        let t13 = -dir[0] * dir[2];
        let t23 = -dir[1] * dir[2];

        a[0][0] += t11;
        a[0][1] += t12;
        a[0][2] += t13;

        a[1][0] += t12;
        a[1][1] += t22;
        a[1][2] += t23;

        a[2][0] += t13;
        a[2][1] += t23;
        a[2][2] += t33;

        b[0] += t11 * origin[0] + t12 * origin[1] + t13 * origin[2];
        b[1] += t12 * origin[0] + t22 * origin[1] + t23 * origin[2];
        b[2] += t13 * origin[0] + t23 * origin[1] + t33 * origin[2];
    }

    let a = DMatrix::from_fn(3, 3, |r, c| a[r][c]);
    let b = DMatrix::from_column_slice(3, 1, &b);
    let f = solve_svd(&a, &b);
    Point3::new(f[(0, 0)], f[(1, 0)], f[(2, 0)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_values_map_as_the_cpp_switch_does() {
        assert_eq!(
            TriangulationMethod::from_setting(0),
            TriangulationMethod::Zisserman
        );
        assert_eq!(
            TriangulationMethod::from_setting(1),
            TriangulationMethod::ZissermanIncremental
        );
        assert_eq!(
            TriangulationMethod::from_setting(2),
            TriangulationMethod::ZissermanMatlab
        );
        assert_eq!(
            TriangulationMethod::from_setting(3),
            TriangulationMethod::ZissermanIncrementalMatlab
        );
        assert_eq!(
            TriangulationMethod::from_setting(4),
            TriangulationMethod::RayIntersection
        );
        assert_eq!(
            TriangulationMethod::from_setting(9),
            TriangulationMethod::Zisserman
        );
        assert_eq!(
            TriangulationMethod::from_setting(-1),
            TriangulationMethod::Zisserman
        );
        assert_eq!(
            TriangulationMethod::default(),
            TriangulationMethod::ZissermanIncremental
        );
    }

    #[test]
    fn convert_to_scaling_rules() {
        // Scaled as v * (1/s) + 0, which is not always v / s.
        let mut v = [3.0, -0.0, 1.0];
        scale_like_convert_to(&mut v, 3.0);
        assert_eq!(v[0], 3.0 * (1.0 / 3.0));
        assert!(v[1] == 0.0 && v[1].is_sign_positive());
        // |1/s - 1| < DBL_EPSILON: copied unchanged.
        let mut u = [5.0, -0.0];
        scale_like_convert_to(&mut u, 1.0 + f64::EPSILON / 4.0);
        assert_eq!(u[0], 5.0);
        assert!(u[1].is_sign_negative());
    }

    #[test]
    fn gemm_and_norm_order() {
        let m = Matrix4::from_fn(|r, c| (r * 4 + c) as f64 + 0.5);
        let d = gemm_4x4_by_4x1(&m, &[1.0, 2.0, 3.0, 4.0], 1.0, None, 0.0);
        assert_eq!(d[0], 0.5 + 1.5 * 2.0 + 2.5 * 3.0 + 3.5 * 4.0);
        let e = gemm_4x4_by_4x1(&m, &[1.0, 2.0, 3.0, 4.0], 1.0, Some(&d), -1.0);
        assert_eq!(e, [0.0; 4]);
        assert_eq!(norm_l2(&[3.0, 4.0, 0.0, 0.0]), 5.0);
    }
}
