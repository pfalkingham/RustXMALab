//! The camera model: rotation conversion, projection, and lens ("model") distortion.
//!
//! Work package: geometry primitives. Ported from `core/Camera.cpp` and OpenCV 4.13's
//! `Rodrigues`, `projectPoints` and `undistortPoints` (`modules/calib3d/src/`).
//!
//! # Floating point
//!
//! Every function follows its source operation by operation, in the same order, at the same
//! width, and never fuses a multiply and an add. Where the C++ goes through a `cv::Mat`
//! product, the summation order of OpenCV's `gemm` for that shape is reproduced (see
//! [`projection_matrix`] and [`CameraModel::project_point`]).
//!
//! # Distortion coefficients
//!
//! OpenCV order throughout: `k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y`.
//! XMALab always holds eight (`core/Camera.cpp:67`, the rational model of
//! `processing/Calibration.cpp`). The OpenCV ports here also take 4, 5, 12 or 14, as OpenCV
//! does, but not the tilted model: a non-zero `tau_x` or `tau_y` panics (XMALab never sets
//! them, and `computeTiltProjectionMatrix` is not ported).

use crate::undistortion::{GridModel, transform_point};
use crate::{Matrix3, Matrix3x4, Point2, Point3, Vector3};

/// Camera matrix and lens distortion.
#[derive(Clone, Debug, PartialEq)]
pub struct Intrinsics {
    pub camera_matrix: Matrix3,
    /// Distortion coefficients in OpenCV order (k1, k2, p1, p2, k3, k4, k5, k6), or `None`
    /// when the camera has no model distortion (C++ `hasModelDistortion() == false`).
    pub distortion: Option<[f64; 8]>,
}

/// A rigid transform as XMALab stores it: a Rodrigues rotation vector and a translation.
/// For a camera this maps world coordinates to camera coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidTransform {
    pub rotation: Vector3,
    pub translation: Vector3,
}

/// Everything needed to move a point between the raw image and the ideal pinhole image.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraModel {
    pub intrinsics: Intrinsics,
    /// The fitted undistortion-grid model, when the camera has a computed grid.
    pub grid: Option<GridModel>,
    /// Which faults of XMALab 3 to reproduce when moving points (ledger A11, A19).
    pub compat: ModelCompat,
}

/// Which faults of XMALab 3's `Camera::undistortPoint` to reproduce (parity ledger A11,
/// A19; decided 14 November 2026: fixed by default). Each field is `true` for the C++
/// behaviour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelCompat {
    /// A11: distorting a point applies the grid first and the lens model second, the same
    /// order as undistorting. Fixed: the lens model first, then the grid, the inverse of
    /// undistorting.
    pub grid_first_when_distorting: bool,
    /// A19: removing lens distortion runs `cv::undistortPoints`' default 5 iterations.
    /// Fixed: it iterates until converged ([`TermCriteria::CONVERGED`]).
    pub five_iterations: bool,
}

impl ModelCompat {
    /// Both faults fixed: the default.
    pub const FIXED: ModelCompat = ModelCompat {
        grid_first_when_distorting: false,
        five_iterations: false,
    };

    /// Both as XMALab 3 does it.
    pub const XMALAB3: ModelCompat = ModelCompat {
        grid_first_when_distorting: true,
        five_iterations: true,
    };
}

/// Termination criteria of [`undistort_points`]: OpenCV's `cv::TermCriteria` as
/// `cvUndistortPointsInternal` reads it.
///
/// `max_count: Some(n)` is the `COUNT` flag with `maxCount = n`; `epsilon: Some(e)` is the
/// `EPS` flag with `epsilon = e`.
///
/// Validity is OpenCV's `TermCriteria::isValid()`
/// (`modules/core/include/opencv2/core/types.hpp:914-919`): `COUNT` with `maxCount > 0`, or
/// `EPS` with an `epsilon` that is not NaN. [`undistort_points`] panics otherwise, as OpenCV's
/// `CV_Assert` throws. One usable criterion is enough, and the other one still takes part
/// in the iteration exactly as in OpenCV:
///
/// - `max_count: Some(n)` with `n <= 0` beside a valid `epsilon` stops before the first
///   iteration (`j >= maxCount` at `j = 0`), returning the normalised point;
/// - an `epsilon` that is negative or NaN beside a valid `max_count` never stops the loop
///   (`error < epsilon` is false), so exactly `max_count` iterations run; the error is
///   still computed each iteration and still drives the damping.
///
/// With only `epsilon` set (or `max_count` unusable), the iteration runs until the
/// reprojection error falls below `epsilon`. If it never does, OpenCV loops forever, and so
/// does this faithful port: give a `max_count` unless convergence is certain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TermCriteria {
    pub max_count: Option<i32>,
    pub epsilon: Option<f64>,
}

impl TermCriteria {
    /// The stopping rule used to remove lens distortion when ledger A19 is fixed: iterate
    /// until the point, distorted again, lies within 1e-10 pixel of the observed point
    /// (OpenCV's `EPS` test on the reprojection error, which also turns on its step
    /// damping), and at most 1000 iterations. 1e-10 pixel is about a thousand times the
    /// rounding of a coordinate near 1000 pixels, so it is reached wherever the iteration
    /// converges; the count only bounds an iteration that does not.
    pub const CONVERGED: TermCriteria = TermCriteria {
        max_count: Some(1000),
        epsilon: Some(1e-10),
    };
}

impl Default for TermCriteria {
    /// The default of the six-argument `cv::undistortPoints`, which XMALab uses
    /// (`core/Camera.cpp:695`): `TermCriteria(MAX_ITER, 5, 0.01)`, so only the count is
    /// active and the epsilon is ignored (parity ledger, item A19).
    fn default() -> Self {
        TermCriteria {
            max_count: Some(5),
            epsilon: None,
        }
    }
}

/// Rotation vector to rotation matrix.
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`, `cv::Rodrigues`
/// (vector to matrix, `CV_64F`, no Jacobian). Tier A.
/// A vector shorter than `DBL_EPSILON` gives the identity.
/// Oracle: primitive cases for `Rodrigues`, including rotations near 0 and near pi.
#[must_use]
pub fn rodrigues_to_matrix(rotation: &Vector3) -> Matrix3 {
    let (mut rx, mut ry, mut rz) = (rotation[0], rotation[1], rotation[2]);
    // norm(Point3d): sqrt(x*x + y*y + z*z).
    let theta = (rx * rx + ry * ry + rz * rz).sqrt();
    if theta < f64::EPSILON {
        return Matrix3::identity();
    }
    let c = theta.cos();
    let s = theta.sin();
    let c1 = 1.0 - c;
    let itheta = if theta != 0.0 { 1.0 / theta } else { 0.0 };

    // r *= itheta
    rx *= itheta;
    ry *= itheta;
    rz *= itheta;

    // Row-major Matx33d values.
    let rrt = [
        rx * rx,
        rx * ry,
        rx * rz,
        rx * ry,
        ry * ry,
        ry * rz,
        rx * rz,
        ry * rz,
        rz * rz,
    ];
    let r_x = [0.0, -rz, ry, rz, 0.0, -rx, -ry, rx, 0.0];
    let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    // R = c*I + c1*rrt + s*r_x, element by element (Matx_ScaleOp then Matx_AddOp, left to
    // right).
    let mut out = [0.0; 9];
    for i in 0..9 {
        out[i] = (eye[i] * c + rrt[i] * c1) + r_x[i] * s;
    }
    Matrix3::from_row_slice(&out)
}

/// Rotation matrix to rotation vector.
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`, `cv::Rodrigues`
/// (matrix to vector, `CV_64F`, no Jacobian), including:
///
/// - the range check: if any element is NaN or outside `[-100, 100)`
///   (`checkRange(R, true, NULL, -100, 100)`) the result is (0, 0, 0);
/// - the clean-up of the input with OpenCV's Jacobi SVD, `R = U * Vt`;
/// - the special handling when `sin(theta) < 1e-5`: zero for angles near 0, and the
///   diagonal-based axis with its sign rules for angles near pi, which the rigid-body code
///   relies on.
///
/// Tier A. Oracle: primitive cases for `Rodrigues`.
#[must_use]
pub fn matrix_to_rodrigues(matrix: &Matrix3) -> Vector3 {
    // checkRange with CV_TOGGLE_DBL compares as ordered doubles: v >= -100 && v < 100,
    // NaN failing.
    if matrix.iter().any(|&v| !(-100.0..100.0).contains(&v)) {
        return Vector3::zeros();
    }

    let usv = crate::linalg::svd(matrix, crate::linalg::SvdMode::Thin);
    // R = U*Vt, Matx_MatMulOp: s = 0; s += a(i,k)*b(k,j).
    let mut r = [[0.0f64; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in 0..3 {
                s += usv.u[(i, k)] * usv.vt[(k, j)];
            }
            *v = s;
        }
    }

    let mut rx = r[2][1] - r[1][2];
    let mut ry = r[0][2] - r[2][0];
    let mut rz = r[1][0] - r[0][1];

    let s = ((rx * rx + ry * ry + rz * rz) * 0.25).sqrt();
    let mut c = (r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5;
    // c > 1 ? 1 : c < -1 ? -1 : c; identical to clamp, NaN included.
    c = c.clamp(-1.0, 1.0);
    let mut theta = c.acos();

    if s < 1e-5 {
        if c > 0.0 {
            rx = 0.0;
            ry = 0.0;
            rz = 0.0;
        } else {
            let mut t = (r[0][0] + 1.0) * 0.5;
            rx = t.max(0.0).sqrt();
            t = (r[1][1] + 1.0) * 0.5;
            ry = t.max(0.0).sqrt() * if r[0][1] < 0.0 { -1.0 } else { 1.0 };
            t = (r[2][2] + 1.0) * 0.5;
            rz = t.max(0.0).sqrt() * if r[0][2] < 0.0 { -1.0 } else { 1.0 };
            if rx.abs() < ry.abs() && rx.abs() < rz.abs() && (r[1][2] > 0.0) != (ry * rz > 0.0) {
                rz = -rz;
            }
            theta /= (rx * rx + ry * ry + rz * rz).sqrt();
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    } else {
        let mut vth = 1.0 / (2.0 * s);
        vth *= theta;
        rx *= vth;
        ry *= vth;
        rz *= vth;
    }
    Vector3::new(rx, ry, rz)
}

/// The rotation vector of the rotation nearest to `matrix` (`U Vt` of its SVD, as
/// [`matrix_to_rodrigues`] orthogonalises it), exact to rounding for every axis and angle:
/// [`rodrigues_to_matrix`] of the result gives that rotation back to a few units in the last
/// place. `None` when `matrix` has a value that is not finite, or its nearest orthogonal
/// matrix is a reflection (determinant not positive), which no rotation vector describes.
///
/// Not a port: [`matrix_to_rodrigues`] is OpenCV's `cv::Rodrigues`, whose branches for
/// `sin(theta) < 1e-5` give the zero vector for a rotation by up to about 1e-5 rad and an
/// approximate axis within about 1e-5 rad of 180 degrees (parity ledger A165). Here the
/// rotation goes through its unit quaternion (Shepperd's method: from the largest of the
/// trace and the diagonal, so that no component is found by cancellation), and the angle
/// is `2 atan2(|v|, w)`, accurate at 0 and at 180 degrees alike. The angle is at most 180
/// degrees.
#[must_use]
pub fn nearest_rotation_to_rodrigues(matrix: &Matrix3) -> Option<Vector3> {
    if matrix.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let usv = crate::linalg::svd(matrix, crate::linalg::SvdMode::Thin);
    let mut r = [[0.0f64; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in 0..3 {
                s += usv.u[(i, k)] * usv.vt[(k, j)];
            }
            *v = s;
        }
    }
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if det.is_nan() || det <= 0.0 {
        return None;
    }
    // Shepperd: (w, x, y, z) with the largest of 1 + trace and 1 + 2 r_ii - trace first.
    let tr = r[0][0] + r[1][1] + r[2][2];
    let (w, x, y, z);
    if tr >= r[0][0] && tr >= r[1][1] && tr >= r[2][2] {
        let t = (1.0 + tr).sqrt() * 2.0; // 4 w
        w = 0.25 * t;
        x = (r[2][1] - r[1][2]) / t;
        y = (r[0][2] - r[2][0]) / t;
        z = (r[1][0] - r[0][1]) / t;
    } else if r[0][0] >= r[1][1] && r[0][0] >= r[2][2] {
        let t = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0; // 4 x
        w = (r[2][1] - r[1][2]) / t;
        x = 0.25 * t;
        y = (r[0][1] + r[1][0]) / t;
        z = (r[0][2] + r[2][0]) / t;
    } else if r[1][1] >= r[2][2] {
        let t = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0; // 4 y
        w = (r[0][2] - r[2][0]) / t;
        x = (r[0][1] + r[1][0]) / t;
        y = 0.25 * t;
        z = (r[1][2] + r[2][1]) / t;
    } else {
        let t = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0; // 4 z
        w = (r[1][0] - r[0][1]) / t;
        x = (r[0][2] + r[2][0]) / t;
        y = (r[1][2] + r[2][1]) / t;
        z = 0.25 * t;
    }
    // The same rotation with w >= 0, so that the angle is at most 180 degrees.
    let sign = if w < 0.0 { -1.0 } else { 1.0 };
    let (w, x, y, z) = (w * sign, x * sign, y * sign, z * sign);
    let n = (x * x + y * y + z * z).sqrt();
    if n == 0.0 {
        return Some(Vector3::zeros());
    }
    let theta = 2.0 * n.atan2(w);
    let k = theta / n;
    Some(Vector3::new(x * k, y * k, z * k))
}

/// The 3x4 projection matrix of a camera for one calibration image: `K [R | t]`.
///
/// Ported from `core/Camera.cpp:389-404`, `Camera::getProjectionMatrix`. `R` comes from
/// [`rodrigues_to_matrix`]; the product `cameramatrix * tmp` (3x3 by 3x4) goes through
/// OpenCV's `gemmImpl` small-matrix branch (`modules/core/src/matmul.simd.hpp`, `len == 3`,
/// `len == d_size.height`), which sums `K(i,0)*T(0,j) + K(i,1)*T(1,j) + K(i,2)*T(2,j)` left to
/// right and stores `t*alpha + c*beta` with `alpha = 1`, `beta = 0`. Tier A.
/// Oracle: `trial_<t>/projection_matrices.csv`.
#[must_use]
pub fn projection_matrix(camera_matrix: &Matrix3, pose: &RigidTransform) -> Matrix3x4 {
    let r = rodrigues_to_matrix(&pose.rotation);
    let mut tmp = Matrix3x4::zeros();
    for i in 0..3 {
        for j in 0..3 {
            tmp[(i, j)] = r[(i, j)];
        }
        tmp[(i, 3)] = pose.translation[i];
    }
    let k = camera_matrix;
    let mut p = Matrix3x4::zeros();
    for j in 0..4 {
        for i in 0..3 {
            let t = k[(i, 0)] * tmp[(0, j)] + k[(i, 1)] * tmp[(1, j)] + k[(i, 2)] * tmp[(2, j)];
            // d = t*alpha + c*beta, with alpha = 1, c = 0, beta = 0.
            p[(i, j)] = t * 1.0 + 0.0 * 0.0;
        }
    }
    p
}

/// Checks a distortion vector's length as OpenCV does and rejects the tilted model.
fn check_distortion(distortion: &[f64]) {
    assert!(
        matches!(distortion.len(), 4 | 5 | 8 | 12 | 14),
        "distortion coefficients must have 4, 5, 8, 12 or 14 elements, got {}",
        distortion.len()
    );
    if distortion.len() == 14 {
        assert!(
            distortion[12] == 0.0 && distortion[13] == 0.0,
            "the tilted distortion model (tau_x, tau_y != 0) is not ported"
        );
    }
}

/// Copies distortion coefficients into OpenCV's zero-padded `k[14]`.
fn padded(distortion: &[f64]) -> [f64; 14] {
    let mut k = [0.0; 14];
    k[..distortion.len()].copy_from_slice(distortion);
    k
}

/// `Matx33d::eye() * Vec3d(a, b, 1)` with `Matx_MatMulOp` (s = 0; s += m(i,k)*v(k)): the
/// untilted tilt step of `undistortPoints` and `projectPoints`.
fn eye_times(a: f64, b: f64) -> [f64; 3] {
    const EYE: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let v = [a, b, 1.0];
    let mut out = [0.0; 3];
    for i in 0..3 {
        let mut s = 0.0;
        for k in 0..3 {
            s += EYE[i][k] * v[k];
        }
        out[i] = s;
    }
    out
}

/// Removes lens distortion from points, returning normalised coordinates (no camera matrix
/// re-applied): OpenCV `cv::undistortPoints` with no `R` and no `P`.
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/undistort.dispatch.cpp`,
/// `cvUndistortPointsInternal`, double precision, including the damped fixed-point
/// iteration, the early exit when the rational factor goes negative, and the `EPS` branch.
/// The skew entry `camera_matrix[(0, 1)]` is ignored, as in OpenCV.
///
/// An empty `distortion` is OpenCV's empty `distCoeffs`: the points are only normalised,
/// with no iteration. Otherwise it must have 4, 5, 8, 12 or 14 elements. XMALab calls this
/// with eight coefficients and [`TermCriteria::default`] (`core/Camera.cpp:695`).
/// Tier B. Oracle: primitive cases for `undistortPoints`.
#[must_use]
pub fn undistort_points(
    points: &[Point2],
    camera_matrix: &Matrix3,
    distortion: &[f64],
    criteria: TermCriteria,
) -> Vec<Point2> {
    // CV_Assert(criteria.isValid()): (COUNT && maxCount > 0) || (EPS && !isnan(epsilon)).
    // Either criterion alone suffices; the other keeps its flag and is tested in the loop
    // as it is.
    let count_valid = matches!(criteria.max_count, Some(n) if n > 0);
    let eps_valid = matches!(criteria.epsilon, Some(e) if !e.is_nan());
    assert!(count_valid || eps_valid, "invalid termination criteria");
    let has_dist = !distortion.is_empty();
    if has_dist {
        check_distortion(distortion);
    }
    let k = padded(distortion);

    let fx = camera_matrix[(0, 0)];
    let fy = camera_matrix[(1, 1)];
    let ifx = 1.0 / fx;
    let ify = 1.0 / fy;
    let cx = camera_matrix[(0, 2)];
    let cy = camera_matrix[(1, 2)];

    points
        .iter()
        .map(|p| {
            let (mut x, mut y) = (p.x, p.y);
            let (u, v) = (x, y);
            x = (x - cx) * ifx;
            y = (y - cy) * ify;

            if has_dist {
                let vec_untilt = eye_times(x, y);
                let mut inv_proj = if vec_untilt[2] != 0.0 {
                    1.0 / vec_untilt[2]
                } else {
                    1.0
                };
                x = inv_proj * vec_untilt[0];
                y = inv_proj * vec_untilt[1];
                let x0 = x;
                let y0 = y;

                let mut error = f64::MAX;
                let mut prev_error = f64::MAX;
                let mut alpha = 1.0;

                let mut j: i32 = 0;
                loop {
                    if let Some(n) = criteria.max_count
                        && j >= n
                    {
                        break;
                    }
                    if let Some(eps) = criteria.epsilon
                        && error < eps
                    {
                        break;
                    }
                    let r2 = x * x + y * y;
                    let icdist = (1.0 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2)
                        / (1.0 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
                    if icdist < 0.0 {
                        x = (u - cx) * ifx;
                        y = (v - cy) * ify;
                        break;
                    }
                    let delta_x =
                        2.0 * k[2] * x * y + k[3] * (r2 + 2.0 * x * x) + k[8] * r2 + k[9] * r2 * r2;
                    let delta_y = k[2] * (r2 + 2.0 * y * y)
                        + 2.0 * k[3] * x * y
                        + k[10] * r2
                        + k[11] * r2 * r2;
                    let new_x = (1.0 - alpha) * x + alpha * (x0 - delta_x) * icdist;
                    let new_y = (1.0 - alpha) * y + alpha * (y0 - delta_y) * icdist;

                    if criteria.epsilon.is_some() {
                        let r2 = new_x * new_x + new_y * new_y;
                        let r4 = r2 * r2;
                        let r6 = r4 * r2;
                        let a1 = 2.0 * new_x * new_y;
                        let a2 = r2 + 2.0 * new_x * new_x;
                        let a3 = r2 + 2.0 * new_y * new_y;
                        let cdist = 1.0 + k[0] * r2 + k[1] * r4 + k[4] * r6;
                        let icdist2 = 1.0 / (1.0 + k[5] * r2 + k[6] * r4 + k[7] * r6);
                        let xd0 =
                            new_x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
                        let yd0 = new_y * cdist * icdist2
                            + k[2] * a3
                            + k[3] * a1
                            + k[10] * r2
                            + k[11] * r4;
                        let vec_tilt = eye_times(xd0, yd0);
                        inv_proj = if vec_tilt[2] != 0.0 {
                            1.0 / vec_tilt[2]
                        } else {
                            1.0
                        };
                        let xd = inv_proj * vec_tilt[0];
                        let yd = inv_proj * vec_tilt[1];
                        let x_proj = xd * fx + cx;
                        let y_proj = yd * fy + cy;
                        // sqrt(pow(dx, 2) + pow(dy, 2)); GCC folds pow(d, 2) to d*d.
                        let dx = x_proj - u;
                        let dy = y_proj - v;
                        error = (dx * dx + dy * dy).sqrt();
                    }
                    if error > prev_error {
                        alpha *= 0.5;
                    } else {
                        x = new_x;
                        y = new_y;
                    }
                    prev_error = error;
                    j += 1;
                }
            }
            Point2::new(x, y)
        })
        .collect()
}

/// Projects object points into the image: OpenCV `cv::projectPoints` (points only, no
/// Jacobian, no aspect-ratio constraint), double precision.
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`, `cv::projectPoints`
/// (the generic per-point loop, which is what runs: its HAL shortcut is never taken because
/// it tests `ipoints.type() == CV_32F` against a two-channel output). The rotation goes
/// through [`rodrigues_to_matrix`]; `z == 0` is replaced by 1 before dividing, as in OpenCV.
/// The skew entry `camera_matrix[(0, 1)]` is ignored.
///
/// An empty `distortion` is OpenCV's empty `distCoeffs`, which it replaces by five zeros;
/// otherwise it must have 4, 5, 8, 12 or 14 elements.
/// For single-precision input and output, as `core/RigidBody.cpp:2418-2439` uses it, see
/// [`project_points_f32`]. Tier A. Oracle: primitive cases for `projectPoints`.
#[must_use]
pub fn project_points(
    object_points: &[Point3],
    rvec: &Vector3,
    tvec: &Vector3,
    camera_matrix: &Matrix3,
    distortion: &[f64],
) -> Vec<Point2> {
    if !distortion.is_empty() {
        check_distortion(distortion);
    }
    let k = padded(distortion);
    let rm = rodrigues_to_matrix(rvec);
    let r = [
        rm[(0, 0)],
        rm[(0, 1)],
        rm[(0, 2)],
        rm[(1, 0)],
        rm[(1, 1)],
        rm[(1, 2)],
        rm[(2, 0)],
        rm[(2, 1)],
        rm[(2, 2)],
    ];
    let t = [tvec[0], tvec[1], tvec[2]];
    let fx = camera_matrix[(0, 0)];
    let fy = camera_matrix[(1, 1)];
    let cx = camera_matrix[(0, 2)];
    let cy = camera_matrix[(1, 2)];

    object_points
        .iter()
        .map(|m| {
            let (xw, yw, zw) = (m.x, m.y, m.z);
            let mut x = r[0] * xw + r[1] * yw + r[2] * zw + t[0];
            let mut y = r[3] * xw + r[4] * yw + r[5] * zw + t[1];
            let mut z = r[6] * xw + r[7] * yw + r[8] * zw + t[2];

            z = if z != 0.0 { 1.0 / z } else { 1.0 };
            x *= z;
            y *= z;

            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let a1 = 2.0 * x * y;
            let a2 = r2 + 2.0 * x * x;
            let a3 = r2 + 2.0 * y * y;
            let cdist = 1.0 + k[0] * r2 + k[1] * r4 + k[4] * r6;
            let icdist2 = 1.0 / (1.0 + k[5] * r2 + k[6] * r4 + k[7] * r6);
            let xd0 = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
            let yd0 = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;

            let vec_tilt = eye_times(xd0, yd0);
            let inv_proj = if vec_tilt[2] != 0.0 {
                1.0 / vec_tilt[2]
            } else {
                1.0
            };
            let xd = inv_proj * vec_tilt[0];
            let yd = inv_proj * vec_tilt[1];

            Point2::new(xd * fx + cx, yd * fy + cy)
        })
        .collect()
}

/// [`project_points`] with single-precision object and image points, as
/// `core/RigidBody.cpp:2418-2439` calls `cv::projectPoints` (`std::vector<cv::Point3f>` in,
/// `std::vector<cv::Point2f>` out, an empty `std::vector<float>` of distortion
/// coefficients, double `rvec`, `tvec` and camera matrix).
///
/// OpenCV widens the `float` object points to `double`, computes in double, and narrows
/// the result to `float` (`_m.convertTo(_imagePoints, CV_32F)`, round to nearest); so does
/// this. The caller narrows its own `double` points to `f32` first, as the C++ does when it
/// builds the `cv::Point3f`s. `distortion` holds the caller's coefficients widened to
/// `f64` (empty in XMALab).
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`, `cv::projectPoints`.
/// Tier A. Oracle: primitive case `projectpoints-f32_nodist_cpp`.
#[must_use]
pub fn project_points_f32(
    object_points: &[nalgebra::Point3<f32>],
    rvec: &Vector3,
    tvec: &Vector3,
    camera_matrix: &Matrix3,
    distortion: &[f64],
) -> Vec<nalgebra::Point2<f32>> {
    let wide: Vec<Point3> = object_points
        .iter()
        .map(|p| Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z)))
        .collect();
    project_points(&wide, rvec, tvec, camera_matrix, distortion)
        .into_iter()
        .map(|p| nalgebra::Point2::new(p.x as f32, p.y as f32))
        .collect()
}

/// Applies (`undistort == false`) or removes (`undistort == true`) lens distortion, in pixels.
///
/// Ported from `core/Camera.cpp:686-713`, `Camera::applyModelDistortion`.
///
/// - Removing calls [`undistort_points`] (OpenCV `cv::undistortPoints` with the full
///   camera matrix, the eight coefficients and the default five iterations, ledger A19) and
///   re-applies `fx * x + cx`, `fy * y + cy` (no skew). Tier B.
/// - Applying is the C++'s own closed form of the rational model (not OpenCV's): it ignores
///   the skew, and evaluates `kr` in Horner form. Tier A.
///
/// Returns the point unchanged when `intrinsics.distortion` is `None` (the C++ never calls
/// this then: `Camera::undistortPoint` checks `hasModelDistortion()` first).
/// Oracle: primitive cases for `undistortPoints`; `undistortion/camera_undistort_samples.csv`.
#[must_use]
pub fn apply_model_distortion(intrinsics: &Intrinsics, point: Point2, undistort: bool) -> Point2 {
    apply_model_distortion_with(intrinsics, point, undistort, TermCriteria::default())
}

/// [`apply_model_distortion`] with the termination criteria of the removal (ledger A19:
/// XMALab 3 uses [`TermCriteria::default`], the fixed program
/// [`TermCriteria::CONVERGED`]). Applying distortion is closed-form and does not read
/// `criteria`.
///
/// With [`TermCriteria::CONVERGED`] the removal is [`undistort_lens_converged`], and a
/// point it finds no result for comes back as (NaN, NaN); use that function to get the
/// `None` instead.
#[must_use]
pub fn apply_model_distortion_with(
    intrinsics: &Intrinsics,
    point: Point2,
    undistort: bool,
    criteria: TermCriteria,
) -> Point2 {
    let Some(d) = intrinsics.distortion else {
        return point;
    };
    let k = &intrinsics.camera_matrix;
    if undistort && criteria == TermCriteria::CONVERGED {
        return undistort_lens_converged(intrinsics, point)
            .unwrap_or_else(|| Point2::new(f64::NAN, f64::NAN));
    }
    if undistort {
        let n = undistort_points(&[point], k, &d, criteria)[0];
        Point2::new(k[(0, 0)] * n.x + k[(0, 2)], k[(1, 1)] * n.y + k[(1, 2)])
    } else {
        let x = (point.x - k[(0, 2)]) / k[(0, 0)];
        let y = (point.y - k[(1, 2)]) / k[(1, 1)];

        let x2 = x * x;
        let y2 = y * y;
        let r2 = x2 + y2;
        let xy2 = 2.0 * x * y;

        let kr = (1.0 + ((d[4] * r2 + d[1]) * r2 + d[0]) * r2)
            / (1.0 + ((d[7] * r2 + d[6]) * r2 + d[5]) * r2);
        Point2::new(
            k[(0, 0)] * (x * kr + d[2] * xy2 + d[3] * (r2 + 2.0 * x2)) + k[(0, 2)],
            k[(1, 1)] * (y * kr + d[2] * (r2 + 2.0 * y2) + d[3] * xy2) + k[(1, 2)],
        )
    }
}

/// The reprojection tolerance of the fixed lens undistortion, in pixels: the iteration's
/// stopping rule ([`TermCriteria::CONVERGED`]) and the test of its result.
pub const UNDISTORT_TOLERANCE_PX: f64 = 1e-10;

/// The tolerance a result of the Newton fallback of [`undistort_lens_converged`] must meet,
/// in pixels: ten times [`UNDISTORT_TOLERANCE_PX`], so that a root found to the last bits
/// is not refused for rounding in the reprojection.
pub const UNDISTORT_ACCEPT_PX: f64 = 1e-9;

/// The lens model in normalised coordinates, `(x, y)` to the distorted `(x', y')`, and its
/// Jacobian: OpenCV's rational model with k1, k2, k3 over k4, k5, k6 and the tangential
/// p1, p2 (the eight coefficients XMALab holds), evaluated as [`apply_model_distortion`]
/// evaluates it.
fn lens_forward(d: &[f64; 8], x: f64, y: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    let x2 = x * x;
    let y2 = y * y;
    let r2 = x2 + y2;
    let xy2 = 2.0 * x * y;
    let num = 1.0 + ((d[4] * r2 + d[1]) * r2 + d[0]) * r2;
    let den = 1.0 + ((d[7] * r2 + d[6]) * r2 + d[5]) * r2;
    let kr = num / den;
    let f = [
        x * kr + d[2] * xy2 + d[3] * (r2 + 2.0 * x2),
        y * kr + d[2] * (r2 + 2.0 * y2) + d[3] * xy2,
    ];
    // d kr / d r2.
    let dnum = d[0] + (2.0 * d[1] + 3.0 * d[4] * r2) * r2;
    let dden = d[5] + (2.0 * d[6] + 3.0 * d[7] * r2) * r2;
    let dkr = (dnum * den - num * dden) / (den * den);
    let j = [
        [
            kr + 2.0 * x2 * dkr + 2.0 * d[2] * y + 6.0 * d[3] * x,
            2.0 * x * y * dkr + 2.0 * d[2] * x + 2.0 * d[3] * y,
        ],
        [
            2.0 * x * y * dkr + 2.0 * d[2] * x + 2.0 * d[3] * y,
            kr + 2.0 * y2 * dkr + 6.0 * d[2] * y + 2.0 * d[3] * x,
        ],
    ];
    (f, j)
}

/// The reprojection error in pixels of the normalised point `(x, y)` against the
/// normalised observation `(u, v)`, with focal lengths `fx`, `fy`.
fn lens_error(d: &[f64; 8], x: f64, y: f64, u: f64, v: f64, fx: f64, fy: f64) -> f64 {
    let (f, _) = lens_forward(d, x, y);
    let ex = fx * (f[0] - u);
    let ey = fy * (f[1] - v);
    (ex * ex + ey * ey).sqrt()
}

/// Newton's method on the lens model from `(x, y)` towards the normalised observation
/// `(u, v)`; `None` when the Jacobian is singular or not finite, or, with `keep_orientation`,
/// when an iterate leaves the region where the model keeps its orientation (determinant not
/// positive) or the residual does not fall below 1e-13 (normalised). At most `iterations`
/// steps; stops when the step is below 1e-15 (normalised).
fn lens_newton(
    d: &[f64; 8],
    mut x: f64,
    mut y: f64,
    (u, v): (f64, f64),
    iterations: usize,
    keep_orientation: bool,
) -> Option<(f64, f64)> {
    for _ in 0..iterations {
        let (f, j) = lens_forward(d, x, y);
        let (rx, ry) = (f[0] - u, f[1] - v);
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det == 0.0 || !det.is_finite() || (keep_orientation && det < 0.0) {
            return None;
        }
        if rx.abs() < 1e-15 && ry.abs() < 1e-15 {
            break;
        }
        let dx = (j[1][1] * rx - j[0][1] * ry) / det;
        let dy = (-j[1][0] * rx + j[0][0] * ry) / det;
        if !(dx.is_finite() && dy.is_finite()) {
            return None;
        }
        x -= dx;
        y -= dy;
        if dx.abs() < 1e-15 && dy.abs() < 1e-15 {
            break;
        }
    }
    if keep_orientation {
        let (f, j) = lens_forward(d, x, y);
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det <= 0.0 || (f[0] - u).abs() > 1e-13 || (f[1] - v).abs() > 1e-13 {
            return None;
        }
    }
    Some((x, y))
}

/// True when the lens model keeps its orientation (positive Jacobian determinant) on the
/// whole segment from the centre to `(x, y)`: the point lies on the principal branch, the
/// one connected to the undistorted centre.
fn on_principal_branch(d: &[f64; 8], x: f64, y: f64) -> bool {
    (0..=128).all(|i| {
        let t = f64::from(i) / 128.0;
        let (_, j) = lens_forward(d, t * x, t * y);
        j[0][0] * j[1][1] - j[0][1] * j[1][0] > 0.0
    })
}

/// Removes lens distortion from a pixel position until it has converged, or finds that it
/// cannot (parity ledger A19, fixed).
///
/// 1. OpenCV's iteration ([`undistort_points`]) with [`TermCriteria::CONVERGED`]: until the
///    point, distorted again, lies within [`UNDISTORT_TOLERANCE_PX`] of the observation, at
///    most 1000 iterations. Its result is taken when it has converged and the model keeps
///    its orientation (positive Jacobian determinant) on the straight segment from the
///    centre to it, which is the case for every point of ordinary cameras; a converged
///    result that fails this test goes to step 3, so that a root beyond the fold is not
///    given.
/// 2. If its reprojection error is still at or above the tolerance (the damped fixed-point
///    iteration can stall, for example near the edge of a camera with a rational term and
///    strong tangential distortion): Newton's method on the lens model with its analytic
///    Jacobian, from that iterate, accepted when the result reprojects within
///    [`UNDISTORT_ACCEPT_PX`] and the Jacobian keeps its sign on the straight segment from
///    the centre to it.
/// 3. Otherwise Newton's method with continuation: the target moves from the image centre
///    to the observation along the straight segment in the distorted image (400 steps),
///    and the root is followed from the centre, so it stays on the principal branch (the
///    one connected to the undistorted centre); refused if an iterate meets the fold of
///    the model (Jacobian determinant not positive), a step does not converge (residual
///    above 1e-13, normalised), or the final point does not reproject within
///    [`UNDISTORT_ACCEPT_PX`].
/// 4. Otherwise `None`: the point has no preimage on the principal branch (it lies beyond
///    the fold of a strongly distorted lens model) and no undistorted position is given,
///    instead of a wrong one.
///
/// Returns the point unchanged when `intrinsics.distortion` is `None`.
#[must_use]
pub fn undistort_lens_converged(intrinsics: &Intrinsics, point: Point2) -> Option<Point2> {
    let Some(d) = intrinsics.distortion else {
        return Some(point);
    };
    let k = &intrinsics.camera_matrix;
    let (fx, fy, cx, cy) = (k[(0, 0)], k[(1, 1)], k[(0, 2)], k[(1, 2)]);
    let n = undistort_points(&[point], k, &d, TermCriteria::CONVERGED)[0];
    let to_px = |x: f64, y: f64| Point2::new(fx * x + cx, fy * y + cy);
    let (u, v) = ((point.x - cx) / fx, (point.y - cy) / fy);
    if !(u.is_finite() && v.is_finite()) {
        return None;
    }
    let converged = lens_error(&d, n.x, n.y, u, v, fx, fy) < UNDISTORT_TOLERANCE_PX;
    if converged && on_principal_branch(&d, n.x, n.y) {
        return Some(to_px(n.x, n.y));
    }
    let accept = |(x, y): (f64, f64)| lens_error(&d, x, y, u, v, fx, fy) < UNDISTORT_ACCEPT_PX;
    // Newton from the iterate, kept only if it lands where the model keeps its orientation
    // all the way from the centre (so it cannot be a root beyond the fold).
    if !converged
        && n.x.is_finite()
        && n.y.is_finite()
        && let Some(r) = lens_newton(&d, n.x, n.y, (u, v), 50, false)
            .filter(|r| accept(*r) && on_principal_branch(&d, r.0, r.1))
    {
        return Some(to_px(r.0, r.1));
    }
    // Continuation from the centre: the observation is approached along the straight
    // segment in the distorted image, and the root is followed from the centre, so the
    // result is on the principal branch by construction.
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let steps = 400;
    for s in 1..=steps {
        let t = f64::from(s) / f64::from(steps);
        let (nx, ny) = lens_newton(&d, x, y, (t * u, t * v), 60, true)?;
        x = nx;
        y = ny;
    }
    accept((x, y)).then(|| to_px(x, y))
}

impl CameraModel {
    /// [`CameraModel::undistort_point`], with `None` where the fixed lens undistortion finds
    /// no result ([`undistort_lens_converged`], ledger A19). Always `Some` in the XMALab 3
    /// mode ([`ModelCompat::five_iterations`]) and when distorting.
    #[must_use]
    pub fn try_undistort_point(
        &self,
        point: Point2,
        undistort: bool,
        with_model: bool,
        with_refine: bool,
    ) -> Option<Point2> {
        let q = self.undistort_point(point, undistort, with_model, with_refine);
        if undistort
            && !self.compat.five_iterations
            && self.intrinsics.distortion.is_some()
            && with_model
            && !(q.x.is_finite() && q.y.is_finite())
            && point.x.is_finite()
            && point.y.is_finite()
        {
            return None;
        }
        Some(q)
    }

    /// Moves a point between the raw image and the ideal image.
    ///
    /// Ported from `core/Camera.cpp:668-680`, `Camera::undistortPoint`. Undistorting applies
    /// the grid transform first and the lens model second. XMALab 3 distorts in the same
    /// order ([`ModelCompat::grid_first_when_distorting`], ledger A11); fixed, distorting
    /// applies the lens model first, the inverse. Removing lens distortion iterates 5 times
    /// in XMALab 3 and until converged when fixed ([`ModelCompat::five_iterations`], ledger
    /// A19, [`undistort_lens_converged`]); where the fixed removal has no result the point
    /// comes back as (NaN, NaN) ([`CameraModel::try_undistort_point`] gives `None`).
    /// `with_model == false` skips the lens model; `with_refine == false` skips the iterative
    /// refinement of the grid undistortion. The C++ defaults are `with_model = true`,
    /// `with_refine = true` (`core/Camera.h:116`).
    /// Oracle: `undistortion/camera_undistort_samples.csv`, `trial_<t>/points2d_undistorted.csv`.
    #[must_use]
    pub fn undistort_point(
        &self,
        point: Point2,
        undistort: bool,
        with_model: bool,
        with_refine: bool,
    ) -> Point2 {
        let criteria = if self.compat.five_iterations {
            TermCriteria::default()
        } else {
            TermCriteria::CONVERGED
        };
        let lens = |p: Point2| {
            if self.intrinsics.distortion.is_some() && with_model {
                apply_model_distortion_with(&self.intrinsics, p, undistort, criteria)
            } else {
                p
            }
        };
        let grid = |p: Point2| match &self.grid {
            Some(grid) => transform_point(grid, p, undistort, with_refine),
            None => p,
        };
        if undistort || self.compat.grid_first_when_distorting {
            lens(grid(point))
        } else {
            grid(lens(point))
        }
    }

    /// Projects a world point into the raw (distorted) image.
    ///
    /// Ported from `core/Camera.cpp:165-186`, `Camera::projectPoint`: `P = K [R | t]` from
    /// [`projection_matrix`] for `pose` (the C++ takes the reference calibration image's
    /// pose), `P * (X, Y, Z, 1)`, division by the third component, then
    /// [`Self::undistort_point`] in the distorting direction with the defaults
    /// (`with_model = true`, `with_refine = true`).
    ///
    /// The product is OpenCV's `gemm` for a 3x4 by 4x1 matrix: `gemmImpl` turns it into
    /// `A * Bt` and sums the four products as `((p0 + p1) + p2) + p3`.
    /// When the projected depth is exactly zero the C++ returns (0, 0); so does this. Tier A.
    /// Oracle: `trial_<t>/reprojection.csv`.
    #[must_use]
    pub fn project_point(&self, pose: &RigidTransform, point: Point3) -> Point2 {
        let p = projection_matrix(&self.intrinsics.camera_matrix, pose);
        let x = [point.x, point.y, point.z, 1.0];
        let mut out = [0.0; 3];
        for (i, o) in out.iter_mut().enumerate() {
            // GEMMSingleMul, A * Bt branch, unrolled by four: s_k = 0 + a_k*b_k.
            let s0 = 0.0 + p[(i, 0)] * x[0];
            let s1 = 0.0 + p[(i, 1)] * x[1];
            let s2 = 0.0 + p[(i, 2)] * x[2];
            let s3 = 0.0 + p[(i, 3)] * x[3];
            *o = (s0 + s1 + s2 + s3) * 1.0;
        }
        let z = out[2];
        if z != 0.0 {
            let pt = Point2::new(out[0] / z, out[1] / z);
            self.undistort_point(pt, false, true, true)
        } else {
            Point2::new(0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkrange_rejects_nan_and_large() {
        let mut m = Matrix3::identity();
        m[(1, 2)] = f64::NAN;
        assert_eq!(matrix_to_rodrigues(&m), Vector3::zeros());
        let mut m = Matrix3::identity();
        m[(0, 0)] = 100.0;
        assert_eq!(matrix_to_rodrigues(&m), Vector3::zeros());
    }
}
