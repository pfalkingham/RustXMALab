//! Rigid-body poses: the least-squares fit, the pose from 2D rays, the per-frame pose
//! computation with its fallbacks, rotation continuity and the error measures.
//!
//! Work package: rigid-body poses. Ported from `core/RigidBody.cpp` (`fitRigidTransform`,
//! `computePose`, `makeRotationsContinous`, `updateError`, `projectToImage`,
//! `transformPoint`, `addDummyPointsForOptimization`) and
//! `processing/RigidBodyPoseFrom2D.cpp`. The refinement on 2D reprojection error is
//! [`crate::pose_refine`].
//!
//! # Conventions
//!
//! A pose ([`RigidTransform`]) is what the C++ stores in `rotationvectors[f]` and
//! `translationvectors[f]`: it maps world coordinates to body (reference) coordinates,
//! `x_body = R x_world + t`. [`body_to_world`] goes the other way.
//!
//! Statuses are not passed: a marker's 3D point is `Some` exactly when its `status3D` is
//! above `UNDEFINED` (0), and a 2D view is `Some` exactly when its `status2D` is above
//! `UNDEFINED`. Those are the only tests the C++ makes here (`> UNDEFINED`, `> 0` and
//! `<= UNDEFINED`, which are the same thing).
//!
//! # What the caller does
//!
//! The functions here see one body in one frame. The caller (the engine) gathers a
//! [`RigidBodyFrame`] and the [`PoseCamera`]s from the project, and keeps what the C++ keeps
//! around the computation:
//!
//! - `RigidBody::computePose` returns before any of this for a project with
//!   `NO_CALIBRATION` (without updating the errors), and computes nothing (but does update
//!   the errors) when the body is not `initialised` (`core/RigidBody.cpp:1071-1074`).
//! - A virtual point that refers to another body needs that body's pose in the same frame;
//!   `transformPoint` computes it on demand (`core/RigidBody.cpp:297-300`). The caller
//!   resolves this and passes the world position ([`VirtualPointFrame::world`]).
//! - `poseComputed[frame]` is the `is_some()` of [`PoseOutcome::pose`]. Where the C++
//!   leaves the stored pose untouched (no pose computed), the caller keeps its old value.
//! - After all frames, [`make_rotations_continuous`], then filtering ([`crate::filter`]).
//! - [`pose_errors`] for every frame after `computePose` (with the pose just computed,
//!   before [`make_rotations_continuous`]) and, with the filtered pose, after filtering.
//!
//! # Floating point
//!
//! Every function follows the C++ operation by operation, never fuses a multiply and an
//! add, and goes through `f32` where the C++ goes through `cv::Point3f` or `cv::Point2f`
//! ([`through_f32`], ledger A23). The OpenCV matrix expressions are written out as OpenCV
//! 4.13 evaluates them (`modules/core/src/matrix_expressions.cpp`, `matmul.simd.hpp`):
//!
//! | C++ | Evaluated as | Here |
//! |---|---|---|
//! | `A * B`, 3x3 by 3x3 (and 4x4 by 4x4) | `gemmImpl` small-matrix branch, row by row: `t = a0*b0 + a1*b1 + a2*b2` left to right, `t*1 + 0*0` | [`gemm_small`] |
//! | `A * v`, 3x3 by 3x1, with `alpha` from a unary minus | small-matrix branch, `t*alpha + 0*0` | [`gemm_3x3_3x1`] |
//! | `y - A * v` | one `gemm(A, v, -1, y, 1)`: `t*(-1) + y*1` | [`gemm_3x3_3x1`] |
//! | `R.t() * (x - t)` | `GEMM_1_T` is not the small branch: `GEMMSingleMul`, `A * Bt` loop, `s = 0 + a0*b0 + a1*b1 + a2*b2`, then `(s + 0 + 0 + 0) * 1` | [`gemm_t_3x3_3x1`] |
//! | `P * X`, 3x4 by 4x1 | `GEMMSingleMul`, `A * Bt`, unrolled by four: `(p0 + p1 + p2 + p3) * 1`, each `p_k = 0 + a_k*b_k` | [`gemm_3x4_4x1`] |
//! | `v = v / cv::norm(v)` | `convertTo(v, -1, 1./n)`: `v*(1/n) + 0`, a copy when `|1/n - 1| < DBL_EPSILON`, `v + 0` when `1/n == 1` | [`divide_by_norm`] |
//! | `cv::norm` of a 3-vector (`Mat`, `Vec3d`) | `sqrt(((0 + x*x) + y*y) + z*z)` | [`norm3`] |

// The ports keep the C++ index loops so they can be checked line by line against it.
#![allow(clippy::needless_range_loop)]

use crate::camera::{
    CameraModel, RigidTransform, matrix_to_rodrigues, project_points_f32, projection_matrix,
    rodrigues_to_matrix,
};
use crate::linalg::{SvdMode, determinant, invert, svd};
use crate::{Matrix3, Point2, Point3, Vector3};
use nalgebra::Matrix4;

// ---------------------------------------------------------------------------------------
// Inputs and outputs
// ---------------------------------------------------------------------------------------

/// One camera as the rigid-body code reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseCamera {
    /// Camera matrix, lens distortion and undistortion grid (`Camera::getCameraMatrix`,
    /// `getDistortionCoefficiants`, `undistortPoint`). When the camera has no model
    /// distortion the C++ holds eight zero coefficients; the refinement reads those
    /// ([`crate::pose_refine::RefineCamera::from_pose_camera`]).
    pub model: CameraModel,
    /// The rotation and translation vectors of the camera's calibration image at the
    /// trial's reference index (`getCalibrationImages()[trial->getReferenceCalibrationImage()]`),
    /// world to camera.
    pub pose: RigidTransform,
    /// `Camera::isCalibrated()`. Only virtual points look at it (ledger A29).
    pub calibrated: bool,
    /// `CalibrationImage::isCalibrated() > 0` for that calibration image. Only the EPnP
    /// fallback looks at it: for an uncalibrated image the C++ inverts an uninitialised
    /// matrix ([`compute_pose`]).
    pub reference_image_calibrated: bool,
}

/// One body marker in one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerFrame {
    /// The marker's 3D point (`getPoints3D()[frame]`) when `status3D[frame] > UNDEFINED`.
    pub point3d: Option<Point3>,
    /// One entry per camera, in camera order: the 2D point when
    /// `status2D[camera][frame] > UNDEFINED`.
    pub views: Vec<Option<View>>,
}

/// A marker's 2D point in one camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// The point as stored (`getPoints2D()[camera][frame]`), in the raw (distorted) image.
    pub raw: Point2,
    /// `Camera::undistortPoint(raw, true)`: [`CameraModel::undistort_point`] with
    /// `undistort = true` and the defaults `with_model = true`, `with_refine = true`.
    pub undistorted: Point2,
}

/// One virtual ("dummy") point of the body in one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualPointFrame {
    /// The point in this body's reference frame (`dummypoints[i]`).
    pub reference: Point3,
    /// The point in world coordinates for this frame, as the C++ finds it
    /// (`core/RigidBody.cpp:1094-1118`, `:486-509`): if the point refers to another body
    /// (`dummyRBIndex[i] >= 0`), that body's `transformPoint(dummypoints2[i], frame)`
    /// ([`body_to_world`] with that body's pose), or `None` when that body has no pose;
    /// otherwise `dummypointsCoords[i][frame]` when `dummypointsCoordsSet[i][frame]`, else
    /// `None`. (A point that refers to another body never falls back to the coordinates.)
    pub world: Option<Point3>,
}

/// Everything about one body in one frame that the pose computation reads.
#[derive(Clone, Debug, PartialEq)]
pub struct RigidBodyFrame {
    /// The body's working reference points (`RigidBody::points3D`, which
    /// `getReferencePoints()` returns: the optimised set when one is loaded), one per body
    /// marker.
    pub reference_points: Vec<Point3>,
    /// The body's markers (`pointsIdx` order), one per reference point.
    pub markers: Vec<MarkerFrame>,
    /// The body's virtual points.
    pub virtual_points: Vec<VirtualPointFrame>,
}

/// Which faults of XMALab 3's rigid-body computation to reproduce (parity ledger A2, A3,
/// A4, A20, A23, A26, A27, A28, A29, A30, A210; decided 14 November 2026, A210 on 21
/// November 2026: fixed by default). Each
/// field is `true` for the C++ behaviour. [`Default`] and [`RigidCompat::FIXED`] fix all of
/// them; [`RigidCompat::XMALAB3`] is the C++.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct RigidCompat {
    /// A2: in the refinement each virtual point is added once per camera for every
    /// calibrated camera (cameras x calibrated cameras observations). Fixed: one
    /// observation per calibrated camera.
    pub virtual_points_per_camera_pair: bool,
    /// A3: a virtual point with no world position in the frame is added to the refinement
    /// at the origin, with its reference position at the origin. Fixed: left out, as
    /// `computePose` leaves it out.
    pub virtual_points_at_origin: bool,
    /// A4: the refinement's residual applies the camera's k1, k2, p1, p2, k3 to the
    /// projection, although the targets are fully undistorted. Fixed: the coefficients are
    /// taken as zero, the plain pinhole projection (exactly what the C++ computes for a
    /// camera without a lens model).
    pub lens_distortion: bool,
    /// A20: an error standard deviation with one contributing value is NaN (`count - 1`).
    /// Fixed: 0, as for markers.
    pub error_sd_nan: bool,
    /// A23: the inputs of the rigid fit, the virtual points of the refinement, the points
    /// `projectToImage` projects and the EPnP inputs pass through single precision. Fixed:
    /// double throughout.
    pub single_precision: bool,
    /// A26: the EPnP fallback uses the last camera with at least 4 markers, and a failed
    /// EPnP gives a zero pose marked as computed. Fixed: the camera with the most markers
    /// (at least 4; the first on a tie), and no pose when EPnP fails.
    pub epnp_last_camera: bool,
    /// A27: in the pose from 2D rays, a ray with no candidate empties the list of
    /// combinations and later indices no longer match their rays. Fixed: such a ray (its
    /// marker) is left out for the frame.
    pub rays_misaligned: bool,
    /// A28: the solver's refusal or failure is ignored and the refined parameters are used
    /// (a refusal returns the pose inverted twice). Fixed: the pose is left exactly as it
    /// was.
    pub refine_failure_ignored: bool,
    /// A29: the refinement uses marker observations from uncalibrated cameras too. Fixed:
    /// calibrated cameras only.
    pub uncalibrated_cameras_in_refinement: bool,
    /// A30: the refinement's projection ignores the camera matrix's skew. Fixed: the skew
    /// term is included.
    pub refinement_without_skew: bool,
    /// A210: the pose from 2D rays scores every combination of ray candidates, however
    /// many there are (up to 4^n for n rays). Fixed: a frame with more than
    /// [`MAX_RAY_COMBINATIONS`] combinations gets no correspondences from the rays
    /// ([`From2dOutcome::capped`]).
    pub pose_from_2d_uncapped: bool,
}

impl RigidCompat {
    /// Every fault fixed: the default.
    pub const FIXED: RigidCompat = RigidCompat {
        virtual_points_per_camera_pair: false,
        virtual_points_at_origin: false,
        lens_distortion: false,
        error_sd_nan: false,
        single_precision: false,
        epnp_last_camera: false,
        rays_misaligned: false,
        refine_failure_ignored: false,
        uncalibrated_cameras_in_refinement: false,
        refinement_without_skew: false,
        pose_from_2d_uncapped: false,
    };

    /// Everything as XMALab 3 does it.
    pub const XMALAB3: RigidCompat = RigidCompat {
        virtual_points_per_camera_pair: true,
        virtual_points_at_origin: true,
        lens_distortion: true,
        error_sd_nan: true,
        single_precision: true,
        epnp_last_camera: true,
        rays_misaligned: true,
        refine_failure_ignored: true,
        uncalibrated_cameras_in_refinement: true,
        refinement_without_skew: true,
        pose_from_2d_uncapped: true,
    };

    /// [`through_f32`] when [`RigidCompat::single_precision`], else the point itself.
    #[must_use]
    pub fn round(&self, p: Point3) -> Point3 {
        if self.single_precision {
            through_f32(p)
        } else {
            p
        }
    }
}

/// The two settings `computePose` reads, and which faults of the computation to reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoseSettings {
    /// `DisableRBComputeAdvanced`: no pose from 2D rays, no EPnP. Default false.
    pub disable_advanced: bool,
    /// `OptimizeRigidBody`: refine every pose found on 2D reprojection error. Default true.
    pub optimize: bool,
    /// Which faults of XMALab 3 to reproduce. Default: none.
    pub compat: RigidCompat,
}

impl Default for PoseSettings {
    /// The defaults registered in `core/Settings.cpp:104-105`, with every fault fixed
    /// ([`RigidCompat::FIXED`]).
    fn default() -> Self {
        PoseSettings {
            disable_advanced: false,
            optimize: true,
            compat: RigidCompat::FIXED,
        }
    }
}

/// Which path of `computePose` produced the pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseSource {
    /// The rigid fit on 3D correspondences. `from_2d` is the number of correspondences
    /// [`find_from_2d_with_3d`] added (0 when it did not run or added none).
    RigidFit {
        /// Correspondences added from 2D rays.
        from_2d: usize,
    },
    /// The EPnP fallback, from this camera.
    Epnp {
        /// The camera used.
        camera: usize,
    },
    /// No pose (`poseComputed[frame] = 0`).
    None,
}

/// The result of [`compute_pose`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseOutcome {
    /// The pose, after refinement when that is on; `None` where the C++ sets
    /// `poseComputed[frame] = 0`.
    pub pose: Option<RigidTransform>,
    /// The path taken.
    pub source: PoseSource,
    /// True when [`find_from_2d_with_3d_checked`] ran and found more than
    /// [`MAX_RAY_COMBINATIONS`] combinations, so that the rays gave no correspondence
    /// (ledger A210; never in the XMALab 3 mode).
    pub combinations_capped: bool,
}

/// The four per-frame error measures of `RigidBody::updateError`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseErrors {
    /// `errorMean2D`: mean distance in pixels between the stored 2D points and the
    /// projected reference points.
    pub mean_2d: f64,
    /// `errorSd2D`, dividing by `count - 1` (NaN for one value, ledger A20).
    pub sd_2d: f64,
    /// `errorMean3D`: mean distance between the markers' 3D points and the reference points
    /// mapped to world.
    pub mean_3d: f64,
    /// `errorSd3D`, as `sd_2d`.
    pub sd_3d: f64,
}

impl PoseErrors {
    /// All four zero, as the C++ writes them when there is no pose.
    pub const ZERO: PoseErrors = PoseErrors {
        mean_2d: 0.0,
        sd_2d: 0.0,
        mean_3d: 0.0,
        sd_3d: 0.0,
    };
}

// ---------------------------------------------------------------------------------------
// OpenCV arithmetic helpers
// ---------------------------------------------------------------------------------------

/// `cv::Point3f(p.x, p.y, p.z)` stored back into a `cv::Point3d`: each coordinate rounded
/// to `f32` (round to nearest) and widened again. Ledger A23.
#[must_use]
pub fn through_f32(p: Point3) -> Point3 {
    Point3::new(
        f64::from(p.x as f32),
        f64::from(p.y as f32),
        f64::from(p.z as f32),
    )
}

/// `cv::gemm(a, b, 1, noArray(), 0, d, 0)` for square `n x n` matrices with `n` of 3 or 4:
/// the small-matrix branch of `gemmImpl` with `len == d_size.width`, row by row,
/// `t = a(i,0)*b(0,j) + a(i,1)*b(1,j) + ...` left to right, stored as `t*1 + 0*0`.
///
/// Ported from `modules/core/src/matmul.simd.hpp`, `gemmImpl` (`CV_64F`, `case 3` and
/// `case 4`, first branch).
pub fn gemm_small<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut d = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut t = a[i][0] * b[0][j];
            for k in 1..N {
                t += a[i][k] * b[k][j];
            }
            d[i][j] = t * 1.0 + 0.0 * 0.0;
        }
    }
    d
}

/// `cv::gemm(a, v, alpha, c, beta, d, 0)` for a 3x3 `a` and a 3x1 `v`: the small-matrix
/// branch of `gemmImpl` (`len == d_size.height`), `t = a(i,0)*v0 + a(i,1)*v1 + a(i,2)*v2`
/// left to right, stored as `t*alpha + c*beta`. With no `c` OpenCV reads zeros.
///
/// Ported from `modules/core/src/matmul.simd.hpp`, `gemmImpl` (`CV_64F`, `case 3`, the
/// `a != d` branch).
pub(crate) fn gemm_3x3_3x1(
    a: &Matrix3,
    v: [f64; 3],
    alpha: f64,
    c: Option<[f64; 3]>,
    beta: f64,
) -> [f64; 3] {
    let c = c.unwrap_or([0.0; 3]);
    let mut d = [0.0f64; 3];
    for i in 0..3 {
        let t = a[(i, 0)] * v[0] + a[(i, 1)] * v[1] + a[(i, 2)] * v[2];
        d[i] = t * alpha + c[i] * beta;
    }
    d
}

/// `r.t() * v` for a 3x3 `r` and a 3x1 `v`: `cv::gemm(r, v, 1, noArray(), 0, d, GEMM_1_T)`.
/// With a transposed operand `gemmImpl` skips its small-matrix branch; a one-column result
/// turns into `GEMMSingleMul`'s `A * Bt` loop, which for three terms sums
/// `s0 = 0 + a0*b0 + a1*b1 + a2*b2` and stores `(s0 + s1 + s2 + s3) * alpha` with
/// `s1 = s2 = s3 = 0`.
///
/// Ported from `modules/core/src/matmul.simd.hpp`, `gemmImpl` and `GEMMSingleMul`.
pub(crate) fn gemm_t_3x3_3x1(r: &Matrix3, v: [f64; 3]) -> [f64; 3] {
    let mut d = [0.0f64; 3];
    for i in 0..3 {
        let mut s0 = 0.0f64;
        for k in 0..3 {
            s0 += r[(k, i)] * v[k];
        }
        let (s1, s2, s3) = (0.0f64, 0.0f64, 0.0f64);
        d[i] = (s0 + s1 + s2 + s3) * 1.0;
    }
    d
}

/// `p * x` for a 3x4 `p` and a 4x1 `x`: `gemmImpl` skips the small-matrix branch
/// (`len == 4` matches neither dimension of the 3x1 result) and `GEMMSingleMul`'s `A * Bt`
/// loop, unrolled by four, gives `(s0 + s1 + s2 + s3) * 1` with `s_k = 0 + p_k*x_k`.
///
/// Ported from `modules/core/src/matmul.simd.hpp`, `GEMMSingleMul` (as
/// [`CameraModel::project_point`] does).
pub(crate) fn gemm_3x4_4x1(p: &crate::Matrix3x4, x: [f64; 4]) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let s0 = 0.0 + p[(i, 0)] * x[0];
        let s1 = 0.0 + p[(i, 1)] * x[1];
        let s2 = 0.0 + p[(i, 2)] * x[2];
        let s3 = 0.0 + p[(i, 3)] * x[3];
        *o = (s0 + s1 + s2 + s3) * 1.0;
    }
    out
}

/// `cv::norm` of three doubles, `Mat` (`normL2_`, scalar tail) or `Vec3d`
/// (`normL2Sqr<double, double>`): `sqrt(((0 + x*x) + y*y) + z*z)`.
///
/// Ported from `modules/core/include/opencv2/core/base.hpp`, `normL2Sqr`, and
/// `modules/core/src/norm.simd.hpp`, `normL2_`.
pub(crate) fn norm3(v: [f64; 3]) -> f64 {
    let mut s = 0.0f64;
    for x in v {
        s += x * x;
    }
    s.sqrt()
}

/// `v = v / s` for a `Mat` (`MatOp_AddEx` with `alpha = 1./s` assigned onto its own
/// operand): `convertTo(v, -1, alpha)` when `|alpha| != 1`, which copies unchanged when
/// `|alpha - 1| < DBL_EPSILON` and otherwise stores `x*alpha + 0`; `cv::add(v, 0)` when
/// `alpha == 1`; `cv::subtract(0, v)` when `alpha == -1`.
///
/// Ported from `modules/core/src/matrix_expressions.cpp`, `MatOp_AddEx::assign`, and
/// `modules/core/src/convert.dispatch.cpp`, `Mat::convertTo`.
pub(crate) fn divide_by_norm(v: &mut [f64; 3], s: f64) {
    let alpha = 1.0 / s;
    if alpha.abs() != 1.0 {
        if (alpha - 1.0).abs() < f64::EPSILON {
            return;
        }
        for x in v.iter_mut() {
            *x = *x * alpha + 0.0;
        }
    } else if alpha == 1.0 {
        for x in v.iter_mut() {
            *x += 0.0;
        }
    } else {
        for x in v.iter_mut() {
            *x = 0.0 - *x;
        }
    }
}

/// `Mat::inv()` (`DECOMP_LU`) of a 3x3 matrix; zeros for a singular one, as OpenCV.
pub(crate) fn inv3(m: &Matrix3) -> Matrix3 {
    let x = invert(m).x;
    Matrix3::from_fn(|r, c| x[(r, c)])
}

fn rows3(m: &Matrix3) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

fn from_rows3(a: &[[f64; 3]; 3]) -> Matrix3 {
    Matrix3::from_fn(|r, c| a[r][c])
}

fn arr(p: Point3) -> [f64; 3] {
    [p.x, p.y, p.z]
}

fn varr(v: &Vector3) -> [f64; 3] {
    [v[0], v[1], v[2]]
}

// ---------------------------------------------------------------------------------------
// Rigid fit
// ---------------------------------------------------------------------------------------

/// Kabsch fit and the sum of distances, the common part of `fitRigidTransform`,
/// `computeAlignementError` and the `Filter3DPoints` refit. Returns the rotation matrix and
/// the translation. No size check.
fn kabsch(src: &[Point3], dst: &[Point3]) -> (Matrix3, [f64; 3]) {
    let mut xg = [0.0f64; 3];
    let mut yg = [0.0f64; 3];
    // Sums in point order (`xg[m] = xg[m] + vnl_tmp[m]`).
    for (x, y) in src.iter().zip(dst) {
        let (x, y) = (arr(*x), arr(*y));
        for m in 0..3 {
            xg[m] += x[m];
            yg[m] += y[m];
        }
    }
    // `xg[m] /= X.size()`: the size_t count converted to double.
    let n = src.len() as f64;
    for m in 0..3 {
        xg[m] /= n;
    }
    let n = dst.len() as f64;
    for m in 0..3 {
        yg[m] /= n;
    }

    // K(l, j) = vnl_tmp[l] + K(l, j) with vnl_tmp[l] = x[i][j] * y[i][l], centred values.
    let mut k = [[0.0f64; 3]; 3];
    for (x, y) in src.iter().zip(dst) {
        let (x, y) = (arr(*x), arr(*y));
        let xc = [x[0] - xg[0], x[1] - xg[1], x[2] - xg[2]];
        let yc = [y[0] - yg[0], y[1] - yg[1], y[2] - yg[2]];
        for j in 0..3 {
            for l in 0..3 {
                let tmp = xc[j] * yc[l];
                // `vnl_tmp[l] + K(l, j)`; IEEE addition is commutative.
                k[l][j] += tmp;
            }
        }
    }

    // svd.compute(K, W, U, VT); V = VT.t(); Sd = diag(1, 1, det(U) * det(V)).
    let usv = svd(&from_rows3(&k), SvdMode::Thin);
    let u = Matrix3::from_fn(|r, c| usv.u[(r, c)]);
    let vt = Matrix3::from_fn(|r, c| usv.vt[(r, c)]);
    let v = vt.transpose();
    let det_u = determinant(&u);
    let det_v = determinant(&v);
    let mut sd = [[0.0f64; 3]; 3];
    sd[0][0] = 1.0;
    sd[1][1] = 1.0;
    sd[2][2] = det_u * det_v;

    // rotMatTmp = U * Sd * VT: (U * Sd) is evaluated first, then times VT.
    let usd = gemm_small(&rows3(&u), &sd);
    let r = from_rows3(&gemm_small(&usd, &rows3(&vt)));

    // t = ygmat - rotMatTmp * xgmat: one gemm(R, xg, -1, yg, 1).
    let t = gemm_3x3_3x1(&r, xg, -1.0, Some(yg), 1.0);
    (r, t)
}

/// Least-squares rigid transform taking `src` onto `dst`: `dst ≈ R src + t`.
///
/// Ported from `core/RigidBody.cpp:980-1063`, `RigidBody::fitRigidTransform`: centroids,
/// covariance `K = Σ y_c x_cᵀ`, OpenCV's Jacobi SVD `K = U W Vᵀ`, `R = U diag(1, 1,
/// det U det V) Vᵀ`, Rodrigues vector of `R`, `t = ȳ − R x̄`. All in double; the caller
/// rounds the inputs through `f32` where the C++ does ([`through_f32`]).
///
/// Returns `None` where the C++ returns false: fewer than 3 pairs, or slices of different
/// lengths. Collinear or coincident points are not rejected (nor in the C++): the SVD then
/// has repeated or zero singular values and the rotation is whatever it gives.
///
/// The per-frame refit of `RigidBody::filterTransformations` in `Filter3DPoints` mode
/// (`core/RigidBody.cpp:1767-1885`) is the same code inline, with the same arithmetic
/// (its inputs also pass through `cv::Point3f`); use this function for it. Tier A.
/// Oracle: `trial_<t>/rb_pose_kabsch.csv`.
#[must_use]
pub fn fit_rigid_transform(src: &[Point3], dst: &[Point3]) -> Option<RigidTransform> {
    if src.len() < 3 || src.len() != dst.len() {
        return None;
    }
    let (r, t) = kabsch(src, dst);
    Some(RigidTransform {
        rotation: matrix_to_rodrigues(&r),
        translation: Vector3::new(t[0], t[1], t[2]),
    })
}

/// Makes a sequence of rotation vectors continuous: where the angle between consecutive
/// vectors exceeds 150 degrees, the later one is replaced by the same rotation expressed
/// on the other side of pi, `-r / |r| * (2 pi - |r|)`.
///
/// Ported from `core/RigidBody.cpp:1594-1613`, `RigidBody::makeRotationsContinous`; the
/// same loop runs on the filtered poses in `Filter3DPoints` mode (`:1888-1903`). For
/// `i = 1..n-1` in order, only when `valid[i-1]` and `valid[i]` (so a gap breaks the
/// chain), comparing with the already adjusted `rotations[i-1]`. The cosine
/// `dot / |a| / |b|` is not clamped: a zero vector or a cosine outside [-1, 1] gives NaN and
/// leaves the vector as it is. Tier A.
/// Oracle: `trial_<t>/rb_pose_kabsch.csv`, `rb_pose_optimized.csv` (stored after this step).
///
/// Panics if the slices differ in length.
pub fn make_rotations_continuous(rotations: &mut [Vector3], valid: &[bool]) {
    assert_eq!(rotations.len(), valid.len(), "one valid flag per rotation");
    for i in 1..rotations.len() {
        if valid[i - 1] && valid[i] {
            let a = varr(&rotations[i - 1]);
            let b = varr(&rotations[i]);
            // Matx::dot: s = 0; s += a[k]*b[k].
            let mut d = 0.0f64;
            for k in 0..3 {
                d += a[k] * b[k];
            }
            let diffangle = d / norm3(a) / norm3(b);
            if 180.0 / std::f64::consts::PI * diffangle.acos() > 150.0 {
                let angle = norm3(b);
                let two_pi = 2.0 * std::f64::consts::PI;
                for k in 0..3 {
                    rotations[i][k] = -rotations[i][k] / angle * (two_pi - angle);
                }
            }
        }
    }
}

/// Maps a point from body (reference) coordinates to world coordinates with a pose:
/// `Rᵀ (x − t)`.
///
/// Ported from `core/RigidBody.cpp:295-317`, `RigidBody::transformPoint` (the arithmetic;
/// the on-demand `computePose` and the pose test are the caller's). The same expression
/// is used by `updateError` and `projectToImage`. Tier A.
#[must_use]
pub fn body_to_world(pose: &RigidTransform, point: Point3) -> Point3 {
    let r = rodrigues_to_matrix(&pose.rotation);
    let t = varr(&pose.translation);
    let diff = [point.x - t[0], point.y - t[1], point.z - t[2]];
    let w = gemm_t_3x3_3x1(&r, diff);
    Point3::new(w[0], w[1], w[2])
}

// ---------------------------------------------------------------------------------------
// Pose from 2D rays (processing/RigidBodyPoseFrom2D.cpp)
// ---------------------------------------------------------------------------------------

/// A viewing ray of a marker whose 3D point is missing, in the form
/// `RigidBodyPoseFrom2D` uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// Unit direction in world coordinates (Plücker `Q1..Q3`).
    pub direction: [f64; 3],
    /// The camera centre (Plücker `Q4..Q6`, a point on the line, not the moment).
    pub origin: [f64; 3],
    /// The marker's reference point (`Pts3D`).
    pub reference: Point3,
}

/// One ray per (camera, body marker) with a 2D point and no 3D point, cameras outer, markers
/// inner.
///
/// Ported from `processing/RigidBodyPoseFrom2D.cpp:320-359`,
/// `RigidBodyPoseFrom2D::getPluckerForMissing`: `K⁻¹ (u, v, 1)` normalised, then
/// `Rᵀ` (as `Rot.inv()`) times it, normalised again; the origin is `Rot.inv() * -t`. Tier A.
#[must_use]
pub fn rays_for_missing(frame: &RigidBodyFrame, cameras: &[PoseCamera]) -> Vec<Ray> {
    let mut rays = Vec::new();
    for (c, cam) in cameras.iter().enumerate() {
        for (p, marker) in frame.markers.iter().enumerate() {
            let Some(view) = marker.views.get(c).copied().flatten() else {
                continue;
            };
            if marker.point3d.is_some() {
                continue;
            }
            let rot = rodrigues_to_matrix(&cam.pose.rotation);
            let trans = varr(&cam.pose.translation);
            let k_inv = inv3(&cam.model.intrinsics.camera_matrix);

            let pt2d = view.undistorted;
            let pt3 = [pt2d.x, pt2d.y, 1.0];
            let mut pt3_world = gemm_3x3_3x1(&k_inv, pt3, 1.0, None, 0.0);
            let n = norm3(pt3_world);
            divide_by_norm(&mut pt3_world, n);
            let rot = inv3(&rot);

            // q2 = Rot * -trans: gemm(Rot, trans, -1).
            let q2 = gemm_3x3_3x1(&rot, trans, -1.0, None, 0.0);
            let mut q1 = gemm_3x3_3x1(&rot, pt3_world, 1.0, None, 0.0);
            let n = norm3(q1);
            divide_by_norm(&mut q1, n);
            rays.push(Ray {
                direction: q1,
                origin: q2,
                reference: frame.reference_points[p],
            });
        }
    }
    rays
}

/// The points on a ray at the distance from a known world point that the reference points
/// say they must have: the intersections of the line with the sphere around `known_3d` of
/// radius `|ray.reference − known_reference|`, kept only for line parameter `l > 0`; if
/// the line misses the sphere (discriminant not positive), the closest point on the line,
/// again only if `l > 0`. Empty when the direction is zero.
///
/// Ported from `processing/RigidBodyPoseFrom2D.cpp:241-318`, `RigidBodyPoseFrom2D::get3DPoint`,
/// term by term. Tier A.
#[must_use]
pub fn ray_points_at_distance(ray: &Ray, known_3d: Point3, known_reference: Point3) -> Vec<Point3> {
    let mut points = Vec::new();
    let [q1, q2, q3] = ray.direction;
    let [q4, q5, q6] = ray.origin;
    // cv::norm(Point3d): sqrt(x*x + y*y + z*z).
    let d = ray.reference - known_reference;
    let dist12 = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();

    let pt3d1 = known_3d.x;
    let pt3d2 = known_3d.y;
    let pt3d3 = known_3d.z;

    let t2 = q1 * q1;
    let t3 = q2 * q2;
    let t4 = q3 * q3;
    let t5 = q4 * q4;
    let t6 = q6 * q6;
    let t7 = q5 * q5;
    let t8 = dist12 * dist12;
    let t9 = pt3d1 * pt3d1;
    let t10 = pt3d3 * pt3d3;
    let t11 = pt3d2 * pt3d2;
    let t12 = t2 + t3 + t4;
    if t12 == 0.0 {
        return points;
    }
    let t13 = 1.0 / t12;
    let t14 = t2 * t8;
    let t15 = t3 * t8;
    let t16 = t4 * t8;
    let t17 = q4 * pt3d1 * t3 * 2.0;
    let t18 = q5 * pt3d2 * t2 * 2.0;
    let t19 = q4 * pt3d1 * t4 * 2.0;
    let t20 = q6 * pt3d3 * t2 * 2.0;
    let t21 = q5 * pt3d2 * t4 * 2.0;
    let t22 = q6 * pt3d3 * t3 * 2.0;
    let t23 = q1 * q2 * q4 * q5 * 2.0;
    let t24 = q1 * q3 * q4 * q6 * 2.0;
    let t25 = q2 * q3 * q5 * q6 * 2.0;
    let t26 = q1 * q2 * pt3d1 * pt3d2 * 2.0;
    let t27 = q1 * q3 * pt3d1 * pt3d3 * 2.0;
    let t28 = q2 * q3 * pt3d2 * pt3d3 * 2.0;
    let t29 =
        t14 + t15 + t16 + t17 + t18 + t19 + t20 + t21 + t22 + t23 + t24 + t25 + t26 + t27 + t28
            - t2 * t6
            - t3 * t5
            - t2 * t7
            - t3 * t6
            - t4 * t5
            - t4 * t7
            - t2 * t10
            - t3 * t9
            - t2 * t11
            - t3 * t10
            - t4 * t9
            - t4 * t11
            - q1 * q2 * q4 * pt3d2 * 2.0
            - q1 * q2 * q5 * pt3d1 * 2.0
            - q1 * q3 * q4 * pt3d3 * 2.0
            - q1 * q3 * q6 * pt3d1 * 2.0
            - q2 * q3 * q5 * pt3d3 * 2.0
            - q2 * q3 * q6 * pt3d2 * 2.0;

    let t31 = q1 * pt3d1;
    let t32 = q2 * pt3d2;
    let t33 = q3 * pt3d3;
    if t29 > 0.0 {
        let t30 = t29.sqrt();
        let l1 = t13 * (t30 + t31 + t32 + t33 - q1 * q4 - q2 * q5 - q3 * q6);
        let pt1 = Point3::new(q4 + q1 * l1, q5 + q2 * l1, q6 + q3 * l1);
        if l1 > 0.0 {
            points.push(pt1);
        }
        let l2 = -t13 * (t30 - t31 - t32 - t33 + q1 * q4 + q2 * q5 + q3 * q6);
        let pt2 = Point3::new(q4 + q1 * l2, q5 + q2 * l2, q6 + q3 * l2);
        if l2 > 0.0 {
            points.push(pt2);
        }
    } else {
        // The C++ prints t29 to stderr here.
        let l1 = t13 * (t31 + t32 + t33 - q1 * q4 - q2 * q5 - q3 * q6);
        let pt1 = Point3::new(q4 + q1 * l1, q5 + q2 * l1, q6 + q3 * l1);
        if l1 > 0.0 {
            points.push(pt1);
        }
    }
    points
}

/// Sum (not mean) of the distances between `src` and `dst` mapped back with the Kabsch fit
/// of `src` onto `dst`: `Σ |src_i − Rᵀ (dst_i − t)|`.
///
/// Ported from `processing/RigidBodyPoseFrom2D.cpp:142-239`,
/// `RigidBodyPoseFrom2D::computeAlignementError`. No minimum size (the C++ has none), so
/// one or two pairs give a degenerate but defined fit. Tier A.
#[must_use]
pub fn alignment_error(src: &[Point3], dst: &[Point3]) -> f64 {
    let (r, t) = kabsch(src, dst);
    let mut dist = 0.0f64;
    for (s, d) in src.iter().zip(dst) {
        let diff_in = [d.x - t[0], d.y - t[1], d.z - t[2]];
        let p = gemm_t_3x3_3x1(&r, diff_in);
        let diff = [s.x - p[0], s.y - p[1], s.z - p[2]];
        dist += (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    }
    dist
}

/// Adds correspondences from 2D rays when one or two 3D points are known: for each ray of a
/// marker seen in 2D but without a 3D point ([`rays_for_missing`]), every candidate point at
/// the right distance from each known point ([`ray_points_at_distance`]); every combination
/// of one candidate per ray is scored by [`alignment_error`] over the known pairs plus the
/// candidates, and the best one below 100000 is appended to `src` (candidates) and `dst`
/// (their reference points). Returns the number of pairs added.
///
/// Ported from `processing/RigidBodyPoseFrom2D.cpp:58-140`,
/// `RigidBodyPoseFrom2D::findFrom2Dwith3D`, including the behaviour in ledger A27: a ray
/// with no candidate empties the list of combinations and the next ray starts it afresh,
/// after which the `p`-th entry of a combination is still read as a candidate of the `p`-th
/// ray and paired with the `p`-th ray's reference point, although it was built from a
/// later ray. Where such an index runs past the end of the `p`-th ray's candidates the C++
/// reads outside the vector (undefined behaviour); this port skips that combination.
///
/// Returns 0 and changes nothing when rays plus known pairs number fewer than 3.
///
/// Without [`RigidCompat::rays_misaligned`] (ledger A27 fixed) a ray with no candidate is
/// left out (its marker gets no point in this frame) before the combinations are formed,
/// so every entry of a combination belongs to its own ray.
///
/// Without [`RigidCompat::pose_from_2d_uncapped`] (ledger A210 fixed) a frame with more
/// than [`MAX_RAY_COMBINATIONS`] combinations adds nothing and returns 0; use
/// [`find_from_2d_with_3d_checked`] to tell that case apart.
pub fn find_from_2d_with_3d(
    frame: &RigidBodyFrame,
    cameras: &[PoseCamera],
    src: &mut Vec<Point3>,
    dst: &mut Vec<Point3>,
    compat: RigidCompat,
) -> usize {
    find_from_2d_with_3d_checked(frame, cameras, src, dst, compat).added
}

/// The most combinations of ray candidates [`find_from_2d_with_3d`] scores for one frame
/// without [`RigidCompat::pose_from_2d_uncapped`] (the author, 21 November 2026: "Cap at 1
/// million"; ledger A210). Exactly this many are still scored.
pub const MAX_RAY_COMBINATIONS: usize = 1_000_000;

/// What [`find_from_2d_with_3d_checked`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct From2dOutcome {
    /// The number of pairs appended to `src` and `dst` (what [`find_from_2d_with_3d`]
    /// returns).
    pub added: usize,
    /// True when the frame had more than [`MAX_RAY_COMBINATIONS`] combinations and
    /// [`RigidCompat::pose_from_2d_uncapped`] is off: nothing was scored or added.
    pub capped: bool,
}

/// The number of combinations [`for_each_combination`] calls back with for these candidate
/// counts: the product of the counts after the last zero (none when the last count is zero
/// or there is none), saturating at `usize::MAX` instead of overflowing.
#[must_use]
pub fn ray_combination_count(counts: &[usize]) -> usize {
    let start = counts.iter().rposition(|&c| c == 0).map_or(0, |i| i + 1);
    let radices = &counts[start..];
    if radices.is_empty() {
        return 0;
    }
    radices.iter().fold(1usize, |n, &c| n.saturating_mul(c))
}

/// [`find_from_2d_with_3d`], also saying whether the frame was capped
/// ([`From2dOutcome::capped`], ledger A210).
///
/// Ported from `processing/RigidBodyPoseFrom2D.cpp:58-140`,
/// `RigidBodyPoseFrom2D::findFrom2Dwith3D` (see [`find_from_2d_with_3d`]); the cap is not in
/// the C++.
pub fn find_from_2d_with_3d_checked(
    frame: &RigidBodyFrame,
    cameras: &[PoseCamera],
    src: &mut Vec<Point3>,
    dst: &mut Vec<Point3>,
    compat: RigidCompat,
) -> From2dOutcome {
    let mut rays = rays_for_missing(frame, cameras);
    if rays.len() + dst.len() < 3 {
        return From2dOutcome::default();
    }

    let mut solutions: Vec<Vec<Point3>> = rays
        .iter()
        .map(|ray| {
            let mut sol = Vec::new();
            for r in 0..dst.len() {
                sol.extend(ray_points_at_distance(ray, src[r], dst[r]));
            }
            sol
        })
        .collect();
    if !compat.rays_misaligned {
        let keep: Vec<bool> = solutions.iter().map(|s| !s.is_empty()).collect();
        let mut k = keep.iter();
        rays.retain(|_| *k.next().unwrap_or(&false));
        solutions.retain(|s| !s.is_empty());
    }

    // The C++ builds the whole list of combinations before scoring any (its size is the
    // product of the candidate counts); they are scored here one at a time, in its order.
    let counts: Vec<usize> = solutions.iter().map(Vec::len).collect();
    if !compat.pose_from_2d_uncapped && ray_combination_count(&counts) > MAX_RAY_COMBINATIONS {
        return From2dOutcome {
            added: 0,
            capped: true,
        };
    }
    let mut best: Vec<usize> = Vec::new();
    let mut best_dist = 100000.0f64;
    let (n_src, n_dst) = (src.len(), dst.len());
    let mut src_tmp = src.clone();
    let mut dst_tmp = dst.clone();
    for_each_combination(&counts, |combination| {
        src_tmp.truncate(n_src);
        dst_tmp.truncate(n_dst);
        for (p, &k) in combination.iter().enumerate() {
            let Some(&candidate) = solutions[p].get(k) else {
                // The C++ reads past the end of src_Solutions[p] here.
                return;
            };
            src_tmp.push(candidate);
            dst_tmp.push(rays[p].reference);
        }
        let dist = alignment_error(&src_tmp, &dst_tmp);
        if dist < best_dist {
            best_dist = dist;
            best.clear();
            best.extend_from_slice(combination);
        }
    });

    for (i, &k) in best.iter().enumerate() {
        dst.push(rays[i].reference);
        src.push(solutions[i][k]);
    }
    From2dOutcome {
        added: best.len(),
        capped: false,
    }
}

/// Calls `f` with every combination of one candidate index per ray that
/// `RigidBodyPoseFrom2D::findFrom2Dwith3D` puts in its list `combinations`, in the list's
/// order, without building the list (its length is the product of the counts).
///
/// The C++ extends the list ray by ray, the new ray's index in the outer loop and the
/// combinations so far in the inner one, so the earliest ray's index varies fastest. A ray
/// with no candidate empties the list and the next ray starts it afresh with one-entry
/// combinations; so the combinations hold one index for each ray after the last ray with
/// no candidate, and there are none when the last ray has none or there is no ray.
///
/// Ported from `processing/RigidBodyPoseFrom2D.cpp:87-106`.
fn for_each_combination(counts: &[usize], mut f: impl FnMut(&[usize])) {
    let start = counts.iter().rposition(|&c| c == 0).map_or(0, |i| i + 1);
    let radices = &counts[start..];
    if radices.is_empty() {
        return;
    }
    let mut index = vec![0usize; radices.len()];
    loop {
        f(&index);
        let mut p = 0;
        loop {
            index[p] += 1;
            if index[p] < radices[p] {
                break;
            }
            index[p] = 0;
            p += 1;
            if p == radices.len() {
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// computePose
// ---------------------------------------------------------------------------------------

/// `CalibrationImage::getTransformationMatrix()`: the 4x4 `[R | t; 0 0 0 1]` of the
/// calibration image (world to camera), **inverted** (`return trans.inv()`, so camera to
/// world). `None` for an uncalibrated image, where the C++ inverts an uninitialised matrix.
///
/// Ported from `core/CalibrationImage.cpp:369-401`.
fn calibration_transformation_matrix(cam: &PoseCamera) -> Option<[[f64; 4]; 4]> {
    if !cam.reference_image_calibrated {
        return None;
    }
    let r = rodrigues_to_matrix(&cam.pose.rotation);
    let mut trans = Matrix4::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            trans[(i, j)] = r[(i, j)];
        }
        trans[(i, 3)] = cam.pose.translation[i];
    }
    trans[(3, 3)] = 1.0;
    let inv = invert(&trans).x;
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| inv[(i, j)])
    }))
}

/// The camera EPnP runs in: at least 4 of the body's markers seen there. XMALab 3 takes the
/// last such camera (`core/RigidBody.cpp:1140-1155`, `bestCameracount` never raised);
/// fixed (ledger A26), the one with the most markers, the first on a tie.
fn epnp_camera(frame: &RigidBodyFrame, n_cameras: usize, compat: RigidCompat) -> Option<usize> {
    let mut good_camera = None;
    let mut best_camera_count = 4;
    for c in 0..n_cameras {
        let count = frame
            .markers
            .iter()
            .filter(|m| m.views.get(c).copied().flatten().is_some())
            .count();
        if count >= best_camera_count {
            if compat.epnp_last_camera {
                good_camera = Some(c);
            } else if good_camera.is_none() || count > best_camera_count {
                good_camera = Some(c);
                best_camera_count = count;
            }
        }
    }
    good_camera
}

/// The EPnP fallback of `computePose` from camera `good_camera`; `None` where this port
/// cannot reproduce the C++ (uncalibrated reference image).
///
/// Ported from `core/RigidBody.cpp:1156-1216`.
///
/// Fixed (ledger A23, A26): the points stay in double ([`crate::epnp::solve_pnp_epnp_f64`])
/// and a failed EPnP (OpenCV's exception, or a non-finite result) gives no pose instead of
/// the zero transform.
fn epnp_pose(
    frame: &RigidBodyFrame,
    cameras: &[PoseCamera],
    good_camera: usize,
    compat: RigidCompat,
) -> Option<RigidTransform> {
    let cam = &cameras[good_camera];
    let mut object_points = Vec::new();
    let mut image_points = Vec::new();
    let mut object_f64 = Vec::new();
    let mut image_f64 = Vec::new();
    for (p, marker) in frame.markers.iter().enumerate() {
        if let Some(view) = marker.views.get(good_camera).copied().flatten() {
            // Point3d -> Point3f and Point2d -> Point2f conversions.
            let r = frame.reference_points[p];
            object_points.push(nalgebra::Point3::new(r.x as f32, r.y as f32, r.z as f32));
            image_points.push(nalgebra::Point2::new(
                view.undistorted.x as f32,
                view.undistorted.y as f32,
            ));
            object_f64.push(r);
            image_f64.push(view.undistorted);
        }
    }
    let distortion = [0.0f64; 8];
    let k = &cam.model.intrinsics.camera_matrix;
    let solved = if compat.single_precision {
        // As `core/RigidBody.cpp:1170` calls it: `cv::solvePnP(object_points, image_points,
        // cameraMatrix, distortion_coeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP)` with
        // `cv::Point3f` reference points, `cv::Point2f` undistorted image points, the camera
        // matrix as `CV_64F` and eight zero coefficients. `None` where OpenCV throws (it then
        // writes nothing; the C++ ignores the return value and its `Vec3d` outputs stay
        // zero, which `compute_pose` reproduces).
        crate::epnp::solve_pnp_epnp(&object_points, &image_points, k, &distortion)
    } else {
        crate::epnp::solve_pnp_epnp_f64(&object_f64, &image_f64, k, &distortion)
    };
    let pnp = match solved {
        Some(t) => t,
        None if compat.epnp_last_camera => RigidTransform {
            rotation: Vector3::zeros(),
            translation: Vector3::zeros(),
        },
        None => return None,
    };
    if !compat.epnp_last_camera
        && (pnp.rotation.iter().chain(pnp.translation.iter())).any(|v| !v.is_finite())
    {
        return None;
    }

    // trans1 = [Rodrigues(rvec) | tvec; 0 0 0 1].
    let rot1 = rodrigues_to_matrix(&pnp.rotation);
    let mut trans1 = [[0.0f64; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            trans1[i][j] = rot1[(i, j)];
        }
        trans1[i][3] = pnp.translation[i];
    }
    trans1[3][3] = 1.0;

    // out = getTransformationMatrix() * trans1; out = out.inv(). With getTransformationMatrix
    // returning camera-to-world, this is world to body.
    let cam_to_world = calibration_transformation_matrix(cam)?;
    let out = gemm_small(&cam_to_world, &trans1);
    let out = invert(&Matrix4::from_fn(|r, c| out[r][c])).x;

    let rot = Matrix3::from_fn(|r, c| out[(r, c)]);
    Some(RigidTransform {
        rotation: matrix_to_rodrigues(&rot),
        translation: Vector3::new(out[(0, 3)], out[(1, 3)], out[(2, 3)]),
    })
}

/// The pose of a body in one frame, as `RigidBody::computePose` computes it (steps 4 to 8 of
/// `docs/cpp-recompute.md` section 4.1).
///
/// Ported from `core/RigidBody.cpp:1076-1225`:
///
/// 1. Correspondences: each marker with a 3D point (world) and its reference point, then
///    each virtual point with a world position and its reference position, all through
///    `cv::Point3f` ([`through_f32`], ledger A23).
/// 2. Unless `disable_advanced`, with 1 or 2 correspondences: [`find_from_2d_with_3d`].
/// 3. With 3 or more: [`fit_rigid_transform`].
/// 4. Otherwise, unless `disable_advanced`: EPnP from the **last** camera in which at least
///    4 body markers have a 2D point, with the reference points and the undistorted 2D
///    points through `f32` and zero distortion; the camera pose composition and inversion
///    of `:1175-1215` (ledger A26). No camera with 4 points: no pose.
/// 5. With a pose and `optimize`: [`crate::pose_refine::refine_pose`] on the observations of
///    [`crate::pose_refine::refinement_observations`], with `settings.compat` (ledger A3,
///    A4).
///
/// The EPnP step calls [`crate::epnp::solve_pnp_epnp`]. When the chosen camera's reference
/// calibration image is not calibrated the C++ inverts uninitialised memory; this returns
/// no pose there.
///
/// Panics if `frame.markers` and `frame.reference_points` differ in length. Tier A
/// (tier B with refinement where the solver's last bits differ).
/// Oracle: `trial_<t>/rb_pose_kabsch.csv` (`optimize` false), `rb_pose_optimized.csv`.
#[must_use]
pub fn compute_pose(
    frame: &RigidBodyFrame,
    cameras: &[PoseCamera],
    settings: PoseSettings,
) -> PoseOutcome {
    assert_eq!(
        frame.markers.len(),
        frame.reference_points.len(),
        "one reference point per body marker"
    );
    let compat = settings.compat;
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for (marker, reference) in frame.markers.iter().zip(&frame.reference_points) {
        if let Some(p) = marker.point3d {
            src.push(compat.round(p));
            dst.push(compat.round(*reference));
        }
    }
    for v in &frame.virtual_points {
        if let Some(w) = v.world {
            src.push(compat.round(w));
            dst.push(compat.round(v.reference));
        }
    }

    let mut from_2d = From2dOutcome::default();
    if !settings.disable_advanced && dst.len() < 3 && !dst.is_empty() {
        from_2d = find_from_2d_with_3d_checked(frame, cameras, &mut src, &mut dst, compat);
    }
    let combinations_capped = from_2d.capped;
    let from_2d = from_2d.added;

    let mut outcome = PoseOutcome {
        pose: None,
        source: PoseSource::None,
        combinations_capped,
    };
    if dst.len() >= 3 {
        if let Some(t) = fit_rigid_transform(&src, &dst) {
            outcome = PoseOutcome {
                pose: Some(t),
                source: PoseSource::RigidFit { from_2d },
                combinations_capped,
            };
        }
    } else if !settings.disable_advanced
        && let Some(c) = epnp_camera(frame, cameras.len(), compat)
    {
        let pose = epnp_pose(frame, cameras, c, compat);
        outcome = PoseOutcome {
            pose,
            source: if pose.is_some() || compat.epnp_last_camera {
                PoseSource::Epnp { camera: c }
            } else {
                PoseSource::None
            },
            combinations_capped,
        };
    }

    if settings.optimize
        && let Some(pose) = outcome.pose
    {
        let obs = crate::pose_refine::refinement_observations(frame, cameras, settings.compat);
        let refine_cams: Vec<_> = cameras
            .iter()
            .map(|c| crate::pose_refine::RefineCamera::from_pose_camera(c, settings.compat))
            .collect();
        outcome.pose = Some(crate::pose_refine::refine_pose(
            &pose,
            &obs,
            &refine_cams,
            compat,
        ));
    }
    outcome
}

// ---------------------------------------------------------------------------------------
// Projection and errors
// ---------------------------------------------------------------------------------------

/// Projects world points into a camera's raw image the way `RigidBody::projectToImage`
/// does: through `cv::Point3f`, `cv::projectPoints` with the reference calibration image's
/// pose, the camera matrix and no distortion coefficients into `cv::Point2f`, then
/// `Camera::undistortPoint(pt, false)` (grid, then lens model).
///
/// Ported from `core/RigidBody.cpp:2416-2455` (the part after the points are in world
/// coordinates; with `dummy_frame` the C++ passes world points directly). Tier A.
///
/// Without [`RigidCompat::single_precision`] (ledger A23 fixed) the points are projected in
/// double ([`crate::camera::project_points`]).
#[must_use]
pub fn project_world_points(
    world: &[Point3],
    camera: &PoseCamera,
    compat: RigidCompat,
) -> Vec<Point2> {
    if world.is_empty() {
        return Vec::new();
    }
    if !compat.single_precision {
        return crate::camera::project_points(
            world,
            &camera.pose.rotation,
            &camera.pose.translation,
            &camera.model.intrinsics.camera_matrix,
            &[],
        )
        .into_iter()
        .map(|pt| camera.model.undistort_point(pt, false, true, true))
        .collect();
    }
    let object: Vec<nalgebra::Point3<f32>> = world
        .iter()
        .map(|p| nalgebra::Point3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let image = project_points_f32(
        &object,
        &camera.pose.rotation,
        &camera.pose.translation,
        &camera.model.intrinsics.camera_matrix,
        &[],
    );
    image
        .into_iter()
        .map(|p| {
            let pt = Point2::new(f64::from(p.x), f64::from(p.y));
            camera.model.undistort_point(pt, false, true, true)
        })
        .collect()
}

/// Projects points given in body coordinates (reference points, virtual points, the body
/// centre) into a camera's raw image with a pose: [`body_to_world`] then
/// [`project_world_points`].
///
/// Ported from `core/RigidBody.cpp:2321-2455`, `RigidBody::projectToImage`. Tier A.
/// Oracle: through [`pose_errors`], `trial_<t>/rb_errors.csv`.
#[must_use]
pub fn project_to_image(
    pose: &RigidTransform,
    body_points: &[Point3],
    camera: &PoseCamera,
    compat: RigidCompat,
) -> Vec<Point2> {
    let world: Vec<Point3> = body_points
        .iter()
        .map(|p| body_to_world(pose, *p))
        .collect();
    project_world_points(&world, camera, compat)
}

/// Mean and `count - 1` standard deviation of `dist` as `updateError` forms them: the sum
/// in order divided by the int count, and NaN for a single value (ledger A20; with
/// `sd_nan` false, 0 for a single value, as for markers). Zeros when empty.
fn mean_sd(dist: &[f64], sd_nan: bool) -> (f64, f64) {
    if !sd_nan && dist.len() == 1 {
        return (dist[0], 0.0);
    }
    let count = dist.len() as i32;
    let mut mean = 0.0f64;
    for d in dist {
        mean += d;
    }
    if count != 0 {
        mean /= f64::from(count);
    }
    let mut sd = 0.0f64;
    for d in dist {
        // pow(x, 2): the C library's, as the reference build calls it.
        sd += crate::cmath::pow(d - mean, 2.0);
    }
    if count != 0 {
        sd = (sd / f64::from(count - 1)).sqrt();
    }
    (mean, sd)
}

/// The per-frame error measures of a pose (unfiltered or filtered).
///
/// Ported from `core/RigidBody.cpp:1231-1338`, `RigidBody::updateError`:
///
/// - 2D: for every camera and every body marker with a 2D point, the distance between the
///   stored (raw) 2D point and the reference point projected with [`project_to_image`];
/// - 3D: for every body marker with a 3D point, the distance between it and the reference
///   point mapped to world with [`body_to_world`].
///
/// Each mean is the sum in order divided by the count, each standard deviation divides by
/// `count - 1` (NaN for one value with [`RigidCompat::error_sd_nan`], ledger A20; 0
/// otherwise); a measure with no values is 0. With no pose
/// (`pose` `None`) or no reference points, all four are 0. The marker positions are the
/// unfiltered ones in both cases, as in the C++. Tier A.
/// Oracle: `trial_<t>/rb_errors.csv`, `rb_errors_filtered.csv`.
///
/// Panics if `frame.markers` and `frame.reference_points` differ in length.
#[must_use]
pub fn pose_errors(
    frame: &RigidBodyFrame,
    cameras: &[PoseCamera],
    pose: Option<&RigidTransform>,
    compat: RigidCompat,
) -> PoseErrors {
    assert_eq!(
        frame.markers.len(),
        frame.reference_points.len(),
        "one reference point per body marker"
    );
    let Some(pose) = pose else {
        return PoseErrors::ZERO;
    };
    if frame.reference_points.is_empty() {
        return PoseErrors::ZERO;
    }

    let mut dist = Vec::new();
    for (c, cam) in cameras.iter().enumerate() {
        let image_points = project_to_image(pose, &frame.reference_points, cam, compat);
        for (i, marker) in frame.markers.iter().enumerate() {
            if let Some(view) = marker.views.get(c).copied().flatten() {
                let dx = view.raw.x - image_points[i].x;
                let dy = view.raw.y - image_points[i].y;
                dist.push((dx * dx + dy * dy).sqrt());
            }
        }
    }
    let (mean_2d, sd_2d) = mean_sd(&dist, compat.error_sd_nan);

    let r = rodrigues_to_matrix(&pose.rotation);
    let t = varr(&pose.translation);
    dist.clear();
    for (reference, marker) in frame.reference_points.iter().zip(&frame.markers) {
        let w = gemm_t_3x3_3x1(
            &r,
            [reference.x - t[0], reference.y - t[1], reference.z - t[2]],
        );
        if let Some(p) = marker.point3d {
            let dx = p.x - w[0];
            let dy = p.y - w[1];
            let dz = p.z - w[2];
            dist.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    let (mean_3d, sd_3d) = mean_sd(&dist, compat.error_sd_nan);

    PoseErrors {
        mean_2d,
        sd_2d,
        mean_3d,
        sd_3d,
    }
}

/// The 3x4 projection matrix `K [R | t]` of a camera's reference calibration image, as
/// `Camera::getProjectionMatrix` gives it (used for virtual points in the refinement).
pub(crate) fn camera_projection(cam: &PoseCamera) -> crate::Matrix3x4 {
    projection_matrix(&cam.model.intrinsics.camera_matrix, &cam.pose)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_helpers_follow_opencv_order() {
        let a = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0);
        let v = [1.0, -1.0, 0.5];
        assert_eq!(gemm_3x3_3x1(&a, v, 1.0, None, 0.0), [0.5, 2.0, 4.0]);
        assert_eq!(
            gemm_3x3_3x1(&a, v, -1.0, Some([1.0, 1.0, 1.0]), 1.0),
            [0.5, -1.0, -3.0]
        );
        assert_eq!(gemm_t_3x3_3x1(&a, v), [0.5, 1.0, 2.0]);
        let r = gemm_small(&rows3(&a), &rows3(&Matrix3::identity()));
        assert_eq!(from_rows3(&r), a);
    }

    #[test]
    fn divide_by_norm_cases() {
        let mut v = [3.0, 4.0, 0.0];
        let n = norm3(v);
        divide_by_norm(&mut v, n);
        assert_eq!(v, [3.0 * 0.2, 4.0 * 0.2, 0.0]);
        let mut v = [1.0, -0.0, 0.0];
        divide_by_norm(&mut v, 1.0);
        // alpha == 1: v + 0 turns -0 into +0.
        assert!(v[1].is_sign_positive());
    }

    #[test]
    fn through_f32_rounds() {
        let p = through_f32(Point3::new(0.1, 1.0 / 3.0, -2.5));
        assert_eq!(p.x, f64::from(0.1f32));
        assert_eq!(p.y, f64::from((1.0f64 / 3.0) as f32));
        assert_eq!(p.z, -2.5);
    }

    /// The list the C++ (and this port before review 11) built in
    /// `findFrom2Dwith3D`, kept to check [`for_each_combination`] against.
    fn combination_list(counts: &[usize]) -> Vec<Vec<usize>> {
        let mut combinations: Vec<Vec<usize>> = Vec::new();
        for &count in counts {
            let old = std::mem::take(&mut combinations);
            for k in 0..count {
                if old.is_empty() {
                    combinations.push(vec![k]);
                } else {
                    for c in &old {
                        let mut c = c.clone();
                        c.push(k);
                        combinations.push(c);
                    }
                }
            }
        }
        combinations
    }

    #[test]
    fn review11_combinations_in_the_order_of_the_list() {
        let patterns: &[&[usize]] = &[
            &[],
            &[0],
            &[1],
            &[2],
            &[4],
            &[0, 0],
            &[1, 1, 1],
            &[2, 3],
            &[3, 2],
            &[4, 4, 4],
            &[2, 1, 4, 2],
            &[0, 2, 3],
            &[2, 0, 3],
            &[2, 3, 0],
            &[4, 0, 0, 2, 1],
            &[1, 2, 0, 4, 2, 0, 3, 1, 2],
            &[0, 4, 4, 0, 1],
            &[4, 4, 4, 4, 4, 4],
            &[2, 2, 2, 2, 2, 2, 2, 2],
            &[1, 0, 1],
        ];
        for counts in patterns {
            let mut seen: Vec<Vec<usize>> = Vec::new();
            for_each_combination(counts, |c| seen.push(c.to_vec()));
            assert_eq!(seen, combination_list(counts), "counts {counts:?}");
        }
        // Every pattern of up to five rays with 0 to 4 candidates each.
        for n in 0..=5u32 {
            for code in 0..5usize.pow(n) {
                let counts: Vec<usize> = (0..n).map(|i| code / 5usize.pow(i) % 5).collect();
                let mut seen: Vec<Vec<usize>> = Vec::new();
                for_each_combination(&counts, |c| seen.push(c.to_vec()));
                assert_eq!(seen, combination_list(&counts), "counts {counts:?}");
            }
        }
    }
}
