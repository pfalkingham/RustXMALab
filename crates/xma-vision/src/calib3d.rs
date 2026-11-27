//! `cv::calibrateCamera` as XMALab calls it: intrinsics, lens coefficients and one pose per
//! view from matched 3D and image points.
//!
//! Work package "calib3d" (phase 4). Serves the cube calibration
//! (`processing/CubeCalibration.cpp`, `calibrateOpenCV`) and the per-camera refinement over
//! all images (`processing/Calibration.cpp`, `Calibration::process`), including its planar
//! (checkerboard) call: `use_intrinsic_guess` false, the lens-model flags as for the cube
//! (`FIX_K1..K6` + `ZERO_TANGENT_DIST`, or `rational_model`), optionally
//! `fix_principal_point`; eight starting coefficients (not read), one view per image with
//! detected corners, object points `(x s, y s, 0)` (checked against the 4.13 libraries on
//! detected corners, `crates/xma-vision/tests/chessboard_calib.rs`).
//!
//! Ported from OpenCV 4.13 `modules/calib3d/src/calibration.cpp` (`cv::calibrateCameraRO`,
//! `calibrateCameraInternal`, `initIntrinsicParams2D`), `calibration_base.cpp`
//! (`findExtrinsicCameraParams2`, `projectPoints` and `Rodrigues` with their derivatives),
//! `compat_ptsetreg.cpp` (`CvLevMarq`), `fundam.cpp` and `levmarq.cpp` (`findHomography`
//! with method 0) and the `core` routines they reach (submodule `cvcore`, and
//! [`crate::linalg`] for the SVD, eigen decomposition and solves).
//!
//! # Parity
//!
//! - Every flag set XMALab uses is ported: `CALIB_USE_INTRINSIC_GUESS` with `FIX_K1..K6` and
//!   `ZERO_TANGENT_DIST` (lens model off) or `CALIB_RATIONAL_MODEL`, optionally
//!   `CALIB_FIX_PRINCIPAL_POINT`, and the planar case without an intrinsic guess.
//! - Wherever the reference OpenCV runs its own code, the results are bit-identical: every
//!   recorded call of the calibration runs and every probe case below the sizes in the next
//!   point (tests `calib3d_runs.rs`, `calib3d_primitives.rs`).
//! - The reference hands `J^T J` products of views with 50 or more points to BLAS and the
//!   normal-equation solve with 25 or more free parameters (four views without the lens
//!   model, three with it) to LAPACK; this port always runs OpenCV's own code, so results
//!   there agree to rounding amplified by the iterations (parity ledger A37).
//! - Planar objects: `findHomography` takes an eigenvector from `cv::eigen`, which the
//!   reference OpenCV (built with Eigen) computes with Eigen; the port uses OpenCV's own
//!   Jacobi method (module [`homography`]), so planar calibrations agree to rounding.
//! - Not ported: the tilted model (`tau_x`, `tau_y` non-zero), which XMALab never uses:
//!   [`CalibrateError::Rejected`].

use crate::Matrix3;
use crate::camera::RigidTransform;

/// The `cv::CALIB_*` flags XMALab passes (`processing/CubeCalibration.cpp:325-421`,
/// `processing/Calibration.cpp:238-316`). Any other flag of OpenCV is off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CalibrateFlags {
    /// `CALIB_USE_INTRINSIC_GUESS`: `camera_matrix` holds the starting values. Without it
    /// (XMALab's planar case) OpenCV starts from `initCameraMatrix2D`.
    pub use_intrinsic_guess: bool,
    /// `CALIB_FIX_PRINCIPAL_POINT` (the setting `FixPrincipal`).
    pub fix_principal_point: bool,
    /// `CALIB_ZERO_TANGENT_DIST`.
    pub zero_tangent_dist: bool,
    /// `CALIB_FIX_K1` to `CALIB_FIX_K6`.
    pub fix_k: [bool; 6],
    /// `CALIB_RATIONAL_MODEL`: k4, k5 and k6 are part of the model (eight coefficients).
    pub rational_model: bool,
}

/// What `cv::calibrateCamera` writes.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrateOutput {
    /// The return value: the root mean square reprojection error over every point.
    pub rms: f64,
    pub camera_matrix: Matrix3,
    /// The coefficients in OpenCV order, as many as OpenCV writes for the flags (five, or
    /// eight with the rational model).
    pub distortion: Vec<f64>,
    /// One pose per view, object to camera, in the order of the views.
    pub poses: Vec<RigidTransform>,
}

/// Why no calibration was returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrateError {
    /// Where OpenCV throws a `cv::Exception` (XMALab catches it and keeps the previous
    /// values): the text says which check failed.
    Rejected(String),
    /// The routine is not ported yet (no longer returned; kept for the callers written
    /// against the interface).
    NotPorted,
}

impl std::fmt::Display for CalibrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrateError::Rejected(t) => write!(f, "calibrateCamera: {t}"),
            CalibrateError::NotPorted => write!(f, "calibrateCamera is not ported yet"),
        }
    }
}

impl std::error::Error for CalibrateError {}

pub(crate) mod cvcore;
pub mod extrinsic;
pub mod homography;
mod levmarq;
pub mod project;

use levmarq::{LevMarq, State};
use project::{JacOut, project_points_jac};

/// `CALIB_NINTRINSIC`: fx, fy, cx, cy and 14 distortion coefficients.
const NINTRINSIC: usize = 18;

fn rejected<T>(text: impl Into<String>) -> Result<T, CalibrateError> {
    Err(CalibrateError::Rejected(text.into()))
}

/// `cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
/// rvecs, tvecs, flags)` with OpenCV's default termination criteria
/// (`TermCriteria(COUNT + EPS, 30, DBL_EPSILON)`).
///
/// - `object_points[v]` and `image_points[v]` are the matched points of view `v`, in single
///   precision as XMALab passes them (`std::vector<cv::Point3f>`, `std::vector<cv::Point2f>`).
/// - `image_size` is (width, height).
/// - `camera_matrix` is read as `CV_64F`; `distortion` holds the starting coefficients
///   (XMALab passes five zeros in single precision for one view, and eight doubles for the
///   refinement over all images; a `float` value widened to `f64` is exactly what OpenCV's
///   `convertTo` gives).
/// - The returned `distortion` has five coefficients, or eight with the rational model
///   (OpenCV writes fourteen there; the last six are the thin-prism and tilt terms, which
///   these flags always keep at zero).
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration.cpp`, `cv::calibrateCameraRO`
/// (with `prepareCameraMatrix`, `prepareDistCoeffs`, `collectCalibrationData`) and
/// `calibrateCameraInternal`; the pieces it calls are in the submodules.
///
/// Non-finite values (ledger A157, both behaviours): a NaN or infinite object point, image
/// point, camera-matrix entry or starting coefficient is refused before OpenCV's
/// computation (the camera matrix and the coefficients only with an intrinsic guess, since
/// without one OpenCV does not read them, ledger A172), and a result with a non-finite RMS,
/// camera-matrix entry, coefficient or pose is refused after it. OpenCV's own checks are comparisons that NaN passes, so OpenCV
/// returns NaN values there; XMALab would store them.
///
/// # Errors
/// [`CalibrateError::Rejected`] with OpenCV's message where OpenCV throws; also for the
/// tilted model (a non-zero 13th or 14th coefficient), which is not ported; and, with a
/// sentence saying which, for a non-finite input or result (A157).
pub fn calibrate_camera(
    object_points: &[Vec<nalgebra::Point3<f32>>],
    image_points: &[Vec<nalgebra::Point2<f32>>],
    image_size: (usize, usize),
    camera_matrix: &Matrix3,
    distortion: &[f64],
    flags: CalibrateFlags,
) -> Result<CalibrateOutput, CalibrateError> {
    for (v, view) in object_points.iter().enumerate() {
        if let Some(j) = view
            .iter()
            .position(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
        {
            return rejected(format!(
                "object point {j} of view {v} is not finite (NaN or infinite)"
            ));
        }
    }
    for (v, view) in image_points.iter().enumerate() {
        if let Some(j) = view
            .iter()
            .position(|p| !(p.x.is_finite() && p.y.is_finite()))
        {
            return rejected(format!(
                "image point {j} of view {v} is not finite (NaN or infinite)"
            ));
        }
    }
    // Without an intrinsic guess OpenCV reads neither the camera matrix nor the starting
    // coefficients (the planar call of Calibration.cpp passes whatever its DLT left there),
    // so only a guess is checked (parity ledger A172).
    if flags.use_intrinsic_guess {
        if let Some(i) = camera_matrix.iter().position(|v| !v.is_finite()) {
            return rejected(format!(
                "camera matrix entry ({}, {}) is not finite (NaN or infinite)",
                i % 3,
                i / 3
            ));
        }
        if let Some(i) = distortion.iter().position(|v| !v.is_finite()) {
            return rejected(format!(
                "distortion coefficient {i} is not finite (NaN or infinite)"
            ));
        }
    }
    let out = calibrate_camera_opencv(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        distortion,
        flags,
    )?;
    if !out.rms.is_finite() {
        return rejected("the result is not finite: the reprojection error is NaN or infinite");
    }
    if out.camera_matrix.iter().any(|v| !v.is_finite()) {
        return rejected("the result is not finite: the camera matrix has a NaN or infinite entry");
    }
    if out.distortion.iter().any(|v| !v.is_finite()) {
        return rejected("the result is not finite: a distortion coefficient is NaN or infinite");
    }
    if out.poses.iter().any(|p| {
        p.rotation.iter().any(|v| !v.is_finite()) || p.translation.iter().any(|v| !v.is_finite())
    }) {
        return rejected("the result is not finite: a view's pose is NaN or infinite");
    }
    Ok(out)
}

/// The body of [`calibrate_camera`]: OpenCV's computation, without the checks of non-finite
/// values.
#[allow(
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::needless_range_loop
)]
fn calibrate_camera_opencv(
    object_points: &[Vec<nalgebra::Point3<f32>>],
    image_points: &[Vec<nalgebra::Point2<f32>>],
    image_size: (usize, usize),
    camera_matrix: &Matrix3,
    distortion: &[f64],
    flags: CalibrateFlags,
) -> Result<CalibrateOutput, CalibrateError> {
    // --- calibrateCameraRO ---
    // prepareDistCoeffs(distCoeffs, CV_64F, 14).
    let n_in = distortion.len();
    if n_in > 14 {
        return rejected("(int)distCoeffs0.total() <= outputSize");
    }
    let mut dist14 = [0.0f64; 14];
    if matches!(n_in, 4 | 5 | 8 | 12 | 14) {
        dist14[..n_in].copy_from_slice(distortion);
    }
    let ndist = if flags.rational_model { 14 } else { 5 };
    let nimages = object_points.len();
    if nimages == 0 {
        return rejected("nimages > 0");
    }
    // collectCalibrationData.
    if nimages != image_points.len() {
        return rejected(format!(
            ">  (expected: 'nimages == (int)imagePoints1.total()'), where\n>     'nimages' is {nimages}\n> must be equal to\n>     '(int)imagePoints1.total()' is {}\n",
            image_points.len()
        ));
    }
    for i in 0..nimages {
        if object_points[i].is_empty() {
            return rejected("objectPoints should not contain empty vector of vectors of points");
        }
        if image_points[i].is_empty() {
            return rejected("imagePoints1 should not contain empty vector of vectors of points");
        }
        let (no, ni) = (object_points[i].len(), image_points[i].len());
        if no != ni {
            return rejected(format!(
                "> Number of object and image points must be equal (expected: 'numberOfObjectPoints == numberOfImagePoints'), where\n>     'numberOfObjectPoints' is {no}\n> must be equal to\n>     'numberOfImagePoints' is {ni}\n"
            ));
        }
    }

    // --- calibrateCameraInternal ---
    // Only the rational model keeps all 14 coefficients, and only an intrinsic guess reads
    // them: then a non-zero tau_x or tau_y would need the tilted model.
    let flags_tilt_ok = !(flags.rational_model
        && flags.use_intrinsic_guess
        && (dist14[12] != 0.0 || dist14[13] != 0.0));
    if image_size.0 == 0 || image_size.1 == 0 {
        return rejected("image width and height must be positive");
    }
    let npoints: Vec<usize> = object_points.iter().map(Vec::len).collect();
    let mut total = 0usize;
    let mut max_points = 0usize;
    for (i, &ni) in npoints.iter().enumerate() {
        if ni < 4 {
            return rejected(format!("The number of points in the view #{i} is < 4"));
        }
        max_points = max_points.max(ni);
        total += ni;
    }
    let mut mat_m: Vec<[f64; 3]> = object_points
        .iter()
        .flatten()
        .map(|p| [f64::from(p.x), f64::from(p.y), f64::from(p.z)])
        .collect();
    let m_img: Vec<[f64; 2]> = image_points
        .iter()
        .flatten()
        .map(|p| [f64::from(p.x), f64::from(p.y)])
        .collect();

    let nparams = NINTRINSIC + nimages * 6;
    let mut k = [0.0f64; 14];
    let mut fix_k = flags.fix_k;
    // distCoeffs.total() < 8: k4..k6 fixed (5 coefficients without the rational model).
    if ndist < 8 {
        fix_k[3] = true;
        fix_k[4] = true;
        fix_k[5] = true;
    }

    let mut a = *camera_matrix;
    if flags.use_intrinsic_guess {
        if a[(0, 0)] <= 0.0 || a[(1, 1)] <= 0.0 {
            return rejected("Focal length (fx and fy) must be positive");
        }
        #[allow(clippy::cast_precision_loss)]
        let (w, h) = (image_size.0 as f64, image_size.1 as f64);
        if a[(0, 2)] < 0.0 || a[(0, 2)] >= w || a[(1, 2)] < 0.0 || a[(1, 2)] >= h {
            return rejected("Principal point must be within the image");
        }
        if a[(0, 1)].abs() > 1e-5 {
            return rejected("Non-zero skew is not supported by the function");
        }
        if a[(1, 0)].abs() > 1e-5
            || a[(2, 0)].abs() > 1e-5
            || a[(2, 1)].abs() > 1e-5
            || (a[(2, 2)] - 1.0).abs() > 1e-5
        {
            return rejected("The intrinsic matrix must have [fx 0 cx; 0 fy cy; 0 0 1] shape");
        }
        a[(0, 1)] = 0.0;
        a[(1, 0)] = 0.0;
        a[(2, 0)] = 0.0;
        a[(2, 1)] = 0.0;
        a[(2, 2)] = 1.0;
        k[..ndist].copy_from_slice(&dist14[..ndist]);
    } else {
        // meanStdDev(matM): a planar rig must have z == 0.
        #[allow(clippy::cast_precision_loss)]
        let nt = total as f64;
        let mut s = 0.0f64;
        let mut sq = 0.0f64;
        for p in &mat_m {
            s += p[2];
            sq += p[2] * p[2];
        }
        let mean = s * (1.0 / nt);
        let var = (sq * (1.0 / nt) - mean * mean).max(0.0);
        if mean.abs() > 1e-5 || var.sqrt() > 1e-5 {
            return rejected(
                "For non-planar calibration rigs the initial intrinsic matrix must be specified",
            );
        }
        for p in &mut mat_m {
            p[2] = 0.0;
        }
        let a0 = homography::init_intrinsic_params_2d(&mat_m, &m_img, &npoints, image_size)
            .map_err(CalibrateError::Rejected)?;
        a = Matrix3::from_row_slice(&a0);
    }
    if !flags_tilt_ok {
        return rejected("the tilted distortion model (tau_x, tau_y) is not ported");
    }

    let mut solver = LevMarq::new(nparams, 0, 30, f64::EPSILON, false);
    {
        let param = &mut solver.param;
        let mask = &mut solver.mask;
        param[0] = a[(0, 0)];
        param[1] = a[(1, 1)];
        param[2] = a[(0, 2)];
        param[3] = a[(1, 2)];
        param[4..18].copy_from_slice(&k);
        if flags.fix_principal_point {
            mask[2] = false;
            mask[3] = false;
        }
        if flags.zero_tangent_dist {
            param[6] = 0.0;
            param[7] = 0.0;
            mask[6] = false;
            mask[7] = false;
        }
        if !flags.rational_model {
            fix_k[3] = true;
            fix_k[4] = true;
            fix_k[5] = true;
        }
        mask[4] = !fix_k[0];
        mask[5] = !fix_k[1];
        mask[8] = !fix_k[2];
        mask[9] = !fix_k[3];
        mask[10] = !fix_k[4];
        mask[11] = !fix_k[5];
        // CALIB_FIX_S1_S2_S3_S4 and CALIB_FIX_TAUX_TAUY (no thin-prism or tilted model).
        for m in &mut mask[12..18] {
            *m = false;
        }
    }
    let nparams_nz = solver.mask.iter().filter(|&&m| m).count();
    if nparams_nz >= 2 * total {
        return rejected(format!(
            "There should be less vars to optimize (having {nparams_nz}) than the number of residuals ({} = 2 per point)",
            2 * total
        ));
    }

    // 2. Initial extrinsics of every view.
    let a9 = [
        a[(0, 0)],
        a[(0, 1)],
        a[(0, 2)],
        a[(1, 0)],
        a[(1, 1)],
        a[(1, 2)],
        a[(2, 0)],
        a[(2, 1)],
        a[(2, 2)],
    ];
    let mut pos = 0usize;
    for (i, &ni) in npoints.iter().enumerate() {
        let s = NINTRINSIC + i * 6;
        let (r, t) = extrinsic::find_extrinsic_camera_params2(
            &mat_m[pos..pos + ni],
            &m_img[pos..pos + ni],
            &a9,
            &k[..ndist],
        )
        .map_err(CalibrateError::Rejected)?;
        solver.param[s..s + 3].copy_from_slice(&r);
        solver.param[s + 3..s + 6].copy_from_slice(&t);
        pos += ni;
    }

    // 3. The optimisation.
    let mut ji_buf = vec![0.0f64; max_points * 2 * NINTRINSIC];
    let mut je_buf = vec![0.0f64; max_points * 2 * 6];
    let mut err_buf = vec![0.0f64; max_points * 2];
    let mut reproj_err = 0.0f64;
    loop {
        let st = solver.update_alt();
        let calc_j = solver.state == State::CalcJ;
        if !st.proceed {
            break;
        }
        let p = solver.param.clone();
        let intrin = [p[0], 0.0, p[2], 0.0, p[1], p[3], 0.0, 0.0, 1.0];
        let dist = &p[4..18];
        reproj_err = 0.0;
        let mut pos = 0usize;
        for (i, &ni) in npoints.iter().enumerate() {
            let si = NINTRINSIC + i * 6;
            let ri = [p[si], p[si + 1], p[si + 2]];
            let ti = [p[si + 3], p[si + 4], p[si + 5]];
            let mi = &mat_m[pos..pos + ni];
            let rows = ni * 2;
            let err = &mut err_buf[..rows];
            if calc_j {
                let mut dr = vec![0.0f64; rows * 3];
                let mut dt = vec![0.0f64; rows * 3];
                let mut df = vec![0.0f64; rows * 2];
                let mut dc = vec![0.0f64; rows * 2];
                let mut dk = vec![0.0f64; rows * 14];
                project_points_jac(
                    mi,
                    &ri,
                    &ti,
                    &intrin,
                    dist,
                    err,
                    Some(JacOut {
                        buf: &mut dr,
                        stride: 3,
                        col: 0,
                    }),
                    Some(JacOut {
                        buf: &mut dt,
                        stride: 3,
                        col: 0,
                    }),
                    Some(JacOut {
                        buf: &mut df,
                        stride: 2,
                        col: 0,
                    }),
                    if flags.fix_principal_point {
                        None
                    } else {
                        Some(JacOut {
                            buf: &mut dc,
                            stride: 2,
                            col: 0,
                        })
                    },
                    Some(JacOut {
                        buf: &mut dk,
                        stride: 14,
                        col: 0,
                    }),
                );
                for r in 0..rows {
                    let je = &mut je_buf[r * 6..r * 6 + 6];
                    je[..3].copy_from_slice(&dr[r * 3..r * 3 + 3]);
                    je[3..].copy_from_slice(&dt[r * 3..r * 3 + 3]);
                    let ji = &mut ji_buf[r * NINTRINSIC..(r + 1) * NINTRINSIC];
                    ji[..2].copy_from_slice(&df[r * 2..r * 2 + 2]);
                    if !flags.fix_principal_point {
                        ji[2..4].copy_from_slice(&dc[r * 2..r * 2 + 2]);
                    }
                    ji[4..].copy_from_slice(&dk[r * 14..r * 14 + 14]);
                }
            } else {
                project_points_jac(
                    mi, &ri, &ti, &intrin, dist, err, None, None, None, None, None,
                );
            }
            // subtract(_mp, _mi, _mp).
            for (q, e) in err.iter_mut().enumerate() {
                *e -= m_img[pos + q / 2][q % 2];
            }
            if calc_j {
                let ji = &ji_buf[..rows * NINTRINSIC];
                let je = &je_buf[..rows * 6];
                let n = nparams;
                // JtJ(Rect(0, 0, 18, 18)) += Ji^T Ji.
                let t =
                    cvcore::gemm_at_b(ji, NINTRINSIC, NINTRINSIC, ji, NINTRINSIC, NINTRINSIC, rows);
                for r in 0..NINTRINSIC {
                    for c in 0..NINTRINSIC {
                        solver.jtj[r * n + c] += t[r * NINTRINSIC + c];
                    }
                }
                // JtJ(Rect(si, si, 6, 6)) = Je^T Je.
                let t = cvcore::gemm_at_b(je, 6, 6, je, 6, 6, rows);
                for r in 0..6 {
                    for c in 0..6 {
                        solver.jtj[(si + r) * n + si + c] = t[r * 6 + c];
                    }
                }
                // JtJ(Rect(si, 0, 6, 18)) = Ji^T Je.
                let t = cvcore::gemm_at_b(ji, NINTRINSIC, NINTRINSIC, je, 6, 6, rows);
                for r in 0..NINTRINSIC {
                    for c in 0..6 {
                        solver.jtj[r * n + si + c] = t[r * 6 + c];
                    }
                }
                // JtErr.rowRange(0, 18) += Ji^T err; JtErr.rowRange(si, si + 6) = Je^T err.
                let t = cvcore::gemm_at_v(ji, NINTRINSIC, NINTRINSIC, err, rows);
                for r in 0..NINTRINSIC {
                    solver.jt_err[r] += t[r];
                }
                let t = cvcore::gemm_at_v(je, 6, 6, err, rows);
                solver.jt_err[si..si + 6].copy_from_slice(&t);
            }
            reproj_err += cvcore::norm_l2sqr(err);
            pos += ni;
        }
        if st.want_err {
            solver.err_norm = reproj_err;
        }
    }

    // 4. The results.
    let p = &solver.param;
    let camera_matrix = Matrix3::new(p[0], 0.0, p[2], 0.0, p[1], p[3], 0.0, 0.0, 1.0);
    let nout = if flags.rational_model { 8 } else { 5 };
    let distortion = p[4..4 + nout].to_vec();
    let poses = (0..nimages)
        .map(|i| {
            let s = NINTRINSIC + i * 6;
            RigidTransform {
                rotation: crate::Vector3::new(p[s], p[s + 1], p[s + 2]),
                translation: crate::Vector3::new(p[s + 3], p[s + 4], p[s + 5]),
            }
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let rms = (reproj_err / total as f64).sqrt();
    Ok(CalibrateOutput {
        rms,
        camera_matrix,
        distortion,
        poses,
    })
}
