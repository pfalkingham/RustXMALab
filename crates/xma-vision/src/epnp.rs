//! Camera pose from 3D to 2D point correspondences by the EPnP method.
//!
//! Work package: EPnP. Ported from OpenCV 4.13, `cv::solvePnP` with `SOLVEPNP_EPNP`
//! (`modules/calib3d/src/solvepnp.cpp` and `epnp.cpp`). The C++ uses it as the last fallback
//! for a rigid-body pose (`core/RigidBody.cpp:1172`) and for pose-only cube calibration
//! (`processing/CubeCalibration.cpp:708`).
//!
//! # What runs
//!
//! [`solve_pnp_epnp`] is the path `cv::solvePnP` → `cv::solvePnPGeneric` takes for
//! `SOLVEPNP_EPNP` with single-precision points (`cv::Point3f`, `cv::Point2f`), as both
//! XMALab call sites pass them:
//!
//! 1. the point-count check (at least 4, same count of object and image points);
//! 2. `cv::undistortPoints` of the image points to normalised coordinates with the given
//!    camera matrix and distortion and the default criteria (5 iterations). The output has
//!    the input's depth, so the normalised points are rounded to `f32`;
//! 3. the `epnp` class with that camera matrix, which multiplies the normalised points back
//!    by `fx`, `fy` and adds `cx`, `cy` (in `f64`), so the pose is computed on the
//!    undistorted pixel positions;
//! 4. `cv::Rodrigues` of the rotation matrix.
//!
//! # Decompositions
//!
//! All go through [`crate::linalg`], and all are far below the sizes at which the reference
//! OpenCV hands them to LAPACK (SVD at `max(rows, cols) >= 25`):
//!
//! | Where | OpenCV call | Size | Rust |
//! |---|---|---|---|
//! | `choose_control_points` | `cvSVD(.., CV_SVD_MODIFY_A \| CV_SVD_U_T)` | 3 x 3 | [`linalg::svd`] |
//! | `compute_barycentric_coordinates` | `cvInvert(.., CV_SVD)` | 3 x 3 | [`linalg::invert_svd`] |
//! | `compute_pose` | `cvSVD(.., CV_SVD_MODIFY_A \| CV_SVD_U_T)` | 12 x 12 | [`linalg::svd`] |
//! | `find_betas_approx_1/2/3` | `cvSolve(.., CV_SVD)` | 6 x 4, 6 x 3, 6 x 5 | [`linalg::solve_svd`] |
//! | `qr_solve` (Gauss-Newton) | EPnP's own Householder QR | 6 x 4 | ported here |
//! | `estimate_R_and_t` | `cvSVD(.., CV_SVD_MODIFY_A)` | 3 x 3 | [`linalg::svd`] |
//! | `cv::Rodrigues` | `SVD::compute` | 3 x 3 | [`camera::matrix_to_rodrigues`] |
//!
//! The products `cvMulTransposed(M, MtM, 1)` (`(2n x 12)^T (2n x 12)`) and
//! `(n x 3)^T (n x 3)` are OpenCV's `MulTransposedR`
//! (`calib3d::cvcore::mul_transposed_ata`; below `gemm_level = 100`, so never `gemm`).
//!
//! # Floating point
//!
//! Every function follows its source operation by operation, in the same order, at the same
//! width, and never fuses a multiply and an add.

// The port keeps OpenCV's index loops so it can be checked line by line against the C++.
#![allow(clippy::needless_range_loop)]

use crate::calib3d::cvcore;
use crate::camera::{self, RigidTransform, TermCriteria};
use crate::linalg::{self, SvdMode};
use crate::{Matrix3, Point2, Vector3};
use nalgebra::DMatrix;

/// Camera pose of an object from at least four 3D-2D correspondences: OpenCV
/// `cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false,
/// SOLVEPNP_EPNP)` with `std::vector<cv::Point3f>` and `std::vector<cv::Point2f>`.
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/solvepnp.cpp`, `cv::solvePnP` and
/// `cv::solvePnPGeneric` (the `SOLVEPNP_EPNP` branch), and `modules/calib3d/src/epnp.cpp`,
/// class `epnp`. Serves `core/RigidBody.cpp:1172` (eight zero distortion coefficients,
/// image points undistorted beforehand) and `processing/CubeCalibration.cpp:708` (an empty
/// `std::vector<float>` of coefficients, at least five points). Tier A.
///
/// - `camera_matrix` is read as `CV_64F` (`Mat_<double>`); only `fx`, `fy`, `cx`, `cy` are
///   used (the skew is ignored, by `undistortPoints` and by `epnp`).
/// - `distortion` is OpenCV's `distCoeffs`: empty (no distortion, no iteration) or 4, 5, 8,
///   12 or 14 coefficients in OpenCV order. A `std::vector<float>` is passed widened to
///   `f64`, which is what `Mat_<double>` does.
/// - Returns the rotation vector and translation that `solvePnP` writes (the object-to-camera
///   transform), or `None` where OpenCV throws a `cv::Exception` instead of returning:
///   fewer than 4 points, different numbers of object and image points, or a distortion
///   vector of another length. When it does not throw, `solvePnP` with `SOLVEPNP_EPNP`
///   always returns `true` (one solution), so there is no other failure. Degenerate input
///   (collinear or repeated points) gives whatever numbers EPnP computes, NaN included,
///   exactly as OpenCV does.
///
/// Panics, as [`camera::undistort_points`] does, for the tilted model (a non-zero 13th or
/// 14th coefficient), which is not ported.
///
/// Oracle: primitive cases `solvepnp-epnp-*`; the differential fixture
/// `tests/data/epnp/fixture.txt` from the reference build's OpenCV 4.13.
#[must_use]
pub fn solve_pnp_epnp(
    object_points: &[nalgebra::Point3<f32>],
    image_points: &[nalgebra::Point2<f32>],
    camera_matrix: &Matrix3,
    distortion: &[f64],
) -> Option<RigidTransform> {
    // solvePnPGeneric: CV_Assert(npoints >= 4 && npoints == number of image points).
    let npoints = object_points.len();
    if npoints < 4 || npoints != image_points.len() {
        return None;
    }
    // cvUndistortPointsInternal: CV_Assert on the number of coefficients.
    if !distortion.is_empty() && !matches!(distortion.len(), 4 | 5 | 8 | 12 | 14) {
        return None;
    }

    // undistortPoints(ipoints, undistortedPoints, cameraMatrix, distCoeffs): CV_32FC2 in,
    // CV_32FC2 out, computed in double.
    let pixels: Vec<Point2> = image_points
        .iter()
        .map(|p| Point2::new(f64::from(p.x), f64::from(p.y)))
        .collect();
    let normalised =
        camera::undistort_points(&pixels, camera_matrix, distortion, TermCriteria::default());
    #[allow(clippy::cast_possible_truncation)]
    let normalised: Vec<[f64; 2]> = normalised
        .iter()
        .map(|p| [f64::from(p.x as f32), f64::from(p.y as f32)])
        .collect();
    let opoints: Vec<[f64; 3]> = object_points
        .iter()
        .map(|p| [f64::from(p.x), f64::from(p.y), f64::from(p.z)])
        .collect();

    // epnp PnP(cameraMatrix, opoints, undistortedPoints); PnP.compute_pose(R, tvec);
    let mut pnp = Epnp::new(camera_matrix, &opoints, &normalised);
    Some(finish(pnp.compute_pose()))
}

/// [`solve_pnp_epnp`] with the points in double precision throughout: what `cv::solvePnP`
/// does with `std::vector<cv::Point3d>` and `std::vector<cv::Point2d>` (the normalised
/// points keep the input's depth). Used where the rigid-body fallback no longer rounds its
/// inputs to single precision (parity ledger A23 fixed). Same failures as
/// [`solve_pnp_epnp`].
#[must_use]
pub fn solve_pnp_epnp_f64(
    object_points: &[crate::Point3],
    image_points: &[Point2],
    camera_matrix: &Matrix3,
    distortion: &[f64],
) -> Option<RigidTransform> {
    let npoints = object_points.len();
    if npoints < 4 || npoints != image_points.len() {
        return None;
    }
    if !distortion.is_empty() && !matches!(distortion.len(), 4 | 5 | 8 | 12 | 14) {
        return None;
    }
    let normalised: Vec<[f64; 2]> = camera::undistort_points(
        image_points,
        camera_matrix,
        distortion,
        TermCriteria::default(),
    )
    .iter()
    .map(|p| [p.x, p.y])
    .collect();
    let opoints: Vec<[f64; 3]> = object_points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let mut pnp = Epnp::new(camera_matrix, &opoints, &normalised);
    Some(finish(pnp.compute_pose()))
}

/// `Rodrigues(R, rvec)` of EPnP's result; the copies into the caller's `CV_64F` outputs
/// change nothing.
fn finish((r, t): ([[f64; 3]; 3], [f64; 3])) -> RigidTransform {
    let rmat = Matrix3::new(
        r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2],
    );
    RigidTransform {
        rotation: camera::matrix_to_rodrigues(&rmat),
        translation: Vector3::new(t[0], t[1], t[2]),
    }
}

/// `cvSVD(a, w, ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T)` on a square row-major `n x n`
/// matrix: the singular values and `U^T` row-major (row `i` is the `i`-th left singular
/// vector).
///
/// Ported from OpenCV 4.13 `modules/core/src/lapack.cpp`, `cvSVD`: `SVD::compute` (here
/// [`linalg::svd`]; square, so neither `FULL_UV` nor `NO_UV`), then
/// `cv::transpose(svd.u, u)` for `CV_SVD_U_T`.
fn svd_u_t(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = DMatrix::from_row_slice(n, n, a);
    let s = linalg::svd(&m, SvdMode::Thin);
    let w = s.w.iter().copied().collect();
    let mut ut = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            ut[i * n + j] = s.u[(j, i)];
        }
    }
    (w, ut)
}

/// `cvSolve(L, Rho, B, CV_SVD)` for a row-major `6 x cols` `L`: `cv::solve(.., DECOMP_SVD)`.
///
/// Ported from OpenCV 4.13 `modules/core/src/lapack.cpp`, `cvSolve` (`CV_SVD` maps to
/// `DECOMP_SVD`), through [`linalg::solve_svd`].
fn solve_svd_6(l: &[f64], cols: usize, rho: &[f64; 6]) -> Vec<f64> {
    let a = DMatrix::from_row_slice(6, cols, l);
    let b = DMatrix::from_row_slice(6, 1, rho);
    linalg::solve_svd(&a, &b).iter().copied().collect()
}

/// `epnp::dot`.
fn dot(v1: &[f64], v2: &[f64]) -> f64 {
    v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
}

/// `epnp::dist2`.
fn dist2(p1: &[f64], p2: &[f64]) -> f64 {
    (p1[0] - p2[0]) * (p1[0] - p2[0])
        + (p1[1] - p2[1]) * (p1[1] - p2[1])
        + (p1[2] - p2[2]) * (p1[2] - p2[2])
}

/// The state of OpenCV's `epnp` class (`modules/calib3d/src/epnp.h`).
struct Epnp {
    uc: f64,
    vc: f64,
    fu: f64,
    fv: f64,
    pws: Vec<f64>,
    us: Vec<f64>,
    alphas: Vec<f64>,
    pcs: Vec<f64>,
    number_of_correspondences: usize,
    cws: [[f64; 3]; 4],
    ccs: [[f64; 3]; 4],
    /// `A1`, `A2` of `qr_solve`: heap buffers that live as long as the object.
    a1: Vec<f64>,
    a2: Vec<f64>,
}

impl Epnp {
    /// Ported from `epnp.cpp`, `epnp::epnp`, with `init_camera_parameters<double>` and
    /// `init_points<Point3f, Point2f>` (`epnp.h`): `us = x * fu + uc` in double from the
    /// normalised point (single precision widened, or double for [`solve_pnp_epnp_f64`]).
    fn new(camera_matrix: &Matrix3, opoints: &[[f64; 3]], ipoints: &[[f64; 2]]) -> Self {
        let uc = camera_matrix[(0, 2)];
        let vc = camera_matrix[(1, 2)];
        let fu = camera_matrix[(0, 0)];
        let fv = camera_matrix[(1, 1)];
        let n = opoints.len();
        let mut pws = vec![0.0f64; 3 * n];
        let mut us = vec![0.0f64; 2 * n];
        for i in 0..n {
            pws[3 * i] = opoints[i][0];
            pws[3 * i + 1] = opoints[i][1];
            pws[3 * i + 2] = opoints[i][2];
            us[2 * i] = ipoints[i][0] * fu + uc;
            us[2 * i + 1] = ipoints[i][1] * fv + vc;
        }
        Epnp {
            uc,
            vc,
            fu,
            fv,
            pws,
            us,
            alphas: vec![0.0; 4 * n],
            pcs: vec![0.0; 3 * n],
            number_of_correspondences: n,
            cws: [[0.0; 3]; 4],
            ccs: [[0.0; 3]; 4],
            a1: Vec::new(),
            a2: Vec::new(),
        }
    }

    /// `number_of_correspondences` as the `int` the C++ divides by.
    #[allow(clippy::cast_precision_loss)]
    fn n_f64(&self) -> f64 {
        self.number_of_correspondences as f64
    }

    /// Ported from `epnp.cpp`, `epnp::choose_control_points`: the centroid, then the PCA
    /// axes from the SVD of `PW0^T PW0`.
    fn choose_control_points(&mut self) {
        let n = self.number_of_correspondences;
        self.cws[0] = [0.0; 3];
        for i in 0..n {
            for j in 0..3 {
                self.cws[0][j] += self.pws[3 * i + j];
            }
        }
        let nf = self.n_f64();
        for j in 0..3 {
            self.cws[0][j] /= nf;
        }

        let mut pw0 = vec![0.0f64; n * 3];
        for i in 0..n {
            for j in 0..3 {
                pw0[3 * i + j] = self.pws[3 * i + j] - self.cws[0][j];
            }
        }
        let pw0tpw0 = cvcore::mul_transposed_ata(&pw0, n, 3);
        let (dc, uct) = svd_u_t(&pw0tpw0, 3);

        for i in 1..4 {
            let k = (dc[i - 1] / nf).sqrt();
            for j in 0..3 {
                self.cws[i][j] = self.cws[0][j] + k * uct[3 * (i - 1) + j];
            }
        }
    }

    /// Ported from `epnp.cpp`, `epnp::compute_barycentric_coordinates`
    /// (`cvInvert(.., CV_SVD)`, the SVD pseudo-inverse).
    fn compute_barycentric_coordinates(&mut self) {
        let mut cc = [0.0f64; 9];
        for i in 0..3 {
            for j in 1..4 {
                cc[3 * i + j - 1] = self.cws[j][i] - self.cws[0][i];
            }
        }
        let (inv, _) = linalg::invert_svd(&Matrix3::from_row_slice(&cc));
        let mut ci = [0.0f64; 9];
        for r in 0..3 {
            for c in 0..3 {
                ci[3 * r + c] = inv[(r, c)];
            }
        }
        for i in 0..self.number_of_correspondences {
            let pi = &self.pws[3 * i..3 * i + 3];
            let a = &mut self.alphas[4 * i..4 * i + 4];
            for j in 0..3 {
                a[1 + j] = ci[3 * j] * (pi[0] - self.cws[0][0])
                    + ci[3 * j + 1] * (pi[1] - self.cws[0][1])
                    + ci[3 * j + 2] * (pi[2] - self.cws[0][2]);
            }
            a[0] = 1.0 - a[1] - a[2] - a[3];
        }
    }

    /// Ported from `epnp.cpp`, `epnp::fill_M`: rows `row` and `row + 1` of the row-major
    /// `2n x 12` matrix `m`.
    fn fill_m(&self, m: &mut [f64], row: usize, alphas: &[f64], u: f64, v: f64) {
        let (m1, rest) = m[row * 12..].split_at_mut(12);
        let m2 = &mut rest[..12];
        for i in 0..4 {
            m1[3 * i] = alphas[i] * self.fu;
            m1[3 * i + 1] = 0.0;
            m1[3 * i + 2] = alphas[i] * (self.uc - u);

            m2[3 * i] = 0.0;
            m2[3 * i + 1] = alphas[i] * self.fv;
            m2[3 * i + 2] = alphas[i] * (self.vc - v);
        }
    }

    /// Ported from `epnp.cpp`, `epnp::compute_ccs`.
    fn compute_ccs(&mut self, betas: &[f64; 4], ut: &[f64]) {
        for i in 0..4 {
            self.ccs[i] = [0.0; 3];
        }
        for i in 0..4 {
            let v = &ut[12 * (11 - i)..12 * (11 - i) + 12];
            for j in 0..4 {
                for k in 0..3 {
                    self.ccs[j][k] += betas[i] * v[3 * j + k];
                }
            }
        }
    }

    /// Ported from `epnp.cpp`, `epnp::compute_pcs`.
    fn compute_pcs(&mut self) {
        for i in 0..self.number_of_correspondences {
            let a = &self.alphas[4 * i..4 * i + 4];
            for j in 0..3 {
                self.pcs[3 * i + j] = a[0] * self.ccs[0][j]
                    + a[1] * self.ccs[1][j]
                    + a[2] * self.ccs[2][j]
                    + a[3] * self.ccs[3][j];
            }
        }
    }

    /// Ported from `epnp.cpp`, `epnp::compute_pose`: returns `R` (row-major) and `t` of the
    /// solution with the smallest mean reprojection error, the first one on ties.
    fn compute_pose(&mut self) -> ([[f64; 3]; 3], [f64; 3]) {
        self.choose_control_points();
        self.compute_barycentric_coordinates();

        let n = self.number_of_correspondences;
        let mut m = vec![0.0f64; 2 * n * 12];
        for i in 0..n {
            let alphas = self.alphas[4 * i..4 * i + 4].to_vec();
            self.fill_m(&mut m, 2 * i, &alphas, self.us[2 * i], self.us[2 * i + 1]);
        }

        let mtm = cvcore::mul_transposed_ata(&m, 2 * n, 12);
        let (_d, ut) = svd_u_t(&mtm, 12);

        let l_6x10 = compute_l_6x10(&ut);
        let rho = self.compute_rho();

        let mut betas = [[0.0f64; 4]; 4];
        let mut rep_errors = [0.0f64; 4];
        let mut rs = [[[0.0f64; 3]; 3]; 4];
        let mut ts = [[0.0f64; 3]; 4];

        find_betas_approx_1(&l_6x10, &rho, &mut betas[1]);
        self.gauss_newton(&l_6x10, &rho, &mut betas[1]);
        rep_errors[1] = self.compute_r_and_t(&ut, &betas[1], &mut rs[1], &mut ts[1]);

        find_betas_approx_2(&l_6x10, &rho, &mut betas[2]);
        self.gauss_newton(&l_6x10, &rho, &mut betas[2]);
        rep_errors[2] = self.compute_r_and_t(&ut, &betas[2], &mut rs[2], &mut ts[2]);

        find_betas_approx_3(&l_6x10, &rho, &mut betas[3]);
        self.gauss_newton(&l_6x10, &rho, &mut betas[3]);
        rep_errors[3] = self.compute_r_and_t(&ut, &betas[3], &mut rs[3], &mut ts[3]);

        let mut best = 1;
        if rep_errors[2] < rep_errors[1] {
            best = 2;
        }
        if rep_errors[3] < rep_errors[best] {
            best = 3;
        }
        (rs[best], ts[best])
    }

    /// Ported from `epnp.cpp`, `epnp::estimate_R_and_t`: Procrustes alignment of the camera
    /// and world points by the SVD of `A B^T` (`cvSVD(.., CV_SVD_MODIFY_A)`, so `u` is `U`
    /// and `v` is `V`), `R = U V^T`, the third row negated when the determinant is negative.
    fn estimate_r_and_t(&self, r: &mut [[f64; 3]; 3], t: &mut [f64; 3]) {
        let n = self.number_of_correspondences;
        let mut pc0 = [0.0f64; 3];
        let mut pw0 = [0.0f64; 3];
        for i in 0..n {
            let pc = &self.pcs[3 * i..3 * i + 3];
            let pw = &self.pws[3 * i..3 * i + 3];
            for j in 0..3 {
                pc0[j] += pc[j];
                pw0[j] += pw[j];
            }
        }
        let nf = self.n_f64();
        for j in 0..3 {
            pc0[j] /= nf;
            pw0[j] /= nf;
        }

        let mut abt = [0.0f64; 9];
        for i in 0..n {
            let pc = &self.pcs[3 * i..3 * i + 3];
            let pw = &self.pws[3 * i..3 * i + 3];
            for j in 0..3 {
                abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
                abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
                abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
            }
        }

        let s = linalg::svd(&Matrix3::from_row_slice(&abt), SvdMode::Thin);
        // abt_u = U (row-major), abt_v = transpose(vt) = V (row-major).
        let mut abt_u = [0.0f64; 9];
        let mut abt_v = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                abt_u[3 * i + j] = s.u[(i, j)];
                abt_v[3 * i + j] = s.vt[(j, i)];
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = dot(&abt_u[3 * i..], &abt_v[3 * j..]);
            }
        }

        let det =
            r[0][0] * r[1][1] * r[2][2] + r[0][1] * r[1][2] * r[2][0] + r[0][2] * r[1][0] * r[2][1]
                - r[0][2] * r[1][1] * r[2][0]
                - r[0][1] * r[1][0] * r[2][2]
                - r[0][0] * r[1][2] * r[2][1];

        if det < 0.0 {
            r[2][0] = -r[2][0];
            r[2][1] = -r[2][1];
            r[2][2] = -r[2][2];
        }

        t[0] = pc0[0] - dot(&r[0], &pw0);
        t[1] = pc0[1] - dot(&r[1], &pw0);
        t[2] = pc0[2] - dot(&r[2], &pw0);
    }

    /// Ported from `epnp.cpp`, `epnp::solve_for_sign`: flips everything when the first
    /// point's depth is negative.
    fn solve_for_sign(&mut self) {
        if self.pcs[2] < 0.0 {
            for i in 0..4 {
                for j in 0..3 {
                    self.ccs[i][j] = -self.ccs[i][j];
                }
            }
            for v in &mut self.pcs {
                *v = -*v;
            }
        }
    }

    /// Ported from `epnp.cpp`, `epnp::compute_R_and_t`.
    fn compute_r_and_t(
        &mut self,
        ut: &[f64],
        betas: &[f64; 4],
        r: &mut [[f64; 3]; 3],
        t: &mut [f64; 3],
    ) -> f64 {
        self.compute_ccs(betas, ut);
        self.compute_pcs();
        self.solve_for_sign();
        self.estimate_r_and_t(r, t);
        self.reprojection_error(r, t)
    }

    /// Ported from `epnp.cpp`, `epnp::reprojection_error`: the mean (not RMS) pixel
    /// distance.
    fn reprojection_error(&self, r: &[[f64; 3]; 3], t: &[f64; 3]) -> f64 {
        let mut sum2 = 0.0f64;
        for i in 0..self.number_of_correspondences {
            let pw = &self.pws[3 * i..3 * i + 3];
            let xc = dot(&r[0], pw) + t[0];
            let yc = dot(&r[1], pw) + t[1];
            let inv_zc = 1.0 / (dot(&r[2], pw) + t[2]);
            let ue = self.uc + self.fu * xc * inv_zc;
            let ve = self.vc + self.fv * yc * inv_zc;
            let u = self.us[2 * i];
            let v = self.us[2 * i + 1];
            sum2 += ((u - ue) * (u - ue) + (v - ve) * (v - ve)).sqrt();
        }
        sum2 / self.n_f64()
    }

    /// Ported from `epnp.cpp`, `epnp::compute_rho`.
    fn compute_rho(&self) -> [f64; 6] {
        let c = &self.cws;
        [
            dist2(&c[0], &c[1]),
            dist2(&c[0], &c[2]),
            dist2(&c[0], &c[3]),
            dist2(&c[1], &c[2]),
            dist2(&c[1], &c[3]),
            dist2(&c[2], &c[3]),
        ]
    }

    /// Ported from `epnp.cpp`, `epnp::gauss_newton`: five iterations. `x` lives across the
    /// iterations, so when [`Epnp::qr_solve`] returns early on a zero column the previous
    /// step is added again, as in OpenCV.
    fn gauss_newton(&mut self, l_6x10: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
        const ITERATIONS_NUMBER: usize = 5;
        let mut a = [0.0f64; 24];
        let mut b = [0.0f64; 6];
        let mut x = [0.0f64; 4];
        for _ in 0..ITERATIONS_NUMBER {
            compute_a_and_b_gauss_newton(l_6x10, rho, betas, &mut a, &mut b);
            self.qr_solve(&mut a, 6, 4, &mut b, &mut x);
            for i in 0..4 {
                betas[i] += x[i];
            }
        }
    }

    /// Ported from `epnp.cpp`, `epnp::qr_solve`: Householder QR of the row-major
    /// `nr x nc` matrix `a` in place, `b <- Q^T b`, then back substitution into `x`.
    ///
    /// Copied as it is, including two quirks:
    /// - the column maximum `eta` reads row `k` twice and never reads the last row
    ///   (`epnp.cpp:548-554`: the pointer is advanced after the read);
    /// - when `eta == 0` it returns without touching `b` or `x`.
    fn qr_solve(&mut self, a: &mut [f64], nr: usize, nc: usize, b: &mut [f64], x: &mut [f64]) {
        if nc == 0 || nr == 0 {
            return;
        }
        // max_nr bookkeeping: buffers of nr doubles, kept between calls.
        if self.a1.len() < nr {
            self.a1 = vec![0.0; nr];
            self.a2 = vec![0.0; nr];
        }
        let a1 = &mut self.a1;
        let a2 = &mut self.a2;

        let mut kk = 0usize; // ppAkk
        for k in 0..nc {
            let mut ik1 = kk; // ppAik1
            let mut eta = a[ik1].abs();
            for _i in k + 1..nr {
                let elt = a[ik1].abs();
                if eta < elt {
                    eta = elt;
                }
                ik1 += nc;
            }
            if eta == 0.0 {
                a1[k] = 0.0;
                a2[k] = 0.0;
                return;
            }
            let mut ik2 = kk;
            let mut sum2 = 0.0f64;
            let inv_eta = 1.0 / eta;
            for _i in k..nr {
                a[ik2] *= inv_eta;
                sum2 += a[ik2] * a[ik2];
                ik2 += nc;
            }
            let mut sigma = sum2.sqrt();
            if a[kk] < 0.0 {
                sigma = -sigma;
            }
            a[kk] += sigma;
            a1[k] = sigma * a[kk];
            a2[k] = -eta * sigma;
            for j in k + 1..nc {
                let mut ik = kk;
                let mut sum = 0.0f64;
                for _i in k..nr {
                    sum += a[ik] * a[ik + j - k];
                    ik += nc;
                }
                let tau = sum / a1[k];
                ik = kk;
                for _i in k..nr {
                    a[ik + j - k] -= tau * a[ik];
                    ik += nc;
                }
            }
            kk += nc + 1;
        }

        // b <- Qt b
        let mut jj = 0usize;
        for j in 0..nc {
            let mut ij = jj;
            let mut tau = 0.0f64;
            for i in j..nr {
                tau += a[ij] * b[i];
                ij += nc;
            }
            tau /= a1[j];
            ij = jj;
            for i in j..nr {
                b[i] -= tau * a[ij];
                ij += nc;
            }
            jj += nc + 1;
        }

        // X = R-1 b
        x[nc - 1] = b[nc - 1] / a2[nc - 1];
        for i in (0..nc - 1).rev() {
            // ppAij starts at A(i, i + 1) and walks along row i.
            let mut sum = 0.0f64;
            for j in i + 1..nc {
                sum += a[i * nc + j] * x[j];
            }
            x[i] = (b[i] - sum) / a2[i];
        }
    }
}

/// Ported from `epnp.cpp`, `epnp::compute_L_6x10`: from the last four rows of `U^T`
/// (the four smallest singular vectors of `M^T M`).
fn compute_l_6x10(ut: &[f64]) -> [f64; 60] {
    let v: [&[f64]; 4] = [
        &ut[12 * 11..12 * 12],
        &ut[12 * 10..12 * 11],
        &ut[12 * 9..12 * 10],
        &ut[12 * 8..12 * 9],
    ];
    let mut dv = [[[0.0f64; 3]; 6]; 4];
    for i in 0..4 {
        let (mut a, mut b) = (0usize, 1usize);
        for j in 0..6 {
            dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];
            b += 1;
            if b > 3 {
                a += 1;
                b = a + 1;
            }
        }
    }
    let mut l = [0.0f64; 60];
    for i in 0..6 {
        let row = &mut l[10 * i..10 * i + 10];
        row[0] = dot(&dv[0][i], &dv[0][i]);
        row[1] = 2.0 * dot(&dv[0][i], &dv[1][i]);
        row[2] = dot(&dv[1][i], &dv[1][i]);
        row[3] = 2.0 * dot(&dv[0][i], &dv[2][i]);
        row[4] = 2.0 * dot(&dv[1][i], &dv[2][i]);
        row[5] = dot(&dv[2][i], &dv[2][i]);
        row[6] = 2.0 * dot(&dv[0][i], &dv[3][i]);
        row[7] = 2.0 * dot(&dv[1][i], &dv[3][i]);
        row[8] = 2.0 * dot(&dv[2][i], &dv[3][i]);
        row[9] = dot(&dv[3][i], &dv[3][i]);
    }
    l
}

/// Columns `cols` of `L_6x10`, row-major `6 x cols.len()`.
fn l_columns(l_6x10: &[f64; 60], cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(6 * cols.len());
    for i in 0..6 {
        for &c in cols {
            out.push(l_6x10[10 * i + c]);
        }
    }
    out
}

/// Ported from `epnp.cpp`, `epnp::find_betas_approx_1`
/// (`betas_approx_1 = [B11 B12 B13 B14]`).
fn find_betas_approx_1(l_6x10: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
    let l_6x4 = l_columns(l_6x10, &[0, 1, 3, 6]);
    let b4 = solve_svd_6(&l_6x4, 4, rho);
    if b4[0] < 0.0 {
        betas[0] = (-b4[0]).sqrt();
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = b4[0].sqrt();
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

/// Ported from `epnp.cpp`, `epnp::find_betas_approx_2` (`betas_approx_2 = [B11 B12 B22]`).
fn find_betas_approx_2(l_6x10: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
    let l_6x3 = l_columns(l_6x10, &[0, 1, 2]);
    let b3 = solve_svd_6(&l_6x3, 3, rho);
    if b3[0] < 0.0 {
        betas[0] = (-b3[0]).sqrt();
        betas[1] = if b3[2] < 0.0 { (-b3[2]).sqrt() } else { 0.0 };
    } else {
        betas[0] = b3[0].sqrt();
        betas[1] = if b3[2] > 0.0 { b3[2].sqrt() } else { 0.0 };
    }
    if b3[1] < 0.0 {
        betas[0] = -betas[0];
    }
    betas[2] = 0.0;
    betas[3] = 0.0;
}

/// Ported from `epnp.cpp`, `epnp::find_betas_approx_3`
/// (`betas_approx_3 = [B11 B12 B22 B13 B23]`).
fn find_betas_approx_3(l_6x10: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
    let l_6x5 = l_columns(l_6x10, &[0, 1, 2, 3, 4]);
    let b5 = solve_svd_6(&l_6x5, 5, rho);
    if b5[0] < 0.0 {
        betas[0] = (-b5[0]).sqrt();
        betas[1] = if b5[2] < 0.0 { (-b5[2]).sqrt() } else { 0.0 };
    } else {
        betas[0] = b5[0].sqrt();
        betas[1] = if b5[2] > 0.0 { b5[2].sqrt() } else { 0.0 };
    }
    if b5[1] < 0.0 {
        betas[0] = -betas[0];
    }
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}

/// Ported from `epnp.cpp`, `epnp::compute_A_and_b_gauss_newton`: row-major `6 x 4` `a`
/// and `b`.
fn compute_a_and_b_gauss_newton(
    l_6x10: &[f64; 60],
    rho: &[f64; 6],
    betas: &[f64; 4],
    a: &mut [f64; 24],
    b: &mut [f64; 6],
) {
    for i in 0..6 {
        let l = &l_6x10[i * 10..i * 10 + 10];
        let row = &mut a[i * 4..i * 4 + 4];
        row[0] = 2.0 * l[0] * betas[0] + l[1] * betas[1] + l[3] * betas[2] + l[6] * betas[3];
        row[1] = l[1] * betas[0] + 2.0 * l[2] * betas[1] + l[4] * betas[2] + l[7] * betas[3];
        row[2] = l[3] * betas[0] + l[4] * betas[1] + 2.0 * l[5] * betas[2] + l[8] * betas[3];
        row[3] = l[6] * betas[0] + l[7] * betas[1] + l[8] * betas[2] + 2.0 * l[9] * betas[3];

        b[i] = rho[i]
            - (l[0] * betas[0] * betas[0]
                + l[1] * betas[0] * betas[1]
                + l[2] * betas[1] * betas[1]
                + l[3] * betas[0] * betas[2]
                + l[4] * betas[1] * betas[2]
                + l[5] * betas[2] * betas[2]
                + l[6] * betas[0] * betas[3]
                + l[7] * betas[1] * betas[3]
                + l[8] * betas[2] * betas[3]
                + l[9] * betas[3] * betas[3]);
    }
}
