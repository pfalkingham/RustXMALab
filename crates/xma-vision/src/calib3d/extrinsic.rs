//! `cv::findExtrinsicCameraParams2`: the pose of one view from its object and image points
//! with a known camera matrix, as `cv::calibrateCamera` initialises each view.
//!
//! Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`,
//! `cv::findExtrinsicCameraParams2` (no extrinsic guess): normalise the image points with
//! `cv::undistortPoints` (default criteria, `P` the identity), then either the planar
//! initialisation (a homography, [`super::homography`]) or the DLT on `L^T L`, then 20
//! iterations at most of `CvLevMarq` on the reprojection error.

#![allow(clippy::neg_multiply)]

use nalgebra::{DMatrix, Matrix3};

use super::cvcore;
use super::levmarq::LevMarq;
use super::project::{JacOut, project_points_jac};
use crate::Point2;
use crate::camera::{TermCriteria, matrix_to_rodrigues, rodrigues_to_matrix, undistort_points};
use crate::linalg::{self, SvdMode};

/// `cv::undistortPoints(src, dst, A, dist, Mat(), I)` of `CV_64F` points: the normalised
/// points, then the identity `P` applied literally (`xx = RR(0,0) x + RR(0,1) y + RR(0,2)`,
/// `ww = 1 / (RR(2,0) x + RR(2,1) y + RR(2,2))`, `x = xx ww`).
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/undistort.dispatch.cpp`,
/// `cvUndistortPointsInternal` (the `matR || matP` step after [`undistort_points`]).
fn undistort_identity_p(points: &[[f64; 2]], a: &[f64; 9], dist: &[f64]) -> Vec<[f64; 2]> {
    let pts: Vec<Point2> = points.iter().map(|p| Point2::new(p[0], p[1])).collect();
    let k = Matrix3::from_row_slice(a);
    let n = undistort_points(&pts, &k, dist, TermCriteria::default());
    // RR = PP * RR with both the identity: cvMatMul of 3x3 identities is the identity.
    let rr = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    n.iter()
        .map(|p| {
            let (x, y) = (p.x, p.y);
            let xx = rr[0][0] * x + rr[0][1] * y + rr[0][2];
            let yy = rr[1][0] * x + rr[1][1] * y + rr[1][2];
            let ww = 1.0 / (rr[2][0] * x + rr[2][1] * y + rr[2][2]);
            [xx * ww, yy * ww]
        })
        .collect()
}

/// OpenCV's `MAX(a, b)` macro: `a < b ? b : a`.
fn cmax(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

fn row9(m: &DMatrix<f64>) -> [f64; 9] {
    let mut o = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] = m[(i, j)];
        }
    }
    o
}

/// `cv::findExtrinsicCameraParams2(objectPoints, imagePoints, A, distCoeffs, rvec, tvec,
/// false)` for `CV_64F` points: returns `(rvec, tvec)`, or OpenCV's error text where it
/// throws.
///
/// `a` is the camera matrix row-major; `dist` the coefficients exactly as passed (4, 5, 8,
/// 12 or 14 values; `calibrateCamera` passes 5 or 14).
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`,
/// `cv::findExtrinsicCameraParams2`.
///
/// # Errors
/// OpenCV's exception text: fewer than 4 points; a non-planar object with fewer than 6;
/// a degenerate DLT.
#[allow(clippy::too_many_lines, clippy::similar_names)]
pub fn find_extrinsic_camera_params2(
    obj: &[[f64; 3]],
    img: &[[f64; 2]],
    a: &[f64; 9],
    dist: &[f64],
) -> Result<([f64; 3], [f64; 3]), String> {
    let count = obj.len();
    if count < 4 {
        return Err("(count >= 4) || (count == 3 && useExtrinsicGuess)".to_string());
    }
    let mn = undistort_identity_p(img, a, dist);
    let mut param: [f64; 6];

    let mc = cvcore::mean3(obj);
    let flat: Vec<f64> = obj.iter().flat_map(|p| p.iter().copied()).collect();
    let mm = cvcore::mul_transposed_ata_delta(&flat, count, 3, &mc);
    let svd = linalg::svd(&DMatrix::from_row_slice(3, 3, &mm), SvdMode::Thin);
    let w = [svd.w[0], svd.w[1], svd.w[2]];
    let v = row9(&svd.vt);

    if w[2] / w[1] < 1e-3 {
        // A planar structure.
        let mut r_transform = v;
        if v[2] * v[2] + v[5] * v[5] < 1e-10 {
            r_transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        }
        let rt_m = Matrix3::from_row_slice(&r_transform);
        if linalg::determinant(&rt_m) < 0.0 {
            for x in &mut r_transform {
                *x = cvcore::scale(*x, -1.0);
            }
        }
        // gemm(R_transform, Mc, -1, Mat(), 0, T, GEMM_2_T): 3x3 by (1x3)^T. gemmImpl takes
        // GEMMSingleMul (flags != 0), A * Bt branch, n = 3 < 4: s0 = sum, times alpha.
        let mut tt = [0.0f64; 3];
        for i in 0..3 {
            let mut s0 = 0.0f64;
            for k in 0..3 {
                s0 += r_transform[i * 3 + k] * mc[k];
            }
            tt[i] = (s0 + 0.0 + 0.0 + 0.0) * -1.0;
        }
        let mxy: Vec<[f64; 2]> = obj
            .iter()
            .map(|s| {
                let rp = &r_transform;
                [
                    rp[0] * s[0] + rp[1] * s[1] + rp[2] * s[2] + tt[0],
                    rp[3] * s[0] + rp[4] * s[1] + rp[5] * s[2] + tt[1],
                ]
            })
            .collect();
        let h = super::homography::find_homography(&mxy, &mn);
        let mat_r;
        let mut t;
        if let Some(mut h) = h.filter(|h| h.iter().all(|x| x.is_finite())) {
            let h1_norm = (h[0] * h[0] + h[3] * h[3] + h[6] * h[6]).sqrt();
            let h2_norm = (h[1] * h[1] + h[4] * h[4] + h[7] * h[7]).sqrt();
            let s1 = 1.0 / cmax(h1_norm, f64::EPSILON);
            let s2 = 1.0 / cmax(h2_norm, f64::EPSILON);
            for r in 0..3 {
                h[r * 3] = cvcore::scale(h[r * 3], s1);
                h[r * 3 + 1] = cvcore::scale(h[r * 3 + 1], s2);
            }
            let s3 = 2.0 / cmax(h1_norm + h2_norm, f64::EPSILON);
            t = [
                cvcore::scale(h[2], s3),
                cvcore::scale(h[5], s3),
                cvcore::scale(h[8], s3),
            ];
            // _h1.cross(_h2) into _h3 (Mat::cross for 3x1 double: a1 b2 - a2 b1, ...).
            let (a0, a1, a2) = (h[0], h[3], h[6]);
            let (b0, b1, b2) = (h[1], h[4], h[7]);
            h[2] = a1 * b2 - a2 * b1;
            h[5] = a2 * b0 - a0 * b2;
            h[8] = a0 * b1 - a1 * b0;
            let r = matrix_to_rodrigues(&Matrix3::from_row_slice(&h));
            let hr = rodrigues_to_matrix(&r);
            let hr9 = [
                hr[(0, 0)],
                hr[(0, 1)],
                hr[(0, 2)],
                hr[(1, 0)],
                hr[(1, 1)],
                hr[(1, 2)],
                hr[(2, 0)],
                hr[(2, 1)],
                hr[(2, 2)],
            ];
            // _t += matH*T_transform: gemm 3x3 by 3x1 (len 3 == d_size.height): column
            // branch t_i = a(i,0) b0 + a(i,1) b1 + a(i,2) b2, stored t*1 + 0*0; then added.
            for i in 0..3 {
                let p = hr9[i * 3] * tt[0] + hr9[i * 3 + 1] * tt[1] + hr9[i * 3 + 2] * tt[2];
                t[i] += p * 1.0 + 0.0 * 0.0;
            }
            mat_r = cvcore::gemm3(&hr9, &r_transform);
        } else {
            mat_r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            t = [0.0; 3];
        }
        let r = matrix_to_rodrigues(&Matrix3::from_row_slice(&mat_r));
        param = [r[0], r[1], r[2], t[0], t[1], t[2]];
    } else {
        // Non-planar: DLT.
        if count < 6 {
            return Err(format!(
                "> DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences. (expected: 'count >= 6'), where\n>     'count' is {count}\n> must be greater than or equal to\n>     '6' is 6\n"
            ));
        }
        let mut l = vec![0.0f64; 2 * count * 12];
        for i in 0..count {
            let row = &mut l[i * 24..i * 24 + 24];
            let x = -mn[i][0];
            let y = -mn[i][1];
            let m = obj[i];
            row[0] = m[0];
            row[16] = m[0];
            row[1] = m[1];
            row[17] = m[1];
            row[2] = m[2];
            row[18] = m[2];
            row[3] = 1.0;
            row[19] = 1.0;
            for q in [4, 5, 6, 7, 12, 13, 14, 15] {
                row[q] = 0.0;
            }
            row[8] = x * m[0];
            row[9] = x * m[1];
            row[10] = x * m[2];
            row[11] = x;
            row[20] = y * m[0];
            row[21] = y * m[1];
            row[22] = y * m[2];
            row[23] = y;
        }
        let ll = cvcore::mul_transposed_ata(&l, 2 * count, 12);
        let lsvd = linalg::svd(&DMatrix::from_row_slice(12, 12, &ll), SvdMode::Thin);
        let mut rrt = [0.0f64; 12];
        for (c, x) in rrt.iter_mut().enumerate() {
            *x = lsvd.vt[(11, c)];
        }
        let rr_of = |rrt: &[f64; 12]| -> [f64; 9] {
            [
                rrt[0], rrt[1], rrt[2], rrt[4], rrt[5], rrt[6], rrt[8], rrt[9], rrt[10],
            ]
        };
        if linalg::determinant(&Matrix3::from_row_slice(&rr_of(&rrt))) < 0.0 {
            for x in &mut rrt {
                *x = cvcore::scale(*x, -1.0);
            }
        }
        let rr = rr_of(&rrt);
        let sc = cvcore::norm_l2_rows(&[&rrt[0..3], &rrt[4..7], &rrt[8..11]]);
        if sc.abs() <= f64::EPSILON {
            return Err("fabs(sc) > DBL_EPSILON".to_string());
        }
        let s = linalg::svd(&DMatrix::from_row_slice(3, 3, &rr), SvdMode::Thin);
        let mat_r = cvcore::gemm3(&row9(&s.u), &row9(&s.vt));
        let f = cvcore::norm_l2(&mat_r) / sc;
        let r = matrix_to_rodrigues(&Matrix3::from_row_slice(&mat_r));
        param = [
            r[0],
            r[1],
            r[2],
            cvcore::scale(rrt[3], f),
            cvcore::scale(rrt[7], f),
            cvcore::scale(rrt[11], f),
        ];
    }

    // Refine with CvLevMarq(6, count*2, (EPS+ITER, 20, FLT_EPSILON), true).
    let mut solver = LevMarq::new(6, count * 2, 20, f64::from(f32::EPSILON), true);
    solver.param.copy_from_slice(&param);
    loop {
        let st = solver.update();
        param.copy_from_slice(&solver.param);
        if !st.proceed || !st.want_err {
            break;
        }
        let rv = [param[0], param[1], param[2]];
        let tv = [param[3], param[4], param[5]];
        let mut proj = vec![0.0f64; 2 * count];
        if st.want_j {
            let mut jr = vec![0.0f64; 2 * count * 3];
            let mut jt = vec![0.0f64; 2 * count * 3];
            project_points_jac(
                obj,
                &rv,
                &tv,
                a,
                dist,
                &mut proj,
                Some(JacOut {
                    buf: &mut jr,
                    stride: 3,
                    col: 0,
                }),
                Some(JacOut {
                    buf: &mut jt,
                    stride: 3,
                    col: 0,
                }),
                None,
                None,
                None,
            );
            for row in 0..2 * count {
                for c in 0..3 {
                    solver.j[row * 6 + c] = jr[row * 3 + c];
                    solver.j[row * 6 + 3 + c] = jt[row * 3 + c];
                }
            }
        } else {
            project_points_jac(
                obj, &rv, &tv, a, dist, &mut proj, None, None, None, None, None,
            );
        }
        for i in 0..count {
            solver.err[2 * i] = proj[2 * i] - img[i][0];
            solver.err[2 * i + 1] = proj[2 * i + 1] - img[i][1];
        }
    }
    param.copy_from_slice(&solver.param);
    Ok((
        [param[0], param[1], param[2]],
        [param[3], param[4], param[5]],
    ))
}
