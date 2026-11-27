//! `cv::findHomography(src, dst)` with method 0 (every point, the normalised DLT, then
//! `LMSolver` refinement), as `findExtrinsicCameraParams2` (planar objects) and
//! `initIntrinsicParams2D` call it, and `initIntrinsicParams2D` itself.
//!
//! Ported from OpenCV 4.13 `modules/calib3d/src/fundam.cpp` (`cv::findHomography`,
//! `HomographyEstimatorCallback::runKernel`, `HomographyRefineCallback::compute`),
//! `modules/calib3d/src/levmarq.cpp` (`LMSolverImpl::run`) and
//! `modules/calib3d/src/calibration.cpp` (`initIntrinsicParams2D`).
//!
//! **Departure:** `runKernel` takes the eigenvector of the smallest eigenvalue of the 9x9
//! `L^T L` from `cv::eigen`, which the reference build (OpenCV built with Eigen 5.0.1) hands
//! to Eigen's `SelfAdjointEigenSolver`. Eigen's code is not available to port, so this port
//! uses OpenCV's own Jacobi method ([`crate::linalg::eigen_jacobi`], what `cv::eigen` runs
//! in a build without Eigen). The two agree to rounding (the sign is removed by the
//! normalisation `H / H(2,2)`), so homographies, and the planar calibrations built on them,
//! agree to rounding rather than bit for bit (parity ledger A37 notes).

#![allow(clippy::neg_multiply)]

use super::cvcore;
use crate::linalg;

/// `scaleFor(double)` of `fundam.cpp`: `1/x` when `|x| > FLT_EPSILON`, else 1.
fn scale_for(x: f64) -> f64 {
    if x.abs() > f64::from(f32::EPSILON) {
        1.0 / x
    } else {
        1.0
    }
}

/// `HomographyEstimatorCallback::runKernel` on float points: the normalised DLT. `None`
/// where it returns 0 (the points do not spread in x or y).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn run_kernel(big_m: &[[f32; 2]], m: &[[f32; 2]]) -> Option<[f64; 9]> {
    let count = big_m.len();
    #[allow(clippy::cast_precision_loss)]
    let countf = count as f64;
    let (mut cm_x, mut cm_y, mut cbm_x, mut cbm_y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..count {
        cm_x += f64::from(m[i][0]);
        cm_y += f64::from(m[i][1]);
        cbm_x += f64::from(big_m[i][0]);
        cbm_y += f64::from(big_m[i][1]);
    }
    cm_x /= countf;
    cm_y /= countf;
    cbm_x /= countf;
    cbm_y /= countf;
    let (mut sm_x, mut sm_y, mut sbm_x, mut sbm_y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..count {
        sm_x += (f64::from(m[i][0]) - cm_x).abs();
        sm_y += (f64::from(m[i][1]) - cm_y).abs();
        sbm_x += (f64::from(big_m[i][0]) - cbm_x).abs();
        sbm_y += (f64::from(big_m[i][1]) - cbm_y).abs();
    }
    if sm_x.abs() < f64::EPSILON
        || sm_y.abs() < f64::EPSILON
        || sbm_x.abs() < f64::EPSILON
        || sbm_y.abs() < f64::EPSILON
    {
        return None;
    }
    sm_x = countf / sm_x;
    sm_y = countf / sm_y;
    sbm_x = countf / sbm_x;
    sbm_y = countf / sbm_y;
    let inv_hnorm = [1.0 / sm_x, 0.0, cm_x, 0.0, 1.0 / sm_y, cm_y, 0.0, 0.0, 1.0];
    let hnorm2 = [
        sbm_x,
        0.0,
        -cbm_x * sbm_x,
        0.0,
        sbm_y,
        -cbm_y * sbm_y,
        0.0,
        0.0,
        1.0,
    ];
    let mut ltl = [0.0f64; 81];
    for i in 0..count {
        let x = (f64::from(m[i][0]) - cm_x) * sm_x;
        let y = (f64::from(m[i][1]) - cm_y) * sm_y;
        let bx = (f64::from(big_m[i][0]) - cbm_x) * sbm_x;
        let by = (f64::from(big_m[i][1]) - cbm_y) * sbm_y;
        let lx = [bx, by, 1.0, 0.0, 0.0, 0.0, -x * bx, -x * by, -x];
        let ly = [0.0, 0.0, 0.0, bx, by, 1.0, -y * bx, -y * by, -y];
        for j in 0..9 {
            for k in j..9 {
                ltl[j * 9 + k] += lx[j] * lx[k] + ly[j] * ly[k];
            }
        }
    }
    cvcore::complete_symm_upper(&mut ltl, 9);
    // eigen(_LtL, matW, matV): see the module documentation (departure).
    let (_w, v) = linalg::eigen_jacobi(&ltl, 9);
    let mut h0 = [0.0f64; 9];
    h0.copy_from_slice(&v[72..81]);
    let htemp = cvcore::gemm3(&inv_hnorm, &h0);
    let h0 = cvcore::gemm3(&htemp, &hnorm2);
    let s = scale_for(h0[8]);
    let mut out = [0.0f64; 9];
    for i in 0..9 {
        out[i] = cvcore::scale(h0[i], s);
    }
    Some(out)
}

/// `HomographyRefineCallback::compute`: residuals (`2n`) and, when asked, the `2n x 9`
/// Jacobian.
fn refine_compute(
    src: &[[f32; 2]],
    dst: &[[f32; 2]],
    h: &[f64],
    jac: Option<&mut Vec<f64>>,
) -> Vec<f64> {
    let count = src.len();
    let mut err = vec![0.0f64; count * 2];
    let mut jp = jac;
    if let Some(j) = jp.as_mut() {
        j.clear();
        j.resize(count * 2 * 9, 0.0);
    }
    for i in 0..count {
        let mx = f64::from(src[i][0]);
        let my = f64::from(src[i][1]);
        let mut ww = h[6] * mx + h[7] * my + h[8];
        ww = if ww.abs() > f64::EPSILON {
            1.0 / ww
        } else {
            0.0
        };
        let xi = (h[0] * mx + h[1] * my + h[2]) * ww;
        let yi = (h[3] * mx + h[4] * my + h[5]) * ww;
        err[i * 2] = xi - f64::from(dst[i][0]);
        err[i * 2 + 1] = yi - f64::from(dst[i][1]);
        if let Some(j) = jp.as_mut() {
            let o = i * 18;
            j[o] = mx * ww;
            j[o + 1] = my * ww;
            j[o + 2] = ww;
            j[o + 6] = -mx * ww * xi;
            j[o + 7] = -my * ww * xi;
            j[o + 8] = -ww * xi;
            j[o + 12] = mx * ww;
            j[o + 13] = my * ww;
            j[o + 14] = ww;
            j[o + 15] = -mx * ww * yi;
            j[o + 16] = -my * ww * yi;
            j[o + 17] = -ww * yi;
        }
    }
    err
}

/// `Mat::dot` of two continuous `CV_64F` arrays: `dotProd_<double>` (blocks of four summed
/// left to right, then added; the rest one by one).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0usize;
    let mut r = 0.0f64;
    while i + 4 <= len {
        r += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        r += a[i] * b[i];
        i += 1;
    }
    r
}

/// `LMSolverImpl::run` with `HomographyRefineCallback` (`maxIters` 10, `eps` `FLT_EPSILON`).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn lm_refine(src: &[[f32; 2]], dst: &[[f32; 2]], h: &mut [f64; 9]) {
    let lx = 9usize;
    let max_iters = 10;
    let eps = f64::from(f32::EPSILON);
    let rows = src.len() * 2;
    let mut x = h.to_vec();
    let mut j = Vec::new();
    let mut r = refine_compute(src, dst, &x, Some(&mut j));
    let mut s = cvcore::norm_l2sqr(&r);
    let mut a = cvcore::mul_transposed_ata(&j, rows, lx);
    let mut v = cvcore::gemm_at_v(&j, lx, lx, &r, rows);
    let d_diag: Vec<f64> = (0..lx).map(|i| a[i * lx + i]).collect();
    let (rlo, rhi) = (0.25, 0.75);
    let mut lambda = 1.0f64;
    let mut lc = 0.75f64;
    let mut iter = 0;
    loop {
        let mut ap = a.clone();
        for i in 0..lx {
            ap[i * lx + i] += lambda * d_diag[i];
        }
        let d = linalg::solve_eig(&ap, lx, &v, 1);
        let xd: Vec<f64> = x.iter().zip(&d).map(|(p, q)| p - q).collect();
        let rd = refine_compute(src, dst, &xd, None);
        let sd = cvcore::norm_l2sqr(&rd);
        // gemm(A, d, -1, v, 2, temp_d): one-column result, A * Bt branch.
        let mut temp_d = vec![0.0f64; lx];
        for i in 0..lx {
            let row = &a[i * lx..(i + 1) * lx];
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut k = 0usize;
            while k + 4 <= lx {
                s0 += row[k] * d[k];
                s1 += row[k + 1] * d[k + 1];
                s2 += row[k + 2] * d[k + 2];
                s3 += row[k + 3] * d[k + 3];
                k += 4;
            }
            while k < lx {
                s0 += row[k] * d[k];
                k += 1;
            }
            s0 = (s0 + s1 + s2 + s3) * -1.0;
            temp_d[i] = s0 + v[i] * 2.0;
        }
        let ds = dot(&d, &temp_d);
        let big_r = (s - sd) / if ds.abs() > f64::EPSILON { ds } else { 1.0 };
        if big_r > rhi {
            lambda *= 0.5;
            if lambda < lc {
                lambda = 0.0;
            }
        } else if big_r < rlo {
            let t = dot(&d, &v);
            let mut nu = (sd - s) / if t.abs() > f64::EPSILON { t } else { 1.0 } + 2.0;
            // std::min(std::max(nu, 2.), 10.), NaN kept as the C++ keeps it.
            nu = if nu < 2.0 { 2.0 } else { nu };
            nu = if 10.0 < nu { 10.0 } else { nu };
            if lambda == 0.0 {
                let inv_diag = linalg::invert_eig(&a, lx);
                let mut maxval = f64::EPSILON;
                for i in 0..lx {
                    maxval = maxval.max(inv_diag[i * lx + i].abs());
                }
                lambda = 1.0 / maxval;
                lc = lambda;
                nu *= 0.5;
            }
            lambda *= nu;
        }
        if sd < s {
            s = sd;
            x = xd;
            r = refine_compute(src, dst, &x, Some(&mut j));
            a = cvcore::mul_transposed_ata(&j, rows, lx);
            v = cvcore::gemm_at_v(&j, lx, lx, &r, rows);
        }
        iter += 1;
        let dinf = d.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let rinf = r.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let proceed = iter < max_iters && dinf >= eps && rinf >= eps;
        if !proceed {
            break;
        }
    }
    h.copy_from_slice(&x);
}

/// `cv::findHomography(src, dst)` (method 0) with the points converted to `float` as OpenCV
/// does: the homography row-major, normalised to `H(2,2) = 1`, or `None` where OpenCV
/// returns an empty matrix (the points do not spread in x or y), and also where OpenCV
/// throws: the two point sets differ in length or have fewer than 4 points.
#[must_use]
pub fn find_homography(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<[f64; 9]> {
    if src.len() != dst.len() || src.len() < 4 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    let to_f32 = |p: &[[f64; 2]]| -> Vec<[f32; 2]> {
        p.iter().map(|q| [q[0] as f32, q[1] as f32]).collect()
    };
    let s = to_f32(src);
    let d = to_f32(dst);
    let mut h = run_kernel(&s, &d)?;
    if s.len() > 4 {
        lm_refine(&s, &d, &mut h);
        let sc = scale_for(h[8]);
        for x in &mut h {
            *x = cvcore::scale(*x, sc);
        }
    }
    Some(h)
}

/// `initIntrinsicParams2D(objectPoints, imagePoints, npoints, imageSize, cameraMatrix, 0)`:
/// the starting camera matrix from one homography per view (the object's `z` is ignored),
/// row-major. `Err` with OpenCV's text where it throws.
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration.cpp`, `initIntrinsicParams2D`
/// (with `solve(.., DECOMP_NORMAL + DECOMP_SVD)`: the Jacobi SVD for one view, and for more
/// the normal equations solved by OpenCV's Jacobi eigen decomposition).
///
/// # Errors
/// `matH0.size() == Size(3, 3)` when a view's homography is empty.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn init_intrinsic_params_2d(
    obj: &[[f64; 3]],
    img: &[[f64; 2]],
    npoints: &[usize],
    image_size: (usize, usize),
) -> Result<[f64; 9], String> {
    let nimages = npoints.len();
    #[allow(clippy::cast_precision_loss)]
    let cx = if image_size.0 == 0 {
        0.5
    } else {
        (image_size.0 as f64 - 1.0) * 0.5
    };
    #[allow(clippy::cast_precision_loss)]
    let cy = if image_size.1 == 0 {
        0.5
    } else {
        (image_size.1 as f64 - 1.0) * 0.5
    };
    let mut mat_a = vec![0.0f64; 2 * nimages * 2];
    let mut mat_b = vec![0.0f64; 2 * nimages];
    let mut pos = 0usize;
    for (i, &ni) in npoints.iter().enumerate() {
        // findHomography of the three-channel object points: convertPointsFromHomogeneous
        // with z = 0 (scaleFor(0) = 1) gives (x * 1, y * 1).
        let src: Vec<[f64; 2]> = obj[pos..pos + ni]
            .iter()
            .map(|p| {
                let sc = scale_for(p[2]);
                [p[0] * sc, p[1] * sc]
            })
            .collect();
        let h0 = find_homography(&src, &img[pos..pos + ni])
            .ok_or_else(|| "matH0.size() == Size(3, 3)".to_string())?;
        pos += ni;
        let mut h = h0;
        h[0] -= h[6] * cx;
        h[1] -= h[7] * cx;
        h[2] -= h[8] * cx;
        h[3] -= h[6] * cy;
        h[4] -= h[7] * cy;
        h[5] -= h[8] * cy;
        let (mut hv, mut vv, mut d1, mut d2) = ([0.0f64; 3], [0.0f64; 3], [0.0f64; 3], [0.0f64; 3]);
        let mut n = [0.0f64; 4];
        for j in 0..3 {
            let t0 = h[j * 3];
            let t1 = h[j * 3 + 1];
            hv[j] = t0;
            vv[j] = t1;
            d1[j] = (t0 + t1) * 0.5;
            d2[j] = (t0 - t1) * 0.5;
            n[0] += t0 * t0;
            n[1] += t1 * t1;
            n[2] += d1[j] * d1[j];
            n[3] += d2[j] * d2[j];
        }
        for x in &mut n {
            *x = 1.0 / x.sqrt();
        }
        for j in 0..3 {
            hv[j] *= n[0];
            vv[j] *= n[1];
            d1[j] *= n[2];
            d2[j] *= n[3];
        }
        mat_a[(i * 2) * 2] = hv[0] * vv[0];
        mat_a[(i * 2) * 2 + 1] = hv[1] * vv[1];
        mat_a[(i * 2 + 1) * 2] = d1[0] * d2[0];
        mat_a[(i * 2 + 1) * 2 + 1] = d1[1] * d2[1];
        mat_b[i * 2] = -hv[2] * vv[2];
        mat_b[i * 2 + 1] = -d1[2] * d2[2];
    }
    let f = if nimages == 1 {
        let a = nalgebra::DMatrix::from_row_slice(2, 2, &mat_a);
        let b = nalgebra::DMatrix::from_row_slice(2, 1, &mat_b);
        let x = linalg::solve_svd(&a, &b);
        [x[(0, 0)], x[(1, 0)]]
    } else {
        // DECOMP_NORMAL: a = mulTransposed(A, true), b = gemm(A, b, GEMM_1_T), DECOMP_EIG.
        let rows = 2 * nimages;
        let ata = cvcore::mul_transposed_ata(&mat_a, rows, 2);
        let atb = cvcore::gemm_at_v(&mat_a, 2, 2, &mat_b, rows);
        let x = linalg::solve_eig(&ata, 2, &atb, 1);
        [x[0], x[1]]
    };
    let fx = (1.0 / f[0]).abs().sqrt();
    let fy = (1.0 / f[1]).abs().sqrt();
    Ok([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])
}
