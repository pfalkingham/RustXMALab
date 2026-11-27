//! `cv::Rodrigues` (vector to matrix) with its Jacobian and `cv::projectPoints` with its
//! derivatives, as `cv::calibrateCamera` and `findExtrinsicCameraParams2` call them.
//!
//! Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp` (`cv::Rodrigues`,
//! `cv::projectPoints`, the per-point loop). Only the untilted model is ported: the 13th
//! and 14th coefficients (`tau_x`, `tau_y`) must be zero, as they always are on XMALab's
//! path; the tilt matrix is then the identity and its products are evaluated literally.
//! Every operation is in the source's order, unfused.

/// Rotation vector to rotation matrix (row-major) and the 3x9 Jacobian `dR/dr` (row-major,
/// row `i` is the derivative of the nine entries of `R` by `r_i`).
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`, `cv::Rodrigues`
/// (`v2m`, `CV_64F`, Jacobian requested).
#[must_use]
pub fn rodrigues_jacobian(r_in: &[f64; 3]) -> ([f64; 9], [f64; 27]) {
    let (mut rx, mut ry, mut rz) = (r_in[0], r_in[1], r_in[2]);
    let theta = (rx * rx + ry * ry + rz * rz).sqrt();
    let mut j = [0.0f64; 27];
    if theta < f64::EPSILON {
        j[5] = -1.0;
        j[15] = -1.0;
        j[19] = -1.0;
        j[7] = 1.0;
        j[11] = 1.0;
        j[21] = 1.0;
        return ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], j);
    }
    let c = theta.cos();
    let s = theta.sin();
    let c1 = 1.0 - c;
    let itheta = if theta != 0.0 { 1.0 / theta } else { 0.0 };
    rx *= itheta;
    ry *= itheta;
    rz *= itheta;
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
    let mut rm = [0.0f64; 9];
    for i in 0..9 {
        rm[i] = (eye[i] * c + rrt[i] * c1) + r_x[i] * s;
    }
    let drrt = [
        rx + rx,
        ry,
        rz,
        ry,
        0.0,
        0.0,
        rz,
        0.0,
        0.0,
        0.0,
        rx,
        0.0,
        rx,
        ry + ry,
        rz,
        0.0,
        rz,
        0.0,
        0.0,
        0.0,
        rx,
        0.0,
        0.0,
        ry,
        rx,
        ry,
        rz + rz,
    ];
    let d_r_x = [
        0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    for i in 0..3 {
        let ri = if i == 0 {
            rx
        } else if i == 1 {
            ry
        } else {
            rz
        };
        let a0 = -s * ri;
        let a1 = (s - 2.0 * c1 * itheta) * ri;
        let a2 = c1 * itheta;
        let a3 = (c - s * itheta) * ri;
        let a4 = s * itheta;
        for k in 0..9 {
            j[i * 9 + k] = a0 * eye[k]
                + a1 * rrt[k]
                + a2 * drrt[i * 9 + k]
                + a3 * r_x[k]
                + a4 * d_r_x[i * 9 + k];
        }
    }
    (rm, j)
}

/// Where `project_points_jac` writes derivatives: row-major buffers with `2 * n` rows, a
/// row stride, and the first column to write. `None` is OpenCV's `noArray()`.
pub struct JacOut<'a> {
    pub buf: &'a mut [f64],
    pub stride: usize,
    pub col: usize,
}

/// `cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints,
/// dpdr, dpdt, dpdf, dpdc, dpdk, noArray(), 0)` for `CV_64F` points.
///
/// - `a` is the camera matrix, row-major (only `fx`, `fy`, `cx`, `cy` are read).
/// - `dist` holds `ktotal` coefficients (4, 5, 8, 12 or 14); `tau_x`, `tau_y` must be 0.
/// - `out` receives `x0, y0, x1, y1, ...`.
/// - Each derivative block gets two rows per point (`x` then `y`); `dpdk` gets `ktotal`
///   columns, `dpdr`, `dpdt` three, `dpdf`, `dpdc` two.
///
/// Ported from OpenCV 4.13 `modules/calib3d/src/calibration_base.cpp`, `cv::projectPoints`
/// (the generic per-point loop with `calc_derivatives`).
///
/// # Panics
/// If `tau_x` or `tau_y` is not zero.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn project_points_jac(
    obj: &[[f64; 3]],
    rvec: &[f64; 3],
    tvec: &[f64; 3],
    a: &[f64; 9],
    dist: &[f64],
    out: &mut [f64],
    mut dpdr: Option<JacOut<'_>>,
    mut dpdt: Option<JacOut<'_>>,
    mut dpdf: Option<JacOut<'_>>,
    mut dpdc: Option<JacOut<'_>>,
    mut dpdk: Option<JacOut<'_>>,
) {
    let mut k = [0.0f64; 14];
    k[..dist.len()].copy_from_slice(dist);
    assert!(
        k[12] == 0.0 && k[13] == 0.0,
        "the tilted model is not ported"
    );
    let ktotal = dist.len();
    let (rm, d_rdr) = rodrigues_jacobian(rvec);
    let r = rm;
    let t = *tvec;
    let fx = a[0];
    let fy = a[4];
    let cx = a[2];
    let cy = a[5];
    // matTilt = eye; dMatTiltdTauX = (0,0,0,0,0,0,0,-1,0); dMatTiltdTauY = (0,0,0,0,0,0,1,0,0).
    let mat_tilt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let d_tau_x = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
    let d_tau_y = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let calc_derivatives =
        dpdr.is_some() || dpdt.is_some() || dpdf.is_some() || dpdc.is_some() || dpdk.is_some();

    // Matx33d * Vec3d (Matx_MatMulOp: s = 0; s += m(i,k)*v(k)).
    let mul3 = |m: &[[f64; 3]; 3], v: [f64; 3]| -> [f64; 3] {
        let mut o = [0.0f64; 3];
        for i in 0..3 {
            let mut s = 0.0f64;
            for kk in 0..3 {
                s += m[i][kk] * v[kk];
            }
            o[i] = s;
        }
        o
    };
    // Matx22d * Vec2d.
    let mul2 = |m: &[[f64; 2]; 2], v: [f64; 2]| -> [f64; 2] {
        let mut o = [0.0f64; 2];
        for i in 0..2 {
            let mut s = 0.0f64;
            for kk in 0..2 {
                s += m[i][kk] * v[kk];
            }
            o[i] = s;
        }
        o
    };

    for (i, m) in obj.iter().enumerate() {
        let (xw, yw, zw) = (m[0], m[1], m[2]);
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

        let vec_tilt = mul3(&mat_tilt, [xd0, yd0, 1.0]);
        let inv_proj = if vec_tilt[2] != 0.0 {
            1.0 / vec_tilt[2]
        } else {
            1.0
        };
        let xd = inv_proj * vec_tilt[0];
        let yd = inv_proj * vec_tilt[1];

        out[2 * i] = xd * fx + cx;
        out[2 * i + 1] = yd * fy + cy;

        if !calc_derivatives {
            continue;
        }
        let r0 = 2 * i;
        let r1 = 2 * i + 1;
        if let Some(o) = dpdc.as_mut() {
            o.buf[r0 * o.stride + o.col] = 1.0;
            o.buf[r0 * o.stride + o.col + 1] = 0.0;
            o.buf[r1 * o.stride + o.col] = 0.0;
            o.buf[r1 * o.stride + o.col + 1] = 1.0;
        }
        if let Some(o) = dpdf.as_mut() {
            o.buf[r0 * o.stride + o.col] = xd;
            o.buf[r0 * o.stride + o.col + 1] = 0.0;
            o.buf[r1 * o.stride + o.col] = 0.0;
            o.buf[r1 * o.stride + o.col + 1] = yd;
        }
        let mut d_mat_tilt = [[0.0f64; 2]; 2];
        for row in 0..2 {
            for col in 0..2 {
                d_mat_tilt[row][col] =
                    mat_tilt[row][col] * vec_tilt[2] - mat_tilt[2][col] * vec_tilt[row];
            }
        }
        let inv_proj_square = inv_proj * inv_proj;
        for row in d_mat_tilt.iter_mut() {
            for v in row.iter_mut() {
                *v *= inv_proj_square;
            }
        }
        if let Some(o) = dpdk.as_mut() {
            let mut put = |c: usize, d: [f64; 2]| {
                o.buf[r0 * o.stride + o.col + c] = fx * d[0];
                o.buf[r1 * o.stride + o.col + c] = fy * d[1];
            };
            put(0, mul2(&d_mat_tilt, [x * icdist2 * r2, y * icdist2 * r2]));
            put(1, mul2(&d_mat_tilt, [x * icdist2 * r4, y * icdist2 * r4]));
            if ktotal > 2 {
                put(2, mul2(&d_mat_tilt, [a1, a3]));
                put(3, mul2(&d_mat_tilt, [a2, a1]));
                if ktotal > 4 {
                    put(4, mul2(&d_mat_tilt, [x * icdist2 * r6, y * icdist2 * r6]));
                    if ktotal > 5 {
                        put(
                            5,
                            mul2(
                                &d_mat_tilt,
                                [
                                    x * cdist * (-icdist2) * icdist2 * r2,
                                    y * cdist * (-icdist2) * icdist2 * r2,
                                ],
                            ),
                        );
                        put(
                            6,
                            mul2(
                                &d_mat_tilt,
                                [
                                    x * cdist * (-icdist2) * icdist2 * r4,
                                    y * cdist * (-icdist2) * icdist2 * r4,
                                ],
                            ),
                        );
                        put(
                            7,
                            mul2(
                                &d_mat_tilt,
                                [
                                    x * cdist * (-icdist2) * icdist2 * r6,
                                    y * cdist * (-icdist2) * icdist2 * r6,
                                ],
                            ),
                        );
                        if ktotal > 8 {
                            put(8, mul2(&d_mat_tilt, [r2, 0.0]));
                            put(9, mul2(&d_mat_tilt, [r4, 0.0]));
                            put(10, mul2(&d_mat_tilt, [0.0, r2]));
                            put(11, mul2(&d_mat_tilt, [0.0, r4]));
                            if ktotal > 12 {
                                for (c, dm) in [(12usize, &d_tau_x), (13usize, &d_tau_y)] {
                                    let dv = mul3(dm, [xd0, yd0, 1.0]);
                                    o.buf[r0 * o.stride + o.col + c] = fx
                                        * inv_proj_square
                                        * (dv[0] * vec_tilt[2] - dv[2] * vec_tilt[0]);
                                    o.buf[r1 * o.stride + o.col + c] = fy
                                        * inv_proj_square
                                        * (dv[1] * vec_tilt[2] - dv[2] * vec_tilt[1]);
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(o) = dpdt.as_mut() {
            let dxdt = [z, 0.0, -x * z];
            let dydt = [0.0, z, -y * z];
            for jj in 0..3 {
                let dr2dt = 2.0 * x * dxdt[jj] + 2.0 * y * dydt[jj];
                let dcdist_dt = k[0] * dr2dt + 2.0 * k[1] * r2 * dr2dt + 3.0 * k[4] * r4 * dr2dt;
                let dicdist2_dt = -icdist2
                    * icdist2
                    * (k[5] * dr2dt + 2.0 * k[6] * r2 * dr2dt + 3.0 * k[7] * r4 * dr2dt);
                let da1dt = 2.0 * (x * dydt[jj] + y * dxdt[jj]);
                let dmxdt = dxdt[jj] * cdist * icdist2
                    + x * dcdist_dt * icdist2
                    + x * cdist * dicdist2_dt
                    + k[2] * da1dt
                    + k[3] * (dr2dt + 4.0 * x * dxdt[jj])
                    + k[8] * dr2dt
                    + 2.0 * r2 * k[9] * dr2dt;
                let dmydt = dydt[jj] * cdist * icdist2
                    + y * dcdist_dt * icdist2
                    + y * cdist * dicdist2_dt
                    + k[2] * (dr2dt + 4.0 * y * dydt[jj])
                    + k[3] * da1dt
                    + k[10] * dr2dt
                    + 2.0 * r2 * k[11] * dr2dt;
                let d = mul2(&d_mat_tilt, [dmxdt, dmydt]);
                o.buf[r0 * o.stride + o.col + jj] = fx * d[0];
                o.buf[r1 * o.stride + o.col + jj] = fy * d[1];
            }
        }
        if let Some(o) = dpdr.as_mut() {
            let d = &d_rdr;
            let dx0dr = [
                xw * d[0] + yw * d[1] + zw * d[2],
                xw * d[9] + yw * d[10] + zw * d[11],
                xw * d[18] + yw * d[19] + zw * d[20],
            ];
            let dy0dr = [
                xw * d[3] + yw * d[4] + zw * d[5],
                xw * d[12] + yw * d[13] + zw * d[14],
                xw * d[21] + yw * d[22] + zw * d[23],
            ];
            let dz0dr = [
                xw * d[6] + yw * d[7] + zw * d[8],
                xw * d[15] + yw * d[16] + zw * d[17],
                xw * d[24] + yw * d[25] + zw * d[26],
            ];
            for jj in 0..3 {
                let dxdr = z * (dx0dr[jj] - x * dz0dr[jj]);
                let dydr = z * (dy0dr[jj] - y * dz0dr[jj]);
                let dr2dr = 2.0 * x * dxdr + 2.0 * y * dydr;
                let dcdist_dr = (k[0] + 2.0 * k[1] * r2 + 3.0 * k[4] * r4) * dr2dr;
                let dicdist2_dr =
                    -icdist2 * icdist2 * (k[5] + 2.0 * k[6] * r2 + 3.0 * k[7] * r4) * dr2dr;
                let da1dr = 2.0 * (x * dydr + y * dxdr);
                let dmxdr = dxdr * cdist * icdist2
                    + x * dcdist_dr * icdist2
                    + x * cdist * dicdist2_dr
                    + k[2] * da1dr
                    + k[3] * (dr2dr + 4.0 * x * dxdr)
                    + (k[8] + 2.0 * r2 * k[9]) * dr2dr;
                let dmydr = dydr * cdist * icdist2
                    + y * dcdist_dr * icdist2
                    + y * cdist * dicdist2_dr
                    + k[2] * (dr2dr + 4.0 * y * dydr)
                    + k[3] * da1dr
                    + (k[10] + 2.0 * r2 * k[11]) * dr2dr;
                let dd = mul2(&d_mat_tilt, [dmxdr, dmydr]);
                o.buf[r0 * o.stride + o.col + jj] = fx * dd[0];
                o.buf[r1 * o.stride + o.col + jj] = fy * dd[1];
            }
        }
    }
}
