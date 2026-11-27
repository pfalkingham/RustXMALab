//! OpenCV `core` routines as `cv::calibrateCamera` reaches them, with the summation order of
//! the reference build (Fedora's OpenCV 4.13.0, baseline SSE2, run on a CPU with AVX2 and
//! AVX-512, so the dispatched kernels are the AVX2 or AVX-512 builds).
//!
//! Matrices are row-major `f64` slices with an explicit row stride.
//!
//! - `cv::gemm` with `GEMM_1_T` (`A^T B`): `gemmImpl` takes `GEMMSingleMul`. For a result
//!   wider than one column each element is the plain sequential dot product
//!   (`s = 0; s += a*b`); for a one-column result `gemmImpl` switches to the `A * Bt` branch,
//!   which sums in four interleaved partial sums. Above 99 rows of `A` the reference hands
//!   the product to BLAS (`hal_internal.cpp`, `HAL_GEMM_SMALL_MATRIX_THRESH`), which this
//!   port does not reproduce (parity ledger A37).
//! - `cv::norm` of `CV_64F` data with `NORM_L2` / `NORM_L2SQR`: the AVX2 kernel
//!   (`NormL2_SIMD<double, double>`, `NormDiffL2_SIMD<double, double>`), four 4-lane
//!   accumulators with fused multiply-adds over blocks of 16, a fixed reduction, then the
//!   remaining elements one by one.
//! - `cv::mean` of three-channel `CV_64F` data: sequential sums, times `1/n`.
//! - `cv::mulTransposed(src, dst, true, delta)`: `MulTransposedR` (sequential sums), then
//!   `completeSymm`.
//!
//! Every multiply and add outside the documented fused kernels is a separate operation.

/// `cv::gemm(A, B, 1, noArray(), 0, D, GEMM_1_T)` for `A` (`rows x a_cols`, stride
/// `a_stride`) and `B` (`rows x b_cols`, stride `b_stride`), `b_cols > 1`: `D = A^T B`,
/// `a_cols x b_cols`, row-major.
///
/// Ported from OpenCV 4.13 `modules/core/src/matmul.simd.hpp`, `gemmImpl` and
/// `GEMMSingleMul<double, double>` (the `GEMM_1_T` case with `d_size.width * 8 <= 1600`):
/// each element is `s = 0; s += A(k, i) * B(k, j)` over `k` in order, then `s * alpha`
/// with `alpha = 1`.
pub(crate) fn gemm_at_b(
    a: &[f64],
    a_stride: usize,
    a_cols: usize,
    b: &[f64],
    b_stride: usize,
    b_cols: usize,
    rows: usize,
) -> Vec<f64> {
    debug_assert!(b_cols > 1);
    let mut d = vec![0.0f64; a_cols * b_cols];
    let mut a_buf = vec![0.0f64; rows];
    for i in 0..a_cols {
        for k in 0..rows {
            a_buf[k] = a[k * a_stride + i];
        }
        for j in 0..b_cols {
            let mut s = 0.0f64;
            for k in 0..rows {
                s += a_buf[k] * b[k * b_stride + j];
            }
            d[i * b_cols + j] = s * 1.0;
        }
    }
    d
}

/// `cv::gemm(A, v, 1, noArray(), 0, d, GEMM_1_T)` for `A` (`rows x a_cols`, stride
/// `a_stride`) and a continuous column `v` (`rows` values): `d = A^T v`.
///
/// Ported from OpenCV 4.13 `modules/core/src/matmul.simd.hpp`, `gemmImpl` (a one-column
/// result with a continuous `B` sets `GEMM_2_T` and a zero `b_step`) and
/// `GEMMSingleMul<double, double>`, `A * Bt` branch: four partial sums over `k` stepping by
/// four (`CV_ENABLE_UNROLLED`), the remainder added to the first, then
/// `(s0 + s1 + s2 + s3) * alpha`.
pub(crate) fn gemm_at_v(
    a: &[f64],
    a_stride: usize,
    a_cols: usize,
    v: &[f64],
    rows: usize,
) -> Vec<f64> {
    let mut d = vec![0.0f64; a_cols];
    let mut a_buf = vec![0.0f64; rows];
    for i in 0..a_cols {
        for k in 0..rows {
            a_buf[k] = a[k * a_stride + i];
        }
        d[i] = dot_unrolled4(&a_buf, &v[..rows]);
    }
    d
}

/// The `A * Bt` inner loop of `GEMMSingleMul` for one element, `alpha = 1`.
fn dot_unrolled4(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut k = 0usize;
    while k + 4 <= n {
        s0 += a[k] * b[k];
        s1 += a[k + 1] * b[k + 1];
        s2 += a[k + 2] * b[k + 2];
        s3 += a[k + 3] * b[k + 3];
        k += 4;
    }
    while k < n {
        s0 += a[k] * b[k];
        k += 1;
    }
    (s0 + s1 + s2 + s3) * 1.0
}

/// The AVX2 `v_reduce_sum` of a four-lane `v_float64x4`: `_mm256_hadd_pd(a, a)`, then the
/// low and high halves added: `(a0 + a1) + (a2 + a3)`.
fn reduce4(r: [f64; 4]) -> f64 {
    (r[0] + r[1]) + (r[2] + r[3])
}

/// `NormL2_SIMD<double, double>::operator()` (AVX2 build): the sum of squares of `v`.
///
/// Ported from OpenCV 4.13 `modules/core/src/norm.simd.hpp`. Blocks of 16 go to four
/// 4-lane accumulators with `v_fma` (`_mm256_fmadd_pd`), reduced as
/// `v_reduce_sum(((r00 + r01) + r10) + r11)`; the rest is added one square at a time.
fn norm_l2_simd(v: &[f64]) -> f64 {
    let n = v.len();
    let mut j = 0usize;
    let mut s = 0.0f64;
    let mut r = [[0.0f64; 4]; 4];
    while j + 16 <= n {
        for (q, acc) in r.iter_mut().enumerate() {
            for (l, lane) in acc.iter_mut().enumerate() {
                let x = v[j + 4 * q + l];
                *lane = x.mul_add(x, *lane);
            }
        }
        j += 16;
    }
    let mut sum = [0.0f64; 4];
    for l in 0..4 {
        sum[l] = ((r[0][l] + r[1][l]) + r[2][l]) + r[3][l];
    }
    s += reduce4(sum);
    while j < n {
        let x = v[j];
        s += x * x;
        j += 1;
    }
    s
}

/// `NormDiffL2_SIMD<double, double>::operator()` (AVX2 build): the sum of squared
/// differences of `a` and `b`.
///
/// Ported from OpenCV 4.13 `modules/core/src/norm.simd.hpp`: blocks of 16 into four
/// accumulators of `v_fma(|a - b|, |a - b|, r)`, reduced as
/// `v_reduce_sum(((r0 + r1) + r2) + r3)`, then the rest one by one.
fn norm_diff_l2_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut j = 0usize;
    let mut s = 0.0f64;
    let mut r = [[0.0f64; 4]; 4];
    while j + 16 <= n {
        for (q, acc) in r.iter_mut().enumerate() {
            for (l, lane) in acc.iter_mut().enumerate() {
                let d = (a[j + 4 * q + l] - b[j + 4 * q + l]).abs();
                *lane = d.mul_add(d, *lane);
            }
        }
        j += 16;
    }
    let mut sum = [0.0f64; 4];
    for l in 0..4 {
        sum[l] = ((r[0][l] + r[1][l]) + r[2][l]) + r[3][l];
    }
    s += reduce4(sum);
    while j < n {
        let d = a[j] - b[j];
        s += d * d;
        j += 1;
    }
    s
}

/// `cv::norm(v, NORM_L2SQR)` of a continuous `CV_64F` array.
///
/// Ported from OpenCV 4.13 `modules/core/src/norm.dispatch.cpp`, `cv::norm` (one plane of
/// the `NAryMatIterator`: `result = 0; result += normL2_64f(...)`).
pub(crate) fn norm_l2sqr(v: &[f64]) -> f64 {
    let mut result = 0.0f64;
    result += norm_l2_simd(v);
    result
}

/// `cv::norm(v, NORM_L2)` of a continuous `CV_64F` array: `sqrt` of [`norm_l2sqr`].
pub(crate) fn norm_l2(v: &[f64]) -> f64 {
    norm_l2sqr(v).sqrt()
}

/// `cv::norm(m, NORM_L2)` of a `CV_64F` matrix that is not continuous (a column range):
/// the `NAryMatIterator` visits it one row at a time and adds each row's kernel result.
pub(crate) fn norm_l2_rows(rows: &[&[f64]]) -> f64 {
    let mut result = 0.0f64;
    for row in rows {
        result += norm_l2_simd(row);
    }
    result.sqrt()
}

/// `cv::norm(a, b, NORM_RELATIVE | NORM_L2)` of continuous `CV_64F` arrays:
/// `norm(a, b, NORM_L2) / (norm(b, NORM_L2) + DBL_EPSILON)`.
///
/// Ported from OpenCV 4.13 `modules/core/src/norm.dispatch.cpp`, `cv::norm` (two arrays).
pub(crate) fn norm_relative_l2(a: &[f64], b: &[f64]) -> f64 {
    let mut diff = 0.0f64;
    diff += norm_diff_l2_simd(a, b);
    let num = diff.sqrt();
    num / (norm_l2(b) + f64::EPSILON)
}

/// `cv::mean` of `n` three-channel `CV_64F` points, as `[x, y, z]`.
///
/// Ported from OpenCV 4.13 `modules/core/src/mean.dispatch.cpp`, `cv::mean`, and
/// `modules/core/src/sum.simd.hpp`, `sum_<double, double>` (no SIMD kernel for doubles;
/// `cn == 3`: three sequential sums), then `s * (1. / n)`.
pub(crate) fn mean3(points: &[[f64; 3]]) -> [f64; 3] {
    let (mut s0, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0f64);
    for p in points {
        s0 += p[0];
        s1 += p[1];
        s2 += p[2];
    }
    let n = points.len();
    #[allow(clippy::cast_precision_loss)]
    let scale = if n != 0 { 1.0 / n as f64 } else { 0.0 };
    [s0 * scale, s1 * scale, s2 * scale]
}

/// `cv::mulTransposed(src, dst, true)` for a row-major `rows x cols` `CV_64F` matrix with
/// no delta and scale 1: `dst = src^T src`, `cols x cols`.
///
/// Ported from OpenCV 4.13 `modules/core/src/matmul.dispatch.cpp`, `cv::mulTransposed`
/// (below `gemm_level` it calls `MulTransposedR`, then `completeSymm(dst, false)`), and
/// `modules/core/src/matmul.simd.hpp`, `MulTransposedR<double, double>`: each upper element
/// is `s = 0; s += col_i[k] * src[k][j]` over `k` in order (the two-lane SIMD branch keeps
/// one sequential sum per lane), times the scale.
pub(crate) fn mul_transposed_ata(src: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut dst = vec![0.0f64; cols * cols];
    let mut col_buf = vec![0.0f64; rows];
    for i in 0..cols {
        for k in 0..rows {
            col_buf[k] = src[k * cols + i];
        }
        for j in i..cols {
            let mut s = 0.0f64;
            for k in 0..rows {
                s += col_buf[k] * src[k * cols + j];
            }
            dst[i * cols + j] = s * 1.0;
        }
    }
    complete_symm_upper(&mut dst, cols);
    dst
}

/// `cv::mulTransposed(src, dst, true, delta)` with a one-row `delta` of `cols` values and
/// scale 1: `dst = (src - delta)^T (src - delta)`.
///
/// Ported from OpenCV 4.13 `modules/core/src/matmul.simd.hpp`, `MulTransposedR<double,
/// double>`, delta branch (`delta_cols == size.width`, row step 0): `col_buf[k] =
/// src[k][i] - delta[i]`, then `s += col_buf[k] * (src[k][j] - delta[j])`.
pub(crate) fn mul_transposed_ata_delta(
    src: &[f64],
    rows: usize,
    cols: usize,
    delta: &[f64],
) -> Vec<f64> {
    let mut dst = vec![0.0f64; cols * cols];
    let mut col_buf = vec![0.0f64; rows];
    for i in 0..cols {
        for k in 0..rows {
            col_buf[k] = src[k * cols + i] - delta[i];
        }
        for j in i..cols {
            let mut s = 0.0f64;
            for k in 0..rows {
                s += col_buf[k] * (src[k * cols + j] - delta[j]);
            }
            dst[i * cols + j] = s * 1.0;
        }
    }
    complete_symm_upper(&mut dst, cols);
    dst
}

/// `matU * matV` for 3x3 matrices stored row-major as nine values, through `cv::gemm`'s
/// small-matrix branch (`gemmImpl`, `len == 3`, `len == d_size.width`): `t = a0 b0 + a1 b1
/// + a2 b2` left to right, stored as `t * 1 + 0 * 0`. The arithmetic is
/// [`crate::rigid::gemm_small`]'s; this only changes the storage.
///
/// Ported from OpenCV 4.13 `modules/core/src/matmul.simd.hpp`, `gemmImpl` (`CV_64F`,
/// `case 3`, first branch).
pub(crate) fn gemm3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let rows = |m: &[f64; 9]| [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
    let d = crate::rigid::gemm_small(&rows(a), &rows(b));
    [
        d[0][0], d[0][1], d[0][2], d[1][0], d[1][1], d[1][2], d[2][0], d[2][1], d[2][2],
    ]
}

/// `cv::completeSymm(m, false)`: the lower triangle copied from the upper one.
///
/// Ported from OpenCV 4.13 `modules/core/src/matrix_operations.cpp`, `cv::completeSymm`.
pub(crate) fn complete_symm_upper(m: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            m[i * n + j] = m[j * n + i];
        }
    }
}

/// `Mat::convertTo(dst, CV_64F, alpha, 0)` of one `CV_64F` value: `x * alpha + 0`.
///
/// Ported from OpenCV 4.13 `modules/core/src/convert_scale.simd.hpp` (`cvt_64f`): with a
/// zero `beta` the fused and unfused forms give the same result.
pub(crate) fn scale(x: f64, alpha: f64) -> f64 {
    x * alpha + 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `gemm3` as `extrinsic.rs` and `homography.rs` each had it before it was joined onto
    /// `rigid::gemm_small` (third performance round, finding 7).
    fn gemm3_before(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
        let mut d = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let t = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
                d[i * 3 + j] = t * 1.0 + 0.0 * 0.0;
            }
        }
        d
    }

    /// `epnp::mul_transposed_ata` as it was before `epnp` called this module's copy (third
    /// performance round, finding 6).
    fn epnp_mul_transposed_ata_before(src: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let scale = 1.0f64;
        let mut dst = vec![0.0f64; cols * cols];
        let mut col_buf = vec![0.0f64; rows];
        for i in 0..cols {
            for k in 0..rows {
                col_buf[k] = src[k * cols + i];
            }
            for j in i..cols {
                let mut s = 0.0f64;
                for k in 0..rows {
                    s += col_buf[k] * src[k * cols + j];
                }
                dst[i * cols + j] = s * scale;
            }
        }
        for i in 0..cols {
            for j in 0..i {
                dst[i * cols + j] = dst[j * cols + i];
            }
        }
        dst
    }

    #[test]
    fn review13_mul_transposed_ata_same_bits_as_epnp_copy() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let edges = [
            0.0,
            -0.0,
            1e-200,
            -1e-200,
            f64::INFINITY,
            f64::NAN,
            2.5,
            -7.0,
        ];
        for case in 0..400 {
            let rows = (next() % 30) as usize;
            let cols = 1 + (next() % 12) as usize;
            let src: Vec<f64> = (0..rows * cols)
                .map(|_| {
                    let r = next();
                    if case % 3 == 0 {
                        edges[(r % edges.len() as u64) as usize]
                    } else {
                        (r >> 11) as f64 / (1u64 << 52) as f64 * 20.0 - 10.0
                    }
                })
                .collect();
            let got = mul_transposed_ata(&src, rows, cols);
            let want = epnp_mul_transposed_ata_before(&src, rows, cols);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.to_bits(), w.to_bits(), "case {case}");
            }
        }
    }

    #[test]
    fn review13_gemm3_same_bits_as_before() {
        let edges = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1e-300,
            -1e-300,
            5e-324,
            -5e-324,
            1e308,
            -1e308,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0xfff8_0000_0000_0007),
            0.1,
            -3.7,
        ];
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for case in 0..200_000 {
            let mut a = [0.0; 9];
            let mut b = [0.0; 9];
            for v in a.iter_mut().chain(b.iter_mut()) {
                let r = next();
                *v = if case % 2 == 0 {
                    edges[(r % edges.len() as u64) as usize]
                } else {
                    (r >> 11) as f64 / (1u64 << 52) as f64 * 200.0 - 100.0
                };
            }
            let got = gemm3(&a, &b);
            let want = gemm3_before(&a, &b);
            for k in 0..9 {
                assert_eq!(got[k].to_bits(), want[k].to_bits(), "{a:?} {b:?} entry {k}");
            }
        }
    }

    #[test]
    fn gemm_shapes() {
        // A = [[1, 2], [3, 4], [5, 6]] (3 x 2), B = A.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = gemm_at_b(&a, 2, 2, &a, 2, 2, 3);
        assert_eq!(d, vec![35.0, 44.0, 44.0, 56.0]);
        let v = gemm_at_v(&a, 2, 2, &[1.0, 1.0, 1.0], 3);
        assert_eq!(v, vec![9.0, 12.0]);
    }

    #[test]
    fn jacobi_eigen() {
        let (w, v) = crate::linalg::eigen_jacobi(&[2.0, 1.0, 1.0, 2.0], 2);
        assert!((w[0] - 3.0).abs() < 1e-15 && (w[1] - 1.0).abs() < 1e-15);
        assert!((v[0].abs() - v[1].abs()).abs() < 1e-15);
        let x = crate::linalg::solve_eig(&[2.0, 1.0, 1.0, 2.0], 2, &[3.0, 3.0], 1);
        assert!((x[0] - 1.0).abs() < 1e-15 && (x[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn norms() {
        let v: Vec<f64> = (0..37).map(f64::from).collect();
        let want: f64 = (0..37).map(|i| f64::from(i * i)).sum();
        assert_eq!(norm_l2sqr(&v), want);
        let z = vec![0.0; 37];
        assert_eq!(norm_relative_l2(&v, &v), 0.0);
        assert!((norm_relative_l2(&z, &v) - 1.0).abs() < 1e-15);
    }
}
