//! Per-pixel arithmetic as OpenCV 4.13 computes it: `Mat::convertTo` between 8-bit and
//! 32-bit float, subtraction, `normalize` with `NORM_MINMAX`, `minMaxLoc`, `threshold` and
//! `Scalar::all(255) - image`.
//!
//! Called from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`, and the
//! 2D tracker (`processing/MarkerTracking.cpp`).

use crate::image::{Image, ImageF32, ImageU8, ImageU16};

/// `saturate_cast<uchar>(float)`: `cvRound` (half to even) then saturation to 0..255. A value
/// outside the `int` range, or NaN, gives 0: x86's `cvtss2si` / `cvtps2dq` return `INT_MIN`
/// for those, which saturates to 0 (so `1e10` becomes 0, not 255).
///
/// Ported from `modules/core/include/opencv2/core/saturate.hpp`, `saturate_cast<uchar>(float)`
/// and `modules/core/src/convert.hpp`, `v_store_pair_as(uchar*, v_float32, v_float32)`.
#[must_use]
pub fn saturate_u8(v: f32) -> u8 {
    let r = v.round_ties_even();
    // cvRound: in range of i32, else INT_MIN.
    if !(-2_147_483_648.0..2_147_483_648.0).contains(&r) {
        return 0;
    }
    r.clamp(0.0, 255.0) as u8
}

/// `src.convertTo(dst, CV_32F)` for an 8-bit image: exact.
///
/// Ported from `modules/core/src/convert.dispatch.cpp`, `Mat::convertTo` (`cvt8u32f`).
#[must_use]
pub fn u8_to_f32(src: &ImageU8) -> ImageF32 {
    src.map(f32::from)
}

/// `src.convertTo(dst, CV_8U)` for a float image (`alpha = 1`, `beta = 0`): each value
/// rounded half to even and saturated, see [`saturate_u8`].
///
/// Ported from `modules/core/src/convert.dispatch.cpp`, `Mat::convertTo` with
/// `convert.simd.hpp`, `cvt32f8u` (`v_pack_u(v_pack(v_round(a), v_round(b)))` in the vector
/// loop, `saturate_cast<uchar>` in the tail; the two agree).
#[must_use]
pub fn f32_to_u8(src: &ImageF32) -> ImageU8 {
    src.map(saturate_u8)
}

/// `src.convertTo(dst, CV_8U, alpha, beta)` for a 16-bit image: `alpha` and `beta` rounded
/// to `f32`, each value `fma(v, alpha, beta)` in `f32`, then [`saturate_u8`]. Works per value,
/// so an interleaved multi-channel image can be passed as one channel `channels` times wider.
///
/// Ported from `modules/core/src/convert_scale.simd.hpp`, `cvt_32f` (`cvtScale16u8u`); the
/// scalar tail `saturate_cast<uchar>(src[j]*a + b)` is contracted to the same fused
/// multiply-add (see [`convert_scale_f32`]).
#[must_use]
pub fn u16_to_u8_scaled(src: &ImageU16, alpha: f64, beta: f64) -> ImageU8 {
    let (a, b) = (alpha as f32, beta as f32);
    src.map(|v| saturate_u8(f32::from(v).mul_add(a, b)))
}

/// `a - b` (the `MatExpr` subtraction, `cv::subtract`) for float images: the IEEE difference
/// of each pair.
///
/// Ported from `modules/core/src/arithm.cpp`, `cv::subtract` (`sub32f`).
///
/// # Panics
/// If the sizes differ.
#[must_use]
pub fn subtract_f32(a: &ImageF32, b: &ImageF32) -> ImageF32 {
    assert_eq!(
        (a.width(), a.height()),
        (b.width(), b.height()),
        "image sizes differ"
    );
    let data = a.data().iter().zip(b.data()).map(|(x, y)| x - y).collect();
    ImageF32::from_vec(a.width(), a.height(), data)
}

/// `Scalar::all(255) - src` for an 8-bit image: `255 - v` for each value (the saturating
/// subtraction never saturates for 8-bit input).
///
/// Ported from `modules/core/src/arithm.cpp`, `cv::subtract` with a scalar first operand.
#[must_use]
pub fn invert_u8(src: &ImageU8) -> ImageU8 {
    src.map(|v| 255 - v)
}

/// `src.convertTo(dst, CV_32F, alpha, beta)` for a float image with scaling: `alpha` and
/// `beta` are rounded to `f32` and each value is `fma(v, alpha, beta)` in `f32`.
///
/// Ported from `modules/core/src/convert_scale.simd.hpp`, `cvt_32f` (`cvtScale32f`). The
/// vector loop uses `v_fma`; the scalar tail `src[j]*a + b` runs only when the conversion is
/// in place (or the image has fewer than 16 values) and is contracted to the same fused
/// multiply-add by the compiler: the reference OpenCV gives identical results in place and
/// not in place (checked by the `normalize-*` cases of `oracle/py/gen_primitives_imgops.py`).
#[must_use]
pub fn convert_scale_f32(src: &ImageF32, alpha: f64, beta: f64) -> ImageF32 {
    let (a, b) = (alpha as f32, beta as f32);
    src.map(|v| v.mul_add(a, b))
}

/// The scale and shift `cv::normalize(src, dst, a, b, NORM_MINMAX)` applies to a float
/// image whose minimum and maximum are `smin` and `smax`, as `f64` (already rounded as OpenCV
/// rounds them for a `CV_32F` result).
///
/// Ported from `modules/core/src/norm.dispatch.cpp`, `cv::normalize`:
/// `scale = (dmax - dmin) * (smax - smin > DBL_EPSILON ? 1/(smax - smin) : 0)`, then
/// `scale = (float)scale` and `shift = (float)dmin - (float)(smin * scale)` in `double`.
#[must_use]
pub fn normalize_minmax_params(smin: f64, smax: f64, a: f64, b: f64) -> (f64, f64) {
    let (dmin, dmax) = (a.min(b), a.max(b));
    let inv = if smax - smin > f64::EPSILON {
        1.0 / (smax - smin)
    } else {
        0.0
    };
    let scale = f64::from(((dmax - dmin) * inv) as f32);
    let shift = f64::from(dmin as f32) - f64::from((smin * scale) as f32);
    (scale, shift)
}

/// `cv::normalize(src, dst, a, b, NORM_MINMAX)` for a float image, result float (`dtype` -1,
/// no mask): maps the minimum to `min(a, b)` and the maximum to `max(a, b)` (approximately:
/// the scale and shift are rounded to `f32`). A flat image (range at most `DBL_EPSILON`)
/// becomes `min(a, b)` everywhere. Detection uses 0..255; the tracker 0..(100 - maxPenalty)
/// and 0..1.
///
/// Ported from `modules/core/src/norm.dispatch.cpp`, `cv::normalize`, with
/// [`normalize_minmax_params`] and [`convert_scale_f32`]. The minimum and maximum are those of
/// `minMaxIdx` (NaN not handled).
#[must_use]
pub fn normalize_minmax_f32(src: &ImageF32, a: f64, b: f64) -> ImageF32 {
    if src.is_empty() {
        return src.clone();
    }
    let m = min_max_loc_f32(src);
    let (scale, shift) = normalize_minmax_params(f64::from(m.min_val), f64::from(m.max_val), a, b);
    convert_scale_f32(src, scale, shift)
}

/// Result of `cv::minMaxLoc`: the extreme values and their positions `(x, y)` (column, row).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxLoc<T> {
    /// Smallest value.
    pub min_val: T,
    /// Largest value.
    pub max_val: T,
    /// Position `(x, y)` of the first smallest value in row order.
    pub min_loc: (usize, usize),
    /// Position `(x, y)` of the first largest value in row order.
    pub max_loc: (usize, usize),
}

fn min_max_loc<T: Copy + Default + PartialOrd>(src: &Image<T>) -> MinMaxLoc<T> {
    let w = src.width().max(1);
    let d = src.data();
    if d.is_empty() {
        // OpenCV reports 0 values and (-1, -1); there is no such position here.
        return MinMaxLoc {
            min_val: T::default(),
            max_val: T::default(),
            min_loc: (0, 0),
            max_loc: (0, 0),
        };
    }
    let (mut imin, mut imax) = (0, 0);
    for (i, &v) in d.iter().enumerate().skip(1) {
        if v < d[imin] {
            imin = i;
        }
        if v > d[imax] {
            imax = i;
        }
    }
    MinMaxLoc {
        min_val: d[imin],
        max_val: d[imax],
        min_loc: (imin % w, imin / w),
        max_loc: (imax % w, imax / w),
    }
}

/// `cv::minMaxLoc` on an 8-bit image: the values and the first position (row order) of
/// each.
///
/// Ported from `modules/core/src/minmax.cpp`, `cv::minMaxIdx` / `minMaxIdx_8u`.
#[must_use]
pub fn min_max_loc_u8(src: &ImageU8) -> MinMaxLoc<u8> {
    min_max_loc(src)
}

/// `cv::minMaxLoc` on a float image: the values and the first position (row order) of each,
/// comparing with `<` and `>` (so `-0` and `0` tie, and the value reported is the one at the
/// first position). NaN not handled.
///
/// Ported from `modules/core/src/minmax.cpp`, `cv::minMaxIdx` / `minMaxIdx_32f`.
#[must_use]
pub fn min_max_loc_f32(src: &ImageF32) -> MinMaxLoc<f32> {
    min_max_loc(src)
}

/// The `cv::threshold` types ported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdType {
    /// `THRESH_BINARY`: `src > thresh ? maxval : 0`.
    Binary,
    /// `THRESH_BINARY_INV`: `src > thresh ? 0 : maxval`.
    BinaryInv,
}

/// `cv::threshold(src, dst, thresh, maxval, type)` on an 8-bit image. Returns the image and
/// OpenCV's return value, the threshold actually used.
///
/// For 8-bit input OpenCV floors `thresh` to an integer (`cvFloor`) and rounds `maxval`
/// (`cvRound`, then saturated to 0..255); a floored threshold below 0 or at least 255 fills
/// the image with one value (`BINARY_INV`: 0 below, `maxval` at or above 255; `BINARY` the
/// opposite).
///
/// Ported from `modules/imgproc/src/thresh.cpp`, `cv::threshold` and `thresh_8u`.
#[must_use]
pub fn threshold_u8(
    src: &ImageU8,
    thresh: f64,
    maxval: f64,
    kind: ThresholdType,
) -> (ImageU8, f64) {
    let t = thresh.floor();
    // cvFloor of NaN or of a value outside int is INT_MIN on x86.
    let ithresh: i64 = if t.is_nan() || t < -2_147_483_648.0 || t >= 2_147_483_648.0 {
        i64::from(i32::MIN)
    } else {
        t as i64
    };
    let imax = maxval.round_ties_even();
    let imax: u8 = if imax.is_nan() || imax < -2_147_483_648.0 || imax >= 2_147_483_648.0 {
        0
    } else {
        imax.clamp(0.0, 255.0) as u8
    };
    let ret = ithresh as f64;
    if !(0..255).contains(&ithresh) {
        let v = match kind {
            ThresholdType::Binary => {
                if ithresh >= 255 {
                    0
                } else {
                    imax
                }
            }
            ThresholdType::BinaryInv => {
                if ithresh >= 255 {
                    imax
                } else {
                    0
                }
            }
        };
        return (ImageU8::filled(src.width(), src.height(), v), ret);
    }
    let th = ithresh as u8;
    let out = match kind {
        ThresholdType::Binary => src.map(|v| if v > th { imax } else { 0 }),
        ThresholdType::BinaryInv => src.map(|v| if v > th { 0 } else { imax }),
    };
    (out, ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturation_and_rounding() {
        assert_eq!(saturate_u8(0.5), 0);
        assert_eq!(saturate_u8(1.5), 2);
        assert_eq!(saturate_u8(254.5), 254);
        assert_eq!(saturate_u8(-3.0), 0);
        assert_eq!(saturate_u8(300.0), 255);
        assert_eq!(saturate_u8(1e10), 0);
        assert_eq!(saturate_u8(f32::NAN), 0);
    }

    #[test]
    fn threshold_edges() {
        let src = ImageU8::from_vec(4, 1, vec![0, 20, 21, 255]);
        let (d, r) = threshold_u8(&src, 20.9, 255.0, ThresholdType::BinaryInv);
        assert_eq!(d.data(), &[255, 255, 0, 0]);
        assert_eq!(r, 20.0);
        let (d, _) = threshold_u8(&src, 20.0, 100.0, ThresholdType::Binary);
        assert_eq!(d.data(), &[0, 0, 100, 100]);
        let (d, _) = threshold_u8(&src, -0.5, 255.0, ThresholdType::BinaryInv);
        assert!(d.data().iter().all(|&v| v == 0));
        let (d, _) = threshold_u8(&src, 255.0, 255.0, ThresholdType::Binary);
        assert!(d.data().iter().all(|&v| v == 0));
    }

    #[test]
    fn flat_normalize_gives_lower_bound() {
        let src = ImageF32::filled(3, 3, 7.0);
        let n = normalize_minmax_f32(&src, 0.0, 255.0);
        assert!(n.data().iter().all(|&v| v == 0.0));
        let n = normalize_minmax_f32(&src, 5.0, 1.0);
        assert!(n.data().iter().all(|&v| v == 1.0));
    }
}
