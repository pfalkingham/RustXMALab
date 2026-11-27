//! OpenCV's corner detector (`goodFeaturesToTrack`, with `cornerMinEigenVal` or `cornerHarris`)
//! and sub-pixel refinement (`cornerSubPix`), and XMALab's detection method 3 built on them.
//!
//! # What XMALab's method 3 runs
//!
//! `processing/MarkerDetection.cpp:323` calls
//! `goodFeaturesToTrack(subimage, corners, 50, 0.0001, 3, cv::Mat(), 7, use_harris)` with
//! `int use_harris = true`. Since OpenCV 3.4 there is an overload
//! `goodFeaturesToTrack(..., InputArray mask, int blockSize, int gradientSize,
//! bool useHarrisDetector = false, double k = 0.04)`, and an `int` eighth argument matches it
//! better than the `bool useHarrisDetector` of the older overload. So XMALab 3 (built against
//! OpenCV 4.13) runs the **minimum-eigenvalue** detector with a **gradient aperture of 1**, not
//! the Harris detector its source names. This was confirmed by compiling the literal C++ call
//! against the 4.13 headers and libraries of the XMALab 3 AppImage
//! (`oracle/py/cv413_blobcorner/probe.cpp`, command `X`). [`XMALAB_METHOD3`] holds the
//! parameters actually used.
//!
//! # Rounding
//!
//! Sobel and box filters are reproduced as OpenCV's generic (non-IPP) code computes them on
//! x86-64 with its AVX2 dispatch, and checked against OpenCV's own eigenvalue images
//! (`cornereig-bc-*`). With gradient aperture 1 every step is a single rounding, so no path
//! choice matters; with aperture 3 the multiply-adds are fused as described at
//! [`sobel_u8`]. `cornerHarris` mixes three formulas by position (see [`corner_harris`]).
//! Only block sizes above 5 are ported: OpenCV filters smaller float blocks with a different
//! routine (`blockSum`).
//!
//! # Images deeper than 8 bits
//!
//! The deep path (parity ledger V1, `crate::depth`) runs method 3 on the patch as `f32` on
//! the 8-bit scale ([`detect_corner_marker_f32`]) with the 8-bit path's arithmetic: the
//! derivative scale keeps the factor 1/255 OpenCV applies to 8-bit input, the Sobel rows
//! are rounded as for 8-bit input, and `cornerSubPix` samples its window with the scheme
//! OpenCV uses for 8-bit input (`getRectSubPix_8u32f`). On an 8-bit image promoted to a
//! deeper one the values are the same, so the result is the 8-bit path's exactly. With
//! gradient aperture 1 (what method 3 runs) every Sobel and covariance value is also what
//! OpenCV computes for a `CV_32F` image apart from the 1/255 factor
//! ([`corner_min_eigen_val_f32`] is OpenCV's float form, checked against it).

use crate::detection::Detection;
use crate::image::{GreySource, ImageF32, ImageU8};
use crate::imgproc::blob::SelectionCompat;
use crate::imgproc::blur::reflect101;

/// The arguments of `cv::goodFeaturesToTrack` (no mask).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GfttParams {
    /// Most corners returned; 0 or less: no limit.
    pub max_corners: i32,
    /// Corners weaker than `quality_level` times the strongest are dropped (must be > 0).
    pub quality_level: f64,
    /// Smallest distance between returned corners (must be >= 0).
    pub min_distance: f64,
    /// Neighbourhood of the covariance sum (`blockSize`); only values above 5 are ported.
    pub block_size: usize,
    /// Sobel aperture (`gradientSize`): 1 or 3.
    pub gradient_size: i32,
    /// Harris response instead of the minimum eigenvalue.
    pub use_harris: bool,
    /// Harris `k`.
    pub k: f64,
}

/// The detector XMALab 3's method 3 actually runs: 50 corners, quality 0.0001, minimum
/// distance 3, block size 7, gradient aperture 1, minimum eigenvalue (see the module
/// documentation for why not Harris).
pub const XMALAB_METHOD3: GfttParams = GfttParams {
    max_corners: 50,
    quality_level: 0.0001,
    min_distance: 3.0,
    block_size: 7,
    gradient_size: 1,
    use_harris: false,
    k: 0.04,
};

/// `cv::TermCriteria` for [`corner_sub_pix`]: `None` leaves that criterion out of the type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TermCriteria {
    /// `MAX_ITER` with this count.
    pub max_iter: Option<i32>,
    /// `EPS` with this epsilon.
    pub epsilon: Option<f64>,
}

/// Why `cv::cornerSubPix` throws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CornerSubPixError {
    /// A window size is not positive.
    Window,
    /// The image is smaller than `2 * win + 5` in a direction.
    ImageTooSmall,
    /// A corner lies outside the image.
    CornerOutside,
}

/// Lanes of OpenCV's AVX2 `v_uint8` (`RowVec_8u32f` processes this many pixels per step).
const ROW_LANES_U8: usize = 32;

/// The two kernels of `getSobelKernels` (`normalize = false`) for a first derivative, as
/// `f32`, with `scale` applied to the smoothing kernel as `cv::Sobel` does (`kx *= scale` for
/// `dx == 0`, else `ky *= scale`; `Mat` scaling: the scale rounded to `f32`, one product).
///
/// Ported from imgproc/src/deriv.cpp, `getSobelKernels` and `cv::Sobel`.
fn sobel_kernels(dx: bool, ksize: i32, scale: f64) -> (Vec<f32>, Vec<f32>) {
    let deriv = vec![-1.0f32, 0.0, 1.0];
    let smooth: Vec<f32> = match ksize {
        1 => vec![1.0],
        3 => vec![1.0, 2.0, 1.0],
        _ => panic!("Sobel aperture {ksize} is not ported (1 or 3)"),
    };
    let s = scale as f32;
    let scaled: Vec<f32> = smooth.iter().map(|v| v * s).collect();
    let keep = scale == 1.0;
    if dx {
        (deriv, if keep { smooth } else { scaled })
    } else {
        (if keep { smooth } else { scaled }, deriv)
    }
}

/// `cv::Sobel(src, dst, CV_32F, dx, dy, ksize, scale, 0, BORDER_REFLECT_101)` for a first
/// derivative in x (`dx`) or y, `ksize` 1 or 3, on an 8-bit image.
///
/// The separable filter runs rows first (`RowFilter<uchar, float, RowVec_8u32f>`), then
/// columns (`SymmColumnFilter` for one tap, `SymmColumnSmallFilter` for three). Rounding,
/// as found against OpenCV's own output (`cornereig-bc-*`): the row taps form a fused chain
/// `fma(x2, k2, fma(x1, k1, x0 * k0))` for the pixels the vector loop covers
/// (`x < 32 * (w / 32)`) and `(x0 k0 + x1 k1) + x2 k2`, each step rounded, for the rest; a
/// symmetric 3-tap column is `fma(S-1 + S+1, k1, S0 * k0)`, an antisymmetric one
/// `S+1 - S-1`; a 1-tap column `S0 * k0`. With `ksize = 1` all of these are single roundings
/// (the row kernel is `-1, 0, 1` or one tap), so no choice of path matters.
///
/// Ported from imgproc/src/deriv.cpp, `cv::Sobel`, with filter.dispatch.cpp,
/// `createSeparableLinearFilter`, and filter.simd.hpp, `RowFilter`, `RowVec_8u32f`,
/// `SymmColumnFilter`, `SymmColumnSmallFilter`, `SymmColumnSmallVec_32f`.
///
/// # Panics
/// If `ksize` is not 1 or 3.
#[must_use]
pub fn sobel_u8(src: &ImageU8, dx: bool, ksize: i32, scale: f64) -> ImageF32 {
    sobel(src, dx, ksize, scale)
}

/// `cv::Sobel(src, dst, CV_32F, dx, dy, ksize, scale, 0, BORDER_REFLECT_101)` on a float
/// image, computed as [`sobel_u8`] computes it. For `ksize = 1` (what XMALab's method 3 runs)
/// every value is a single rounding of the same operands, so this is OpenCV's float `Sobel`
/// exactly; for `ksize = 3` the rows are rounded as for 8-bit input, which OpenCV's float
/// row filter (`SymmRowSmallVec_32f`) does not promise.
///
/// Ported from imgproc/src/deriv.cpp, `cv::Sobel` for `CV_32F` (extends [`sobel_u8`]).
///
/// # Panics
/// If `ksize` is not 1 or 3.
#[must_use]
pub fn sobel_f32(src: &ImageF32, dx: bool, ksize: i32, scale: f64) -> ImageF32 {
    sobel(src, dx, ksize, scale)
}

/// [`sobel_u8`] on any [`GreySource`], with the rounding of the 8-bit row filter. For
/// `ksize = 1` every value is a single rounding of the same operands, so this is also
/// OpenCV's `Sobel` on a `CV_32F` image of the source's values.
///
/// Ported from imgproc/src/deriv.cpp, `cv::Sobel` (as [`sobel_u8`]).
fn sobel<S: GreySource + ?Sized>(src: &S, dx: bool, ksize: i32, scale: f64) -> ImageF32 {
    let (w, h) = (src.width(), src.height());
    let (kx, ky) = sobel_kernels(dx, ksize, scale);
    let (rx, ry) = (kx.len() / 2, ky.len() / 2);
    if w == 0 || h == 0 {
        return ImageF32::new(w, h);
    }
    // Row pass: RowVec_8u32f handles 32 pixels at a time (AVX2 `v_uint8`).
    let vec_end = w / ROW_LANES_U8 * ROW_LANES_U8;
    let mut rows = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let px =
                |t: usize| src.value(y * w + reflect101(x as isize + t as isize - rx as isize, w));
            let mut s = px(0) * kx[0];
            for (t, &k) in kx.iter().enumerate().skip(1) {
                s = if x < vec_end {
                    px(t).mul_add(k, s)
                } else {
                    s + px(t) * k
                };
            }
            rows[y * w + x] = s;
        }
    }
    // Column pass.
    let mut dst = ImageF32::new(w, h);
    for y in 0..h {
        let r = |d: isize| &rows[reflect101(y as isize + d, h) * w..][..w];
        let out = dst.row_mut(y);
        if ky.len() == 1 {
            let s0 = r(0);
            for x in 0..w {
                out[x] = s0[x] * ky[0] + 0.0;
            }
        } else {
            debug_assert_eq!(ry, 1);
            let (sa, s0, sb) = (r(-1), r(0), r(1));
            if ky[1] == 0.0 {
                // [-1, 0, 1]
                for x in 0..w {
                    out[x] = sb[x] - sa[x];
                }
            } else {
                for x in 0..w {
                    out[x] = (sa[x] + sb[x]).mul_add(ky[2], s0[x] * ky[1]);
                }
            }
        }
    }
    dst
}

/// `cv::boxFilter(src, dst, CV_32F, Size(k, k), Point(-1, -1), false, BORDER_REFLECT_101)` on
/// a float image, `k > 5`: unnormalised sums, kept in `double` and rounded to `float` at the
/// end, with OpenCV's running sums (each row sum adds the entering value minus the leaving
/// one; each column sum adds the entering row, writes, then subtracts the leaving row).
///
/// Ported from imgproc/src/box_filter.simd.hpp, `RowSum<float, double>` and
/// `ColumnSum<double, float>` (no scale), driven as `createBoxFilter` / `FilterEngine` do.
fn box_sum(src: &[f32], w: usize, h: usize, k: usize) -> Vec<f32> {
    let a = k / 2;
    // Row sums.
    let mut rs = vec![0f64; w * h];
    let mut ext = vec![0f64; w + k - 1];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for (j, e) in ext.iter_mut().enumerate() {
            *e = f64::from(row[reflect101(j as isize - a as isize, w)]);
        }
        let mut s = 0f64;
        for e in &ext[..k] {
            s += e;
        }
        let d = &mut rs[y * w..(y + 1) * w];
        d[0] = s;
        for i in 0..w - 1 {
            s += ext[i + k] - ext[i];
            d[i + 1] = s;
        }
    }
    // Column sums.
    let ext_row = |j: usize| &rs[reflect101(j as isize - a as isize, h) * w..][..w];
    let mut sum = vec![0f64; w];
    for j in 0..k - 1 {
        for (s, v) in sum.iter_mut().zip(ext_row(j)) {
            *s += v;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let sp = ext_row(y + k - 1);
        let sm = ext_row(y);
        let d = &mut out[y * w..(y + 1) * w];
        for i in 0..w {
            let s0 = sum[i] + sp[i];
            d[i] = s0 as f32;
            sum[i] = s0 - sm[i];
        }
    }
    out
}

/// The three covariance images of `cornerEigenValsVecs` after the box filter: `sum(dx^2)`,
/// `sum(dx dy)`, `sum(dy^2)` over the block.
///
/// Ported from imgproc/src/corner.cpp, `cornerEigenValsVecs` (up to `boxFilter`).
///
/// `scale_255` is OpenCV's `depth == CV_8U` test: the derivatives are divided by 255 for
/// 8-bit input and for the deep path's values on the 8-bit scale, not for a float image.
fn covariance<S: GreySource + ?Sized>(
    src: &S,
    block_size: usize,
    ksize: i32,
    scale_255: bool,
) -> [Vec<f32>; 3] {
    assert!(
        block_size > 5,
        "block sizes up to 5 use OpenCV's blockSum, which is not ported"
    );
    // Before the scale: `1 << (ksize - 1)` overflows for an aperture above 32, which would
    // otherwise panic with a message the functions do not document.
    assert!(
        ksize == 1 || ksize == 3,
        "Sobel aperture {ksize} is not ported (1 or 3)"
    );
    let mut scale = f64::from(1 << ((if ksize > 0 { ksize } else { 3 }) - 1)) * block_size as f64;
    if scale_255 {
        scale *= 255.0;
    }
    let scale = 1.0 / scale;
    let dx = sobel(src, true, ksize, scale);
    let dy = sobel(src, false, ksize, scale);
    let (w, h) = (src.width(), src.height());
    let mut a = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    let mut c = vec![0f32; w * h];
    for (i, (&x, &y)) in dx.data().iter().zip(dy.data()).enumerate() {
        a[i] = x * x;
        b[i] = x * y;
        c[i] = y * y;
    }
    [
        box_sum(&a, w, h, block_size),
        box_sum(&b, w, h, block_size),
        box_sum(&c, w, h, block_size),
    ]
}

/// `cv::cornerMinEigenVal(src, dst, blockSize, ksize)` (default border) on an 8-bit image: the
/// smaller eigenvalue of the gradient covariance over each block,
/// `(a + c) - sqrt((a - c)^2 + b^2)` with `a`, `c` the halved diagonal, in `float`.
///
/// Ported from imgproc/src/corner.cpp, `cv::cornerMinEigenVal`, `cornerEigenValsVecs` and
/// `calcMinEigenVal` (every code path gives the same result).
///
/// # Panics
/// If `ksize` is not 1 or 3, or `block_size` is 5 or less.
#[must_use]
pub fn corner_min_eigen_val(src: &ImageU8, block_size: usize, ksize: i32) -> ImageF32 {
    min_eigen_val(src, block_size, ksize, true)
}

/// [`corner_min_eigen_val`] on a float image, as OpenCV computes it for `CV_32F` input: the
/// same arithmetic without the factor 1/255 in the derivative scale. Identical to OpenCV for
/// `ksize = 1` (every Sobel value a single rounding); for `ksize = 3` the rows are rounded as
/// for 8-bit input, which OpenCV's float row filter does not promise.
///
/// Ported from imgproc/src/corner.cpp, `cv::cornerMinEigenVal` for `CV_32F` (extends
/// [`corner_min_eigen_val`]).
///
/// # Panics
/// As [`corner_min_eigen_val`].
#[must_use]
pub fn corner_min_eigen_val_f32(src: &ImageF32, block_size: usize, ksize: i32) -> ImageF32 {
    min_eigen_val(src, block_size, ksize, false)
}

/// The response the deep path's method 3 computes ([`detect_corner_marker_f32`]): the float
/// image with the derivative scale OpenCV uses for 8-bit input (the factor 1/255 kept).
/// Exposed for the comparison with OpenCV (`cornereig255-fulldepth-*`).
///
/// Ported from imgproc/src/corner.cpp, `cornerEigenValsVecs` (`depth == CV_8U` scale) and
/// `calcMinEigenVal` (extends [`corner_min_eigen_val`]).
///
/// # Panics
/// As [`corner_min_eigen_val`].
#[must_use]
pub fn corner_min_eigen_val_8bit_scale(src: &ImageF32, block_size: usize, ksize: i32) -> ImageF32 {
    min_eigen_val(src, block_size, ksize, true)
}

/// The body of [`corner_min_eigen_val`] and [`corner_min_eigen_val_f32`].
///
/// Ported from imgproc/src/corner.cpp, `cornerEigenValsVecs` and `calcMinEigenVal`.
fn min_eigen_val<S: GreySource + ?Sized>(
    src: &S,
    block_size: usize,
    ksize: i32,
    scale_255: bool,
) -> ImageF32 {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return ImageF32::new(w, h);
    }
    let [ca, cb, cc] = covariance(src, block_size, ksize, scale_255);
    let data = (0..w * h)
        .map(|i| {
            let a = ca[i] * 0.5;
            let b = cb[i];
            let c = cc[i] * 0.5;
            let t = a - c;
            (a + c) - (b * b + t * t).sqrt()
        })
        .collect();
    ImageF32::from_vec(w, h, data)
}

/// `cv::cornerHarris(src, dst, blockSize, ksize, k)` (default border) on an 8-bit image:
/// `det - k trace^2` of the gradient covariance over each block.
///
/// The image is processed as one line of `n = w * h` values and the formula depends on the
/// position `j` (the AVX line function, then OpenCV's 128-bit universal intrinsics, then the
/// scalar loop):
/// - `j < 8 * (n / 8)`: `(a c - b b) - k_f (ac ac)` in `float`, `k_f = (float)k`;
/// - the next 4 if at least 4 remain: `(a c - b b) - (k_f ac) ac` in `float`;
/// - the last `n % 4` (at most 3, in the image's last row): `a c - b b` in `float`, then
///   `- k (a + c)(a + c)` in `double`, rounded to `float`.
///
/// Ported from imgproc/src/corner.cpp, `cv::cornerHarris`, `cornerEigenValsVecs` and
/// `calcHarris`, with corner.avx.cpp, `calcHarrisLine_AVX`.
///
/// # Panics
/// If `ksize` is not 1 or 3, or `block_size` is 5 or less.
#[must_use]
pub fn corner_harris(src: &ImageU8, block_size: usize, ksize: i32, k: f64) -> ImageF32 {
    harris(src, block_size, ksize, k, true)
}

/// The body of [`corner_harris`], on any [`GreySource`] (see [`covariance`] for
/// `scale_255`).
///
/// Ported from imgproc/src/corner.cpp, `cornerEigenValsVecs` and `calcHarris`.
fn harris<S: GreySource + ?Sized>(
    src: &S,
    block_size: usize,
    ksize: i32,
    k: f64,
    scale_255: bool,
) -> ImageF32 {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return ImageF32::new(w, h);
    }
    let [ca, cb, cc] = covariance(src, block_size, ksize, scale_255);
    let n = w * h;
    let avx_end = n / 8 * 8;
    let sse_end = if n - avx_end >= 4 {
        avx_end + 4
    } else {
        avx_end
    };
    let kf = k as f32;
    let data = (0..n)
        .map(|j| {
            let (a, b, c) = (ca[j], cb[j], cc[j]);
            let ac_bb = a * c - b * b;
            let ac = a + c;
            if j < avx_end {
                ac_bb - kf * (ac * ac)
            } else if j < sse_end {
                ac_bb - (kf * ac) * ac
            } else {
                let s = f64::from(a + c);
                (f64::from(ac_bb) - k * s * s) as f32
            }
        })
        .collect();
    ImageF32::from_vec(w, h, data)
}

/// `cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, noArray(),
/// blockSize, gradientSize, useHarrisDetector, k)` on an 8-bit image: local maxima (3 x 3) of
/// the corner response above `qualityLevel` times the largest, not on the image's outer rows
/// and columns, strongest first (ties: the later pixel in row order first), each at least
/// `minDistance` from every stronger corner kept (checked in a grid of `cvRound(minDistance)`
/// cells, only when `minDistance >= 1`), at most `maxCorners` of them. Positions are whole
/// pixels.
///
/// Ported from imgproc/src/featureselect.cpp, `cv::goodFeaturesToTrack` (the CPU path, no
/// mask), with [`corner_min_eigen_val`] or [`corner_harris`], `minMaxLoc`,
/// `threshold(THRESH_TOZERO)` and `dilate` with a 3 x 3 rectangle.
///
/// # Panics
/// If `quality_level <= 0`, `min_distance < 0`, `max_corners < 0` (OpenCV's assertions), or
/// the block or gradient size is not ported.
#[must_use]
pub fn good_features_to_track(image: &ImageU8, p: &GfttParams) -> Vec<(f32, f32)> {
    gftt(image, p, true)
}

/// [`good_features_to_track`] on a float image as OpenCV computes it for `CV_32F` input (the
/// response without the factor 1/255; see [`corner_min_eigen_val_f32`]).
///
/// Ported from imgproc/src/featureselect.cpp, `cv::goodFeaturesToTrack` for `CV_32F`
/// (extends [`good_features_to_track`]).
///
/// # Panics
/// As [`good_features_to_track`].
#[must_use]
pub fn good_features_to_track_f32(image: &ImageF32, p: &GfttParams) -> Vec<(f32, f32)> {
    gftt(image, p, false)
}

/// The body of [`good_features_to_track`] on any [`GreySource`] (see [`covariance`] for
/// `scale_255`).
///
/// Ported from imgproc/src/featureselect.cpp, `cv::goodFeaturesToTrack`.
fn gftt<S: GreySource + ?Sized>(image: &S, p: &GfttParams, scale_255: bool) -> Vec<(f32, f32)> {
    assert!(
        p.quality_level > 0.0 && p.min_distance >= 0.0 && p.max_corners >= 0,
        "goodFeaturesToTrack arguments"
    );
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let eig = if p.use_harris {
        harris(image, p.block_size, p.gradient_size, p.k, scale_255)
    } else {
        min_eigen_val(image, p.block_size, p.gradient_size, scale_255)
    };
    let e = eig.data();
    let max_val = f64::from(
        e.iter()
            .copied()
            .fold(e[0], |m, v| if v > m { v } else { m }),
    );
    let thresh = (max_val * p.quality_level) as f32;
    let eig: Vec<f32> = e
        .iter()
        .map(|&v| if v > thresh { v } else { 0.0 })
        .collect();

    let mut cand: Vec<usize> = Vec::new();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w - 1 {
            let i = y * w + x;
            let val = eig[i];
            if val == 0.0 {
                continue;
            }
            // dilate with a 3 x 3 rectangle: the neighbourhood is inside the image here.
            let mut m = f32::NEG_INFINITY;
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    let v = eig[yy * w + xx];
                    if v > m {
                        m = v;
                    }
                }
            }
            if val == m {
                cand.push(i);
            }
        }
    }
    if cand.is_empty() {
        return Vec::new();
    }
    // greaterThanPtr: larger value first, then larger address.
    cand.sort_by(|&a, &b| {
        eig[b]
            .partial_cmp(&eig[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.cmp(&a))
    });

    let max_corners = p.max_corners as usize;
    let mut corners: Vec<(f32, f32)> = Vec::new();
    if p.min_distance >= 1.0 {
        let cell = p.min_distance.round_ties_even() as usize;
        let gw = w.div_ceil(cell);
        let gh = h.div_ceil(cell);
        let mut grid: Vec<Vec<(f32, f32)>> = vec![Vec::new(); gw * gh];
        let min_d2 = p.min_distance * p.min_distance;
        for &i in &cand {
            let (y, x) = (i / w, i % w);
            let (xc, yc) = (x / cell, y / cell);
            let x1 = xc.saturating_sub(1);
            let y1 = yc.saturating_sub(1);
            let x2 = (xc + 1).min(gw - 1);
            let y2 = (yc + 1).min(gh - 1);
            let mut good = true;
            'out: for yy in y1..=y2 {
                for xx in x1..=x2 {
                    for q in &grid[yy * gw + xx] {
                        let dx = x as f32 - q.0;
                        let dy = y as f32 - q.1;
                        if f64::from(dx * dx + dy * dy) < min_d2 {
                            good = false;
                            break 'out;
                        }
                    }
                }
            }
            if good {
                grid[yc * gw + xc].push((x as f32, y as f32));
                corners.push((x as f32, y as f32));
                if max_corners > 0 && corners.len() == max_corners {
                    break;
                }
            }
        }
    } else {
        for &i in &cand {
            corners.push(((i % w) as f32, (i / w) as f32));
            if max_corners > 0 && corners.len() == max_corners {
                break;
            }
        }
    }
    corners
}

/// `cv::getRectSubPix(image, Size(win_w, win_h), center, patch, CV_32F)` for an 8-bit image:
/// the window sampled bilinearly into `float`, edges replicated outside the image.
///
/// `image` may be any [`GreySource`]: the deep path of method 3 samples its float patch with
/// this same arithmetic (the module documentation says why), which is not what OpenCV's
/// float-to-float `getRectSubPix` computes inside the image
/// ([`crate::imgproc::subpix::get_rect_sub_pix_32f`]).
///
/// Ported from imgproc/src/samplers.cpp, `getRectSubPix_8u32f` (the window entirely inside
/// the image, which uses a running-weight scheme of its own) and
/// `getRectSubPix_Cn_<uchar, float, float, nop<float>, nop<float>>` (otherwise).
#[must_use]
pub fn get_rect_sub_pix_f32<S: GreySource + ?Sized>(
    image: &S,
    win_w: usize,
    win_h: usize,
    center: (f32, f32),
) -> Vec<f32> {
    let (w, h) = (image.width() as i64, image.height() as i64);
    let step = w;
    let px = |i: i64| image.value(usize::try_from(i).expect("read before image"));
    let mut dst = vec![0f32; win_w * win_h];
    let cx = center.0 - (win_w as i32 - 1) as f32 * 0.5;
    let cy = center.1 - (win_h as i32 - 1) as f32 * 0.5;
    let ipx = cx.floor() as i64;
    let ipy = cy.floor() as i64;
    let (ww, wh) = (win_w as i64, win_h as i64);
    if 0 <= ipx && ipx + ww < w && 0 <= ipy && ipy + wh < h && ww > 0 && wh > 0 {
        let mut a = cx - ipx as f32;
        let b = cy - ipy as f32;
        a = a.max(0.0001);
        let a12 = a * (1.0 - b);
        let a22 = a * b;
        let b1 = 1.0 - b;
        let b2 = b;
        let s = (1.0 - f64::from(a)) / f64::from(a);
        let mut src = ipy * step + ipx;
        for row in dst.chunks_mut(win_w) {
            let mut prev = (1.0 - a) * (b1 * px(src) + b2 * px(src + step));
            for (j, o) in row.iter_mut().enumerate() {
                let j = j as i64;
                let t = a12 * px(src + j + 1) + a22 * px(src + j + 1 + step);
                *o = prev + t;
                prev = (f64::from(t) * s) as f32;
            }
            src += step;
        }
        return dst;
    }
    // Generic code with float weights: the window is not entirely inside the image.
    let a = cx - ipx as f32;
    let b = cy - ipy as f32;
    let a11 = (1.0 - a) * (1.0 - b);
    let a12 = a * (1.0 - b);
    let a21 = (1.0 - a) * b;
    let a22 = a * b;
    let b1 = 1.0 - b;
    let b2 = b;
    // The window reaches outside the image (or is empty): adjustRect and edge replication.
    // adjustRect
    let mut src: i64 = 0;
    let rx = if ipx >= 0 {
        src += ipx;
        0
    } else {
        (-ipx).min(ww)
    };
    let rw = if ipx < w - ww {
        ww
    } else {
        let mut rw = w - ipx - 1;
        if rw < 0 {
            src += rw;
            rw = 0;
        }
        rw
    };
    let ry = if ipy >= 0 {
        src += ipy * step;
        0
    } else {
        -ipy
    };
    let rh = if ipy < h - wh {
        wh
    } else {
        let mut rh = h - ipy - 1;
        if rh < 0 {
            src += rh * step;
            rh = 0;
        }
        rh
    };
    let mut src = src - rx;
    for i in 0..wh {
        let mut src2 = src + step;
        if i < ry || i >= rh {
            src2 -= step;
        }
        let row = &mut dst[(i as usize) * win_w..][..win_w];
        let s0 = px(src + rx) * b1 + px(src2 + rx) * b2;
        for o in &mut row[..rx as usize] {
            *o = s0;
        }
        if rw < ww {
            let s0 = px(src + rw) * b1 + px(src2 + rw) * b2;
            for o in &mut row[rw as usize..] {
                *o = s0;
            }
        }
        for j in rx..rw {
            row[j as usize] = px(src + j) * a11
                + px(src + j + 1) * a12
                + px(src2 + j) * a21
                + px(src2 + j + 1) * a22;
        }
        if i < rh {
            src = src2;
        }
    }
    dst
}

/// `cv::cornerSubPix(image, corners, win, zeroZone, criteria)` on an 8-bit image: moves each
/// corner to where the image gradients in a Gaussian-weighted `(2 win + 1)` window are
/// orthogonal to the vectors from the corner, iterating until the step is at most `epsilon`
/// or `max_iter` steps (at most 100) have run. A step that leaves the image, or a singular
/// system, stops the iteration; a result further than `win` from the start returns the start.
///
/// An empty `corners` returns at once. (In OpenCV an empty `std::vector` fails the
/// `count >= 0` assertion and throws: XMALab's method 3 throws there when no corner is found.)
///
/// Ported from imgproc/src/cornersubpix.cpp, `cv::cornerSubPix`, with
/// [`get_rect_sub_pix_f32`].
///
/// # Errors
/// What OpenCV asserts: positive window, an image of at least `2 win + 5` in each direction,
/// every corner inside the image. Corners before the failing one are already refined.
pub fn corner_sub_pix<S: GreySource + ?Sized>(
    image: &S,
    corners: &mut [(f32, f32)],
    win: (i32, i32),
    zero_zone: (i32, i32),
    criteria: TermCriteria,
) -> Result<(), CornerSubPixError> {
    const MAX_ITERS: i32 = 100;
    let win_w = (win.0 * 2 + 1) as usize;
    let win_h = (win.1 * 2 + 1) as usize;
    let max_iters = criteria
        .max_iter
        .map_or(MAX_ITERS, |m| m.clamp(1, MAX_ITERS));
    let eps = criteria.epsilon.map_or(0.0, |e| e.max(0.0));
    let eps = eps * eps;
    if corners.is_empty() {
        return Ok(());
    }
    if win.0 <= 0 || win.1 <= 0 {
        return Err(CornerSubPixError::Window);
    }
    let (cols, rows) = (image.width() as i64, image.height() as i64);
    if cols < i64::from(win.0) * 2 + 5 || rows < i64::from(win.1) * 2 + 5 {
        return Err(CornerSubPixError::ImageTooSmall);
    }
    let mut mask = vec![0f32; win_w * win_h];
    for i in 0..win_h {
        let y = (i as i32 - win.1) as f32 / win.1 as f32;
        let vy = (-y * y).exp();
        for j in 0..win_w {
            let x = (j as i32 - win.0) as f32 / win.0 as f32;
            mask[i * win_w + j] = vy * (-x * x).exp();
        }
    }
    if zero_zone.0 >= 0
        && zero_zone.1 >= 0
        && ((zero_zone.0 * 2 + 1) as usize) < win_w
        && ((zero_zone.1 * 2 + 1) as usize) < win_h
    {
        for i in win.1 - zero_zone.1..=win.1 + zero_zone.1 {
            for j in win.0 - zero_zone.0..=win.0 + zero_zone.0 {
                mask[i as usize * win_w + j as usize] = 0.0;
            }
        }
    }
    let contains = |p: (f32, f32)| {
        let (x, y) = (f64::from(p.0), f64::from(p.1));
        0.0 <= x && x < cols as f64 && 0.0 <= y && y < rows as f64
    };
    let bw = win_w + 2;
    for corner in corners.iter_mut() {
        let ct = *corner;
        if !contains(ct) {
            return Err(CornerSubPixError::CornerOutside);
        }
        let mut ci = ct;
        let mut iter = 0;
        loop {
            let buf = get_rect_sub_pix_f32(image, bw, win_h + 2, ci);
            let (mut a, mut b, mut c, mut bb1, mut bb2) = (0f64, 0f64, 0f64, 0f64, 0f64);
            let mut k = 0;
            for i in 0..win_h {
                let py = i as f64 - f64::from(win.1);
                let base = (i + 1) * bw + 1;
                for j in 0..win_w {
                    let m = f64::from(mask[k]);
                    let p = base + j;
                    let tgx = f64::from(buf[p + 1] - buf[p - 1]);
                    let tgy = f64::from(buf[p + bw] - buf[p - bw]);
                    let gxx = tgx * tgx * m;
                    let gxy = tgx * tgy * m;
                    let gyy = tgy * tgy * m;
                    let px = j as f64 - f64::from(win.0);
                    a += gxx;
                    b += gxy;
                    c += gyy;
                    bb1 += gxx * px + gxy * py;
                    bb2 += gxy * px + gyy * py;
                    k += 1;
                }
            }
            let det = a * c - b * b;
            if det.abs() <= f64::EPSILON * f64::EPSILON {
                break;
            }
            let scale = 1.0 / det;
            let x2 = (f64::from(ci.0) + c * scale * bb1 - b * scale * bb2) as f32;
            let y2 = (f64::from(ci.1) - b * scale * bb1 + a * scale * bb2) as f32;
            let (ex, ey) = (x2 - ci.0, y2 - ci.1);
            let err = f64::from(ex * ex + ey * ey);
            if !contains((x2, y2)) {
                break;
            }
            ci = (x2, y2);
            iter += 1;
            if !(iter < max_iters && err > eps) {
                break;
            }
        }
        if (ci.0 - ct.0).abs() > win.0 as f32 || (ci.1 - ct.1).abs() > win.1 as f32 {
            ci = ct;
        }
        *corner = ci;
    }
    Ok(())
}

/// The corners XMALab's method 3 refines: [`good_features_to_track`] with `params`, then
/// [`corner_sub_pix`] with half window 7, no zero zone, 100 iterations, epsilon 0.001.
/// `None` where the C++ throws: no corner found (cornerSubPix's assertion on the empty
/// vector) or a patch smaller than 19 pixels with corners in it.
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint`
/// (lines 317-329).
#[must_use]
pub fn method3_corners(patch: &ImageU8, params: &GfttParams) -> Option<Vec<(f32, f32)>> {
    method3_corners_impl(patch, params, true)
}

/// The body of [`method3_corners`] on any [`GreySource`] (see [`covariance`] for
/// `scale_255`).
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint`
/// (lines 317-329).
fn method3_corners_impl<S: GreySource + ?Sized>(
    patch: &S,
    params: &GfttParams,
    scale_255: bool,
) -> Option<Vec<(f32, f32)>> {
    let mut corners = gftt(patch, params, scale_255);
    if corners.is_empty() {
        return None;
    }
    let criteria = TermCriteria {
        max_iter: Some(100),
        epsilon: Some(0.001),
    };
    corner_sub_pix(patch, &mut corners, (7, 7), (-1, -1), criteria).ok()?;
    Some(corners)
}

/// Detection method 3 (corner) on the cut patch: [`method3_corners`] with
/// [`XMALAB_METHOD3`], then the corner nearest the search centre (the seed, or XMALab 3's
/// patch index `searchArea + 1`, [`SelectionCompat::search_centre_plus_one`]) within
/// `searchArea` (strictly closer; the first wins a tie), with size 15 (`half_win_size * 2 + 1`). The
/// C++ fills no candidate list for this method, so `all_centres` and `all_sizes` stay empty.
/// Nothing found gives `center` and size -1.
///
/// The C++ loop visits 50 entries of a vector that usually holds fewer and reads past its
/// end (parity ledger A18); here only the corners found are visited. Where the C++ throws (no
/// corner in the patch; a patch under 19 pixels), nothing is found.
///
/// `patch` is `getSubImage(subimage, searchArea, off_x, off_y)`.
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint` (the branch
/// `method == 3`, lines 315-346).
#[must_use]
pub fn detect_corner_marker(
    patch: &ImageU8,
    search_area: i32,
    off_x: i32,
    off_y: i32,
    center: (f64, f64),
    compat: SelectionCompat,
) -> Detection {
    detect_corner_marker_with(
        patch,
        search_area,
        off_x,
        off_y,
        center,
        &XMALAB_METHOD3,
        compat,
    )
}

/// [`detect_corner_marker`] with other detector parameters (for example the Harris detector
/// with aperture 3 that the C++ source names but does not run).
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint` (lines
/// 315-346).
#[must_use]
pub fn detect_corner_marker_with(
    patch: &ImageU8,
    search_area: i32,
    off_x: i32,
    off_y: i32,
    center: (f64, f64),
    params: &GfttParams,
    compat: SelectionCompat,
) -> Detection {
    corner_marker(patch, search_area, off_x, off_y, center, params, compat)
}

/// [`detect_corner_marker`] on the patch of a deeper image, as `f32` on the 8-bit scale
/// (parity ledger V1): the 8-bit path's arithmetic on those values (see the module
/// documentation), so an 8-bit patch promoted to a deeper image gives the 8-bit result
/// exactly.
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint` (the branch
/// `method == 3`; extends [`detect_corner_marker`]).
#[must_use]
pub fn detect_corner_marker_f32(
    patch: &ImageF32,
    search_area: i32,
    off_x: i32,
    off_y: i32,
    center: (f64, f64),
    compat: SelectionCompat,
) -> Detection {
    corner_marker(
        patch,
        search_area,
        off_x,
        off_y,
        center,
        &XMALAB_METHOD3,
        compat,
    )
}

/// The body of [`detect_corner_marker_with`] and [`detect_corner_marker_f32`]: the
/// derivative scale is the 8-bit one (`scale_255`) for both.
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint` (lines
/// 315-346).
fn corner_marker<S: GreySource + ?Sized>(
    patch: &S,
    search_area: i32,
    off_x: i32,
    off_y: i32,
    center: (f64, f64),
    params: &GfttParams,
    compat: SelectionCompat,
) -> Detection {
    let mut out = Detection {
        point: center,
        size: -1.0,
        all_centres: Vec::new(),
        all_sizes: Vec::new(),
    };
    let Some(corners) = method3_corners_impl(patch, params, true) else {
        return out;
    };
    let sc = compat.search_centre(search_area, center, (off_x, off_y));
    let mut dist_min_sq = f64::from(search_area) * f64::from(search_area);
    for c in &corners {
        let dx = f64::from(c.0) - sc.0;
        let dy = f64::from(c.1) - sc.1;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < dist_min_sq {
            out.point = (compat.place(off_x, c.0), compat.place(off_y, c.1));
            out.size = 15.0;
            dist_min_sq = dist_sq;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobel_kernels_scale_smoothing_side() {
        let (kx, ky) = sobel_kernels(true, 1, 0.5);
        assert_eq!((kx, ky), (vec![-1.0, 0.0, 1.0], vec![0.5]));
        let (kx, ky) = sobel_kernels(false, 3, 1.0);
        assert_eq!((kx, ky), (vec![1.0, 2.0, 1.0], vec![-1.0, 0.0, 1.0]));
    }

    #[test]
    fn box_sum_of_ones() {
        let s = box_sum(&[1.0; 12 * 9], 12, 9, 7);
        assert!(s.iter().all(|&v| v == 49.0));
    }

    #[test]
    fn no_corners_on_flat_patch() {
        let flat = ImageU8::filled(21, 21, 100);
        assert!(good_features_to_track(&flat, &XMALAB_METHOD3).is_empty());
        let d = detect_corner_marker(&flat, 10, 3, 4, (13.5, 14.5), SelectionCompat::XMALAB3);
        assert_eq!(d.point, (13.5, 14.5));
        assert_eq!(d.size, -1.0);
    }

    #[test]
    fn sub_pix_argument_checks() {
        let im = ImageU8::filled(18, 30, 7);
        let crit = TermCriteria {
            max_iter: Some(10),
            epsilon: None,
        };
        let mut c = [(5.0, 5.0)];
        assert_eq!(
            corner_sub_pix(&im, &mut c, (7, 7), (-1, -1), crit),
            Err(CornerSubPixError::ImageTooSmall)
        );
        let im = ImageU8::filled(19, 19, 7);
        let mut c = [(19.0, 5.0)];
        assert_eq!(
            corner_sub_pix(&im, &mut c, (7, 7), (-1, -1), crit),
            Err(CornerSubPixError::CornerOutside)
        );
        assert_eq!(corner_sub_pix(&im, &mut [], (7, 7), (-1, -1), crit), Ok(()));
        // A flat window is singular: the corner stays.
        let mut c = [(9.5, 9.25)];
        corner_sub_pix(&im, &mut c, (7, 7), (-1, -1), crit).unwrap();
        assert_eq!(c, [(9.5, 9.25)]);
    }
}
