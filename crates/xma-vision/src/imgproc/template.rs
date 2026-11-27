//! `cv::matchTemplate` with `TM_CCORR_NORMED` for 8-bit and 32-bit float images and
//! templates, the correlation the 2D marker tracker runs (`processing/MarkerTracking.cpp`;
//! float for sources deeper than 8 bits, parity ledger V1).
//!
//! OpenCV's CPU code computes the correlation through its own discrete Fourier transform
//! (`crossCorr`): single precision for 8-bit inputs, double precision for float inputs
//! (`maxDepth` is `CV_64F` for any depth above `CV_8S`), the result rounded to `float`. It
//! then divides by the norms from an integral image of squares and the template's
//! `meanStdDev` (`common_matchTemplate`). Both are ported operation for operation ([`dxt`]
//! holds the transform), so the `f32` result is OpenCV's to the bit. Tested against
//! `matchtemplate-*` and `tracking-step-*` in the primitive oracle (the `.noipp` outputs;
//! the C++ reference build has no IPP) and, for float input, `matchtemplate-fulldepth-*`.

mod dxt;

use crate::image::{Image, ImageF32, ImageU8};

use dxt::{Dft2dOf, Real, mul_spectrums_conj, optimal_dft_size};

/// An element type `cv::matchTemplate` accepts, with the precision `crossCorr` transforms it
/// in (`maxDepth`).
pub(crate) trait TemplatePixel: Copy {
    /// `f32` for `CV_8U`, `f64` for `CV_32F`.
    type Work: Real;
    /// `convertTo(maxDepth)`: exact.
    fn work(self) -> Self::Work;
    /// The value as `double` (exact), as the integral image and `meanStdDev` read it.
    fn to_f64(self) -> f64;
}

impl TemplatePixel for u8 {
    type Work = f32;
    fn work(self) -> f32 {
        f32::from(self)
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl TemplatePixel for f32 {
    type Work = f64;
    fn work(self) -> f64 {
        f64::from(self)
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

/// `cv::matchTemplate(image, templ, result, TM_CCORR_NORMED)` for 8-bit single-channel
/// inputs: for each placement `(x, y)` of the template's top-left corner inside the image,
/// `sum(T * I) / sqrt(sum(T^2) * sum(I^2))` over the window, as `f32`. The result is
/// `(W - w + 1)` by `(H - h + 1)`. As in OpenCV, an image smaller than the template in both
/// directions is swapped with it.
///
/// A window or template with no energy gives 0; a quotient that rounding pushes up to
/// 1.125 is clamped to 1 (OpenCV's rules).
///
/// Ported from imgproc/src/templmatch.cpp, cv::matchTemplate (CPU path, no IPP, no mask),
/// crossCorr and common_matchTemplate.
///
/// # Panics
/// If either input is empty, or the image is smaller than the template in one direction
/// and larger in the other (OpenCV's assertion).
pub fn match_template_ccorr_normed(image: &ImageU8, templ: &ImageU8) -> ImageF32 {
    match_template_impl(image, templ)
}

/// [`match_template_ccorr_normed`] for 32-bit float image and template (OpenCV accepts
/// `CV_32F` for both; the tracker's deep path, parity ledger V1). The correlation is
/// transformed in double precision and rounded to `f32` (`crossCorr` with `maxDepth =
/// CV_64F`); the window sums of squares come from a `double` integral image and the
/// template's norm from `meanStdDev`, both summed in `double` in OpenCV's order.
///
/// Ported from imgproc/src/templmatch.cpp, cv::matchTemplate (CPU path, no IPP, no mask),
/// crossCorr and common_matchTemplate, for `CV_32F` inputs; it extends
/// [`match_template_ccorr_normed`].
///
/// # Panics
/// As [`match_template_ccorr_normed`].
#[must_use]
pub fn match_template_ccorr_normed_f32(image: &ImageF32, templ: &ImageF32) -> ImageF32 {
    match_template_impl(image, templ)
}

/// The body of both entry points.
///
/// Ported from imgproc/src/templmatch.cpp, cv::matchTemplate.
pub(crate) fn match_template_impl<P: TemplatePixel + Default>(
    image: &Image<P>,
    templ: &Image<P>,
) -> ImageF32 {
    assert!(
        !image.is_empty() && !templ.is_empty(),
        "matchTemplate of an empty image"
    );
    let needswap = image.height() < templ.height() || image.width() < templ.width();
    let (img, templ) = if needswap {
        assert!(
            image.height() <= templ.height() && image.width() <= templ.width(),
            "matchTemplate: the image must be larger than the template in both directions"
        );
        (templ, image)
    } else {
        (image, templ)
    };
    let mut result = cross_corr(img, templ);
    normalize_ccorr(img, templ, &mut result);
    result
}

/// The plain correlation `sum(T * I)` by blocks of OpenCV's DFT size, as `f32`.
///
/// Ported from imgproc/src/templmatch.cpp, crossCorr (one channel; single-precision
/// transforms for 8-bit inputs, double precision for float inputs, the result converted to
/// `float`; anchor (0, 0), delta 0, whole images, so no border is ever added).
fn cross_corr<P: TemplatePixel + Default>(img: &Image<P>, templ: &Image<P>) -> ImageF32 {
    const BLOCK_SCALE: f64 = 4.5;
    const MIN_BLOCK_SIZE: usize = 256;
    let (tw, th) = (templ.width(), templ.height());
    let corr_w = img.width() - tw + 1;
    let corr_h = img.height() - th + 1;
    let mut corr = ImageF32::new(corr_w, corr_h);

    let block = |t: usize, c: usize| {
        let b = (t as f64 * BLOCK_SCALE).round_ties_even() as usize;
        b.max((MIN_BLOCK_SIZE + 1).saturating_sub(t)).min(c)
    };
    let bw = block(tw, corr_w);
    let bh = block(th, corr_h);
    let dft_w = optimal_dft_size(bw + tw - 1)
        .expect("matchTemplate: the input arrays are too big")
        .max(2);
    let dft_h = optimal_dft_size(bh + th - 1).expect("matchTemplate: the input arrays are too big");
    // recompute block size
    let bw = (dft_w - tw + 1).min(corr_w);
    let bh = (dft_h - th + 1).min(corr_h);

    // DFT of the template, zero-padded to the right; rows below it are zeroed by the
    // transform (`nonzero_rows`).
    let zero = <P::Work as Real>::ZERO;
    let mut dft_templ = vec![zero; dft_w * dft_h];
    for y in 0..th {
        for (x, &v) in templ.row(y).iter().enumerate() {
            dft_templ[y * dft_w + x] = v.work();
        }
    }
    Dft2dOf::new(dft_w, dft_h, false, th).apply(&mut dft_templ);

    let forward = Dft2dOf::new(dft_w, dft_h, false, bh + th - 1);
    let inverse = Dft2dOf::new(dft_w, dft_h, true, bh);

    let tiles_x = corr_w.div_ceil(bw);
    let tiles_y = corr_h.div_ceil(bh);
    let mut dft_img = vec![zero; dft_w * dft_h];
    for i in 0..tiles_x * tiles_y {
        let x = (i % tiles_x) * bw;
        let y = (i / tiles_x) * bh;
        let bsz_w = bw.min(corr_w - x);
        let bsz_h = bh.min(corr_h - y);
        let dsz_w = bsz_w + tw - 1;
        let dsz_h = bsz_h + th - 1;
        dft_img.fill(zero);
        for r in 0..dsz_h {
            for (c, &v) in img.row(y + r)[x..x + dsz_w].iter().enumerate() {
                dft_img[r * dft_w + c] = v.work();
            }
        }
        // A last, shorter tile goes through `cv::dft` with `nonzero_rows = dsz.height`
        // (and `bsz.height` for the inverse): the same transform on the same non-zero rows.
        if bsz_h == bh {
            forward.apply(&mut dft_img);
        } else {
            Dft2dOf::new(dft_w, dft_h, false, dsz_h).apply(&mut dft_img);
        }
        mul_spectrums_conj(&mut dft_img, &dft_templ, dft_w, dft_h);
        if bsz_h == bh {
            inverse.apply(&mut dft_img);
        } else {
            Dft2dOf::new(dft_w, dft_h, true, bsz_h).apply(&mut dft_img);
        }
        // `src.convertTo(cdst, CV_32F)`: exact for `f32`, rounded to nearest for `f64`.
        for r in 0..bsz_h {
            for (d, &v) in corr.row_mut(y + r)[x..x + bsz_w]
                .iter_mut()
                .zip(&dft_img[r * dft_w..r * dft_w + bsz_w])
            {
                *d = v.to_f64() as f32;
            }
        }
    }
    corr
}

/// Divides the correlation by `sqrt(window sum I^2) * |T|`, with OpenCV's guards.
///
/// The sums are OpenCV's in `double` and in its order: `meanStdDev` adds the template's
/// values and squares one after the other (no vector code for these types reaches it:
/// 8-bit sums are integers, exact in `double`), the integral image of squares adds each
/// row's running sum to the row above, and a window's sum is `q0 - q1 - q2 + q3`.
///
/// Ported from imgproc/src/templmatch.cpp, common_matchTemplate (`TM_CCORR_NORMED`, one
/// channel), with core/src/mean.dispatch.cpp, cv::meanStdDev (`sumsqr_`), and
/// imgproc/src/sumpixels.dispatch.cpp, `integral_` (`CV_64F` sums).
fn normalize_ccorr<P: TemplatePixel + Default>(
    img: &Image<P>,
    templ: &Image<P>,
    result: &mut ImageF32,
) {
    let (tw, th) = (templ.width(), templ.height());
    let inv_area = 1.0 / (th as f64 * tw as f64);

    // meanStdDev: sums in double, then mean and standard deviation.
    let (mut s, mut sq) = (0f64, 0f64);
    for &v in templ.data() {
        let v = v.to_f64();
        s += v;
        sq += v * v;
    }
    let scale = 1.0 / (templ.data().len() as f64);
    let mean = s * scale;
    let sdv = (sq * scale - mean * mean).max(0.0).sqrt();
    // templNorm = sdv^2 (+ three zero channels); templSum2 = templNorm + mean^2.
    let templ_sum2 = sdv * sdv + mean * mean;
    let templ_norm = templ_sum2.sqrt() / inv_area.sqrt();

    // The integral image of squares, `CV_64F`.
    let (w, h) = (img.width(), img.height());
    let mut sqsum = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            let v = img.at(x, y).to_f64();
            row += v * v;
            sqsum[(y + 1) * (w + 1) + x + 1] = sqsum[y * (w + 1) + x + 1] + row;
        }
    }
    let at = |x: usize, y: usize| sqsum[y * (w + 1) + x];
    let eps = 10.0 * f64::from(f32::EPSILON);
    for i in 0..result.height() {
        let row = result.row_mut(i);
        for (j, r) in row.iter_mut().enumerate() {
            let mut num = f64::from(*r);
            let wnd_sum2 = at(j, i) - at(j + tw, i) - at(j, i + th) + at(j + tw, i + th);
            let diff2 = wnd_sum2.max(0.0);
            let t = if diff2 <= 0.5f64.min(eps * wnd_sum2) {
                0.0 // avoid rounding errors
            } else {
                diff2.sqrt() * templ_norm
            };
            if num.abs() < t {
                num /= t;
            } else if num.abs() < t * 1.125 {
                num = if num > 0.0 { 1.0 } else { -1.0 };
            } else {
                num = 0.0;
            }
            *r = num as f32;
        }
    }
}
