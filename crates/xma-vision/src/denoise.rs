//! Non-local means denoising of an image sequence: `cv::fastNlMeansDenoisingMulti` for
//! 8-bit single-channel images, as XMALab's "Denoise Images" calls it
//! (`processing/DenoiseTrial.cpp`).
//!
//! Ported from OpenCV 4.13.0 `modules/photo/src/denoising.cpp`
//! (`fastNlMeansDenoisingMulti`, `fastNlMeansDenoisingMultiCheckPreconditions`,
//! `fastNlMeansDenoisingMulti_` with `uchar`, `int`, `unsigned`, `DistSquared` and one
//! filter strength), `fast_nlmeans_multi_denoising_invoker.hpp`
//! (`FastNlMeansMultiDenoisingInvoker`) and `fast_nlmeans_denoising_invoker_commons.hpp`
//! (`DistSquared::calcWeight`, `incWithWeight`, `divByWeightsSum`), with
//! `modules/core/src/copy.cpp` (`copyMakeBorder` with `BORDER_DEFAULT`, that is
//! `BORDER_REFLECT_101`, and `borderInterpolate`).
//!
//! What OpenCV computes, for the frame `index` of `frames`, with `T` the temporal window,
//! `S` the search window and `P` the template window (each odd), `t = T / 2`, `s = S / 2`,
//! `p = P / 2`:
//!
//! 1. The frames `index - t ..= index + t` are extended by `s + p` pixels on every side by
//!    reflection about the edge pixel (`BORDER_REFLECT_101`, repeated when the border is
//!    longer than the image).
//! 2. For each pixel and each of the `T × S × S` candidate pixels (the same position offset
//!    by `-s ..= s` in each direction, in each frame of the window) the distance is the sum
//!    of squared differences of the two `P × P` patches, an exact integer.
//! 3. The distance is divided by `P²` with a shift: `dist >> ceil(log2(P²))`, and looked up
//!    in a table of weights made for every such value `a`:
//!    `w = cvRound(m * exp(-a * 2^shift / P² / (h * h)))` (`h * h` in single precision),
//!    `w = 1` where the exponent is NaN (`h = 0` with a zero distance), and 0 when below
//!    `0.001 * m`, with the fixed-point multiplier `m = INT_MAX / (T * S * S * 255)`.
//! 4. The result is `(Σ w * pixel + Σ w / 2) / Σ w` in unsigned integers, saturated to
//!    8 bits.
//!
//! OpenCV visits the pixels row by row and keeps running column sums of the distances
//! between rows (and restarts them at the first row of each block of rows it gives a
//! thread); every sum is exact in integers and none overflows within the limits checked
//! here, so the order of the additions does not matter. This port computes the same sums
//! candidate by candidate over a block of rows, with a sliding window over the template,
//! which needs working memory proportional to the image (besides the frames extended by
//! `s + p` on each side, as OpenCV extends them), and runs the
//! blocks of rows on threads: the bytes are OpenCV's whatever the number of threads
//! (`crates/xma-vision/tests/denoise.rs`, against OpenCV 4.13.0 on the cases of
//! `oracle/data/primitives/denoise`).
//!
//! Where OpenCV refuses (it throws `cv::Exception`), [`fast_nl_means_denoising_multi`]
//! returns a [`DenoiseError`]: no frames, an even window size, a temporal window that
//! reaches past either end of the sequence, frames of different sizes. It also refuses what
//! OpenCV computes with signed overflow or a division by zero (undefined behaviour in C++):
//! a window size of zero or below, `T × S × S × 255` above `INT_MAX` (OpenCV's multiplier
//! is then 0 and every weight 0) and `P² × 255²` above `INT_MAX` (a distance overflows).

use std::fmt;

/// The parameters of `fastNlMeansDenoisingMulti`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseParams {
    /// `temporalWindowSize`: frames used, centred on the frame denoised (odd).
    pub temporal: i32,
    /// `h`: the filter strength.
    pub h: f32,
    /// `templateWindowSize`: the side of the patches compared (odd).
    pub template: i32,
    /// `searchWindowSize`: the side of the area searched in each frame (odd).
    pub search: i32,
}

impl Default for DenoiseParams {
    /// XMALab's defaults (`DenoiseDialog.ui`): temporal 5, strength 5, template 11,
    /// search 35.
    fn default() -> Self {
        DenoiseParams {
            temporal: 5,
            h: 5.0,
            template: 11,
            search: 35,
        }
    }
}

/// Why nothing was denoised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DenoiseError {
    /// "Input images vector should not be empty!"
    NoImages,
    /// "All windows sizes should be odd!"
    EvenWindow,
    /// A window size of zero or below (undefined behaviour in OpenCV for a negative odd
    /// size; zero is even).
    WindowNotPositive,
    /// "imgToDenoiseIndex and temporalWindowSize should be chosen corresponding srcImgs
    /// size!": the window `index - T/2 ..= index + T/2` is not inside `0..n`.
    WindowOutOfRange {
        index: usize,
        temporal: i32,
        frames: usize,
    },
    /// "Input images should have the same size and type!" (or a frame whose length is not
    /// `width × height`).
    SizeMismatch,
    /// The window sizes are too large for OpenCV's integer arithmetic.
    TooLarge,
}

impl fmt::Display for DenoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenoiseError::NoImages => write!(f, "Input images vector should not be empty!"),
            DenoiseError::EvenWindow => write!(f, "All windows sizes should be odd!"),
            DenoiseError::WindowNotPositive => write!(f, "window sizes must be at least 1"),
            DenoiseError::WindowOutOfRange {
                index,
                temporal,
                frames,
            } => write!(
                f,
                "imgToDenoiseIndex and temporalWindowSize should be chosen corresponding \
                 srcImgs size! (frame {index}, temporal window {temporal}, {frames} frames)"
            ),
            DenoiseError::SizeMismatch => {
                write!(f, "Input images should have the same size and type!")
            }
            DenoiseError::TooLarge => write!(
                f,
                "the window sizes are too large (temporal × search² × 255 and template² × \
                 255² must stay below 2^31)"
            ),
        }
    }
}

impl std::error::Error for DenoiseError {}

/// `cv::borderInterpolate(p, len, BORDER_REFLECT_101)`.
///
/// Ported from OpenCV 4.13.0 modules/core/src/copy.cpp, cv::borderInterpolate.
///
/// The reflections repeat with period 2 (len - 1); taken modulo it, so any `p` (and any
/// `len`) is answered at once, with the value OpenCV's loop reaches
/// ([`crate::imgproc::blur::reflect101`] for an `isize`).
#[must_use]
pub fn reflect101(p: i64, len: usize) -> usize {
    crate::imgproc::blur::reflect101_wide(i128::from(p), len)
}

/// The checked parameters and the tables both the single- and multi-threaded paths use.
struct Plan {
    t_half: usize,
    s_half: usize,
    p_half: usize,
    border: usize,
    shift: u32,
    weights: Vec<u32>,
}

/// Checks the arguments as `fastNlMeansDenoisingMultiCheckPreconditions` does (and the
/// limits of the module documentation), and makes the weight table of the invoker's
/// constructor.
fn plan(
    frames: &[&[u8]],
    width: usize,
    height: usize,
    index: usize,
    p: &DenoiseParams,
) -> Result<Plan, DenoiseError> {
    if frames.is_empty() {
        return Err(DenoiseError::NoImages);
    }
    if p.temporal % 2 == 0 || p.search % 2 == 0 || p.template % 2 == 0 {
        return Err(DenoiseError::EvenWindow);
    }
    if p.temporal < 1 || p.search < 1 || p.template < 1 {
        return Err(DenoiseError::WindowNotPositive);
    }
    let t_half = (p.temporal / 2) as usize;
    if index < t_half || index.checked_add(t_half).is_none_or(|e| e >= frames.len()) {
        return Err(DenoiseError::WindowOutOfRange {
            index,
            temporal: p.temporal,
            frames: frames.len(),
        });
    }
    let n = width
        .checked_mul(height)
        .ok_or(DenoiseError::SizeMismatch)?;
    if frames.iter().any(|f| f.len() != n) {
        return Err(DenoiseError::SizeMismatch);
    }
    // `max_estimate_sum_value` and the template's largest distance, in `int`.
    let (t, s, pt) = (
        i64::from(p.temporal),
        i64::from(p.search),
        i64::from(p.template),
    );
    let max_dist: i64 = 255 * 255;
    let limit = i64::from(i32::MAX);
    let max_estimate = t
        .checked_mul(s)
        .and_then(|v| v.checked_mul(s))
        .and_then(|v| v.checked_mul(255))
        .filter(|&v| v <= limit)
        .ok_or(DenoiseError::TooLarge)?;
    if pt
        .checked_mul(pt)
        .and_then(|v| v.checked_mul(max_dist))
        .is_none_or(|v| v > limit)
    {
        return Err(DenoiseError::TooLarge);
    }
    let fpm = i64::from(i32::MAX) / max_estimate;
    let s_half = (p.search / 2) as usize;
    let p_half = (p.template / 2) as usize;
    let template_sq = pt * pt;
    let mut shift = 0u32;
    while (1i64 << shift) < template_sq {
        shift += 1;
    }
    let mult = (1i64 << shift) as f64 / template_sq as f64;
    let almost_max = (max_dist as f64 / mult + 1.0) as usize;
    // h[0] * h[0] * pixelInfo<uchar>::channels, in float.
    let hh = f64::from(p.h * p.h * 1.0f32);
    let weights = (0..almost_max)
        .map(|a| {
            let dist = a as f64 * mult;
            let mut w = (-dist / hh).exp();
            if w.is_nan() {
                w = 1.0;
            }
            // cvRound: nearest, ties to even.
            let weight = (fpm as f64 * w).round_ties_even() as i64;
            if (weight as f64) < 0.001 * fpm as f64 {
                0
            } else {
                weight as u32
            }
        })
        .collect();
    Ok(Plan {
        t_half,
        s_half,
        p_half,
        border: s_half + p_half,
        shift,
        weights,
    })
}

/// `copyMakeBorder(src, dst, b, b, b, b, BORDER_REFLECT_101)` of a `width × height` image.
fn extend(src: &[u8], width: usize, height: usize, b: usize) -> Vec<u8> {
    let (ew, eh) = (width + 2 * b, height + 2 * b);
    let cols: Vec<usize> = (0..ew)
        .map(|c| reflect101(c as i64 - b as i64, width))
        .collect();
    let mut out = vec![0u8; ew * eh];
    for (r, row) in out.chunks_exact_mut(ew).enumerate() {
        let sr = reflect101(r as i64 - b as i64, height);
        let srow = &src[sr * width..(sr + 1) * width];
        for (o, &c) in row.iter_mut().zip(&cols) {
            *o = srow[c];
        }
    }
    out
}

/// Denoises frame `index` of `frames` (each `width × height` 8-bit grey pixels, row by
/// row), as `cv::fastNlMeansDenoisingMulti(frames, dst, index, temporal, h, template,
/// search)`, with the work spread over the available processors.
///
/// Ported from OpenCV 4.13.0 modules/photo/src/denoising.cpp, cv::fastNlMeansDenoisingMulti
/// (see the module documentation).
///
/// # Errors
/// See [`DenoiseError`]; nothing is computed then.
pub fn fast_nl_means_denoising_multi(
    frames: &[&[u8]],
    width: usize,
    height: usize,
    index: usize,
    params: &DenoiseParams,
) -> Result<Vec<u8>, DenoiseError> {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    fast_nl_means_denoising_multi_threads(frames, width, height, index, params, threads)
}

/// [`fast_nl_means_denoising_multi`] on at most `threads` threads (0 or 1: the calling
/// thread only). The result does not depend on the number of threads.
///
/// # Errors
/// See [`DenoiseError`].
pub fn fast_nl_means_denoising_multi_threads(
    frames: &[&[u8]],
    width: usize,
    height: usize,
    index: usize,
    params: &DenoiseParams,
    threads: usize,
) -> Result<Vec<u8>, DenoiseError> {
    let plan = plan(frames, width, height, index, params)?;
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let b = plan.border;
    let ext: Vec<Vec<u8>> = (index - plan.t_half..=index + plan.t_half)
        .map(|i| extend(frames[i], width, height, b))
        .collect();
    let mut dst = vec![0u8; width * height];
    let threads = threads.clamp(1, height);
    let per = height.div_ceil(threads);
    if threads == 1 {
        denoise_rows(&plan, &ext, width, height, 0, &mut dst);
    } else {
        std::thread::scope(|scope| {
            for (k, chunk) in dst.chunks_mut(per * width).enumerate() {
                let (plan, ext) = (&plan, &ext);
                // A thread the system refuses: that block on this thread.
                let cell = std::sync::Arc::new(std::sync::Mutex::new(Some(chunk)));
                let mine = std::sync::Arc::clone(&cell);
                let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                    let c = mine.lock().ok().and_then(|mut g| g.take());
                    if let Some(c) = c {
                        denoise_rows(plan, ext, width, height, k * per, c);
                    }
                });
                if spawned.is_err() {
                    let c = cell.lock().ok().and_then(|mut g| g.take());
                    if let Some(c) = c {
                        denoise_rows(plan, ext, width, height, k * per, c);
                    }
                }
            }
        });
    }
    Ok(dst)
}

/// The rows `row0 .. row0 + out.len() / width` of the result, into `out`.
///
/// Ported from fast_nlmeans_multi_denoising_invoker.hpp,
/// FastNlMeansMultiDenoisingInvoker::operator() (the same sums, candidate by candidate).
fn denoise_rows(
    plan: &Plan,
    ext: &[Vec<u8>],
    width: usize,
    height: usize,
    row0: usize,
    out: &mut [u8],
) {
    let rows = out.len() / width;
    if rows == 0 {
        return;
    }
    let _ = height;
    let b = plan.border;
    let ew = width + 2 * b;
    let (sh, ph) = (plan.s_half, plan.p_half);
    let s = 2 * sh + 1;
    let main = &ext[plan.t_half];
    let mut est = vec![0u32; rows * width];
    let mut wsum = vec![0u32; rows * width];
    // Column sums of the squared differences over the template's rows, for the columns
    // b - p ..= b + width - 1 + p of the extended image.
    let cw = width + 2 * ph;
    let mut col = vec![0i32; cw];
    let c0 = b - ph;
    let weights = &plan.weights;
    let last = weights.len().saturating_sub(1);
    for frame in ext {
        for y in 0..s {
            for x in 0..s {
                // Candidate offset (y - s/2, x - s/2) in this frame: the template row `r`
                // of the main frame (extended coordinates, columns from `c0`) and the
                // matching row of the candidate.
                let rows_at = |r: usize| -> (&[u8], &[u8]) {
                    (
                        &main[r * ew + c0..r * ew + c0 + cw],
                        &frame[(r + y - sh) * ew + c0 + x - sh..][..cw],
                    )
                };
                // Rows of the template around the first row of the block.
                let r_first = b + row0 - ph;
                col.iter_mut().for_each(|v| *v = 0);
                for r in r_first..=r_first + 2 * ph {
                    let (a, c) = rows_at(r);
                    for ((v, &a), &c) in col.iter_mut().zip(a).zip(c) {
                        let d = i32::from(a) - i32::from(c);
                        *v += d * d;
                    }
                }
                for i in 0..rows {
                    if i > 0 {
                        let (ai, ci) = rows_at(b + row0 + i + ph);
                        let (ao, co) = rows_at(b + row0 + i - 1 - ph);
                        for ((((v, &ai), &ci), &ao), &co) in
                            col.iter_mut().zip(ai).zip(ci).zip(ao).zip(co)
                        {
                            let din = i32::from(ai) - i32::from(ci);
                            let dout = i32::from(ao) - i32::from(co);
                            *v += din * din - dout * dout;
                        }
                    }
                    let mut dist: i32 = col[..=2 * ph].iter().sum();
                    let prow = &frame[(b + row0 + i + y - sh) * ew + b + x - sh..];
                    let e = &mut est[i * width..(i + 1) * width];
                    let ws = &mut wsum[i * width..(i + 1) * width];
                    for j in 0..width {
                        if j > 0 {
                            dist += col[j + 2 * ph] - col[j - 1];
                        }
                        let a = (dist >> plan.shift) as usize;
                        let w = weights[a.min(last)];
                        e[j] += w * u32::from(prow[j]);
                        ws[j] += w;
                    }
                }
            }
        }
    }
    for ((o, &e), &w) in out.iter_mut().zip(&est).zip(&wsum) {
        // divByWeightsSum in unsigned, then saturate_cast<uchar>(int). The centre candidate
        // has distance 0 and weight m >= 1, so the sum of weights is never 0.
        let v = (e + w / 2) / w.max(1);
        *o = v.min(255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_101_as_border_interpolate() {
        assert_eq!(reflect101(-1, 5), 1);
        assert_eq!(reflect101(-4, 5), 4);
        assert_eq!(reflect101(-5, 5), 3);
        assert_eq!(reflect101(5, 5), 3);
        assert_eq!(reflect101(9, 5), 1);
        assert_eq!(reflect101(-7, 2), 1);
        assert_eq!(reflect101(17, 1), 0);
        assert_eq!(reflect101(3, 5), 3);
        assert_eq!(reflect101(i64::MIN, 2), 0);
        assert_eq!(reflect101(i64::MAX, 2), 1);
        assert_eq!(reflect101(i64::MAX, 7), 5);
        // Against OpenCV's loop, for every small case.
        for len in 2..9usize {
            for p in -40i64..40 {
                let mut q = p;
                let l = len as i64;
                while q < 0 || q >= l {
                    q = if q < 0 { -q } else { 2 * l - 2 - q };
                }
                assert_eq!(reflect101(p, len), q as usize, "{p} {len}");
            }
        }
    }

    #[test]
    fn refusals() {
        let f = vec![0u8; 12];
        let fr: Vec<&[u8]> = vec![&f; 5];
        let p = DenoiseParams::default();
        let run =
            |fr: &[&[u8]], i, p: &DenoiseParams| fast_nl_means_denoising_multi(fr, 4, 3, i, p);
        assert_eq!(run(&[], 0, &p), Err(DenoiseError::NoImages));
        for bad in [
            DenoiseParams { template: 10, ..p },
            DenoiseParams { search: 0, ..p },
            DenoiseParams { temporal: 4, ..p },
        ] {
            assert_eq!(run(&fr, 2, &bad), Err(DenoiseError::EvenWindow));
        }
        assert_eq!(
            run(&fr, 2, &DenoiseParams { template: -3, ..p }),
            Err(DenoiseError::WindowNotPositive)
        );
        assert!(matches!(
            run(&fr, 3, &p),
            Err(DenoiseError::WindowOutOfRange { .. })
        ));
        assert!(matches!(
            run(&fr, 1, &p),
            Err(DenoiseError::WindowOutOfRange { .. })
        ));
        assert!(matches!(
            run(&fr, 0, &DenoiseParams { temporal: 99, ..p }),
            Err(DenoiseError::WindowOutOfRange { .. })
        ));
        let short = vec![0u8; 11];
        let mixed: Vec<&[u8]> = vec![&f, &f, &short, &f, &f];
        assert_eq!(run(&mixed, 2, &p), Err(DenoiseError::SizeMismatch));
        assert_eq!(
            run(&fr, 2, &DenoiseParams { search: 30001, ..p }),
            Err(DenoiseError::TooLarge)
        );
        assert_eq!(
            run(&fr, 2, &DenoiseParams { template: 183, ..p }),
            Err(DenoiseError::TooLarge)
        );
        let extreme = DenoiseParams {
            temporal: i32::MAX,
            search: i32::MAX,
            template: i32::MIN + 1,
            h: f32::NAN,
        };
        assert!(run(&fr, 2, &extreme).is_err());
    }

    #[test]
    fn flat_frames_stay_flat_with_any_strength() {
        let f = vec![93u8; 7 * 6];
        let fr: Vec<&[u8]> = vec![&f; 3];
        for h in [0.0, 1e-30, 5.0, f32::INFINITY, f32::NAN, -4.0] {
            let p = DenoiseParams {
                temporal: 3,
                h,
                template: 3,
                search: 5,
            };
            let out = fast_nl_means_denoising_multi(&fr, 7, 6, 1, &p).unwrap();
            assert!(out.iter().all(|&v| v == 93), "h {h}");
        }
    }

    #[test]
    fn empty_image() {
        let f: Vec<u8> = Vec::new();
        let fr: Vec<&[u8]> = vec![&f; 1];
        let p = DenoiseParams {
            temporal: 1,
            ..DenoiseParams::default()
        };
        assert_eq!(
            fast_nl_means_denoising_multi(&fr, 0, 5, 0, &p),
            Ok(Vec::new())
        );
    }
}
