//! Checkerboard corner detection as XMALab's checkerboard calibration runs it
//! (`processing/CheckerboardDetection.cpp`, `CheckerboardDetection::detectCorner_thread`):
//! OpenCV's `cv::findChessboardCorners`, the `cv::cornerSubPix` refinement XMALab applies to
//! its result, and the manual mode that predicts every corner from four clicks.
//!
//! [`find_chessboard_corners`] is a port of OpenCV 4.13's `cv::findChessboardCorners`
//! (`modules/calib3d/src/calibinit.cpp`): the histogram-based binarisation and its dilation
//! loop, then the older adaptive-threshold loop, the quad search of `ChessBoardDetector`
//! (submodule `detector`), the monotony and border checks, the order of the corners and the
//! final `cornerSubPix(5 x 5, 15 iterations, 0.1)`. The image routines it reaches are in
//! [`crate::imgproc`] (`adaptive`, `morph`, `contours`, `approx`, `shape`, `rotcalipers`).
//!
//! # Parity with OpenCV 4.13
//!
//! Every step is integer or single-precision arithmetic that OpenCV specifies, ported
//! operation for operation (no fused multiply-adds, as OpenCV's baseline x86-64 build), and
//! the result is OpenCV's to the bit on every reference image (`chessboard-*` in the
//! primitive oracle, made by the 4.13 libraries of the XMALab 3 AppImage). Two points needed
//! a choice:
//!
//! - The neighbour search uses a k-d tree (`cvflann::KDTreeSingleIndex`, exact search); its
//!   result is every point strictly inside the radius, sorted by distance and then by index
//!   (`RadiusUniqueResultSet`), which this port computes directly.
//! - `boxFilter` rounds the mean in `float` in its vector loops and in `double` in the
//!   scalar loop for the last `width % 8` columns (module `imgproc::adaptive`).
//!
//! Where OpenCV throws (a pattern side of 2 or less; an image with a side below 15 pixels
//! when the adaptive loop is reached, whose block size is then 1), this port returns
//! `found == false` and no corners (parity ledger A170): XMALab's detection thread then
//! stores nothing, which is what it does after OpenCV's exception too.

mod detector;
mod fastcheck;

use crate::image::ImageU8;
use crate::imgproc::adaptive::{
    adaptive_threshold_mean_u8, draw_border3, equalize_hist_u8, mean_u8,
};
use crate::imgproc::arith::{ThresholdType, threshold_u8};
use crate::imgproc::corners::{CornerSubPixError, TermCriteria, corner_sub_pix};
use crate::imgproc::morph::dilate_u8;
use detector::Detector;

/// The `CALIB_CB_*` flags of `cv::findChessboardCorners`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChessboardFlags {
    /// `CALIB_CB_ADAPTIVE_THRESH` (XMALab passes only this one).
    pub adaptive_thresh: bool,
    /// `CALIB_CB_NORMALIZE_IMAGE`.
    pub normalize_image: bool,
    /// `CALIB_CB_FILTER_QUADS`.
    pub filter_quads: bool,
    /// `CALIB_CB_FAST_CHECK`.
    pub fast_check: bool,
}

impl ChessboardFlags {
    /// The flags XMALab passes (`CheckerboardDetection.cpp:100`).
    pub const XMALAB: ChessboardFlags = ChessboardFlags {
        adaptive_thresh: true,
        normalize_image: false,
        filter_quads: false,
        fast_check: false,
    };

    /// The flags from OpenCV's integer (`ADAPTIVE_THRESH` 1, `NORMALIZE_IMAGE` 2,
    /// `FILTER_QUADS` 4, `FAST_CHECK` 8; other bits are ignored).
    #[must_use]
    pub fn from_bits(bits: i32) -> Self {
        ChessboardFlags {
            adaptive_thresh: bits & 1 != 0,
            normalize_image: bits & 2 != 0,
            filter_quads: bits & 4 != 0,
            fast_check: bits & 8 != 0,
        }
    }
}

/// What `cv::findChessboardCorners` returns.
#[derive(Clone, Debug, PartialEq)]
pub struct ChessboardCorners {
    /// The return value.
    pub found: bool,
    /// What OpenCV leaves in its output array: with `found`, the `cols * rows` corners row
    /// by row (refined to sub-pixel by OpenCV's own small `cornerSubPix`); without, the
    /// corners of the best partial group or of a board that failed the last checks
    /// (unrefined), or none.
    pub corners: Vec<[f32; 2]>,
}

impl ChessboardCorners {
    fn none() -> Self {
        ChessboardCorners {
            found: false,
            corners: Vec::new(),
        }
    }
}

/// `icvBinarizationHistogramBased`: a global threshold chosen from the peaks of the
/// smoothed histogram (pixels at or above it become 255, the others 0); the image is
/// returned unchanged when the chosen threshold is 0.
///
/// Ported from OpenCV 4.13 calib3d/src/calibinit.cpp, `icvBinarizationHistogramBased`
/// (with `icvGetIntensityHistogram256`, `icvSmoothHistogram256<1>`,
/// `icvGradientOfHistogram256`).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn binarize_histogram(img: &ImageU8) -> ImageU8 {
    let n = img.data().len();
    let max_pix = i64::try_from(n).unwrap_or(i64::MAX);
    let max_pix1 = max_pix / 100;
    let mut hist = [0i64; 256];
    for &v in img.data() {
        hist[usize::from(v)] += 1;
    }
    let mut smooth = [0i64; 256];
    for i in 0..256usize {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(255);
        let s: i64 = hist[lo..=hi].iter().sum();
        smooth[i] = s / (hi - lo + 1) as i64;
    }
    let mut grad = [0i64; 256];
    let mut prev_grad = 0i64;
    for i in 1..255 {
        let mut g = smooth[i - 1] - smooth[i + 1];
        if g.abs() < 100 {
            g = if prev_grad == 0 { -100 } else { prev_grad };
        }
        grad[i] = g;
        prev_grad = g;
    }
    let mut max_pos: Vec<usize> = Vec::with_capacity(20);
    let mut i = 254usize;
    while i > 2 && max_pos.len() < 20 {
        if grad[i - 1] < 0 && grad[i] > 0 {
            let sum_around = smooth[i - 1] + smooth[i] + smooth[i + 1];
            if !(sum_around < max_pix1 && i < 64) {
                max_pos.push(i);
            }
        }
        i -= 1;
    }
    let cnt = max_pos.len();
    let thresh: i64 = match cnt {
        0 => {
            let half = max_pix / 2;
            let mut sum = 0i64;
            let mut t = 0i64;
            for (v, &h) in hist.iter().enumerate() {
                sum += h;
                if sum > half {
                    t = v as i64;
                    break;
                }
            }
            t
        }
        1 => max_pos[0] as i64 / 2,
        2 => (max_pos[0] as i64 + max_pos[1] as i64) / 2,
        _ => {
            let mut idx_acc_sum = 0usize;
            let mut accum = 0i64;
            for v in (1..256).rev() {
                accum += hist[v];
                if accum > max_pix / 18 {
                    idx_acc_sum = v;
                    break;
                }
            }
            let mut idx_bg_max = 0usize;
            let mut bright_max = max_pos[0];
            for (nn, &p) in max_pos.iter().enumerate().take(cnt - 1) {
                idx_bg_max = nn + 1;
                if p < idx_acc_sum {
                    break;
                }
                bright_max = p;
            }
            let mut max_val = hist[max_pos[idx_bg_max]];
            if max_pos[idx_bg_max] >= 250 && idx_bg_max + 1 < cnt {
                idx_bg_max += 1;
                max_val = hist[max_pos[idx_bg_max]];
            }
            for nn in idx_bg_max + 1..cnt {
                if hist[max_pos[nn]] >= max_val {
                    max_val = hist[max_pos[nn]];
                    idx_bg_max = nn;
                }
            }
            let dist2 = (bright_max as i64 - max_pos[idx_bg_max] as i64) / 2;
            bright_max as i64 - dist2
        }
    };
    if thresh > 0 {
        img.map(|v| if i64::from(v) >= thresh { 255 } else { 0 })
    } else {
        img.clone()
    }
}

/// `cv::findChessboardCorners(image, Size(cols, rows), corners, flags)` on an 8-bit grey
/// image. `cols` and `rows` are the inner corners per row and per column (OpenCV's
/// `Size(width, height)`).
///
/// Returns `found == false` with no corners where OpenCV throws (a side of 2 or less, an
/// empty image, an adaptive block size of 1) and where `cols * rows` does not fit OpenCV's
/// `int` (module documentation).
///
/// Ported from OpenCV 4.13 calib3d/src/calibinit.cpp, `cv::findChessboardCorners`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn find_chessboard_corners(
    image: &ImageU8,
    cols: usize,
    rows: usize,
    flags: ChessboardFlags,
) -> ChessboardCorners {
    if cols <= 2 || rows <= 2 || image.is_empty() {
        return ChessboardCorners::none();
    }
    let (Ok(width), Ok(height)) = (i32::try_from(cols), i32::try_from(rows)) else {
        return ChessboardCorners::none();
    };
    if width.checked_mul(height).is_none() {
        return ChessboardCorners::none();
    }
    let area = cols * rows;
    let mut img = image.clone();
    let mut prev_sqr_size = 0i32;
    let mut found = false;
    let mut out_corners: Vec<(f32, f32)> = Vec::new();

    let mut thresh_img_new = binarize_histogram(&img);
    if flags.fast_check
        && !fastcheck::check_chessboard_binary(&thresh_img_new, width, height)
        && !fastcheck::check_chessboard(&img, width, height)
    {
        return ChessboardCorners::none();
    }

    let mut detector = Detector::new(width, height);
    const MAX_DILATIONS: i32 = 7;
    for dilations in 0..=MAX_DILATIONS {
        if dilations > 0 {
            thresh_img_new = dilate_u8(&thresh_img_new, 1);
        }
        draw_border3(&mut thresh_img_new);
        detector.generate_quads(&thresh_img_new, flags.filter_quads, dilations);
        if detector.process_quads(&mut out_corners, &mut prev_sqr_size) {
            found = true;
            break;
        }
    }

    // revert to old, slower, method if detection failed
    if !found {
        if flags.normalize_image {
            img = equalize_hist_u8(&img);
        }
        prev_sqr_size = 0;
        let use_adaptive = flags.adaptive_thresh;
        let mut thresh_img = ImageU8::new(0, 0);
        if !use_adaptive {
            let mean = mean_u8(&img);
            let thresh_level = ((mean - 10.0).round_ties_even() as i64).max(10);
            #[allow(clippy::cast_precision_loss)]
            let (t, _) = threshold_u8(&img, thresh_level as f64, 255.0, ThresholdType::Binary);
            thresh_img = t;
        }
        let max_k = if use_adaptive { 6 } else { 1 };
        let mut prev_thresh_img = ImageU8::new(0, 0);
        let (w, h) = (img.width(), img.height());
        'k: for k in 0..max_k {
            if found {
                break;
            }
            let mut prev_block_size: i64 = -1;
            for dilations in 0..=MAX_DILATIONS {
                if use_adaptive {
                    #[allow(clippy::cast_precision_loss)]
                    let bs = if prev_sqr_size == 0 {
                        w.min(h) as f64 * (if k % 2 == 0 { 0.2 } else { 0.1 })
                    } else {
                        f64::from(prev_sqr_size) * 2.0
                    };
                    let block_size = bs.round_ties_even() as i64 | 1;
                    if block_size != prev_block_size {
                        let Some(t) = usize::try_from(block_size).ok().and_then(|b| {
                            adaptive_threshold_mean_u8(&img, b, f64::from((k / 2) * 5))
                        }) else {
                            // OpenCV: CV_Assert(blockSize % 2 == 1 && blockSize > 1) throws.
                            return ChessboardCorners::none();
                        };
                        thresh_img = dilate_u8(&t, dilations as usize);
                        prev_thresh_img = thresh_img.clone();
                    } else if dilations > 0 {
                        prev_thresh_img = dilate_u8(&prev_thresh_img, 1);
                        thresh_img = prev_thresh_img.clone();
                    }
                    prev_block_size = block_size;
                } else if dilations > 0 {
                    thresh_img = dilate_u8(&thresh_img, 1);
                }
                draw_border3(&mut thresh_img);
                detector.generate_quads(&thresh_img, flags.filter_quads, dilations);
                if detector.process_quads(&mut out_corners, &mut prev_sqr_size) {
                    found = true;
                    continue 'k;
                }
            }
        }
    }

    if found {
        found = detector.check_board_monotony(&out_corners);
    }

    // check that none of the found corners is too close to the image boundary
    if found {
        const BORDER: f32 = 8.0;
        #[allow(clippy::cast_precision_loss)]
        let (fw, fh) = (img.width() as f32, img.height() as f32);
        for p in out_corners.iter().take(area) {
            if p.0 <= BORDER || p.0 > fw - BORDER || p.1 <= BORDER || p.1 > fh - BORDER {
                found = false;
                break;
            }
        }
    }

    if found {
        if rows.is_multiple_of(2) && cols.is_multiple_of(2) {
            let last_row = (rows - 1) * cols;
            let dy0 = f64::from(out_corners[last_row].1 - out_corners[0].1);
            if dy0 < 0.0 {
                out_corners[..area].reverse();
            }
        }
        if corner_sub_pix(
            &img,
            &mut out_corners,
            (2, 2),
            (-1, -1),
            TermCriteria {
                max_iter: Some(15),
                epsilon: Some(0.1),
            },
        )
        .is_err()
        {
            return ChessboardCorners::none();
        }
    }

    ChessboardCorners {
        found,
        corners: out_corners.iter().map(|p| [p.0, p.1]).collect(),
    }
}

/// XMALab's `cornerSubPix(image, corners, Size(11, 11), Size(-1, -1),
/// TermCriteria(MAX_ITER + EPS, 30, 0.1))` (`CheckerboardDetection.cpp:103` and `:127`).
///
/// # Errors
/// Where OpenCV throws: an image smaller than 27 pixels in a direction, or a corner outside
/// the image (corners before it are already refined). XMALab's detection then stores
/// nothing.
pub fn refine_corners_xmalab(
    image: &ImageU8,
    corners: &mut [[f32; 2]],
) -> Result<(), CornerSubPixError> {
    let mut pts: Vec<(f32, f32)> = corners.iter().map(|p| (p[0], p[1])).collect();
    let r = corner_sub_pix(
        image,
        &mut pts,
        (11, 11),
        (-1, -1),
        TermCriteria {
            max_iter: Some(30),
            epsilon: Some(0.1),
        },
    );
    for (c, p) in corners.iter_mut().zip(&pts) {
        *c = [p.0, p.1];
    }
    r
}

/// XMALab's automatic mode (`CheckerboardDetection::detectCorner_thread`, the branch without
/// homography): [`find_chessboard_corners`] with [`ChessboardFlags::XMALAB`], then
/// [`refine_corners_xmalab`] if the board was found. XMALab stores the result only when it
/// has `cols * rows` corners (`detectCorner_threadFinished`); it keeps an unrefined result
/// that OpenCV reported as not found but left complete (parity ledger A171).
///
/// Where the refinement fails (OpenCV throws), no corners are returned.
///
/// Ported from processing/CheckerboardDetection.cpp, `detectCorner_thread`.
#[must_use]
pub fn detect_corners_xmalab(image: &ImageU8, cols: usize, rows: usize) -> ChessboardCorners {
    let mut r = find_chessboard_corners(image, cols, rows, ChessboardFlags::XMALAB);
    if r.found && refine_corners_xmalab(image, &mut r.corners).is_err() {
        return ChessboardCorners::none();
    }
    r
}

/// The manual mode's homography (`CheckerboardDetection.cpp:115`):
/// `cv::findHomography(gridPoints, selectedPoints)` from the four outer inner corners of the
/// grid, `(0, 0)`, `((cols - 1) s, 0)`, `((cols - 1) s, (rows - 1) s)`, `(0, (rows - 1) s)`
/// with `s = square_size` (as `float`), to the four clicks (as `float`), in that order.
/// Row-major, normalised to `H(2,2) = 1`; `None` where OpenCV returns an empty matrix.
///
/// Ported from processing/CheckerboardDetection.cpp, `CheckerboardDetection` (the
/// constructor with references) and `detectCorner_thread`; the homography is
/// [`crate::calib3d::homography::find_homography`] (method 0), which agrees with the
/// reference OpenCV to rounding, not to the bit (its eigenvector comes from Eigen there).
#[must_use]
pub fn manual_homography(
    cols: usize,
    rows: usize,
    square_size: f64,
    clicks: &[[f64; 2]; 4],
) -> Option<[f64; 9]> {
    #[allow(clippy::cast_precision_loss)]
    let (gx, gy) = (
        ((cols as f64 - 1.0) * square_size) as f32,
        ((rows as f64 - 1.0) * square_size) as f32,
    );
    let grid = [
        [0.0, 0.0],
        [f64::from(gx), 0.0],
        [f64::from(gx), f64::from(gy)],
        [0.0, f64::from(gy)],
    ];
    if clicks.iter().flatten().any(|v| !v.is_finite()) || !gx.is_finite() || !gy.is_finite() {
        return None;
    }
    crate::calib3d::homography::find_homography(&grid, clicks)
}

/// The manual mode's prediction of each point of the calibration object through `h`, as the
/// C++ computes it: `x' = (h00 x + h01 y + h02) / (h20 x + h21 y + h22)` and likewise for
/// `y'` in `double`, then stored as `float` (`std::vector<cv::Point2f>`).
///
/// Ported from processing/CheckerboardDetection.cpp, `detectCorner_thread` (lines 116-124).
#[must_use]
pub fn project_with_homography(h: &[f64; 9], points: &[[f64; 2]]) -> Vec<[f32; 2]> {
    points
        .iter()
        .map(|p| {
            let denom = p[0] * h[6] + p[1] * h[7] + h[8];
            let x = (p[0] * h[0] + p[1] * h[1] + h[2]) / denom;
            let y = (p[0] * h[3] + p[1] * h[4] + h[5]) / denom;
            #[allow(clippy::cast_possible_truncation)]
            let r = [x as f32, y as f32];
            r
        })
        .collect()
}

/// XMALab's manual mode for a board whose object points are `(x s, y s)`, rows outer
/// (`CalibrationObject::setCheckerboard` without an inverted axis): the homography of
/// [`manual_homography`], every corner predicted by [`project_with_homography`], then
/// [`refine_corners_xmalab`] if `refine` (the setting `DisableCheckerboardRefinement`
/// off). `None` where the homography is empty or the refinement fails (OpenCV throws: a
/// predicted corner outside the image); XMALab then stores nothing.
///
/// Ported from processing/CheckerboardDetection.cpp, `detectCorner_thread`.
#[must_use]
pub fn manual_corners(
    image: &ImageU8,
    cols: usize,
    rows: usize,
    square_size: f64,
    clicks: &[[f64; 2]; 4],
    refine: bool,
) -> Option<Vec<[f32; 2]>> {
    let h = manual_homography(cols, rows, square_size, clicks)?;
    let n = cols.checked_mul(rows)?;
    if n > image.data().len().max(1 << 16) {
        // More corners than pixels: no board of that size can be refined (bounded
        // allocation for any argument).
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let obj: Vec<[f64; 2]> = (0..rows)
        .flat_map(|y| (0..cols).map(move |x| [x as f64 * square_size, y as f64 * square_size]))
        .collect();
    let mut pts = project_with_homography(&h, &obj);
    if refine && refine_corners_xmalab(image, &mut pts).is_err() {
        return None;
    }
    Some(pts)
}
