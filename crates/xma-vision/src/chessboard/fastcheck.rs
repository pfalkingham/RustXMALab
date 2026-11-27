//! The quick test `CALIB_CB_FAST_CHECK` makes before the search: `checkChessboardBinary` on
//! the histogram-binarised image and, if it fails, `cv::checkChessboard` on the image.
//! Ported from OpenCV 4.13 `modules/calib3d/src/checkchessboard.cpp`.

use crate::image::ImageU8;
use crate::imgproc::arith::{ThresholdType, threshold_u8};
use crate::imgproc::contours::{ChainApprox, PointI, find_contours_ccomp};
use crate::imgproc::morph::{dilate_u8, erode_u8};
use crate::imgproc::rotcalipers::min_area_rect;

/// `icvGetQuadrangleHypotheses`: the size of each outer contour whose rotated bounding box
/// is at least 10 pixels and not too elongated, with `class_id`.
fn quadrangle_hypotheses(image: &ImageU8, quads: &mut Vec<(f32, usize)>, class_id: usize) {
    const MIN_ASPECT_RATIO: f32 = 0.3;
    const MAX_ASPECT_RATIO: f32 = 3.0;
    const MIN_BOX_SIZE: f32 = 10.0;
    let (contours, hierarchy) = find_contours_ccomp(image, ChainApprox::Simple, PointI::default());
    for (c, h) in contours.iter().zip(&hierarchy) {
        if h[3] != -1 {
            continue; // skip holes
        }
        let b = min_area_rect(c);
        let box_size = b.width.max(b.height);
        if box_size < MIN_BOX_SIZE {
            continue;
        }
        let aspect_ratio = b.width / b.height.max(1.0);
        if aspect_ratio < MIN_ASPECT_RATIO || aspect_ratio > MAX_ASPECT_RATIO {
            continue;
        }
        quads.push((box_size, class_id));
    }
}

/// `fillQuads`.
fn fill_quads(
    white: &ImageU8,
    black: &ImageU8,
    white_thresh: f64,
    black_thresh: f64,
    quads: &mut Vec<(f32, usize)>,
) {
    let (t, _) = threshold_u8(white, white_thresh, 255.0, ThresholdType::Binary);
    quadrangle_hypotheses(&t, quads, 1);
    let (t, _) = threshold_u8(black, black_thresh, 255.0, ThresholdType::BinaryInv);
    quadrangle_hypotheses(&t, quads, 0);
}

/// `checkQuads`: whether enough hypotheses of similar size, black and white, are found.
fn check_quads(quads: &mut [(f32, usize)], width: i32, height: i32) -> bool {
    let min_quads_count = (i64::from(width) * i64::from(height) / 2) as usize;
    // std::sort on the size alone; the result does not depend on the order of equal sizes
    // (an equal-size run's first element sees a superset of what the others see).
    quads.sort_by(|a, b| a.0.total_cmp(&b.0));
    const SIZE_REL_DEV: f32 = 0.4;
    let black_count = ((f64::from(width) / 2.0).ceil() * (f64::from(height) / 2.0).ceil())
        .round_ties_even() as i64;
    let white_count = ((f64::from(width) / 2.0).floor() * (f64::from(height) / 2.0).floor())
        .round_ties_even() as i64;
    for i in 0..quads.len() {
        let mut j = i + 1;
        while j < quads.len() {
            if quads[j].0 / quads[i].0 > 1.0 + SIZE_REL_DEV {
                break;
            }
            j += 1;
        }
        if j + 1 > min_quads_count + i {
            let mut counts = [0i64; 2];
            for q in &quads[i..j] {
                counts[q.1] += 1;
            }
            #[allow(clippy::cast_precision_loss)]
            if (counts[0] as f64) < black_count as f64 * 0.75
                || (counts[1] as f64) < white_count as f64 * 0.75
            {
                continue;
            }
            return true;
        }
    }
    false
}

/// `cv::checkChessboard(img, size)`.
///
/// Ported from checkchessboard.cpp, `cv::checkChessboard`.
pub(super) fn check_chessboard(img: &ImageU8, width: i32, height: i32) -> bool {
    let white = erode_u8(img, 1);
    let black = dilate_u8(img, 1);
    let mut thresh_level = 20.0f32;
    while thresh_level < 130.0 {
        let mut quads = Vec::new();
        fill_quads(
            &white,
            &black,
            f64::from(thresh_level + 70.0),
            f64::from(thresh_level),
            &mut quads,
        );
        if check_quads(&mut quads, width, height) {
            return true;
        }
        thresh_level += 20.0;
    }
    false
}

/// `checkChessboardBinary(img, size)` on a binary image: 1 if a board can be there.
///
/// Ported from checkchessboard.cpp, `checkChessboardBinary`.
pub(super) fn check_chessboard_binary(img: &ImageU8, width: i32, height: i32) -> bool {
    let mut white = img.clone();
    let mut black = img.clone();
    for erosion_count in 0..=3 {
        if erosion_count != 0 {
            white = erode_u8(&white, 1);
            black = dilate_u8(&black, 1);
        }
        let mut quads = Vec::new();
        fill_quads(&white, &black, 128.0, 128.0, &mut quads);
        if check_quads(&mut quads, width, height) {
            return true;
        }
    }
    false
}
