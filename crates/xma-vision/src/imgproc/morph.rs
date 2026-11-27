//! Grey-level dilation and erosion with OpenCV's default structuring element, as
//! `cv::findChessboardCorners` and `cv::checkChessboard` call them:
//! `dilate(src, dst, Mat(), Point(-1, -1), iterations)` and the same for `erode`.
//!
//! With an empty kernel and `iterations` n, OpenCV uses one `(2n + 1)` square rectangle
//! kernel once (`morphOp` in `modules/imgproc/src/morph.dispatch.cpp`), with
//! `BORDER_CONSTANT` and `morphologyDefaultBorderValue()`: pixels outside the image never
//! win the maximum (dilation) or the minimum (erosion). Both are exact integer operations, so
//! the result is OpenCV's whatever the code path (checked against the 4.13 libraries,
//! `chessops-*` in the primitive oracle).

use crate::image::ImageU8;

/// `cv::dilate(src, dst, Mat(), Point(-1, -1), iterations)` on an 8-bit image: each pixel
/// becomes the largest value in the `(2 iterations + 1)` square around it, clipped to the
/// image. `iterations == 0` returns a copy (OpenCV's `src.copyTo(dst)`).
///
/// Ported from OpenCV 4.13 imgproc/src/morph.dispatch.cpp, `cv::dilate` / `morphOp`.
#[must_use]
pub fn dilate_u8(src: &ImageU8, iterations: usize) -> ImageU8 {
    square_filter(src, iterations, u8::max)
}

/// `cv::erode(src, dst, Mat(), Point(-1, -1), iterations)` on an 8-bit image: the smallest
/// value in the `(2 iterations + 1)` square, clipped to the image.
///
/// Ported from OpenCV 4.13 imgproc/src/morph.dispatch.cpp, `cv::erode` / `morphOp`.
#[must_use]
pub fn erode_u8(src: &ImageU8, iterations: usize) -> ImageU8 {
    square_filter(src, iterations, u8::min)
}

/// The separable square max or min filter of half width `r`, clipped to the image.
fn square_filter(src: &ImageU8, r: usize, op: fn(u8, u8) -> u8) -> ImageU8 {
    let (w, h) = (src.width(), src.height());
    if r == 0 || w == 0 || h == 0 {
        return src.clone();
    }
    // A half width beyond the image's size changes nothing further.
    let r = r.min(w.max(h));
    let mut tmp = ImageU8::new(w, h);
    for y in 0..h {
        let row = src.row(y);
        let out = tmp.row_mut(y);
        for (x, o) in out.iter_mut().enumerate() {
            let lo = x.saturating_sub(r);
            let hi = x.saturating_add(r).min(w - 1);
            *o = row[lo..=hi].iter().copied().fold(row[lo], op);
        }
    }
    let mut dst = ImageU8::new(w, h);
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = y.saturating_add(r).min(h - 1);
        let mut acc = tmp.row(lo).to_vec();
        for yy in lo + 1..=hi {
            for (a, &b) in acc.iter_mut().zip(tmp.row(yy)) {
                *a = op(*a, b);
            }
        }
        dst.row_mut(y).copy_from_slice(&acc);
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilate_and_erode_small() {
        let mut im = ImageU8::new(5, 4);
        im.set(2, 1, 200);
        im.set(0, 3, 7);
        let d = dilate_u8(&im, 1);
        assert_eq!(d.at(1, 0), 200);
        assert_eq!(d.at(3, 2), 200);
        assert_eq!(d.at(4, 3), 0);
        assert_eq!(d.at(1, 2), 200);
        assert_eq!(d.at(0, 3), 7);
        let e = erode_u8(&ImageU8::filled(3, 3, 9), 2);
        assert_eq!(e, ImageU8::filled(3, 3, 9));
        assert_eq!(dilate_u8(&im, 0), im);
        let empty = ImageU8::new(0, 0);
        assert!(dilate_u8(&empty, 3).is_empty());
        // A huge iteration count is the whole image.
        assert_eq!(dilate_u8(&im, usize::MAX).at(0, 0), 200);
    }
}
