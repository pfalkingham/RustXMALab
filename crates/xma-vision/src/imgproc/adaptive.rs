//! The 8-bit image routines `cv::findChessboardCorners` runs before it looks for quads:
//! `adaptiveThreshold` with `ADAPTIVE_THRESH_MEAN_C` (and the normalised `boxFilter` it is
//! built on), `equalizeHist`, the mean of an image (`cv::mean`) and the border it draws
//! (`rectangle` with thickness 3).
//!
//! The box filter reproduces OpenCV 4.13's three summing paths for 8-bit input and output
//! (`createBoxFilter` in `modules/imgproc/src/box_filter.simd.hpp`), which round the mean
//! differently: up to 256 pixels in the window, 16-bit sums divided by a fixed-point
//! reciprocal (`ColumnSum<ushort, uchar>`); up to 2^23 pixels, 32-bit sums scaled in
//! `float` by the SIMD loop and in `double` by the scalar loop for the last `width % 8`
//! columns of each row (`ColumnSum<int, uchar>`; the vector loops of every x86 dispatch
//! cover whole multiples of 8 columns); beyond that, `double` sums (`ColumnSum<double,
//! uchar>`). Every result is checked against the 4.13 libraries (`chessops-*`).

use crate::image::ImageU8;

/// OpenCV's `cvRound` of a `double`: round half to even (`lrint` in the default mode).
fn cv_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `saturate_cast<uchar>` of an integer.
fn sat_u8(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

/// The largest window side [`box_filter_mean_u8`] uses.
const MAX_SIDE: usize = 1 << 26;

/// The sum of `src` over the `kw` x `kh` window centred on each pixel (anchor `kw / 2`,
/// `kh / 2`), with `BORDER_REPLICATE`: row `y`, column `x` of the result is element
/// `y * width + x`.
fn box_sums(src: &ImageU8, kw: usize, kh: usize) -> Vec<i64> {
    let (w, h) = (src.width(), src.height());
    let (ax, ay) = (kw / 2, kh / 2);
    // Row sums.
    let mut rows = vec![0i64; w * h];
    let mut pre = vec![0i64; w + 1];
    for y in 0..h {
        let r = src.row(y);
        for x in 0..w {
            pre[x + 1] = pre[x] + i64::from(r[x]);
        }
        for x in 0..w {
            // window [x - ax, x - ax + kw - 1], clamped copies of the end pixels
            let lo = x as i64 - ax as i64;
            let hi = lo + kw as i64 - 1;
            rows[y * w + x] = clamped_sum(&pre, lo, hi, i64::from(r[0]), i64::from(r[w - 1]));
        }
    }
    // Column sums of the row sums, row by row (a column at a time strides through the whole
    // image for every pixel): `pre` holds, for each column, the prefix
    // sums of the row sums down the column (row `y` of `pre` is the sum of rows `0..y`), and
    // each output row takes the window's rows from it as `clamped_sum` does, its counts of
    // clamped rows computed once per row. The same exact integers as summing each column.
    let mut pre = vec![0i64; (h + 1) * w];
    for y in 0..h {
        let (above, below) = pre.split_at_mut((y + 1) * w);
        let above = &above[y * w..];
        for ((p, &a), &r) in below[..w]
            .iter_mut()
            .zip(above)
            .zip(&rows[y * w..(y + 1) * w])
        {
            *p = a + r;
        }
    }
    // The result goes into the row sums' buffer; only the first and last rows are read again.
    let (first, last) = (rows[..w].to_vec(), rows[(h - 1) * w..].to_vec());
    let n = h as i64;
    let mut out = rows;
    for (y, o) in out.chunks_exact_mut(w).enumerate() {
        let lo = y as i64 - ay as i64;
        let hi = lo + kh as i64 - 1;
        let n_first = if lo < 0 { (-lo).min(hi - lo + 1) } else { 0 };
        let n_last = if hi > n - 1 {
            (hi - (n - 1)).min(hi - lo + 1)
        } else {
            0
        };
        let (a, b) = (lo.max(0), hi.min(n - 1));
        let inner = (a <= b).then(|| {
            (
                &pre[(b + 1) as usize * w..][..w],
                &pre[a as usize * w..][..w],
            )
        });
        for x in 0..w {
            let mut s = first[x] * n_first + last[x] * n_last;
            if let Some((pb, pa)) = inner {
                s += pb[x] - pa[x];
            }
            o[x] = s;
        }
    }
    out
}

/// The sum of `v[clamp(i)]` for `i` in `lo..=hi`, from the prefix sums `pre` of `v`
/// (`pre[0] = 0`), `first = v[0]` and `last = v[n - 1]`.
fn clamped_sum(pre: &[i64], lo: i64, hi: i64, first: i64, last: i64) -> i64 {
    let n = (pre.len() - 1) as i64;
    let mut s = 0i64;
    if lo < 0 {
        s += first * ((-lo).min(hi - lo + 1));
    }
    if hi > n - 1 {
        s += last * (hi - (n - 1)).min(hi - lo + 1);
    }
    let a = lo.max(0);
    let b = hi.min(n - 1);
    if a <= b {
        s += pre[(b + 1) as usize] - pre[a as usize];
    }
    s
}

/// `cv::boxFilter(src, dst, CV_8U, Size(kw, kh), Point(-1, -1), true,
/// BORDER_REPLICATE | BORDER_ISOLATED)`: the mean over the window, rounded as OpenCV's
/// summing path for that window size rounds it (module documentation). A window of 0 in
/// either direction returns a copy; a side beyond 2^26 counts as 2^26.
///
/// Ported from OpenCV 4.13 imgproc/src/box_filter.dispatch.cpp, `cv::boxFilter`, and
/// box_filter.simd.hpp, `createBoxFilter`, `RowSum`, `ColumnSum<ushort, uchar>`,
/// `ColumnSum<int, uchar>`, `ColumnSum<double, uchar>`.
#[must_use]
pub fn box_filter_mean_u8(src: &ImageU8, kw: usize, kh: usize) -> ImageU8 {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 || kw == 0 || kh == 0 {
        return src.clone();
    }
    // BORDER_ISOLATED with normalisation: a one-row or one-column image is filtered along
    // the other direction only.
    let kh = if h == 1 { 1 } else { kh };
    let kw = if w == 1 { 1 } else { kw };
    // Windows beyond 2^26 pixels a side are taken as 2^26 (a sum of 255 over the window
    // must fit in 64 bits; OpenCV's own sums lose exactness long before that).
    let (kw, kh) = (kw.min(MAX_SIDE), kh.min(MAX_SIDE));
    let area = (kw as u128).saturating_mul(kh as u128);
    let sums = box_sums(src, kw, kh);
    let scale = 1.0 / (kw as f64 * kh as f64);
    let mut dst = ImageU8::new(w, h);
    let out = dst.data_mut();
    if area <= 256 {
        // ColumnSum<ushort, uchar>: (s + divDelta) * divScale >> 23.
        const SHIFT: u32 = 23;
        let (div_scale, div_delta) = if scale == 1.0 {
            (1i64, 0i64)
        } else {
            let d = cv_round(1.0 / scale);
            let scalef = f64::from(1u32 << SHIFT) / d as f64;
            let mut ds = scalef.floor() as i64;
            let frac = scalef - ds as f64;
            let mut dd = d / 2;
            if frac < 0.5 {
                dd += 1;
            } else {
                ds += 1;
            }
            (ds, dd)
        };
        for (o, &s) in out.iter_mut().zip(&sums) {
            *o = if scale == 1.0 {
                sat_u8(s)
            } else {
                (((s + div_delta) * div_scale) >> SHIFT) as u8
            };
        }
    } else if area <= 1 << 23 {
        // ColumnSum<int, uchar>: float in the vector loops, double in the scalar tail.
        #[allow(clippy::cast_possible_truncation)]
        let fscale = scale as f32;
        let simd_cols = w - w % 8;
        for y in 0..h {
            for x in 0..w {
                let s = sums[y * w + x] as i32;
                out[y * w + x] = if x < simd_cols {
                    #[allow(clippy::cast_precision_loss)]
                    let v = (s as f32) * fscale;
                    sat_u8(v.round_ties_even() as i64)
                } else {
                    sat_u8(cv_round(f64::from(s) * scale))
                };
            }
        }
    } else {
        // ColumnSum<double, uchar>.
        for (o, &s) in out.iter_mut().zip(&sums) {
            #[allow(clippy::cast_precision_loss)]
            let v = s as f64 * scale;
            *o = sat_u8(cv_round(v));
        }
    }
    dst
}

/// `cv::adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size,
/// delta)`: 255 where `src - mean > -ceil(delta)`, the mean taken over the
/// `block_size` square by [`box_filter_mean_u8`], else 0.
///
/// `None` where OpenCV throws: `block_size` even or not above 1.
///
/// Ported from OpenCV 4.13 imgproc/src/thresh.cpp, `cv::adaptiveThreshold`.
#[must_use]
pub fn adaptive_threshold_mean_u8(src: &ImageU8, block_size: usize, delta: f64) -> Option<ImageU8> {
    if block_size % 2 != 1 || block_size <= 1 {
        return None;
    }
    let mean = box_filter_mean_u8(src, block_size, block_size);
    // idelta = cvCeil(delta); tab[i] = i - 255 > -idelta ? 255 : 0 for i = src - mean + 255.
    let idelta = if delta.is_nan() {
        i64::MIN
    } else {
        delta.ceil().clamp(-1e15, 1e15) as i64
    };
    let mut dst = ImageU8::new(src.width(), src.height());
    for ((o, &s), &m) in dst.data_mut().iter_mut().zip(src.data()).zip(mean.data()) {
        let d = i64::from(s) - i64::from(m);
        *o = if idelta != i64::MIN && d > -idelta {
            255
        } else {
            0
        };
    }
    Some(dst)
}

/// `cv::equalizeHist(src, dst)` on an 8-bit image. An empty image is returned unchanged;
/// an image of one value keeps that value.
///
/// Ported from OpenCV 4.13 imgproc/src/histogram.cpp, `cv::equalizeHist`.
#[must_use]
pub fn equalize_hist_u8(src: &ImageU8) -> ImageU8 {
    if src.is_empty() {
        return src.clone();
    }
    let mut hist = [0i64; 256];
    for &v in src.data() {
        hist[usize::from(v)] += 1;
    }
    let mut i = 0usize;
    while hist[i] == 0 {
        i += 1;
    }
    let total = src.data().len() as i64;
    if hist[i] == total {
        return ImageU8::filled(src.width(), src.height(), i as u8);
    }
    // float scale = 255.f / (total - hist[i]); lut[i] = saturate_cast<uchar>(sum * scale)
    #[allow(clippy::cast_precision_loss)]
    let scale = 255.0f32 / ((total - hist[i]) as i32) as f32;
    let mut lut = [0u8; 256];
    let mut sum = 0i32;
    lut[i] = 0;
    for j in i + 1..256 {
        sum += hist[j] as i32;
        #[allow(clippy::cast_precision_loss)]
        let v = sum as f32 * scale;
        lut[j] = sat_u8(v.round_ties_even() as i64);
    }
    src.map(|v| lut[usize::from(v)])
}

/// `cv::mean(src).val[0]` of an 8-bit image: the sum over the number of pixels (0 for an
/// empty image).
///
/// Ported from OpenCV 4.13 core/src/mean.dispatch.cpp, `cv::mean` (the sums are exact
/// integers, multiplied once by `1. / n` in `double`).
#[must_use]
pub fn mean_u8(src: &ImageU8) -> f64 {
    let n = src.data().len();
    if n == 0 {
        return 0.0;
    }
    let s: u64 = src.data().iter().map(|&v| u64::from(v)).sum();
    // return s * (nz0 ? 1./nz0 : 0)
    #[allow(clippy::cast_precision_loss)]
    let r = s as f64 * (1.0 / n as f64);
    r
}

/// `cv::rectangle(img, Point(0, 0), Point(cols - 1, rows - 1), Scalar(255), 3, LINE_8)`
/// as `cv::findChessboardCorners` draws it: every pixel within 3 pixels of the image's edge
/// set to 255 (the thick polyline OpenCV draws covers exactly that band; checked for every
/// size from 1 x 1 to 39 x 39 and on the larger images of the oracle).
///
/// Ported from OpenCV 4.13 imgproc/src/drawing.cpp, `cv::rectangle` (its result for this
/// call).
pub fn draw_border3(img: &mut ImageU8) {
    let (w, h) = (img.width(), img.height());
    for y in 0..h {
        let row = img.row_mut(y);
        if y < 3 || y + 3 >= h {
            row.fill(255);
        } else {
            for v in row.iter_mut().take(3) {
                *v = 255;
            }
            for v in row.iter_mut().skip(w.saturating_sub(3)) {
                *v = 255;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_mean_constant_and_degenerate() {
        let im = ImageU8::filled(7, 5, 77);
        for k in [1usize, 3, 15, 17, 99, 3001] {
            assert_eq!(box_filter_mean_u8(&im, k, k), im, "k = {k}");
        }
        assert!(adaptive_threshold_mean_u8(&im, 4, 0.0).is_none());
        assert!(adaptive_threshold_mean_u8(&im, 1, 0.0).is_none());
        let t = adaptive_threshold_mean_u8(&im, 3, 0.0).unwrap();
        assert!(t.data().iter().all(|&v| v == 0));
        let t = adaptive_threshold_mean_u8(&im, 3, 0.5).unwrap();
        assert!(t.data().iter().all(|&v| v == 255));
        assert!(adaptive_threshold_mean_u8(&im, 3, f64::NAN).is_some());
        let e = ImageU8::new(0, 0);
        assert!(box_filter_mean_u8(&e, 3, 3).is_empty());
        assert!(equalize_hist_u8(&e).is_empty());
        assert_eq!(mean_u8(&e), 0.0);
        assert_eq!(equalize_hist_u8(&im), im);
    }

    /// `box_sums` as it was before the third review round (the column pass a column at a
    /// time), kept to compare with.
    fn old_box_sums(src: &ImageU8, kw: usize, kh: usize) -> Vec<i64> {
        let (w, h) = (src.width(), src.height());
        let (ax, ay) = (kw / 2, kh / 2);
        let mut rows = vec![0i64; w * h];
        let mut pre = vec![0i64; w + 1];
        for y in 0..h {
            let r = src.row(y);
            for x in 0..w {
                pre[x + 1] = pre[x] + i64::from(r[x]);
            }
            for x in 0..w {
                let lo = x as i64 - ax as i64;
                let hi = lo + kw as i64 - 1;
                rows[y * w + x] = clamped_sum(&pre, lo, hi, i64::from(r[0]), i64::from(r[w - 1]));
            }
        }
        let mut out = vec![0i64; w * h];
        let mut col = vec![0i64; h + 1];
        for x in 0..w {
            for y in 0..h {
                col[y + 1] = col[y] + rows[y * w + x];
            }
            for y in 0..h {
                let lo = y as i64 - ay as i64;
                let hi = lo + kh as i64 - 1;
                out[y * w + x] = clamped_sum(&col, lo, hi, rows[x], rows[(h - 1) * w + x]);
            }
        }
        out
    }

    fn noise(w: usize, h: usize, seed: u64) -> ImageU8 {
        let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(7);
        let data = (0..w * h)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (s >> 56) as u8
            })
            .collect();
        ImageU8::from_vec(w, h, data)
    }

    /// Finding 3: the row-major column pass gives the old sums, and so the old means and
    /// thresholds, on odd sizes, 1-pixel-wide and -high images and windows larger than the
    /// image (up to the 2^26 side, where the sums are largest).
    #[test]
    fn review13_box_sums_equal_old() {
        let sizes = [
            (1, 1),
            (1, 9),
            (9, 1),
            (2, 3),
            (7, 5),
            (13, 31),
            (31, 13),
            (64, 64),
            (101, 77),
        ];
        let windows = [
            (1, 1),
            (1, 3),
            (3, 1),
            (2, 2),
            (3, 3),
            (4, 6),
            (5, 5),
            (15, 15),
            (17, 17),
            (40, 3),
            (99, 99),
            (409, 409),
            (3001, 5),
            (MAX_SIDE, MAX_SIDE),
        ];
        for (seed, &(w, h)) in sizes.iter().enumerate() {
            for v in [None, Some(0u8), Some(255)] {
                let im = match v {
                    None => noise(w, h, seed as u64),
                    Some(v) => ImageU8::filled(w, h, v),
                };
                for &(kw, kh) in &windows {
                    assert_eq!(
                        box_sums(&im, kw, kh),
                        old_box_sums(&im, kw, kh),
                        "{w}x{h} {kw}x{kh}"
                    );
                }
            }
        }
    }

    #[test]
    fn border_band() {
        let mut im = ImageU8::new(8, 7);
        draw_border3(&mut im);
        assert_eq!(im.at(3, 3), 0);
        assert_eq!(im.at(4, 3), 0);
        assert_eq!(im.at(2, 3), 255);
        assert_eq!(im.at(5, 3), 255);
        let mut im = ImageU8::new(2, 1);
        draw_border3(&mut im);
        assert_eq!(im.data(), &[255, 255]);
    }
}
