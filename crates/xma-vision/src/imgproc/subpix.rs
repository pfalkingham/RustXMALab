//! `cv::getRectSubPix` for an 8-bit image and an 8-bit patch and for a float image and a
//! float patch, and XMALab's two `Image::getSubImage` wrappers on top of each (the template
//! and the search region of the 2D tracker, the detection window). The float form is the
//! one images deeper than 8 bits take (parity ledger V1, `crate::depth`).
//!
//! The patch is sampled bilinearly with OpenCV's 16-bit fixed-point weights. Pixels outside
//! the image take the value of the nearest edge (OpenCV's own replication, which also
//! interpolates along the edge). Tested against `getrectsubpix-*` in the primitive
//! oracle; where Intel IPP gives a different answer, the target is the `.noipp` output (the
//! C++ reference build has no IPP).

use crate::image::{GreySource, Image, ImageF32, ImageU8};

/// `SUBPIX_SHIFT` in `samplers.cpp`.
const SUBPIX_SHIFT: i32 = 16;

/// The bound to which [`get_rect_sub_pix`] holds a window origin outside the `int` range
/// (no effect on an origin in it).
const IP_LIMIT: i64 = 1 << 40;

/// `cvFloor(c)` and the fraction `c - cvFloor(c)` of a window origin. For an origin whose
/// window is inside the `int` range (every one XMALab passes; [`window_in_int_range`])
/// these are OpenCV's values. An origin beyond +-2^40, infinite or NaN (undefined in the
/// C++) is held to +-2^40 (0 for NaN) with a fraction of 0, so that the edge replication
/// computes without overflow: the patch is then the replicated edge.
fn split_origin(c: f32) -> (i64, f32) {
    let ip = c.floor() as i64;
    if c.is_finite() && (-IP_LIMIT..=IP_LIMIT).contains(&ip) {
        (ip, c - ip as f32)
    } else {
        (ip.clamp(-IP_LIMIT, IP_LIMIT), 0.0)
    }
}

/// The smallest window half-width [`max_window_half`] allows, whatever the image (the
/// half-width up to which the refinement also reproduces the C++ on small images,
/// `crate::detection::REFINE_MAX_HALF_WIDTH_CPP`).
pub const WINDOW_HALF_FLOOR: i64 = 1024;

/// The largest half-width `size` of a `2 * size + 1` window that [`get_sub_image`],
/// [`get_sub_image_at`] and their float forms cut from a `width` by `height` image, and the
/// largest window the detection and the tracker work on (parity ledger A167): the larger of
/// the image's larger side and [`WINDOW_HALF_FLOOR`] (1024). A window beyond it is mostly the
/// replicated image edge; the C++ allocates it whatever its size (`(2 size + 1)^2` pixels for
/// any `int` size, terabytes for a size of 10^6), so here such a window is not cut and the
/// callers find nothing. The largest window cut is therefore `(2 max(L, 1024) + 1)^2` pixels
/// (`L` the larger side): 16.8 MB as `f32` for an image up to 1024 pixels wide.
#[must_use]
pub fn max_window_half(width: usize, height: usize) -> i64 {
    i64::try_from(width.max(height))
        .unwrap_or(i64::MAX)
        .max(WINDOW_HALF_FLOOR)
}

/// The largest patch side [`get_rect_sub_pix`] and [`get_rect_sub_pix_32f`] compute for a
/// `width` by `height` image: `2 max_window_half + 1` ([`max_window_half`]).
fn max_patch_side(width: usize, height: usize) -> usize {
    usize::try_from(2 * max_window_half(width, height) + 1).unwrap_or(usize::MAX)
}

/// True when the `int` arithmetic XMALab and OpenCV do for the `2 * size + 1` square whose
/// top-left pixel is `(off_x, off_y)` is defined: the corners (`off + size`, the far edge
/// `off + 2 * size + 1`) are `int`s, and so is `cvFloor` of the `f32` window origin
/// `getRectSubPix` computes from the `cv::Point2f` centre (and that origin plus the
/// window's width); that origin is also not `INT_MIN`, which OpenCV's `adjustRect` negates
/// (`rect.x = -ip.x`). Outside it the C++ overflows (undefined behaviour); the callers return
/// "nothing found" there (parity ledger A160). Defined for every `i64` origin and every size
/// (no overflow of its own).
#[must_use]
pub fn window_in_int_range(off_x: i64, off_y: i64, size: i32) -> bool {
    let n = 2 * i64::from(size) + 1;
    let fits = |off: i64| -> bool {
        let Some(far) = off.checked_add(n) else {
            return false;
        };
        if off < i64::from(i32::MIN) || far > i64::from(i32::MAX) {
            return false;
        }
        // `cv::Point2f(off + size)`, then `center - (n - 1) * 0.5f` and `cvFloor`.
        let c = (off + i64::from(size)) as f32;
        let ip = f64::from((c - (n - 1) as f32 * 0.5f32).floor());
        ip > f64::from(i32::MIN) && ip + n as f64 <= f64::from(i32::MAX)
    };
    fits(off_x) && fits(off_y)
}

/// [`window_in_int_range`] for a window given by its centre (`getSubImage(Mat&, int, double,
/// double)`, which converts the centre to `cv::Point2f`): false for a centre that is not
/// finite, and for an origin of `INT_MIN` (negated by `adjustRect`).
#[must_use]
pub fn centre_in_int_range(x: f64, y: f64, size: i32) -> bool {
    let n = 2 * i64::from(size) + 1;
    let fits = |v: f64| -> bool {
        let c = v as f32;
        if !c.is_finite() {
            return false;
        }
        let ip = f64::from((c - (n - 1) as f32 * 0.5f32).floor());
        ip > f64::from(i32::MIN) && ip + n as f64 <= f64::from(i32::MAX)
    };
    fits(x) && fits(y)
}

/// `scale_fixpt`: a weight in 16-bit fixed point, `cvRound(a * 65536)`.
///
/// Ported from imgproc/src/samplers.cpp, scale_fixpt::operator().
fn scale_fixpt(a: f32) -> i32 {
    (a * (1 << SUBPIX_SHIFT) as f32).round_ties_even() as i32
}

/// `cast_8u`: back from fixed point with rounding; the C cast to `uchar` keeps the low
/// 8 bits.
///
/// Ported from imgproc/src/samplers.cpp, cast_8u::operator().
fn cast_8u(a: i32) -> u8 {
    ((a + (1 << (SUBPIX_SHIFT - 1))) >> SUBPIX_SHIFT) as u8
}

/// The part of the window that lies inside the image (`cv::Rect` of `adjustRect`) and the
/// offset, in pixels from the image's first pixel, of the pointer `adjustRect` returns
/// (which may point before the row or before the image).
///
/// Ported from imgproc/src/samplers.cpp, adjustRect (one channel).
fn adjust_rect(
    width: i64,
    height: i64,
    step: i64,
    win_w: i64,
    win_h: i64,
    ip: (i64, i64),
) -> (i64, [i64; 4]) {
    let mut src: i64 = 0;
    // `rect.x`, `rect.y`, `rect.width`, `rect.height` of the C++.
    let rx = if ip.0 >= 0 {
        src += ip.0;
        0
    } else {
        (-ip.0).min(win_w)
    };
    let rw = if ip.0 < width - win_w {
        win_w
    } else {
        let mut rw = width - ip.0 - 1;
        if rw < 0 {
            src += rw;
            rw = 0;
        }
        assert!(rw <= win_w);
        rw
    };
    let ry = if ip.1 >= 0 {
        src += ip.1 * step;
        0
    } else {
        -ip.1
    };
    let rh = if ip.1 < height - win_h {
        win_h
    } else {
        let mut rh = height - ip.1 - 1;
        if rh < 0 {
            src += rh * step;
            rh = 0;
        }
        rh
    };
    (src - rx, [rx, ry, rw, rh])
}

/// The arithmetic of one `getRectSubPix_Cn_` instantiation: the type the weights and sums
/// are in, how a weight is made from its `float` value, and how a sum becomes a pixel.
trait Bilinear {
    /// `WT` of the C++ (`int` for 8-bit, `float` for float).
    type W: Copy + std::ops::Add<Output = Self::W> + std::ops::Mul<Output = Self::W>;
    /// `DT` of the C++.
    type Out: Copy + Default;
    /// `ScaleOp`: the weight from its `float` value.
    fn weight(a: f32) -> Self::W;
    /// `CastOp`: the pixel from a weighted sum.
    fn cast(s: Self::W) -> Self::Out;
}

/// `getRectSubPix_Cn_<uchar, uchar, int, scale_fixpt, cast_8u>`.
struct Fixed8;

impl Bilinear for Fixed8 {
    type W = i32;
    type Out = u8;
    fn weight(a: f32) -> i32 {
        scale_fixpt(a)
    }
    fn cast(s: i32) -> u8 {
        cast_8u(s)
    }
}

/// `getRectSubPix_Cn_<float, float, float, nop<float>, nop<float>>`.
struct Float32;

impl Bilinear for Float32 {
    type W = f32;
    type Out = f32;
    fn weight(a: f32) -> f32 {
        a
    }
    fn cast(s: f32) -> f32 {
        s
    }
}

/// The body of [`get_rect_sub_pix`] and [`get_rect_sub_pix_32f`]: the patch of a `width` by
/// `height` image whose pixel at index `i` (row-major) is `px(i)`, in `B`'s arithmetic. Each
/// sample is `p00 a11 + p01 a12 + p10 a21 + p11 a22` (or `p0 b1 + p1 b2` along a replicated
/// edge), evaluated left to right in `B::W`.
///
/// Ported from imgproc/src/samplers.cpp, cv::getRectSubPix and getRectSubPix_Cn_ (generic
/// code, one channel).
///
/// # Panics
/// If the image is empty.
fn rect_sub_pix<B: Bilinear>(
    width: usize,
    height: usize,
    px: impl Fn(usize) -> B::W,
    patch_width: usize,
    patch_height: usize,
    center: (f32, f32),
) -> Image<B::Out> {
    let (w, h) = (width as i64, height as i64);
    assert!(w > 0 && h > 0, "getRectSubPix of an empty image");
    let limit = max_patch_side(width, height);
    if patch_width > limit || patch_height > limit {
        return Image::new(0, 0);
    }
    let mut dst = Image::new(patch_width, patch_height);
    let step = w;
    let (win_w, win_h) = (patch_width as i64, patch_height as i64);

    let cx = center.0 - (patch_width as i32 - 1) as f32 * 0.5f32;
    let cy = center.1 - (patch_height as i32 - 1) as f32 * 0.5f32;
    let (ipx, a) = split_origin(cx);
    let (ipy, b) = split_origin(cy);
    let a11 = B::weight((1.0 - a) * (1.0 - b));
    let a12 = B::weight(a * (1.0 - b));
    let a21 = B::weight((1.0 - a) * b);
    let a22 = B::weight(a * b);
    let b1 = B::weight(1.0 - b);
    let b2 = B::weight(b);

    let px = |i: i64| -> B::W { px(usize::try_from(i).expect("read before image")) };

    if 0 <= ipx && ipx < w - win_w && 0 <= ipy && ipy < h - win_h {
        // Entirely inside the image.
        let mut src = ipy * step + ipx;
        for i in 0..patch_height {
            let row = dst.row_mut(i);
            for (j, d) in row.iter_mut().enumerate() {
                let j = j as i64;
                let s0 = px(src + j) * a11
                    + px(src + j + 1) * a12
                    + px(src + j + step) * a21
                    + px(src + j + step + 1) * a22;
                *d = B::cast(s0);
            }
            src += step;
        }
    } else {
        let (mut src, [rx, ry, rw, rh]) = adjust_rect(w, h, step, win_w, win_h, (ipx, ipy));
        for i in 0..win_h {
            let mut src2 = src + step;
            if i < ry || i >= rh {
                src2 -= step;
            }
            let row = dst.row_mut(i as usize);
            let s0 = px(src + rx) * b1 + px(src2 + rx) * b2;
            for d in &mut row[..rx as usize] {
                *d = B::cast(s0);
            }
            if rw < win_w {
                let s0 = px(src + rw) * b1 + px(src2 + rw) * b2;
                for d in &mut row[rw as usize..] {
                    *d = B::cast(s0);
                }
            }
            for j in rx..rw {
                let s0 = px(src + j) * a11
                    + px(src + j + 1) * a12
                    + px(src2 + j) * a21
                    + px(src2 + j + 1) * a22;
                row[j as usize] = B::cast(s0);
            }
            if i < rh {
                src = src2;
            }
        }
    }
    dst
}

/// `cv::getRectSubPix(image, Size(patch_width, patch_height), center, patch)` for an 8-bit
/// image and an 8-bit patch: the `patch_width` by `patch_height` window whose centre is at
/// `center` (pixel centres at integers), sampled bilinearly.
///
/// The centre is a `cv::Point2f`, as in OpenCV: a caller holding a `double` converts it to
/// `f32` first (see [`get_sub_image`]).
///
/// Ported from imgproc/src/samplers.cpp, cv::getRectSubPix and
/// getRectSubPix_Cn_<uchar, uchar, int, scale_fixpt, cast_8u> (the generic code; the IPP
/// branch is not ported).
///
/// A patch wider or higher than `2 max_window_half + 1` ([`max_window_half`]) is not
/// computed: the result is then an empty (0 by 0) image (parity ledger A167).
///
/// # Panics
/// If the image is empty.
pub fn get_rect_sub_pix(
    image: &ImageU8,
    patch_width: usize,
    patch_height: usize,
    center: (f32, f32),
) -> ImageU8 {
    let src_data = image.data();
    rect_sub_pix::<Fixed8>(
        image.width(),
        image.height(),
        |i| i32::from(src_data[i]),
        patch_width,
        patch_height,
        center,
    )
}

/// `Image::getSubImage(Mat&, int size, int off_x, int off_y)`: the `2 * size + 1` square
/// whose top-left pixel is `(off_x, off_y)`, that is centred on `(off_x + size,
/// off_y + size)`.
///
/// A `size` beyond [`max_window_half`] gives an empty (0 by 0) image (parity ledger A167).
///
/// Ported from core/Image.cpp, Image::getSubImage(cv::Mat&, int, int, int).
pub fn get_sub_image_at(image: &ImageU8, size: i32, off_x: i32, off_y: i32) -> ImageU8 {
    let n = usize::try_from(2 * i64::from(size) + 1).expect("negative sub-image size");
    get_rect_sub_pix(image, n, n, sub_image_centre(size, off_x, off_y))
}

/// `cv::Point2f(off_x + size, off_y + size)`: the sum in `i64`, which is the C++ `int` sum
/// wherever that is defined ([`window_in_int_range`]) and cannot overflow elsewhere.
fn sub_image_centre(size: i32, off_x: i32, off_y: i32) -> (f32, f32) {
    let s = i64::from(size);
    ((i64::from(off_x) + s) as f32, (i64::from(off_y) + s) as f32)
}

/// `Image::getSubImage(Mat&, int size, double x, double y)`: the `2 * size + 1` square
/// centred on `(x, y)`. The centre is converted to `f32` (a `cv::Point2f`) before sampling,
/// as in the C++.
///
/// A `size` beyond [`max_window_half`] gives an empty (0 by 0) image (parity ledger A167).
///
/// Ported from core/Image.cpp, Image::getSubImage(cv::Mat&, int, double, double).
pub fn get_sub_image(image: &ImageU8, size: i32, x: f64, y: f64) -> ImageU8 {
    let n = usize::try_from(2 * i64::from(size) + 1).expect("negative sub-image size");
    get_rect_sub_pix(image, n, n, (x as f32, y as f32))
}

/// `cv::getRectSubPix(image, Size(patch_width, patch_height), center, patch)` for a float
/// image and a float patch (`CV_32F` to `CV_32F`): the window sampled bilinearly with `f32`
/// weights, `p00 a11 + p01 a12 + p10 a21 + p11 a22` evaluated left to right with a rounding
/// after each operation, edges replicated as in the 8-bit form. At a whole-pixel centre the
/// weights are 1 and 0, so the patch is an exact copy.
///
/// `image` is any [`GreySource`]: an [`ImageF32`], or a 16-bit image on the 8-bit scale
/// (`crate::depth::on_8bit_scale`), which gives what OpenCV gives for the float image of
/// those values.
///
/// Ported from imgproc/src/samplers.cpp, cv::getRectSubPix and
/// getRectSubPix_Cn_<float, float, float, nop<float>, nop<float>> (generic code; the IPP
/// branch is not ported). The float counterpart of [`get_rect_sub_pix`].
///
/// A patch wider or higher than `2 max_window_half + 1` ([`max_window_half`]) is not
/// computed: the result is then an empty (0 by 0) image (parity ledger A167).
///
/// # Panics
/// If the image is empty.
pub fn get_rect_sub_pix_32f<S: GreySource + ?Sized>(
    image: &S,
    patch_width: usize,
    patch_height: usize,
    center: (f32, f32),
) -> ImageF32 {
    rect_sub_pix::<Float32>(
        image.width(),
        image.height(),
        |i| image.value(i),
        patch_width,
        patch_height,
        center,
    )
}

/// [`get_sub_image_at`] for a float image: the `2 * size + 1` square whose top-left pixel is
/// `(off_x, off_y)`, an exact copy (edges replicated).
///
/// A `size` beyond [`max_window_half`] gives an empty (0 by 0) image (parity ledger A167).
///
/// Ported from core/Image.cpp, Image::getSubImage(cv::Mat&, int, int, int), on a `CV_32F`
/// image ([`get_rect_sub_pix_32f`]).
pub fn get_sub_image_at_32f<S: GreySource + ?Sized>(
    image: &S,
    size: i32,
    off_x: i32,
    off_y: i32,
) -> ImageF32 {
    let n = usize::try_from(2 * i64::from(size) + 1).expect("negative sub-image size");
    get_rect_sub_pix_32f(image, n, n, sub_image_centre(size, off_x, off_y))
}

/// [`get_sub_image`] for a float image: the `2 * size + 1` square centred on `(x, y)`
/// (converted to `f32`), sampled bilinearly in `f32` without rounding to whole grey levels.
///
/// A `size` beyond [`max_window_half`] gives an empty (0 by 0) image (parity ledger A167).
///
/// Ported from core/Image.cpp, Image::getSubImage(cv::Mat&, int, double, double), on a
/// `CV_32F` image ([`get_rect_sub_pix_32f`]).
pub fn get_sub_image_32f<S: GreySource + ?Sized>(image: &S, size: i32, x: f64, y: f64) -> ImageF32 {
    let n = usize::try_from(2 * i64::from(size) + 1).expect("negative sub-image size");
    get_rect_sub_pix_32f(image, n, n, (x as f32, y as f32))
}
