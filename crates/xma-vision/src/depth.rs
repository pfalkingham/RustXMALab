//! Images deeper than 8 bits (parity ledger V1): the scale that puts them on the 8-bit
//! scale, and the reduction to 8 bits that XMALab 3 applies.
//!
//! # The full-depth rule
//!
//! A `bit_depth`-bit image (9 to 16 bits carry information, held in `u16`) enters the
//! detection, the tracker and the polynomial refinement as floating-point values on the
//! 8-bit scale, `min(value, 2^bit_depth - 1) / 2^(bit_depth - 8)`, without rounding
//! ([`on_8bit_scale`], [`ScaledU16`]; the division by a power of two is exact in `f32` for
//! every 16-bit value). A value above the depth's largest, `2^bit_depth - 1`, is read as
//! that largest value, as the reduction to 8 bits saturates it to 255; this happens in one
//! place, [`ScaledU16`], through which every deep entry point reads its pixels. From that
//! point each algorithm is the 8-bit algorithm unchanged, with the same later quantisation
//! points: the normalised difference image of the contour methods is still converted to 8
//! bits before the median and the threshold, the blob detector's binary images and every
//! mask stay 8-bit. The entry points are `detection::detect_point_u16`,
//! `detection::detect_point_preview_u16`, `detection::refine_point_polynomial_fit_u16`,
//! `tracking::cut_template_u16` and `tracking::track_step_u16_with` (with
//! `tracking::track_step_surfaces_u16_with`). Like every entry point of the detection and the
//! tracker, each takes the mode from its caller (`DetectOptions`, `DetectCompat` or
//! `TrackCompat`): none of them chooses XMALab 3's behaviour by itself.
//!
//! Parameters are not rescaled: grey levels (the threshold offset, the blob detector's
//! thresholds) are on the 0..255 scale, sizes in pixels. The threshold offset of the
//! contour methods acts on the normalised difference after its conversion to 8 bits, so it
//! means the same on both paths. The blob detector's thresholds differ by half a grey
//! level: for a threshold `t`, the image reduced to 8 bits keeps the pixels whose value on
//! the 8-bit scale is at least `floor(t) + 0.5` (a reduced value `round(v)` above
//! `floor(t)`, the 8-bit `cv::threshold`), the deep path the pixels whose value is above `t`
//! (the float `cv::threshold`). For the whole-number thresholds the blob detector steps
//! through, the deep path keeps pixels half a grey level darker than the reduced path does.
//!
//! An 8-bit image promoted to a deeper one (each value times `2^(bit_depth - 8)`) gives on
//! the 8-bit scale exactly its own values, so every stage that reads them at whole-pixel
//! positions computes what the 8-bit path computes; where the two paths differ by design
//! (a sub-pixel cut kept in floating point, the float form of `matchTemplate`), the entry
//! points say so.
//!
//! In the XMALab 3 compatibility mode a deeper image is reduced to 8 bits first, as the C++
//! does ([`reduce_to_u8`]), and the 8-bit entry points run on the result.
//!
//! # What depth gains, and where it does not
//!
//! Processing at depth is not uniformly better than processing the image reduced to 8
//! bits. Measured on synthetic 12-bit images (`crates/xma-vision/tests/fulldepth_measured.rs`,
//! 600 trials per cell: a dark disc of radius 4 at random sub-pixel positions, contrast 1, 4
//! or 16 grey levels, noise 0, 0.5 or 2 levels, flat or sloping background):
//!
//! - Depth helps where the marker's contrast is near the size of an 8-bit step, or the noise
//!   is half a grey level or more: the contour methods (0, 5) miss far less on faint marks
//!   with noise (method 0, contrast 1, noise 0.5, sloped: 49 misses against 141) and are
//!   more precise (contrast 4, noise 0.5: 0.39 against 0.52 px); method 3 on a flat
//!   background is much more precise (contrast 4: 0.048 against 0.555 px); the refinement
//!   without noise and the tracker on faint markers are clearly better (tracker, contrast 1,
//!   noise 0.5: 84 misses against 275).
//! - Depth does not help on strong markers without noise, and at noise of 2 levels the two
//!   paths are within the trials' scatter. For the contour methods on a noise-free marker
//!   the reduced path's error depends on the background's grey level relative to the 8-bit
//!   steps (method 0, contrast 16: from 0.16 to 0.28 px as the background goes from 120.37
//!   to 120.0), while the deep path's does not (0.22 to 0.23 px); averaged over background
//!   levels the deep path is 0.005 to 0.016 px behind, and in a single
//!   scene it can be up to 0.07 px behind (contrast 16, sloped: 0.24 against 0.17 px) or
//!   ahead. Neither the float blur nor the normalisation nor the conversion of the
//!   normalised difference to 8 bits causes this (swapping each between the paths changes
//!   the error by less than 0.01 px); the 3 x 3 median and the threshold on a noise-free,
//!   smoothly varying difference image set the deep path's error (without the median it
//!   would be 0.14 px, but the median is what keeps the methods working in noise: method 0,
//!   contrast 4, noise 2, 42 misses with it and 246 without). On a faint noise-free marker
//!   (contrast 1) the reduction to 8 bits turns the marker into a two-level image, already
//!   thresholded, which the contour methods locate better (0.13 against 0.26 px, flat) as
//!   long as the reduction keeps the marker at all (with a random background level it loses
//!   it in 44 of 600 trials).
//! - The corner method (3) on a shaded background is worse at depth: on a background sloping
//!   by 0.013 and -0.009 grey levels per pixel, without noise, a marker of 1 grey level is
//!   missed in 585 of 600 trials at depth and in none reduced, and a marker of 4 levels is
//!   found with 0.64 px mean error against 0.33 px. `cornerSubPix` is pulled by the
//!   background's gradient, which the reduction to 8 bits flattens into steps (the independent
//!   review found OpenCV's own float `cornerSubPix` on the deep patch gives the same biased
//!   points).
//!
//! # Out-of-range and misdeclared data
//!
//! The rule takes the declared bit depth at its word. Two kinds of data break it:
//!
//! - Samples stored in the high bits of the 16-bit word (a 12-bit sensor whose values are
//!   shifted left by 4), declared as 12-bit: almost every value is above `2^12 - 1` and is
//!   read as the largest value, so the image is nearly uniformly white. The contour methods
//!   (0, 2, 5) then return the whole search window as their contour, a wrong point; the blob
//!   and corner methods and the refinement find nothing. The XMALab 3 reduction saturates
//!   the same way. Such data has to be declared as 16-bit.
//! - A 16-bit file holding fewer bits than its container (12-bit data in a 16-bit PNG or
//!   TIFF, declared as 16-bit): the values read as nearly black on the 8-bit scale (12-bit
//!   data from 0 to 16), as XMALab 3 reads them (it divides by 256 whatever the true depth).
//!   At depth the contour and corner methods give the same result as with the right
//!   declaration (their normalisation and relative quality level remove the scale); the
//!   blob methods (1, 6), whose thresholds start at 10 grey levels, find nothing, at depth
//!   as in XMALab.
//!
//! Both are checked in `crates/xma-vision/tests/fulldepth.rs` (`misdeclared_depth`).
//!
//! What a user should check: the largest value in a frame against `2^bit_depth - 1`. A
//! largest value well below it (under a sixteenth for 12-bit data in a 16-bit file) means
//! the declared depth is too high; values above it mean it is too low (or the data is
//! shifted into the high bits).

use crate::image::{ImageF32, ImageU8, ImageU16, ScaledU16};
use crate::imgproc::arith::u16_to_u8_scaled;

/// `2^-(bit_depth - 8)`: the factor that puts a `bit_depth`-bit value on the 8-bit scale.
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[must_use]
pub fn scale_to_8bit(bit_depth: u8) -> f32 {
    assert!(
        (8..=16).contains(&bit_depth),
        "bit depth {bit_depth} outside 8..=16"
    );
    1.0 / f32::from(1u16 << (bit_depth - 8))
}

/// `2^bit_depth - 1`: the largest value of a `bit_depth`-bit image. The deep path reads any
/// larger value as this one.
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[must_use]
pub fn max_value(bit_depth: u8) -> u16 {
    assert!(
        (8..=16).contains(&bit_depth),
        "bit depth {bit_depth} outside 8..=16"
    );
    u16::MAX >> (16 - bit_depth)
}

/// The image read on the 8-bit scale: `min(value, 2^bit_depth - 1) / 2^(bit_depth - 8)`
/// ([`scale_to_8bit`], [`max_value`]); see the module documentation. Every deep entry point
/// reads its pixels through this.
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[must_use]
pub fn on_8bit_scale(image: &ImageU16, bit_depth: u8) -> ScaledU16<'_> {
    ScaledU16 {
        image,
        scale: scale_to_8bit(bit_depth),
        max: max_value(bit_depth),
    }
}

/// The whole image as `f32` on the 8-bit scale, as [`on_8bit_scale`] reads it:
/// `min(value, 2^bit_depth - 1) / 2^(bit_depth - 8)`, exact.
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[must_use]
pub fn to_f32_8bit_scale(image: &ImageU16, bit_depth: u8) -> ImageF32 {
    let s = scale_to_8bit(bit_depth);
    let m = max_value(bit_depth);
    image.map(|v| f32::from(v.min(m)) * s)
}

/// The 8-bit image XMALab 3 makes of a `bit_depth`-bit one, for the compatibility mode:
/// `convertTo(CV_8U, 2^-(bit_depth - 8))`, that is `value / 2^(bit_depth - 8)` rounded half
/// to even and saturated to 255 (1/256 for 16-bit stills in `core/Image.cpp`,
/// `1 / 2^(RealBPP - 8)` for `.cine` in `core/CineVideo.cpp`). For bit depth 8 the values
/// are taken as they are (saturated). The same values as `xma_video::Frame::to_gray8` for
/// every 16-bit value and depth (checked exhaustively in
/// `crates/xma-engine/tests/fulldepth_reduction.rs`).
///
/// Ported from core/Image.cpp, Image::Image, and core/CineVideo.cpp,
/// CineVideo::setActiveFrame, with [`u16_to_u8_scaled`] (OpenCV `cvtScale16u8u`).
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[must_use]
pub fn reduce_to_u8(image: &ImageU16, bit_depth: u8) -> ImageU8 {
    u16_to_u8_scaled(image, f64::from(scale_to_8bit(bit_depth)), 0.0)
}

/// An 8-bit image promoted to `bit_depth` bits: each value times `2^(bit_depth - 8)`. On the
/// 8-bit scale it reads back as exactly its own values; used to check the deep path against
/// the 8-bit path.
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[must_use]
pub fn promote_u8(image: &ImageU8, bit_depth: u8) -> ImageU16 {
    assert!(
        (8..=16).contains(&bit_depth),
        "bit depth {bit_depth} outside 8..=16"
    );
    let shift = bit_depth - 8;
    image.map(|v| u16::from(v) << shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GreySource;

    #[test]
    fn scale_and_round_trip() {
        assert_eq!(scale_to_8bit(8), 1.0);
        assert_eq!(scale_to_8bit(12), 1.0 / 16.0);
        assert_eq!(scale_to_8bit(16), 1.0 / 256.0);
        let im = ImageU8::from_vec(256, 1, (0..=255).collect());
        for d in 8..=16 {
            let p = promote_u8(&im, d);
            let s = on_8bit_scale(&p, d);
            for i in 0..256 {
                assert_eq!(s.value(i), i as f32);
            }
            assert_eq!(reduce_to_u8(&p, d), im);
            assert_eq!(to_f32_8bit_scale(&p, d), im.map(f32::from));
        }
    }

    #[test]
    fn reduction_rounds_half_to_even_and_saturates() {
        let im = ImageU16::from_vec(5, 1, vec![8, 24, 4087, 4088, 65535]);
        assert_eq!(reduce_to_u8(&im, 12).data(), &[0, 2, 255, 255, 255]);
        let im = ImageU16::from_vec(4, 1, vec![128, 384, 383, 65535]);
        assert_eq!(reduce_to_u8(&im, 16).data(), &[0, 2, 1, 255]);
    }

    #[test]
    fn values_above_the_depth_saturate() {
        assert_eq!(max_value(8), 255);
        assert_eq!(max_value(12), 4095);
        assert_eq!(max_value(16), u16::MAX);
        let im = ImageU16::from_vec(4, 1, vec![4094, 4095, 4096, u16::MAX]);
        let s = on_8bit_scale(&im, 12);
        let top = 4095.0 / 16.0;
        assert_eq!(
            [s.value(0), s.value(1), s.value(2), s.value(3)],
            [4094.0 / 16.0, top, top, top]
        );
        assert_eq!(
            to_f32_8bit_scale(&im, 12).data(),
            &[4094.0 / 16.0, top, top, top]
        );
        // The XMALab 3 reduction makes all three 255 as well.
        assert_eq!(reduce_to_u8(&im, 12).data(), &[255, 255, 255, 255]);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn depth_out_of_range() {
        let _ = scale_to_8bit(17);
    }
}
