//! Reduction of decoded images to the single 8-bit channel the C++ program
//! holds (parity ledger V1 and V2). Ported from `core/Image.cpp` and
//! `core/CineVideo.cpp`, with the arithmetic of the OpenCV 4.13 routines they
//! call (`Mat::convertTo`, `cvtColor`).

use crate::GreyWeights;

/// `Mat::convertTo(dst, CV_8U, 2^-shift)` on one 16-bit value: the product is
/// formed in single precision (exact for a power of two), rounded to nearest
/// with ties to even (`cvRound`), and saturated to 255.
///
/// Ported from OpenCV `modules/core/src/convert_scale.simd.hpp`, `cvt_32f`
/// instantiated for `16u8u` (`DEF_CVT_SCALE_FUNC(16u8u, cvt_32f, ushort, uchar, float)`).
/// A shift of 32 or more gives a scale that rounds every value to 0.
#[inline]
pub fn scale_u16_to_u8(value: u16, shift: u32) -> u8 {
    if shift == 0 {
        return value.min(255) as u8;
    }
    if shift > 16 {
        // value * 2^-shift < 0.5 for every 16-bit value.
        return 0;
    }
    let v = u32::from(value);
    let q = v >> shift;
    let r = v & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let up = r > half || (r == half && q & 1 == 1);
    (q + u32::from(up)).min(255) as u8
}

/// Reduces 16-bit pixels with `convertTo(CV_8U, 2^-shift)`.
///
/// Ported from core/CineVideo.cpp, CineVideo::setActiveFrame (`convertTo(imageWithData, CV_8U, mod)`
/// with `mod = 1 / 2^(RealBPP - 8)`).
pub fn scale16_to_gray8(data: &[u16], shift: u32) -> Vec<u8> {
    data.iter().map(|&v| scale_u16_to_u8(v, shift)).collect()
}

/// Weights of `cvtColor(COLOR_RGB2GRAY)` for 8-bit data in OpenCV 4.13: 15-bit
/// fixed point (`RY15`, `GY15`, `BY15` in `imgproc/src/color.simd_helpers.hpp`).
const RY15: u32 = 9798;
const GY15: u32 = 19235;
const BY15: u32 = 3735;

/// The grey value the C++ computes for one pixel whose true colour is `(r, g, b)`.
///
/// The C++ calls `cvtColor(image_color, image, COLOR_RGB2GRAY)` on data that OpenCV
/// decoded in B, G, R order (ledger V2), so the blue sample gets the red weight and the
/// red sample the blue weight: `(b * 9798 + g * 19235 + r * 3735 + 2^14) >> 15`.
///
/// Ported from core/Image.cpp, Image::Image (`cvtColor(image_color, image, cv::COLOR_RGB2GRAY)`),
/// with the arithmetic of OpenCV `imgproc/src/color_rgb.simd.hpp`, `RGB2Gray<uchar>`.
#[inline]
pub fn bgr_as_rgb_to_gray(r: u8, g: u8, b: u8) -> u8 {
    let sum = u32::from(b) * RY15 + u32::from(g) * GY15 + u32::from(r) * BY15 + (1 << 14);
    (sum >> 15) as u8
}

/// The correct grey value of `(r, g, b)` with the same 15-bit weights:
/// `(r * 9798 + g * 19235 + b * 3735 + 2^14) >> 15` (OpenCV's `COLOR_BGR2GRAY` on
/// B, G, R data). Used unless XMALab 3's behaviour is asked for (ledger V2).
///
/// Ported from OpenCV `imgproc/src/color_rgb.simd.hpp`, `RGB2Gray<uchar>`.
#[inline]
pub fn rgb_to_gray(r: u8, g: u8, b: u8) -> u8 {
    let sum = u32::from(r) * RY15 + u32::from(g) * GY15 + u32::from(b) * BY15 + (1 << 14);
    (sum >> 15) as u8
}

/// The grey value of `(r, g, b)` with the chosen weights.
#[inline]
pub fn gray_of(r: u8, g: u8, b: u8, weights: GreyWeights) -> u8 {
    match weights {
        GreyWeights::Correct => rgb_to_gray(r, g, b),
        GreyWeights::Swapped => bgr_as_rgb_to_gray(r, g, b),
    }
}

/// [`gray_of`] over interleaved samples: `channels` samples per pixel, the first
/// three being R, G, B (any further sample, alpha, is ignored).
///
/// Ported from core/Image.cpp, Image::Image (colour branch).
pub fn rgb_to_gray8(data: &[u8], channels: usize, weights: GreyWeights) -> Vec<u8> {
    assert!(
        channels >= 3,
        "colour data needs at least 3 samples per pixel"
    );
    data.chunks_exact(channels)
        .map(|p| gray_of(p[0], p[1], p[2], weights))
        .collect()
}

/// 16-bit colour to grey as `core/Image.cpp` does it: each sample reduced with
/// `convertTo(CV_8UC3, 1.0 / 256.0)` first, then [`gray_of`].
///
/// Ported from core/Image.cpp, Image::Image (16-bit colour branch).
pub fn rgb16_to_gray8(data: &[u16], channels: usize, weights: GreyWeights) -> Vec<u8> {
    assert!(
        channels >= 3,
        "colour data needs at least 3 samples per pixel"
    );
    data.chunks_exact(channels)
        .map(|p| {
            gray_of(
                scale_u16_to_u8(p[0], 8),
                scale_u16_to_u8(p[1], 8),
                scale_u16_to_u8(p[2], 8),
                weights,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_round_to_even() {
        assert_eq!(scale_u16_to_u8(128, 8), 0);
        assert_eq!(scale_u16_to_u8(129, 8), 1);
        assert_eq!(scale_u16_to_u8(384, 8), 2);
        assert_eq!(scale_u16_to_u8(383, 8), 1);
        assert_eq!(scale_u16_to_u8(65535, 8), 255);
        assert_eq!(scale_u16_to_u8(65407, 8), 255);
        assert_eq!(scale_u16_to_u8(4095, 4), 255);
        assert_eq!(scale_u16_to_u8(4096 + 8, 4), 255);
        assert_eq!(scale_u16_to_u8(300, 0), 255);
        assert_eq!(scale_u16_to_u8(65535, 17), 0);
    }

    #[test]
    fn equal_channels_keep_their_value() {
        for v in 0..=255u8 {
            assert_eq!(bgr_as_rgb_to_gray(v, v, v), v);
        }
    }

    #[test]
    fn correct_weights_put_red_on_red() {
        assert_eq!(rgb_to_gray(255, 0, 0), 76);
        assert_eq!(rgb_to_gray(0, 0, 255), 29);
        assert_eq!(gray_of(255, 0, 0, GreyWeights::Swapped), 29);
        for v in 0..=255u8 {
            assert_eq!(rgb_to_gray(v, v, v), v);
        }
    }

    #[test]
    fn red_and_blue_are_swapped() {
        // Pure red gets the blue weight 0.114, pure blue the red weight 0.299.
        assert_eq!(bgr_as_rgb_to_gray(255, 0, 0), 29);
        assert_eq!(bgr_as_rgb_to_gray(0, 0, 255), 76);
    }
}
