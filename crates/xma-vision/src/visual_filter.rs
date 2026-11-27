//! XMALab's visual filter: the contrast enhancement applied to the image shown in the
//! marker-tracking workspace. Display only; the algorithms never see it.
//!
//! Ported from `processing/FilterImage.cpp`, `FilterImage::run`, with the OpenCV 4.13
//! routines it calls on an 8-bit grey image:
//!
//! 1. `GaussianBlur(src, blur, Size(k, k), gsigma)` with `k = 2 * krad + 1` and the default
//!    border (`BORDER_REFLECT_101`): the bit-exact fixed-point path every 8-bit blur takes
//!    (`modules/imgproc/src/smooth.dispatch.cpp`, `getGaussianKernelBitExact`,
//!    `getGaussianKernelFixedPoint_ED` with 8 fraction bits; `smooth.simd.hpp`,
//!    `fixedSmoothInvoker` with `hlineSmoothONa_yzy_a` and `vlineSmoothONa_yzy_a`). Rows
//!    are filtered into 16-bit fixed point (8 fraction bits), columns into 32-bit (16
//!    fraction bits), rounded with `+ 2^15 >> 16`. Every sum is exact in integers, so the
//!    order of the additions does not matter.
//! 2. `addWeighted(src, img_wt, blur, blur_wt, 0, out)`: per pixel in `f32`,
//!    `fma(src, img_wt, fma(blur, blur_wt, 0))` rounded half to even and saturated, as
//!    OpenCV's vectorised loop computes it on a machine with FMA (the reference machine;
//!    without FMA OpenCV rounds the products separately, which can change a pixel by one).
//! 3. A gamma lookup table: `lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma)
//!    * 255.0f)`.
//!
//! The settings are `VisualFilter_krad` (17), `VisualFilter_gsigma` (10),
//! `VisualFilter_img_wt` (3.6), `VisualFilter_blur_wt` (-2.9) and `VisualFilter_gamma`
//! (0.4), read as `float` (`core/Settings.cpp`).
//!
//! The Gaussian kernel is computed in `f64` with the platform's `exp` where OpenCV uses its
//! software `softdouble`; the kernel is then rounded to multiples of 1/256, so a difference
//! in the last bit of `exp` would matter only at an exact rounding tie. Compared with OpenCV
//! on the 8-bit GaussianBlur primitive cases and on 10 pig frames
//! (`oracle/py/display_visual_filter.py`, `oracle/py/DISPLAY_CHECKS.md`): identical.

/// The largest kernel radius applied: beyond XMALab's spin box (99), far beyond any image.
pub const MAX_KRAD: i32 = 1000;

/// The visual filter settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisualFilter {
    /// `VisualFilter_krad`: the kernel is `2 * krad + 1` pixels wide.
    pub krad: i32,
    /// `VisualFilter_gsigma`.
    pub gsigma: f32,
    /// `VisualFilter_img_wt`: weight of the image.
    pub img_wt: f32,
    /// `VisualFilter_blur_wt`: weight of the blurred image.
    pub blur_wt: f32,
    /// `VisualFilter_gamma`.
    pub gamma: f32,
}

impl Default for VisualFilter {
    /// The C++ defaults (`core/Settings.cpp`).
    fn default() -> Self {
        Self {
            krad: 17,
            gsigma: 10.0,
            img_wt: 3.6,
            blur_wt: -2.9,
            gamma: 0.4,
        }
    }
}

impl VisualFilter {
    /// Filters an 8-bit grey image of `width × height` pixels.
    ///
    /// Ported from `processing/FilterImage.cpp`, `FilterImage::run`. XMALab's dialog keeps
    /// `krad` in 1 to 99; any value is taken here without overflow or an unbounded kernel: a
    /// radius below 0 is 0 (no blur) and one above [`MAX_KRAD`] is [`MAX_KRAD`] (parity
    /// ledger U164). Non-finite weights, sigma or gamma give a defined image (OpenCV's
    /// saturating casts); they cannot be entered in the dialog.
    #[must_use]
    pub fn apply(&self, src: &[u8], width: usize, height: usize) -> Vec<u8> {
        let k = (2 * self.krad.clamp(0, MAX_KRAD) + 1) as usize;
        let blur = gaussian_blur_u8(src, width, height, k, f64::from(self.gsigma));
        let mut out = add_weighted_u8(
            src,
            f64::from(self.img_wt),
            &blur,
            f64::from(self.blur_wt),
            0.0,
        );
        let lut = gamma_lut(self.gamma);
        for v in &mut out {
            *v = lut[*v as usize];
        }
        out
    }
}

pub use crate::imgproc::blur::{gaussian_kernel, gaussian_kernel_fixed};

/// `cv::GaussianBlur` of an 8-bit grey image with a square `k × k` kernel (`k` odd) and
/// `BORDER_REFLECT_101`, OpenCV's bit-exact fixed-point path. See the module documentation;
/// the code is [`crate::imgproc::blur::gaussian_blur_u8_raw`].
#[must_use]
pub fn gaussian_blur_u8(src: &[u8], width: usize, height: usize, k: usize, sigma: f64) -> Vec<u8> {
    crate::imgproc::blur::gaussian_blur_u8_raw(src, width, height, k, sigma)
}

/// `cv::addWeighted(a, alpha, b, beta, gamma, out)` on 8-bit images (FMA form, see the
/// module documentation).
#[must_use]
pub fn add_weighted_u8(a: &[u8], alpha: f64, b: &[u8], beta: f64, gamma: f64) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "image sizes");
    let (al, be, ga) = (alpha as f32, beta as f32, gamma as f32);
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let v = f32::from(x).mul_add(al, f32::from(y).mul_add(be, ga));
            v.round_ties_even().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// The gamma table of `FilterImage::run`.
#[must_use]
pub fn gamma_lut(gamma: f32) -> [u8; 256] {
    std::array::from_fn(|i| {
        let v = ((i as f64 / 255.0) as f32).powf(gamma) * 255.0f32;
        v.round_ties_even().clamp(0.0, 255.0) as u8
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_blur_matches_opencv_primitives() {
        let mut n = 0;
        for case in xma_oracle::cases("gaussianblur") {
            let src = case.image("src");
            let xma_oracle::ImageData::U8(s) = src.data else {
                continue;
            };
            let ks = case.arg("ksize");
            let k = ks[0].as_u64().unwrap_or(0) as usize;
            if ks[1].as_u64().unwrap_or(0) as usize != k {
                continue;
            }
            let sigma = case.arg_f64("sigmaX");
            if !case
                .arg("borderType")
                .as_str()
                .unwrap_or("")
                .contains("REFLECT_101")
            {
                continue;
            }
            let xma_oracle::ImageData::U8(want) = case.image("dst").data else {
                panic!("u8 output");
            };
            let got = gaussian_blur_u8(&s, src.cols, src.rows, k, sigma);
            assert_eq!(got, want, "{}", case.name);
            n += 1;
        }
        assert!(n > 0, "no 8-bit Gaussian blur case was usable");
    }

    #[test]
    fn fixed_kernel_sums_to_256() {
        let k = gaussian_kernel_fixed(&gaussian_kernel(35, 10.0));
        assert_eq!(k.iter().map(|&v| u32::from(v)).sum::<u32>(), 256);
        assert_eq!(k.len(), 35);
        assert_eq!(
            gaussian_kernel_fixed(&gaussian_kernel(3, 0.0)),
            vec![64, 128, 64]
        );
    }

    #[test]
    fn any_setting_gives_an_image() {
        let src: Vec<u8> = (0..23 * 17).map(|i| (i * 37 % 251) as u8).collect();
        for krad in [i32::MIN, -1, 0, 1, 99, 5000, i32::MAX] {
            for v in [
                0.0f32,
                -1.0,
                99.99,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ] {
                let f = VisualFilter {
                    krad,
                    gsigma: v,
                    img_wt: v,
                    blur_wt: -v,
                    gamma: v,
                };
                assert_eq!(f.apply(&src, 23, 17).len(), src.len());
            }
        }
        // Gamma 0: every pixel 255 (pow(x, 0) = 1), as OpenCV's table gives it.
        let f = VisualFilter {
            gamma: 0.0,
            ..VisualFilter::default()
        };
        assert!(f.apply(&src, 23, 17).iter().all(|&v| v == 255));
        assert_eq!(
            VisualFilter {
                krad: 5000,
                ..VisualFilter::default()
            }
            .apply(&src, 23, 17),
            VisualFilter {
                krad: MAX_KRAD,
                ..VisualFilter::default()
            }
            .apply(&src, 23, 17)
        );
    }

    #[test]
    fn gamma_table_ends() {
        let l = gamma_lut(0.4);
        assert_eq!(l[0], 0);
        assert_eq!(l[255], 255);
        assert!(l.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn flat_image_stays_flat_through_the_blur() {
        let src = vec![77u8; 40 * 30];
        assert!(
            gaussian_blur_u8(&src, 40, 30, 35, 10.0)
                .iter()
                .all(|&v| v == 77)
        );
    }

    /// The display check directory, or `None` (skip) when absent, unless the reference data
    /// is required (`XMA_REQUIRE_ORACLE=1`).
    fn display_data(name: &str) -> Option<std::path::PathBuf> {
        let dir = xma_oracle::oracle_root().join("data/display").join(name);
        if dir.exists() {
            return Some(dir);
        }
        assert!(
            !xma_oracle::oracle_required(),
            "{} is missing: run oracle/py/display_{name}.py (oracle/py/DISPLAY_CHECKS.md)",
            dir.display()
        );
        eprintln!("SKIPPED: {} absent", dir.display());
        None
    }

    /// Pig frames filtered here and by OpenCV (`oracle/py/display_visual_filter.py`) are
    /// identical (measured: 10 frames, 0 pixels differ).
    #[test]
    fn pig_frames_match_opencv() {
        let Some(dir) = display_data("visual_filter") else {
            return;
        };
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            let name = p.to_string_lossy().into_owned();
            let Some(stem) = name.strip_suffix(".in.u8") else {
                continue;
            };
            let size = std::fs::read_to_string(format!("{stem}.size")).unwrap();
            let mut it = size.split_whitespace().map(|v| v.parse::<usize>().unwrap());
            let (w, h) = (it.next().unwrap(), it.next().unwrap());
            let src = std::fs::read(&p).unwrap();
            let want = std::fs::read(format!("{stem}.cv.u8")).unwrap();
            let got = VisualFilter::default().apply(&src, w, h);
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{stem}: {differ} pixels differ");
            n += 1;
        }
        assert!(n > 0, "no frames in {}", dir.display());
    }
}
