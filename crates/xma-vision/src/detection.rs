//! Marker detection, a port of `processing/MarkerDetection.cpp`:
//! `MarkerDetection::detectionPoint` and `MarkerDetection::refinePointPolynomialFit`.
//!
//! - [`detect_point`] is the whole of `detectionPoint` on an 8-bit image: the window
//!   offsets ([`window_origin`]), the patch (`Image::getSubImage`), the method branch and the
//!   returned [`Detection`]. [`detect_point_preview`] also returns the images the C++ hands
//!   to the Detection Settings preview.
//! - Methods 0, 2 and 5: the image stages ([`detection_image_stages`]) and the contour stage
//!   ([`select_contour`]: `findContours` with the offset, the contour nearest the seed by its
//!   `minEnclosingCircle` centre or, for method 5, its moment centre).
//! - Methods 1, 6 (`SimpleBlobDetector`) and 3 (`goodFeaturesToTrack` + `cornerSubPix`) are
//!   another work package (`imgproc::blob`, `imgproc::corners`); [`detect_point`] has one
//!   marked place where they are dispatched and returns "not found" for them until then.
//! - Method 4 detects nothing.
//! - [`refine_point_polynomial_fit`]: the polynomial point refinement of the Points panel.
//!
//! Faults of XMALab 3 are switched by [`DetectCompat`] (parity ledger A65, A66, A90) and by
//! the selection switches of [`DetectOptions`] (A14, A73); the default fixes them.
//!
//! # Images deeper than 8 bits
//!
//! [`detect_point_u16`], [`detect_point_preview_u16`] and
//! [`refine_point_polynomial_fit_u16`] take a `bit_depth`-bit image (parity ledger V1; the
//! rule is in `crate::depth`): the window is cut from the image on the 8-bit scale,
//! `value / 2^(bit_depth - 8)`, as `f32` without rounding (an exact copy, since the window
//! is at whole-pixel offsets), and each method runs its 8-bit algorithm on those values:
//!
//! - methods 0, 2, 5 ([`detection_image_stages_f32`]): method 2 inverts as `255 - v`; the
//!   blur, difference and normalisation are float stages already; the normalised difference
//!   is converted to 8 bits before the median and threshold, as in the 8-bit path;
//! - methods 1, 6: the blob detector with each threshold applied to the float values
//!   (`imgproc::blob::detect_blob_marker_f32`);
//! - method 3: the corner detector on the float values with the 8-bit path's arithmetic
//!   (`imgproc::corners::detect_corner_marker_f32`);
//! - the refinement fits the float values (`255 - v` for a dark marker).
//!
//! Parameters are not rescaled: the threshold offset is a percentage of the 0..255 range of
//! the normalised difference (the same on both paths), the blob thresholds are on the
//! 0..255 scale, sizes are in pixels. The blob thresholds differ by half a grey level
//! between the paths: for a threshold `t` the image reduced to 8 bits keeps values of at
//! least `floor(t) + 0.5` on the 8-bit scale, the deep path values above `t`
//! (`crate::depth`). For an 8-bit image promoted to a deeper one (each value times
//! `2^(bit_depth - 8)`) every one of these entry points returns exactly what the 8-bit
//! entry point returns on the 8-bit image (`crates/xma-vision/tests/fulldepth.rs`).

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

use crate::depth::on_8bit_scale;
use crate::image::{ImageF32, ImageU8, ImageU16};
use crate::imgproc::blob::{self, BlobSettings, SelectionCompat};
use crate::imgproc::contours::{ChainApprox, PointI, RetrievalMode, find_contours};
use crate::imgproc::shape::{min_enclosing_circle, moments};
use crate::imgproc::subpix::{get_sub_image_at, get_sub_image_at_32f};
use crate::imgproc::{arith, blur, corners};
use crate::linalg;

/// Which faults of XMALab 3's detection code to reproduce. Each field is `true` for the C++
/// behaviour. [`Default`] and [`DetectCompat::FIXED`] fix all three; [`DetectCompat::XMALAB3`]
/// is the C++. (xma-model's `CppCompat` holds the same switches under the same names.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DetectCompat {
    /// Ledger A65: window offsets are `(int)(v + 0.5)`, which rounds towards zero for a
    /// negative `v`, so a window reaching past the left or top image edge is one pixel off.
    /// Fixed: `floor(v + 0.5)`. Used by [`detect_point`] and
    /// [`refine_point_polynomial_fit`] (`MarkerDetection.cpp:111-112`, `:474-475`).
    pub window_round_toward_zero: bool,
    /// Ledger A66: the polynomial refinement's ellipse rotation is the mis-bracketed
    /// `0.5 * (pi/2 - atan((c - a)/2/b) + (a - c < 0)) * pi/2` (`MarkerDetection.cpp:597`),
    /// which changes the refined radius. Fixed: `0.5 * atan2(2b, c - a)`, the principal axes
    /// of the quadratic part, defined everywhere ([`polyfit_rotation`]).
    pub polyfit_rotation_bracket: bool,
    /// Ledger A90: the polynomial refinement returns a refined point and radius when its last
    /// cycle's quadratic part is not a peak (`J = ac - b^2 <= 0`: the cycle's clamped
    /// correction is applied and the radius of that fit returned), when it is a minimum of
    /// the fitted quantity (`J > 0` and `a + c > 0`: a dip where the marker's polarity needs
    /// a maximum), and when the last cycle's radius is `+inf` or 0
    /// (`MarkerDetection.cpp:590-606`, `:682-691`). Fixed: no result in those cases
    /// ([`refine_point_polynomial_fit`]). The same switch sets the window bound (A90): fixed,
    /// no result once a cycle's window half-width exceeds the image's larger side; XMALab 3,
    /// once it exceeds the larger of that side and [`REFINE_MAX_HALF_WIDTH_CPP`] (1024), so
    /// that the C++'s results for windows larger than the image are reproduced.
    pub refine_accepts_non_peak: bool,
}

impl DetectCompat {
    /// Both faults fixed: the default.
    pub const FIXED: DetectCompat = DetectCompat {
        window_round_toward_zero: false,
        polyfit_rotation_bracket: false,
        refine_accepts_non_peak: false,
    };

    /// Everything as XMALab 3 does it.
    pub const XMALAB3: DetectCompat = DetectCompat {
        window_round_toward_zero: true,
        polyfit_rotation_bracket: true,
        refine_accepts_non_peak: true,
    };
}

/// What [`detect_point`] needs besides its per-call arguments: which faults of XMALab 3 to
/// reproduce, and the blob detector's settings for methods 1 and 6.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DetectOptions {
    /// Window rounding (ledger A65).
    pub compat: DetectCompat,
    /// How methods 1, 3 and 6 choose among the features they find (ledger A14, A73), and how
    /// methods 0, 2 and 5 place the selected contour's centre (ledger A73,
    /// [`SelectionCompat::offset_sum_f32`]).
    pub selection: SelectionCompat,
    /// The `BlobDetector*` settings, read by methods 1 and 6.
    pub blob: BlobSettings,
}

impl DetectOptions {
    /// Every fault fixed and XMALab's default blob settings: the same as [`Default`].
    #[must_use]
    pub fn fixed() -> Self {
        DetectOptions::default()
    }

    /// Everything as XMALab 3 does it, with XMALab's default blob settings.
    #[must_use]
    pub fn xmalab3() -> Self {
        DetectOptions {
            compat: DetectCompat::XMALAB3,
            selection: SelectionCompat::XMALAB3,
            blob: BlobSettings::default(),
        }
    }
}

/// The top-left pixel of a window of half-width `half` around `centre`:
/// `(int)(centre - half + 0.5)` in XMALab 3 (truncating towards zero, ledger A65),
/// `floor(centre - half + 0.5)` when fixed. A value outside the `i32` range saturates (the
/// C++ cast is undefined there) and NaN gives 0.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`
/// (`:111-112`) and `MarkerDetection::refinePointPolynomialFit` (`:474-475`).
#[must_use]
pub fn window_origin(centre: f64, half: f64, compat: DetectCompat) -> i32 {
    let v = centre - half + 0.5;
    if compat.window_round_toward_zero {
        v as i32
    } else {
        v.floor() as i32
    }
}

/// True when the detection window of half-width `search_area` around `centre` is one the
/// C++ computes with defined `int` arithmetic: the origin `(int)(centre - search_area +
/// 0.5)` is an `int` (the centre finite) and the window's corners and `getRectSubPix`'s
/// origin are too ([`crate::imgproc::subpix::window_in_int_range`]). Where it is false,
/// [`detect_point`] and its variants find nothing (ledger A160).
#[must_use]
pub fn window_fits(centre: (f64, f64), search_area: i32, compat: DetectCompat) -> bool {
    let sa = f64::from(search_area);
    let int_range = f64::from(i32::MIN)..f64::from(i32::MAX);
    if !int_range.contains(&(centre.0 - sa + 0.5)) || !int_range.contains(&(centre.1 - sa + 0.5)) {
        return false;
    }
    let off_x = window_origin(centre.0, sa, compat);
    let off_y = window_origin(centre.1, sa, compat);
    crate::imgproc::subpix::window_in_int_range(i64::from(off_x), i64::from(off_y), search_area)
}

/// XMALab's detection method (`Marker::method`), with the codes the project files use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DetectionMethod {
    /// 0: dark marker on a lighter background, contour closest to the seed
    /// (`minEnclosingCircle` centre).
    Default = 0,
    /// 1: blob detector for dark blobs.
    Blob = 1,
    /// 2: as 0 for a white marker (the patch is inverted first).
    White = 2,
    /// 3: corner detector (`goodFeaturesToTrack` + `cornerSubPix`).
    Corner = 3,
    /// 4: no detection (the point is set by hand).
    NoDetection = 4,
    /// 5: as 0, but the contour's centre is its centroid (moments).
    Centroid = 5,
    /// 6: blob detector for white blobs.
    WhiteBlob = 6,
}

impl DetectionMethod {
    /// The method for an XMALab code, or `None` outside 0..=6.
    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => Self::Default,
            1 => Self::Blob,
            2 => Self::White,
            3 => Self::Corner,
            4 => Self::NoDetection,
            5 => Self::Centroid,
            6 => Self::WhiteBlob,
            _ => return None,
        })
    }

    /// The XMALab code.
    #[must_use]
    pub fn code(self) -> i32 {
        self as i32
    }

    /// True for the methods whose image stages are [`detection_image_stages`] (0, 2, 5).
    #[must_use]
    pub fn uses_contours(self) -> bool {
        matches!(self, Self::Default | Self::White | Self::Centroid)
    }
}

/// The blur radius and sigma of `detectionPoint`: `radius = (int)(1.5 * masksize + 0.5)`,
/// `sigma = radius * sqrt(2 * log(255)) - 1`. The kernel is `2 * radius + 1` wide.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`.
#[must_use]
pub fn blur_radius_sigma(masksize: i32) -> (i32, f64) {
    let radius = (1.5 * f64::from(masksize) + 0.5) as i32;
    let sigma = f64::from(radius) * (2.0 * 255f64.ln()).sqrt() - 1.0;
    (radius, sigma)
}

/// The smallest blur radius [`blur_radius_within_bound`] allows, whatever the window.
pub const BLUR_RADIUS_FLOOR: i64 = 4096;

/// True when `detectionPoint`'s background blur for `masksize` on a window of half-width
/// `search_area` is computed (parity ledger A168): the radius `1.5 * masksize + 0.5` is an
/// `int` (the C++ cast is undefined otherwise) and at most the larger of the window's side
/// `2 * search_area + 1` and [`BLUR_RADIUS_FLOOR`] (4096). The C++ allocates and convolves a
/// kernel of `2 radius + 1` taps whatever the window (a mask size of 10^8 asks for 2.4 GB
/// and hours), so beyond it the detection finds nothing and [`detection_image_stages`]
/// returns `None`. A marker's mask size is its size in pixels (at most 50 from the measured
/// sizes; a size override from the project file can hold any `int`).
#[must_use]
pub fn blur_radius_within_bound(masksize: i32, search_area: usize) -> bool {
    let r = 1.5 * f64::from(masksize) + 0.5;
    // `(int)r` is defined for -2^31 - 1 < r < 2^31.
    if r.is_nan() || r <= f64::from(i32::MIN) - 1.0 || r >= 2_147_483_648.0 {
        return false;
    }
    let side = i64::try_from(search_area)
        .ok()
        .and_then(|sa| sa.checked_mul(2))
        .map_or(i64::MAX, |v| v + 1);
    i64::from(r as i32) <= side.max(BLUR_RADIUS_FLOOR)
}

/// Every image the image part of `detectionPoint` (methods 0, 2, 5) produces.
///
/// The six 8-bit images in [`DetectionStages::preview_images`] are what the C++ pushes to the
/// `images` vector for the Detection Settings preview, in its order; the C++ then adds a
/// seventh, the original patch cut again, only when a contour was found.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionStages {
    /// The patch after the method-2 inversion (`images[0]`). For a deeper image
    /// ([`detection_image_stages_f32`]) this is [`DetectionStages::input_f32`] rounded to 8
    /// bits, for the preview only: no stage reads it.
    pub input: ImageU8,
    /// The patch as float (`img_float`).
    pub input_f32: ImageF32,
    /// `GaussianBlur(img_float, Size(2r+1, 2r+1), sigma)` (`blurred`).
    pub blurred_f32: ImageF32,
    /// `blurred.convertTo(CV_8U)` (`images[1]`).
    pub blurred_u8: ImageU8,
    /// `img_float - blurred` before normalising.
    pub diff_f32: ImageF32,
    /// The difference normalised to 0..255 (`normalize(diff, diff, 0, 255, NORM_MINMAX)`).
    pub diff_norm_f32: ImageF32,
    /// The normalised difference as 8-bit (`images[2]`).
    pub diff_u8: ImageU8,
    /// `medianBlur(.., 3)` (`images[3]`).
    pub median: ImageU8,
    /// Minimum of the median image (`minMaxLoc`).
    pub min_val: f64,
    /// Maximum of the median image.
    pub max_val: f64,
    /// The median image at the patch centre `(searchArea, searchArea)`.
    pub centre_val: u8,
    /// The threshold passed to `cv::threshold`:
    /// `0.5 * min + 0.5 * centre + threshold * 0.01 * 255`.
    pub thres: f64,
    /// `threshold(.., thres, 255, THRESH_BINARY_INV)` (`images[4]`).
    pub threshold: ImageU8,
    /// `GaussianBlur(.., Size(3, 3), 1.3)` (`images[5]`): the image `findContours` receives.
    pub thresh_blur: ImageU8,
}

impl DetectionStages {
    /// The images the C++ pushes to its `images` vector before the contour stage, in order.
    #[must_use]
    pub fn preview_images(&self) -> [&ImageU8; 6] {
        [
            &self.input,
            &self.blurred_u8,
            &self.diff_u8,
            &self.median,
            &self.threshold,
            &self.thresh_blur,
        ]
    }

    /// The image `cv::findContours` receives.
    #[must_use]
    pub fn contour_input(&self) -> &ImageU8 {
        &self.thresh_blur
    }
}

/// The image stages of `detectionPoint` for methods 0, 2 and 5: from the cut patch to the
/// image `findContours` receives.
///
/// `subimage` is the C++ `subimage` after `getSubImage`, `(2 * search_area + 1)` pixels
/// square; `masksize` the marker size as the C++ `int`; `threshold` the threshold offset in
/// percent of 255.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint` (the branch
/// `method == 0 || method == 2 || method == 5`, up to `cv::findContours`), with the OpenCV
/// routines of [`crate::imgproc`]. `normalize(diff, diff, ..)` runs in place in the C++; in
/// place and not give the same values (see [`arith::convert_scale_f32`]).
///
/// `None` when the blur for `masksize` is beyond its bound ([`blur_radius_within_bound`],
/// parity ledger A168): the detection then finds nothing.
///
/// # Panics
/// If `method` is not 0, 2 or 5, if the patch is not square or `search_area` is not its
/// centre.
#[must_use]
pub fn detection_image_stages(
    subimage: &ImageU8,
    method: DetectionMethod,
    search_area: usize,
    masksize: i32,
    threshold: f64,
) -> Option<DetectionStages> {
    assert!(
        method.uses_contours(),
        "method {method:?} has no image stages"
    );
    assert_eq!(
        (subimage.width(), subimage.height()),
        (2 * search_area + 1, 2 * search_area + 1),
        "the patch must be 2 * searchArea + 1 pixels square"
    );
    let input = if method == DetectionMethod::White {
        arith::invert_u8(subimage)
    } else {
        subimage.clone()
    };
    let input_f32 = arith::u8_to_f32(&input);
    stages_from_float(input, input_f32, search_area, masksize, threshold)
}

/// [`detection_image_stages`] for the patch of a deeper image, as `f32` on the 8-bit scale
/// (parity ledger V1; `crate::depth`). Method 2 inverts it as `255 - v` (exact for these
/// values); from the float patch on, every stage is the 8-bit path's, including the
/// conversion of the normalised difference to 8 bits before the median and the threshold.
/// [`DetectionStages::input`] is the patch rounded to 8 bits for the preview.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint` (the branch
/// `method == 0 || method == 2 || method == 5`, up to `cv::findContours`; extends
/// [`detection_image_stages`]).
///
/// `None` as [`detection_image_stages`].
///
/// # Panics
/// As [`detection_image_stages`].
#[must_use]
pub fn detection_image_stages_f32(
    subimage: &ImageF32,
    method: DetectionMethod,
    search_area: usize,
    masksize: i32,
    threshold: f64,
) -> Option<DetectionStages> {
    assert!(
        method.uses_contours(),
        "method {method:?} has no image stages"
    );
    assert_eq!(
        (subimage.width(), subimage.height()),
        (2 * search_area + 1, 2 * search_area + 1),
        "the patch must be 2 * searchArea + 1 pixels square"
    );
    let input_f32 = if method == DetectionMethod::White {
        subimage.map(|v| 255.0 - v)
    } else {
        subimage.clone()
    };
    let input = arith::f32_to_u8(&input_f32);
    stages_from_float(input, input_f32, search_area, masksize, threshold)
}

/// The stages from the float patch on, shared by [`detection_image_stages`] and
/// [`detection_image_stages_f32`].
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint` (from
/// `GaussianBlur` to `cv::findContours`).
fn stages_from_float(
    input: ImageU8,
    input_f32: ImageF32,
    search_area: usize,
    masksize: i32,
    threshold: f64,
) -> Option<DetectionStages> {
    if !blur_radius_within_bound(masksize, search_area) {
        return None;
    }
    let (radius, sigma) = blur_radius_sigma(masksize);
    // In `i64`: the same as the `int` sum for every radius within the bound, and defined for
    // a negative one (`max(1)`: no blur, as before).
    let k = (2 * i64::from(radius) + 1).max(1) as usize;
    let blurred_f32 = blur::gaussian_blur_f32(&input_f32, k, sigma);
    let blurred_u8 = arith::f32_to_u8(&blurred_f32);
    let diff_f32 = arith::subtract_f32(&input_f32, &blurred_f32);
    let diff_norm_f32 = arith::normalize_minmax_f32(&diff_f32, 0.0, 255.0);
    let diff_u8 = arith::f32_to_u8(&diff_norm_f32);
    let median = blur::median_blur3_u8(&diff_u8);
    let mm = arith::min_max_loc_u8(&median);
    let centre_val = median.at(search_area, search_area);
    let thres =
        0.5 * f64::from(mm.min_val) + 0.5 * f64::from(centre_val) + threshold * 0.01 * 255.0;
    let (thr, _) = arith::threshold_u8(&median, thres, 255.0, arith::ThresholdType::BinaryInv);
    let thresh_blur = blur::gaussian_blur_u8(&thr, 3, 1.3);
    Some(DetectionStages {
        input,
        input_f32,
        blurred_f32,
        blurred_u8,
        diff_f32,
        diff_norm_f32,
        diff_u8,
        median,
        min_val: f64::from(mm.min_val),
        max_val: f64::from(mm.max_val),
        centre_val,
        thres,
        threshold: thr,
        thresh_blur,
    })
}

/// What a detection returns (`MarkerDetection::detectionPoint`): the point, its size, and
/// every candidate the method found (the C++ `allCentres` and `allSizes`, which the merge
/// monitor reads).
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// The detected point in image coordinates; the input centre when nothing was found.
    pub point: (f64, f64),
    /// The detected size (a radius for methods 0, 2, 5; see parity ledger A69 for methods 1
    /// and 6); -1 when nothing was found.
    pub size: f64,
    /// Every candidate's centre in image coordinates, in the order the method found them.
    pub all_centres: Vec<(f64, f64)>,
    /// Every candidate's size, parallel to `all_centres`.
    pub all_sizes: Vec<f64>,
}

impl Detection {
    /// Nothing found: the input centre, size -1, no candidates (`detectionPoint`'s initial
    /// `point_out` and `tmp_size`).
    #[must_use]
    pub fn not_found(center: (f64, f64)) -> Self {
        Detection {
            point: center,
            size: -1.0,
            all_centres: Vec::new(),
            all_sizes: Vec::new(),
        }
    }
}

/// One contour of the contour stage, with what `detectionPoint` computes for it.
#[derive(Clone, Debug, PartialEq)]
pub struct ContourCandidate {
    /// The contour in image coordinates (the window offset added), as `findContours` gives
    /// it (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`).
    pub contour: Vec<PointI>,
    /// `minEnclosingCircle` centre, computed on the contour in image coordinates in `float`
    /// as the C++ computes it.
    pub circle_centre: (f32, f32),
    /// `minEnclosingCircle` radius, as the C++ computes it.
    pub circle_radius: f32,
    /// The C++ `cv::Point2f` centre: the circle centre for methods 0 and 2, the moment
    /// centre `(m10 / m00, m01 / m00)` rounded to `f32` for method 5 (NaN for a contour of
    /// zero area).
    pub centre: (f32, f32),
    /// The centre the selection measures from the seed and the detection stores, in image
    /// coordinates (ledger A73, [`SelectionCompat::offset_sum_f32`]). XMALab 3:
    /// [`ContourCandidate::centre`]. Fixed, with no single-precision rounding at image
    /// scale: for methods 0 and 2 the `minEnclosingCircle` centre of the contour in patch
    /// coordinates (the window offset taken off, so the `float` circle is computed at
    /// magnitudes below `2 searchArea + 1`) with the integer offset added in `f64`; for
    /// method 5 the moment centre `m10 / m00`, `m01 / m00` in `f64`.
    pub position: (f64, f64),
    /// The candidate's size (`allSizes`, and the detected size when selected): XMALab 3
    /// [`ContourCandidate::circle_radius`]; fixed, the radius of the circle in patch
    /// coordinates.
    pub size: f64,
    /// Distance from the seed to [`ContourCandidate::position`], in `f64` (NaN for a NaN
    /// centre, which is then never selected).
    pub dist: f64,
}

/// The contour stage of methods 0, 2 and 5: every contour and the one selected.
#[derive(Clone, Debug, PartialEq)]
pub struct ContourSelection {
    /// The contours in `findContours` order.
    pub candidates: Vec<ContourCandidate>,
    /// Index of the selected contour (`bestIdx`); `None` when there is none closer than
    /// 1000 pixels (no contour, or only NaN distances).
    pub best: Option<usize>,
}

impl ContourSelection {
    /// The [`Detection`] this selection gives for the seed `center`: the selected candidate's
    /// position and size, and every candidate's position and size.
    ///
    /// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`
    /// (`:239-273`).
    #[must_use]
    pub fn detection(&self, center: (f64, f64)) -> Detection {
        let mut d = Detection::not_found(center);
        for c in &self.candidates {
            d.all_centres.push(c.position);
            d.all_sizes.push(c.size);
        }
        if let Some(b) = self.best {
            let c = &self.candidates[b];
            d.point = c.position;
            d.size = c.size;
        }
        d
    }
}

/// The contour stage of `detectionPoint` for methods 0, 2 and 5: finds the outer contours of
/// `contour_input` (the image [`DetectionStages::contour_input`]) shifted by the window
/// offset `offset`, and selects the one whose centre is nearest the seed `center` (strictly
/// nearer than 1000 pixels and than every earlier contour, so the first of equals wins).
///
/// The centre is the `minEnclosingCircle` centre, or for [`DetectionMethod::Centroid`] the
/// moment centre. In XMALab 3 (`selection.offset_sum_f32`) it is the C++ `cv::Point2f`: the
/// circle computed in `float` on the contour in image coordinates, or the moment centre
/// computed in `f64` and stored as `f32`; the distance is computed in `f64` from that `f32`
/// centre. Fixed, it is computed without single-precision rounding at image scale
/// ([`ContourCandidate::position`]). A contour of zero area (one pixel, or a line) has a
/// NaN moment centre and a NaN distance under method 5 and is never selected, but it stays
/// in the candidate list with its NaN centre, as in `allCentres`.
///
/// An `offset` at which a point of the image plus the offset (less 1 for `findContours`'
/// border) is not an `int` gives no candidates (parity ledger A168).
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`
/// (`:217-273`).
///
/// # Panics
/// If `method` is not 0, 2 or 5.
#[must_use]
pub fn select_contour(
    contour_input: &ImageU8,
    method: DetectionMethod,
    center: (f64, f64),
    offset: (i32, i32),
    selection: SelectionCompat,
) -> ContourSelection {
    assert!(
        method.uses_contours(),
        "method {method:?} has no contour stage"
    );
    // Ledger A168: `findContours` adds the offset (and -1 for its border) to every point in
    // `int`; where a point of the image would leave the `int` range (undefined behaviour in
    // the C++) there is no contour.
    let fits = |off: i32, len: usize| {
        let len = i64::try_from(len).unwrap_or(i64::MAX);
        off > i32::MIN && i64::from(off).saturating_add(len) <= i64::from(i32::MAX)
    };
    if !fits(offset.0, contour_input.width()) || !fits(offset.1, contour_input.height()) {
        return ContourSelection {
            candidates: Vec::new(),
            best: None,
        };
    }
    let contours = find_contours(
        contour_input,
        RetrievalMode::External,
        ChainApprox::Simple,
        PointI::new(offset.0, offset.1),
    );
    let mut dist = 1000.0f64;
    let mut best = None;
    let mut candidates = Vec::with_capacity(contours.len());
    for (i, contour) in contours.into_iter().enumerate() {
        let (circle_centre, circle_radius) = min_enclosing_circle(&contour);
        let mu = (method == DetectionMethod::Centroid).then(|| moments(&contour));
        let centre = match &mu {
            Some(mu) => ((mu.m10 / mu.m00) as f32, (mu.m01 / mu.m00) as f32),
            None => circle_centre,
        };
        let (position, size) = if selection.offset_sum_f32 {
            (
                (f64::from(centre.0), f64::from(centre.1)),
                f64::from(circle_radius),
            )
        } else {
            // Ledger A73: the circle in patch coordinates, the offset added in double.
            let local: Vec<PointI> = contour
                .iter()
                .map(|p| PointI::new(p.x.wrapping_sub(offset.0), p.y.wrapping_sub(offset.1)))
                .collect();
            let (c, r) = min_enclosing_circle(&local);
            let position = match &mu {
                Some(mu) => (mu.m10 / mu.m00, mu.m01 / mu.m00),
                None => (
                    f64::from(offset.0) + f64::from(c.0),
                    f64::from(offset.1) + f64::from(c.1),
                ),
            };
            (position, f64::from(r))
        };
        let dx = center.0 - position.0;
        let dy = center.1 - position.1;
        let dist_tmp = (dx * dx + dy * dy).sqrt();
        if dist_tmp < dist {
            best = Some(i);
            dist = dist_tmp;
        }
        candidates.push(ContourCandidate {
            contour,
            circle_centre,
            circle_radius,
            centre,
            position,
            size,
            dist: dist_tmp,
        });
    }
    ContourSelection { candidates, best }
}

/// Grey level of the preview crosshair (`cv::Scalar(127)`).
const CROSSHAIR_GREY: u8 = 127;

/// Draws the preview crosshair at `cent` (patch coordinates): two 5-pixel lines,
/// `cv::line(im, cent -/+ (2, 0))` and `cv::line(im, cent -/+ (0, 2))` in grey 127, each end
/// rounded to the nearest pixel (ties to even, as `cv::Point2d` to `cv::Point`), clipped to
/// the image.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`
/// (`:290-296`), with OpenCV `cv::line` (`LINE_8`, thickness 1) for axis-parallel lines.
fn draw_crosshair(im: &mut ImageU8, cent: (f64, f64)) {
    let r = |v: f64| v.round_ties_even();
    let (w, h) = (im.width() as f64, im.height() as f64);
    // Horizontal: (cent.x - 2, cent.y) to (cent.x + 2, cent.y).
    let (x0, x1, y) = (r(cent.0 - 2.0), r(cent.0 + 2.0), r(cent.1));
    if y >= 0.0 && y < h {
        let mut x = x0.max(0.0);
        while x <= x1.min(w - 1.0) {
            im.set(x as usize, y as usize, CROSSHAIR_GREY);
            x += 1.0;
        }
    }
    // Vertical: (cent.x, cent.y - 2) to (cent.x, cent.y + 2).
    let (y0, y1, x) = (r(cent.1 - 2.0), r(cent.1 + 2.0), r(cent.0));
    if x >= 0.0 && x < w {
        let mut y = y0.max(0.0);
        while y <= y1.min(h - 1.0) {
            im.set(x as usize, y as usize, CROSSHAIR_GREY);
            y += 1.0;
        }
    }
}

/// `MarkerDetection::detectionPoint` on an 8-bit image: detects the marker near `center`
/// with `method` and returns the point, its size and every candidate.
///
/// - `search_area`: half-width of the square searched (the C++ `int`; the `MarkerDetection`
///   constructor clamps it to at least 10, or 50 for methods 1 and 6, before the call);
/// - `masksize`: the marker size as the C++ `int` parameter (truncated by the caller, ledger
///   A16);
/// - `threshold`: the threshold offset in percent of 255 (the threaded C++ path passes it
///   through a `uchar`, ledger A15; that conversion is the caller's).
///
/// The patch is the `2 * search_area + 1` square at [`window_origin`]`(center,
/// search_area)` (`Image::getSubImage`, border pixels replicated). Method 4, and methods 1,
/// 3 and 6 until they are ported, return [`Detection::not_found`].
///
/// Nothing is found ([`Detection::not_found`]) where the C++ overflows `int` or allocates
/// without bound: a window outside the `int` range ([`window_fits`], ledger A160), a
/// `search_area` beyond [`crate::imgproc::subpix::max_window_half`] of the image (A167),
/// and, for methods 0, 2 and 5, a blur radius beyond [`blur_radius_within_bound`] (A168).
/// The largest buffer is then a few windows of `(2 max(L, 1024) + 1)^2` values (`L` the
/// image's larger side) and a kernel of at most `2 max(2 search_area + 1, 4096) + 1` taps.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`.
///
/// # Panics
/// If `search_area` is negative.
#[must_use]
pub fn detect_point(
    image: &ImageU8,
    method: DetectionMethod,
    center: (f64, f64),
    search_area: i32,
    masksize: i32,
    threshold: f64,
    options: &DetectOptions,
) -> Detection {
    detect_point_impl(
        &Source::U8(image),
        method,
        center,
        search_area,
        masksize,
        threshold,
        None,
        options,
    )
}

/// [`detect_point`] with the images of the Detection Settings preview (the C++ `images`
/// vector): for methods 0, 2 and 5 the six stage images ([`DetectionStages::preview_images`])
/// and, when a contour was selected, a seventh, the original patch cut again (before the
/// method-2 inversion). With `draw_crosshairs` and a selected contour, a grey crosshair is
/// drawn on all seven at the detected point. Other methods give no images.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint` (the
/// `images != NULL` paths).
///
/// # Panics
/// If `search_area` is negative.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn detect_point_preview(
    image: &ImageU8,
    method: DetectionMethod,
    center: (f64, f64),
    search_area: i32,
    masksize: i32,
    threshold: f64,
    draw_crosshairs: bool,
    options: &DetectOptions,
) -> (Detection, Vec<ImageU8>) {
    let mut images = Vec::new();
    let d = detect_point_impl(
        &Source::U8(image),
        method,
        center,
        search_area,
        masksize,
        threshold,
        Some((&mut images, draw_crosshairs)),
        options,
    );
    (d, images)
}

/// [`detect_point`] for a `bit_depth`-bit image (8 to 16 bits in `u16`, parity ledger V1):
/// the window is cut from the image on the 8-bit scale (`value / 2^(bit_depth - 8)`) as
/// `f32` and the method runs on those values (see the module documentation); the arguments
/// and the result are those of [`detect_point`].
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint` (extends
/// [`detect_point`]).
///
/// # Panics
/// If `search_area` is negative or `bit_depth` is not 8 to 16.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn detect_point_u16(
    image: &ImageU16,
    bit_depth: u8,
    method: DetectionMethod,
    center: (f64, f64),
    search_area: i32,
    masksize: i32,
    threshold: f64,
    options: &DetectOptions,
) -> Detection {
    detect_point_impl(
        &Source::Deep(image, bit_depth),
        method,
        center,
        search_area,
        masksize,
        threshold,
        None,
        options,
    )
}

/// [`detect_point_preview`] for a `bit_depth`-bit image (see [`detect_point_u16`]). The
/// preview images are 8-bit: the input patch (and the seventh image, the patch cut again)
/// rounded to 8 bits for display; the other five are the 8-bit stages themselves.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint` (the
/// `images != NULL` paths; extends [`detect_point_preview`]).
///
/// # Panics
/// If `search_area` is negative or `bit_depth` is not 8 to 16.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn detect_point_preview_u16(
    image: &ImageU16,
    bit_depth: u8,
    method: DetectionMethod,
    center: (f64, f64),
    search_area: i32,
    masksize: i32,
    threshold: f64,
    draw_crosshairs: bool,
    options: &DetectOptions,
) -> (Detection, Vec<ImageU8>) {
    let mut images = Vec::new();
    let d = detect_point_impl(
        &Source::Deep(image, bit_depth),
        method,
        center,
        search_area,
        masksize,
        threshold,
        Some((&mut images, draw_crosshairs)),
        options,
    );
    (d, images)
}

/// The image a detection or refinement reads: 8-bit, or deeper with its bit depth.
enum Source<'a> {
    U8(&'a ImageU8),
    Deep(&'a ImageU16, u8),
}

/// A window cut from a [`Source`].
enum Patch {
    U8(ImageU8),
    /// On the 8-bit scale.
    F32(ImageF32),
}

impl Source<'_> {
    /// The larger of the image's width and height.
    fn larger_side(&self) -> usize {
        match *self {
            Source::U8(im) => im.width().max(im.height()),
            Source::Deep(im, _) => im.width().max(im.height()),
        }
    }

    /// The image's width and height.
    fn size(&self) -> (usize, usize) {
        match *self {
            Source::U8(im) => (im.width(), im.height()),
            Source::Deep(im, _) => (im.width(), im.height()),
        }
    }

    /// `getSubImage(subimage, size, off_x, off_y)`: 8-bit, or `f32` on the 8-bit scale.
    fn window(&self, size: i32, off_x: i32, off_y: i32) -> Patch {
        match *self {
            Source::U8(im) => Patch::U8(get_sub_image_at(im, size, off_x, off_y)),
            Source::Deep(im, d) => Patch::F32(get_sub_image_at_32f(
                &on_8bit_scale(im, d),
                size,
                off_x,
                off_y,
            )),
        }
    }
}

impl Patch {
    /// The window as 8 bits, for the preview (the deep window rounded).
    fn to_u8(&self) -> ImageU8 {
        match self {
            Patch::U8(p) => p.clone(),
            Patch::F32(p) => arith::f32_to_u8(p),
        }
    }
}

/// The body of [`detect_point`], [`detect_point_preview`] and their deep counterparts.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::detectionPoint`.
#[allow(clippy::too_many_arguments)]
fn detect_point_impl(
    image: &Source,
    method: DetectionMethod,
    center: (f64, f64),
    search_area: i32,
    masksize: i32,
    threshold: f64,
    images: Option<(&mut Vec<ImageU8>, bool)>,
    options: &DetectOptions,
) -> Detection {
    let compat = options.compat;
    assert!(search_area >= 0, "negative search area {search_area}");
    let sa = f64::from(search_area);
    // Ledger A160: a centre whose window the C++ cannot compute in `int` (NaN, infinite, or
    // beyond about +-2^31: undefined behaviour) finds nothing.
    if !window_fits(center, search_area, compat) {
        return Detection::not_found(center);
    }
    // Ledger A167: a window larger than the image allows is not cut; nothing found.
    let (w, h) = image.size();
    if i64::from(search_area) > crate::imgproc::subpix::max_window_half(w, h) {
        return Detection::not_found(center);
    }
    let off_x = window_origin(center.0, sa, compat);
    let off_y = window_origin(center.1, sa, compat);
    let subimage = image.window(search_area, off_x, off_y);

    match method {
        DetectionMethod::Default | DetectionMethod::White | DetectionMethod::Centroid => {
            let stages = match &subimage {
                Patch::U8(p) => {
                    detection_image_stages(p, method, search_area as usize, masksize, threshold)
                }
                Patch::F32(p) => {
                    detection_image_stages_f32(p, method, search_area as usize, masksize, threshold)
                }
            };
            // Ledger A168: a blur beyond its bound finds nothing.
            let Some(stages) = stages else {
                return Detection::not_found(center);
            };
            let sel = select_contour(
                stages.contour_input(),
                method,
                center,
                (off_x, off_y),
                options.selection,
            );
            let d = sel.detection(center);
            if let Some((images, draw)) = images {
                images.extend(stages.preview_images().into_iter().cloned());
                if let Some(b) = sel.best {
                    // The patch is cut again from the image: the original, not inverted.
                    images.push(subimage.to_u8());
                    if draw {
                        // `cv::Point2d cent = detected_center - cv::Point2f(off_x, off_y)`:
                        // the subtraction is in float in XMALab 3; fixed (ledger A73), the
                        // stored position less the offset, in double.
                        let k = &sel.candidates[b];
                        let cent = if options.selection.offset_sum_f32 {
                            let c = k.centre;
                            (f64::from(c.0 - off_x as f32), f64::from(c.1 - off_y as f32))
                        } else {
                            (
                                k.position.0 - f64::from(off_x),
                                k.position.1 - f64::from(off_y),
                            )
                        };
                        for im in images.iter_mut() {
                            draw_crosshair(im, cent);
                        }
                    }
                }
            }
            d
        }
        // Methods 1 and 6 (`MarkerDetection.cpp:347-408`). They push no preview images. Blob
        // settings that OpenCV refuses make the C++ throw an exception nothing catches; here
        // nothing is found (ledger A72).
        DetectionMethod::Blob | DetectionMethod::WhiteBlob => match &subimage {
            Patch::U8(p) => blob::detect_blob_marker(
                p,
                search_area,
                off_x,
                off_y,
                center,
                &options.blob,
                method.code(),
                options.selection,
            ),
            Patch::F32(p) => blob::detect_blob_marker_f32(
                p,
                search_area,
                off_x,
                off_y,
                center,
                &options.blob,
                method.code(),
                options.selection,
            ),
        }
        .unwrap_or_else(|_| Detection::not_found(center)),
        // Method 3 (`MarkerDetection.cpp:315-346`).
        DetectionMethod::Corner => match &subimage {
            Patch::U8(p) => corners::detect_corner_marker(
                p,
                search_area,
                off_x,
                off_y,
                center,
                options.selection,
            ),
            Patch::F32(p) => corners::detect_corner_marker_f32(
                p,
                search_area,
                off_x,
                off_y,
                center,
                options.selection,
            ),
        },
        DetectionMethod::NoDetection => Detection::not_found(center),
    }
}

/// `sign` of `MarkerDetection.cpp`: `(a >= 0) ? 1 : -1` (so NaN gives -1).
fn sign(a: f64) -> f64 {
    if a >= 0.0 { 1.0 } else { -1.0 }
}

/// `min` of `MarkerDetection.cpp`: `(a < b) ? a : b` (so `min(NaN, b)` is `b`).
fn min_macro(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// The window half-width up to which [`refine_point_polynomial_fit`] in the XMALab 3 mode
/// ([`DetectCompat::refine_accepts_non_peak`]) reproduces the C++ on images smaller than the
/// window (ledger A90): the half-width up to which the reference probe
/// (`oracle/py/refine_probe`, `PROBE_WCAP`) runs the C++.
pub const REFINE_MAX_HALF_WIDTH_CPP: usize = 1024;

/// The result of [`refine_point_polynomial_fit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refinement {
    /// The refined centre.
    pub point: (f64, f64),
    /// The refined radius (the geometric mean of the fitted peak's two radii), which the
    /// refinement action stores as the marker size.
    pub radius: f64,
    /// How many fitting cycles ran (`refinementcount`).
    pub cycles: u32,
}

/// The rotation of the fitted ellipse from the quadratic coefficients `a` (`x^2`), `b` (half
/// the `xy` coefficient) and `c` (`y^2`).
///
/// XMALab 3 (`compat`): `0.5 * (pi/2 - atan((c - a)/2/b) + (a - c < 0)) * pi/2`, where the
/// comparison adds 0 or 1 and the whole angle is then scaled by `pi/2` (ledger A66).
/// Fixed: `0.5 * atan2(2b, c - a)`. With `t` the result, the rotated axes `(cos t, -sin t)`
/// and `(sin t, cos t)` that `Q1`, `Q2`, `P1`, `P2` are evaluated on are the principal axes
/// of `a x^2 + 2b xy + c y^2` (`tan 2t = 2b / (c - a)`). The intended expression of the C++,
/// `0.5 * (pi/2 - atan((c - a)/2/b)) + (a < c) * pi/2`, solves the same equation and differs
/// from this one by a multiple of `pi/2` (which exchanges or reverses the axes and leaves the
/// radius unchanged), but it is `0/0` when `a = c` and `b = 0`, an isotropic quadratic part,
/// where every direction is a principal axis; `atan2(0, 0)` is 0 there.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::refinePointPolynomialFit`
/// (`:597`).
#[must_use]
pub fn polyfit_rotation(a: f64, b: f64, c: f64, compat: DetectCompat) -> f64 {
    if compat.polyfit_rotation_bracket {
        let lt = a - c < 0.0;
        0.5 * (PI / 2.0 - ((c - a) / 2.0 / b).atan() + f64::from(u8::from(lt))) * PI / 2.0
    } else {
        0.5 * (2.0 * b).atan2(c - a)
    }
}

/// The polynomial point refinement ("Refine center of selected Points by polynomial fit"):
/// fits a weighted 4th-order polynomial surface to the image around `point`, moves the point
/// towards the centre of the fitted quadratic peak by at most half a pixel per cycle, and
/// estimates the peak's radius. Returns `None` where the C++ returns false (the window has
/// fewer than 15 pixels, or the point or radius is NaN at the end) and where the C++ would
/// go on with a radius that is not finite (undefined behaviour in the C++, ledger A66: the
/// refinement ends without a result in both modes).
///
/// Window bound (ledger A90): the refinement ends with no result as soon as a cycle's window
/// half-width `w = (int)(1.6 * radius + 0.5)` (`:471`) exceeds the bound, before the window
/// is cut (`:479`) and the system of `(2w + 1)^2` rows of 15 doubles is built (`:492-494`).
/// On an image that is exactly quadratic around the point the fitted quartic terms are
/// rounding noise, the radius of `:604` comes out near `1e8`, and the C++ fails in the next
/// cycle with OpenCV's out-of-memory exception (`A.create`, `:492`): a crash, which is not
/// reproduced in either mode. The bound depends on
/// [`DetectCompat::refine_accepts_non_peak`] (the C++ outcome for a refinement that has left
/// the marker):
/// - fixed: the larger of the image's width and height `N`. A window that large holds no
///   more of the image than the image itself (the rest is its replicated border) and means
///   the fit has diverged. Each cycle allocates at most `(2N + 1)^2` rows of 15 doubles.
/// - XMALab 3: the larger of `N` and [`REFINE_MAX_HALF_WIDTH_CPP`] (1024). The C++ returns a
///   defined result for a window larger than the image whenever it can allocate it (it fits
///   the replicated border), and those results are reproduced up to the half-width the
///   reference covers; beyond it whether the C++ returns depends on the machine's memory.
///   On an image whose larger side is at most 1024, a cycle can allocate up to
///   `(2 * 1024 + 1)^2` rows of 15 doubles, about 0.5 GB (and the solve's copy of them),
///   whatever the image size.
///
/// Also in both modes, a point that is not finite, or whose window origin
/// `(int)(x - w + 0.5)` (`:474-475`) is not an `int` (undefined behaviour in the C++), ends
/// the refinement with no result, and a negative `w` (a negative radius; OpenCV raises an
/// exception for the negative window size) too. The number of cycles is at most
/// `1.6 * radius0 / 0.5 + 3` for a starting radius `radius0` whose first window passes the
/// bound.
///
/// Fits that are not a peak (ledger A90, [`DetectCompat::refine_accepts_non_peak`]): the
/// loop stops as soon as a cycle's quadratic part is not a peak (`J <= 0`), but the C++
/// has already applied that cycle's clamped correction and computed its radius, and
/// returns both. It also returns a last-cycle radius of `+inf` (a quartic term of zero) or
/// 0, which the caller stores as the marker size, and a last fit whose quadratic part is a
/// minimum of the fitted quantity. The fitted quantity is `255 - v` for a dark marker and
/// `v` for a bright one (`:513-520`), so in both polarities the marker is a maximum of it;
/// the quadratic part `a x^2 + 2b xy + c y^2` (`a = p3`, `b = p4 / 2`, `c = p5`, `:583-585`)
/// is a maximum when `J = ac - b^2 > 0` and `a + c < 0` (with `J > 0`, `a` and `c` have the
/// same sign) and a minimum when `J > 0` and `a + c > 0`: a light spot fitted as a dark
/// marker, or a dark one fitted as bright. The C++ moves the point to that minimum's centre
/// and returns it. Fixed, the refinement returns no result when the last cycle's `J` is not
/// positive, its `a + c` is not negative, or its radius is not finite and positive.
///
/// Flat windows (ledger A37), in both modes: when every pixel of a cycle's window (the
/// `2w + 1` square `getSubImage` cuts, `:479`) has the same value, the refinement ends with
/// no result, before the solve. On such a window every `B` is the same multiple of the
/// weight column of `A` (`:513-521`), so the exact least-squares solution of `:539` is that
/// multiple times the first unit vector: all quadratic terms are zero, `J = 0` (`:590`),
/// `xc`, `yc` and the rotation are `0/0`, the radius NaN (`:591-604`) and the loop stops
/// (`:606`), so the C++ in exact arithmetic returns false (`:682-691`). The C++ as built
/// does so only when its solver returns those zeros exactly (in the reference runs: when
/// `B` is zero, a white window fitted as dark or a black one fitted as bright); otherwise
/// the solver's rounding leaves tiny quadratic terms and the C++ (with LAPACK as with
/// OpenCV's own QR) returns a point and radius made of rounding noise, which this function
/// does not reproduce. The test is exact equality of the 8-bit values; a window that differs
/// in a single pixel is fitted.
///
/// Arguments, as `MarkerTreeWidget::action_RefinePointsPolynomialFit_triggered` passes them:
/// `image` is the camera's 8-bit image at the frame (`VideoStream::getImage`); `point` the
/// stored 2D point; `radius` the marker's size at that point (5 when it is -1);
/// `dark_marker` is `method != 2 && method != 6`. Nothing else is read from the project.
///
/// The constants are the C++'s: window half-width `(int)(1.6 * radius + 0.5)`, weighting
/// `exp(-(dx^2 + dy^2) / radius^2)`, at most `1.6 * radius0 / 0.5` cycles, stop after three
/// cycles in which neither correction exceeds 0.5, or as soon as the quadratic part is not a
/// peak (`J <= 0`). The least-squares solve is OpenCV's own `QRImpl` ([`linalg::solve_qr`]);
/// XMALab 3's OpenCV hands systems of 30 or more rows to LAPACK (ledger A37), which agrees
/// to rounding. The unused parts of the C++ (the quadric surface, the semi-axes, eccentricity
/// and skewness of the last cycle, the dead `!subpixpeak` branch) are not computed.
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::refinePointPolynomialFit`.
#[must_use]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn refine_point_polynomial_fit(
    image: &ImageU8,
    point: (f64, f64),
    radius: f64,
    dark_marker: bool,
    compat: DetectCompat,
) -> Option<Refinement> {
    refine_impl(&Source::U8(image), point, radius, dark_marker, compat)
}

/// [`refine_point_polynomial_fit`] for a `bit_depth`-bit image (8 to 16 bits in `u16`,
/// parity ledger V1): each cycle's window is cut from the image on the 8-bit scale
/// (`value / 2^(bit_depth - 8)`, an exact copy at whole-pixel offsets) and the surface is
/// fitted to those values, `255 - v` for a dark marker, in `f64`. The flat-window test
/// compares the float values. Arguments and result as [`refine_point_polynomial_fit`].
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::refinePointPolynomialFit`
/// (extends [`refine_point_polynomial_fit`]).
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[must_use]
pub fn refine_point_polynomial_fit_u16(
    image: &ImageU16,
    bit_depth: u8,
    point: (f64, f64),
    radius: f64,
    dark_marker: bool,
    compat: DetectCompat,
) -> Option<Refinement> {
    let _ = crate::depth::scale_to_8bit(bit_depth);
    refine_impl(
        &Source::Deep(image, bit_depth),
        point,
        radius,
        dark_marker,
        compat,
    )
}

/// The body of [`refine_point_polynomial_fit`] and [`refine_point_polynomial_fit_u16`].
///
/// Ported from `processing/MarkerDetection.cpp`, `MarkerDetection::refinePointPolynomialFit`.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn refine_impl(
    image: &Source,
    point: (f64, f64),
    radius: f64,
    dark_marker: bool,
    compat: DetectCompat,
) -> Option<Refinement> {
    const LIMMULT: f64 = 1.6;
    const MASKMULT: f64 = 1.0;
    const IMPROVERTHRESH: f64 = 0.5;

    // Ledger A90: the window bound (see the doc comment).
    let max_half_width = if compat.refine_accepts_non_peak {
        image.larger_side().max(REFINE_MAX_HALF_WIDTH_CPP)
    } else {
        image.larger_side()
    };
    let mut radius = radius;
    let (mut x, mut y) = point;
    let mut doextracycles = 3;
    let mut stillgood = true;
    let mut refinementcount: u32 = 0;
    let maxrefinements = LIMMULT * radius / IMPROVERTHRESH;
    // `J` and `a + c` of the last cycle (ledger A90).
    let mut last_j = f64::NAN;
    let mut last_trace = f64::NAN;

    while stillgood && doextracycles > 0 {
        // Ledger A66: the C++ converts a NaN (or infinite) window size to `int` here.
        let wf = LIMMULT * radius + 0.5;
        if !wf.is_finite() || wf >= f64::from(i32::MAX / 2) || wf <= f64::from(i32::MIN / 2) {
            return None;
        }
        refinementcount += 1;
        let w = f64::from(wf as i32);
        // `getSubImage(subimage, w, off_x, off_y)`: the integer overload (`double w` converts
        // to `int` exactly).
        let w_i = wf as i32;
        // Ledger A90: the window bound, and a negative window (see the doc comment).
        if w_i < 0 || w_i.unsigned_abs() as usize > max_half_width {
            return None;
        }
        // `:485`: a window of fewer than 15 pixels (checked before the cut; the cut is
        // always `(2w + 1)^2` pixels).
        if (2 * i64::from(w_i) + 1).pow(2) < 15 {
            return None;
        }
        // The window origin `(int)(v - w + 0.5)` must be an `int`, and the window's far
        // edge too (see the doc comment).
        let int_range = f64::from(i32::MIN)..f64::from(i32::MAX);
        if !int_range.contains(&(x - w + 0.5)) || !int_range.contains(&(y - w + 0.5)) {
            return None;
        }
        let off_x = window_origin(x, w, compat);
        let off_y = window_origin(y, w, compat);
        if off_x.checked_add(2 * w_i + 1).is_none() || off_y.checked_add(2 * w_i + 1).is_none() {
            return None;
        }
        // The window's values: whole grey levels for an 8-bit image, the 8-bit scale's
        // fractions for a deeper one (both exact in `f64`).
        let (cols, rows, sub): (usize, usize, Vec<f64>) = match image.window(w_i, off_x, off_y) {
            Patch::U8(p) => (
                p.width(),
                p.height(),
                p.data().iter().map(|&v| f64::from(v)).collect(),
            ),
            Patch::F32(p) => (
                p.width(),
                p.height(),
                p.data().iter().map(|&v| f64::from(v)).collect(),
            ),
        };
        debug_assert!(cols * rows >= 15);
        // A window whose pixels are all equal has no peak: no result (see the doc comment).
        let first = sub[0];
        if sub.iter().all(|&v| v == first) {
            return None;
        }

        let n = cols * rows;
        let mut a_rows = vec![0.0f64; n * 15];
        let mut b_vec = vec![0.0f64; n];
        let inv_radius_sq_maskmult = -1.0 / (radius * radius * MASKMULT);
        let mut count = 0;
        // Pixels in column order (j outer), as the C++ fills the rows of A.
        for j in 0..cols {
            for i in 0..rows {
                let tmpx = f64::from(off_x) - x + j as f64;
                let tmpy = f64::from(off_y) - y + i as f64;
                let tmpw = ((tmpx * tmpx + tmpy * tmpy) * inv_radius_sq_maskmult).exp();
                let v = sub[i * cols + j];
                b_vec[count] = if dark_marker {
                    tmpw * (255.0 - v)
                } else {
                    tmpw * v
                };
                let r = &mut a_rows[count * 15..count * 15 + 15];
                r[0] = tmpw;
                let mut ocol = 1;
                for order in 1..=4 {
                    for ocol2 in ocol..ocol + order {
                        r[ocol2] = tmpx * r[ocol2 - order];
                    }
                    ocol += order;
                    r[ocol] = tmpy * r[ocol - order - 1];
                    ocol += 1;
                }
                count += 1;
            }
        }
        let am = DMatrix::from_row_slice(n, 15, &a_rows);
        let bm = DVector::from_vec(b_vec);
        let p = linalg::solve_qr(&am, &bm).x;
        let pk = |k: usize| p[(k, 0)];

        let a = pk(3);
        let b = pk(4) / 2.0;
        let c = pk(5);
        let d = pk(1) / 2.0;
        let f = pk(2) / 2.0;

        let jj = a * c - b * b;
        last_j = jj;
        last_trace = a + c;
        let xc = (b * f - c * d) / jj;
        let yc = (b * d - a * f) / jj;

        x += sign(xc) * min_macro(xc.abs(), IMPROVERTHRESH);
        y += sign(yc) * min_macro(yc.abs(), IMPROVERTHRESH);

        let rotation = polyfit_rotation(a, b, c, compat);
        let ct = rotation.cos();
        let st = rotation.sin();
        let p1 = pk(10) * ct.powf(4.0) - pk(11) * ct.powf(3.0) * st + pk(12) * ct * ct * st * st
            - pk(13) * ct * st.powf(3.0)
            + pk(14) * st.powf(4.0);
        let p2 = pk(10) * st.powf(4.0)
            + pk(11) * st.powf(3.0) * ct
            + pk(12) * st * st * ct * ct
            + pk(13) * st * ct.powf(3.0)
            + pk(14) * ct.powf(4.0);
        let q1 = pk(3) * ct * ct - pk(4) * ct * st + pk(5) * st * st;
        let q2 = pk(3) * st * st + pk(4) * st * ct + pk(5) * ct * ct;
        radius = (q1 * q2 / p1 / p2 / 36.0).sqrt().sqrt().abs();

        stillgood = f64::from(refinementcount) <= maxrefinements && jj > 0.0;
        let improverswitch = xc.abs() > IMPROVERTHRESH || yc.abs() > IMPROVERTHRESH;
        if !improverswitch {
            doextracycles -= 1;
        }
    }

    if radius.is_nan() || x.is_nan() || y.is_nan() {
        None
    } else if !compat.refine_accepts_non_peak
        && !(last_j > 0.0 && last_trace < 0.0 && radius.is_finite() && radius > 0.0)
    {
        // Ledger A90: the last fit is not a maximum, or its radius is not a size.
        None
    } else {
        Some(Refinement {
            point: (x, y),
            radius,
            cycles: refinementcount,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_codes_round_trip() {
        for c in 0..=6 {
            assert_eq!(DetectionMethod::from_code(c).unwrap().code(), c);
        }
        assert!(DetectionMethod::from_code(7).is_none());
    }

    #[test]
    fn radius_and_sigma() {
        assert_eq!(blur_radius_sigma(5).0, 8);
        assert_eq!(blur_radius_sigma(12).0, 18);
        assert_eq!(blur_radius_sigma(1).0, 2);
        assert_eq!(blur_radius_sigma(0).0, 0);
        assert_eq!(blur_radius_sigma(5).1, 25.632343753043564);
    }

    #[test]
    fn window_origin_both_modes() {
        let (cpp, fixed) = (DetectCompat::XMALAB3, DetectCompat::FIXED);
        assert_eq!(DetectCompat::default(), fixed);
        // Positive: both round half up.
        assert_eq!(window_origin(48.0, 15.0, cpp), 33);
        assert_eq!(window_origin(48.0, 15.0, fixed), 33);
        assert_eq!(window_origin(47.5, 15.0, cpp), 33);
        assert_eq!(window_origin(47.4, 15.0, fixed), 32);
        // Negative: the C++ truncates towards zero (ledger A65).
        assert_eq!(window_origin(-3.6, 10.0, cpp), -13);
        assert_eq!(window_origin(-3.6, 10.0, fixed), -14);
        assert_eq!(window_origin(3.0, 10.0, cpp), -6);
        assert_eq!(window_origin(3.0, 10.0, fixed), -7);
        assert_eq!(window_origin(9.6, 10.0, cpp), 0);
        assert_eq!(window_origin(9.6, 10.0, fixed), 0);
        assert_eq!(window_origin(9.4, 10.0, cpp), 0);
        assert_eq!(window_origin(9.4, 10.0, fixed), -1);
    }

    /// The XMALab 3 expression is reproduced literally.
    #[test]
    fn rotation_compat_is_the_cpp_expression() {
        let cpp = DetectCompat::XMALAB3;
        for &(a, b, c) in &[
            (-3.0f64, 0.4f64, -1.0f64),
            (-1.0, 0.4, -3.0),
            (-2.0, -0.7, -2.5),
        ] {
            let lt = if a - c < 0.0 { 1.0 } else { 0.0 };
            let want = 0.5 * (PI / 2.0 - ((c - a) / 2.0 / b).atan() + lt) * PI / 2.0;
            assert_eq!(polyfit_rotation(a, b, c, cpp).to_bits(), want.to_bits());
        }
    }

    /// The fixed rotation puts the axes on the principal axes of `a x^2 + 2b xy + c y^2`: the
    /// cross term of the form on the rotated axes `(cos t, -sin t)`, `(sin t, cos t)` vanishes,
    /// and Q1, Q2 are its eigenvalues. The XMALab 3 expression does not.
    #[test]
    fn fixed_rotation_diagonalises() {
        let mut worst_cpp = 0.0f64;
        for &(a, b, c) in &[
            (-3.0f64, 0.4f64, -1.0f64),
            (-1.0, 0.4, -3.0),
            (-2.0, -0.7, -2.5),
            (-2.5, -0.7, -2.0),
            (-1.0, 1e-3, -1.2),
            (-4.0, 1.9, -1.0),
        ] {
            let cross = |t: f64| {
                let (ct, st) = (t.cos(), t.sin());
                // Bilinear form B(u, v) for u = (ct, -st), v = (st, ct).
                a * ct * st + b * (ct * ct - st * st) - c * st * ct
            };
            let t = polyfit_rotation(a, b, c, DetectCompat::FIXED);
            assert!(cross(t).abs() < 1e-12, "{a} {b} {c}: {}", cross(t));
            let (ct, st) = (t.cos(), t.sin());
            let q1 = a * ct * ct - 2.0 * b * ct * st + c * st * st;
            let q2 = a * st * st + 2.0 * b * st * ct + c * ct * ct;
            let tr = a + c;
            let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
            let (e1, e2) = ((tr - disc) / 2.0, (tr + disc) / 2.0);
            let (lo, hi) = (q1.min(q2), q1.max(q2));
            assert!((lo - e1).abs() < 1e-12 && (hi - e2).abs() < 1e-12);
            worst_cpp =
                worst_cpp.max(cross(polyfit_rotation(a, b, c, DetectCompat::XMALAB3)).abs());
        }
        assert!(worst_cpp > 0.1, "{worst_cpp}");
    }

    #[test]
    fn crosshair_as_cv_line() {
        let mut im = ImageU8::new(7, 7);
        draw_crosshair(&mut im, (3.0, 3.0));
        let set: Vec<(usize, usize)> = (0..7)
            .flat_map(|y| (0..7).map(move |x| (x, y)))
            .filter(|&(x, y)| im.at(x, y) == 127)
            .collect();
        assert_eq!(set.len(), 9);
        // Half-pixel centre: each end rounds to even (10.5 - 2 = 8.5 -> 8, 12.5 -> 12).
        let mut im = ImageU8::new(16, 16);
        draw_crosshair(&mut im, (10.5, 3.0));
        let row: Vec<usize> = (0..16).filter(|&x| im.at(x, 3) == 127).collect();
        assert_eq!(row, vec![8, 9, 10, 11, 12]);
        let col: Vec<usize> = (0..16).filter(|&y| im.at(10, y) == 127).collect();
        assert_eq!(col, vec![1, 2, 3, 4, 5]);
        // Clipped at the edge.
        let mut im = ImageU8::new(4, 4);
        draw_crosshair(&mut im, (0.0, 0.0));
        let n = im.data().iter().filter(|&&v| v == 127).count();
        assert_eq!(n, 5);
        // Outside: nothing.
        let mut im = ImageU8::new(4, 4);
        draw_crosshair(&mut im, (-3.0, 10.0));
        assert!(im.data().iter().all(|&v| v == 0));
    }

    fn dark_disc(w: usize, cx: f64, cy: f64, r: f64) -> ImageU8 {
        let mut im = ImageU8::filled(w, w, 200);
        for y in 0..w {
            for x in 0..w {
                let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                if d <= r {
                    im.set(x, y, 40);
                }
            }
        }
        im
    }

    #[test]
    fn method_4_finds_nothing_and_feature_methods_push_no_images() {
        let im = dark_disc(64, 32.0, 32.0, 4.0);
        for m in [
            DetectionMethod::NoDetection,
            DetectionMethod::Blob,
            DetectionMethod::WhiteBlob,
            DetectionMethod::Corner,
        ] {
            let (d, images) = detect_point_preview(
                &im,
                m,
                (31.0, 33.0),
                10,
                5,
                8.0,
                true,
                &DetectOptions::fixed(),
            );
            if m == DetectionMethod::NoDetection {
                assert_eq!(d, Detection::not_found((31.0, 33.0)));
            }
            // Only the contour methods fill the preview (`MarkerDetection.cpp:128-297`).
            assert!(images.is_empty());
        }
    }

    #[test]
    fn corner_method_finds_the_disc_edge() {
        // A dark disc on a flat background has corners (strong gradients) on its edge.
        let im = dark_disc(64, 32.0, 32.0, 4.0);
        let d = detect_point(
            &im,
            DetectionMethod::Corner,
            (31.0, 33.0),
            10,
            5,
            8.0,
            &DetectOptions::fixed(),
        );
        assert!(d.size > 0.0, "no corner found: {d:?}");
        let dist = ((d.point.0 - 32.0).powi(2) + (d.point.1 - 32.0).powi(2)).sqrt();
        assert!(dist < 8.0, "corner far from the disc: {d:?}");
    }

    #[test]
    fn preview_images_and_crosshair() {
        let im = dark_disc(64, 32.0, 32.0, 4.0);
        let (d, images) = detect_point_preview(
            &im,
            DetectionMethod::Default,
            (31.0, 33.0),
            10,
            4,
            8.0,
            false,
            &DetectOptions::fixed(),
        );
        assert_eq!(d.point, (32.0, 32.0));
        assert_eq!(images.len(), 7);
        assert_eq!(images[6], get_sub_image_at(&im, 10, 21, 23));
        assert_eq!(
            d,
            detect_point(
                &im,
                DetectionMethod::Default,
                (31.0, 33.0),
                10,
                4,
                8.0,
                &DetectOptions::fixed()
            )
        );
        let (_, drawn) = detect_point_preview(
            &im,
            DetectionMethod::Default,
            (31.0, 33.0),
            10,
            4,
            8.0,
            true,
            &DetectOptions::fixed(),
        );
        // The detected point (32, 32) is (11, 9) in the patch: each drawn image is the
        // undrawn one with exactly the crosshair's nine pixels set to grey 127.
        let cross: Vec<(usize, usize)> = (9..=13)
            .map(|x| (x, 9))
            .chain((7..=11).filter(|&y| y != 9).map(|y| (11, y)))
            .collect();
        assert_eq!(cross.len(), 9);
        assert_eq!(drawn.len(), 7);
        for (a, b) in images.iter().zip(&drawn) {
            let mut want = a.clone();
            for &(x, y) in &cross {
                want.set(x, y, 127);
            }
            assert_eq!(b, &want);
        }
        // Nothing found (a flat image with a negative threshold offset: the threshold is
        // below every value, so the thresholded image is empty): the six stage images, no
        // seventh, no crosshair.
        let flat = ImageU8::filled(64, 64, 0);
        for draw in [false, true] {
            let (d, six) = detect_point_preview(
                &flat,
                DetectionMethod::Default,
                (31.0, 33.0),
                10,
                4,
                -10.0,
                draw,
                &DetectOptions::fixed(),
            );
            assert_eq!(d, Detection::not_found((31.0, 33.0)));
            assert_eq!(six.len(), 6);
            assert!(six.iter().all(|im| im.data().iter().all(|&v| v != 127)));
        }
    }

    #[test]
    fn refinement_finds_a_disc_centre_and_rejects_nan() {
        // A smooth dark spot at (30.3, 29.6).
        let mut im = ImageU8::filled(64, 64, 220);
        for y in 0..64 {
            for x in 0..64 {
                let d2 = (x as f64 - 30.3).powi(2) + (y as f64 - 29.6).powi(2);
                let v = 220.0 - 160.0 * (-d2 / (2.0 * 2.5 * 2.5)).exp();
                im.set(x, y, v.round() as u8);
            }
        }
        for compat in [DetectCompat::FIXED, DetectCompat::XMALAB3] {
            let r = refine_point_polynomial_fit(&im, (31.0, 29.0), 3.0, true, compat).unwrap();
            assert!((r.point.0 - 30.3).abs() < 0.1, "{r:?}");
            assert!((r.point.1 - 29.6).abs() < 0.1, "{r:?}");
            assert!(r.radius.is_finite() && r.radius > 0.0);
        }
        // A NaN or too small radius: no result.
        assert!(
            refine_point_polynomial_fit(&im, (31.0, 29.0), f64::NAN, true, DetectCompat::FIXED)
                .is_none()
        );
        assert!(
            refine_point_polynomial_fit(&im, (31.0, 29.0), 0.5, true, DetectCompat::FIXED)
                .is_none()
        );
        assert!(
            refine_point_polynomial_fit(&im, (31.0, 29.0), -4.0, true, DetectCompat::XMALAB3)
                .is_none()
        );
    }
}
