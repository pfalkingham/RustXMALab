//! OpenCV's `SimpleBlobDetector` and XMALab's detection methods 1 and 6 built on it.
//!
//! [`detect_blobs`] reproduces `cv::SimpleBlobDetector::detect` (OpenCV 4.13,
//! `modules/features2d/src/blobdetector.cpp`) on an 8-bit grey image: the keypoints come back
//! in OpenCV's order with OpenCV's `float` positions and sizes. It is used by marker detection
//! (`processing/MarkerDetection.cpp`, methods 1 and 6, [`detect_blob_marker`]) and by the
//! undistortion-grid and calibration-bead detection (`processing/BlobDetection.cpp`).
//!
//! Tested against the primitive oracle (`simpleblob-*`, written by OpenCV 4.14 and checked
//! against the 4.13 libraries XMALab 3 links: no case differs).
//!
//! Images deeper than 8 bits (parity ledger V1, `crate::depth`): OpenCV's detector accepts
//! only 8-bit images. The deep path ([`detect_blobs_f32`], [`detect_blob_marker_f32`]) reads
//! the image as `f32` on the 8-bit scale and changes only the thresholding: each binary
//! image is `v > t ? 255 : 0` ([`threshold_binary_8bit_scale`]) with the same thresholds
//! `t` on the 0..255 scale; everything after it (contours, filters, grouping) is the 8-bit
//! detector unchanged. On whole-number values `v > t` and the 8-bit `v > floor(t)` agree,
//! so an 8-bit image promoted to a deeper one gives the 8-bit keypoints exactly.

use crate::detection::Detection;
use crate::image::{ImageF32, ImageU8};
use crate::imgproc::arith::{ThresholdType, threshold_u8};
use crate::imgproc::contours::{ChainApprox, PointI, RetrievalMode, find_contours};
use crate::imgproc::shape::{arc_length, contour_area, convex_hull, moments};

/// `cv::SimpleBlobDetector::Params`, field by field with OpenCV's types. [`Default`] gives
/// OpenCV's defaults (`SimpleBlobDetector::Params::Params()`), not XMALab's settings (see
/// [`BlobSettings`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlobParams {
    /// Step between thresholds (default 10).
    pub threshold_step: f32,
    /// First threshold (default 50).
    pub min_threshold: f32,
    /// Thresholds stay below this (default 220).
    pub max_threshold: f32,
    /// Number of thresholds a blob must be found at (default 2; `size_t` in OpenCV).
    pub min_repeatability: u64,
    /// Blobs closer than this are the same blob (default 10).
    pub min_dist_between_blobs: f32,
    /// Keep only blobs whose centre pixel in the binary image equals `blob_color` (default on).
    pub filter_by_color: bool,
    /// 0 for dark blobs, 255 for light ones (default 0).
    pub blob_color: u8,
    /// Filter by `m00` (default on).
    pub filter_by_area: bool,
    /// Smallest area kept (default 25).
    pub min_area: f32,
    /// Areas kept stay below this (default 5000).
    pub max_area: f32,
    /// Filter by `4 pi area / perimeter^2` (default off).
    pub filter_by_circularity: bool,
    /// Smallest circularity kept (default 0.8).
    pub min_circularity: f32,
    /// Circularities kept stay below this (default `FLT_MAX`).
    pub max_circularity: f32,
    /// Filter by the ratio of the smallest to the largest second moment (default on).
    pub filter_by_inertia: bool,
    /// Smallest inertia ratio kept (default 0.1).
    pub min_inertia_ratio: f32,
    /// Inertia ratios kept stay below this (default `FLT_MAX`).
    pub max_inertia_ratio: f32,
    /// Filter by area over convex-hull area (default on).
    pub filter_by_convexity: bool,
    /// Smallest convexity kept (default 0.95).
    pub min_convexity: f32,
    /// Convexities kept stay below this (default `FLT_MAX`).
    pub max_convexity: f32,
    /// OpenCV keeps each blob's contour for `getBlobContours` (default off). Has no effect on
    /// the keypoints; kept so the struct holds every field of OpenCV's.
    pub collect_contours: bool,
}

impl Default for BlobParams {
    /// Ported from features2d/src/blobdetector.cpp, `SimpleBlobDetector::Params::Params`.
    fn default() -> Self {
        BlobParams {
            threshold_step: 10.0,
            min_threshold: 50.0,
            max_threshold: 220.0,
            min_repeatability: 2,
            min_dist_between_blobs: 10.0,
            filter_by_color: true,
            blob_color: 0,
            filter_by_area: true,
            min_area: 25.0,
            max_area: 5000.0,
            filter_by_circularity: false,
            min_circularity: 0.8,
            max_circularity: f32::MAX,
            filter_by_inertia: true,
            min_inertia_ratio: 0.1,
            max_inertia_ratio: f32::MAX,
            filter_by_convexity: true,
            min_convexity: 0.95,
            max_convexity: f32::MAX,
            collect_contours: false,
        }
    }
}

/// Why OpenCV refuses a set of [`BlobParams`] (it throws `cv::Exception`; XMALab does not
/// catch it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobParamError {
    /// `thresholdStep > 0` fails.
    ThresholdStep,
    /// `0 <= minThreshold <= maxThreshold` fails.
    Thresholds,
    /// `minDistBetweenBlobs > 0` fails.
    MinDist,
    /// `0 < minArea <= maxArea` fails.
    Area,
    /// `0 < minCircularity <= maxCircularity` fails.
    Circularity,
    /// `0 < minInertiaRatio <= maxInertiaRatio` fails.
    Inertia,
    /// `0 < minConvexity <= maxConvexity` fails.
    Convexity,
    /// `minRepeatability != 0` fails (checked by `detect`, not by `create`).
    Repeatability,
    /// The minimum, maximum or step of the thresholds is not finite (infinite or NaN). Not an
    /// OpenCV check (parity ledger A217).
    ThresholdsNotFinite,
    /// The thresholds below `min(maxThreshold, 256)` ([`THRESHOLD_CAP`]) number more than
    /// [`MAX_THRESHOLDS`], or the step does not move the first threshold
    /// (`minThreshold + thresholdStep == minThreshold` with `minThreshold < maxThreshold`).
    /// OpenCV runs the loop for hours or for ever there. Not an OpenCV check (parity ledger
    /// A217).
    TooManyThresholds,
}

/// A sentence for the user: what is wrong and which `BlobDetector*` settings to look at (it
/// reaches the interface and `xma` as the reason bead or grid detection was refused).
impl std::fmt::Display for BlobParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BlobParamError::ThresholdStep => {
                "the blob detector's threshold step (BlobDetectorThresholdStep) must be \
                 greater than 0"
            }
            BlobParamError::Thresholds => {
                "the blob detector's thresholds must satisfy 0 <= minimum <= maximum \
                 (BlobDetectorMinThreshold, BlobDetectorMaxThreshold)"
            }
            BlobParamError::MinDist => {
                "the blob detector's minimum distance between blobs \
                 (BlobDetectorMinDistBetweenBlobs) must be greater than 0"
            }
            BlobParamError::Area => {
                "the blob detector's area limits must satisfy 0 < minimum <= maximum \
                 (BlobDetectorMinArea, BlobDetectorMaxArea)"
            }
            BlobParamError::Circularity => {
                "the blob detector's circularity limits must satisfy 0 < minimum <= maximum \
                 (BlobDetectorMinCircularity, BlobDetectorMaxCircularity)"
            }
            BlobParamError::Inertia => {
                "the blob detector's inertia ratio limits must satisfy 0 < minimum <= maximum \
                 (BlobDetectorMinInertiaRatio, BlobDetectorMaxInertiaRatio)"
            }
            BlobParamError::Convexity => {
                "the blob detector's convexity limits must satisfy 0 < minimum <= maximum \
                 (BlobDetectorMinConvexity, BlobDetectorMaxConvexity)"
            }
            BlobParamError::Repeatability => {
                "the blob detector's minimum repeatability (BlobDetectorMinRepeatability) \
                 must not be 0"
            }
            BlobParamError::ThresholdsNotFinite => {
                "the blob detector's threshold settings (minimum, maximum, step: \
                 BlobDetectorMinThreshold, BlobDetectorMaxThreshold, \
                 BlobDetectorThresholdStep) are not finite numbers"
            }
            BlobParamError::TooManyThresholds => {
                "the blob detector's threshold settings (minimum, maximum, step: \
                 BlobDetectorMinThreshold, BlobDetectorMaxThreshold, \
                 BlobDetectorThresholdStep) would need more than 100,000 thresholds; use a \
                 larger step or a narrower range"
            }
        })
    }
}

/// Thresholds of this value or more are not run: on the 8-bit path the binary image is empty
/// from threshold 255 (`floor(t) >= 255`), on the deep path values on the 8-bit scale are at
/// most 255.996 (`crate::depth::to_f32_8bit_scale`), and an empty binary image adds no blob.
/// (The one exception was OpenCV's: on the 8-bit path a threshold of 2^31 or more overflows
/// `cvFloor` and gives an all-white image.) Parity ledger A217.
pub const THRESHOLD_CAP: f64 = 256.0;

/// The most thresholds the detector runs (below [`THRESHOLD_CAP`]); settings that need more
/// are refused ([`BlobParamError::TooManyThresholds`], parity ledger A217).
pub const MAX_THRESHOLDS: u32 = 100_000;

impl BlobParams {
    /// The checks `SimpleBlobDetector::create` makes (in OpenCV's order), and the
    /// `minRepeatability != 0` assertion of `detect`. The comparisons are those of the C++, so
    /// a NaN bound passes where OpenCV lets it pass. Then, not in OpenCV (parity ledger A217):
    /// the thresholds must be finite ([`BlobParamError::ThresholdsNotFinite`]), and the loop
    /// must end within [`MAX_THRESHOLDS`] thresholds below [`THRESHOLD_CAP`]
    /// ([`BlobParamError::TooManyThresholds`]).
    ///
    /// Ported from features2d/src/blobdetector.cpp, `SimpleBlobDetectorImpl::validateParameters`
    /// and the start of `SimpleBlobDetectorImpl::detect`.
    ///
    /// # Errors
    /// The first check that fails.
    pub fn validate(&self) -> Result<(), BlobParamError> {
        let p = self;
        if p.threshold_step <= 0.0 {
            return Err(BlobParamError::ThresholdStep);
        }
        if p.min_threshold > p.max_threshold || p.min_threshold < 0.0 {
            return Err(BlobParamError::Thresholds);
        }
        if p.min_dist_between_blobs <= 0.0 {
            return Err(BlobParamError::MinDist);
        }
        if p.min_area > p.max_area || p.min_area <= 0.0 {
            return Err(BlobParamError::Area);
        }
        if p.min_circularity > p.max_circularity || p.min_circularity <= 0.0 {
            return Err(BlobParamError::Circularity);
        }
        if p.min_inertia_ratio > p.max_inertia_ratio || p.min_inertia_ratio <= 0.0 {
            return Err(BlobParamError::Inertia);
        }
        if p.min_convexity > p.max_convexity || p.min_convexity <= 0.0 {
            return Err(BlobParamError::Convexity);
        }
        if p.min_repeatability == 0 {
            return Err(BlobParamError::Repeatability);
        }
        // Parity ledger A217: the bound on the threshold loop, after OpenCV's own checks.
        if !(p.min_threshold.is_finite()
            && p.max_threshold.is_finite()
            && p.threshold_step.is_finite())
        {
            return Err(BlobParamError::ThresholdsNotFinite);
        }
        let (min, max, step) = threshold_range(p);
        if min < max && min + step == min {
            return Err(BlobParamError::TooManyThresholds);
        }
        // Count the thresholds the loop of `detect_blobs_with` runs, by the same repeated
        // addition, so that the count is exactly the loop's.
        let cap = max.min(THRESHOLD_CAP);
        let mut thresh = min;
        let mut n = 0u32;
        while thresh < cap {
            n += 1;
            if n > MAX_THRESHOLDS {
                return Err(BlobParamError::TooManyThresholds);
            }
            thresh += step;
        }
        Ok(())
    }
}

/// The first threshold, `maxThreshold` and the step, in `double` as OpenCV's loop holds them.
fn threshold_range(p: &BlobParams) -> (f64, f64, f64) {
    (
        f64::from(p.min_threshold),
        f64::from(p.max_threshold),
        f64::from(p.threshold_step),
    )
}

/// A `cv::KeyPoint` as `SimpleBlobDetector` fills it: `angle` -1, `response` 0, `octave` 0 and
/// `class_id` -1 always.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyPoint {
    /// Centre `(x, y)` in pixels (OpenCV `Point2f`).
    pub pt: (f32, f32),
    /// Diameter: twice the blob's median radius.
    pub size: f32,
    /// -1.
    pub angle: f32,
    /// 0.
    pub response: f32,
    /// 0.
    pub octave: i32,
    /// -1.
    pub class_id: i32,
}

/// `SimpleBlobDetectorImpl::Center`.
#[derive(Clone, Copy, Debug)]
struct Center {
    location: (f64, f64),
    radius: f64,
    confidence: f64,
}

/// `cv::norm(Point2d)`.
fn norm2(dx: f64, dy: f64) -> f64 {
    (dx * dx + dy * dy).sqrt()
}

/// `cvRound(double)`: half to even, as x86's `cvtsd2si`. Positions here are inside the image.
fn cv_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// The blobs of one binary image that pass the filters, in contour order.
///
/// Ported from features2d/src/blobdetector.cpp, `SimpleBlobDetectorImpl::findBlobs`.
fn find_blobs(binary: &ImageU8, p: &BlobParams) -> Vec<Center> {
    let contours = find_contours(
        binary,
        RetrievalMode::List,
        ChainApprox::None,
        PointI::new(0, 0),
    );
    let mut centers = Vec::new();
    for contour in &contours {
        let mut confidence = 1.0;
        let moms = moments(contour);
        if p.filter_by_area {
            let area = moms.m00;
            if area < f64::from(p.min_area) || area >= f64::from(p.max_area) {
                continue;
            }
        }
        if p.filter_by_circularity {
            let area = moms.m00;
            let perimeter = arc_length(contour, true);
            let ratio = 4.0 * std::f64::consts::PI * area / (perimeter * perimeter);
            if ratio < f64::from(p.min_circularity) || ratio >= f64::from(p.max_circularity) {
                continue;
            }
        }
        if p.filter_by_inertia {
            let (mu20, mu11, mu02) = (moms.mu20, moms.mu11, moms.mu02);
            // std::pow(x, 2) is x * x rounded once.
            let t = 2.0 * mu11;
            let d = mu20 - mu02;
            let denominator = (t * t + d * d).sqrt();
            let eps = 1e-2;
            let ratio = if denominator > eps {
                let cosmin = (mu20 - mu02) / denominator;
                let sinmin = 2.0 * mu11 / denominator;
                let cosmax = -cosmin;
                let sinmax = -sinmin;
                let imin = 0.5 * (mu20 + mu02) - 0.5 * (mu20 - mu02) * cosmin - mu11 * sinmin;
                let imax = 0.5 * (mu20 + mu02) - 0.5 * (mu20 - mu02) * cosmax - mu11 * sinmax;
                imin / imax
            } else {
                1.0
            };
            if ratio < f64::from(p.min_inertia_ratio) || ratio >= f64::from(p.max_inertia_ratio) {
                continue;
            }
            confidence = ratio * ratio;
        }
        if p.filter_by_convexity {
            let hull = convex_hull(contour);
            let area = moms.m00;
            let hull_area = contour_area(&hull, false);
            if hull_area.abs() < f64::EPSILON {
                continue;
            }
            let ratio = area / hull_area;
            if ratio < f64::from(p.min_convexity) || ratio >= f64::from(p.max_convexity) {
                continue;
            }
        }
        if moms.m00 == 0.0 {
            continue;
        }
        let location = (moms.m10 / moms.m00, moms.m01 / moms.m00);
        if p.filter_by_color {
            let (x, y) = (cv_round(location.0), cv_round(location.1));
            if binary.at(x as usize, y as usize) != p.blob_color {
                continue;
            }
        }
        let mut dists: Vec<f64> = contour
            .iter()
            .map(|q| norm2(location.0 - f64::from(q.x), location.1 - f64::from(q.y)))
            .collect();
        dists.sort_by(f64::total_cmp);
        let n = dists.len();
        let radius = (dists[(n - 1) / 2] + dists[n / 2]) / 2.0;
        centers.push(Center {
            location,
            radius,
            confidence,
        });
    }
    centers
}

/// `cv::SimpleBlobDetector::create(params)->detect(image, keypoints)` on an 8-bit grey image,
/// no mask. Returns the keypoints in OpenCV's order (the order in which each blob was first
/// seen, over increasing thresholds).
///
/// For each threshold `t = minThreshold, minThreshold + step, ...` below `maxThreshold`
/// (in `double`), the image is thresholded (`THRESH_BINARY`, 255), its contours found
/// (`RETR_LIST`, `CHAIN_APPROX_NONE`) and filtered; a blob within `minDistBetweenBlobs` (and
/// within both blobs' median radii) of a known blob's middle entry joins it, kept sorted by
/// radius. Blobs seen at `minRepeatability` thresholds or more become keypoints at their
/// confidence-weighted mean position, with twice the middle entry's radius as size.
///
/// Unlike OpenCV, thresholds of 256 or more are not run ([`THRESHOLD_CAP`]; they find
/// nothing), and settings that need more than [`MAX_THRESHOLDS`] thresholds, or have a
/// non-finite threshold setting, are refused (parity ledger A217).
///
/// Ported from features2d/src/blobdetector.cpp, `SimpleBlobDetectorImpl::detect` and
/// `SimpleBlobDetectorImpl::findBlobs`, with the image routines of [`crate::imgproc`].
///
/// # Errors
/// What `create` or `detect` would throw for these parameters ([`BlobParams::validate`]).
pub fn detect_blobs(image: &ImageU8, params: &BlobParams) -> Result<Vec<KeyPoint>, BlobParamError> {
    detect_blobs_with(
        |t| threshold_u8(image, t, 255.0, ThresholdType::Binary).0,
        params,
    )
}

/// The binary image the deep blob detector finds contours in at threshold `thresh`:
/// 255 where `v > thresh`, 0 elsewhere, compared in `f64` (exact for both). For a threshold
/// that is a `float` (XMALab's defaults 10, 20, ..., 210 are) this is OpenCV's
/// `threshold(src, dst, thresh, 255, THRESH_BINARY)` on a `CV_32F` image, which compares in
/// `float`, converted to 8 bits; on whole-number values it is the 8-bit threshold
/// (`v > floor(thresh)`). On a deep image this keeps values half a grey level darker than
/// the image reduced to 8 bits does: there a value `v` passes when `round(v) > floor(t)`,
/// that is from `floor(t) + 0.5` (`crate::depth`).
///
/// Ported from imgproc/src/thresh.cpp, `cv::threshold` (`thresh_32f`, `THRESH_BINARY`), as
/// the deep counterpart of the 8-bit threshold in
/// `SimpleBlobDetectorImpl::detect` (features2d/src/blobdetector.cpp).
#[must_use]
pub fn threshold_binary_8bit_scale(image: &ImageF32, thresh: f64) -> ImageU8 {
    image.map(|v| if f64::from(v) > thresh { 255 } else { 0 })
}

/// [`detect_blobs`] on a deeper image read as `f32` on the 8-bit scale: the thresholds are
/// applied with [`threshold_binary_8bit_scale`] and the rest is the 8-bit detector (see the
/// module documentation). The values are expected below 256, as
/// `crate::depth::to_f32_8bit_scale` gives them: thresholds of 256 or more are not run
/// ([`THRESHOLD_CAP`]).
///
/// Ported from features2d/src/blobdetector.cpp, `SimpleBlobDetectorImpl::detect` (extends
/// [`detect_blobs`]; OpenCV itself refuses images that are not 8-bit).
///
/// # Errors
/// As [`detect_blobs`].
pub fn detect_blobs_f32(
    image: &ImageF32,
    params: &BlobParams,
) -> Result<Vec<KeyPoint>, BlobParamError> {
    detect_blobs_with(|t| threshold_binary_8bit_scale(image, t), params)
}

/// The body of [`detect_blobs`] and [`detect_blobs_f32`]; `binarise(t)` is the binary image
/// at threshold `t`.
///
/// Ported from features2d/src/blobdetector.cpp, `SimpleBlobDetectorImpl::detect`.
fn detect_blobs_with(
    binarise: impl Fn(f64) -> ImageU8,
    params: &BlobParams,
) -> Result<Vec<KeyPoint>, BlobParamError> {
    params.validate()?;
    let p = params;
    let mut centers: Vec<Vec<Center>> = Vec::new();
    let (mut thresh, max, step) = threshold_range(p);
    // Parity ledger A217: thresholds of 256 or more find nothing and are not run; `validate`
    // has checked that at most `MAX_THRESHOLDS` remain.
    let max = max.min(THRESHOLD_CAP);
    while thresh < max {
        let binary = binarise(thresh);
        let cur = find_blobs(&binary, p);
        let mut new_centers: Vec<Vec<Center>> = Vec::new();
        for c in &cur {
            let mut is_new = true;
            for group in &mut centers {
                let mid = group[group.len() / 2];
                let dist = norm2(mid.location.0 - c.location.0, mid.location.1 - c.location.1);
                is_new = dist >= f64::from(p.min_dist_between_blobs)
                    && dist >= mid.radius
                    && dist >= c.radius;
                if !is_new {
                    group.push(*c);
                    let mut k = group.len() - 1;
                    while k > 0 && c.radius < group[k - 1].radius {
                        group[k] = group[k - 1];
                        k -= 1;
                    }
                    group[k] = *c;
                    break;
                }
            }
            if is_new {
                new_centers.push(vec![*c]);
            }
        }
        centers.extend(new_centers);
        thresh += step;
    }

    let mut keypoints = Vec::new();
    for group in &centers {
        if (group.len() as u64) < p.min_repeatability {
            continue;
        }
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        let mut normalizer = 0.0f64;
        for c in group {
            sx += c.confidence * c.location.0;
            sy += c.confidence * c.location.1;
            normalizer += c.confidence;
        }
        let inv = 1.0 / normalizer;
        sx *= inv;
        sy *= inv;
        keypoints.push(KeyPoint {
            pt: (sx as f32, sy as f32),
            size: (group[group.len() / 2].radius as f32) * 2.0,
            angle: -1.0,
            response: 0.0,
            octave: 0,
            class_id: -1,
        });
    }
    Ok(keypoints)
}

/// XMALab's blob-detector settings as `core/Settings.cpp` holds them (`BlobDetector*`), with
/// the settings' own types: `getFloatSetting` returns `float`, `getIntSetting` `int`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlobSettings {
    /// `BlobDetectorThresholdStep` (10).
    pub threshold_step: f32,
    /// `BlobDetectorMinThreshold` (10).
    pub min_threshold: f32,
    /// `BlobDetectorMaxThreshold` (220).
    pub max_threshold: f32,
    /// `BlobDetectorMinRepeatability` (2).
    pub min_repeatability: i32,
    /// `BlobDetectorMinDistBetweenBlobs` (5).
    pub min_dist_between_blobs: f32,
    /// `BlobDetectorFilterByColor` (true).
    pub filter_by_color: bool,
    /// `BlobDetectorBlobColor` (255).
    pub blob_color: i32,
    /// `BlobDetectorFilterByArea` (true).
    pub filter_by_area: bool,
    /// `BlobDetectorMinArea` (25).
    pub min_area: f32,
    /// `BlobDetectorMaxArea` (5000).
    pub max_area: f32,
    /// `BlobDetectorFilterByCircularity` (true).
    pub filter_by_circularity: bool,
    /// `BlobDetectorMinCircularity` (0.5).
    pub min_circularity: f32,
    /// `BlobDetectorMaxCircularity` (5000000).
    pub max_circularity: f32,
    /// `BlobDetectorFilterByInertia` (true).
    pub filter_by_inertia: bool,
    /// `BlobDetectorMinInertiaRatio` (0.5).
    pub min_inertia_ratio: f32,
    /// `BlobDetectorMaxInertiaRatio` (5000000).
    pub max_inertia_ratio: f32,
    /// `BlobDetectorFilterByConvexity` (true).
    pub filter_by_convexity: bool,
    /// `BlobDetectorMinConvexity` (0.9f).
    pub min_convexity: f32,
    /// `BlobDetectorMaxConvexity` (5000000).
    pub max_convexity: f32,
}

impl Default for BlobSettings {
    /// XMALab's defaults.
    ///
    /// Ported from core/Settings.cpp, `Settings::Settings` (lines 138-160).
    fn default() -> Self {
        BlobSettings {
            threshold_step: 10.0,
            min_threshold: 10.0,
            max_threshold: 220.0,
            min_repeatability: 2,
            min_dist_between_blobs: 5.0,
            filter_by_color: true,
            blob_color: 255,
            filter_by_area: true,
            min_area: 25.0,
            max_area: 5000.0,
            filter_by_circularity: true,
            min_circularity: 0.5,
            max_circularity: 5_000_000.0,
            filter_by_inertia: true,
            min_inertia_ratio: 0.5,
            max_inertia_ratio: 5_000_000.0,
            filter_by_convexity: true,
            min_convexity: 0.9,
            max_convexity: 5_000_000.0,
        }
    }
}

impl BlobSettings {
    /// The `SimpleBlobDetector::Params` XMALab builds from the settings. `invert_color` gives
    /// `blobColor = 255 - BlobDetectorBlobColor` (method 1, and calibration images without
    /// white blobs); otherwise `blobColor = BlobDetectorBlobColor` (method 6, the undistortion
    /// grid, calibration images with white blobs). The `int` is converted to `uchar` modulo
    /// 256 and `minRepeatability` to `size_t` as C++ does (a negative value becomes huge, so no
    /// blob is kept; 0 makes `detect` throw).
    ///
    /// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint`
    /// (lines 349-381), and processing/BlobDetection.cpp, `BlobDetection::process`
    /// (lines 72-104).
    #[must_use]
    pub fn to_params(&self, invert_color: bool) -> BlobParams {
        let s = self;
        let color = if invert_color {
            255i32.wrapping_sub(s.blob_color)
        } else {
            s.blob_color
        };
        BlobParams {
            threshold_step: s.threshold_step,
            min_threshold: s.min_threshold,
            max_threshold: s.max_threshold,
            min_repeatability: i64::from(s.min_repeatability) as u64,
            min_dist_between_blobs: s.min_dist_between_blobs,
            filter_by_color: s.filter_by_color,
            blob_color: color as u8,
            filter_by_area: s.filter_by_area,
            min_area: s.min_area,
            max_area: s.max_area,
            filter_by_circularity: s.filter_by_circularity,
            min_circularity: s.min_circularity,
            max_circularity: s.max_circularity,
            filter_by_inertia: s.filter_by_inertia,
            min_inertia_ratio: s.min_inertia_ratio,
            max_inertia_ratio: s.max_inertia_ratio,
            filter_by_convexity: s.filter_by_convexity,
            min_convexity: s.min_convexity,
            max_convexity: s.max_convexity,
            collect_contours: false,
        }
    }
}

/// How the candidate selection of detection methods 1, 3 and 6 works: the fixed behaviour or
/// XMALab 3's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionCompat {
    /// XMALab 3 measures each candidate's distance from patch index `searchArea + 1`, one
    /// pixel off the patch centre `searchArea` (which is itself the seed rounded to a whole
    /// pixel) (parity ledger A14). Fixed: from the seed itself, at `center - offset` in patch
    /// coordinates, as methods 0, 2 and 5 do (`MarkerDetection.cpp:248`).
    pub search_centre_plus_one: bool,
    /// XMALab 3 adds the integer patch offset to the `float` coordinate in `float`
    /// (`off_x + keypoints[i].pt.x`), rounding the image position to single precision.
    /// Fixed: the sum is taken in `double` (exact).
    pub offset_sum_f32: bool,
}

impl SelectionCompat {
    /// The fixed behaviour.
    pub const FIXED: Self = SelectionCompat {
        search_centre_plus_one: false,
        offset_sum_f32: false,
    };
    /// XMALab 3's behaviour.
    pub const XMALAB3: Self = SelectionCompat {
        search_centre_plus_one: true,
        offset_sum_f32: true,
    };

    /// The image position of a patch coordinate.
    pub(crate) fn place(self, off: i32, v: f32) -> f64 {
        if self.offset_sum_f32 {
            f64::from(off as f32 + v)
        } else {
            f64::from(off) + f64::from(v)
        }
    }

    /// The point distances are measured from, in patch coordinates: XMALab 3's
    /// `(searchArea + 1, searchArea + 1)`, or fixed the seed `center` less the patch's
    /// top-left corner `off`.
    pub(crate) fn search_centre(
        self,
        search_area: i32,
        center: (f64, f64),
        off: (i32, i32),
    ) -> (f64, f64) {
        if self.search_centre_plus_one {
            let c = f64::from(search_area) + 1.0;
            (c, c)
        } else {
            (center.0 - f64::from(off.0), center.1 - f64::from(off.1))
        }
    }
}

impl Default for SelectionCompat {
    fn default() -> Self {
        Self::FIXED
    }
}

/// Detection methods 1 (dark blob) and 6 (white blob) on the cut patch: the blob detector with
/// XMALab's settings (`blobColor = 255 - setting` for method 1), then the keypoint nearest the
/// search centre ([`SelectionCompat::search_centre_plus_one`]: the seed, or XMALab 3's patch
/// index `searchArea + 1`) within `searchArea` (strictly closer; the first wins a tie). Every keypoint
/// is a candidate, in OpenCV's order, at its image position, with the keypoint's size (a
/// diameter, parity ledger A69). Nothing found gives `center` and size -1.
///
/// `patch` is `getSubImage(subimage, searchArea, off_x, off_y)`, `(2 * searchArea + 1)`
/// pixels square; `center` the input point (returned when nothing is found).
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint` (the branch
/// `method == 1 || method == 6`, lines 347-408).
///
/// # Errors
/// The parameter check OpenCV makes ([`BlobParams::validate`]); the C++ throws an exception
/// there, which XMALab does not catch.
///
/// # Panics
/// If `method` is not 1 or 6.
#[allow(clippy::too_many_arguments)]
pub fn detect_blob_marker(
    patch: &ImageU8,
    search_area: i32,
    off_x: i32,
    off_y: i32,
    center: (f64, f64),
    settings: &BlobSettings,
    method: i32,
    compat: SelectionCompat,
) -> Result<Detection, BlobParamError> {
    assert!(
        method == 1 || method == 6,
        "blob detection is method 1 or 6"
    );
    let params = settings.to_params(method == 1);
    let keypoints = detect_blobs(patch, &params)?;
    Ok(select_blob(
        &keypoints,
        search_area,
        off_x,
        off_y,
        center,
        compat,
    ))
}

/// [`detect_blob_marker`] on the patch of a deeper image, as `f32` on the 8-bit scale
/// ([`detect_blobs_f32`]; parity ledger V1). The settings are used unchanged: thresholds on
/// the 0..255 scale, areas and distances in pixels.
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint` (the branch
/// `method == 1 || method == 6`; extends [`detect_blob_marker`]).
///
/// # Errors
/// As [`detect_blob_marker`].
///
/// # Panics
/// If `method` is not 1 or 6.
#[allow(clippy::too_many_arguments)]
pub fn detect_blob_marker_f32(
    patch: &ImageF32,
    search_area: i32,
    off_x: i32,
    off_y: i32,
    center: (f64, f64),
    settings: &BlobSettings,
    method: i32,
    compat: SelectionCompat,
) -> Result<Detection, BlobParamError> {
    assert!(
        method == 1 || method == 6,
        "blob detection is method 1 or 6"
    );
    let params = settings.to_params(method == 1);
    let keypoints = detect_blobs_f32(patch, &params)?;
    Ok(select_blob(
        &keypoints,
        search_area,
        off_x,
        off_y,
        center,
        compat,
    ))
}

/// The selection of methods 1 and 6 among the keypoints (see [`detect_blob_marker`]).
///
/// Ported from processing/MarkerDetection.cpp, `MarkerDetection::detectionPoint` (lines
/// 383-408).
fn select_blob(
    keypoints: &[KeyPoint],
    search_area: i32,
    off_x: i32,
    off_y: i32,
    center: (f64, f64),
    compat: SelectionCompat,
) -> Detection {
    let mut out = Detection {
        point: center,
        size: -1.0,
        all_centres: Vec::with_capacity(keypoints.len()),
        all_sizes: Vec::with_capacity(keypoints.len()),
    };
    let sc = compat.search_centre(search_area, center, (off_x, off_y));
    let mut dist_min_sq = f64::from(search_area) * f64::from(search_area);
    for k in keypoints {
        let dx = f64::from(k.pt.0) - sc.0;
        let dy = f64::from(k.pt.1) - sc.1;
        let dist_sq = dx * dx + dy * dy;
        let p = (compat.place(off_x, k.pt.0), compat.place(off_y, k.pt.1));
        out.all_centres.push(p);
        out.all_sizes.push(f64::from(k.size));
        if dist_sq < dist_min_sq {
            out.point = p;
            out.size = f64::from(k.size);
            dist_min_sq = dist_sq;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_conversion() {
        let s = BlobSettings::default();
        assert_eq!(s.to_params(true).blob_color, 0);
        assert_eq!(s.to_params(false).blob_color, 255);
        let mut t = s;
        t.blob_color = 300;
        assert_eq!(t.to_params(false).blob_color, 44);
        assert_eq!(t.to_params(true).blob_color, 211); // -45 as uchar
        t.min_repeatability = -1;
        assert_eq!(t.to_params(false).min_repeatability, u64::MAX);
        t.min_repeatability = 0;
        assert_eq!(
            detect_blobs(&ImageU8::new(4, 4), &t.to_params(false)),
            Err(BlobParamError::Repeatability)
        );
    }

    #[test]
    fn parameter_checks() {
        let ok = BlobParams::default();
        assert!(ok.validate().is_ok());
        let bad = |f: &dyn Fn(&mut BlobParams)| {
            let mut p = ok;
            f(&mut p);
            p.validate().unwrap_err()
        };
        assert_eq!(
            bad(&|p| p.threshold_step = 0.0),
            BlobParamError::ThresholdStep
        );
        assert_eq!(
            bad(&|p| p.min_threshold = 230.0),
            BlobParamError::Thresholds
        );
        assert_eq!(bad(&|p| p.min_threshold = -1.0), BlobParamError::Thresholds);
        assert_eq!(
            bad(&|p| p.min_dist_between_blobs = 0.0),
            BlobParamError::MinDist
        );
        assert_eq!(bad(&|p| p.min_area = 0.0), BlobParamError::Area);
        assert_eq!(
            bad(&|p| p.min_circularity = 0.0),
            BlobParamError::Circularity
        );
        assert_eq!(
            bad(&|p| p.max_inertia_ratio = 0.05),
            BlobParamError::Inertia
        );
        assert_eq!(bad(&|p| p.min_convexity = -0.1), BlobParamError::Convexity);
    }

    #[test]
    fn selection_offsets() {
        let v = 30.889_065_f32;
        assert_eq!(SelectionCompat::FIXED.place(513, v), 513.0 + f64::from(v));
        assert_ne!(SelectionCompat::FIXED.place(513, v), f64::from(513.0 + v));
        assert_eq!(SelectionCompat::XMALAB3.place(513, v), f64::from(513.0 + v));
        assert_eq!(
            SelectionCompat::XMALAB3.search_centre(10, (512.4, 300.6), (502, 291)),
            (11.0, 11.0)
        );
        // Fixed: the seed in patch coordinates, fraction included.
        let (x, y) = SelectionCompat::FIXED.search_centre(10, (512.4, 300.6), (502, 291));
        assert_eq!((x, y), (512.4 - 502.0, 300.6 - 291.0));
        assert!((x - 10.4).abs() < 1e-12 && (y - 9.6).abs() < 1e-12);
    }
}
