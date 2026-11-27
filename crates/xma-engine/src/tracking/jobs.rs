//! The two jobs of the C++ tracking loop as plain values: a tracker for one (marker, camera)
//! pair and one frame step ([`TrackerJob`], `processing/MarkerTracking.cpp`), and a
//! detection at one point ([`DetectionJob`], `processing/MarkerDetection.cpp`).
//!
//! Each job is built from the project as the C++ constructor reads it (on the main thread,
//! before any job of its batch runs), then run on an image without touching the project, so
//! that a batch can run on several threads and its results be written afterwards in a fixed
//! order. What a job writes back is the caller's ([`super`]).
//!
//! # Image depth (parity ledger V1)
//!
//! A frame becomes the image the algorithms read in one place, [`algorithm_image_for`]
//! ([`AlgorithmImage`]), and the jobs choose the algorithm from what it holds:
//!
//! - an 8-bit frame: the 8-bit image, in every mode (an 8-bit source is processed exactly as
//!   XMALab 3 processes it);
//! - a deeper frame in the XMALab 3 mode: reduced to 8 bits as the C++ does
//!   ([`algorithm_image`], `Frame::to_gray8`), then the 8-bit algorithms;
//! - a deeper frame in any other mode ([`full_depth`]): the samples at the frame's own depth
//!   (values above `2^bit_depth - 1` saturated to it, as the reduction saturates them), and
//!   the full-depth algorithms of xma-vision (`xma_vision::depth`).
//!
//! The tracker's template is cut at one frame and searched for at the next. When both
//! images are of one kind the pair takes that kind's path. When a source changes kind
//! between the two frames (an image sequence mixing 8-bit and deeper files, read in the
//! default mode), the step runs on the full-depth path with the 8-bit side taken exactly onto
//! it (an 8-bit template as `f32`, an 8-bit image as 8-bit samples of depth 8), so nothing is
//! rounded and the result is the deep tracker's on those values. Two deep frames of different
//! depths need nothing: both are read on the 8-bit scale.

use std::sync::Arc;

use xma_model::{CppCompat, DetectionMethod, Marker, MarkerStatus, Point2};
use xma_video::{Frame, Pixels};
use xma_vision::detection::{self, DetectCompat, DetectOptions, Detection, Refinement};
use xma_vision::image::{ImageU8, ImageU16};
use xma_vision::imgproc::blob::{BlobSettings, SelectionCompat};
use xma_vision::tracking::{MarkerTemplate, Prediction, TrackCompat, TrackStep};

use crate::document::Document;

/// The 8-bit image XMALab 3 holds for a frame (parity ledger V1, as the C++ `Image::getImage`
/// holds it: a deeper frame scaled by `2^-(bit_depth - 8)`, rounded half to even and
/// saturated). This is the image of the XMALab 3 mode; the tracking code reads frames through
/// [`algorithm_image_for`], which keeps a deeper frame at its depth in the default mode.
///
/// Ported from core/Image.cpp, Image::setImage (the `convertTo(CV_8U, ...)` of a 16-bit
/// image), through [`Frame::to_gray8`].
#[must_use]
pub fn algorithm_image(frame: &Frame) -> ImageU8 {
    ImageU8::from_vec(
        frame.width as usize,
        frame.height as usize,
        frame.to_gray8().into_owned(),
    )
}

/// Whether a mode processes a frame deeper than 8 bits at its own depth (parity ledger V1):
/// every mode but XMALab 3's, which reduces it to 8 bits as the C++ does. The same test
/// chooses the XMALab 3 readers ([`super::TrialSources::new`]).
#[must_use]
pub fn full_depth(compat: CppCompat) -> bool {
    compat != CppCompat::XMALAB3
}

/// The image the algorithms read for a frame: 8-bit, or deeper samples with their depth.
#[derive(Clone, Debug, PartialEq)]
pub enum AlgorithmImage {
    /// 8-bit grey: an 8-bit frame, or a deeper one reduced as XMALab 3 reduces it.
    U8(ImageU8),
    /// The samples of a deeper frame, below `2^bit_depth` (9 to 16 bits).
    Deep { image: ImageU16, bit_depth: u8 },
}

impl AlgorithmImage {
    /// A borrowed view, as the jobs take it.
    #[must_use]
    pub fn view(&self) -> ImageRef<'_> {
        match self {
            AlgorithmImage::U8(im) => ImageRef::U8(im),
            AlgorithmImage::Deep { image, bit_depth } => ImageRef::Deep(image, *bit_depth),
        }
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        match self {
            AlgorithmImage::U8(im) => im.width(),
            AlgorithmImage::Deep { image, .. } => image.width(),
        }
    }
}

/// The image the algorithms read for `frame` in a mode (see the module documentation): an
/// 8-bit frame as it is; a deeper frame at its own depth when `full_depth`, else reduced to
/// 8 bits as XMALab 3 does ([`algorithm_image`]). Deep samples above `2^bit_depth - 1` (a
/// `.cine` can hold some) are saturated to that value, as the reduction saturates them.
#[must_use]
pub fn algorithm_image_in(frame: &Frame, full_depth: bool) -> AlgorithmImage {
    match &frame.pixels {
        Pixels::Gray16(data) if full_depth => {
            let bit_depth = frame.bit_depth.clamp(8, 16);
            let top = u16::MAX >> (16 - bit_depth);
            let image = ImageU16::from_vec(
                frame.width as usize,
                frame.height as usize,
                data.iter().map(|&v| v.min(top)).collect(),
            );
            AlgorithmImage::Deep { image, bit_depth }
        }
        _ => AlgorithmImage::U8(algorithm_image(frame)),
    }
}

/// [`algorithm_image_in`] for a document's mode ([`full_depth`]).
#[must_use]
pub fn algorithm_image_for(frame: &Frame, compat: CppCompat) -> AlgorithmImage {
    algorithm_image_in(frame, full_depth(compat))
}

/// [`algorithm_image_for`] of a shared frame.
pub(crate) fn algorithm_image_arc(frame: &Arc<Frame>, compat: CppCompat) -> AlgorithmImage {
    algorithm_image_for(frame, compat)
}

/// An image as the jobs read it: 8-bit, or deep samples with their bit depth. Made from an
/// [`AlgorithmImage`] or an 8-bit image (shared or not).
#[derive(Clone, Copy, Debug)]
pub enum ImageRef<'a> {
    U8(&'a ImageU8),
    Deep(&'a ImageU16, u8),
}

impl<'a> From<&'a ImageU8> for ImageRef<'a> {
    fn from(im: &'a ImageU8) -> Self {
        ImageRef::U8(im)
    }
}

impl<'a> From<&'a Arc<ImageU8>> for ImageRef<'a> {
    fn from(im: &'a Arc<ImageU8>) -> Self {
        ImageRef::U8(im)
    }
}

impl<'a> From<&'a AlgorithmImage> for ImageRef<'a> {
    fn from(im: &'a AlgorithmImage) -> Self {
        im.view()
    }
}

impl<'a> From<&'a Arc<AlgorithmImage>> for ImageRef<'a> {
    fn from(im: &'a Arc<AlgorithmImage>) -> Self {
        im.view()
    }
}

/// The polynomial point refinement on an image of either kind: 8-bit
/// (`refine_point_polynomial_fit`) or at full depth (`refine_point_polynomial_fit_u16`).
///
/// Ported from processing/MarkerDetection.cpp, MarkerDetection::refinePointPolynomialFit,
/// through xma-vision.
#[must_use]
pub fn refine_point<'a>(
    image: impl Into<ImageRef<'a>>,
    point: (f64, f64),
    radius: f64,
    dark_marker: bool,
    compat: DetectCompat,
) -> Option<Refinement> {
    match image.into() {
        ImageRef::U8(im) => {
            detection::refine_point_polynomial_fit(im, point, radius, dark_marker, compat)
        }
        ImageRef::Deep(im, d) => {
            detection::refine_point_polynomial_fit_u16(im, d, point, radius, dark_marker, compat)
        }
    }
}

/// The marker detection at one point on an image of either kind: 8-bit (`detect_point`) or
/// at full depth (`detect_point_u16`).
///
/// Ported from processing/MarkerDetection.cpp, MarkerDetection::detectionPoint, through
/// xma-vision.
#[must_use]
pub fn detect_point<'a>(
    image: impl Into<ImageRef<'a>>,
    method: detection::DetectionMethod,
    center: (f64, f64),
    search_area: i32,
    mask_size: i32,
    threshold: f64,
    options: &DetectOptions,
) -> Detection {
    match image.into() {
        ImageRef::U8(im) => detection::detect_point(
            im,
            method,
            center,
            search_area,
            mask_size,
            threshold,
            options,
        ),
        ImageRef::Deep(im, depth) => detection::detect_point_u16(
            im,
            depth,
            method,
            center,
            search_area,
            mask_size,
            threshold,
            options,
        ),
    }
}

/// The Detection Settings preview's detection on an image of either kind
/// (`detect_point_preview` or `detect_point_preview_u16`); the stage images are 8-bit in
/// both.
///
/// Ported from processing/MarkerDetection.cpp, MarkerDetection::detectionPoint (preview),
/// through xma-vision.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn detect_preview<'a>(
    image: impl Into<ImageRef<'a>>,
    method: detection::DetectionMethod,
    center: (f64, f64),
    search_area: i32,
    mask_size: i32,
    threshold: f64,
    crosshair: bool,
    options: &DetectOptions,
) -> (Detection, Vec<ImageU8>) {
    match image.into() {
        ImageRef::U8(im) => detection::detect_point_preview(
            im,
            method,
            center,
            search_area,
            mask_size,
            threshold,
            crosshair,
            options,
        ),
        ImageRef::Deep(im, d) => detection::detect_point_preview_u16(
            im,
            d,
            method,
            center,
            search_area,
            mask_size,
            threshold,
            crosshair,
            options,
        ),
    }
}

/// The tracker's switches for a document's mode (ledger A64, A65).
#[must_use]
pub fn track_compat(compat: CppCompat) -> TrackCompat {
    TrackCompat {
        penalty_off_centre: compat.penalty_off_centre,
        window_round_toward_zero: compat.window_round_toward_zero,
    }
}

/// The detection options for a document: its mode (ledger A14, A65, A66, A73, A90) and the
/// `BlobDetector*` settings, which the C++ reads as `float` and `int`
/// (`processing/MarkerDetection.cpp:350-379`).
#[must_use]
pub fn detect_options(doc: &Document) -> DetectOptions {
    let compat = doc.compat();
    let b = &doc.settings().blob_detector;
    DetectOptions {
        compat: DetectCompat {
            window_round_toward_zero: compat.window_round_toward_zero,
            polyfit_rotation_bracket: compat.polyfit_rotation_bracket,
            refine_accepts_non_peak: compat.refine_accepts_non_peak,
        },
        selection: SelectionCompat {
            search_centre_plus_one: compat.search_centre_plus_one,
            offset_sum_f32: compat.offset_sum_f32,
        },
        blob: BlobSettings {
            threshold_step: b.threshold_step as f32,
            min_threshold: b.min_threshold as f32,
            max_threshold: b.max_threshold as f32,
            min_repeatability: b.min_repeatability as i32,
            min_dist_between_blobs: b.min_dist_between_blobs as f32,
            filter_by_color: b.filter_by_color,
            blob_color: i32::from(b.blob_color),
            filter_by_area: b.filter_by_area,
            min_area: b.min_area as f32,
            max_area: b.max_area as f32,
            filter_by_circularity: b.filter_by_circularity,
            min_circularity: b.min_circularity as f32,
            max_circularity: b.max_circularity as f32,
            filter_by_inertia: b.filter_by_inertia,
            min_inertia_ratio: b.min_inertia_ratio as f32,
            max_inertia_ratio: b.max_inertia_ratio as f32,
            filter_by_convexity: b.filter_by_convexity,
            min_convexity: b.min_convexity as f32,
            max_convexity: b.max_convexity as f32,
        },
    }
}

/// What `Marker::getMarkerPrediction` returns: 2 with the constant-velocity position from
/// the two previous frames in the direction of travel, 1 with the previous frame's position,
/// 0 (and no position) otherwise. "Previous" frames count when their status is above
/// `UNDEFINED` (so `PREDICTED` and `INTERPOLATED` points count).
///
/// Ported from core/Marker.cpp, Marker::getMarkerPrediction (`:179-207`).
#[must_use]
pub fn marker_prediction(
    marker: &Marker,
    camera: usize,
    frame: usize,
    forward: bool,
) -> (i32, Option<Point2>) {
    let n = marker.n_frames();
    if frame >= n {
        return (0, None);
    }
    let back = |k: usize| -> Option<usize> {
        if forward {
            frame.checked_sub(k)
        } else {
            Some(frame + k).filter(|&f| f < n)
        }
    };
    let Some(p1) = back(1).and_then(|f| marker.point2d(camera, f)) else {
        return (0, None);
    };
    match back(2).and_then(|f| marker.point2d(camera, f)) {
        Some(p2) => (2, Some(Point2::new(2.0 * p1.x - p2.x, 2.0 * p1.y - p2.y))),
        None => (1, Some(p1)),
    }
}

/// A tracker's template: 8-bit, or `f32` on the 8-bit scale when cut from a deeper image.
#[derive(Clone, Debug, PartialEq)]
pub enum TemplateImage {
    U8(MarkerTemplate),
    Deep(MarkerTemplate<f32>),
}

/// The template a tracker cut at its start frame.
#[derive(Clone, Debug, PartialEq)]
pub struct JobTemplate {
    /// `size` of the C++ (`MarkerTracking::size`).
    pub size: i32,
    pub image: TemplateImage,
}

impl JobTemplate {
    /// Cuts the template around `(x, y)`: `cut_template` on an 8-bit image, `cut_template_u16`
    /// on a deeper one.
    ///
    /// Ported from processing/MarkerTracking.cpp, MarkerTracking::MarkerTracking, through
    /// xma-vision.
    #[must_use]
    pub fn cut(image: ImageRef<'_>, x: f64, y: f64, marker_size: f64) -> JobTemplate {
        let image =
            match image {
                ImageRef::U8(im) => {
                    TemplateImage::U8(xma_vision::tracking::cut_template(im, x, y, marker_size))
                }
                ImageRef::Deep(im, d) => TemplateImage::Deep(
                    xma_vision::tracking::cut_template_u16(im, d, x, y, marker_size),
                ),
            };
        let size = match &image {
            TemplateImage::U8(t) => t.size,
            TemplateImage::Deep(t) => t.size,
        };
        JobTemplate { size, image }
    }
}

/// One tracker: a (marker, camera) pair from one frame to the next.
#[derive(Clone, Debug)]
pub struct TrackerJob {
    pub marker: usize,
    pub camera: usize,
    /// The frame the template was cut at (`frame_from`: the start frame).
    pub frame_from: usize,
    pub frame_to: usize,
    pub forward: bool,
    /// The position the template was cut around (`x_from`, `y_from`).
    pub from: Point2,
    pub template: JobTemplate,
    /// `Marker::getMaxPenalty()`.
    pub max_penalty: i32,
    /// A position that replaces the prediction (`setPredictionOverride`; the loop sets none
    /// since merge-aware tracking, its only user, was removed, parity ledger A226).
    pub prediction_override: Option<Point2>,
}

/// What a tracker found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerResult {
    /// `getMarkerPrediction`'s return value (1 with an override).
    pub prediction: i32,
    /// The search centre (`x_to`, `y_to` before the search).
    pub predicted: Point2,
    /// The step's details; `None` when the search region is smaller than the template.
    pub step: Option<TrackStep>,
    /// The position written (`x_to`, `y_to` after the search): whole pixels.
    pub point: Point2,
}

impl TrackerJob {
    /// The constructor: the marker's point at `frame_from`, its mean size and penalty, and
    /// the template cut from `image_from` (the image the camera holds when the tracker is
    /// constructed), 8-bit or deep as the image is (module documentation).
    ///
    /// Ported from processing/MarkerTracking.cpp, MarkerTracking::MarkerTracking
    /// (`:124-142`).
    #[must_use]
    pub fn new<'a>(
        marker: &Marker,
        marker_index: usize,
        camera: usize,
        frame_from: usize,
        frame_to: usize,
        forward: bool,
        image_from: impl Into<ImageRef<'a>>,
    ) -> TrackerJob {
        let from = marker.views[camera].points[frame_from];
        TrackerJob {
            marker: marker_index,
            camera,
            frame_from,
            frame_to,
            forward,
            from,
            template: JobTemplate::cut(image_from.into(), from.x, from.y, marker.mean_size),
            max_penalty: marker.max_penalty,
            prediction_override: None,
        }
    }

    /// The tracker thread: the prediction at `frame_to` from the marker's points as they are
    /// now, then the template search in `image_to`, on the 8-bit or the full-depth path as
    /// the template and the image are (module documentation).
    ///
    /// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_thread (CPU
    /// branch; the OpenCL branch is not ported, ledger A6).
    #[must_use]
    pub fn run<'a>(
        &self,
        marker: &Marker,
        image_to: impl Into<ImageRef<'a>>,
        compat: TrackCompat,
    ) -> TrackerResult {
        let (code, predicted) = marker_prediction(marker, self.camera, self.frame_to, self.forward);
        // `x_to`, `y_to` are not set when the prediction returns 0 (never in the loop, where
        // the start frame always holds a point): the template position stands in.
        let (prediction, predicted, kind) = match self.prediction_override {
            Some(p) => (1, p, Prediction::Override),
            None => (
                code,
                predicted.unwrap_or(self.from),
                Prediction::Returned(code),
            ),
        };
        let (x, y, pen) = (predicted.x, predicted.y, self.max_penalty);
        let deep = |t: &MarkerTemplate<f32>, im: &ImageU16, d: u8| {
            xma_vision::tracking::track_step_u16_with(t, im, d, x, y, kind, pen, compat)
        };
        let step = match (&self.template.image, image_to.into()) {
            (TemplateImage::U8(t), ImageRef::U8(im)) => {
                xma_vision::tracking::track_step_with(t, im, x, y, kind, pen, compat)
            }
            (TemplateImage::Deep(t), ImageRef::Deep(im, d)) => deep(t, im, d),
            // A source that changed kind between the two frames: the 8-bit side taken
            // exactly onto the full-depth path.
            (TemplateImage::Deep(t), ImageRef::U8(im)) => deep(t, &im.map(u16::from), 8),
            (TemplateImage::U8(t), ImageRef::Deep(im, d)) => {
                let t = MarkerTemplate {
                    size: t.size,
                    image: t.image.map(f32::from),
                };
                deep(&t, im, d)
            }
        };
        let point = step.map_or(predicted, |s| Point2::new(f64::from(s.x), f64::from(s.y)));
        TrackerResult {
            prediction,
            predicted,
            step,
            point,
        }
    }
}

/// The input size a detection reads for a marker (`m_input_size`): the size override if
/// positive, else the mean size if positive, else 5.
///
/// Ported from processing/MarkerDetection.cpp, MarkerDetection::MarkerDetection (`:61-87`).
pub(crate) fn input_size(marker: &Marker) -> f64 {
    if marker.size_override > 0 {
        f64::from(marker.size_override)
    } else if marker.mean_size > 0.0 {
        marker.mean_size
    } else {
        5.0
    }
}

/// One detection at a stored point.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionJob {
    pub marker: usize,
    pub camera: usize,
    pub frame: usize,
    pub method: DetectionMethod,
    /// The point detected around (`m_x`, `m_y`).
    pub center: Point2,
    /// `m_searchArea`.
    pub search_area: i32,
    /// The `int` mask size `detectionPoint` receives (`m_input_size` truncated, ledger A16).
    pub mask_size: i32,
    /// The threshold offset as `detectionPoint` receives it.
    pub threshold: f64,
    /// The status written with the result: `TRACKED` after tracking, else `SET`.
    pub status: MarkerStatus,
}

/// What a detection writes: the point and the size.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionResult {
    pub point: Point2,
    /// The detected size; -1 when nothing was found, and for method 4 (no detection; ledger
    /// A22, see [`DetectionJob::run`]).
    pub size: f64,
    /// The full detection, `None` for method 4.
    pub detection: Option<Detection>,
}

impl DetectionJob {
    /// The constructor: the marker's method and point at `frame`, the search area
    /// `(int)(search_area + 0.5)` raised to 50 for methods 1 and 6 and to 10 for the
    /// others, the input size (size override if positive, else the mean size if positive,
    /// else 5) and the threshold offset, held in an unsigned 8-bit value when
    /// `threshold_wraps` (ledger A15).
    ///
    /// Ported from processing/MarkerDetection.cpp, MarkerDetection::MarkerDetection
    /// (`:61-87`).
    #[must_use]
    pub fn new(
        marker: &Marker,
        marker_index: usize,
        camera: usize,
        frame: usize,
        search_area: f64,
        after_tracking: bool,
        threshold_wraps: bool,
    ) -> DetectionJob {
        let method = marker.method;
        let mut sa = (search_area + 0.5) as i32;
        if matches!(method, DetectionMethod::Blob | DetectionMethod::WhiteBlob) {
            if sa < 50 {
                sa = 50;
            }
        } else if sa < 10 {
            sa = 10;
        }
        let input_size = input_size(marker);
        let threshold = if threshold_wraps {
            f64::from(marker.threshold_offset as u8)
        } else {
            f64::from(marker.threshold_offset)
        };
        DetectionJob {
            marker: marker_index,
            camera,
            frame,
            method,
            center: marker.views[camera].points[frame],
            search_area: sa,
            mask_size: input_size as i32,
            threshold,
            status: if after_tracking {
                MarkerStatus::Tracked
            } else {
                MarkerStatus::Set
            },
        }
    }

    /// The detection thread. Method 4 detects nothing and leaves the point; its size is -1
    /// in both modes (ledger A22: the C++ writes its uninitialised `m_size`, which cannot be
    /// reproduced; -1 is the value for "no size measured", which `updateMeanSize` ignores, as
    /// it ignores the C++ values, all below 1 in the reference runs).
    ///
    /// The image is read 8-bit or at full depth as it is (module documentation).
    ///
    /// Ported from processing/MarkerDetection.cpp, MarkerDetection::detectMarker_thread
    /// (`:420-427`).
    #[must_use]
    pub fn run<'a>(
        &self,
        image: impl Into<ImageRef<'a>>,
        options: &DetectOptions,
    ) -> DetectionResult {
        let Some(method) = detection::DetectionMethod::from_code(self.method.code())
            .filter(|m| *m != detection::DetectionMethod::NoDetection)
        else {
            return DetectionResult {
                point: self.center,
                size: -1.0,
                detection: None,
            };
        };
        let d = detect_point(
            image,
            method,
            (self.center.x, self.center.y),
            self.search_area,
            self.mask_size,
            self.threshold,
            options,
        );
        DetectionResult {
            point: Point2::new(d.point.0, d.point.1),
            size: d.size,
            detection: Some(d),
        }
    }
}

/// Maps `f` over `items` on up to `available_parallelism` threads, results in the order of
/// `items` whatever order the threads finish in.
pub(crate) fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(items.len());
    if threads <= 1 {
        return items.iter().map(&f).collect();
    }
    let chunk = items.len().div_ceil(threads);
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|c| s.spawn(move || c.iter().map(f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a tracking thread panicked"))
            .collect()
    })
}
