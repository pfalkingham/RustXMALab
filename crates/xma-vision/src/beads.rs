//! Bead detection on undistortion-grid and calibration images.
//!
//! Ported from `processing/BlobDetection.cpp` (`BlobDetection::process` and
//! `process_finished`): XMALab builds `cv::SimpleBlobDetector::Params` from its 17
//! `BlobDetector*` settings, runs the detector once on the whole image, and keeps the
//! keypoint centres, converted from `float` to `double`, in detection order. The grid image
//! (`image = -1`) replaces `UndistortionObject::points_detected`; a calibration image
//! replaces that image's detected points (`CalibrationImage::setDetectedPoints`).
//!
//! The settings are [`BlobSettings`] (the same struct marker detection uses; field names
//! follow the settings `BlobDetectorThresholdStep` ... `BlobDetectorMaxConvexity`, and the
//! defaults are those of `core/Settings.cpp`). None of them is in XMALab's Settings dialog.
//!
//! The images are 8-bit, as XMALab holds them (parity ledger V6: the grid image is held
//! and stored as 8-bit grey; calibration images likewise). The detector is
//! [`detect_blobs`], the port of OpenCV 4.13's `SimpleBlobDetector::detect`.

use crate::Point2;
use crate::image::ImageU8;
use crate::imgproc::blob::{BlobParamError, BlobParams, BlobSettings, detect_blobs};

/// Which image the beads are detected in: decides the blob colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeadImage {
    /// The undistortion grid (`BlobDetection(camera, -1)`): `blobColor` is
    /// `BlobDetectorBlobColor` (default 255, light beads).
    Grid,
    /// A calibration image. `white_blobs` is `CalibrationObject::hasWhiteBlobs` (the frame
    /// specification file says `white`): `blobColor` is `BlobDetectorBlobColor` then, and
    /// `255 - BlobDetectorBlobColor` otherwise (default 0, dark beads).
    Calibration {
        /// The calibration object has white beads.
        white_blobs: bool,
    },
}

/// The `SimpleBlobDetector::Params` `BlobDetection::process` builds from the settings for
/// this image.
///
/// Ported from processing/BlobDetection.cpp, `BlobDetection::process` (lines 72-104).
#[must_use]
pub fn bead_blob_params(settings: &BlobSettings, image: BeadImage) -> BlobParams {
    let invert = match image {
        BeadImage::Grid => false,
        BeadImage::Calibration { white_blobs } => !white_blobs,
    };
    settings.to_params(invert)
}

/// Detects the beads of a grid or calibration image: the keypoint centres of
/// `cv::SimpleBlobDetector`, as `double`, in detection order (the order XMALab stores them
/// in, and the order the grid matching and the calibration read them in).
///
/// Ported from processing/BlobDetection.cpp, `BlobDetection::process` and
/// `BlobDetection::process_finished`.
///
/// # Errors
/// The check OpenCV's `SimpleBlobDetector` fails for these settings (it throws there, and
/// XMALab does not catch it in the job).
pub fn detect_beads(
    image: &ImageU8,
    settings: &BlobSettings,
    kind: BeadImage,
) -> Result<Vec<Point2>, BlobParamError> {
    let params = bead_blob_params(settings, kind);
    let keypoints = detect_blobs(image, &params)?;
    Ok(keypoints
        .iter()
        .map(|k| Point2::new(f64::from(k.pt.0), f64::from(k.pt.1)))
        .collect())
}
