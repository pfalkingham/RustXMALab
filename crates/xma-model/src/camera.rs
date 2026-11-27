//! Cameras: intrinsics, calibration images and the undistortion grid.
//!
//! Ported from `core/Camera.h`, `core/CalibrationImage.h`, `core/CalibrationSequence.h` and
//! `core/UndistortionObject.h`. Only stored data lives here. Fitted undistortion models,
//! lookup maps, projected points and errors are derived and belong to the engine.

use crate::{Matrix3, Point2, Vector3};

/// Display scale and bias for one view of a camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageAdjustment {
    pub scale: f64,
    pub bias: f64,
}

impl Default for ImageAdjustment {
    fn default() -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
        }
    }
}

/// The three views that keep their own scale and bias (C++ `Camera::AdjustedView`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdjustedView {
    Calibration = 0,
    Digitization = 1,
    Detail = 2,
}

/// One calibration image of one camera.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationImage {
    /// File name of the image, or empty when the image is a frame of a calibration sequence.
    pub filename: String,
    /// The C++ stores an int: 0 not calibrated, 1 calibrated. Other values are kept as read.
    pub calibrated: i32,
    /// Rodrigues rotation, world to camera.
    pub rotation: Vector3,
    /// Translation, world to camera.
    pub translation: Vector3,
    /// Every point detected in the image (C++ `detectedPoints_ALL`).
    pub detected_all: Vec<Point2>,
    /// Detected position of each calibration-object point, in object order; unset entries
    /// hold the C++ placeholder as read from the file (C++ `detectedPoints`).
    pub detected: Vec<Point2>,
    /// Inlier flag for each calibration-object point, as the C++ stores it.
    pub inlier: Vec<i32>,
}

impl CalibrationImage {
    #[must_use]
    pub fn new(filename: String) -> Self {
        Self {
            filename,
            calibrated: 0,
            rotation: Vector3::zeros(),
            translation: Vector3::zeros(),
            detected_all: Vec::new(),
            detected: Vec::new(),
            inlier: Vec::new(),
        }
    }

    #[must_use]
    pub fn is_calibrated(&self) -> bool {
        self.calibrated > 0
    }
}

/// A calibration video, when calibration images are frames of one file
/// (C++ `CalibrationSequence`).
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationSequence {
    pub filename: String,
    pub n_frames: usize,
    pub width: u32,
    pub height: u32,
}

/// The stored data of an undistortion grid image.
#[derive(Clone, Debug, PartialEq)]
pub struct UndistortionGrid {
    pub image_filename: String,
    /// True when the grid has been matched and the mapping can be built.
    pub computed: bool,
    /// Bead centres detected in the grid image.
    pub points_detected: Vec<Point2>,
    /// Matched bead positions in the distorted image.
    pub grid_distorted: Vec<Point2>,
    /// The ideal position of each matched bead.
    pub grid_references: Vec<Point2>,
    pub grid_inlier: Vec<bool>,
    /// Centre chosen by the user, if any.
    pub center: Option<Point2>,
}

/// A camera.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// A visible-light camera, as opposed to an X-ray camera.
    pub is_light_camera: bool,
    pub calibrated: bool,
    /// True when intrinsics were refined by multi-camera optimisation.
    pub optimized: bool,
    /// Images are mirrored horizontally on load.
    pub flipped: bool,
    pub camera_matrix: Matrix3,
    /// Lens distortion in OpenCV order: k1, k2, p1, p2, k3, k4, k5, k6.
    pub distortion: [f64; 8],
    /// True when `distortion` is in use (C++ `model_distortion`).
    pub has_model_distortion: bool,
    pub undistortion: Option<UndistortionGrid>,
    pub calibration_sequence: Option<CalibrationSequence>,
    pub calibration_images: Vec<CalibrationImage>,
    /// Scale and bias per [`AdjustedView`], indexed by its value.
    pub image_adjustments: [ImageAdjustment; 3],
    /// Identifier on the XMA Portal, or -1.
    pub portal_id: i32,
}

impl Camera {
    /// A new, uncalibrated camera, as `Camera::Camera` creates it.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            width: 0,
            height: 0,
            is_light_camera: false,
            calibrated: false,
            optimized: false,
            flipped: false,
            camera_matrix: Matrix3::identity(),
            distortion: [0.0; 8],
            has_model_distortion: false,
            undistortion: None,
            calibration_sequence: None,
            calibration_images: Vec::new(),
            image_adjustments: [ImageAdjustment::default(); 3],
            portal_id: -1,
        }
    }

    #[must_use]
    pub fn adjustment(&self, view: AdjustedView) -> ImageAdjustment {
        self.image_adjustments[view as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_camera_matches_cpp_defaults() {
        let camera = Camera::new("Camera 1".to_owned());
        assert_eq!(camera.camera_matrix, Matrix3::identity());
        assert_eq!(camera.distortion, [0.0; 8]);
        assert!(!camera.calibrated);
        assert_eq!(camera.portal_id, -1);
        assert_eq!(camera.adjustment(AdjustedView::Detail).scale, 1.0);
        assert_eq!(camera.adjustment(AdjustedView::Detail).bias, 0.0);
    }

    #[test]
    fn calibration_image_flag() {
        let mut image = CalibrationImage::new("cube_01.tif".to_owned());
        assert!(!image.is_calibrated());
        image.calibrated = 1;
        assert!(image.is_calibrated());
    }
}
