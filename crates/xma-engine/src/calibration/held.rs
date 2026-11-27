//! The point lists XMALab holds per calibration image besides the ones a project stores,
//! and the `CalibrationImage` operations that keep them.
//!
//! A project stores, per calibration image, the detected points (distorted), the inlier
//! flags, every bead detected and the pose ([`xma_model::CalibrationImage`]). The C++ object
//! also holds the detected points undistorted, the projected points (distorted and
//! undistorted) and both error lists, and each action sets them in its own way: the cube
//! calibration stores its undistorted points and computes the distorted ones from them,
//! `MultiCameraCalibration::reproject` projects with the camera matrix, an undistortion
//! recomputes the undistorted ones from the distorted ones, and a few operations change
//! one list and leave the others. [`HeldPoints`] keeps those lists as the C++ holds them,
//! per image, in the document's [`crate::DerivedCamera`] (not saved, restored by undo with
//! the project).
//!
//! An image without held lists (a project just opened, or a camera whose lists were
//! dropped with [`Document::invalidate_held_points`]) is shown as XMALab shows it after
//! loading ([`crate::view::calibration::calibration_image`]); the first calibration action
//! on a camera takes those values as its starting point ([`ensure_held`]).
//!
//! Ported from core/CalibrationImage.cpp (`reset`, `init`, `setPointsUndistorted`,
//! `undistortPoints`, `setPointsProjectedUndistorted`, `computeError`, `setMatrices`,
//! `toggleInlier`, `setInlier`, `setPoint`, `setPointManual`).

use std::sync::Arc;

use xma_model::{CalibrationImage, Point2, Vector3};
use xma_vision::camera::CameraModel;

use crate::document::{DerivedCamera, Document};

/// The lists of one calibration image that a project does not store, as the C++ holds them
/// (`detectedPointsUndistorted`, `projectedPoints`, `projectedPointsUndistorted`, `error`,
/// `errorUndistorted`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeldPoints {
    pub detected_undistorted: Vec<Point2>,
    pub projected: Vec<Point2>,
    pub projected_undistorted: Vec<Point2>,
    pub error: Vec<f64>,
    pub error_undistorted: Vec<f64>,
}

/// The point (-1, -1), XMALab's unset point.
pub(crate) const UNSET: Point2 = Point2::new(-1.0, -1.0);

/// `Camera::undistortPoint(p, true)`: the grid, then the lens model, refined. Where the
/// fixed lens undistortion has no result (ledger A19) the point is (-1, -1), as the views
/// show it.
pub(crate) fn undistort(model: &CameraModel, p: Point2) -> Point2 {
    model
        .try_undistort_point(p, true, true, true)
        .filter(|q| q.x.is_finite() && q.y.is_finite())
        .unwrap_or(UNSET)
}

/// `Camera::undistortPoint(p, false)`: back into the raw image.
pub(crate) fn distort(model: &CameraModel, p: Point2) -> Point2 {
    model.undistort_point(p, false, true, true)
}

fn dist(a: Point2, b: Point2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::reset: the pose cleared (the
/// C++ re-creates it without values; here zeros, as a new image has), not calibrated,
/// every list cleared.
pub(crate) fn reset(im: &mut CalibrationImage, h: &mut HeldPoints) {
    im.rotation = Vector3::zeros();
    im.translation = Vector3::zeros();
    im.calibrated = 0;
    im.detected_all.clear();
    im.detected.clear();
    im.inlier.clear();
    *h = HeldPoints::default();
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::init: every point list sized to
/// `n` unset points, `n` inlier flags 0; the bead list emptied; the error lists left as
/// they are.
pub(crate) fn init(im: &mut CalibrationImage, h: &mut HeldPoints, n: usize) {
    im.detected_all.clear();
    im.detected = vec![UNSET; n];
    im.inlier = vec![0; n];
    h.projected = vec![UNSET; n];
    h.detected_undistorted = vec![UNSET; n];
    h.projected_undistorted = vec![UNSET; n];
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::computeError: nothing when the
/// detected and projected lists differ in length.
pub(crate) fn compute_error(im: &CalibrationImage, h: &mut HeldPoints) {
    if im.detected.len() != h.projected.len() {
        return;
    }
    h.error = im
        .detected
        .iter()
        .zip(&h.projected)
        .map(|(a, b)| dist(*a, *b))
        .collect();
    h.error_undistorted = h
        .detected_undistorted
        .iter()
        .zip(&h.projected_undistorted)
        .map(|(a, b)| dist(*a, *b))
        .collect();
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::setPointsUndistorted: the
/// undistorted lists as given, the distorted ones computed from them, each inlier flag 1
/// where `inlier`, -1 where the detected point is exactly (-1, -1), else 0.
pub(crate) fn set_points_undistorted(
    im: &mut CalibrationImage,
    h: &mut HeldPoints,
    model: &CameraModel,
    detected: &[Point2],
    projected: &[Point2],
    inlier: &[bool],
) {
    h.detected_undistorted = detected.to_vec();
    h.projected_undistorted = projected.to_vec();
    im.detected = detected.iter().map(|&p| distort(model, p)).collect();
    h.projected = projected.iter().map(|&p| distort(model, p)).collect();
    im.inlier = inlier
        .iter()
        .zip(detected)
        .map(|(&i, p)| {
            if i {
                1
            } else if p.x == -1.0 && p.y == -1.0 {
                -1
            } else {
                0
            }
        })
        .collect();
    compute_error(im, h);
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::undistortPoints: both
/// undistorted lists recomputed from the distorted ones.
pub(crate) fn undistort_points(im: &CalibrationImage, h: &mut HeldPoints, model: &CameraModel) {
    h.detected_undistorted = im.detected.iter().map(|&p| undistort(model, p)).collect();
    h.projected_undistorted = h.projected.iter().map(|&p| undistort(model, p)).collect();
    compute_error(im, h);
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::setPointsProjectedUndistorted.
pub(crate) fn set_points_projected_undistorted(
    im: &CalibrationImage,
    h: &mut HeldPoints,
    model: &CameraModel,
    projected: Vec<Point2>,
) {
    h.projected = projected.iter().map(|&p| distort(model, p)).collect();
    h.projected_undistorted = projected;
    compute_error(im, h);
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::setMatrices.
pub(crate) fn set_matrices(im: &mut CalibrationImage, rotation: Vector3, translation: Vector3) {
    im.rotation = rotation;
    im.translation = translation;
    im.calibrated = 1;
}

/// C23. Returns true when a flag changed (the caller then flags the camera).
///
/// Ported from core/CalibrationImage.cpp, CalibrationImage::toggleInlier: the detected
/// point (both coordinates not -1) nearest to the click within 20 pixels (strict), in the
/// view's coordinates; inlier 0 becomes 1, 1 or more becomes 0, -1 stays.
pub(crate) fn toggle_inlier(
    im: &mut CalibrationImage,
    h: &HeldPoints,
    x: f64,
    y: f64,
    distorted_view: bool,
) -> bool {
    let pt = Point2::new(x, y);
    let mut idx = None;
    let mut mindist = 20.0;
    for (i, p) in im.detected.iter().enumerate() {
        if p.x != -1.0 && p.y != -1.0 {
            let q = if distorted_view {
                *p
            } else {
                h.detected_undistorted.get(i).copied().unwrap_or(UNSET)
            };
            let d = dist(pt, q);
            if d < mindist {
                mindist = d;
                idx = Some(i);
            }
        }
    }
    let Some(i) = idx else { return false };
    match im.inlier.get(i).copied() {
        Some(0) => {
            im.inlier[i] = 1;
            true
        }
        Some(v) if v >= 1 => {
            im.inlier[i] = 0;
            true
        }
        _ => false,
    }
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::setInlier.
pub(crate) fn set_inlier(im: &mut CalibrationImage, idx: usize, value: bool) {
    im.inlier[idx] = i32::from(value);
}

/// Ported from core/CalibrationImage.cpp, CalibrationImage::setPoint: the detected point
/// `idx` set in the view's coordinates, the other list computed.
pub(crate) fn set_point(
    im: &mut CalibrationImage,
    h: &mut HeldPoints,
    model: &CameraModel,
    idx: usize,
    x: f64,
    y: f64,
    distorted_view: bool,
) {
    if distorted_view {
        im.detected[idx] = Point2::new(x, y);
        h.detected_undistorted[idx] = undistort(model, im.detected[idx]);
    } else {
        h.detected_undistorted[idx] = Point2::new(x, y);
        im.detected[idx] = distort(model, h.detected_undistorted[idx]);
    }
}

/// C24. Returns the point moved, if any.
///
/// Ported from core/CalibrationImage.cpp, CalibrationImage::setPointManual: the projected
/// point nearest to the click within 20 pixels (strict) in the view's coordinates becomes
/// an inlier detected at the click; its two errors are updated.
pub(crate) fn set_point_manual(
    im: &mut CalibrationImage,
    h: &mut HeldPoints,
    model: &CameraModel,
    x: f64,
    y: f64,
    distorted_view: bool,
) -> Option<usize> {
    let pt = Point2::new(x, y);
    let mut idx = None;
    let mut mindist = 20.0;
    let list = if distorted_view {
        &h.projected
    } else {
        &h.projected_undistorted
    };
    for (i, p) in list.iter().enumerate() {
        let d = dist(pt, *p);
        if d < mindist {
            mindist = d;
            idx = Some(i);
        }
    }
    let i = idx?;
    if i >= im.detected.len()
        || i >= im.inlier.len()
        || i >= h.detected_undistorted.len()
        || i >= h.projected.len()
        || i >= h.projected_undistorted.len()
    {
        // The C++ writes out of range here (the lists never differ in its own runs).
        return None;
    }
    im.inlier[i] = 1;
    if distorted_view {
        im.detected[i] = pt;
        h.detected_undistorted[i] = undistort(model, pt);
    } else {
        h.detected_undistorted[i] = pt;
        im.detected[i] = distort(model, pt);
    }
    if i < h.error.len() {
        h.error[i] = dist(im.detected[i], h.projected[i]);
    }
    if i < h.error_undistorted.len() {
        h.error_undistorted[i] = dist(h.detected_undistorted[i], h.projected_undistorted[i]);
    }
    Some(i)
}

/// Gives every calibration image of `camera` its held lists, taken from what the views show
/// for an image without them (the state after loading), so that an action can change them
/// as the C++ does.
pub(crate) fn ensure_held(doc: &mut Document, camera: usize) {
    let n = doc.project.cameras[camera].calibration_images.len();
    let missing: Vec<usize> = {
        let d = &doc.cameras[camera];
        (0..n)
            .filter(|&i| d.images.get(i).is_none_or(Option::is_none))
            .collect()
    };
    if missing.is_empty() && doc.cameras[camera].images.len() == n {
        return;
    }
    let computed: Vec<(usize, HeldPoints)> = missing
        .iter()
        .map(|&i| {
            let v = crate::view::calibration::calibration_image_as_loaded(doc, camera, i)
                .unwrap_or_default();
            (
                i,
                HeldPoints {
                    detected_undistorted: v.detected_undistorted,
                    projected: v.projected,
                    projected_undistorted: v.projected_undistorted,
                    error: v.error,
                    error_undistorted: v.error_undistorted,
                },
            )
        })
        .collect();
    let d = Arc::make_mut(&mut doc.cameras[camera]);
    d.images.resize(n, None);
    for (i, h) in computed {
        d.images[i] = Some(Arc::new(h));
    }
}

/// Calls `f` with calibration image `image` of `camera`, its held lists and the camera's
/// model, for changing. [`ensure_held`] must have run for the camera.
pub(crate) fn with_image<R>(
    doc: &mut Document,
    camera: usize,
    image: usize,
    f: impl FnOnce(&mut CalibrationImage, &mut HeldPoints, &CameraModel) -> R,
) -> R {
    ensure_held(doc, camera);
    let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
    let DerivedCamera { model, images, .. } = Arc::make_mut(&mut doc.cameras[camera]);
    let slot = images[image].get_or_insert_with(Default::default);
    let h = Arc::make_mut(slot);
    f(&mut cam.calibration_images[image], h, model)
}

impl Document {
    /// Drops the held point lists of `camera`'s calibration images ([`HeldPoints`]): the
    /// views then show them as after loading (projections with the stored poses and camera
    /// matrix, undistorted points from the distorted ones). For an action that sets poses or
    /// the camera matrix without keeping the lists itself.
    pub fn invalidate_held_points(&mut self, camera: usize) {
        if let Some(d) = self.cameras.get_mut(camera)
            && d.images.iter().any(Option::is_some)
        {
            Arc::make_mut(d).images.clear();
        }
    }
}
