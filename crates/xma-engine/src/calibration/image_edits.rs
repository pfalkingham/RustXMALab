//! The edits of a calibration image and of a camera in the Calibration workspace (C23 to
//! C28, C32 to C36) and the pending-changes recalibration (C39).
//!
//! Ported from ui/GLCameraView.cpp (`mouseDoubleClickEvent`, the Calibration branch of
//! `mousePressEvent`), ui/WizardCalibrationCubeFrame.cpp (`addCalibrationReference` in
//! manual mode, `checkBoxManualReference_clicked`, `on_pushButtonResetFrame_clicked`,
//! `on_pushButtonResetCamera_clicked`, `on_pushButtonDeleteFrame_clicked`,
//! `on_checkBoxDistortion_clicked`, `on_checkBoxOptimized_clicked`, `checkForPendingChanges`),
//! core/Camera.cpp (`reset`, `deleteFrame`) and core/CalibrationSequence.cpp (`reset`,
//! `deleteFrame`).

use std::sync::Arc;

use xma_model::Matrix3;

use super::all_frames::{
    calibration_jobs, flag_trials, flagged_cameras, object_points, reload_manual_points,
    reset_distortion, run_calibration_camera_all_frames, set_dialog, set_distortion,
};
use super::cube::{check_image, compute_calibration};
use super::held;
use super::undistortion::set_camera_pending;
use crate::document::Document;
use crate::edit::{EditError, EditOutcome, EditResult};
use crate::session::Workspace;

fn is_calibrated_image(doc: &Document, camera: usize, image: usize) -> bool {
    doc.project().cameras[camera].calibration_images[image].calibrated == 1
}

/// C23, a double click on a calibrated image: the nearest detected point within 20 pixels
/// changes between inlier and outlier, and the camera is flagged. On an image that is not
/// calibrated (1) nothing happens.
///
/// Ported from ui/GLCameraView.cpp, mouseDoubleClickEvent, and core/CalibrationImage.cpp,
/// CalibrationImage::toggleInlier.
pub(crate) fn toggle_inlier(
    doc: &mut Document,
    camera: usize,
    image: usize,
    x: f64,
    y: f64,
    undistorted: bool,
) -> EditResult {
    check_image(doc, camera, image)?;
    if !is_calibrated_image(doc, camera, image) {
        return Ok(EditOutcome::none());
    }
    let changed = held::with_image(doc, camera, image, |im, h, _| {
        held::toggle_inlier(im, h, x, y, !undistorted)
    });
    if changed {
        set_camera_pending(doc, camera, 1);
    }
    Ok(EditOutcome::none())
}

/// C24, Ctrl + click on a calibrated image (manual mode off): the projected point nearest to
/// the click within 20 pixels becomes an inlier detected at the click, and the camera is
/// flagged.
///
/// Ported from core/CalibrationImage.cpp, CalibrationImage::setPointManual.
///
/// # Errors
/// [`EditError::Invalid`] for an image that is not calibrated (the click is a reference
/// click there, [`crate::Session::click_reference`]).
pub(crate) fn move_point(
    doc: &mut Document,
    camera: usize,
    image: usize,
    x: f64,
    y: f64,
    undistorted: bool,
) -> EditResult {
    check_image(doc, camera, image)?;
    if !is_calibrated_image(doc, camera, image) {
        return Err(EditError::Invalid(
            "a point is moved only on a calibrated image".to_owned(),
        ));
    }
    let moved = held::with_image(doc, camera, image, |im, h, model| {
        held::set_point_manual(im, h, model, x, y, !undistorted)
    });
    if moved.is_some() {
        set_camera_pending(doc, camera, 1);
    }
    Ok(EditOutcome::none())
}

/// The image made active, and given its point lists as manual mode does
/// (`reloadManualPoints`).
fn prepare_manual(doc: &mut Document, camera: usize, image: usize) -> usize {
    let s = doc.session_mut();
    s.active_camera = camera;
    s.active_frame_calibration = image;
    reload_manual_points(doc);
    object_points(doc).len()
}

/// C27, a click in manual mode: calibration-object point `point` is detected at the click
/// (in the view's coordinates) and becomes an inlier; the camera is flagged; the next point
/// is chosen for the next click ([`crate::Session::manual_point`]).
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, addCalibrationReference (manual branch),
/// and core/CalibrationImage.cpp, setPoint and setInlier.
pub(crate) fn set_manual_point(
    doc: &mut Document,
    camera: usize,
    image: usize,
    point: usize,
    x: f64,
    y: f64,
    undistorted: bool,
) -> EditResult {
    check_image(doc, camera, image)?;
    let n = object_points(doc).len();
    if point >= n {
        return Err(EditError::Invalid(format!(
            "calibration-object point {point} out of range"
        )));
    }
    prepare_manual(doc, camera, image);
    held::with_image(doc, camera, image, |im, h, model| {
        held::set_point(im, h, model, point, x, y, !undistorted);
        held::set_inlier(im, point, true);
    });
    set_camera_pending(doc, camera, 1);
    reload_manual_points(doc);
    if point + 1 < n {
        doc.session_mut().manual_point = point + 1;
    }
    Ok(EditOutcome::none())
}

/// C28, a manual-mode check box: every point's inlier flag written from the check boxes (1
/// or 0, so an undetected point, -1, becomes 0), and the camera flagged.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, checkBoxManualReference_clicked.
pub(crate) fn set_manual_inliers(
    doc: &mut Document,
    camera: usize,
    image: usize,
    inliers: &[bool],
) -> EditResult {
    check_image(doc, camera, image)?;
    let n = object_points(doc).len();
    if inliers.len() != n {
        return Err(EditError::Invalid(format!(
            "{} check boxes for {n} calibration-object points",
            inliers.len()
        )));
    }
    prepare_manual(doc, camera, image);
    held::with_image(doc, camera, image, |im, _, _| {
        for (i, &v) in inliers.iter().enumerate() {
            held::set_inlier(im, i, v);
        }
    });
    if n > 0 {
        set_camera_pending(doc, camera, 1);
    }
    Ok(EditOutcome::none())
}

/// `Camera::reset`: the camera matrix (re-created without values in the C++, ledger A136;
/// the identity here, as a new camera has), not calibrated, nothing pending, every
/// calibration image reset and given its point lists, the lens model off.
///
/// Ported from core/Camera.cpp, Camera::reset, and core/CalibrationSequence.cpp,
/// CalibrationSequence::reset.
pub(crate) fn camera_reset(doc: &mut Document, camera: usize) {
    let n = object_points(doc).len();
    {
        let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
        cam.camera_matrix = Matrix3::identity();
        cam.calibrated = false;
    }
    super::all_frames::refresh_intrinsics(doc, camera);
    set_camera_pending(doc, camera, 0);
    for i in 0..doc.project().cameras[camera].calibration_images.len() {
        held::with_image(doc, camera, i, |im, h, _| {
            held::reset(im, h);
            held::init(im, h, n);
        });
    }
    reset_distortion(doc, camera);
}

/// C32, "Reset Frame" (confirmed): a camera with one calibrated image is reset; with more,
/// this image is reset and the camera calibrated again over the others.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_pushButtonResetFrame_clicked.
pub(crate) fn reset_frame(doc: &mut Document, camera: usize, image: usize) -> EditResult {
    check_image(doc, camera, image)?;
    let count = doc.project().cameras[camera]
        .calibration_images
        .iter()
        .filter(|im| im.calibrated == 1)
        .count();
    let mut outcome = EditOutcome::none();
    if count == 1 {
        camera_reset(doc, camera);
        set_dialog(doc);
    } else if count > 1 {
        let n = object_points(doc).len();
        held::with_image(doc, camera, image, |im, h, _| {
            held::reset(im, h);
            held::init(im, h, n);
        });
        set_camera_pending(doc, camera, 1);
        outcome
            .messages
            .extend(run_calibration_camera_all_frames(doc));
    }
    Ok(outcome)
}

/// C33, "Reset Camera" (confirmed). In the default mode every trial is flagged for
/// recomputation (ledger A133).
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_pushButtonResetCamera_clicked.
pub(crate) fn reset_camera(doc: &mut Document, camera: usize) -> EditResult {
    if camera >= doc.project().cameras.len() {
        return Err(EditError::Invalid(format!("camera {camera} out of range")));
    }
    camera_reset(doc, camera);
    set_dialog(doc);
    if !doc.compat().recalibration_trials_not_flagged {
        flag_trials(doc);
    }
    Ok(EditOutcome::none())
}

/// C34, "Delete Frame" (confirmed), for every camera: a still image is removed, a frame of a
/// calibration video is reset; a camera left with no calibrated image is reset, the others
/// are calibrated again over their images. In the default mode the trials' reference
/// calibration images follow the deletion (ledger A131).
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_pushButtonDeleteFrame_clicked;
/// core/CalibrationSequence.cpp, deleteFrame; core/Project.cpp, recountFrames.
///
/// # Errors
/// [`EditError::Invalid`] when the project has one calibration frame (XMALab does nothing
/// then) or `image` is out of range.
pub(crate) fn delete_frame(doc: &mut Document, image: usize) -> EditResult {
    if doc.session().n_calibration_images <= 1 {
        return Err(EditError::Invalid(
            "a project with one calibration frame keeps it".to_owned(),
        ));
    }
    let n_cams = doc.project().cameras.len();
    for c in 0..n_cams {
        check_image(doc, c, image)?;
    }
    let n = object_points(doc).len();
    for c in 0..n_cams {
        if doc.project().cameras[c].calibration_sequence.is_some() {
            held::with_image(doc, c, image, |im, h, _| {
                held::reset(im, h);
                held::init(im, h, n);
            });
        } else {
            held::ensure_held(doc, c);
            Arc::make_mut(&mut doc.project.cameras[c])
                .calibration_images
                .remove(image);
            let d = Arc::make_mut(&mut doc.cameras[c]);
            if image < d.images.len() {
                d.images.remove(image);
            }
        }
    }
    // Project::recountFrames: camera 0's count.
    let count = doc
        .project()
        .cameras
        .first()
        .map_or(0, |c| c.calibration_images.len());
    doc.session_mut().n_calibration_images = count;
    if !doc.compat().delete_frame_keeps_trial_references {
        for t in 0..doc.project().trials.len() {
            // XMALab's -1 ("no reference", ledger F68) is not an index: it stays -1.
            let Some(r) = doc.project().trials[t].reference_image() else {
                continue;
            };
            let new = if r > image {
                r - 1
            } else {
                r.min(count.saturating_sub(1))
            };
            if new != r {
                Arc::make_mut(&mut doc.project.trials[t]).reference_calibration_image = new;
            }
        }
    }
    // The active frame (the one deleted) moves back by one, to at least 0.
    doc.session_mut().active_frame_calibration = image.saturating_sub(1);
    for c in 0..n_cams {
        let calibrated = doc.project().cameras[c]
            .calibration_images
            .iter()
            .filter(|im| im.calibrated == 1)
            .count();
        if calibrated == 0 {
            camera_reset(doc, c);
        } else {
            set_camera_pending(doc, c, 1);
        }
    }
    set_dialog(doc);
    let mut outcome = EditOutcome::none();
    outcome
        .messages
        .extend(run_calibration_camera_all_frames(doc));
    Ok(outcome)
}

/// C35, "use camera distortion model": on, the lens model with the camera's coefficients
/// (zeros for a camera that never had any); off, no lens model. The camera is flagged.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_checkBoxDistortion_clicked.
pub(crate) fn set_lens_model(doc: &mut Document, camera: usize, on: bool) -> EditResult {
    if camera >= doc.project().cameras.len() {
        return Err(EditError::Invalid(format!("camera {camera} out of range")));
    }
    if on {
        let d = doc.project().cameras[camera].distortion;
        set_distortion(doc, camera, d);
    } else {
        reset_distortion(doc, camera);
    }
    set_camera_pending(doc, camera, 1);
    Ok(EditOutcome::none())
}

/// C36, unchecking "is optimized" (confirmed): every camera flagged, then, in the XMALab 3
/// mode, the main button's action on the active image (`camera`, `image`), which
/// recalibrates only when that image is calibrated (1); in the default mode every camera is
/// calibrated again over all its images (ledger A134).
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_checkBoxOptimized_clicked.
pub(crate) fn remove_optimization(doc: &mut Document, camera: usize, image: usize) -> EditResult {
    check_image(doc, camera, image)?;
    for c in 0..doc.project().cameras.len() {
        set_camera_pending(doc, c, 1);
    }
    if doc.compat().remove_optimization_on_active_image {
        return compute_calibration(doc, camera, image);
    }
    let mut outcome = EditOutcome::none();
    outcome
        .messages
        .extend(run_calibration_camera_all_frames(doc));
    Ok(outcome)
}

/// The "yes" of the pending-changes question (C11, C39), in the workspace shown: the grids
/// with pending changes in the Undistortion workspace, the cameras with pending changes in
/// the Calibration workspace (one calibration over all images each; in the default mode the
/// trials are then flagged, ledger A133). Nothing in Marker tracking.
///
/// Ported from ui/WizardUndistortionFrame.cpp and ui/WizardCalibrationCubeFrame.cpp,
/// checkForPendingChanges.
pub(crate) fn recompute_pending(doc: &mut Document) -> EditResult {
    let mut outcome = EditOutcome::none();
    match doc.session().workspace {
        Workspace::Undistortion => super::undistortion::recompute_pending_grids(doc)?,
        Workspace::Calibration => {
            let flagged = flagged_cameras(doc);
            if flagged.is_empty() {
                return Ok(outcome);
            }
            outcome.messages.extend(calibration_jobs(doc, &flagged));
            if !doc.compat().recalibration_trials_not_flagged {
                flag_trials(doc);
            }
        }
        Workspace::Tracking => {}
    }
    Ok(outcome)
}
