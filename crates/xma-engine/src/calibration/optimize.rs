//! C37 and C42, "optimize camera setup": the multi-camera optimisation as an edit.
//!
//! Ported from `ui/WizardCalibrationCubeFrame.cpp:964-997`
//! (`on_pushButtonOptimize_clicked`, `optimizationDone`) and
//! `processing/MultiCameraCalibration.cpp` (constructor `:264-339` for the input,
//! `optimizeCameraSetup_threadFinished` `:706-789` and `reproject` `:824-880` for what is
//! written back). The algorithm is [`xma_vision::multicam`]; this module builds its input
//! from the project and writes its result into the project.
//!
//! # What is written back
//!
//! Per camera: fx, fy, cx, cy into the camera matrix (the other entries kept), the camera
//! calibrated and optimised, `camera_recalibration = 0`; method 2: the coefficients and the
//! lens model on; methods 0 and 1: the lens model off (coefficients zero) when it was on.
//! Per listed frame and every camera: the pose, and `calibrated = 2` for an image that was
//! not 1. The projections and reprojection errors of the calibrated images
//! (`MultiCameraCalibration::reproject`) are not stored: [`crate::view::calibration`]
//! computes them from the pose and the camera matrix. In the XMALab 3 mode (ledger A124)
//! the camera keeps the previous camera matrix for the undistortion of its calibration
//! images' detected points ([`crate::DerivedCamera::lens_points_camera_matrix`]).
//!
//! # Refusals
//!
//! Every refusal is [`EditError::Rejected`] with a sentence for the user and changes
//! nothing: a camera not calibrated or a method other than 0 to 2 (ledger A146, CD19),
//! anything [`xma_vision::multicam::setup`] refuses (ledger A120), and in the default mode a
//! solver refusal or non-finite result (ledger A127) and checkerboard cameras that number
//! the board differently (ledger A189, [`super::numbering`]).
//!
//! # Warning
//!
//! Ledger A129: a camera matrix with a skew entry gives a warning first ([`skew_warning`],
//! both modes), the first of the outcome's messages.

use std::sync::Arc;

use xma_model::CppCompat;
use xma_vision::camera::RigidTransform;
use xma_vision::multicam::{
    Method, MulticamCamera, MulticamCompat, MulticamError, MulticamImage, MulticamInput,
    MulticamResult, optimize,
};

use crate::document::Document;
use crate::edit::{EditError, EditOutcome, EditResult};

/// The [`MulticamCompat`] switches of a compatibility mode (ledger A121 to A127).
#[must_use]
pub fn multicam_compat(compat: CppCompat) -> MulticamCompat {
    MulticamCompat {
        chain_skips_frame_zero: compat.multicam_chain_skips_frame_zero,
        chain_uses_reference_image_pose: compat.multicam_chain_uses_reference_image_pose,
        jacobian_nan_at_zero_difference: compat.multicam_jacobian_nan_at_zero_difference,
        lens_maps_with_previous_camera_matrix: compat
            .multicam_lens_maps_with_previous_camera_matrix,
        keeps_rational_coefficients: compat.multicam_keeps_rational_coefficients,
        trials_not_flagged: compat.multicam_trials_not_flagged,
        failure_written_back: compat.multicam_failure_written_back,
        // Ledger A158 (an under-determined camera or frame): follows the same switch as
        // A127, a result that should not have been written.
        underdetermined_written_back: compat.multicam_failure_written_back,
    }
}

/// The sentence shown for a refusal of the optimisation.
fn refusal(e: &MulticamError) -> String {
    format!("The camera setup cannot be optimized: {e}.")
}

/// The optimisation's input as `MultiCameraCalibration`'s constructor reads the project:
/// each camera's matrix and coefficients, and for each of the project's calibration frames
/// each image's state, pose, inliers and detected points passed through the camera's
/// undistortion grid only (`UndistortionObject::transformPoint(pt, true)` when the grid is
/// computed).
///
/// Ported from `processing/MultiCameraCalibration.cpp:296-339`.
///
/// # Errors
/// A sentence when a camera has fewer calibration images than the project's number of
/// calibration frames (the C++ reads past its list).
pub fn multicam_images(
    doc: &Document,
) -> Result<(Vec<MulticamCamera>, Vec<Vec<MulticamImage>>), String> {
    let n_frames = doc.session().n_calibration_images;
    let mut cameras = Vec::new();
    let mut images = Vec::new();
    for (c, cam) in doc.project().cameras.iter().enumerate() {
        cameras.push(MulticamCamera {
            camera_matrix: cam.camera_matrix,
            distortion: cam.distortion,
            model_distortion: cam.has_model_distortion,
        });
        if cam.calibration_images.len() < n_frames {
            return Err(format!(
                "The camera setup cannot be optimized: {} has {} calibration images and the \
                 project has {n_frames} calibration frames.",
                cam.name,
                cam.calibration_images.len()
            ));
        }
        let model = &doc.camera(c).model;
        let row = cam.calibration_images[..n_frames]
            .iter()
            .map(|im| MulticamImage {
                calibrated: im.calibrated,
                pose: (im.calibrated > 0).then_some(RigidTransform {
                    rotation: im.rotation,
                    translation: im.translation,
                }),
                inliers: im.inlier.clone(),
                // Only inliers are read; the grid transform of the others is not needed,
                // but computing it is harmless and keeps the indices aligned.
                points: if im.calibrated == 1 {
                    im.detected
                        .iter()
                        .zip(&im.inlier)
                        .map(|(p, &inl)| {
                            if inl > 0 {
                                model.undistort_point(*p, true, false, true)
                            } else {
                                *p
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                },
            })
            .collect();
        images.push(row);
    }
    Ok((cameras, images))
}

/// Runs the optimisation on the project as it stands, without changing it: the input as
/// [`multicam_images`] builds it, [`xma_vision::multicam::optimize`] with the document's
/// mode. Used by the edit, and by `xma parity` to compare the hook tables.
///
/// # Errors
/// The sentence of a refusal, as [`CalibrationEdit::Optimize`](super::CalibrationEdit::Optimize)
/// gives it.
pub fn run_optimization(
    doc: &Document,
    method: i32,
    iterations: usize,
    initial: f64,
) -> Result<MulticamResult, String> {
    let project = doc.project();
    // Ledger A146 (CD19): the C++ starts without checking; an uncalibrated camera has no
    // values to start from (undefined behaviour). Both modes.
    if let Some(cam) = project.cameras.iter().find(|c| !c.calibrated) {
        return Err(format!(
            "The camera setup cannot be optimized: {} is not calibrated. Calibrate every \
             camera first.",
            cam.name
        ));
    }
    // Ledger A146: the dialog's three radio buttons give 0, 1 or 2; any other value is
    // refused (both modes), not run as method 0.
    if !(0..=2).contains(&method) {
        return Err(format!(
            "The camera setup cannot be optimized: method {method} is not one of 0, 1 and 2."
        ));
    }
    if iterations == 0 {
        return Err("The number of iterations must be at least 1.".to_owned());
    }
    if !initial.is_finite() || initial < 0.0 {
        return Err("The scale factor for the initial lambda must be 0 or more.".to_owned());
    }
    let object_points: Vec<xma_model::Point3> = match &project.calibration_object {
        // A frame's beads or a checkerboard's corners (`getFrameSpecifications`).
        Some(o) => o.points(),
        None => {
            return Err(
                "The camera setup cannot be optimized: the project has no calibration object."
                    .to_owned(),
            );
        }
    };
    let (cameras, images) = multicam_images(doc)?;
    let input = MulticamInput {
        cameras: &cameras,
        images: &images,
        object_points: &object_points,
        method: Method::from_index(method),
        iterations,
        initial_mu: initial,
    };
    optimize(&input, multicam_compat(doc.compat())).map_err(|e| refusal(&e))
}

/// C37, C42: "optimize camera setup".
///
/// Ported from `ui/WizardCalibrationCubeFrame.cpp:964-997` and
/// `processing/MultiCameraCalibration.cpp:706-789`, `optimizeCameraSetup_threadFinished`.
pub(crate) fn apply_optimize(
    doc: &mut Document,
    method: i32,
    iterations: usize,
    initial: f64,
) -> EditResult {
    // Ledger A189 (default mode): a warning when cameras number a checkerboard differently
    // (nothing changed, never refused).
    let numbering = super::numbering::before_optimization(doc);
    // Ledger A129: both modes keep the arithmetic and warn first, about the cameras as they
    // are optimised.
    let mut messages: Vec<String> = skew_warning(doc).into_iter().collect();
    messages.extend(numbering);
    let result = run_optimization(doc, method, iterations, initial).map_err(EditError::Rejected)?;
    write_back(doc, &result);
    let info = result.info.as_ref().map_or_else(
        || "Levenberg-Marquardt refused; the initial values were written back".to_owned(),
        |i| {
            format!(
                "Levenberg-Marquardt stopped after {} iterations ({:?}); squared error {} \
                 before, {} after",
                i.iterations, i.stop, i.initial_error, i.error
            )
        },
    );
    messages.push(info);
    Ok(EditOutcome {
        messages,
        refused: None,
    })
}

/// Ledger A129: the warning given before "optimize camera setup" when a camera matrix has a
/// skew entry (row 1, column 2) that is not zero, as an external calibration can leave: the
/// optimisation treats those cameras as having no skew, and writes the skew back unchanged.
/// `None` when every camera's skew is zero.
#[must_use]
pub fn skew_warning(doc: &Document) -> Option<String> {
    let names: Vec<&str> = doc
        .project()
        .cameras
        .iter()
        .filter(|c| c.camera_matrix[(0, 1)] != 0.0)
        .map(|c| c.name.as_str())
        .collect();
    if names.is_empty() {
        return None;
    }
    let (list, has, it) = match names.as_slice() {
        [one] => ((*one).to_owned(), "has a camera matrix with skew", "it"),
        [rest @ .., last] => (
            format!("{} and {last}", rest.join(", ")),
            "have camera matrices with skew",
            "them",
        ),
        [] => return None,
    };
    Some(format!(
        "Warning: {list} {has}. The optimization treats {it} as having no skew: it optimizes \
         the other values as if the skew were zero and keeps the skew value unchanged, so the \
         optimized camera does not exactly match the values it was optimized with."
    ))
}

/// Writes a result into the project, the derived cameras and the session.
fn write_back(doc: &mut Document, result: &MulticamResult) {
    let mut lens_points = Vec::new();
    for (c, up) in result.cameras.iter().enumerate() {
        let cam = Arc::make_mut(&mut doc.project.cameras[c]);
        let old_k = cam.camera_matrix;
        match up.distortion {
            // Camera::setDistortionCoefficiants.
            Some(d) => {
                cam.distortion = d;
                cam.has_model_distortion = true;
            }
            // Camera::resetDistortion: only when the model is on.
            None => {
                if cam.has_model_distortion {
                    cam.distortion = [0.0; 8];
                    cam.has_model_distortion = false;
                }
            }
        }
        // Camera::setCameraMatrix, setRecalibrationRequired(false), setOptimized(true).
        cam.camera_matrix = up.camera_matrix;
        cam.calibrated = true;
        cam.optimized = true;
        lens_points.push(
            up.lens_maps_camera_matrix
                .filter(|k| *k == old_k && *k != up.camera_matrix),
        );
    }
    for im in &result.images {
        let cam = Arc::make_mut(&mut doc.project.cameras[im.camera]);
        let image = &mut cam.calibration_images[im.frame];
        image.rotation = im.pose.rotation;
        image.translation = im.pose.translation;
        image.calibrated = im.calibrated;
    }
    doc.sync_camera_intrinsics();
    for (c, k) in lens_points.into_iter().enumerate() {
        doc.camera_mut(c).lens_points_camera_matrix = k;
        // The poses and camera matrix changed: the point lists a calibration of this session
        // left (`HeldPoints`) no longer belong to them; the views compute them as after
        // loading.
        doc.invalidate_held_points(c);
    }
    doc.session_mut().camera_recalibration.fill(0);
    // optimizationDone -> setDialog.
    let project = doc.project.clone();
    doc.session_mut().reload_references(&project);
    // Ledger A126.
    if result.trials_need_recompute {
        for t in 0..doc.project.trials.len() {
            doc.set_trial_requires_recomputation(t, true);
        }
    }
}
