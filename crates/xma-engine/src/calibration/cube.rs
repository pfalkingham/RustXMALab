//! "compute calibration" / "recompute calibration" (C17, C18, C22, C29) and everything that
//! follows a single-image cube calibration (C20 and C21).
//!
//! Ported from ui/WizardCalibrationCubeFrame.cpp (`on_pushButton_clicked`, `runCalibration`,
//! `runCalibrationFinished`, `checkForCalibrationError`, `calibrateOtherFrames`,
//! `setTransformationMatrix`), processing/CubeCalibration.cpp (the constructor and the
//! `_threadFinished` handlers; the computation is `xma_vision::cube`),
//! processing/BlobDetection.cpp (calibration images) and core/CalibrationImage.cpp
//! (`getTransformationMatrix`).

use std::sync::Arc;

use nalgebra::Matrix4;
use xma_model::{CalibrationObject, CalibrationType, Point2};
use xma_vision::beads::BeadImage;
use xma_vision::camera::rodrigues_to_matrix;
use xma_vision::crand::CRand;
use xma_vision::cube::{
    CubeCompat, CubeError, CubeInput, CubeMode, CubeReference, CubeResult, CubeSettings,
    compute_pose, compute_pose_and_cam, set_pose,
};
use xma_vision::linalg;
use xma_vision::rigid::gemm_small;

use super::all_frames::{
    n_frames, object_points, run_calibration_camera_all_frames, set_camera_matrix, set_dialog,
};
use super::undistortion::set_camera_pending;
use super::{held, images};
use crate::document::Document;
use crate::edit::{EditError, EditOutcome, EditResult};

/// The cube calibration's part of a document's mode (ledger A110 to A112).
pub(crate) fn cube_compat(doc: &Document) -> CubeCompat {
    let c = doc.compat();
    CubeCompat {
        reference_from_undistorted_position: c.cube_reference_from_undistorted_position,
        success_after_failed_refit: c.cube_success_after_failed_refit,
        principal_point_one_based: c.principal_point_one_based,
        // Ledger A156 (the matches returned belong to the camera before the last refit):
        // follows the same switch as A111, the other fault of `refineResults`.
        inliers_of_previous_camera: c.cube_success_after_failed_refit,
    }
}

/// XMALab's error sentences of the "compute calibration" button.
pub const NEED_THREE_REFERENCES: &str = "You need to at least select 3 reference points";
/// After the bead detection, fewer than 6 beads.
pub const NOT_ENOUGH_POINTS: &str = "Not enough points found to run calibration";
/// Manual mode with fewer than 7 points checked.
pub const NEED_SEVEN_POINTS: &str = "You need to at least select 7 points for a manual calibration";

/// `BlobDetection(camera, image)`: the beads of a calibration image (distorted), stored as
/// its `detectedPoints_ALL` by the caller.
fn detect_calibration_beads(
    doc: &Document,
    camera: usize,
    image: usize,
    videos: &mut images::CalibrationVideos,
) -> Result<Vec<Point2>, EditError> {
    let frame =
        images::calibration_frame_in(doc, camera, image, videos).map_err(EditError::Rejected)?;
    images::detect(
        doc,
        &frame,
        BeadImage::Calibration {
            white_blobs: images::white_blobs(doc),
        },
    )
    .map_err(EditError::Rejected)
}

fn store_beads(doc: &mut Document, camera: usize, image: usize, beads: Vec<Point2>) {
    let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
    cam.calibration_images[image].detected_all = beads;
}

/// The button's count of reference clicks: raw `x >= 0 && y >= 0`
/// (ui/WizardCalibrationCubeFrame.cpp:607-610).
fn click_set(p: Option<(f64, f64)>) -> bool {
    p.is_some_and(|(x, y)| x >= 0.0 && y >= 0.0)
}

/// What the `CubeCalibration` constructor computes, and the calibration of `mode` on it.
///
/// Ported from processing/CubeCalibration.cpp, CubeCalibration::CubeCalibration (the
/// clicks, unset ones as (-1, -1), and every bead undistorted with
/// `Camera::undistortPoint(p, true)`), with the computation of `mode` (`computePoseAndCam`,
/// `computePose`, `setPose`).
fn run_cube(
    doc: &mut Document,
    camera: usize,
    image: usize,
    mode: CubeMode,
    transformation: Option<&Matrix4<f64>>,
) -> Result<CubeResult, CubeError> {
    let model = doc.camera(camera).model.clone();
    let pts = object_points(doc);
    let cam = &doc.project().cameras[camera];
    let detected: Vec<Point2> = cam.calibration_images[image]
        .detected_all
        .iter()
        .map(|&p| model.undistort_point(p, true, true, true))
        .collect();
    let refs = doc.session().references;
    let references: [CubeReference; 4] = std::array::from_fn(|i| {
        let r = refs[i];
        let raw = r.position.unwrap_or((-1.0, -1.0));
        CubeReference {
            point_id: i32::try_from(r.point_id).unwrap_or(i32::MAX),
            position: model.undistort_point(Point2::new(raw.0, raw.1), true, true, true),
            set: click_set(r.position),
        }
    });
    let s = doc.settings();
    #[allow(clippy::cast_possible_wrap)]
    let settings = CubeSettings {
        identification_threshold: s.identification_threshold_calibration as i32,
        outlier_threshold: s.outlier_threshold_for_calibration as i32,
        fix_principal: s.fix_principal,
        width: cam.width as i32,
        height: cam.height as i32,
    };
    let k = cam.camera_matrix;
    let input = CubeInput {
        object_points: &pts,
        detected: &detected,
        references,
        settings,
        compat: cube_compat(doc),
        search_loops: None,
    };
    // Ledger A5: the session's generator in the XMALab 3 mode, a fresh one otherwise.
    let shared = doc.compat().calibration_rand_continues;
    let mut rng = if shared {
        doc.session().rng.clone()
    } else {
        CRand::new(1)
    };
    let r = match mode {
        CubeMode::PoseAndCam => compute_pose_and_cam(&input, &mut rng, None),
        CubeMode::Pose => compute_pose(&input, &k, &mut rng, None),
        CubeMode::SetPose => set_pose(
            &input,
            &k,
            transformation.expect("set_pose needs a transformation"),
            None,
        ),
    };
    if shared {
        doc.session_mut().rng = rng;
    }
    r
}

/// What `computePoseAndCam_threadFinished` / `computePose_threadFinished` write: on success
/// the camera matrix (first image of a camera), the image's points and pose, the camera
/// flagged for recalibration; otherwise the image marked failed (-1).
///
/// Ported from processing/CubeCalibration.cpp, computePoseAndCam_threadFinished and
/// computePose_threadFinished.
fn write_result(doc: &mut Document, camera: usize, image: usize, r: &CubeResult) {
    if r.succeeded() {
        if r.mode == CubeMode::PoseAndCam {
            set_camera_matrix(doc, camera, r.camera_matrix);
        }
        held::with_image(doc, camera, image, |im, h, model| {
            held::set_points_undistorted(im, h, model, &r.detected, &r.projected, &r.inlier);
            held::set_matrices(im, r.pose.rotation, r.pose.translation);
        });
        set_camera_pending(doc, camera, 1);
    } else {
        Arc::make_mut(&mut doc.project.cameras[camera]).calibration_images[image].calibrated = -1;
    }
}

/// `checkForCalibrationError`: every image marked failed is listed, set back to 0, and
/// reported unless `HideWarningsDuringCalibration`.
fn check_for_calibration_error(doc: &mut Document) -> Option<String> {
    let mut text =
        String::from("There were problems during Calibration with the following frames:\n");
    let mut has_error = false;
    for j in 0..doc.project().cameras.len() {
        for k in 0..doc.project().cameras[j].calibration_images.len() {
            if doc.project().cameras[j].calibration_images[k].calibrated < 0 {
                has_error = true;
                text.push_str(&format!("Camera {} Frame {}\n", j + 1, k + 1));
                Arc::make_mut(&mut doc.project.cameras[j]).calibration_images[k].calibrated = 0;
            }
        }
    }
    if !has_error {
        return None;
    }
    text.push_str(
        "Please try to select the references again. If you keep getting this error please \
         have a look at the 3D view and check if the already computed cameras are correct. If \
         necessary reset the erroneous frames or reset the camera!",
    );
    (!doc.session().wizard.hide_warnings_during_calibration).then_some(text)
}

/// `CalibrationImage::getTransformationMatrix`: `[R | t; 0 0 0 1]` of the image's pose,
/// inverted (camera to world).
fn transformation_matrix(doc: &Document, camera: usize, image: usize) -> Matrix4<f64> {
    let im = &doc.project().cameras[camera].calibration_images[image];
    let r = rodrigues_to_matrix(&im.rotation);
    let mut t = Matrix4::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            t[(i, j)] = r[(i, j)];
        }
        t[(i, 3)] = im.translation[i];
    }
    t[(3, 3)] = 1.0;
    let inv = linalg::invert(&t);
    Matrix4::from_iterator(inv.x.iter().copied())
}

fn to_rows(m: &Matrix4<f64>) -> [[f64; 4]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| m[(i, j)]))
}

/// C20, `calibrateOtherFrames`: for each frame calibrated (1) in camera j and not in a
/// calibrated camera k, the pose of k there from j's pose and the transformation between the
/// two cameras at the first frame both have; returns (k, frame, camera to world of k).
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, calibrateOtherFrames. The products are
/// OpenCV's: `A.inv() * B` is `cv::solve(A, B, DECOMP_LU)` ([`linalg::solve_lu`]) and `A *
/// B` of two 4x4 matrices `cv::gemm` ([`gemm_small`]).
fn calibrate_other_frames(doc: &Document) -> Vec<(usize, usize, Matrix4<f64>)> {
    let n = doc.project().cameras.len();
    let calibrated: Vec<bool> = doc.project().cameras.iter().map(|c| c.calibrated).collect();
    let state = |c: usize, m: usize| doc.project().cameras[c].calibration_images[m].calibrated;
    let frames = |a: usize, b: usize| n_frames(doc, a).min(n_frames(doc, b));
    let mut t: Vec<Vec<Option<Matrix4<f64>>>> = vec![vec![None; n]; n];
    for j in 0..n {
        for k in 0..n {
            if j != k && calibrated[k] && calibrated[j] {
                for m in 0..frames(j, k) {
                    if state(k, m) == 1 && state(j, m) == 1 {
                        let fk = transformation_matrix(doc, k, m);
                        let fj = transformation_matrix(doc, j, m);
                        let s = linalg::solve_lu(&fk, &fj);
                        t[j][k] = Some(Matrix4::from_iterator(s.x.iter().copied()));
                        break;
                    }
                }
            }
        }
    }
    let mut queue: Vec<(usize, usize, Matrix4<f64>)> = Vec::new();
    for j in 0..n {
        for k in 0..n {
            if calibrated[k] && calibrated[j] {
                for m in 0..frames(j, k) {
                    if j != k && state(k, m) != 1 && state(j, m) == 1 {
                        let save = !queue.iter().any(|q| q.0 == k && q.1 == m);
                        if let (true, Some(tkj)) = (save, t[k][j]) {
                            let fj = transformation_matrix(doc, j, m);
                            let p = gemm_small(&to_rows(&fj), &to_rows(&tkj));
                            queue.push((k, m, Matrix4::from_fn(|r, c| p[r][c])));
                        }
                    }
                }
            }
        }
    }
    queue
}

/// `runCalibrationFinished`, repeated as long as poses can be carried to other cameras'
/// images: the failed frames reported, the poses from other cameras set (each such image's
/// beads detected and `CubeCalibration::setPose` run), and, when none is left, `setDialog`
/// and the calibration over all images of every camera flagged.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, runCalibrationFinished,
/// calibrateOtherFrames and setTransformationMatrix.
fn run_calibration_finished(doc: &mut Document) -> Result<Vec<String>, EditError> {
    let mut messages = Vec::new();
    let mut temporary: Vec<(usize, usize)> = Vec::new();
    // The calibration videos stay open for the whole edit (finding 11).
    let mut videos = images::CalibrationVideos::default();
    loop {
        messages.extend(check_for_calibration_error(doc));
        let mut failed = false;
        for &(k, m) in &temporary {
            if doc.project().cameras[k].calibration_images[m].calibrated != 1 {
                if !doc.session().wizard.hide_warnings_during_calibration {
                    messages.push(format!(
                        "Could not set Camera {} Frame {} from other Frames. Please check your calibration.",
                        k + 1,
                        m + 1
                    ));
                }
                failed = true;
            }
        }
        temporary.clear();
        let queue = if failed {
            Vec::new()
        } else {
            calibrate_other_frames(doc)
        };
        if queue.is_empty() {
            set_dialog(doc);
            messages.extend(run_calibration_camera_all_frames(doc));
            return Ok(messages);
        }
        for &(k, m, _) in &queue {
            let beads = detect_calibration_beads(doc, k, m, &mut videos)?;
            store_beads(doc, k, m, beads);
        }
        for (k, m, trans) in &queue {
            let r = run_cube(doc, *k, *m, CubeMode::SetPose, Some(trans))
                .map_err(|e| EditError::Rejected(e.to_string()))?;
            write_result(doc, *k, *m, &r);
            temporary.push((*k, *m));
        }
    }
}

/// C17, C18, C22, C29: the calibration wizard's main button for calibration image `image` of
/// `camera` (made the active ones).
///
/// - Manual mode (C29): at least 7 points checked, else "You need to at least select 7
///   points for a manual calibration"; the image is marked calibrated and every flagged
///   camera is calibrated over all its images.
/// - Image not calibrated (C17, C18): at least 3 reference clicks, else "You need to at
///   least select 3 reference points"; the image's beads are detected (fewer than 6: "Not
///   enough points found to run calibration"); the cube calibration of the image
///   (`computePoseAndCam` for a camera not yet calibrated, `computePose` otherwise); then
///   `runCalibrationFinished`.
/// - Image calibrated (C22): every flagged camera calibrated over all its images.
/// - A checkerboard's image not calibrated (C40, C41): `checkerboard::compute_uncalibrated`.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_pushButton_clicked and runCalibration.
pub(crate) fn compute_calibration(doc: &mut Document, camera: usize, image: usize) -> EditResult {
    check_image(doc, camera, image)?;
    if doc.project().calibration_type != CalibrationType::Internal {
        return Err(EditError::Invalid(
            "the project has no internal calibration".to_owned(),
        ));
    }
    if doc.project().calibration_object.is_none() {
        return Err(EditError::Invalid(
            "the project has no calibration object".to_owned(),
        ));
    }
    if !doc.session().undistorted {
        return Err(EditError::Invalid(
            "You first have to perform an undistortion!".to_owned(),
        ));
    }
    {
        let s = doc.session_mut();
        s.active_camera = camera;
        s.active_frame_calibration = image;
    }
    let mut outcome = EditOutcome::none();
    if doc.session().manual_mode {
        let count = doc.project().cameras[camera].calibration_images[image]
            .inlier
            .iter()
            .filter(|&&v| v > 0)
            .count();
        if count < 7 {
            return Err(EditError::Rejected(NEED_SEVEN_POINTS.to_owned()));
        }
        Arc::make_mut(&mut doc.project.cameras[camera]).calibration_images[image].calibrated = 1;
        outcome
            .messages
            .extend(run_calibration_camera_all_frames(doc));
        return Ok(outcome);
    }
    if doc.project().cameras[camera].calibration_images[image].calibrated == 1 {
        outcome
            .messages
            .extend(run_calibration_camera_all_frames(doc));
        return Ok(outcome);
    }
    if matches!(
        doc.project().calibration_object,
        Some(CalibrationObject::Checkerboard { .. })
    ) {
        return super::checkerboard::compute_uncalibrated(doc, camera, image);
    }
    let count = doc
        .session()
        .references
        .iter()
        .filter(|r| click_set(r.position))
        .count();
    if count < 3 {
        return Err(EditError::Rejected(NEED_THREE_REFERENCES.to_owned()));
    }
    let beads = detect_calibration_beads(doc, camera, image, &mut Default::default())?;
    if beads.len() < 6 {
        if doc.compat().refused_calibration_keeps_beads {
            store_beads(doc, camera, image, beads);
            return Ok(EditOutcome::refused(NOT_ENOUGH_POINTS));
        }
        return Err(EditError::Rejected(NOT_ENOUGH_POINTS.to_owned()));
    }
    store_beads(doc, camera, image, beads);
    let mode = if doc.project().cameras[camera].calibrated {
        CubeMode::Pose
    } else {
        CubeMode::PoseAndCam
    };
    let r =
        run_cube(doc, camera, image, mode, None).map_err(|e| EditError::Rejected(e.to_string()))?;
    write_result(doc, camera, image, &r);
    outcome.messages.extend(run_calibration_finished(doc)?);
    Ok(outcome)
}

pub(crate) fn check_image(doc: &Document, camera: usize, image: usize) -> Result<(), EditError> {
    let Some(cam) = doc.project().cameras.get(camera) else {
        return Err(EditError::Invalid(format!("camera {camera} out of range")));
    };
    if image >= cam.calibration_images.len() {
        return Err(EditError::Invalid(format!(
            "calibration image {image} out of range"
        )));
    }
    Ok(())
}
