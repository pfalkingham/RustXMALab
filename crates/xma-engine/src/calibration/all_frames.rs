//! The calibration of a camera over all its calibration images (C21, `Calibration`), what
//! starts it (`runCalibrationCameraAllFrames`) and what follows it, the projections it
//! stores (`MultiCameraCalibration::reproject`), and the wizard's `setDialog`.
//!
//! Ported from processing/Calibration.cpp (the constructor, `setInitialByReferences`,
//! `process`, `process_finished`), ui/WizardCalibrationCubeFrame.cpp
//! (`runCalibrationCameraAllFrames`, `runCalibrationCameraAllFramesFinished`, `setDialog`,
//! `resetReferences`, `reloadManualPoints`), processing/MultiCameraCalibration.cpp
//! (`reproject`) and core/Camera.cpp (`setCameraMatrix`, `setDistortionCoefficiants`,
//! `resetDistortion`).

use std::sync::Arc;

use nalgebra::DMatrix;
use xma_model::{CalibrationObject, CalibrationType, Matrix3, Point3};
use xma_vision::calib3d::{CalibrateFlags, calibrate_camera};
use xma_vision::camera::RigidTransform;
use xma_vision::linalg;

use super::held;
use super::undistortion::{camera_undistort, set_camera_pending};
use crate::document::Document;

/// The calibration object's points (`getFrameSpecifications`): a frame's beads or a
/// checkerboard's corners; empty for none.
pub(crate) fn object_points(doc: &Document) -> Vec<Point3> {
    doc.project()
        .calibration_object
        .as_ref()
        .map(CalibrationObject::points)
        .unwrap_or_default()
}

/// `Project::getNbImagesCalibration` as far as `camera` has images (the C++ reads past a
/// shorter camera's list; ledger F35).
pub(crate) fn n_frames(doc: &Document, camera: usize) -> usize {
    doc.session()
        .n_calibration_images
        .min(doc.project().cameras[camera].calibration_images.len())
}

/// Updates the camera model's matrix and lens coefficients from the project's camera, after
/// `setCameraMatrix`, `setDistortionCoefficiants` or `resetDistortion`.
pub(crate) fn refresh_intrinsics(doc: &mut Document, camera: usize) {
    let (k, dist) = {
        let cam = &doc.project().cameras[camera];
        (
            cam.camera_matrix,
            cam.has_model_distortion.then_some(cam.distortion),
        )
    };
    let d = &doc.cameras[camera].model.intrinsics;
    if d.camera_matrix == k && d.distortion == dist {
        return;
    }
    let d = Arc::make_mut(&mut doc.cameras[camera]);
    d.model.intrinsics.camera_matrix = k;
    d.model.intrinsics.distortion = dist;
}

/// `Camera::setCameraMatrix`: the matrix, and the camera calibrated.
pub(crate) fn set_camera_matrix(doc: &mut Document, camera: usize, k: Matrix3) {
    let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
    cam.camera_matrix = k;
    cam.calibrated = true;
    refresh_intrinsics(doc, camera);
}

/// `Camera::setDistortionCoefficiants`: the coefficients, the lens model on, then
/// `Camera::undistort` (the lens maps are the views').
pub(crate) fn set_distortion(doc: &mut Document, camera: usize, d: [f64; 8]) {
    let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
    cam.distortion = d;
    cam.has_model_distortion = true;
    refresh_intrinsics(doc, camera);
    camera_undistort(doc, camera);
}

/// `Camera::resetDistortion`: only when the lens model is on: coefficients zero, model off,
/// `Camera::undistort`.
pub(crate) fn reset_distortion(doc: &mut Document, camera: usize) {
    if !doc.project().cameras[camera].has_model_distortion {
        return;
    }
    let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
    cam.distortion = [0.0; 8];
    cam.has_model_distortion = false;
    refresh_intrinsics(doc, camera);
    camera_undistort(doc, camera);
}

/// Whether a calibration image takes part in the calibration over all frames and in
/// `reproject` for a planar object: it has detected points (`getDetectedPoints().size() >
/// 0`); ledger A188: in the default mode also at least one inlier.
fn planar_used(doc: &Document, im: &xma_model::CalibrationImage) -> bool {
    !im.detected.is_empty()
        && (doc.compat().planar_frames_without_points_used || im.inlier.iter().any(|&v| v > 0))
}

/// `MultiCameraCalibration::reproject(camera)`: every calibration image with a pose (cube)
/// or with detected points (planar object) gets the projections of the calibration object
/// with its pose and the camera matrix, without lens distortion, in single precision.
///
/// Ported from processing/MultiCameraCalibration.cpp, MultiCameraCalibration::reproject.
pub(crate) fn reproject(doc: &mut Document, camera: usize) {
    let pts = object_points(doc);
    if pts.is_empty() {
        return;
    }
    let planar = xma_io::calibration_object::is_planar(&pts);
    let k = doc.project().cameras[camera].camera_matrix;
    for f in 0..n_frames(doc, camera) {
        let im = &doc.project().cameras[camera].calibration_images[f];
        let used = (!planar && im.calibrated > 0) || (planar && planar_used(doc, im));
        if !used {
            continue;
        }
        let pose = RigidTransform {
            rotation: im.rotation,
            translation: im.translation,
        };
        let proj = xma_vision::multicam::reproject(&pts, &pose, &k);
        held::with_image(doc, camera, f, |im, h, model| {
            held::set_points_projected_undistorted(im, h, model, proj);
        });
    }
}

/// The camera matrix from the DLT of the first image's points, for a camera that is not
/// calibrated yet.
///
/// Ported from processing/Calibration.cpp, Calibration::setInitialByReferences. The products
/// `-u * X` of the C++ are of two `float`s, rounded to `float` before they are stored.
fn initial_by_references(object: &[[f32; 3]], image: &[[f32; 2]]) -> Matrix3 {
    let n = image.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        let [x, y, z] = object[i];
        let [u, v] = image[i];
        let r = 2 * i;
        a[(r, 0)] = f64::from(x);
        a[(r, 1)] = f64::from(y);
        a[(r, 2)] = f64::from(z);
        a[(r, 3)] = 1.0;
        a[(r, 8)] = f64::from(-u * x);
        a[(r, 9)] = f64::from(-u * y);
        a[(r, 10)] = f64::from(-u * z);
        a[(r, 11)] = f64::from(-u);
        a[(r + 1, 4)] = f64::from(x);
        a[(r + 1, 5)] = f64::from(y);
        a[(r + 1, 6)] = f64::from(z);
        a[(r + 1, 7)] = 1.0;
        a[(r + 1, 8)] = f64::from(-v * x);
        a[(r + 1, 9)] = f64::from(-v * y);
        a[(r + 1, 10)] = f64::from(-v * z);
        a[(r + 1, 11)] = f64::from(-v);
    }
    let f = linalg::solve_z(&a);
    // B, C = B * B^T, u0, v0, ku, kc, kv, the -999 branches and C(2,2) / C(2,2) are
    // CubeCalibration's decomposition, operation for operation.
    let projection = xma_vision::Matrix3x4::from_fn(|r, c| f[r * 4 + c]);
    let (mut k, _) = xma_vision::cube::decompose(&projection);
    // intrinsic_matrix(0, 1) = 0 at the end.
    k[(0, 1)] = 0.0;
    k
}

/// Ledger A204: the sentence after a camera's name when a checkerboard camera has fewer than
/// two frames with corners.
pub const NEED_TWO_FRAMES: &str = "could not be calibrated over all frames: a checkerboard camera needs at least two frames with detected corners";

/// Why the calibration over all images of a camera wrote nothing.
#[derive(Debug)]
enum AllFramesFailure {
    /// The camera is not calibrated and no image is used: the C++ reads the first image's
    /// points out of range (survey CD16; ledger A132).
    NoImage,
    /// `cv::calibrateCamera` threw.
    Refused(String),
}

/// One `Calibration(camera, planar)`: the constructor (which images are used, their points,
/// the starting camera matrix; unused images get `calibrated = 0`) and `process` (the
/// refinement and what it writes).
///
/// Ported from processing/Calibration.cpp, Calibration::Calibration and Calibration::process.
fn calibrate_camera_all_frames(doc: &mut Document, camera: usize) -> Result<(), AllFramesFailure> {
    let pts = object_points(doc);
    let planar = xma_io::calibration_object::is_planar(&pts);
    let lens = doc.project().cameras[camera].has_model_distortion;
    let distortion: Vec<f64> = if lens {
        doc.project().cameras[camera].distortion.to_vec()
    } else {
        vec![0.0; 8]
    };
    let model = doc.camera(camera).model.clone();
    let mut object = Vec::new();
    let mut image = Vec::new();
    let mut used_frames = Vec::new();
    for f in 0..n_frames(doc, camera) {
        let im = &doc.project().cameras[camera].calibration_images[f];
        let used = (!planar && im.calibrated == 1) || (planar && planar_used(doc, im));
        if used {
            let mut o = Vec::new();
            let mut i2 = Vec::new();
            for (k, p) in pts.iter().enumerate() {
                if im.inlier.get(k).is_some_and(|&v| v > 0)
                    && let Some(&d) = im.detected.get(k)
                {
                    #[allow(clippy::cast_possible_truncation)]
                    o.push(nalgebra::Point3::new(p.x as f32, p.y as f32, p.z as f32));
                    let u = model.undistort_point(d, true, false, true);
                    #[allow(clippy::cast_possible_truncation)]
                    i2.push(nalgebra::Point2::new(u.x as f32, u.y as f32));
                }
            }
            object.push(o);
            image.push(i2);
            used_frames.push(f);
        } else if im.calibrated != 0 {
            Arc::make_mut(&mut doc.project.cameras[camera]).calibration_images[f].calibrated = 0;
        }
    }
    let cam = &doc.project().cameras[camera];
    let (w, h) = (cam.width, cam.height);
    let mut k = if cam.calibrated {
        let mut k = cam.camera_matrix;
        k[(0, 1)] = 0.0;
        k
    } else {
        if object.is_empty() {
            return Err(AllFramesFailure::NoImage);
        }
        // Review 11: the first image used has no point (in the XMALab 3 mode a planar frame
        // used without an inlier, ledger A188; in either mode a project whose lists do not
        // match). XMALab's `setInitialByReferences` then runs `cv::SVD::solveZ` on a 0 x 12
        // matrix, which in OpenCV 4.13 writes through a null pointer (`_SVDcompute` copies
        // the empty matrix into a released buffer, then `JacobiSVDImpl_` fills it with random
        // vectors): XMALab crashes. Had it not, `cv::calibrateCamera` would have thrown on
        // the empty view and nothing would have been written. Refused here, before anything
        // of the calibration is written, in both modes.
        if object[0].is_empty() {
            return Err(AllFramesFailure::Refused(
                "the first frame used has no inlier point, so no starting camera matrix can \
                 be computed"
                    .to_owned(),
            ));
        }
        let o: Vec<[f32; 3]> = object[0].iter().map(|p| [p.x, p.y, p.z]).collect();
        let i: Vec<[f32; 2]> = image[0].iter().map(|p| [p.x, p.y]).collect();
        initial_by_references(&o, &i)
    };
    // process()
    let (wf, hf) = (f64::from(w), f64::from(h));
    if k[(0, 2)] < 0.0 {
        k[(0, 2)] = 1.0;
    }
    if k[(0, 2)] > wf {
        k[(0, 2)] = wf - 1.0;
    }
    if k[(1, 2)] < 0.0 {
        k[(1, 2)] = 0.0;
    }
    if k[(1, 2)] > hf {
        k[(1, 2)] = hf - 1.0;
    }
    let mut flags = CalibrateFlags {
        use_intrinsic_guess: !planar,
        ..CalibrateFlags::default()
    };
    if lens {
        flags.rational_model = true;
    } else {
        flags.fix_k = [true; 6];
        flags.zero_tangent_dist = true;
    }
    if doc.settings().fix_principal {
        flags.fix_principal_point = true;
        // Ledger A112 (survey CD28): XMALab fixes the 1-based image centre.
        let off = if doc.compat().principal_point_one_based {
            1.0
        } else {
            -1.0
        };
        k[(0, 2)] = 0.5 * (wf + off);
        k[(1, 2)] = 0.5 * (hf + off);
    }
    // Without `CALIB_USE_INTRINSIC_GUESS` (a planar object) OpenCV reads neither the
    // matrix nor the coefficients; `calibrate_camera` ignores them too (ledger A172), so the
    // DLT of a plane's points, which has no finite decomposition, is passed as it is.
    let out = calibrate_camera(
        &object,
        &image,
        (w as usize, h as usize),
        &k,
        &distortion,
        flags,
    )
    .map_err(|e| AllFramesFailure::Refused(e.to_string()))?;
    set_camera_matrix(doc, camera, out.camera_matrix);
    Arc::make_mut(&mut doc.project.cameras[camera]).optimized = false;
    if lens {
        let mut d = [0.0; 8];
        for (t, v) in d.iter_mut().zip(&out.distortion) {
            *t = *v;
        }
        set_distortion(doc, camera, d);
    } else {
        reset_distortion(doc, camera);
    }
    for (count, f) in used_frames.into_iter().enumerate() {
        let pose = out.poses[count];
        let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
        held::set_matrices(
            &mut cam.calibration_images[f],
            pose.rotation,
            pose.translation,
        );
    }
    reproject(doc, camera);
    Ok(())
}

/// The calibration over all images of `camera` and its `process_finished`: the camera's
/// pending recalibration cleared. A failure is reported; in the default mode the camera
/// stays pending (ledger A132), in the XMALab 3 mode it is cleared and nothing is said.
///
/// Ledger A189: in the default mode a checkerboard camera calibrated here is then compared
/// with the other cameras' numbering of the board ([`super::numbering`]: a warning).
pub(crate) fn calibration_job(doc: &mut Document, camera: usize) -> Vec<String> {
    calibration_jobs(doc, &[camera])
}

/// [`calibration_job`] for several cameras in one edit: each camera's calibration, then the
/// numbering compared once, on the state after all of them (review 12: once per camera
/// repeated each warning and compared a half-updated rig), when one of them ended
/// calibrated and no longer pending.
pub(crate) fn calibration_jobs(doc: &mut Document, cameras: &[usize]) -> Vec<String> {
    let mut messages = Vec::new();
    for &camera in cameras {
        messages.extend(calibration_job_without_numbering(doc, camera));
    }
    if cameras.iter().any(|&camera| {
        !doc.session()
            .camera_recalibration
            .get(camera)
            .is_some_and(|&v| v != 0)
            && doc.project().cameras[camera].calibrated
    }) {
        messages.extend(super::numbering::after_calibration(doc));
    }
    messages
}

/// [`calibration_job`] without the comparison of the board's numbering (ledger A189).
pub(crate) fn calibration_job_without_numbering(doc: &mut Document, camera: usize) -> Vec<String> {
    let mut messages = Vec::new();
    // Ledger A204: a checkerboard camera needs two frames with corners on every path, as
    // `runCalibrationCameraAllFrames` requires (a single view of a plane does not determine
    // the camera without an intrinsic guess). Not calibrated, still pending, reported.
    if !doc.compat().checkerboard_single_frame_calibrated
        && matches!(
            doc.project().calibration_object,
            Some(CalibrationObject::Checkerboard { .. })
        )
        && super::checkerboard::frames_with_corners(doc, camera) < 2
    {
        let name = doc.project().cameras[camera].name.clone();
        messages.push(format!("{name} {NEED_TWO_FRAMES}"));
        return messages;
    }
    match calibrate_camera_all_frames(doc, camera) {
        Ok(()) => set_camera_pending(doc, camera, 0),
        Err(e) => {
            let name = doc.project().cameras[camera].name.clone();
            let why = match e {
                AllFramesFailure::NoImage => {
                    format!("{name} could not be calibrated over all frames: no calibrated frame")
                }
                AllFramesFailure::Refused(t) => {
                    format!("{name} could not be calibrated over all frames: {t}")
                }
            };
            if doc.compat().failed_recalibration_cleared {
                set_camera_pending(doc, camera, 0);
            } else {
                messages.push(why);
            }
        }
    }
    messages
}

/// C21, `runCalibrationCameraAllFrames`: one calibration over all images for every camera
/// flagged for recalibration; when there was at least one, `setDialog` and every trial
/// flagged for recomputation. Returns the messages of failed calibrations.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, runCalibrationCameraAllFrames (cube; the
/// checkerboard's branch is `checkerboard::run_all_frames`) and
/// runCalibrationCameraAllFramesFinished.
pub(crate) fn run_calibration_camera_all_frames(doc: &mut Document) -> Vec<String> {
    match doc.project().calibration_object {
        Some(CalibrationObject::Frame(_)) => {}
        Some(CalibrationObject::Checkerboard { .. }) => {
            return super::checkerboard::run_all_frames(doc);
        }
        None => return Vec::new(),
    }
    let flagged = flagged_cameras(doc);
    if flagged.is_empty() {
        return Vec::new();
    }
    let messages = calibration_jobs(doc, &flagged);
    set_dialog(doc);
    flag_trials(doc);
    messages
}

/// The cameras flagged for recalibration (`cameraRequiresRecalibration` not 0), in order.
pub(crate) fn flagged_cameras(doc: &Document) -> Vec<usize> {
    (0..doc.project().cameras.len())
        .filter(|&c| {
            doc.session()
                .camera_recalibration
                .get(c)
                .is_some_and(|&v| v != 0)
        })
        .collect()
}

/// Every trial flagged for recomputation (`Trial::setRequiresRecomputation(true)`).
pub(crate) fn flag_trials(doc: &mut Document) {
    for t in 0..doc.project().trials.len() {
        doc.set_trial_requires_recomputation(t, true);
    }
}

/// What the wizard's `setDialog` does to the session and the project: for a project with an
/// internal calibration, once undistorted, the reference clicks cleared and their point ids
/// reloaded, and in manual mode the active image given its point lists when it has fewer
/// than the calibration object has points.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, setDialog, resetReferences and
/// reloadManualPoints.
pub(crate) fn set_dialog(doc: &mut Document) {
    if doc.project().calibration_type != CalibrationType::Internal || !doc.session().undistorted {
        return;
    }
    let project = doc.project.clone();
    doc.session_mut().reload_references(&project);
    if doc.session().manual_mode {
        reload_manual_points(doc);
    }
}

/// Ported from ui/WizardCalibrationCubeFrame.cpp, reloadManualPoints: the active image is
/// given `n` unset points when its inlier list is shorter than the calibration object.
///
/// Inside a step (a calibration edit) the lists are created at once. Outside one (the
/// window's check box or a click on another image) the project is not changed: the image is
/// recorded in [`crate::Session::manual_lists_pending`] and the next calibration edit
/// creates its lists ([`create_pending_manual_lists`], ledger A164).
pub(crate) fn reload_manual_points(doc: &mut Document) {
    let (c, i) = (
        doc.session().active_camera,
        doc.session().active_frame_calibration,
    );
    if needs_manual_lists(doc, c, i).is_none() {
        return;
    }
    if doc.step_open() {
        init_manual_lists(doc, c, i);
    } else {
        doc.session_mut().manual_lists_pending.insert((c, i));
    }
}

/// Ledger A164: keeps manual mode's pending point lists true to the project after a step
/// is committed or taken back, an undo or a redo: images that no longer exist or no longer
/// need their lists leave [`crate::Session::manual_lists_pending`], and in manual mode the
/// active image is given its lists again (recorded as pending) when it has fewer than the
/// calibration object, so that the window always reads one entry per point, as XMALab,
/// which has no undo, holds them. Session only: the project is not changed.
pub(crate) fn sync_manual_lists(doc: &mut Document) {
    if doc.step_open() {
        return;
    }
    let keep: std::collections::BTreeSet<(usize, usize)> = doc
        .session()
        .manual_lists_pending
        .iter()
        .copied()
        .filter(|&(c, i)| needs_manual_lists(doc, c, i).is_some())
        .collect();
    doc.session_mut().manual_lists_pending = keep;
    if doc.session().manual_mode {
        reload_manual_points(doc);
    }
}

/// The calibration object's point count when image `image` of `camera` exists and has fewer
/// inlier flags than that (`reloadManualPoints`' condition), or fewer detected points
/// (review 11: the two lists are read from separate files; a detected list shorter than
/// the object would be indexed past its end by a manual-mode click. XMALab writes them with
/// equal lengths, so this changes nothing for its projects).
fn needs_manual_lists(doc: &Document, camera: usize, image: usize) -> Option<usize> {
    let n = object_points(doc).len();
    let im = doc
        .project()
        .cameras
        .get(camera)?
        .calibration_images
        .get(image)?;
    (n > im.inlier.len() || n > im.detected.len()).then_some(n)
}

/// `CalibrationImage::init(n)` on an image that needs it.
fn init_manual_lists(doc: &mut Document, camera: usize, image: usize) {
    if let Some(n) = needs_manual_lists(doc, camera, image) {
        held::with_image(doc, camera, image, |im, h, _| held::init(im, h, n));
    }
}

/// Ledger A164: creates the point lists manual mode gave images outside an edit
/// ([`crate::Session::manual_lists_pending`]), for a calibration edit to run on what XMALab
/// holds. Returns the images it recorded (the set is emptied).
pub(crate) fn create_pending_manual_lists(doc: &mut Document) -> Vec<(usize, usize)> {
    let pending: Vec<(usize, usize)> = std::mem::take(&mut doc.session_mut().manual_lists_pending)
        .into_iter()
        .collect();
    for &(c, i) in &pending {
        init_manual_lists(doc, c, i);
    }
    pending
}

impl Document {
    /// The point lists of calibration image `image` of `camera` as XMALab holds them: the
    /// stored image, or, for an image manual mode gave its lists outside an edit
    /// ([`crate::Session::manual_lists_pending`], ledger A164), the image with `n` unset
    /// points (-1, -1) and `n` inlier flags 0 (`CalibrationImage::init`), which the project
    /// receives with the next calibration edit. `None` for an image that does not exist.
    #[must_use]
    pub fn calibration_image_lists(
        &self,
        camera: usize,
        image: usize,
    ) -> Option<std::borrow::Cow<'_, xma_model::CalibrationImage>> {
        let im = self
            .project()
            .cameras
            .get(camera)?
            .calibration_images
            .get(image)?;
        if !self
            .session()
            .manual_lists_pending
            .contains(&(camera, image))
        {
            return Some(std::borrow::Cow::Borrowed(im));
        }
        let Some(n) = needs_manual_lists(self, camera, image) else {
            return Some(std::borrow::Cow::Borrowed(im));
        };
        let mut copy = im.clone();
        let mut h = held::HeldPoints::default();
        held::init(&mut copy, &mut h, n);
        Some(std::borrow::Cow::Owned(copy))
    }
}

#[cfg(test)]
mod tests {
    //! Review 13 (simplification round, calibration.md finding 9): `initial_by_references`
    //! through `xma_vision::cube::decompose` against the function as it was, kept here.

    use super::*;
    use xma_vision::rigid::gemm_small;

    /// The decomposition of `initial_by_references` before 20 November 2026.
    #[allow(clippy::many_single_char_names)]
    fn reference_decomposition(f: &[f64]) -> Matrix3 {
        let mut b = [[0.0f64; 3]; 3];
        let mut bt = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i][j] = f[i * 4 + j];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                bt[j][i] = b[i][j];
            }
        }
        let c = gemm_small(&b, &bt);
        let u0 = c[0][2] / c[2][2];
        let v0 = c[1][2] / c[2][2];
        let ku = c[0][0] / c[2][2];
        let kc = c[0][1] / c[2][2];
        let kv = c[1][1] / c[2][2];
        let bb = if (kv - v0 * v0) > 0.0 {
            (kv - v0 * v0).sqrt()
        } else {
            -999.0
        };
        let g = (kc - u0 * v0) / bb;
        let aa = if ku - u0 * u0 - g * g > 0.0 {
            (ku - u0 * u0 - g * g).sqrt()
        } else {
            -999.0
        };
        #[allow(clippy::eq_op)]
        let k22 = c[2][2] / c[2][2];
        // intrinsic_matrix(0, 1) = 0 at the end.
        Matrix3::new(aa, 0.0, u0, 0.0, bb, v0, 0.0, 0.0, k22)
    }

    /// `initial_by_references` before 20 November 2026.
    fn reference(object: &[[f32; 3]], image: &[[f32; 2]]) -> Matrix3 {
        let n = image.len();
        let mut a = DMatrix::<f64>::zeros(2 * n, 12);
        for i in 0..n {
            let [x, y, z] = object[i];
            let [u, v] = image[i];
            let r = 2 * i;
            a[(r, 0)] = f64::from(x);
            a[(r, 1)] = f64::from(y);
            a[(r, 2)] = f64::from(z);
            a[(r, 3)] = 1.0;
            a[(r, 8)] = f64::from(-u * x);
            a[(r, 9)] = f64::from(-u * y);
            a[(r, 10)] = f64::from(-u * z);
            a[(r, 11)] = f64::from(-u);
            a[(r + 1, 4)] = f64::from(x);
            a[(r + 1, 5)] = f64::from(y);
            a[(r + 1, 6)] = f64::from(z);
            a[(r + 1, 7)] = 1.0;
            a[(r + 1, 8)] = f64::from(-v * x);
            a[(r + 1, 9)] = f64::from(-v * y);
            a[(r + 1, 10)] = f64::from(-v * z);
            a[(r + 1, 11)] = f64::from(-v);
        }
        reference_decomposition(linalg::solve_z(&a).as_slice())
    }

    fn same(a: &Matrix3, b: &Matrix3) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The new decomposition of a 12-vector, as `initial_by_references` does it.
    fn decomposition(f: &[f64]) -> Matrix3 {
        let projection = xma_vision::Matrix3x4::from_fn(|r, c| f[r * 4 + c]);
        let (mut k, _) = xma_vision::cube::decompose(&projection);
        k[(0, 1)] = 0.0;
        k
    }

    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let v = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            v
        }
    }

    const SPECIAL: [f64; 9] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE / 4.0,
        1e300,
    ];

    /// The decomposition alone, on random 12-vectors, vectors with special values, and
    /// rank-deficient B (both -999 branches).
    #[test]
    fn review13_initial_by_references_decomposition() {
        let mut r = Lcg(1);
        let (mut b999, mut a999) = (0, 0);
        for case in 0..200_000 {
            let mut f: Vec<f64> = (0..12).map(|_| 2.0 * r.unit() - 1.0).collect();
            match case % 5 {
                1 => {
                    // B's second row a multiple of the third: kv - v0^2 = 0 (or rounding).
                    let s = 2.0 * r.unit() - 1.0;
                    for j in 0..3 {
                        f[4 + j] = s * f[8 + j];
                    }
                }
                2 => {
                    // B's first row a multiple of the third.
                    let s = 2.0 * r.unit() - 1.0;
                    for j in 0..3 {
                        f[j] = s * f[8 + j];
                    }
                }
                3 =>
                {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    for _ in 0..1 + (r.unit() * 3.0) as usize {
                        let i = (r.unit() * 12.0) as usize % 12;
                        f[i] = SPECIAL[(r.unit() * 9.0) as usize % 9];
                    }
                }
                4 => {
                    #[allow(clippy::cast_possible_truncation)]
                    let e = (r.unit() * 600.0 - 300.0) as i32;
                    for v in &mut f {
                        *v *= 10f64.powi(e);
                    }
                }
                _ => {}
            }
            let want = reference_decomposition(&f);
            if want[(1, 1)] == -999.0 {
                b999 += 1;
            }
            if want[(0, 0)] == -999.0 {
                a999 += 1;
            }
            assert!(same(&decomposition(&f), &want), "{f:?}");
        }
        assert!(b999 > 100 && a999 > 100, "branches: {b999} {a999}");
    }

    /// The whole function on random point sets (planar like a checkerboard, and not),
    /// degenerate sets and special values.
    #[test]
    fn review13_initial_by_references_points() {
        let mut r = Lcg(2);
        #[allow(clippy::cast_possible_truncation)]
        for case in 0..3000 {
            let n = 1 + (r.unit() * 40.0) as usize;
            let planar = case % 3 == 0;
            let mut object: Vec<[f32; 3]> = (0..n)
                .map(|_| {
                    let z = if planar {
                        0.0
                    } else {
                        (r.unit() * 50.0) as f32
                    };
                    [(r.unit() * 100.0) as f32, (r.unit() * 100.0) as f32, z]
                })
                .collect();
            let mut image: Vec<[f32; 2]> = (0..n)
                .map(|_| [(r.unit() * 1024.0) as f32, (r.unit() * 1024.0) as f32])
                .collect();
            if case % 7 == 1 {
                let p = object[0];
                object.iter_mut().for_each(|o| *o = p);
            }
            if case % 11 == 2 {
                let i = (r.unit() * n as f64) as usize % n;
                image[i][0] = SPECIAL[case % 9] as f32;
                object[i][2] = SPECIAL[(case / 9) % 9] as f32;
            }
            let want = reference(&object, &image);
            assert!(
                same(&initial_by_references(&object, &image), &want),
                "case {case}"
            );
        }
    }
}
