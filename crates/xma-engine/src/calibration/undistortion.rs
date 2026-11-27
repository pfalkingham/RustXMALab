//! The Undistortion workspace's actions (C1, C2, C6, C7, C8, C11) and `Camera::undistort`.
//!
//! Ported from ui/WizardUndistortionFrame.cpp (`on_pushButton_clicked`,
//! `computeUndistortion`, `recomputeUndistortion`, `undistortionFinished`,
//! `on_pushButtonRemoveOutlierAutomatically_clicked`, `checkForPendingChanges`),
//! processing/LocalUndistortion.cpp (`computeUndistortion`, `localUndistortion_thread`),
//! processing/BlobDetection.cpp (the grid image, `image = -1`), core/UndistortionObject.cpp
//! (`toggleOutlier`, `setCenter`, `removeOutlier`, the setters `localUndistortion_thread`
//! calls) and core/Camera.cpp (`Camera::undistort`), with the algorithms of
//! `xma_vision::grid_match`.

use std::sync::Arc;

use xma_model::{CalibrationType, Point2};
use xma_vision::beads::BeadImage;
use xma_vision::grid_match::{
    GridMatchCompat, OutlierCompat, PendingRecompute, compute_grid_undistortion, fit_grid,
    remove_outliers, toggle_outlier,
};
use xma_vision::image::ImageU8;

use super::{GridPointList, all_frames, held, images};
use crate::document::Document;
use crate::edit::{EditError, EditOutcome, EditResult};

/// The grid matching's part of a document's mode (ledger A104 to A106).
pub(crate) fn grid_compat(doc: &Document) -> GridMatchCompat {
    let c = doc.compat();
    GridMatchCompat {
        six_neighbours_any_target: c.grid_neighbour_for_two_targets,
        lattice_site_by_coordinates: c.grid_sites_compared_by_coordinates,
        unchecked_reference_scale: c.grid_reference_scale_unchecked,
    }
}

/// "remove outlier at border"'s part of a document's mode (ledger A107, A108).
pub(crate) fn outlier_compat(doc: &Document) -> OutlierCompat {
    let c = doc.compat();
    OutlierCompat {
        circle_sets_inliers: c.outlier_circle_sets_inliers,
        circle_radius_signed: c.outlier_circle_radius_signed,
    }
}

fn pending(doc: &Document, camera: usize) -> PendingRecompute {
    match doc
        .session()
        .grid_recalibration
        .get(camera)
        .copied()
        .unwrap_or(0)
    {
        0 => PendingRecompute::None,
        1 => PendingRecompute::Refit,
        _ => PendingRecompute::Rematch,
    }
}

fn set_pending(doc: &mut Document, camera: usize, p: PendingRecompute) {
    let v = match p {
        PendingRecompute::None => 0,
        PendingRecompute::Refit => 1,
        PendingRecompute::Rematch => 2,
    };
    let s = doc.session_mut();
    if s.grid_recalibration.len() <= camera {
        s.grid_recalibration.resize(camera + 1, 0);
    }
    s.grid_recalibration[camera] = v;
}

/// Sets `Camera::requiresRecalibration` of `camera`.
pub(crate) fn set_camera_pending(doc: &mut Document, camera: usize, v: i32) {
    let s = doc.session_mut();
    if s.camera_recalibration.len() <= camera {
        s.camera_recalibration.resize(camera + 1, 0);
    }
    s.camera_recalibration[camera] = v;
}

fn has_grid(doc: &Document, camera: usize) -> bool {
    doc.project().cameras[camera].undistortion.is_some()
}

/// Refuses a grid whose stored point lists (`GridPointsDistorted`, `GridPointsReferences`,
/// `GridPointsInlier`, read from three files) differ in length: a refit cannot use them,
/// and "toggle outlier" and "remove outlier at border" would only set a refit that fails.
/// XMALab writes the three with equal lengths; another length comes from an altered
/// project (review 11; the load warns that the grid is not used).
fn check_grid_lists(doc: &Document, camera: usize) -> Result<(), EditError> {
    let cam = &doc.project().cameras[camera];
    let Some(g) = cam.undistortion.as_ref() else {
        return Ok(());
    };
    let (d, r, i) = (
        g.grid_distorted.len(),
        g.grid_references.len(),
        g.grid_inlier.len(),
    );
    if d == r && r == i {
        return Ok(());
    }
    Err(EditError::Rejected(format!(
        "{}: the undistortion grid's stored lists differ in length ({d} distorted points, \
         {r} reference points, {i} inlier flags), so it cannot be refitted; compute the \
         undistortion again from a new centre or reload the grid.",
        cam.name
    )))
}

/// [`check_grid_lists`] for every grid whose pending change is a refit, before any grid is
/// recomputed.
fn check_pending_refits(doc: &Document) -> Result<(), EditError> {
    for c in 0..doc.project().cameras.len() {
        if has_grid(doc, c) && pending(doc, c) == PendingRecompute::Refit {
            check_grid_lists(doc, c)?;
        }
    }
    Ok(())
}

/// One `LocalUndistortion` run on a grid: the matching from the stored beads (`rematch`,
/// `computeUndistortion(false)`) or the refit from the stored grid points and inliers
/// (`computeUndistortion(true)`), then everything `localUndistortion_thread` stores, and
/// `Camera::undistort`.
///
/// Ported from processing/LocalUndistortion.cpp, LocalUndistortion::computeUndistortion and
/// localUndistortion_thread.
fn local_undistortion(doc: &mut Document, camera: usize, rematch: bool) -> Result<(), EditError> {
    let frame = images::grid_frame(doc, camera).map_err(EditError::Rejected)?;
    let (w, h) = (frame.width as usize, frame.height as usize);
    let neighbours = doc.settings().local_undistortion_neighbours as usize;
    let compat = grid_compat(doc);
    let g = doc.project().cameras[camera]
        .undistortion
        .clone()
        .expect("a camera with a grid");
    let (distorted, references, inlier, fit) = if rematch {
        let r = compute_grid_undistortion(&g.points_detected, g.center, w, h, neighbours, compat)
            .map_err(|e| {
            EditError::Rejected(format!("{}: {e}", doc.project().cameras[camera].name))
        })?;
        (
            r.matching.distorted,
            r.matching.references,
            r.matching.inlier,
            r.fit,
        )
    } else {
        check_grid_lists(doc, camera)?;
        let fit = fit_grid(
            &g.grid_distorted,
            &g.grid_references,
            &g.grid_inlier,
            w,
            h,
            neighbours,
        );
        (g.grid_distorted, g.grid_references, g.grid_inlier, fit)
    };
    // Ledger A161: a matching or refit that gives no grid model.
    if fit.model.is_none() {
        let inliers = inlier.iter().filter(|&&v| v).count();
        // The C++ is defined only without inliers (empty matrices: every point maps to
        // itself, the grid computed); with 1 to `neighbours - 1` inliers, or 0 neighbours,
        // it reads past its arrays (ledger A44).
        let defined = inliers == 0 && neighbours > 0;
        if !(defined && doc.compat().grid_fit_without_inliers_accepted) {
            let name = &doc.project().cameras[camera].name;
            return Err(EditError::Rejected(if inliers == 0 {
                format!(
                    "{name}: the undistortion grid has no inlier point, so no undistortion \
                     can be fitted."
                )
            } else {
                format!(
                    "{name}: the undistortion grid has {inliers} inlier points, fewer than the \
                     {neighbours} neighbours (LocalUndistortionNeighbours) each point of the \
                     fit needs, so no undistortion can be fitted."
                )
            }));
        }
    }
    // Only what changes is written, so that a refit of a computed grid shares the
    // project's camera with the undo step (ledger A166).
    let stored = doc.project().cameras[camera]
        .undistortion
        .as_ref()
        .expect("grid");
    if !stored.computed || rematch {
        let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
        let grid = cam.undistortion.as_mut().expect("grid");
        grid.computed = true;
        if rematch {
            grid.grid_distorted = distorted;
            grid.grid_references = references;
            grid.grid_inlier = inlier;
        }
    }
    {
        // The model's source: the stored grid points just written (ledger A166).
        let source = fit.model.is_some().then(|| crate::document::GridSource {
            camera: doc.project.cameras[camera].clone(),
            neighbours,
        });
        let d = Arc::make_mut(&mut doc.cameras[camera]);
        d.model.grid = fit.model;
        d.lookup = Some(Arc::new(fit.lookup));
        d.grid_source = source;
    }
    set_pending(doc, camera, PendingRecompute::None);
    camera_undistort(doc, camera);
    Ok(())
}

/// `Camera::undistort`: a calibrated camera is flagged for recalibration, and every
/// calibration image's undistorted points are recomputed from its distorted ones (only in
/// a project with an internal calibration). The undistorted images are made by the views.
///
/// Ported from core/Camera.cpp, Camera::undistort, and core/CalibrationSequence.cpp,
/// CalibrationSequence::undistort.
pub(crate) fn camera_undistort(doc: &mut Document, camera: usize) {
    if doc.project().cameras[camera].calibrated {
        set_camera_pending(doc, camera, 1);
    }
    if doc.project().calibration_type != CalibrationType::Internal {
        return;
    }
    let n = doc.project().cameras[camera].calibration_images.len();
    for i in 0..n {
        held::with_image(doc, camera, i, |im, h, model| {
            held::undistort_points(im, h, model);
        });
    }
}

/// C1 and C2, "compute undistortion" / "recompute undistortion".
///
/// Ported from ui/WizardUndistortionFrame.cpp, on_pushButton_clicked, computeUndistortion,
/// recomputeUndistortion and undistortionFinished (with `WizardDockWidget::updateWizard`):
/// before the first computation the beads of every grid are detected and every grid is
/// matched and fitted; after it, each grid with pending changes is matched again (a new
/// centre) or refitted (changed inliers). When at least one grid was computed the state
/// becomes "undistorted" and every camera flagged for recalibration is calibrated again
/// over all its images.
pub(crate) fn compute_undistortion(doc: &mut Document) -> EditResult {
    let n = doc.project().cameras.len();
    let grids: Vec<usize> = (0..n).filter(|&c| has_grid(doc, c)).collect();
    let mut ran = Vec::new();
    if doc.session().undistorted {
        check_pending_refits(doc)?;
        for &c in &grids {
            match pending(doc, c) {
                PendingRecompute::None => {}
                PendingRecompute::Refit => {
                    local_undistortion(doc, c, false)?;
                    ran.push(c);
                }
                PendingRecompute::Rematch => {
                    local_undistortion(doc, c, true)?;
                    ran.push(c);
                }
            }
        }
    } else {
        // BlobDetection(i, -1) for every grid, then, when the last has finished, one
        // LocalUndistortion(i).computeUndistortion() per grid.
        for &c in &grids {
            let frame = images::grid_frame(doc, c).map_err(EditError::Rejected)?;
            let beads =
                images::detect(doc, &frame, BeadImage::Grid).map_err(EditError::Rejected)?;
            let cam = Arc::make_mut(&mut doc.project.cameras[c]);
            cam.undistortion.as_mut().expect("grid").points_detected = beads;
        }
        for &c in &grids {
            local_undistortion(doc, c, true)?;
            ran.push(c);
        }
    }
    let mut outcome = EditOutcome::none();
    if !ran.is_empty() {
        doc.session_mut().undistorted = true;
        outcome
            .messages
            .extend(all_frames::run_calibration_camera_all_frames(doc));
    }
    Ok(outcome)
}

/// C11, the "yes" of the pending-changes question in the Undistortion workspace: each grid
/// with pending changes is recomputed as C2 does it, without the recalibration that
/// follows C2 (the question's handler does not call `updateWizard`).
///
/// Ported from ui/WizardUndistortionFrame.cpp, checkForPendingChanges.
pub(crate) fn recompute_pending_grids(doc: &mut Document) -> Result<(), EditError> {
    check_pending_refits(doc)?;
    let n = doc.project().cameras.len();
    for c in 0..n {
        if !has_grid(doc, c) {
            continue;
        }
        match pending(doc, c) {
            PendingRecompute::None => {}
            PendingRecompute::Refit => local_undistortion(doc, c, false)?,
            PendingRecompute::Rematch => local_undistortion(doc, c, true)?,
        }
    }
    Ok(())
}

/// C6, a click in "toggle outlier" mode. A camera without a grid, or a click with no grid
/// point within 10 pixels, changes nothing.
///
/// Ported from ui/GLCameraView.cpp, mousePressEvent (`:236-247`), and
/// core/UndistortionObject.cpp, UndistortionObject::toggleOutlier.
pub(crate) fn toggle_grid_outlier(
    doc: &mut Document,
    camera: usize,
    list: GridPointList,
    x: f64,
    y: f64,
) -> EditResult {
    check_camera(doc, camera)?;
    if !has_grid(doc, camera) {
        return Ok(EditOutcome::none());
    }
    check_grid_lists(doc, camera)?;
    let points: Vec<Point2> = {
        let g = doc.project().cameras[camera]
            .undistortion
            .as_ref()
            .expect("grid");
        match list {
            GridPointList::Distorted => g.grid_distorted.clone(),
            GridPointList::References => g.grid_references.clone(),
            GridPointList::Undistorted => crate::view::calibration::undistortion_grid(doc, camera)
                .map(|v| v.grid_undistorted)
                .unwrap_or_default(),
        }
    };
    let mut inlier = doc.project().cameras[camera]
        .undistortion
        .as_ref()
        .expect("grid")
        .grid_inlier
        .clone();
    if points.len() != inlier.len() {
        return Ok(EditOutcome::none());
    }
    if toggle_outlier(&points, &mut inlier, x, y).is_some() {
        let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
        cam.undistortion.as_mut().expect("grid").grid_inlier = inlier;
        let p = pending(doc, camera).after_toggle();
        set_pending(doc, camera, p);
    }
    Ok(EditOutcome::none())
}

/// C7, a click in "set center" mode on the distorted grid image.
///
/// Ported from core/UndistortionObject.cpp, UndistortionObject::setCenter.
pub(crate) fn set_grid_center(doc: &mut Document, camera: usize, x: f64, y: f64) -> EditResult {
    check_camera(doc, camera)?;
    if !has_grid(doc, camera) {
        return Ok(EditOutcome::none());
    }
    let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
    cam.undistortion.as_mut().expect("grid").center = Some(Point2::new(x, y));
    let p = pending(doc, camera).after_set_center();
    set_pending(doc, camera, p);
    Ok(EditOutcome::none())
}

/// C8, "remove outlier at border" with its two thresholds, on every camera with a grid.
///
/// Ported from ui/WizardUndistortionFrame.cpp,
/// on_pushButtonRemoveOutlierAutomatically_clicked, and core/UndistortionObject.cpp,
/// UndistortionObject::removeOutlier.
pub(crate) fn remove_grid_outliers(doc: &mut Document, border: i32, circle: i32) -> EditResult {
    if !(0..=100).contains(&border) || !(0..=100).contains(&circle) {
        return Err(EditError::Invalid(format!(
            "thresholds {border} and {circle}: the dialogs take 0 to 100"
        )));
    }
    let compat = outlier_compat(doc);
    let n = doc.project().cameras.len();
    for c in 0..n {
        check_grid_lists(doc, c)?;
    }
    for c in 0..n {
        if !has_grid(doc, c) {
            continue;
        }
        let frame = images::grid_frame(doc, c).map_err(EditError::Rejected)?;
        let image = ImageU8::from_vec(
            frame.width as usize,
            frame.height as usize,
            frame.to_gray8().into_owned(),
        );
        let cam = Arc::make_mut(&mut doc.project.cameras[c]);
        let g = cam.undistortion.as_mut().expect("grid");
        let mut inlier = std::mem::take(&mut g.grid_inlier);
        if inlier.len() == g.grid_distorted.len() {
            remove_outliers(
                &image,
                &g.grid_distorted,
                &mut inlier,
                f64::from(circle),
                f64::from(border),
                compat,
            );
        }
        g.grid_inlier = inlier;
        let p = pending(doc, c).after_remove_outliers();
        set_pending(doc, c, p);
    }
    Ok(EditOutcome::none())
}

fn check_camera(doc: &Document, camera: usize) -> Result<(), EditError> {
    if camera >= doc.project().cameras.len() {
        return Err(EditError::Invalid(format!("camera {camera} out of range")));
    }
    Ok(())
}
