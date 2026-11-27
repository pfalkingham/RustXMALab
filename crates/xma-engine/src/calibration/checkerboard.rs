//! Checkerboard calibration: "compute calibration" on a calibration image that is not
//! calibrated (C40 automatic detection, C41 manual corners) and what follows it, and the
//! board's part of the calibration over all frames (`runCalibrationCameraAllFrames`).
//!
//! Ported from ui/WizardCalibrationCubeFrame.cpp (`on_pushButton_clicked`, `runCalibration`,
//! `runCalibrationCameraAllFrames` and `addCalibrationReference`, the checkerboard branches),
//! processing/CheckerboardDetection.cpp (`detectCorner_thread`, `detectCorner_threadFinished`)
//! and core/CalibrationImage.cpp (`setDetectedPoints`, `sortGridByReference`).
//!
//! # What XMALab does
//!
//! - **Automatic** (`DisableCheckerboardDetection` off, and no failed detection since the
//!   last change of camera, frame or workspace): the click on the interior upper-left corner
//!   is reference 1 (`selectedReferencePoints[0]`); with `AutoCalibAfterReference` it runs
//!   "compute calibration". The detection is `cv::findChessboardCorners(image, (H, V),
//!   CALIB_CB_ADAPTIVE_THRESH)` on the 8-bit distorted image, then, when found,
//!   `cv::cornerSubPix` (window 11 x 11, no zero zone, 30 iterations or 0.1).
//! - **Manual** (the setting, or after "Could not detect chessboard"): four clicks on the
//!   board's outer interior corners, upper left first and then clockwise as the board's
//!   points run (`(0, 0)`, `(H-1, 0)`, `(H-1, V-1)`, `(0, V-1)`), in the reference rows;
//!   the fourth runs "compute calibration" with `AutoCalibAfterReference`. A homography
//!   (`cv::findHomography`, method 0) from those four board corners to the clicks predicts
//!   every corner (rounded to `float`), refined by the same `cv::cornerSubPix` unless
//!   `DisableCheckerboardRefinement`.
//! - The corners are stored (`detectedPoints_ALL`) only when there are `H x V` of them.
//!   Then, with fewer than 6 stored: "Could not detect chessboard. Please try the manual
//!   mode" and manual corner mode on. Otherwise `sortGridByReference(click 1)`: the corners
//!   keep their order when the click is nearer the first than the last (squared distance,
//!   strict), else are reversed; every one an inlier. Then, once the camera has at least two
//!   frames with corners, its calibration over all frames (`Calibration(camera, planar)`:
//!   no intrinsic guess, every frame with detected points), `setDialog` and every trial
//!   flagged.
//! - No pose is carried to other cameras' frames for a board (`calibrateOtherFrames` skips
//!   it).
//!
//! # Defects fixed (XMALab 3 mode keeps them)
//!
//! Ledger A180 (the undistorted points after sorting), A181 (corners stored although the
//! board was not found), A182 (frames counted on the active camera), A183 (the manual
//! homography ignores an inverted axis), A184 (manual mode with fewer than four clicks),
//! A185 (a refused calibration keeps the corners it stored, switch of A135), A186 (sorting
//! one corner never ends), A189 (cameras that number the board differently: matched after
//! each calibration, [`super::numbering`]).

use std::sync::Arc;

use xma_model::{CalibrationObject, Point2};
use xma_vision::image::ImageU8;

use super::all_frames::{
    calibration_job, calibration_jobs, flag_trials, flagged_cameras, n_frames, object_points,
    set_dialog,
};
use super::{held, images};
use crate::document::Document;
use crate::edit::{EditError, EditOutcome, EditResult};
pub use xma_vision::chessboard::{self, ChessboardCorners, ChessboardFlags};

/// The button's error with no click in automatic mode.
pub const NEED_UPPER_LEFT_CORNER: &str =
    "You need to select the interior upper left corner of the chessboard";
/// Fewer than 6 corners stored after the detection.
pub const COULD_NOT_DETECT: &str = "Could not detect chessboard. Please try the manual mode";
/// Ledger A184: manual corner mode with fewer than four corners clicked (default mode).
/// Ledger A202: four clicks that do not span a proper quadrilateral (default mode).
pub const NEED_PROPER_CORNERS: &str = "The four corners do not form the board: click the four interior corners of the checkerboard in order around the board, starting with the upper left corner";
pub const NEED_FOUR_CORNERS: &str = "You need to select the four interior corners of the checkerboard, starting with the upper left corner";

/// A replacement for `cv::findChessboardCorners` (8-bit image, corners per row, corners per
/// column): what [`crate::Session::chessboard_detector`] holds for tests that supply the
/// corners of synthetic images themselves.
#[derive(Clone)]
pub struct ChessboardDetector(pub Arc<ChessboardDetectorFn>);

/// The function a [`ChessboardDetector`] holds.
pub type ChessboardDetectorFn = dyn Fn(&ImageU8, usize, usize) -> ChessboardCorners + Send + Sync;

impl PartialEq for ChessboardDetector {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for ChessboardDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChessboardDetector")
    }
}

/// The board's size (corners per row, per column), when the project's object is one.
pub(crate) fn board(doc: &Document) -> Option<(usize, usize)> {
    match &doc.project().calibration_object {
        Some(CalibrationObject::Checkerboard {
            horizontal_squares,
            vertical_squares,
            ..
        }) => Some((*horizontal_squares as usize, *vertical_squares as usize)),
        _ => None,
    }
}

/// `cv::findChessboardCorners(image, (cols, rows), CALIB_CB_ADAPTIVE_THRESH)`: the
/// session's replacement when set (tests), else [`chessboard::find_chessboard_corners`].
fn find_corners(doc: &Document, image: &ImageU8, cols: usize, rows: usize) -> ChessboardCorners {
    if let Some(d) = &doc.session().chessboard_detector {
        return (d.0)(image, cols, rows);
    }
    chessboard::find_chessboard_corners(image, cols, rows, ChessboardFlags::XMALAB)
}

fn to_points(c: &[[f32; 2]]) -> Vec<Point2> {
    c.iter()
        .map(|p| Point2::new(f64::from(p[0]), f64::from(p[1])))
        .collect()
}

/// What the detection of calibration image `image` of `camera` gives: the corners in
/// detection order (`tmpPoints`), and whether the board counts as found (always for the
/// manual corners). Where XMALab's `cornerSubPix` throws (a corner outside the image), the
/// worker thread ends with no corners.
///
/// Ported from processing/CheckerboardDetection.cpp, CheckerboardDetection::detectCorner_thread,
/// with `xma_vision::chessboard` (`find_chessboard_corners`, `refine_corners_xmalab`,
/// `manual_homography`, `project_with_homography`). The image is the 8-bit copy in both
/// modes (ledger A187).
fn detect(
    doc: &Document,
    camera: usize,
    image: usize,
    manual: bool,
) -> Result<(Vec<Point2>, bool), EditError> {
    let (h, v) = board(doc).ok_or_else(|| EditError::Invalid("not a checkerboard".to_owned()))?;
    let frame = images::calibration_frame(doc, camera, image).map_err(EditError::Rejected)?;
    let im = crate::tracking::jobs::algorithm_image(&frame);
    if !manual {
        // detect_corners_xmalab, with the session's replacement for the search.
        let mut found = find_corners(doc, &im, h, v);
        if found.found && chessboard::refine_corners_xmalab(&im, &mut found.corners).is_err() {
            return Ok((Vec::new(), false));
        }
        return Ok((to_points(&found.corners), found.found));
    }
    let pts = object_points(doc);
    let s = doc.session();
    let clicks: [[f64; 2]; 4] = std::array::from_fn(|i| {
        let (x, y) = s.references[i].position.unwrap_or((-1.0, -1.0));
        [x, y]
    });
    // Ledger A202: a board with a single row or column of corners has no four corners to
    // click (no homography can be made from them): a failed detection. Otherwise the four
    // clicks must span a proper quadrilateral.
    if !doc.compat().checkerboard_degenerate_clicks_accepted && (h < 2 || v < 2) {
        return Ok((Vec::new(), true));
    }
    if !doc.compat().checkerboard_degenerate_clicks_accepted && !proper_quadrilateral(&clicks, h, v)
    {
        return Err(EditError::Rejected(NEED_PROPER_CORNERS.to_owned()));
    }
    let hm = if doc.compat().checkerboard_homography_ignores_inverted_axis {
        let size = match &doc.project().calibration_object {
            Some(CalibrationObject::Checkerboard { square_size, .. }) => *square_size,
            _ => 0.0,
        };
        // The grid without the inverted axis (`CheckerboardDetection`'s constructor).
        chessboard::manual_homography(h, v, size, &clicks)
    } else {
        // Ledger A183: the board's own outer corners, as `cv::Point2f`.
        #[allow(clippy::cast_possible_truncation)]
        let corner = |i: usize| {
            pts.get(i).map_or([0.0, 0.0], |p| {
                [f64::from(p.x as f32), f64::from(p.y as f32)]
            })
        };
        let grid = [
            corner(0),
            corner(h.saturating_sub(1)),
            corner((h * v).saturating_sub(1)),
            corner((v.saturating_sub(1)) * h),
        ];
        #[allow(clippy::cast_possible_truncation)]
        let clicks_f: Vec<[f64; 2]> = clicks
            .iter()
            .map(|c| [f64::from(c[0] as f32), f64::from(c[1] as f32)])
            .collect();
        if clicks_f.iter().flatten().any(|v| !v.is_finite()) {
            None
        } else {
            xma_vision::calib3d::homography::find_homography(&grid, &clicks_f)
        }
    };
    // An empty homography (the points do not spread) is read out of range by the C++;
    // here no corner is predicted (the detection fails).
    let Some(hm) = hm else {
        return Ok((Vec::new(), true));
    };
    let obj: Vec<[f64; 2]> = pts.iter().map(|p| [p.x, p.y]).collect();
    let mut predicted = chessboard::project_with_homography(&hm, &obj);
    // Ledger A202: the predicted corners must be finite and inside the image.
    if !doc.compat().checkerboard_degenerate_clicks_accepted
        && !inside_image(&predicted, im.width(), im.height())
    {
        return Ok((Vec::new(), true));
    }
    if !doc.settings().disable_checkerboard_refinement
        && chessboard::refine_corners_xmalab(&im, &mut predicted).is_err()
    {
        return Ok((Vec::new(), true));
    }
    Ok((to_points(&predicted), true))
}

/// Ledger A202: whether the four clicks (the board's corners `(0, 0)`, `(H-1, 0)`,
/// `(H-1, V-1)`, `(0, V-1)` in that order) span a proper quadrilateral: finite, convex with
/// every turn in the same direction (either winding: a mirrored image reverses it) and a
/// non-zero area, and each side at least 2 pixels per square it spans (2 (H-1) for the
/// first and third, 2 (V-1) for the second and fourth). Inner corners less than 2 pixels
/// apart cannot be placed or refined (`cornerSubPix` works in an 11 x 11 window), and any
/// image in which a board can be clicked has far more pixels per square.
fn proper_quadrilateral(c: &[[f64; 2]; 4], h: usize, v: usize) -> bool {
    if c.iter().flatten().any(|x| !x.is_finite()) || h < 2 || v < 2 {
        return false;
    }
    #[allow(clippy::cast_precision_loss)]
    let min = [2.0 * (h - 1) as f64, 2.0 * (v - 1) as f64];
    let mut sign = 0.0;
    for i in 0..4 {
        let (a, b, n) = (c[i], c[(i + 1) % 4], c[(i + 2) % 4]);
        let e = [b[0] - a[0], b[1] - a[1]];
        if (e[0] * e[0] + e[1] * e[1]).sqrt() < min[i % 2] {
            return false;
        }
        let f = [n[0] - b[0], n[1] - b[1]];
        let cross = e[0] * f[1] - e[1] * f[0];
        if cross == 0.0 || (sign != 0.0 && cross.signum() != sign) {
            return false;
        }
        sign = cross.signum();
    }
    true
}

/// Ledger A202: every predicted corner finite and inside the image (pixel centres at whole
/// numbers, so the image covers -0.5 to `width - 0.5`).
fn inside_image(p: &[[f32; 2]], width: usize, height: usize) -> bool {
    #[allow(clippy::cast_precision_loss)]
    let (w, h) = (width as f32 - 0.5, height as f32 - 0.5);
    p.iter().all(|q| {
        q[0].is_finite()
            && q[1].is_finite()
            && q[0] >= -0.5
            && q[0] <= w
            && q[1] >= -0.5
            && q[1] <= h
    })
}

/// `CalibrationImage::sortGridByReference(x, y)` on the stored corners: kept in order when
/// the click is nearer the first corner than the last (squared distance, strict, from
/// 10000000), else reversed; every point an inlier. Ledger A180: the undistorted points are
/// the corners undistorted (XMALab 3 mode: the distorted corners, and the error lists left
/// as they were). Ledger A186: with fewer than two corners the order is kept (the C++ loop
/// never ends with one).
///
/// Ported from core/CalibrationImage.cpp, CalibrationImage::sortGridByReference.
fn sort_grid_by_reference(doc: &mut Document, camera: usize, image: usize, x: f64, y: f64) {
    let raw = doc.compat().checkerboard_undistorted_points_raw;
    held::with_image(doc, camera, image, |im, h, model| {
        let all = &im.detected_all;
        let n = all.len();
        let mut idx = 0;
        if n >= 2 {
            let mut dist = 10_000_000.0;
            for i in [0, n - 1] {
                let d = (x - all[i].x) * (x - all[i].x) + (y - all[i].y) * (y - all[i].y);
                if d < dist {
                    dist = d;
                    idx = i;
                }
            }
        }
        let mut sorted = all.clone();
        if n >= 2 && idx == n - 1 {
            sorted.reverse();
        }
        im.inlier = vec![1; sorted.len()];
        im.detected = sorted.clone();
        if raw {
            h.detected_undistorted = sorted.clone();
            h.projected = sorted.clone();
            h.projected_undistorted = sorted;
        } else {
            h.detected_undistorted = sorted.iter().map(|&p| held::undistort(model, p)).collect();
            h.projected = sorted;
            h.projected_undistorted = h.detected_undistorted.clone();
            held::compute_error(im, h);
        }
    });
}

/// The frames of `camera` with stored corners (`getDetectedPointsAll().size() > 0`).
pub(crate) fn frames_with_corners(doc: &Document, camera: usize) -> usize {
    (0..n_frames(doc, camera))
        .filter(|&m| {
            !doc.project().cameras[camera].calibration_images[m]
                .detected_all
                .is_empty()
        })
        .count()
}

/// `runCalibrationCameraAllFrames` for a checkerboard: for every camera flagged for
/// recalibration with at least two frames with corners, its calibration over all frames;
/// when one ran, `setDialog` and every trial flagged. Ledger A182: XMALab 3 counts the
/// frames on the active camera.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, runCalibrationCameraAllFrames (the
/// checkerboard branch) and runCalibrationCameraAllFramesFinished.
pub(crate) fn run_all_frames(doc: &mut Document) -> Vec<String> {
    let flagged = flagged_cameras(doc);
    let active = doc.session().active_camera;
    let on_active = doc.compat().checkerboard_frames_counted_on_active_camera
        && active < doc.project().cameras.len();
    let run: Vec<usize> = flagged
        .into_iter()
        .filter(|&j| frames_with_corners(doc, if on_active { active } else { j }) >= 2)
        .collect();
    // The numbering compared once, after every camera (review 12).
    let messages = calibration_jobs(doc, &run);
    if !run.is_empty() {
        set_dialog(doc);
        flag_trials(doc);
    }
    messages
}

/// C40, C41: "compute calibration" on calibration image `image` of `camera` (the active
/// ones), not calibrated, manual mode off: the button's check of the clicks, the detection,
/// and `runCalibration`.
///
/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_pushButton_clicked (checkerboard),
/// processing/CheckerboardDetection.cpp, detectCorner_threadFinished, and
/// ui/WizardCalibrationCubeFrame.cpp, runCalibration (checkerboard).
pub(crate) fn compute_uncalibrated(doc: &mut Document, camera: usize, image: usize) -> EditResult {
    let (h, v) = board(doc).ok_or_else(|| EditError::Invalid("not a checkerboard".to_owned()))?;
    let manual = doc.session().checkerboard_manual;
    let refs = doc.session().references;
    let set =
        |r: &crate::session::ReferenceClick| r.position.is_some_and(|(x, y)| x >= 0.0 && y >= 0.0);
    let count = refs.iter().filter(|r| set(r)).count();
    if count < 1 {
        return Err(EditError::Rejected(NEED_UPPER_LEFT_CORNER.to_owned()));
    }
    // Ledger A184.
    if manual && count < 4 && !doc.compat().checkerboard_manual_with_unset_clicks {
        return Err(EditError::Rejected(NEED_FOUR_CORNERS.to_owned()));
    }
    let (corners, found) = detect(doc, camera, image, manual)?;
    // detectCorner_threadFinished: stored only when there are as many as the board has
    // (ledger A181: and, in the default mode, only when the board was found).
    let keep =
        corners.len() == h * v && (found || doc.compat().checkerboard_unconfirmed_corners_stored);
    // Ledger A203: in the default mode a detection that gives no board is a failure, also
    // when the frame holds corners from an earlier detection (XMALab 3 sorts and calibrates
    // those).
    let stored = if keep {
        corners.len()
    } else if !doc.compat().checkerboard_failed_detection_reuses_corners {
        0
    } else {
        doc.project().cameras[camera].calibration_images[image]
            .detected_all
            .len()
    };
    // runCalibration.
    if stored < 6 {
        doc.session_mut().checkerboard_manual = true;
        // Ledger A185: XMALab 3 has stored the corners (a board of fewer than 6) before it
        // refuses.
        if keep && doc.compat().refused_calibration_keeps_beads {
            Arc::make_mut(&mut doc.project.cameras[camera]).calibration_images[image]
                .detected_all = corners;
        }
        set_dialog(doc);
        return Ok(EditOutcome::refused(COULD_NOT_DETECT));
    }
    if keep {
        Arc::make_mut(&mut doc.project.cameras[camera]).calibration_images[image].detected_all =
            corners;
    }
    let (x, y) = refs[0].position.unwrap_or((-1.0, -1.0));
    sort_grid_by_reference(doc, camera, image, x, y);
    let mut outcome = EditOutcome::none();
    if frames_with_corners(doc, camera) >= 2 {
        outcome.messages.extend(calibration_job(doc, camera));
        set_dialog(doc);
        flag_trials(doc);
    }
    Ok(outcome)
}

impl Document {
    /// C15, C40, C41: a click in the Calibration workspace, manual mode off, on a calibration
    /// image that is not calibrated, once undistorted (the caller checks those): for a
    /// checkerboard in automatic mode the click is the interior upper-left corner
    /// (reference 1); otherwise (a cube, or the manual corners) the checked reference takes
    /// it and the next one is checked. Returns true when "compute calibration" is to run
    /// next (`AutoCalibAfterReference` with the board's corner or the fourth reference).
    /// Session only: no undo step.
    ///
    /// Ported from ui/WizardCalibrationCubeFrame.cpp, addCalibrationReference.
    pub fn click_calibration_reference(&mut self, x: f64, y: f64) -> bool {
        let auto = self.session().wizard.auto_calib_after_reference;
        if board(self).is_some() && !self.session().checkerboard_manual {
            self.session_mut().references[0].position = Some((x, y));
            return auto;
        }
        self.session_mut().click_reference(x, y) && auto
    }

    /// True when the project's object is a checkerboard and its corners are clicked by hand
    /// (`checkerboardManual`): the setting `DisableCheckerboardDetection`, or a detection
    /// that failed since the last change of camera, calibration frame or workspace.
    #[must_use]
    pub fn checkerboard_manual(&self) -> bool {
        board(self).is_some() && self.session().checkerboard_manual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create::{CalibrationTarget, NewCameraSpec, NewProjectSpec};
    use xma_model::CppCompat;

    /// A two-image light camera with a board, and a lens model (k1 = -0.2) so that the
    /// undistorted points differ from the distorted ones.
    fn doc(compat: CppCompat) -> Document {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("xma-checker-unit-{}-{n}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let images: Vec<std::path::PathBuf> = (0..2)
            .map(|i| {
                let p = dir.join(format!("u{i}.png"));
                image::GrayImage::from_fn(64, 48, |x, y| image::Luma([((x + y + i) % 256) as u8]))
                    .save(&p)
                    .expect("png");
                p
            })
            .collect();
        let spec = NewProjectSpec {
            cameras: vec![NewCameraSpec {
                name: "Camera 1".into(),
                images,
                light_camera: true,
                grid: None,
                flipped: false,
            }],
            target: CalibrationTarget::Checkerboard {
                horizontal_squares: 3,
                vertical_squares: 2,
                square_size: 1.0,
            },
        };
        let options = crate::OpenOptions {
            compat,
            ..crate::OpenOptions::default()
        };
        let mut doc = Document::new_project(&spec, &options).expect("created");
        let cam = Arc::make_mut(&mut doc.project.cameras[0]);
        cam.camera_matrix =
            xma_model::Matrix3::new(50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0);
        cam.distortion = [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        cam.has_model_distortion = true;
        super::super::all_frames::refresh_intrinsics(&mut doc, 0);
        let corners: Vec<Point2> = (0..6)
            .map(|i| Point2::new(10.0 + 8.0 * f64::from(i % 3), 10.0 + 9.0 * f64::from(i / 3)))
            .collect();
        Arc::make_mut(&mut doc.project.cameras[0]).calibration_images[0].detected_all = corners;
        doc
    }

    #[test]
    fn a180_sorted_corners_undistorted() {
        let mut d = doc(CppCompat::FIXED);
        sort_grid_by_reference(&mut d, 0, 0, 9.0, 9.0);
        let im = d.project().cameras[0].calibration_images[0].clone();
        let h = d.camera_held_points(0, 0).expect("held").clone();
        let model = d.camera(0).model.clone();
        for (p, u) in im.detected.iter().zip(&h.detected_undistorted) {
            assert_eq!(*u, held::undistort(&model, *p));
            assert!((u.x - p.x).abs() > 1e-3, "the lens model moves it");
        }
        assert_eq!(h.error_undistorted, vec![0.0; 6]);
        // XMALab 3: the distorted corners as the undistorted ones.
        let mut d = doc(CppCompat::XMALAB3);
        sort_grid_by_reference(&mut d, 0, 0, 9.0, 9.0);
        let im = d.project().cameras[0].calibration_images[0].clone();
        let h = d.camera_held_points(0, 0).expect("held").clone();
        assert_eq!(h.detected_undistorted, im.detected);
        assert_eq!(h.projected_undistorted, im.detected);
    }

    #[test]
    fn a186_one_corner_is_kept() {
        let mut d = doc(CppCompat::FIXED);
        Arc::make_mut(&mut d.project.cameras[0]).calibration_images[0].detected_all =
            vec![Point2::new(3.0, 4.0)];
        sort_grid_by_reference(&mut d, 0, 0, 100.0, 100.0);
        let im = &d.project().cameras[0].calibration_images[0];
        assert_eq!(im.detected, vec![Point2::new(3.0, 4.0)]);
        assert_eq!(im.inlier, vec![1]);
    }

    #[test]
    fn sorting_reverses_toward_the_last_corner() {
        let mut d = doc(CppCompat::FIXED);
        sort_grid_by_reference(&mut d, 0, 0, 27.0, 20.0);
        let im = &d.project().cameras[0].calibration_images[0];
        assert_eq!(im.detected[0], Point2::new(26.0, 19.0));
        // Far from both (squared distance of 1e7 or more): the order is kept.
        let mut d = doc(CppCompat::FIXED);
        sort_grid_by_reference(&mut d, 0, 0, 5000.0, 5000.0);
        let im = &d.project().cameras[0].calibration_images[0];
        assert_eq!(im.detected[0], Point2::new(10.0, 10.0));
    }
}
