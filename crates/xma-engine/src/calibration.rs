//! The undistortion and calibration actions: what the Undistortion and Calibration
//! workspaces, the external-calibration frame and the optimisation dialog do to a project.
//!
//! Work packages "engcal" (undistortion, cube calibration, the edits of a calibrated image:
//! submodules `undistortion`, `cube`, `all_frames`, `image_edits`, `held`, `images`),
//! "engopt" (multi-camera optimisation, external calibration, MayaCam import: [`optimize`],
//! [`external`]) of phase 4, and "p6checker" (checkerboard calibration, C40 and C41:
//! submodule `checkerboard`, entered from `ComputeCalibration` and the calibration over all
//! frames; [`Document::click_calibration_reference`] for the clicks).
//! The names, fields and signatures here are fixed so that the engine and the window can be
//! written against them. What each action does in the C++ is in `docs/cpp-calibration.md`
//! (the C numbers below are its action numbers).
//!
//! # Where the state lives
//!
//! - What a project stores (grids, calibration images, camera matrices) is in the model and
//!   is changed only by a [`CalibrationEdit`], through [`crate::Edit::Calibration`]. An edit
//!   that changes the project, the derived state below or the session values undo restores
//!   is one undo step; one that fails or changes none of them records nothing (ledger A159).
//! - What XMALab keeps outside the project and never saves is in [`crate::Session`]: the
//!   reference clicks, the custom reference indices, manual mode, the interface settings
//!   these actions read, the random generator of the XMALab 3 mode, the calibration-frame
//!   count, and the flags that say a grid or camera has changes not yet recomputed. Those
//!   that an edit writes and a later edit reads go back with undo ([`crate::session`] lists
//!   them).
//! - What XMALab computes from the project and holds without saving is in
//!   [`crate::DerivedCamera`]: the camera model (grid fit, camera matrix, lens model), the
//!   grid's lookup table, and per calibration image the undistorted and projected points and
//!   their errors ([`HeldPoints`]). It goes back with undo.
//!
//! # Errors and messages
//!
//! - [`EditError::Rejected`] carries XMALab's error-dialog sentence, and the document is
//!   unchanged: "You need to at least select 3 reference points", "Not enough points found
//!   to run calibration" (in the XMALab 3 mode this one is an [`crate::EditOutcome::refused`]
//!   after the beads were stored, ledger A135), "You need to at least select 7 points for a
//!   manual calibration", and the reasons a grid cannot be matched or an image read.
//! - [`crate::EditOutcome::messages`] carries the sentences XMALab shows after a
//!   calibration that went through: the frames that could not be calibrated ("There were
//!   problems during Calibration with the following frames: ..."), the frames that could not
//!   be set from other cameras, both left out with `HideWarningsDuringCalibration`
//!   ([`crate::WizardSettings`]), and, in the default mode, a calibration over all frames
//!   that failed (ledger A132).
//!
//! # Long computations
//!
//! "compute undistortion", "compute calibration" and "optimize camera setup" take from a
//! fraction of a second to about a minute. They are ordinary edits: a caller that must
//! stay responsive applies the edit to a copy of the document on another thread and puts
//! the copy in place when it is done ([`CalibrationEdit::progress_text`] is the text
//! XMALab shows meanwhile). XMALab offers no cancel for them.
//!
//! # Coordinates
//!
//! Positions are image coordinates as `GLCameraView::mousePressEvent` computes them:
//! 0-based pixels, of the distorted image unless a field says the view shows the
//! undistorted one.

use std::path::Path;

pub mod external;
pub mod optimize;

use crate::document::Document;
use crate::edit::EditResult;

mod all_frames;
mod checkerboard;
mod cube;
mod held;
mod image_edits;
mod images;
pub mod numbering;
mod undistortion;
pub(crate) use all_frames::sync_manual_lists;

pub use checkerboard::{
    COULD_NOT_DETECT, ChessboardCorners, ChessboardDetector, ChessboardFlags, NEED_FOUR_CORNERS,
    NEED_PROPER_CORNERS, NEED_UPPER_LEFT_CORNER,
};
pub use cube::{NEED_SEVEN_POINTS, NEED_THREE_REFERENCES, NOT_ENOUGH_POINTS};
pub use held::HeldPoints;

/// Which of the undistortion wizard's point lists a click refers to (its points combo: 2,
/// 3 or 4; C6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridPointList {
    /// "grid points (distorted)".
    Distorted,
    /// "grid points (undistorted)".
    Undistorted,
    /// "reference points".
    References,
}

/// What the external-calibration frame holds for one camera (C43, C44): the values of a
/// MayaCam 2.0 file.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalCalibration {
    pub width: u32,
    pub height: u32,
    /// Row by row.
    pub camera_matrix: [[f64; 3]; 3],
    /// Row by row, world to camera.
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    /// The eight lens coefficients (k1, k2, p1, p2, k3, k4, k5, k6) when "model
    /// undistortion" is checked.
    pub distortion: Option<[f64; 8]>,
}

/// One undistortion or calibration action, applied with
/// `Document::apply(&Edit::Calibration(..))`.
#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationEdit {
    // ----- Undistortion workspace ---------------------------------------------------------
    /// C1 and C2: the wizard's "compute undistortion" / "recompute undistortion" button,
    /// according to [`crate::Session::undistorted`].
    ComputeUndistortion,
    /// C6: a click in "toggle outlier" mode at (`x`, `y`) of `camera`, with the point list
    /// the view shows.
    ToggleGridOutlier {
        camera: usize,
        list: GridPointList,
        x: f64,
        y: f64,
    },
    /// C7: a click in "set center" mode on the distorted grid image.
    SetGridCenter { camera: usize, x: f64, y: f64 },
    /// C8: "remove outlier at border" with its two thresholds.
    RemoveGridOutliers { border: i32, circle: i32 },
    /// The "Yes" of the pending-changes question (C11 in the Undistortion workspace, C39
    /// in the Calibration workspace): recomputes what has pending changes.
    RecomputePending,

    // ----- Calibration workspace ----------------------------------------------------------
    /// C17, C18, C22, C29: the wizard's "compute calibration" / "recompute calibration"
    /// button for the active camera and calibration image, with the session's reference
    /// clicks or manual points; and everything that follows it (poses from other cameras,
    /// the calibration over all frames).
    ComputeCalibration { camera: usize, image: usize },
    /// C23: a double click on a calibrated image; `undistorted` when the view shows the
    /// undistorted image.
    ToggleInlier {
        camera: usize,
        image: usize,
        x: f64,
        y: f64,
        undistorted: bool,
    },
    /// C24: Ctrl + click on a calibrated image: the nearest projected point's detected
    /// position becomes the click.
    MovePoint {
        camera: usize,
        image: usize,
        x: f64,
        y: f64,
        undistorted: bool,
    },
    /// C27: manual mode, a click setting calibration-object point `point`.
    SetManualPoint {
        camera: usize,
        image: usize,
        point: usize,
        x: f64,
        y: f64,
        undistorted: bool,
    },
    /// C28: manual mode, the check boxes: the inlier state of every calibration-object
    /// point.
    SetManualInliers {
        camera: usize,
        image: usize,
        inliers: Vec<bool>,
    },
    /// C32: "Reset Frame".
    ResetFrame { camera: usize, image: usize },
    /// C33: "Reset Camera".
    ResetCamera { camera: usize },
    /// C34: "Delete Frame" (for all cameras).
    DeleteFrame { image: usize },
    /// C35: "use camera distortion model".
    SetLensModel { camera: usize, on: bool },
    /// C36: unchecking "is optimized", confirmed.
    RemoveOptimization { camera: usize, image: usize },
    /// C37, C42: "optimize camera setup": `method` 0, 1 or 2 (the dialog's radio buttons),
    /// the number of iterations and the scale factor for the initial lambda. Any other
    /// method is refused (ledger A146).
    Optimize {
        method: i32,
        iterations: usize,
        initial: f64,
    },

    // ----- External calibration -----------------------------------------------------------
    /// C44: "APPLY CHANGES" of the external-calibration frame of `camera`.
    ApplyExternal {
        camera: usize,
        calibration: Box<ExternalCalibration>,
    },
}

impl CalibrationEdit {
    /// Ledger F51: the first number of the edit that is NaN or infinite, named for a
    /// sentence ([`crate::Edit::non_finite_input`]); `None` when every number is finite.
    /// An external calibration's values are checked by the edit itself.
    #[must_use]
    pub fn non_finite_input(&self) -> Option<&'static str> {
        let bad = |v: &[f64]| v.iter().any(|x| !x.is_finite());
        match self {
            CalibrationEdit::ToggleGridOutlier { x, y, .. }
            | CalibrationEdit::SetGridCenter { x, y, .. }
            | CalibrationEdit::ToggleInlier { x, y, .. }
            | CalibrationEdit::MovePoint { x, y, .. }
            | CalibrationEdit::SetManualPoint { x, y, .. }
                if bad(&[*x, *y]) =>
            {
                Some("a point coordinate")
            }
            // Optimize refuses a non-finite initial value itself (`Rejected`).
            // ApplyExternal refuses non-finite values itself (`Rejected`, ledger A150).
            _ => None,
        }
    }

    /// The undo step's name.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            CalibrationEdit::ComputeUndistortion => "Compute Undistortion",
            CalibrationEdit::ToggleGridOutlier { .. } => "Toggle Outlier",
            CalibrationEdit::SetGridCenter { .. } => "Set Center",
            CalibrationEdit::RemoveGridOutliers { .. } => "Remove Outliers",
            CalibrationEdit::RecomputePending => "Recompute",
            CalibrationEdit::ComputeCalibration { .. } => "Compute Calibration",
            CalibrationEdit::ToggleInlier { .. } => "Toggle Inlier",
            CalibrationEdit::MovePoint { .. } | CalibrationEdit::SetManualPoint { .. } => {
                "Set Calibration Point"
            }
            CalibrationEdit::SetManualInliers { .. } => "Set Inliers",
            CalibrationEdit::ResetFrame { .. } => "Reset Frame",
            CalibrationEdit::ResetCamera { .. } => "Reset Camera",
            CalibrationEdit::DeleteFrame { .. } => "Delete Frame",
            CalibrationEdit::SetLensModel { .. } => "Camera Distortion Model",
            CalibrationEdit::RemoveOptimization { .. } => "Remove Optimization",
            CalibrationEdit::Optimize { .. } => "Optimize Camera Setup",
            CalibrationEdit::ApplyExternal { .. } => "External Calibration",
        }
    }

    /// The edit's name in scripts (`xma edit`, `xma parity`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            CalibrationEdit::ComputeUndistortion => "ComputeUndistortion",
            CalibrationEdit::ToggleGridOutlier { .. } => "ToggleGridOutlier",
            CalibrationEdit::SetGridCenter { .. } => "SetGridCenter",
            CalibrationEdit::RemoveGridOutliers { .. } => "RemoveGridOutliers",
            CalibrationEdit::RecomputePending => "RecomputePending",
            CalibrationEdit::ComputeCalibration { .. } => "ComputeCalibration",
            CalibrationEdit::ToggleInlier { .. } => "ToggleInlier",
            CalibrationEdit::MovePoint { .. } => "MovePoint",
            CalibrationEdit::SetManualPoint { .. } => "SetManualPoint",
            CalibrationEdit::SetManualInliers { .. } => "SetManualInliers",
            CalibrationEdit::ResetFrame { .. } => "ResetFrame",
            CalibrationEdit::ResetCamera { .. } => "ResetCamera",
            CalibrationEdit::DeleteFrame { .. } => "DeleteFrame",
            CalibrationEdit::SetLensModel { .. } => "SetLensModel",
            CalibrationEdit::RemoveOptimization { .. } => "RemoveOptimization",
            CalibrationEdit::Optimize { .. } => "Optimize",
            CalibrationEdit::ApplyExternal { .. } => "ApplyExternal",
        }
    }

    /// For an edit that can take long, the text XMALab's progress dialog shows first
    /// (`docs/cpp-calibration.md` section 9); `None` for an edit that is immediate.
    #[must_use]
    pub fn progress_text(&self) -> Option<&'static str> {
        match self {
            CalibrationEdit::ComputeUndistortion | CalibrationEdit::ComputeCalibration { .. } => {
                Some("Detect Points")
            }
            CalibrationEdit::RecomputePending => Some("Local Undistortion"),
            CalibrationEdit::ResetFrame { .. }
            | CalibrationEdit::DeleteFrame { .. }
            | CalibrationEdit::RemoveOptimization { .. } => {
                Some("Calibrate camera internal parameters and pose for all frames")
            }
            CalibrationEdit::Optimize { .. } => Some("Optimize camera setup"),
            _ => None,
        }
    }
}

/// Applies a calibration edit (`Edit::Calibration`).
///
/// Ledger A164: the point lists manual mode gave images outside an edit
/// ([`crate::Session::manual_lists_pending`]) are created first, as part of this edit's step,
/// so that the edit reads what XMALab's would. When the edit fails, or succeeds without
/// changing anything else, they are not created after all (the images stay pending), so an
/// edit that changes nothing still records nothing.
pub(crate) fn apply(doc: &mut Document, edit: &CalibrationEdit) -> EditResult {
    if doc.session().manual_lists_pending.is_empty() {
        return apply_edit(doc, edit);
    }
    let before = (doc.project.clone(), doc.computed_state());
    let pending = all_frames::create_pending_manual_lists(doc);
    let created = (doc.project.clone(), doc.computed_state());
    let result = apply_edit(doc, edit);
    let unchanged = result.is_ok()
        && crate::history::same_project(&created.0, &doc.project)
        && created.1.same_as(&doc.computed_state());
    if unchanged {
        doc.project = before.0;
        doc.set_computed_state(before.1);
    }
    if unchanged || result.is_err() {
        doc.session_mut().manual_lists_pending.extend(pending);
    }
    result
}

/// [`apply`] after the pending manual lists are created.
fn apply_edit(doc: &mut Document, edit: &CalibrationEdit) -> EditResult {
    match edit {
        CalibrationEdit::ComputeUndistortion => undistortion::compute_undistortion(doc),
        CalibrationEdit::ToggleGridOutlier { camera, list, x, y } => {
            undistortion::toggle_grid_outlier(doc, *camera, *list, *x, *y)
        }
        CalibrationEdit::SetGridCenter { camera, x, y } => {
            undistortion::set_grid_center(doc, *camera, *x, *y)
        }
        CalibrationEdit::RemoveGridOutliers { border, circle } => {
            undistortion::remove_grid_outliers(doc, *border, *circle)
        }
        CalibrationEdit::RecomputePending => image_edits::recompute_pending(doc),
        CalibrationEdit::ComputeCalibration { camera, image } => {
            cube::compute_calibration(doc, *camera, *image)
        }
        CalibrationEdit::ToggleInlier {
            camera,
            image,
            x,
            y,
            undistorted,
        } => image_edits::toggle_inlier(doc, *camera, *image, *x, *y, *undistorted),
        CalibrationEdit::MovePoint {
            camera,
            image,
            x,
            y,
            undistorted,
        } => image_edits::move_point(doc, *camera, *image, *x, *y, *undistorted),
        CalibrationEdit::SetManualPoint {
            camera,
            image,
            point,
            x,
            y,
            undistorted,
        } => image_edits::set_manual_point(doc, *camera, *image, *point, *x, *y, *undistorted),
        CalibrationEdit::SetManualInliers {
            camera,
            image,
            inliers,
        } => image_edits::set_manual_inliers(doc, *camera, *image, inliers),
        CalibrationEdit::ResetFrame { camera, image } => {
            image_edits::reset_frame(doc, *camera, *image)
        }
        CalibrationEdit::ResetCamera { camera } => image_edits::reset_camera(doc, *camera),
        CalibrationEdit::DeleteFrame { image } => image_edits::delete_frame(doc, *image),
        CalibrationEdit::SetLensModel { camera, on } => {
            image_edits::set_lens_model(doc, *camera, *on)
        }
        CalibrationEdit::RemoveOptimization { camera, image } => {
            image_edits::remove_optimization(doc, *camera, *image)
        }
        CalibrationEdit::Optimize {
            method,
            iterations,
            initial,
        } => optimize::apply_optimize(doc, *method, *iterations, *initial),
        CalibrationEdit::ApplyExternal {
            camera,
            calibration,
        } => external::apply_external(doc, *camera, calibration),
    }
}

/// C43, "Load MayaCam 2.0": the values of a MayaCam 2.0 file, for the external-calibration
/// frame to show (nothing is applied until [`CalibrationEdit::ApplyExternal`]).
///
/// # Errors
/// A sentence for the user when the file cannot be read or is not a MayaCam 2.0 file.
pub fn read_mayacam2(path: &Path) -> Result<ExternalCalibration, String> {
    external::read_mayacam2(path)
}

impl Document {
    /// What the calibration wizard's `setDialog` does, for the window to call where XMALab
    /// calls it (a change of active camera, calibration frame or workspace, after manual
    /// mode is switched): [`crate::Session::reload_references`] and, in manual mode, the
    /// active image given its point lists when it has fewer than the calibration object
    /// (`reloadManualPoints`). Not an undo step and no change of the project (those lists
    /// are recorded in [`crate::Session::manual_lists_pending`] and created by the next
    /// calibration edit, ledger A164); nothing for a project without an internal
    /// calibration or before the undistortion. The calibration edits call it themselves
    /// where XMALab does.
    ///
    /// Ported from ui/WizardCalibrationCubeFrame.cpp, setDialog.
    pub fn calibration_dialog(&mut self) {
        all_frames::set_dialog(self);
    }

    /// A click on a camera view or the calibration frame bar: the active camera and
    /// calibration image (`State::changeActiveCamera`, `changeActiveFrameCalibration`), with,
    /// when either changes, the checkerboard's manual corner mode reset from the setting
    /// `DisableCheckerboardDetection` ([`crate::Session::checkerboard_manual`]) and
    /// [`Document::calibration_dialog`].
    pub fn set_active_calibration_image(&mut self, camera: usize, image: usize) {
        let s = self.session();
        if s.active_camera == camera && s.active_frame_calibration == image {
            return;
        }
        let manual = self.settings().disable_checkerboard_detection;
        let s = self.session_mut();
        s.active_camera = camera;
        s.active_frame_calibration = image;
        // activeCameraChanged, activeFrameCalibrationChanged: `checkerboardManual` from
        // `DisableCheckerboardDetection`, then setDialog.
        s.checkerboard_manual = manual;
        self.calibration_dialog();
    }

    /// C25, the "manual mode" check box: the session's manual mode, then
    /// `reloadManualPoints` and `setDialog`. Not an undo step, and the project is not
    /// changed: the active image's manual point lists are created by the next calibration
    /// edit (ledger A164, [`crate::Session::manual_lists_pending`]).
    ///
    /// Ported from ui/WizardCalibrationCubeFrame.cpp, on_checkBoxManual_clicked.
    pub fn set_manual_mode(&mut self, on: bool) {
        self.session_mut().manual_mode = on;
        all_frames::reload_manual_points(self);
        self.calibration_dialog();
    }

    /// C16, Alt + click in the Calibration workspace on the distorted image: the click moved
    /// to the marker that the setting `DetectionMethodForCalibration` finds near it in the
    /// calibration image (the click itself when the setting is "none" or nothing is found).
    /// [`Document::snap_calibration_click_in`] for a view that may show the undistorted image.
    #[must_use]
    pub fn snap_calibration_click(
        &self,
        camera: usize,
        image: usize,
        x: f64,
        y: f64,
    ) -> (f64, f64) {
        self.snap_calibration_click_in(camera, image, x, y, false)
    }

    /// C16 for a view that shows the distorted (`undistorted` false) or undistorted image:
    /// the setting's detection (1 to 4 detection method 0 to 3, 5 the white blob detector),
    /// searching 40 pixels around the click with mask size 5 and threshold 8, in the distorted
    /// calibration image.
    ///
    /// In the XMALab 3 mode the click is searched for as given, also on the undistorted view,
    /// and the result is rounded to whole pixels (`cv::Point`); in the default mode an
    /// undistorted click is moved into the distorted image first and the result back, and
    /// the position is kept as found (ledger A130).
    ///
    /// A click that cannot be snapped comes back as given, in both modes: one whose search
    /// window is out of the `int` range of the C++ (NaN, infinite, beyond about +-2^31;
    /// ledger A160); in the default mode also one the detection leaves where it was (nothing
    /// found).
    ///
    /// Ported from ui/GLCameraView.cpp, mousePressEvent (`:265-282`), with
    /// `MarkerDetection::detectionPoint` ([`xma_vision::detection::detect_point`]).
    #[must_use]
    pub fn snap_calibration_click_in(
        &self,
        camera: usize,
        image: usize,
        x: f64,
        y: f64,
        undistorted: bool,
    ) -> (f64, f64) {
        use xma_vision::detection::{DetectionMethod as M, detect_point, detect_point_u16};
        let setting = self.settings().detection_method_for_calibration.code();
        if setting <= 0 || camera >= self.project().cameras.len() {
            return (x, y);
        }
        let method = match setting {
            1 => M::Default,
            2 => M::Blob,
            3 => M::White,
            4 => M::Corner,
            5 => M::WhiteBlob,
            _ => return (x, y),
        };
        let Ok(frame) = images::calibration_frame(self, camera, image) else {
            return (x, y);
        };
        let rounded = self.compat().calibration_snap_rounded;
        let model = &self.camera(camera).model;
        let seed = if undistorted && !rounded {
            let p = held::distort(model, xma_model::Point2::new(x, y));
            (p.x, p.y)
        } else {
            (x, y)
        };
        let options = crate::tracking::jobs::detect_options(self);
        // Ledger A160: a click whose search window the C++ cannot compute in `int` (NaN,
        // infinite, or beyond about +-2^31) cannot be snapped: it comes back as given.
        if !xma_vision::detection::window_fits(seed, 40, options.compat) {
            return (x, y);
        }
        let found = match crate::tracking::jobs::algorithm_image_for(&frame, self.compat()) {
            crate::tracking::jobs::AlgorithmImage::U8(im) => {
                detect_point(&im, method, seed, 40, 5, 8.0, &options)
            }
            crate::tracking::jobs::AlgorithmImage::Deep { image, bit_depth } => {
                detect_point_u16(&image, bit_depth, method, seed, 40, 5, 8.0, &options)
            }
        };
        let (px, py) = found.point;
        // Ledger A160: in the default mode a click the detection leaves where it was
        // (nothing found) comes back as given, not moved into the distorted image and back.
        if !rounded && found.point == seed {
            return (x, y);
        }
        if rounded {
            // cv::Point(cv::Point2d): cvRound, half to even.
            return (px.round_ties_even(), py.round_ties_even());
        }
        if undistorted {
            let q = held::undistort(model, xma_model::Point2::new(px, py));
            return (q.x, q.y);
        }
        (px, py)
    }
}
