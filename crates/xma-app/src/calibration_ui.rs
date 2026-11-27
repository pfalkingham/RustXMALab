//! The Undistortion and Calibration workspaces' controls and clicks: what each one asks
//! for, as plain data and functions of the state and the document.
//!
//! Ported from ui/WizardUndistortionFrame.cpp, ui/WizardCalibrationCubeFrame.cpp and the
//! `UNDISTORTION` and `CALIBRATION` branches of ui/GLCameraView.cpp (`mousePressEvent`,
//! `mouseDoubleClickEvent`); the action numbers are those of `docs/cpp-calibration.md`.
//!
//! The Toolbox pages and the camera views raise [`CalibAction`]s
//! (`Action::Calib`); `XmaApp::dispatch` carries them out (`app/calibration.rs`): it asks
//! the questions XMALab asks, changes the session state the engine keeps
//! (`xma_engine::Session`: reference clicks, manual mode) and requests the
//! `xma_engine::calibration::CalibrationEdit`s. Edits that take long run on a worker thread
//! (`app/long_edit.rs`).
//!
//! Differences from XMALab 3 (interface defects fixed; proposed ledger rows in the appcal
//! report):
//!
//! - Alt + click (C16) is the engine's `Document::snap_calibration_click_in` (ledger A130):
//!   the detected position is kept to the sub-pixel, and in a view that shows the
//!   undistorted image the click is distorted before the detection and the result
//!   undistorted again (CD5).
//! - A double click (C23) is placed with the press formula (ledger U3).
//! - Delete Frame's question names the frame 1-based, as everywhere else (CD13).
//! - Unchecking "is optimized" requests the removal of the optimisation for every camera
//!   whatever the active image is (CD23).
//! - A reference index set by hand ("CUSTOM INDEX", C14) is kept by the window until the
//!   project is closed: after every `setDialog` the engine runs (a camera, frame or
//!   workspace change, manual mode, every calibration), which reloads the reference file's
//!   indices as XMALab 3 does, the window sets it again. XMALab 3 loses it.
//! - A manual-mode click (C27) and Ctrl + click (C24) are in the coordinates of the image
//!   the view shows: XMALab 3 takes the "undistorted" display choice even when the image is
//!   not calibrated and so shown distorted.

use xma_engine::Document;
use xma_engine::calibration::{CalibrationEdit, GridPointList};
use xma_model::CalibrationObject;

use crate::actions::Action;
use crate::state::{
    AppState, CalibrationVisImage, UndistortionMouseMode, UndistortionVisPoints, Workspace,
};

/// C5's question (`on_radioButtonMouseClickCenter_clicked`).
pub const SET_CENTER_WARNING: &str =
    "Warning: Setting a new Center will reset all inliers. Are you sure you want to proceed?";
/// C6's error for a points display other than the grid or reference points.
pub const TOGGLE_OUTLIER_ERROR: &str = "You either have to display the grid points (distorted or undistorted) or the reference points to toggle outlier";
/// C7's error for an image display other than the distorted one.
pub const SET_CENTER_ERROR: &str = "You have to display the distorted image to set the center";
/// C11's question (`WizardUndistortionFrame::checkForPendingChanges`).
pub const PENDING_UNDISTORTION: &str = "You have modified the undistortion input and before continuing you first have to recompute the undistortion. If you want to recompute click 'yes', if you want to continue changing undistortion parameters click 'cancel' ";
/// C39's question (`WizardCalibrationCubeFrame::checkForPendingChanges`).
pub const PENDING_CALIBRATION: &str = "You have modified the calibration input and before continuing you first have to recompute the calibration. If you want to recompute click 'yes', if you want to continue changing calibration parameters click 'cancel' ";
/// C36's question.
pub const REMOVE_OPTIMIZATION: &str = "Are you sure you want to remove the optimization?";

/// C8's first prompt: title, label, value, minimum, maximum.
pub const BORDER_PROMPT: (&str, &str, i32, i32, i32) = (
    "Threshold: Distance to border",
    "Threshold (Border)",
    10,
    0,
    100,
);
/// C8's second prompt.
pub const CIRCLE_PROMPT: (&str, &str, i32, i32, i32) = (
    "Threshold: Distance to fitted circle",
    "Threshold (Circle)",
    25,
    0,
    100,
);

/// C32's question for the active camera and calibration frame (0-based).
#[must_use]
pub fn reset_frame_text(frame: usize, camera_name: &str) -> String {
    format!(
        "Are you sure you want to reset Frame {} for {camera_name}",
        frame + 1
    )
}

/// C33's question.
#[must_use]
pub fn reset_camera_text(camera_name: &str) -> String {
    format!("Are you sure you want to reset {camera_name} and all detected frames")
}

/// C34's question for calibration frame `frame` (0-based). XMALab 3 shows the 0-based
/// number (CD13); here it is 1-based, as the frame list and the frame bar show it.
#[must_use]
pub fn delete_frame_text(frame: usize) -> String {
    format!(
        "Are you sure you want to delete frame {}  for all cameras?",
        frame + 1
    )
}

/// C14's prompt title for reference `slot` (0-based).
#[must_use]
pub fn reference_index_title(slot: usize) -> String {
    format!("Set Custom Idx for Reference {}", slot + 1)
}

/// What a control of the two pages or a click in their camera views asks for
/// (`Action::Calib`). Handled by `XmaApp::dispatch` (`app/calibration.rs`).
#[derive(Clone, Debug, PartialEq)]
pub enum CalibAction {
    // ----- Undistortion workspace ---------------------------------------------------------
    /// C3, C4, C5: a mouse-click radio button. "set center" asks first once undistorted.
    MouseMode(UndistortionMouseMode),
    /// C6, C7: a left click in `camera`'s view at (`x`, `y`) of the image (press formula).
    UndistortionClick { camera: usize, x: f64, y: f64 },
    /// C8: "remove outlier at border" (its two prompts).
    RemoveOutliers,

    // ----- Calibration workspace ----------------------------------------------------------
    /// C13: a reference radio button.
    ChooseReference(usize),
    /// C14: a reference tool button (its prompt).
    ReferenceIndex(usize),
    /// C15, C16, C19, C24, C27: a left click in `camera`'s view at (`x`, `y`), with Ctrl
    /// (Command on macOS) or Alt held.
    CalibrationClick {
        camera: usize,
        x: f64,
        y: f64,
        ctrl: bool,
        alt: bool,
    },
    /// C23: a double click in `camera`'s view.
    DoubleClick { camera: usize, x: f64, y: f64 },
    /// C25: "manual mode".
    ManualMode(bool),
    /// C26: a manual-reference radio button (the calibration-object point, 0-based).
    ManualPoint(usize),
    /// C28: a manual-reference check box: every check box's state.
    ManualInliers(Vec<bool>),
    /// C17, C18, C22, C29: "compute calibration" / "recompute calibration".
    Compute,
    /// C32.
    ResetFrame,
    /// C33.
    ResetCamera,
    /// C34.
    DeleteFrame,
    /// C35: "use camera distortion model".
    LensModel(bool),
    /// C36: "is optimized" unchecked.
    RemoveOptimization,

    // ----- Both ---------------------------------------------------------------------------
    /// C11, C39: "Yes" to the pending-changes question: recompute, then carry out `then`
    /// (the workspace change or the save that asked), with `after_save` as the action to
    /// follow that save.
    RecomputeThen {
        then: Box<Action>,
        after_save: Option<Box<Action>>,
    },
}

// `Action` is `Eq` and `Hash`: the click positions compare and hash by their bits.
impl Eq for CalibAction {}

impl std::hash::Hash for CalibAction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            CalibAction::MouseMode(m) => (*m as u8).hash(state),
            CalibAction::UndistortionClick { camera, x, y }
            | CalibAction::DoubleClick { camera, x, y } => {
                (camera, x.to_bits(), y.to_bits()).hash(state);
            }
            CalibAction::CalibrationClick {
                camera,
                x,
                y,
                ctrl,
                alt,
            } => (camera, x.to_bits(), y.to_bits(), ctrl, alt).hash(state),
            CalibAction::ChooseReference(i)
            | CalibAction::ReferenceIndex(i)
            | CalibAction::ManualPoint(i) => i.hash(state),
            CalibAction::ManualMode(b) | CalibAction::LensModel(b) => b.hash(state),
            CalibAction::ManualInliers(v) => v.hash(state),
            CalibAction::RecomputeThen { then, after_save } => (then, after_save).hash(state),
            CalibAction::RemoveOutliers
            | CalibAction::Compute
            | CalibAction::ResetFrame
            | CalibAction::ResetCamera
            | CalibAction::DeleteFrame
            | CalibAction::RemoveOptimization => {}
        }
    }
}

impl CalibAction {
    /// The text and C++ origin of the control.
    #[must_use]
    pub fn info(&self) -> (&'static str, &'static str) {
        match self {
            CalibAction::MouseMode(UndistortionMouseMode::None) => (
                "None",
                "WizardUndistortionFrame::on_radioButtonMouseClickNone_clicked",
            ),
            CalibAction::MouseMode(UndistortionMouseMode::ToggleOutlier) => (
                "toggle outlier",
                "WizardUndistortionFrame::on_radioButtonMouseClickOutlier_clicked",
            ),
            CalibAction::MouseMode(UndistortionMouseMode::SetCenter) => (
                "set center",
                "WizardUndistortionFrame::on_radioButtonMouseClickCenter_clicked",
            ),
            CalibAction::UndistortionClick { .. } => {
                ("Undistortion click", "GLCameraView::mousePressEvent")
            }
            CalibAction::RemoveOutliers => (
                "remove outlier at border",
                "WizardUndistortionFrame::on_pushButtonRemoveOutlierAutomatically_clicked",
            ),
            CalibAction::ChooseReference(_) => (
                "Reference",
                "WizardCalibrationCubeFrame::radioButtonReference",
            ),
            CalibAction::ReferenceIndex(_) => (
                "Reference index",
                "WizardCalibrationCubeFrame::on_toolButtonReference_clicked",
            ),
            CalibAction::CalibrationClick { .. } => (
                "Calibration click",
                "GLCameraView::mousePressEvent, WizardCalibrationCubeFrame::addCalibrationReference",
            ),
            CalibAction::DoubleClick { .. } => {
                ("Toggle inlier", "GLCameraView::mouseDoubleClickEvent")
            }
            CalibAction::ManualMode(_) => (
                "manual mode",
                "WizardCalibrationCubeFrame::on_checkBoxManual_clicked",
            ),
            CalibAction::ManualPoint(_) => (
                "Manual reference",
                "WizardCalibrationCubeFrame::setupManualPoints",
            ),
            CalibAction::ManualInliers(_) => (
                "Manual reference inlier",
                "WizardCalibrationCubeFrame::checkBoxManualReference_clicked",
            ),
            CalibAction::Compute => (
                "compute calibration",
                "WizardCalibrationCubeFrame::on_pushButton_clicked",
            ),
            CalibAction::ResetFrame => (
                "Reset Frame",
                "WizardCalibrationCubeFrame::on_pushButtonResetFrame_clicked",
            ),
            CalibAction::ResetCamera => (
                "Reset Camera",
                "WizardCalibrationCubeFrame::on_pushButtonResetCamera_clicked",
            ),
            CalibAction::DeleteFrame => (
                "Delete Frame",
                "WizardCalibrationCubeFrame::on_pushButtonDeleteFrame_clicked",
            ),
            CalibAction::LensModel(_) => (
                "use camera distortion model",
                "WizardCalibrationCubeFrame::on_checkBoxDistortion_clicked",
            ),
            CalibAction::RemoveOptimization => (
                "is optimized",
                "WizardCalibrationCubeFrame::on_checkBoxOptimized_clicked",
            ),
            CalibAction::RecomputeThen { .. } => {
                ("Recompute", "WizardDockWidget::checkForPendingChanges")
            }
        }
    }

    /// The C++ rule for when the control does something (the page shows it only then, or
    /// its handler checks): an internal calibration in its workspace.
    #[must_use]
    pub fn cpp_enabled(&self, state: &AppState, open: bool) -> bool {
        if !open {
            return false;
        }
        match self {
            CalibAction::MouseMode(_)
            | CalibAction::UndistortionClick { .. }
            | CalibAction::RemoveOutliers => state.workspace == Workspace::Undistortion,
            // Raised by the pending-changes question, in either workspace.
            CalibAction::RecomputeThen { .. } => true,
            // GLCameraView::mousePressEvent returns while drawing is disabled; its double
            // click handler does not check.
            CalibAction::CalibrationClick { .. } => {
                state.workspace == Workspace::Calibration && !state.disable_draw
            }
            _ => state.workspace == Workspace::Calibration,
        }
    }
}

/// The point list a click in "toggle outlier" mode refers to (C6): the points display
/// "grid points (distorted)", "grid points (undistorted)" or "reference points"; `None`
/// for the others (XMALab shows [`TOGGLE_OUTLIER_ERROR`]).
#[must_use]
pub fn outlier_list(points: UndistortionVisPoints) -> Option<GridPointList> {
    match points {
        UndistortionVisPoints::GridDistorted => Some(GridPointList::Distorted),
        UndistortionVisPoints::GridUndistorted => Some(GridPointList::Undistorted),
        UndistortionVisPoints::Reference => Some(GridPointList::References),
        UndistortionVisPoints::None | UndistortionVisPoints::Detected => None,
    }
}

/// The calibrated state of `camera`'s calibration image `image` (0 when absent).
#[must_use]
pub fn image_calibrated(doc: &Document, camera: usize, image: usize) -> i32 {
    doc.project()
        .cameras
        .get(camera)
        .and_then(|c| c.calibration_images.get(image))
        .map_or(0, |im| im.calibrated)
}

/// True when `camera`'s view shows the undistorted calibration image: the display choice
/// "undistorted" and an image that is calibrated (`CalibrationImage::bindTexture` shows an
/// image that is not calibrated distorted).
#[must_use]
pub fn shows_undistorted(doc: &Document, state: &AppState, camera: usize) -> bool {
    state.calibration_vis_image == CalibrationVisImage::Undistorted
        && image_calibrated(doc, camera, state.active_frame_calibration.unwrap_or(0)) > 0
}

/// True when the project's calibration object is a checkerboard (section 4 of
/// `docs/cpp-calibration.md`).
#[must_use]
pub fn is_checkerboard(doc: &Document) -> bool {
    matches!(
        doc.project().calibration_object,
        Some(CalibrationObject::Checkerboard { .. })
    )
}

/// The number of points of the calibration object: the cube's beads or the checkerboard's
/// corners (`getFrameSpecifications().size()`).
#[must_use]
pub fn object_points(doc: &Document) -> usize {
    doc.project()
        .calibration_object
        .as_ref()
        .map_or(0, CalibrationObject::n_points)
}

/// The text the Progress panel shows while `edit` runs on the worker thread, `None` for an
/// edit applied at once. `CalibrationEdit::RecomputePending` (the pending-changes question's
/// Yes) shows XMALab's text of the workspace that asked: "Local Undistortion" in the
/// Undistortion workspace, the calibration's text in the Calibration workspace
/// (`docs/cpp-calibration.md` section 9).
#[must_use]
pub fn progress_text(edit: &CalibrationEdit, workspace: Workspace) -> Option<&'static str> {
    match edit {
        CalibrationEdit::RecomputePending if workspace == Workspace::Calibration => {
            Some(RECOMPUTE_CALIBRATION_TEXT)
        }
        e => e.progress_text(),
    }
}

/// [`progress_text`] for a project whose calibration object is a checkerboard (`board`):
/// "compute calibration" shows the checkerboard detection's text, "Detect Corner"
/// (`CheckerboardDetection::detectCorner`).
#[must_use]
pub fn progress_text_for(
    edit: &CalibrationEdit,
    workspace: Workspace,
    board: bool,
) -> Option<&'static str> {
    match edit {
        CalibrationEdit::ComputeCalibration { .. } if board => Some("Detect Corner"),
        e => progress_text(e, workspace),
    }
}

/// The Progress panel's text while the cameras with pending changes are calibrated again.
pub const RECOMPUTE_CALIBRATION_TEXT: &str =
    "Calibrate camera internal parameters and pose for all frames";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delete_frame_question_is_one_based() {
        assert_eq!(
            delete_frame_text(0),
            "Are you sure you want to delete frame 1  for all cameras?"
        );
        assert_eq!(
            reset_frame_text(2, "cam1"),
            "Are you sure you want to reset Frame 3 for cam1"
        );
    }

    #[test]
    fn outlier_lists() {
        assert_eq!(
            outlier_list(UndistortionVisPoints::GridDistorted),
            Some(GridPointList::Distorted)
        );
        assert_eq!(outlier_list(UndistortionVisPoints::Detected), None);
        assert_eq!(outlier_list(UndistortionVisPoints::None), None);
    }
}
