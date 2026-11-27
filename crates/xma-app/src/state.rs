//! What the interface is showing: workspace, view layout, active camera, frame and trial,
//! the display choices of the undistortion and calibration pages, and per trial the view
//! state the C++ keeps in `Trial`.
//!
//! Ported from ui/State.h and ui/State.cpp (the `State` singleton) without the singleton:
//! one [`AppState`] lives in the application and is passed to whatever needs it. Every
//! `change_*` function does what the C++ `State::change*` does (change the value if
//! different or `force`d) and, instead of emitting a Qt signal, returns a [`Changes`] that
//! names what changed. The application then runs its handlers for those changes
//! (`crate::app`), as the slots connected to the signals do in the C++.
//!
//! Also here: [`ProjectFacts`], the few facts about the open project that decide what the
//! window shows (calibrated or not, how many trials...), read from the document.

use std::collections::BTreeSet;

use xma_engine::Document;
use xma_model::{CalibrationType, MarkerStatus};

/// The three workspaces (C++ `work_state`).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Workspace {
    /// `UNDISTORTION`.
    Undistortion,
    /// `CALIBRATION` (the C++ initial value).
    #[default]
    Calibration,
    /// `DIGITIZATION`, shown as "Marker tracking".
    MarkerTracking,
}

impl Workspace {
    pub const ALL: [Workspace; 3] = [
        Workspace::Undistortion,
        Workspace::Calibration,
        Workspace::MarkerTracking,
    ];

    /// The item text in the workspace combo box (`WorkspaceNavigationFrame.ui`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Workspace::Undistortion => "Undistortion",
            Workspace::Calibration => "Calibration",
            Workspace::MarkerTracking => "Marker tracking",
        }
    }

    /// For `--workspace`.
    #[must_use]
    pub fn from_cli(s: &str) -> Option<Workspace> {
        match s {
            "undistortion" => Some(Workspace::Undistortion),
            "calibration" => Some(Workspace::Calibration),
            "tracking" | "marker-tracking" | "digitization" => Some(Workspace::MarkerTracking),
            _ => None,
        }
    }
}

/// The camera-view layouts of the View combo box (C++ `ui_state`).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum DisplayLayout {
    /// `ALL_CAMERAS_FULL_HEIGHT`: side by side at full height, scrolling sideways.
    FullHeight,
    /// `ALL_CAMERAS_1ROW_SCALED` (the C++ initial value).
    #[default]
    OneRow,
    /// `ALL_CAMERAS_2ROW_SCALED`.
    TwoRows,
    /// `ALL_CAMERAS_3ROW_SCALED`.
    ThreeRows,
}

impl DisplayLayout {
    pub const ALL: [DisplayLayout; 4] = [
        DisplayLayout::FullHeight,
        DisplayLayout::OneRow,
        DisplayLayout::TwoRows,
        DisplayLayout::ThreeRows,
    ];

    /// The item text in the View combo box.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            DisplayLayout::FullHeight => "All Cameras - Full Height",
            DisplayLayout::OneRow => "All Cameras - 1 Row Scaled",
            DisplayLayout::TwoRows => "All Cameras - 2 Row Scaled",
            DisplayLayout::ThreeRows => "All Cameras - 3 Row Scaled",
        }
    }
}

/// C++ `undistortion_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UndistortionState {
    #[default]
    NotUndistorted,
    Undistorted,
}

/// C++ `undistortionMouseMode_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UndistortionMouseMode {
    #[default]
    None,
    ToggleOutlier,
    SetCenter,
}

/// C++ `undistortionVisImage_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UndistortionVisImage {
    #[default]
    Distorted,
    Undistorted,
    MapAngleScaled,
    MapAngles,
    MapDistance,
    MapHorizontal,
    MapVertical,
}

/// C++ `undistortionVisPoints_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UndistortionVisPoints {
    #[default]
    None,
    Detected,
    GridDistorted,
    GridUndistorted,
    Reference,
}

/// C++ `calibrationVisImage_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CalibrationVisImage {
    #[default]
    Distorted,
    Undistorted,
}

/// C++ `calibrationVisPoints_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CalibrationVisPoints {
    #[default]
    None,
    DetectedAll,
    DetectedCubeDistorted,
    ProjectedCubeDistorted,
    DetectedCubeUndistorted,
    ProjectedCubeUndistorted,
}

/// C++ `calibrationVisText_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CalibrationVisText {
    #[default]
    None,
    Id,
    Error,
    ErrorColor,
    MarkerName,
}

/// Per-trial view state the C++ keeps in `Trial` (`activeFrame`, `startFrame`, `endFrame`,
/// `activeMarkerIdx`, `activeBodyIdx`), which is interface state, not project data.
///
/// `start_frame` and `end_frame` are 1-based as in the C++ and the project file;
/// `active_frame` is 0-based.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrialView {
    pub active_frame: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    /// Number of frames of the trial (`Trial::getNbImages`).
    pub n_frames: usize,
    pub active_marker: Option<usize>,
    pub active_body: Option<usize>,
}

impl TrialView {
    /// The view state of a freshly loaded trial: frame 0 (`Trial::Trial`, `activeFrame = 0`),
    /// the stored start and end frames, no active marker or body.
    #[must_use]
    pub fn from_trial(t: &xma_model::Trial) -> Self {
        let n = t.n_frames;
        Self {
            active_frame: 0,
            start_frame: t.start_frame.max(1),
            end_frame: if t.end_frame == 0 {
                n.max(1)
            } else {
                t.end_frame
            },
            n_frames: n,
            active_marker: None,
            active_body: None,
        }
    }
}

/// What is selected in the Points panel (markers and rigid bodies, by index in the trial).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub markers: BTreeSet<usize>,
    pub bodies: BTreeSet<usize>,
}

/// Direction of playback (`SequenceNavigationFrame::play_tag`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Playback {
    #[default]
    Stopped,
    Forward,
    Backward,
}

/// The interface state. See the module documentation.
#[derive(Clone, Debug, PartialEq)]
pub struct AppState {
    pub workspace: Workspace,
    pub display: DisplayLayout,
    pub active_camera: Option<usize>,
    pub active_frame_calibration: Option<usize>,
    pub active_trial: Option<usize>,
    pub active_frame_trial: Option<usize>,
    pub undistortion: UndistortionState,
    pub undistortion_vis_image: UndistortionVisImage,
    pub undistortion_vis_points: UndistortionVisPoints,
    pub undistortion_mouse_mode: UndistortionMouseMode,
    pub calibration_vis_image: CalibrationVisImage,
    pub calibration_vis_points: CalibrationVisPoints,
    pub calibration_vis_text: CalibrationVisText,
    /// `State::disableDraw`: set while a job changes data the views draw.
    pub disable_draw: bool,
    /// `State::loading`: a project is being loaded.
    pub loading: bool,

    /// One per trial of the open project.
    pub trials: Vec<TrialView>,
    /// Points-panel selection in the active trial.
    pub selection: Selection,
    /// Camera visibility (Camera Selector dialog, `CameraViewWidget::setIsVisible`).
    pub camera_visible: Vec<bool>,
    /// Playback in the frame bar.
    pub playback: Playback,
    /// The name of the step Edit > Undo would undo, `None` when there is none (parity
    /// ledger U52; XMALab 3 has one `MainWindow::setUndo` flag). Kept in step with the
    /// document by the application.
    pub undo_label: Option<String>,
    /// The name of the step Edit > Redo would redo (an addition, ledger U52).
    pub redo_label: Option<String>,
    /// The tracking button whose run is open (`WizardDigitizationFrame::trackType`,
    /// `trackDirection`); kept by the application while `XmaApp::tracking` is `Some`.
    pub running_track: Option<crate::tracking_ui::TrackButton>,
    /// Whether the multi-camera runs started from the window fit touching dots apart
    /// (`MultiTrackParams::split_touching`, new in this program, ledger U169): "Separate
    /// touching dots" on the Toolbox's tracking page, ticked while "Enable multi-camera
    /// tracking" is ticked. The page keeps the check box's own state
    /// (`DigitizationPage::split_touching`) and writes this each time it is drawn; the
    /// application reads it when it starts a run. No other kind reads it. True at start and
    /// while the multi-camera box is unticked (ledger U171).
    pub multi_split_touching: bool,
    /// Whether the multi-camera runs started from the window recognise a lost marker by the
    /// run's own distance statistics (`MultiTrackParams::stat_links`, ledger A224):
    /// "Recognise a lost marker by its distances" (ledger U170), written by the page as
    /// `multi_split_touching`; true at start and while "Enable multi-camera tracking" is
    /// unticked (ledger U171).
    pub multi_stat_links: bool,
    /// Whether they place a lost marker with no rigid prediction by its mean distances
    /// (`MultiTrackParams::distance_placement`, ledger A225): "Place a lost marker by its
    /// distances" (ledger U170), as above; true while the multi-camera box is unticked.
    pub multi_distance_placement: bool,
    /// Whether they remove a kept older point lying on the dot the run gave to another marker
    /// (`MultiTrackParams::keep_dot_rule`, rule (b) of ledger A221): "Remove old points on
    /// another marker's dot" (ledger U170), as above; true while the multi-camera box is
    /// unticked.
    pub multi_keep_dot_rule: bool,
    /// View > DetectionSettings is open (`DetectionSettings::isVisible`).
    pub detection_settings_open: bool,
}

impl Default for AppState {
    /// `State::State`.
    fn default() -> Self {
        Self {
            workspace: Workspace::Calibration,
            display: DisplayLayout::OneRow,
            active_camera: None,
            active_frame_calibration: None,
            active_trial: None,
            active_frame_trial: None,
            undistortion: UndistortionState::NotUndistorted,
            undistortion_vis_image: UndistortionVisImage::Distorted,
            undistortion_vis_points: UndistortionVisPoints::None,
            undistortion_mouse_mode: UndistortionMouseMode::None,
            calibration_vis_image: CalibrationVisImage::Distorted,
            calibration_vis_points: CalibrationVisPoints::None,
            calibration_vis_text: CalibrationVisText::None,
            disable_draw: false,
            loading: false,
            trials: Vec::new(),
            selection: Selection::default(),
            camera_visible: Vec::new(),
            playback: Playback::Stopped,
            undo_label: None,
            redo_label: None,
            running_track: None,
            multi_split_touching: true,
            multi_stat_links: true,
            multi_distance_placement: true,
            multi_keep_dot_rule: true,
            detection_settings_open: false,
        }
    }
}

/// What a `change_*` call changed: one flag per C++ `State` signal, plus the per-trial
/// view state and selection. Combine with `|=`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Changes {
    pub workspace: bool,
    pub display: bool,
    pub active_camera: bool,
    pub active_frame_calibration: bool,
    pub active_trial: bool,
    pub active_frame_trial: bool,
    pub undistortion: bool,
    pub undistortion_vis: bool,
    pub calibration_vis: bool,
    /// Start or end frame of a trial.
    pub trial_range: bool,
    /// Active marker or body, or the selection.
    pub selection: bool,
    /// Camera visibility.
    pub cameras: bool,
    /// The project's data changed: an edit, an undo or redo, a tracking step (not the
    /// views' scale and bias). Not a C++ `State` signal: what XMALab 3 does with
    /// `redrawGL`, `reloadListFromObject` and `updateMarkers` after each edit.
    pub document: bool,
}

impl Changes {
    pub const NONE: Changes = Changes {
        workspace: false,
        display: false,
        active_camera: false,
        active_frame_calibration: false,
        active_trial: false,
        active_frame_trial: false,
        undistortion: false,
        undistortion_vis: false,
        calibration_vis: false,
        trial_range: false,
        selection: false,
        cameras: false,
        document: false,
    };

    #[must_use]
    pub fn any(&self) -> bool {
        *self != Changes::NONE
    }
}

impl std::ops::BitOrAssign for Changes {
    fn bitor_assign(&mut self, o: Changes) {
        self.workspace |= o.workspace;
        self.display |= o.display;
        self.active_camera |= o.active_camera;
        self.active_frame_calibration |= o.active_frame_calibration;
        self.active_trial |= o.active_trial;
        self.active_frame_trial |= o.active_frame_trial;
        self.undistortion |= o.undistortion;
        self.undistortion_vis |= o.undistortion_vis;
        self.calibration_vis |= o.calibration_vis;
        self.trial_range |= o.trial_range;
        self.selection |= o.selection;
        self.cameras |= o.cameras;
        self.document |= o.document;
    }
}

macro_rules! change_fn {
    ($(#[$m:meta])* $name:ident, $field:ident, $ty:ty, $flag:ident) => {
        $(#[$m])*
        pub fn $name(&mut self, value: $ty, force: bool) -> Changes {
            if force || self.$field != value {
                self.$field = value;
                Changes { $flag: true, ..Changes::NONE }
            } else {
                Changes::NONE
            }
        }
    };
}

impl AppState {
    change_fn!(
        /// Ported from ui/State.cpp, State::changeWorkspace.
        change_workspace,
        workspace,
        Workspace,
        workspace
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeDisplay.
        change_display,
        display,
        DisplayLayout,
        display
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeActiveCamera.
        change_active_camera,
        active_camera,
        Option<usize>,
        active_camera
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeActiveFrameCalibration.
        change_active_frame_calibration,
        active_frame_calibration,
        Option<usize>,
        active_frame_calibration
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeActiveTrial.
        change_active_trial,
        active_trial,
        Option<usize>,
        active_trial
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeUndistortion.
        change_undistortion,
        undistortion,
        UndistortionState,
        undistortion
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeUndistortionVisImage.
        change_undistortion_vis_image,
        undistortion_vis_image,
        UndistortionVisImage,
        undistortion_vis
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeUndistortionVisPoints.
        change_undistortion_vis_points,
        undistortion_vis_points,
        UndistortionVisPoints,
        undistortion_vis
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeUndistortionMouseMode.
        change_undistortion_mouse_mode,
        undistortion_mouse_mode,
        UndistortionMouseMode,
        undistortion_vis
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeCalibrationVisImage.
        change_calibration_vis_image,
        calibration_vis_image,
        CalibrationVisImage,
        calibration_vis
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeCalibrationVisPoints.
        change_calibration_vis_points,
        calibration_vis_points,
        CalibrationVisPoints,
        calibration_vis
    );
    change_fn!(
        /// Ported from ui/State.cpp, State::changeCalibrationVisText.
        change_calibration_vis_text,
        calibration_vis_text,
        CalibrationVisText,
        calibration_vis
    );

    /// Ported from ui/State.cpp, State::changeActiveFrameTrial. Also keeps the active
    /// trial's `TrialView::active_frame` in step, as `SequenceNavigationFrame::changeFrame`
    /// does with `Trial::setActiveFrame` before changing the state.
    pub fn change_active_frame_trial(&mut self, value: Option<usize>, force: bool) -> Changes {
        if let (Some(f), Some(t)) = (value, self.active_trial)
            && let Some(view) = self.trials.get_mut(t)
        {
            view.active_frame = f;
        }
        if force || self.active_frame_trial != value {
            self.active_frame_trial = value;
            Changes {
                active_frame_trial: true,
                ..Changes::NONE
            }
        } else {
            Changes::NONE
        }
    }

    /// The active trial's view state.
    #[must_use]
    pub fn trial_view(&self) -> Option<&TrialView> {
        self.trials.get(self.active_trial?)
    }

    /// The active trial's view state, mutable.
    pub fn trial_view_mut(&mut self) -> Option<&mut TrialView> {
        let t = self.active_trial?;
        self.trials.get_mut(t)
    }

    /// Sets the active marker of the active trial and selects only it (`PointsDockWidget::
    /// selectPoint`).
    pub fn select_marker(&mut self, marker: Option<usize>) -> Changes {
        let Some(view) = self.trial_view_mut() else {
            return Changes::NONE;
        };
        view.active_marker = marker;
        view.active_body = None;
        self.selection.markers = marker.into_iter().collect();
        self.selection.bodies.clear();
        Changes {
            selection: true,
            ..Changes::NONE
        }
    }

    /// Sets the active rigid body of the active trial and selects only it (`PointsDockWidget::
    /// selectBody`).
    pub fn select_body(&mut self, body: Option<usize>) -> Changes {
        let Some(view) = self.trial_view_mut() else {
            return Changes::NONE;
        };
        view.active_body = body;
        view.active_marker = None;
        self.selection.bodies = body.into_iter().collect();
        self.selection.markers.clear();
        Changes {
            selection: true,
            ..Changes::NONE
        }
    }

    /// Resets the state for a newly opened project: one [`TrialView`] per trial, every
    /// camera visible.
    pub fn reset_for_project(&mut self, doc: &Document) {
        let project = doc.project();
        self.trials = project
            .trials
            .iter()
            .map(|t| TrialView::from_trial(t))
            .collect();
        self.camera_visible = vec![true; project.cameras.len()];
        self.selection = Selection::default();
        self.playback = Playback::Stopped;
        self.undo_label = None;
        self.redo_label = None;
    }

    /// Clears what belonged to the closed project (`MainWindow::closeProject` and
    /// `tearDownProjectUI`).
    pub fn reset_for_close(&mut self) -> Changes {
        let mut ch = self.change_undistortion(UndistortionState::NotUndistorted, false);
        ch |= self.change_active_trial(None, false);
        self.trials.clear();
        self.camera_visible.clear();
        self.selection = Selection::default();
        self.playback = Playback::Stopped;
        self.undo_label = None;
        self.redo_label = None;
        ch
    }
}

/// Facts about the open project that decide what the window shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectFacts {
    pub n_cameras: usize,
    pub camera_names: Vec<String>,
    /// `Project::getCalibration`.
    pub calibration: CalibrationType,
    /// `Project::isCalibrated`: every camera calibrated (`Project::checkCalibration`).
    pub calibrated: bool,
    /// Any camera has an undistortion grid (`MainWindow::setupProjectUI`).
    pub has_undistortion: bool,
    /// `Project::getNbImagesCalibration`: the first camera's calibration images.
    pub nb_images_calibration: usize,
    /// `Project::camerasOptimized`.
    pub cameras_optimized: bool,
    pub trial_names: Vec<String>,
    pub trial_is_default: Vec<bool>,
}

impl ProjectFacts {
    /// Reads the facts from a document.
    #[must_use]
    pub fn from_doc(doc: &Document) -> Self {
        let p = doc.project();
        Self {
            n_cameras: p.cameras.len(),
            camera_names: p.cameras.iter().map(|c| c.name.clone()).collect(),
            calibration: p.calibration_type,
            calibrated: !p.cameras.is_empty() && p.cameras.iter().all(|c| c.calibrated),
            has_undistortion: p.cameras.iter().any(|c| c.undistortion.is_some()),
            nb_images_calibration: p.cameras.first().map_or(0, |c| c.calibration_images.len()),
            cameras_optimized: p.cameras.iter().all(|c| c.optimized),
            trial_names: p.trials.iter().map(|t| t.name.clone()).collect(),
            trial_is_default: p.trials.iter().map(|t| t.is_default).collect(),
        }
    }

    /// `project->isCalibrated() || project->getCalibration() == NO_CALIBRATION`: the
    /// condition under which the marker-tracking panels are offered.
    #[must_use]
    pub fn calibrated_or_none(&self) -> bool {
        self.calibrated || self.calibration == CalibrationType::None
    }

    #[must_use]
    pub fn n_trials(&self) -> usize {
        self.trial_names.len()
    }
}

/// Status of a marker in one camera at one frame, or `Undefined` when out of range.
#[must_use]
pub fn marker_status(
    doc: &Document,
    trial: usize,
    marker: usize,
    cam: usize,
    frame: usize,
) -> MarkerStatus {
    doc.project()
        .trials
        .get(trial)
        .and_then(|t| t.markers.get(marker))
        .and_then(|m| m.views.get(cam))
        .and_then(|v| v.status.get(frame))
        .copied()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn change_reports_only_real_changes_unless_forced() {
        let mut s = AppState::default();
        assert!(!s.change_workspace(Workspace::Calibration, false).any());
        assert!(s.change_workspace(Workspace::Calibration, true).workspace);
        let ch = s.change_workspace(Workspace::MarkerTracking, false);
        assert!(ch.workspace && !ch.display);
        assert_eq!(s.workspace, Workspace::MarkerTracking);
    }

    #[test]
    fn active_frame_follows_into_trial_view() {
        let mut s = AppState {
            trials: vec![TrialView {
                active_frame: 0,
                start_frame: 1,
                end_frame: 10,
                n_frames: 10,
                active_marker: None,
                active_body: None,
            }],
            active_trial: Some(0),
            ..AppState::default()
        };
        assert!(
            s.change_active_frame_trial(Some(4), false)
                .active_frame_trial
        );
        assert_eq!(s.trials[0].active_frame, 4);
    }
}
