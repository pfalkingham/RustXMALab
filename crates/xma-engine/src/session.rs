//! Session state: what XMALab 3 keeps outside the project (`ui/State.cpp` and flags of the
//! camera and grid objects that are never saved), as far as creating projects and the
//! calibration workspaces need it.
//!
//! Not saved: the C++ values are not restored by reopening a project either (a reopened
//! project gets [`Session::after_load`]).
//!
//! # Undo
//!
//! Undo and redo put back with the project every value here that an edit writes and a later
//! edit reads (ledger A159): [`Session::undistorted`], [`Session::camera_recalibration`],
//! [`Session::grid_recalibration`] (what has been computed from the project and what is
//! pending), [`Session::n_calibration_images`] (Delete Frame), [`Session::active_camera`] and
//! [`Session::active_frame_calibration`] (set by the calibration edits), and [`Session::rng`]
//! (the XMALab 3 mode's generator, continued by every cube calibration). A change of any of
//! them makes an edit an undo step even when the project is unchanged. Not part of undo: the
//! wizard's input, which the window sets between edits and every `setDialog` clears or
//! reloads (a camera, frame or workspace change): the reference clicks
//! ([`Session::references`], [`Session::next_reference`]; "compute calibration" reads and
//! clears them, and its undo leaves them cleared), [`Session::manual_point`],
//! [`Session::manual_mode`], [`Session::manual_lists_pending`], [`Session::checkerboard_manual`];
//! and [`Session::workspace`]
//! and [`Session::wizard`], which only the window changes. The window shows the workspace
//! and the active camera and calibration frame from here; `xma parity` writes them into
//! `calib_state.json`, `undistortion/grid_state.csv` and `calibration/camera_state.csv`.
//!
//! # The project changes only in a step
//!
//! No function changes the project outside the history: whatever writes the project runs
//! inside an undo step ([`crate::Document::transact`] or an open step), which moves
//! [`crate::Document::changes`] and the revision when it changed something (ledger A164).
//! The one place where XMALab writes project data from the interface's own state is manual
//! mode: `reloadManualPoints` (the "manual mode" check box, a click on another calibration
//! image or camera in manual mode, every `setDialog`) gives the active calibration image `n`
//! unset points and `n` inlier flags 0 (`CalibrationImage::init`) when it has fewer than the
//! calibration object's `n`. Those lists are written to the project only when the user sets
//! a manual point (or another calibration action runs); until then the image is recorded in
//! [`Session::manual_lists_pending`], the project is unchanged, and the views can show the
//! lists through [`crate::Document::calibration_image_lists`]. The first calibration edit
//! after that creates them as part of its own step, before it reads the image, so every
//! calibration action sees what XMALab's sees; an edit that then changes nothing else drops
//! them again and records nothing. Nothing is lost by the wait: XMALab writes an image's
//! lists to disk only once the image is calibrated, so a project saved after switching
//! manual mode on and off is the project saved before, byte for byte.

use std::collections::BTreeSet;

use xma_model::{CalibrationType, Project};
use xma_vision::crand::CRand;

/// The workspace shown (`State::workspace`, `work_state`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Workspace {
    /// `UNDISTORTION`, 0.
    #[default]
    Undistortion = 0,
    /// `CALIBRATION`, 1.
    Calibration = 1,
    /// `DIGITIZATION` ("Marker tracking"), 2.
    Tracking = 2,
}

/// One of the four reference points of the cube calibration (the calibration wizard's
/// `radioButtonReference1..4` and `toolButtonReference1..4`; `docs/cpp-calibration.md`
/// C13 to C15).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReferenceClick {
    /// Where the user clicked, in image coordinates of the distorted image; `None` until
    /// clicked (the C++ holds -1, -1).
    pub position: Option<(f64, f64)>,
    /// The calibration-object point this reference stands for (0-based): from the
    /// reference file, or set by hand (C14).
    pub point_id: usize,
    /// True when `point_id` was set by hand ("CUSTOM INDEX").
    pub custom: bool,
}

/// The settings of XMALab 3 that the Undistortion and Calibration workspaces read and that
/// change no number: per machine in XMALab 3 (`core/Settings.cpp`), held by the window's own
/// settings here. The window copies them into [`Session::wizard`] after opening or creating
/// a project (a new session starts from XMALab's defaults); `xma parity` and `xma calibrate`
/// set them from the script's `Setting` lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct WizardSettings {
    /// `AutoConfirmPendingChanges` (default false): the pending-changes question (C11, C39)
    /// is answered "yes" without being shown.
    pub auto_confirm_pending_changes: bool,
    /// `AutoCalibAfterReference` (default true): the click that sets the fourth reference
    /// runs "compute calibration" (C19).
    pub auto_calib_after_reference: bool,
    /// `ShowAdvancedCalibration` (default false): "use camera distortion model" is shown,
    /// and "optimize camera setup" opens the optimisation dialog.
    pub show_advanced_calibration: bool,
    /// `HideWarningsDuringCalibration` (default false): the sentences about frames that
    /// could not be calibrated are not shown (the engine leaves them out of
    /// [`crate::EditOutcome::messages`]).
    pub hide_warnings_during_calibration: bool,
}

impl Default for WizardSettings {
    fn default() -> Self {
        WizardSettings {
            auto_confirm_pending_changes: false,
            auto_calib_after_reference: true,
            show_advanced_calibration: false,
            hide_warnings_during_calibration: false,
        }
    }
}

impl WizardSettings {
    /// Sets one of the four by its XMALab name; false for another name.
    pub fn set(&mut self, name: &str, value: bool) -> bool {
        let f = match name {
            "AutoConfirmPendingChanges" => &mut self.auto_confirm_pending_changes,
            "AutoCalibAfterReference" => &mut self.auto_calib_after_reference,
            "ShowAdvancedCalibration" => &mut self.show_advanced_calibration,
            "HideWarningsDuringCalibration" => &mut self.hide_warnings_during_calibration,
            _ => return false,
        };
        *f = value;
        true
    }
}

/// The session values of one open project.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Session {
    /// `State::workspace`.
    pub workspace: Workspace,
    /// `State::undistortion`: `UNDISTORTED` (true) or `NOTUNDISTORTED`.
    pub undistorted: bool,
    /// `State::activeCamera`.
    pub active_camera: usize,
    /// `State::activeFrameCalibration`.
    pub active_frame_calibration: usize,
    /// `Project::nbImagesCalibration`: the number of calibration frames, as
    /// `Project::addCamera` sets it (the last camera added; ledger A128 for the external
    /// calibration).
    pub n_calibration_images: usize,
    /// `Camera::requiresRecalibration`, per camera (0 or 1).
    pub camera_recalibration: Vec<i32>,
    /// `UndistortionObject::requiresRecalibration`, per camera (0 none, 1 refit, 2 match
    /// again); 0 for a camera without a grid.
    pub grid_recalibration: Vec<i32>,
    /// The four reference points of the calibration wizard (C13 to C15). Cleared, and the
    /// point ids reloaded from the calibration object, by [`Session::reload_references`].
    pub references: [ReferenceClick; 4],
    /// Which reference the next click sets (the checked radio button, 0 to 3).
    pub next_reference: usize,
    /// The wizard's "manual mode" check box (C25).
    pub manual_mode: bool,
    /// In manual mode, the calibration-object point the next click sets (C26).
    pub manual_point: usize,
    /// The calibration images, as (camera, image), that manual mode's `reloadManualPoints`
    /// gave their point lists outside an edit: the lists are created in the project by the
    /// next calibration edit, inside its step (ledger A164, module documentation). Wizard
    /// input: not part of undo. Kept true to the project after every committed or
    /// taken-back step, undo and redo: an image that no longer exists or no longer needs
    /// its lists leaves it, and in manual mode the active image is added again when it has
    /// fewer lists than the calibration object, so that the window always reads one entry
    /// per point for it.
    pub manual_lists_pending: BTreeSet<(usize, usize)>,
    /// The interface settings the calibration actions read ([`WizardSettings`]).
    pub wizard: WizardSettings,
    /// The C library generator the cube calibration's random search draws from in the
    /// XMALab 3 mode (`rand()`, never seeded by XMALab: `srand(1)` at the start of the
    /// session, ledger A5), continued by every calibration. Set to `CRand::new(1)` when a
    /// project is opened or created; `xma parity` sets it from a script's `Seed`. In the
    /// default mode each calibration starts from its own `CRand::new(1)` and this is not
    /// read.
    pub rng: CRand,
    /// The calibration wizard's `checkerboardManual`: a checkerboard's corners are clicked by
    /// hand (four reference clicks and a homography, C41) instead of detected (C40). Set from
    /// the setting `DisableCheckerboardDetection` when the page is made and at every change
    /// of active camera, calibration frame or workspace (to Calibration), and set by a
    /// detection that fails ("Could not detect chessboard. Please try the manual mode").
    /// Wizard input: not part of undo.
    pub checkerboard_manual: bool,
    /// Replaces `cv::findChessboardCorners` when set (tests that supply the corners of
    /// synthetic images); `None`: `xma_vision`'s routine. Not part of undo.
    pub chessboard_detector: Option<crate::calibration::ChessboardDetector>,
}

impl Session {
    /// The state right after File > New dataset, New dataset (external calibration) or New
    /// trial without calibration: active camera and calibration frame 0; a project without
    /// calibration shows Marker tracking; one with an undistortion grid the Undistortion
    /// workspace, not yet undistorted; otherwise it counts as undistorted and shows the
    /// Calibration workspace.
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::setupProjectUI (`:357-397`), with
    /// `State::undistortion` as `closeProject` leaves it (`NOTUNDISTORTED`).
    #[must_use]
    pub fn new_project(project: &Project) -> Session {
        let has_grid = project.cameras.iter().any(|c| c.undistortion.is_some());
        let (workspace, undistorted) = if project.calibration_type == CalibrationType::None {
            (Workspace::Tracking, false)
        } else if has_grid {
            (Workspace::Undistortion, false)
        } else {
            (Workspace::Calibration, true)
        };
        Session {
            workspace,
            undistorted,
            active_camera: 0,
            active_frame_calibration: 0,
            n_calibration_images: project
                .cameras
                .last()
                .map_or(0, |c| c.calibration_images.len()),
            camera_recalibration: vec![0; project.cameras.len()],
            grid_recalibration: vec![0; project.cameras.len()],
            references: [ReferenceClick::default(); 4],
            next_reference: 0,
            manual_mode: false,
            manual_point: 0,
            manual_lists_pending: BTreeSet::new(),
            wizard: WizardSettings::default(),
            rng: CRand::new(1),
            checkerboard_manual: project.algorithm_settings.disable_checkerboard_detection,
            chessboard_detector: None,
        }
    }

    /// True when an undistortion grid has changes that were not recomputed (C11).
    #[must_use]
    pub fn pending_undistortion(&self) -> bool {
        self.grid_recalibration.iter().any(|&r| r > 0)
    }

    /// True when a camera has calibration changes that were not recomputed (C39).
    #[must_use]
    pub fn pending_calibration(&self) -> bool {
        self.camera_recalibration.iter().any(|&r| r > 0)
    }

    /// C15: a click in the Calibration workspace on an image that is not calibrated: the
    /// checked reference takes the click and the next radio button is checked (after the
    /// fourth, none moves). Returns true when the click set the fourth reference (the
    /// fourth radio button was checked): with `AutoCalibAfterReference`
    /// ([`WizardSettings::auto_calib_after_reference`]) the caller then runs "compute
    /// calibration" (C19), whether or not the other three are set, as XMALab does (with
    /// fewer than three set that gives the error "You need to at least select 3 reference
    /// points").
    ///
    /// Ported from ui/WizardCalibrationCubeFrame.cpp, addCalibrationReference
    /// (`:423-455`).
    pub fn click_reference(&mut self, x: f64, y: f64) -> bool {
        let slot = self.next_reference.min(3);
        self.references[slot].position = Some((x, y));
        if slot < 3 {
            self.next_reference = slot + 1;
        }
        slot == 3
    }

    /// C14: a reference's calibration-object point set by hand (0-based).
    pub fn set_reference_index(&mut self, slot: usize, point_id: usize) {
        if let Some(r) = self.references.get_mut(slot) {
            r.point_id = point_id;
            r.custom = true;
        }
    }

    /// What every `setDialog` of the calibration wizard does (a camera, frame or workspace
    /// change, and after any calibration): the clicks are cleared, the first radio button
    /// is checked, the four point ids are reloaded from the first four ids of the
    /// calibration object's reference file (custom indices are lost), and in manual mode the
    /// first point is chosen again (`setupManualPoints`). [`crate::Document::calibration_dialog`]
    /// does this and what `setDialog` does to the project.
    pub fn reload_references(&mut self, project: &Project) {
        let ids: Vec<usize> = match &project.calibration_object {
            Some(xma_model::CalibrationObject::Frame(f)) => f
                .reference_ids
                .iter()
                .map(|&id| usize::try_from(id).unwrap_or(0))
                .collect(),
            _ => Vec::new(),
        };
        for (slot, r) in self.references.iter_mut().enumerate() {
            r.position = None;
            r.custom = false;
            if ids.len() >= 4 {
                r.point_id = ids[slot];
            }
        }
        self.next_reference = 0;
        // loadCalibrationSettings -> setupManualPoints: the first manual radio button.
        self.manual_point = 0;
    }

    /// The state after a project is opened: [`Session::new_project`], then, for a project
    /// with calibration, undistorted when every grid is computed, and Marker tracking when
    /// every camera has a calibrated image, else Calibration when undistorted.
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::loadProjectFinished (`:790-830`) and
    /// MainWindow::UndistortionAfterloadProjectFinished (`:860-918`).
    #[must_use]
    pub fn after_load(project: &Project) -> Session {
        let mut s = Session::new_project(project);
        if project.calibration_type == CalibrationType::None {
            return s;
        }
        if project
            .cameras
            .iter()
            .all(|c| c.undistortion.as_ref().is_none_or(|g| g.computed))
        {
            s.undistorted = true;
        }
        let all_calibrated = project
            .cameras
            .iter()
            .all(|c| c.calibration_images.iter().any(|im| im.is_calibrated()));
        if all_calibrated {
            s.workspace = Workspace::Tracking;
        } else if s.undistorted {
            s.workspace = Workspace::Calibration;
        }
        s
    }
}
