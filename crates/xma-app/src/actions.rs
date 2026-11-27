//! Every menu action and keyboard shortcut of XMALab 3, as one enum.
//!
//! Sources: `MainWindow.ui` (menu actions, ui-map "Menu bar"), ui/Shortcuts.cpp
//! (`Shortcuts::bindApplicationShortcuts`, ui-map "Keyboard shortcuts"), and the buttons of
//! the workspace and frame bars, which raise the same kind of request.
//!
//! - [`Action::info`] gives the label, the shortcuts and the C++ handler it ports.
//! - [`Action::enabled`] is the enabled-state rule: the C++ rule (from
//!   `MainWindow::workspaceChanged` and the constructor, `ui/MainWindow.cpp:186-280,
//!   1322-1563`) **and** [`Action::implemented`]. Actions that need later phases exist in
//!   the menus in their place but are disabled until their package sets `implemented`.
//! - [`Action::checked`] gives the check mark of checkable actions.
//! - Actions are raised by pushing onto an [`ActionQueue`]; the application dispatches the
//!   queue once per frame in `crate::app::XmaApp::dispatch` (one `match`, the single place
//!   where an action does something).
//! - [`shortcut_table`] lists the key bindings; [`collect_shortcuts`] turns key presses into
//!   actions ([`collect_frame_arrows`] in a floating panel's window).

use std::path::PathBuf;

use egui::{Key, KeyboardShortcut, Modifiers};

use crate::panels::PanelId;
use crate::settings::UiSettings;
use crate::state::{AppState, DisplayLayout, ProjectFacts, Workspace};
use crate::theme::ThemeChoice;
use crate::widgets::view_keys;

/// One action. See the module documentation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    // -- File ------------------------------------------------------------------------------
    NewProject,
    NewDatasetExternalCalibration,
    NewTrialWithoutCalibration,
    LoadProject,
    LoadProjectWithDifferentCalibration,
    CloseProject,
    /// File > Recent Files > the n-th entry (also the start screen's buttons).
    OpenRecentFile(usize),
    /// Open this file (command line, start screen).
    OpenPath(PathBuf),
    SaveProject,
    SaveProjectAs,
    SaveSubdatasetAs,
    Export3DPoints,
    Export2DPoints,
    ExportRigidBodyTransformations,
    ExportEvents,
    ExportMarkerToMarkerDistances,
    ExportPrecisionInfo,
    ExportReprojectionErrors,
    ExportUndistortedTrialImages,
    ExportMayaCams2,
    /// "XRayProject Matlab Format" (`actionExportAll`).
    ExportXRayProjectMatlab,
    /// "Blender scene": not in XMALab 3 (parity ledger U176, the export F85).
    ExportBlenderScene,
    /// Hidden at start-up in the C++ (`MainWindow.cpp:132`); never in the menu.
    ExportXrommVr,
    /// Visible only with the `ExportAllEnabled` setting.
    ExportAllTrialsForExternal,
    Import2DPoints,
    ImportTrial,
    UndistortSequence,
    DeleteMultipleTrials,
    // -- Edit ------------------------------------------------------------------------------
    Undo,
    /// Edit > Redo: not in XMALab 3 (parity ledger U52).
    Redo,
    // -- View ------------------------------------------------------------------------------
    Settings,
    About,
    Help,
    ToggleDisplayOptions,
    ToggleDetailedView,
    TogglePlot,
    Toggle3DWorldView,
    ToggleEvents,
    ProjectMetadata,
    ProjectOverview,
    ToggleDetectionSettings,
    /// View > Output (`actionConsole`): shows or hides the Console panel (the `Console`
    /// setting). Hidden in XMALab 3's menu; shown here, enabled always (parity ledger U174).
    ToggleConsole,
    LockUiPanes,
    Theme(ThemeChoice),
    /// View > Reset layout: every panel and the main window back to the default layout,
    /// what `--reset-layout` gives on a fresh start. Not in XMALab 3 (parity ledger U165).
    ResetLayout,
    // -- Shortcuts (ui/Shortcuts.cpp) ---------------------------------------------------------
    PrevFrame,
    NextFrame,
    PrevMarker,
    NextMarker,
    StopPlayback,
    TrackSelectedForwardAll,
    TrackSelectedNext,
    TrackActiveForwardAll,
    TrackActiveNext,
    TrackSelectedBackwardAll,
    TrackSelectedPrev,
    TrackActiveBackwardAll,
    TrackActivePrev,
    GoToLastTrackedFrame,
    GoToFirstTrackedFrame,
    MoveNFramesBackward,
    MoveNFramesForward,
    SetAndMoveNFramesBackward,
    SetAndMoveNFramesForward,
    NextPartiallySetFrame,
    PrevPartiallySetFrame,
    CenterViews,
    InterpolateActive,
    InterpolateAll,
    SetInterpolationRange,
    SetUntrackableRange,
    SelectAllPoints,
    NextAboveError,
    PrevAboveError,
    DeleteAllAboveError,
    ToggleHideAll,
    SetEventOn,
    SetEventOff,
    // -- Workspace bar and frame bar ------------------------------------------------------------
    SetWorkspace(Workspace),
    SetDisplay(DisplayLayout),
    SetActiveTrial(usize),
    AddTrial,
    TrialSettings,
    /// The Trial dialog's "Delete Trial" for this trial, raised by its confirmation.
    DeleteTrial(usize),
    CameraSettings,
    PlayForward,
    PlayBackward,
    /// Go to a frame (0-based) of the active trial or calibration.
    GoToFrame(usize),
    /// The start-frame button: asks for the start frame.
    SetStartFrame,
    /// The end-frame button: asks for the end frame.
    SetEndFrame,
    // -- Start screen and trial start frame ------------------------------------------------------
    AddDefaultTrial,
    // -- Panels -------------------------------------------------------------------------------
    FloatPanel(PanelId),
    DockPanel(PanelId),
    ClosePanel(PanelId),
    Quit,
    /// An action the user confirmed in a confirmation dialog: carried out without asking
    /// again.
    Confirmed(Box<Action>),
    // -- Points panel ---------------------------------------------------------------------
    /// The Points panel's editing controls (context menu, drop, buttons, "from default",
    /// a marker's settings button): `dialogs::points`.
    Points(crate::dialogs::points::PointsAction),
    /// The settings button of a rigid-body row: opens `RigidBodyDialog` for the body
    /// (0-based) of the active trial (`dialogs::rigid_body`).
    RigidBodySettings(usize),
    /// The Points panel's "Save the 3D data of selected points": File > Export > 3D Points
    /// for the selected markers (`MarkerTreeWidget::action_Save3D_triggered`).
    Export3DPointsSelected,
    /// The Points panel's "Save the transformation of selected rigid bodies"
    /// (`MarkerTreeWidget::action_SaveTransformations_triggered`).
    ExportTransformationsSelected,
    // -- Plot panel files -------------------------------------------------------------------
    /// The Plot panel's "Export": the application asks for the file, then passes
    /// [`Action::PlotExportTo`] to the panels.
    PlotExport,
    /// The Plot panel's extra-plot button: the application asks for the file, then passes
    /// [`Action::PlotExtraFrom`] to the panels.
    PlotExtraData,
    /// The file chosen for the Plot panel's export (handled by the panel).
    PlotExportTo(PathBuf),
    /// The file chosen for the Plot panel's extra plot (handled by the panel).
    PlotExtraFrom(PathBuf),
    // -- Calibration: optimisation and external calibration (`app/calibration_ext.rs`) ----
    /// The Calibration page's "optimize camera setup": the Optimization dialog with the
    /// setting `ShowAdvancedCalibration`, otherwise "Set number of iterations" (C37, C42).
    OptimizeCameraSetup,
    /// "Load MayaCam 2.0" of the external-calibration frame of a camera (C43).
    LoadMayaCam2(usize),
    /// Display Options, the button beside "filter images": the Visual filter dialog (C67).
    VisualFilterSettings,
    // -- Undistortion and Calibration workspaces ----------------------------------------------
    /// A control of the Undistortion or Calibration page, or a click in their camera views
    /// (`crate::calibration_ui`, carried out in `app/calibration.rs`).
    Calib(crate::calibration_ui::CalibAction),
}

/// Static description of an action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionInfo {
    /// Menu text (or a description for actions not in a menu).
    pub label: String,
    /// The C++ code this ports.
    pub cpp: &'static str,
}

/// What the enabled and checked rules read.
#[derive(Clone, Copy, Debug)]
pub struct ActionContext<'a> {
    pub state: &'a AppState,
    /// `None` when no project is open.
    pub facts: Option<&'a ProjectFacts>,
    pub settings: &'a UiSettings,
    /// A job runs that disables the window.
    pub busy: bool,
}

impl ActionContext<'_> {
    fn tracking_with_trials(&self) -> bool {
        self.facts.is_some_and(|f| f.n_trials() > 0)
            && self.state.workspace == Workspace::MarkerTracking
    }

    fn tracking_panels(&self) -> bool {
        self.tracking_with_trials() && self.facts.is_some_and(ProjectFacts::calibrated_or_none)
    }

    fn has_trial(&self) -> bool {
        self.state.workspace == Workspace::MarkerTracking
            && self
                .state
                .active_trial
                .is_some_and(|t| t < self.state.trials.len())
    }
}

impl Action {
    /// True when the action does something in this version of the program.
    #[must_use]
    pub fn implemented(&self) -> bool {
        use Action::*;
        matches!(
            self,
            NewProject
                | NewDatasetExternalCalibration
                | NewTrialWithoutCalibration
                | LoadProjectWithDifferentCalibration
                | AddTrial
                | AddDefaultTrial
                | LoadProject
                | CloseProject
                | SaveProject
                | SaveProjectAs
                | SaveSubdatasetAs
                | UndistortSequence
                | Undo
                | Redo
                | OpenRecentFile(_)
                | OpenPath(_)
                | ToggleDisplayOptions
                | ToggleDetailedView
                | TogglePlot
                | Toggle3DWorldView
                | ToggleEvents
                | ToggleConsole
                | LockUiPanes
                | Theme(_)
                | ResetLayout
                | PrevFrame
                | NextFrame
                | PrevMarker
                | NextMarker
                | StopPlayback
                | GoToLastTrackedFrame
                | GoToFirstTrackedFrame
                | MoveNFramesBackward
                | MoveNFramesForward
                | SetAndMoveNFramesBackward
                | SetAndMoveNFramesForward
                | NextPartiallySetFrame
                | PrevPartiallySetFrame
                | SelectAllPoints
                | CenterViews
                | ToggleHideAll
                | SetWorkspace(_)
                | NextAboveError
                | PrevAboveError
                | InterpolateActive
                | InterpolateAll
                | SetInterpolationRange
                | SetUntrackableRange
                | DeleteAllAboveError
                | SetEventOn
                | SetEventOff
                | SetDisplay(_)
                | SetActiveTrial(_)
                | PlayForward
                | PlayBackward
                | GoToFrame(_)
                | SetStartFrame
                | SetEndFrame
                | TrialSettings
                | DeleteTrial(_)
                | FloatPanel(_)
                | DockPanel(_)
                | ClosePanel(_)
                | Quit
                | Settings
                | About
                | Help
                | ProjectMetadata
                | ProjectOverview
                | Points(_)
                | TrackSelectedForwardAll
                | TrackSelectedNext
                | TrackActiveForwardAll
                | TrackActiveNext
                | TrackSelectedBackwardAll
                | TrackSelectedPrev
                | TrackActiveBackwardAll
                | TrackActivePrev
                | ToggleDetectionSettings
                | Import2DPoints
                | ImportTrial
                | DeleteMultipleTrials
                | RigidBodySettings(_)
                | Export3DPoints
                | Export2DPoints
                | ExportRigidBodyTransformations
                | ExportEvents
                | ExportMarkerToMarkerDistances
                | ExportPrecisionInfo
                | ExportReprojectionErrors
                | ExportMayaCams2
                | ExportUndistortedTrialImages
                | ExportAllTrialsForExternal
                | ExportBlenderScene
                | VisualFilterSettings
                | Export3DPointsSelected
                | ExportTransformationsSelected
                | CameraSettings
                | PlotExport
                | PlotExtraData
                | PlotExportTo(_)
                | PlotExtraFrom(_)
                | OptimizeCameraSetup
                | LoadMayaCam2(_)
                | Calib(_)
        ) || matches!(self, Confirmed(a) if a.implemented())
    }

    /// Label and origin.
    #[must_use]
    pub fn info(&self) -> ActionInfo {
        use Action::*;
        let (label, cpp): (&str, &'static str) = match self {
            NewProject => ("New dataset", "MainWindow::on_actionNew_Project_triggered"),
            NewDatasetExternalCalibration => (
                "New dataset (external calibration)",
                "MainWindow::on_actionNew_dataset_external_calibration_triggered",
            ),
            NewTrialWithoutCalibration => (
                "New trial without calibration",
                "MainWindow::on_actionNew_trial_without_calibration_triggered",
            ),
            LoadProject => (
                "Open dataset",
                "MainWindow::on_actionLoad_Project_triggered",
            ),
            LoadProjectWithDifferentCalibration => (
                "Open Dataset with different calibration",
                "MainWindow::on_actionLoad_Project_with_different_calibration_triggered",
            ),
            CloseProject => (
                "Close dataset",
                "MainWindow::on_actionClose_Project_triggered",
            ),
            OpenRecentFile(_) => ("Recent file", "MainWindow::loadRecentFile"),
            OpenPath(_) => ("Open file", "MainWindow::loadProject"),
            SaveProject => (
                "Save dataset",
                "MainWindow::on_actionSave_Project_triggered",
            ),
            SaveProjectAs => (
                "Save dataset as",
                "MainWindow::on_actionSave_Project_as_triggered",
            ),
            SaveSubdatasetAs => (
                "Save subdataset as",
                "MainWindow::on_actionSave_subdataset_as_triggered",
            ),
            Export3DPoints => (
                "3D Points",
                "MainWindow::on_actionExport3D_Points_triggered",
            ),
            Export2DPoints => (
                "2D Points",
                "MainWindow::on_actionExport2D_Points_triggered",
            ),
            ExportRigidBodyTransformations => (
                "RigidBodyTransformations",
                "MainWindow::on_actionRigidBodyTransformations_triggered",
            ),
            ExportEvents => ("Events", "MainWindow::on_actionExportEvents_triggered"),
            ExportMarkerToMarkerDistances => (
                "Marker to Marker Distances (range)",
                "MainWindow::on_actionMarkertoMarkerDistances_triggered",
            ),
            ExportPrecisionInfo => (
                "Precision Info",
                "MainWindow::on_actionPrecisionInfo_triggered",
            ),
            ExportReprojectionErrors => (
                "Reprojection Errors",
                "MainWindow::on_actionReprojection_Errors_triggered",
            ),
            ExportUndistortedTrialImages => (
                "Undistorted Trial images",
                "MainWindow::on_actionExport_Undistorted_Trial_images_for_Maya_triggered",
            ),
            ExportMayaCams2 => (
                "MayaCams 2.0",
                "MainWindow::on_actionMayaCams_2_0_triggered",
            ),
            ExportXRayProjectMatlab => (
                "XRayProject Matlab Format",
                "MainWindow::on_actionExportAll_triggered",
            ),
            ExportBlenderScene => ("Blender scene", "(new: parity ledger U176)"),
            ExportXrommVr => ("XROMM-VR", "MainWindow::on_actionXROMM_VR_triggered"),
            ExportAllTrialsForExternal => (
                "All Trials for External",
                "MainWindow::on_actionAll_Trials_for_External_triggered",
            ),
            Import2DPoints => (
                "2D Points",
                "MainWindow::on_actionImport2D_Points_triggered",
            ),
            ImportTrial => (
                "Trial from xma-file",
                "MainWindow::on_actionImportTrial_triggered",
            ),
            UndistortSequence => (
                "Undistort sequence",
                "MainWindow::on_actionUndistort_sequence_triggered",
            ),
            DeleteMultipleTrials => (
                "Delete multiple trials",
                "MainWindow::on_actionDelete_multiple_trials_triggered",
            ),
            Undo => ("Undo", "MainWindow::on_actionUndo_triggered"),
            Redo => ("Redo", "(new: parity ledger U52)"),
            Settings => ("Settings", "MainWindow::on_actionSettings_triggered"),
            About => ("About", "MainWindow::on_actionAbout_triggered"),
            Help => ("Help", "MainWindow::on_actionHelp_triggered"),
            ToggleDisplayOptions => (
                "Display Options",
                "MainWindow::on_actionDisplay_Options_triggered",
            ),
            ToggleDetailedView => (
                "Detailed View",
                "MainWindow::on_actionDetailed_View_triggered",
            ),
            TogglePlot => ("Plot", "MainWindow::on_actionPlot_triggered"),
            Toggle3DWorldView => (
                "3D world view",
                "MainWindow::on_action3D_world_view_triggered",
            ),
            ToggleEvents => ("Events", "MainWindow::on_actionEvents_triggered"),
            ProjectMetadata => (
                "Project Metadata",
                "MainWindow::on_actionProject_Metadata_triggered",
            ),
            ProjectOverview => (
                "Project Overview",
                "MainWindow::on_actionProject_Overview_triggered",
            ),
            ToggleDetectionSettings => (
                "DetectionSettings",
                "MainWindow::on_actionDetectionSettings_triggered",
            ),
            ToggleConsole => ("Output", "MainWindow::on_actionConsole_triggered"),
            LockUiPanes => (
                "Lock UI Panes",
                "MainWindow::on_actionLock_UI_Panes_toggled",
            ),
            Theme(c) => (
                c.label(),
                "MainWindow::onThemeLight/onThemeDark/onThemeSystem",
            ),
            ResetLayout => ("Reset layout", "(new: parity ledger U165)"),
            PrevFrame => (
                "Previous frame",
                "SequenceNavigationFrame::on_toolButtonPrev_clicked",
            ),
            NextFrame => (
                "Next frame",
                "SequenceNavigationFrame::on_toolButtonNext_clicked",
            ),
            PrevMarker => (
                "Select previous marker",
                "PointsDockWidget::selectPrevPoint",
            ),
            NextMarker => ("Select next marker", "PointsDockWidget::selectNextPoint"),
            StopPlayback => (
                "Stop playback",
                "SequenceNavigationFrame::on_toolButtonStop_clicked",
            ),
            TrackSelectedForwardAll => (
                "Track selected points forward through all frames",
                "Shortcuts::trackSelectedForw",
            ),
            TrackSelectedNext => (
                "Track selected points to the next frame",
                "Shortcuts::trackSelectedNext",
            ),
            TrackActiveForwardAll => (
                "Track active point forward through all frames",
                "Shortcuts::trackPointForw",
            ),
            TrackActiveNext => (
                "Track active point to the next frame",
                "Shortcuts::trackPointNext",
            ),
            TrackSelectedBackwardAll => (
                "Track selected points backward through all frames",
                "Shortcuts::trackSelectedBack",
            ),
            TrackSelectedPrev => (
                "Track selected points to the previous frame",
                "Shortcuts::trackSelectedPrev",
            ),
            TrackActiveBackwardAll => (
                "Track active point backward through all frames",
                "Shortcuts::trackPointBack",
            ),
            TrackActivePrev => (
                "Track active point to the previous frame",
                "Shortcuts::trackPointPrev",
            ),
            GoToLastTrackedFrame => (
                "Go to the last frame of the active point's tracked run",
                "WizardDigitizationFrame::goToLastTrackedFrame",
            ),
            GoToFirstTrackedFrame => (
                "Go to the first frame of the active point's tracked run",
                "WizardDigitizationFrame::goToFirstTrackedFrame",
            ),
            MoveNFramesBackward => (
                "Move back N frames",
                "SequenceNavigationFrame::moveNFramesBackward",
            ),
            MoveNFramesForward => (
                "Move forward N frames",
                "SequenceNavigationFrame::moveNFramesForward",
            ),
            SetAndMoveNFramesBackward => (
                "Set N, then move back N frames",
                "SequenceNavigationFrame::setAndMoveNFramesBackward",
            ),
            SetAndMoveNFramesForward => (
                "Set N, then move forward N frames",
                "SequenceNavigationFrame::setAndMoveNFramesForward",
            ),
            NextPartiallySetFrame => (
                "Next frame where the active point is set in some cameras but not all",
                "SequenceNavigationFrame::moveFrameToMissingForward",
            ),
            PrevPartiallySetFrame => (
                "Previous frame where the active point is set in some cameras but not all",
                "SequenceNavigationFrame::moveFrameToMissingBackward",
            ),
            CenterViews => (
                "Centre all camera views on the active point",
                "MainWindow::centerViews",
            ),
            InterpolateActive => (
                "Interpolate the active point",
                "WizardDockWidget::interpolateActive",
            ),
            InterpolateAll => ("Interpolate all points", "WizardDockWidget::interpolateAll"),
            SetInterpolationRange => (
                "Set interpolation over the plot's selected range",
                "PlotWindow::setInterpolation",
            ),
            SetUntrackableRange => (
                "Mark the plot's selected range untrackable",
                "PlotWindow::setUntrackable",
            ),
            SelectAllPoints => ("Select all points", "PointsDockWidget::selectAllPoints"),
            NextAboveError => (
                "Next frame above the plot error threshold",
                "PlotWindow::goToNextAboveError",
            ),
            PrevAboveError => (
                "Previous frame above the plot error threshold",
                "PlotWindow::goToPrevAboveError",
            ),
            DeleteAllAboveError => (
                "Delete all data above the plot error threshold",
                "PlotWindow::deleteAllAboveError",
            ),
            ToggleHideAll => (
                "Toggle 'hide all' overlays",
                "DisplayOptionsDockWidget::toggleHideAll",
            ),
            SetEventOn => (
                "Set the active event on over the plot's selected range",
                "PlotWindow::setEventOn",
            ),
            SetEventOff => (
                "Set the active event off over the plot's selected range",
                "PlotWindow::setEventOff",
            ),
            SetWorkspace(_) => (
                "Workspace",
                "WorkspaceNavigationFrame::on_comboBoxWorkspace_currentIndexChanged",
            ),
            SetDisplay(_) => (
                "View",
                "WorkspaceNavigationFrame::on_comboBoxViewspace_currentIndexChanged",
            ),
            SetActiveTrial(_) => (
                "Trial",
                "WorkspaceNavigationFrame::on_comboBoxTrial_currentIndexChanged",
            ),
            AddTrial => (
                "+",
                "WorkspaceNavigationFrame::on_toolButtonAddTrial_clicked",
            ),
            TrialSettings => (
                "Trial settings",
                "WorkspaceNavigationFrame::on_toolButtonTrialSettings_clicked",
            ),
            DeleteTrial(_) => (
                "Delete Trial",
                "TrialDialog::on_pushButton_DeleteTrial_clicked",
            ),
            CameraSettings => (
                "Camera settings",
                "WorkspaceNavigationFrame::on_toolButtonCameraSettings_clicked",
            ),
            PlayForward => ("Play", "SequenceNavigationFrame::on_toolButtonPlay_clicked"),
            PlayBackward => (
                "Play backward",
                "SequenceNavigationFrame::on_toolButtonPlayBackward_clicked",
            ),
            GoToFrame(_) => ("Go to frame", "SequenceNavigationFrame::changeFrame"),
            SetStartFrame => (
                "Set start frame",
                "SequenceNavigationFrame::on_toolButtonFrameStart_clicked",
            ),
            SetEndFrame => (
                "Set end frame",
                "SequenceNavigationFrame::on_toolButtonFrameEnd_clicked",
            ),
            AddDefaultTrial => (
                "Add Default Trial",
                "MainWindow::on_pushButtonDefaultTrial_clicked",
            ),
            FloatPanel(_) => ("Float", "QDockWidget float button"),
            DockPanel(_) => ("Dock", "QDockWidget float button (floating)"),
            ClosePanel(_) => ("Close", "QDockWidget close button"),
            Quit => ("Quit", "MainWindow::closeEvent"),
            Confirmed(a) => return a.info(),
            Points(p) => p.info(),
            RigidBodySettings(_) => (
                "Rigid body settings",
                "MarkerTreeWidgetButton::settingsButtonClicked",
            ),
            Export3DPointsSelected => (
                "Save the 3D data of selected points",
                "MarkerTreeWidget::action_Save3D_triggered",
            ),
            ExportTransformationsSelected => (
                "Save the transformation of selected rigid bodies",
                "MarkerTreeWidget::action_SaveTransformations_triggered",
            ),
            PlotExport | PlotExportTo(_) => ("Export", "PlotWindow::on_pushButtonSave_clicked"),
            PlotExtraData | PlotExtraFrom(_) => {
                ("Extra plot", "PlotWindow::on_toolButtonExtraPlot_clicked")
            }
            Calib(c) => c.info(),
            OptimizeCameraSetup => (
                "optimize camera setup",
                "WizardCalibrationCubeFrame::on_pushButtonOptimize_clicked",
            ),
            LoadMayaCam2(_) => (
                "Load MayaCam 2.0",
                "ExternalCalibrationFrame::on_pushButton_MayaCam_clicked",
            ),
            VisualFilterSettings => (
                "Visual filter settings",
                "DisplayOptionsDockWidget::on_toolButton_VisualFilter_clicked",
            ),
        };
        ActionInfo {
            label: label.to_owned(),
            cpp,
        }
    }

    /// The C++ enabled-state rule, without [`Action::implemented`].
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub fn cpp_enabled(&self, cx: &ActionContext<'_>) -> bool {
        use Action::*;
        let open = cx.facts.is_some();
        match self {
            // Always enabled in the C++ (some crash without a dataset; here they need one).
            NewProject | NewDatasetExternalCalibration | NewTrialWithoutCalibration => true,
            LoadProject | LoadProjectWithDifferentCalibration | OpenRecentFile(_) | OpenPath(_) => {
                true
            }
            CloseProject => true,
            SaveProject | SaveProjectAs | SaveSubdatasetAs => open,
            ExportMayaCams2 | ExportXRayProjectMatlab | ExportAllTrialsForExternal => open,
            // New (ledger U176): any workspace; without a calibrated camera the dialog says so.
            ExportBlenderScene => open,
            UndistortSequence | DeleteMultipleTrials => open,
            // MainWindow::workspaceChanged: only Marker tracking with at least one trial.
            Export3DPoints
            | Export2DPoints
            | ExportRigidBodyTransformations
            | ExportEvents
            | ExportMarkerToMarkerDistances
            | ExportPrecisionInfo
            | ExportReprojectionErrors
            | ExportUndistortedTrialImages
            | ExportXrommVr
            | Import2DPoints => cx.tracking_with_trials(),
            // Calibrated cameras or a dataset without calibration.
            ImportTrial => cx.facts.is_some_and(ProjectFacts::calibrated_or_none),
            // XMALab 3: `MainWindow::setUndo` (one point). Here: the document's undo and
            // redo stacks (parity ledger U52).
            Undo => open && cx.state.undo_label.is_some(),
            Redo => open && cx.state.redo_label.is_some(),
            Settings | About | Help => true,
            ProjectMetadata | ProjectOverview => open,
            ToggleDisplayOptions
            | ToggleDetailedView
            | TogglePlot
            | ToggleEvents
            | ToggleDetectionSettings => cx.tracking_panels(),
            // Also enabled in Calibration (MainWindow.cpp:1395).
            Toggle3DWorldView => {
                (open && cx.state.workspace == Workspace::Calibration) || cx.tracking_panels()
            }
            ToggleConsole | LockUiPanes | Theme(_) => true,
            // Also while the panes are locked: it is how a lost panel is brought back.
            ResetLayout => true,
            // Shortcuts: each handler checks its own conditions and does nothing otherwise.
            PrevFrame
            | NextFrame
            | StopPlayback
            | MoveNFramesBackward
            | MoveNFramesForward
            | SetAndMoveNFramesBackward
            | SetAndMoveNFramesForward
            | GoToFrame(_) => open,
            PlayForward | PlayBackward | SetStartFrame | SetEndFrame => cx.has_trial(),
            PrevMarker
            | NextMarker
            | GoToLastTrackedFrame
            | GoToFirstTrackedFrame
            | NextPartiallySetFrame
            | PrevPartiallySetFrame
            | SelectAllPoints => cx.has_trial(),
            // `trackPointsShortcut` returns while drawing is disabled; between the steps of a
            // run XMALab 3 clicks any enabled button. Here (T7 fixed) a run takes only the
            // key of its own continuous button, which stops it (`tracking_ui::press`).
            TrackSelectedForwardAll
            | TrackSelectedNext
            | TrackActiveForwardAll
            | TrackActiveNext
            | TrackSelectedBackwardAll
            | TrackSelectedPrev
            | TrackActiveBackwardAll
            | TrackActivePrev => {
                cx.has_trial()
                    && match cx.state.running_track {
                        Some(r) => crate::tracking_ui::TrackButton::of_action(self)
                            .is_some_and(|b| b == r && b.continuous),
                        None => !cx.state.disable_draw,
                    }
            }
            CenterViews
            | InterpolateActive
            | InterpolateAll
            | SetInterpolationRange
            | SetUntrackableRange
            | NextAboveError
            | PrevAboveError
            | DeleteAllAboveError
            | SetEventOn
            | SetEventOff => cx.has_trial() && !cx.state.disable_draw,
            ToggleHideAll => true,
            SetWorkspace(_) | SetDisplay(_) => open,
            SetActiveTrial(t) => cx.facts.is_some_and(|f| *t < f.n_trials()),
            AddTrial | AddDefaultTrial => open,
            TrialSettings => cx.has_trial(),
            DeleteTrial(t) => cx.facts.is_some_and(|f| *t < f.n_trials()),
            // WorkspaceNavigationFrame::workspaceChanged: enabled except in Marker tracking
            // without trials.
            CameraSettings => {
                open && (cx.state.workspace != Workspace::MarkerTracking
                    || cx.facts.is_some_and(|f| f.n_trials() > 0))
            }
            FloatPanel(_) | DockPanel(_) | ClosePanel(_) => !cx.settings.lock_ui_panes,
            Quit => true,
            Confirmed(a) => a.cpp_enabled(cx),
            // The Points tree's controls: a trial in Marker tracking, no tracking run.
            Points(_) | RigidBodySettings(_) => cx.has_trial() && !cx.state.disable_draw,
            // The Points tree's context menu (no tracking run) and the Plot panel.
            Export3DPointsSelected | ExportTransformationsSelected => {
                cx.has_trial() && !cx.state.disable_draw
            }
            PlotExport | PlotExtraData | PlotExportTo(_) | PlotExtraFrom(_) => cx.has_trial(),
            Calib(c) => c.cpp_enabled(cx.state, open),
            // "Load MayaCam 2.0" of the external-calibration frame.
            LoadMayaCam2(_) => open && cx.state.workspace == Workspace::Calibration,
            // Disabled with "Hide all" by the panel (DisplayOptionsDockWidget, `:124`).
            VisualFilterSettings => open,
            // Always enabled where the page shows it: the Calibration workspace of an
            // internal calibration (no check that the cameras are calibrated, CD19; the
            // engine refuses with a sentence, ledger A146).
            OptimizeCameraSetup => {
                cx.state.workspace == Workspace::Calibration
                    && cx
                        .facts
                        .is_some_and(|f| f.calibration == xma_model::CalibrationType::Internal)
            }
        }
    }

    /// The enabled-state rule: the C++ rule, implemented in this version, no job that
    /// disables the window running (`ProgressDialog::showProgressbar`), and, for the actions
    /// that change the document or its file ([`Action::blocked_while_tracking`]), no
    /// tracking run going on (`AppState::disable_draw`).
    #[must_use]
    pub fn enabled(&self, cx: &ActionContext<'_>) -> bool {
        let allowed_while_busy = matches!(self, Action::Quit | Action::Theme(_))
            || matches!(self, Action::Confirmed(a) if matches!(**a, Action::Quit));
        self.implemented()
            && self.cpp_enabled(cx)
            && (allowed_while_busy || !cx.busy)
            && !(cx.state.disable_draw && self.blocked_while_tracking())
    }

    /// Undo, redo and saving are not offered while a tracking run is open: the run is one
    /// undo step that is not finished yet. (XMALab 3 has no such rule; its Undo is cleared
    /// by every frame the run moves to.)
    #[must_use]
    pub fn blocked_while_tracking(&self) -> bool {
        match self {
            Action::Undo
            | Action::Redo
            | Action::SaveProject
            | Action::SaveProjectAs
            | Action::SaveSubdatasetAs => true,
            // The exports read (and the rigid-body export recomputes) the trial a run is
            // changing.
            a if a.is_export() => true,
            Action::Confirmed(a) => a.blocked_while_tracking(),
            _ => false,
        }
    }

    /// The File > Export entries that write an export, and the Points panel's two.
    #[must_use]
    pub fn is_export(&self) -> bool {
        use Action::*;
        matches!(
            self,
            Export3DPoints
                | Export2DPoints
                | ExportRigidBodyTransformations
                | ExportEvents
                | ExportMarkerToMarkerDistances
                | ExportPrecisionInfo
                | ExportReprojectionErrors
                | ExportUndistortedTrialImages
                | ExportMayaCams2
                | ExportXRayProjectMatlab
                | ExportBlenderScene
                | ExportXrommVr
                | ExportAllTrialsForExternal
                | Export3DPointsSelected
                | ExportTransformationsSelected
        )
    }

    /// The text of the action in its menu: [`ActionInfo::label`], except "Undo <step>" and
    /// "Redo <step>" with the name of the step (parity ledger U52).
    #[must_use]
    pub fn menu_label(&self, cx: &ActionContext<'_>) -> String {
        match (self, &cx.state.undo_label, &cx.state.redo_label) {
            (Action::Undo, Some(l), _) => format!("Undo {l}"),
            (Action::Redo, _, Some(l)) => format!("Redo {l}"),
            _ => self.info().label,
        }
    }

    /// The check mark of a checkable action, `None` for the others.
    ///
    /// The panel toggles show their setting. (The C++ also unchecks Plot, Events and
    /// DetectionSettings without changing their settings when the workspace becomes
    /// Calibration, `MainWindow.cpp:1386-1388`, so the mark can disagree with what is
    /// shown; here the mark always shows the setting.)
    #[must_use]
    pub fn checked(&self, cx: &ActionContext<'_>) -> Option<bool> {
        use Action::*;
        let s = cx.settings;
        match self {
            ToggleDisplayOptions => Some(s.show_display_options),
            ToggleDetailedView => Some(s.show_detail_view),
            TogglePlot => Some(s.show_plot),
            Toggle3DWorldView => Some(s.show_3d_view),
            ToggleEvents => Some(s.show_events),
            ToggleDetectionSettings => Some(cx.state.detection_settings_open),
            ToggleConsole => Some(s.console),
            LockUiPanes => Some(s.lock_ui_panes),
            Theme(c) => Some(crate::theme::ThemeChoice::from_setting(&s.theme) == *c),
            _ => None,
        }
    }

    /// The key bindings of this action (see [`shortcut_table`]).
    #[must_use]
    pub fn shortcuts(&self) -> Vec<KeyboardShortcut> {
        shortcut_table()
            .into_iter()
            .filter(|(_, a)| a == self)
            .map(|(k, _)| k)
            .collect()
    }

    /// The first shortcut as menu text, for example "Ctrl+N".
    #[must_use]
    pub fn shortcut_text(&self) -> Option<String> {
        self.shortcuts().first().map(shortcut_label)
    }
}

/// Text for a shortcut in Qt's style ("Ctrl+Shift+S").
#[must_use]
pub fn shortcut_label(k: &KeyboardShortcut) -> String {
    let mut s = String::new();
    let m = k.modifiers;
    if m.command || m.ctrl {
        s.push_str(if cfg!(target_os = "macos") {
            "Cmd+"
        } else {
            "Ctrl+"
        });
    }
    if m.alt {
        s.push_str("Alt+");
    }
    if m.shift {
        s.push_str("Shift+");
    }
    s.push_str(k.logical_key.name());
    s
}

/// The ordered queue of raised actions. Panels, menus and bars push; the application
/// takes the queue once per frame and dispatches it.
#[derive(Clone, Debug, Default)]
pub struct ActionQueue {
    queue: Vec<Action>,
}

impl ActionQueue {
    pub fn push(&mut self, a: Action) {
        self.queue.push(a);
    }

    /// Takes everything queued so far.
    pub fn take(&mut self) -> Vec<Action> {
        std::mem::take(&mut self.queue)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    #[must_use]
    pub fn pending(&self) -> &[Action] {
        &self.queue
    }
}

const CTRL: Modifiers = Modifiers::COMMAND;
const NONE: Modifiers = Modifiers::NONE;

fn sc(m: Modifiers, k: Key) -> KeyboardShortcut {
    KeyboardShortcut::new(m, k)
}

/// Every key binding: the menu actions' shortcuts (`MainWindow.ui`), then the application
/// shortcuts of `Shortcuts::bindApplicationShortcuts` in their source order. Ctrl is the
/// command key (Cmd on macOS), as Qt maps it.
///
/// Ctrl+O is both File > Open dataset and "Event off" in the C++ (ambiguous there); here
/// the menu action is listed first and wins while it is enabled.
#[must_use]
pub fn shortcut_table() -> Vec<(KeyboardShortcut, Action)> {
    use Action::*;
    let ca = Modifiers::COMMAND.plus(Modifiers::ALT);
    vec![
        (sc(CTRL, Key::N), NewProject),
        (sc(CTRL, Key::O), LoadProject),
        (sc(CTRL, Key::S), SaveProject),
        (sc(CTRL.plus(Modifiers::SHIFT), Key::S), SaveProjectAs),
        (sc(CTRL, Key::X), CloseProject),
        (sc(CTRL, Key::Z), Undo),
        (sc(CTRL.plus(Modifiers::SHIFT), Key::Z), Redo),
        (sc(CTRL, Key::Y), Redo),
        (sc(NONE, Key::ArrowLeft), PrevFrame),
        (sc(NONE, Key::ArrowRight), NextFrame),
        (sc(NONE, Key::ArrowUp), PrevMarker),
        (sc(NONE, Key::ArrowDown), NextMarker),
        (sc(NONE, Key::Escape), StopPlayback),
        (sc(ca, Key::Num2), TrackSelectedForwardAll),
        (sc(CTRL, Key::Num2), TrackSelectedNext),
        (sc(Modifiers::ALT, Key::Num2), TrackActiveForwardAll),
        (sc(NONE, Key::Num2), TrackActiveNext),
        (sc(ca, Key::Num1), TrackSelectedBackwardAll),
        (sc(CTRL, Key::Num1), TrackSelectedPrev),
        (sc(Modifiers::ALT, Key::Num1), TrackActiveBackwardAll),
        (sc(NONE, Key::Num1), TrackActivePrev),
        (sc(NONE, Key::Num3), GoToLastTrackedFrame),
        (sc(CTRL, Key::Num3), GoToFirstTrackedFrame),
        (sc(NONE, Key::Q), PrevFrame),
        (sc(NONE, Key::W), NextFrame),
        (sc(NONE, Key::Num5), MoveNFramesBackward),
        (sc(NONE, Key::Num6), MoveNFramesForward),
        (sc(CTRL, Key::Num5), SetAndMoveNFramesBackward),
        (sc(CTRL, Key::Num6), SetAndMoveNFramesForward),
        (sc(NONE, Key::Num7), NextPartiallySetFrame),
        (sc(CTRL, Key::Num7), PrevPartiallySetFrame),
        (sc(NONE, Key::C), CenterViews),
        (sc(NONE, Key::I), InterpolateActive),
        (sc(Modifiers::SHIFT, Key::I), InterpolateAll),
        (sc(CTRL, Key::I), SetInterpolationRange),
        (sc(CTRL, Key::U), SetUntrackableRange),
        (sc(CTRL, Key::A), SelectAllPoints),
        (sc(NONE, Key::Num4), NextAboveError),
        (sc(CTRL, Key::Num4), PrevAboveError),
        (sc(Modifiers::SHIFT, Key::Delete), DeleteAllAboveError),
        (sc(Modifiers::SHIFT, Key::Backspace), DeleteAllAboveError),
        (sc(NONE, Key::H), ToggleHideAll),
        (sc(NONE, Key::O), SetEventOn),
        (sc(CTRL, Key::O), SetEventOff),
    ]
}

/// Keys a text field needs for itself: while one has keyboard focus these shortcuts are
/// not taken (ui/Shortcuts.cpp:208-217 lets digits, `.`, `,`, `-`, Backspace and Delete
/// through to spin boxes and line edits; arrows, letters and the editing chords are also
/// left to the field here).
fn text_field_key(k: &KeyboardShortcut) -> bool {
    let m = k.modifiers;
    if m.alt {
        return false;
    }
    if m.command || m.ctrl {
        return matches!(
            k.logical_key,
            Key::A | Key::X | Key::Z | Key::Y | Key::C | Key::V
        );
    }
    true
}

/// Turns this frame's key presses into actions. Exact modifier match (Ctrl+2 is not 2);
/// each key press gives at most one action; a disabled action does not take the key, so a
/// later binding of the same key can (the Ctrl+O pair). While a text field has keyboard
/// focus, keys it needs are left to it. A view that has the focus for a key of its own
/// (the plot) is not a text field ([`view_keys`]): every shortcut works then.
#[must_use]
pub fn collect_shortcuts(ctx: &egui::Context, acx: &ActionContext<'_>) -> Vec<Action> {
    collect(ctx, acx, false)
}

/// The Left and Right arrows pressed in a floating panel's window, by the rules of
/// [`collect_shortcuts`]: the window is the operating system's, which sends it the keys
/// while it has the keyboard, so the main window never sees them. (The other shortcuts
/// are not taken there.)
#[must_use]
pub fn collect_frame_arrows(ctx: &egui::Context, acx: &ActionContext<'_>) -> Vec<Action> {
    collect(ctx, acx, true)
}

fn collect(ctx: &egui::Context, acx: &ActionContext<'_>, frame_arrows_only: bool) -> Vec<Action> {
    let text_focus = ctx.egui_wants_keyboard_input() && !view_keys::view_has_focus(ctx);
    let table = shortcut_table();
    let mut out = Vec::new();
    ctx.input_mut(|i| {
        i.events.retain(|e| {
            let egui::Event::Key {
                key,
                pressed: true,
                modifiers,
                ..
            } = e
            else {
                return true;
            };
            let found = table.iter().find(|(k, a)| {
                k.logical_key == *key
                    && (same_modifiers(k.modifiers, *modifiers)
                        || (takes_shift(k, a) && same_modifiers(Modifiers::SHIFT, *modifiers)))
                    && (!frame_arrows_only || frame_arrow(k, a))
                    && !(text_focus && text_field_key(k))
                    && a.enabled(acx)
            });
            match found {
                Some((_, a)) => {
                    out.push(a.clone());
                    false
                }
                None => true,
            }
        });
    });
    out
}

/// The Left and Right arrows as the previous and next frame keys.
fn frame_arrow(k: &KeyboardShortcut, a: &Action) -> bool {
    matches!(k.logical_key, Key::ArrowLeft | Key::ArrowRight)
        && matches!(a, Action::PrevFrame | Action::NextFrame)
}

/// The previous and next frame keys (arrows, Q, W) also work with Shift held: Shift held
/// while the frame changes selects the Plot panel's range (`PlotWindow::ShiftPressed`), and
/// `Shortcuts::eventFilter` tests no modifier for the arrows and only Meta for Q and W.
fn takes_shift(k: &KeyboardShortcut, a: &Action) -> bool {
    matches!(a, Action::PrevFrame | Action::NextFrame) && k.modifiers == Modifiers::NONE
}

/// Exact match of Ctrl/Cmd, Alt and Shift.
fn same_modifiers(want: Modifiers, got: Modifiers) -> bool {
    (want.command || want.ctrl) == (got.command || got.ctrl)
        && want.alt == got.alt
        && want.shift == got.shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shortcut_has_a_label_and_origin() {
        for (k, a) in shortcut_table() {
            let info = a.info();
            assert!(!info.label.is_empty());
            assert!(!info.cpp.is_empty());
            assert!(!shortcut_label(&k).is_empty());
        }
    }

    #[test]
    fn no_project_disables_tracking_actions() {
        let state = AppState::default();
        let settings = UiSettings::default();
        let cx = ActionContext {
            state: &state,
            facts: None,
            settings: &settings,
            busy: false,
        };
        assert!(Action::LoadProject.enabled(&cx));
        assert!(!Action::Export2DPoints.enabled(&cx));
        assert!(!Action::NextFrame.enabled(&cx));
        assert!(Action::NewProject.enabled(&cx));
        assert!(!Action::AddTrial.enabled(&cx), "needs a dataset");
        assert!(!Action::UndistortSequence.enabled(&cx));
        let busy = ActionContext { busy: true, ..cx };
        assert!(!Action::LoadProject.enabled(&busy));
    }

    /// A floating panel's window as `shell/floating.rs` handles its keys: the frame arrows
    /// are taken, then the window's widgets (here one text field) are drawn.
    #[derive(Default)]
    struct FloatingWindow {
        state: AppState,
        facts: ProjectFacts,
        settings: UiSettings,
        text: String,
        /// The text field asks for the keyboard focus.
        focus_field: bool,
        /// A job disables the window.
        busy: bool,
        taken: Vec<Action>,
        /// Key presses left for the window's widgets.
        left: Vec<Key>,
    }

    impl FloatingWindow {
        fn ui(&mut self, ui: &mut egui::Ui) {
            let ctx = ui.ctx().clone();
            let acx = ActionContext {
                state: &self.state,
                facts: Some(&self.facts),
                settings: &self.settings,
                busy: self.busy,
            };
            self.taken.extend(collect_frame_arrows(&ctx, &acx));
            ctx.input(|i| {
                for e in &i.events {
                    if let egui::Event::Key {
                        key, pressed: true, ..
                    } = e
                    {
                        self.left.push(*key);
                    }
                }
            });
            let field = ui
                .add(egui::TextEdit::singleline(&mut self.text))
                .accessible_name("a text field");
            if self.focus_field {
                field.request_focus();
            }
        }

        /// Presses `keys`, one frame each, and gives what was taken and what was left.
        fn press(h: &mut egui_kittest::Harness<'_, Self>, keys: &[Key]) -> (Vec<Action>, Vec<Key>) {
            h.run_steps(2);
            h.state_mut().taken.clear();
            h.state_mut().left.clear();
            for k in keys {
                h.key_press(*k);
                h.run_steps(2);
            }
            let s = h.state_mut();
            (std::mem::take(&mut s.taken), std::mem::take(&mut s.left))
        }
    }

    fn floating_window(w: FloatingWindow) -> egui_kittest::Harness<'static, FloatingWindow> {
        egui_kittest::Harness::builder().build_ui_state(|ui, w: &mut FloatingWindow| w.ui(ui), w)
    }

    #[test]
    fn a_floating_window_takes_the_frame_arrows_and_no_other_shortcut() {
        let mut h = floating_window(FloatingWindow::default());
        let keys = [
            Key::ArrowRight,
            Key::W,
            Key::ArrowDown,
            Key::ArrowLeft,
            Key::Num2,
        ];
        let (taken, left) = FloatingWindow::press(&mut h, &keys);
        assert_eq!(taken, [Action::NextFrame, Action::PrevFrame]);
        assert_eq!(left, [Key::W, Key::ArrowDown, Key::Num2]);
        // A disabled action does not take its key.
        h.state_mut().busy = true;
        let (taken, left) = FloatingWindow::press(&mut h, &[Key::ArrowRight]);
        assert!(taken.is_empty());
        assert_eq!(left, [Key::ArrowRight]);
    }

    #[test]
    fn a_floating_window_leaves_the_arrows_to_its_focused_text_field() {
        let mut h = floating_window(FloatingWindow {
            focus_field: true,
            ..FloatingWindow::default()
        });
        let keys = [Key::ArrowRight, Key::ArrowLeft];
        let (taken, left) = FloatingWindow::press(&mut h, &keys);
        assert!(h.ctx.text_edit_focused());
        assert!(taken.is_empty(), "the field has the focus: {taken:?}");
        assert_eq!(left, keys);
    }
}
